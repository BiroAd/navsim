from typing import Any, Dict, List, Optional, Union

import math
import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.vision_llm.vision_llm_callback import VisionLLMCallback
from navsim.agents.vision_llm.vision_llm_config import VisionLLMConfig
from navsim.agents.vision_llm.vision_llm_features import VisionLLMFeatureBuilder, VisionLLMTargetBuilder
from navsim.agents.vision_llm.vision_llm_loss import vision_llm_loss
from navsim.agents.vision_llm.vision_llm_model import VisionLLMModel
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory


class VisionLLMAgent(AbstractAgent):
    """Agent interface for VisionLLM baseline."""

    def __init__(
        self,
        config: VisionLLMConfig,
        lr_base: float,
        lr_final: float,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        :param trajectory_sampling: trajectory sampling specification
        """
        super().__init__(trajectory_sampling)

        self._config = config
        self._lr_base = lr_base
        self._lr_final = lr_final

        self._checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._vlm_model = VisionLLMModel(self._trajectory_sampling, config)
        #self._vlm_model = torch.compile(self._vlm_model)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Load checkpoint into the VisionLLMAgent, remapping keys if necessary."""
        # Load checkpoint
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location="cpu")["state_dict"]

        # Fix prefixes: remove "agent." and "_orig_mod" if present
        new_state_dict = {}
        for k, v in state_dict.items():
            k_new = k.replace("agent.", "")
            k_new = k_new.replace("_vlm_model._orig_mod.", "_vlm_model.")
            new_state_dict[k_new] = v

        # Load state dict non-strictly to avoid warnings for any remaining mismatches
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)

        if missing_keys:
            print("Missing keys:", missing_keys)
        if unexpected_keys:
            print("Unexpected keys:", unexpected_keys)

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        # NOTE: Transfuser only uses current frame (with index 3 by default)
        history_steps = [3]
        return SensorConfig(
            cam_f0=history_steps,
            cam_l0=history_steps,
            cam_l1=False,
            cam_l2=history_steps,
            cam_r0=history_steps,
            cam_r1=False,
            cam_r2=history_steps,
            cam_b0=history_steps,
            lidar_pc=False,
        )
        '''return SensorConfig(
                cam_f0=False,
                cam_l0=False,
                cam_l1=False,
                cam_l2=False,
                cam_r0=False,
                cam_r1=False,
                cam_r2=False,
                cam_b0=False,
                lidar_pc=False,
        )'''

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [VisionLLMTargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [VisionLLMFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], teacher_forcing_seq, teacher_forcing_ratio) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._vlm_model(features, teacher_forcing_seq, teacher_forcing_ratio)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return vision_llm_loss(targets, predictions, self._config)

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        '''batch_size = self._config.dataloader.params.batch_size
        num_epochs = self._config.trainer.params.max_epochs
        gradient_accumulation_steps = self._config.trainer.params.accumulate_grad_batches'''
        num_epochs = 10
        gradient_accumulation_steps = 32

        num_training_steps = self._config.total_training_steps

        total_steps = num_training_steps * num_epochs
        total_steps = total_steps // gradient_accumulation_steps


        # Define the AdamW optimizer with initial learning rate of 0
        # The actual LR will be controlled by the scheduler
        optimizer = torch.optim.AdamW(self._vlm_model.parameters(), lr=self._lr_base)
        #optimizer = torch.optim.AdamW(self._vlm_model.parameters(), lr=0.05)

        # Define the learning rate scheduler
        warmup_steps = int(0.1 * total_steps)  # First 10% of steps for warmup
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup from 0 to max_lr
                return step / warmup_steps
            else:
                # Cosine decay from max_lr to final_lr
                decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
                # Scale cosine decay from 1.0 to final_lr/max_lr
                return self._lr_final/self._lr_base + (1.0 - self._lr_final/self._lr_base) * cosine_decay
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": gradient_accumulation_steps,
            }
        }
        #return optimizer

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [VisionLLMCallback(self._config)]

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}

        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        # add batch dimension
        #features = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
        features = {
            k: (v.unsqueeze(0).to(self.device) if isinstance(v, torch.Tensor) else v) 
            for k, v in features.items()
        }

        teacher_forcing_seq = None
        teacher_forcing_ratio = 1.0

        # forward pass
        with torch.no_grad():
            predictions = self.forward(features, teacher_forcing_seq=teacher_forcing_seq, teacher_forcing_ratio=teacher_forcing_ratio)
            poses = predictions["trajectory"].squeeze(0).cpu().numpy()

        # extract trajectory
        return Trajectory(poses, self._trajectory_sampling)
