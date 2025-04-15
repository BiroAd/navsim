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
        self._vlm_model = VisionLLMModel(self._trajectory_sampling, config)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

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

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [VisionLLMTargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [VisionLLMFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._vlm_model(features)

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
        num_epochs = 5
        gradient_accumulation_steps = 2

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
                "frequency": 2,
            }
        }
        #return optimizer

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [VisionLLMCallback(self._config)]
