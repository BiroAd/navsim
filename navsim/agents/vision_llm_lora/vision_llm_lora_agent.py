import math
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.vision_llm_lora.vision_llm_lora_callback import VisionLLMLoRACallback
from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_features import (
    VisionLLMLoRAFeatureBuilder,
    VisionLLMLoRATargetBuilder,
)
from navsim.agents.vision_llm_lora.vision_llm_lora_loss import vision_llm_lora_loss
from navsim.agents.vision_llm_lora.vision_llm_lora_model import VisionLLMLoRAModel
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class VisionLLMLoRAAgent(AbstractAgent):
    """Agent interface for LoRA fine-tuning on top of a Hugging Face CausalLM."""

    def __init__(
        self,
        config: VisionLLMLoRAConfig,
        lr_base: float,
        lr_final: float,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        super().__init__(trajectory_sampling)
        self._config = config
        self._lr_base = lr_base
        self._lr_final = lr_final
        self._checkpoint_path = checkpoint_path
        self._vlm_model = VisionLLMLoRAModel(self._trajectory_sampling, config)

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        """Load checkpoint into agent if path is provided."""
        if self._checkpoint_path is None:
            return

        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict = torch.load(self._checkpoint_path, map_location="cpu")["state_dict"]

        remapped = {}
        for key, value in state_dict.items():
            key_new = key.replace("agent.", "")
            key_new = key_new.replace("_vlm_model._orig_mod.", "_vlm_model.")
            remapped[key_new] = value

        missing_keys, unexpected_keys = self.load_state_dict(remapped, strict=False)
        if missing_keys:
            print("Missing keys:", missing_keys)
        if unexpected_keys:
            print("Unexpected keys:", unexpected_keys)

    def get_sensor_config(self) -> SensorConfig:
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
        return [VisionLLMLoRATargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [VisionLLMLoRAFeatureBuilder(config=self._config)]

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        teacher_forcing_seq: Optional[torch.Tensor],
        teacher_forcing_ratio: float,
    ) -> Dict[str, torch.Tensor]:
        return self._vlm_model(features, teacher_forcing_seq, teacher_forcing_ratio)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return vision_llm_lora_loss(targets, predictions, self._config)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        total_steps = self._config.total_training_steps * self._config.num_epochs
        total_steps = max(1, total_steps // max(1, self._config.gradient_accumulation_steps))
        warmup_steps = int(self._config.warmup_ratio * total_steps)

        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found. Check LoRA setup and freezing strategy.")

        traj_named   = []
        ego_named    = []
        lang_named   = []
        vision_named = []
        other_named  = []


        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if (
                "traj_classifier" in name
                or "planner_queries" in name
                or "query_pos_embed" in name
            ):
                traj_named.append((name, param))

            elif (
                "velocity_encoder" in name
                or "acceleration_encoder" in name
                or "direction_embedding" in name
                or "ego_fusion_mlp" in name
            ):
                ego_named.append((name, param))

            elif "lora" in name and "visual" in name:
                vision_named.append((name, param))

            elif "lora" in name:
                lang_named.append((name, param))

            else:
                # Catch-all: new token embeddings, token_type_embed, etc.
                other_named.append((name, param))
            
        traj_decay, traj_no_decay = self.split_decay_params(traj_named)
        ego_decay, ego_no_decay = self.split_decay_params(ego_named)
        lang_decay, lang_no_decay = self.split_decay_params(lang_named)
        vision_decay, vision_no_decay = self.split_decay_params(vision_named)
        other_decay, other_no_decay = self.split_decay_params(other_named)

        # ── LR multipliers ────────────────────────────────────────────────────
        # Stage 1 (heads only, LLM frozen): lr_base should be ~5e-3
        #   traj and ego are from scratch → full base LR
        #   lang/vision will be empty → no effect
        #
        # Stage 2 (LoRA fine-tune): lr_base should be ~5e-4
        #   traj and ego already warm → reduced multiplier
        #   lang/vision pretrained → conservative multiplier
        traj_lr = self._config.traj_lr_multiplier * self._lr_base   # stage1: 1.0, stage2: 0.4
        ego_lr = self._config.ego_lr_multiplier * self._lr_base   # stage1: 1.0, stage2: 0.5
        lang_lr = self._config.lang_lr_multiplier * self._lr_base   # stage1: 0.0 (empty), stage2: 0.1
        vision_lr = self._config.vision_lr_multiplier * self._lr_base   # stage1: 0.0 (empty), stage2: 0.05

        param_groups = []

        def _add_group(decay_params, no_decay_params, lr, weight_decay):
            if decay_params:
                param_groups.append({
                    "params": decay_params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                })
            if no_decay_params:
                param_groups.append({
                    "params": no_decay_params,
                    "lr": lr,
                    "weight_decay": 0.0,
                })
        
        _add_group(traj_decay, traj_no_decay, traj_lr, 0.05)
        _add_group(ego_decay, ego_no_decay, ego_lr, 0.01)
        _add_group(lang_decay, lang_no_decay, lang_lr, 0.01)
        _add_group(vision_decay, vision_no_decay, vision_lr, 0.01)
        _add_group(other_decay, other_no_decay, ego_lr, 0.01)


        if not param_groups:
            raise RuntimeError("No trainable parameters found.")

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        """optimizer = torch.optim.AdamW([
            {"params": decay_params,    "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=self._lr_base)    """

        total_params = sum(
        p.numel() for g in param_groups for p in g["params"]
        )
        print(f"Optimizer parameters: {total_params:,}")


        def build_lambda(group_lr: float) -> callable:
            final_ratio = self._lr_final / self._lr_base

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                decay_ratio = (step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                decay_ratio = min(max(decay_ratio, 0.0), 1.0)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
                return final_ratio + (1.0 - final_ratio) * cosine_decay

            return lr_lambda
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[build_lambda(g["lr"]) for g in param_groups],
        )

                
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": self._config.gradient_accumulation_steps,
            },
        }

    def split_decay_params(self, named_params):
        """
        Split parameters into:
        - decay
        - no_decay

        No weight decay for:
        - bias
        - LayerNorm / RMSNorm
        - norm scales
        """
        decay = []
        no_decay = []

        for name, param in named_params:
            if not param.requires_grad:
                continue

            name_lower = name.lower()

            if (
                "bias" in name_lower
                or "norm" in name_lower
                or "layernorm" in name_lower
                or "ln_" in name_lower
                or "rmsnorm" in name_lower
            ):
                no_decay.append(param)
            else:
                decay.append(param)

        return decay, no_decay

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [VisionLLMLoRACallback(self._config)]

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()
        features: Dict[str, torch.Tensor] = {}

        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        features = {k: (v.unsqueeze(0).to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features, teacher_forcing_seq=None, teacher_forcing_ratio=1.0)
            poses = predictions["trajectory"].squeeze(0).cpu().numpy()

        return Trajectory(poses, self._trajectory_sampling)
