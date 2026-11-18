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
from navsim.agents.vision_llm_lora.vision_llm_lora_aux import aux_output_channels
from navsim.agents.vision_llm_lora.vision_llm_lora_features import (
    PDMScoreTargetBuilder,
    VisionLLMLoRAAuxTargetBuilder,
    VisionLLMLoRAEgoHistoryBuilder,
    VisionLLMLoRAFeatureBuilder,
    VisionLLMLoRAHistoryFrameBuilder,
    VisionLLMLoRATargetBuilder,
    normalize_uint8_camera,
)
from navsim.agents.vision_llm_lora.vision_llm_lora_loss import (
    V1_MULTIPLICATIVE_PDM_SCORE_KEYS,
    V1_WEIGHTED_PDM_SCORE_WEIGHTS,
    aggregate_pdm_scores,
    vision_llm_lora_loss,
)
from navsim.agents.vision_llm_lora.vision_llm_lora_model import VisionLLMLoRAModel
from navsim.agents.vision_llm_lora.vision_llm_lora_rl import RLReferenceTargetBuilder
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


TRAJECTORY_SELECTIONS = (
    "argmax",
    "score_heads",
    "score_heads_v1",
    "shortlist",
    "shortlist_v1",
    "mixture",
    "mixture_v1",
    "local",
    "local_v1",
)


def select_trajectory(
    predictions: Dict[str, torch.Tensor],
    selection: str,
    codebook: torch.Tensor,
    shortlist_size: int = 64,
    mixture_lambda: float = 10.0,
    local_size: int = 64,
    local_lambda: float = 0.1,
) -> torch.Tensor:
    """
    Output trajectory per config.trajectory_selection.
    :param predictions: model outputs; "trajectory" (B, 8, 3), and for the head-based selections "pdm_score_logits"
        (B, M, K) and, for "shortlist*" / "mixture*", the policy's "trajectory_logits" (B, K)
    :param selection: "argmax" (the model's own output); "score_heads" (entry with the best predicted total, v2 formula
        without two-frame comfort); "shortlist" (the policy's most likely entry among the shortlist_size entries with
        the best predicted total); "mixture" (argmax of log p + mixture_lambda * log predicted total); "local" (a
        discrete version of test-time search around the policy's choice, after TOAD, arXiv 2606.07170: among the
        local_size codebook entries nearest to the policy's most likely entry by ADE over (x, y), itself included, the
        one with the best predicted total - local_lambda * ADE in metres; the anchor term keeps the policy's entry
        unless a neighbour's predicted gain exceeds local_lambda per metre of displacement); the "_v1" variants use the
        NAVSIM v1 PDMS formula for the predicted total
    :param codebook: (K, 8, 3) codebook trajectories
    :return: (B, 8, 3)
    """
    if selection not in TRAJECTORY_SELECTIONS:
        raise ValueError(f"trajectory_selection must be one of {TRAJECTORY_SELECTIONS}, got {selection!r}")
    if selection == "argmax":
        return predictions["trajectory"]
    sub_scores = predictions["pdm_score_logits"].float().sigmoid()
    if selection.endswith("_v1"):
        total = aggregate_pdm_scores(sub_scores, V1_MULTIPLICATIVE_PDM_SCORE_KEYS, V1_WEIGHTED_PDM_SCORE_WEIGHTS)
    else:
        # Same aggregation as head_all in scripts/evaluation/eval_vision_llm_lora_checkpoints.py.
        total = aggregate_pdm_scores(sub_scores)  # (B, K)
    if selection.startswith("score_heads"):
        return codebook[total.argmax(dim=-1)]
    log_p = predictions["trajectory_logits"].float().log_softmax(dim=-1)  # (B, K)
    if selection.startswith("shortlist"):
        short = total.topk(min(shortlist_size, total.shape[-1]), dim=-1).indices  # (B, k)
        best = short.gather(1, log_p.gather(1, short).argmax(dim=-1, keepdim=True))[:, 0]
    elif selection.startswith("local"):
        xy = codebook[..., :2].float().to(total.device)  # (K, 8, 2)
        anchor = xy[log_p.argmax(dim=-1)]  # (B, 8, 2)
        ade = (xy[None] - anchor[:, None]).norm(dim=-1).mean(dim=-1)  # (B, K), 0 at the anchor
        near = ade.topk(min(local_size, ade.shape[-1]), dim=-1, largest=False).indices  # (B, n), anchor included
        objective = total.gather(1, near) - local_lambda * ade.gather(1, near)
        best = near.gather(1, objective.argmax(dim=-1, keepdim=True))[:, 0]
    else:  # mixture
        best = (log_p + mixture_lambda * total.clamp_min(1e-6).log()).argmax(dim=-1)
    return codebook[best]


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
        if config.trajectory_selection not in TRAJECTORY_SELECTIONS:
            raise ValueError(
                f"trajectory_selection must be one of {TRAJECTORY_SELECTIONS}, got {config.trajectory_selection!r}"
            )
        if config.trajectory_selection != "argmax" and config.pdm_score_loss_weight <= 0:
            raise ValueError(
                f"trajectory_selection={config.trajectory_selection!r} needs the sub-score heads "
                "(pdm_score_loss_weight > 0)"
            )
        self._vlm_model = VisionLLMLoRAModel(self._trajectory_sampling, config)

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self._checkpoint_path is None:
            return

        # Safely load to CPU RAM first
        checkpoint = torch.load(
            self._checkpoint_path, 
            map_location="cpu", 
            weights_only=True
        )
        state_dict = checkpoint["state_dict"]

        remapped = {}
        for key, value in state_dict.items():
            # 1. Strip top-level wrappers
            key_new = key.replace("agent.", "")
            key_new = key_new.replace("_vlm_model._orig_mod.", "_vlm_model.")
            
            # 2. DEDUPLICATION LOGIC:
            # If PyTorch saved BOTH vision_encoder and model.visual, 
            # force them all into the exact same model.visual namespace.
            if key_new.startswith("_vlm_model.vision_encoder."):
                key_new = key_new.replace("_vlm_model.vision_encoder.", "_vlm_model.model.visual.")

            # 3. Handle PEFT base_model wrapping if transitioning into a LoRA phase
            # (e.g., if you are wrapping the vision model in PEFT for this phase)
            # key_new = key_new.replace("model.visual.", "model.visual.base_model.model.")

            remapped[key_new] = value

        # Explicitly define ALL components that must survive the filter
        # Explicitly define ONLY the custom heads/embeddings that must survive
        CUSTOM_PREFIXES = (
            "_vlm_model.traj_classifier.",
            "_vlm_model.ego_fusion_mlp.",
            "_vlm_model.velocity_encoder.",
            "_vlm_model.acceleration_encoder.",
            "_vlm_model.direction_embedding.",
            "_vlm_model.ego_state_tokens.",
            "_vlm_model.ego_history_tokens.",
            "_vlm_model.ego_history_missing",
            "_vlm_model.planner_queries",
            "_vlm_model.query_pos_embed",
            "_vlm_model.aux_decoder.",
        )

        # Filter out random optimizer states or discarded pre-training heads
        filtered = {
            k: v for k, v in remapped.items()
            if any(k.startswith(p) for p in CUSTOM_PREFIXES) or "lora_" in k
        }

        print(f"Loaded keys: {list(filtered.keys())}")

        # Inject the weights into the empty architecture
        missing_keys, unexpected_keys = self.load_state_dict(filtered, strict=False)
        
        print(f"Loaded {len(filtered)} keys from checkpoint")
        if missing_keys:
            print(f"Missing keys ({len(missing_keys)}):", missing_keys)
        if unexpected_keys:
            print(f"Unexpected keys ({len(unexpected_keys)}):", unexpected_keys)

        # No device move here: training also calls initialize() for the phase handoff, before Lightning sets each
        # rank's GPU, so .cuda() would put every rank's copy on cuda:0. compute_trajectory moves the model for eval.

    def get_sensor_config(self) -> SensorConfig:
        history_steps = [3]  # the current frame; temporal_history_frame k also loads frame 3 - k
        if self._config.temporal_history_frame:
            history_steps = [3 - self._config.temporal_history_frame, 3]
        # The panorama layout uses the side cameras L1/R1 instead of the rear-side L2/R2.
        panorama = self._config.camera_layout == "panorama"
        return SensorConfig(
            cam_f0=history_steps,
            cam_l0=history_steps,
            cam_l1=history_steps if panorama else False,
            cam_l2=False if panorama else history_steps,
            cam_r0=history_steps,
            cam_r1=history_steps if panorama else False,
            cam_r2=False if panorama else history_steps,
            cam_b0=history_steps,
            lidar_pc=False,
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        builders = [VisionLLMLoRATargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]
        if (
            self._config.pdm_score_loss_weight > 0
            or self._config.score_weighted_labels
            or self._config.log_pdm_score_metrics
        ):
            builders.append(PDMScoreTargetBuilder())  # precomputed, read from the training cache
        if aux_output_channels(self._config):
            if not self._config.aux_target_cache_path:
                raise ValueError(
                    "aux_*_loss_weight > 0 needs aux_target_cache_path (the mounted aux_targets_navtrain.sqfs; "
                    "AUX_TARGETS=1 in run_vision_llm_lora_agent_training_slurm.sh)"
                )
            aux = VisionLLMLoRAAuxTargetBuilder(semantic_classes=self._config.bev_semantic_classes)
            aux.cache_root = self._config.aux_target_cache_path  # read by CacheOnlyDataset instead of cache_path
            builders.append(aux)
        if self._config.rl_loss_weight > 0:
            if not (self._config.rl_reference_cache_path and self._config.rl_reference_head_path):
                raise ValueError(
                    "rl_loss_weight > 0 needs rl_reference_cache_path (scripts/training/cache_rl_reference.py) and "
                    "rl_reference_head_path (rl_features/<run>/head.pt)"
                )
            if self._config.rl_objective not in ("exact", "grpo") or self._config.rl_grpo_group_size < 2:
                raise ValueError("rl_objective must be 'exact' or 'grpo', and rl_grpo_group_size >= 2")
            builders.append(RLReferenceTargetBuilder(self._config.rl_reference_cache_path))
        return builders

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        builders = [VisionLLMLoRAFeatureBuilder(config=self._config)]
        root = self._config.history_cache_path
        if self._config.temporal_history_frame:
            builders.append(VisionLLMLoRAHistoryFrameBuilder(self._config, self._config.temporal_history_frame, root))
        if self._config.ego_history_tokens:
            builders.append(VisionLLMLoRAEgoHistoryBuilder(root))
        if self._ec_in_reward():
            # Training only (read from the history cache): the previous frame's camera tiles and ego status, for the
            # previous plan of the EC reward (compute_loss). Not added at evaluation (history_cache_path unset there).
            if self._config.temporal_history_frame or self._config.ego_history_tokens:
                raise ValueError("rl_ec_weight > 0 is not supported with temporal_history_frame / ego_history_tokens")
            builders.append(VisionLLMLoRAHistoryFrameBuilder(self._config, 1, root))
            builders.append(VisionLLMLoRAEgoHistoryBuilder(root))
        return builders

    def _ec_in_reward(self) -> bool:
        return self._config.rl_loss_weight > 0 and self._config.rl_ec_weight > 0 and bool(self._config.history_cache_path)

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
        if self._ec_in_reward():
            previous_shortlist = None
            if self._config.rl_ec_previous_plan == "shortlist":
                previous_shortlist = targets["rl_previous_shortlist"][:, : self._config.rl_ec_previous_shortlist_size]
            elif self._config.rl_ec_previous_plan != "argmax":
                raise ValueError(f"rl_ec_previous_plan must be argmax or shortlist, got {self._config.rl_ec_previous_plan}")
            ec = self._ec_against_previous_plan(features, targets["rl_multiplier"] > 0, previous_shortlist)
            predictions = dict(predictions, rl_ec=ec)
        return vision_llm_lora_loss(targets, predictions, self._config)

    @torch.no_grad()
    def _ec_against_previous_plan(
        self,
        features: Dict[str, torch.Tensor],
        needed: Optional[torch.Tensor] = None,
        previous_shortlist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        EC (B, K) of every codebook entry at t against the current model's plan at t - 0.5 s: a forward pass on the
        previous frame's camera tiles, velocity, acceleration and driving command in eval mode (no dropout, the
        deterministic output an agent would have produced there), then ec_torch on the GPU. The plan is the argmax, or
        with previous_shortlist (B, k) (the scorer's shortlist at t - 0.5 s, best first) the most likely entry among it,
        as the "shortlist" selection picks it. needed (B, K) bool: simulate only these entries (those with M > 0: EC
        does not change the reward of the others, ~72% of all), 0 elsewhere.
        """
        from navsim.agents.vision_llm_lora.ec_torch import TorchExtendedComfort

        model = self._vlm_model
        previous = dict(features)
        previous["camera_feature"] = normalize_uint8_camera(
            features["camera_history_t1"], self._config.image_normalization
        ).to(features["camera_feature"].dtype)
        previous["velocity"] = features["ego_history_velocity"][:, 0]
        previous["acceleration"] = features["ego_history_acceleration"][:, 0]
        previous["driving_command"] = features["ego_history_command"][:, 0]
        was_training = model.training
        model.eval()
        try:
            previous_logits = model(previous, None, 1.0)["trajectory_logits"]
        finally:
            model.train(was_training)
        device = previous_logits.device
        if previous_shortlist is None:
            previous_entry = previous_logits.float().argmax(-1)
        else:  # as select_trajectory "shortlist": argmax of the policy over the shortlist, ties in shortlist order
            short = previous_shortlist.to(device).long()
            previous_entry = short.gather(1, previous_logits.float().gather(1, short).argmax(-1, keepdim=True))[:, 0]
        if getattr(self, "_ec", None) is None or self._ec.device != device:
            import numpy as np

            tracking = dict(np.load(self._config.rl_ec_tracking_path))
            self._ec = TorchExtendedComfort(tracking, device)
        entries = None
        if needed is not None:
            needed = needed.to(device)
            count = int(needed.sum(-1).max().clamp_min(1))
            # per scene the needed entries first (stable: index order), padded with unneeded ones up to the batch max
            entries = torch.sort(needed.to(torch.int8), dim=-1, descending=True, stable=True).indices[:, :count]
        ec = self._ec.against_previous(
            features["velocity"].to(device),
            features["acceleration"].to(device),
            features["ego_history_velocity"][:, 0].to(device),
            features["ego_history_acceleration"][:, 0].to(device),
            previous_entry,
            entries=entries,
        )
        if entries is None:
            return ec.float()
        full = torch.zeros(needed.shape, dtype=torch.float32, device=device)
        full.scatter_(1, entries, ec.float())
        return full * needed

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        total_steps = self._config.total_training_steps * self._config.num_epochs
        total_steps = max(1, total_steps // max(1, self._config.gradient_accumulation_steps))
        warmup_steps = int(self._config.warmup_ratio * total_steps)

        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found. Check LoRA setup and freezing strategy.")

        traj_named   = []
        ego_named    = []
        ego_gain_named = []
        lang_named   = []
        vision_named = []
        aux_named    = []
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

            elif "ego_state_tokens.norm" in name or "ego_history_tokens.norm" in name:
                ego_gain_named.append((name, param))

            elif "aux_decoder" in name:
                aux_named.append((name, param))

            elif (
                "velocity_encoder" in name
                or "acceleration_encoder" in name
                or "direction_embedding" in name
                or "ego_fusion_mlp" in name
                or "ego_state_tokens" in name
                or "ego_history_tokens" in name
                or "ego_history_missing" in name
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
        traj_multiplier = self._config.traj_lr_multiplier
        ego_multiplier = self._config.ego_lr_multiplier
        if self._config.training_phase == 1 and self._config.phase1_lr_multiplier is not None:
            traj_multiplier = ego_multiplier = self._config.phase1_lr_multiplier
        traj_lr = traj_multiplier * self._lr_base
        ego_lr = ego_multiplier * self._lr_base
        ego_gain_lr = self._config.ego_lr_multiplier * self._lr_base
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
        
        _add_group(traj_decay, traj_no_decay, traj_lr, self._config.traj_weight_decay)
        _add_group(ego_decay, ego_no_decay, ego_lr, 0.01)
        _add_group(*self.split_decay_params(ego_gain_named), ego_gain_lr, 0.01)
        _add_group(lang_decay, lang_no_decay, lang_lr, 0.01)
        _add_group(vision_decay, vision_no_decay, vision_lr, 0.01)
        _add_group(*self.split_decay_params(aux_named), self._config.aux_lr_multiplier * self._lr_base, 0.01)
        _add_group(other_decay, other_no_decay, lang_lr, 0.0)


        if not param_groups:
            raise RuntimeError("No trainable parameters found.")

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
            eps=1e-5,
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
                or "magnitude" in name_lower
                or "lora_b" in name_lower
            ):
                no_decay.append(param)
            else:
                decay.append(param)

        return decay, no_decay

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [VisionLLMLoRACallback(self._config)]

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()
        # Evaluation runs without the Lightning trainer; move the weights once, on the first scene.
        if next(self.parameters()).device.type != self.device.type:
            self.to(self.device)
        features: Dict[str, torch.Tensor] = {}

        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        features = {k: (v.unsqueeze(0).to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in features.items()}

        # Autocast in the weights' dtype (config.torch_dtype), as in training; fp16 autocast over bf16 weights can
        # overflow. float32 weights run without autocast.
        dtype = getattr(torch, self._config.torch_dtype)
        autocast_enabled = dtype in (torch.float16, torch.bfloat16)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled):
            predictions = self.forward(features, teacher_forcing_seq=None, teacher_forcing_ratio=1.0)
            trajectory = select_trajectory(
                predictions,
                self._config.trajectory_selection,
                self._vlm_model.codebook,
                shortlist_size=self._config.selection_shortlist_size,
                mixture_lambda=self._config.selection_mixture_lambda,
                local_size=self._config.selection_local_size,
                local_lambda=self._config.selection_local_lambda,
            )
            poses = trajectory.squeeze(0).float().cpu().numpy()

        return Trajectory(poses, self._trajectory_sampling)
