from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer


@dataclass
class VisionLLMLoRAConfig:
    """Global configuration for the Qwen-VL LoRA trajectory prediction model."""

    # ── Model architecture ────────────────────────────────────────────────────
    architecture: str = "Qwen/Qwen3.5-4B"

    # ── Training schedule ─────────────────────────────────────────────────────
    total_training_steps: int = 10639
    #total_training_steps: int = 21278
    #total_training_steps: int = 21
    num_epochs: int = 10
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.05

    # Stage 1 values:  traj=1.0, ego=1.0, lang=0.0, vision=0.0
    # Stage 2 values:  traj=0.4, ego=0.5, lang=0.1, vision=0.05
    traj_lr_multiplier:   float = 1.0
    ego_lr_multiplier:    float = 1.0
    lang_lr_multiplier:   float = 0.0
    vision_lr_multiplier: float = 0.0

    # ── Trajectory codebook ───────────────────────────────────────────────────
    # Path to an (N, 3) NumPy array; each row is one (x, y, yaw) candidate
    # waypoint.  N is typically 16 384.  All N entries are added to the
    # tokeniser as special tokens so the LLM head produces trajectory vocab IDs
    # directly.
    codebook_path: str = "/root/workdir/biroa/navsim/navsim/agents/vision_llm_lora/codebook/16384_8step.npy"

    # Weight applied to the squared circular yaw error during nearest-neighbour
    # codebook lookup:
    #     d² = (Δx)² + (Δy)² + yaw_distance_weight · (Δyaw_wrapped)²
    # Set to 0.0 to ignore yaw when selecting the nearest codebook entry.
    yaw_distance_weight: float = 2.5

    # Temperature for label smoothing when building soft targets from codebook distances.
    label_temperature: float = 1.0

    # Training loss weights
    trajectory_token_loss_weight: float = 1.0    # cross-entropy on trajectory tokens
    trajectory_continuous_loss_weight: float = 0.2  # optional auxiliary regression loss

    # ── Vision encoder ────────────────────────────────────────────────────────
    # Whether to allow gradients through Qwen's built-in visual tower.
    # Keep False to save memory; enable only for full fine-tuning runs.
    train_vision_encoder: bool = False
    train_language_model: bool = False

    # Input resolution fed to the vision encoder
    camera_width: int = 112
    camera_height: int = 112

    # ── Ego-state encoding ────────────────────────────────────────────────────
    # Number of soft ego-context tokens injected into the prompt sequence
    n_ego_tokens: int = 4

    # ── LoRA ─────────────────────────────────────────────────────────────────
    enable_lora: bool = False
    lora_bias: str = "none"
    print_trainable_parameters: bool = True

    # --- LLM LoRA ---
    llm_lora_r: int = 16
    llm_lora_alpha: int = 32          # alpha/r = 2.0
    llm_lora_dropout: float = 0.05
    llm_use_dora: bool = False        # standard LoRA is enough for routing
    llm_lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # --- Vision DoRA/LoRA ---
    # Independent flag: vision lora only applies when train_vision_encoder=True
    enable_vision_lora: bool = False
    vision_lora_r: int = 32           # higher rank: large domain shift
    vision_lora_alpha: int = 32       # alpha/r = 1.0, conservative with DoRA
    vision_lora_dropout: float = 0.05
    vision_use_dora: bool = True      # DoRA for more expressive adaptation
    vision_lora_target_modules: Tuple[str, ...] = (
        "qkv",
        "linear_fc1",
        "linear_fc2",
    )


    head_dropout: float = 0.3

    # ── Quantised loading (BitsAndBytes) ──────────────────────────────────────
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    torch_dtype: str = "float16"

    # ── LiDAR / BEV settings (kept for downstream feature extraction) ─────────
    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32.0
    lidar_max_x: float = 32.0
    lidar_min_y: float = -32.0
    lidar_max_y: float = 32.0
    lidar_split_height: float = 0.2
    use_ground_plane: bool = False
    lidar_seq_len: int = 1

    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256

    # ── BEV semantic segmentation ─────────────────────────────────────────────
    use_bev_semantic: bool = True
    num_bounding_boxes: int = 30

    bev_semantic_classes: dict = field(default_factory=lambda: {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),
        4: ("box", [
            TrackedObjectType.CZONE_SIGN,
            TrackedObjectType.BARRIER,
            TrackedObjectType.TRAFFIC_CONE,
            TrackedObjectType.GENERIC_OBJECT,
        ]),
        5: ("box", [TrackedObjectType.VEHICLE]),
        6: ("box", [TrackedObjectType.PEDESTRIAN]),
    })

    bev_pixel_width: int = 256
    bev_pixel_height: int = 128          # lidar_resolution_height // 2
    bev_pixel_size: float = 0.25
    num_bev_classes: int = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    # ── Loss weights ──────────────────────────────────────────────────────────
    trajectory_weight: float = 10.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 10.0
    lambda_pos: float = 1.0
    lambda_theta: float = 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Derived properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        return max(
            abs(self.lidar_min_x),
            abs(self.lidar_max_x),
            abs(self.lidar_min_y),
            abs(self.lidar_max_y),
        )