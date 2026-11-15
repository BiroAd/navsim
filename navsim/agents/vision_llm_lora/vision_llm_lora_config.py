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
    total_training_steps: int = 2660
    #total_training_steps: int = 10639
    #total_training_steps: int = 21278
    #total_training_steps: int = 21
    num_epochs: int = 20
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1

    # Stage 1 values:  traj=1.0, ego=1.0, lang=0.0, vision=0.0
    # Stage 2 values:  traj=0.4, ego=0.5, lang=0.1, vision=0.05
    traj_lr_multiplier:   float = 0.10
    ego_lr_multiplier:    float = 0.12 # 0.15 was before
    lang_lr_multiplier:   float = 0.08 # 0.2 was before
    vision_lr_multiplier: float = 0.25
    # Phase 1 trains only newly initialized modules (trajectory head, planner queries, ego encoders/tokens), so if set
    # this replaces traj_lr_multiplier and ego_lr_multiplier in phase 1. The ego-token RMSNorm gain keeps
    # ego_lr_multiplier: it starts at the token-embedding RMS (~0.013) and sets the ego tokens' scale in the prompt.
    # None = the multipliers above in every phase.
    phase1_lr_multiplier: Optional[float] = None

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

    # Weight of an extra loss term E_p[ADE_k] = sum_k p_k * ADE(codebook_k, GT): the predicted distribution's expected
    # xy ADE (metres) over all codebook entries, added to the soft cross-entropy. 0 = cross-entropy only.
    expected_distance_loss_weight: float = 0.0

    # Weight of a BCE loss on predicted per-candidate PDM sub-scores (Hydra-MDP-style), against scores precomputed
    # offline for every codebook entry (scripts/evaluation/score_codebook_trajectories.py, cached per token by
    # scripts/training/cache_pdm_score_targets.py). 0 = no sub-score heads and no score targets.
    # Needs head_type="trajectory_embedding".
    pdm_score_loss_weight: float = 0.0

    # Per-head weights of that BCE in PDM_SCORE_KEYS order (no_at_fault_collisions, drivable_area_compliance,
    # driving_direction_compliance, traffic_light_compliance, ego_progress, time_to_collision_within_bound,
    # lane_keeping, history_comfort): the BCE becomes the weighted mean of the per-head BCEs. None = equal weights.
    pdm_score_head_weights: Optional[Tuple[float, ...]] = None

    # Weight of a BCE between the predicted total score (the sub-score heads aggregated like PDMScorer: product of the
    # multiplicative metrics x weighted mean of the others, without extended comfort) and the same aggregate of the
    # targets. Its gradient reaches each head in proportion to how much that sub-score changes the total, and it trains
    # the quantity used to select a trajectory by the heads. Needs pdm_score_loss_weight > 0. 0 = off.
    pdm_total_score_loss_weight: float = 0.0

    # Pairwise ranking loss between near-identical candidates (after FPD-Drive's safety-aware score correction, arXiv
    # 2610.13232): the per-candidate BCE never asks the heads to order two trajectories 0.5 m apart, and the local
    # search around the policy's choice (selection "local") picked less safe neighbours than the policy's own entry
    # (final_evaluation.md, findings 7-8). Pairs: each of the pairwise_score_anchors codebook entries nearest to the ground-truth trajectory
    # with its pairwise_score_neighbours nearest codebook entries (ADE over x, y); a pair counts for a head in
    # pairwise_score_keys when the two targets differ by >= 0.5; loss = mean softplus(margin - (s_better - s_worse)) on
    # the logits. Logs pdm_scores/pair_acc[_<key>]. Needs pdm_score_loss_weight > 0. 0 = off.
    pairwise_score_loss_weight: float = 0.0
    pairwise_score_keys: Tuple[str, ...] = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "driving_direction_compliance",
    )
    pairwise_score_anchors: int = 512
    pairwise_score_neighbours: int = 32
    pairwise_score_margin: float = 0.5

    # How compute_trajectory (evaluation / inference only; training and the loss are unaffected) picks the output:
    # "argmax" = the policy's most likely codebook entry (refined if the refiner is on); "score_heads" = the entry with
    # the highest predicted total score from the sub-score heads (aggregated like pdm_total_score_loss_weight), over all
    # codebook entries, without refinement. The offline evaluation calls this choice head_all. "score_heads_v1": the
    # same heads combined with the NAVSIM v1 PDMS formula (NC · DAC · (5 EP + 5 TTC + 2 HC) / 12), for navtest v1
    # numbers. "shortlist": the policy's most likely entry among the selection_shortlist_size entries with the best
    # predicted total (heads shortlist, policy picks; offline column shortlist_<k>). "mixture": argmax of
    # log p + selection_mixture_lambda · log predicted total (offline column comb_<lambda>). "shortlist_v1" /
    # "mixture_v1": the same with the v1 formula for the predicted total. All but "argmax" need
    # pdm_score_loss_weight > 0.
    trajectory_selection: str = "argmax"
    selection_shortlist_size: int = 64  # "shortlist*" only
    selection_mixture_lambda: float = 10.0  # "mixture*" only
    # Past inputs (cached by scripts/training/cache_history_features.py into a squashfs image; history_cache_path = its
    # mount point, or several mounts of it joined by os.pathsep, read in turn; set by HISTORY_CACHE=1 in the training SLURM script; at evaluation the builders read the raw history
    # frames). temporal_history_frame k > 0: the vision tower's second temporal slot (temporal_patch_size 2, filled with
    # a copy of the current frame otherwise) gets the frame k * 0.5 s earlier, in video order (earlier, current); same
    # token count. ego_history_tokens: 3 more <EGO_STATE> tokens after velocity / acceleration with the past (x, y)
    # positions at t-0.5 / t-1.0 / t-1.5 s (current ego frame, divided by ego_history_position_scales) and, with
    # ego_history_yaw_scale > 0, the past heading (wrapped to (-pi, pi], divided by that scale in rad; 0 = x, y only), replaced in
    # training by learned "missing" embeddings with probability ego_history_dropout per scene (against extrapolating
    # the past path); needs ego_state_tokens. Defaults: off (current behaviour).
    history_cache_path: Optional[str] = None
    temporal_history_frame: int = 0
    ego_history_tokens: bool = False
    ego_history_dropout: float = 0.5
    ego_history_position_scales: Tuple[float, float, float] = (7.5, 15.0, 22.5)
    ego_history_yaw_scale: float = 0.5

    # "local" / "local_v1": discrete test-time search around the policy's most likely entry (select_trajectory): the
    # selection_local_size nearest codebook entries by ADE, objective predicted total - selection_local_lambda * ADE (m).
    selection_local_size: int = 64  # "local*" only
    selection_local_lambda: float = 0.1  # "local*" only

    # Auxiliary perception losses (vision_llm_lora_aux.py; training only): a BEV decoder on the LLM's last-layer
    # hidden states at the image tokens predicts the cached maps of VisionLLMLoRAAuxTargetBuilder (160 x 128 cells of
    # 0.5 m, x -16..64 m, y -32..32 m). Each weight > 0 adds its output head; all 0 = no decoder (current behaviour).
    # BEV semantics: cross-entropy over bev_semantic_classes + none; drivable area (PDM's drivable surface): BCE; lane
    # heading (travel direction, 36 bins): cross-entropy on lane cells.
    aux_bev_semantic_loss_weight: float = 0.0
    aux_drivable_area_loss_weight: float = 0.0
    aux_lane_heading_loss_weight: float = 0.0
    # Root of the cached aux targets, <root>/<log>/<token>/vision_llm_aux_target.gz: the mounted
    # $OPENSCENE_DATA_ROOT/images/aux_targets_navtrain.sqfs (the training SLURM script mounts it with AUX_TARGETS=1).
    aux_target_cache_path: Optional[str] = None
    aux_decoder_dim: int = 256
    aux_decoder_layers: int = 2
    aux_decoder_heads: int = 8
    aux_query_grid: Tuple[int, int] = (20, 16)  # one learned query per 4 x 4 m cell; x 2^3 upsampling -> 160 x 128
    aux_upsample_steps: int = 3
    aux_lr_multiplier: float = 1.0  # LR of the decoder (trained from scratch) = aux_lr_multiplier x lr_base

    # Score-aware soft labels: multiply each codebook entry's soft label exp(-d/T) by its precomputed total PDM score
    # (aggregate of the cached sub-score targets) and renormalize, so entries near GT that fail a multiplicative metric
    # (e.g. leave the drivable area) get no target mass and slower neighbours get less. Scenes where no entry scores
    # above 0 keep the plain soft labels. Loads the PDM score targets like pdm_score_loss_weight > 0. False = off.
    score_weighted_labels: bool = False

    # Weight of an L1 loss on a residual refinement of the selected codebook entry: a small MLP predicts an
    # (x, y, yaw) offset per waypoint from the head's hidden state and the entry, removing the codebook's quantization
    # error (~0.2 m oracle ADE). 0 = no refiner; the output trajectory is the codebook entry itself.
    refinement_loss_weight: float = 0.0

    # Training loss weights
    trajectory_token_loss_weight: float = 1.0    # weight of the soft-label cross-entropy (0 = expected distance only)
    trajectory_continuous_loss_weight: float = 0.2  # optional auxiliary regression loss

    # ── Training phase ────────────────────────────────────────────────────────
    # What trains besides the heads, ego encoders and planner queries; the Qwen base weights are always frozen.
    #   1: nothing else (LLM and vision tower frozen; gradients still flow through the LLM to the planner queries)
    #   2: + LLM LoRA
    #   3: + LLM LoRA + vision-tower LoRA (gradients through the tower)
    training_phase: int = 1
    # Policy-only fine-tuning: freeze everything except the parameters only the policy logits use (scene projection,
    # per-trajectory bias, logit scale; POLICY_HEAD_PARAMETERS in vision_llm_lora_model.py), e.g. to give a
    # heads-only run (p3_aux) a policy without changing its sub-score heads. Use with the run's training_phase so its
    # LoRA modules exist and load; no backward pass runs through the backbone.
    train_policy_head_only: bool = False
    # Load the cached PDM score targets although no score loss or score-aware labels use them, so that a policy-only run
    # logs the same proxy-score metrics (val/pdm_scores/argmax_score[_gap], expected_score) as the score-head runs and
    # can monitor them. The loss is unchanged (score losses still need pdm_score_loss_weight > 0).
    log_pdm_score_metrics: bool = False
    # RL with the backbone trainable (RL-3; vision_llm_lora_rl.py, docs/theoretical.md): adds rl_loss_weight x
    # (-E_pi[r] + rl_beta * KL(pi || pi_start)) over all K entries, rewards = the unfiltered codebook PDM totals. Needs
    # rl_reference_cache_path (per-scene rewards, start-model fused state and scorer shortlist, written by
    # scripts/training/cache_rl_reference.py) and rl_reference_head_path (the start policy head: rl_features/<run>/head.pt).
    # Set the other loss weights to 0 for pure RL. The scorer (sub-score heads) is not trained: evaluation takes its
    # outputs from the start model (VisionLLMLoRAPrecomputedAgent scores_path). 0 = off.
    rl_loss_weight: float = 0.0
    rl_beta: float = 0.02
    rl_reference_cache_path: Optional[str] = None
    rl_reference_head_path: Optional[str] = None
    # Two-frame extended comfort (EC) in the RL-3 reward (vision_llm_lora_agent.compute_loss, ec_torch.py). The
    # previous plan is the current model's argmax at t - 0.5 s (a no-grad forward on the cached past camera frame and
    # past ego status, eval mode); EC of every entry at t against it is simulated on the GPU as the official scorer does.
    # Reward = M * (5 EP + 5 TTC + 2 LK + 2 HC + rl_ec_weight * EC) / (14 + rl_ec_weight) with M the product of the
    # multiplicative sub-scores (rl_multiplier in the RL reference cache), i.e. the official EPDMS for rl_ec_weight = 2.
    # Needs history_cache_path (HISTORY_CACHE=1) and rl_ec_tracking_path (scripts/training/precompute_ec_tracking.py).
    # Not with temporal_history_frame / ego_history_tokens (their previous frame would need older inputs). 0 = off.
    rl_ec_weight: float = 0.0
    rl_ec_tracking_path: Optional[str] = None
    # Previous plan of the EC reward: "argmax" (the model's argmax at t - 0.5 s, as above) or "shortlist" (the model's
    # most likely entry among the first rl_ec_previous_shortlist_size entries of the scorer's shortlist at t - 0.5 s,
    # i.e. what the "shortlist" selection outputs there; rl_previous_shortlist in the RL reference cache, written by
    # cache_rl_reference.py --previous-shortlists from scripts/training/cache_scorer_shortlists.py).
    rl_ec_previous_plan: str = "argmax"
    rl_ec_previous_shortlist_size: int = 256

    # Input resolution fed to the vision encoder: square tiles, the stitched panorama is 6x2 tiles plus a
    # thumbnail. 128 avoids padding every tile to the 32-px merge grid (docs/vision_llm_lora_findings.md F10).
    camera_width: int = 112
    camera_height: int = 112

    # Cameras on the 6x2-tile canvas (findings F11). "original": central 896x448 crops of L0|F0|R0 over L2|B0|R2,
    # 194 deg of 360 visible. "panorama": TransFuser-style front L0|F0|R0 (full F0 width, central 1088 px of L0/R0)
    # over full-width R1|B0|L1, 332 deg visible, each row cropped vertically to 6:1. Same number of tokens.
    camera_layout: str = "original"
    # 13th image: "canvas" (original: the whole canvas squeezed into one tile), "front_zoom" (a 14 x 14 deg crop of F0
    # around the road ahead at ~9 px/deg) or "none" (12 images, 192 tokens).
    thumbnail: str = "canvas"
    # True reproduces the original input, in which red and blue are swapped (findings F9).
    swap_red_blue: bool = True
    # "imagenet" (original) or "qwen": mean = std = 0.5, as in Qwen3.5's preprocessor_config.json.
    image_normalization: str = "imagenet"
    # Patch order fed to the visual tower: "raster" (original) or "merge_window", where each 2x2 merge window
    # is contiguous, as the Qwen2-VL image processor emits it and the tower's merger expects (findings F13).
    patch_order: str = "raster"
    # How the 12 canvas tiles reach the vision tower (model-side; the cached tiles are exact crops of the canvas).
    # "strips": reassembled into the two canvas rows, one 6:1 image each, so the tower attends across tile seams and
    # positions run along the whole row; "tiles": 12 separate square images (runs before 2026-10-26). A 13th image
    # stays separate. Same pixels and token count (2 × 96 + 16 = 208 at 128-px tiles).
    image_layout: str = "strips"
    # Image tokens in the prompt: "custom" (original: newly added <img>/</img> around all images and space-separated
    # <IMG_CONTEXT> placeholders, 1-D positions, findings F15, F16, F14) or "qwen", Qwen3.5's native format:
    # pretrained <|vision_start|><|image_pad|>...<|vision_end|> around each image, no spaces and no new tokens,
    # multimodal RoPE positions (t, h, w) for the image tokens, and the empty think block before the planner queries.
    image_tokens: str = "custom"

    # ── Ego-state encoding ────────────────────────────────────────────────────
    # Number of soft ego-context tokens injected into the prompt sequence
    n_ego_tokens: int = 4
    # Late fusion: velocity/acceleration/command encoders → EgoFusionMLP → concatenated with the planner queries in the
    # head. False leaves those modules out and the head sees only the planner queries, so ego state reaches the head
    # only through the LLM (command_in_prompt / ego_state_tokens; with both off the model is blind to ego state).
    # The head's input size changes, so checkpoints don't transfer between the two settings.
    late_ego_fusion: bool = True
    # Ego state inside the LLM sequence (findings F3).
    # command_in_prompt writes the driving command as text after the images ("Navigation command: go left.").
    command_in_prompt: bool = False
    # ego_state_tokens adds one soft token each for velocity and acceleration ("Current velocity:<tok>, ...").
    ego_state_tokens: bool = False
    # Scales (m/s, m/s²) dividing the (x, y) velocity and acceleration before the sin/cos features, about the 99th
    # percentile of the magnitudes (250 trainval logs, 120,768 frames: speed p99 13.8 m/s, max 17.0; ax, ay within
    # ±1.6 m/s² at p1/p99, extremes ±5.4).
    ego_velocity_scale: float = 15.0
    ego_acceleration_scale: float = 2.0
    # n sin/cos frequencies pi * 2^k, k = 0 … n-1; the finest period is 2 / 2^(n-1) scaled units
    # (n = 4: 0.25, i.e. 3.75 m/s of velocity).
    ego_fourier_frequencies: int = 4

    # ── LoRA (applied from training_phase 2 on) ──────────────────────────────
    lora_bias: str = "none"
    print_trainable_parameters: bool = True

    # --- LLM LoRA ---
    llm_lora_r: int = 32
    llm_lora_alpha: int = 64          # alpha/r = 2.0
    llm_lora_dropout: float = 0.10
    llm_use_dora: bool = False       # standard LoRA is enough for routing
    # q/k/v/o_proj exist only in Qwen3.5's 8 full-attention layers. The 24 Gated DeltaNet layers mix tokens through
    # in_proj_qkv, in_proj_z, in_proj_a, in_proj_b and out_proj, which these defaults leave frozen (findings F18).
    llm_lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # --- Vision DoRA/LoRA (training_phase 3) ---
    vision_lora_r: int = 64           # higher rank: large domain shift
    vision_lora_alpha: int = 128       # alpha/r = 1.0, conservative with DoRA
    vision_lora_dropout: float = 0.10
    vision_use_dora: bool = False      # DoRA for more expressive adaptation
    vision_lora_target_modules: Tuple[str, ...] = (
        "attn.qkv",
        "attn.proj",   # Safely restores the 3.1M attention output parameters!
        "linear_fc1",  # Double check that this is spelled correctly in your code!
        "linear_fc2",
    )


    head_dropout: float = 0.20
    # Width of the trajectory head's hidden layer (planner queries -> hidden -> codebook logits). Most head parameters
    # sit in hidden x codebook_size, so a narrower layer regularizes the head. Changes the head's shapes.
    head_hidden_dim: int = 2048
    # Last layer of the head: "linear" (one weight row per codebook entry) or "trajectory_embedding" (dot product with
    # a shared MLP embedding of each trajectory, TrajectoryEmbeddingScorer). Changes the head's parameters.
    head_type: str = "linear"
    trajectory_embed_dim: int = 512  # scene / trajectory embedding size for head_type="trajectory_embedding"
    # Weight decay of the trajectory group (head, planner queries); norms/biases keep 0.
    traj_weight_decay: float = 0.01
    # Learned queries appended after the prompt; the head reads their last hidden states. Changes the head's input size.
    num_planner_queries: int = 4

    # ── Quantised loading (BitsAndBytes) ──────────────────────────────────────
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    torch_dtype: str = "float16"
    # Passed to from_pretrained when set, e.g. "sdpa" or "flash_attention_2"; None keeps the transformers default.
    # It covers Qwen3.5's 8 full-attention layers and the vision tower; the 24 linear-attention layers use the fla
    # (and causal_conv1d, if importable) kernels when installed, else a slow pure-torch path.
    attn_implementation: Optional[str] = None
    # Gradient checkpointing on the LoRA-wrapped LLM (and vision tower when trained): less memory, ~20-30% slower.
    gradient_checkpointing: bool = True

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