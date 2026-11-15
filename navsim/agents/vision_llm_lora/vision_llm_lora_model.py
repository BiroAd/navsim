"""
VisionLLMLoRAModel — Qwen-VL backbone with single-token trajectory prediction.

Architecture overview
─────────────────────
  1. Images  → Qwen built-in vision encoder → vision_proj → LLM hidden dim
  2. Ego     → velocity / acceleration / direction encoders
                 → ego_fusion_mlp → LLM hidden dim
  3. Prompt  → tokenizer; ego & image embeddings are injected at placeholder
               positions; prompt ends with <TRAJ_START>
  4. LLM     → Qwen3.5-4B (optionally LoRA-adapted)
  5. Output  → ONE token from the 16 384-entry trajectory codebook

One token = one complete trajectory
─────────────────────────────────────
  Each codebook entry encodes a full 8-step (x, y, yaw) trajectory template.
  The model does NOT generate tokens autoregressively; it predicts a single
  classification index over the K = 16 384 codebook entries by reading the
  logits at the <TRAJ_START> position.

Codebook file  (see codebook/build_codebook.py for the offline step)
──────────────────────────────────────────────────────────────────────
  Expected shape after offline pre-processing: (K, 8, 3)
  — 16 384 trajectory templates, each with 8 waypoints of (x, y, yaw).

  The offline step takes the raw (K, 40, 3) array (5 s @ 8 Hz) and selects
  every 5th frame: indices [0, 5, 10, 15, 20, 25, 30, 35] → 8 waypoints.

  The model also accepts the raw (K, 40, 3) shape at load time and applies
  the same subsampling automatically, but the offline approach is preferred.
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from transformers import AutoModel, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None  # type: ignore[assignment]

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None                       # type: ignore[assignment]
    TaskType = None                         # type: ignore[assignment]
    get_peft_model = None                   # type: ignore[assignment]
    prepare_model_for_kbit_training = None  # type: ignore[assignment]

from navsim.agents.vision_llm_lora.vision_llm_lora_aux import AuxBEVDecoder, aux_output_channels
from navsim.agents.vision_llm_lora.vision_llm_lora_codebook import load_codebook
from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_features import PDM_SCORE_KEYS, normalize_uint8_camera


# ─────────────────────────────────────────────────────────────────────────────
# Special-token name constants
# ─────────────────────────────────────────────────────────────────────────────

_IMG_CTX_TOKEN = "<IMG_CONTEXT>"
_IMG_OPEN_TOKEN = "<img>"
_IMG_CLOSE_TOKEN = "</img>"

# Qwen3.5's pretrained vision tokens (config.image_tokens == "qwen")
_QWEN_VISION_START_TOKEN = "<|vision_start|>"
_QWEN_VISION_END_TOKEN = "<|vision_end|>"
_QWEN_IMAGE_PAD_TOKEN = "<|image_pad|>"
# Qwen3.5's chat template opens a non-thinking assistant turn with an empty think block; the planner queries follow it
_QWEN_EMPTY_THINK = "<think>\n\n</think>\n\n"

# Placeholder replaced by the ego-state tokens (config.ego_state_tokens)
_EGO_STATE_TOKEN = "<EGO_STATE>"

# Driving command text by one-hot index (config.command_in_prompt). Order checked against the logs: over the next 4 s
# the heading changes by +20° on average for index 0, +0.2° for 1 and -22° for 2; 3 is NAVSIM's "unknown".
# All four phrasings tokenize to the same length, so a batch with mixed commands needs no padding.
_COMMAND_TEXT = ("go left", "go straight", "go right", "not given")

# Parameters only the policy logits use (TrajectoryEmbeddingScorer.forward and TrajectoryClassifier.logit_scale);
# trainable alone with config.train_policy_head_only. The trajectory embedding, query fusion and planner queries are
# shared with the sub-score heads and stay frozen.
POLICY_HEAD_PARAMETERS = (
    "traj_classifier.classifier.scene_proj.",
    "traj_classifier.classifier.trajectory_bias",
    "traj_classifier.logit_scale",
)

# ─────────────────────────────────────────────────────────────────────────────
# Trajectory classifier
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryEmbeddingScorer(nn.Module):
    """
    Scores every codebook trajectory as a dot product between a scene vector and a shared embedding of the trajectory
    (config.head_type == "trajectory_embedding"), instead of one independent weight row per trajectory. Similar
    trajectories get similar embeddings and hence similar logits, and the parameter count no longer grows with K.

    Each trajectory's waypoints (normalized per coordinate over the codebook) go through Fourier features at
    pi * 2^k (k < n_frequencies) before the MLP, so it can still separate nearby trajectories (an MLP on raw
    coordinates is biased towards very smooth functions). The trajectory embeddings are RMS-normalized without a gain,
    so every candidate has the same norm and none can score high for every scene by growing its embedding; the scene
    vector is not normalized, so its norm can act as a per-sample confidence. A per-trajectory bias keeps a learnable
    prior (K params).

    With num_score_heads > 0 it also predicts per-candidate sub-scores (e.g. PDM metrics) from the same trajectory
    embeddings, with one scene projection and per-candidate bias per head (score_logits).

    forward(fused: (B, in_dim)) -> logits (B, K)
    score_logits(fused: (B, in_dim)) -> (B, num_score_heads, K)
    """

    def __init__(
        self,
        codebook: torch.Tensor,
        in_dim: int,
        embed_dim: int = 512,
        mlp_dim: int = 256,
        n_frequencies: int = 6,
        num_score_heads: int = 0,
    ):
        super().__init__()
        waypoints = codebook.detach().float()  # (K, T, 3)
        mean = waypoints.mean(dim=(0, 1))
        std = waypoints.std(dim=(0, 1)).clamp_min(1e-3)
        self.register_buffer(
            "trajectories", ((waypoints - mean) / std).flatten(1), persistent=False
        )  # (K, T * 3)
        self.register_buffer(
            "freqs", torch.pi * 2.0 ** torch.arange(n_frequencies, dtype=torch.float32), persistent=False
        )
        traj_features = self.trajectories.shape[1] * (1 + 2 * n_frequencies)
        self.trajectory_mlp = nn.Sequential(
            nn.Linear(traj_features, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.RMSNorm(embed_dim, elementwise_affine=False),
        )
        self.scene_proj = nn.Linear(in_dim, embed_dim)
        self.trajectory_bias = nn.Parameter(torch.zeros(codebook.shape[0]))
        self.scale = embed_dim ** -0.5
        self.num_score_heads = num_score_heads
        if num_score_heads > 0:
            self.score_proj = nn.Linear(in_dim, embed_dim * num_score_heads)
            self.score_bias = nn.Parameter(torch.zeros(num_score_heads, codebook.shape[0]))

    def embed_trajectories(self) -> torch.Tensor:
        x = self.trajectories  # (K, T * 3)
        angles = x[..., None] * self.freqs  # (K, T * 3, F)
        features = torch.cat([x, angles.sin().flatten(1), angles.cos().flatten(1)], dim=-1)
        return self.trajectory_mlp(features)  # (K, D)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        scene = self.scene_proj(fused)  # (B, D)
        trajectories = self.embed_trajectories().to(scene.dtype)  # (K, D)
        return scene @ trajectories.T * self.scale + self.trajectory_bias.to(scene.dtype)

    def score_logits(self, fused: torch.Tensor) -> torch.Tensor:
        scene = self.score_proj(fused).view(fused.shape[0], self.num_score_heads, -1)  # (B, M, D)
        trajectories = self.embed_trajectories().to(scene.dtype)  # (K, D)
        return scene @ trajectories.T * self.scale + self.score_bias.to(scene.dtype)  # (B, M, K)


class TrajectoryRefiner(nn.Module):
    """
    Residual refinement of a selected codebook entry (config.refinement_loss_weight > 0): an MLP maps the head's hidden
    state and the entry's normalized waypoints to an (x, y, yaw) offset per waypoint, in units of the codebook's
    per-coordinate std. The last layer starts at zero, so training starts from the unrefined entry.

    forward(fused: (B, in_dim), anchor: (B, T, 3)) -> offset (B, T, 3), same units as anchor
    """

    def __init__(self, codebook: torch.Tensor, in_dim: int, hidden_dim: int = 512):
        super().__init__()
        waypoints = codebook.detach().float()
        self.register_buffer("mean", waypoints.mean(dim=(0, 1)), persistent=False)
        self.register_buffer("std", waypoints.std(dim=(0, 1)).clamp_min(1e-3), persistent=False)
        num_values = waypoints.shape[1] * 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + num_values, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_values),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, fused: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        normalized = ((anchor.float() - self.mean) / self.std).flatten(1).to(fused.dtype)
        offset = self.mlp(torch.cat([fused, normalized], dim=-1)).view(anchor.shape)
        return offset.float() * self.std


class TrajectoryClassifier(nn.Module):
    """
    Planner-query classifier, with the ego vector late-fused when late_ego_fusion is set. The last layer is a linear
    map to one logit per codebook entry, or a TrajectoryEmbeddingScorer when trajectory_codebook is given.
    Optional parts sit here so they share the head's optimizer group and phase-handoff keys: per-candidate sub-score
    heads (num_score_heads, embedding head only) and a TrajectoryRefiner (refiner_codebook).

    forward -> (logits (B, K), query similarity, query variance, relative variance, fused hidden state (B, H))
    """

    def __init__(
        self,
        llm_hidden_size: int,
        num_trajectories: int,
        num_queries: int = 4,
        bottleneck_dim: int = 1024,
        dropout: float = 0.1,
        late_ego_fusion: bool = True,
        trajectory_codebook: Optional[torch.Tensor] = None,
        trajectory_embed_dim: int = 512,
        num_score_heads: int = 0,
        refiner_codebook: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if num_score_heads > 0 and trajectory_codebook is None:
            raise ValueError("Sub-score heads need the trajectory-embedding head (head_type='trajectory_embedding').")

        self.num_queries = num_queries
        self.late_ego_fusion = late_ego_fusion

        fusion_input_dim = (
            llm_hidden_size * num_queries
            + (llm_hidden_size if late_ego_fusion else 0)
        )

        self.query_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, bottleneck_dim),
            nn.RMSNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if trajectory_codebook is not None:
            self.classifier = TrajectoryEmbeddingScorer(
                trajectory_codebook, bottleneck_dim, embed_dim=trajectory_embed_dim, num_score_heads=num_score_heads
            )
        else:
            self.classifier = nn.Linear(
                bottleneck_dim,
                num_trajectories,
            )
        self.refiner = TrajectoryRefiner(refiner_codebook, bottleneck_dim) if refiner_codebook is not None else None

        self.logit_scale = nn.Parameter(
            torch.tensor(0.0)
        )

    def forward(
        self,
        outputs,
        ego_repr: Optional[torch.Tensor],
    ) -> torch.Tensor:

        query_embeddings = outputs.last_hidden_state[
            :, -self.num_queries:, :
        ]

        planner_repr = query_embeddings.reshape(
            query_embeddings.shape[0],
            -1,
        )

        # similarity
        q = query_embeddings.mean(dim=0).detach()

        sim = F.cosine_similarity(
            q[:, None, :],
            q[None, :, :],
            dim=-1,
            eps=1e-5,
        )

        # Measure variance across batch — should be high if scene-sensitive. A batch of one sample (e.g. the last val
        # batch on a rank) has no spread: correction=0 gives 0 there instead of NaN, which would poison the epoch mean.
        correction = 1 if query_embeddings.shape[0] > 1 else 0
        query_variance = query_embeddings.var(dim=0, correction=correction).mean().detach()

        # Variance scale
        with torch.no_grad():
            hidden_scale = query_embeddings.detach().pow(2).mean().sqrt()  # RMS magnitude
            relative_variance = query_embeddings.var(dim=0, correction=correction).mean() / (hidden_scale ** 2)

        # Return only the mean of off-diagonal elements as a scalar
        mask = ~torch.eye(self.num_queries, dtype=torch.bool, device=q.device)
        sim_scalar = sim[mask].mean()

        if self.late_ego_fusion:
            fused = torch.cat(
                [planner_repr, ego_repr],
                dim=-1,
            )
        else:
            fused = planner_repr

        fused = self.query_fusion(fused)

        logits = self.classifier(fused)
        logits = torch.tanh(logits / 30.0) * 30.0
        scale = self.logit_scale.clamp(-2.0, 2.0).exp()
        logits = logits * scale

        return logits, sim_scalar, query_variance, relative_variance, fused


# Ego Fusion MLP


class EgoFusionMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        bottleneck_dim: int,
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(bottleneck_dim, out_dim),

            # IMPORTANT:
            # stabilize ego representation scale
            nn.RMSNorm(out_dim),
        )

        # learned scaling
        self.scale = nn.Parameter(
            torch.tensor(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.scale.clamp(min=0.0, max=1.0)


# Ego-state tokens


def ego_history_inputs(pose: torch.Tensor, use_yaw: bool) -> torch.Tensor:
    """(B, 3, 3) past poses (x, y, yaw; current ego frame) -> the history tokens' inputs: (x, y, yaw wrapped to
    (-pi, pi]) with use_yaw, else (x, y)."""
    pose = pose.float()
    if not use_yaw:
        return pose[..., :2]
    yaw = torch.atan2(torch.sin(pose[..., 2:]), torch.cos(pose[..., 2:]))
    return torch.cat([pose[..., :2], yaw], dim=-1)


class EgoStateTokens(nn.Module):
    """
    One LLM input token per (x, y) ego quantity (velocity, acceleration).

    Each quantity is divided by a fixed physical scale, so typical values lie in about [-1, 1], and expanded with
    sin/cos features at frequencies pi * 2^k, which let a small MLP separate nearby values (e.g. 5 vs 6 m/s). The MLP
    output is RMS-normalized with a per-dim gain that starts at the RMS of Qwen's token embeddings, so the tokens enter
    the sequence at the same scale as the text around them.
    """

    def __init__(
        self,
        hidden_size: int,
        scales: Sequence,
        n_frequencies: int,
        embed_rms: float,
        mlp_dim: int = 256,
        n_components: int = 2,
    ):
        """
        :param scales: one scale per quantity (Q,), applied to all its components, or per component (Q, n_components)
            (e.g. the history tokens' (x, y, yaw) with metres for x, y and radians for yaw)
        :param n_components: values per quantity (2: x, y; 3: x, y, yaw)
        """
        super().__init__()

        scales = torch.tensor(scales, dtype=torch.float32)
        if scales.dim() == 1:
            scales = scales[:, None].expand(-1, n_components)
        if scales.shape[1] != n_components:
            raise ValueError(f"scales {tuple(scales.shape)} do not match n_components={n_components}")
        self.register_buffer("scales", scales.contiguous(), persistent=False)  # (Q, n_components)
        self.register_buffer(
            "freqs", torch.pi * 2.0 ** torch.arange(n_frequencies, dtype=torch.float32), persistent=False
        )

        in_dim = n_components * (1 + 2 * n_frequencies)  # components × [value, sin, cos]

        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, hidden_size),
            )
            for _ in scales
        ])

        self.norm = nn.RMSNorm(hidden_size)
        nn.init.constant_(self.norm.weight, embed_rms)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        :param states: (B, Q, n_components) raw ego quantities in ego frame, Q = len(scales)
        :return: (B, Q, hidden_size) token embeddings
        """
        x = states.float() / self.scales
        angles = x[..., None] * self.freqs                                     # (B, Q, 2, F)
        feats = torch.cat([x, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)

        tokens = torch.stack([mlp(feats[:, i]) for i, mlp in enumerate(self.mlps)], dim=1)
        return self.norm(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class VisionLLMLoRAModel(nn.Module):
    """
    Qwen-VL backbone predicting a single trajectory codebook token.

    The model classifies over K = 16 384 trajectory templates; each template
    is a complete 8-step (x, y, yaw) trajectory in the ego frame.
    One forward pass → one token → one full trajectory.
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        config: VisionLLMLoRAConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_waypoints: int = trajectory_sampling.num_poses  # 8

        # M-RoPE (image_tokens="qwen") gives image tokens 2-D grid positions, which only match the merge-window order.
        if config.image_tokens == "qwen" and config.patch_order != "merge_window":
            raise ValueError("image_tokens='qwen' requires patch_order='merge_window' (findings F13, F14).")

        # 1: heads only, 2: + LLM LoRA, 3: + vision LoRA (see VisionLLMLoRAConfig.training_phase)
        if config.training_phase not in (1, 2, 3):
            raise ValueError(f"training_phase must be 1, 2 or 3, got {config.training_phase!r}.")

        self._prompt_cache: dict = {}  # cache tokenized prompts by (n_img_tokens, n_ego_tokens) for efficiency

        # ── 1. Codebook ─────────────────────────────────────────────────────
        self._init_codebook(config.codebook_path)
 
        # ── 2. Backbone ──────────────────────────────────────────────────────
        # AutoModel loads Qwen3_5ForConditionalGeneration,
        # which exposes model.visual and model.language_model.
        self.model = self._build_llm(config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.architecture, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── 3. Special tokens (ego, image, K trajectory tokens) ─────────────
        self._register_special_tokens()

        # ── 4. Freeze the whole Qwen backbone (vision tower + LLM) ─────────
        # In every phase; LoRA adapters added below are the only trainable
        # backbone parameters.
        for p in self.model.parameters():
            p.requires_grad_(False)

        # ── 5. LoRA: LLM from phase 2, vision tower in phase 3 ─────────────
        if config.training_phase >= 2:
            self._attach_lora(config)

        # ── 6. Cached references ─────────────────────────────────────────────
        self.embedding = self.model.language_model.get_input_embeddings()
        llm_hidden: int = self._hidden_size(self.model)

        # Derive patch constants from the Conv3d kernel so nothing is hardcoded.
        conv = self.model.visual.patch_embed.proj          # Conv3d
        self._patch_size     = conv.kernel_size[1]           # 16
        self._temporal_size  = conv.kernel_size[0]           # 2  (frames per temporal patch)
        self._patch_dim      = (
            conv.in_channels
            * self._temporal_size
            * self._patch_size
            * self._patch_size
        )  # 3 * 2 * 16 * 16 = 1536
        self._merger_factor = (
            self.model.visual.merger.linear_fc1.in_features
            // self.model.visual.merger.norm.normalized_shape[0]
        )
        self._merge_size = int(round(self._merger_factor ** 0.5))  # 2 (2x2 spatial merge)

        # ── 7. Late-fusion ego encoders (config.late_ego_fusion) ─────────────
        # Not created when late_ego_fusion is off: the head then sees only the planner queries.
        if config.late_ego_fusion:
            self.velocity_encoder = nn.Sequential(
                nn.Linear(2, 256),
                nn.GELU(),
                nn.Linear(256, llm_hidden),
                nn.RMSNorm(llm_hidden),
            )

            self.acceleration_encoder = nn.Sequential(
                nn.Linear(2, 256),
                nn.GELU(),
                nn.Linear(256, llm_hidden),
                nn.RMSNorm(llm_hidden),
            )

            self.direction_embedding = nn.Embedding(
                4,
                llm_hidden,
            )

            # Fuse [velocity | acceleration | direction] → one vector for the head
            bottleneck_dim = 256
            out_dim = llm_hidden

            self.ego_fusion_mlp = EgoFusionMLP(
                in_dim=3 * llm_hidden,
                bottleneck_dim=bottleneck_dim,
                out_dim=out_dim,
                dropout=getattr(config, "head_dropout", 0.0),
            )

        # ── 8. Ego-state tokens in the LLM sequence ──────────────────────────
        # Velocity and acceleration as one token each (independent of late_ego_fusion).
        if config.ego_state_tokens:
            with torch.no_grad():
                embed_rms = self.embedding.weight.pow(2).mean(dtype=torch.float32).sqrt().item()

            self.ego_state_tokens = EgoStateTokens(
                hidden_size=llm_hidden,
                scales=(config.ego_velocity_scale, config.ego_acceleration_scale),
                n_frequencies=config.ego_fourier_frequencies,
                embed_rms=embed_rms,
            )
        # Past (x, y) positions at t-0.5 / t-1.0 / t-1.5 s as 3 more ego tokens (config.ego_history_tokens), with
        # learned "missing" embeddings that replace them in training with probability config.ego_history_dropout.
        if config.ego_history_tokens:
            if not config.ego_state_tokens:
                raise ValueError("ego_history_tokens needs ego_state_tokens (it reuses the <EGO_STATE> placeholder).")
            yaw = config.ego_history_yaw_scale > 0
            self.ego_history_tokens = EgoStateTokens(
                hidden_size=llm_hidden,
                scales=[(s, s, config.ego_history_yaw_scale) if yaw else (s, s)
                        for s in config.ego_history_position_scales],
                n_frequencies=config.ego_fourier_frequencies,
                embed_rms=embed_rms,
                n_components=3 if yaw else 2,
            )
            self.ego_history_missing = nn.Parameter(torch.randn(len(config.ego_history_position_scales), llm_hidden) * embed_rms)

        # ── 9. Prompt template ────────────────────────────────────────────────
        # Uses Qwen3.5's ChatML template structure. <EGO> is empty unless
        # command_in_prompt or ego_state_tokens is set.
        self._prompt_template = (
            "<|im_start|>system\n"
            "You are an expert autonomous driving assistant."
            "<|im_end|>\n"

            "<|im_start|>user\n"

            "Visual context:\n"
            "<IMG>\n"
            "<EGO>"

            "Predict the safest trajectory token.\n"

            "<|im_end|>\n"

            "<|im_start|>assistant\n"
        )

        # ── 10. Trajectory classifier head (for direct token prediction) ─────────
        self.num_planner_queries = config.num_planner_queries
        if config.head_type not in ("linear", "trajectory_embedding"):
            raise ValueError(f"Unknown head_type: {config.head_type!r}")
        self.traj_classifier = TrajectoryClassifier(
            llm_hidden_size=llm_hidden,
            num_trajectories=self.codebook_size,
            bottleneck_dim=config.head_hidden_dim,
            dropout=getattr(config, "head_dropout", 0.0),
            num_queries=self.num_planner_queries,
            late_ego_fusion=config.late_ego_fusion,
            trajectory_codebook=self.codebook if config.head_type == "trajectory_embedding" else None,
            trajectory_embed_dim=config.trajectory_embed_dim,
            num_score_heads=len(PDM_SCORE_KEYS) if config.pdm_score_loss_weight > 0 else 0,
            refiner_codebook=self.codebook if config.refinement_loss_weight > 0 else None,
        )

        # ── 11. Planner queries ─────────
        init_embeds = torch.zeros(self.num_planner_queries, llm_hidden)
        nn.init.orthogonal_(init_embeds)
        # Unit-norm rows × the embedding std: norm ~0.013, ~50× below a token
        # embedding (~0.66). The sum below is dominated by query_pos_embed
        # (norm ~1), so the queries effectively start random at ~1.5× token scale.
        with torch.no_grad():
            emb_std = self.embedding.weight.std().item()
        init_embeds = init_embeds * emb_std

        self.planner_queries = nn.Parameter(init_embeds.float())

        self.query_pos_embed = nn.Parameter(
            torch.randn(
                self.num_planner_queries,
                llm_hidden
            ) * 0.02
        )

        # ── 12. Auxiliary BEV decoder (training only; config.aux_*_loss_weight) ─────────
        # Built last, so runs without it draw the same random initialisation as before.
        aux_outputs = aux_output_channels(config)
        self.aux_decoder = (
            AuxBEVDecoder(
                in_dim=llm_hidden,
                outputs=aux_outputs,
                dim=config.aux_decoder_dim,
                layers=config.aux_decoder_layers,
                heads=config.aux_decoder_heads,
                query_grid=config.aux_query_grid,
                upsample_steps=config.aux_upsample_steps,
            )
            if aux_outputs
            else None
        )

        # ── 13. Policy-only fine-tuning (config.train_policy_head_only) ─────────
        # Everything is frozen except the parameters only the policy logits use; the sub-score heads, the shared
        # trajectory embedding, the planner queries and the LoRA stay as loaded (e.g. from a heads-only run).
        if config.train_policy_head_only:
            self._freeze_all_but_policy_head(config)

        print("--- TRAINABLE PARAMETERS ---")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.numel():,} parameters")
        print("----------------------------")

    # ─────────────────────────────────────────────────────────────────────────
    # Codebook initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_codebook(self, codebook_path: str) -> None:
        """
        Load and register the trajectory codebook buffer.

        Input shapes handled
        ────────────────────
        (K, 8, 3)   — pre-processed, used as-is.  Preferred.
        (K, 40, 3)  — raw navsim 4 s @ 10 Hz (pose i at (i + 1) * 0.1 s); subsampled to
                       indices [4, 9, …, 39] = 0.5 … 4.0 s by ``load_codebook``.

        Subsampling offline is recommended (see codebook/codebook_builder.py).
        ``self.codebook`` is always (K, num_waypoints, 3) after this call.
        """
        arr = load_codebook(codebook_path, self.num_waypoints)

        self.codebook_size: int = arr.shape[0]
        self.register_buffer("codebook", torch.from_numpy(arr))  # (K, 8, 3)

    # ─────────────────────────────────────────────────────────────────────────
    # Special-token registration
    # ─────────────────────────────────────────────────────────────────────────

    def _register_special_tokens(self) -> None:

        if self.config.image_tokens == "qwen":
            # Pretrained tokens: nothing is added and the embedding matrix stays as loaded.
            self.img_ctx_id = self.tokenizer.convert_tokens_to_ids(_QWEN_IMAGE_PAD_TOKEN)
        elif self.config.image_tokens == "custom":
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": [
                    _IMG_OPEN_TOKEN,
                    _IMG_CLOSE_TOKEN,
                    _IMG_CTX_TOKEN,
                ]
            })

            self.model.resize_token_embeddings(
                len(self.tokenizer)
            )

            self.img_ctx_id = self.tokenizer.convert_tokens_to_ids(
                _IMG_CTX_TOKEN
            )
        else:
            raise ValueError(f"Unknown image_tokens: {self.config.image_tokens!r}")

        if self.config.ego_state_tokens:
            # Placeholder only: its embedding is replaced by the ego tokens, so the row's value never matters. Qwen's
            # matrix has spare rows past the tokenizer, so a resize is only needed when they are used up.
            self.tokenizer.add_tokens([_EGO_STATE_TOKEN], special_tokens=True)
            self.ego_state_id = self.tokenizer.convert_tokens_to_ids(_EGO_STATE_TOKEN)
            if self.ego_state_id >= self.model.get_input_embeddings().num_embeddings:
                self.model.resize_token_embeddings(len(self.tokenizer))

    # ─────────────────────────────────────────────────────────────────────────
    # LLM / LoRA construction
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_dtype(self, name: str) -> torch.dtype:
        if not hasattr(torch, name):
            raise ValueError(f"Unknown torch dtype: {name!r}")
        return getattr(torch, name)

    def _build_llm(self, config: VisionLLMLoRAConfig) -> nn.Module:
        kwargs: Dict = {"trust_remote_code": True}

        if config.load_in_4bit or config.load_in_8bit:
            if BitsAndBytesConfig is None:
                raise ImportError("Update transformers to get BitsAndBytesConfig.")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=config.load_in_4bit,
                load_in_8bit=config.load_in_8bit,
                bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=self._resolve_dtype(config.bnb_4bit_compute_dtype),
            )
            if torch.cuda.is_available():
                kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = self._resolve_dtype(config.torch_dtype)
        if config.attn_implementation is not None:
            kwargs["attn_implementation"] = config.attn_implementation
        return AutoModel.from_pretrained(config.architecture, **kwargs)

    def _attach_lora(self, config: VisionLLMLoRAConfig) -> None:
        if get_peft_model is None:
            raise ImportError("pip install peft")

        if config.load_in_4bit or config.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # ── 1. PEFT wrapping FIRST (restructures the model before any hooks) ───

        llm_targets = (
            list(config.llm_lora_target_modules)
            if config.llm_lora_target_modules
            else list(self._default_lora_targets())
        )

        llm_lora_config = LoraConfig(
            r=config.llm_lora_r,
            lora_alpha=config.llm_lora_alpha,
            lora_dropout=config.llm_lora_dropout,
            bias=config.lora_bias,
            use_dora=config.llm_use_dora,
            target_modules=llm_targets,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.model.language_model = get_peft_model(
            self.model.language_model,
            llm_lora_config,
        )

        if config.print_trainable_parameters:
            print("=== LLM trainable parameters ===")
            self.model.language_model.print_trainable_parameters()

        if config.training_phase == 3:
            vision_targets = (
                list(config.vision_lora_target_modules)
                if config.vision_lora_target_modules
                else ["qkv", "linear_fc1", "linear_fc2"]
            )
            vision_lora_config = LoraConfig(
                r=config.vision_lora_r,
                lora_alpha=config.vision_lora_alpha,
                lora_dropout=config.vision_lora_dropout,
                bias=config.lora_bias,
                use_dora=config.vision_use_dora,
                target_modules=vision_targets,
                # No task_type: get_peft_model then returns a plain PeftModel, which forwards only the given
                # arguments. FEATURE_EXTRACTION's forward adds input_ids/attention_mask/inputs_embeds=None, and the
                # tower passes attention_mask on to SDPA alongside its own, which raises a TypeError.
            )
            self.model.visual = get_peft_model(
                self.model.visual,
                vision_lora_config,
            )

            if config.print_trainable_parameters:
                print("=== Vision encoder trainable parameters ===")
                self.model.visual.print_trainable_parameters()

        # ── 2. Gradient checkpointing AFTER PEFT wrapping ─────────────────────

        if config.gradient_checkpointing:
            self.model.language_model.gradient_checkpointing_enable()

        if config.training_phase == 3:
            if config.gradient_checkpointing:
                self.model.visual.gradient_checkpointing_enable()

            # ── 3. Input-require-grad hook on the correct patch_embed ──────────
            # Must go through base_model after PEFT wrapping
            try:
                patch_proj = self.model.visual.base_model.model.patch_embed.proj
            except AttributeError:
                patch_proj = self.model.visual.patch_embed.proj

            def _make_inputs_require_grad(module, input, output):
                output.requires_grad = True   # property, not in-place op

            self._vision_grad_hook = patch_proj.register_forward_hook(
                _make_inputs_require_grad
            )

        # ── 4. enable_input_require_grads LAST (after grad checkpointing) ──────
        if hasattr(self.model.language_model, "enable_input_require_grads"):
            self.model.language_model.enable_input_require_grads()

        # For vision: the patch_embed hook above already handles this,
        # but also call the method if available for safety
        if config.training_phase == 3 and hasattr(self.model.visual, "enable_input_require_grads"):
            self.model.visual.enable_input_require_grads()

    def _freeze_all_but_policy_head(self, config: VisionLLMLoRAConfig) -> None:
        """
        Leave only POLICY_HEAD_PARAMETERS trainable (the policy's scene projection, per-trajectory bias and logit
        scale), and drop the hooks that make the backbone inputs require grad, so no backward pass runs through the
        LLM or the vision tower.
        """
        if config.head_type != "trajectory_embedding":
            raise ValueError("train_policy_head_only needs head_type='trajectory_embedding' (policy-only parameters).")
        trainable = []
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith(POLICY_HEAD_PARAMETERS))
            if param.requires_grad:
                trainable.append(name)
        if not trainable:
            raise RuntimeError("train_policy_head_only: no policy-head parameter found")
        if getattr(self, "_vision_grad_hook", None) is not None:
            self._vision_grad_hook.remove()
            self._vision_grad_hook = None
        for module in (self.model.language_model, self.model.visual):
            if hasattr(module, "disable_input_require_grads"):
                module.disable_input_require_grads()
            if config.gradient_checkpointing and hasattr(module, "gradient_checkpointing_disable"):
                module.gradient_checkpointing_disable()

    @staticmethod
    def _default_lora_targets() -> Sequence[str]:
        return ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")

    @staticmethod
    def _hidden_size(model: nn.Module) -> int:
        if hasattr(model, "language_model"):
            for attr in ("hidden_size", "n_embd", "d_model"):
                if hasattr(model.language_model.config, attr):
                    return int(getattr(model.language_model.config, attr))
        for attr in ("hidden_size", "n_embd", "d_model"):
            if hasattr(model.config, attr):
                return int(getattr(model.config, attr))
        raise ValueError("Cannot infer LLM hidden size from model.config.")


    # ─────────────────────────────────────────────────────────────────────────
    # Codebook lookup (one index per trajectory, not per timestep)
    # ─────────────────────────────────────────────────────────────────────────

    def decode_trajectory(self, codebook_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up full (8, 3) trajectories from single codebook indices.

        Args:
            codebook_ids : (B,)  — one index per batch element
        Returns:
            waypoints : (B, 8, 3)  — (x, y, yaw) trajectories
        """
        return self.codebook[codebook_ids]  # (B, 8, 3)

    # ─────────────────────────────────────────────────────────────────────────
    # Prompt builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_prompt(self, tokens_per_image: Sequence[int], command: Optional[int] = None) -> str:

        if self.config.image_tokens == "qwen":
            # One block per image, as the Qwen processor emits it; no separators, so no space tokens.
            img_str = "".join(
                _QWEN_VISION_START_TOKEN
                + _QWEN_IMAGE_PAD_TOKEN * n_tokens
                + _QWEN_VISION_END_TOKEN
                for n_tokens in tokens_per_image
            )
        else:
            img_block = " ".join(
                [_IMG_CTX_TOKEN] * sum(tokens_per_image)
            )

            img_str = (
                f"{_IMG_OPEN_TOKEN} "
                f"{img_block} "
                f"{_IMG_CLOSE_TOKEN}"
            )

        # No space before a placeholder: the tokenizer would emit a separate space token.
        ego_str = ""
        if self.config.ego_state_tokens:
            ego_str += f"Current velocity:{_EGO_STATE_TOKEN}, acceleration:{_EGO_STATE_TOKEN}.\n"
        if getattr(self.config, "ego_history_tokens", False):
            ego_str += f"Past positions:{_EGO_STATE_TOKEN * len(self.config.ego_history_position_scales)}.\n"
        if command is not None:
            ego_str += f"Navigation command: {_COMMAND_TEXT[command]}.\n"

        prompt = (
            self._prompt_template
            .replace("<IMG>", img_str)
            .replace("<EGO>", ego_str)
        )

        if self.config.image_tokens == "qwen":
            prompt += _QWEN_EMPTY_THINK

        return prompt


    # ─────────────────────────────────────────────────────────────────────────
    # Input encoding  (shared between training and inference)
    # ─────────────────────────────────────────────────────────────────────────

    def _preprocess_images(
        self, images: torch.Tensor, previous: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        """
        previous: optional (B, P, C, H, W) images of an earlier frame, arranged exactly like images; they fill the
        second temporal slot of every patch (config.temporal_history_frame, _patchify).
        Convert a batch of multi-camera images into the flat-patch format that
        Qwen3_5VisionModel.forward() expects.

        Input shape : (B, P, C, H, W)
          B — batch size
          P — cached images per sample: 12 canvas tiles (2 rows of 6), plus an
              optional 13th image (thumbnail / front zoom)
          C — channels (3)
          H, W — tile resolution

        config.image_layout decides what the vision tower sees per sample:
          "tiles"  — every cached image on its own (P square images)
          "strips" — tiles 0-5 and 6-11 reassembled into the two canvas rows
                     (H x 6W each; the tiles are exact crops of the canvas),
                     any further image on its own

        Qwen3_5VisionModel.forward(pixel_values, grid_thw):
          pixel_values : (sum over images of h * w, patch_dim), sample-major
                         patch_dim = C * temporal_size * patch_size²  = 1536
          grid_thw     : (B * images per sample, 3), each row [t=1, h, w]

        Returns:
            pixel_values     : (total patches, patch_dim)
            grid_thw         : (B * n_images, 3)
            tokens_per_image : merged tokens of each image of one sample, e.g.
                               (96, 96, 16) for strips + zoom at 128 px tiles
        """
        B, P, C, H, W = images.shape
        if previous is not None and previous.shape != images.shape:
            raise ValueError(f"previous frame {tuple(previous.shape)} does not match images {tuple(images.shape)}")

        def arrange(x: torch.Tensor) -> List[torch.Tensor]:
            if self.config.image_layout == "strips":
                if P < 12:
                    raise ValueError(f"image_layout='strips' needs the 12 canvas tiles, got {P} images.")
                # (B, 12, C, H, W) → (B, 2 rows, 6, C, H, W) → (B, 2, C, H, 6 * W)
                rows = x[:, :12].reshape(B, 2, 6, C, H, W).permute(0, 1, 3, 4, 2, 5).reshape(B, 2, C, H, 6 * W)
                return [rows] + ([x[:, 12:]] if P > 12 else [])
            if self.config.image_layout == "tiles":
                return [x]
            raise ValueError(f"Unknown image_layout: {self.config.image_layout!r}")

        groups = arrange(images)
        previous_groups = arrange(previous) if previous is not None else [None] * len(groups)

        # Patchify each group of same-sized images; keep every sample's
        # images together, in order, as the vision tower output is split per
        # sample afterwards.
        pixel_values, grids = [], []
        for group, previous_group in zip(groups, previous_groups):
            patches, grid_h, grid_w = self._patchify(group, previous_group)
            pixel_values.append(patches)
            grids += [[1, grid_h, grid_w]] * group.shape[1]

        pixel_values = torch.cat(pixel_values, dim=1).reshape(-1, self._patch_dim)

        # t=1: 2 temporal frames / Conv3d temporal stride 2 = 1 temporal patch
        grid_thw = torch.tensor(
            grids, dtype=torch.long, device=images.device
        ).repeat(B, 1)   # (B * n_images, 3)

        # Qwen's merger compresses 2×2 spatial patches → 1 token (factor 4).
        tokens_per_image = tuple(h * w // self._merger_factor for _, h, w in grids)

        return pixel_values, grid_thw, tokens_per_image

    def _patchify(
        self, images: torch.Tensor, previous: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int, int]:
        """
        (B, M, C, H, W) same-sized images → (B, M * grid_h * grid_w, patch_dim)
        patches, sample-major, plus the patch grid (grid_h, grid_w).

        For a still image we duplicate it along the temporal axis to satisfy
        the Conv3d temporal kernel size of 2, giving t=1 temporal patch. With
        previous (same shape: the same images of an earlier frame,
        config.temporal_history_frame) the two temporal slots hold (previous,
        current), the order of video frames; still t=1, same token count.
        """
        B, M, C, H, W = images.shape
        ps = self._patch_size     # 16
        ts = self._temporal_size  # 2

        if H % ps != 0 or W % ps != 0:
            raise ValueError(
                f"Image size ({H}x{W}) must be divisible by patch_size={ps}."
            )

        N = B * M   # treat every (batch, image) pair as one image sequence

        # Flatten B and M into a single image-sequence axis
        imgs_flat = images.reshape(N, C, H, W)   # (N, C, H, W)
        prev_flat = None if previous is None else previous.reshape(N, C, H, W)

        # ── Pad to even patch grid ───────────────────────────────────────────
        # Qwen3.5's merger does a 2×2 spatial merge (merge_size=2), so it calls
        # .view(t, h//2, 2, w//2, 2, -1).  Both grid_h and grid_w must therefore
        # be even, which means H and W must be divisible by patch_size * 2 = 32.
        # Example failure: 112×112 → grid 7×7 (odd) → merger view crashes.
        required = ps * 2   # 32
        pad_h = (required - H % required) % required   # 0 if already aligned
        pad_w = (required - W % required) % required
        if pad_h > 0 or pad_w > 0:
            # Pad right / bottom only so the top-left pixel origin is unchanged.
            imgs_flat = F.pad(imgs_flat, (0, pad_w, 0, pad_h))
            if prev_flat is not None:
                prev_flat = F.pad(prev_flat, (0, pad_w, 0, pad_h))
            H = H + pad_h
            W = W + pad_w
 
        grid_h = H // ps   # now guaranteed even
        grid_w = W // ps
 
        # Duplicate along temporal axis: (N, C, H, W) → (N, C, ts, H, W)
        # .contiguous() required before .view() because .expand() is lazy/non-contiguous.
        if prev_flat is None:
            imgs_t = imgs_flat.unsqueeze(2).expand(-1, -1, ts, -1, -1).contiguous()
        else:
            if ts != 2:
                raise ValueError(f"previous frame needs temporal_patch_size 2, got {ts}")
            imgs_t = torch.stack([prev_flat, imgs_flat], dim=2)  # (N, C, 2, H, W): earlier, current
 
        # Rearrange into non-overlapping spatio-temporal patches then flatten. Each patch is laid out as
        # (C, ts, ps, ps) in both orders.
        if self.config.patch_order == "merge_window":
            # Qwen2-VL image-processor order: the 2x2 patches of every merge window are contiguous, which is
            # what the tower's position ids and merger (consecutive groups of 4) assume.
            #   (N, C, ts, grid_h/m, m, ps, grid_w/m, m, ps) → (N, grid_h/m, grid_w/m, m, m, C, ts, ps, ps)
            m = self._merge_size
            patches = imgs_t.view(N, C, ts, grid_h // m, m, ps, grid_w // m, m, ps)
            patches = patches.permute(0, 3, 6, 4, 7, 1, 2, 5, 8).contiguous()
        elif self.config.patch_order == "raster":
            #   (N, C, ts, grid_h, ps, grid_w, ps) → (N, grid_h, grid_w, C, ts, ps, ps)
            patches = imgs_t.view(N, C, ts, grid_h, ps, grid_w, ps)
            patches = patches.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
        else:
            raise ValueError(f"Unknown patch_order: {self.config.patch_order!r}")

        return patches.view(B, M * grid_h * grid_w, self._patch_dim), grid_h, grid_w


    def _encode_inputs(
        self,
        features: Dict[str, torch.Tensor]
    ):

        dev = self.embedding.weight.device
        dtype = self.embedding.weight.dtype

        # ============================================================
        # VISION
        # ============================================================

        images = features["camera_feature"].to(dev)
        previous = None
        if self.config.temporal_history_frame:
            # uint8 tiles of the earlier frame, normalized like camera_feature (VisionLLMLoRAFeatureBuilder)
            previous = normalize_uint8_camera(
                features[f"camera_history_t{self.config.temporal_history_frame}"].to(dev),
                self.config.image_normalization,
            ).to(images.dtype)

        B = images.shape[0]

        pixel_values, grid_thw, tokens_per_image = (
            self._preprocess_images(images, previous)
        )
        n_images = len(tokens_per_image)

        grad_ctx = (
            torch.enable_grad
            if self.config.training_phase == 3
            else torch.no_grad
        )

        with grad_ctx():

            vis_out = self.model.visual(
                hidden_states=pixel_values,
                grid_thw=grid_thw,
            )

            # pooler_output is the merger's output (last_hidden_state is the
            # pre-merger 1024-dim state, so re-merging it ran the merger twice).
            if not isinstance(vis_out, torch.Tensor):
                vis_out = vis_out.pooler_output

        vis_out = vis_out.view(
            B,
            -1,
            vis_out.shape[-1]
        ).to(dtype)

        # ============================================================
        # EGO
        # ============================================================

        velocity = features["velocity"].to(dev)
        acceleration = features["acceleration"].to(dev)
        direction = features["driving_command"].to(dev)

        ego_repr = None

        if self.config.late_ego_fusion:
            enc_v = self.velocity_encoder(velocity)
            enc_a = self.acceleration_encoder(acceleration)
            enc_d = self.direction_embedding(direction)

            ego_flat = torch.cat(
                [enc_v, enc_a, enc_d],
                dim=-1,
            )

            ego_repr = self.ego_fusion_mlp(
                ego_flat
            ).to(dtype)

        if self.config.ego_state_tokens:
            ego_tokens = self.ego_state_tokens(
                torch.stack([velocity, acceleration], dim=1)
            ).to(dtype)   # (B, 2, D)
        if self.config.ego_history_tokens:
            pose = ego_history_inputs(features["ego_history_pose"].to(dev), self.config.ego_history_yaw_scale > 0)
            history = self.ego_history_tokens(pose)  # (B, 3, D)
            if self.training and self.config.ego_history_dropout > 0:
                drop = torch.rand(history.shape[0], 1, 1, device=history.device) < self.config.ego_history_dropout
                history = torch.where(drop, self.ego_history_missing.to(history.dtype)[None], history)
            # after velocity / acceleration: the prompt order of the <EGO_STATE> placeholders (masked_scatter)
            ego_tokens = torch.cat([ego_tokens, history.to(dtype)], dim=1)   # (B, 5, D)

        # ============================================================
        # PROMPT
        # ============================================================

        cache_key = tokens_per_image

        if cache_key not in self._prompt_cache:

            # One row per driving command when the command is written into
            # the prompt, else a single row shared by the whole batch.
            commands = (
                range(len(_COMMAND_TEXT))
                if self.config.command_in_prompt
                else (None,)
            )

            tok = self.tokenizer(
                [self._build_prompt(tokens_per_image, command) for command in commands],
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )

            if not bool(tok["attention_mask"].all()):
                raise ValueError("Command phrasings must tokenize to the same length.")

            # Qwen3.5's multimodal RoPE (image_tokens == "qwen"): each image
            # token gets (t, h, w) positions on its merged grid (4x4 per tile,
            # 4x24 per strip) and the text continues after each image
            # (findings F14). Same call the HF model makes; (3, rows, L).
            # "custom" passes no position_ids, so the LLM falls back to 1-D
            # positions as before.
            prompt_pos = None
            if self.config.image_tokens == "qwen":
                prompt_pos, _ = self.model.get_rope_index(
                    tok["input_ids"],
                    mm_token_type_ids=(tok["input_ids"] == self.img_ctx_id).int(),
                    image_grid_thw=grid_thw[:n_images].cpu().repeat(len(commands), 1),
                )
                prompt_pos = prompt_pos.to(dev)

            self._prompt_cache[cache_key] = (
                tok["input_ids"].to(dev),
                tok["attention_mask"].to(dev),
                prompt_pos,
            )

        prompt_ids, prompt_mask, prompt_pos = (
            self._prompt_cache[cache_key]
        )

        if self.config.command_in_prompt:
            prompt_ids  = prompt_ids.to(dev)[direction]
            prompt_mask = prompt_mask.to(dev)[direction]
            if prompt_pos is not None:
                prompt_pos = prompt_pos.to(dev)[:, direction]
        else:
            prompt_ids  = prompt_ids.to(dev).expand(B, -1)
            prompt_mask = prompt_mask.to(dev).expand(B, -1)
            if prompt_pos is not None:
                prompt_pos = prompt_pos.to(dev).expand(-1, B, -1)

        # ============================================================
        # IMAGE INJECTION ONLY
        # ============================================================

        img_pos = prompt_ids == self.img_ctx_id

        prompt_embeds = self.embedding(
            prompt_ids
        ).clone()

        img_expanded = torch.zeros_like(
            prompt_embeds
        )

        img_expanded[img_pos] = vis_out.reshape(
            -1,
            vis_out.shape[-1]
        )

        img_mask = img_pos.unsqueeze(-1).expand_as(
            prompt_embeds
        )

        prompt_embeds = torch.where(
            img_mask,
            img_expanded,
            prompt_embeds,
        )

        # ============================================================
        # EGO-STATE INJECTION
        # ============================================================

        if self.config.ego_state_tokens:
            # masked_scatter fills row-major: velocity, then acceleration, per sample.
            ego_mask = (prompt_ids == self.ego_state_id).unsqueeze(-1).expand_as(
                prompt_embeds
            )

            prompt_embeds = prompt_embeds.masked_scatter(
                ego_mask,
                ego_tokens,
            )

        # ============================================================
        # APPEND PLANNER QUERIES
        # ============================================================

        planner_queries = (
            self.planner_queries + self.query_pos_embed
        ).unsqueeze(0).expand(B, -1, -1).to(dtype)

        prompt_embeds = torch.cat(
            [prompt_embeds, planner_queries],
            dim=1,
        )

        planner_mask = torch.ones(
            B,
            self.num_planner_queries,
            device=prompt_mask.device,
            dtype=prompt_mask.dtype,
        )

        prompt_mask = torch.cat(
            [prompt_mask, planner_mask],
            dim=1,
        )

        # The queries continue the text positions after the last prompt
        # token (text, so its t, h, w positions are equal).
        position_ids = None
        if prompt_pos is not None:
            query_pos = prompt_pos[..., -1:] + 1 + torch.arange(
                self.num_planner_queries, device=prompt_pos.device
            )

            position_ids = torch.cat(
                [prompt_pos, query_pos],
                dim=-1,
            )

        return (
            prompt_embeds,
            prompt_mask,
            ego_repr,
            position_ids,
            img_pos,
        )



    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        features,
        teacher_forcing_seq=None,
        teacher_forcing_ratio=1.0,
    ):

        # Policy-only training: nothing before the head trains, so the backbone runs without autograd. Removing the
        # input-grad hooks is not enough: transformers keeps only the last enable_input_require_grads() call's hooks
        # (gradient_checkpointing_enable() registered one before), so one stayed and the LLM kept all activations
        # (~40 GB at batch 8; same values, only memory and time). Same forward values either way.
        backbone_ctx = torch.no_grad if self.config.train_policy_head_only else contextlib.nullcontext
        with backbone_ctx():
            (
                prompt_embeds,
                prompt_mask,
                ego_repr,
                position_ids,
                img_pos,
            ) = self._encode_inputs(features)

            outputs = self.model.language_model(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_mask,
                position_ids=position_ids,
                return_dict=True,
                use_cache=False,
            )

        traj_logits, sim, query_variance, relative_variance, fused = self.traj_classifier(
            outputs,
            ego_repr,
        )

        best_cb_ids = traj_logits.argmax(dim=-1)

        trajectory = self.decode_trajectory(
            best_cb_ids
        )

        predictions = {
            "trajectory_logits": traj_logits,
            "trajectory_tokens": best_cb_ids,
            "similarity": sim,
            "query_variance": query_variance,
            "relative_variance": relative_variance,
        }
        if self.traj_classifier.refiner is not None:
            predictions["anchor_trajectory"] = trajectory
            trajectory = trajectory + self.traj_classifier.refiner(fused, trajectory)
        if getattr(self.traj_classifier.classifier, "num_score_heads", 0) > 0:
            predictions["pdm_score_logits"] = self.traj_classifier.classifier.score_logits(fused)  # (B, M, K)
        predictions["trajectory"] = trajectory
        if self.aux_decoder is not None:
            # Last-layer hidden states at the image tokens; every prompt has the same image tokens in the same order.
            hidden = outputs.last_hidden_state[:, : img_pos.shape[1]]
            image_states = hidden[img_pos].view(img_pos.shape[0], -1, hidden.shape[-1])
            predictions.update(self.aux_decoder(image_states))
        return predictions