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

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
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

from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig


# ─────────────────────────────────────────────────────────────────────────────
# Special-token name constants
# ─────────────────────────────────────────────────────────────────────────────

_IMG_CTX_TOKEN = "<IMG_CONTEXT>"
_IMG_OPEN_TOKEN = "<img>"
_IMG_CLOSE_TOKEN = "</img>"

# ─────────────────────────────────────────────────────────────────────────────
# Trajectory classifier
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryClassifier(nn.Module):
    """
    Planner-query + ego late fusion classifier.
    """

    def __init__(
        self,
        llm_hidden_size: int,
        num_trajectories: int,
        num_queries: int = 4,
        bottleneck_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_queries = num_queries

        #self.query_attention = nn.MultiheadAttention(
        #    embed_dim=llm_hidden_size,
        #    num_heads=8,
        #    dropout=0,
        #    batch_first=True,
        #)

        fusion_input_dim = (
            llm_hidden_size * num_queries
            + llm_hidden_size
        )

        self.query_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, bottleneck_dim),
            nn.RMSNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(
            bottleneck_dim,
            num_trajectories,
        )

        #self.query_norm = nn.RMSNorm(
        #    llm_hidden_size * num_queries
        #)

        self.logit_scale = nn.Parameter(
            torch.tensor(0.0)
        )

        #self.query_post_norm = nn.RMSNorm(llm_hidden_size)

    def forward(
        self,
        outputs,
        ego_repr: torch.Tensor,
    ) -> torch.Tensor:

        query_embeddings = outputs.last_hidden_state[
            :, -self.num_queries:, :
        ]

        #attn_out, attn_weights = self.query_attention(
        #    query_embeddings,
        #    query_embeddings,
        #    query_embeddings,
        #)

        #query_embeddings = self.query_post_norm(
        #    query_embeddings + attn_out
        #)

        planner_repr = query_embeddings.reshape(
            query_embeddings.shape[0],
            -1,
        )

        q = query_embeddings.mean(dim=0).detach()

        sim = F.cosine_similarity(
            q[:, None, :],
            q[None, :, :],
            dim=-1,
        )

        # Return only the mean of off-diagonal elements as a scalar
        mask = ~torch.eye(self.num_queries, dtype=torch.bool, device=q.device)
        sim_scalar = sim[mask].mean()

        #planner_repr = self.query_norm(planner_repr)

        fused = torch.cat(
            [planner_repr, ego_repr],
            dim=-1,
        )

        fused = self.query_fusion(fused)

        logits = self.classifier(fused)
        logits = torch.tanh(logits / 30.0) * 30.0
        scale = self.logit_scale.clamp(-2.0, 2.0).exp()
        logits = logits * scale

        return logits, sim_scalar


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

        # ── 4. Vision encoder extraction & Base Freezing ────────────────────
        self.vision_encoder: nn.Module = self.model.visual
        if not config.train_vision_encoder:

            for p in self.vision_encoder.parameters():
                p.requires_grad_(False)

        # ── 5. Language part extraction & Base Freezing ────────────────────
        if not config.train_language_model:

            for p in self.model.language_model.parameters():
                p.requires_grad_(False)


        # ── 6. Optional LoRA ────────────────────────────────────────────────
        if config.enable_lora:
            self._attach_lora(config)
            # Re-sync: _attach_lora may have replaced self.model.visual
            # with a PeftModel wrapper; keep the reference consistent.
            self.vision_encoder = self.model.visual

        # ── 7. Cached references ─────────────────────────────────────────────
        self.embedding = self.model.language_model.get_input_embeddings()
        llm_hidden: int = self._hidden_size(self.model)

        # Derive patch constants from the Conv3d kernel so nothing is hardcoded.
        conv = self.vision_encoder.patch_embed.proj          # Conv3d
        self._patch_size     = conv.kernel_size[1]           # 16
        self._temporal_size  = conv.kernel_size[0]           # 2  (frames per temporal patch)
        self._patch_dim      = (
            conv.in_channels
            * self._temporal_size
            * self._patch_size
            * self._patch_size
        )  # 3 * 2 * 16 * 16 = 1536
        self._merger_factor = (
            self.vision_encoder.merger.linear_fc1.in_features
            // self.vision_encoder.merger.norm.normalized_shape[0]
        )

        #self._unfreeze_new_token_embeddings(self._old_vocab_size)

        # ── 8. Token-type bias: 0=text, 1=ego, 2=image ───────────────────────
        # self.token_type_embed = nn.Embedding(3, llm_hidden)

        # ── 9. Ego encoders ──────────────────────────────────────────────────
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

        # Fuse [velocity | acceleration | direction] → n_ego_tokens context slots
        bottleneck_dim = 256
        out_dim = llm_hidden

        self.ego_fusion_mlp = EgoFusionMLP(
            in_dim=3 * llm_hidden,
            bottleneck_dim=bottleneck_dim,
            out_dim=out_dim,
            dropout=getattr(config, "head_dropout", 0.0),
        )

        # ── 10. Prompt template ───────────────────────────────────────────────
        # Uses Qwen3.5's ChatML template structure
        self._prompt_template = (
            "<|im_start|>system\n"
            "You are an expert autonomous driving assistant."
            "<|im_end|>\n"

            "<|im_start|>user\n"

            "Visual context:\n"
            "<IMG>\n"

            "Predict the safest trajectory token.\n"

            "<|im_end|>\n"

            "<|im_start|>assistant\n"
        )

        # ── 11. Trajectory classifier head (for direct token prediction) ─────────
        self.num_planner_queries = 4
        self.traj_classifier = TrajectoryClassifier(
            llm_hidden_size=llm_hidden,
            num_trajectories=self.codebook_size,
            bottleneck_dim=2048,
            dropout=getattr(config, "head_dropout", 0.0),
            num_queries=self.num_planner_queries,
        )

        # ── 12. Planner queries ─────────
        init_embeds = torch.zeros(self.num_planner_queries, llm_hidden)
        nn.init.orthogonal_(init_embeds)
        # scale to match typical embedding magnitude
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
        (K, 40, 3)  — raw navsim 5 s @ 8 Hz; subsampled here with step=5:
                       indices [0, 5, 10, 15, 20, 25, 30, 35] → 8 waypoints.

        Subsampling offline is recommended (see codebook/build_codebook.py).
        ``self.codebook`` is always (K, num_waypoints, 3) after this call.
        """
        arr = np.load(codebook_path).astype(np.float32)

        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"Codebook must have shape (K, T, 3), got {arr.shape}. "
                "Each entry is a full trajectory of T (x, y, yaw) waypoints."
            )

        K, T_full, _ = arr.shape

        if T_full == self.num_waypoints:
            pass  # already correct resolution

        elif T_full > self.num_waypoints:
            # Take every (T_full // num_waypoints)-th frame.
            # For T_full=40, num_waypoints=8: step=5 → [0,5,10,15,20,25,30,35]
            step    = T_full // self.num_waypoints
            indices = np.arange(0, step * self.num_waypoints, step)
            arr     = arr[:, indices, :]   # (K, num_waypoints, 3)

        else:
            raise ValueError(
                f"Codebook has {T_full} timesteps but num_waypoints={self.num_waypoints}. "
                "Upsampling is not supported; pre-process the codebook offline."
            )

        self.codebook_size: int = K
        self.register_buffer("codebook", torch.from_numpy(arr))  # (K, 8, 3)

    # ─────────────────────────────────────────────────────────────────────────
    # Special-token registration
    # ─────────────────────────────────────────────────────────────────────────

    def _register_special_tokens(self) -> None:

        old_vocab_size = len(self.tokenizer)

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

        self._old_vocab_size = old_vocab_size

    def _unfreeze_new_token_embeddings(self, old_vocab_size: int) -> None:
        """Allow only the new structural delimiter embeddings to train."""
        emb = self.model.get_input_embeddings()
        emb.weight.requires_grad_(True)

        def _mask_old_tokens(grad: torch.Tensor) -> torch.Tensor:
            grad[:old_vocab_size] = 0.0   # freeze original vocab
            # new rows [old_vocab_size:] pass through unchanged
            return grad

        emb.weight.register_hook(_mask_old_tokens)

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
        return AutoModel.from_pretrained(config.architecture, **kwargs)

    def _attach_lora(self, config: VisionLLMLoRAConfig) -> None:
        if get_peft_model is None:
            raise ImportError("pip install peft")

        if config.load_in_4bit or config.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # ── 1. Gradient checkpointing ──────────────────────────────────────────
        self.model.language_model.gradient_checkpointing_enable()

        if config.train_vision_encoder:
            self.vision_encoder.gradient_checkpointing_enable()

            def _make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self._vision_grad_hook = (
                self.vision_encoder.patch_embed.proj
                .register_forward_hook(_make_inputs_require_grad)
            )

        # ── 2. Input grad hooks (must happen before PEFT wrapping) ────────────
        # Required for gradient checkpointing to keep the chain connected
        # through the embedding → first layer boundary.
        if hasattr(self.model.language_model, "enable_input_require_grads"):
            self.model.language_model.enable_input_require_grads()

        # ── 3. LLM LoRA ────────────────────────────────────────────────────────
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

        # ── 4. Vision DoRA / LoRA (only if both flags are set) ─────────────────
        # enable_vision_lora is independent of enable_lora so you can do
        # LLM-only LoRA in phase 2 and add vision DoRA only in phase 3.
        if config.enable_vision_lora and config.train_vision_encoder:
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
                task_type=TaskType.FEATURE_EXTRACTION,
            )

            self.model.visual = get_peft_model(
                self.model.visual,
                vision_lora_config,
            )

            if config.print_trainable_parameters:
                print("=== Vision encoder trainable parameters ===")
                self.model.visual.print_trainable_parameters()

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
    # Codebook nearest-neighbour (one index per trajectory, not per timestep)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_trajectory(self, waypoints: torch.Tensor) -> torch.Tensor:
        """
        Find the single nearest codebook entry for each trajectory in the batch.

        Distance is summed over all T=8 timesteps:

            D(gt, entry_k) = Σ_t [ (Δx_t)² + (Δy_t)² + w·(Δyaw_wrapped_t)² ]

        Returns ONE index per batch element (the whole trajectory is one token).

        Args:
            waypoints : (B, 8, 3)  — GT trajectories, each step is (x, y, yaw)
        Returns:
            codebook_ids : (B,)    — indices in [0, K)
        """
        w  = float(getattr(self.config, "yaw_distance_weight", 2.5))
        gt = waypoints.float()       # (B, 8, 3)
        cb = self.codebook.float()   # (K, 8, 3)

        # Squared Euclidean on (x, y), summed over timesteps → (B, K)
        xy_dist2 = (
            (gt[:, None, :, :2] - cb[None, :, :, :2]) ** 2
        ).sum(dim=(-2, -1))   # sum over T and xy dims

        # Wrapped angular distance on yaw, summed over timesteps → (B, K)
        delta_yaw = gt[:, None, :, 2] - cb[None, :, :, 2]   # (B, K, T)
        delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi
        yaw_dist2 = (delta_yaw ** 2).sum(dim=-1)              # (B, K)

        return (xy_dist2 + w * yaw_dist2).argmin(dim=-1)      # (B,)

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

    def _build_prompt(self, n_img_tokens: int) -> str:

        img_block = " ".join(
            [_IMG_CTX_TOKEN] * n_img_tokens
        )

        img_str = (
            f"{_IMG_OPEN_TOKEN} "
            f"{img_block} "
            f"{_IMG_CLOSE_TOKEN}"
        )

        return (
            self._prompt_template
            .replace("<IMG>", img_str)
        )


    # ─────────────────────────────────────────────────────────────────────────
    # Input encoding  (shared between training and inference)
    # ─────────────────────────────────────────────────────────────────────────

    def _preprocess_images(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Convert a batch of multi-camera images into the flat-patch format that
        Qwen3_5VisionModel.forward() expects.
 
        Input shape : (B, P, C, H, W)
          B — batch size
          P — number of camera perspectives
          C — channels (3)
          H, W — spatial resolution
 
        Each (B, P) image is treated as an independent sequence entry, so the
        effective number of image sequences is N = B * P.  The vision encoder
        processes all of them in one call and the caller reshapes the output
        back to (B, P * n_vis_per_image, D) before injecting into the prompt.
 
        Qwen3_5VisionModel.forward(pixel_values, grid_thw):
          pixel_values : (N * h * w, patch_dim)
                         N = B * P,  h = H // patch_size,  w = W // patch_size
                         patch_dim = C * temporal_size * patch_size²  = 1536
          grid_thw     : (N, 3)   each row is [t=1, h, w]
 
        For a still image we duplicate it along the temporal axis to satisfy
        the Conv3d temporal kernel size of 2, giving t=1 temporal patch.
 
        Returns:
            pixel_values  : (N * h * w, patch_dim)
            grid_thw      : (N, 3)
            N             : B * P  (total image sequences)
            n_vis_per_cam : h * w // merger_factor  (tokens per camera after merger)
        """
        B, P, C, H, W = images.shape
        ps = self._patch_size     # 16
        ts = self._temporal_size  # 2
 
        if H % ps != 0 or W % ps != 0:
            raise ValueError(
                f"Image size ({H}x{W}) must be divisible by patch_size={ps}."
            )
 
        N = B * P   # treat every (batch, perspective) pair as one image sequence
 
        # Flatten B and P into a single image-sequence axis
        imgs_flat = images.reshape(N, C, H, W)   # (N, C, H, W)
 
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
            H = H + pad_h
            W = W + pad_w
 
        grid_h = H // ps   # now guaranteed even
        grid_w = W // ps
 
        # Duplicate along temporal axis: (N, C, H, W) → (N, C, ts, H, W)
        # .contiguous() required before .view() because .expand() is lazy/non-contiguous.
        imgs_t = imgs_flat.unsqueeze(2).expand(-1, -1, ts, -1, -1).contiguous()
 
        # Rearrange into non-overlapping spatio-temporal patches then flatten:
        #   (N, C, ts, grid_h, ps, grid_w, ps)
        #   → permute → (N, grid_h, grid_w, C, ts, ps, ps)
        #   → view    → (N * grid_h * grid_w, patch_dim)
        patches = imgs_t.view(N, C, ts, grid_h, ps, grid_w, ps)
        patches = patches.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
        pixel_values = patches.view(N * grid_h * grid_w, self._patch_dim)
 
        # t=1: 2 temporal frames / Conv3d temporal stride 2 = 1 temporal patch
        grid_thw = torch.tensor(
            [[1, grid_h, grid_w]], dtype=torch.long, device=images.device
        ).expand(N, -1).contiguous()   # (N, 3)
 
        # Qwen's merger compresses 2×2 spatial patches → 1 token (factor 4).
        # Derive the factor from the merger's first linear layer input dim.
        n_vis_per_cam = (grid_h * grid_w) // self._merger_factor
 
        return pixel_values, grid_thw, N, n_vis_per_cam


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

        B = images.shape[0]

        pixel_values, grid_thw, N, n_vis_per_cam = (
            self._preprocess_images(images)
        )

        grad_ctx = (
            torch.enable_grad
            if self.config.train_vision_encoder
            else torch.no_grad
        )

        with grad_ctx():

            vis_out = self.vision_encoder(
                pixel_values,
                grid_thw=grid_thw,
            )

            if not isinstance(vis_out, torch.Tensor):
                vis_out = vis_out.last_hidden_state

            if vis_out.shape[-1] != self.embedding.weight.shape[-1]:
                vis_out = self.vision_encoder.merger(vis_out)

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

        # ============================================================
        # PROMPT
        # ============================================================

        n_img = vis_out.shape[1]

        cache_key = n_img

        if cache_key not in self._prompt_cache:

            prompt_str = self._build_prompt(n_img)

            tok = self.tokenizer(
                prompt_str,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )

            self._prompt_cache[cache_key] = (
                tok["input_ids"].to(dev),
                tok["attention_mask"].to(dev),
            )

        prompt_ids, prompt_mask = (
            self._prompt_cache[cache_key]
        )

        prompt_ids  = prompt_ids.to(dev).expand(B, -1)
        prompt_mask = prompt_mask.to(dev).expand(B, -1)

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
        # APPEND PLANNER QUERIES
        # ============================================================

        planner_queries = (
            self.planner_queries + self.query_pos_embed
        ).unsqueeze(0).expand(B, -1, -1)

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

        return (
            prompt_embeds,
            prompt_mask,
            ego_repr,
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

        (
            prompt_embeds,
            prompt_mask,
            ego_repr,
        ) = self._encode_inputs(features)

        outputs = self.model.language_model(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_mask,
            return_dict=True,
            use_cache=False,
        )

        traj_logits, sim = self.traj_classifier(
            outputs,
            ego_repr,
        )

        best_cb_ids = traj_logits.argmax(dim=-1)

        trajectory = self.decode_trajectory(
            best_cb_ids
        )

        return {
            "trajectory_logits": traj_logits,
            "trajectory_tokens": best_cb_ids,
            "trajectory": trajectory,
            #"attn_weights": attn_weights,
            "similarity": sim,
        }