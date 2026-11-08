"""
Auxiliary perception losses for vision_llm_lora (training only): a small BEV decoder reads the LLM's last-layer hidden
states at the image tokens and predicts the cached perception maps of VisionLLMLoRAAuxTargetBuilder. Its gradient
reaches the LoRA adapters only through the image-token states, which the planner queries attend to inside the LLM.

Map layout (VisionLLMLoRAAuxTargetBuilder): H x W = 160 x 128 cells of 0.5 m, rows along x (forward, -16 to 64 m), columns
along y (left to right, 32 to -32 m). The decoder's learned query grid (default 20 x 16, one query per 4 x 4 m cell)
is upsampled by 2 per step (default 3 steps) to that size.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_features import NO_LANE_HEADING

AUX_HEADING_BINS = 36  # VisionLLMLoRAAuxTargetBuilder default, used for the cached targets
AUX_MAP_SHAPE = (160, 128)  # cached targets (x, y); the decoder output must match
_MAX_IMAGE_TOKENS = 512  # rows of the image-token position embedding; strips and tiles both use 208


def aux_output_channels(config: VisionLLMLoRAConfig) -> Dict[str, int]:
    """Output name -> channels, for each auxiliary loss with a weight > 0 (none: no decoder is built)."""
    outputs = {}
    if config.aux_bev_semantic_loss_weight > 0:
        outputs["aux_bev_semantic"] = len(config.bev_semantic_classes) + 1  # + class 0 = none
    if config.aux_drivable_area_loss_weight > 0:
        outputs["aux_drivable_area"] = 1
    if config.aux_lane_heading_loss_weight > 0:
        outputs["aux_lane_heading"] = AUX_HEADING_BINS
    return outputs


class AuxBEVDecoder(nn.Module):
    """
    Image-token hidden states (B, N, in_dim) -> BEV maps {name: (B, C, H, W)}.

    1. LayerNorm + Linear in_dim -> dim, plus a learned embedding per image-token position (camera and grid cell; the
       token order is fixed by the prompt).
    2. Learned BEV queries, one per ground cell of query_grid, through `layers` transformer decoder layers
       (self-attention between cells, cross-attention to the image tokens, MLP).
    3. The query grid as a feature map, `upsample_steps` x (bilinear x2, 3x3 conv, GroupNorm, GELU), halving the
       channels each step (at least 32).
    4. One 1x1 conv per output.
    """

    def __init__(
        self,
        in_dim: int,
        outputs: Dict[str, int],
        dim: int = 256,
        layers: int = 2,
        heads: int = 8,
        query_grid: Tuple[int, int] = (20, 16),
        upsample_steps: int = 3,
    ):
        super().__init__()
        self.query_grid = tuple(query_grid)
        self.input_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim))
        self.token_pos = nn.Parameter(torch.randn(_MAX_IMAGE_TOKENS, dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(self.query_grid[0] * self.query_grid[1], dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=4 * dim, dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        blocks, channels = [], dim
        for _ in range(upsample_steps):
            out = max(channels // 2, 32)
            blocks += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(channels, out, kernel_size=3, padding=1),
                nn.GroupNorm(8, out),
                nn.GELU(),
            ]
            channels = out
        self.upsample = nn.Sequential(*blocks)
        self.heads = nn.ModuleDict({name: nn.Conv2d(channels, c, kernel_size=1) for name, c in outputs.items()})

    def forward(self, image_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, num_tokens, _ = image_states.shape
        if num_tokens > self.token_pos.shape[0]:
            raise ValueError(f"{num_tokens} image tokens, the position embedding has {self.token_pos.shape[0]} rows")
        memory = self.input_proj(image_states) + self.token_pos[:num_tokens]
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        x = self.norm(self.decoder(queries, memory))  # (B, G, dim)
        x = x.transpose(1, 2).reshape(batch, -1, *self.query_grid)  # (B, dim, gx, gy)
        x = self.upsample(x)
        return {name: head(x) for name, head in self.heads.items()}


def _iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    union = (pred | target).sum()
    return (pred & target).sum() / union.clamp_min(1)


def aux_losses(
    predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], config: VisionLLMLoRAConfig
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Weighted sum of the auxiliary losses with a weight > 0, and logged values: unweighted losses (loss_terms/aux_*) and
    metrics (aux/*). BEV semantics: cross-entropy over all cells (class 0 = none). Drivable area: BCE over all cells.
    Lane heading: cross-entropy over the direction bins, on lane cells only (overlapping lanes in intersections make
    the direction multi-modal, which a classification handles and a regression would average).
    """
    loss = torch.zeros((), device=next(iter(predictions.values())).device)
    extra: Dict[str, torch.Tensor] = {}
    if config.aux_bev_semantic_loss_weight > 0:
        logits = predictions["aux_bev_semantic"].float()
        target = targets["bev_semantic_map"].long()
        _check_shape(logits, target)
        bev = F.cross_entropy(logits, target)
        loss = loss + config.aux_bev_semantic_loss_weight * bev
        extra["loss_terms/aux_bev_semantic"] = bev.detach()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            ious = [_iou(pred == c, target == c) for c in range(1, logits.shape[1]) if (target == c).any()]
            extra["aux/bev_miou"] = torch.stack(ious).mean() if ious else torch.zeros((), device=loss.device)
    if config.aux_drivable_area_loss_weight > 0:
        logits = predictions["aux_drivable_area"].float()[:, 0]
        target = targets["drivable_area_map"].float()
        _check_shape(logits, target)
        drivable = F.binary_cross_entropy_with_logits(logits, target)
        loss = loss + config.aux_drivable_area_loss_weight * drivable
        extra["loss_terms/aux_drivable_area"] = drivable.detach()
        with torch.no_grad():
            extra["aux/drivable_iou"] = _iou(logits > 0, target > 0.5)
    if config.aux_lane_heading_loss_weight > 0:
        logits = predictions["aux_lane_heading"].float()
        target = targets["lane_heading_map"].long()
        _check_shape(logits, target)
        on_lane = target != NO_LANE_HEADING
        # Sum / count instead of the mean: a batch without lane cells gives 0, not NaN.
        heading = F.cross_entropy(logits, target, ignore_index=NO_LANE_HEADING, reduction="sum")
        heading = heading / on_lane.sum().clamp_min(1)
        loss = loss + config.aux_lane_heading_loss_weight * heading
        extra["loss_terms/aux_lane_heading"] = heading.detach()
        with torch.no_grad():
            diff = (logits.argmax(dim=1) - target).abs()
            diff = torch.minimum(diff, AUX_HEADING_BINS - diff)  # circular bin distance
            extra["aux/lane_heading_acc"] = (diff[on_lane] <= 1).float().mean() if on_lane.any() else loss * 0
    return loss, extra


def _check_shape(logits: torch.Tensor, target: torch.Tensor) -> None:
    if logits.shape[-2:] != target.shape[-2:]:
        raise ValueError(
            f"aux decoder output {tuple(logits.shape[-2:])} != target map {tuple(target.shape[-2:])}: "
            "query_grid x 2^upsample_steps must equal the cached map size"
        )
