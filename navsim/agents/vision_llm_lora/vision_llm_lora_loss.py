from typing import Dict

import torch
import torch.nn.functional as F

from navsim.agents.vision_llm.vision_llm_loss import vision_llm_loss
from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_tokenizer import CoordinateKNNTokenizer


import numpy as np

# Cache codebook globally to avoid reloading it on every batch
_GLOBAL_CODEBOOK = None
_GLOBAL_CODEBOOK_DEVICE = None

def _get_codebook(config: VisionLLMLoRAConfig, device: torch.device) -> torch.Tensor:
    global _GLOBAL_CODEBOOK, _GLOBAL_CODEBOOK_DEVICE
    if _GLOBAL_CODEBOOK is None:
        arr = np.load(config.codebook_path).astype(np.float32)
        if arr.shape[1] == 40:
            arr = arr[:, [0, 5, 10, 15, 20, 25, 30, 35], :]
        _GLOBAL_CODEBOOK = torch.from_numpy(arr).to(device)
        _GLOBAL_CODEBOOK_DEVICE = device
    elif _GLOBAL_CODEBOOK_DEVICE != device:
        _GLOBAL_CODEBOOK = _GLOBAL_CODEBOOK.to(device)
        _GLOBAL_CODEBOOK_DEVICE = device
    return _GLOBAL_CODEBOOK

def vision_llm_lora_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: VisionLLMLoRAConfig,
) -> torch.Tensor:
    """
    Soft-label cross-entropy trajectory loss.

    Instead of penalising all wrong tokens equally, we build a soft target
    distribution from the physical distance between the GT trajectory and
    every codebook entry.  Tokens that are geometrically close to GT receive
    a high soft-label weight; tokens that are far away receive nearly zero.

    This gives the model a smooth gradient signal that reflects the actual
    geometry of the codebook rather than an arbitrary token ordering.
    """
    if "trajectory_logits" not in predictions:
        return vision_llm_loss(targets, predictions, config)

    token_logits = predictions["trajectory_logits"]   # (B, K)
    target_trajectory = targets["trajectory"].float()      # (B, 8, 3)

    cb  = _get_codebook(config, token_logits.device).float()  # (K, 8, 3)

    # ── Physical distance: GT vs every codebook entry ────────────────────────
    # xy: sum squared error over 8 timesteps → (B, K)
    xy_dist2 = (
        (target_trajectory[:, None, :, :2] - cb[None, :, :, :2]) ** 2
    ).sum(dim=(-2, -1))

    # yaw: wrapped into [-π, π] before squaring → (B, K)
    delta_yaw = target_trajectory[:, None, :, 2] - cb[None, :, :, 2]
    delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi
    yaw_dist2 = (delta_yaw ** 2).sum(dim=-1)

    distances = xy_dist2 + float(getattr(config, "yaw_distance_weight", 2.5)) * yaw_dist2  # (B, K)
    distances = distances.float()

    '''target_ids = distances.argmin(dim=-1)

    loss = F.cross_entropy(token_logits, target_ids)'''

    # ── Soft labels: nearby codebook entries get high probability ────────────
    # Temperature controls sharpness:
    #   low  (0.1) → nearly one-hot, only the closest token is rewarded
    #   high (2.0) → flat, many tokens share credit
    LABEL_TEMPERATURE = float(getattr(config, "label_temperature", 0.7))
    soft_labels_fp32 = F.softmax(-distances / LABEL_TEMPERATURE, dim=-1).float()  # (B, K)
    soft_labels = soft_labels_fp32.to(token_logits.dtype)  # (B, K)

    # ── Soft cross-entropy ───────────────────────────────────────────────────
    # log_softmax is numerically stable — critical with K = 16 384 entries
    log_probs = F.log_softmax(token_logits, dim=-1)                   # (B, K)
    loss = -(soft_labels * log_probs).sum(dim=-1).mean()

    pred_probs = F.softmax(token_logits.detach().float(), dim=-1)
    pred_entropy = -(pred_probs * torch.log(pred_probs + 1e-7)).sum(dim=-1).mean()

    ## ADE loss
    pred_xy = predictions["trajectory"][..., :2]
    gt_xy = targets["trajectory"][..., :2]

    ade = torch.norm(
        pred_xy - gt_xy,
        dim=-1,
    ).mean()

    ## FDE loss

    pred_final = predictions["trajectory"][:, -1, :2]
    gt_final = targets["trajectory"][:, -1, :2]

    fde = torch.norm(
        pred_final - gt_final,
        dim=-1,
    ).mean()

    ## TOP-K accuracy for analysis

    #target_ids = distances.argmin(dim=-1)

    #pred_topk = token_logits.topk(k=10, dim=-1).indices

    #top1 = (pred_topk[:, :1] == target_ids[:, None]).any(dim=-1).float().mean()

    #top5 = (pred_topk[:, :5] == target_ids[:, None]).any(dim=-1).float().mean()

    #top10 = (pred_topk[:, :10] == target_ids[:, None]).any(dim=-1).float().mean()

    #return loss, entropy, predictions["attn_weights"], predictions["similarity"], ade, fde
    return loss, pred_entropy, predictions["similarity"], ade, fde, predictions["query_variance"], predictions["relative_variance"]
