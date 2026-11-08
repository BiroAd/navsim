from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from navsim.agents.vision_llm_lora.vision_llm_lora_aux import aux_losses, aux_output_channels
from navsim.agents.vision_llm_lora.vision_llm_lora_codebook import load_codebook
from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_features import PDM_SCORE_KEYS
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorerConfig

# Sub-scores that multiply the total (any 0 zeroes it); the others enter a weighted mean with PDMScorerConfig's weights.
MULTIPLICATIVE_PDM_SCORE_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
)
_SCORER_CONFIG = PDMScorerConfig()
WEIGHTED_PDM_SCORE_WEIGHTS = {
    "ego_progress": _SCORER_CONFIG.progress_weight,
    "time_to_collision_within_bound": _SCORER_CONFIG.ttc_weight,
    "lane_keeping": _SCORER_CONFIG.lane_keeping_weight,
    "history_comfort": _SCORER_CONFIG.history_comfort_weight,
}

# Sub-scores that only take the values 0 and 1 (driving direction also has 0.5, ego progress is continuous): their
# accuracy at 0.5 is logged next to BCE and MAE.
BINARY_PDM_SCORE_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "traffic_light_compliance",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
)

# Cache codebook globally to avoid reloading it on every batch
_GLOBAL_CODEBOOK = None
_GLOBAL_CODEBOOK_DEVICE = None

def _get_codebook(config: VisionLLMLoRAConfig, device: torch.device) -> torch.Tensor:
    global _GLOBAL_CODEBOOK, _GLOBAL_CODEBOOK_DEVICE
    if _GLOBAL_CODEBOOK is None:
        arr = load_codebook(config.codebook_path)
        _GLOBAL_CODEBOOK = torch.from_numpy(arr).to(device)
        _GLOBAL_CODEBOOK_DEVICE = device
    elif _GLOBAL_CODEBOOK_DEVICE != device:
        _GLOBAL_CODEBOOK = _GLOBAL_CODEBOOK.to(device)
        _GLOBAL_CODEBOOK_DEVICE = device
    return _GLOBAL_CODEBOOK

# NAVSIM v1 PDMS = NC · DAC · (5 EP + 5 TTC + 2 C) / 12 (v1 paper): no driving-direction / traffic-light multiplier,
# no lane keeping. v2 history comfort stands in for v1 comfort; the weights are the same values as in v2.
V1_MULTIPLICATIVE_PDM_SCORE_KEYS = ("no_at_fault_collisions", "drivable_area_compliance")
V1_WEIGHTED_PDM_SCORE_WEIGHTS = {
    "ego_progress": _SCORER_CONFIG.progress_weight,
    "time_to_collision_within_bound": _SCORER_CONFIG.ttc_weight,
    "history_comfort": _SCORER_CONFIG.history_comfort_weight,
}


def aggregate_pdm_scores(
    sub_scores: torch.Tensor,
    multiplicative_keys: Tuple[str, ...] = MULTIPLICATIVE_PDM_SCORE_KEYS,
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Total score from sub-scores, as PDMScorer aggregates them without the two-frame extended comfort term (the
    codebook scoring's per-frame score): product of the multiplicative metrics x weighted mean of the others.
    With V1_MULTIPLICATIVE_PDM_SCORE_KEYS / V1_WEIGHTED_PDM_SCORE_WEIGHTS: the NAVSIM v1 PDMS from the same sub-scores.
    :param sub_scores: (B, len(PDM_SCORE_KEYS), K) values in [0, 1], rows in PDM_SCORE_KEYS order
    :return: (B, K)
    """
    weights = WEIGHTED_PDM_SCORE_WEIGHTS if weights is None else weights
    index = {key: i for i, key in enumerate(PDM_SCORE_KEYS)}
    multi = torch.stack([sub_scores[:, index[k]] for k in multiplicative_keys]).prod(dim=0)
    weighted = sum(w * sub_scores[:, index[k]] for k, w in weights.items())
    return multi * weighted / sum(weights.values())


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
    # Total PDM score of every entry from the cached sub-score targets (present when pdm_score_loss_weight > 0 or
    # score_weighted_labels); used for the score-aware labels and the proxy-score metrics below.
    total_targets = aggregate_pdm_scores(targets["pdm_scores"].float()) if "pdm_scores" in targets else None
    label_score_mass = None
    if config.score_weighted_labels:
        # Keep the label mass on entries that also score well; scenes where nothing scores keep the plain labels.
        weighted = soft_labels_fp32 * total_targets
        label_score_mass = weighted.sum(dim=-1, keepdim=True)  # share of the plain label mass that survives, x score
        soft_labels_fp32 = torch.where(
            label_score_mass > 1e-6, weighted / label_score_mass.clamp_min(1e-12), soft_labels_fp32
        )
    soft_labels = soft_labels_fp32.to(token_logits.dtype)  # (B, K)

    # ── Soft cross-entropy ───────────────────────────────────────────────────
    # log_softmax is numerically stable — critical with K = 16 384 entries
    log_probs = F.log_softmax(token_logits, dim=-1)                   # (B, K)
    soft_ce = -(soft_labels * log_probs).sum(dim=-1).mean()
    loss = config.trajectory_token_loss_weight * soft_ce
    # Unweighted loss terms and diagnostics, logged by agent_lightning_module as <train|val>/<name>.
    extra: Dict[str, torch.Tensor] = {"loss_terms/soft_ce": soft_ce.detach()}
    if label_score_mass is not None:
        extra["labels/score_mass"] = label_score_mass.mean().detach()  # E_label[score]: 1 = all label mass scores 1
    if total_targets is not None:
        # Proxy PDM score of the argmax entry (what the agent drives), as in scripts/evaluation/
        # eval_vision_llm_lora_checkpoints.py but from the cached sub-scores, whose human-penalty-filtered metrics read 1
        # (the offline and official totals are computed without that filter): ~0.03 higher than the eval script's score
        # (p1d_scorelabels epoch 10: 0.741 vs 0.713, 11% of val scenes differ). The gap 1 - score is
        # logged too, so checkpointing and early stopping (mode "min") can monitor val/pdm_scores/argmax_score_gap.
        with torch.no_grad():
            argmax_score = total_targets.gather(1, token_logits.detach().argmax(dim=-1, keepdim=True)).mean()
            extra["pdm_scores/argmax_score"] = argmax_score
            extra["pdm_scores/argmax_score_gap"] = 1.0 - argmax_score
            extra["pdm_scores/expected_score"] = (log_probs.detach().float().exp() * total_targets).sum(-1).mean()

    # ── Expected-distance term ───────────────────────────────────────────────
    # The soft CE also scores how probability is spread over wrong entries; E_p[ADE] directly rewards putting mass on
    # trajectories close to GT (gradient per logit: p_k * (ADE_k - E_p[ADE])), closer to what argmax decoding is
    # evaluated on. Logged val/loss then includes this term. Always logged; only in the loss with a weight > 0.
    with torch.set_grad_enabled(config.expected_distance_loss_weight > 0 and torch.is_grad_enabled()):
        entry_ade = torch.linalg.norm(
            target_trajectory[:, None, :, :2] - cb[None, :, :, :2], dim=-1
        ).mean(dim=-1)  # (B, K) metres
        expected_ade = (log_probs.float().exp() * entry_ade).sum(dim=-1).mean()
    extra["loss_terms/expected_ade"] = expected_ade.detach()
    if config.expected_distance_loss_weight > 0:
        loss = loss + config.expected_distance_loss_weight * expected_ade

    # ── Optional PDM sub-score heads ─────────────────────────────────────────
    # BCE of every candidate's predicted sub-scores against the precomputed ones (soft targets in [0, 1]).
    if config.pdm_score_loss_weight > 0:
        score_logits = predictions["pdm_score_logits"].float()  # (B, M, K)
        score_targets = targets["pdm_scores"].float()
        bce = F.binary_cross_entropy_with_logits(score_logits, score_targets, reduction="none")
        if config.pdm_score_head_weights is not None:
            if len(config.pdm_score_head_weights) != len(PDM_SCORE_KEYS):
                raise ValueError(f"pdm_score_head_weights needs {len(PDM_SCORE_KEYS)} values, in PDM_SCORE_KEYS order.")
            head_weights = torch.tensor(config.pdm_score_head_weights, dtype=bce.dtype, device=bce.device)
            score_loss = (bce.mean(dim=(0, 2)) * head_weights).sum() / head_weights.sum()
        else:
            score_loss = bce.mean()
        loss = loss + config.pdm_score_loss_weight * score_loss
        extra["loss_terms/score_bce"] = score_loss.detach()

        # BCE of the aggregated total, written out instead of F.binary_cross_entropy, which autocast refuses.
        total_pred = aggregate_pdm_scores(score_logits.sigmoid()).clamp(1e-6, 1 - 1e-6)
        total_target = aggregate_pdm_scores(score_targets)
        total_bce = -(total_target * total_pred.log() + (1 - total_target) * (1 - total_pred).log()).mean()
        if config.pdm_total_score_loss_weight > 0:
            loss = loss + config.pdm_total_score_loss_weight * total_bce
        extra["loss_terms/score_total_bce"] = total_bce.detach()
        extra["pdm_scores/total_mae"] = (total_pred - total_target).abs().mean().detach()
        # Proxy score of the entry the heads rank best (selection by the predicted total); the gap 1 - score is the
        # monitor (mode "min") for runs judged by head selection, e.g. phase 3 with score losses only.
        head_score = total_target.gather(1, total_pred.detach().argmax(dim=-1, keepdim=True)).mean().detach()
        extra["pdm_scores/head_score"] = head_score
        extra["pdm_scores/head_score_gap"] = 1.0 - head_score
        with torch.no_grad():
            probs = score_logits.sigmoid()
            for m, key in enumerate(PDM_SCORE_KEYS):
                extra[f"pdm_scores/{key}_bce"] = bce[:, m].mean()
                extra[f"pdm_scores/{key}_mae"] = (probs[:, m] - score_targets[:, m]).abs().mean()
                if key in BINARY_PDM_SCORE_KEYS:
                    extra[f"pdm_scores/{key}_acc"] = ((probs[:, m] > 0.5) == (score_targets[:, m] > 0.5)).float().mean()

    # ── Optional refinement of the selected entry ────────────────────────────
    # L1 between the refined trajectory and GT (x, y in m; yaw wrapped, in rad).
    if config.refinement_loss_weight > 0:
        diff = predictions["trajectory"].float() - target_trajectory
        diff = torch.cat([diff[..., :2], ((diff[..., 2:] + torch.pi) % (2 * torch.pi) - torch.pi)], dim=-1)
        refine_l1 = diff.abs().mean()
        loss = loss + config.refinement_loss_weight * refine_l1
        extra["loss_terms/refine_l1"] = refine_l1.detach()
    if "anchor_trajectory" in predictions:  # the unrefined argmax entry, to separate the refiner's correction
        with torch.no_grad():
            anchor_err = torch.linalg.norm(predictions["anchor_trajectory"][..., :2].float() - target_trajectory[..., :2], dim=-1)
            extra["refine/anchor_ade"] = anchor_err.mean()
            extra["refine/anchor_fde"] = anchor_err[:, -1].mean()

    # ── Optional auxiliary perception losses (BEV decoder on the image-token states) ──
    if aux_output_channels(config):
        aux_loss, aux_extra = aux_losses(predictions, targets, config)
        loss = loss + aux_loss
        extra.update(aux_extra)

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
    return (
        loss, pred_entropy, predictions["similarity"], ade, fde, predictions["query_variance"],
        predictions["relative_variance"], extra,
    )
