"""
CPU-only checks for the vision_llm_lora soft-label loss and its optional expected-distance term.
No network, dataset or HF weights: the global codebook is replaced by a tiny one.
Run: pytest tests/test_vision_llm_lora_loss.py
"""

import torch

from navsim.agents.vision_llm_lora import vision_llm_lora_loss as loss_module
from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig

B, K, T = 2, 3, 8


def _codebook():
    # Entry 0 = GT, entry 1 shifted by 1 m in x, entry 2 by 5 m.
    cb = torch.zeros(K, T, 3)
    cb[1, :, 0] = 1.0
    cb[2, :, 0] = 5.0
    return cb


def _loss(monkeypatch, logits, preds_extra=None, targets_extra=None, **flags):
    monkeypatch.setattr(loss_module, "_get_codebook", lambda config, device: _codebook())
    preds = {
        "trajectory_logits": logits,
        "trajectory": torch.zeros(B, T, 3),
        "similarity": torch.tensor(0.0),
        "query_variance": torch.tensor(0.0),
        "relative_variance": torch.tensor(0.0),
        **(preds_extra or {}),
    }
    targets = {"trajectory": torch.zeros(B, T, 3), **(targets_extra or {})}
    return loss_module.vision_llm_lora_loss(targets, preds, VisionLLMLoRAConfig(**flags))[0]


def test_cross_entropy_weight_zero_leaves_only_the_expected_distance(monkeypatch):
    logits = torch.randn(B, K)
    expected_ade = (torch.softmax(logits, -1) * torch.tensor([0.0, 1.0, 5.0])).sum(-1).mean()
    loss = _loss(monkeypatch, logits, trajectory_token_loss_weight=0.0, expected_distance_loss_weight=1.0)
    assert torch.allclose(loss, expected_ade, atol=1e-5)


def test_pdm_score_loss_adds_the_weighted_bce(monkeypatch):
    logits = torch.randn(B, K)
    score_logits, score_targets = torch.randn(B, 8, K), torch.rand(B, 8, K)
    ce = _loss(monkeypatch, logits)
    total = _loss(monkeypatch, logits, {"pdm_score_logits": score_logits}, {"pdm_scores": score_targets},
                  pdm_score_loss_weight=0.5)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(score_logits, score_targets)
    assert torch.allclose(total, ce + 0.5 * bce, atol=1e-5)


def test_equal_head_weights_reproduce_the_unweighted_bce(monkeypatch):
    logits = torch.randn(B, K)
    score_logits, score_targets = torch.randn(B, 8, K), torch.rand(B, 8, K)
    args = (monkeypatch, logits, {"pdm_score_logits": score_logits}, {"pdm_scores": score_targets})
    plain = _loss(*args, pdm_score_loss_weight=1.0)
    assert torch.allclose(_loss(*args, pdm_score_loss_weight=1.0, pdm_score_head_weights=(3.0,) * 8), plain, atol=1e-6)


def test_head_weights_give_the_weighted_mean_of_per_head_bce(monkeypatch):
    logits = torch.randn(B, K)
    score_logits, score_targets = torch.randn(B, 8, K), torch.rand(B, 8, K)
    weights = (1.0, 4.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0)
    ce = _loss(monkeypatch, logits)
    total = _loss(monkeypatch, logits, {"pdm_score_logits": score_logits}, {"pdm_scores": score_targets},
                  pdm_score_loss_weight=1.0, pdm_score_head_weights=weights)
    per_head = torch.nn.functional.binary_cross_entropy_with_logits(score_logits, score_targets, reduction="none")
    w = torch.tensor(weights)
    assert torch.allclose(total, ce + (per_head.mean(dim=(0, 2)) * w).sum() / w.sum(), atol=1e-5)


def test_aggregate_matches_the_pdm_formula():
    sub = torch.ones(1, 8, 3)
    sub[0, 1, 1] = 0.0          # entry 1 leaves the drivable area: total 0
    sub[0, 4, 2] = 0.0          # entry 2 makes no progress: total (5 + 2 + 2) / 14
    total = loss_module.aggregate_pdm_scores(sub)
    assert torch.allclose(total, torch.tensor([[1.0, 0.0, 9.0 / 14.0]]))


def test_total_score_loss_adds_the_bce_of_the_aggregate(monkeypatch):
    logits = torch.randn(B, K)
    score_logits, score_targets = torch.randn(B, 8, K), torch.rand(B, 8, K)
    args = (monkeypatch, logits, {"pdm_score_logits": score_logits}, {"pdm_scores": score_targets})
    base = _loss(*args, pdm_score_loss_weight=1.0)
    total = _loss(*args, pdm_score_loss_weight=1.0, pdm_total_score_loss_weight=0.5)
    pred = loss_module.aggregate_pdm_scores(score_logits.sigmoid()).clamp(1e-6, 1 - 1e-6)
    target = loss_module.aggregate_pdm_scores(score_targets)
    expected = torch.nn.functional.binary_cross_entropy(pred, target)
    assert torch.allclose(total, base + 0.5 * expected, atol=1e-5)


def test_total_score_gradient_is_largest_for_the_failing_multiplicative_head(monkeypatch):
    # Every sub-score predicted ~1, but the target says the entry leaves the drivable area: the total-score term
    # pushes the drivable-area head much harder than lane keeping, which also misses but barely changes the total.
    score_logits = torch.full((B, 8, K), 4.0, requires_grad=True)
    targets = torch.ones(B, 8, K)
    targets[:, 1] = 0.0                       # drivable area fails
    targets[:, 6] = 0.0                       # lane keeping fails
    monkeypatch.setattr(loss_module, "_get_codebook", lambda config, device: _codebook())
    preds = {"trajectory_logits": torch.randn(B, K), "trajectory": torch.zeros(B, T, 3), "similarity": torch.tensor(0.0),
             "query_variance": torch.tensor(0.0), "relative_variance": torch.tensor(0.0), "pdm_score_logits": score_logits}
    flags = dict(pdm_score_loss_weight=1e-9, pdm_total_score_loss_weight=1.0, trajectory_token_loss_weight=0.0)
    loss = loss_module.vision_llm_lora_loss({"trajectory": torch.zeros(B, T, 3), "pdm_scores": targets}, preds,
                                            VisionLLMLoRAConfig(**flags))[0]
    loss.backward()
    grad = score_logits.grad.abs().mean(dim=(0, 2))
    assert grad[1] > 5 * grad[6]


def test_score_weighted_labels_remove_failing_entries_from_the_target(monkeypatch):
    # Entry 0 (= GT) leaves the drivable area: its label mass moves to entry 1 (1 m off, scores 1); entry 2 scores 1
    # too but is 5 m away, so it keeps only its tiny soft-label share.
    logits = torch.randn(B, K)
    scores = torch.ones(B, 8, K)
    scores[:, 1, 0] = 0.0
    plain = torch.softmax(-torch.tensor([0.0, 8.0, 200.0]), -1)          # d = sum over 8 steps of dx^2, T = 1
    target = plain * torch.tensor([0.0, 1.0, 1.0])
    target = target / target.sum()
    expected = -(target * torch.log_softmax(logits, -1)).sum(-1).mean()
    loss = _loss(monkeypatch, logits, targets_extra={"pdm_scores": scores}, score_weighted_labels=True)
    assert torch.allclose(loss, expected, atol=1e-5)


def test_score_weighted_labels_keep_plain_labels_when_nothing_scores(monkeypatch):
    logits = torch.randn(B, K)
    scores = torch.ones(B, 8, K)
    scores[:, 0] = 0.0                        # every entry collides
    plain = _loss(monkeypatch, logits)
    weighted = _loss(monkeypatch, logits, targets_extra={"pdm_scores": scores}, score_weighted_labels=True)
    assert torch.allclose(weighted, plain, atol=1e-6)


def test_score_weighted_labels_with_perfect_scores_change_nothing(monkeypatch):
    logits = torch.randn(B, K)
    plain = _loss(monkeypatch, logits)
    weighted = _loss(monkeypatch, logits, targets_extra={"pdm_scores": torch.ones(B, 8, K)}, score_weighted_labels=True)
    assert torch.allclose(weighted, plain, atol=1e-6)


def test_proxy_scores_of_argmax_and_head_choice_are_logged(monkeypatch):
    logits = torch.tensor([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]])    # argmax: entry 1, then entry 0
    scores = torch.ones(B, 8, K)
    scores[0, 1, 1] = 0.0                                          # scene 0: the argmax entry leaves the road
    score_logits = torch.full((B, 8, K), 5.0)
    score_logits[:, 1, 1] = -5.0                                   # heads predict entry 1 off-road: they pick 0 or 2
    _, extra = _extra(monkeypatch, logits, {"pdm_score_logits": score_logits}, {"pdm_scores": scores},
                      pdm_score_loss_weight=1.0)
    assert torch.isclose(extra["pdm_scores/argmax_score"], torch.tensor(0.5))
    assert torch.isclose(extra["pdm_scores/argmax_score_gap"], torch.tensor(0.5))
    assert torch.isclose(extra["pdm_scores/head_score"], torch.tensor(1.0))
    assert torch.isclose(extra["pdm_scores/head_score_gap"], torch.tensor(0.0))
    expected = (torch.softmax(logits, -1) * loss_module.aggregate_pdm_scores(scores)).sum(-1).mean()
    assert torch.isclose(extra["pdm_scores/expected_score"], expected, atol=1e-6)


def test_refinement_loss_is_the_l1_error_with_wrapped_yaw(monkeypatch):
    logits = torch.randn(B, K)
    refined = torch.zeros(B, T, 3)
    refined[..., 0] = 0.3                     # 0.3 m off in x
    refined[..., 2] = 2 * torch.pi - 0.1      # 0.1 rad off in yaw, after wrapping
    ce = _loss(monkeypatch, logits)
    total = _loss(monkeypatch, logits, {"trajectory": refined}, refinement_loss_weight=1.0)
    assert torch.allclose(total - ce, torch.tensor((0.3 + 0.0 + 0.1) / 3), atol=1e-4)


def test_expected_distance_term_is_off_by_default(monkeypatch):
    logits = torch.randn(B, K)
    assert torch.equal(_loss(monkeypatch, logits), _loss(monkeypatch, logits, expected_distance_loss_weight=0.0))


def test_expected_distance_term_adds_the_expected_ade(monkeypatch):
    logits = torch.randn(B, K)
    ce = _loss(monkeypatch, logits)
    expected_ade = (torch.softmax(logits, -1) * torch.tensor([0.0, 1.0, 5.0])).sum(-1).mean()
    total = _loss(monkeypatch, logits, expected_distance_loss_weight=0.5)
    assert torch.allclose(total, ce + 0.5 * expected_ade, atol=1e-5)


def test_expected_distance_gradient_moves_mass_towards_the_closest_entry(monkeypatch):
    logits = torch.zeros(B, K, requires_grad=True)
    monkeypatch.setattr(loss_module, "_get_codebook", lambda config, device: _codebook())
    ce_only = _loss(monkeypatch, logits)
    with_term = _loss(monkeypatch, logits, expected_distance_loss_weight=1.0)
    (with_term - ce_only).backward()
    grad = logits.grad[0]
    assert grad[0] < 0 < grad[2]  # descent raises the GT entry's logit and lowers the far entry's


def _extra(monkeypatch, logits, preds_extra=None, targets_extra=None, **flags):
    monkeypatch.setattr(loss_module, "_get_codebook", lambda config, device: _codebook())
    preds = {
        "trajectory_logits": logits, "trajectory": torch.zeros(B, T, 3), "similarity": torch.tensor(0.0),
        "query_variance": torch.tensor(0.0), "relative_variance": torch.tensor(0.0), **(preds_extra or {}),
    }
    targets = {"trajectory": torch.zeros(B, T, 3), **(targets_extra or {})}
    out = loss_module.vision_llm_lora_loss(targets, preds, VisionLLMLoRAConfig(**flags))
    assert len(out) == 8
    return out[0], out[7]


def test_loss_terms_are_logged_unweighted_and_sum_to_the_loss(monkeypatch):
    logits = torch.randn(B, K)
    loss, extra = _extra(monkeypatch, logits, trajectory_token_loss_weight=0.5, expected_distance_loss_weight=2.0)
    expected = 0.5 * extra["loss_terms/soft_ce"] + 2.0 * extra["loss_terms/expected_ade"]
    assert torch.allclose(loss, expected, atol=1e-5)
    # expected ADE is logged even when it is not in the loss
    _, extra_off = _extra(monkeypatch, logits)
    assert torch.allclose(extra_off["loss_terms/expected_ade"], extra["loss_terms/expected_ade"])


def test_per_metric_sub_score_quality_is_logged(monkeypatch):
    targets = torch.zeros(B, 8, K)
    targets[:, 0] = 1.0                                   # no_at_fault_collisions: all candidates collision-free
    score_logits = torch.full((B, 8, K), -10.0)
    score_logits[:, 0] = 10.0                             # perfect prediction for every metric
    _, extra = _extra(monkeypatch, torch.randn(B, K), {"pdm_score_logits": score_logits}, {"pdm_scores": targets},
                      pdm_score_loss_weight=1.0)
    assert extra["pdm_scores/no_at_fault_collisions_acc"] == 1.0
    assert extra["pdm_scores/no_at_fault_collisions_mae"] < 1e-3
    assert "pdm_scores/ego_progress_acc" not in extra     # continuous: BCE and MAE only
    assert "pdm_scores/ego_progress_mae" in extra and "loss_terms/score_bce" in extra


def test_anchor_errors_are_logged_next_to_the_refined_ones(monkeypatch):
    anchor = torch.zeros(B, T, 3)
    anchor[..., 0] = 2.0                                  # unrefined entry 2 m off, refined output exact
    _, extra = _extra(monkeypatch, torch.randn(B, K), {"anchor_trajectory": anchor}, refinement_loss_weight=1.0)
    assert torch.allclose(extra["refine/anchor_ade"], torch.tensor(2.0))
    assert torch.allclose(extra["refine/anchor_fde"], torch.tensor(2.0))
    assert torch.allclose(extra["loss_terms/refine_l1"], torch.tensor(0.0))


def test_score_targets_for_logging_only_leave_the_loss_unchanged(monkeypatch):
    # log_pdm_score_metrics loads the cached score targets for a policy-only run: the loss must stay the plain
    # policy loss, and the proxy-score metrics must appear.
    monkeypatch.setattr(loss_module, "_get_codebook", lambda config, device: _codebook())
    logits = torch.randn(B, K)
    preds = {"trajectory_logits": logits, "trajectory": torch.zeros(B, T, 3), "similarity": torch.tensor(0.0),
             "query_variance": torch.tensor(0.0), "relative_variance": torch.tensor(0.0)}
    config = VisionLLMLoRAConfig(expected_distance_loss_weight=1.0, log_pdm_score_metrics=True)
    plain = loss_module.vision_llm_lora_loss({"trajectory": torch.zeros(B, T, 3)}, preds, config)
    targets = {"trajectory": torch.zeros(B, T, 3), "pdm_scores": torch.rand(B, 8, K)}
    logged = loss_module.vision_llm_lora_loss(targets, preds, config)
    assert torch.allclose(logged[0], plain[0])
    assert "pdm_scores/argmax_score_gap" in logged[-1] and "pdm_scores/argmax_score_gap" not in plain[-1]
