"""
Pairwise score loss (config.pairwise_score_loss_weight): the neighbour table and the loss match brute force, correct /
reversed orderings give accuracy 1 / 0, pairs need targets that differ by >= 0.5, and vision_llm_lora_loss adds the
term only when enabled.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_features import PDM_SCORE_KEYS
from navsim.agents.vision_llm_lora.vision_llm_lora_loss import (
    codebook_neighbours,
    pairwise_score_loss,
    vision_llm_lora_loss,
)

CODEBOOK = Path(__file__).parents[1] / "navsim/agents/vision_llm_lora/codebook/16384_8step.npy"
KEYS = ("no_at_fault_collisions", "drivable_area_compliance")


def _small(seed=0, b=2, k=40):
    torch.manual_seed(seed)
    codebook = torch.randn(k, 8, 3)
    logits = torch.randn(b, len(PDM_SCORE_KEYS), k, requires_grad=True)
    targets = torch.randint(0, 3, (b, len(PDM_SCORE_KEYS), k)).float() / 2  # {0, 0.5, 1}
    entry_ade = torch.rand(b, k)
    return codebook, logits, targets, entry_ade


def test_neighbour_table_matches_brute_force():
    codebook = torch.randn(30, 8, 3)
    table = codebook_neighbours(codebook, 5, chunk=7)
    for i in range(30):
        ade = [float((codebook[j, :, :2] - codebook[i, :, :2]).norm(dim=-1).mean()) if j != i else np.inf
               for j in range(30)]
        assert set(table[i].tolist()) == set(np.argsort(ade)[:5].tolist())


def test_pairwise_loss_matches_brute_force():
    codebook, logits, targets, entry_ade = _small()
    anchors, n, margin = 10, 4, 0.5
    loss, extra = pairwise_score_loss(logits, targets, entry_ade, codebook, KEYS, anchors, n, margin)
    table = codebook_neighbours(codebook, n)
    terms, right = [], []
    for b in range(logits.shape[0]):
        for i in entry_ade[b].argsort()[:anchors].tolist():
            for j in table[i].tolist():
                for key in KEYS:
                    m = PDM_SCORE_KEYS.index(key)
                    dt = float(targets[b, m, i] - targets[b, m, j])
                    if abs(dt) >= 0.5:
                        gap = np.sign(dt) * (logits[b, m, i] - logits[b, m, j])
                        terms.append(F.softplus(margin - gap))
                        right.append(float(gap) > 0)
    torch.testing.assert_close(loss, torch.stack(terms).mean())
    assert abs(float(extra["pdm_scores/pair_acc"]) - np.mean(right)) < 1e-6
    loss.backward()
    assert logits.grad.abs().sum() > 0


def test_pairwise_orderings_and_no_pairs():
    codebook, _, targets, entry_ade = _small(1)
    ordered = (targets * 10).clone().requires_grad_(True)  # logits follow the targets: every pair right
    _, extra = pairwise_score_loss(ordered, targets, entry_ade, codebook, KEYS, 10, 4, 0.5)
    assert float(extra["pdm_scores/pair_acc"]) == 1.0
    _, extra = pairwise_score_loss(-ordered, targets, entry_ade, codebook, KEYS, 10, 4, 0.5)
    assert float(extra["pdm_scores/pair_acc"]) == 0.0
    same = torch.ones_like(targets)  # equal targets: no pairs, zero loss with a gradient path
    logits = torch.randn_like(targets, requires_grad=True)
    loss, extra = pairwise_score_loss(logits, same, entry_ade, codebook, KEYS, 10, 4, 0.5)
    assert float(loss) == 0.0 and float(extra["pdm_scores/pairs_per_scene"]) == 0.0
    loss.backward()


def test_loss_adds_pairwise_term_only_when_enabled():
    k, b = 16384, 2
    torch.manual_seed(3)
    cb = torch.from_numpy(np.load(CODEBOOK)).float()
    predictions = {"trajectory_logits": torch.randn(b, k), "trajectory": torch.zeros(b, 8, 3),
                   "pdm_score_logits": torch.randn(b, len(PDM_SCORE_KEYS), k, requires_grad=True),
                   "similarity": torch.zeros(()), "query_variance": torch.zeros(()), "relative_variance": torch.zeros(())}
    targets = {"trajectory": cb[torch.tensor([5, 900])].clone(),
               "pdm_scores": (torch.rand(b, len(PDM_SCORE_KEYS), k) > 0.3).float()}
    base = dict(codebook_path=str(CODEBOOK), trajectory_token_loss_weight=0.0, pdm_score_loss_weight=1.0)
    off = vision_llm_lora_loss(targets, predictions, VisionLLMLoRAConfig(**base))
    assert "loss_terms/score_pairwise" not in off[-1]
    on = vision_llm_lora_loss(targets, predictions, VisionLLMLoRAConfig(**base, pairwise_score_loss_weight=2.0))
    torch.testing.assert_close(on[0] - off[0], 2.0 * on[-1]["loss_terms/score_pairwise"])
    assert {"pdm_scores/pair_acc", "pdm_scores/pair_acc_drivable_area_compliance"} <= set(on[-1])
    assert float(on[-1]["pdm_scores/pairs_per_scene"]) > 0
