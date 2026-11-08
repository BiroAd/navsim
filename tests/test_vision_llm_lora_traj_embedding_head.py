"""
CPU-only checks for the trajectory-embedding head (config.head_type == "trajectory_embedding").
No network, dataset or HF weights: the head runs on random planner-query states and a tiny random codebook.
Run: pytest tests/test_vision_llm_lora_traj_embedding_head.py
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from navsim.agents.vision_llm_lora.vision_llm_lora_model import (
    TrajectoryClassifier,
    TrajectoryEmbeddingScorer,
    TrajectoryRefiner,
)

D, B, K, T = 32, 3, 16, 8


def _head(codebook=None):
    return TrajectoryClassifier(
        D, num_trajectories=K, num_queries=4, bottleneck_dim=24, late_ego_fusion=False,
        trajectory_codebook=codebook, trajectory_embed_dim=16,
    )


def _outputs():
    return SimpleNamespace(last_hidden_state=torch.randn(B, 10, D))


def test_default_head_keeps_the_linear_classifier():
    assert isinstance(_head().classifier, nn.Linear)


def test_embedding_head_gives_one_logit_per_trajectory():
    torch.manual_seed(0)
    head = _head(torch.randn(K, T, 3))
    assert isinstance(head.classifier, TrajectoryEmbeddingScorer)
    logits, *_ = head(_outputs(), None)
    assert logits.shape == (B, K)
    assert torch.isfinite(logits).all()


def test_identical_trajectories_get_identical_logits():
    torch.manual_seed(0)
    codebook = torch.randn(K, T, 3)
    codebook[5] = codebook[4]
    logits, *_ = _head(codebook)(_outputs(), None)
    assert torch.allclose(logits[:, 4], logits[:, 5])
    assert not torch.allclose(logits[:, 4], logits[:, 6])


def test_gradients_reach_scene_and_trajectory_branches():
    torch.manual_seed(0)
    head = _head(torch.randn(K, T, 3))
    with torch.no_grad():
        head.classifier.trajectory_bias.normal_()  # non-zero, so the bias gradient is not trivially uniform
    logits, *_ = head(_outputs(), None)
    torch.log_softmax(logits, -1)[:, 0].sum().backward()
    for name in ("query_fusion.0.weight", "classifier.scene_proj.weight",
                 "classifier.trajectory_mlp.0.weight", "classifier.trajectory_bias"):
        grad = dict(head.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, name


def test_trajectory_embeddings_all_have_the_same_norm():
    torch.manual_seed(0)
    scorer = TrajectoryEmbeddingScorer(torch.randn(K, T, 3) * 10, in_dim=24, embed_dim=16)
    norms = scorer.embed_trajectories().norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, 16 ** 0.5), rtol=1e-3)


def test_embedding_parameters_do_not_grow_with_the_codebook():
    def count(k):
        scorer = TrajectoryEmbeddingScorer(torch.randn(k, T, 3), in_dim=24, embed_dim=16)
        return sum(p.numel() for p in scorer.parameters()) - k  # minus the per-trajectory bias

    assert count(16) == count(1024)


def test_score_heads_give_one_logit_per_metric_and_candidate():
    torch.manual_seed(0)
    head = TrajectoryClassifier(
        D, num_trajectories=K, num_queries=4, bottleneck_dim=24, late_ego_fusion=False,
        trajectory_codebook=torch.randn(K, T, 3), trajectory_embed_dim=16, num_score_heads=8,
    )
    *_, fused = head(_outputs(), None)
    scores = head.classifier.score_logits(fused)
    assert scores.shape == (B, 8, K) and torch.isfinite(scores).all()


def test_score_heads_need_the_embedding_head():
    with pytest.raises(ValueError):
        TrajectoryClassifier(D, num_trajectories=K, num_queries=4, bottleneck_dim=24, late_ego_fusion=False,
                             num_score_heads=8)


def test_refiner_starts_as_identity_and_learns_offsets():
    torch.manual_seed(0)
    codebook = torch.randn(K, T, 3)
    refiner = TrajectoryRefiner(codebook, in_dim=24)
    fused, anchor = torch.randn(B, 24), codebook[:B]
    assert torch.equal(refiner(fused, anchor), torch.zeros(B, T, 3))  # zero-initialized last layer
    (refiner(fused, anchor) - 1.0).abs().mean().backward()
    assert refiner.mlp[-1].weight.grad.abs().sum() > 0


def test_codebook_derived_buffers_are_not_saved_in_checkpoints():
    state = _head(torch.randn(K, T, 3)).state_dict()
    assert not any(key.endswith(("trajectories", "freqs")) for key in state)
