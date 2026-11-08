"""
CPU-only check of config.train_policy_head_only (VisionLLMLoRAModel._freeze_all_but_policy_head): only the parameters
the policy logits use stay trainable, the sub-score heads and the shared parts get no gradient, and the backbone
input-grad hooks are removed. Uses a real TrajectoryClassifier on a tiny codebook and stub backbone modules.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from navsim.agents.vision_llm_lora.vision_llm_lora_model import (
    POLICY_HEAD_PARAMETERS,
    TrajectoryClassifier,
    VisionLLMLoRAModel,
)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)
        self.hooks_removed = False
        self.checkpointing_off = False

    def disable_input_require_grads(self):
        self.hooks_removed = True

    def gradient_checkpointing_disable(self):
        self.checkpointing_off = True


class _Stub(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        codebook = torch.randn(32, 8, 3)
        self.traj_classifier = TrajectoryClassifier(
            llm_hidden_size=16, num_trajectories=32, num_queries=2, bottleneck_dim=24, late_ego_fusion=False,
            trajectory_codebook=codebook, trajectory_embed_dim=8, num_score_heads=3,
        )
        self.planner_queries = nn.Parameter(torch.randn(2, 16))
        self.model = SimpleNamespace(language_model=_Backbone(), visual=_Backbone())
        self.lm, self.vis = self.model.language_model, self.model.visual  # registered, so they are in named_parameters
        hook_owner = nn.Linear(2, 2)
        self._vision_grad_hook = hook_owner.register_forward_hook(lambda *a: None)


def _freeze(stub, **overrides):
    config = SimpleNamespace(head_type="trajectory_embedding", gradient_checkpointing=True, **overrides)
    VisionLLMLoRAModel._freeze_all_but_policy_head(stub, config)


def test_only_policy_parameters_stay_trainable():
    stub = _Stub()
    _freeze(stub)
    trainable = {n for n, p in stub.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith(POLICY_HEAD_PARAMETERS) for n in trainable)
    assert any("scene_proj" in n for n in trainable) and any("trajectory_bias" in n for n in trainable)
    assert "traj_classifier.logit_scale" in trainable
    assert not any("score_proj" in n or "score_bias" in n or "trajectory_mlp" in n for n in trainable)
    assert stub._vision_grad_hook is None
    assert stub.lm.hooks_removed and stub.vis.hooks_removed and stub.lm.checkpointing_off


def test_policy_loss_reaches_only_the_policy_and_scores_are_unchanged():
    stub = _Stub()
    _freeze(stub)
    fused = torch.randn(5, 24)
    scores_before = stub.traj_classifier.classifier.score_logits(fused).detach().clone()
    logits = stub.traj_classifier.classifier(fused)
    nn.functional.cross_entropy(logits, torch.randint(0, 32, (5,))).backward()
    for name, p in stub.named_parameters():
        if name.startswith(POLICY_HEAD_PARAMETERS) and "logit_scale" not in name:
            assert p.grad is not None, name
        elif not name.startswith(POLICY_HEAD_PARAMETERS):
            assert p.grad is None, name
    torch.optim.SGD([p for p in stub.parameters() if p.requires_grad], lr=1.0).step()
    assert torch.equal(stub.traj_classifier.classifier.score_logits(fused), scores_before)


def test_needs_the_trajectory_embedding_head():
    config = SimpleNamespace(head_type="linear", gradient_checkpointing=False)
    with pytest.raises(ValueError):
        VisionLLMLoRAModel._freeze_all_but_policy_head(_Stub(), config)
