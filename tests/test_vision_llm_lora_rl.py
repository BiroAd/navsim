"""
CPU checks of navsim/agents/vision_llm_lora/vision_llm_lora_rl.py: PolicyHead reproduces TrajectoryClassifier's policy
logits from the fused state, the exact objective's gradient equals the expected GRPO gradient direction, training with
the exact objective converges to pi_ref * exp(r / beta), and the selection metrics pick what they should.
"""
from types import SimpleNamespace

import torch

from navsim.agents.vision_llm_lora.vision_llm_lora_model import TrajectoryClassifier
from navsim.agents.vision_llm_lora.vision_llm_lora_rl import (
    POLICY_HEAD_KEYS,
    PolicyHead,
    expected_reward_kl_loss,
    grpo_loss,
    selection_metrics,
)


def _classifier():
    torch.manual_seed(0)
    head = TrajectoryClassifier(
        16, num_trajectories=32, num_queries=2, bottleneck_dim=24, late_ego_fusion=False,
        trajectory_codebook=torch.randn(32, 8, 3), trajectory_embed_dim=8, num_score_heads=3,
    )
    with torch.no_grad():
        head.logit_scale.fill_(0.7)
        head.classifier.trajectory_bias.normal_()
    return head.eval()


def _policy_head(classifier):
    head = PolicyHead(classifier.classifier.embed_trajectories().detach(), in_dim=24)
    params = dict(classifier.named_parameters())
    state = {key: params[key.removeprefix("agent._vlm_model.traj_classifier.")] for key in POLICY_HEAD_KEYS.values()}
    head.load_from_checkpoint_state(state)
    return head


def test_policy_head_reproduces_the_classifier_logits():
    classifier = _classifier()
    outputs = SimpleNamespace(last_hidden_state=torch.randn(5, 6, 16))
    logits, *_, fused = classifier(outputs, None)
    head = _policy_head(classifier)
    torch.testing.assert_close(head(fused), logits)
    roundtrip = {k.removeprefix("agent._vlm_model.traj_classifier."): v for k, v in head.checkpoint_state().items()}
    params = dict(classifier.named_parameters())
    assert all(torch.equal(v, params[k]) for k, v in roundtrip.items())


def test_exact_gradient_is_the_expected_policy_gradient():
    torch.manual_seed(1)
    logits = torch.randn(3, 10, requires_grad=True)
    rewards = torch.rand(3, 10)
    loss, stats = expected_reward_kl_loss(logits, rewards, logits.detach(), beta=0.0)
    loss.backward()
    p = logits.detach().softmax(-1)
    expected = (p * rewards).sum(-1, keepdim=True)
    # d(-E[r]) / d logit_k = -p_k (r_k - E[r]), averaged over the batch
    torch.testing.assert_close(logits.grad, -(p * (rewards - expected)) / 3)
    torch.testing.assert_close(stats["expected_reward"], expected.mean())
    assert stats["kl"].abs() < 1e-6


def test_exact_objective_converges_to_the_kl_regularized_optimum():
    torch.manual_seed(2)
    ref_logits = torch.randn(2, 6)
    rewards = torch.rand(2, 6)
    beta = 0.2
    logits = ref_logits.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=0.05)
    for _ in range(3000):
        loss, _ = expected_reward_kl_loss(logits, rewards, ref_logits, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    optimum = (ref_logits.log_softmax(-1) + rewards / beta).softmax(-1)
    torch.testing.assert_close(logits.detach().softmax(-1), optimum, atol=2e-3, rtol=0)


def test_grpo_pushes_up_the_better_sampled_entry():
    logits = torch.zeros(1, 4, requires_grad=True)
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    generator = torch.Generator().manual_seed(0)
    loss, stats = grpo_loss(logits, rewards, logits.detach(), beta=0.0, group_size=64, generator=generator)
    loss.backward()
    assert logits.grad[0, 0] < 0 and (logits.grad[0, 1:] > 0).all()  # gradient descent raises entry 0
    assert 0.1 < stats["sampled_reward"] < 0.4


def test_grpo_with_equal_rewards_has_only_the_kl():
    logits = torch.randn(2, 5, requires_grad=True)
    loss, _ = grpo_loss(logits, torch.ones(2, 5), logits.detach(), beta=0.0, group_size=8)
    loss.backward()
    assert torch.allclose(logits.grad, torch.zeros_like(logits.grad))


def test_selection_metrics():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    rewards = torch.tensor([[0.0, 0.5, 0.9, 1.0]])
    shortlist = torch.tensor([[3, 2, 1, 0]])  # heads' ranking, best first
    out = selection_metrics(logits, rewards, shortlist, (1, 2, 4), ref_logits=logits.flip(-1))
    assert out["argmax"].item() == 0.0 and out["argmax_zero"].item() == 1.0
    assert out["shortlist_1"].item() == 1.0  # entry 3
    assert abs(out["shortlist_2"].item() - 0.9) < 1e-6  # most likely of {3, 2}
    assert out["shortlist_4"].item() == 0.0  # most likely overall
    assert out["changed"].item() == 1.0 and out["kl"].item() > 0


def test_grpo_without_std_normalization_scales_with_the_reward_gap():
    logits = torch.zeros(1, 4)
    grads = []
    for gap in (1.0, 0.1):
        x = logits.clone().requires_grad_(True)
        rewards = torch.tensor([[gap, 0.0, 0.0, 0.0]])
        loss, _ = grpo_loss(x, rewards, x.detach(), beta=0.0, group_size=64,
                            generator=torch.Generator().manual_seed(0), normalize_std=False)
        loss.backward()
        grads.append(x.grad.clone())
    torch.testing.assert_close(grads[1], grads[0] * 0.1)
