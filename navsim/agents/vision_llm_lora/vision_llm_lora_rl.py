"""
RL on the policy logits of vision_llm_lora with rewards looked up from the precomputed codebook scores
(exp/codebook_scores_navtrain: the PDM score of every codebook entry in every navtrain scene, log-replay traffic, no
two-frame comfort). One decision per scene (a contextual bandit), and the reward of every action is known, so:

- expected_reward_kl_loss: exact on-policy objective, -E_pi[r] + beta * KL(pi || pi_ref), summed over all K entries (no
  sampling). Its optimum is pi* proportional to pi_ref * exp(r / beta).
- grpo_loss: the sampled variant (GRPO): G entries per scene drawn from pi, group-normalized advantages, policy
  gradient on their log-probabilities, plus the exact KL. One gradient step per sample (on-policy), so the PPO ratio is
  1 and clipping never acts; it is left out.

PolicyHead reproduces the policy logits of TrajectoryClassifier / TrajectoryEmbeddingScorer from the fused hidden state
(the query_fusion output) and the trajectory embeddings, so that a policy-head-only run can train on cached features
(scripts/training/cache_policy_head_features.py, scripts/training/rl_policy_head.py). Parameter names follow the model
checkpoints (POLICY_HEAD_PARAMETERS).
"""
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Checkpoint key of each PolicyHead parameter (Lightning state dict of the agent).
POLICY_HEAD_KEYS = {
    "scene_proj.weight": "agent._vlm_model.traj_classifier.classifier.scene_proj.weight",
    "scene_proj.bias": "agent._vlm_model.traj_classifier.classifier.scene_proj.bias",
    "trajectory_bias": "agent._vlm_model.traj_classifier.classifier.trajectory_bias",
    "logit_scale": "agent._vlm_model.traj_classifier.logit_scale",
}


class PolicyHead(nn.Module):
    """
    Policy logits from the fused hidden state, as TrajectoryClassifier.forward computes them with a
    TrajectoryEmbeddingScorer: scene_proj(fused) @ E^T / sqrt(D) + trajectory_bias, soft-capped at 30 by tanh, times
    exp(logit_scale clamped to [-2, 2]). E (K, D) = the frozen trajectory embeddings (embed_trajectories()).

    forward(fused: (B, in_dim)) -> logits (B, K)
    """

    def __init__(self, trajectory_embeddings: torch.Tensor, in_dim: int):
        super().__init__()
        num_trajectories, embed_dim = trajectory_embeddings.shape
        self.register_buffer("trajectory_embeddings", trajectory_embeddings.detach().float().clone())
        self.scene_proj = nn.Linear(in_dim, embed_dim)
        self.trajectory_bias = nn.Parameter(torch.zeros(num_trajectories))
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.scale = embed_dim ** -0.5

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        logits = self.scene_proj(fused.float()) @ self.trajectory_embeddings.T * self.scale + self.trajectory_bias
        logits = torch.tanh(logits / 30.0) * 30.0
        return logits * self.logit_scale.clamp(-2.0, 2.0).exp()

    def load_from_checkpoint_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Copy the policy parameters from an agent state dict (POLICY_HEAD_KEYS); fails if one is missing."""
        own = self.state_dict()
        missing = [key for key in POLICY_HEAD_KEYS.values() if key not in state]
        if missing:
            raise KeyError(f"policy parameters missing from the checkpoint: {missing}")
        own.update({name: state[key].float() for name, key in POLICY_HEAD_KEYS.items()})
        self.load_state_dict(own)

    def checkpoint_state(self) -> Dict[str, torch.Tensor]:
        """The policy parameters under their agent checkpoint keys."""
        params = dict(self.named_parameters())
        return {key: params[name].detach().float().cpu().clone() for name, key in POLICY_HEAD_KEYS.items()}


def expected_reward_kl_loss(
    logits: torch.Tensor, rewards: torch.Tensor, ref_logits: torch.Tensor, beta: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Exact objective over all K entries: mean over scenes of -sum_k pi_k r_k + beta * KL(pi || pi_ref).

    logits, ref_logits: (B, K); rewards: (B, K) in [0, 1]. Returns the loss and detached statistics.
    """
    log_p = logits.float().log_softmax(-1)
    p = log_p.exp()
    expected = (p * rewards.float()).sum(-1)
    kl = (p * (log_p - ref_logits.float().log_softmax(-1))).sum(-1)
    loss = (-expected + beta * kl).mean()
    return loss, {"expected_reward": expected.mean().detach(), "kl": kl.mean().detach()}


def grpo_loss(
    logits: torch.Tensor,
    rewards: torch.Tensor,
    ref_logits: torch.Tensor,
    beta: float,
    group_size: int,
    generator: Optional[torch.Generator] = None,
    eps: float = 1e-6,
    normalize_std: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    GRPO with looked-up rewards: per scene, group_size entries drawn from pi (with replacement); advantage = (r - group
    mean) / (group std + eps); loss = -mean(advantage * log pi(entry)) + beta * KL(pi || pi_ref) (exact KL over all K).
    A group with equal rewards has zero advantage and only the KL acts. normalize_std=False keeps the advantage at
    r - group mean (as in Dr. GRPO), so the gradient scales with the reward differences, like the exact objective's.
    """
    log_p = logits.float().log_softmax(-1)
    with torch.no_grad():
        samples = torch.multinomial(log_p.exp(), group_size, replacement=True, generator=generator)  # (B, G)
        sampled_rewards = rewards.float().gather(1, samples)
        advantage = sampled_rewards - sampled_rewards.mean(1, keepdim=True)
        if normalize_std:
            advantage = advantage / (sampled_rewards.std(1, keepdim=True) + eps)
    policy_gradient = -(advantage * log_p.gather(1, samples)).mean()
    p = log_p.exp()
    kl = (p * (log_p - ref_logits.float().log_softmax(-1))).sum(-1)
    loss = policy_gradient + beta * kl.mean()
    stats = {
        "expected_reward": (p * rewards.float()).sum(-1).mean().detach(),
        "kl": kl.mean().detach(),
        "sampled_reward": sampled_rewards.mean(),
    }
    return loss, stats


@torch.no_grad()
def selection_metrics(
    logits: torch.Tensor,
    rewards: torch.Tensor,
    shortlist: Optional[torch.Tensor] = None,
    shortlist_sizes: Sequence[int] = (),
    ref_logits: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Per-scene evaluation of a policy against looked-up rewards (B scenes):
    argmax (score of the most likely entry), expected (E_pi[r]), entropy, argmax_zero (argmax scored 0); with
    shortlist (B, >= k) = entry ids sorted by the heads' predicted total, best first: shortlist_<k> = score of the most
    likely entry among the first k, and shortlist_<k>_zero; with ref_logits: kl (KL(pi || pi_ref)) and changed (argmax
    differs from the reference argmax).
    """
    log_p = logits.float().log_softmax(-1)
    p = log_p.exp()
    rewards = rewards.float()
    best = log_p.argmax(-1, keepdim=True)
    argmax = rewards.gather(1, best)[:, 0]
    out = {
        "argmax": argmax,
        "argmax_zero": (argmax == 0).float(),
        "expected": (p * rewards).sum(-1),
        "entropy": -(p * log_p).sum(-1),
    }
    for k in shortlist_sizes:
        short = shortlist[:, :k].long()
        pick = short.gather(1, log_p.gather(1, short).argmax(-1, keepdim=True))
        score = rewards.gather(1, pick)[:, 0]
        out[f"shortlist_{k}"] = score
        out[f"shortlist_{k}_zero"] = (score == 0).float()
    if ref_logits is not None:
        ref_log_p = ref_logits.float().log_softmax(-1)
        out["kl"] = (p * (log_p - ref_log_p)).sum(-1)
        out["changed"] = (best[:, 0] != ref_log_p.argmax(-1)).float()
    return out
