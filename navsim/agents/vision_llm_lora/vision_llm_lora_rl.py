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


# ── RL with the backbone trainable (RL-3; config.rl_loss_weight > 0) ───────────────────────────────────────────────
# The policy side of the model (LoRA adapters, planner queries, query fusion, policy head; config.training_phase sets
# which) is trained with expected_reward_kl_loss inside the normal training loop. Per scene it reads, from a separate
# cache root written by scripts/training/cache_rl_reference.py (<root>/<log>/<token>/vision_llm_rl_reference.gz):
#   rl_reward (K,) float16: the unfiltered PDM total of every codebook entry (the codebook score table's "score", as in
#       RL-1; the training cache's pdm_scores have the human-penalty filter applied and are not used)
#   rl_reference_fused (D,) float16: the start model's fused state (rl_features cache); the reference policy is the
#       start model's policy head (rl_reference_head_path) on it, so no frozen copy of the backbone is needed
#   rl_shortlist (S,) int16: the start model's sub-score heads' shortlist, best first; the scorer stays the start model
#       at evaluation, so shortlist_<k> on it is the selection proxy
RL_REFERENCE_NAME = "vision_llm_rl_reference"
_REFERENCE_HEADS: Dict[Tuple[str, str], PolicyHead] = {}


class RLReferenceTargetBuilder:
    """Cache-only target builder of the RL-3 per-scene data (see above); CacheOnlyDataset reads it from cache_root."""

    def __init__(self, cache_root: str):
        self.cache_root = cache_root

    def get_unique_name(self) -> str:
        return RL_REFERENCE_NAME

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("RL reference targets are written by scripts/training/cache_rl_reference.py")


def reference_head(path: str, device: torch.device) -> PolicyHead:
    """The frozen start policy head from an rl_features head.pt (trajectory_embeddings, policy_state), float32."""
    key = (str(path), str(device))
    if key not in _REFERENCE_HEADS:
        info = torch.load(path, map_location="cpu", weights_only=False)
        state = info["policy_state"]
        in_dim = state[POLICY_HEAD_KEYS["scene_proj.weight"]].shape[1]
        head = PolicyHead(info["trajectory_embeddings"], in_dim=in_dim)
        head.load_from_checkpoint_state(state)
        _REFERENCE_HEADS[key] = head.to(device).eval().requires_grad_(False)
    return _REFERENCE_HEADS[key]


def rl_loss_terms(
    token_logits: torch.Tensor, targets: Dict[str, torch.Tensor], beta: float, reference_head_path: str,
    shortlist_sizes: Sequence[int] = (64, 256), ec: Optional[torch.Tensor] = None, ec_weight: float = 0.0,
    objective: str = "exact", group_size: int = 32, normalize_std: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    RL-3 loss: expected_reward_kl_loss of the policy logits (B, K) against the cached rewards, KL to the start policy.
    objective "grpo": grpo_loss instead (group_size samples per scene from pi, normalize_std = standard GRPO, False =
    Dr. GRPO), same rewards and KL; adds rl/sampled_reward.
    Metrics (logged as <train|val>/rl/<name>): expected_reward, kl, argmax, shortlist_<k> (score of the policy's most
    likely entry among the start model's top-k), argmax_zero, changed (argmax differs from the start policy's), entropy,
    and shortlist_<k>_gap = 1 - shortlist_<k> (a "min" checkpoint monitor).
    With ec (B, K) in {0, 1} (EC of every entry against the previous plan) and ec_weight > 0, the reward becomes
    ec_reward(): the official EPDMS weighting for ec_weight = 2; all metrics above then use it, and ec_argmax, ec_expected
    (EC of the argmax / under pi) and argmax_no_ec (the argmax under the reward without EC) are added.
    """
    rewards = targets["rl_reward"].float()
    base_rewards = rewards
    if ec is not None and ec_weight > 0:
        rewards = ec_reward(rewards, targets["rl_multiplier"].float(), ec.float(), ec_weight)
    with torch.no_grad():
        ref_logits = reference_head(reference_head_path, token_logits.device)(targets["rl_reference_fused"].float())
    if objective == "exact":
        loss, stats = expected_reward_kl_loss(token_logits, rewards, ref_logits, beta)
    elif objective == "grpo":
        loss, stats = grpo_loss(token_logits, rewards, ref_logits, beta, group_size, normalize_std=normalize_std)
    else:
        raise ValueError(f"RL objective must be 'exact' or 'grpo', got {objective!r}")
    with torch.no_grad():
        metrics = selection_metrics(token_logits.detach(), rewards, targets["rl_shortlist"], shortlist_sizes, ref_logits)
    extra = {f"rl/{name}": value for name, value in stats.items()}
    for name in ("argmax", "argmax_zero", "changed", "entropy", *(f"shortlist_{k}" for k in shortlist_sizes)):
        extra[f"rl/{name}"] = metrics[name].mean()
    for k in shortlist_sizes:
        extra[f"rl/shortlist_{k}_gap"] = 1.0 - extra[f"rl/shortlist_{k}"]
    if ec is not None and ec_weight > 0:
        with torch.no_grad():
            log_p = token_logits.detach().float().log_softmax(-1)
            best = log_p.argmax(-1, keepdim=True)
            extra["rl/ec_argmax"] = ec.float().gather(1, best).mean()
            extra["rl/ec_expected"] = (log_p.exp() * ec.float()).sum(-1).mean()
            extra["rl/ec_any"] = (ec > 0).any(-1).float().mean()
            extra["rl/argmax_no_ec"] = base_rewards.gather(1, best).mean()
    return loss, extra


def ec_reward(rewards: torch.Tensor, multiplier: torch.Tensor, ec: torch.Tensor, ec_weight: float) -> torch.Tensor:
    """
    M * (5 EP + 5 TTC + 2 LK + 2 HC + w * EC) / (14 + w) from the cached total r = M * (5 EP + 5 TTC + 2 LK + 2 HC) / 14:
    (14 r + w M EC) / (14 + w). multiplier = M (rl_multiplier: the product of the stored multiplicative sub-scores, set
    to 0 where r = 0; the stored sub-scores have the human-penalty filter applied and r does not, so where the filter
    changed a sub-score M can be off; cache_rl_reference.py).
    """
    return (14.0 * rewards + ec_weight * multiplier * ec) / (14.0 + ec_weight)
