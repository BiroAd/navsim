"""
CPU checks of RL-3 (config.rl_loss_weight > 0): rl_loss_terms uses the cached start-model fused state through the
frozen reference head (KL 0 when the policy equals the start policy), its shortlist metric picks the policy's most
likely entry among the start model's top k, vision_llm_lora_loss adds the term only when enabled, and the replay agent
combines a policy from one prediction folder with the sub-scores of another (scores_path).
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.vision_llm_lora.vision_llm_lora_agent import select_trajectory
from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_loss import vision_llm_lora_loss
from navsim.agents.vision_llm_lora.vision_llm_lora_precomputed_agent import VisionLLMLoRAPrecomputedAgent
from navsim.agents.vision_llm_lora.vision_llm_lora_rl import (
    POLICY_HEAD_KEYS,
    PolicyHead,
    expected_reward_kl_loss,
    rl_loss_terms,
)

REPO = Path(__file__).parents[1]
CODEBOOK = REPO / "navsim/agents/vision_llm_lora/codebook/16384_8step.npy"


def _reference(tmp_path, k, dim=6, embed=4):
    torch.manual_seed(0)
    head = PolicyHead(torch.randn(k, embed), in_dim=dim)
    with torch.no_grad():
        head.trajectory_bias.normal_()
        head.logit_scale.fill_(0.3)
    path = tmp_path / "head.pt"
    torch.save({"trajectory_embeddings": head.trajectory_embeddings, "policy_state": head.checkpoint_state()}, path)
    return head, str(path)


def test_rl_loss_terms_match_the_exact_objective(tmp_path):
    k, b = 32, 3
    head, path = _reference(tmp_path, k)
    fused = torch.randn(b, 6).half()
    rewards = torch.rand(b, k).half()
    shortlist = torch.stack([torch.randperm(k) for _ in range(b)]).short()
    targets = {"rl_reward": rewards, "rl_reference_fused": fused, "rl_shortlist": shortlist}
    ref_logits = head(fused.float()).detach()

    # policy = start policy: KL 0, nothing changed
    loss, extra = rl_loss_terms(ref_logits.clone().requires_grad_(True), targets, 0.02, path, (4, k))
    assert extra["rl/kl"].abs() < 1e-5 and extra["rl/changed"] == 0
    assert torch.isclose(extra["rl/shortlist_32"], extra["rl/argmax"])  # the full list = plain argmax

    logits = torch.randn(b, k, requires_grad=True)
    loss, extra = rl_loss_terms(logits, targets, 0.1, path, (4,))
    expected, _ = expected_reward_kl_loss(logits, rewards.float(), ref_logits, 0.1)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    top4 = shortlist[:, :4].long()
    pick = top4.gather(1, logits.detach().gather(1, top4).argmax(1, keepdim=True))
    torch.testing.assert_close(extra["rl/shortlist_4"], rewards.float().gather(1, pick).mean())
    torch.testing.assert_close(extra["rl/shortlist_4_gap"], 1 - extra["rl/shortlist_4"])


def test_loss_adds_the_rl_term_only_when_enabled(tmp_path):
    k, b = 16384, 2
    _, path = _reference(tmp_path, k)
    torch.manual_seed(1)
    logits = torch.randn(b, k, requires_grad=True)
    predictions = {"trajectory_logits": logits, "trajectory": torch.zeros(b, 8, 3), "similarity": torch.zeros(()),
                   "query_variance": torch.zeros(()), "relative_variance": torch.zeros(())}
    targets = {"trajectory": torch.zeros(b, 8, 3), "rl_reward": torch.rand(b, k).half(),
               "rl_reference_fused": torch.randn(b, 6).half(), "rl_shortlist": torch.arange(1024).repeat(b, 1).short()}
    base = dict(codebook_path=str(CODEBOOK), trajectory_token_loss_weight=0.0)
    off = vision_llm_lora_loss(targets, predictions, VisionLLMLoRAConfig(**base))
    assert not any(name.startswith("rl/") for name in off[-1])
    on = vision_llm_lora_loss(targets, predictions, VisionLLMLoRAConfig(
        **base, rl_loss_weight=2.0, rl_beta=0.05, rl_reference_head_path=path, rl_reference_cache_path="unused"))
    torch.testing.assert_close(on[0] - off[0], 2.0 * on[-1]["loss_terms/rl"])
    assert {"rl/kl", "rl/expected_reward", "rl/shortlist_256", "rl/shortlist_256_gap"} <= set(on[-1])


def _save_predictions(folder: Path, outputs, codebook, tokens):
    spec = importlib.util.spec_from_file_location("predict", REPO / "scripts/evaluation/predict_vision_llm_lora.py")
    predict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict)
    folder.mkdir()
    for key, value in outputs.items():
        np.save(folder / f"{key}.npy", predict.to_numpy(value))
    np.save(folder / "codebook.npy", codebook.numpy())
    (folder / "tokens.json").write_text(json.dumps(tokens))
    (folder / "meta.json").write_text(json.dumps({"logits_dtype": "bfloat16"}))


def test_replay_agent_takes_sub_scores_from_scores_path(tmp_path):
    torch.manual_seed(2)
    n, m, k = 4, 8, 64
    codebook = torch.randn(k, 8, 3)
    policy = {"trajectory_logits": torch.randn(n, k).bfloat16(), "trajectory": torch.zeros(n, 8, 3),
              "pdm_score_logits": torch.randn(n, m, k).bfloat16()}  # the RL model's untrained heads
    scorer = {"trajectory_logits": torch.randn(n, k).bfloat16(), "trajectory": torch.zeros(n, 8, 3),
              "pdm_score_logits": torch.randn(n, m, k).bfloat16()}
    tokens = [f"t{i}" for i in range(n)]
    _save_predictions(tmp_path / "rl", policy, codebook, tokens)
    _save_predictions(tmp_path / "start", scorer, codebook, list(reversed(tokens)))  # other row order
    agent = VisionLLMLoRAPrecomputedAgent(TrajectorySampling(time_horizon=4, interval_length=0.5), str(tmp_path / "rl"),
                                          "shortlist", selection_shortlist_size=8, scores_path=str(tmp_path / "start"))
    agent.initialize()
    combined = {"trajectory_logits": policy["trajectory_logits"], "trajectory": policy["trajectory"],
                "pdm_score_logits": scorer["pdm_score_logits"].flip(0)}  # back to the policy folder's order
    expected = select_trajectory(combined, "shortlist", codebook, shortlist_size=8)
    for i, token in enumerate(tokens):
        out = agent.compute_trajectory(None, SimpleNamespace(scene_metadata=SimpleNamespace(initial_token=token)))
        np.testing.assert_array_equal(out.poses, expected[i].numpy())
