"""
select_trajectory "local" / "local_v1" (discrete test-time search around the policy's entry): matches a brute-force
reference, reduces to argmax for local_size 1 or a huge anchor weight and to score_heads for local_size K with no
anchor, uses the v1 formula for "local_v1", and is passed through by the replay agent.
"""
import json
from types import SimpleNamespace

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.vision_llm_lora.vision_llm_lora_agent import select_trajectory
from navsim.agents.vision_llm_lora.vision_llm_lora_loss import (
    V1_MULTIPLICATIVE_PDM_SCORE_KEYS,
    V1_WEIGHTED_PDM_SCORE_WEIGHTS,
    aggregate_pdm_scores,
)
from navsim.agents.vision_llm_lora.vision_llm_lora_precomputed_agent import VisionLLMLoRAPrecomputedAgent

B, M, K = 3, 8, 50


def _inputs(seed=0):
    torch.manual_seed(seed)
    codebook = torch.randn(K, 8, 3) * 3
    logits = torch.randn(B, K) * 2
    predictions = {"trajectory_logits": logits, "trajectory": codebook[logits.argmax(-1)],
                   "pdm_score_logits": torch.randn(B, M, K) * 2}
    return predictions, codebook


def _reference(predictions, codebook, n, lam, v1=False):
    sub = predictions["pdm_score_logits"].sigmoid()
    total = (aggregate_pdm_scores(sub, V1_MULTIPLICATIVE_PDM_SCORE_KEYS, V1_WEIGHTED_PDM_SCORE_WEIGHTS) if v1
             else aggregate_pdm_scores(sub))
    out = []
    for b in range(B):
        a = int(predictions["trajectory_logits"][b].argmax())
        ade = [float(np.linalg.norm(codebook[k, :, :2] - codebook[a, :, :2], axis=-1).mean()) for k in range(K)]
        near = sorted(range(K), key=lambda k: ade[k])[:n]
        out.append(codebook[max(near, key=lambda k: float(total[b, k]) - lam * ade[k])])
    return torch.stack(out)


def test_local_matches_brute_force():
    predictions, codebook = _inputs()
    for n, lam in ((5, 0.0), (10, 0.05), (20, 0.3)):
        got = select_trajectory(predictions, "local", codebook, local_size=n, local_lambda=lam)
        torch.testing.assert_close(got, _reference(predictions, codebook, n, lam))
    got = select_trajectory(predictions, "local_v1", codebook, local_size=10, local_lambda=0.05)
    torch.testing.assert_close(got, _reference(predictions, codebook, 10, 0.05, v1=True))


def test_local_limits():
    predictions, codebook = _inputs(1)
    argmax = select_trajectory(predictions, "argmax", codebook)
    torch.testing.assert_close(select_trajectory(predictions, "local", codebook, local_size=1), argmax)
    torch.testing.assert_close(select_trajectory(predictions, "local", codebook, local_size=K, local_lambda=1e6), argmax)
    torch.testing.assert_close(select_trajectory(predictions, "local", codebook, local_size=K, local_lambda=0.0),
                               select_trajectory(predictions, "score_heads", codebook))


def test_replay_agent_passes_local_parameters(tmp_path):
    predictions, codebook = _inputs(2)
    folder = tmp_path / "pred"
    folder.mkdir()
    np.save(folder / "trajectory_logits.npy", predictions["trajectory_logits"].numpy())
    np.save(folder / "pdm_score_logits.npy", predictions["pdm_score_logits"].numpy())
    np.save(folder / "trajectory.npy", predictions["trajectory"].numpy())
    np.save(folder / "codebook.npy", codebook.numpy())
    tokens = [f"t{i}" for i in range(B)]
    (folder / "tokens.json").write_text(json.dumps(tokens))
    (folder / "meta.json").write_text(json.dumps({"logits_dtype": "float32"}))
    agent = VisionLLMLoRAPrecomputedAgent(TrajectorySampling(time_horizon=4, interval_length=0.5), str(folder), "local",
                                          selection_local_size=7, selection_local_lambda=0.2)
    agent.initialize()
    expected = select_trajectory(predictions, "local", codebook, local_size=7, local_lambda=0.2)
    for i, token in enumerate(tokens):
        out = agent.compute_trajectory(None, SimpleNamespace(scene_metadata=SimpleNamespace(initial_token=token)))
        np.testing.assert_allclose(out.poses, expected[i].numpy(), rtol=1e-6, atol=1e-6)
