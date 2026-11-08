"""
CPU checks of the split evaluation: outputs saved by scripts/evaluation/predict_vision_llm_lora.py (bfloat16 stored
bit-exact as int16) are replayed by VisionLLMLoRAPrecomputedAgent with the same trajectory as select_trajectory on the
original outputs, for every selection rule.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.vision_llm_lora.vision_llm_lora_agent import TRAJECTORY_SELECTIONS, select_trajectory
from navsim.agents.vision_llm_lora.vision_llm_lora_precomputed_agent import VisionLLMLoRAPrecomputedAgent

_spec = importlib.util.spec_from_file_location(
    "predict_vision_llm_lora", Path(__file__).parents[1] / "scripts" / "evaluation" / "predict_vision_llm_lora.py"
)
predict = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(predict)

N, M, K = 5, 8, 64


def _outputs():
    torch.manual_seed(0)
    codebook = torch.randn(K, 8, 3)
    logits = (torch.randn(N, K) * 10).to(torch.bfloat16)
    logits[0, :2] = 37.0  # an exact tie, as bf16 produces
    return codebook, {
        "trajectory_logits": logits,
        "pdm_score_logits": (torch.randn(N, M, K) * 3).to(torch.bfloat16),
        "trajectory": codebook[logits.float().argmax(-1)],
    }


def _save(tmp_path: Path, codebook, outputs, tokens):
    for key, value in outputs.items():
        np.save(tmp_path / f"{key}.npy", predict.to_numpy(value))
    np.save(tmp_path / "codebook.npy", codebook.numpy())
    (tmp_path / "tokens.json").write_text(json.dumps(tokens))
    (tmp_path / "meta.json").write_text(json.dumps({"logits_dtype": "bfloat16"}))


@pytest.mark.parametrize("selection", TRAJECTORY_SELECTIONS)
def test_replay_matches_select_trajectory(tmp_path, selection):
    codebook, outputs = _outputs()
    tokens = [f"token{i}" for i in range(N)]
    _save(tmp_path, codebook, outputs, tokens)
    agent = VisionLLMLoRAPrecomputedAgent(
        TrajectorySampling(time_horizon=4, interval_length=0.5), str(tmp_path), selection, selection_shortlist_size=16
    )
    agent.initialize()
    assert agent.requires_scene
    expected = select_trajectory(outputs, selection, codebook, shortlist_size=16)
    for i, token in enumerate(tokens):
        scene = SimpleNamespace(scene_metadata=SimpleNamespace(initial_token=token))
        trajectory = agent.compute_trajectory(None, scene)
        np.testing.assert_array_equal(trajectory.poses, expected[i].numpy())


def test_bfloat16_is_stored_bit_exact():
    x = (torch.randn(3, 7) * 40).to(torch.bfloat16)
    stored = predict.to_numpy(x)
    assert stored.dtype == np.int16
    assert torch.equal(torch.from_numpy(stored).view(torch.bfloat16), x)


def test_unknown_token_fails(tmp_path):
    codebook, outputs = _outputs()
    _save(tmp_path, codebook, outputs, [f"token{i}" for i in range(N)])
    agent = VisionLLMLoRAPrecomputedAgent(TrajectorySampling(time_horizon=4, interval_length=0.5), str(tmp_path))
    agent.initialize()
    with pytest.raises(KeyError):
        agent.compute_trajectory(None, SimpleNamespace(scene_metadata=SimpleNamespace(initial_token="other")))
