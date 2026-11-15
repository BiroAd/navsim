"""The base rankings of scripts/evaluation/ec_aware_selection.py start with select_trajectory's choice."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

from navsim.agents.vision_llm_lora.vision_llm_lora_agent import select_trajectory
from navsim.agents.vision_llm_lora.vision_llm_lora_loss import PDM_SCORE_KEYS

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("ec_aware_selection", ROOT / "scripts/evaluation/ec_aware_selection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pred(key, values):
    n = values.shape[0]
    return {"row": {f"t{i}": i for i in range(n)}, "arrays": {key: values.numpy()}, "meta": {"logits_dtype": "float32"}}


def test_rankings_match_select_trajectory():
    torch.manual_seed(0)
    b, k, m = 5, 40, len(PDM_SCORE_KEYS)
    policy_logits = torch.randn(b, k) * 3
    score_logits = torch.randn(b, m, k) * 2
    codebook = torch.randn(k, 8, 3)
    script = _script()
    top_policy, top_short, top_mix = script.rankings(
        _pred("trajectory_logits", policy_logits), _pred("pdm_score_logits", score_logits), shortlist_size=8, n_max=6,
        chunk=2, mixture_lambda=10.0,
    )
    predictions = {"trajectory": codebook[policy_logits.argmax(-1)], "trajectory_logits": policy_logits,
                   "pdm_score_logits": score_logits}
    for ranking, selection in ((top_short, "shortlist"), (top_mix, "mixture")):
        chosen = select_trajectory(predictions, selection, codebook, shortlist_size=8, mixture_lambda=10.0)
        assert torch.equal(codebook[torch.from_numpy(ranking[:, 0])], chosen), selection
    assert np.array_equal(top_policy[:, 0], policy_logits.argmax(-1).numpy())
    # mixture ranking is ordered by the objective
    log_p = policy_logits.log_softmax(-1)
    from navsim.agents.vision_llm_lora.vision_llm_lora_loss import aggregate_pdm_scores

    objective = log_p + 10.0 * aggregate_pdm_scores(score_logits.sigmoid()).clamp_min(1e-6).log()
    ordered = objective.gather(1, torch.from_numpy(top_mix))
    assert (ordered[:, :-1] >= ordered[:, 1:]).all()
