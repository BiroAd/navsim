"""
CPU-only checks for the precomputed PDM sub-score targets (scripts/training/cache_pdm_score_targets.py) and the
agent's target builders. No network, dataset or HF weights.
Run: pytest tests/test_vision_llm_lora_pdm_score_targets.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from navsim.agents.vision_llm_lora.vision_llm_lora_features import PDM_SCORE_KEYS, PDMScoreTargetBuilder
from navsim.planning.training.dataset import load_feature_target_from_pickle


def _script():
    path = Path(__file__).parents[1] / "scripts/training/cache_pdm_score_targets.py"
    spec = importlib.util.spec_from_file_location("cache_pdm_score_targets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _log_scores(tokens, k=6):
    rng = np.random.default_rng(0)
    scores = {key: rng.random((len(tokens), k)).astype(np.float32) for key in PDM_SCORE_KEYS}
    scores["tokens"] = np.array(tokens)
    return scores


def test_scene_targets_stack_metrics_in_key_order():
    scores = _log_scores(["a1", "b2"])
    targets = _script().scene_targets(scores)

    assert list(targets) == ["a1", "b2"]
    assert targets["b2"]["pdm_scores"].shape == (len(PDM_SCORE_KEYS), 6)
    assert targets["b2"]["pdm_scores"].dtype == torch.float16
    for m, key in enumerate(PDM_SCORE_KEYS):
        assert np.allclose(targets["b2"]["pdm_scores"][m].numpy(), scores[key][1], atol=1e-3)


def test_targets_land_next_to_the_cached_features(tmp_path):
    script = _script()
    scores_dir, cache = tmp_path / "scores", tmp_path / "cache"
    scores_dir.mkdir()
    np.savez_compressed(scores_dir / "log_x.npz", **_log_scores(["a1", "b2"]))
    (cache / "log_x" / "a1").mkdir(parents=True)  # b2 has no folder in the cache

    written, present, missing = script._write_log((scores_dir / "log_x.npz", cache))
    again = script._write_log((scores_dir / "log_x.npz", cache))

    assert (written, present, missing) == (1, 0, ["b2"])
    assert again[:2] == (0, 1)  # existing files are skipped
    loaded = load_feature_target_from_pickle(cache / "log_x" / "a1" / script.FILE_NAME)
    assert loaded["pdm_scores"].shape == (len(PDM_SCORE_KEYS), 6)


def test_builder_name_matches_the_written_file_and_cannot_compute_live():
    assert _script().FILE_NAME == PDMScoreTargetBuilder().get_unique_name() + ".gz"
    with pytest.raises(NotImplementedError):
        PDMScoreTargetBuilder().compute_targets(None)
