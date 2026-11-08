"""
CPU-only checks for offline codebook scoring against the official evaluation code paths.
Run: pytest tests/test_vision_llm_lora_codebook_scoring.py
"""

from types import SimpleNamespace

import numpy as np
import pytest
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.vision_llm_lora.vision_llm_lora_codebook_scoring import codebook_to_state_arrays, pairwise_scores
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex

TRAJECTORY_SAMPLING = TrajectorySampling(time_horizon=4, interval_length=0.5)
PROPOSAL_SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)


def _ego_state(heading: float, time_us: int) -> EgoState:
    return EgoState.build_from_rear_axle(
        StateSE2(312.4, -87.1, heading),
        StateVector2D(7.3, 0.2),
        StateVector2D(0.6, -0.3),
        0.04,
        TimePoint(time_us),
        get_pacifica_parameters(),
        True,
        0.07,
        0.01,
        0.002,
    )


# Headings near +-pi make absolute headings cross the wrap; odd time stamps exercise the float -> microsecond
# rounding. Yaw steps between poses stay well below pi (the 16384 codebook's largest is 0.34 rad): at exactly pi,
# unwrap's direction depends on the rounding of the pose transform in either implementation.
@pytest.mark.parametrize("heading, time_us", [(0.3, 1_630_000_000_000_000), (3.1, 1_624_458_381_049_917), (-3.05, 7)])
def test_state_arrays_match_official_transform(heading, time_us):
    rng = np.random.default_rng(0)
    steps = np.cumsum(rng.uniform(0.0, 3.0, (64, 8, 1)), axis=1)
    yaw = np.cumsum(rng.normal(0.0, 0.15, (64, 8, 1)), axis=1)
    poses = np.concatenate([steps, rng.normal(0.0, 1.5, (64, 8, 1)) * steps / 10, yaw], axis=2).astype(np.float32)
    ego = _ego_state(heading, time_us)

    vectorized = codebook_to_state_arrays(poses, ego, TRAJECTORY_SAMPLING, PROPOSAL_SAMPLING)
    official = np.stack(
        [
            get_trajectory_as_array(
                transform_trajectory(Trajectory(p, TRAJECTORY_SAMPLING), ego), PROPOSAL_SAMPLING, ego.time_point
            )
            for p in poses
        ]
    )

    assert vectorized.shape == official.shape == (64, 41, 11)
    heading_diff = np.angle(np.exp(1j * (vectorized[..., 2] - official[..., 2])))
    assert np.abs(heading_diff).max() < 1e-9
    assert np.abs(np.delete(vectorized - official, 2, axis=-1)).max() < 1e-6


def _official_pair(multi, weighted, progress_raw, config, k):
    """PDMScorer's own aggregation on the pair [reference, proposal k]."""
    scorer = PDMScorer(PROPOSAL_SAMPLING, config)
    scorer._multi_metrics = multi[:, [0, k]].copy()
    scorer._weighted_metrics = weighted[:, [0, k]].copy()
    scorer._progress_raw = progress_raw[[0, k]].copy()
    score = scorer._aggregate_pdm_scores()[1]
    return scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, 1], score


def test_pairwise_scores_match_scorer_on_each_pair():
    rng = np.random.default_rng(1)
    n = 40
    multi = rng.choice([0.0, 0.5, 1.0], (len(MultiMetricIndex), n + 1), p=[0.1, 0.1, 0.8])
    weighted = rng.choice([0.0, 1.0], (len(WeightedMetricIndex), n + 1), p=[0.2, 0.8])
    progress_raw = rng.uniform(0.0, 40.0, n + 1)
    progress_raw[1:6] = rng.uniform(0.0, 4.0, 5)  # below the 5 m threshold
    config = PDMScorerConfig()

    pairwise = pairwise_scores(multi, weighted, progress_raw, config)

    for k in range(1, n + 1):
        ego_progress, score = _official_pair(multi, weighted, progress_raw, config, k)
        assert pairwise["ego_progress"][k - 1] == pytest.approx(ego_progress, abs=1e-12)
        assert pairwise["score"][k - 1] == pytest.approx(score, abs=1e-12)


def test_pairwise_progress_differs_from_one_batch():
    # Why entries are not scored in one batch: there, every entry is normalized by the best entry's progress.
    multi = np.ones((len(MultiMetricIndex), 3))
    weighted = np.ones((len(WeightedMetricIndex), 3))
    progress_raw = np.array([10.0, 20.0, 40.0])  # reference, entry 1, entry 2

    pairwise = pairwise_scores(multi, weighted, progress_raw, PDMScorerConfig())

    assert pairwise["ego_progress"].tolist() == [1.0, 1.0]  # each entry beats the reference


def _scoring_script():
    """scripts/evaluation/score_codebook_trajectories.py, which is not inside a package."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts/evaluation/score_codebook_trajectories.py"
    spec = importlib.util.spec_from_file_location("score_codebook_trajectories", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_log_file_stacks_scenes_and_lists_failures(tmp_path):
    script = _scoring_script()
    rng = np.random.default_rng(0)
    scenes = {token: {key: rng.random(5) for key in script.METRIC_KEYS} for token in ("a1", "b2")}

    arrays = script.stack_log_scores([("a1", scenes["a1"]), ("x9", None), ("b2", scenes["b2"])], num_entries=5)
    np.savez_compressed(tmp_path / "log.npz", **arrays)
    loaded = np.load(tmp_path / "log.npz")  # no pickle needed

    assert loaded["tokens"].tolist() == ["a1", "b2"]
    assert loaded["failed_tokens"].tolist() == ["x9"]
    for key in script.METRIC_KEYS:
        dtype = np.float32 if key in script.FLOAT32_KEYS else np.float16
        assert loaded[key].dtype == dtype and loaded[key].shape == (2, 5)
    assert np.allclose(loaded["score"][1], scenes["b2"]["score"])

    empty = script.stack_log_scores([("x9", None)], num_entries=5)
    assert empty["tokens"].shape == (0,) and empty["score"].shape == (0, 5)


def test_large_logs_are_split_into_equal_contiguous_chunks():
    script = _scoring_script()
    tokens = [f"t{i}" for i in range(70)]

    chunks = script.chunk_tokens(tokens, scenes_per_job=32)

    assert [len(c) for c in chunks] == [24, 24, 22]  # 3 parts of at most 32, balanced
    assert sum(chunks, []) == tokens
    assert script.chunk_tokens(tokens[:5], 32) == [tokens[:5]]
    assert script.chunk_tokens([], 32) == [[]]  # a log without scenes still gets a (empty) file


def test_merged_parts_equal_scoring_the_log_at_once():
    script = _scoring_script()
    rng = np.random.default_rng(0)
    results = [(t, None if t == "x9" else {k: rng.random(5) for k in script.METRIC_KEYS}) for t in ("a1", "x9", "b2", "c3")]

    whole = script.stack_log_scores(results, num_entries=5)
    merged = script.merge_log_parts(
        [script.stack_log_scores(results[:2], num_entries=5), script.stack_log_scores(results[2:], num_entries=5)]
    )

    assert merged.keys() == whole.keys()
    for key in whole:
        assert np.array_equal(merged[key], whole[key]) and merged[key].dtype == whole[key].dtype


def test_split_mode_lists_navtrain_logs(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENSCENE_DATA_ROOT", str(tmp_path))
    script = _scoring_script()

    scene_filter, log_path = script._load_split("navtrain")
    jobs = script._log_jobs(SimpleNamespace(metric_cache_path=None, split="navtrain"))

    assert log_path == tmp_path / "navsim_logs" / "trainval"
    assert scene_filter.num_history_frames == 4 and len(scene_filter.tokens) > 100_000
    assert [name for name, _ in jobs] == sorted(scene_filter.log_names)
    assert all(paths is None for _, paths in jobs)
