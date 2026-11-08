"""
CPU checks of scripts/evaluation/run_pdm_score_balanced.py: log chunking and packing keep every token exactly once, and
the index-based synthetic scene filter applies NAVSIM's rules (token filter, log filter, stage one final frames).
"""
import importlib.util
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

_spec = importlib.util.spec_from_file_location(
    "run_pdm_score_balanced", Path(__file__).parents[1] / "scripts" / "evaluation" / "run_pdm_score_balanced.py"
)
balanced = importlib.util.module_from_spec(_spec)
sys.modules["run_pdm_score_balanced"] = balanced  # the index builder's worker pool pickles its function by module
_spec.loader.exec_module(balanced)


def _items(sizes):
    return [{"cfg": None, "log_file": f"log{i}", "tokens": [f"{i}_{j}" for j in range(n)]} for i, n in enumerate(sizes)]


def test_split_and_pack_keep_every_token_once():
    items = _items([250, 30, 7, 426, 4, 142])
    tokens = sorted(t for item in items for t in item["tokens"])
    chunks = balanced.split_logs(items, 100)
    assert max(len(c["tokens"]) for c in chunks) == 100
    assert sorted(t for c in chunks for t in c["tokens"]) == tokens
    tasks = balanced.pack_logs(items, 3)
    assert len(tasks) == 3 and sorted(t for task in tasks for item in task for t in item["tokens"]) == tokens
    # largest first, each to the lightest task: 426 | 250 | 142 + 30 + 7 + 4
    assert sorted(sum(len(item["tokens"]) for item in task) for task in tasks) == [183, 250, 426]


def test_fast_synthetic_filter_rules(tmp_path, monkeypatch):
    monkeypatch.setenv("NAVSIM_EXP_ROOT", str(tmp_path / "exp"))
    data = tmp_path / "navhard" / "synthetic_scene_pickles"
    data.mkdir(parents=True)
    scenes = [("a", "logA", "orig1"), ("b", "logA", "orig2"), ("c", "logB", "orig1"), ("d", "logC", "other")]
    for token, log, original in scenes:
        metadata = {"log_name": log, "initial_token": token, "corresponding_original_scene": original, "map_name": "m"}
        (data / f"{token}.pkl").write_bytes(pickle.dumps({"scene_metadata": metadata, "frames": []}))
    run = lambda **kw: balanced.fast_filter_synthetic_scenes(  # noqa: E731
        data, SimpleNamespace(log_names=kw.get("logs"), synthetic_scene_tokens=kw.get("tokens")), ["orig1", "orig2"]
    )
    assert sorted(run()) == ["a", "b", "c"]  # only scenes of loaded stage one frames
    assert sorted(run(logs=["logA"])) == ["a", "b"]
    assert sorted(run(tokens=["d", "a"])) == ["a", "d"]  # explicit tokens bypass the stage one rule
    assert run()["a"] == [data / "a.pkl", "logA"]
    cache = list((tmp_path / "exp" / "synthetic_scene_index").glob("*.json"))
    assert len(cache) == 1
    (data / "e.pkl").write_bytes(pickle.dumps({"scene_metadata": {"log_name": "logA", "initial_token": "e",
                                                                  "corresponding_original_scene": "orig1"}}))
    assert "e" in run()  # a changed file list rebuilds the index
