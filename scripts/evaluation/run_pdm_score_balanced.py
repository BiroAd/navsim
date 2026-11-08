"""
Run a NAVSIM scoring script unchanged, with balanced work distribution.

run_pdm_score_one_stage.py and run_pdm_score.py hand their per-log work items to nuplan's worker_map, which splits them
into exactly one fixed chunk per worker process (chunk_list). Logs differ a lot in their number of scenes, so most
workers finish early and the job waits for the one with the heaviest chunk (job 14926361: 31 of 32 processes idle after
~4 min, CPU efficiency 41%). Here every log is its own task, largest first, and the process pool hands the next log to
whichever process is free. The scoring itself (per-log function, aggregation, output CSV) is the script's own code.

run_pdm_score_one_stage additionally splits logs into chunks of at most CHUNK_SCENES tokens: each scene is scored on its
own and the two-frame comfort is computed afterwards from all rows, so the split does not change any score, and the
largest navtest log (568 scenes, ~4 min on one process) no longer sets the job's run time.

run_pdm_score (two stage) keeps whole logs (a log's stage two scenes are loaded through its stage one scenes) and gets
exactly one task per worker process: every task builds a SceneLoader, which reads all 5,462 navhard synthetic scene
pickles (~97 s on one core, job 14926363), so one task per log would repeat that per log. The logs are packed into the
tasks by size instead (largest first, each to the task with the fewest scenes so far), not in list order as chunk_list
does (stage two scenes per process ranged from 4 to 426 in job 14926363).

Synthetic scenes: navsim.common.dataloader.filter_synthetic_scenes builds the full Scene of every one of the 5,462
pickles (855 MB on /project, ~33 ms each to read and unpickle) only to read three metadata fields. It is replaced at
runtime by fast_filter_synthetic_scenes: the same filter rules on a metadata index (log_name, initial_token,
corresponding_original_scene per file name) built once into $NAVSIM_EXP_ROOT/synthetic_scene_index/ and rebuilt when
the folder's file list changes. Checked against the original on navhard_two_stage (2026-11-08): identical result and
order for the full filter (5,462 scenes; 208 s → 0.05 s, index build 24 s once) and for a 3-log filter. Rules and
packing: tests/test_pdm_score_balanced.py.

    python scripts/evaluation/run_pdm_score_balanced.py run_pdm_score_one_stage <hydra overrides...>
    python scripts/evaluation/run_pdm_score_balanced.py run_pdm_score <hydra overrides...>
"""
import hashlib
import importlib
import json
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool

SCRIPTS = ("run_pdm_score_one_stage", "run_pdm_score")
CHUNK_SCENES = 100


METADATA_FIELDS = ("log_name", "initial_token", "corresponding_original_scene")


def _read_metadata(path: Path) -> Tuple[str, Dict[str, str]]:
    with open(path, "rb") as f:
        metadata = pickle.load(f)["scene_metadata"]
    return path.name, {key: metadata[key] for key in METADATA_FIELDS}


def synthetic_scene_index(data_path: Path) -> Dict[str, Dict[str, str]]:
    """File name -> scene metadata fields of every synthetic scene pickle in data_path, cached as JSON."""
    data_path = Path(data_path)
    names = sorted(p.name for p in data_path.iterdir())
    cache_dir = Path(os.environ["NAVSIM_EXP_ROOT"]) / "synthetic_scene_index"
    cache = cache_dir / f"{data_path.parent.name}_{hashlib.sha1(str(data_path.resolve()).encode()).hexdigest()[:8]}.json"
    if cache.is_file():
        index = json.loads(cache.read_text())
        if sorted(index) == names:
            return index
    with Pool(min(8, os.cpu_count() or 1)) as pool:
        index = dict(pool.map(_read_metadata, [data_path / name for name in names], chunksize=64))
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(index))
    tmp.replace(cache)
    return index


def fast_filter_synthetic_scenes(data_path: Path, scene_filter, stage1_scenes_final_frames_tokens: List[str]):
    """navsim.common.dataloader.filter_synthetic_scenes on the metadata index: same rules, same iteration order, same
    {initial_token: [scene_path, log_name]} result, without loading the scenes."""
    index = synthetic_scene_index(data_path)
    filter_logs = scene_filter.log_names is not None
    filter_tokens = scene_filter.synthetic_scene_tokens is not None
    final_frames = set(stage1_scenes_final_frames_tokens)
    loaded_scenes = {}
    for scene_path in Path(data_path).iterdir():
        metadata = index[scene_path.name]
        if filter_tokens and metadata["initial_token"] not in scene_filter.synthetic_scene_tokens:
            continue
        log_name = metadata["log_name"]
        if filter_logs and log_name not in scene_filter.log_names:
            continue
        if not filter_tokens and metadata["corresponding_original_scene"] not in final_frames:
            continue
        loaded_scenes.update({metadata["initial_token"]: [scene_path, log_name]})
    return loaded_scenes


def split_logs(input_objects: List[Any], max_tokens: int) -> List[Any]:
    """Split each per-log item {"cfg", "log_file", "tokens"} into items of at most max_tokens tokens."""
    return [
        {**item, "tokens": item["tokens"][i : i + max_tokens]}
        for item in input_objects
        for i in range(0, len(item["tokens"]), max_tokens)
    ]


def pack_logs(input_objects: List[Any], num_tasks: int) -> List[List[Any]]:
    """Greedy largest-first packing of per-log items into num_tasks lists with similar numbers of tokens."""
    tasks: List[List[Any]] = [[] for _ in range(min(num_tasks, len(input_objects)))]
    sizes = [0] * len(tasks)
    for item in sorted(input_objects, key=lambda item: len(item["tokens"]), reverse=True):
        i = sizes.index(min(sizes))
        tasks[i].append(item)
        sizes[i] += len(item["tokens"])
    return [task for task in tasks if task]


def balanced_worker_map(
    worker: WorkerPool, fn: Callable[..., List[Any]], input_objects: List[Any], max_tokens: int = 0
) -> List[Any]:
    """Same results as nuplan's worker_map. max_tokens > 0: one task per chunk of at most max_tokens tokens of a log,
    largest first; max_tokens = 0: one task per worker, logs packed by size."""
    if worker.number_of_threads == 0:
        return fn(input_objects)
    if max_tokens:
        order = sorted(split_logs(input_objects, max_tokens), key=lambda item: len(item["tokens"]), reverse=True)
        tasks = [[item] for item in order]
    else:
        tasks = pack_logs(input_objects, worker.number_of_threads)
    results = worker.map(Task(fn=fn), tasks)
    return [row for rows in results for row in rows]


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in SCRIPTS:
        raise SystemExit(f"usage: {sys.argv[0]} {{{','.join(SCRIPTS)}}} <hydra overrides...>")
    import navsim.common.dataloader as dataloader  # before the script, so its SceneLoaders use the fast filter

    dataloader.filter_synthetic_scenes = fast_filter_synthetic_scenes
    module = importlib.import_module(f"navsim.planning.script.{sys.argv[1]}")
    max_tokens = CHUNK_SCENES if sys.argv[1] == "run_pdm_score_one_stage" else 0
    module.worker_map = lambda worker, fn, items: balanced_worker_map(worker, fn, items, max_tokens)
    sys.argv = [module.__file__, *sys.argv[2:]]
    module.main()


if __name__ == "__main__":
    main()
