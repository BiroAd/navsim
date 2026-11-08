"""
Scores every codebook trajectory in every scene of a split with the NAVSIM PDM scorer, offline.
Logic and caveats (log-replay traffic, no two-frame comfort):
navsim/agents/vision_llm_lora/vision_llm_lora_codebook_scoring.py

Writes one <out-dir>/<log_name>.npz per log: `tokens` [N] and one [N, K] array per metric (row = scene, column =
codebook entry):
  no_at_fault_collisions, drivable_area_compliance, driving_direction_compliance, traffic_light_compliance,
  ego_progress, time_to_collision_within_bound, lane_keeping, history_comfort, score, progress_raw
plus `failed_tokens`, the scenes of the log that could not be scored. One file per log instead of per scene keeps
navtrain at ~1.2k files: /project has a project quota of 1M files. Scoring parallelizes over jobs of at most
--scenes-per-job scenes (navtrain logs have 1 to 1080 scenes, at ~2 min per scene per worker): a log's jobs write
<out-dir>/parts/<log_name>.partNNN.npz, merged into <log_name>.npz once all exist. Existing log and part files are
skipped, so an interrupted run continues where it stopped, losing at most the jobs that were running. With array
shards, logs are dealt to shards largest first (by log file size or metric cache count), so shards get similar work.

Input, one of:
  --metric-cache-path DIR  existing metric caches (run_metric_caching.py), e.g. navtest or navhard, which official
                           evaluation needs anyway.
  --split NAME             builds each scene's metric cache in memory from the navsim logs and maps, as
                           run_metric_caching.py would, and never writes it (~2 files per scene saved). Original log
                           scenes only, no synthetic ones. Needs OPENSCENE_DATA_ROOT and NUPLAN_MAPS_ROOT.
No images. Start with a few logs to measure the time:
  OMP_NUM_THREADS=1 python scripts/evaluation/score_codebook_trajectories.py --split navtrain \
      --out-dir $NAVSIM_EXP_ROOT/codebook_scores_navtrain --max-logs 2 --workers 2
"""

import argparse
import logging
import lzma
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.agents.vision_llm_lora.vision_llm_lora_codebook import load_codebook
from navsim.agents.vision_llm_lora.vision_llm_lora_codebook_scoring import (
    MULTI_METRICS,
    WEIGHTED_METRICS,
    score_codebook,
)
from navsim.common.dataclasses import Scene, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

CONFIG_DIR = Path(__file__).resolve().parents[2] / "navsim/planning/script/config"
METRIC_KEYS = (*MULTI_METRICS, *WEIGHTED_METRICS, "score", "progress_raw")  # as score_codebook returns them
FLOAT32_KEYS = ("ego_progress", "score", "progress_raw")  # continuous; the others are 0 / 0.5 / 1
MAP_VERSION = "nuplan-maps-v1.0"  # as navsim/planning/metric_caching/caching.py

logger = logging.getLogger(__name__)
_worker: Dict = {}


def _load_split(split: str) -> Tuple[object, Path]:
    """Scene filter and navsim log directory of a train_test_split config, as Hydra would compose them."""
    split_cfg = OmegaConf.load(CONFIG_DIR / f"common/train_test_split/{split}.yaml")
    filter_name = next(d["scene_filter"] for d in split_cfg.defaults if "scene_filter" in d)
    scene_filter = instantiate(OmegaConf.load(CONFIG_DIR / f"common/train_test_split/scene_filter/{filter_name}.yaml"))
    return scene_filter, Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs" / split_cfg.data_split


def _build(codebook_path: str, chunk_size: int, split: Optional[str]) -> None:
    """Per-process setup: simulator, scorer and traffic policy from the evaluation configs, and the codebook."""
    common = OmegaConf.load(CONFIG_DIR / "common/default_common.yaml")
    cfg = OmegaConf.create(
        {
            "proposal_sampling": common.proposal_sampling,
            "scorer": OmegaConf.load(CONFIG_DIR / "pdm_scoring/scorer/pdm_scorer.yaml"),
            "simulator": OmegaConf.load(CONFIG_DIR / "pdm_scoring/default_run_pdm_score.yaml").simulator,
        }
    )
    agent_cfg = OmegaConf.load(CONFIG_DIR / "common/agent/vision_llm_lora_agent.yaml")
    simulator = instantiate(cfg.simulator)
    traffic = OmegaConf.load(CONFIG_DIR / "common/traffic_agents_policy/log_replay_traffic_agents.yaml")
    _worker.update(
        simulator=simulator,
        scorer=instantiate(cfg.scorer),
        traffic_agents_policy=instantiate(traffic.non_reactive, simulator.proposal_sampling),
        trajectory_sampling=instantiate(agent_cfg.trajectory_sampling),
        poses=load_codebook(codebook_path),
        chunk_size=chunk_size,
    )
    if split is not None:
        scene_filter, log_path = _load_split(split)
        _worker.update(
            scene_filter=scene_filter,
            log_path=log_path,
            processor=MetricCacheProcessor(None, False, instantiate(cfg.proposal_sampling)),
        )


def _metric_caches_from_files(paths: List[Path]) -> Iterator[Tuple[str, Optional[MetricCache]]]:
    for path in paths:
        try:
            with lzma.open(path, "rb") as f:  # as MetricCacheLoader.get_from_token
                yield path.parent.name, pickle.load(f)
        except Exception:
            logger.exception(f"Loading metric cache {path} failed")
            yield path.parent.name, None


def _metric_caches_from_log(
    log_name: str, tokens: Optional[List[str]] = None
) -> Iterator[Tuple[str, Optional[MetricCache]]]:
    """
    Metric caches of a log's scenes (only the given tokens, if any), computed in memory as
    navsim/planning/metric_caching/caching.py does.
    """
    scene_filter = deepcopy(_worker["scene_filter"])
    scene_filter.log_names = [log_name]
    if tokens is not None:
        scene_filter.tokens = tokens
    loader = SceneLoader(
        data_path=_worker["log_path"],
        original_sensor_path=None,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    for token, scene_dict in loader.scene_frames_dicts.items():
        try:
            scene = Scene.from_scene_dict_list(
                scene_dict,
                None,
                num_history_frames=scene_filter.num_history_frames,
                num_future_frames=scene_filter.num_future_frames,
                sensor_config=SensorConfig.build_no_sensors(),
            )
            scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version=MAP_VERSION)
            yield scenario.token, _worker["processor"].compute_metric_cache(scenario)
        except Exception:
            logger.exception(f"Building the metric cache of token {token} failed")
            yield token, None


def stack_log_scores(
    results: List[Tuple[str, Optional[Dict[str, npt.NDArray]]]], num_entries: int
) -> Dict[str, npt.NDArray]:
    """
    One log's per-scene results as the arrays of its .npz.
    :param results: (token, {metric: [K]} or None if the scene failed) per scene
    :param num_entries: K, the codebook size, for the shape of a log without any scored scene
    :return: tokens [N], failed_tokens [F] and one [N, K] array per metric
    """
    scored = [(token, scores) for token, scores in results if scores is not None]
    arrays = {
        "tokens": np.array([token for token, _ in scored], dtype=str),
        "failed_tokens": np.array([token for token, scores in results if scores is None], dtype=str),
    }
    for key in METRIC_KEYS:
        dtype = np.float32 if key in FLOAT32_KEYS else np.float16
        rows = [scores[key] for _, scores in scored]
        arrays[key] = np.stack(rows).astype(dtype) if rows else np.zeros((0, num_entries), dtype)
    return arrays


def merge_log_parts(parts: List[Dict[str, npt.NDArray]]) -> Dict[str, npt.NDArray]:
    """One log's .npz arrays from the arrays of its parts (as stack_log_scores returns them), in part order."""
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def chunk_tokens(tokens: List[str], scenes_per_job: int) -> List[List[str]]:
    """Contiguous chunks of at most scenes_per_job tokens; a log without scenes is one empty chunk."""
    if not tokens:
        return [[]]
    num_parts = -(-len(tokens) // scenes_per_job)
    size = -(-len(tokens) // num_parts)  # equal sizes rather than full chunks plus a small remainder
    return [tokens[i : i + size] for i in range(0, len(tokens), size)]


def _part_path(out_dir: Path, log_name: str, part: int) -> Path:
    return out_dir / "parts" / f"{log_name}.part{part:03d}.npz"


def _log_tokens(job: Tuple[str, Optional[List[Path]]]) -> Tuple[str, List[str]]:
    """(log name, scene tokens of the log), from the metric cache paths or the split's scene filter."""
    log_name, metric_cache_paths = job
    if metric_cache_paths is not None:
        return log_name, [path.parent.name for path in metric_cache_paths]
    scene_filter = deepcopy(_worker["scene_filter"])
    scene_filter.log_names = [log_name]
    loader = SceneLoader(
        data_path=_worker["log_path"],
        original_sensor_path=None,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    return log_name, list(loader.scene_frames_dicts)


def _write_npz(arrays: Dict[str, npt.NDArray], out_path: Path) -> None:
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp_path, out_path)  # a killed job never leaves a half-written .npz


def _score_log(
    job: Tuple[str, Optional[List[Path]], Path, Optional[List[str]]]
) -> Tuple[str, List[float], List[str], float]:
    """
    Scores one log, or the given scene tokens of it, and writes the .npz to out_path.
    Returns (log name, best score per scene, failed tokens, seconds).
    """
    log_name, metric_cache_paths, out_path, tokens = job
    start = time.perf_counter()
    if metric_cache_paths is None:
        metric_caches = _metric_caches_from_log(log_name, tokens)
    else:
        if tokens is not None:
            wanted = set(tokens)
            metric_cache_paths = [path for path in metric_cache_paths if path.parent.name in wanted]
        metric_caches = _metric_caches_from_files(metric_cache_paths)

    results = []
    for token, metric_cache in metric_caches:
        scores = None
        if metric_cache is not None:
            try:
                scores = score_codebook(
                    metric_cache,
                    _worker["poses"],
                    _worker["trajectory_sampling"],
                    _worker["simulator"],
                    _worker["scorer"],
                    _worker["traffic_agents_policy"],
                    _worker["chunk_size"],
                )
            except Exception:
                logger.exception(f"Scoring failed for token {token}")
        results.append((token, scores))

    arrays = stack_log_scores(results, len(_worker["poses"]))
    _write_npz(arrays, out_path)
    best = arrays["score"].max(axis=1).tolist() if len(arrays["tokens"]) else []
    return log_name, best, arrays["failed_tokens"].tolist(), time.perf_counter() - start


def _log_jobs(args: argparse.Namespace) -> List[Tuple[str, Optional[List[Path]]]]:
    """(log name, metric cache files or None to build them in memory) for every log of the input, sorted by name."""
    if args.metric_cache_path is not None:
        loader = MetricCacheLoader(args.metric_cache_path)
        per_log: Dict[str, List[Path]] = {}
        for token in sorted(loader.tokens):
            path = Path(loader.metric_cache_paths[token])  # <cache>/<log>/<scenario type>/<token>/metric_cache.pkl
            per_log.setdefault(path.parents[2].name.removesuffix(".db"), []).append(path)
        return sorted(per_log.items())
    scene_filter, log_path = _load_split(args.split)
    log_names = scene_filter.log_names or [p.name.removesuffix(".pkl") for p in log_path.iterdir()]
    return [(name, None) for name in sorted(log_names)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metric-cache-path", type=Path)
    source.add_argument("--split", help="train_test_split config name, e.g. navtrain")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--codebook", default=None, help="default: codebook_path from the agent YAML")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=1024, help="entries per simulator/scorer batch")
    parser.add_argument("--max-logs", type=int, default=None)
    parser.add_argument(
        "--scenes-per-job",
        type=int,
        default=32,
        help="larger logs are split into jobs of at most this many scenes, scored in parallel and merged per log",
    )
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    codebook = args.codebook
    if codebook is None:
        codebook = OmegaConf.load(CONFIG_DIR / "common/agent/vision_llm_lora_agent.yaml").config.codebook_path
    logger.info(f"Codebook {codebook}")

    all_logs = _log_jobs(args)
    if args.metric_cache_path is None:
        log_path = _load_split(args.split)[1]
        pkl = {name: log_path / f"{name}.pkl" for name, _ in all_logs}
        size = {name: path.stat().st_size if path.exists() else 0 for name, path in pkl.items()}  # ~ scene count
    else:
        size = {name: len(paths) for name, paths in all_logs}
    # Largest logs first, dealt round-robin: shards get similar work, and big logs start early within a shard.
    all_logs.sort(key=lambda log: (-size[log[0]], log[0]))
    logs = all_logs[args.shard_index :: args.num_shards][: args.max_logs]
    (args.out_dir / "parts").mkdir(parents=True, exist_ok=True)
    todo = [(name, paths) for name, paths in logs if not (args.out_dir / f"{name}.npz").exists()]
    logger.info(
        f"Shard {args.shard_index + 1}/{args.num_shards}: {len(logs)} logs, {len(logs) - len(todo)} already "
        f"scored, {len(todo)} to score with {args.workers} worker(s), at most {args.scenes_per_job} scenes per job"
    )

    best, seconds_per_scene, failed = [], [], []
    with ProcessPoolExecutor(args.workers, initializer=_build, initargs=(codebook, args.chunk_size, args.split)) as ex:
        paths_of = dict(todo)
        log_parts = {name: chunk_tokens(tokens, args.scenes_per_job) for name, tokens in ex.map(_log_tokens, todo)}

        def merge_if_complete(name: str) -> None:
            part_paths = [_part_path(args.out_dir, name, p) for p in range(len(log_parts[name]))]
            if all(path.exists() for path in part_paths):
                _write_npz(merge_log_parts([dict(np.load(path)) for path in part_paths]), args.out_dir / f"{name}.npz")
                for path in part_paths:
                    path.unlink()

        jobs = []
        for name, chunks in log_parts.items():  # largest logs first, as sorted above
            for p, tokens in enumerate(chunks):
                if not _part_path(args.out_dir, name, p).exists():  # parts of an interrupted run are kept
                    jobs.append((name, paths_of[name], _part_path(args.out_dir, name, p), tokens))
            merge_if_complete(name)  # e.g. killed after its last part, before merging
        logger.info(f"{len(jobs)} jobs ({sum(len(c) for c in log_parts.values())} parts in total)")

        futures = [ex.submit(_score_log, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            name, part_best, part_failed, duration = future.result()
            num_scenes = len(part_best) + len(part_failed)
            best += part_best
            failed += part_failed
            if num_scenes:
                seconds_per_scene.append(duration / num_scenes)
            merge_if_complete(name)
            logger.info(
                f"[{i}/{len(jobs)}] {name}: {num_scenes} scenes, {len(part_failed)} failed, {duration:.0f} s, "
                f"mean best codebook score {np.mean(part_best) if part_best else float('nan'):.4f}"
            )

    if seconds_per_scene:
        logger.info(f"Median {np.median(seconds_per_scene):.1f} s per scene per worker")
    if best:
        logger.info(f"Mean best codebook score over {len(best)} scenes (oracle, this run): {np.mean(best):.4f}")
    if failed:
        logger.error(f"{len(failed)} scene(s) failed, listed in failed_tokens of their log's .npz: {failed[:10]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
