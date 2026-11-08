"""
Reactive (IDM) vs non-reactive (log-replay) PDM scores of plausible codebook entries, on a sample of scenes.

The offline codebook scores (score_codebook_trajectories.py) use log-replay traffic: other agents ignore the ego, so
one traffic rollout per scene serves all 16384 entries. Reactive traffic needs one IDM rollout per entry (~1.7-6.6 s
each, measured on navtest), which is too slow for the whole codebook. This script measures whether it matters: for
each sampled scene it takes the --candidates entries closest to the human trajectory (ADE, where a trained policy puts
its mass), scores each one paired with the PDM-Closed reference (as score_codebook does) once with log-replay and once
with IDM traffic, and stores both. Only collisions (NC), time to collision (TTC) and, through the progress
normalization, ego progress can differ; the other metrics do not depend on traffic.

No human-penalty filter (strict, like the offline `score`). Run as a SLURM array (--shard/--num-shards), then:
  python scripts/evaluation/compare_reactive_scores.py --summarize --out-dir <dir>
"""
import argparse
import copy
import glob
import logging
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

import score_codebook_trajectories as scb  # same directory; sets up simulator, scorer and codebook as the offline run
from navsim.agents.vision_llm_lora.vision_llm_lora_codebook_scoring import (
    MULTI_METRICS,
    WEIGHTED_METRICS,
    codebook_to_state_arrays,
    pairwise_scores,
)
from navsim.evaluate.pdm_score import get_trajectory_as_array

logger = logging.getLogger(__name__)
METRICS = [*MULTI_METRICS, *WEIGHTED_METRICS, "score"]
MODES = ("nonreactive", "reactive")
_policies: Dict = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric-cache-path", type=Path, default=None)
    parser.add_argument("--codebook-path", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-scenes", type=int, default=512, help="scenes sampled uniformly (fixed seed)")
    parser.add_argument("--candidates", type=int, default=64, help="entries per scene closest to the human trajectory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summarize", action="store_true", help="only print statistics of the shard files")
    return parser.parse_args()


def _init(codebook_path: str) -> None:
    scb._build(codebook_path, 1024, None)
    sim = scb._worker["simulator"]
    cfg = OmegaConf.load(scb.CONFIG_DIR / "common/traffic_agents_policy/navsim_IDM_traffic_agents.yaml")
    _policies["reactive"] = instantiate(cfg.reactive, sim.proposal_sampling)
    _policies["nonreactive"] = scb._worker["traffic_agents_policy"]


def _candidates(metric_cache, poses: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Indices and ADE of the k codebook entries closest to the human trajectory (ego frame, x/y)."""
    human = metric_cache.human_trajectory
    assert abs(human.trajectory_sampling.interval_length - 0.5) < 1e-6, "codebook is sampled at 0.5 s"
    steps = min(len(human.poses), poses.shape[1])
    ade = np.linalg.norm(poses[:, :steps, :2] - human.poses[None, :steps, :2], axis=-1).mean(axis=-1)
    idx = np.argsort(ade)[:k]
    return idx, ade[idx]


def _score_scene(path: str):
    """Both modes for one scene: (token, candidate idx, ADE, {mode: {metric: [k]}}, IDM seconds) or None."""
    w = scb._worker
    sim, scorer = w["simulator"], w["scorer"]
    try:
        token, mc = next(scb._metric_caches_from_files([Path(path)]))
        if mc is None or mc.human_trajectory is None:
            return None
        ego = mc.ego_state
        idx, ade = _candidates(mc, w["poses"], _K)
        ref = get_trajectory_as_array(mc.trajectory, sim.proposal_sampling, ego.time_point)
        entries = codebook_to_state_arrays(w["poses"][idx], ego, w["trajectory_sampling"], sim.proposal_sampling)
        out = {mode: {m: np.zeros(len(idx)) for m in METRICS} for mode in MODES}
        idm_seconds = 0.0
        for j, entry in enumerate(entries):
            simulated = sim.simulate_proposals(np.stack([ref, entry]), ego)
            for mode in MODES:
                # Fresh copy per scoring call: scoring with reactive (IDM) tracks leaves state in mc.observation that
                # changes every later score_proposals call on the same metric cache, log-replay ones included
                # (found 2026-11-05: the first version reused one metric cache per scene, so only candidate 0 was
                # clean). The official evaluation loads a fresh metric cache per scene, i.e. the first-call result.
                mc_call = copy.deepcopy(mc)
                start = time.perf_counter()
                tracks = _policies[mode].simulate_environment(simulated[1], mc_call)  # as navsim/evaluate/pdm_score.py
                if mode == "reactive":
                    idm_seconds += time.perf_counter() - start
                scorer.score_proposals(simulated, mc_call.observation, mc_call.centerline, mc_call.route_lane_ids,
                                       mc_call.drivable_area_map, mc_call.map_parameters, tracks,
                                       mc_call.past_human_trajectory)
                multi, weighted, progress = scorer._multi_metrics, scorer._weighted_metrics, scorer._progress_raw
                pair = pairwise_scores(multi, weighted, progress, scorer._config)
                for name, i in MULTI_METRICS.items():
                    out[mode][name][j] = multi[i, 1]
                for name, i in WEIGHTED_METRICS.items():
                    out[mode][name][j] = pair["ego_progress"][0] if name == "ego_progress" else weighted[i, 1]
                out[mode]["score"][j] = pair["score"][0]
        return token, idx, ade, out, idm_seconds
    except Exception:
        logger.exception(f"scene {path} failed")
        return None


def summarize(out_dir: Path) -> None:
    files = sorted(glob.glob(str(out_dir / "reactive_vs_nonreactive_shard*.npz")))
    data = [np.load(f) for f in files]
    get = lambda key: np.concatenate([d[key] for d in data])  # noqa: E731
    n, k = get("candidates").shape
    print(f"{len(files)} shards, {n} scenes, {k} candidates each (closest to the human trajectory)")
    print(f"IDM rollout: {get('idm_seconds').sum() / (n * k):.2f} s per entry on average")
    print(f"{'metric':32s} {'non-reactive':>12s} {'reactive':>9s} {'entries changed':>16s} {'scenes with a change':>21s}")
    for m in METRICS:
        nr, r = get(f"nonreactive/{m}"), get(f"reactive/{m}")
        changed = np.abs(nr - r) > 1e-6
        print(f"{m:32s} {nr.mean():12.4f} {r.mean():9.4f} {changed.mean():16.2%} {changed.any(axis=1).mean():21.2%}")
    nr, r = get("nonreactive/score"), get("reactive/score")
    for label, col in [(lb, c) for lb, c in (("closest entry to the human trajectory", 0), ("5th closest", 4),
                                               ("20th closest", 19)) if c < k]:
        print(f"{label:40s} score {nr[:, col].mean():.4f} -> {r[:, col].mean():.4f}")
    best_nr, best_r = nr.argmax(axis=1), r.argmax(axis=1)
    print(f"best candidate (by score) differs in {(best_nr != best_r).mean():.2%} of scenes; reactive score of the"
          f" non-reactive best: {r[np.arange(n), best_nr].mean():.4f} vs reactive best {r.max(axis=1).mean():.4f}")
    flips_up = ((r > nr + 1e-6).sum(), (r < nr - 1e-6).sum())
    print(f"entries scored higher with reactive traffic: {flips_up[0]}, lower: {flips_up[1]}")


def main() -> None:
    global _K
    args = parse_args()
    if args.summarize:
        summarize(args.out_dir)
        return
    _K = args.candidates
    codebook = args.codebook_path or str(
        Path(__file__).resolve().parents[2] / "navsim/agents/vision_llm_lora/codebook/16384_8step.npy"
    )
    paths = sorted(str(p) for p in args.metric_cache_path.glob("*/*/*/metric_cache.pkl"))
    rng = np.random.default_rng(args.seed)
    sample = sorted(rng.choice(len(paths), min(args.num_scenes, len(paths)), replace=False))
    paths = [paths[i] for i in sample][args.shard :: args.num_shards]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"shard {args.shard}/{args.num_shards}: {len(paths)} scenes, {args.candidates} candidates, "
          f"{args.workers} workers", flush=True)

    start, results = time.time(), []
    with Pool(args.workers, initializer=_init, initargs=(codebook,)) as pool:
        for i, res in enumerate(pool.imap_unordered(_score_scene, paths)):
            if res is not None:
                results.append(res)
            if (i + 1) % 10 == 0 or i + 1 == len(paths):
                print(f"{i + 1}/{len(paths)} scenes, {time.time() - start:.0f} s", flush=True)

    arrays: Dict[str, np.ndarray] = {
        "tokens": np.array([r[0] for r in results]),
        "candidates": np.stack([r[1] for r in results]),
        "ade": np.stack([r[2] for r in results]),
        "idm_seconds": np.array([r[4] for r in results]),
    }
    for mode in MODES:
        for m in METRICS:
            arrays[f"{mode}/{m}"] = np.stack([r[3][mode][m] for r in results])
    np.savez(args.out_dir / f"reactive_vs_nonreactive_shard{args.shard:02d}of{args.num_shards:02d}.npz", **arrays)
    print(f"done: {len(results)} of {len(paths)} scenes in {time.time() - start:.0f} s", flush=True)


_K = 64
if __name__ == "__main__":
    main()
