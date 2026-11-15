"""
Per-scene data for RL-3 (vision_llm_lora_rl.RLReferenceTargetBuilder, config.rl_loss_weight > 0), written into a
separate tree with the training cache layout: <out>/<log>/<token>/vision_llm_rl_reference.gz with
  rl_reward (K,) float16           the unfiltered PDM total of every codebook entry ("score" of the codebook score table,
                                   exp/codebook_scores_navtrain/<log>.npz, as RL-1 used it)
  rl_multiplier (K,) float16       NC * DAC * DDC * TLC of every entry (stored sub-scores), 0 where rl_reward is 0; used
                                   only by the EC reward (config.rl_ec_weight > 0)
  rl_reference_fused (D,) float16  the start model's fused state (rl_features/<run>/shard_*.pt)
  rl_shortlist (S,) int16          the start model's sub-score heads' shortlist, best first (same shards)
  rl_previous_shortlist (P,) int16 only with --previous-shortlists: a scorer's shortlist at t - 0.5 s (frame 1 of
                                   scripts/training/cache_scorer_shortlists.py), best first; the previous plan of the EC
                                   reward with config.rl_ec_previous_plan = shortlist. Scenes without one are skipped.
Scenes without a score row are skipped; CacheOnlyDataset then leaves them out of training and validation.

    python scripts/training/cache_rl_reference.py --features-dir $NAVSIM_EXP_ROOT/rl_features/p3_aux_policy_ep03 \
        --scores-dir $NAVSIM_EXP_ROOT/codebook_scores_navtrain --out-dir $NAVSIM_EXP_ROOT/rl_reference/p3_aux_policy_ep03
"""
import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from navsim.agents.vision_llm_lora.vision_llm_lora_rl import RL_REFERENCE_NAME
from navsim.planning.training.dataset import dump_feature_target_to_pickle

_SHARED: Dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", type=Path, required=True, help="rl_features/<run> (shard_*.pt)")
    parser.add_argument("--scores-dir", type=Path, required=True, help="codebook score table, one <log>.npz per log")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2 * int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
                        help="processes (default 2 x CPUs: while one waits for /project I/O another compresses)")
    parser.add_argument("--max-logs", type=int, default=None, help="smoke test: first N logs")
    parser.add_argument("--previous-shortlists", type=Path, default=None,
                        help="scorer_shortlists/<name> (shard_*.pt with frame 1): adds rl_previous_shortlist")
    parser.add_argument("--previous-shortlist-size", type=int, default=256)
    return parser.parse_args()


def write_log(job: Tuple[str, List[int]]) -> Tuple[int, int]:
    """Write the files of one log; returns (written, skipped without a score row or a previous shortlist)."""
    log, indices = job
    tokens, fused, shortlist = _SHARED["token"], _SHARED["fused"], _SHARED["shortlist"]
    previous = _SHARED.get("previous")  # token -> (P,) int16, or None
    path = _SHARED["scores_dir"] / f"{log}.npz"
    if not path.exists():
        return 0, len(indices)
    data = np.load(path)
    rows = {str(t): i for i, t in enumerate(data["tokens"])}
    score = data["score"]
    # product of the multiplicative sub-scores, 0 where the total is 0 (the stored sub-scores have the human-penalty
    # filter applied, the total does not; vision_llm_lora_rl.ec_reward)
    multiplier = np.ones_like(score)
    for key in ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
                "traffic_light_compliance"):
        multiplier = multiplier * data[key].astype(np.float32)
    multiplier[score <= 0] = 0.0
    written = 0
    for index in indices:
        row = rows.get(tokens[index])
        if row is None or not np.isfinite(score[row]).all():
            continue
        if previous is not None and tokens[index] not in previous:
            continue
        folder = _SHARED["out_dir"] / log / tokens[index]
        folder.mkdir(parents=True, exist_ok=True)
        dump_feature_target_to_pickle(
            folder / f"{RL_REFERENCE_NAME}.gz",
            {
                "rl_reward": torch.from_numpy(score[row].astype(np.float16)),
                "rl_multiplier": torch.from_numpy(multiplier[row].astype(np.float16)),
                "rl_reference_fused": fused[index].clone(),
                "rl_shortlist": shortlist[index].clone(),
                **({"rl_previous_shortlist": previous[tokens[index]].clone()} if previous is not None else {}),
            },
        )
        written += 1
    return written, len(indices) - written


def load_previous_shortlists(folder: Path, size: int) -> Dict[str, torch.Tensor]:
    """token -> the first `size` entries of the frame-1 (t - 0.5 s) shortlist of cache_scorer_shortlists.py shards."""
    shards = sorted(folder.glob("shard_*.pt"))
    assert shards, f"no shard_*.pt in {folder}"
    out: Dict[str, torch.Tensor] = {}
    for path in shards:
        part = torch.load(path, map_location="cpu", weights_only=False)
        frame = part["frames"].index(1)
        assert part["shortlist"].shape[-1] >= size, f"{path}: shortlists shorter than {size}"
        rows = part["shortlist"][:, frame, :size].to(torch.int16)
        out.update(zip(part["token"], rows))
    return out


def main() -> None:
    args = parse_args()
    shards = sorted(args.features_dir.glob("shard_*.pt"))
    assert shards, f"no shard_*.pt in {args.features_dir}"
    parts = [torch.load(s, map_location="cpu", weights_only=False) for s in shards]
    _SHARED.update(
        token=[t for p in parts for t in p["token"]],
        fused=torch.cat([p["fused"] for p in parts]),
        shortlist=torch.cat([p["shortlist"] for p in parts]),
        scores_dir=args.scores_dir,
        out_dir=args.out_dir,
    )
    if args.previous_shortlists is not None:
        _SHARED["previous"] = load_previous_shortlists(args.previous_shortlists, args.previous_shortlist_size)
        print(f"previous-frame shortlists of {len(_SHARED['previous'])} scenes from {args.previous_shortlists}", flush=True)
    logs = [log for p in parts for log in p["log"]]
    by_log: Dict[str, List[int]] = {}
    for index, log in enumerate(logs):
        by_log.setdefault(log, []).append(index)
    jobs = sorted(by_log.items())[: args.max_logs]
    jobs.sort(key=lambda job: len(job[1]), reverse=True)  # largest logs first, handed out one at a time: no tail
    print(f"{len(_SHARED['token'])} scenes in {len(by_log)} logs from {len(shards)} shards; writing {len(jobs)} logs "
          f"with {args.workers} processes", flush=True)
    written = skipped = 0
    start = time.time()
    with Pool(args.workers) as pool:  # forked workers share the loaded shards
        for i, (w, s) in enumerate(pool.imap_unordered(write_log, jobs, chunksize=1), 1):
            written, skipped = written + w, skipped + s
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} logs, {written} scenes, {written / (time.time() - start):.0f} scenes/s", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "meta.json").write_text(json.dumps({
        "features_dir": str(args.features_dir), "scores_dir": str(args.scores_dir), "written": written,
        "skipped_without_score": skipped, "head": str(args.features_dir / "head.pt"),
        "previous_shortlists": str(args.previous_shortlists) if args.previous_shortlists else None,
    }, indent=1))
    what = "a score row" + (" or a previous shortlist" if args.previous_shortlists else "")
    print(f"wrote {written} scenes, skipped {skipped} without {what} -> {args.out_dir}")


if __name__ == "__main__":
    main()
