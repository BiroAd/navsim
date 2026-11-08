"""
Per-scene data for RL-3 (vision_llm_lora_rl.RLReferenceTargetBuilder, config.rl_loss_weight > 0), written into a
separate tree with the training cache layout: <out>/<log>/<token>/vision_llm_rl_reference.gz with
  rl_reward (K,) float16           the unfiltered PDM total of every codebook entry ("score" of the codebook score table,
                                   exp/codebook_scores_navtrain/<log>.npz, as RL-1 used it)
  rl_reference_fused (D,) float16  the start model's fused state (rl_features/<run>/shard_*.pt)
  rl_shortlist (S,) int16          the start model's sub-score heads' shortlist, best first (same shards)
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
    return parser.parse_args()


def write_log(job: Tuple[str, List[int]]) -> Tuple[int, int]:
    """Write the files of one log; returns (written, skipped without a score row)."""
    log, indices = job
    tokens, fused, shortlist = _SHARED["token"], _SHARED["fused"], _SHARED["shortlist"]
    path = _SHARED["scores_dir"] / f"{log}.npz"
    if not path.exists():
        return 0, len(indices)
    data = np.load(path)
    rows = {str(t): i for i, t in enumerate(data["tokens"])}
    score = data["score"]
    written = 0
    for index in indices:
        row = rows.get(tokens[index])
        if row is None or not np.isfinite(score[row]).all():
            continue
        folder = _SHARED["out_dir"] / log / tokens[index]
        folder.mkdir(parents=True, exist_ok=True)
        dump_feature_target_to_pickle(
            folder / f"{RL_REFERENCE_NAME}.gz",
            {
                "rl_reward": torch.from_numpy(score[row].astype(np.float16)),
                "rl_reference_fused": fused[index].clone(),
                "rl_shortlist": shortlist[index].clone(),
            },
        )
        written += 1
    return written, len(indices) - written


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
    }, indent=1))
    print(f"wrote {written} scenes, skipped {skipped} without a score row -> {args.out_dir}")


if __name__ == "__main__":
    main()
