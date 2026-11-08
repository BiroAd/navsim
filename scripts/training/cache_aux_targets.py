"""
Computes VisionLLMLoRAAuxTargetBuilder targets (BEV semantic map, drivable-area map, lane-heading map, agent boxes) for
every scene of the training cache and writes them as <out-dir>/<log>/<token>/vision_llm_aux_target.gz, the training
cache layout. No sensors are read: only the navsim logs (annotations) and the nuPlan maps ($NUPLAN_MAPS_ROOT).
The SLURM wrapper scripts/training/run_cache_aux_targets_slurm.sh stages the output in /dev/shm and packs it into one
squashfs image, so the ~103k files do not count against the project's file quota.

Usage:
  python scripts/training/cache_aux_targets.py --run-config exp/p1c_scores/<date>/code/hydra/config.yaml \
      --out-dir /dev/shm/aux --workers 24 [--max-logs 2]
"""
import argparse
import json
import os
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_features import VisionLLMLoRAAuxTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import dump_feature_target_to_pickle

_STATE: Dict = {}  # per worker process: scene filter config, paths, builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-config", type=Path, required=True,
                        help="a training run's code/hydra/config.yaml (scene filter, cache_path)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-logs", type=int, default=None, help="only the first n logs (test)")
    parser.add_argument("--shard", type=int, default=0, help="process logs[shard::num_shards] (parallel jobs)")
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def _init_worker(scene_filter_cfg, data_path: str, cache_path: str, out_dir: str) -> None:
    _STATE.update(
        scene_filter_cfg=scene_filter_cfg,
        data_path=Path(data_path),
        cache_path=Path(cache_path),
        out_dir=Path(out_dir),
        builder=VisionLLMLoRAAuxTargetBuilder(VisionLLMLoRAConfig().bev_semantic_classes),
    )


def _process_log(log: str) -> Tuple[str, int, List[str], List[str]]:
    """Targets for every cached scene of one log; returns (log, written, tokens not found in the log, errors)."""
    builder = _STATE["builder"]
    cached = sorted(p.name for p in (_STATE["cache_path"] / log).iterdir() if p.is_dir())
    scene_filter = instantiate(_STATE["scene_filter_cfg"])
    scene_filter.log_names = [log]
    loader = SceneLoader(
        _STATE["data_path"], _STATE["data_path"], scene_filter, sensor_config=SensorConfig.build_no_sensors()
    )
    available = set(loader.tokens)
    written, missing, errors = 0, [], []
    for token in cached:
        if token not in available:
            missing.append(token)
            continue
        path = _STATE["out_dir"] / log / token / f"{builder.get_unique_name()}.gz"
        if path.exists():  # resumed run: written by an earlier job (files are written to a .tmp name, then renamed)
            written += 1
            continue
        try:
            targets = builder.compute_targets(loader.get_scene_from_token(token))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            dump_feature_target_to_pickle(tmp, targets)
            tmp.rename(path)  # a job killed mid-write leaves only a .tmp, which a rerun overwrites
            written += 1
        except Exception:  # keep going; failures are listed in the summary
            errors.append(f"{token}: {traceback.format_exc(limit=2)}")
    return log, written, missing, errors


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.run_config)
    cache_path = Path(cfg.cache_path)
    data_path = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs" / cfg.train_test_split.data_split
    logs = sorted(p.name for p in cache_path.iterdir() if p.is_dir())[: args.max_logs][args.shard :: args.num_shards]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(logs)} logs from {cache_path}, navsim logs {data_path}, {args.workers} workers", flush=True)

    start, done, total_written, all_missing, all_errors = time.time(), 0, 0, [], []
    scene_filter_cfg = OmegaConf.to_container(cfg.train_test_split.scene_filter, resolve=True)
    init_args = (scene_filter_cfg, str(data_path), str(cache_path), str(args.out_dir))
    with Pool(args.workers, initializer=_init_worker, initargs=init_args, maxtasksperchild=50) as pool:
        for log, written, missing, errors in pool.imap_unordered(_process_log, logs):
            done += 1
            total_written += written
            all_missing += [f"{log}/{t}" for t in missing]
            all_errors += [f"{log}/{e}" for e in errors]
            if done % 20 == 0 or done == len(logs):
                rate = total_written / (time.time() - start)
                print(f"{done}/{len(logs)} logs, {total_written} scenes, {rate:.1f} scenes/s, "
                      f"{len(all_missing)} missing, {len(all_errors)} errors", flush=True)

    builder = VisionLLMLoRAAuxTargetBuilder(VisionLLMLoRAConfig().bev_semantic_classes)
    summary = {
        "target_name": builder.get_unique_name(),
        "logs": len(logs), "scenes_written": total_written, "missing": all_missing, "errors": all_errors,
        "grid": {"x_range": [builder._x_min, builder._x_max], "y_range": [builder._y_min, builder._y_max],
                 "resolution": builder._resolution, "shape": list(builder._shape)},
        "bev_semantic_classes": {str(k): [v[0], [str(layer) for layer in v[1]]]
                                 for k, v in VisionLLMLoRAConfig().bev_semantic_classes.items()},
        "max_agents": builder._max_agents, "heading_bins": builder._heading_bins,
        "source_cache": str(cache_path), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    suffix = f"_shard{args.shard}of{args.num_shards}" if args.num_shards > 1 else ""
    (args.out_dir / f"aux_targets_info{suffix}.json").write_text(json.dumps(summary, indent=1))
    print(f"done: {total_written} scenes in {time.time() - start:.0f} s, {len(all_missing)} missing, "
          f"{len(all_errors)} errors", flush=True)


if __name__ == "__main__":
    main()
