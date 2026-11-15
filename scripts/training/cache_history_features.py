"""
Past-frame inputs for every scene of the training cache: the camera tiles of the 3 history frames (t-0.5, t-1.0,
t-1.5 s; VisionLLMLoRAHistoryFrameBuilder, uint8, one file per frame) and the 3 past ego statuses
(VisionLLMLoRAEgoHistoryBuilder), in the training cache layout <log>/<token>/<name>.gz:
  vision_llm_history_t1.gz, vision_llm_history_t2.gz, vision_llm_history_t3.gz, vision_llm_ego_history.gz
The current frame stays in the existing training cache (not recomputed). Tiles use the run config's camera settings
(layout, thumbnail, tile size, channel order), so they match its cached current frame pixel for pixel.

Project file quota: the files are written to --stage-dir (RAM, /dev/shm), then this shard is packed into ONE tar
(<out-dir>/history_shard<i>of<n>.tar) and the stage is removed; scripts/training/run_cache_history_features_slurm.sh
streams all shard tars into one squashfs image. A shard whose tar exists is skipped (resubmit failed shards alone).

  python scripts/training/cache_history_features.py --run-config exp/p3_aux/<date>/code/hydra/config.yaml \
      --sensor-path <mount> [<mount> ...] --stage-dir /dev/shm/hist --out-dir exp/history_cache_stage --workers 8
"""
import argparse
import json
import os
import shutil
import tarfile
import time
import traceback
from multiprocessing import Pool, current_process
from pathlib import Path
from typing import Dict, List, Tuple

from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.agents.vision_llm_lora.vision_llm_lora_features import (
    VisionLLMLoRAEgoHistoryBuilder,
    VisionLLMLoRAHistoryFrameBuilder,
)
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import dump_feature_target_to_pickle

_STATE: Dict = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-config", type=Path, required=True,
                        help="a training run's code/hydra/config.yaml (scene filter, cache_path, camera settings)")
    parser.add_argument("--sensor-path", type=Path, nargs="+", required=True,
                        help="mount(s) of the navtrain sensor-blob image; workers share them round-robin")
    parser.add_argument("--stage-dir", type=Path, required=True, help="per-scene files before packing (RAM)")
    parser.add_argument("--out-dir", type=Path, required=True, help="where the shard tar is written")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-logs", type=int, default=None, help="only the first n logs (test)")
    parser.add_argument("--shard", type=int, default=0, help="process logs[shard::num_shards]")
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def history_sensor_config(config) -> SensorConfig:
    """The cameras of the agent's layout, history frames 0-2 (the current frame 3 is already cached)."""
    steps = [0, 1, 2]
    panorama = config.camera_layout == "panorama"
    return SensorConfig(
        cam_f0=steps, cam_l0=steps, cam_r0=steps, cam_b0=steps,
        cam_l1=steps if panorama else False, cam_r1=steps if panorama else False,
        cam_l2=False if panorama else steps, cam_r2=False if panorama else steps,
        lidar_pc=False,
    )


def _init_worker(agent_config, scene_filter_cfg, data_path: str, cache_path: str, stage: str, mounts: List[str]) -> None:
    index = (current_process()._identity or (1,))[0] - 1
    _STATE.update(
        scene_filter_cfg=scene_filter_cfg, data_path=Path(data_path), cache_path=Path(cache_path), stage=Path(stage),
        mount=Path(mounts[index % len(mounts)]), sensor_config=history_sensor_config(agent_config),
        builders=[VisionLLMLoRAHistoryFrameBuilder(agent_config, k) for k in (1, 2, 3)] + [VisionLLMLoRAEgoHistoryBuilder()],
    )


def _process_log(log: str) -> Tuple[str, int, List[str], List[str]]:
    """History features of every cached scene of one log; returns (log, written, tokens not in the log, errors)."""
    cached = sorted(p.name for p in (_STATE["cache_path"] / log).iterdir() if p.is_dir())
    scene_filter = instantiate(_STATE["scene_filter_cfg"])
    scene_filter.log_names = [log]
    loader = SceneLoader(_STATE["data_path"], _STATE["mount"], scene_filter, sensor_config=_STATE["sensor_config"])
    available = set(loader.tokens)
    written, missing, errors = 0, [], []
    for token in cached:
        if token not in available:
            missing.append(token)
            continue
        folder = _STATE["stage"] / log / token
        if all((folder / f"{b.get_unique_name()}.gz").exists() for b in _STATE["builders"]):
            written += 1
            continue
        try:
            agent_input = loader.get_agent_input_from_token(token)
            folder.mkdir(parents=True, exist_ok=True)
            for builder in _STATE["builders"]:
                path = folder / f"{builder.get_unique_name()}.gz"
                tmp = path.with_suffix(".tmp")
                dump_feature_target_to_pickle(tmp, builder.compute_features(agent_input))
                tmp.rename(path)
            written += 1
        except Exception:
            errors.append(f"{token}: {traceback.format_exc(limit=2)}")
    return log, written, missing, errors


def main() -> None:
    args = parse_args()
    shard_tar = args.out_dir / f"history_shard{args.shard}of{args.num_shards}.tar"
    if shard_tar.exists():
        print(f"{shard_tar} exists, nothing to do")
        return
    cfg = OmegaConf.load(args.run_config)
    agent_config = instantiate(cfg.agent.config)
    cache_path = Path(cfg.cache_path)
    data_path = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs" / cfg.train_test_split.data_split
    logs = sorted(p.name for p in cache_path.iterdir() if p.is_dir())[: args.max_logs][args.shard :: args.num_shards]
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"shard {args.shard}/{args.num_shards}: {len(logs)} logs, {args.workers} workers, "
          f"{len(args.sensor_path)} sensor mounts", flush=True)

    start, done, total, all_missing, all_errors = time.time(), 0, 0, [], []
    init_args = (agent_config, OmegaConf.to_container(cfg.train_test_split.scene_filter, resolve=True), str(data_path),
                 str(cache_path), str(args.stage_dir), [str(m) for m in args.sensor_path])
    with Pool(args.workers, initializer=_init_worker, initargs=init_args, maxtasksperchild=20) as pool:
        for log, written, missing, errors in pool.imap_unordered(_process_log, logs):
            done, total = done + 1, total + written
            all_missing += [f"{log}/{t}" for t in missing]
            all_errors += [f"{log}/{e}" for e in errors]
            if done % 5 == 0 or done == len(logs):
                print(f"  {done}/{len(logs)} logs, {total} scenes, {total / (time.time() - start):.1f} scenes/s, "
                      f"{len(all_missing)} missing, {len(all_errors)} errors", flush=True)
    if all_errors:
        print("\n".join(all_errors[:5]), flush=True)
        raise SystemExit(f"{len(all_errors)} scenes failed; the stage is kept, rerun this shard")
    summary = {"shard": args.shard, "num_shards": args.num_shards, "logs": len(logs), "scenes_written": total,
               "missing": all_missing, "source_cache": str(cache_path), "run_config": str(args.run_config),
               "seconds": time.time() - start, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (args.stage_dir / f"history_info_shard{args.shard}of{args.num_shards}.json").write_text(json.dumps(summary, indent=1))
    tmp = shard_tar.with_suffix(".tar.tmp")
    with tarfile.open(tmp, "w") as tar:  # uncompressed: the .gz members are already compressed
        for entry in sorted(args.stage_dir.iterdir()):
            tar.add(entry, arcname=entry.name)
    tmp.rename(shard_tar)
    shutil.rmtree(args.stage_dir)
    print(f"done: {total} scenes in {time.time() - start:.0f} s -> {shard_tar} "
          f"({shard_tar.stat().st_size / 1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
