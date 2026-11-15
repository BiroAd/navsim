"""
Model outputs on the navtrain validation scenes (val_logs of default_train_val_test_log_split.yaml) from the training
feature cache, in the format of predict_vision_llm_lora.py (<out-dir>/navtrain_val/: tokens.json, trajectory_logits.npy,
trajectory.npy, pdm_score_logits.npy if the model has sub-score heads, codebook.npy, meta.json). The cached features
are the ones training reads (no camera images, no squashfs mounts); the inputs equal those of the training-time
validation. Used to choose selection rules on held-out data (scripts/evaluation/eval_selection_rules_val.py) instead of
on navtest / navhard.

    python scripts/evaluation/predict_vision_llm_lora_cache.py --run-dir $NAVSIM_EXP_ROOT/rl3_b0.02/2026.11.08.21.48.32 \
        --checkpoint checkpoints/best/best-epoch=01.ckpt --out-dir $NAVSIM_EXP_ROOT/predictions/rl3_b0.02_ep01
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_vision_llm_lora import inference_only, run_overrides, to_numpy  # noqa: E402

from navsim.planning.training.dataset import CacheOnlyDataset  # noqa: E402

SPLIT = "navtrain_val"
LOG_SPLIT = Path(__file__).resolve().parents[2] / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="training run folder (has code/hydra/overrides.yaml)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint, relative to --run-dir or absolute")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, default=Path(os.environ.get("NAVSIM_EXP_ROOT", ".")) / "training_cache")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=None, help="smoke test: first N scenes")
    parser.add_argument("overrides", nargs="*", help="extra Hydra overrides, e.g. ++agent.config.x=y")
    return parser.parse_args()


class CachedFeatures(Dataset):
    def __init__(self, cache: CacheOnlyDataset, tokens: List[str]):
        self._cache, self._tokens = cache, tokens

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, index: int):
        features, _ = self._cache._load_scene_with_token(self._tokens[index])
        return index, features


def main() -> None:
    start = time.time()
    args = parse_args()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else args.run_dir / args.checkpoint
    assert checkpoint.is_file(), checkpoint
    overrides = ["agent=vision_llm_lora_agent", f"agent.checkpoint_path='{checkpoint}'",
                 *run_overrides(args.run_dir), *args.overrides]
    with initialize_config_module(config_module="navsim.planning.script.config.pdm_scoring", version_base=None):
        cfg = compose("default_run_pdm_score", overrides=["experiment_name=predict", *overrides])
    agent = instantiate(cfg.agent)
    agent.initialize()
    inference_only(agent)
    agent.eval().to("cuda")

    val_logs = yaml.safe_load(LOG_SPLIT.read_text())["val_logs"]
    cache = CacheOnlyDataset(str(args.cache_path), agent.get_feature_builders(), [], log_names=val_logs)
    tokens = sorted(cache.tokens)[: args.max_scenes]
    print(f"setup done after {time.time() - start:.0f} s; {SPLIT}: {len(tokens)} scenes in {len(cache.log_names)} logs",
          flush=True)

    out = args.out_dir / SPLIT
    out.mkdir(parents=True, exist_ok=True)
    data = DataLoader(CachedFeatures(cache, tokens), batch_size=args.batch_size, num_workers=args.num_workers,
                      shuffle=False, pin_memory=True)
    dtype = getattr(torch, agent._config.torch_dtype)
    autocast = dtype in (torch.float16, torch.bfloat16)
    arrays: Dict[str, np.ndarray] = {}
    stored_dtype = None
    t0 = time.time()
    for step, (index, features) in enumerate(data):
        features = {k: v.to("cuda", non_blocking=True) for k, v in features.items()}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast):
            predictions = agent.forward(features, teacher_forcing_seq=None, teacher_forcing_ratio=1.0)
        keep = {"trajectory_logits": predictions["trajectory_logits"], "trajectory": predictions["trajectory"].float()}
        if "pdm_score_logits" in predictions:
            keep["pdm_score_logits"] = predictions["pdm_score_logits"]
        if stored_dtype is None:
            stored_dtype = str(keep["trajectory_logits"].dtype).removeprefix("torch.")
            for key, value in keep.items():
                first = to_numpy(value[:1])
                arrays[key] = open_memmap(out / f"{key}.npy", mode="w+", dtype=first.dtype,
                                          shape=(len(tokens), *first.shape[1:]))
        rows = index.numpy()
        for key, value in keep.items():
            arrays[key][rows] = to_numpy(value)
        if step % 10 == 0:
            done = rows[-1] + 1
            print(f"  {SPLIT} {done}/{len(tokens)} ({done / (time.time() - t0):.1f} scenes/s)", flush=True)
    for array in arrays.values():
        array.flush()
    np.save(out / "codebook.npy", agent._vlm_model.codebook.float().cpu().numpy())
    (out / "tokens.json").write_text(json.dumps(tokens))
    (out / "meta.json").write_text(json.dumps({
        "run_dir": str(args.run_dir), "checkpoint": str(checkpoint), "overrides": overrides, "split": SPLIT,
        "cache_path": str(args.cache_path), "num_scenes": len(tokens), "logits_dtype": stored_dtype,
        "batch_size": args.batch_size, "seconds": time.time() - t0}, indent=1))
    print(f"{SPLIT}: wrote {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
