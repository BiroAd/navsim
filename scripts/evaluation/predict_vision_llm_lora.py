"""
GPU half of the split official evaluation: run a vision_llm_lora checkpoint over the scenes of navtest and / or
navhard_two_stage in batches and save the raw model outputs per scene. The CPU half
(scripts/evaluation/run_vision_llm_lora_score_predictions_slurm.sh) applies a trajectory selection to them and runs the
unchanged NAVSIM scoring scripts with VisionLLMLoRAPrecomputedAgent. The simulation never feeds back into the model's
inputs (NAVSIM is open loop per scene; navhard's stage two scenes are pre-generated), so splitting gives the same
scores as the combined run, up to bf16 differences between batched and single-scene inference.

The agent is built exactly like run_vision_llm_lora_pdm_score_slurm.sh builds it: agent=vision_llm_lora_agent, the
run's ++agent.config.* training overrides, agent.checkpoint_path, agent.initialize().

Output, per split, in <out-dir>/<split>/:
  tokens.json               scene tokens, row order of the arrays
  trajectory_logits.npy     (N, K) policy logits
  pdm_score_logits.npy      (N, M, K) sub-score head logits (if the model has heads)
  trajectory.npy            (N, 8, 3) the model's own output (argmax entry, refined if the model has a refiner)
  codebook.npy              (K, 8, 3) codebook the selections index into
  meta.json                 checkpoint, run, overrides, dtype of the logit arrays, batch size
Logit arrays are stored in the dtype the model produced; bfloat16 is stored bit-exact as int16 ("dtype": "bfloat16" in
meta.json, read back with torch.from_numpy(a).view(torch.bfloat16)).

    python scripts/evaluation/predict_vision_llm_lora.py --run-dir $NAVSIM_EXP_ROOT/p3_aux_policy/2026.11.06.23.42.11 \
        --checkpoint checkpoints/merged/epoch-epoch=03.ckpt --original-sensor-path <mounted test_sensor_blobs> \
        --splits navtest navhard_two_stage --out-dir $NAVSIM_EXP_ROOT/predictions/p3_aux_policy_ep03
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset

import navsim.common.dataloader as dataloader
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))  # run_pdm_score_balanced, also when imported by tests
from run_pdm_score_balanced import fast_filter_synthetic_scenes  # noqa: E402

# navhard's synthetic scenes from a metadata index instead of unpickling all 5,462 scenes (~2 min on one core).
dataloader.filter_synthetic_scenes = fast_filter_synthetic_scenes

SPLITS = ("navtest", "navhard_two_stage")
START = time.time()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="training run folder (has code/hydra/overrides.yaml)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint, relative to --run-dir or absolute")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--original-sensor-path", type=Path, nargs="+", required=True,
                        help="mounted test_sensor_blobs; several mounts of the same image are shared out over the "
                             "DataLoader workers (squashfuse serves one read at a time per mount)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader processes building the image features")
    parser.add_argument("--max-scenes", type=int, default=None, help="smoke test: first N scenes per split")
    parser.add_argument("overrides", nargs="*", help="extra Hydra overrides, e.g. ++agent.config.x=y")
    return parser.parse_args()


def run_overrides(run_dir: Path) -> List[str]:
    """The run's ++agent.config.* training overrides (training-only ones are harmless at inference)."""
    overrides = yaml.safe_load((run_dir / "code" / "hydra" / "overrides.yaml").read_text())
    return [o for o in overrides if o.startswith("++agent.config.")]


def split_config(split: str, overrides: List[str]):
    data_root = os.environ["OPENSCENE_DATA_ROOT"]
    split_overrides = [f"train_test_split={split}", f"metric_cache_path={os.environ['NAVSIM_EXP_ROOT']}/metric_cache_{split}"]
    if split == "navhard_two_stage":
        split_overrides += [
            f"synthetic_scenes_path={data_root}/navhard_two_stage/synthetic_scene_pickles",
            f"synthetic_sensor_path={data_root}/navhard_two_stage/sensor_blobs",
        ]
    return compose("default_run_pdm_score", overrides=["experiment_name=predict", *split_overrides, *overrides])


def scene_tokens(cfg, scene_filter, loader: SceneLoader) -> List[str]:
    """The tokens the scoring script evaluates: run_pdm_score_one_stage.py scores loader.tokens, run_pdm_score.py the
    stage one tokens and the reactive stage two tokens; both only where a metric cache exists."""
    cached = set(MetricCacheLoader(Path(cfg.metric_cache_path)).tokens)
    if scene_filter.include_synthetic_scenes:
        tokens = list(loader.tokens_stage_one) + list(loader.reactive_tokens_stage_two)
    else:
        tokens = list(loader.tokens)
    return sorted(t for t in set(tokens) if t in cached)


class FeatureDataset(Dataset):
    def __init__(self, loader: SceneLoader, tokens: List[str], builders):
        self._loader, self._tokens, self._builders = loader, tokens, builders

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, index: int):
        agent_input = self._loader.get_agent_input_from_token(self._tokens[index])
        features: Dict[str, torch.Tensor] = {}
        for builder in self._builders:
            features.update(builder.compute_features(agent_input))
        return index, features


def use_own_mount(worker_id: int) -> None:
    """DataLoader worker_init_fn: worker i reads the camera images through mount i % number of mounts."""
    info = torch.utils.data.get_worker_info()
    mounts = info.dataset.mounts
    info.dataset._loader._original_sensor_path = mounts[worker_id % len(mounts)]


def inference_only(agent) -> None:
    """
    Freeze every parameter and drop the hooks that make the backbone inputs require grad (as
    VisionLLMLoRAModel._freeze_all_but_policy_head does). Phase-3 models run the vision tower under torch.enable_grad
    (vision_llm_lora_model.py, forward: grad_ctx), which overrides the caller's no_grad: with trainable LoRA weights and
    the patch-embedding hook, autograd kept every vision-tower activation (batch 64: 40 GB, CUDA OOM in jobs
    14927247/50/54/61; policy-head-only models, already frozen, used 15.7 GB). Forward values are unchanged.
    """
    agent.requires_grad_(False)
    model = agent._vlm_model
    hook = getattr(model, "_vision_grad_hook", None)
    if hook is not None:
        hook.remove()
        model._vision_grad_hook = None
    for module in (model.model.language_model, model.model.visual):
        if hasattr(module, "disable_input_require_grads"):
            module.disable_input_require_grads()


def to_numpy(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu()
    return (x.view(torch.int16) if x.dtype == torch.bfloat16 else x).numpy()


def build_scenes(cfg, args, sensor_config) -> Tuple[SceneLoader, List[str]]:
    """Scene loader and the tokens to predict of one split (CPU only; ~1 min navtest, ~2 min navhard)."""
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    synthetic = scene_filter.include_synthetic_scenes
    loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        original_sensor_path=args.original_sensor_path[0],
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path) if synthetic else None,
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path) if synthetic else None,
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )
    return loader, scene_tokens(cfg, scene_filter, loader)[: args.max_scenes]


def predict_split(agent, split: str, loader: SceneLoader, tokens: List[str], args, meta: dict) -> None:
    out = args.out_dir / split
    out.mkdir(parents=True, exist_ok=True)
    print(f"{split}: {len(tokens)} scenes")
    dataset = FeatureDataset(loader, tokens, agent.get_feature_builders())
    dataset.mounts = args.original_sensor_path
    data = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                      pin_memory=True, worker_init_fn=use_own_mount)
    dtype = getattr(torch, agent._config.torch_dtype)
    autocast = dtype in (torch.float16, torch.bfloat16)
    arrays: Dict[str, np.ndarray] = {}
    stored_dtype = None
    start = time.time()
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
                array = to_numpy(value[:1])
                arrays[key] = open_memmap(out / f"{key}.npy", mode="w+", dtype=array.dtype,
                                          shape=(len(tokens), *array.shape[1:]))
        rows = index.numpy()
        for key, value in keep.items():
            arrays[key][rows] = to_numpy(value)
        if step % 10 == 0:
            done = rows[-1] + 1
            print(f"  {split} {done}/{len(tokens)} ({done / (time.time() - start):.1f} scenes/s)", flush=True)
    for array in arrays.values():
        array.flush()
    np.save(out / "codebook.npy", agent._vlm_model.codebook.float().cpu().numpy())
    (out / "tokens.json").write_text(json.dumps(tokens))
    (out / "meta.json").write_text(json.dumps({**meta, "split": split, "num_scenes": len(tokens),
                                               "logits_dtype": stored_dtype, "seconds": time.time() - start}, indent=1))
    print(f"{split}: wrote {out} in {time.time() - start:.0f} s")


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else args.run_dir / args.checkpoint
    assert checkpoint.is_file(), checkpoint
    overrides = ["agent=vision_llm_lora_agent", f"agent.checkpoint_path='{checkpoint}'",
                 *run_overrides(args.run_dir), *args.overrides]
    meta = {"run_dir": str(args.run_dir), "checkpoint": str(checkpoint), "overrides": overrides,
            "batch_size": args.batch_size}
    with initialize_config_module(config_module="navsim.planning.script.config.pdm_scoring", version_base=None):
        cfgs = {split: split_config(split, overrides) for split in args.splits}
    agent = instantiate(cfgs[args.splits[0]].agent)
    # Build the scene lists in a thread while the weights load and move to the GPU; joined before the DataLoader
    # forks its workers (no thread may run during a fork).
    with ThreadPoolExecutor(max_workers=1) as pool:
        scenes = {split: pool.submit(build_scenes, cfgs[split], args, agent.get_sensor_config()) for split in args.splits}
        agent.initialize()
        inference_only(agent)
        agent.eval().to("cuda")
        scenes = {split: future.result() for split, future in scenes.items()}
    print(f"setup done after {time.time() - START:.0f} s", flush=True)
    for split in args.splits:
        predict_split(agent, split, *scenes[split], args, meta)


if __name__ == "__main__":
    main()
