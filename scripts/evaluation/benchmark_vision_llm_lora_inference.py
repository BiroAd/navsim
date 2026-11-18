"""
Inference latency, throughput and GPU memory of vision_llm_lora checkpoints at several batch sizes, plus the CPU-side
input preparation (scene loading with JPEG decoding, feature building) per scene.

GPU part: real inputs (the cached features of navtrain validation scenes, the ones training and
predict_vision_llm_lora_cache.py read), the model as the official evaluation runs it (inference_only, eval mode,
autocast in config.torch_dtype). Per batch size: warm-up iterations, then timed iterations with CUDA synchronisation;
latency of one batch (mean / median / p90 / p99), scenes per second, the vision tower / LLM share (CUDA events in forward
hooks, no extra synchronisation) and the peak allocated / reserved GPU memory (torch.cuda peak statistics, reset per
batch size; nvidia-smi additionally shows the CUDA context, ~0.5 GB). Selection (argmax / mixture / shortlist over the
16384 codebook entries) is timed on the policy logits and, with --scorer-run-dir, on the scorer model's sub-score logits
(the RL-3 models have no sub-score heads; shortlist / mixture take them from the start model, i.e. a second forward).

CPU part (--prep-scenes N): SceneLoader.get_agent_input_from_token (reads and decodes the 6 camera JPEGs) and
feature building for N navtest scenes.

    python scripts/evaluation/benchmark_vision_llm_lora_inference.py \
        --run-dir $NAVSIM_EXP_ROOT/rl3_b0.005/2026.11.08.23.46.54 --checkpoint checkpoints/epochs/epoch-epoch=01.ckpt \
        --scorer-run-dir $NAVSIM_EXP_ROOT/p3_aux_policy/2026.11.06.23.42.11 \
        --scorer-checkpoint checkpoints/merged/epoch-epoch=03.ckpt --batch-sizes 1 2 4 8 --out benchmark.json
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
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_vision_llm_lora import inference_only, run_overrides  # noqa: E402

from navsim.agents.vision_llm_lora.vision_llm_lora_agent import select_trajectory  # noqa: E402
from navsim.planning.training.dataset import CacheOnlyDataset  # noqa: E402

LOG_SPLIT = Path(__file__).resolve().parents[2] / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="policy run folder (has code/hydra/overrides.yaml)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="relative to --run-dir or absolute")
    parser.add_argument("--scorer-run-dir", type=Path, default=None, help="optional model with sub-score heads")
    parser.add_argument("--scorer-checkpoint", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=Path(os.environ.get("NAVSIM_EXP_ROOT", ".")) / "training_cache")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--scenes", type=int, default=64, help="distinct cached scenes cycled through")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--prep-scenes", type=int, default=0, help="CPU input preparation on N navtest scenes (0 = off)")
    parser.add_argument("--sensor-path", type=Path, default=None,
                        help="navtest camera images (default $OPENSCENE_DATA_ROOT/sensor_blobs/test; the test images "
                        "are in images/test_sensor_blobs.sqfs, mount it with squashfuse)")
    parser.add_argument("--out", type=Path, default=None, help="results as JSON")
    return parser.parse_args()


def load_agent(run_dir: Path, checkpoint: Path):
    checkpoint = checkpoint if checkpoint.is_absolute() else run_dir / checkpoint
    assert checkpoint.is_file(), checkpoint
    overrides = ["agent=vision_llm_lora_agent", f"agent.checkpoint_path='{checkpoint}'", *run_overrides(run_dir)]
    with initialize_config_module(config_module="navsim.planning.script.config.pdm_scoring", version_base=None):
        cfg = compose("default_run_pdm_score", overrides=["experiment_name=benchmark", *overrides])
    agent = instantiate(cfg.agent)
    agent.initialize()
    inference_only(agent)
    return agent.eval().to("cuda"), str(checkpoint)


class StageTimer:
    """CUDA events around the vision tower and the language model (forward hooks; read after synchronize)."""

    def __init__(self, agent):
        model = agent._vlm_model.model
        self.events: Dict[str, List] = {"vision": [], "llm": []}
        for name, module in (("vision", model.visual), ("llm", model.language_model)):
            module.register_forward_pre_hook(self._pre(name))
            module.register_forward_hook(self._post(name))

    def _pre(self, name):
        def hook(module, args):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self.events[name].append([start, None])
        return hook

    def _post(self, name):
        def hook(module, args, output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.events[name][-1][1] = end
        return hook

    def reset(self):
        for value in self.events.values():
            value.clear()

    def per_iteration_ms(self, iterations: int) -> Dict[str, float]:
        return {name: sum(s.elapsed_time(e) for s, e in pairs) / max(1, iterations) for name, pairs in self.events.items()}


def stats(ms: List[float]) -> Dict[str, float]:
    a = np.asarray(ms)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)), "p90_ms": float(np.percentile(a, 90)),
            "p99_ms": float(np.percentile(a, 99)), "min_ms": float(a.min()), "max_ms": float(a.max())}


def benchmark_model(agent, scenes: List[Dict[str, torch.Tensor]], batch_sizes: List[int], warmup: int,
                    iterations: int) -> Dict:
    dtype = getattr(torch, agent._config.torch_dtype)
    autocast = dtype in (torch.float16, torch.bfloat16)
    timer = StageTimer(agent)
    codebook = agent._vlm_model.codebook
    results = {}
    for batch_size in batch_sizes:
        batches = []
        for i in range(0, len(scenes) - batch_size + 1, batch_size):
            batch = default_collate(scenes[i : i + batch_size])
            batches.append({k: v.to("cuda") for k, v in batch.items()})
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        latencies, outputs = [], None
        for it in range(warmup + iterations):
            features = batches[it % len(batches)]
            if it == warmup:
                timer.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast):
                outputs = agent.forward(features, teacher_forcing_seq=None, teacher_forcing_ratio=1.0)
            torch.cuda.synchronize()
            if it >= warmup:
                latencies.append((time.perf_counter() - t0) * 1000)
        entry = {"batch": stats(latencies), "per_scene_ms": float(np.mean(latencies)) / batch_size,
                 "scenes_per_s": 1000.0 * batch_size / float(np.mean(latencies)),
                 "stages_ms": timer.per_iteration_ms(iterations),
                 "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                 "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}
        # selection on the last outputs (GPU), argmax always; shortlist / mixture where sub-score logits exist
        selection_ms = {}
        rules = ["argmax"] + (["shortlist", "mixture", "mixture_v1"] if "pdm_score_logits" in outputs else [])
        for rule in rules:
            times = []
            for _ in range(20):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                select_trajectory(outputs, rule, codebook, shortlist_size=256, mixture_lambda=10.0)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
            selection_ms[rule] = float(np.median(times))
        entry["selection_ms"] = selection_ms
        results[batch_size] = entry
        print(f"  batch {batch_size}: {entry['batch']['mean_ms']:.1f} ms (median {entry['batch']['median_ms']:.1f}, "
              f"p99 {entry['batch']['p99_ms']:.1f}), {entry['scenes_per_s']:.1f} scenes/s, vision "
              f"{entry['stages_ms']['vision']:.1f} ms, LLM {entry['stages_ms']['llm']:.1f} ms, peak "
              f"{entry['peak_allocated_gb']:.2f} GB allocated / {entry['peak_reserved_gb']:.2f} GB reserved, "
              f"selection {selection_ms}", flush=True)
    return results


def prep_benchmark(agent, n: int, sensor_path=None) -> Dict:
    """CPU: scene loading (JPEG decoding of the cameras the agent asks for) and feature building, per scene."""
    from omegaconf import OmegaConf

    from navsim.common.dataloader import SceneLoader

    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    config_dir = Path(__file__).resolve().parents[2] / "navsim/planning/script/config/common"
    scene_filter = instantiate(OmegaConf.load(config_dir / "train_test_split/scene_filter/navtest.yaml"))
    scene_filter.max_scenes = n
    loader = SceneLoader(data_path=data_root / "navsim_logs/test", original_sensor_path=sensor_path or data_root / "sensor_blobs/test",
                         scene_filter=scene_filter, sensor_config=agent.get_sensor_config())
    builders = agent.get_feature_builders()
    load_ms, build_ms = [], []
    for token in list(loader.tokens)[:n]:
        t0 = time.perf_counter()
        agent_input = loader.get_agent_input_from_token(token)
        t1 = time.perf_counter()
        for builder in builders:
            builder.compute_features(agent_input)
        t2 = time.perf_counter()
        load_ms.append((t1 - t0) * 1000)
        build_ms.append((t2 - t1) * 1000)
    result = {"scenes": len(load_ms), "load_and_decode": stats(load_ms[1:] or load_ms),
              "feature_building": stats(build_ms[1:] or build_ms)}
    print(f"  CPU input preparation ({len(load_ms)} scenes, first excluded): load + decode "
          f"{result['load_and_decode']['median_ms']:.0f} ms, features {result['feature_building']['median_ms']:.0f} ms "
          f"(median per scene, one process)", flush=True)
    return result


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    info = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "batch_sizes": args.batch_sizes,
            "warmup": args.warmup, "iterations": args.iterations, "cpus": os.environ.get("SLURM_CPUS_PER_TASK")}
    print(info, flush=True)
    models = [("policy", args.run_dir, args.checkpoint)]
    if args.scorer_run_dir:
        models.append(("scorer", args.scorer_run_dir, args.scorer_checkpoint))
    scenes = None
    results = {"info": info}
    for role, run_dir, checkpoint in models:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        agent, path = load_agent(run_dir, checkpoint)
        torch.cuda.synchronize()
        weights_gb = torch.cuda.memory_allocated() / 2**30
        print(f"{role}: {path}; weights on GPU {weights_gb:.2f} GB", flush=True)
        if scenes is None:
            val_logs = yaml.safe_load(LOG_SPLIT.read_text())["val_logs"]
            cache = CacheOnlyDataset(str(args.cache_path), agent.get_feature_builders(), [], log_names=val_logs)
            tokens = sorted(cache.tokens)[: args.scenes]
            scenes = [cache._load_scene_with_token(t)[0] for t in tokens]
            shapes = {k: list(v.shape) for k, v in scenes[0].items()}
            print(f"{len(scenes)} cached scenes, features {shapes}", flush=True)
            results["info"]["features"] = shapes
        results[role] = {"checkpoint": path, "weights_gb": weights_gb,
                         "batches": benchmark_model(agent, scenes, args.batch_sizes, args.warmup, args.iterations)}
        if role == "policy" and args.prep_scenes:
            # kept on the CPU for the input-preparation part below (it needs only the feature builders and sensor
            # config; run last because it reads the camera images and may fail)
            prep_agent = agent.cpu()
        else:
            del agent
    if args.prep_scenes:
        try:
            results["cpu_input_preparation"] = prep_benchmark(prep_agent, args.prep_scenes, args.sensor_path)
        except Exception as error:  # noqa: BLE001
            print(f"CPU input preparation failed: {error!r}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
