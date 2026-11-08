"""
Pre-caching check for vision_llm_lora. Runs the caching code path on a few real navtrain scenes: agent YAML config,
the agent's sensor config, the feature/target builders and the gzip cache format. The 4B model is not loaded.
Saves a PNG per scene showing what the model will see, and prints READY / NOT READY for dataset caching.

Run where the navtrain image is mounted (a CPU allocation or the login node), with the env active:
  source bashrc && source /project/nr_e2e_vlm/envs/navsim_vlm/bin/activate
  squashfuse $OPENSCENE_DATA_ROOT/images/navtrain_sensor_blobs.sqfs $OPENSCENE_DATA_ROOT/sensor_blobs/trainval
  OMP_NUM_THREADS=4 python scripts/training/preview_vision_llm_lora_features.py --num-scenes 4
(OMP_NUM_THREADS: on the 96-core login node, OpenMP's default one-thread-per-core start fails and exits silently.)
  fusermount -u $OPENSCENE_DATA_ROOT/sensor_blobs/trainval
"""

import argparse
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List

import cv2
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.agents.vision_llm_lora.vision_llm_lora_agent import VisionLLMLoRAAgent
from navsim.agents.vision_llm_lora.vision_llm_lora_features import (
    IMAGE_NORMALIZATION,
    VisionLLMLoRAFeatureBuilder,
    VisionLLMLoRATargetBuilder,
)
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import dump_feature_target_to_pickle, load_feature_target_from_pickle

CONFIG_DIR = Path(__file__).resolve().parents[2] / "navsim/planning/script/config/common"
NAVTRAIN_SCENES = 103288
VALUE_RANGE = {"qwen": (-1.0, 1.0), "imagenet": (-2.2, 2.7)}


class Report:
    """Collects check results; any failure makes the verdict NOT READY."""

    def __init__(self) -> None:
        self.failures: List[str] = []
        self.warnings: List[str] = []

    def check(self, ok: bool, message: str) -> bool:
        print(f"  [{'ok' if ok else 'FAIL'}] {message}")
        if not ok:
            self.failures.append(message)
        return ok

    def warn(self, message: str) -> None:
        print(f"  [warn] {message}")
        self.warnings.append(message)


def check_environment(report: Report, cache_path: Path) -> bool:
    print("Environment")
    names = ("NAVSIM_DEVKIT_ROOT", "OPENSCENE_DATA_ROOT", "NUPLAN_MAPS_ROOT", "NAVSIM_EXP_ROOT")
    if not all(report.check(bool(os.environ.get(name)), f"${name} is set") for name in names):
        return False

    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    logs = data_root / "navsim_logs/trainval"
    mount = data_root / "sensor_blobs/trainval"
    report.check(logs.is_dir() and len(list(logs.glob("*.pkl"))) == 1310, f"{logs} holds the 1310 trainval logs")
    report.check((Path(os.environ["NUPLAN_MAPS_ROOT"]) / "nuplan-maps-v1.0.json").exists(), "nuPlan maps found")
    mounted = mount.is_dir() and any(mount.iterdir())
    report.check(mounted, f"navtrain image mounted at {mount} (see the docstring for the squashfuse command)")

    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache("Qwen/Qwen3.5-4B", "model.safetensors.index.json")
        report.check(isinstance(cached, str), "Qwen/Qwen3.5-4B is in the HF cache (caching workers load the model)")
    except ImportError:
        report.check(False, "huggingface_hub importable")

    if cache_path.is_dir() and any(cache_path.iterdir()):
        report.warn(
            f"{cache_path} is not empty: caching skips tokens that already have files there, so old features would "
            "be kept. Use an empty cache_path for the new preprocessing."
        )
    return mounted


def preview(scene_loader, tokens, feature_builder, target_builder, config, out_dir: Path, report: Report) -> None:
    tile = config.camera_width
    n_images = 12 if config.thumbnail == "none" else 13
    low, high = VALUE_RANGE[config.image_normalization]
    mean, std = (torch.tensor(v).view(3, 1, 1) for v in IMAGE_NORMALIZATION[config.image_normalization])
    sizes, seconds = [], []

    for token in tokens:
        print(f"Scene {token}")
        start = time.perf_counter()
        scene = scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()
        features = feature_builder.compute_features(agent_input)
        targets = target_builder.compute_targets(scene)
        seconds.append(time.perf_counter() - start)

        camera = features["camera_feature"]
        report.check(camera.shape == (n_images, 3, tile, tile), f"camera_feature {tuple(camera.shape)}")
        report.check(camera.dtype == torch.float32 and bool(torch.isfinite(camera).all()), "camera_feature finite")
        report.check(
            low - 1e-3 <= camera.min().item() and camera.max().item() <= high + 1e-3,
            f"camera_feature range [{camera.min().item():.2f}, {camera.max().item():.2f}] within [{low}, {high}]",
        )
        for key in ("velocity", "acceleration"):
            value = features[key]
            report.check(value.shape == (2,) and bool(torch.isfinite(value).all()), f"{key} {value.tolist()}")
        report.check(0 <= int(features["driving_command"]) <= 3, f"driving_command {int(features['driving_command'])}")
        trajectory = targets["trajectory"]
        report.check(
            trajectory.shape == (8, 3) and bool(torch.isfinite(trajectory).all()),
            f"trajectory {tuple(trajectory.shape)}, last pose {[round(v, 2) for v in trajectory[-1].tolist()]}",
        )

        with tempfile.TemporaryDirectory() as tmp:
            for name, data in (("vision_llm_feature", features), ("vision_llm_target", targets)):
                path = Path(tmp) / f"{name}.gz"
                dump_feature_target_to_pickle(path, data)
                sizes.append(path.stat().st_size)
                loaded = load_feature_target_from_pickle(path)
                report.check(all(torch.equal(loaded[k], data[k]) for k in data), f"{name}.gz round trip")

        # What the model sees: channels as the model reads them (RGB), denormalized.
        images = ((camera * std + mean).clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
        grid = np.concatenate([np.concatenate(list(images[row * 6 : row * 6 + 6]), axis=1) for row in range(2)], 0)
        side = np.zeros((2 * tile, tile, 3), dtype=np.uint8)
        if n_images == 13:
            side[:tile] = images[12]
        panel = np.concatenate([grid, np.full((2 * tile, 8, 3), 255, np.uint8), side], axis=1)
        raw_f0 = agent_input.cameras[-1].cam_f0.image  # RGB, straight from the JPEG, for a colour comparison
        raw_f0 = cv2.resize(raw_f0, (panel.shape[1], round(raw_f0.shape[0] * panel.shape[1] / raw_f0.shape[1])))
        banner = np.full((28, panel.shape[1], 3), 255, np.uint8)
        text = (
            f"{token}  cmd {int(features['driving_command'])}  v {features['velocity'][0].item():.1f} m/s  "
            f"top: tiles 1-12 + 13th image | bottom: raw CAM_F0"
        )
        cv2.putText(banner, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        figure = np.concatenate([banner, panel, raw_f0], axis=0)
        out_path = out_dir / f"{token}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(figure, cv2.COLOR_RGB2BGR))
        print(f"  saved {out_path}")

    per_scene = sum(sizes) / len(tokens)
    print("Estimates for the full navtrain cache")
    print(f"  disk  ~{per_scene * NAVTRAIN_SCENES / 1e9:.0f} GB ({per_scene / 1e6:.2f} MB per scene)")
    print(f"  files ~{3 * NAVTRAIN_SCENES:,} (one folder + two .gz per scene)")
    per_scene_s = float(np.median(seconds))
    print(f"  CPU   ~{per_scene_s * NAVTRAIN_SCENES / 3600:.1f} core-hours ({per_scene_s:.2f} s per scene)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-scenes", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: $NAVSIM_EXP_ROOT/preview_features")
    parser.add_argument("--cache-path", type=Path, default=None, help="default: $NAVSIM_EXP_ROOT/training_cache")
    args = parser.parse_args()

    report = Report()
    exp_root = Path(os.environ.get("NAVSIM_EXP_ROOT", "."))
    cache_path = args.cache_path or exp_root / "training_cache"
    if check_environment(report, cache_path):
        agent_cfg = OmegaConf.load(CONFIG_DIR / "agent/vision_llm_lora_agent.yaml")
        config = instantiate(agent_cfg.config)
        trajectory_sampling = instantiate(agent_cfg.trajectory_sampling)
        print("Agent config (from the YAML)")
        keys = ("camera_layout", "thumbnail", "camera_width", "swap_red_blue", "image_normalization", "patch_order")
        for key in keys:
            print(f"  {key} = {getattr(config, key)}")
        report.check(Path(config.codebook_path).exists(), f"codebook {config.codebook_path}")

        # The agent's own sensor config, called without building the agent (which would load the 4B model).
        sensor_config = VisionLLMLoRAAgent.get_sensor_config(SimpleNamespace(_config=config))
        scene_filter = instantiate(
            OmegaConf.load(CONFIG_DIR / "train_test_split/scene_filter/navtrain.yaml"), max_scenes=args.num_scenes
        )
        data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
        scene_loader = SceneLoader(
            data_path=data_root / "navsim_logs/trainval",
            original_sensor_path=data_root / "sensor_blobs/trainval",
            scene_filter=scene_filter,
            sensor_config=sensor_config,
        )
        out_dir = args.out_dir or exp_root / "preview_features"
        out_dir.mkdir(parents=True, exist_ok=True)
        preview(
            scene_loader,
            scene_loader.tokens[: args.num_scenes],
            VisionLLMLoRAFeatureBuilder(config),
            VisionLLMLoRATargetBuilder(trajectory_sampling, config),
            config,
            out_dir,
            report,
        )

    print()
    if report.failures:
        print(f"NOT READY: {len(report.failures)} check(s) failed")
        for failure in report.failures:
            print(f"  - {failure}")
    else:
        print("READY for dataset caching" + (f" ({len(report.warnings)} warning(s) above)" if report.warnings else ""))
        print("Also look at the PNGs: natural colours, sky blue, no grey seams, front zoom on the road ahead.")
    raise SystemExit(1 if report.failures else 0)


if __name__ == "__main__":
    main()
