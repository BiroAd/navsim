"""Figures of the aux BEV decoder (p3_aux) on navtrain val scenes, from the maps saved by
scripts/evaluation/eval_aux_maps.py (maps_<name>.npz):
  scene figure (one PNG per scene): front camera; ground truth vs prediction of the BEV semantics, the drivable area
               and the lane travel direction (hue = direction relative to the ego heading, off-lane cells blank).
  drift figure (--compare): predicted semantics and drivable area of the same decoder on other backbones
               (e.g. RL-3, and p3_heads as the no-aux floor), one row per scene.
Map layout (VisionLLMLoRAAuxTargetBuilder): rows x = -16..64 m forward, columns y = 32..-32 m (left to right), 0.5 m.
Needs the navtrain sensor image mounted (squashfuse $OPENSCENE_DATA_ROOT/images/navtrain_sensor_blobs.sqfs <dir>).
  python scripts/visualization/plot_aux_maps.py --maps $E/aux_eval/maps_p3_aux_ep05.npz --sensor-path <mount> \
      --out-dir $E/qualitative/aux [--compare $E/aux_eval/maps_p3_aux_ep05_on_rl3_b0.005_ep01.npz ...]
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from navsim.common.dataclasses import SensorConfig  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402

D = Path(os.environ["OPENSCENE_DATA_ROOT"])
REPO = Path(__file__).resolve().parents[2]
EXTENT = [32, -32, -16, 64]  # imshow (left, right, bottom, top) with origin="lower": forward up, left = +y
NAMES = ["none", "road", "walkway", "lane centreline", "static object", "vehicle", "pedestrian"]
COLORS = ["#ffffff", "#d9d9d9", "#d8cf9b", "#4d4d4d", "#ff7f0e", "#1f77b4", "#e377c2"]
SEM_CMAP = ListedColormap(COLORS)
NO_LANE = 255
BINS = 36


def loader_for(tokens, logs, sensor_path):
    cfg = OmegaConf.load(REPO / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml")
    sf = instantiate(cfg)
    sf.tokens = list(tokens)
    sf.log_names = sorted(set(logs))
    cams = dict(cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False)
    return SceneLoader(D / "navsim_logs/trainval", Path(sensor_path), sf,
                       sensor_config=SensorConfig(cam_f0=[3], lidar_pc=False, **cams))


def bev_ax(ax, title):
    ax.add_patch(Rectangle((-0.95, -1.0), 1.9, 4.9, facecolor="#d62728", edgecolor="black", lw=0.6, zorder=5))  # ego
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(-16, 64)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def show_sem(ax, m, title):
    ax.imshow(m, origin="lower", extent=EXTENT, cmap=SEM_CMAP, vmin=-0.5, vmax=6.5, interpolation="nearest")
    bev_ax(ax, title)


def show_drv(ax, prob, target, title):
    ax.imshow(prob.astype(np.float32), origin="lower", extent=EXTENT, cmap="Greens", vmin=0, vmax=1)
    ax.contour(np.linspace(32, -32, target.shape[1]), np.linspace(-16, 64, target.shape[0]), target.astype(float),
               levels=[0.5], colors="black", linewidths=0.8)
    bev_ax(ax, title)


def show_head(ax, m, title):
    a = np.where(m == NO_LANE, np.nan, (m.astype(float) + 0.5) / BINS * 360 - 180)
    im = ax.imshow(a, origin="lower", extent=EXTENT, cmap="twilight_shifted", vmin=-180, vmax=180,
                   interpolation="nearest")
    bev_ax(ax, title)
    return im


def camera(ax, scene):
    img = scene.frames[3].cameras.cam_f0.image
    ax.imshow(img)
    h = img.shape[0]
    ax.set_ylim(h * 0.99, h * 0.25)
    ax.axis("off")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maps", type=Path, required=True)
    p.add_argument("--compare", type=Path, nargs="*", default=[], help="maps of the same decoder on other backbones")
    p.add_argument("--labels", nargs="*", default=[], help="labels of --maps and each --compare")
    p.add_argument("--sensor-path", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    m = np.load(args.maps)
    tokens, logs = list(m["token"]), list(m["log"])
    comp = [np.load(c) for c in args.compare]
    labels = args.labels or [args.maps.stem.removeprefix("maps_")] + [c.stem.removeprefix("maps_") for c in args.compare]
    metrics = {}
    for f in [args.maps, *args.compare]:
        mf = f.parent / f.name.replace("maps_", "metrics_").replace(".npz", ".json")
        if mf.exists():
            metrics[f] = json.loads(mf.read_text())
    loader = loader_for(tokens, logs, args.sensor_path)
    out = args.out_dir / "scenes"
    out.mkdir(parents=True, exist_ok=True)
    legend = [Patch(facecolor=c, edgecolor="0.5", label=n) for n, c in zip(NAMES[1:], COLORS[1:])]
    index = ["# Aux BEV decoder: ground truth vs prediction (navtrain val)", "",
             f"Maps: `{args.maps}` (decoder and backbone {labels[0]}). Scenes: the random val scenes saved by "
             "eval_aux_maps.py (--seed). Metrics of the run over its evaluated scenes: "
             + json.dumps({k: round(v, 3) for k, v in metrics.get(args.maps, {}).items()
                           if k in ("bev_miou", "drivable_iou", "lane_heading_acc")}), "",
             "| File | Token | Log |", "|---|---|---|"]
    for i, (tok, log) in enumerate(zip(tokens, logs)):
        scene = loader.get_scene_from_token(tok)
        fig = plt.figure(figsize=(16, 8.2))
        gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.55], hspace=0.12, wspace=0.04)
        camera(fig.add_subplot(gs[0, :]), scene)
        show_sem(fig.add_subplot(gs[1, 0]), m["sem_target"][i], "BEV semantics: ground truth")
        show_sem(fig.add_subplot(gs[1, 1]), m["sem_pred"][i], "BEV semantics: predicted")
        show_drv(fig.add_subplot(gs[1, 2]), m["drv_target"][i], m["drv_target"][i], "drivable area: ground truth")
        show_drv(fig.add_subplot(gs[1, 3]), m["drv_prob"][i], m["drv_target"][i],
                 "drivable area: predicted p\n(black: ground-truth edge)")
        show_head(fig.add_subplot(gs[1, 4]), m["head_target"][i], "lane direction: ground truth")
        im = show_head(fig.add_subplot(gs[1, 5]), m["head_pred"][i], "lane direction: predicted")
        cb = fig.colorbar(im, ax=fig.axes[-2:], orientation="horizontal", fraction=0.04, pad=0.02)
        cb.set_label("travel direction relative to ego heading (deg; 0 = same as ego)", fontsize=8)
        fig.legend(handles=legend, loc="lower left", ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.01, 0.0))
        fig.suptitle(f"aux decoder ({labels[0]}) | {log} | token {tok}", fontsize=10, y=0.995)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.08)
        f = out / f"{i:02d}_{tok}.png"
        fig.savefig(f, dpi=args.dpi)
        plt.close(fig)
        index.append(f"| `scenes/{f.name}` | {tok} | {log} |")
        print(f, flush=True)
    (args.out_dir / "index.md").write_text("\n".join(index) + "\n")
    if comp:  # drift figure: rows = scenes (first 6), columns = backbones x (semantics, drivable)
        n = min(6, len(tokens))
        cols = 2 + 2 * (1 + len(comp))  # camera, GT semantics, then (semantics, drivable) per backbone
        fig, axes = plt.subplots(n, cols, figsize=(2.1 * cols + 2.5, 3.0 * n),
                                 gridspec_kw={"width_ratios": [2.4] + [1] * (cols - 1)})
        for r in range(n):
            camera(axes[r, 0], loader.get_scene_from_token(tokens[r]))
            show_sem(axes[r, 1], m["sem_target"][r], "GT semantics" if r == 0 else "")
            for k, (mm, lab) in enumerate(zip([m, *comp], labels)):
                j = [list(mm["token"]).index(tokens[r])]
                show_sem(axes[r, 2 + 2 * k], mm["sem_pred"][j[0]], f"semantics: {lab}" if r == 0 else "")
                if 3 + 2 * k < cols:
                    show_drv(axes[r, 3 + 2 * k], mm["drv_prob"][j[0]], m["drv_target"][r],
                             f"drivable: {lab}" if r == 0 else "")
        rows = [f"{lab}: mIoU {metrics[f]['bev_miou']:.3f}, drivable IoU {metrics[f]['drivable_iou']:.3f}, "
                f"heading acc {metrics[f]['lane_heading_acc']:.3f}" for f, lab in zip([args.maps, *args.compare], labels)
                if f in metrics]
        fig.suptitle("same aux decoder (p3_aux ep 5), different backbones | " + " | ".join(rows), fontsize=9)
        fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=8, frameon=False)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.04, wspace=0.04, hspace=0.1)
        f = args.out_dir / "drift_backbones.png"
        fig.savefig(f, dpi=args.dpi)
        plt.close(fig)
        print(f)


if __name__ == "__main__":
    main()
