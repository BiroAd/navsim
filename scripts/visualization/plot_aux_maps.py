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
COLORS = ["#ffffff", "#D3D3D3", "#d4d19e", "#4d4d4d", "#edc948", "#699CDB", "#b07aa1"]  # NAVSIM BEV colours (config.py)
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


def show_head(ax, m, title, mask=None):
    a = np.where(m == NO_LANE, np.nan, (m.astype(float) + 0.5) / BINS * 360 - 180)
    if mask is not None:  # the decoder outputs a direction everywhere; it is trained (and meaningful) on lanes only
        a = np.where(mask, a, np.nan)
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
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--drift-scenes", type=int, nargs="*", default=[0], help="scene indices of the backbone figures")
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
        gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.55], hspace=0.14, wspace=0.04)
        camera(fig.add_subplot(gs[0, :]), scene)
        show_sem(fig.add_subplot(gs[1, 0]), m["sem_target"][i], "BEV semantics: ground truth")
        show_sem(fig.add_subplot(gs[1, 1]), m["sem_pred"][i], "BEV semantics: predicted")
        show_drv(fig.add_subplot(gs[1, 2]), m["drv_target"][i], m["drv_target"][i], "drivable area: ground truth")
        show_drv(fig.add_subplot(gs[1, 3]), m["drv_prob"][i], m["drv_target"][i],
                 "drivable area: predicted p\n(black: ground-truth edge)")
        show_head(fig.add_subplot(gs[1, 4]), m["head_target"][i], "lane direction: ground truth")
        im = show_head(fig.add_subplot(gs[1, 5]), m["head_pred"][i],
                       "lane direction: predicted\n(shown where predicted drivable p > 0.5)",
                       mask=m["drv_prob"][i].astype(np.float32) > 0.5)
        cb = fig.colorbar(im, cax=fig.add_axes([0.70, 0.075, 0.25, 0.018]), orientation="horizontal")
        cb.set_label("travel direction relative to ego heading (deg; 0 = same as ego)", fontsize=8)
        fig.legend(handles=legend, loc="lower left", ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.01, 0.0))
        fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.14)
        f = out / f"{i:02d}_{tok}.png"
        fig.savefig(f, dpi=args.dpi)
        plt.close(fig)
        index.append(f"| `scenes/{f.name}` | {tok} | {log} |")
        print(f, flush=True)
    (args.out_dir / "index.md").write_text("\n".join(index) + "\n")
    if comp:  # backbone comparison, one figure per --drift-scenes index, sized for 0.8 x the thesis text width:
        # columns = ground truth and each backbone (--labels), rows = semantics, drivable area, lane direction
        plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 6.5})
        for r in args.drift_scenes:
            scene = loader.get_scene_from_token(tokens[r])
            sources = [("ground truth", None)] + list(zip(labels, [m, *comp]))
            n = len(sources)
            fig = plt.figure(figsize=(4.8, 6.7))
            gs = fig.add_gridspec(5, n, height_ratios=[0.80, 1.25, 1.25, 1.25, 0.50], hspace=0.22, wspace=0.03)
            cam = fig.add_subplot(gs[0, :])
            camera(cam, scene)
            for c, (lab, mm) in enumerate(sources):
                axs = [fig.add_subplot(gs[1 + k, c]) for k in range(3)]
                if mm is None:
                    show_sem(axs[0], m["sem_target"][r], "")
                    show_drv(axs[1], m["drv_target"][r], m["drv_target"][r], "")
                    im = show_head(axs[2], m["head_target"][r], "")
                else:
                    j = list(mm["token"]).index(tokens[r])
                    show_sem(axs[0], mm["sem_pred"][j], "")
                    show_drv(axs[1], mm["drv_prob"][j], m["drv_target"][r], "")
                    show_head(axs[2], mm["head_pred"][j], "", mask=mm["drv_prob"][j].astype(np.float32) > 0.5)
                axs[0].set_title(lab.replace(" auxiliary losses", "\naux losses").replace("after RL fine-tuning",
                                 "after RL\nfine-tuning").replace("ground truth", "ground\ntruth"), fontsize=7, pad=2)
                if c == 0:
                    for ax, t in zip(axs, ["BEV semantics", "drivable area", "lane direction"]):
                        ax.set_ylabel(t, fontsize=7.5)
            lax = fig.add_subplot(gs[4, :])
            lax.axis("off")
            lax.legend(handles=legend, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(-0.02, 1.05),
                       handlelength=1.4, columnspacing=0.8)
            cb = fig.colorbar(im, cax=fig.add_axes([0.58, 0.06, 0.38, 0.012]), orientation="horizontal")
            cb.set_label("lane direction relative to ego heading (deg)", fontsize=6.5)
            cb.ax.tick_params(labelsize=6)
            fig.subplots_adjust(left=0.07, right=0.99, top=0.99, bottom=0.005)
            f = args.out_dir / f"backbones_scene{r:02d}.png"
            fig.savefig(f, dpi=args.dpi)
            plt.close(fig)
            print(f)


if __name__ == "__main__":
    main()
