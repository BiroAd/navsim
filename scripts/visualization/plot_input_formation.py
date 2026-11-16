"""
Thesis figure: how vision_llm_lora forms its visual input from the surround cameras of one scene.

  (a) the six cameras the panorama layout uses, raw, with the region kept for the strips and the front-zoom crop;
      a top-down sketch of the rig shows each camera's full and kept horizontal field of view
  (b) the three images the vision tower sees (front strip, rear strip, front zoom), at true relative scale, with the
      32 px grid of merged visual tokens (16 px patches, 2x2 merge)
  (c) the visual token sequence in prompt order, each token drawn in the mean colour of its 32x32 px region

Images come from VisionLLMLoRAFeatureBuilder with the agent YAML config (the pixels the model is trained on); the
crop boxes drawn in (a) are checked to reproduce the builder's strips bit-exactly. Only images, no ego state/command.

Run on the login node or a CPU allocation with the navtrain image mounted, env active:
  source bashrc
  squashfuse $OPENSCENE_DATA_ROOT/images/navtrain_sensor_blobs.sqfs <mount>
  OMP_NUM_THREADS=4 python scripts/visualization/plot_input_formation.py --sensor-path <mount> \
      --tokens 72b2f93b877f5c7c e9bc97fbf494563b
  fusermount -u <mount>
Writes <out-dir>/input_formation_<token>.{pdf,png} and the three model images as PNGs.
"""

import argparse
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge  # noqa
from omegaconf import OmegaConf  # noqa: E402

from navsim.agents.vision_llm_lora.vision_llm_lora_agent import VisionLLMLoRAAgent  # noqa: E402
from navsim.agents.vision_llm_lora.vision_llm_lora_features import VisionLLMLoRAFeatureBuilder  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[2] / "navsim/planning/script/config/common"

# Mirrors VisionLLMLoRAFeatureBuilder._camera_images (camera_layout="panorama", thumbnail="front_zoom"); a mismatch
# fails the bit-exact check in main().
SIDE_CROP = 416  # columns cut from each side of L0 and R0
ZOOM_BOX = (768, 290, 1152, 674)  # x0, y0, x1, y1 on F0
FRONT_ROW = ("cam_l0", "cam_f0", "cam_r0")
REAR_ROW = ("cam_r1", "cam_b0", "cam_l1")
TOKEN_PX = 32  # merged token = 2x2 patches of 16 px

# Okabe-Ito based, readable on white and on photos.
COLORS = {
    "cam_l0": "#E69F00",
    "cam_f0": "#0072B2",
    "cam_r0": "#009E73",
    "cam_r1": "#CC79A7",
    "cam_b0": "#56B4E9",
    "cam_l1": "#8C6D31",
    "zoom": "#D7263D",
}
INK = "#222222"
MUTED = "#6B6B6B"
SPECIAL = "#3A3A3A"


def label(name: str) -> str:
    return name.upper().replace("CAM_", "CAM_")


# ── data ────────────────────────────────────────────────────────────────────────────────────────────────────────────


def row_geometry(cameras, names):
    """Per camera: (x0, x1, y0, y1) kept in raw pixels, and its column span in the 768 px strip."""
    spans = []
    for name in names:
        width = getattr(cameras, name).image.shape[1]
        x0, x1 = (SIDE_CROP, width - SIDE_CROP) if name in ("cam_l0", "cam_r0") else (0, width)
        spans.append((name, x0, x1))
    total = sum(x1 - x0 for _, x0, x1 in spans)
    raw_h = getattr(cameras, names[0]).image.shape[0]
    height = round(total / 6)
    y0 = (raw_h - height) // 2
    geometry, offset = {}, 0
    for name, x0, x1 in spans:
        geometry[name] = dict(box=(x0, x1, y0, y0 + height), strip=(offset / total, (offset + x1 - x0) / total))
        offset += x1 - x0
    return geometry


def reproduce_strip(cameras, names, geometry, tile):
    """The strip rebuilt from the boxes in `geometry` (checks that the drawn boxes are the builder's)."""
    parts = []
    for name in names:
        x0, x1, y0, y1 = geometry[name]["box"]
        parts.append(getattr(cameras, name).image[y0:y1, x0:x1])
    return cv2.resize(np.concatenate(parts, 1), (6 * tile, tile), interpolation=cv2.INTER_LANCZOS4)


def camera_fov(camera, x0, x1):
    """Yaw of the optical axis and of the kept column range, degrees in the lidar frame (x forward, y left).

    The raw images keep their lens distortion (k1 -0.36: strong barrel), so a column is undistorted before it is turned
    into a ray; the pinhole model alone gives 63.7 deg per camera instead of ~72 deg and draws 4-6 deg gaps between
    neighbours that the images do not have (SIFT matches put the L0 | F0 and F0 | R0 seams of the strips within 1.4 deg)."""
    rot, k = np.asarray(camera.sensor2lidar_rotation), np.asarray(camera.intrinsics)
    dist = np.asarray(camera.distortion, dtype=np.float64)
    width = camera.image.shape[1]

    def yaw(u):
        x, y = cv2.undistortPoints(np.array([[[u, k[1, 2]]]], dtype=np.float64), k, dist)[0, 0]
        d = rot @ np.array([x, y, 1.0])
        return np.degrees(np.arctan2(d[1], d[0]))

    axis = yaw(k[0, 2])

    def span(u0, u1):  # (low, high) as axis + offsets, so a range across +-180 deg stays one interval
        return tuple(axis + (yaw(u) - axis + 180.0) % 360.0 - 180.0 for u in (u1, u0))

    return dict(full=span(0, width), kept=span(x0, x1), axis=axis)


# ── drawing ─────────────────────────────────────────────────────────────────────────────────────────────────────────


class Canvas:
    """Axes placed in inches from the top-left corner of a fixed-size figure."""

    def __init__(self, width, height):
        self.w, self.h = width, height
        self.fig = plt.figure(figsize=(width, height))
        self.overlay = self.fig.add_axes([0, 0, 1, 1], zorder=-1)
        self.overlay.set_xlim(0, width)
        self.overlay.set_ylim(height, 0)
        self.overlay.axis("off")

    def axes(self, x, y, w, h, **kwargs):
        ax = self.fig.add_axes([x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h], **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def text(self, x, y, s, **kwargs):
        kwargs.setdefault("color", INK)
        return self.overlay.text(x, y, s, **kwargs)

    def panel_title(self, x, y, letter, title):
        self.text(x, y, f"({letter})", fontsize=9.5, fontweight="bold", va="baseline")
        self.text(x + 0.27, y, title, fontsize=9.5, va="baseline")


def frame(ax, color, lw=1.6):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


def show(ax, image, extent=None):
    h, w = image.shape[:2]
    extent = extent or (0, w, h, 0)
    ax.imshow(image, extent=extent, interpolation="lanczos")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])


def token_grid(ax, w, h):
    for x in range(TOKEN_PX, w, TOKEN_PX):
        ax.axvline(x, color="white", lw=0.35, alpha=0.55)
    for y in range(TOKEN_PX, h, TOKEN_PX):
        ax.axhline(y, color="white", lw=0.35, alpha=0.55)


def draw_raw(ax, name, image, box, zoom=False):
    """Raw frame (downscaled for the PDF) with the discarded area veiled and the kept box outlined."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (960, round(960 * h / w)), interpolation=cv2.INTER_AREA)
    show(ax, small, extent=(0, w, h, 0))
    x0, x1, y0, y1 = box
    veil = dict(facecolor="white", alpha=0.62, lw=0)
    for rect in ((0, 0, w, y0), (0, y1, w, h - y1), (0, y0, x0, y1 - y0), (x1, y0, w - x1, y1 - y0)):
        if rect[2] > 0 and rect[3] > 0:
            ax.add_patch(Rectangle(rect[:2], rect[2], rect[3], **veil))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=COLORS[name], lw=1.4))
    if zoom:
        zx0, zy0, zx1, zy1 = ZOOM_BOX
        ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec=COLORS["zoom"], lw=1.2, ls=(0, (3, 1.5))))
    frame(ax, "#BBBBBB", lw=0.5)


def draw_rig(ax, fovs):
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    for name, fov in fovs.items():
        # Matplotlib angles: 0 deg = +x (right). Plot forward (lidar +x) as up and left (lidar +y) as left.
        lo, hi = (a + 90.0 for a in fov["full"])
        ax.add_patch(Wedge((0, 0), 1.0, lo, hi, width=0.1, facecolor=COLORS[name], alpha=0.25, lw=0))
        lo, hi = (a + 90.0 for a in fov["kept"])
        ax.add_patch(Wedge((0, 0), 0.88, lo, hi, facecolor=COLORS[name], alpha=0.6, ec="white", lw=0.8))
        a = np.radians(fov["axis"] + 90.0)
        ax.text(1.12 * np.cos(a), 1.12 * np.sin(a), label(name)[4:], ha="center", va="center", fontsize=6.5,
                color=COLORS[name], fontweight="bold")
    # Ego vehicle, nose up.
    ax.add_patch(FancyBboxPatch((-0.09, -0.19), 0.18, 0.38, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="white", ec=INK, lw=0.8, zorder=5))
    ax.add_patch(Polygon([(-0.06, 0.09), (0.06, 0.09), (0, 0.17)], closed=True, fc=INK, lw=0, zorder=6))


def segment_bar(canvas, x, y, w, geometry, names, height=0.05):
    """Coloured bar under a strip marking which camera each column range came from, with names."""
    for name in names:
        a, b = geometry[name]["strip"]
        canvas.overlay.add_patch(Rectangle((x + a * w, y), (b - a) * w, height, fc=COLORS[name], lw=0))
        canvas.text(x + (a + b) / 2 * w, y + height + 0.035, label(name), fontsize=6.5, ha="center", va="top",
                    color=COLORS[name], fontweight="bold")


def token_patches(image):
    """The 32x32 px regions of the merged tokens, row-major (= merge-window order of the merged tokens)."""
    h, w = image.shape[:2]
    blocks = image.reshape(h // TOKEN_PX, TOKEN_PX, w // TOKEN_PX, TOKEN_PX, 3).transpose(0, 2, 1, 3, 4)
    return blocks.reshape(-1, TOKEN_PX, TOKEN_PX, 3)


def draw_sequence(canvas, x, y, size, groups, head=3, step=0.022, group_gap=0.16):
    """Per image: <|vision_start|>, the first `head` tokens, ..., the last token, <|vision_end|>; tokens drawn as
    their 32x32 px region. Returns the right edge."""
    ax = canvas.overlay
    cursor = x
    for title, color, patches in groups:
        start = cursor
        n = len(patches)
        slots = ["vs"] + list(range(head)) + ["..."] + [n - 1] + ["ve"]
        for slot in slots:
            if slot in ("vs", "ve"):
                ax.add_patch(FancyBboxPatch((cursor, y), size, size, boxstyle="round,pad=0,rounding_size=0.025",
                                            fc=SPECIAL, ec="none"))
            elif slot == "...":
                canvas.text(cursor + size / 2, y + size / 2, "···", ha="center", va="center", fontsize=9,
                            color=color)
            else:
                ax.imshow(patches[slot], extent=(cursor, cursor + size, y + size, y), interpolation="lanczos",
                          aspect="auto", zorder=2)
                ax.add_patch(Rectangle((cursor, y), size, size, fill=False, ec=color, lw=0.9, zorder=3))
                canvas.text(cursor + size / 2, y + size + 0.035, str(slot + 1), ha="center", va="top", fontsize=5.8,
                            color=MUTED)
            cursor += size + step
        cursor -= step
        yb = y - 0.06
        ax.plot([start, start, cursor, cursor], [yb + 0.03, yb, yb, yb + 0.03], color=color, lw=0.8,
                solid_capstyle="butt")
        canvas.text((start + cursor) / 2, yb - 0.035, title, ha="center", va="bottom", fontsize=7, color=color)
        cursor += group_gap
    return cursor - group_gap


def make_figure(token, cameras, tiles, out_dir: Path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "image.composite_image": False,
        "savefig.dpi": 300,
    })
    tile = tiles[0].shape[0]
    front = np.concatenate(tiles[:6], 1)
    rear = np.concatenate(tiles[6:12], 1)
    zoom = tiles[12]
    geo = {**row_geometry(cameras, FRONT_ROW), **row_geometry(cameras, REAR_ROW)}
    fovs = {n: camera_fov(getattr(cameras, n), *geo[n]["box"][:2]) for n in FRONT_ROW + REAR_ROW}
    n_front, n_rear, n_zoom = (img.shape[0] // TOKEN_PX * img.shape[1] // TOKEN_PX for img in (front, rear, zoom))

    # ── layout (inches from the top) ──
    W, L, R = 6.3, 0.02, 6.28
    gx = 1.42  # raw-camera grid
    gap = 0.06
    iw = (R - gx - 2 * gap) / 3
    ih = iw * 9 / 16
    a_title = 0.17
    rows_y = (0.42, 0.42 + ih + 0.2)
    arrow_y = rows_y[1] + ih + 0.1
    b_title = arrow_y + 0.55
    sx, gz = 0.42, 0.16  # strips / zoom
    sw = (R - sx - gz) * 6 / 7
    sh = sw / 6
    fy = b_title + 0.14
    ry = fy + sh + 0.29
    c_title = ry + sh + 0.47
    seq_y, seq_h = c_title + 0.36, 0.2
    height = seq_y + seq_h + 0.3

    c = Canvas(W, height)

    # (a) raw cameras + rig
    c.panel_title(L, a_title, "a", "Surround-view cameras (1920 × 1080 px each) and the regions kept")
    for r, names in enumerate((FRONT_ROW, REAR_ROW)):
        for i, name in enumerate(names):
            x = gx + i * (iw + gap)
            ax = c.axes(x, rows_y[r], iw, ih)
            draw_raw(ax, name, getattr(cameras, name).image, geo[name]["box"], zoom=name == "cam_f0")
            c.text(x, rows_y[r] - 0.04, label(name), fontsize=7, fontweight="bold", color=COLORS[name],
                   va="baseline")
    for r, text in enumerate(("front row", "rear row")):
        c.text(R, rows_y[r] - 0.04, text, fontsize=7, color=MUTED, ha="right", va="baseline", style="italic")
    rig = 1.34
    rig_y = (rows_y[0] + rows_y[1] + ih) / 2 - rig / 2
    draw_rig(c.axes(L, rig_y, rig, rig), fovs)
    c.text(L + rig / 2, rig_y + rig + 0.07, "camera rig, top view", fontsize=6.5, color=MUTED, ha="center",
           va="top", style="italic")

    # arrow + operations
    ax0 = gx + iw / 2
    c.overlay.add_patch(FancyArrowPatch((ax0, arrow_y), (ax0, arrow_y + 0.3), arrowstyle="-|>", mutation_scale=9,
                                        color=MUTED, lw=1.0))
    c.text(ax0 + 0.1, arrow_y + 0.15,
           f"each row: concatenate → centre-crop to 6 : 1 → resize to {6 * tile} × {tile}\n"
           f"front zoom: 384 × 384 px crop of CAM_F0 around the horizon straight ahead → {tile} × {tile}",
           fontsize=6.8, color=MUTED, va="center", linespacing=1.3)

    # (b) model images at true relative scale
    c.panel_title(L, b_title, "b", f"Vision-encoder input: two {6 * tile} × {tile} strips and a {tile} × {tile} "
                  "front zoom (true relative scale)")
    for img, y, title, names in ((front, fy, "front", FRONT_ROW), (rear, ry, "rear", REAR_ROW)):
        ax = c.axes(sx, y, sw, sh)
        show(ax, img)
        token_grid(ax, img.shape[1], img.shape[0])
        frame(ax, INK, lw=0.6)
        c.text(sx - 0.07, y + sh / 2, f"{title} strip\n{img.shape[1]} × {img.shape[0]}", rotation=90,
               fontsize=6.8, ha="center", va="center", linespacing=1.1)
        segment_bar(c, sx, y + sh + 0.02, sw, geo, names)
    zx = sx + sw + gz
    zax = c.axes(zx, fy, sh, sh)
    show(zax, zoom)
    token_grid(zax, tile, tile)
    frame(zax, COLORS["zoom"], lw=1.4)
    c.text(zx + sh / 2, fy + sh + 0.06, f"front zoom\n{tile} × {tile}", fontsize=6.8, ha="center", va="top",
           color=COLORS["zoom"], linespacing=1.1, fontweight="bold")
    c.text(zx - 0.04, ry + 0.02, f"grid cell = {TOKEN_PX} × {TOKEN_PX} px\n= 1 visual token\n(16 px patches,\n"
           "2 × 2 merge)", fontsize=6.4, color=MUTED, va="top", linespacing=1.15)

    # (c) token sequence
    c.panel_title(L, c_title, "c", f"Visual token sequence in the prompt ({n_front + n_rear + n_zoom} image tokens)")
    end = draw_sequence(
        c, sx, seq_y, seq_h,
        [
            (f"front strip · 4 × 24 = {n_front} tokens", COLORS["cam_f0"], token_patches(front)),
            (f"rear strip · 4 × 24 = {n_rear} tokens", "#2C7FB8", token_patches(rear)),
            (f"front zoom · 4 × 4 = {n_zoom} tokens", COLORS["zoom"], token_patches(zoom)),
        ],
    )
    mid = seq_y + seq_h / 2
    c.overlay.add_patch(FancyArrowPatch((end + 0.05, mid), (end + 0.27, mid), arrowstyle="-|>", mutation_scale=9,
                                        color=MUTED, lw=1.0))
    bx = end + 0.3
    c.overlay.add_patch(FancyBboxPatch((bx, seq_y - 0.04), R - bx, seq_h + 0.08,
                                       boxstyle="round,pad=0,rounding_size=0.05", fc="#F2F2F2", ec=INK, lw=0.7))
    c.text((bx + R) / 2, mid, "Qwen3.5\nLLM", fontsize=7.2, ha="center", va="center", linespacing=1.1)
    ly = seq_y + seq_h + 0.2
    c.overlay.add_patch(FancyBboxPatch((sx, ly - 0.04), 0.08, 0.08, boxstyle="round,pad=0,rounding_size=0.015",
                                       fc=SPECIAL, ec="none"))
    c.text(sx + 0.12, ly, "<|vision_start|> / <|vision_end|>        image tokens: their 32 × 32 px region, numbered "
           "in row-major order", fontsize=6.4, color=MUTED, va="center")
    c.overlay.set_xlim(0, W)
    c.overlay.set_ylim(height, 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"input_formation_{token}"
    c.fig.savefig(f"{stem}.pdf")
    c.fig.savefig(f"{stem}.png")
    plt.close(c.fig)
    for name, img in (("front_strip", front), ("rear_strip", rear), ("front_zoom", zoom)):
        cv2.imwrite(str(out_dir / f"{token}_{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"saved {stem}.pdf / .png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", nargs="+", required=True)
    parser.add_argument("--sensor-path", type=Path, default=None,
                        help="default: $OPENSCENE_DATA_ROOT/sensor_blobs/trainval")
    parser.add_argument("--logs", type=Path, default=None, help="default: $OPENSCENE_DATA_ROOT/navsim_logs/trainval")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: $NAVSIM_EXP_ROOT/thesis_figures")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="keep the loaded cameras and tiles here (one small pickle per token), so re-plotting "
                        "skips the scene loading")
    args = parser.parse_args()
    out_dir = args.out_dir or Path(os.environ.get("NAVSIM_EXP_ROOT", ".")) / "thesis_figures"
    cached = {}
    if args.cache_dir:
        for token in args.tokens:
            path = args.cache_dir / f"{token}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    cached[token] = pickle.load(f)
    for token, (cameras, tiles) in cached.items():
        make_figure(token, cameras, tiles, out_dir)
    tokens = [t for t in args.tokens if t not in cached]
    if not tokens:
        return

    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    agent_cfg = OmegaConf.load(CONFIG_DIR / "agent/vision_llm_lora_agent.yaml")
    config = instantiate(agent_cfg.config)
    assert config.camera_layout == "panorama" and config.thumbnail == "front_zoom", "figure drawn for these settings"
    sensor_config = VisionLLMLoRAAgent.get_sensor_config(SimpleNamespace(_config=config))
    scene_filter = instantiate(
        OmegaConf.load(CONFIG_DIR / "train_test_split/scene_filter/navtrain.yaml"),
        tokens=tokens,
        max_scenes=len(tokens),
    )
    scene_loader = SceneLoader(
        data_path=args.logs or data_root / "navsim_logs/trainval",
        original_sensor_path=args.sensor_path or data_root / "sensor_blobs/trainval",
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )
    builder = VisionLLMLoRAFeatureBuilder(config)
    missing = set(tokens) - set(scene_loader.tokens)
    if missing:
        print(f"not found in navtrain: {sorted(missing)}")

    for token in [t for t in tokens if t in scene_loader.tokens]:
        agent_input = scene_loader.get_agent_input_from_token(token)
        cameras = agent_input.cameras[-1]
        tiles = [np.asarray(image) for image in builder._camera_images(agent_input, frame=-1)]
        tile = tiles[0].shape[0]
        for names, strip in ((FRONT_ROW, np.concatenate(tiles[:6], 1)), (REAR_ROW, np.concatenate(tiles[6:12], 1))):
            rebuilt = reproduce_strip(cameras, names, row_geometry(cameras, names), tile)
            assert np.array_equal(rebuilt, strip), f"drawn crop boxes do not match the builder ({names})"
        x0, y0, x1, y1 = ZOOM_BOX
        zoom = cv2.resize(cameras.cam_f0.image[y0:y1, x0:x1], (tile, tile), interpolation=cv2.INTER_LANCZOS4)
        assert np.array_equal(zoom, tiles[12]), "drawn zoom box does not match the builder"
        if args.cache_dir:
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            for camera in vars(cameras).values():
                camera.image = None if camera.image is None else np.ascontiguousarray(camera.image)
            with open(args.cache_dir / f"{token}.pkl", "wb") as f:
                pickle.dump((cameras, tiles), f)
        make_figure(token, cameras, tiles, out_dir)


if __name__ == "__main__":
    main()
