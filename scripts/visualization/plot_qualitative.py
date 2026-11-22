"""Qualitative figures of the vision_llm_lora planner on navtest: front camera with the trajectories projected onto the
ground + bird's-eye view (map, agents, ego, candidates, trajectories), one PNG per scene, plus an index.md.

Figures (--figure):
  rules        one scene, several selection rules of one model (default: the RL-3 model rl3_b0.005_ep01 with its own
               scorer p3_aux_policy_ep03): human, argmax, heads (score_heads), shortlist k 256, mixture lambda 10,
               mixture + EC rule n 64 (sequential). BEV also shows the policy's top candidates (end points coloured by
               probability) and the scorer's top-256 shortlist (grey end points).
  consecutive  the plans at t - 0.5 s and at t for heads selection vs mixture + EC rule (EC compares the two).
Scenes are chosen by stated rules from the per-scene official scores (no hand picking); rules figure categories:
  shortlist_rescues, heads_choice_unsafe, all_rules_safe, beats_human, perfect_turns, best_rule_fails (definitions in
  CATEGORIES below and in rules/index.md)
  consecutive: heads EC = 0 and EC-rule EC = 1 at t, both without collision / off-road (top by how far the heads plan's
               end point jumps between t - 0.5 s and t minus the EC rule's jump)
Needs the test sensor image mounted (squashfuse $OPENSCENE_DATA_ROOT/images/test_sensor_blobs.sqfs <dir>) and the
environment of bashrc. CPU only; reads predictions/<name>/navtest and pdm_navtest/<name>_<rule>_non_reactive_split.
  python scripts/visualization/plot_qualitative.py --figure rules --sensor-path <mount> --out-dir $NAVSIM_EXP_ROOT/qualitative
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib
import matplotlib.patheffects

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from navsim.agents.vision_llm_lora.vision_llm_lora_agent import select_trajectory  # noqa: E402
from navsim.agents.vision_llm_lora.vision_llm_lora_loss import aggregate_pdm_scores  # noqa: E402
from navsim.agents.vision_llm_lora.vision_llm_lora_precomputed_agent import load_predictions  # noqa: E402
from navsim.common.dataclasses import SensorConfig  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402
from navsim.visualization.bev import add_configured_bev_on_ax  # noqa: E402
from navsim.visualization.camera import _transform_points_to_image  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType  # noqa: E402
from navsim.visualization.config import AGENT_CONFIG  # noqa: E402

PRINT_W = 6.0  # in: thesis text width 150 mm; fonts 7-8 pt at that size
plt.rcParams.update({"font.size": 8, "axes.titlesize": 7.5, "legend.fontsize": 6.8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7})

# NAVSIM draws every agent box with a black 1.0 pt outline and heading line, which hides the fill of small boxes at
# print size; thinner outlines for the agents in these figures only (the ego box keeps NAVSIM's style; the upstream
# config file is not changed)
for _t, _c in AGENT_CONFIG.items():
    if _t != TrackedObjectType.EGO:
        AGENT_CONFIG[_t] = {**_c, "line_width": 0.4}

E = Path(os.environ["NAVSIM_EXP_ROOT"])
D = Path(os.environ["OPENSCENE_DATA_ROOT"])
REPO = Path(__file__).resolve().parents[2]
SUB = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance",
       "ego_progress", "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
AB = dict(zip(SUB, ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]))
COMMANDS = {0: "left", 1: "straight", 2: "right", 3: "unknown"}
STYLE = {  # label -> (colour, line style, width, z-order)
    "human": ("black", "--", 1.8, 26),
    "argmax": ("#1f77b4", "-", 2.0, 27),
    "heads": ("#ff7f0e", "-", 2.0, 27),
    "shortlist k256": ("#2ca02c", "-", 2.0, 28),
    "mixture λ10": ("#9467bd", "-", 2.0, 28),
    "mixture + EC n64": ("#d62728", "-", 2.6, 29),
    "argmax + EC n64": ("#17becf", "-", 2.2, 28),
    "shortlist + EC n64": ("#006d2c", "-", 2.2, 28),
}
RULE5 = ["argmax", "heads", "shortlist k256", "mixture λ10", "mixture + EC n64", "shortlist + EC n64"]  # rules panels


# ----------------------------------------------------------------------------------------------- data
def latest_csv(folder: Path):
    fs = [f for f in glob.glob(f"{folder}/*/*.csv") if re.search(r"/\d{4}(\.\d{2}){5}\.csv$", f)]
    return max(fs, key=os.path.getmtime) if fs else None


def per_scene(name: str, suffix: str = "_non_reactive_split") -> pd.DataFrame:
    f = latest_csv(E / "pdm_navtest" / f"{name}{suffix}")
    if f is None:
        raise FileNotFoundError(name)
    df = pd.read_csv(f)
    df = df[(df.token.astype(str) != "average_all_frames") & (df.valid.astype(str) == "True")]
    return df.set_index("token")[SUB + ["score"]]


def bf16(pred, key, row):
    a = torch.from_numpy(np.array(pred["arrays"][key][row : row + 1]))
    return a.view(torch.bfloat16).float() if pred["meta"]["logits_dtype"] == "bfloat16" else a.float()


class Rules:
    """Trajectories of every rule for one token, recomputed from the stored logits (select_trajectory)."""

    def __init__(self, model: str, scorer: str, ec_folder: str):
        self.model, self.scorer = load_predictions(E / "predictions" / model / "navtest"), load_predictions(
            E / "predictions" / scorer / "navtest")
        self.ec = load_predictions(E / "predictions" / ec_folder / "navtest")
        self.ec_more = {k: load_predictions(E / "predictions" / f"{model}_own_ecseq_{b}_n64" / "navtest")
                        for k, b in [("argmax + EC n64", "argmax"), ("shortlist + EC n64", "short256")]}
        self.codebook = self.model["codebook"].float()
        # drawing: the original 40-step (10 Hz) codebook; its poses at 0.5, 1.0, ..., 4.0 s are the 8-step entries
        cb_dir = REPO / "navsim/agents/vision_llm_lora/codebook"
        self.cb8, self.cb40 = np.load(cb_dir / "16384_8step.npy"), np.load(cb_dir / "16384.npy")
        assert np.array_equal(self.cb40[:, 4::5], self.cb8), "16384.npy and 16384_8step.npy do not match"
        self._key = {e.tobytes(): k for k, e in enumerate(self.cb8)}

    def fine(self, traj):
        """The 40-step codebook entry of an 8-step codebook trajectory (unchanged if it is not one)."""
        t = np.ascontiguousarray(traj, dtype=np.float32)
        k = self._key.get(t.tobytes())
        if k is None:
            d = np.abs(self.cb8 - t).reshape(len(self.cb8), -1).max(1)
            k = int(d.argmin()) if d.min() < 1e-4 else None
        return traj if k is None else self.cb40[k].astype(np.float64)

    def __call__(self, token: str, top_n: int = 200):
        r, s = self.model["row"][token], self.scorer["row"][token]
        pred = {"trajectory": torch.from_numpy(np.array(self.model["arrays"]["trajectory"][r : r + 1])),
                "trajectory_logits": bf16(self.model, "trajectory_logits", r),
                "pdm_score_logits": bf16(self.scorer, "pdm_score_logits", s)}
        out = {"argmax": pred["trajectory"][0].numpy(),
               "heads": select_trajectory(pred, "score_heads", self.codebook)[0].numpy(),
               "shortlist k256": select_trajectory(pred, "shortlist", self.codebook, shortlist_size=256)[0].numpy(),
               "mixture λ10": select_trajectory(pred, "mixture", self.codebook, mixture_lambda=10.0)[0].numpy(),
               "mixture + EC n64": np.array(self.ec["arrays"]["trajectory"][self.ec["row"][token]])}
        for k, pr in self.ec_more.items():
            out[k] = np.array(pr["arrays"]["trajectory"][pr["row"][token]])
        out = {k: self.fine(v) for k, v in out.items()}
        p = pred["trajectory_logits"].log_softmax(-1).exp()[0]
        top = p.topk(top_n)
        total = aggregate_pdm_scores(pred["pdm_score_logits"].sigmoid())[0]
        short = total.topk(256).indices
        cand = {"policy_xy": self.codebook[top.indices, -1, :2].numpy(), "policy_p": top.values.numpy(),
                "short_xy": self.codebook[short, -1, :2].numpy(), "p_argmax": float(top.values[0])}
        return out, cand


def scene_loader(tokens, sensor_path: Path, frames=(3,)):
    cfg = OmegaConf.load(REPO / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml")
    sf = instantiate(cfg)
    sf.tokens = list(tokens)
    cams = dict(cam_l1=False, cam_l2=False, cam_r1=False, cam_r2=False, cam_b0=False)
    return SceneLoader(D / "navsim_logs/test", sensor_path, sf,
                       sensor_config=SensorConfig(cam_f0=list(frames), cam_l0=list(frames), cam_r0=list(frames),
                                                  lidar_pc=False, **cams))


def previous_token(loader: SceneLoader, token: str):
    frames = loader.scene_frames_dicts[token]
    t = frames[2]["token"]  # the frame 0.5 s earlier (frame_interval 1); a navtest scene itself if it is in the split
    return t


# ----------------------------------------------------------------------------------------------- drawing
def ground_z(frame) -> float:
    """Ground height in the ego/lidar frame from the annotated boxes (box bottom), as NAVSIM projects the boxes."""
    b = frame.annotations.boxes
    if len(b) == 0:
        return 0.0
    near = np.hypot(b[:, 0], b[:, 1]) < 40
    sel = b[near] if near.sum() >= 2 else b
    return float(np.median(sel[:, 2] - sel[:, 5] / 2))


def densify(traj, n=8):
    """Smooth drawing curve through the ego origin and the poses: cubic Hermite segments whose tangents follow each
    pose's heading (length = segment length), so the drawn path is tangent to the planned headings instead of a
    polyline with corners at the 0.5 s waypoints. Drawing only; scoring uses the poses. Without a heading column:
    linear segments."""
    traj = np.asarray(traj)
    pts = np.concatenate([[[0.0, 0.0]], traj[:, :2]])
    if traj.shape[1] < 3:
        return np.concatenate([[pts[0]]] + [a + np.linspace(0, 1, n + 1)[1:, None] * (b - a)
                                            for a, b in zip(pts[:-1], pts[1:])])
    yaw = np.concatenate([[0.0], traj[:, 2]])
    s = np.linspace(0, 1, n + 1)[1:, None]
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    out = [pts[0]]
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        L = np.linalg.norm(b - a)
        if L < 0.05:  # standing still: no tangent
            out.append(np.repeat(b[None], n, 0))
            continue
        ta, tb = L * np.array([np.cos(yaw[i]), np.sin(yaw[i])]), L * np.array([np.cos(yaw[i + 1]), np.sin(yaw[i + 1])])
        out.append(h00 * a + h10 * ta + h01 * b + h11 * tb)
    return np.vstack([np.atleast_2d(o) for o in out])


def project(camera, xy, z):
    pts = np.concatenate([xy, np.full((len(xy), 1), z)], 1)
    r, t = camera.sensor2lidar_rotation, camera.sensor2lidar_translation
    cam = (pts - t) @ r  # lidar -> camera: R^T (p - t), as _transform_annotations_to_camera
    keep = cam[:, 2] > 1.0
    uv, inside = _transform_points_to_image(cam, camera.intrinsics, camera.image.shape[:2])
    return uv, keep


def draw_camera(ax, frame, trajs, title=None, cam_name="cam_f0"):
    cam = getattr(frame.cameras, cam_name)
    img = cam.image
    ax.imshow(img)
    z = ground_z(frame)
    for label, traj in trajs.items():
        c, ls, lw, zo = STYLE[label]
        uv, keep = project(cam, densify(traj), z)
        uv = uv[keep]
        ax.plot(uv[:, 0], uv[:, 1], color=c, linestyle=ls, linewidth=lw * 1.0, zorder=zo, solid_capstyle="round",
                clip_on=True)
    h, w = img.shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h * 0.99, h * 0.30)  # crop the sky
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def bev_extent(trajs, extra=None, min_span=32.0, margin=8.0):
    """Square view (forward up) around the ego and every drawn trajectory."""
    pts = np.concatenate([np.zeros((1, 2))] + [t[:, :2] for t in trajs.values()] +
                         [t[:, :2] for t, _, _ in (extra or {}).values()])
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    span = max(min_span, *(hi - lo))
    c = (lo + hi) / 2
    return (c[1] - span / 2, c[1] + span / 2, c[0] - span / 2, c[0] + span / 2)  # y_min, y_max, x_min, x_max


def draw_bev(ax, scene, trajs, cand=None, extra=None, lim=None):
    lim = lim or bev_extent(trajs, extra)
    frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    add_configured_bev_on_ax(ax, scene.map_api, frame)
    if cand is not None and len(cand["short_xy"]):
        ax.scatter(cand["short_xy"][:, 1], cand["short_xy"][:, 0], s=4, c="0.45", marker="x", linewidths=0.5,
                   alpha=0.45, zorder=20, label="heads top-256 (end points)")
    if cand is not None and len(cand["policy_xy"]):
        o = np.argsort(cand["policy_p"])
        ax.scatter(cand["policy_xy"][o, 1], cand["policy_xy"][o, 0], s=4, c=np.log10(cand["policy_p"][o]),
                   cmap="viridis", vmin=np.log10(cand["policy_p"]).max() - 3, zorder=21, alpha=0.55, linewidths=0,
                   label="policy top-200 (colour: log p)")
    for label, traj in trajs.items():
        c, ls, lw, zo = STYLE[label]
        pts = densify(traj, 10)
        ax.plot(pts[:, 1], pts[:, 0], color=c, linestyle=ls, linewidth=lw * 0.75, zorder=zo, label=label,
                solid_capstyle="round")
        ax.plot(pts[-1, 1], pts[-1, 0], marker="*", markersize=7.5, color=c, markeredgecolor="white",
                markeredgewidth=0.5, zorder=zo + 10)  # end point: a stop shows as a star at the ego
    for label, (traj, c, ls) in (extra or {}).items():
        pts = densify(traj, 10)
        ax.plot(pts[:, 1], pts[:, 0], color=c, linestyle=ls, linewidth=1.3, zorder=27, label=label)
        ax.plot(pts[-1, 1], pts[-1, 0], marker="*", markersize=7, color="white", markeredgecolor=c, zorder=37)
    ax.set_aspect("equal")
    ax.set_xlim(lim[1], lim[0])  # left = +y
    ax.set_ylim(lim[2], lim[3])
    ax.set_xticks([])
    ax.set_yticks([])


AGENT_COLORS = {"vehicle": "#699CDB", "pedestrian": "#b07aa1", "bicycle": "#E38C47"}  # NAVSIM BEV colours


def darker(col, f=0.75):
    """the colour scaled towards black (future outlines: visible on the grey road in print)"""
    return tuple(np.array(matplotlib.colors.to_rgb(col)) * f)


def poses8(traj):
    """the 8 scored poses (0.5 ... 4.0 s) of a drawn plan: the 40-step codebook entry at indices 4::5"""
    t = np.asarray(traj, dtype=float)
    return t[4::5] if len(t) == 40 else t


class CollisionSnapshot:
    """NAVSIM's own at-fault collision for a plan: pdm_score_from_interpolated_trajectory (navsim/evaluate/pdm_score.py)
    replicated up to the first scorer.score_proposals call (the human-filter call after it would overwrite the scorer's
    state), with the official scoring config (default_run_pdm_score; navtest: non-reactive traffic). Returns the first
    at-fault collision time, the simulated ego polygon and the polygons of the agents it intersects at that time, in
    the ego frame of the current frame, or None if the plan does not collide. NAVSIM code is used unchanged."""

    def __init__(self, metric_cache_path, traffic="non_reactive"):
        from hydra import compose, initialize_config_dir
        from navsim.common.dataloader import MetricCacheLoader
        with initialize_config_dir(config_dir=str(REPO / "navsim/planning/script/config/pdm_scoring"), version_base=None):
            cfg = compose(config_name="default_run_pdm_score", overrides=[f"traffic_agents={traffic}"])
        self.simulator, self.scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
        self.policy = instantiate(cfg.traffic_agents_policy[traffic], self.simulator.proposal_sampling)
        self.loader = MetricCacheLoader(Path(metric_cache_path))

    def __call__(self, token, poses8):
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        mc = self.loader.get_from_token(token)
        sampling = self.simulator.proposal_sampling
        pred = transform_trajectory(Trajectory(np.asarray(poses8, dtype=np.float32),
                                               TrajectorySampling(time_horizon=4, interval_length=0.5)), mc.ego_state)
        states = np.concatenate([get_trajectory_as_array(mc.trajectory, sampling, mc.ego_state.time_point)[None],
                                 get_trajectory_as_array(pred, sampling, mc.ego_state.time_point)[None]], axis=0)
        sim = self.simulator.simulate_proposals(states, mc.ego_state)
        tracks = self.policy.simulate_environment(sim[1], mc)
        res = self.scorer.score_proposals(sim, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map,
                                          mc.map_parameters, tracks, mc.past_human_trajectory)[1]
        sc = self.scorer
        nc = float(np.asarray(res.no_at_fault_collisions).reshape(-1)[0])
        if not np.isfinite(sc._collision_time_idcs[1]):
            return {"nc": nc, "t": None}
        ti = int(sc._collision_time_idcs[1])
        ego_poly = sc._ego_polygons[1, ti]
        hit = sc._observation[ti].query(ego_poly, predicate="intersects")
        toks = [sc._observation[ti].tokens[g] for g in np.atleast_1d(hit)]
        toks = [t for t in toks if sc._observation.red_light_token not in t]
        o = mc.ego_state.rear_axle  # the ego frame of the figures (rear axle at the current frame)
        c, s_ = np.cos(o.heading), np.sin(o.heading)

        def local(poly):
            xy = np.asarray(poly.exterior.coords) - [o.x, o.y]
            return np.stack([c * xy[:, 0] + s_ * xy[:, 1], -s_ * xy[:, 0] + c * xy[:, 1]], 1)

        return {"nc": nc, "t": ti * sampling.interval_length, "ego": local(ego_poly),
                "agents": [local(sc._observation[ti][t]) for t in toks]}


def draw_collision(ax, snap, colour):
    """the ego (plan's colour) and the agent(s) it hits (dark), at NAVSIM's first at-fault collision, with its time"""
    for a in snap["agents"]:
        ax.fill(a[:, 1], a[:, 0], facecolor="none", edgecolor="#1a1a1a", linewidth=1.0, zorder=44)
    e = snap["ego"]
    ax.plot(e[:, 1], e[:, 0], color=colour, linewidth=1.1, zorder=45)
    # time label beside the ego outline, on the side away from the agent hit (lateral = data y, the plot's x axis)
    ey = e[:-1, 1].mean()
    side = np.sign(ey - np.mean([a[:-1, 1].mean() for a in snap["agents"]])) if snap["agents"] else 1.0
    side = side or 1.0
    y_lab = (e[:, 1].max() if side > 0 else e[:, 1].min()) + 0.4 * side
    right = ax.transData.transform((y_lab, 0))[0] > ax.transData.transform((ey, 0))[0]  # axis may be inverted
    txt = ax.text(y_lab, e[:-1, 0].mean(), f"{snap['t']:.1f} s", fontsize=6, color=colour, va="center",
                  ha="left" if right else "right", zorder=46)
    txt.set_path_effects([matplotlib.patheffects.withStroke(linewidth=1.5, foreground="white")])


def draw_future_agents(ax, scene, times=(1.0, 2.0, 3.0, 4.0)):
    """Moving agents at +1 ... +4 s (fading dashed outlines) in the current ego frame: makes collisions with agents that
    are not yet next to the ego at t visible."""
    h = scene.scene_metadata.num_history_frames - 1
    pc = np.array(scene.frames[h].ego_status.ego_pose, dtype=float)
    for k, t in enumerate(times):
        i = h + int(round(t / 0.5))
        if i >= len(scene.frames):
            break
        f = scene.frames[i]
        pf = np.array(f.ego_status.ego_pose, dtype=float)
        for b, n in zip(f.annotations.boxes, f.annotations.names):
            col = AGENT_COLORS.get(n)
            if col is None:
                continue
            x, y, hd = to_frame(np.array([[b[0], b[1], b[6]]], dtype=float), pf, pc)[0]
            L, W = b[3], b[4]
            cx = np.array([L, L, -L, -L, L]) / 2
            cy = np.array([W, -W, -W, W, W]) / 2
            px, py = x + np.cos(hd) * cx - np.sin(hd) * cy, y + np.sin(hd) * cx + np.cos(hd) * cy
            ax.plot(py, px, color=darker(col), linewidth=1.1 - 0.15 * k, linestyle=(0, (2, 1.4)),
                    alpha=0.95 - 0.17 * k, zorder=15)  # +1 s darkest and thickest, fading to +4 s


def map_legend_handles(future=True, footprint=False):
    """Legend entries for the BEV map and agent colours (navsim/visualization/config.py)."""
    h = [Patch(facecolor="#D3D3D3", label="road"), Patch(facecolor="#d4d19e", label="walkway"),
         Patch(facecolor="#b9d3b4", label="car park"), Patch(facecolor="#b07aa1", alpha=0.3, label="crosswalk"),
         plt.Line2D([], [], color="#666666", linestyle="--", linewidth=0.8, label="lane centreline"),
         Patch(facecolor="#699CDB", edgecolor="black", linewidth=0.4, label="vehicle"),
         Patch(facecolor="#b07aa1", edgecolor="black", linewidth=0.4, label="pedestrian"),
         Patch(facecolor="#E38C47", edgecolor="black", linewidth=0.4, label="bicycle"),
         Patch(facecolor="#edc948", edgecolor="black", linewidth=0.4, label="static object"),
         Patch(facecolor="#DE7061", edgecolor="black", linewidth=0.5, label="ego vehicle")]
    if future:
        h.append(plt.Line2D([], [], color=darker("#699CDB"), linestyle=(0, (2, 1.4)), linewidth=1.0,
                            label="agents at +1 … +4 s (fading)"))
    if footprint:
        h.append(Patch(facecolor="none", edgecolor="0.3", linewidth=1.0, label="collision: ego, agent hit"))
    h.append(plt.Line2D([], [], color="0.3", marker="*", linestyle="", markersize=7, label="end point"))
    return h


def fmt_scores(row):
    """EPDMS with every sub-score below 1 and its value, e.g. 0.75 (HC 0, EC 0) or 0.98 (EP 0.95)."""
    bad = [f"{AB[k]} {row[k]:.2f}".replace(" 0.00", " 0") for k in SUB if row[k] < 0.995]
    return f"{row['score']:.2f}" + (f" ({', '.join(bad)})" if bad else "")


def scene_info(scene):
    f = scene.frames[scene.scene_metadata.num_history_frames - 1]
    cmd = COMMANDS.get(int(np.argmax(f.ego_status.driving_command)), "?")
    v = float(np.hypot(*f.ego_status.ego_velocity[:2]))
    return f"{scene.scene_metadata.log_name} | command: {cmd} | speed {v:.1f} m/s"


# ----------------------------------------------------------------------------------------------- figures
SHORT = {"argmax": "Policy (argmax)", "heads": "Heads only", "shortlist k256": "Shortlist (k = 256)",
         "mixture λ10": "Mixture (λ = 10)", "mixture + EC n64": "Mixture + consistency",
         "argmax + EC n64": "Policy + consistency", "shortlist + EC n64": "Shortlist + consistency"}
RULE_TITLES = {  # panel titles: what each rule does
    "argmax": "Policy only (argmax)\nmost likely trajectory",
    "heads": "Sub-score heads only\nbest predicted score",
    "shortlist k256": "Shortlist (k = 256)\npolicy's pick among the heads' 256 best",
    "mixture λ10": "Mixture (λ = 10)\nmax log p + 10 · log predicted score",
    "mixture + EC n64": "Mixture + consistency rule\nkeeps the plan of t − 0.5 s if possible",
    "argmax + EC n64": "Policy + consistency rule\namong the policy's 64 most likely",
    "shortlist + EC n64": "Shortlist + consistency rule\namong the shortlist's 64 best for the policy",
}
CATEGORIES = {  # folder -> how the scenes were chosen (also in index.md)
    "shortlist_rescues": "policy argmax scored < 0.3, shortlist >= 0.8 (top by shortlist - argmax)",
    "heads_choice_unsafe": "heads-only choice >= 0.3 below an argmax scored >= 0.8 (top by argmax - heads)",
    "all_rules_safe": "every rule > 0.5 and the rules choose different trajectories (random)",
    "beats_human": "best rule (mixture + EC) >= 0.95 while the logged human <= 0.8 (top by model - human)",
    "perfect_turns": "best rule scored >= 0.99 in a turn (its plan ends > 35 deg from the current heading; random)",
    "best_rule_fails": "best rule (mixture + EC) scored 0 (random)",
}  # every category: at most one scene per log


def first_per_log(tokens, log_of, n):
    """The first n tokens in order, at most one per log (scenes of one log are consecutive frames of one drive)."""
    out, seen = [], set()
    for t in tokens:
        if log_of(t) not in seen:
            out.append(t)
            seen.add(log_of(t))
        if len(out) == n:
            break
    return out


def select_rules(scores, human, rules, rng, n_top, n_rand, log_of):
    keys = ["argmax", "heads", "shortlist k256", "mixture λ10", "mixture + EC n64"]
    idx = scores[keys[0]].index
    for k in keys[1:]:
        idx = idx.intersection(scores[k].index)
    idx = idx.intersection(human.index)
    a, h, s, m, me = (scores[k]["score"][idx] for k in keys)
    hu = human["score"][idx]
    out = {}
    q = (a < 0.3) & (s >= 0.8)
    out["shortlist_rescues"] = first_per_log((s - a)[q].sort_values(ascending=False).index, log_of, n_top)
    q = (a >= 0.8) & (h < a - 0.3)
    out["heads_choice_unsafe"] = first_per_log((a - h)[q].sort_values(ascending=False).index, log_of, n_top)
    q = (a > 0.5) & (h > 0.5) & (s > 0.5) & (me > 0.5)
    typ = []
    seen = set()
    for t in rng.choice(np.array(idx[q]), size=min(int(q.sum()), n_rand * 10), replace=False):
        if log_of(t) in seen:
            continue
        tr, _ = rules(t)
        if np.ptp(np.array([tr[k][-1, :2] for k in RULE5]), axis=0).max() > 1.0:  # end points > 1 m apart
            typ.append(t)
            seen.add(log_of(t))
        if len(typ) == n_rand:
            break
    out["all_rules_safe"] = typ
    q = (me >= 0.95) & (hu <= 0.8)
    out["beats_human"] = first_per_log((me - hu)[q].sort_values(ascending=False).index, log_of, n_top)
    turns = [t for t in idx[me >= 0.99] if abs(rules.ec["arrays"]["trajectory"][rules.ec["row"][t]][-1, 2]) > 0.61]
    out["perfect_turns"] = first_per_log(rng.permutation(np.array(turns)), log_of, n_rand)
    q = me == 0
    out["best_rule_fails"] = first_per_log(rng.permutation(np.array(idx[q])), log_of, n_rand)
    return out


# print layout, 6 in wide, margins 0.01: three map columns (wspace 0.04) are 1.91 in wide; three camera images
# (wspace 0.02, sky cropped: 1920 x 745 px) are 1.93 x 0.75 in, with a 0.14 in label row above them
MAP_W, CAM_H, CAM_LABEL_H = 1.91, 0.75, 0.18
CAM_LABELS = {"cam_l0": "front left", "cam_f0": "front", "cam_r0": "front right"}


def camera_row(fig, spec, prefix=""):
    """front-left, front, front-right camera axes side by side in one gridspec cell, each with a small label
    (prefix goes before the first label, e.g. the scene name)"""
    sub = spec.subgridspec(1, 3, wspace=0.02)
    axes = [fig.add_subplot(sub[0, j]) for j in range(3)]
    for j, (ax, lab) in enumerate(zip(axes, CAM_LABELS.values())):
        ax.set_title(prefix + lab if j == 0 else lab, fontsize=6.5, pad=2.5, color="#333333")
    return axes


def three_cameras(axes, frame, trajs, dotted=None, cams=("cam_l0", "cam_f0", "cam_r0")):
    """Front-left, front, front-right cameras with the trajectories projected (each where that camera sees it) and
    their end points as stars; dotted: {label: (traj, colour)} drawn dotted (e.g. the plan of t - 0.5 s)."""
    z = ground_z(frame)
    for ax, name in zip(axes, cams):
        draw_camera(ax, frame, trajs, cam_name=name)
        cam = getattr(frame.cameras, name)
        h, w = cam.image.shape[:2]
        items = [(STYLE[k][0], t, "-") for k, t in trajs.items()] + [(c, t, ":") for t, c in (dotted or {}).values()]
        for colour, traj, ls in items:
            if ls == ":":
                uv, keep = project(cam, densify(traj), z)
                ax.plot(uv[keep, 0], uv[keep, 1], color=colour, linestyle=":", linewidth=2.6, zorder=30)
            uv, keep = project(cam, traj[-1:, :2], z)
            if keep[0] and 0 < uv[0, 0] < w and h * 0.3 < uv[0, 1] < h:
                ax.plot(uv[0, 0], uv[0, 1], marker="*", markersize=8, color=colour if ls == "-" else "white",
                        markeredgecolor="white" if ls == "-" else colour, markeredgewidth=0.9, zorder=40)


def run_rules(args):
    rules = Rules(args.model, args.scorer, args.ec_folder)
    snap = CollisionSnapshot(E / "metric_cache_navtest")  # navtest: official non-reactive scoring
    names = {"argmax": f"{args.model}_argmax", "heads": f"{args.heads_eval}_score_heads",
             "shortlist k256": f"{args.model}_shortlist_k256", "mixture λ10": f"{args.model}_mixture_lam10",
             "mixture + EC n64": f"{args.model}_own_ecseq_mix10_n64_argmax",
             "shortlist + EC n64": f"{args.model}_own_ecseq_short256_n64_argmax"}
    scores = {k: per_scene(v) for k, v in names.items()}
    human = per_scene(args.human_eval, suffix="")
    rng = np.random.default_rng(args.seed)
    loader = scene_loader(list(scores["argmax"].index), Path(args.sensor_path))  # all navtest scenes (logs, rendering)
    log_of = lambda t: loader.scene_frames_dicts[t][3]["log_name"]
    sel = select_rules(scores, human, rules, rng, args.n_top, args.n_random, log_of)
    out_root = Path(args.out_dir) / "rules"
    index = ["# Qualitative: selection rules on one scene", "",
             f"Model `{args.model}`, scorer `{args.scorer}` (heads-only score from `{args.heads_eval}_score_heads`), "
             f"EC rule `{args.ec_folder}`, human = `{args.human_eval}`. Seed {args.seed}.", "",
             "Categories (how the scenes were chosen, from the per-scene official navtest v2 scores):", ""] + \
            [f"- `{c}`: {d}" for c, d in CATEGORIES.items()] + \
            ["", "| Category | File | Token | human | " + " | ".join(names) + " |",
             "|---|---|---|---|" + "---|" * len(names)]
    for cat, toks in sel.items():
        (out_root / cat).mkdir(parents=True, exist_ok=True)
        for i, tok in enumerate(toks):
            scene = loader.get_scene_from_token(tok)
            frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
            tr, cand = rules(tok)
            tr = {k: tr[k] for k in RULE5}
            hum = scene.get_future_trajectory().poses
            snaps = {k: snap(tok, poses8(tr[k])) for k in RULE5 if scores[k].loc[tok, "no_at_fault_collisions"] == 0}
            for k, sn in snaps.items():  # NAVSIM's collision for every NC 0 plan (recomputed NC must match)
                assert sn["nc"] == 0 and sn["t"] is not None, (tok, k, sn)
            geo = {f"_{k}{j}": g for k, sn in snaps.items() for j, g in enumerate([sn["ego"], *sn["agents"]])}
            lim = bev_extent({"human": hum, **tr, **geo})  # the collision (ego, agent hit) stays inside the map
            # inches: camera labels, cameras, title, maps, title, maps, gap, legend
            rows = [CAM_LABEL_H, CAM_H, 0.40, MAP_W, 0.40, MAP_W, 0.06, 0.76]
            fig = plt.figure(figsize=(PRINT_W, sum(rows) / 0.98))
            gs = fig.add_gridspec(len(rows), 3, height_ratios=rows, hspace=0, wspace=0.04)
            three_cameras(camera_row(fig, gs[1, :]), frame, {"human": hum, "mixture + EC n64": tr["mixture + EC n64"]})
            order = ["argmax", "heads", "shortlist k256", "mixture λ10", "mixture + EC n64", "shortlist + EC n64"]
            for j, k in enumerate(order):
                ax = fig.add_subplot(gs[3 + 2 * (j // 3), j % 3])
                draw_bev(ax, scene, {"human": hum, k: tr[k]}, None, lim=lim)
                draw_future_agents(ax, scene)
                if k in snaps:  # NAVSIM's collision: ego, agent hit, time
                    draw_collision(ax, snaps[k], STYLE[k][0])
                ax.set_title(f"{SHORT[k]}\nEPDMS {fmt_scores(scores[k].loc[tok])}",
                             color="#b00000" if scores[k].loc[tok, "score"] < 0.5 else "black")
            lax = fig.add_subplot(gs[7, :])
            lax.axis("off")
            lax.legend([plt.Line2D([], [], color="black", linestyle="--", linewidth=1.4)] +
                       map_legend_handles(footprint=True), [f"human (logged): EPDMS {fmt_scores(human.loc[tok])}"] +
                       [h.get_label() for h in map_legend_handles(footprint=True)], loc="upper center", ncol=4, frameon=False,
                       handlelength=1.6, columnspacing=1.0)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.985, bottom=0.005)
            f = out_root / cat / f"{i:02d}_{tok}.png"
            fig.savefig(f, dpi=args.dpi)
            plt.close(fig)
            index.append(f"| {cat} | `{f.relative_to(args.out_dir)}` | {tok} | {human.loc[tok, 'score']:.2f} | " +
                         " | ".join(f"{scores[k].loc[tok, 'score']:.2f}" for k in names) + " |")
            print(f, flush=True)
    (out_root / "index.md").write_text("\n".join(index) + "\n")


def to_frame(traj_prev, pose_prev, pose_cur):
    """Plan given in the ego frame at pose_prev -> ego frame at pose_cur (global poses x, y, heading)."""
    c, s = np.cos(pose_prev[2]), np.sin(pose_prev[2])
    g = np.stack([pose_prev[0] + c * traj_prev[:, 0] - s * traj_prev[:, 1],
                  pose_prev[1] + s * traj_prev[:, 0] + c * traj_prev[:, 1]], 1)
    c, s = np.cos(pose_cur[2]), np.sin(pose_cur[2])
    d = g - pose_cur[:2]
    xy = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)
    if traj_prev.shape[1] < 3:
        return xy
    return np.concatenate([xy, (traj_prev[:, 2:3] + pose_prev[2] - pose_cur[2])], 1)  # heading too


CONSEC_ROWS = [["heads", "argmax", "shortlist k256", "mixture λ10"],
               ["argmax + EC n64", "shortlist + EC n64", "mixture + EC n64"]]
CONSEC_CATS = {  # folder -> (rule that flips, its EC-rule version, description)
    "heads_flips": ("heads", "mixture + EC n64",
                    "heads-only choice fails EC, mixture + consistency passes, both without collision / off-road"),
    "shortlist_flips": ("shortlist k256", "shortlist + EC n64",
                        "shortlist fails EC, shortlist + consistency passes, both without collision / off-road"),
}


def run_consecutive(args):
    rules = Rules(args.model, args.scorer, args.ec_folder)
    names = {"heads": f"{args.heads_eval}_score_heads", "argmax": f"{args.model}_argmax",
             "shortlist k256": f"{args.model}_shortlist_k256", "mixture λ10": f"{args.model}_mixture_lam10",
             "argmax + EC n64": f"{args.model}_own_ecseq_argmax_n64_argmax",
             "shortlist + EC n64": f"{args.model}_own_ecseq_short256_n64_argmax",
             "mixture + EC n64": f"{args.model}_own_ecseq_mix10_n64_argmax"}
    scores = {k: per_scene(v) for k, v in names.items()}
    human = per_scene(args.human_eval, suffix="")
    idx = scores["heads"].index.intersection(human.index)
    for k in names:
        idx = idx.intersection(scores[k].index)
    rng = np.random.default_rng(args.seed)
    ok = lambda k: (scores[k].loc[idx, "no_at_fault_collisions"] == 1) & (scores[k].loc[idx, "drivable_area_compliance"] == 1)
    ec = lambda k: scores[k].loc[idx, "two_frame_extended_comfort"]
    pools = {}
    for cat, (bad, good, _) in CONSEC_CATS.items():
        q = (ec(bad) == 0) & (ec(good) == 1) & ok(bad) & ok(good)
        pools[cat] = list(rng.choice(np.array(idx[q]), size=min(int(q.sum()), 300), replace=False))
    loader = scene_loader([t for v in pools.values() for t in v], Path(args.sensor_path))

    def poses(tok):
        fr = loader.scene_frames_dicts[tok]
        return [np.array([fr[i]["ego2global_translation"][0], fr[i]["ego2global_translation"][1],
                          _yaw(fr[i]["ego2global_rotation"])]) for i in (3, 2)]

    def jump(tr, trp, k, pc, pp):
        return float(np.linalg.norm(to_frame(trp[k], pp, pc)[-1, :2] - tr[k][-1, :2]))

    chosen = {}
    for cat, toks in pools.items():
        bad, good, _ = CONSEC_CATS[cat]
        rows = []
        for tok in toks:
            prev = previous_token(loader, tok)
            if prev not in rules.model["row"] or any(prev not in pr["row"] for pr in [rules.ec, *rules.ec_more.values()]):
                continue
            tr, _ = rules(tok)
            trp, _ = rules(prev)
            pc, pp = poses(tok)
            rows.append((jump(tr, trp, bad, pc, pp) - jump(tr, trp, good, pc, pp), tok, prev))
        rows.sort(reverse=True)
        top = first_per_log([t for _, t, _ in rows], lambda t: loader.scene_frames_dicts[t][3]["log_name"], args.n_top)
        prev_of = {t: p for _, t, p in rows}
        chosen[cat] = [(t, prev_of[t]) for t in top]
    loader2 = scene_loader([t for v in chosen.values() for pair in v for t in pair], Path(args.sensor_path))
    out_root = Path(args.out_dir) / "consecutive"
    index = ["# Qualitative: consecutive frames (t − 0.5 s and t) under every selection rule", "",
             f"Model `{args.model}`, scorer `{args.scorer}`. Every panel shows the rule's plan at t (solid) and the "
             "plan the same rule output at t − 0.5 s (dotted, moved into the frame at t). EC (two-frame extended "
             "comfort) passes when the two agree. Categories (top by the flipping rule's end-point jump minus its "
             f"consistency version's, from a random pool of up to 300 scenes, seed {args.seed}):", ""] + \
            [f"- `{c}`: {d}" for c, (_, _, d) in CONSEC_CATS.items()] + \
            ["", "| Category | File | Token (t) | previous | " + " | ".join(f"{k} EC / jump" for k in names) + " |",
             "|---|---|---|---|" + "---|" * len(names)]
    for cat, pairs in chosen.items():
        (out_root / cat).mkdir(parents=True, exist_ok=True)
        for i, (tok, prev) in enumerate(pairs):
            scene = loader2.get_scene_from_token(tok)
            tr, _ = rules(tok)
            trp, _ = rules(prev)
            fr = loader2.scene_frames_dicts[tok]
            pc, pp = [np.array([fr[j]["ego2global_translation"][0], fr[j]["ego2global_translation"][1],
                                _yaw(fr[j]["ego2global_rotation"])]) for j in (3, 2)]
            prev_in = {k: to_frame(trp[k], pp, pc) for k in names}
            hum = scene.get_future_trajectory().poses
            lim = bev_extent({"human": hum, **{k: tr[k] for k in names}}, {k: (v, "k", ":") for k, v in prev_in.items()})
            fig = plt.figure(figsize=(PRINT_W, 7.3))
            gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.04)
            grid = [["argmax", "shortlist k256", "mixture λ10"],
                    ["argmax + EC n64", "shortlist + EC n64", "mixture + EC n64"], ["heads", "human", None]]
            for r, keys in enumerate(grid):
                for j, k in enumerate(keys):
                    if k is None:
                        continue
                    ax = fig.add_subplot(gs[r, j])
                    if k == "human":
                        hum_prev = to_frame(loader2.get_scene_from_token(prev).get_future_trajectory().poses, pp, pc)
                        draw_bev(ax, scene, {"human": hum}, extra={"t − 0.5 s": (hum_prev, "0.35", ":")}, lim=lim)
                        row, name = human.loc[tok], "Human (logged)"
                        jm = float(np.linalg.norm(hum_prev[-1, :2] - hum[-1, :2]))
                    else:
                        draw_bev(ax, scene, {"human": hum, k: tr[k]}, extra={"t − 0.5 s": (prev_in[k], STYLE[k][0], ":")},
                                 lim=lim)
                        row, name = scores[k].loc[tok], SHORT[k]
                        jm = float(np.linalg.norm(prev_in[k][-1, :2] - tr[k][-1, :2]))
                    passed = row["two_frame_extended_comfort"] == 1
                    ax.set_title(f"{name}\nEC {'pass' if passed else 'fail'}, moved {jm:.1f} m · EPDMS {row['score']:.2f}",
                                 color="black" if passed else "#b00000")
            lax = fig.add_subplot(gs[2, 2])
            lax.axis("off")
            lines = [plt.Line2D([], [], color="0.2", linewidth=1.4), plt.Line2D([], [], color="0.2", linestyle=":", linewidth=1.4),
                     plt.Line2D([], [], color="black", linestyle="--", linewidth=1.4)]
            lax.legend(lines + map_legend_handles(future=False),
                       ["plan at t", "plan at t − 0.5 s", "human (logged)"] + [h.get_label() for h in map_legend_handles(future=False)],
                       loc="center", ncol=1, frameon=False, handlelength=1.6, labelspacing=0.35)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.945, bottom=0.01)
            f = out_root / cat / f"{i:02d}_{tok}.png"
            fig.savefig(f, dpi=args.dpi)
            plt.close(fig)
            index.append(f"| {cat} | `{f.relative_to(args.out_dir)}` | {tok} | {prev} | " +
                         " | ".join(f"{scores[k].loc[tok, 'two_frame_extended_comfort']:.0f} / "
                                    f"{np.linalg.norm(prev_in[k][-1, :2] - tr[k][-1, :2]):.1f} m" for k in names) + " |")
            print(f, flush=True)
    (out_root / "index.md").write_text("\n".join(index) + "\n")


def _yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figure", choices=["rules", "consecutive"], required=True)
    p.add_argument("--sensor-path", required=True, help="mounted test sensor image (squashfuse)")
    p.add_argument("--out-dir", default=str(E / "qualitative"))
    p.add_argument("--model", default="rl3_b0.005_ep01")
    p.add_argument("--scorer", default="p3_aux_policy_ep03", help="prediction folder with the scorer's sub-score logits")
    p.add_argument("--heads-eval", default="p3_aux_ep05", help="official score_heads evaluation of the scorer's heads")
    p.add_argument("--ec-folder", default="rl3_b0.005_ep01_own_ecseq_mix10_n64")
    p.add_argument("--human-eval", default="human_agent_non_reactive", help="pdm_navtest folder of the logged human")
    p.add_argument("--n-top", type=int, default=4)
    p.add_argument("--n-random", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    {"rules": run_rules, "consecutive": run_consecutive}[args.figure](args)


if __name__ == "__main__":
    main()
