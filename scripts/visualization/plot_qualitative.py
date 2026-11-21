"""Qualitative figures of the vision_llm_lora planner on navtest: front camera with the trajectories projected onto the
ground + bird's-eye view (map, agents, ego, candidates, trajectories), one PNG per scene, plus an index.md.

Figures (--figure):
  rules        one scene, several selection rules of one model (default: the RL-3 model rl3_b0.005_ep01 with its own
               scorer p3_aux_policy_ep03): human, argmax, heads (score_heads), shortlist k 256, mixture lambda 10,
               mixture + EC rule n 64 (sequential). BEV also shows the policy's top candidates (end points coloured by
               probability) and the scorer's top-256 shortlist (grey end points).
  consecutive  the plans at t - 0.5 s and at t for heads selection vs mixture + EC rule (EC compares the two).
Scenes are chosen by stated rules from the per-scene official scores (no hand picking):
  rules:       rescue   argmax scored < 0.3, shortlist >= 0.8 (top by shortlist - argmax)
               heads_fail  heads selection >= 0.3 below an argmax scored >= 0.8 (top by argmax - heads)
               typical  all rules > 0.5 and the rules choose different trajectories (random, --seed)
               failure  mixture + EC rule (the best navtest rule) scored 0 (random, --seed)
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
}


# ----------------------------------------------------------------------------------------------- data
def latest_csv(folder: Path):
    fs = [f for f in glob.glob(f"{folder}/*/*.csv") if re.search(r"/\d{4}(\.\d{2}){5}\.csv$", f)]
    return max(fs, key=os.path.getmtime) if fs else None


def per_scene(name: str) -> pd.DataFrame:
    f = latest_csv(E / "pdm_navtest" / f"{name}_non_reactive_split")
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
        self.codebook = self.model["codebook"].float()

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
    cams = dict(cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False)
    return SceneLoader(D / "navsim_logs/test", sensor_path, sf,
                       sensor_config=SensorConfig(cam_f0=list(frames), lidar_pc=False, **cams))


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


def densify(traj_xy, n=8):
    pts = np.concatenate([[[0.0, 0.0]], traj_xy[:, :2]])
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, n + 1)[1:]:
            out.append(a + t * (b - a))
    return np.array(out)


def project(camera, xy, z):
    pts = np.concatenate([xy, np.full((len(xy), 1), z)], 1)
    r, t = camera.sensor2lidar_rotation, camera.sensor2lidar_translation
    cam = (pts - t) @ r  # lidar -> camera: R^T (p - t), as _transform_annotations_to_camera
    keep = cam[:, 2] > 1.0
    uv, inside = _transform_points_to_image(cam, camera.intrinsics, camera.image.shape[:2])
    return uv, keep


def draw_camera(ax, frame, trajs, title=None):
    cam = frame.cameras.cam_f0
    img = cam.image
    ax.imshow(img)
    z = ground_z(frame)
    for label, traj in trajs.items():
        c, ls, lw, zo = STYLE[label]
        uv, keep = project(cam, densify(traj), z)
        uv = uv[keep]
        ax.plot(uv[:, 0], uv[:, 1], color=c, linestyle=ls, linewidth=lw * 1.4, zorder=zo, solid_capstyle="round")
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
    if cand is not None:
        ax.scatter(cand["short_xy"][:, 1], cand["short_xy"][:, 0], s=4, c="0.45", marker="x", linewidths=0.5,
                   alpha=0.45, zorder=20, label="heads top-256 (end points)")
        o = np.argsort(cand["policy_p"])
        ax.scatter(cand["policy_xy"][o, 1], cand["policy_xy"][o, 0], s=4, c=np.log10(cand["policy_p"][o]),
                   cmap="viridis", vmin=np.log10(cand["policy_p"]).max() - 3, zorder=21, alpha=0.55, linewidths=0,
                   label="policy top-200 (colour: log p)")
    for label, traj in trajs.items():
        c, ls, lw, zo = STYLE[label]
        pts = densify(traj, 1)
        ax.plot(pts[:, 1], pts[:, 0], color=c, linestyle=ls, linewidth=lw, zorder=zo, label=label,
                marker="o" if label != "human" else None, markersize=2.5)
        ax.plot(pts[-1, 1], pts[-1, 0], marker="*", markersize=11, color=c, markeredgecolor="white",
                markeredgewidth=0.7, zorder=zo + 10)  # end point: a stop shows as a star at the ego
    for label, (traj, c, ls) in (extra or {}).items():
        pts = traj[:, :2]
        ax.plot(pts[:, 1], pts[:, 0], color=c, linestyle=ls, linewidth=1.6, zorder=27, label=label, marker=".",
                markersize=3)
        ax.plot(pts[-1, 1], pts[-1, 0], marker="*", markersize=9, color="white", markeredgecolor=c, zorder=37)
    ax.set_aspect("equal")
    ax.set_xlim(lim[1], lim[0])  # left = +y
    ax.set_ylim(lim[2], lim[3])
    ax.set_xticks([])
    ax.set_yticks([])


def fmt_scores(row):
    bad = [AB[k] for k in SUB if k not in ("ego_progress",) and row[k] < 1]
    return f"{row['score']:.2f}" + (f" ({', '.join(bad)})" if bad else "")


def scene_info(scene):
    f = scene.frames[scene.scene_metadata.num_history_frames - 1]
    cmd = COMMANDS.get(int(np.argmax(f.ego_status.driving_command)), "?")
    v = float(np.hypot(*f.ego_status.ego_velocity[:2]))
    return f"{scene.scene_metadata.log_name} | command: {cmd} | speed {v:.1f} m/s"


# ----------------------------------------------------------------------------------------------- figures
def select_rules(scores, rng, n_top, n_rand):
    a, h, s, m, me = (scores[k]["score"] for k in ["argmax", "heads", "shortlist k256", "mixture λ10", "mixture + EC n64"])
    idx = a.index.intersection(h.index).intersection(s.index).intersection(m.index).intersection(me.index)
    a, h, s, m, me = a[idx], h[idx], s[idx], m[idx], me[idx]
    out = {}
    q = (a < 0.3) & (s >= 0.8)
    out["rescue"] = list((s - a)[q].sort_values(ascending=False).index[:n_top])
    q = (a >= 0.8) & (h < a - 0.3)
    out["heads_fail"] = list((a - h)[q].sort_values(ascending=False).index[:n_top])
    q = (a > 0.5) & (h > 0.5) & (s > 0.5) & (me > 0.5)
    out["typical"] = list(rng.choice(np.array(idx[q]), size=n_rand * 3, replace=False))  # filtered to differing below
    q = me == 0
    out["failure"] = list(rng.choice(np.array(idx[q]), size=n_rand, replace=False))
    return out


def run_rules(args):
    rules = Rules(args.model, args.scorer, args.ec_folder)
    names = {"argmax": f"{args.model}_argmax", "heads": f"{args.heads_eval}_score_heads",
             "shortlist k256": f"{args.model}_shortlist_k256", "mixture λ10": f"{args.model}_mixture_lam10",
             "mixture + EC n64": f"{args.model}_own_ecseq_mix10_n64_argmax"}
    scores = {k: per_scene(v) for k, v in names.items()}
    rng = np.random.default_rng(args.seed)
    sel = select_rules(scores, rng, args.n_top, args.n_random)
    # typical: keep scenes where the rules' choices differ (end points > 1 m apart), first n_random
    typ = []
    for t in sel["typical"]:
        tr, _ = rules(t)
        ends = np.array([v[-1, :2] for v in tr.values()])
        if np.ptp(ends, axis=0).max() > 1.0:
            typ.append(t)
        if len(typ) == args.n_random:
            break
    sel["typical"] = typ
    tokens = [t for v in sel.values() for t in v]
    loader = scene_loader(tokens, Path(args.sensor_path))
    out_root = Path(args.out_dir) / "rules"
    index = ["# Qualitative: selection rules on one scene", "",
             f"Model `{args.model}`, scorer `{args.scorer}` (heads score from `{args.heads_eval}_score_heads`), EC rule "
             f"`{args.ec_folder}`. Selection: see the script docstring; seed {args.seed}.", "",
             "| Category | File | Token | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
    for cat, toks in sel.items():
        (out_root / cat).mkdir(parents=True, exist_ok=True)
        for i, tok in enumerate(toks):
            scene = loader.get_scene_from_token(tok)
            frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
            tr, cand = rules(tok)
            trajs = {"human": scene.get_future_trajectory().poses, **tr}
            fig = plt.figure(figsize=(15, 5.6))
            gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.02)
            ax_c, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
            draw_camera(ax_c, frame, trajs)
            draw_bev(ax_b, scene, trajs, cand)
            same = {k: [j for j in list(tr)[: list(tr).index(k)] if np.abs(tr[j] - tr[k]).max() < 1e-3] for k in tr}
            labels = ["human"] + [f"{k}: {fmt_scores(scores[k].loc[tok])}" + (f" (= {same[k][0]})" if same[k] else "")
                                  for k in tr]
            handles = [plt.Line2D([], [], color=STYLE[k][0], linestyle=STYLE[k][1], linewidth=STYLE[k][2]) for k in trajs]
            handles += [plt.Line2D([], [], color="0.55", marker="x", linestyle=""),
                        plt.Line2D([], [], color=plt.cm.viridis(0.9), marker="o", linestyle="")]
            labels += ["heads top-256 end points", f"policy top-200 end points (colour log p; argmax p = {cand['p_argmax']:.2f})"]
            fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, frameon=False,
                       title="per-scene PDMS (v2), failed terms in brackets", title_fontsize=8.5)
            fig.suptitle(f"{cat}: {scene_info(scene)} | token {tok}", fontsize=10, y=0.98)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.17)
            f = out_root / cat / f"{i:02d}_{tok}.png"
            fig.savefig(f, dpi=args.dpi)
            plt.close(fig)
            index.append(f"| {cat} | `{f.relative_to(args.out_dir)}` | {tok} | " +
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
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


def run_consecutive(args):
    rules = Rules(args.model, args.scorer, args.ec_folder)
    heads, ecr = per_scene(f"{args.heads_eval}_score_heads"), per_scene(f"{args.model}_own_ecseq_mix10_n64_argmax")
    idx = heads.index.intersection(ecr.index)
    q = ((heads.loc[idx, "two_frame_extended_comfort"] == 0) & (ecr.loc[idx, "two_frame_extended_comfort"] == 1)
         & (heads.loc[idx, "no_at_fault_collisions"] == 1) & (heads.loc[idx, "drivable_area_compliance"] == 1)
         & (ecr.loc[idx, "no_at_fault_collisions"] == 1) & (ecr.loc[idx, "drivable_area_compliance"] == 1))
    cands = list(idx[q])
    rng = np.random.default_rng(args.seed)
    cands = list(rng.choice(np.array(cands), size=min(len(cands), 400), replace=False))
    # need the previous frame: load all candidate logs once
    loader = scene_loader(cands, Path(args.sensor_path))
    rows = []
    for tok in cands:
        prev = previous_token(loader, tok)
        if prev not in rules.model["row"] or prev not in rules.ec["row"]:
            continue
        tr, _ = rules(tok)
        trp, _ = rules(prev)
        fr = loader.scene_frames_dicts[tok]
        pc, pp = (np.array([fr[i]["ego2global_translation"][0], fr[i]["ego2global_translation"][1],
                            _yaw(fr[i]["ego2global_rotation"])]) for i in (3, 2))
        jh = np.linalg.norm(to_frame(trp["heads"][:, :2], pp, pc)[-1] - tr["heads"][-1, :2])
        je = np.linalg.norm(to_frame(trp["mixture + EC n64"][:, :2], pp, pc)[-1] - tr["mixture + EC n64"][-1, :2])
        rows.append((tok, prev, jh - je, jh, je))
    rows.sort(key=lambda r: -r[2])
    out_root = Path(args.out_dir) / "consecutive"
    out_root.mkdir(parents=True, exist_ok=True)
    index = ["# Qualitative: consecutive frames (t − 0.5 s and t), heads selection vs mixture + EC rule", "",
             f"Model `{args.model}`, scorer `{args.scorer}`. Scenes: heads EC = 0, EC rule EC = 1, both without collision "
             f"or off-road; top by (heads end-point jump − EC-rule end-point jump), from {len(rows)} candidates "
             f"(random subset of 400, seed {args.seed}).", "",
             "| File | Token (t) | previous | heads jump (m) | EC-rule jump (m) | heads PDMS | EC-rule PDMS |",
             "|---|---|---|---|---|---|---|"]
    top = rows[: args.n_top]
    loader2 = scene_loader([t for r in top for t in r[:2]], Path(args.sensor_path))
    for i, (tok, prev, _, jh, je) in enumerate(top):
        scene, scene_p = loader2.get_scene_from_token(tok), loader2.get_scene_from_token(prev)
        tr, _ = rules(tok)
        trp, _ = rules(prev)
        fr = loader.scene_frames_dicts[tok]
        pc, pp = (np.array([fr[k]["ego2global_translation"][0], fr[k]["ego2global_translation"][1],
                            _yaw(fr[k]["ego2global_rotation"])]) for k in (3, 2))
        fig = plt.figure(figsize=(15, 9.6))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.75, 1.0, 1.0], wspace=0.02, hspace=0.12)
        for r, (label, col) in enumerate([("heads", "#ff7f0e"), ("mixture + EC n64", "#d62728")]):
            ax_c = fig.add_subplot(gs[r, 0])
            frame = scene.frames[3]
            prev_in_cur = to_frame(trp[label][:, :2], pp, pc)
            draw_camera(ax_c, frame, {label: tr[label]})
            uv, keep = project(frame.cameras.cam_f0, densify(prev_in_cur), ground_z(frame))
            ax_c.plot(uv[keep, 0], uv[keep, 1], color=col, linestyle=":", linewidth=2.4, zorder=9)
            ax_c.set_title(f"{label} at t (solid) and its plan from t − 0.5 s (dotted)", fontsize=9)
            ax_p = fig.add_subplot(gs[r, 1])
            draw_bev(ax_p, scene_p, {label: trp[label]})
            ax_p.set_title(f"t − 0.5 s: {label}", fontsize=9)
            ax_b = fig.add_subplot(gs[r, 2])
            draw_bev(ax_b, scene, {label: tr[label]}, extra={"plan from t − 0.5 s": (prev_in_cur, col, ":")})
            row = (heads if label == "heads" else ecr).loc[tok]
            ax_b.set_title(f"t: {label}, PDMS {fmt_scores(row)}, end-point jump {jh if label == 'heads' else je:.1f} m",
                           fontsize=9)
        fig.suptitle(f"consecutive: {scene_info(scene)} | t {tok}, t − 0.5 s {prev}", fontsize=10, y=0.995)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
        f = out_root / f"{i:02d}_{tok}.png"
        fig.savefig(f, dpi=args.dpi)
        plt.close(fig)
        index.append(f"| `{f.relative_to(args.out_dir)}` | {tok} | {prev} | {jh:.1f} | {je:.1f} | "
                     f"{heads.loc[tok, 'score']:.2f} | {ecr.loc[tok, 'score']:.2f} |")
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
    p.add_argument("--n-top", type=int, default=4)
    p.add_argument("--n-random", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    {"rules": run_rules, "consecutive": run_consecutive}[args.figure](args)


if __name__ == "__main__":
    main()
