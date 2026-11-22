"""navhard two-stage qualitative figure: one original (stage one) scene and two of its synthetic stage-two follow-ups
(re-rendered from a perturbed ego start), one the model handles and one it fails, with the plans of the policy's
argmax, the shortlist (k 256) and the best navhard rule (shortlist + consistency rule n 64), scored with the stage's
EPDMS. Groups chosen from the official per-scene scores of the best rule: stage one >= 0.9 and among its stage-two
scenes one >= 0.9 and one <= 0.3 (random groups, --seed). Uses the drawing helpers of plot_qualitative.py.
Needs the test sensor image mounted (original scenes); the synthetic images are read from
$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs.
  python scripts/visualization/plot_navhard.py --sensor-path <mount> --out-dir $NAVSIM_EXP_ROOT/qualitative_v2/navhard
"""
import argparse
import glob
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("pq", HERE / "plot_qualitative.py")
pq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pq)
plt = pq.plt
sys.path.insert(0, str(REPO / "scripts/evaluation"))
from run_pdm_score_balanced import fast_filter_synthetic_scenes  # noqa: E402

from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from navsim.agents.vision_llm_lora.vision_llm_lora_agent import select_trajectory  # noqa: E402
from navsim.agents.vision_llm_lora.vision_llm_lora_precomputed_agent import load_predictions  # noqa: E402
from navsim.common import dataloader  # noqa: E402
from navsim.common.dataclasses import SensorConfig  # noqa: E402

dataloader.filter_synthetic_scenes = fast_filter_synthetic_scenes
E, D = Path(os.environ["NAVSIM_EXP_ROOT"]), Path(os.environ["OPENSCENE_DATA_ROOT"])
SPLIT = "navhard_two_stage"
RULES = {"argmax": "Policy (argmax)", "shortlist k256": "Shortlist (k = 256)",
         "shortlist + EC n64": "Shortlist + consistency"}
COLORS = {"argmax": "#1f77b4", "shortlist k256": "#2ca02c", "shortlist + EC n64": "#006d2c"}  # as in plot_qualitative


def per_scene(name):
    fs = [f for f in glob.glob(f"{E}/pdm_navhard_two_stage/{name}_reactive_split/*/*.csv")
          if re.search(r"/\d{4}(\.\d{2}){5}\.csv$", f)]
    d = pd.read_csv(max(fs, key=os.path.getmtime)).set_index("token")
    return d[~d.index.astype(str).str.startswith("extended")]


def fmt(row, stage):
    bad = [f"{pq.AB[k]} {row[f'{k}_{stage}']:.2f}".replace(" 0.00", " 0") for k in pq.SUB
           if pd.notna(row.get(f"{k}_{stage}")) and row[f"{k}_{stage}"] < 0.995]
    return f"{row['score']:.2f}" + (f" ({', '.join(bad)})" if bad else "")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensor-path", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model", default="rl3_b0.005_ep01")
    p.add_argument("--scorer", default="p3_aux_policy_ep03")
    p.add_argument("--n-groups", type=int, default=4)
    p.add_argument("--original-human", action="store_true",
                   help="synthetic scenes have no future frames; draw the original drive's human path instead "
                        "(moved into the synthetic frame; not a valid reference from the perturbed start)")
    p.add_argument("--mode", choices=["rescue", "mixed"], default="rescue",
                   help="rescue: original + a synthetic follow-up where argmax <= 0.3 and the shortlist >= 0.8; "
                        "mixed: original + one handled and one failed synthetic follow-up (best rule)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    names = {"argmax": f"{args.model}_argmax", "shortlist k256": f"{args.model}_shortlist_k256",
             "shortlist + EC n64": f"{args.model}_own_ec_short256_n64_argmax"}
    scores = {k: per_scene(v) for k, v in names.items()}
    best = scores["shortlist + EC n64"]
    mapping = yaml.safe_load((REPO / "navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml")
                             .read_text())["reactive_all_mapping"]
    groups = []
    amax, short = scores["argmax"], scores["shortlist k256"]
    for now, _, second in mapping:
        if now not in best.index:
            continue
        if args.mode == "rescue":  # original handled by the shortlist; a synthetic follow-up argmax fails, shortlist not
            if short.loc[now, "score"] < 0.8:
                continue
            res = [(sh - am, a) for a, _ in second if a in amax.index
                   for am, sh in [(amax.loc[a, "score"], short.loc[a, "score"])] if am <= 0.3 and sh >= 0.8]
            if res:
                groups.append((now, max(res)[1]))
            continue
        sc = [(a, best.loc[a, "score"]) for a, _ in second if a in best.index]
        if sc and best.loc[now, "score"] >= 0.9:
            hi, lo = max(sc, key=lambda x: x[1]), min(sc, key=lambda x: x[1])
            if hi[1] >= 0.9 and lo[1] <= 0.3:
                groups.append((now, hi[0], lo[0]))
    rng = np.random.default_rng(args.seed)
    groups = [groups[i] for i in rng.permutation(len(groups))[: args.n_groups]]
    pred = {k: load_predictions(E / "predictions" / f / SPLIT) for k, f in
            [("model", args.model), ("scorer", args.scorer), ("ec", f"{args.model}_own_ec_short256_n64")]}
    codebook = pred["model"]["codebook"].float()
    cb_dir = REPO / "navsim/agents/vision_llm_lora/codebook"
    cb8, cb40 = np.load(cb_dir / "16384_8step.npy"), np.load(cb_dir / "16384.npy")
    key = {e.tobytes(): k for k, e in enumerate(cb8)}
    fine = lambda t: cb40[key[np.ascontiguousarray(t, dtype=np.float32).tobytes()]].astype(float) \
        if np.ascontiguousarray(t, dtype=np.float32).tobytes() in key else t

    def plans(tok):
        r, s = pred["model"]["row"][tok], pred["scorer"]["row"][tok]
        x = {"trajectory": torch.from_numpy(np.array(pred["model"]["arrays"]["trajectory"][r: r + 1])),
             "trajectory_logits": pq.bf16(pred["model"], "trajectory_logits", r),
             "pdm_score_logits": pq.bf16(pred["scorer"], "pdm_score_logits", s)}
        out = {"argmax": x["trajectory"][0].numpy(),
               "shortlist k256": select_trajectory(x, "shortlist", codebook, shortlist_size=256)[0].numpy(),
               "shortlist + EC n64": np.array(pred["ec"]["arrays"]["trajectory"][pred["ec"]["row"][tok]])}
        return {k: fine(v) for k, v in out.items()}

    cfg = OmegaConf.load(REPO / "navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml")
    sf = instantiate(cfg)
    sf.tokens = [t for g in groups for t in g]
    cams = dict(cam_l1=False, cam_l2=False, cam_r1=False, cam_r2=False, cam_b0=False)
    loader = dataloader.SceneLoader(D / "navsim_logs/test", Path(args.sensor_path), sf,
                                    synthetic_sensor_path=D / SPLIT / "sensor_blobs",
                                    synthetic_scenes_path=D / SPLIT / "synthetic_scene_pickles",
                                    sensor_config=SensorConfig(cam_f0=[3], cam_l0=[3], cam_r0=[3], lidar_pc=False, **cams))
    # synthetic scenes have no future frames: show the logged human of the corresponding original scene instead,
    # moved into the synthetic start's frame (where the human drove, seen from the perturbed start)
    sys.path.insert(0, str(REPO / "scripts/evaluation"))
    from run_pdm_score_balanced import synthetic_scene_index  # noqa: E402
    from navsim.common.dataclasses import SceneFilter  # noqa: E402
    syn_meta = {v["initial_token"]: v for v in synthetic_scene_index(D / SPLIT / "synthetic_scene_pickles").values()}

    def human_from_original(tok, scene):
        meta = syn_meta.get(tok)
        if meta is None:
            return None
        try:
            f = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=False,
                            max_scenes=None, log_names=[meta["log_name"]], tokens=[meta["corresponding_original_scene"]])
            orig = dataloader.SceneLoader(D / "navsim_logs/test", Path(args.sensor_path), f,
                                          sensor_config=SensorConfig.build_no_sensors()).get_scene_from_token(
                meta["corresponding_original_scene"])
        except Exception as e:  # noqa: BLE001
            print(f"no original scene for {tok}: {e}")
            return None
        po = np.array(orig.frames[3].ego_status.ego_pose, dtype=float)
        ps = np.array(scene.frames[scene.scene_metadata.num_history_frames - 1].ego_status.ego_pose, dtype=float)
        return pq.to_frame(orig.get_future_trajectory().poses, po, ps)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = [f"# navhard ({args.mode}): original scene and synthetic follow-up(s)", "", "| File | tokens |", "|---|---|"]
    for i, toks in enumerate(groups):
        cols = [("original scene (stage one)", "stage_one"), ("synthetic follow-up, handled", "stage_two"),
                ("synthetic follow-up, failed", "stage_two")]
        if args.mode == "rescue":
            cols = [("original scene (stage one)", "stage_one"), ("synthetic follow-up (stage two)", "stage_two")]
        fig = plt.figure(figsize=(6.2 * len(cols), 10.5))
        gs = fig.add_gridspec(2, len(cols), height_ratios=[0.5, 1.0], hspace=0.10, wspace=0.04)
        for c, (tok, (head, stage)) in enumerate(zip(toks, cols)):
            scene = loader.get_scene_from_token(tok)
            frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
            pl = plans(tok)
            pq.STYLE.update({"argmax": (COLORS["argmax"], "-", 5.2, 26), "shortlist k256": (COLORS["shortlist k256"], "-", 3.4, 27),
                             "shortlist + EC n64": (COLORS["shortlist + EC n64"], "-", 1.6, 28)})
            cam_ax = fig.add_subplot(gs[0, c])
            ax = fig.add_subplot(gs[1, c])
            trajs = dict(pl)
            hum = None
            if stage == "stage_one":
                hum = scene.get_future_trajectory().poses
            elif args.original_human:  # off by default: that path does not start at the perturbed pose
                hum = human_from_original(tok, scene)
            if hum is not None and len(hum) == 8:
                trajs = {"human": hum, **pl}
            pq.draw_bev(ax, scene, trajs, None, lim=pq.bev_extent(trajs))
            pq.draw_camera(cam_ax, frame, {k: v for k, v in trajs.items() if k in ("human", "shortlist + EC n64")})
            same = {k: [j for j in list(pl)[: list(pl).index(k)] if np.abs(pl[j] - pl[k]).max() < 1e-3] for k in pl}
            lines = [f"{RULES[k]}: {fmt(scores[k].loc[tok], stage)}" + (f" (= {RULES[same[k][-1]]})" if same[k] else "")
                     for k in RULES]
            ax.set_title(f"{head}\n" + "\n".join(lines), fontsize=9.5)
        handles = [plt.Line2D([], [], color=COLORS[k], linewidth=w) for k, w in zip(RULES, [5.2, 3.4, 1.6])] + \
                  [plt.Line2D([], [], color="black", linestyle="--", linewidth=1.8)]
        fig.legend(handles, list(RULES.values()) + ["human (logged; original scenes only)"],
                   loc="lower center",
                   ncol=4, fontsize=10, frameon=False)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05)
        f = args.out_dir / f"{i:02d}_{toks[0]}.png"
        fig.savefig(f, dpi=args.dpi)
        plt.close(fig)
        index.append(f"| `{f.name}` | {', '.join(toks)} |")
        print(f, flush=True)
    (args.out_dir / "index.md").write_text("\n".join(index) + "\n")


if __name__ == "__main__":
    main()
