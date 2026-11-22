"""One navtest scene through the training stages (the per-stage table of docs/final_evaluation.md, section "Stage 1
official evaluation and one row per training stage"): three front cameras with the human and the final plan, then one
BEV panel per stage at the same scale, each with the logged human for reference.
  stage 1  phase 1 (p1d_total ep 10): policy argmax, and the sub-score heads' best entry
  stage 2  phase 3 with aux losses (p3_aux ep 5): heads selection
  stage 3  policy head on top (p3_aux_policy ep 3): shortlist k 256
  stage 4  RL fine-tuning (rl3_b0.005 ep 1): mixture + consistency rule
Scenes (fixed rules on the official per-scene navtest v2 EPDMS, at most one per log): "progress" = the stage 1 best of
the two <= 0.5, stage 4 >= 0.95 and stages 2-4 never below the previous stage (random, --seed); "typical" = random
scenes where every stage scores > 0.5.
  python scripts/visualization/plot_stages.py --sensor-path <mount> --out-dir $NAVSIM_EXP_ROOT/qualitative_v2/stages
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pq", HERE / "plot_qualitative.py")
pq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pq)
plt = pq.plt
from navsim.agents.vision_llm_lora.vision_llm_lora_agent import select_trajectory  # noqa: E402
from navsim.agents.vision_llm_lora.vision_llm_lora_precomputed_agent import load_predictions  # noqa: E402

STAGES = [  # key, title, prediction folder, selection, per-scene score folder, colour of the rule (as in every figure)
    ("s1_policy", "Stage 1, frozen backbone: policy", "p1d_total_ep10", "argmax", "p1d_total_ep10_argmax", "#1f77b4"),
    ("s1_heads", "Stage 1, frozen backbone: heads", "p1d_total_ep10", "score_heads", "p1d_total_ep10_score_heads", "#ff7f0e"),
    ("s2", "Stage 2, LoRA fine-tuning: heads", "p3_aux_ep05", "score_heads", "p3_aux_ep05_score_heads", "#ff7f0e"),
    ("s3", "Stage 3, policy head: shortlist", "p3_aux_policy_ep03", "shortlist", "p3_aux_policy_ep03_shortlist_k256",
     "#2ca02c"),
    ("s4", "Stage 4, RL: mixture + consistency", None, None, "rl3_b0.005_ep01_own_ecseq_mix10_n64_argmax", "#d62728"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensor-path", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    scores = {k: pq.per_scene(sc) for k, _, _, _, sc, _ in STAGES}
    human = pq.per_scene("human_agent_non_reactive", suffix="")
    idx = human.index
    for v in scores.values():
        idx = idx.intersection(v.index)
    sc = {k: v.loc[idx, "score"] for k, v in scores.items()}
    s1 = np.maximum(sc["s1_policy"], sc["s1_heads"])
    prog = (s1 <= 0.5) & (sc["s4"] >= 0.95) & (sc["s2"] >= s1) & (sc["s3"] >= sc["s2"]) & (sc["s4"] >= sc["s3"])
    typ = (s1 > 0.5) & (sc["s2"] > 0.5) & (sc["s3"] > 0.5) & (sc["s4"] > 0.5)
    rng = np.random.default_rng(args.seed)
    loader = pq.scene_loader(list(idx), Path(args.sensor_path))
    snap = pq.CollisionSnapshot(pq.E / "metric_cache_navtest")
    log_of = lambda t: loader.scene_frames_dicts[t][3]["log_name"]
    sel = {"progress": pq.first_per_log(rng.permutation(np.array(idx[prog])), log_of, args.n),
           "typical": pq.first_per_log(rng.permutation(np.array(idx[typ])), log_of, args.n)}
    preds = {f: load_predictions(pq.E / "predictions" / f / "navtest") for f in
             {s[2] for s in STAGES if s[2]}}
    ec = load_predictions(pq.E / "predictions" / "rl3_b0.005_ep01_own_ecseq_mix10_n64" / "navtest")
    cb_dir = pq.REPO / "navsim/agents/vision_llm_lora/codebook"
    cb8, cb40 = np.load(cb_dir / "16384_8step.npy"), np.load(cb_dir / "16384.npy")
    key = {e.tobytes(): k for k, e in enumerate(cb8)}

    def fine(t):
        b = np.ascontiguousarray(t, dtype=np.float32).tobytes()
        return cb40[key[b]].astype(float) if b in key else np.asarray(t, dtype=float)

    def plan(stage, tok):
        k, _, folder, selection, _, _ = stage
        if folder is None:
            return fine(np.array(ec["arrays"]["trajectory"][ec["row"][tok]]))
        pr = preds[folder]
        r = pr["row"][tok]
        x = {"trajectory": torch.from_numpy(np.array(pr["arrays"]["trajectory"][r: r + 1])),
             "trajectory_logits": pq.bf16(pr, "trajectory_logits", r),
             "pdm_score_logits": pq.bf16(pr, "pdm_score_logits", r)}
        return fine(select_trajectory(x, selection, pr["codebook"].float(), shortlist_size=256)[0].numpy())

    index = ["# One scene through the training stages", "", __doc__.split("\n\n")[0], "",
             "| Category | File | Token | human | " + " | ".join(s[1] for s in STAGES) + " |",
             "|---|---|---|---|" + "---|" * len(STAGES)]
    for cat, toks in sel.items():
        (args.out_dir / cat).mkdir(parents=True, exist_ok=True)
        for i, tok in enumerate(toks):
            scene = loader.get_scene_from_token(tok)
            frame = scene.frames[3]
            hum = scene.get_future_trajectory().poses
            plans = {s[0]: plan(s, tok) for s in STAGES}
            for s in STAGES:
                pq.STYLE[s[0]] = (s[5], "-", 2.2, 28)
            snaps = {}
            for st in STAGES:  # NAVSIM's collision for every NC 0 plan (recomputed NC must match)
                if scores[st[0]].loc[tok, "no_at_fault_collisions"] == 0:
                    snaps[st[0]] = snap(tok, pq.poses8(plans[st[0]]))
                    assert snaps[st[0]]["nc"] == 0 and snaps[st[0]]["t"] is not None, (tok, st[0], snaps[st[0]])
            geo = {f"_{k}{j}": g for k, sn in snaps.items() for j, g in enumerate([sn["ego"], *sn["agents"]])}
            lim = pq.bev_extent({"human": hum, **plans, **geo})
            # inches: camera labels, cameras, title, maps, title, maps
            rows = [pq.CAM_LABEL_H, pq.CAM_H, 0.40, pq.MAP_W, 0.40, pq.MAP_W]
            fig = plt.figure(figsize=(pq.PRINT_W, sum(rows) / 0.98))
            gs = fig.add_gridspec(len(rows), 3, height_ratios=rows, hspace=0, wspace=0.04)
            pq.three_cameras(pq.camera_row(fig, gs[1, :]), frame, {"human": hum, "s4": plans["s4"]})
            for j, s in enumerate(STAGES):
                ax = fig.add_subplot(gs[3 + 2 * (j // 3), j % 3])
                pq.draw_bev(ax, scene, {"human": hum, s[0]: plans[s[0]]}, None, lim=lim)
                pq.draw_future_agents(ax, scene)
                row = scores[s[0]].loc[tok]
                if s[0] in snaps:  # NAVSIM's collision: ego, agent hit, time
                    pq.draw_collision(ax, snaps[s[0]], s[5])
                ax.set_title(f"{s[1]}\nEPDMS {pq.fmt_scores(row)}", color="#b00000" if row["score"] < 0.5 else "black", fontsize=7)
            lax = fig.add_subplot(gs[5, 2])
            lax.axis("off")
            lax.legend([plt.Line2D([], [], color="black", linestyle="--", linewidth=1.4)] +
                       pq.map_legend_handles(footprint=bool(snaps)), [f"human (logged): EPDMS {pq.fmt_scores(human.loc[tok])}"] +
                       [h.get_label() for h in pq.map_legend_handles(footprint=bool(snaps))], loc="center", ncol=1, frameon=False,
                       handlelength=1.6, labelspacing=0.3)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.985, bottom=0.005)
            f = args.out_dir / cat / f"{i:02d}_{tok}.png"
            fig.savefig(f, dpi=args.dpi)
            plt.close(fig)
            index.append(f"| {cat} | `{cat}/{f.name}` | {tok} | {human.loc[tok, 'score']:.2f} | " +
                         " | ".join(f"{scores[s[0]].loc[tok, 'score']:.2f}" for s in STAGES) + " |")
            print(f, flush=True)
    (args.out_dir / "index.md").write_text("\n".join(index) + "\n")


if __name__ == "__main__":
    main()
