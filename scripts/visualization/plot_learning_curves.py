"""Learning curves of the vision_llm_lora training stages from the W&B run logs (offline `.wandb` files in each run
folder): validation metrics on navtrain val per validation pass, one figure per stage, one measure per panel (small
multiples, no dual axes). Seeds: thin lines; mean over seeds: bold; dot: the epoch the official evaluation uses (best
val head_all / val argmax by the offline evaluation, final epoch for RL-3).
  stage 1  phase 1 (p1d_total): val ADE, FDE, imitation loss (soft CE), score-prediction error (total MAE)
  stage 2  phase 3, aux vs heads (4 seeds each): val head score, val argmax score; aux runs: BEV mIoU, drivable IoU,
           lane-direction accuracy
  stage 3  policy head (4 seeds each): val argmax score
  stage 4  RL-3, matched aux / heads pairs (4 each): val argmax, shortlist 256, expected reward, KL to the start policy
The val scores in the W&B logs are the training-time ("lenient") proxy; the offline checkpoint evaluation (strict) and
the official benchmarks are in docs/checkpoint_evaluation.md and docs/final_evaluation.md.
  python scripts/visualization/plot_learning_curves.py --out-dir $NAVSIM_EXP_ROOT/qualitative_v2/learning_curves
"""
import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from wandb.proto import wandb_internal_pb2 as pb  # noqa: E402
from wandb.sdk.internal import datastore  # noqa: E402

E = Path(os.environ["NAVSIM_EXP_ROOT"])
AUX, HEADS = "#2a78d6", "#eb6834"  # reference categorical slots 1 and 2 (blue, orange)
INK, MUTED = "#0b0b0b", "#52514e"


def run_dir(exp):
    for d in sorted(glob.glob(f"{E}/{exp}/*/"), reverse=True):
        if os.path.isdir(f"{d}/checkpoints") and glob.glob(f"{d}/wandb/latest-run/*.wandb"):
            return d
    raise FileNotFoundError(exp)


def history(exp):
    """validation rows (dicts) of a run, in logging order"""
    ds = datastore.DataStore()
    ds.open_for_scan(glob.glob(f"{run_dir(exp)}/wandb/latest-run/*.wandb")[0])
    rows = []
    while (d := ds.scan_data()) is not None:
        r = pb.Record()
        r.ParseFromString(d)
        if r.WhichOneof("record_type") == "history":
            h = {i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in r.history.item}
            if any(k.endswith("_epoch") and "val" in k for k in h):
                rows.append(h)
    return rows


def series(rows, key, x="epoch"):
    pts = [(r.get(x), r[key]) for r in rows if key in r and r.get(x) is not None and r[key] is not None]
    return np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)


def style(ax, title, ylabel=None, xlabel="epoch"):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")
    ax.tick_params(colors=MUTED, labelsize=8)


def family(ax, curves, color, label, chosen=None):
    """thin line per seed, bold mean (on the common x), dot at the chosen epoch of each seed"""
    xs = sorted(set.intersection(*[set(np.round(x, 3)) for x, _ in curves])) if curves else []
    for i, (x, y) in enumerate(curves):
        ax.plot(x, y, color=color, linewidth=1.0, alpha=0.45)
        if chosen is not None and chosen[i] is not None:
            m = np.isclose(x, chosen[i])
            ax.plot(x[m], y[m], "o", color=color, markersize=6, markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    if xs:
        mean = [np.mean([y[np.isclose(x, v)][0] for x, y in curves]) for v in xs]
        ax.plot(xs, mean, color=color, linewidth=2.2, label=label)


def stage1(out):
    rows = history("p1d_total")
    keys = [("stabilize/val/ade_loss_epoch", "ADE (m)"), ("stabilize/val/fde_loss_epoch", "FDE (m)"),
            ("val/loss_terms/soft_ce_epoch", "imitation loss (soft CE)"),
            ("val/pdm_scores/total_mae_epoch", "score-prediction error (MAE)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, (k, t) in zip(axes, keys):
        x, y = series(rows, k)
        ax.plot(x, y, color=AUX, linewidth=2.0)
        m = np.isclose(x, 10)
        ax.plot(x[m], y[m], "o", color=AUX, markersize=6, markeredgecolor="white", markeredgewidth=1.2)
        style(ax, t)
    fig.tight_layout()
    fig.savefig(out / "stage1_phase1.png", dpi=170)
    plt.close(fig)


def stage2(out):
    aux = ["p3_aux", "p3_aux_s1", "p3_aux_s2", "p3_aux_s3"]
    heads = ["p3_heads", "p3_heads_s1", "p3_heads_s2", "p3_heads_s3"]
    ch_aux, ch_heads = [5, 6, 5, 6], [5, 6, 5, 4]  # best val head_all (offline evaluation)
    ra, rh = [history(e) for e in aux], [history(e) for e in heads]
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.6))
    for ax, (k, t) in zip(axes[:2], [("val/pdm_scores/head_score_epoch", "heads selection (val score)"),
                                     ("val/pdm_scores/argmax_score_epoch", "policy argmax (val score)")]):
        family(ax, [series(r, k) for r in ra], AUX, "with auxiliary losses", ch_aux)
        family(ax, [series(r, k) for r in rh], HEADS, "without auxiliary losses", ch_heads)
        style(ax, t)
    for ax, (k, t) in zip(axes[2:], [("val/aux/bev_miou_epoch", "aux: BEV semantics mIoU"),
                                     ("val/aux/drivable_iou_epoch", "aux: drivable-area IoU"),
                                     ("val/aux/lane_heading_acc_epoch", "aux: lane-direction accuracy")]):
        family(ax, [series(r, k) for r in ra], AUX, "with auxiliary losses", ch_aux)
        style(ax, t)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "stage2_phase3.png", dpi=170)
    plt.close(fig)


def stage3(out):
    aux = ["p3_aux_policy", "p3_aux_policy_s1", "p3_aux_policy_s2", "p3_aux_policy_s3"]
    heads = ["p3_heads_policy", "p3_heads_policy_s1", "p3_heads_policy_s2", "p3_heads_policy_s3"]
    ch_aux, ch_heads = [3, 1, 2, 2], [2, 3, 2, 3]  # best val argmax (offline evaluation)
    k = "val/pdm_scores/argmax_score_epoch"
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    family(ax, [series(history(e), k) for e in aux], AUX, "with auxiliary losses", ch_aux)
    family(ax, [series(history(e), k) for e in heads], HEADS, "without auxiliary losses", ch_heads)
    style(ax, "policy head: argmax (val score)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "stage3_policy.png", dpi=170)
    plt.close(fig)


def stage4(out):
    aux = ["rl3_b0.005"] + [f"rl3_b0.005_pipe_s{s}" for s in (1, 2, 3)]
    heads = ["rl3_heads_b0.005"] + [f"rl3_heads_b0.005_pipe_s{s}" for s in (1, 2, 3)]
    keys = [("val/rl/argmax_epoch", "policy argmax (val score)"), ("val/rl/shortlist_256_epoch", "shortlist 256 (val score)"),
            ("val/rl/expected_reward_epoch", "expected reward E_π[r]"), ("val/rl/kl_epoch", "KL to the start policy")]

    def curves(exps, k):
        out_ = []
        for e in exps:
            x, y = series(history(e), k, x="trainer/global_step")
            out_.append((np.round(x / x.max() * 2.0, 2), y))  # 4 val passes per run -> 0.5, 1.0, 1.5, 2.0 epochs
        return out_

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (k, t) in zip(axes, keys):
        family(ax, curves(aux, k), AUX, "start: trained with auxiliary losses", [2.0] * 4)
        family(ax, curves(heads, k), HEADS, "start: trained without auxiliary losses", [2.0] * 4)
        style(ax, t)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "stage4_rl.png", dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for f in (stage1, stage2, stage3, stage4):
        f(args.out_dir)
        print(f.__name__, "done", flush=True)


if __name__ == "__main__":
    main()
