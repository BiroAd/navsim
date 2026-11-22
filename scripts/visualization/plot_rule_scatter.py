"""navtest vs navhard for the selection rules of the RL model over the 4 full-pipeline runs (the runs of the stage chart,
plot_stage_scores.py: rl3_b0.005 and rl3_b0.005_pipe_s1-3, each from its own stage 2 / stage 3 seed and scored with
its own start model's heads): x = navtest v2 EPDMS, y = navhard EPDMS, one point per rule (mean over the 4 runs) with
error bars = SD over the runs. "Heads only" = the 4 runs' scorers alone (stage 2 heads selection, p3_aux ep 5 and
p3_aux_s1-3), drawn hollow. Colours as in the trajectory figures. Latest timestamped CSV per output folder.
  python scripts/visualization/plot_rule_scatter.py --out <png>
"""
import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

E = Path(os.environ["NAVSIM_EXP_ROOT"])
SEEDS = ["rl3_b0.005_ep01"] + [f"rl3_b0.005_pipe_s{s}_ep01" for s in (1, 2, 3)]
RULES = [  # label, navtest suffix, navhard suffix, colour (plot_qualitative STYLE)
    ("Policy (argmax)", "argmax", "argmax", "#1f77b4"),
    ("Shortlist (k = 256)", "shortlist_k256", "shortlist_k256", "#2ca02c"),
    ("Mixture (λ = 10)", "mixture_lam10", "mixture_lam10", "#9467bd"),
    ("Mixture + consistency", "own_ecseq_mix10_n64_argmax", "own_ec_mix10_n64_argmax", "#d62728"),
    ("Shortlist + consistency", "own_ecseq_short256_n64_argmax", "own_ec_short256_n64_argmax", "#006d2c"),
]
HEADS = ("Heads only (scorer)", ["p3_aux_ep05", "p3_aux_s1_ep06", "p3_aux_s2_ep05", "p3_aux_s3_ep06"], "#ff7f0e")


def latest(folder):
    fs = [f for f in glob.glob(f"{folder}/*/*.csv") if re.search(r"/\d{4}(\.\d{2}){5}\.csv$", f)]
    return max(fs, key=os.path.getmtime) if fs else None


def navtest(name):
    f = latest(E / "pdm_navtest" / f"{name}_non_reactive_split")
    if f is None:
        return np.nan
    d = pd.read_csv(f)
    return float(d[d.token.astype(str) == "average_all_frames"].iloc[-1].score)


def navhard(name):
    f = latest(E / "pdm_navhard_two_stage" / f"{name}_reactive_split")
    if f is None:
        return np.nan
    return float(pd.read_csv(f).set_index("token").loc["extended_pdm_score_combined", "score"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for lab, st, sh, col in RULES:
        x = np.array([navtest(f"{s}_{st}") for s in SEEDS])
        y = np.array([navhard(f"{s}_{sh}") for s in SEEDS])
        if np.isnan(y).any() or np.isnan(x).any():
            print(f"{lab}: missing values {x} {y}")
        ax.errorbar(np.nanmean(x), np.nanmean(y), xerr=np.nanstd(x, ddof=1), yerr=np.nanstd(y, ddof=1), color=col,
                    marker="o", markersize=4.5, capsize=2, elinewidth=0.9, markeredgecolor="white", markeredgewidth=0.6,
                    label=lab)
        print(f"{lab}: navtest {np.nanmean(x):.4f} ± {np.nanstd(x, ddof=1):.4f}, navhard {np.nanmean(y):.4f} ± "
              f"{np.nanstd(y, ddof=1):.4f} (n {np.sum(~np.isnan(y))})")
    hxs = np.array([navtest(f"{h}_score_heads") for h in HEADS[1]])
    hys = np.array([navhard(f"{h}_score_heads") for h in HEADS[1]])
    hx, hy = hxs.mean(), hys.mean()
    ax.errorbar(hx, hy, xerr=hxs.std(ddof=1), yerr=hys.std(ddof=1), marker="o", markersize=5, color=HEADS[2],
                markerfacecolor="white", markeredgewidth=1.3, capsize=2, elinewidth=0.9, linestyle="", label=HEADS[0])
    h, l = ax.get_legend_handles_labels()
    names = [RULES[0][0], HEADS[0]] + [r[0] for r in RULES[1:]]  # rule order of the trajectory figures
    order = [l.index(n) for n in names]
    ax.legend([h[i] for i in order], names, loc="upper left", fontsize=7, frameon=False, handletextpad=0.4,
              borderaxespad=0.2)  # a legend, not point labels: labels would cross the error bars
    print(f"{HEADS[0]}: navtest {hx:.4f} ± {hxs.std(ddof=1):.4f}, navhard {hy:.4f} ± {hys.std(ddof=1):.4f}")
    ax.set_xlabel("navtest v2 EPDMS", fontsize=8, color="#52514e")
    ax.set_ylabel("navhard EPDMS", fontsize=8, color="#52514e")
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=7)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(args.out)


if __name__ == "__main__":
    main()
