"""Official benchmark scores per training stage (the per-stage table of docs/final_evaluation.md, finding 70) as small
multiples: navtest v2 EPDMS, navtest v1 PDMS (v1 comfort), navhard EPDMS; x = training stage, one line for the path
with auxiliary losses and one without (mean +- SD over 4 seeds; stage 1 is the shared start, one run), and the
imitation-only phase 3 (same start) as a reference point. Best evaluated rule per stage, as in finding 70:
  stage 1 phase 1: policy argmax (v2), heads (v1, navhard)     stage 2 phase 3: heads selection
  stage 3 policy head: shortlist k 256 (v1: shortlist_v1)      stage 4 RL: mixture + consistency (navtest, v1:
  mixture_v1), shortlist + consistency (navhard)
Latest timestamped CSV per output folder.
  python scripts/visualization/plot_stage_scores.py --out <png>
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
AUX, HEADS, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def latest(folder, suffix=""):
    fs = [f for f in glob.glob(f"{folder}/*/*.csv") if re.search(rf"/\d{{4}}(\.\d{{2}}){{5}}{suffix}\.csv$", f)]
    return max(fs, key=os.path.getmtime)


def v2(name):
    d = pd.read_csv(latest(E / "pdm_navtest" / f"{name}_non_reactive_split"))
    return float(d[d.token.astype(str) == "average_all_frames"].iloc[-1].score)


def v1(name):
    d = pd.read_csv(latest(E / "pdm_navtest" / f"{name}_non_reactive_split", "_v1_comfort"))
    return float(d[~d.token.astype(str).str.startswith("average")].v1_pdms.mean())


def hard(name):
    try:
        f = latest(E / "pdm_navhard_two_stage" / f"{name}_reactive_split")
    except ValueError:  # p3_aux_policy_ep03 shortlist k256: 2026-11-08 re-evaluation folder (as collect_seed_study.py)
        f = latest(E / "pdm_navhard_two_stage" / f"{name}_reactive_split_cpu4_fastload")
    d = pd.read_csv(f).set_index("token")
    return float(d.loc["extended_pdm_score_combined", "score"])


S1 = {"v2": v2("p1d_total_ep10_argmax"), "v1": v1("p1d_total_ep10_score_heads_v1"),
      "hard": hard("p1d_total_ep10_score_heads")}
PATHS = {
    "aux": {2: ["p3_aux_ep05", "p3_aux_s1_ep06", "p3_aux_s2_ep05", "p3_aux_s3_ep06"],
            3: ["p3_aux_policy_ep03", "p3_aux_policy_s1_ep01", "p3_aux_policy_s2_ep02", "p3_aux_policy_s3_ep02"],
            4: ["rl3_b0.005_ep01"] + [f"rl3_b0.005_pipe_s{s}_ep01" for s in (1, 2, 3)]},
    "heads": {2: ["p3_heads_ep05", "p3_heads_s1_ep06", "p3_heads_s2_ep05", "p3_heads_s3_ep04"],
              3: ["p3_heads_policy_ep02", "p3_heads_policy_s1_ep03", "p3_heads_policy_s2_ep02",
                  "p3_heads_policy_s3_ep03"],
              4: ["rl3_heads_b0.005_ep01"] + [f"rl3_heads_b0.005_pipe_s{s}_ep01" for s in (1, 2, 3)]},
}
RULE = {2: {"v2": lambda n: v2(f"{n}_score_heads"), "v1": lambda n: v1(f"{n}_score_heads_v1"),
            "hard": lambda n: hard(f"{n}_score_heads")},
        3: {"v2": lambda n: v2(f"{n}_shortlist_k256"), "v1": lambda n: v1(f"{n}_shortlist_v1_k256"),
            "hard": lambda n: hard(f"{n}_shortlist_k256")},
        4: {"v2": lambda n: v2(f"{n}_own_ecseq_mix10_n64_argmax"), "v1": lambda n: v1(f"{n}_mixture_v1_lam10"),
            "hard": lambda n: hard(f"{n}_own_ec_short256_n64_argmax")}}
IMIT = {"v2": v2("p3_imitation_ep07_argmax"), "v1": v1("p3_imitation_ep07_argmax"), "hard": hard("p3_imitation_ep07_argmax")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    bench = [("v2", "navtest v2 (EPDMS)"), ("v1", "navtest v1 (PDMS)"), ("hard", "navhard (EPDMS)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    rows = []
    for ax, (b, title) in zip(axes, bench):
        for path, col, lab, dx in [("aux", AUX, "with auxiliary losses", -0.06), ("heads", HEADS, "without auxiliary losses", 0.06)]:
            m, sd = [S1[b]], [0.0]
            for st in (2, 3, 4):
                vals = np.array([RULE[st][b](n) for n in PATHS[path][st]])
                m.append(vals.mean())
                sd.append(vals.std(ddof=1))
                rows.append((b, path, st, vals.mean(), vals.std(ddof=1)))
            x = np.array([1, 2, 3, 4]) + np.r_[0, [dx] * 3]
            ax.errorbar(x, m, yerr=sd, color=col, linewidth=2.0, marker="o", markersize=6, capsize=3,
                        markeredgecolor="white", label=lab)
        ax.plot([2], [IMIT[b]], marker="D", markersize=7, color="white", markeredgecolor=MUTED, markeredgewidth=1.5,
                linestyle="", label="imitation only (phase 3, same start)")
        ax.set_xticks([1, 2, 3, 4], ["1\nphase 1", "2\nphase 3", "3\npolicy head", "4\nRL"], fontsize=8.5)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.grid(True, axis="y", color="#e6e5e1", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(args.out)
    print(f"stage 1: {S1}; imitation: {IMIT}")
    for r in rows:
        print(f"{r[0]:4s} {r[1]:5s} stage {r[2]}: {r[3]:.4f} ± {r[4]:.4f}")


if __name__ == "__main__":
    main()
