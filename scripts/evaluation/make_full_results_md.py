"""Write docs/full.md: every official evaluation result on disk, one row per output folder, all values.

Sources (latest timestamped CSV of each folder, as the collectors in exp/ use):
  $NAVSIM_EXP_ROOT/pdm_navtest/<name>/<run>/<YYYY.MM.DD.HH.MM.SS>.csv              navtest v2 (EPDMS) and sub-scores
  $NAVSIM_EXP_ROOT/pdm_navtest/<name>/<run>/<...>_v1_comfort.csv                   navtest v1 PDMS with the v1 comfort
                                                                                   term (v1_pdms_official_comfort.py)
  $NAVSIM_EXP_ROOT/pdm_navhard_two_stage/<name>/<run>/<...>.csv                    navhard two-stage EPDMS, per stage
Folders named *_non_reactive_split / *_reactive_split are the split pipeline (GPU predictions + CPU scoring, the
pipeline of every number since 2026-11-08); every other folder (filterfix, filteroff, chunked, cpu4, ...) is listed in
separate tables as history / pipeline checks. Interpretation and which numbers are pre-fix: docs/final_evaluation.md.
  python scripts/evaluation/make_full_results_md.py            # -> docs/full.md
"""
import argparse
import glob
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd

E = Path(os.environ["NAVSIM_EXP_ROOT"])
REPO = Path(__file__).resolve().parents[2]
SUB = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance",
       "ego_progress", "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
AB = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]
TS = r"\d{4}(\.\d{2}){5}"


def latest(folder: Path, suffix: str = ""):
    fs = [f for f in glob.glob(f"{folder}/*/*.csv") if re.search(rf"/{TS}{suffix}\.csv$", f)]
    return max(fs, key=os.path.getmtime) if fs else None


def navtest_row(folder: Path):
    f = latest(folder)
    if f is None:
        return None
    df = pd.read_csv(f)
    avg = df[df.token.astype(str) == "average_all_frames"]
    sc = df[df.token.astype(str) != "average_all_frames"]
    valid = sc[sc.valid.astype(str) == "True"]
    if len(avg) == 0:
        return None
    a = avg.iloc[-1]
    return dict(PDMS=a.score, **{k: a[c] for k, c in zip(AB, SUB) if c in a}, valid=len(valid), n=len(sc),
                zero=100 * (valid.score == 0).mean(), csv=Path(f).name[:19])


def v1_row(folder: Path):
    f = latest(folder, "_v1_comfort")
    if f is None:
        return None
    df = pd.read_csv(f)
    df = df[~df.token.astype(str).str.startswith("average")]
    return dict(v1=df.v1_pdms.mean(), v1hc=df.v1_pdms_history_comfort.mean(), NC=df.no_at_fault_collisions.mean(),
                DAC=df.drivable_area_compliance.mean(), TTC=df.time_to_collision_within_bound.mean(),
                EP=df.ego_progress.mean(), C=df.v1_comfort.mean(), HC=df.history_comfort.mean(), n=len(df),
                zero=100 * (df.v1_pdms == 0).mean(), csv=Path(f).name[:19])


def navhard_row(folder: Path):
    f = latest(folder)
    if f is None:
        return None
    df = pd.read_csv(f)
    if "token" not in df.columns:
        return None
    df = df.set_index("token")
    rows = [r for r in ["extended_pdm_score_combined", "extended_pdm_score_stage_one", "extended_pdm_score_stage_two"]
            if r in df.index]
    if len(rows) < 3:
        return None
    out = dict(comb=df.loc[rows[0], "score"], s1=df.loc[rows[1], "score"], s2=df.loc[rows[2], "score"],
               csv=Path(f).name[:19])
    for st, key in [("stage_one", rows[1]), ("stage_two", rows[2])]:
        out[st] = [df.loc[key, f"{c}_{st}"] if f"{c}_{st}" in df.columns else float("nan") for c in SUB]
    sc = df[~df.index.astype(str).str.startswith("extended_pdm_score")]
    out["n"], out["invalid"] = len(sc), int((sc.valid.astype(str) != "True").sum()) if "valid" in sc else -1
    return out


def f4(x, d=4):
    return "–" if x is None or pd.isna(x) else f"{x:.{d}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(REPO / "docs" / "full.md"))
    args = p.parse_args()
    nt = sorted(d for d in (E / "pdm_navtest").iterdir() if d.is_dir() and d.name != "logs")
    nh = sorted(d for d in (E / "pdm_navhard_two_stage").iterdir() if d.is_dir() and d.name != "logs")
    L = [f"# Full evaluation results (generated {date.today().isoformat()})", "",
         "Every official evaluation on disk, one row per output folder, latest timestamped CSV of the folder. Generated "
         "by `scripts/evaluation/make_full_results_md.py`; do not edit by hand (rerun it). Interpretation, chosen "
         "checkpoints, selection rules and which numbers are pre-fix: [final_evaluation.md](final_evaluation.md).", "",
         "Folder names: `<prediction>_<rule>[_k<K>|_lam<λ>]_<traffic>_split`. `<prediction>` is the model and epoch "
         "(e.g. `rl3_b0.005_ep01`); `__ctrlscorer_` / `__pairscorer_` = scored with another model's heads; "
         "`_own_ec[seq]_<base>_n<N>` = EC rule with the own scorer (seq = sequential); `_ec[seq]_` without `own` = "
         "control scorer. Abbreviations: NC no at-fault collision, DAC drivable area, DDC driving direction, TLC traffic "
         "light, EP ego progress, TTC time to collision, LK lane keeping, HC history comfort, EC two-frame extended "
         "comfort; \"score 0\" = share of valid scenes scored 0; CSV = timestamp of the source file.", ""]
    for title, pred in [("navtest v2 (EPDMS), split pipeline", lambda d: d.name.endswith("_split")),
                        ("navtest v2, other pipelines and checks (history)", lambda d: not d.name.endswith("_split"))]:
        rows = [(d.name, navtest_row(d)) for d in nt if pred(d)]
        rows = [(n, r) for n, r in rows if r is not None]
        L += [f"## {title} ({len(rows)} folders)", "",
              "| Folder | PDMS | " + " | ".join(AB) + " | valid / scenes | score 0 | CSV |",
              "|---|---|" + "---|" * len(AB) + "---|---|---|"]
        for n, r in rows:
            L.append(f"| {n} | **{f4(r['PDMS'])}** | " + " | ".join(f4(r.get(k)) for k in AB) +
                     f" | {r['valid']} / {r['n']} | {r['zero']:.2f}% | {r['csv']} |")
        L.append("")
    rows = [(d.name, v1_row(d)) for d in nt]
    rows = [(n, r) for n, r in rows if r is not None]
    L += [f"## navtest v1 PDMS with the v1 comfort term ({len(rows)} folders)", "",
          "v1 PDMS = NC · DAC · (5 EP + 5 TTC + 2 C) / 12 with C = the v1 comfort term; \"v1 (HC)\" = the earlier proxy "
          "with history comfort. DDC has weight 0 in v1.", "",
          "| Folder | **v1 PDMS (v1 comfort)** | v1 (HC) | NC | DAC | TTC | EP | v1 comfort | HC | scenes | score 0 | CSV |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in rows:
        L.append(f"| {n} | **{f4(r['v1'])}** | {f4(r['v1hc'])} | {f4(r['NC'])} | {f4(r['DAC'])} | {f4(r['TTC'])} | "
                 f"{f4(r['EP'])} | {f4(r['C'])} | {f4(r['HC'])} | {r['n']} | {r['zero']:.2f}% | {r['csv']} |")
    L.append("")
    for title, pred in [("navhard two-stage (EPDMS), split pipeline", lambda d: d.name.endswith("_split")),
                        ("navhard, other pipelines and checks (history)", lambda d: not d.name.endswith("_split"))]:
        rows = [(d.name, navhard_row(d)) for d in nh if pred(d)]
        rows = [(n, r) for n, r in rows if r is not None]
        L += [f"## {title} ({len(rows)} folders)", "",
              "Combined = mean over groups of stage one × stage two (final_evaluation.md, \"How navhard aggregates\").", "",
              "| Folder | **combined** | stage one | stage two | stage one " + " / ".join(AB) + " | stage two " +
              " / ".join(AB) + " | scenes (invalid) | CSV |", "|---|---|---|---|---|---|---|---|"]
        for n, r in rows:
            s1 = " / ".join(f4(x, 3) for x in r["stage_one"])
            s2 = " / ".join(f4(x, 3) for x in r["stage_two"])
            L.append(f"| {n} | **{f4(r['comb'])}** | {f4(r['s1'])} | {f4(r['s2'])} | {s1} | {s2} | "
                     f"{r['n']} ({r['invalid']}) | {r['csv']} |")
        L.append("")
    Path(args.out).write_text("\n".join(L))
    print(f"wrote {args.out}: {len(L)} lines")


if __name__ == "__main__":
    main()
