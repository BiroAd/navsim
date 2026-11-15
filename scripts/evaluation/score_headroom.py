"""Per-metric headroom of a scored run: how much the final score rises if one sub-score were perfect.

Three subcommands:

  start-points  read the ego start position (rear axle) of every navhard_two_stage scene from the metric cache and
                save it as a CSV (once; ~0.4 s per scene, so use --workers).
  navtest       one-stage CSV (run_pdm_score_one_stage.py), optionally with the human agent's CSV for "human level".
  navhard       two-stage run folder (run_pdm_score.py with vision_llm_lora_precomputed_agent). The stage-two weights
                are not saved in the CSV; they depend only on the planned stage-one endpoint and the stage-two start
                points (SceneAggregator.calculate_pseudo_closed_loop_weights), so they are rebuilt exactly by replaying
                the run's selection rule on the saved predictions. The rebuilt stage one / stage two / combined scores
                are checked against the CSV summary rows before any headroom is reported.

The sub-scores in the CSVs are already after the human-penalty filter of that run, so "perfect" means perfect under the
run's own filter setting.
"""

from __future__ import annotations

import argparse
import glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

MULT = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance"]
WEIGHTED = {"ego_progress": 5, "time_to_collision_within_bound": 5, "lane_keeping": 2, "history_comfort": 2,
            "two_frame_extended_comfort": 2}
METRICS = MULT + list(WEIGHTED)
SHORT = dict(zip(METRICS, ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]))
SUMMARY_TOKENS = {"average_all_frames", "extended_pdm_score_stage_one", "extended_pdm_score_stage_two",
                  "extended_pdm_score_combined"}
SIGMA_SQUARED = 0.1  # SceneAggregator.sigma_squared
NAVHARD_SPLIT = "navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml"


def epdms(df: pd.DataFrame, sfx: str = "") -> np.ndarray:
    """EPDMS per row; EC drops out of the weights where it is NaN (as in the one-stage script)."""
    mult = np.prod([df[c + sfx].to_numpy(float) for c in MULT], axis=0)
    values = {c: df[c + sfx].to_numpy(float) for c in WEIGHTED}
    num = sum(w * np.nan_to_num(values[c]) for c, w in WEIGHTED.items())
    den = sum(w * ~np.isnan(values[c]) for c, w in WEIGHTED.items())
    return mult * num / den


def perfect(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = np.where(np.isnan(df[c].to_numpy(float)), np.nan, 1.0)
    return out


def read_csv(pattern: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(glob.glob(pattern))
    if len(paths) != 1:
        raise SystemExit(f"{pattern}: expected one CSV, found {paths}")
    df = pd.read_csv(paths[0], index_col=0)
    summary = df[df.token.isin(SUMMARY_TOKENS)].set_index("token")
    return df[~df.token.isin(SUMMARY_TOKENS)].reset_index(drop=True), summary


# ---------------------------------------------------------------- start points
def _start_point(args):
    cache_dir, tokens = args
    from navsim.common.dataloader import MetricCacheLoader

    loader = MetricCacheLoader(Path(cache_dir))
    rows = []
    for token in tokens:
        axle = loader.get_from_token(token).ego_state.rear_axle
        rows.append((token, axle.x, axle.y, axle.heading))
    return rows


def cmd_start_points(args):
    from navsim.common.dataloader import MetricCacheLoader

    tokens = sorted(MetricCacheLoader(Path(args.metric_cache)).tokens)
    chunks = [(args.metric_cache, tokens[i :: args.workers * 8]) for i in range(args.workers * 8)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for part in pool.map(_start_point, chunks) for r in part]
    out = pd.DataFrame(rows, columns=["token", "start_point_x", "start_point_y", "start_point_heading"]).sort_values("token")
    out.to_csv(args.out, index=False)
    print(f"{len(out)} start points -> {args.out}")


# ---------------------------------------------------------------- navtest
def cmd_navtest(args):
    df, _ = read_csv(args.csv)
    base = epdms(df)
    if args.check and not np.allclose(base, df.score.to_numpy(float)):
        raise SystemExit("recomputed EPDMS differs from the CSV score column")
    human = None
    if args.human_csv:
        human, _ = read_csv(args.human_csv)
        human = human.set_index("token").loc[df.token].reset_index()
    b = base.mean()
    print(f"navtest {args.csv}\n  EPDMS {b:.4f} (lost {1 - b:.4f}); scenes scored 0: {np.mean(base == 0) * 100:.1f}%")
    print("  metric  mean   failing  +if perfect" + ("  +at human level" if human is not None else ""))
    for c in METRICS:
        v = df[c].to_numpy(float)
        line = f"  {SHORT[c]:4s}  {np.nanmean(v):.3f}  {np.mean(v[~np.isnan(v)] < 1) * 100:5.1f}%   +{epdms(perfect(df, [c])).mean() - b:.4f}"
        if human is not None:
            hl = df.copy()
            hl[c] = np.fmax(v, human[c].to_numpy(float))
            line += f"      +{epdms(hl).mean() - b:.4f}"
        print(line)
    for name, cols in (("all multipliers", MULT), ("all weighted", list(WEIGHTED))):
        print(f"  {name}: +{epdms(perfect(df, cols)).mean() - b:.4f}")


# ---------------------------------------------------------------- navhard
def build_agent(run_dir: Path):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from navsim.agents.vision_llm_lora.vision_llm_lora_precomputed_agent import VisionLLMLoRAPrecomputedAgent

    overrides = yaml.safe_load((run_dir / "code/hydra/overrides.yaml").read_text())
    kv = dict(o.split("=", 1) for o in overrides)
    if kv.get("agent") != "vision_llm_lora_precomputed_agent":
        raise SystemExit(f"{run_dir}: agent={kv.get('agent')}; only vision_llm_lora_precomputed_agent runs are supported")
    kwargs = {}
    for key, value in kv.items():
        if key.startswith("agent."):
            value = value.strip("'\"")
            for cast in (int, float):
                try:
                    value = cast(value)
                    break
                except ValueError:
                    pass
            kwargs[key[len("agent."):]] = None if value == "null" else value
    agent = VisionLLMLoRAPrecomputedAgent(TrajectorySampling(time_horizon=4, interval_length=0.5), **kwargs)
    agent.initialize()
    return agent, kv


def stage_two_weights(mapping, endpoints: dict, start: pd.DataFrame) -> dict:
    """token -> weight, replaying create_scene_aggregators (later groups overwrite earlier ones, as DataFrame.update)."""
    weights = {}
    for now, prev, pairs in mapping:
        weights[now] = 1.0
        weights[prev] = 1.0
        for first, idx in ((now, 0), (prev, 1)):
            tokens = [p[idx] for p in pairs]
            ex, ey = endpoints[first]
            d2 = (start.loc[tokens, "start_point_x"].to_numpy() - ex) ** 2 + (start.loc[tokens, "start_point_y"].to_numpy() - ey) ** 2
            w = np.exp(-d2 / (2 * SIGMA_SQUARED))
            total = w.sum()
            w = np.full(len(tokens), 1.0 / len(tokens)) if (np.isclose(total, 0.0) or np.isnan(total)) else w / total
            weights.update(zip(tokens, w))
    return weights


def combined_score(per_token: pd.Series, weights: dict, mapping):
    """calculate_individual_mapping_scores on the total score: (combined, stage one, stage two)."""
    groups, s1, s2 = [], [], []
    for now, prev, pairs in mapping:
        pair_scores = []
        for first, idx in ((now, 0), (prev, 1)):
            tokens = [p[idx] for p in pairs]
            w = np.array([weights[t] for t in tokens])
            a = per_token[first]
            b = (per_token.loc[tokens].to_numpy() * w).sum() / w.sum() if w.sum() > 0 else np.nan
            s1.append(a)
            s2.append(b)
            pair_scores.append(a * b)
        groups.append(np.mean(pair_scores))
    return np.nanmean(groups), np.nanmean(s1), np.nanmean(s2)


def per_token_score(df: pd.DataFrame) -> pd.Series:
    s1, s2 = epdms(df, "_stage_one"), epdms(df, "_stage_two")
    return pd.Series(np.where(np.isnan(s1), s2, s1), index=df.token)


def cmd_navhard(args):
    root = Path(__file__).resolve().parents[2]
    mapping = yaml.safe_load((root / NAVHARD_SPLIT).read_text())["reactive_all_mapping"]
    start = pd.read_csv(args.start_points).set_index("token")
    for run in args.runs:
        run_dir = Path(sorted(glob.glob(f"{run}/2*/"))[-1]) if not (Path(run) / "code").exists() else Path(run)
        df, summary = read_csv(f"{run_dir}/2*.csv")
        agent, _ = build_agent(run_dir)
        stage_one = df.token[df["score"].notna() & df["no_at_fault_collisions_stage_one"].notna()].tolist()
        from nuplan.common.actor_state.state_representation import StateSE2
        from nuplan.common.geometry.convert import relative_to_absolute_poses

        endpoints = {}  # as run_pdm_score.py: endpoint_x / endpoint_y of the planned stage-one trajectory
        for token in stage_one:
            traj = agent.compute_trajectory(None, SimpleNamespace(scene_metadata=SimpleNamespace(initial_token=token)))
            end = StateSE2(*(float(v) for v in traj.poses[-1, :3]))
            origin = StateSE2(*start.loc[token, ["start_point_x", "start_point_y", "start_point_heading"]].astype(float))
            absolute = relative_to_absolute_poses(origin, [end])[0]
            endpoints[token] = (absolute.x, absolute.y)
        weights = stage_two_weights(mapping, endpoints, start)
        pd.Series(weights, name="weight").rename_axis("token").to_csv(run_dir / "stage_two_weights.csv")

        score = per_token_score(df)
        if not np.allclose(score.to_numpy(), df.score.to_numpy(float), equal_nan=True):
            raise SystemExit(f"{run_dir}: recomputed per-scene EPDMS differs from the CSV score column")
        base = combined_score(score, weights, mapping)
        official = tuple(summary.at[t, "score"] for t in
                         ("extended_pdm_score_combined", "extended_pdm_score_stage_one", "extended_pdm_score_stage_two"))
        ok = np.allclose(base, official, atol=1e-9)
        print(f"\nnavhard {run_dir}\n  rebuilt combined / stage one / stage two {base[0]:.6f} / {base[1]:.6f} / {base[2]:.6f}"
              f"\n  CSV summary rows                       {official[0]:.6f} / {official[1]:.6f} / {official[2]:.6f}"
              f"  -> {'exact' if ok else 'MISMATCH'}")
        if not ok:
            continue
        print("  metric  stage one: mean / failing   stage two: mean / failing   +combined if perfect (both stages; "
              "stage one only; stage two only)")
        for c in METRICS:
            r = []
            for sfx in (("_stage_one", "_stage_two"), ("_stage_one",), ("_stage_two",)):
                r.append(combined_score(per_token_score(perfect(df, [c + s for s in sfx])), weights, mapping)[0] - base[0])
            v1 = df[c + "_stage_one"].dropna().to_numpy()
            v2 = df[c + "_stage_two"].dropna().to_numpy()
            m1, m2 = summary.at["extended_pdm_score_stage_one", c + "_stage_one"], summary.at["extended_pdm_score_stage_two", c + "_stage_two"]
            print(f"  {SHORT[c]:4s}    {m1:.3f} / {np.mean(v1 < 1) * 100:4.1f}%              {m2:.3f} / {np.mean(v2 < 1) * 100:4.1f}%"
                  f"            +{r[0]:.4f}  (+{r[1]:.4f}; +{r[2]:.4f})")
        for name, cols in (("all multipliers", MULT), ("all weighted", list(WEIGHTED))):
            full = [c + s for c in cols for s in ("_stage_one", "_stage_two")]
            print(f"  {name}: +{combined_score(per_token_score(perfect(df, full)), weights, mapping)[0] - base[0]:.4f}")
        s1 = epdms(df, "_stage_one")
        s2 = epdms(df, "_stage_two")
        print(f"  scenes scored 0: stage one {np.mean(s1[~np.isnan(s1)] == 0) * 100:.1f}%, "
              f"stage two {np.mean(s2[~np.isnan(s2)] == 0) * 100:.1f}% (unweighted)")
        # stage one x stage two decomposition: the combined score if one stage were perfect
        ones = pd.Series(1.0, index=score.index)
        s1_only = score.where(df.set_index("token")["no_at_fault_collisions_stage_one"].notna(), ones)
        s2_only = score.where(df.set_index("token")["no_at_fault_collisions_stage_two"].notna(), ones)
        print(f"  combined if stage two were perfect: {combined_score(s1_only, weights, mapping)[0]:.4f}; "
              f"if stage one were perfect: {combined_score(s2_only, weights, mapping)[0]:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start-points")
    s.add_argument("--metric-cache", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("navtest")
    s.add_argument("--csv", required=True, help="glob of the one-stage CSV")
    s.add_argument("--human-csv", help="glob of the human agent's CSV (same filter setting)")
    s.add_argument("--check", action="store_true", help="fail unless the recomputed EPDMS equals the score column")
    s = sub.add_parser("navhard")
    s.add_argument("--start-points", required=True)
    s.add_argument("runs", nargs="+", help="navhard scoring folders (the experiment folder or its timestamp folder)")
    args = p.parse_args()
    {"start-points": cmd_start_points, "navtest": cmd_navtest, "navhard": cmd_navhard}[args.cmd](args)


if __name__ == "__main__":
    main()
