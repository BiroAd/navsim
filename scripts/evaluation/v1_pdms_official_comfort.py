"""
NAVSIM v1 PDMS with the v1 comfort term, for a navtest result CSV of this devkit.

v1 PDMS = NC x DAC x (5 EP + 5 TTC + 2 C) / 12. This devkit (v2) has no v1 comfort column; until now the v1 numbers used
v2 history comfort, which runs the same ego_is_comfortable check on [human past states + simulated plan]
(pdm_scorer.py, _calculate_history_comfort) and is stricter than v1 comfort, which checks the simulated plan alone
(the human gets 0.981 vs the official 0.999). This script recomputes C exactly as v1 did: each selected trajectory is
simulated the same way as in navsim/evaluate/pdm_score.py (transform_trajectory, get_trajectory_as_array,
PDMSimulator.simulate_proposals) and ego_is_comfortable is applied without the history padding. NC, DAC, EP, TTC come
from the given CSV (same run). Upstream code is only called, not changed.

    python scripts/evaluation/v1_pdms_official_comfort.py --csv <pdm_navtest/<run>/<ts>/<ts>.csv> \
        --predictions $NAVSIM_EXP_ROOT/predictions/<pred>/navtest [--scores ...] --selection mixture_v1 --mixture-lambda 10
    python scripts/evaluation/v1_pdms_official_comfort.py --csv <human agent CSV> --human
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_comfortable

CONFIG_DIR = Path(__file__).resolve().parents[2] / "navsim/planning/script/config/pdm_scoring"
_W = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True, help="result CSV of the run (NC, DAC, EP, TTC per token)")
    parser.add_argument("--human", action="store_true", help="the logged human trajectory (metric_cache.human_trajectory)")
    parser.add_argument("--predictions", type=Path, help="predictions/<pred>/navtest (policy)")
    parser.add_argument("--scores", type=Path, default=None, help="predictions/<pred>/navtest with the sub-scores")
    parser.add_argument("--selection", default="argmax")
    parser.add_argument("--shortlist-size", type=int, default=64)
    parser.add_argument("--mixture-lambda", type=float, default=10.0)
    parser.add_argument("--metric-cache", type=Path,
                        default=Path(os.environ.get("NAVSIM_EXP_ROOT", ".")) / "metric_cache_navtest")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    parser.add_argument("--out", type=Path, default=None, help="per-token CSV (default: next to --csv)")
    return parser.parse_args()


def _init(args: argparse.Namespace) -> None:
    overrides = []
    if not args.human:
        overrides = ["agent=vision_llm_lora_precomputed_agent", f"agent.predictions_path='{args.predictions}'",
                     f"agent.trajectory_selection={args.selection}", f"agent.selection_shortlist_size={args.shortlist_size}",
                     f"agent.selection_mixture_lambda={args.mixture_lambda}"]
        overrides += [f"agent.scores_path='{args.scores}'"] if args.scores else []
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="default_run_pdm_score", overrides=overrides)
    _W["simulator"] = instantiate(cfg.simulator)
    _W["cache"] = MetricCacheLoader(args.metric_cache)
    _W["human"] = args.human
    if not args.human:
        agent = instantiate(cfg.agent)
        agent.initialize()
        _W["agent"] = agent


def _comfort(token: str):
    metric_cache = _W["cache"].get_from_token(token)
    if _W["human"]:
        trajectory = metric_cache.human_trajectory
    else:
        trajectory = _W["agent"].compute_trajectory(None, SimpleNamespace(scene_metadata=SimpleNamespace(initial_token=token)))
    simulator = _W["simulator"]
    ego = metric_cache.ego_state
    states = get_trajectory_as_array(transform_trajectory(trajectory, ego), simulator.proposal_sampling, ego.time_point)
    simulated = simulator.simulate_proposals(states[None, ...], ego)  # (1, T, StateIndex.size())
    time_point_s = np.arange(simulated.shape[1], dtype=np.float64) * simulator.proposal_sampling.interval_length
    return token, float(ego_is_comfortable(simulated, time_point_s).all(axis=-1)[0])


def main() -> None:
    args = parse_args()
    if not args.human and args.predictions is None:
        raise SystemExit("give --predictions or --human")
    df = pd.read_csv(args.csv)
    df = df[(df.token != "average_all_frames") & (df.valid == True)].set_index("token")  # noqa: E712
    with Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
        comfort = dict(pool.imap_unordered(_comfort, df.index, chunksize=32))
    df["v1_comfort"] = pd.Series(comfort)
    weighted = lambda c: (5 * df.ego_progress + 5 * df.time_to_collision_within_bound + 2 * c) / 12  # noqa: E731
    mult = df.no_at_fault_collisions * df.drivable_area_compliance
    df["v1_pdms"] = mult * weighted(df.v1_comfort)
    df["v1_pdms_history_comfort"] = mult * weighted(df.history_comfort)
    out = args.out or args.csv.with_name(args.csv.stem + "_v1_comfort.csv")
    df[["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound",
        "history_comfort", "v1_comfort", "v1_pdms", "v1_pdms_history_comfort"]].to_csv(out)
    print(f"{args.csv}: {len(df)} scenes -> {out}")
    print(f"NC {df.no_at_fault_collisions.mean():.4f}  DAC {df.drivable_area_compliance.mean():.4f}  "
          f"TTC {df.time_to_collision_within_bound.mean():.4f}  EP {df.ego_progress.mean():.4f}")
    print(f"comfort: v1 {df.v1_comfort.mean():.4f}  (v2 history comfort {df.history_comfort.mean():.4f})")
    print(f"v1 PDMS (v1 comfort) {df.v1_pdms.mean():.4f}   previous proxy (history comfort) {df.v1_pdms_history_comfort.mean():.4f}")


if __name__ == "__main__":
    main()
