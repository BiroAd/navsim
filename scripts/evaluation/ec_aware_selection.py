"""EC-aware trajectory selection on saved predictions (step 0 of the two-frame extended comfort work).

For every scene that has a previous frame (navtest: the scene starting 0.5 s earlier in the same log, as
run_pdm_score_one_stage.py pairs them; navhard_two_stage: the (now, previous) pairs of the split's
reactive_all_mapping), the previous plan is the base rule's choice at the previous frame (no EC there; an agent can
recompute it from the past frames of its own input). At the current frame the candidates are ranked by the policy
(top-N of the policy, or the top-N by policy inside the scorer's shortlist, or, with --bases mixture, the top-N by
the mixture objective log p + lambda * log predicted total); the first candidate whose two-frame
extended comfort against the previous plan passes is chosen, and the base choice is kept when none passes. EC is
computed exactly as the official scorer does: both plans simulated by PDMSimulator from their own logged ego states,
then SceneAggregator._compute_two_frame_comfort's overlap and ego_is_two_frame_extended_comfort.

Output: one prediction folder per (base, N) with only trajectory.npy (+ tokens, codebook, meta), scored with
trajectory_selection=argmax (run_vision_llm_lora_score_predictions_slurm.sh SELECTION=argmax), so the official
pipeline does the scoring. Diagnostics (also in <out>/ec_selection_stats.json):
  - EC of the unchanged base choice per pair, to compare with the official EC of the base run (validates the
    replication);
  - agreement of a "fast" EC from the planned (interpolated, not simulated) trajectories with the simulated EC, the
    question for computing EC inside RL training.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

SPLIT_YAML = "navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml"
INTERVAL = 0.1  # proposal_sampling.interval_length (default_common.yaml)
NUM_POSES = 40


def load(pred_dir: Path):
    from navsim.agents.vision_llm_lora.vision_llm_lora_precomputed_agent import load_predictions

    return load_predictions(pred_dir)


def logits(pred, key, rows):
    a = torch.from_numpy(np.array(pred["arrays"][key][rows]))
    return a.view(torch.bfloat16).float() if pred["meta"]["logits_dtype"] == "bfloat16" else a.float()


def rankings(policy, scores, shortlist_size: int, n_max: int, chunk: int = 256, mixture_lambda: float = 10.0):
    """Per token: policy top-n_max (argmax base), the top-n_max by policy inside the scorer's shortlist, and the
    top-n_max by the mixture objective log p + mixture_lambda * log total (select_trajectory's "mixture"; first entry =
    its choice)."""
    from navsim.agents.vision_llm_lora.vision_llm_lora_loss import aggregate_pdm_scores

    n = len(policy["row"])
    top_policy, top_short = np.zeros((n, n_max), np.int64), np.zeros((n, n_max), np.int64)
    top_mix = np.zeros((n, n_max), np.int64)
    srow = [scores["row"][t] for t in sorted(policy["row"], key=policy["row"].get)] if scores else None
    for s in range(0, n, chunk):
        rows = np.arange(s, min(s + chunk, n))
        log_p = logits(policy, "trajectory_logits", rows).log_softmax(-1)
        # stable sort: among tied (bf16) logits the lowest index first, as argmax picks it
        top_policy[rows] = torch.sort(log_p, dim=-1, descending=True, stable=True).indices[:, :n_max].numpy()
        if scores is not None:
            total = aggregate_pdm_scores(logits(scores, "pdm_score_logits", [srow[r] for r in rows]).sigmoid())
            short = total.topk(shortlist_size, -1).indices
            order = torch.sort(log_p.gather(1, short), dim=-1, descending=True, stable=True).indices  # ties: argmax order
            top_short[rows] = short.gather(1, order)[:, :n_max].numpy()
            mix = log_p + mixture_lambda * total.clamp_min(1e-6).log()
            top_mix[rows] = torch.sort(mix, dim=-1, descending=True, stable=True).indices[:, :n_max].numpy()
    return top_policy, top_short, top_mix


def pairs_for(split: str, tokens: list, cache_paths: dict, root: Path):
    """(current, previous) token pairs scored with EC, grouped for cache reuse."""
    if split == "navhard_two_stage":
        mapping = yaml.safe_load((root / SPLIT_YAML).read_text())["reactive_all_mapping"]
        pairs = []
        for now, prev, second in mapping:
            pairs.append((now, prev))
            pairs += [(a, b) for a, b in second]
        return [p for p in pairs if p[0] in cache_paths and p[1] in cache_paths]
    # navtest: adjacency by start time within a log (run_pdm_score_one_stage.infer_start_adjacent_mapping); needs
    # each token's log and start time, read by the workers (pairs built after the first pass)
    return None


def _read_times(args):
    import lzma
    import pickle

    out = []
    for token, path in args:
        with lzma.open(path, "rb") as f:
            mc = pickle.load(f)
        out.append((token, mc.log_name, mc.timepoint.time_s))
    return out


def _work(args):
    """EC of the candidates of each pair against the previous base plan(s)."""
    import lzma
    import pickle

    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        ego_is_two_frame_extended_comfort,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    jobs, cache_paths, codebook = args
    sampling = TrajectorySampling(num_poses=NUM_POSES, interval_length=INTERVAL)
    traj_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    simulator = PDMSimulator(sampling)
    memo = {}

    def ego(token):
        if token not in memo:
            with lzma.open(cache_paths[token], "rb") as f:
                mc = pickle.load(f)
            memo[token] = (mc.ego_state, mc.timepoint.time_s)  # start_time as run_pdm_score*.py
        return memo[token]

    def states(poses_list, state):
        planned = np.stack([
            get_trajectory_as_array(transform_trajectory(Trajectory(p, traj_sampling), state), sampling, state.time_point)
            for p in poses_list
        ])
        return planned, simulator.simulate_proposals(planned, state)

    def ec(cur, prev, dt):
        k = round(dt / INTERVAL)
        c, p = cur[:, :-k], np.broadcast_to(prev[None, k:], cur[:, :-k].shape)
        return ego_is_two_frame_extended_comfort(c, p, np.arange(c.shape[1]) * INTERVAL)

    out = []
    for cur_token, prev_token, cand_idx, prev_idx in jobs:
        (cur_state, cur_t), (prev_state, prev_t) = ego(cur_token), ego(prev_token)
        dt = cur_t - prev_t
        assert 0 < dt < 0.55, (cur_token, prev_token, dt)
        prev_plan, prev_sim = states([codebook[i] for i in prev_idx], prev_state)  # one per base
        cand_plan, cand_sim = states([codebook[i] for i in cand_idx], cur_state)
        sim = np.stack([ec(cand_sim, prev_sim[b], dt) for b in range(len(prev_idx))])
        fast = np.stack([ec(cand_plan, prev_plan[b], dt) for b in range(len(prev_idx))])
        out.append((cur_token, sim, fast))
        if len(memo) > 64:
            memo.clear()
    return out



def _work_sequential(args):
    """--previous chosen: each log in time order; the previous plan is what was actually chosen at t - 0.5 s."""
    import lzma
    import pickle

    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        ego_is_two_frame_extended_comfort,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    logs, cache_paths, codebook, ranks, ns = args  # ranks: base -> {token: (n_max,) policy-ordered candidates}
    sampling = TrajectorySampling(num_poses=NUM_POSES, interval_length=INTERVAL)
    traj_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    simulator = PDMSimulator(sampling)
    variants = [(b, n) for b in ranks for n in ns]
    chosen = {v: {} for v in variants}
    ec_of_choice = {v: [] for v in variants}
    for items in logs:  # [(time, token)] of one log
        frames = []
        for _, token in items:
            with lzma.open(cache_paths[token], "rb") as f:
                mc = pickle.load(f)
            frames.append((mc.timepoint.time_s, token, mc.ego_state))
        frames.sort(key=lambda x: x[0])
        prev_t, prev_sim = None, {}
        for t, token, state in frames:
            cand = np.unique(np.concatenate([ranks[b][token] for b in ranks]))
            pos = {c: i for i, c in enumerate(cand)}
            planned = np.stack([
                get_trajectory_as_array(transform_trajectory(Trajectory(codebook[c], traj_sampling), state), sampling,
                                        state.time_point)
                for c in cand
            ])
            sim = simulator.simulate_proposals(planned, state)
            has_prev = prev_t is not None and abs(t - prev_t) <= 0.55  # infer_start_adjacent_mapping
            new_sim = {}
            for v in variants:
                b, n = v
                rank = ranks[b][token]
                choice = rank[0]
                if has_prev:
                    dt = t - prev_t
                    assert 0 < dt < 0.55, (token, dt)
                    k = round(dt / INTERVAL)
                    c_ = sim[:, :-k]
                    ok = ego_is_two_frame_extended_comfort(c_, np.broadcast_to(prev_sim[v][None, k:], c_.shape),
                                                           np.arange(c_.shape[1]) * INTERVAL)
                    for c in rank[:n]:
                        if ok[pos[c]]:
                            choice = c
                            break
                    ec_of_choice[v].append(bool(ok[pos[choice]]))
                chosen[v][token] = int(choice)
                new_sim[v] = sim[pos[choice]]
            prev_t, prev_sim = t, new_sim
    return chosen, ec_of_choice


def write_folder(folder: Path, trajectories, codebook, tokens, meta):
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "trajectory.npy", trajectories.astype(np.float32))
    np.save(folder / "codebook.npy", codebook)
    (folder / "tokens.json").write_text(json.dumps(tokens))
    (folder / "meta.json").write_text(json.dumps(dict(meta, logits_dtype="float32"), indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, choices=["navtest", "navhard_two_stage"])
    p.add_argument("--predictions", required=True, help="policy prediction folder (predictions/<name>/<split>)")
    p.add_argument("--scores", help="scorer prediction folder for the shortlist base (e.g. p3_aux_ctrl_ep02/<split>)")
    p.add_argument("--shortlist-size", type=int, default=256)
    p.add_argument("--candidates", type=int, nargs="+", default=[16, 64])
    p.add_argument("--metric-cache", required=True)
    p.add_argument("--out-root", required=True, help="predictions root; folders <name>_ec_<base>_n<N>/<split>")
    p.add_argument("--name", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--previous", choices=["base", "chosen"], default="base",
                   help="previous plan: the base choice at t - 0.5 s, or (navtest) the choice actually made there, "
                        "processing each log in time order (folders <name>_ecseq_<base>_n<N>)")
    p.add_argument("--bases", nargs="+", choices=["argmax", "shortlist", "mixture"], default=["argmax", "shortlist"],
                   help="base rules (shortlist and mixture need --scores; default argmax + shortlist as before); "
                        "folders ..._<argmax|short<k>|mix<lambda>>_n<N>")
    p.add_argument("--mixture-lambda", type=float, default=10.0)
    p.add_argument("--limit", type=int, help="smoke test: only the first N pairs (outputs then incomplete)")
    args = p.parse_args()

    from navsim.common.dataloader import MetricCacheLoader

    root = Path(__file__).resolve().parents[2]
    policy = load(Path(args.predictions))
    scores = load(Path(args.scores)) if args.scores else None
    tokens = sorted(policy["row"], key=policy["row"].get)
    codebook = policy["codebook"].float().numpy()
    n_max = max(args.candidates)
    if scores is None and "mixture" in args.bases:
        raise SystemExit("--bases mixture needs --scores")
    top_policy, top_short, top_mix = rankings(policy, scores, args.shortlist_size, n_max,
                                              mixture_lambda=args.mixture_lambda)
    argmax_traj = np.array(policy["arrays"]["trajectory"])
    assert np.allclose(argmax_traj, codebook[top_policy[:, 0]], atol=1e-4), "trajectory.npy is not the argmax entry"
    bases = {}
    if "argmax" in args.bases:
        bases["argmax"] = top_policy
    if "shortlist" in args.bases and scores is not None:
        bases["short%d" % args.shortlist_size] = top_short
    if "mixture" in args.bases:
        bases["mix%g" % args.mixture_lambda] = top_mix
    row = policy["row"]

    cache_paths = {t: str(pth) for t, pth in MetricCacheLoader(Path(args.metric_cache)).metric_cache_paths.items()}
    pairs = pairs_for(args.split, tokens, cache_paths, root)
    if pairs is None and args.previous == "chosen":  # navtest, sequential: logs from the cache layout <root>/<log>/...
        by_log = defaultdict(list)
        for t in tokens:
            by_log[Path(cache_paths[t]).relative_to(args.metric_cache).parts[0]].append((0.0, t))  # workers sort by time
        pairs = []
    elif pairs is None:  # navtest
        todo = [(t, cache_paths[t]) for t in tokens]
        with ProcessPoolExecutor(args.workers) as pool:
            info = [r for part in pool.map(_read_times, [todo[i :: args.workers * 4] for i in range(args.workers * 4)]) for r in part]
        by_log = defaultdict(list)
        for token, log, t in info:
            by_log[log].append((t, token))
        pairs = []
        for log, items in by_log.items():
            items.sort()
            pairs += [(cur, prev) for (tp, prev), (tc, cur) in zip(items, items[1:]) if tc - tp <= 0.55]
    pairs = [(c, pr) for c, pr in pairs if c in row and pr in row][: args.limit]

    if args.previous == "chosen":
        if args.split != "navtest":
            raise SystemExit("--previous chosen is for navtest (navhard previous frames are never changed)")
        logs = [items for items in by_log.values()][: args.limit]
        logs.sort(key=len, reverse=True)
        chunks = [logs[i :: args.workers * 4] for i in range(args.workers * 4)]
        ranks_all = {b: {t: r[row[t]] for t in tokens} for b, r in bases.items()}
        work = []
        for chunk in chunks:
            toks = {tok for items in chunk for _, tok in items}
            work.append((chunk, cache_paths, codebook, {b: {t: r[t] for t in toks} for b, r in ranks_all.items()},
                         args.candidates))
        merged, ecs = defaultdict(dict), defaultdict(list)
        with ProcessPoolExecutor(args.workers) as pool:
            for chosen, ec_list in pool.map(_work_sequential, work):
                for v, d in chosen.items():
                    merged[v].update(d)
                for v, lst in ec_list.items():
                    ecs[v] += lst
        stats = {"tokens": len(tokens), "previous": "chosen"}
        for (base, n), d in sorted(merged.items()):
            idx = bases[base][:, 0].copy()
            for t, c in d.items():
                idx[row[t]] = c
            moved = int((idx != bases[base][:, 0]).sum())
            folder = Path(args.out_root) / f"{args.name}_ecseq_{base}_n{n}" / args.split
            write_folder(folder, codebook[idx], codebook, tokens,
                         dict(policy["meta"], ec_selection={"base": base, "candidates": n, "previous": "chosen",
                                                            "source": args.predictions, "scores": args.scores}))
            stats[f"{base}_n{n}"] = {"changed": moved, "ec_vs_actual_previous": float(np.mean(ecs[(base, n)])),
                                     "pairs": len(ecs[(base, n)])}
            print(f"{base} n{n} (sequential): {moved} / {len(tokens)} changed, EC vs the actual previous choice "
                  f"{np.mean(ecs[(base, n)]):.4f} over {len(ecs[(base, n)])} pairs -> {folder}")
        (Path(args.out_root) / f"{args.name}_ecseq_stats_{args.split}.json").write_text(json.dumps(stats, indent=1))
        return

    jobs = []
    for cur, prev in pairs:
        cand = np.unique(np.concatenate([b[row[cur]] for b in bases.values()]))
        jobs.append((cur, prev, cand, [b[row[prev], 0] for b in bases.values()]))
    order = sorted(range(len(jobs)), key=lambda i: cache_paths[jobs[i][0]])  # same log together
    chunks = [[jobs[i] for i in order[s :: args.workers * 4]] for s in range(args.workers * 4)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = [r for part in pool.map(_work, [(c, cache_paths, codebook) for c in chunks]) for r in part]
    cand_of = {j[0]: j[2] for j in jobs}

    stats = {"pairs": len(pairs), "tokens": len(tokens)}
    for b, (base, ranking) in enumerate(bases.items()):
        chosen = ranking[:, 0].copy()
        base_ec, agree = [], []
        changed = {n: chosen.copy() for n in args.candidates}
        for cur, sim, fast in results:
            cand = cand_of[cur]
            pos = {c: i for i, c in enumerate(cand)}
            ok = sim[b]
            base_ec.append(ok[pos[ranking[row[cur], 0]]])
            agree.append((ok == fast[b]).mean())
            for n in args.candidates:
                for c in ranking[row[cur], :n]:
                    if ok[pos[c]]:
                        changed[n][row[cur]] = c
                        break
        stats[base] = {"base_ec_mean": float(np.mean(base_ec)), "fast_vs_simulated_agreement": float(np.mean(agree))}
        for n in args.candidates:
            folder = Path(args.out_root) / f"{args.name}_ec_{base}_n{n}" / args.split
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "trajectory.npy", codebook[changed[n]].astype(np.float32))
            np.save(folder / "codebook.npy", codebook)
            (folder / "tokens.json").write_text(json.dumps(tokens))
            meta = dict(policy["meta"], ec_selection={"base": base, "candidates": n, "source": args.predictions,
                                                      "scores": args.scores, "pairs": len(pairs)})
            meta["logits_dtype"] = "float32"
            (folder / "meta.json").write_text(json.dumps(meta, indent=1))
            moved = int((changed[n] != ranking[:, 0]).sum())
            stats[base][f"n{n}_changed"] = moved
            print(f"{base} n{n}: {moved} / {len(tokens)} choices changed -> {folder}")
        print(f"{base}: EC of the base choice {stats[base]['base_ec_mean']:.4f} over {len(base_ec)} pairs; "
              f"fast EC agrees with simulated on {stats[base]['fast_vs_simulated_agreement'] * 100:.1f}% of candidates")
    out = Path(args.out_root) / f"{args.name}_ec_stats_{args.split}.json"
    out.write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
