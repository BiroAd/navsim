"""
Selection rules on the navtrain validation scenes, without simulation: every rule is applied with the same
select_trajectory as the official evaluation (policy logits from --policy, sub-score logits from --scores, as the
replay agent combines them), and the chosen codebook entry is looked up in the codebook score table
(exp/codebook_scores_navtrain/<log>.npz: log-replay PDM scores of every entry, no two-frame comfort, no human filter;
the RL reward). v2 rules are graded with the table's "score" (EPDMS without EC), "_v1" rules with the v1 PDMS from the
same sub-scores (NC * DAC * (5 EP + 5 TTC + 2 HC) / 12). Held-out data for choosing k / lambda / N instead of navtest
and navhard; blind to EC and to navhard's shifted ego poses.

    python scripts/evaluation/eval_selection_rules_val.py --policy $NAVSIM_EXP_ROOT/predictions/rl3_b0.02_ep01 \
        --scores $NAVSIM_EXP_ROOT/predictions/p3_aux_policy_ep03 --out $NAVSIM_EXP_ROOT/selection_val/rl3_b0.02.csv
"""
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from navsim.agents.vision_llm_lora.vision_llm_lora_agent import select_trajectory
from navsim.agents.vision_llm_lora.vision_llm_lora_precomputed_agent import load_predictions

SPLIT = "navtrain_val"
LOG_SPLIT = Path(__file__).resolve().parents[2] / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
# (selection, shortlist k, mixture lambda, local N, local lambda)
RULES: List[Tuple[str, int, float, int, float]] = (
    [("argmax", 0, 0, 0, 0), ("score_heads", 0, 0, 0, 0), ("score_heads_v1", 0, 0, 0, 0)]
    + [(s, k, 0, 0, 0) for s in ("shortlist", "shortlist_v1") for k in (16, 64, 128, 256, 512, 1024, 2048)]
    + [(s, 0, lam, 0, 0) for s in ("mixture", "mixture_v1") for lam in (1, 3, 10, 30, 100)]
    + [(s, 0, 0, n, lam) for s in ("local", "local_v1") for n in (64, 128) for lam in (0.0, 0.02, 0.05, 0.1, 0.2)]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", type=Path, required=True, help="prediction folder with navtrain_val/ (policy logits)")
    parser.add_argument("--scores", type=Path, default=None, help="prediction folder for the sub-scores (default: --policy)")
    parser.add_argument("--table", type=Path, default=Path(os.environ.get("NAVSIM_EXP_ROOT", ".")) / "codebook_scores_navtrain")
    parser.add_argument("--cache-path", type=Path, default=Path(os.environ.get("NAVSIM_EXP_ROOT", ".")) / "training_cache")
    parser.add_argument("--out", type=Path, required=True, help="CSV with one row per rule")
    parser.add_argument("--chunk", type=int, default=128)
    return parser.parse_args()


def token_logs(cache_path: Path) -> Dict[str, str]:
    val_logs = yaml.safe_load(LOG_SPLIT.read_text())["val_logs"]
    return {t.name: log for log in val_logs if (cache_path / log).is_dir() for t in (cache_path / log).iterdir()}


def reward_rows(table: Path, tokens: List[str], logs: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    """(N, K) v2 reward ("score") and v1 PDMS of every codebook entry per scene, NaN where the table has no row."""
    v2 = v1 = None
    by_log: Dict[str, List[int]] = {}
    for i, t in enumerate(tokens):
        by_log.setdefault(logs[t], []).append(i)
    for log, idx in by_log.items():
        data = np.load(table / f"{log}.npz")
        row = {str(t): j for j, t in enumerate(data["tokens"])}
        if v2 is None:
            k = data["score"].shape[1]
            v2 = np.full((len(tokens), k), np.nan, np.float32)
            v1 = np.full((len(tokens), k), np.nan, np.float32)
        sub = {name: data[name].astype(np.float32) for name in
               ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound",
                "history_comfort")}
        for i in idx:
            j = row.get(tokens[i])
            if j is None:
                continue
            v2[i] = data["score"][j]
            v1[i] = (sub["no_at_fault_collisions"][j] * sub["drivable_area_compliance"][j]
                     * (5 * sub["ego_progress"][j] + 5 * sub["time_to_collision_within_bound"][j]
                        + 2 * sub["history_comfort"][j]) / 12)
    return v2, v1


def main() -> None:
    args = parse_args()
    policy = load_predictions(args.policy / SPLIT)
    scores = load_predictions((args.scores or args.policy) / SPLIT)
    tokens = json.loads((args.policy / SPLIT / "tokens.json").read_text())
    score_row = scores["row"]
    assert all(t in score_row for t in tokens), "policy and score folders cover different scenes"
    v2, v1 = reward_rows(args.table, tokens, token_logs(args.cache_path))
    valid = ~np.isnan(v2[:, 0])
    print(f"{len(tokens)} scenes, {int(valid.sum())} with a score row")
    codebook = policy["codebook"].float()
    flat = codebook.reshape(len(codebook), -1)
    picks = {rule: np.zeros(len(tokens), np.int64) for rule in RULES}
    bf16 = policy["meta"]["logits_dtype"] == "bfloat16"
    for start in range(0, len(tokens), args.chunk):
        rows = list(range(start, min(start + args.chunk, len(tokens))))
        logits = torch.from_numpy(np.array(policy["arrays"]["trajectory_logits"][rows]))
        sub = torch.from_numpy(np.stack([np.array(scores["arrays"]["pdm_score_logits"][score_row[tokens[i]]]) for i in rows]))
        if bf16:
            logits, sub = logits.view(torch.bfloat16), sub.view(torch.bfloat16)
        predictions = {"trajectory_logits": logits.float(), "pdm_score_logits": sub.float(),
                       "trajectory": codebook[logits.float().argmax(-1)]}
        for rule in RULES:
            selection, k, mix, n, lam = rule
            out = select_trajectory(predictions, selection, codebook, shortlist_size=k or 64, mixture_lambda=mix or 10.0,
                                    local_size=n or 64, local_lambda=lam)
            index = torch.cdist(out.reshape(len(rows), -1), flat).argmin(-1)  # exact codebook rows (distance 0)
            picks[rule][rows] = index.numpy()
        if start % (args.chunk * 20) == 0:
            print(f"  {start + len(rows)}/{len(tokens)}", flush=True)

    base = v2[np.arange(len(tokens)), picks[RULES[0]]]
    base_v1 = v1[np.arange(len(tokens)), picks[RULES[0]]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["selection", "k", "mixture_lambda", "local_n", "local_lambda", "metric", "score", "vs_argmax",
                         "vs_argmax_se", "zero_share", "changed_vs_argmax"])
        for rule in RULES:
            selection = rule[0]
            table, ref, metric = (v1, base_v1, "v1") if selection.endswith("_v1") else (v2, base, "v2")
            got = table[np.arange(len(tokens)), picks[rule]][valid]
            diff = got - ref[valid]
            writer.writerow([*rule, metric, f"{got.mean():.4f}", f"{diff.mean():+.4f}",
                             f"{diff.std(ddof=1) / np.sqrt(len(diff)):.4f}", f"{(got == 0).mean():.4f}",
                             f"{(picks[rule] != picks[RULES[0]])[valid].mean():.3f}"])
    print(f"wrote {args.out}")
    for line in open(args.out):
        print(line.rstrip())


if __name__ == "__main__":
    main()
