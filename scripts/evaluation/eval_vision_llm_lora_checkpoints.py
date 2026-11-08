"""
Offline evaluation of vision_llm_lora checkpoints on the navtrain val split (and optionally a subset of the train
split), on one GPU, from the training cache. No simulation: every codebook entry of every navtrain scene is already
PDM-scored (exp/codebook_scores_navtrain, scripts/evaluation/score_codebook_trajectories.py), so the score of the
chosen entry is a lookup.

Per checkpoint and split it reports:
- ade / fde of the output trajectory (refined when the run has a refiner; anchor_* = the argmax entry),
- score: PDM score of the argmax entry, as the codebook scoring computes it (non-reactive log replay, no extended
  comfort; per-frame EPDMS-style score with DDC/TLC/LK/HC), and the mean of each sub-score of that entry,
- expected_score: E_p[score] under the predicted distribution (what an expected-reward objective would maximize),
- oracle_score: best entry per scene (upper bound of the codebook),
- entropy of the predicted distribution,
- with PDM sub-score heads (pdm_score_loss_weight > 0): the true score of the entry the heads rank best, over all
  entries (head_all) and among the policy's top-k (head_topk), plus the heads' MAE on the policy's top-k. These runs
  also write <out-dir>/heads_<run>.csv: comb_<lambda> = true score of argmax(log p + lambda * log predicted total),
  filter_<tau> = the most likely entry among those whose predicted multiplicative product is >= tau,
  shortlist_<k> = the most likely entry among the k entries with the best predicted total (the policy picks from the
  heads' shortlist; k = 1 is head_all), each with _ego_progress and _changed (share of scenes not on the argmax entry),
  and the shortlist's headroom: _oracle (best true score in it), _uniform (its mean), _expected (policy renormalized
  to it) and _top4_oracle (best of the policy's 4 most likely entries in it),
  headsel_<metric> = sub-scores of the head_all entry, and per head (headdiag_<metric>_*) the BCE (as in the training
  loss), MAE over all entries and over the policy's top-k, accuracy for binary metrics, and the mean target.
The refined trajectory has no precomputed score, so score columns always describe the argmax entry.

Each checkpoint is evaluated with the Hydra config saved in its run folder; only the trainable parameters are loaded
from the checkpoint (the frozen Qwen weights come from HF_HOME, as in the phase handoff). Per-scene results (token,
argmax id, top-k ids and logits, score, ADE, FDE) go to <out-dir>/<run>_ep<NN>_<split>.npz for paired comparisons,
the summary rows to <out-dir>/summary_<run>.csv.

Usage (one GPU; see scripts/evaluation/run_vision_llm_lora_checkpoint_eval_slurm.sh):
  python scripts/evaluation/eval_vision_llm_lora_checkpoints.py \
      --run-dir $NAVSIM_EXP_ROOT/p1c_scores/2026.11.01.00.34.10 --epochs 5 7 10 14 19 \
      --out-dir $NAVSIM_EXP_ROOT/checkpoint_eval --train-log-stride 20
"""
import argparse
import csv
import dataclasses
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_features import PDM_SCORE_KEYS
from navsim.agents.vision_llm_lora.vision_llm_lora_loss import (
    BINARY_PDM_SCORE_KEYS,
    MULTIPLICATIVE_PDM_SCORE_KEYS,
    aggregate_pdm_scores,
)
from navsim.planning.training.dataset import CacheOnlyDataset

# DataLoader workers hand tensors to the main process through shared memory files instead of passing descriptors over a
# socket in /tmp: on the cluster nodes that socket vanished mid-run twice (FileNotFoundError in resource_sharer).
torch.multiprocessing.set_sharing_strategy("file_system")

SCORE_FIELDS = (*PDM_SCORE_KEYS, "score")
PER_SCENE_METRICS = ("ade", "fde", "anchor_ade", "anchor_fde", "expected_score", "oracle_score", "entropy", "head_all",
                     "head_topk", "head_mae", *(f"sel_{k}" for k in SCORE_FIELDS))
CSV_FIELDS = ("run", "checkpoint", "split", "n", "n_scored", "sel_score", "score_se", *PER_SCENE_METRICS[:-1])
HEAD_PREFIXES = ("comb_", "filter_", "shortlist_", "headsel_", "headdiag_")  # score-head diagnostics, written to heads_<run>.csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="exp/<run>/<date> folder with code/ and checkpoints/")
    parser.add_argument("--epochs", type=int, nargs="*", default=[], help="epochs from checkpoints/epochs/")
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=[], help="further checkpoint files")
    parser.add_argument("--name", default=None, help="run name in the output (default: the run folder's parent)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, default=None, help="default: $NAVSIM_EXP_ROOT/codebook_scores_navtrain")
    parser.add_argument("--train-log-stride", type=int, default=0, help="also evaluate every n-th train log (0: off)")
    parser.add_argument("--max-scenes", type=int, default=None, help="limit scenes per split (smoke test)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--combine-lambdas", type=float, nargs="*", default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
                        help="score-head runs: weights of log predicted total added to log p (0 = argmax, inf = head_all)")
    parser.add_argument("--filter-thresholds", type=float, nargs="*", default=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
                        help="score-head runs: argmax of p among entries whose predicted multiplicative product >= tau")
    parser.add_argument("--shortlist-sizes", type=int, nargs="*", default=[4, 16, 64, 256, 1024],
                        help="score-head runs: argmax of p among the k entries with the best predicted total")
    return parser.parse_args()


def build_agent(cfg) -> torch.nn.Module:
    """The run's agent without a checkpoint, dropping config fields the current config class no longer has."""
    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)
    known = {f.name for f in dataclasses.fields(VisionLLMLoRAConfig)} | {"_target_", "_convert_"}
    dropped = sorted(set(agent_cfg["config"]) - known)
    if dropped:
        print(f"Dropping config fields unknown to VisionLLMLoRAConfig: {dropped}")
    agent_cfg["config"] = {k: v for k, v in agent_cfg["config"].items() if k in known}
    # A policy-only run (train_policy_head_only) saved only the policy head: build the agent with the run's training
    # phase fully trainable, so every LoRA / head / planner parameter is loaded (from the start checkpoint, see main).
    agent_cfg["config"]["train_policy_head_only"] = False
    agent_cfg["checkpoint_path"] = None
    return instantiate(agent_cfg)


def base_checkpoints(cfg) -> List[Path]:
    """Checkpoints to load before a run's own: its start checkpoint if it trained only the policy head (its checkpoints
    hold only the parameters it trained; checkpoints leave frozen parameters out)."""
    if cfg.agent.config.get("train_policy_head_only", False):
        return [Path(str(cfg.agent.checkpoint_path).strip("'"))]
    return []


def load_trainable(agent: torch.nn.Module, checkpoints: Sequence[Path]) -> None:
    """Load every trainable parameter from Lightning checkpoints, later ones overriding earlier ones; fail if any is
    missing."""
    state = {}
    for checkpoint in checkpoints:
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
        state.update(
            {k.removeprefix("agent.").replace("_vlm_model._orig_mod.", "_vlm_model."): v for k, v in loaded.items()}
        )
    trainable = [name for name, p in agent.named_parameters() if p.requires_grad]
    missing = [name for name in trainable if name not in state]
    if missing:
        raise KeyError(f"{list(checkpoints)}: {len(missing)} trainable parameters missing, e.g. {missing[:5]}")
    result = agent.load_state_dict({name: state[name] for name in trainable}, strict=False)
    assert not result.unexpected_keys, result.unexpected_keys


class ScoreTable:
    """Per-entry PDM scores of each scene, read per log from the codebook scoring output (one .npz per log)."""

    def __init__(self, scores_dir: Path):
        self._dir = scores_dir
        self._log: Optional[str] = None
        self._rows: Dict[str, int] = {}
        self._arrays: Dict[str, np.ndarray] = {}

    def get(self, log: str, token: str) -> Optional[Dict[str, np.ndarray]]:
        if log != self._log:  # the dataset lists scenes log by log, so each log is read once
            path = self._dir / f"{log}.npz"
            self._log, self._rows, self._arrays = log, {}, {}
            if path.exists():
                data = np.load(path)
                self._rows = {str(t): i for i, t in enumerate(data["tokens"])}
                self._arrays = {key: data[key] for key in SCORE_FIELDS}
        row = self._rows.get(token)
        return None if row is None else {key: values[row] for key, values in self._arrays.items()}


def predicted_total(score_logits: torch.Tensor) -> torch.Tensor:
    """(B, M, K) sub-score logits in PDM_SCORE_KEYS order → (B, K) aggregated predicted score."""
    return aggregate_pdm_scores(score_logits.float().sigmoid())


@torch.no_grad()
def evaluate(agent, dataset: CacheOnlyDataset, scores: ScoreTable, args, dtype: torch.dtype) -> Dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    codebook = agent._vlm_model.codebook.float().cpu()
    out: Dict[str, List] = {k: [] for k in ("token", "argmax", "topk_ids", "topk_logits", *PER_SCENE_METRICS)}
    no_scores = {k: np.zeros(len(codebook), dtype=np.float32) for k in SCORE_FIELDS}
    start, seen = time.time(), 0
    for batch_index, (features, targets) in enumerate(loader):
        tokens = dataset.tokens[seen : seen + len(targets["trajectory"])]
        seen += len(tokens)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
            predictions = agent.forward(features, None, 1.0)
        logits = predictions["trajectory_logits"].float()
        ids = logits.argmax(dim=-1).cpu()
        gt = targets["trajectory"][..., :2].float()
        err = torch.linalg.norm(predictions["trajectory"][..., :2].float().cpu() - gt, dim=-1)
        anchor_err = torch.linalg.norm(codebook[ids][..., :2] - gt, dim=-1)
        probs = logits.softmax(dim=-1)
        topk = logits.topk(args.top_k, dim=-1)

        rows = [scores.get(dataset._valid_cache_paths[t].parent.name, t) for t in tokens]
        valid = torch.tensor([r is not None for r in rows])
        nan = torch.full((len(tokens),), float("nan"))
        if valid.any():
            table = {
                k: torch.from_numpy(np.stack([(r or no_scores)[k] for r in rows]).astype(np.float32))
                for k in SCORE_FIELDS
            }
            score = table["score"].to(logits.device)  # (B, K)
            expected = (probs * score).sum(-1).cpu()
            oracle = table["score"].max(-1).values
            for k in SCORE_FIELDS:
                out[f"sel_{k}"].append(torch.where(valid, table[k].gather(1, ids[:, None])[:, 0], nan))
            out["expected_score"].append(torch.where(valid, expected, nan))
            out["oracle_score"].append(torch.where(valid, oracle, nan))
            if "pdm_score_logits" in predictions:
                total = predicted_total(predictions["pdm_score_logits"])  # (B, K)
                best_all = total.argmax(-1)
                best_topk = topk.indices.gather(1, total.gather(1, topk.indices).argmax(-1, keepdim=True))[:, 0]
                mae = (total.gather(1, topk.indices) - score.gather(1, topk.indices)).abs().mean(-1).cpu()
                out["head_all"].append(torch.where(valid, score.gather(1, best_all[:, None])[:, 0].cpu(), nan))
                out["head_topk"].append(torch.where(valid, score.gather(1, best_topk[:, None])[:, 0].cpu(), nan))
                out["head_mae"].append(torch.where(valid, mae, nan))

                def add(key: str, values: torch.Tensor) -> None:
                    out.setdefault(key, []).append(torch.where(valid, values.float().cpu(), nan))

                log_p, log_total = logits.log_softmax(-1), total.clamp_min(1e-6).log()
                for lam in args.combine_lambdas:
                    best = (log_p + lam * log_total).argmax(-1, keepdim=True)
                    add(f"comb_{lam:g}", score.gather(1, best)[:, 0])
                    add(f"comb_{lam:g}_ego_progress", table["ego_progress"].to(score.device).gather(1, best)[:, 0])
                    add(f"comb_{lam:g}_changed", (best[:, 0] != ids.to(best.device)).float())
                # Heads shortlist, policy picks: the most likely of the k entries with the best predicted total.
                for k_short in args.shortlist_sizes:
                    short = total.topk(min(k_short, total.shape[-1]), dim=-1).indices  # (B, k)
                    pick = short.gather(1, log_p.gather(1, short).argmax(-1, keepdim=True))
                    add(f"shortlist_{k_short}", score.gather(1, pick)[:, 0])
                    ego_progress_pick = table["ego_progress"].to(score.device).gather(1, pick)[:, 0]
                    add(f"shortlist_{k_short}_ego_progress", ego_progress_pick)
                    add(f"shortlist_{k_short}_changed", (pick[:, 0] != ids.to(pick.device)).float())
                    # Headroom of a policy restricted to the shortlist (what RL on the policy head could reach):
                    # _oracle = best true score in the shortlist, _uniform = its mean, _expected = E[score] under the
                    # policy renormalized to the shortlist, _top4_oracle = best of the policy's 4 most likely entries
                    # in the shortlist (reachable by small changes to the ranking).
                    short_score = score.gather(1, short)
                    short_log_p = log_p.gather(1, short)
                    add(f"shortlist_{k_short}_oracle", short_score.max(-1).values)
                    add(f"shortlist_{k_short}_uniform", short_score.mean(-1))
                    add(f"shortlist_{k_short}_expected", (short_log_p.softmax(-1) * short_score).sum(-1))
                    top4 = short_log_p.topk(min(4, short.shape[-1]), dim=-1).indices
                    add(f"shortlist_{k_short}_top4_oracle", short_score.gather(1, top4).max(-1).values)
                for k in SCORE_FIELDS:
                    add(f"headsel_{k}", table[k].gather(1, best_all.cpu()[:, None])[:, 0])
                head_probs = predictions["pdm_score_logits"].float().sigmoid()  # (B, M, K)

                # Safety filter: the most likely entry among those the heads predict to pass the multiplicative metrics
                # (falls back to head_all when no entry passes).
                multi_index = [PDM_SCORE_KEYS.index(k) for k in MULTIPLICATIVE_PDM_SCORE_KEYS]
                safe = head_probs[:, multi_index].prod(dim=1)  # (B, K)
                ego_progress = table["ego_progress"].to(score.device)
                for tau in args.filter_thresholds:
                    passing = safe >= tau
                    pick = log_p.masked_fill(~passing, float("-inf")).argmax(-1)
                    pick = torch.where(passing.any(-1), pick, best_all)[:, None]
                    add(f"filter_{tau:g}", score.gather(1, pick)[:, 0])
                    add(f"filter_{tau:g}_ego_progress", ego_progress.gather(1, pick)[:, 0])
                    add(f"filter_{tau:g}_changed", (pick[:, 0] != ids.to(pick.device)).float())

                for m, key in enumerate(PDM_SCORE_KEYS):
                    p, target = head_probs[:, m], table[key].to(head_probs.device)
                    # Signed error on the policy's top-k: < 0 = the head is pessimistic about plausible entries.
                    add(f"headdiag_{key}_bias_topk", (p - target).gather(1, topk.indices).mean(-1))
                    add(f"headdiag_{key}_bce", F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), target,
                                                                      reduction="none").mean(-1))
                    add(f"headdiag_{key}_mae", (p - target).abs().mean(-1))
                    add(f"headdiag_{key}_mae_topk", (p - target).abs().gather(1, topk.indices).mean(-1))
                    add(f"headdiag_{key}_target_mean", target.mean(-1))
                    if key in BINARY_PDM_SCORE_KEYS:
                        add(f"headdiag_{key}_acc", ((p > 0.5) == (target > 0.5)).float().mean(-1))
        else:
            for k in ("expected_score", "oracle_score", *(f"sel_{k}" for k in SCORE_FIELDS)):
                out[k].append(nan)

        out["token"] += tokens
        out["argmax"].append(ids)
        out["topk_ids"].append(topk.indices.cpu().int())
        out["topk_logits"].append(topk.values.cpu().half())
        out["ade"].append(err.mean(-1))
        out["fde"].append(err[:, -1])
        out["anchor_ade"].append(anchor_err.mean(-1))
        out["anchor_fde"].append(anchor_err[:, -1])
        out["entropy"].append(-(probs * torch.log(probs + 1e-12)).sum(-1).cpu())
        if batch_index % 100 == 0:
            print(f"  {seen}/{len(dataset)} scenes, {seen / (time.time() - start):.1f} scenes/s", flush=True)
        if args.max_scenes is not None and seen >= args.max_scenes:
            break
    print(f"  done: {seen} scenes, {seen / (time.time() - start):.1f} scenes/s, "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    return {k: (np.asarray(v) if k == "token" else torch.cat(v).numpy()) for k, v in out.items() if v}


def summarize(result: Dict[str, np.ndarray]) -> Dict[str, float]:
    summary = {"n": len(result["token"]), "n_scored": int(np.isfinite(result["sel_score"]).sum())}
    for key, values in result.items():
        if key in ("token", "argmax", "topk_ids", "topk_logits"):
            continue
        summary[key] = float(np.nanmean(values))
    summary["score_se"] = float(np.nanstd(result["sel_score"]) / np.sqrt(max(summary["n_scored"], 1)))
    return summary


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.run_dir / "code" / "hydra" / "config.yaml")
    name = args.name or args.run_dir.parent.name
    scores_dir = args.scores_dir or Path(cfg.cache_path).parent / "codebook_scores_navtrain"
    checkpoints = [args.run_dir / "checkpoints" / "epochs" / f"epoch-epoch={e:02d}.ckpt" for e in args.epochs]
    checkpoints += list(args.checkpoints)
    for path in checkpoints:
        if not path.exists():
            raise FileNotFoundError(path)

    agent = build_agent(cfg)
    agent.eval().cuda()
    dtype = getattr(torch, agent._config.torch_dtype)
    feature_builders, target_builders = agent.get_feature_builders(), agent.get_target_builders()[:1]  # trajectory only

    splits = {"val": list(cfg.val_logs)}
    if args.train_log_stride > 0:
        splits["train"] = list(cfg.train_logs)[:: args.train_log_stride]
    datasets = {
        split: CacheOnlyDataset(cfg.cache_path, feature_builders, target_builders, log_names=logs)
        for split, logs in splits.items()
    }
    for split, dataset in datasets.items():
        print(f"{split}: {len(dataset)} scenes in {len(dataset.log_names)} logs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"summary_{name}.csv"  # one file per run, so parallel jobs never share one
    base = base_checkpoints(cfg)
    if base:
        print(f"Policy-only run: loading {base[0]} first, then each checkpoint's policy head", flush=True)
    for checkpoint in checkpoints:
        load_trainable(agent, [*base, checkpoint])
        label = checkpoint.stem.replace("epoch-epoch=", "ep").replace("best-epoch=", "best_ep")
        for split, dataset in datasets.items():
            print(f"{name} {label} {split}", flush=True)
            result = evaluate(agent, dataset, ScoreTable(scores_dir), args, dtype)
            np.savez_compressed(args.out_dir / f"{name}_{label}_{split}.npz", **result)
            row = {"run": name, "checkpoint": label, "split": split, **summarize(result)}
            print("  " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
            new_file = not csv_path.exists()
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            head_fields = sorted(k for k in row if k.startswith(HEAD_PREFIXES))
            if head_fields:
                heads_path = args.out_dir / f"heads_{name}.csv"
                new_file = not heads_path.exists()
                with open(heads_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["run", "checkpoint", "split", *head_fields],
                                            extrasaction="ignore")
                    if new_file:
                        writer.writeheader()
                    writer.writerow(row)


if __name__ == "__main__":
    main()
