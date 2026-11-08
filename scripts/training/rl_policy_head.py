"""
RL on the policy head alone (scene projection, trajectory bias, logit scale) of a vision_llm_lora checkpoint, from the
cached fused states (scripts/training/cache_policy_head_features.py), with rewards looked up from the precomputed
codebook scores (exp/codebook_scores_navtrain, the offline proxy's table: log replay, no two-frame comfort, no human
filter). Everything else of the model is frozen, so the sub-score heads and their shortlist stay those of the start
checkpoint. Objectives (navsim/agents/vision_llm_lora/vision_llm_lora_rl.py):

  exact: -E_pi[r] + beta * KL(pi || pi_start) over all K codebook entries
  grpo:  group-normalized policy gradient on --group-size entries sampled per scene, + beta * KL(pi || pi_start)

The candidate set is the full codebook, not the heads' shortlist: the heads were trained on the train scenes, so their
train-time shortlist is far better than at test time (checkpoint_evaluation.md, "Shortlist headroom for RL"). As in
policy-only training, dropout 0.1 is applied to the fused state (the query fusion's dropout). The start policy pi_start
is a frozen copy of the initial head.

Each epoch (and before training, epoch 0) the policy is evaluated on the val and train scenes with the same lookup as
eval_vision_llm_lora_checkpoints.py: argmax, E_pi[r], shortlist_<k> (most likely entry among the heads' k best), the
share of scenes scored 0, entropy, KL to the start and the share of changed argmax picks. Epoch 0 must reproduce the
offline evaluation of the start checkpoint. Outputs in --out-dir: summary.csv, policy_epoch<NN>.pt (the policy
parameters under their checkpoint keys). --export-epoch N writes a full checkpoint (the start checkpoint with that
epoch's policy head) to <out-dir>/checkpoints/epoch-epoch=NN.ckpt for the official evaluation and the offline script.

  python scripts/training/rl_policy_head.py --features-dir $NAVSIM_EXP_ROOT/rl_features/p3_aux_policy_ep03 \
      --out-dir $NAVSIM_EXP_ROOT/rl_policy/p3_aux_policy_exact_b0.1 --objective exact --beta 0.1
"""
import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from navsim.agents.vision_llm_lora.vision_llm_lora_rl import (
    PolicyHead,
    expected_reward_kl_loss,
    grpo_loss,
    selection_metrics,
)

SHORTLIST_SIZES = (4, 16, 64, 256, 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, default=None,
                        help="default: <features-dir>/../../codebook_scores_navtrain")
    parser.add_argument("--objective", choices=("exact", "grpo"), default="exact")
    parser.add_argument("--beta", type=float, default=0.1, help="KL weight")
    parser.add_argument("--group-size", type=int, default=8, help="grpo: entries sampled per scene")
    parser.add_argument("--advantage", choices=("std", "mean"), default="std",
                        help="grpo: (r - group mean) / group std (GRPO) or r - group mean (Dr. GRPO)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export-epoch", type=int, default=None,
                        help="only write the full checkpoint of this (already trained) epoch, no training")
    return parser.parse_args()


def load_features(features_dir: Path) -> Dict:
    shards = sorted(features_dir.glob("shard_*_of_*.pt"))
    counts = {p.name.split("_of_")[1] for p in shards}
    if len(counts) != 1 or len(shards) != int(counts.pop().removesuffix(".pt")):
        raise FileNotFoundError(f"incomplete shards in {features_dir}: {[p.name for p in shards]}")
    parts = [torch.load(p, weights_only=False) for p in shards]
    merged = {k: (sum((part[k] for part in parts), []) if isinstance(parts[0][k], list) else
                  torch.cat([part[k] for part in parts])) for k in parts[0]}
    return merged


def load_rewards(scores_dir: Path, tokens: List[str], logs: List[str]) -> torch.Tensor:
    """(N, K) float16 score of every codebook entry per scene; NaN rows for scenes the table does not have. Each log's
    file is read once, straight into one preallocated array (N x K float16, ~3.4 GB for navtrain)."""
    by_log: Dict[str, List[int]] = {}
    for index, log in enumerate(logs):
        by_log.setdefault(log, []).append(index)
    rewards = None
    for log, indices in sorted(by_log.items()):
        path = scores_dir / f"{log}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        rows = {str(t): i for i, t in enumerate(data["tokens"])}
        score = data["score"]  # decompressed once per log
        if rewards is None:
            rewards = np.full((len(tokens), score.shape[1]), np.nan, dtype=np.float16)
        for index in indices:
            row = rows.get(tokens[index])
            if row is not None:
                rewards[index] = score[row]
    return torch.from_numpy(rewards)


@torch.no_grad()
def evaluate(head: PolicyHead, ref: PolicyHead, data: Dict[str, torch.Tensor], batch_size: int = 1024) -> Dict:
    sums: Dict[str, float] = {}
    n = len(data["fused"])
    for i in range(0, n, batch_size):
        fused = data["fused"][i : i + batch_size].float()
        metrics = selection_metrics(head(fused), data["rewards"][i : i + batch_size], data["shortlist"][i : i + batch_size],
                                    SHORTLIST_SIZES, ref_logits=ref(fused))
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + float(v.sum())
    return {k: v / n for k, v in sums.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    head_info = torch.load(args.features_dir / "head.pt", weights_only=False)
    if args.export_epoch is not None:
        policy = torch.load(args.out_dir / f"policy_epoch{args.export_epoch:02d}.pt", weights_only=True)
        state: Dict[str, torch.Tensor] = {}
        for checkpoint in head_info["checkpoints"]:
            state.update(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
        assert set(policy) <= set(state), sorted(set(policy) - set(state))
        state.update(policy)
        out = args.out_dir / "checkpoints" / f"epoch-epoch={args.export_epoch:02d}.ckpt"
        out.parent.mkdir(exist_ok=True)
        torch.save({"state_dict": state, "merged_from": [*head_info["checkpoints"], f"rl policy epoch {args.export_epoch}"]},
                   out)
        print(f"{out}: {len(state)} keys")
        return

    device = torch.device("cuda")
    features = load_features(args.features_dir)
    scores_dir = args.scores_dir or args.features_dir.parent.parent / "codebook_scores_navtrain"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    head = PolicyHead(head_info["trajectory_embeddings"], in_dim=features["fused"].shape[1])
    head.load_from_checkpoint_state(head_info["policy_state"])
    head.to(device)

    rewards = load_rewards(scores_dir, features["token"], features["log"])
    split = np.array(features["split"])
    scored = ~torch.isnan(rewards[:, 0]).numpy()
    data = {}
    for name in ("train", "val"):
        index = torch.from_numpy(np.flatnonzero((split == name) & scored))
        data[name] = {
            "fused": features["fused"][index].to(device),
            "shortlist": features["shortlist"][index].to(device),
            "argmax": features["argmax"][index].to(device),
            "rewards": rewards[index].to(device),
        }
        print(f"{name}: {len(index)} scenes with rewards ({int(((split == name) & ~scored).sum())} without)")

    ref = PolicyHead(head_info["trajectory_embeddings"], in_dim=features["fused"].shape[1]).to(device)
    ref.load_state_dict(head.state_dict())
    ref.requires_grad_(False)
    with torch.no_grad():  # PolicyHead must reproduce the full model's policy (bf16 there, fp32 here)
        agree = float(torch.cat([(head(data["val"]["fused"][i : i + 1024].float()).argmax(-1)
                                  == data["val"]["argmax"][i : i + 1024]).float()
                                 for i in range(0, len(data["val"]["fused"]), 1024)]).mean())
    print(f"argmax agreement of PolicyHead with the model (val): {agree:.4f}")

    decay = [p for n, p in head.named_parameters() if n == "scene_proj.weight"]
    no_decay = [p for n, p in head.named_parameters() if n != "scene_proj.weight"]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95), eps=1e-5)
    train = data["train"]
    steps_per_epoch = (len(train["fused"]) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(args.warmup_fraction * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, 0.5 * (1 + np.cos(np.pi * min(s, total_steps) / total_steps))))
    generator = torch.Generator(device=device).manual_seed(args.seed)
    dropout = torch.nn.Dropout(args.dropout)

    summary_path = args.out_dir / "summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    def log_epoch(epoch: int, train_stats: Dict[str, float]) -> None:
        head.eval()
        row = {"epoch": epoch, "objective": args.objective, "beta": args.beta, "lr": args.lr, "epochs": args.epochs,
               "group_size": args.group_size if args.objective == "grpo" else 0,
               "advantage": args.advantage if args.objective == "grpo" else "", **train_stats}
        for name in ("val", "train"):
            row.update({f"{name}_{k}": v for k, v in evaluate(head, ref, data[name]).items()})
        print("  " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        new_file = not summary_path.exists()
        with open(summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        torch.save(head.checkpoint_state(), args.out_dir / f"policy_epoch{epoch:02d}.pt")

    stat_keys = ["loss", "expected_reward", "kl", *(["sampled_reward"] if args.objective == "grpo" else [])]
    log_epoch(0, {f"train_batch_{k}": float("nan") for k in stat_keys})
    for epoch in range(1, args.epochs + 1):
        head.train()
        start, totals = time.time(), {}
        order = torch.randperm(len(train["fused"]), device=device, generator=generator)
        for step in range(steps_per_epoch):
            batch = order[step * args.batch_size : (step + 1) * args.batch_size]
            fused = train["fused"][batch].float()
            with torch.no_grad():
                ref_logits = ref(fused)
            logits = head(dropout(fused))
            if args.objective == "exact":
                loss, stats = expected_reward_kl_loss(logits, train["rewards"][batch], ref_logits, args.beta)
            else:
                loss, stats = grpo_loss(logits, train["rewards"][batch], ref_logits, args.beta, args.group_size,
                                        generator=generator, normalize_std=args.advantage == "std")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            for k, v in {"loss": loss.detach(), **stats}.items():
                totals[f"train_batch_{k}"] = totals.get(f"train_batch_{k}", 0.0) + float(v)
        print(f"epoch {epoch}: {steps_per_epoch} steps in {time.time() - start:.0f} s", flush=True)
        log_epoch(epoch, {k: v / steps_per_epoch for k, v in totals.items()})


if __name__ == "__main__":
    main()
