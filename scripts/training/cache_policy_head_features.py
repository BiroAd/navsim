"""
Cache the policy head's inputs of a vision_llm_lora checkpoint for RL on the policy head alone
(scripts/training/rl_policy_head.py). With the LoRA, planner queries, query fusion and sub-score heads frozen, the policy
logits depend on the scene only through the fused hidden state (TrajectoryClassifier.query_fusion output, (1024,)), so
one forward pass of the 4B model per scene is enough for any number of RL epochs.

Per scene (navtrain train and val logs of the run's config, eval mode, i.e. without dropout): token, log, split, the fused
state (fp16), the 1024 entries with the best predicted total of the sub-score heads (best first; the shortlist of any
k <= 1024) and the full model's argmax entry (to check PolicyHead against the model). Once per checkpoint: the trajectory
embeddings (K, D), the policy parameters (vision_llm_lora_rl.POLICY_HEAD_KEYS) and the checkpoints the agent was loaded
from. The scenes are split into shards by log (--shard i --num-shards n), one file each, so several GPU jobs can share
the work:

  <out-dir>/head.pt, <out-dir>/shard_<i>_of_<n>.pt

  python scripts/training/cache_policy_head_features.py --run-dir $NAVSIM_EXP_ROOT/p3_aux_policy/2026.11.06.23.42.11 \
      --checkpoint checkpoints/merged/epoch-epoch=03.ckpt --out-dir $NAVSIM_EXP_ROOT/rl_features/p3_aux_policy_ep03 \
      --shard 0 --num-shards 3
"""
import argparse
import importlib.util
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from navsim.agents.vision_llm_lora.vision_llm_lora_rl import POLICY_HEAD_KEYS
from navsim.planning.training.dataset import CacheOnlyDataset

torch.multiprocessing.set_sharing_strategy("file_system")  # as in eval_vision_llm_lora_checkpoints.py

_EVAL_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "eval_vision_llm_lora_checkpoints.py"
_spec = importlib.util.spec_from_file_location("eval_vision_llm_lora_checkpoints", _EVAL_PATH)
evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluation)

SHORTLIST_SIZE = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="exp/<run>/<date> folder with code/ (model config)")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="full checkpoint (for a policy-only run: the merged one), relative to --run-dir or absolute")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=None, help="per split (smoke test)")
    parser.add_argument("--history-cache-path", default=None,
                        help="past-frame cache (one or more squashfs mounts joined by os.pathsep) for runs with "
                             "temporal_history_frame / ego_history_tokens; the run config holds its training job's "
                             "temporary mounts")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.run_dir / "code" / "hydra" / "config.yaml")
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else args.run_dir / args.checkpoint
    model_cfg = cfg.agent.config
    if model_cfg.get("temporal_history_frame", 0) or model_cfg.get("ego_history_tokens", False):
        if args.history_cache_path is None:
            raise SystemExit("this run reads past frames: give --history-cache-path (HISTORY_CACHE=1 in the SLURM script)")
        model_cfg.history_cache_path = args.history_cache_path
    agent = evaluation.build_agent(cfg)
    checkpoints = [*evaluation.base_checkpoints(cfg), checkpoint]
    evaluation.load_trainable(agent, checkpoints)
    agent.eval().cuda()
    dtype = getattr(torch, agent._config.torch_dtype)
    model = agent._vlm_model
    classifier = model.traj_classifier
    if getattr(classifier.classifier, "num_score_heads", 0) == 0:
        raise ValueError("the checkpoint has no sub-score heads (no shortlist)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    head_path = args.out_dir / "head.pt"
    if args.shard == 0 or not head_path.exists():
        params = dict(agent.named_parameters())
        state = {key: params[key.removeprefix("agent.")].detach().float().cpu() for key in POLICY_HEAD_KEYS.values()}
        torch.save(
            {
                "trajectory_embeddings": classifier.classifier.embed_trajectories().float().cpu(),
                "policy_state": state,
                "checkpoints": [str(c) for c in checkpoints],
                "run_dir": str(args.run_dir),
                "train_logs": list(cfg.train_logs),
                "val_logs": list(cfg.val_logs),
            },
            head_path,
        )

    captured = {}
    hook = classifier.query_fusion.register_forward_hook(lambda module, inputs, output: captured.update(fused=output))
    logs = sorted([("train", log) for log in cfg.train_logs] + [("val", log) for log in cfg.val_logs], key=lambda x: x[1])
    mine = logs[args.shard :: args.num_shards]
    feature_builders, target_builders = agent.get_feature_builders(), agent.get_target_builders()[:1]
    out = {"token": [], "log": [], "split": [], "fused": [], "shortlist": [], "argmax": []}
    for split in ("train", "val"):
        log_names = [log for s, log in mine if s == split]
        if not log_names:
            continue
        dataset = CacheOnlyDataset(cfg.cache_path, feature_builders, target_builders, log_names=log_names)
        print(f"shard {args.shard}/{args.num_shards} {split}: {len(dataset)} scenes in {len(log_names)} logs", flush=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        start, seen = time.time(), 0
        for batch_index, (features, targets) in enumerate(loader):
            tokens = dataset.tokens[seen : seen + len(targets["trajectory"])]
            seen += len(tokens)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
                predictions = agent.forward(features, None, 1.0)
            total = evaluation.predicted_total(predictions["pdm_score_logits"])  # (B, K)
            out["token"] += tokens
            out["log"] += [dataset._valid_cache_paths[t].parent.name for t in tokens]
            out["split"] += [split] * len(tokens)
            out["fused"].append(captured["fused"].half().cpu())
            out["shortlist"].append(total.topk(SHORTLIST_SIZE, dim=-1).indices.short().cpu())
            out["argmax"].append(predictions["trajectory_logits"].argmax(-1).int().cpu())
            if batch_index % 100 == 0:
                print(f"  {seen}/{len(dataset)} scenes, {seen / (time.time() - start):.1f} scenes/s", flush=True)
            if args.max_scenes is not None and seen >= args.max_scenes:
                break
    hook.remove()
    shard_path = args.out_dir / f"shard_{args.shard}_of_{args.num_shards}.pt"
    torch.save({k: (v if isinstance(v[0], str) else torch.cat(v)) for k, v in out.items()}, shard_path)
    print(f"{shard_path}: {len(out['token'])} scenes, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
