"""
Scorer shortlists of the current and past frames of every navtrain scene, for an RL previous plan chosen as at a
shortlist evaluation ("the policy's most likely entry among the scorer's top k at t - 0.5 s") and for runs with history.

A scorer checkpoint (sub-score heads, e.g. p3_aux_ctrl best epoch 2, the control scorer of the shortlist evaluations)
is run, in eval mode, on each requested frame of each scene of the run config's train and val logs:
  frame 0: the current frame (training cache)
  frame k (1, 2, 3 = t - 0.5, t - 1.0, t - 1.5 s): camera tiles vision_llm_history_t<k> and velocity, acceleration and
           driving command of the k-th past ego status (vision_llm_ego_history, each frame's own ego frame) from the
           past-frame cache, i.e. the inputs the agent would have had at that frame.
Per frame the --shortlist-size entries with the best predicted total (aggregate_pdm_scores of the sub-score heads, as
the "shortlist" selection), best first; the shortlist of any k <= --shortlist-size. Predicted scores only: the
evaluator's scores are not used (the agent has none at test time). Sharded by log like cache_policy_head_features.py:

  <out-dir>/shard_<i>_of_<n>.pt  {token, log, split: lists; frames: list; shortlist: (N, F, S) int16}

  python scripts/training/cache_scorer_shortlists.py --run-dir $NAVSIM_EXP_ROOT/p3_aux_ctrl/2026.11.09.23.55.17 \
      --checkpoint checkpoints/best/best-epoch=02.ckpt --out-dir $NAVSIM_EXP_ROOT/scorer_shortlists/p3_aux_ctrl_ep02 \
      --frames 0 1 2 --history-cache-path <mount>[:<mount> ...] --shard 0 --num-shards 6
"""
import argparse
import importlib.util
import time
from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from navsim.agents.vision_llm_lora.vision_llm_lora_features import (
    VisionLLMLoRAEgoHistoryBuilder,
    VisionLLMLoRAHistoryFrameBuilder,
    normalize_uint8_camera,
)
from navsim.planning.training.dataset import CacheOnlyDataset

torch.multiprocessing.set_sharing_strategy("file_system")  # as in eval_vision_llm_lora_checkpoints.py

_EVAL_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "eval_vision_llm_lora_checkpoints.py"
_spec = importlib.util.spec_from_file_location("eval_vision_llm_lora_checkpoints", _EVAL_PATH)
evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluation)


def frame_features(features: Dict[str, torch.Tensor], k: int, normalization: str) -> Dict[str, torch.Tensor]:
    """The model inputs of past frame k (1 = t - 0.5 s) from the history features; k = 0: the current frame."""
    if k == 0:
        return features
    past = dict(features)
    past["camera_feature"] = normalize_uint8_camera(features[f"camera_history_t{k}"], normalization).to(
        features["camera_feature"].dtype
    )
    past["velocity"] = features["ego_history_velocity"][:, k - 1]
    past["acceleration"] = features["ego_history_acceleration"][:, k - 1]
    past["driving_command"] = features["ego_history_command"][:, k - 1]
    return past


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="scorer run exp/<run>/<date> with code/ (config)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="relative to --run-dir or absolute")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 1, 2], choices=[0, 1, 2, 3])
    parser.add_argument("--shortlist-size", type=int, default=1024)
    parser.add_argument("--history-cache-path", default=None,
                        help="past-frame cache (squashfs mounts joined by os.pathsep); needed for frames > 0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=None, help="per split (smoke test)")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    past = sorted(k for k in set(args.frames) if k > 0)
    if past and not args.history_cache_path:
        raise SystemExit("frames > 0 read the past-frame cache: give --history-cache-path (HISTORY_CACHE=1)")
    cfg = OmegaConf.load(args.run_dir / "code" / "hydra" / "config.yaml")
    model_cfg = cfg.agent.config
    if model_cfg.get("temporal_history_frame", 0) or model_cfg.get("ego_history_tokens", False):
        raise SystemExit("the scorer reads past frames itself; only current-frame scorers are supported")
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else args.run_dir / args.checkpoint
    agent = evaluation.build_agent(cfg)
    checkpoints = [*evaluation.base_checkpoints(cfg), checkpoint]
    evaluation.load_trainable(agent, checkpoints)
    agent.eval().cuda()
    dtype = getattr(torch, agent._config.torch_dtype)
    if getattr(agent._vlm_model.traj_classifier.classifier, "num_score_heads", 0) == 0:
        raise ValueError("the checkpoint has no sub-score heads (no shortlist)")

    feature_builders = agent.get_feature_builders()
    feature_builders += [VisionLLMLoRAHistoryFrameBuilder(agent._config, k, args.history_cache_path) for k in past]
    if past:
        feature_builders.append(VisionLLMLoRAEgoHistoryBuilder(args.history_cache_path))
    target_builders = agent.get_target_builders()[:1]

    logs = sorted([("train", log) for log in cfg.train_logs] + [("val", log) for log in cfg.val_logs], key=lambda x: x[1])
    mine = logs[args.shard :: args.num_shards]
    out = {"token": [], "log": [], "split": [], "shortlist": []}
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
            per_frame = []
            for k in args.frames:
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
                    predictions = agent.forward(frame_features(features, k, agent._config.image_normalization), None, 1.0)
                total = evaluation.predicted_total(predictions["pdm_score_logits"])  # (B, K)
                per_frame.append(total.topk(args.shortlist_size, dim=-1).indices.short().cpu())
            out["token"] += tokens
            out["log"] += [dataset._valid_cache_paths[t].parent.name for t in tokens]
            out["split"] += [split] * len(tokens)
            out["shortlist"].append(torch.stack(per_frame, dim=1))  # (B, F, S)
            if batch_index % 100 == 0:
                print(f"  {seen}/{len(dataset)} scenes, {seen / (time.time() - start):.1f} scenes/s", flush=True)
            if args.max_scenes is not None and seen >= args.max_scenes:
                break
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = args.out_dir / f"shard_{args.shard}_of_{args.num_shards}.pt"
    torch.save(
        {
            "token": out["token"], "log": out["log"], "split": out["split"], "frames": list(args.frames),
            "shortlist": torch.cat(out["shortlist"]), "checkpoints": [str(c) for c in checkpoints],
        },
        shard_path,
    )
    print(f"{shard_path}: {len(out['token'])} scenes, frames {args.frames}, "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
