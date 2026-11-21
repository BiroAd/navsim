"""Aux BEV decoder on navtrain val: metrics, and predicted / target maps of a few scenes for figures.

The decoder (vision_llm_lora_aux.py) exists only in runs trained with aux losses (p3_aux and its seeds). With
--backbone-checkpoint the run's LoRA / heads / planner parameters are replaced by another model's (e.g. RL-3) while the
aux decoder stays the aux run's: this measures whether the other model's image-token states still carry what the
decoder reads (a drift check; nothing is trained).

Metrics, over all evaluated scenes:
  bev_miou            dataset-level IoU per semantic class (summed intersections / unions), mean over classes
  bev_miou_batchmean  the training log's aux/bev_miou (mean over batches of the per-batch mIoU), for comparison with W&B
  drivable_iou        dataset-level IoU of drivable area (logit > 0 vs target > 0.5)
  lane_heading_acc    share of lane cells whose predicted direction bin is within 1 bin (10 deg) of the target
  python scripts/evaluation/eval_aux_maps.py --aux-run-dir $E/p3_aux/2026.11.05.00.00.51 --aux-epoch 5 \
      --aux-target-root <mounted aux_targets_navtrain.sqfs> --out-dir $E/aux_eval --name p3_aux_ep05 [--backbone-checkpoint ...]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import sys

from navsim.agents.vision_llm_lora.vision_llm_lora_aux import AUX_HEADING_BINS, aux_output_channels
from navsim.agents.vision_llm_lora.vision_llm_lora_features import NO_LANE_HEADING, VisionLLMLoRAAuxTargetBuilder
from navsim.planning.training.dataset import CacheOnlyDataset
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_vision_llm_lora_checkpoints import build_agent, load_trainable  # noqa: E402


# bev_semantic_classes of VisionLLMLoRAConfig (TransFuser's): index -> name (0 = none)
SEMANTIC_NAMES = {1: "road (lanes, intersections)", 2: "walkways", 3: "lane centrelines", 4: "static objects",
                  5: "vehicles", 6: "pedestrians"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aux-run-dir", type=Path, required=True)
    p.add_argument("--aux-epoch", type=int, required=True)
    p.add_argument("--aux-target-root", required=True, help="mounted aux_targets_navtrain.sqfs")
    p.add_argument("--backbone-checkpoint", type=Path, nargs="*", default=[],
                   help="checkpoints whose non-aux parameters replace the aux run's (loaded in order)")
    p.add_argument("--name", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-scenes", type=int, default=3000, help="evenly spaced val scenes (0: all 18,179)")
    p.add_argument("--save-scenes", type=int, default=24, help="random val scenes whose maps are saved")
    p.add_argument("--save-tokens", nargs="*", default=[], help="further tokens to save (must be among the evaluated)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    cfg = OmegaConf.load(args.aux_run_dir / "code" / "hydra" / "config.yaml")
    cfg.agent.config.aux_target_cache_path = args.aux_target_root
    agent = build_agent(cfg)
    assert agent._vlm_model.aux_decoder is not None, "the aux run has no aux decoder (aux_*_loss_weight all 0)"
    load_trainable(agent, [args.aux_run_dir / "checkpoints" / "epochs" / f"epoch-epoch={args.aux_epoch:02d}.ckpt"])
    replaced = 0
    for ckpt in args.backbone_checkpoint:
        state = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
        state = {k.removeprefix("agent.").replace("_vlm_model._orig_mod.", "_vlm_model."): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if "aux_decoder" not in k}
        own = dict(agent.named_parameters())
        common = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        result = agent.load_state_dict(common, strict=False)
        lora = sum("lora_" in k for k in common)
        print(f"backbone {ckpt}: {len(common)} tensors loaded ({lora} LoRA), {len(state) - len(common)} not in this model")
        assert lora > 0, "no LoRA tensors loaded from the backbone checkpoint"
        replaced += len(common)
    agent.eval().cuda()
    dtype = getattr(torch, agent._config.torch_dtype)
    aux_builder = [b for b in agent.get_target_builders() if isinstance(b, VisionLLMLoRAAuxTargetBuilder)]
    dataset = CacheOnlyDataset(cfg.cache_path, agent.get_feature_builders(), aux_builder, log_names=list(cfg.val_logs))
    n = len(dataset)
    idx = np.arange(n) if args.max_scenes in (0, None) or args.max_scenes >= n else \
        np.linspace(0, n - 1, args.max_scenes).round().astype(int)
    tokens = [dataset.tokens[i] for i in idx]
    rng = np.random.default_rng(args.seed)
    save = set(rng.choice(tokens, size=min(args.save_scenes, len(tokens)), replace=False)) | set(args.save_tokens)
    print(f"val: {n} scenes, evaluating {len(idx)}, saving maps of {len(save)}", flush=True)

    C = aux_output_channels(agent._config)["aux_bev_semantic"]
    inter, union = np.zeros(C), np.zeros(C)
    d_int = d_uni = h_ok = h_n = 0
    batch_miou, saved = [], {k: [] for k in ["token", "log", "sem_pred", "sem_target", "drv_prob", "drv_target",
                                              "head_pred", "head_target"]}
    loader = DataLoader(Subset(dataset, idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    seen, t0 = 0, time.time()
    for features, targets in loader:
        toks = tokens[seen : seen + len(targets["bev_semantic_map"])]
        seen += len(toks)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
            pred = agent.forward(features, None, 1.0)
        sem = pred["aux_bev_semantic"].float().argmax(1).cpu()
        sem_t = targets["bev_semantic_map"].long()
        ious = []
        for c in range(1, C):
            i_c = ((sem == c) & (sem_t == c)).sum().item()
            u_c = ((sem == c) | (sem_t == c)).sum().item()
            inter[c] += i_c
            union[c] += u_c
            if (sem_t == c).any():
                ious.append(i_c / max(u_c, 1))
        batch_miou.append(np.mean(ious) if ious else 0.0)
        drv = pred["aux_drivable_area"].float()[:, 0].cpu()
        drv_t = targets["drivable_area_map"].float()
        d_int += ((drv > 0) & (drv_t > 0.5)).sum().item()
        d_uni += ((drv > 0) | (drv_t > 0.5)).sum().item()
        head = pred["aux_lane_heading"].float().argmax(1).cpu()
        head_t = targets["lane_heading_map"].long()
        on = head_t != NO_LANE_HEADING
        diff = (head - head_t).abs()
        diff = torch.minimum(diff, AUX_HEADING_BINS - diff)
        h_ok += (diff[on] <= 1).sum().item()
        h_n += on.sum().item()
        for j, t in enumerate(toks):
            if t in save:
                saved["token"].append(t)
                saved["log"].append(dataset._valid_cache_paths[t].parent.name)
                saved["sem_pred"].append(sem[j].numpy().astype(np.uint8))
                saved["sem_target"].append(sem_t[j].numpy().astype(np.uint8))
                saved["drv_prob"].append(drv[j].sigmoid().numpy().astype(np.float16))
                saved["drv_target"].append(drv_t[j].numpy().astype(np.uint8))
                saved["head_pred"].append(head[j].numpy().astype(np.uint8))
                saved["head_target"].append(head_t[j].numpy().astype(np.uint8))
        if seen % (args.batch_size * 40) == 0:
            print(f"  {seen}/{len(idx)} scenes, {seen / (time.time() - t0):.1f} scenes/s", flush=True)
    per_class = {int(c): float(inter[c] / union[c]) for c in range(1, C) if union[c] > 0}
    metrics = {"name": args.name, "aux_run": str(args.aux_run_dir), "aux_epoch": args.aux_epoch,
               "backbone": [str(c) for c in args.backbone_checkpoint], "backbone_tensors_replaced": replaced,
               "scenes": int(seen), "bev_miou": float(np.mean(list(per_class.values()))),
               "bev_miou_batchmean": float(np.mean(batch_miou)), "bev_iou_per_class": per_class,
               "drivable_iou": d_int / max(d_uni, 1), "lane_heading_acc": h_ok / max(h_n, 1),
               "semantic_classes": SEMANTIC_NAMES}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"metrics_{args.name}.json").write_text(json.dumps(metrics, indent=1))
    np.savez_compressed(args.out_dir / f"maps_{args.name}.npz", **{k: np.array(v) for k, v in saved.items()})
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("bev_iou_per_class", "semantic_classes")}))


if __name__ == "__main__":
    main()
