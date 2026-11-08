"""
Loss balance probe for vision_llm_lora (1 GPU, a few minutes): how large each loss term is and how hard it pushes the
shared parameters, to choose loss weights.

1. Builds the model from a training run's Hydra config with all three auxiliary losses on, loads --checkpoint the way
   a phase handoff does (agent.initialize(): heads, planner queries, LoRA; missing keys keep their initialisation, e.g.
   a phase 1 checkpoint has no LoRA and lora_B stays 0) and reads train scenes from the training cache plus the
   mounted aux targets.
2. Warm-up: trains only the new aux decoder for --warmup-steps batches (everything else frozen), printing each aux
   loss, so the decoder is not random when its gradients are measured.
3. Probe: on --probe-batches further batches, with the run's trainable parameters unfrozen, computes every loss term
   separately (unweighted) and the norm of its gradient on the LLM LoRA, the vision LoRA and the planner queries.
   The weighted push of a term is weight x gradient norm; terms with equal pushes on the LoRA are balanced.

  python scripts/training/probe_vision_llm_lora_loss_balance.py --run-dir $NAVSIM_EXP_ROOT/p3_heads/<date> \
      --checkpoint <ckpt> --aux-root <mounted aux_targets_navtrain.sqfs> --out <json>
"""
import argparse
import dataclasses
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_loss import vision_llm_lora_loss
from navsim.planning.training.dataset import CacheOnlyDataset

# Each term on its own: config overrides that leave only that term in the loss (all weights 1 unless listed).
_ALL_OFF = dict(
    trajectory_token_loss_weight=0.0, expected_distance_loss_weight=0.0, pdm_score_loss_weight=1e-12,
    pdm_total_score_loss_weight=0.0, aux_bev_semantic_loss_weight=0.0, aux_drivable_area_loss_weight=0.0,
    aux_lane_heading_loss_weight=0.0,
)
TERMS = {
    "soft_ce": dict(trajectory_token_loss_weight=1.0),
    "expected_ade": dict(expected_distance_loss_weight=1.0),
    "score_bce": dict(pdm_score_loss_weight=1.0),
    "score_total_bce": dict(pdm_total_score_loss_weight=1.0),
    "aux_bev_semantic": dict(aux_bev_semantic_loss_weight=1.0),
    "aux_drivable_area": dict(aux_drivable_area_loss_weight=1.0),
    "aux_lane_heading": dict(aux_lane_heading_loss_weight=1.0),
}
AUX_KEYS = ("aux_bev_semantic", "aux_drivable_area", "aux_lane_heading")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="training run whose Hydra config builds the model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="loaded like agent.checkpoint_path")
    parser.add_argument("--aux-root", type=Path, required=True, help="mounted aux_targets_navtrain.sqfs")
    parser.add_argument("--out", type=Path, required=True, help="JSON with the per-term statistics")
    parser.add_argument("--warmup-steps", type=int, default=600)
    parser.add_argument("--probe-batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-stride", type=int, default=10, help="every n-th train log (default ~100 logs)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_agent(cfg, checkpoint: Path, aux_root: Path):
    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)
    known = {f.name for f in dataclasses.fields(VisionLLMLoRAConfig)} | {"_target_", "_convert_"}
    agent_cfg["config"] = {k: v for k, v in agent_cfg["config"].items() if k in known}
    agent_cfg["config"].update(
        aux_bev_semantic_loss_weight=1.0, aux_drivable_area_loss_weight=1.0, aux_lane_heading_loss_weight=1.0,
        aux_target_cache_path=str(aux_root),
    )
    if agent_cfg["config"].get("pdm_score_loss_weight", 0) <= 0:
        raise ValueError("the run needs score heads (pdm_score_loss_weight > 0) and cached PDM score targets")
    agent_cfg["checkpoint_path"] = str(checkpoint)
    agent = instantiate(agent_cfg)
    agent.initialize()
    return agent


def param_groups(agent) -> Dict[str, List[torch.nn.Parameter]]:
    groups = defaultdict(list)
    for name, p in agent.named_parameters():
        if not p.requires_grad:
            continue
        if "lora" in name and "visual" in name:
            groups["vision_lora"].append(p)
        elif "lora" in name:
            groups["llm_lora"].append(p)
        elif "planner_queries" in name or "query_pos_embed" in name:
            groups["planner_queries"].append(p)
    return dict(groups)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = OmegaConf.load(args.run_dir / "code" / "hydra" / "config.yaml")
    agent = build_agent(cfg, args.checkpoint, args.aux_root).cuda()
    config: VisionLLMLoRAConfig = agent._config
    dtype = getattr(torch, config.torch_dtype)

    logs = list(cfg.train_logs)[:: args.log_stride]
    dataset = CacheOnlyDataset(cfg.cache_path, agent.get_feature_builders(), agent.get_target_builders(), logs)
    print(f"{len(dataset)} train scenes in {len(dataset.log_names)} logs", flush=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True,
        persistent_workers=True, generator=torch.Generator().manual_seed(args.seed),
    )
    batches = iter(loader)

    trainable = {name: p for name, p in agent.named_parameters() if p.requires_grad}
    decoder_params = [p for name, p in trainable.items() if "aux_decoder" in name]
    print(f"trainable {sum(p.numel() for p in trainable.values()):,}, aux decoder {sum(p.numel() for p in decoder_params):,}")
    result = {"run_dir": str(args.run_dir), "checkpoint": str(args.checkpoint), "batch_size": args.batch_size,
              "warmup": [], "terms": {}}

    # ── 1. decoder warm-up, everything else frozen ──
    for p in trainable.values():
        p.requires_grad_(False)
    for p in decoder_params:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(decoder_params, lr=config.aux_lr_multiplier * 5e-4, weight_decay=0.01)
    aux_on = dict(aux_bev_semantic_loss_weight=1.0, aux_drivable_area_loss_weight=1.0, aux_lane_heading_loss_weight=1.0)
    warmup_config = dataclasses.replace(config, **{**_ALL_OFF, **aux_on})
    agent.train()
    running, start = defaultdict(float), time.time()
    for step in range(1, args.warmup_steps + 1):
        features, targets = (to_device(x, "cuda") for x in next(batches))
        with torch.autocast("cuda", dtype=dtype):
            predictions = agent.forward(features, None, 1.0)
        loss, *_, extra = vision_llm_lora_loss(targets, predictions, warmup_config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for key in AUX_KEYS:
            running[key] += extra[f"loss_terms/{key}"].item()
        for key in ("aux/bev_miou", "aux/drivable_iou", "aux/lane_heading_acc"):
            running[key] += extra[key].item()
        if step % 50 == 0:
            row = {"step": step, **{k: v / 50 for k, v in running.items()}}
            result["warmup"].append(row)
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  f"({time.time() - start:.0f} s)", flush=True)
            running.clear()

    # ── 2. per-term loss and gradient norms on the shared parameters ──
    for p in trainable.values():
        p.requires_grad_(True)
    groups = param_groups(agent)
    print({g: f"{sum(p.numel() for p in ps):,}" for g, ps in groups.items()}, flush=True)
    stats = {t: defaultdict(list) for t in TERMS}
    for i in range(args.probe_batches):
        features, targets = (to_device(x, "cuda") for x in next(batches))
        with torch.autocast("cuda", dtype=dtype):
            predictions = agent.forward(features, None, 1.0)
        for term, weights in TERMS.items():
            loss, *_ = vision_llm_lora_loss(targets, predictions, dataclasses.replace(config, **{**_ALL_OFF, **weights}))
            stats[term]["loss"].append(loss.item())
            for group, params in groups.items():
                grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                norm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads if g is not None) + 0.0)
                stats[term][f"grad_{group}"].append(float(norm))
        del predictions
        print(f"probe batch {i + 1}/{args.probe_batches}", flush=True)

    print(f"\n{'term':20s} {'loss':>8s} " + " ".join(f"{'grad ' + g:>22s}" for g in groups))
    for term, values in stats.items():
        mean = {k: sum(v) / len(v) for k, v in values.items()}
        result["terms"][term] = mean
        print(f"{term:20s} {mean['loss']:8.4f} " + " ".join(f"{mean['grad_' + g]:22.3e}" for g in groups))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
