"""
Full checkpoints for a policy-only run (agent.config.train_policy_head_only), for the official evaluation.

Checkpoints leave frozen parameters out (agent_lightning_module.on_save_checkpoint), so a policy-only run's checkpoints
hold only the policy head (scene projection, trajectory bias, logit scale); its LoRA, sub-score heads and planner
queries are those of its start checkpoint (agent.checkpoint_path of the run). This script writes, per epoch, the start
checkpoint's state dict with the policy head of that epoch on top, to <run>/checkpoints/merged/epoch-epoch=NN.ckpt,
which agent.initialize() loads like any phase checkpoint (the offline evaluation, eval_vision_llm_lora_checkpoints.py,
does the same merge itself).

  python scripts/evaluation/merge_policy_only_checkpoint.py --run-dir $NAVSIM_EXP_ROOT/p3_aux_policy/<date> --epochs 1 3
"""
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.run_dir / "code" / "hydra" / "config.yaml")
    if not cfg.agent.config.get("train_policy_head_only", False):
        raise ValueError(f"{args.run_dir} is not a policy-only run (train_policy_head_only is not set)")
    start = Path(str(cfg.agent.checkpoint_path).strip("'"))
    base = torch.load(start, map_location="cpu", weights_only=True)["state_dict"]
    out_dir = args.run_dir / "checkpoints" / "merged"
    out_dir.mkdir(exist_ok=True)
    for epoch in args.epochs:
        source = args.run_dir / "checkpoints" / "epochs" / f"epoch-epoch={epoch:02d}.ckpt"
        policy = torch.load(source, map_location="cpu", weights_only=True)["state_dict"]
        missing = sorted(set(policy) - set(base))
        if missing:
            raise KeyError(f"{source}: keys not in the start checkpoint {start}: {missing}")
        merged = {**base, **policy}
        out = out_dir / source.name
        torch.save({"state_dict": merged, "merged_from": [str(start), str(source)]}, out)
        print(f"{out}: {len(base)} keys from {start.name}, {len(policy)} replaced from {source.name}")


if __name__ == "__main__":
    main()
