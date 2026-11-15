"""
Tracker reference of every codebook entry for the EC reward (ec_torch.precompute_tracking; config.rl_ec_tracking_path):
interpolated poses (K, 41, 3) and the velocity / curvature profiles the official LQR tracker fits to them (K, 40 each),
computed once with the official numpy functions. ~1-2 min on one CPU.
    python scripts/training/precompute_ec_tracking.py \
        --codebook navsim/agents/vision_llm_lora/codebook/16384_8step.npy \
        --out $NAVSIM_EXP_ROOT/codebook/16384_8step_ec_tracking.npz
"""
import argparse
import time
from pathlib import Path

import numpy as np

from navsim.agents.vision_llm_lora.ec_torch import precompute_tracking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = time.time()
    tracking = precompute_tracking(np.load(args.codebook))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **tracking)
    print(f"{len(tracking['ref_poses'])} entries -> {args.out} ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
