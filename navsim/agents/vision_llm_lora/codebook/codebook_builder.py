"""
codebook/build_codebook.py
──────────────────────────
Offline pre-processing: subsample the raw navsim trajectory codebook from
40 timesteps (4 s @ 10 Hz) down to 8 waypoints by taking every 5th frame.
 
Usage:
    python codebook/build_codebook.py \
        --input  codebook/16384.npy \
        --output codebook/16384_8step.npy
 
Input shape  : (K, 40, 3)  — K trajectory templates, 40 (x, y, yaw) steps
Output shape : (K,  8, 3)  — same templates at indices [4, 9, 14, …, 39] (pose i is at (i + 1) * 0.1 s)
"""
 
import argparse
 
import numpy as np
 
 
# Every 5th frame from a 40-step trajectory gives 8 waypoints.
# Index 4 = 0.5 s, index 9 = 1 s, …, index 39 = 4 s ahead.
STEP = 5
NUM_WAYPOINTS = 8
 
 
def build(input_path: str, output_path: str) -> None:
    arr = np.load(input_path)
    print(f"Loaded  : {arr.shape}  dtype={arr.dtype}  from {input_path!r}")
 
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"Expected shape (K, T, 3), got {arr.shape}. "
            "Each entry should be a full trajectory of (x, y, yaw) steps."
        )
 
    K, T_full, _ = arr.shape
 
    if T_full < NUM_WAYPOINTS * STEP:
        raise ValueError(
            f"Need at least {NUM_WAYPOINTS * STEP} timesteps to extract "
            f"{NUM_WAYPOINTS} waypoints with step={STEP}, but T={T_full}."
        )
 
    indices = np.arange(4, STEP * NUM_WAYPOINTS, STEP)   # [4, 9, 14, …, 39]
    print(f"Sampling indices: {indices.tolist()}")
 
    subsampled = arr[:, indices, :].astype(np.float32)    # (K, 8, 3)
    print(f"Saving  : {subsampled.shape}  to {output_path!r}")
 
    np.save(output_path, subsampled)
    print("Done.")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample trajectory codebook.")
    parser.add_argument("--input",  default="/root/workdir/biroa/navsim/navsim/agents/vision_llm_lora/codebook/16384.npy",
                        help="Path to raw (K, 40, 3) codebook.")
    parser.add_argument("--output", default="/root/workdir/biroa/navsim/navsim/agents/vision_llm_lora/codebook/16384_8step.npy",
                        help="Path for the (K, 8, 3) output.")
    args = parser.parse_args()
    build(args.input, args.output)
