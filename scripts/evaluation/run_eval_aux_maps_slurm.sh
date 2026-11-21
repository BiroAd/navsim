#!/bin/bash
#SBATCH --job-name=vlm_aux_maps
#SBATCH --partition=gpu,ai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=01:00:00
# Aux BEV decoder on navtrain val (scripts/evaluation/eval_aux_maps.py): mounts the cached aux targets, then evaluates
# p3_aux ep 5 (its own backbone) and the same decoder on the backbone of each checkpoint in BACKBONES (drift check:
# does RL-3 keep what the decoder reads?). Same scenes for every run. Memory as the checkpoint evaluation (31-53 GB).
#   sbatch scripts/evaluation/run_eval_aux_maps_slurm.sh
set -euo pipefail
cd "${NAVSIM_DEVKIT_ROOT:-/project/nr_e2e_vlm/navsim}"
source ./bashrc >/dev/null 2>&1
export PATH=/project/nr_e2e_vlm/envs/navsim_vlm/bin:$PATH HF_HUB_OFFLINE=1
E=$NAVSIM_EXP_ROOT
MOUNT=$E/mnt/aux_eval_${SLURM_JOB_ID:-$$}
mkdir -p "$MOUNT"
squashfuse "$OPENSCENE_DATA_ROOT/images/aux_targets_navtrain.sqfs" "$MOUNT"
trap 'fusermount -u "$MOUNT"; rmdir "$MOUNT"' EXIT
AUX_RUN=${AUX_RUN:-$E/p3_aux/2026.11.05.00.00.51}; AUX_EPOCH=${AUX_EPOCH:-5}; OUT=${OUT:-$E/aux_eval}
COMMON=(--aux-run-dir "$AUX_RUN" --aux-epoch "$AUX_EPOCH" --aux-target-root "$MOUNT" --out-dir "$OUT" "$@")
python scripts/evaluation/eval_aux_maps.py "${COMMON[@]}" --name "${AUX_NAME:-p3_aux_ep05}"
for spec in ${BACKBONES:-rl3_b0.005_ep01:$E/rl3_b0.005/2026.11.08.23.46.54/checkpoints/epochs/epoch-epoch=01.ckpt}; do
    python scripts/evaluation/eval_aux_maps.py "${COMMON[@]}" --name "${AUX_NAME:-p3_aux_ep05}_on_${spec%%:*}" \
        --backbone-checkpoint "${spec#*:}"
done
