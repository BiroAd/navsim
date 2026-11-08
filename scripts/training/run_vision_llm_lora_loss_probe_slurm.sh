#!/bin/bash
#SBATCH --job-name=vlm_loss_probe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.log
# Loss balance probe (scripts/training/probe_vision_llm_lora_loss_balance.py) for each <checkpoint> argument, with the
# model of training run RUN (relative to $NAVSIM_EXP_ROOT) and all auxiliary losses on. Mounts the cached aux targets.
#   RUN=p3_heads/2026.11.04.00.53.01 sbatch scripts/training/run_vision_llm_lora_loss_probe_slurm.sh <ckpt> [<ckpt> ...]
# Results: $OUT_DIR/<run>__<checkpoint>.json and the log.
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${NAVSIM_EXP_ROOT:?}" "${OPENSCENE_DATA_ROOT:?}" "${RUN:?}"

OUT_DIR=${OUT_DIR:-$NAVSIM_EXP_ROOT/loss_probe}
EXTRA_ARGS=${EXTRA_ARGS:-}  # e.g. "--warmup-steps 50 --probe-batches 4" for a smoke test
AUX_IMAGE=${AUX_IMAGE:-$OPENSCENE_DATA_ROOT/images/aux_targets_navtrain.sqfs}
AUX_MOUNT=$NAVSIM_EXP_ROOT/mnt/aux_${SLURM_JOB_ID:-$$}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

mkdir -p "$AUX_MOUNT" "$OUT_DIR"
squashfuse "$AUX_IMAGE" "$AUX_MOUNT"
trap 'fusermount -u "$AUX_MOUNT"; rmdir "$AUX_MOUNT"' EXIT
echo "Mounted $AUX_IMAGE at $AUX_MOUNT: $(ls "$AUX_MOUNT" | wc -l) logs"

for ckpt in "$@"; do
    label=$(echo "${ckpt#$NAVSIM_EXP_ROOT/}" | tr '/=' '__')
    echo "=== $ckpt"
    python $NAVSIM_DEVKIT_ROOT/scripts/training/probe_vision_llm_lora_loss_balance.py \
        --run-dir $NAVSIM_EXP_ROOT/$RUN \
        --checkpoint "$ckpt" \
        --aux-root "$AUX_MOUNT" \
        --out "$OUT_DIR/${RUN%%/*}__${label%.ckpt}.json" \
        $EXTRA_ARGS
done
