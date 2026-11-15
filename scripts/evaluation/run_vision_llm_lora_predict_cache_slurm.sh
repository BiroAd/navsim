#!/bin/bash
#SBATCH --job-name=vlm_predict_val
#SBATCH --partition=gpu,ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.log
# Model outputs on the 18,179 navtrain validation scenes from the training feature cache
# (scripts/evaluation/predict_vision_llm_lora_cache.py), for choosing selection rules on held-out data.
#   RUN=rl3_b0.02/2026.11.08.21.48.32 CKPT=checkpoints/best/best-epoch=01.ckpt NAME=rl3_b0.02_ep01 \
#       sbatch scripts/evaluation/run_vision_llm_lora_predict_cache_slurm.sh
# Output: $NAVSIM_EXP_ROOT/predictions/<NAME>/navtrain_val/. 40G as the image-based prediction job (host peak ~21.5 GB).
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${NAVSIM_EXP_ROOT:?}" "${RUN:?}" "${CKPT:?}" "${NAME:?}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 15 \
    > "$NAVSIM_EXP_ROOT/predictions/logs/gpu_util_val_${NAME}_${SLURM_JOB_ID:-$$}.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
python $NAVSIM_DEVKIT_ROOT/scripts/evaluation/predict_vision_llm_lora_cache.py \
    --run-dir "$NAVSIM_EXP_ROOT/$RUN" --checkpoint "$CKPT" --out-dir "$NAVSIM_EXP_ROOT/predictions/$NAME" \
    --num-workers ${WORKERS:-8} "$@"
awk -F', ' '{gsub(/ %/, "", $4); s += $4; n++} END {if (n) printf "mean GPU utilization %.0f%% over %d samples\n", s / n, n}' \
    "$NAVSIM_EXP_ROOT/predictions/logs/gpu_util_val_${NAME}_${SLURM_JOB_ID:-$$}.csv"
