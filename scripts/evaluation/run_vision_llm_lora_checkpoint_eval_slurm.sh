#!/bin/bash
#SBATCH --job-name=vlm_lora_ckpt_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.log
# Offline val evaluation (ADE/FDE, proxy PDM score from exp/codebook_scores_navtrain, entropy, train/val gap) of
# vision_llm_lora checkpoints on one GPU, from the training cache; see eval_vision_llm_lora_checkpoints.py.
# Each argument is <run folder relative to $NAVSIM_EXP_ROOT>:<comma-separated epochs>, e.g.
#   sbatch scripts/evaluation/run_vision_llm_lora_checkpoint_eval_slurm.sh \
#       p1c_scores/2026.11.01.00.34.10:3,5,7,10,14,19 p1c_base/2026.11.01.00.05.04:5,7,10
# Results are appended to $OUT_DIR/summary_<run>.csv; per-scene arrays go next to it. Several jobs can run at once.
# TRAIN_LOG_STRIDE=20 also evaluates every 20th train log (~4% of the train split) for the train/val gap in eval mode.
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${NAVSIM_EXP_ROOT:?}"

OUT_DIR=${OUT_DIR:-$NAVSIM_EXP_ROOT/checkpoint_eval}
TRAIN_LOG_STRIDE=${TRAIN_LOG_STRIDE:-20}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-8}
EXTRA_ARGS=${EXTRA_ARGS:-}  # further script options, e.g. "--max-scenes 64" for a smoke test
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

for spec in "$@"; do
    run=${spec%%:*}
    epochs=${spec#*:}
    echo "=== $run epochs $epochs"
    python $NAVSIM_DEVKIT_ROOT/scripts/evaluation/eval_vision_llm_lora_checkpoints.py \
        --run-dir $NAVSIM_EXP_ROOT/$run \
        --epochs ${epochs//,/ } \
        --out-dir $OUT_DIR \
        --train-log-stride $TRAIN_LOG_STRIDE \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        $EXTRA_ARGS
done
