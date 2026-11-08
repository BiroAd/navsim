#!/bin/bash
#SBATCH --job-name=vlm_lora_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=7-00:00:00
#SBATCH --output=%x-%j.log
# Trains vision_llm_lora on navtrain from the feature cache, with bf16 and flash attention (set in the agent YAML).
# Lightning under SLURM expects one srun task per GPU, hence ntasks-per-node = number of GPUs.
# Submit from a shell with bashrc sourced and the env active; phase settings go in as extra Hydra args, e.g.
#   sbatch scripts/training/run_vision_llm_lora_agent_training_slurm.sh \
#       agent.checkpoint_path=\'$NAVSIM_EXP_ROOT/<phase1_run>/checkpoints/best/<ckpt>\' ++agent.config.training_phase=2
# BATCH_SIZE / ACCUM change the per-GPU batch at a fixed effective batch; total_training_steps follows so the LR
# schedule keeps its length (2660 = 85112 / (8 x 4) for the defaults; 10639 for the original batch 2, accum 16).
# Batch 8 fits phase 1 only (39.5 of 40.5 GB per A100-40GB, batch_probe jobs 14677362/14677458); phases 2 and 3 run
# out of memory at 8, so use BATCH_SIZE=4 ACCUM=8 for them (30.3 / 35.0 GB).
# AUX_TARGETS=1 mounts the cached auxiliary perception targets ($OPENSCENE_DATA_ROOT/images/aux_targets_navtrain.sqfs)
# for this job and passes their path; needed with ++agent.config.aux_*_loss_weight > 0.
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${OPENSCENE_DATA_ROOT:?}" "${NUPLAN_MAPS_ROOT:?}" "${NAVSIM_EXP_ROOT:?}"

NUM_GPUS=${SLURM_NTASKS_PER_NODE:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
ACCUM=${ACCUM:-4}
NUM_EPOCHS=${NUM_EPOCHS:-20}
PATIENCE=${PATIENCE:-10}  # early stopping; the cosine schedule often improves val/loss late, the best checkpoint is kept
# Metric for the best checkpoint and early stopping. Val FDE tracks the argmax output; val/loss kept rising after
# epoch ~7 while val FDE improved to the end in every phase 1 run (2026-10-30 sweep).
MONITOR=${MONITOR:-val/loss}
NUM_WORKERS=${NUM_WORKERS:-12}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-training_vision_llm_lora_agent}  # output folder and W&B run name
TRAIN_SAMPLES=${TRAIN_SAMPLES:-85109}  # navtrain training split (978 train logs; val 18179 scenes in 214 logs), from the training log
STEPS_PER_EPOCH=$(( (TRAIN_SAMPLES + BATCH_SIZE * NUM_GPUS - 1) / (BATCH_SIZE * NUM_GPUS) ))
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

AUX_ARGS=()
if [[ ${AUX_TARGETS:-0} == 1 ]]; then
    AUX_IMAGE=${AUX_IMAGE:-$OPENSCENE_DATA_ROOT/images/aux_targets_navtrain.sqfs}
    AUX_MOUNT=$NAVSIM_EXP_ROOT/mnt/aux_${SLURM_JOB_ID:-$$}  # FUSE mounts are node-local; one folder per job
    mkdir -p "$AUX_MOUNT"
    squashfuse "$AUX_IMAGE" "$AUX_MOUNT"
    trap 'fusermount -u "$AUX_MOUNT"; rmdir "$AUX_MOUNT"' EXIT
    echo "Mounted $AUX_IMAGE at $AUX_MOUNT: $(ls "$AUX_MOUNT" | wc -l) logs"
    AUX_ARGS=(++agent.config.aux_target_cache_path=$AUX_MOUNT)
fi

echo "GPUs=$NUM_GPUS batch=$BATCH_SIZE accum=$ACCUM effective=$((BATCH_SIZE * ACCUM * NUM_GPUS))"
echo "total_training_steps (batches per GPU per epoch) = $STEPS_PER_EPOCH"

srun python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=vision_llm_lora_agent \
    experiment_name=$EXPERIMENT_NAME \
    train_test_split=navtrain \
    cache_path=$NAVSIM_EXP_ROOT/training_cache \
    trainer.params.precision=bf16-mixed \
    ++trainer.params.devices=$NUM_GPUS \
    trainer.params.max_epochs=$NUM_EPOCHS \
    early_stopping_patience=$PATIENCE \
    checkpoint_monitor=$MONITOR \
    trainer.params.accumulate_grad_batches=$ACCUM \
    dataloader.params.batch_size=$BATCH_SIZE \
    dataloader.params.num_workers=$NUM_WORKERS \
    ++agent.config.num_epochs=$NUM_EPOCHS \
    ++agent.config.gradient_accumulation_steps=$ACCUM \
    ++agent.config.total_training_steps=$STEPS_PER_EPOCH \
    "${AUX_ARGS[@]}" \
    "$@"
