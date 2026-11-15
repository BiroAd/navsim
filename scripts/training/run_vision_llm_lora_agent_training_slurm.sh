#!/bin/bash
#SBATCH --job-name=vlm_lora_train
#SBATCH --partition=gpu,ai
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000M
#SBATCH --time=7-00:00:00
#SBATCH --output=%x-%j.log
# Trains vision_llm_lora on navtrain from the feature cache, with bf16 and flash attention (set in the agent YAML).
# Lightning under SLURM expects one srun task per GPU, hence ntasks-per-node = number of GPUs.
# --cpus-per-task is PER TASK (x 4 tasks): -c 2 = 8 CPUs per job. Measured: phase 1 with 48 CPUs (-c 12) used 11-12%
# (~5-6 cores, GPU 92-93%, host memory 25-29 GB; p1hist 14949469-72); phase 3 with 8 CPUs used 58% (GPU 81%, ~19 GB;
# p3_aux_ctrl 14935958). gpu/ai give at most 4000 MB per CPU, so --mem=32000M fits 8 CPUs (32G would add a 9th). A
# 32-CPU request (-c 8) could not start on nodes with free GPUs but only 16 free CPUs (14967646-49).
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

# HISTORY_CACHE=1 mounts the past-frame cache (scripts/training/run_cache_history_features_slurm.sh:
# $OPENSCENE_DATA_ROOT/images/history_navtrain.sqfs) and passes it as agent.config.history_cache_path; needed with
# ++agent.config.temporal_history_frame > 0 or ++agent.config.ego_history_tokens=true.
# HISTORY_MOUNTS copies are mounted and used in turn: squashfuse serves one read at a time per mount (~11 past-frame
# files/s each on Lustre), and job 14949462 (p1hist_prev, one mount) ran at 0.32 it/s instead of ~0.9.
HISTORY_ARGS=()
HISTORY_DIRS=()
if [[ ${HISTORY_CACHE:-0} == 1 ]]; then
    HISTORY_IMAGE=${HISTORY_IMAGE:-$OPENSCENE_DATA_ROOT/images/history_navtrain.sqfs}
    HISTORY_BASE=$NAVSIM_EXP_ROOT/mnt/history_${SLURM_JOB_ID:-$$}
    for i in $(seq 1 "${HISTORY_MOUNTS:-16}"); do
        mkdir -p "$HISTORY_BASE/$i"
        squashfuse "$HISTORY_IMAGE" "$HISTORY_BASE/$i"
        HISTORY_DIRS+=("$HISTORY_BASE/$i")
    done
    echo "Mounted $HISTORY_IMAGE ${#HISTORY_DIRS[@]} times under $HISTORY_BASE"
    HISTORY_ARGS=("++agent.config.history_cache_path='$(IFS=:; echo "${HISTORY_DIRS[*]}")'")
fi
cleanup_mounts() {
    if [[ ${AUX_TARGETS:-0} == 1 ]]; then fusermount -u "$AUX_MOUNT" || true; rmdir "$AUX_MOUNT" || true; fi
    for d in "${HISTORY_DIRS[@]}"; do fusermount -u "$d" || true; rmdir "$d" || true; done
    if [[ ${#HISTORY_DIRS[@]} -gt 0 ]]; then rmdir "$HISTORY_BASE" || true; fi
    if [[ -n ${RL_REF_DIR:-} ]]; then rm -rf "${RL_REF_DIR:?}"; fi
}
trap cleanup_mounts EXIT

# RL_REFERENCE=<name> (RL-3, rl_loss_weight > 0): builds the per-scene RL data (scripts/training/cache_rl_reference.py:
# rewards, start-model fused state, scorer shortlist; ~1.5 GB in 103k files) from $NAVSIM_EXP_ROOT/rl_features/<name>
# into /dev/shm for this job (never ~100k small files on /project: file quota) and passes rl_reference_cache_path and
# rl_reference_head_path. /dev/shm counts against --mem (~1.5 GB). Deleted at exit.
# RL_PREVIOUS_SHORTLISTS=<name> adds each scene's scorer shortlist at t - 0.5 s from
# $NAVSIM_EXP_ROOT/scorer_shortlists/<name> (scripts/training/cache_scorer_shortlists.py), for
# ++agent.config.rl_ec_previous_plan=shortlist.
RL_ARGS=()
PREVIOUS_ARGS=()
if [[ -n ${RL_PREVIOUS_SHORTLISTS:-} ]]; then
    PREVIOUS_ARGS=(--previous-shortlists "$NAVSIM_EXP_ROOT/scorer_shortlists/$RL_PREVIOUS_SHORTLISTS")
fi
if [[ -n ${RL_REFERENCE:-} ]]; then
    RL_REF_DIR=/dev/shm/rl_reference_${USER}_${SLURM_JOB_ID:-$$}
    python $NAVSIM_DEVKIT_ROOT/scripts/training/cache_rl_reference.py \
        --features-dir "$NAVSIM_EXP_ROOT/rl_features/$RL_REFERENCE" \
        --scores-dir "$NAVSIM_EXP_ROOT/codebook_scores_navtrain" \
        --out-dir "$RL_REF_DIR" --workers "${RL_REFERENCE_WORKERS:-$(nproc)}" "${PREVIOUS_ARGS[@]}"
    echo "RL reference for $RL_REFERENCE in $RL_REF_DIR: $(du -sh "$RL_REF_DIR" | cut -f1)"
    RL_ARGS=(++agent.config.rl_reference_cache_path=$RL_REF_DIR
             ++agent.config.rl_reference_head_path=$NAVSIM_EXP_ROOT/rl_features/$RL_REFERENCE/head.pt)
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
    "${HISTORY_ARGS[@]}" \
    "${RL_ARGS[@]}" \
    "$@"
