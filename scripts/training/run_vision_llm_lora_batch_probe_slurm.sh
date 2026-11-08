#!/bin/bash
#SBATCH --job-name=vlm_lora_batch_probe
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
# Finds the per-GPU batch size that fits: a few optimizer steps of the real training setup (agent YAML, cache, bf16)
# per batch size, logging peak GPU memory, stopping at the first OOM. Runs through the training SLURM script with
# ACCUM=1 by default so the optimizer state is allocated from the first batch; max_steps ends the run before validation
# and checkpointing (Lightning then fails on EarlyStopping's missing val/loss, after the measured steps).
# To check the real multi-GPU setup, override the allocation and set ACCUM, e.g.
#   PHASE=1 BATCH_SIZES=8 ACCUM=16 sbatch --gres=gpu:4 --ntasks-per-node=4 --mem=120G \
#       scripts/training/run_vision_llm_lora_batch_probe_slurm.sh Memory depends on the phase (3 > 2 > 1), so probe the phase you will train, e.g.
#   PHASE=3 BATCH_SIZES="4 8 12" sbatch scripts/training/run_vision_llm_lora_batch_probe_slurm.sh
# Extra arguments are passed to Hydra.
set -uo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${NAVSIM_EXP_ROOT:?}"

PHASE=${PHASE:-1}
BATCH_SIZES=${BATCH_SIZES:-"2 4 8 12 16"}
STEPS=${STEPS:-20}  # optimizer steps, i.e. STEPS x ACCUM batches per GPU
ACCUM=${ACCUM:-1}
PROBE_DIR=$NAVSIM_EXP_ROOT/batch_probe/$SLURM_JOB_ID
export WANDB_MODE=${WANDB_MODE:-disabled}  # no W&B runs for probes
mkdir -p "$PROBE_DIR"
nvidia-smi -L

for bs in $BATCH_SIZES; do
    echo "=== phase $PHASE, batch $bs ==="
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -l 1 > "$PROBE_DIR/mem_bs$bs.csv" &
    smi_pid=$!
    # NUM_EPOCHS=2: AgentLightningModule._step divides by max_epochs - 1; max_steps ends the run within epoch 0.
    BATCH_SIZE=$bs ACCUM=$ACCUM NUM_EPOCHS=2 NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 2)) \
        bash $NAVSIM_DEVKIT_ROOT/scripts/training/run_vision_llm_lora_agent_training_slurm.sh \
        ++agent.config.training_phase=$PHASE \
        ++trainer.params.max_steps=$STEPS \
        output_dir=$PROBE_DIR/bs$bs \
        "$@" 2>&1 | tee "$PROBE_DIR/train_bs$bs.log"
    status=${PIPESTATUS[0]}
    kill $smi_pid
    rm -rf "$PROBE_DIR/bs$bs"  # in case Lightning wrote a checkpoint of the full model
    peak=$(sort -t, -k1 -n "$PROBE_DIR/mem_bs$bs.csv" | tail -1)
    echo "RESULT phase=$PHASE batch=$bs exit=$status peak_used,total_MiB=$peak"
    if grep -qE "OutOfMemoryError|CUDA out of memory" "$PROBE_DIR/train_bs$bs.log"; then
        echo "RESULT batch=$bs ran out of GPU memory; stopping"
        break
    fi
done
