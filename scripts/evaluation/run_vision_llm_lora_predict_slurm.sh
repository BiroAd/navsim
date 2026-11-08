#!/bin/bash
#SBATCH --job-name=vlm_predict
#SBATCH --partition=gpu,ai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:25:00
#SBATCH --output=%x-%j.log
# GPU half of the split official evaluation: batched inference of one checkpoint over navtest and / or
# navhard_two_stage, raw outputs saved to $NAVSIM_EXP_ROOT/predictions/<NAME>/<split>/
# (scripts/evaluation/predict_vision_llm_lora.py). Score them on the cpu partition with
# run_vision_llm_lora_score_predictions_slurm.sh; submit_vision_llm_lora_eval.sh submits both with a dependency.
#   RUN=p3_aux_policy/2026.11.06.23.42.11 CKPT=checkpoints/merged/epoch-epoch=03.ckpt NAME=p3_aux_policy_ep03 \
#       sbatch scripts/evaluation/run_vision_llm_lora_predict_slurm.sh
# RUN        run folder relative to $NAVSIM_EXP_ROOT (its ++agent.config.* overrides rebuild the model)
# CKPT       checkpoint, relative to the run folder or absolute
# NAME       output folder under $NAVSIM_EXP_ROOT/predictions/
# SPLITS     "navtest navhard_two_stage" (default) or one of them
# BATCH_SIZE scenes per forward pass (default 64; batch 8 peaked at 10.4 GB)
# LOADER_WORKERS DataLoader processes (default 30). They mostly wait for squashfuse reads: with 32 CPUs the jobs
#            used ~3 cores on average (CPU efficiency ~10%, 2026-11-08 re-evaluation, 16-18 min, GPU 67-78%), so
#            30 workers share 8 CPUs. 40G: peak ~21.5 GB (job 14926360), 24G was OOM-killed.
# MOUNTS     squashfuse mounts of the image (default 16). squashfuse 0.1.104 serves one read at a time per mount: with
#            one mount 30 loader workers sat in I/O wait at ~4 scenes/s (job 14926280); workers share the mounts round-robin.
# Extra arguments go to predict_vision_llm_lora.py, e.g. --max-scenes 64 for a smoke test.
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${OPENSCENE_DATA_ROOT:?}" "${NUPLAN_MAPS_ROOT:?}" "${NAVSIM_EXP_ROOT:?}" "${RUN:?}" "${CKPT:?}" "${NAME:?}"

SPLITS=${SPLITS:-navtest navhard_two_stage}
BATCH_SIZE=${BATCH_SIZE:-64}
WORKERS=${LOADER_WORKERS:-30}
MOUNTS=${MOUNTS:-16}
IMAGE=${IMAGE:-$OPENSCENE_DATA_ROOT/images/test_sensor_blobs.sqfs}
MOUNT=$NAVSIM_EXP_ROOT/mnt/${SLURM_JOB_ID:-$$}  # FUSE mounts are node-local; one folder per job avoids clashes
OUT=$NAVSIM_EXP_ROOT/predictions/$NAME
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} PYTHONUNBUFFERED=1  # progress lines reach the log as they happen
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

mkdir -p "$OUT"
MOUNT_DIRS=()
for i in $(seq 0 $((MOUNTS - 1))); do
    mkdir -p "$MOUNT/$i"
    squashfuse "$IMAGE" "$MOUNT/$i"
    MOUNT_DIRS+=("$MOUNT/$i")
done
unmount() { for d in "${MOUNT_DIRS[@]}"; do fusermount -u "$d"; rmdir "$d"; done; rmdir "$MOUNT"; }
# GPU use every 15 s, to check the job keeps the GPU busy: <out>/gpu_util.csv
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 15 > "$OUT/gpu_util.csv" &
GPU_LOG_PID=$!
trap 'kill $GPU_LOG_PID 2>/dev/null; unmount' EXIT
echo "Run $RUN, checkpoint $CKPT, splits $SPLITS, batch $BATCH_SIZE, $WORKERS loader workers, $MOUNTS image mounts -> $OUT"

python $NAVSIM_DEVKIT_ROOT/scripts/evaluation/predict_vision_llm_lora.py \
    --run-dir "$NAVSIM_EXP_ROOT/$RUN" --checkpoint "$CKPT" --splits $SPLITS --out-dir "$OUT" \
    --original-sensor-path "${MOUNT_DIRS[@]}" --batch-size "$BATCH_SIZE" --num-workers "$WORKERS" "$@"
awk -F', ' 'NR > 1 { sub(/ %/, "", $4); s += $4; n++ } END { if (n) printf "mean GPU utilization %.0f%% over %d samples\n", s / n, n }' "$OUT/gpu_util.csv"
