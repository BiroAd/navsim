#!/bin/bash
#SBATCH --job-name=history_cache
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
# Past-frame camera tiles (t-0.5 / t-1.0 / t-1.5 s, uint8) and past ego statuses of every training-cache scene
# (scripts/training/cache_history_features.py), packed into ONE squashfs image for the project file quota:
#   $OPENSCENE_DATA_ROOT/images/history_navtrain.sqfs  (<log>/<token>/vision_llm_history_t{1,2,3}.gz,
#   vision_llm_ego_history.gz, history_info.json); mount it with squashfuse and pass the mount as the history cache.
# Job array over NUM_SHARDS log shards; each stages its scenes in /dev/shm (RAM, counted in --mem) and writes one tar
# to STAGE_DIR on project storage; a PACK_ONLY=1 job streams the tars into the image and deletes them as it goes:
#   export RUN_CONFIG=$NAVSIM_EXP_ROOT/p3_aux/2026.11.05.00.00.51/code/hydra/config.yaml NUM_SHARDS=40
#   a=$(sbatch --parsable --array=0-39 scripts/training/run_cache_history_features_slurm.sh)
#   PACK_ONLY=1 sbatch --dependency=afterok:$a -c 4 --mem=8000M -t 03:00:00 scripts/training/run_cache_history_features_slurm.sh
# A failed shard can be resubmitted alone (--array=<id>); shards whose tar exists are skipped.
# Resources: job 14946659 (8 CPUs / 8 workers / 16000M, 40 shards) ran at 25-34% CPU and ~6.6 GB (41%): the
# workers wait for image reads, so 4 CPUs with 8 workers / 8000M (cpu partition: 2000 MB per core).
# Extra arguments go to cache_history_features.py (e.g. --max-logs 2 for a smoke test, without NUM_SHARDS).
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${OPENSCENE_DATA_ROOT:?}" "${NAVSIM_EXP_ROOT:?}"
STAGE_DIR=${STAGE_DIR:-$NAVSIM_EXP_ROOT/history_cache_stage}  # shard tars (one file per shard)
OUT=${OUT:-$OPENSCENE_DATA_ROOT/images/history_navtrain.sqfs}
PACK_ONLY=${PACK_ONLY:-0}
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

if [[ "$PACK_ONLY" == 1 ]]; then
    command -v tar2sqfs > /dev/null || { echo "tar2sqfs not found on $(hostname)" >&2; exit 1; }
    shards=("$STAGE_DIR"/history_shard*.tar)
    echo "packing ${#shards[@]} shard tars into $OUT"
    python $NAVSIM_DEVKIT_ROOT/scripts/training/merge_history_tars.py --delete "${shards[@]}" \
        | tar2sqfs -q -c lz4 -j "${SLURM_CPUS_PER_TASK:-8}" "$OUT.tmp"
    mv "$OUT.tmp" "$OUT"
    echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
    exit 0
fi

: "${RUN_CONFIG:?set RUN_CONFIG to a training run\'s code/hydra/config.yaml}"
IMAGE=${IMAGE:-$OPENSCENE_DATA_ROOT/images/navtrain_sensor_blobs.sqfs}
MOUNTS=${MOUNTS:-8}  # squashfuse serves one read at a time per mount
WORKERS=${WORKERS:-$(( 2 * ${SLURM_CPUS_PER_TASK:-4} ))}  # I/O-bound: 2 processes per core
JOB=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$$}}_${SLURM_ARRAY_TASK_ID:-0}
MNT=/dev/shm/navtrain_mnt_$JOB
STAGE=/dev/shm/history_stage_$JOB
mounts=()
for i in $(seq 1 "$MOUNTS"); do
    mkdir -p "$MNT/$i"
    squashfuse "$IMAGE" "$MNT/$i"
    mounts+=("$MNT/$i")
done
cleanup() { for m in "${mounts[@]}"; do fusermount -u "$m" 2>/dev/null || true; done; rm -rf "$MNT" "$STAGE"; }
trap cleanup EXIT
SHARD_ARGS=()
if [[ -n "${NUM_SHARDS:-}" ]]; then
    SHARD_ARGS=(--shard "${SLURM_ARRAY_TASK_ID:?NUM_SHARDS needs a job array}" --num-shards "$NUM_SHARDS")
fi
python $NAVSIM_DEVKIT_ROOT/scripts/training/cache_history_features.py --run-config "$RUN_CONFIG" \
    --sensor-path "${mounts[@]}" --stage-dir "$STAGE" --out-dir "$STAGE_DIR" --workers "$WORKERS" "${SHARD_ARGS[@]}" "$@"
