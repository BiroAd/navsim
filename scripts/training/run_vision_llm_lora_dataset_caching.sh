#!/bin/bash
#SBATCH --job-name=navtrain_caching
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.log
# Caches vision_llm_lora features/targets for navtrain into $NAVSIM_EXP_ROOT/training_cache.
# Submit from a shell with the env active (source /project/nr_e2e_vlm/envs/navsim_vlm/bin/activate) and the
# NAVSIM env vars set; sbatch exports both.
# Every caching worker instantiates the full agent, including the 4B model, so RAM grows with THREADS.# Extra arguments are passed to Hydra, e.g. a test run:
#   CACHE_PATH=$NAVSIM_EXP_ROOT/training_cache_test THREADS=2 ./run_vision_llm_lora_dataset_caching.sh \
#       train_test_split.scene_filter.max_scenes=200
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${OPENSCENE_DATA_ROOT:?}" "${NUPLAN_MAPS_ROOT:?}" "${NAVSIM_EXP_ROOT:?}"

THREADS=${THREADS:-8}
CACHE_PATH=${CACHE_PATH:-$NAVSIM_EXP_ROOT/training_cache}
IMAGE=${IMAGE:-$OPENSCENE_DATA_ROOT/images/navtrain_sensor_blobs.sqfs}
MOUNT=$OPENSCENE_DATA_ROOT/sensor_blobs/trainval
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}  # compute nodes may have no internet; the model must already be cached
# One OpenMP thread per worker process; by default each would start one per core (128) and oversubscribe the node.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

command -v squashfuse > /dev/null || { echo "squashfuse not found on $(hostname)" >&2; exit 1; }
squashfuse "$IMAGE" "$MOUNT"
trap 'fusermount -u "$MOUNT"' EXIT
echo "Mounted $IMAGE at $MOUNT: $(ls "$MOUNT" | wc -l) logs"

# Process pool instead of the default Ray worker: Ray's raylet cannot start here because /proc of other processes
# (e.g. /proc/1/stat) is not readable on Komondor, so ray.init fails with psutil.AccessDenied.
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
    agent=vision_llm_lora_agent \
    train_test_split=navtrain \
    cache_path=$CACHE_PATH \
    worker=single_machine_thread_pool \
    worker.use_process_pool=true \
    worker.max_workers=$THREADS \
    experiment_name=navtrain_caching \
    "$@"
