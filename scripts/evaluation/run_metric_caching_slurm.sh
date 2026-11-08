#!/bin/bash
#SBATCH --job-name=metric_caching
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log
# Caches the PDM-score metric inputs (map, proposals, agents; no sensors) of an evaluation split into
# $NAVSIM_EXP_ROOT/metric_cache_<split>. Submit with bashrc sourced and the env active:
#   SPLIT=navtest sbatch scripts/evaluation/run_metric_caching_slurm.sh            # NAVSIM v1 (PDMS)
#   SPLIT=navhard_two_stage sbatch scripts/evaluation/run_metric_caching_slurm.sh  # NAVSIM v2 two-stage (EPDMS)
# navhard's synthetic scenes live in $OPENSCENE_DATA_ROOT/navhard_two_stage; default_dataset_paths points the synthetic
# paths to warmup_two_stage, so they are overridden here. Its real scenes come from navsim_logs/test (data_split: test).
# Extra arguments are passed to Hydra, e.g. a test run: train_test_split.scene_filter.max_scenes=20
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${OPENSCENE_DATA_ROOT:?}" "${NUPLAN_MAPS_ROOT:?}" "${NAVSIM_EXP_ROOT:?}"

SPLIT=${SPLIT:?set SPLIT=navtest or SPLIT=navhard_two_stage}
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-8}}
CACHE_PATH=${CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache_$SPLIT}
# One OpenMP thread per worker process; by default each would start one per core and oversubscribe the node.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

EXTRA=()
if [ "$SPLIT" = navhard_two_stage ]; then
    EXTRA+=(synthetic_scenes_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles
            synthetic_sensor_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs)
fi

# Process pool instead of the default Ray worker: Ray's raylet cannot start here because /proc of other processes
# is not readable on Komondor (see run_vision_llm_lora_dataset_caching.sh).
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=$SPLIT \
    metric_cache_path=$CACHE_PATH \
    worker=single_machine_thread_pool \
    worker.use_process_pool=true \
    worker.max_workers=$THREADS \
    "${EXTRA[@]}" \
    "$@"
