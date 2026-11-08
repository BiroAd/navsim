#!/bin/bash
#SBATCH --job-name=codebook_scoring
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.log
# PDM-scores every codebook trajectory in every scene of a split (scripts/evaluation/score_codebook_trajectories.py),
# one .npz per log. Needs no images. Submit from a shell with bashrc sourced and the env active.
# Input: SPLIT builds each scene's metric cache in memory (navtrain), METRIC_CACHE reads existing ones (navtest).
# As a SLURM array, each task scores every N-th log (N = array size; indices must start at 0), e.g.
#   SPLIT=navtrain OUT_DIR=$NAVSIM_EXP_ROOT/codebook_scores_navtrain \
#       sbatch --array=0-15 scripts/evaluation/run_codebook_scoring_slurm.sh
# Extra arguments go to the Python script, e.g. --max-logs 2 for a timing run. Resubmitting skips scored logs.
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${NAVSIM_EXP_ROOT:?}" "${OUT_DIR:?}"

if [ -n "${SPLIT:-}" ]; then
    : "${OPENSCENE_DATA_ROOT:?}" "${NUPLAN_MAPS_ROOT:?}"
    SOURCE=(--split "$SPLIT")
else
    SOURCE=(--metric-cache-path "${METRIC_CACHE:?set SPLIT or METRIC_CACHE}")
fi
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}
# One OpenMP thread per worker process; by default each would start one per core and oversubscribe the node.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

python $NAVSIM_DEVKIT_ROOT/scripts/evaluation/score_codebook_trajectories.py \
    "${SOURCE[@]}" \
    --out-dir "$OUT_DIR" \
    --workers "$WORKERS" \
    "$@"
