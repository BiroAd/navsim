#!/bin/bash
#SBATCH --job-name=vlm_rl_reference
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=00:25:00
#SBATCH --output=%x-%j.log
# Per-scene RL-3 data (rewards, start-model fused state, scorer shortlist) for vision_llm_lora_rl.RLReferenceTargetBuilder:
# scripts/training/cache_rl_reference.py, 2 processes per CPU (I/O to /project overlaps with compression).
# cpu partition: at most 2000 MB per core, so --mem stays at 2000M x CPUs.
#   RUN=p3_aux_policy_ep03 sbatch scripts/training/run_cache_rl_reference_slurm.sh
# I/O-bound (103k small files to /project), not CPU-bound: job 14928904 (4 CPUs, 8 processes, 8000M) took 8.5 min at
# 11% CPU and 2.1 GB. Job 14929818 (2 CPUs, 32 processes, 4000M) also ran at 10% CPU but reached 83% of its memory
# and slowed from 406 to 33 scenes/s (likely writeback throttling: the page cache of the written files counts against
# the job's memory limit); cancelled. The cpu partition gives one core per 2000 MB, so the memory sets the core count.
# Untested lighter option: --cpus-per-task=2 --mem=4000M with 8 processes (2.1 GB measured at 8).
# RUN  rl_features/<RUN> to read; output $NAVSIM_EXP_ROOT/rl_reference/<RUN>/ (~1.8 GB for navtrain, 103,288 scenes)
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${NAVSIM_EXP_ROOT:?}" "${RUN:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python $NAVSIM_DEVKIT_ROOT/scripts/training/cache_rl_reference.py \
    --features-dir "$NAVSIM_EXP_ROOT/rl_features/$RUN" \
    --scores-dir "$NAVSIM_EXP_ROOT/codebook_scores_navtrain" \
    --out-dir "$NAVSIM_EXP_ROOT/rl_reference/$RUN" --workers ${WORKERS:-8} "$@"
