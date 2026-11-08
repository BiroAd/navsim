#!/bin/bash
#SBATCH --job-name=vlm_rl_reference
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.log
# Per-scene RL-3 data (rewards, start-model fused state, scorer shortlist) for vision_llm_lora_rl.RLReferenceTargetBuilder:
# scripts/training/cache_rl_reference.py, 2 processes per CPU (I/O to /project overlaps with compression).
# cpu partition: at most 2000 MB per core, so --mem stays at 2000M x CPUs.
#   RUN=p3_aux_policy_ep03 sbatch scripts/training/run_cache_rl_reference_slurm.sh
# Job 14928904 (4 CPUs, 8 processes): 8.5 min, CPU 11%, 2.1 GB: writing 103k small files to /project is latency-bound,
# so 1 CPU with 32 processes (--workers 32) and 3G.
# RUN  rl_features/<RUN> to read; output $NAVSIM_EXP_ROOT/rl_reference/<RUN>/ (~1.8 GB for navtrain, 103,288 scenes)
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${NAVSIM_EXP_ROOT:?}" "${RUN:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python $NAVSIM_DEVKIT_ROOT/scripts/training/cache_rl_reference.py \
    --features-dir "$NAVSIM_EXP_ROOT/rl_features/$RUN" \
    --scores-dir "$NAVSIM_EXP_ROOT/codebook_scores_navtrain" \
    --out-dir "$NAVSIM_EXP_ROOT/rl_reference/$RUN" --workers ${WORKERS:-32} "$@"
