#!/bin/bash
#SBATCH --job-name=vlm_qualitative
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000M
#SBATCH --time=01:00:00
# Qualitative figures (scripts/visualization/plot_qualitative.py) on a CPU node: mounts the test sensor image
# (squashfuse, node-local, one folder per job), renders FIGURES (default "rules consecutive") into OUT_DIR
# (default $NAVSIM_EXP_ROOT/qualitative), unmounts on exit. One BLAS / torch thread per process: the login node's
# per-user thread limit made OpenBLAS's 64 threads fail there (2026-11-21). Extra arguments go to the script.
#   sbatch scripts/visualization/run_qualitative_slurm.sh [--n-top 4 --n-random 4]
set -euo pipefail
cd "${NAVSIM_DEVKIT_ROOT:-/project/nr_e2e_vlm/navsim}"
source ./bashrc >/dev/null 2>&1
export PATH=/project/nr_e2e_vlm/envs/navsim_vlm/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MOUNT=$NAVSIM_EXP_ROOT/mnt/${SLURM_JOB_ID:-$$}
mkdir -p "$MOUNT"
squashfuse "${IMAGE:-$OPENSCENE_DATA_ROOT/images/test_sensor_blobs.sqfs}" "$MOUNT"
trap 'fusermount -u "$MOUNT"; rmdir "$MOUNT"' EXIT
for fig in ${FIGURES:-rules consecutive}; do
    python scripts/visualization/plot_qualitative.py --figure "$fig" --sensor-path "$MOUNT" \
        --out-dir "${OUT_DIR:-$NAVSIM_EXP_ROOT/qualitative}" "$@"
done
