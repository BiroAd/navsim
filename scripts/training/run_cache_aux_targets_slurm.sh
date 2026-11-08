#!/bin/bash
#SBATCH --job-name=aux_targets
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log
# Caches the auxiliary perception targets (VisionLLMLoRAAuxTargetBuilder: BEV semantic map, drivable-area map,
# lane-heading map, agent boxes) of every training-cache scene into ONE squashfs image, so the ~103k files do not count
# against the project's file quota. Files are staged in /dev/shm (RAM, ~1 GB) and removed afterwards. CPU only: reads
# the navsim logs and nuPlan maps, no sensors. Submit with bashrc sourced and the env active:
#   RUN_CONFIG=$NAVSIM_EXP_ROOT/p1c_scores/2026.11.01.00.34.10/code/hydra/config.yaml \
#       sbatch scripts/training/run_cache_aux_targets_slurm.sh
# The image holds <log>/<token>/vision_llm_aux_target.gz and aux_targets_info.json; mount it with
#   squashfuse $OPENSCENE_DATA_ROOT/images/aux_targets_navtrain.sqfs <dir>
# Extra arguments go to cache_aux_targets.py (e.g. --max-logs 4 for a test).
# Many small jobs (they fit into scheduling gaps): a job array over NUM_SHARDS log shards, resumable staging on project
# storage, then one pack job after all of them:
#   export STAGE=$NAVSIM_EXP_ROOT/aux_targets_stage RUN_CONFIG=... NUM_SHARDS=40 WORKERS=6 PACK=0
#   a=$(sbatch --parsable --array=0-39 -c 8 --mem=16G -t 1:00:00 <this>)
#   PACK_ONLY=1 sbatch -c 8 --mem=16G -t 1:00:00 --dependency=afterok:$a <this>
# A failed or timed-out array task can be resubmitted alone (--array=<id>); finished scenes are skipped.
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${OPENSCENE_DATA_ROOT:?}" "${NUPLAN_MAPS_ROOT:?}" "${NAVSIM_EXP_ROOT:?}"
PACK=${PACK:-1}            # 0: only compute (shards), the image is packed by a later PACK_ONLY=1 job
PACK_ONLY=${PACK_ONLY:-0}  # 1: only pack an existing STAGE
[[ "$PACK_ONLY" == 1 ]] || RUN_CONFIG=${RUN_CONFIG:?set RUN_CONFIG to a training run\'s code/hydra/config.yaml (scene filter and cache_path)}
OUT=${OUT:-$OPENSCENE_DATA_ROOT/images/aux_targets_navtrain.sqfs}  # next to the sensor-blob images
WORKERS=${WORKERS:-40}  # map objects are cached per worker; ~1-2 GB each
# Staging directory of the per-scene files. Default: RAM, removed on exit. Set STAGE to a directory on project storage
# to make the run resumable across jobs (e.g. on the 1 h test partition): scenes already written are skipped, and the
# directory is removed only after the image was packed.
STAGE=${STAGE:-/dev/shm/aux_targets_${SLURM_JOB_ID:-local}}
command -v mksquashfs > /dev/null || { echo "mksquashfs not found on $(hostname)" >&2; exit 1; }
[[ "$STAGE" == /dev/shm/* ]] && trap 'rm -rf "$STAGE"' EXIT
export OMP_NUM_THREADS=1

SHARD_ARGS=()
if [[ -n "${NUM_SHARDS:-}" ]]; then  # job array: this task's shard
    SHARD_ARGS=(--shard "${SLURM_ARRAY_TASK_ID:?NUM_SHARDS needs a job array}" --num-shards "$NUM_SHARDS")
fi
if [[ "$PACK_ONLY" != 1 ]]; then
    python $NAVSIM_DEVKIT_ROOT/scripts/training/cache_aux_targets.py \
        --run-config "$RUN_CONFIG" --out-dir "$STAGE" --workers "$WORKERS" "${SHARD_ARGS[@]}" "$@"
fi
[[ "$PACK" == 1 || "$PACK_ONLY" == 1 ]] || exit 0
if find "$STAGE" -name '*.tmp' | grep -q .; then echo "unfinished .tmp files in $STAGE, rerun the shards" >&2; exit 1; fi
# -no-xattrs: Lustre ACL/striping attributes are unknown to mksquashfs 4.3 (one warning per file otherwise)
mksquashfs "$STAGE" "$OUT.tmp" -noappend -no-xattrs -processors "${SLURM_CPUS_PER_TASK:-8}"
mv "$OUT.tmp" "$OUT"
python - "$STAGE" > "${OUT%.sqfs}_info.json" <<'PY'  # one summary from the per-shard ones
import glob, json, sys
parts = [json.load(open(f)) for f in sorted(glob.glob(f"{sys.argv[1]}/aux_targets_info*.json"))]
merged = {k: parts[0][k] for k in ("target_name", "grid", "bev_semantic_classes", "max_agents", "heading_bins", "source_cache")}
merged.update(logs=sum(p["logs"] for p in parts), scenes_written=sum(p["scenes_written"] for p in parts),
              missing=[m for p in parts for m in p["missing"]], errors=[e for p in parts for e in p["errors"]],
              shards=len(parts), created=max(p["created"] for p in parts))
print(json.dumps(merged, indent=1))
PY
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)), $(find "$STAGE" -name '*.gz' | wc -l) scenes"
rm -rf "$STAGE"
