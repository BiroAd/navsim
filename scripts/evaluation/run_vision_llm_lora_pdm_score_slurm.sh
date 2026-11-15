#!/bin/bash
#SBATCH --job-name=vlm_pdm_navtest
#SBATCH --partition=gpu,ai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.log
# Official PDM scoring of one vision_llm_lora checkpoint from raw camera images, one scene at a time on one GPU.
#   SPLIT=navtest (default): one stage, navsim/planning/script/run_pdm_score_one_stage.py, 12,146 scenes.
#   SPLIT=navhard_two_stage: NAVSIM v2 two-stage pseudo-simulation, navsim/planning/script/run_pdm_score.py (real
#     scenes from navsim_logs/test + synthetic follow-up scenes from $OPENSCENE_DATA_ROOT/navhard_two_stage, 5,912
#     metric caches). Always reactive (IDM): run_pdm_score.py uses traffic_agents_policy.reactive in both stages.
#   RUN=p3_heads/2026.11.04.00.53.01 CKPT=checkpoints/best/best-epoch=05.ckpt SELECTION=score_heads \
#       sbatch scripts/evaluation/run_vision_llm_lora_pdm_score_slurm.sh
# RUN       run folder relative to $NAVSIM_EXP_ROOT; its ++agent.config.* training overrides rebuild the model
# CKPT      checkpoint, relative to the run folder or absolute
# SELECTION argmax (the policy's choice, default) | score_heads (best predicted total, v2 formula without two-frame
#           comfort) | score_heads_v1 (best predicted NAVSIM v1 PDMS); agent config trajectory_selection
# SPLIT     navtest (default) | navhard_two_stage
# TRAFFIC   navtest only: non_reactive (log replay, as NAVSIM v1; default) | reactive (IDM)
# HUMAN_FILTER true (official setting in config/pdm_scoring/scorer/pdm_scorer.yaml; default) | false
# WORKERS   1 (default): sequential worker. N > 1: N worker processes (single_machine_thread_pool with a process pool),
#           each with its own copy of the model on the one GPU (~10.8 GB each: 2 workers peaked at 21.6 GB in smoke test
#           14891440, so at most 3 on an A100-40GB); give the job more CPUs and memory, e.g. --cpus-per-task=16 --mem=120G.
#           Workers split the scenes by log, so a run over few logs starts fewer workers.
# NAME      output folder name under $NAVSIM_EXP_ROOT/pdm_<split>/ (default <run>_<ckpt>_<selection>_<traffic>)
# Extra arguments go to Hydra, e.g. a smoke test: train_test_split.scene_filter.max_scenes=20
#
# Human-penalty filter (pdm_score.py): the human trajectory is scored too, and every metric the human fails is reported
# as 1 for the model. It only changes the per-metric columns: the total is built from multiplicative_metrics_prod and
# weighted_metrics, which the filter skips, so the score is the same either way. With the filter on (official) the
# sub-score columns, and the v1-style PDMS recomputed from them, are lenient; HUMAN_FILTER=false gives raw ones.
# The original camera images come from images/test_sensor_blobs.sqfs, mounted for this job only.
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${OPENSCENE_DATA_ROOT:?}" "${NUPLAN_MAPS_ROOT:?}" "${NAVSIM_EXP_ROOT:?}" "${RUN:?}" "${CKPT:?}"

SELECTION=${SELECTION:-argmax}
WORKERS=${WORKERS:-1}
SPLIT=${SPLIT:-navtest}
HUMAN_FILTER=${HUMAN_FILTER:-true}
case $SPLIT in
    navtest) TRAFFIC=${TRAFFIC:-non_reactive} ;;
    navhard_two_stage)
        [[ ${TRAFFIC:-reactive} = reactive ]] || { echo "navhard_two_stage is always reactive" >&2; exit 1; }
        TRAFFIC=reactive ;;
    *) echo "SPLIT must be navtest or navhard_two_stage" >&2; exit 1 ;;
esac
RUN_DIR=$NAVSIM_EXP_ROOT/$RUN
[[ $CKPT = /* ]] || CKPT=$RUN_DIR/$CKPT
CKPT_LABEL=$(basename "$CKPT" .ckpt | sed 's/epoch-epoch=/ep/; s/best-epoch=/best_ep/')
NAME=${NAME:-${RUN%%/*}_${CKPT_LABEL}_${SELECTION}_${TRAFFIC}}
OUT=pdm_$SPLIT/$NAME
IMAGE=${IMAGE:-$OPENSCENE_DATA_ROOT/images/test_sensor_blobs.sqfs}
MOUNT=$NAVSIM_EXP_ROOT/mnt/${SLURM_JOB_ID:-$$}  # FUSE mounts are node-local; one folder per job avoids clashes
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
[[ -f $CKPT ]] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }

# Model settings of the run: its ++agent.config.* training overrides (training-only ones are harmless here).
mapfile -t AGENT_OVERRIDES < <(sed -n 's/^- \(++agent\.config\..*\)$/\1/p' "$RUN_DIR/code/hydra/overrides.yaml")
echo "Run $RUN, checkpoint $CKPT, split $SPLIT, selection $SELECTION, traffic $TRAFFIC, human filter $HUMAN_FILTER -> $OUT"
printf '  %s\n' "${AGENT_OVERRIDES[@]}"

mkdir -p "$MOUNT"
squashfuse "$IMAGE" "$MOUNT"
trap 'fusermount -u "$MOUNT"; rmdir "$MOUNT"' EXIT
echo "Mounted $IMAGE at $MOUNT: $(ls "$MOUNT" | wc -l) logs"

if (( WORKERS > 1 )); then
    WORKER_ARGS=(worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers=$WORKERS)
    # GPU memory every 15 s (one model copy per worker on the one GPU): $NAVSIM_EXP_ROOT/pdm_<split>/<name>.gpu_mem.csv
    mkdir -p "$NAVSIM_EXP_ROOT/pdm_$SPLIT"
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 15 \
        > "$NAVSIM_EXP_ROOT/$OUT.gpu_mem.csv" &
    GPU_LOG_PID=$!
    trap 'kill $GPU_LOG_PID 2>/dev/null; fusermount -u "$MOUNT"; rmdir "$MOUNT"' EXIT
else
    WORKER_ARGS=(worker=sequential)
fi
if [[ $SPLIT = navtest ]]; then
    SCRIPT=run_pdm_score_one_stage.py
    SPLIT_ARGS=(traffic_agents=$TRAFFIC)
else
    SCRIPT=run_pdm_score.py
    SPLIT_ARGS=(synthetic_scenes_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles
                synthetic_sensor_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs)
fi

# No Ray: it cannot start on Komondor (see run_vision_llm_lora_dataset_caching.sh). Every worker loads the agent (the 4B
# model) itself, inside run_pdm_score.
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/$SCRIPT \
    train_test_split=$SPLIT \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache_$SPLIT \
    original_sensor_path=$MOUNT \
    "${SPLIT_ARGS[@]}" \
    "${WORKER_ARGS[@]}" \
    agent=vision_llm_lora_agent \
    "agent.checkpoint_path='$CKPT'" \
    "${AGENT_OVERRIDES[@]}" \
    ++agent.config.trajectory_selection=$SELECTION \
    scorer.config.human_penalty_filter=$HUMAN_FILTER \
    experiment_name=$OUT \
    "$@"

# Summary of the newest result CSV. navtest: EPDMS (the script's score) and a v1-style PDMS from the reported
# sub-scores (lenient with the filter on). navhard: the script's stage one / stage two / combined EPDMS rows.
python - "$NAVSIM_EXP_ROOT/$OUT" "$SPLIT" <<'PY'
import sys
from pathlib import Path

import pandas as pd

csv = sorted(Path(sys.argv[1]).glob("*/*.csv"))[-1]
df = pd.read_csv(csv)
if sys.argv[2] != "navtest":
    rows = df[df["token"].astype(str).str.startswith("extended_pdm_score")].set_index("token")
    print(f"{csv}: {(~df['token'].astype(str).str.startswith('extended_pdm_score') & df['valid'].astype(bool)).sum()} valid rows")
    print(rows.T.dropna(how="all").to_string())
    sys.exit()
df = df[(df["token"] != "average_all_frames") & df["valid"].astype(bool)]
v1 = (df["no_at_fault_collisions"] * df["drivable_area_compliance"] * df["driving_direction_compliance"]
      * (5 * df["ego_progress"] + 5 * df["time_to_collision_within_bound"] + 2 * df["history_comfort"]) / 12)
print(f"{csv}: {len(df)} valid scenes")
print(f"EPDMS (one stage) {df['score'].mean():.4f}")
print(f"v1-style PDMS     {v1.mean():.4f}  (NC*DAC*DDC*(5EP+5TTC+2HC)/12 from the reported sub-scores; approximation)")
for c in ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
          "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
          "history_comfort", "two_frame_extended_comfort"]:
    if c in df:
        print(f"  {c:32s} {df[c].mean():.4f}")
PY
