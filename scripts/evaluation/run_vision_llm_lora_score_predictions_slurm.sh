#!/bin/bash
#SBATCH --job-name=vlm_score
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=00:25:00
#SBATCH --output=%x-%j.log
# CPU half of the split official evaluation: score saved predictions (run_vision_llm_lora_predict_slurm.sh) with a
# trajectory selection, using the unchanged NAVSIM scripts and VisionLLMLoRAPrecomputedAgent. No GPU, no images.
#   SPLIT=navtest (default): run_pdm_score_one_stage.py (TRAFFIC non_reactive by default); navhard_two_stage:
#   run_pdm_score.py (always reactive). Same settings and output layout as run_vision_llm_lora_pdm_score_slurm.sh.
#   PRED=p3_aux_policy_ep03 SELECTION=shortlist K=256 sbatch scripts/evaluation/run_vision_llm_lora_score_predictions_slurm.sh
# PRED          prediction folder under $NAVSIM_EXP_ROOT/predictions/ (holds <split>/)
# SCORES_PRED   optional: prediction folder whose sub-score logits replace PRED's (RL-3: the start model,
#               e.g. p3_aux_policy_ep03; the RL model's own heads are not trained)
# SELECTION     argmax | score_heads | score_heads_v1 | shortlist | shortlist_v1 | mixture | mixture_v1
# K / LAMBDA    shortlist size (default 64) / mixture lambda (default 10)
# SPLIT         navtest (default) | navhard_two_stage
# TRAFFIC       navtest only: non_reactive (default) | reactive
# HUMAN_FILTER  true (official; default) | false
# SCORE_NAME    output folder under $NAVSIM_EXP_ROOT/pdm_<split>/ (default <PRED>_<selection>[_k<K>]_<traffic>_split).
#               Not NAME: the GPU job's NAME is exported to these jobs by sbatch (both navtest jobs of 14926361/62
#               then wrote into one folder).
# WORKERS       scoring processes (default: the job's CPUs)
# Scoring runs through run_pdm_score_balanced.py: the NAVSIM script unchanged, one task per log (largest first)
# instead of one fixed chunk of logs per process (which left 31 of 32 processes idle in job 14926361).
# Resources (navtest v2, p3_aux_policy, same 12,146 scenes; reportseff CPUEff):
#   32 CPUs 80G 5.4 min 40-43% | 16 CPUs 32G 8.1 min 57% | 8 CPUs 16000M 12.0 min 66% | 4 CPUs 8000M 18.8 min 75%.
#   ~3.8 min per job runs on one process (Python imports ~95 s, worker setup ~35 s, NAVSIM aggregation ~45 s, summary
#   ~40 s, now awk). The cpu partition gives at most 2000 MB per core: a larger --mem adds idle cores (24G on 8 CPUs ->
#   13 CPUs), so keep --mem <= 2000M x CPUs. Memory shows ~100% used at any limit (file cache), no job was killed.
#   Default 4 CPUs / 8000M; navhard also 4 CPUs (submit_vision_llm_lora_eval.sh: 45 min limit; about twice the
#   scoring work: 12 min on 32 CPUs at 36%).
# Extra arguments go to Hydra.
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${OPENSCENE_DATA_ROOT:?}" "${NUPLAN_MAPS_ROOT:?}" "${NAVSIM_EXP_ROOT:?}" "${PRED:?}" "${SELECTION:?}"

SPLIT=${SPLIT:-navtest}
K=${K:-64}
LAMBDA=${LAMBDA:-10.0}
HUMAN_FILTER=${HUMAN_FILTER:-true}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}
case $SPLIT in
    navtest) TRAFFIC=${TRAFFIC:-non_reactive}; SCRIPT=run_pdm_score_one_stage; SPLIT_ARGS=(traffic_agents=$TRAFFIC) ;;
    navhard_two_stage)
        [[ ${TRAFFIC:-reactive} = reactive ]] || { echo "navhard_two_stage is always reactive" >&2; exit 1; }
        TRAFFIC=reactive; SCRIPT=run_pdm_score
        SPLIT_ARGS=(synthetic_scenes_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles
                    synthetic_sensor_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs) ;;
    *) echo "SPLIT must be navtest or navhard_two_stage" >&2; exit 1 ;;
esac
case $SELECTION in
    shortlist*) LABEL=${SELECTION}_k$K ;;
    mixture*) LABEL=${SELECTION}_lam${LAMBDA%.0} ;;
    *) LABEL=$SELECTION ;;
esac
PRED_DIR=$NAVSIM_EXP_ROOT/predictions/$PRED/$SPLIT
[[ -f $PRED_DIR/meta.json ]] || { echo "no predictions in $PRED_DIR" >&2; exit 1; }
SCORES_ARGS=()
if [[ -n ${SCORES_PRED:-} ]]; then
    SCORES_DIR=$NAVSIM_EXP_ROOT/predictions/$SCORES_PRED/$SPLIT
    [[ -f $SCORES_DIR/meta.json ]] || { echo "no predictions in $SCORES_DIR" >&2; exit 1; }
    SCORES_ARGS=("agent.scores_path='$SCORES_DIR'")
fi
SCORE_NAME=${SCORE_NAME:-${PRED}_${LABEL}_${TRAFFIC}_split}
OUT=pdm_$SPLIT/$SCORE_NAME
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1  # one thread per scoring process
echo "Predictions $PRED_DIR${SCORES_PRED:+ (sub-scores from $SCORES_PRED)}, selection $LABEL, split $SPLIT, traffic $TRAFFIC, $WORKERS workers -> $OUT"

# original_sensor_path is never read (the agent asks for no sensors) but the scene loader needs a value.
python $NAVSIM_DEVKIT_ROOT/scripts/evaluation/run_pdm_score_balanced.py $SCRIPT \
    train_test_split=$SPLIT \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache_$SPLIT \
    original_sensor_path=$OPENSCENE_DATA_ROOT \
    "${SPLIT_ARGS[@]}" \
    worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers=$WORKERS \
    agent=vision_llm_lora_precomputed_agent \
    "agent.predictions_path='$PRED_DIR'" \
    agent.trajectory_selection=$SELECTION \
    agent.selection_shortlist_size=$K \
    agent.selection_mixture_lambda=$LAMBDA \
    "${SCORES_ARGS[@]}" \
    scorer.config.human_penalty_filter=$HUMAN_FILTER \
    experiment_name=$OUT \
    "$@"

# Summary of the newest result CSV in awk: a second Python start (pandas import ~40 s on one core) cost ~6% CPU
# efficiency. navtest: valid scenes, EPDMS, v1 PDMS and sub-score means; navhard: the script's stage / combined rows.
CSV=$(ls "$NAVSIM_EXP_ROOT/$OUT"/*/*.csv | sort | tail -1)
awk -F, -v sp="$SPLIT" -v csv="$CSV" '
NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
sp != "navtest" { if ($col["token"] ~ /^extended_pdm_score/) printf "%-32s score %.4f\n", $col["token"], $col["score"]
                     else { rows++; valid += ($col["valid"] == "True") }; next }
$col["token"] == "average_all_frames" { next }
{ scenes++ }
$col["valid"] != "True" { next }
{ n++; s += $col["score"]
  v1 += $col["no_at_fault_collisions"] * $col["drivable_area_compliance"] * (5 * $col["ego_progress"] \
        + 5 * $col["time_to_collision_within_bound"] + 2 * $col["history_comfort"]) / 12
  for (k in col) if (k ~ /^(no_at|drivable|driving|traffic|ego_progress|time_to|lane|history|two_frame)/ && $col[k] != "") { sum[k] += $col[k]; cnt[k]++ } }
END { if (sp != "navtest") { printf "%s: %d valid / %d rows\n", csv, valid, rows; exit }
      printf "%s: %d valid / %d scenes\nEPDMS (one stage) %.4f\nv1 PDMS           %.4f  (NC*DAC*(5EP+5TTC+2HC)/12; use a _v1 selection for the official v1 number)\n", csv, n, scenes, s / n, v1 / n
      for (k in sum) printf "  %-32s %.4f\n", k, sum[k] / cnt[k] }' "$CSV"
