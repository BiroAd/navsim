#!/bin/bash
#SBATCH --job-name=vlm_submission
#SBATCH --partition=gpu,ai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
# Leaderboard submission pickle (submission.pkl) of one vision_llm_lora checkpoint on a two-stage split:
# navsim/planning/script/run_create_submission_pickle.py runs the agent on every stage-one scene and every stage-two
# (synthetic) scene and saves the trajectories, without scoring (the leaderboard scores them). One GPU, sequential.
#   RUN=p3_aux_policy/2026.11.06.23.42.11 CKPT=checkpoints/merged/epoch-epoch=03.ckpt SELECTION=shortlist \
#       sbatch scripts/submission/run_vision_llm_lora_create_submission_slurm.sh ++agent.config.selection_shortlist_size=256
# RUN, CKPT, SELECTION as in scripts/evaluation/run_vision_llm_lora_pdm_score_slurm.sh (the run's ++agent.config.*
#           overrides rebuild the model; policy-only runs need the merged checkpoint)
# SPLIT     two-stage split whose data is under $OPENSCENE_DATA_ROOT/<split> (default navhard_two_stage)
# TEAM_NAME, AUTHORS, EMAIL, INSTITUTION, COUNTRY: submission metadata (shown on the leaderboard; the team name can be
#           changed there later). TEAM_NAME defaults to a placeholder.
# NAME      output folder under $NAVSIM_EXP_ROOT/submission_<split>/ (default <run>_<ckpt>_<selection>)
# Extra arguments go to Hydra.
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${OPENSCENE_DATA_ROOT:?}" "${NUPLAN_MAPS_ROOT:?}" "${NAVSIM_EXP_ROOT:?}" "${RUN:?}" "${CKPT:?}"

SELECTION=${SELECTION:-argmax}
SPLIT=${SPLIT:-navhard_two_stage}
TEAM_NAME=${TEAM_NAME:-Anonymous}
AUTHORS=${AUTHORS:-Anonymous}
EMAIL=${EMAIL:-biro.adam@edu.bme.hu}
INSTITUTION=${INSTITUTION:-Budapest_University_of_Technology_and_Economics}
COUNTRY=${COUNTRY:-Hungary}
RUN_DIR=$NAVSIM_EXP_ROOT/$RUN
[[ $CKPT = /* ]] || CKPT=$RUN_DIR/$CKPT
CKPT_LABEL=$(basename "$CKPT" .ckpt | sed 's/epoch-epoch=/ep/; s/best-epoch=/best_ep/')
NAME=${NAME:-${RUN%%/*}_${CKPT_LABEL}_${SELECTION}}
OUT=submission_$SPLIT/$NAME
IMAGE=${IMAGE:-$OPENSCENE_DATA_ROOT/images/test_sensor_blobs.sqfs}
MOUNT=$NAVSIM_EXP_ROOT/mnt/${SLURM_JOB_ID:-$$}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
[[ -f $CKPT ]] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }
[[ -d $OPENSCENE_DATA_ROOT/$SPLIT ]] || { echo "no synthetic data for $SPLIT under $OPENSCENE_DATA_ROOT" >&2; exit 1; }

mapfile -t AGENT_OVERRIDES < <(sed -n 's/^- \(++agent\.config\..*\)$/\1/p' "$RUN_DIR/code/hydra/overrides.yaml")
echo "Run $RUN, checkpoint $CKPT, split $SPLIT, selection $SELECTION, team '$TEAM_NAME' -> $OUT"
printf '  %s\n' "${AGENT_OVERRIDES[@]}" "$@"

mkdir -p "$MOUNT"
squashfuse "$IMAGE" "$MOUNT"
trap 'fusermount -u "$MOUNT"; rmdir "$MOUNT"' EXIT

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py \
    train_test_split=$SPLIT \
    original_sensor_path=$MOUNT \
    synthetic_scenes_path=$OPENSCENE_DATA_ROOT/$SPLIT/synthetic_scene_pickles \
    synthetic_sensor_path=$OPENSCENE_DATA_ROOT/$SPLIT/sensor_blobs \
    worker=sequential \
    agent=vision_llm_lora_agent \
    "agent.checkpoint_path='$CKPT'" \
    "${AGENT_OVERRIDES[@]}" \
    ++agent.config.trajectory_selection=$SELECTION \
    experiment_name=$OUT \
    "team_name='$TEAM_NAME'" "authors='$AUTHORS'" "email='$EMAIL'" \
    "institution='$INSTITUTION'" "country='$COUNTRY'" \
    "$@"

python - "$NAVSIM_EXP_ROOT/$OUT" <<'PY'
import pickle, sys
from pathlib import Path

pkl = sorted(Path(sys.argv[1]).glob("**/submission.pkl"))[-1]
sub = pickle.load(open(pkl, "rb"))
print(pkl)
print({k: v for k, v in sub.items() if not k.endswith("predictions")})
print("stage one trajectories:", len(sub["first_stage_predictions"][0]),
      "| stage two trajectories:", len(sub["second_stage_predictions"][0]))
PY
