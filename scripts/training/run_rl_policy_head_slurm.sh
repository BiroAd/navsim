#!/bin/bash
#SBATCH --job-name=vlm_rl_policy_head
#SBATCH --partition=gpu,ai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
# RL on the policy head of a vision_llm_lora checkpoint, on one GPU (submit with bashrc sourced and envs/navsim_vlm
# active). MODE=cache: one shard of the fused-state cache (scripts/training/cache_policy_head_features.py, one 4B-model
# pass per scene, ~19 scenes/s); MODE=train: RL runs on a complete cache (scripts/training/rl_policy_head.py, minutes).
#   MODE=cache RUN=p3_aux_policy/2026.11.06.23.42.11 CKPT=checkpoints/merged/epoch-epoch=03.ckpt NAME=p3_aux_policy_ep03 \
#       SHARD=0 NUM_SHARDS=3 sbatch scripts/training/run_rl_policy_head_slurm.sh
#   MODE=train NAME=p3_aux_policy_ep03 sbatch scripts/training/run_rl_policy_head_slurm.sh \
#       "--objective exact --beta 0.1" "--objective grpo --beta 0.1"
#   MODE=shortlist HISTORY_CACHE=1 RUN=p3_aux_ctrl/2026.11.09.23.55.17 CKPT=checkpoints/best/best-epoch=02.ckpt \
#       NAME=p3_aux_ctrl_ep02 SHARD=0 NUM_SHARDS=6 sbatch --time=01:15:00 scripts/training/run_rl_policy_head_slurm.sh
#   (scorer shortlists of frames t, t-0.5, t-1.0 s: scripts/training/cache_scorer_shortlists.py, 3 passes per scene,
#   outputs $NAVSIM_EXP_ROOT/scorer_shortlists/<NAME>; FRAMES overrides "0 1 2")
# NAME      cache folder $NAVSIM_EXP_ROOT/rl_features/<NAME>; train outputs $NAVSIM_EXP_ROOT/rl_policy/<NAME>/<run label>
# HISTORY_CACHE=1 (cache mode, runs with past frames / ego history): mounts the past-frame cache HISTORY_MOUNTS times
# (default 8; one squashfuse mount serves ~11 files/s) and passes it as --history-cache-path, as the training script does.
# Train mode: each argument is one run's options (run label from them, e.g. exact_beta0.1); EXTRA_ARGS go to every run.
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${NAVSIM_EXP_ROOT:?}" "${NAME:?}" "${MODE:?}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
FEATURES=$NAVSIM_EXP_ROOT/rl_features/$NAME
EXTRA_ARGS=${EXTRA_ARGS:-}

HISTORY_ARGS=()
HISTORY_DIRS=()
if [[ ${HISTORY_CACHE:-0} == 1 ]]; then
    HISTORY_IMAGE=${HISTORY_IMAGE:-$OPENSCENE_DATA_ROOT/images/history_navtrain.sqfs}
    HISTORY_BASE=$NAVSIM_EXP_ROOT/mnt/history_${SLURM_JOB_ID:-$$}
    for i in $(seq 1 "${HISTORY_MOUNTS:-8}"); do
        mkdir -p "$HISTORY_BASE/$i"
        squashfuse "$HISTORY_IMAGE" "$HISTORY_BASE/$i"
        HISTORY_DIRS+=("$HISTORY_BASE/$i")
    done
    echo "Mounted $HISTORY_IMAGE ${#HISTORY_DIRS[@]} times under $HISTORY_BASE"
    HISTORY_ARGS=(--history-cache-path "$(IFS=:; echo "${HISTORY_DIRS[*]}")")
fi
cleanup_mounts() {
    for d in "${HISTORY_DIRS[@]}"; do fusermount -u "$d" || true; rmdir "$d" || true; done
    if [[ ${#HISTORY_DIRS[@]} -gt 0 ]]; then rmdir "$HISTORY_BASE" || true; fi
}
trap cleanup_mounts EXIT

case $MODE in
    cache)
        : "${RUN:?}" "${CKPT:?}"
        python $NAVSIM_DEVKIT_ROOT/scripts/training/cache_policy_head_features.py \
            --run-dir $NAVSIM_EXP_ROOT/$RUN --checkpoint "$CKPT" --out-dir $FEATURES \
            --shard ${SHARD:-0} --num-shards ${NUM_SHARDS:-1} "${HISTORY_ARGS[@]}" $EXTRA_ARGS ;;
    shortlist)
        : "${RUN:?}" "${CKPT:?}"
        python $NAVSIM_DEVKIT_ROOT/scripts/training/cache_scorer_shortlists.py \
            --run-dir $NAVSIM_EXP_ROOT/$RUN --checkpoint "$CKPT" --out-dir $NAVSIM_EXP_ROOT/scorer_shortlists/$NAME \
            --frames ${FRAMES:-0 1 2} --shard ${SHARD:-0} --num-shards ${NUM_SHARDS:-1} "${HISTORY_ARGS[@]}" $EXTRA_ARGS ;;
    train)
        for options in "$@"; do
            label=$(echo "$options" | sed 's/--//g; s/objective //; s/ \+/_/g; s/_\([0-9]\)/\1/g')
            echo "=== $label: $options $EXTRA_ARGS"
            python $NAVSIM_DEVKIT_ROOT/scripts/training/rl_policy_head.py --features-dir $FEATURES \
                --out-dir $NAVSIM_EXP_ROOT/rl_policy/$NAME/$label $options $EXTRA_ARGS
        done ;;
    *) echo "MODE must be cache, shortlist or train" >&2; exit 1 ;;
esac
