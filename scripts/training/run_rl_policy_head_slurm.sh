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
# NAME      cache folder $NAVSIM_EXP_ROOT/rl_features/<NAME>; train outputs $NAVSIM_EXP_ROOT/rl_policy/<NAME>/<run label>
# Train mode: each argument is one run's options (run label from them, e.g. exact_beta0.1); EXTRA_ARGS go to every run.
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${NAVSIM_EXP_ROOT:?}" "${NAME:?}" "${MODE:?}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
FEATURES=$NAVSIM_EXP_ROOT/rl_features/$NAME
EXTRA_ARGS=${EXTRA_ARGS:-}

case $MODE in
    cache)
        : "${RUN:?}" "${CKPT:?}"
        python $NAVSIM_DEVKIT_ROOT/scripts/training/cache_policy_head_features.py \
            --run-dir $NAVSIM_EXP_ROOT/$RUN --checkpoint "$CKPT" --out-dir $FEATURES \
            --shard ${SHARD:-0} --num-shards ${NUM_SHARDS:-1} $EXTRA_ARGS ;;
    train)
        for options in "$@"; do
            label=$(echo "$options" | sed 's/--//g; s/objective //; s/ \+/_/g; s/_\([0-9]\)/\1/g')
            echo "=== $label: $options $EXTRA_ARGS"
            python $NAVSIM_DEVKIT_ROOT/scripts/training/rl_policy_head.py --features-dir $FEATURES \
                --out-dir $NAVSIM_EXP_ROOT/rl_policy/$NAME/$label $options $EXTRA_ARGS
        done ;;
    *) echo "MODE must be cache or train" >&2; exit 1 ;;
esac
