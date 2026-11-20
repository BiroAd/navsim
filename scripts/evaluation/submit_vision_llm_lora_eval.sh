#!/bin/bash
# Split official evaluation of one checkpoint: one GPU job (batched inference over navtest + navhard_two_stage), then
# one CPU job per (split, selection) that starts when the GPU job has finished successfully.
#   RUN=p3_aux_policy/2026.11.06.23.42.11 CKPT=checkpoints/merged/epoch-epoch=03.ckpt NAME=p3_aux_policy_ep03 \
#       SELECTIONS="shortlist_v1:256 shortlist:256" scripts/evaluation/submit_vision_llm_lora_eval.sh
# SELECTIONS  navtest selections, each "selection" or "selection:k" (shortlist size k; default K). Default
#             "shortlist_v1 shortlist" (v1 and v2). navhard gets each selection without _v1 once (argmax: one navtest
#             job gives v1 and v2). Output names: <PRED>_<selection>[_k<k>|_lam<lambda>]_<traffic>_split.
# K, LAMBDA   default shortlist size (64) / mixture lambda (10), passed to the CPU jobs
# GPU_PARTITION partition of the GPU job (default: the script's gpu,ai), e.g. test
# PRED        reuse existing predictions under $NAVSIM_EXP_ROOT/predictions/<PRED> instead of running the GPU job
# AFTER       with PRED: job id the CPU jobs wait for (afterok), e.g. a GPU job still running
# SCORES_PRED sub-score logits from this prediction folder instead of the model's own (RL-3: the start model,
#             p3_aux_policy_ep03); passed on to the CPU jobs
# NAVTEST_TIME / NAVHARD_TIME: CPU job limits (default 60 / 90 min since 2026-11-18: 30 / 50 min timed out 43 of ~150
#             jobs submitted at once, on crowded nodes; completed jobs took 16-28 / 29-37 min).
# CPU jobs: 4 CPUs (cpu partition: at most 2000 MB per core, more memory adds idle cores). navtest 8000M / 30 min
# (18-21 min, up to 24.6 min with 20 jobs at once on 2026-11-09; 6G peaked at 5.9 GB and job 14927280 was OOM-killed at the end), navhard 6G / 50 min (31 min, 4.7 GB; 31-40 min with 20+ jobs at once, 14935530 timed out at 40 min).
# Run from the repository root with the environment active (sbatch passes it on).
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${NAVSIM_EXP_ROOT:?}"
SELECTIONS=${SELECTIONS:-shortlist_v1 shortlist}
K=${K:-64}
SCRIPTS=$NAVSIM_DEVKIT_ROOT/scripts/evaluation
LOGS=$NAVSIM_EXP_ROOT/predictions/logs
mkdir -p "$LOGS"

if [[ -n ${PRED:-} ]]; then
    DEPENDENCY=(${AFTER:+--dependency=afterok:$AFTER})
else
    : "${RUN:?}" "${CKPT:?}" "${NAME:?}"
    PRED=$NAME
    GPU_JOB=$(RUN=$RUN CKPT=$CKPT NAME=$NAME sbatch --parsable ${GPU_PARTITION:+-p $GPU_PARTITION} \
        --job-name="vlm_predict_$NAME" --output="$LOGS/%x-%j.log" "$SCRIPTS/run_vision_llm_lora_predict_slurm.sh")
    echo "GPU predictions: job $GPU_JOB -> predictions/$NAME"
    DEPENDENCY=(--dependency=afterok:$GPU_JOB)
fi

# score <split> <selection> <k> <time> <mem>; NAME is unset so the GPU job's NAME cannot become the output name.
score() {
    local job
    job=$(env -u NAME PRED=$PRED SELECTION=$2 K=$3 SPLIT=$1 sbatch --parsable "${DEPENDENCY[@]}" \
        --cpus-per-task=4 --mem=$5 --time=$4 \
        --job-name="vlm_score_${1%%_*}_${PRED}_$2_k$3" --output="$LOGS/%x-%j.log" \
        "$SCRIPTS/run_vision_llm_lora_score_predictions_slurm.sh")
    echo "$1 $2 (k $3): job $job"
}

declare -A NAVHARD=()
for entry in $SELECTIONS; do
    selection=${entry%%:*}
    k=$K; [[ $entry == *:* ]] && k=${entry##*:}
    score navtest "$selection" "$k" "${NAVTEST_TIME:-01:00:00}" 8000M
    NAVHARD["${selection%_v1}:$k"]=1
done
for entry in "${!NAVHARD[@]}"; do
    score navhard_two_stage "${entry%%:*}" "${entry##*:}" "${NAVHARD_TIME:-01:30:00}" 6G
done
