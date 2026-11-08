# vision_llm_lora checkpoint evaluation

Offline evaluation of phase 1 checkpoints with a proxy PDM score, to choose epochs and compare selection rules
without running the simulator. Training runs and their settings: [experiments.md](experiments.md). Last updated
2026-11-04 (round 6).

## Method

- Script: `scripts/evaluation/eval_vision_llm_lora_checkpoints.py`, SLURM wrapper
  `scripts/evaluation/run_vision_llm_lora_checkpoint_eval_slurm.sh <run>/<date>:<epochs> ...` (1 GPU). Options via
  environment: `OUT_DIR` (default `exp/checkpoint_eval`), `TRAIN_LOG_STRIDE` (default 20; 0 = val only), `BATCH_SIZE`,
  `EXTRA_ARGS` (e.g. `--max-scenes 64` for a smoke test).
- Data: the training cache. Val = navtrain val split, 18,179 scenes in 214 logs; train subset = every 20th train log,
  5,681 scenes in 51 logs (eval mode, to measure the train/val gap without dropout or epoch averaging).
- Each checkpoint runs with the Hydra config saved in its run folder; only the trainable parameters come from the
  checkpoint (the job fails if one is missing).
- Proxy score: every codebook entry of every navtrain scene is PDM-scored in `exp/codebook_scores_navtrain`, so the
  score of a chosen entry is a lookup. It uses the v2 sub-scores (NC, DAC, DDC, TLC, EP, TTC, LK, HC), non-reactive
  log-replay traffic and no two-frame extended comfort. It ranks checkpoints; it is **not** comparable to published
  navtest PDMS or EPDMS. First official number (2026-11-05, details in experiments.md, Round 6 decision): p3_heads ep 5
  with score-head selection, navtest one-stage non-reactive, **EPDMS 0.771** vs proxy heads score 0.818 (different
  split and metric); two-frame extended comfort, which the proxy lacks, is 0 in 65% of scenes and costs up to ~0.07.
- Speed: ~26 scenes/s on one A100-40GB at any batch 16–256 (GPU-bound; peak memory 11 / 13.5 / 18 / 28 GB at batch
  32 / 64 / 128 / 256), ~16 min per checkpoint with the train subset.
- bf16 makes argmax metrics depend slightly on batch size (batch 32 vs 64 on 2,048 scenes: score 0.8703 vs 0.8718,
  FDE 1.510 vs 1.505). Treat differences below ~2 standard errors (~0.006) as noise; the per-scene files allow paired
  comparisons, which are tighter.
- Crash `FileNotFoundError` in `multiprocessing/resource_sharer` (the socket the DataLoader uses to pass tensors,
  under /tmp, disappears mid-run): jobs 14808877/80 (right after an `srun --overlap` into them) and 14849801 (no
  attach), so files in /tmp vanish on the compute nodes for an UNKNOWN reason. Since 2026-11-02 the script sets
  `torch.multiprocessing.set_sharing_strategy("file_system")` (shared memory in /dev/shm, no socket). Possibly
  related, UNVERIFIED: evaluation jobs log that transformers falls back from flash-linear-attention (Triton kernels
  may fail to build under /tmp); speed only. Still avoid `srun --overlap` into running jobs.

### Output (`exp/checkpoint_eval/`)

- `summary_<run>.csv`: one row per checkpoint and split. `logs/<job>-<id>.log`: progress and one `run=` line per
  checkpoint and split. `<run>_ep<NN>_<split>.npz`: per scene (token, argmax id, top-64 ids and logits, metrics).
- Score-head runs also write `heads_<run>.csv` (see below).

| Column | Meaning |
|---|---|
| `sel_score` | proxy score of the argmax entry (what the agent drives), mean over scenes; `score_se` its standard error |
| `sel_<metric>` | sub-scores of that entry |
| `ade`, `fde` | of the output trajectory (refined for `p1c_refine`); `anchor_*` = of the argmax entry |
| `expected_score` | Σ_k p_k · score_k, the score of the whole predicted distribution (what an expected-reward RL loss maximizes) |
| `oracle_score` | best entry per scene (0.9817 on val; 84.6% of val scenes have a perfect entry) |
| `entropy` | of the predicted distribution, nats |
| `head_all`, `head_topk` | score-head runs: true score of the entry with the highest predicted total over all entries / among the policy's top 64 |
| `head_mae` | mean \|predicted total − true score\| over the policy's top 64 |
| `comb_<λ>` | true score of argmax(log p + λ · log predicted total) |
| `headsel_<metric>` | sub-scores of the `head_all` entry |
| `headdiag_<metric>_*` | per head: BCE (as in the loss), MAE over all entries and over the top 64, accuracy (binary metrics), mean target |

### Why the oracle is 0.9817, not 1

0.008 comes from 146 val scenes (0.8%) where every entry scores 0. The other 0.010 is spread over ~2,650 scenes,
mostly ego progress (2,058 scenes: no entry progresses as far as the PDM-Closed reference while staying safe), then
history comfort (400) and 337 scenes where all sub-scores read 1 but the total is below 1 (the human-penalty filter
sets those sub-scores to 1; the total is computed without it).

## Results (val proxy score of the argmax entry, ±0.003)

Jobs 14808877–80 (batch 16) and 14809329–32 (batch 256), 2026-11-01.

| Epoch | linear (baseline) | trajemb 4q | r2 q8 ed T2 | p1c_base | p1c_ed05 | p1c_refine | p1c_scores |
|---|---|---|---|---|---|---|---|
| 3 | 0.612 | 0.636 | 0.657 | 0.661 | – | 0.652 | 0.666 |
| 5 | 0.630 | 0.650 | 0.656 | 0.666 | 0.664 | 0.655 | 0.664 |
| 7 | 0.642 | 0.663 | 0.683 | 0.682 | 0.681 | 0.682 | 0.690 |
| 9/10 | 0.646 | 0.673 | 0.684 | 0.684 | **0.695** | 0.681 | 0.696 |
| 12/13 | 0.645 | 0.673 | 0.686 | **0.689** | 0.694 | **0.687** | **0.699** |
| 14/15 | **0.651** | **0.677** | **0.693** | 0.684 | – | 0.687 | 0.689 |
| 17/19 | 0.650 | 0.676 | 0.692 | 0.685 | 0.693 | 0.687 | 0.693 |

Epochs: baseline and p1_trajemb 3,5,7,10,12,15,17; r2 3,5,7,9,12,15,19; p1c_base 3,5,7,10,12,14,19; p1c_ed05
5,7,10,13,19; p1c_refine 3,5,7,10,12,15,19; p1c_scores 3,5,7,10,12,14,19.

## Score-head selection (`p1c_scores`)

| Epoch | argmax | head_all | head_topk | head_mae | train argmax | train head_all |
|---|---|---|---|---|---|---|
| 5 | 0.664 | 0.732 | 0.707 | 0.33 | 0.836 | 0.839 |
| 10 | 0.696 | 0.743 | 0.721 | 0.31 | 0.861 | 0.867 |
| 12 | 0.699 | **0.746** | 0.724 | 0.32 | 0.871 | 0.872 |
| 19 | 0.693 | 0.740 | 0.718 | 0.31 | 0.873 | 0.874 |

Epoch 12, per scene: head_all is better in 31% of scenes, worse in 50%, equal in 19%; mean +0.047 ± 0.003 (paired).
Zero-score scenes drop from 22.6% to 10.0%: the heads mainly avoid leaving the road and collisions, at a small
cost elsewhere. No gain on train, where the argmax already fits.

### Combined rule and per-head quality (epoch 10, val; job 14810651, `exp/checkpoint_eval_heads/heads_p1c_scores.csv`)

Combining does not beat the heads alone: the score rises monotonically with λ in argmax(log p + λ · log predicted
total): λ = 0 (argmax) 0.696, 0.3 → 0.709, 1 → 0.712, 3 → 0.720, 10 → 0.723, 30 → 0.734, ∞ (head_all) 0.743.

Sub-scores of the chosen entry, argmax vs head_all: DAC 0.815 → 0.927, NC 0.968 → 0.983, DDC → 0.991, but ego progress
0.847 → **0.661** and history comfort 0.98 → 0.936. The heads pick safe but slow trajectories.

| Head | BCE (all entries) | MAE all | MAE policy top-64 | Accuracy | Mean target |
|---|---|---|---|---|---|
| no_at_fault_collisions | 0.330 | 0.187 | 0.105 | 0.845 | 0.72 |
| drivable_area_compliance | 0.298 | 0.184 | **0.285** | 0.867 | 0.40 |
| driving_direction_compliance | 0.312 | 0.143 | 0.129 | – | 0.62 |
| traffic_light_compliance | 0.058 | 0.032 | 0.008 | 0.977 | 0.97 |
| ego_progress | 0.387 | 0.079 | 0.091 | – | 0.77 |
| time_to_collision_within_bound | 0.349 | 0.212 | 0.124 | 0.850 | 0.68 |
| lane_keeping | 0.449 | 0.276 | 0.184 | 0.782 | 0.67 |
| history_comfort | 0.096 | 0.056 | 0.029 | 0.963 | 0.63 |

The drivable-area head is the worst on the plausible candidates (top-64 MAE 0.285, higher than over all entries),
exactly where the choice is made and where the argmax fails most; lane keeping is second. Traffic light and comfort
are nearly solved. This motivated runs `p1d_total` and `p1d_headw` (experiments.md).

## Round 4: score-head loss variants (val, jobs 14849802/03, 14850096; batch 64)

All are `p1c_scores` plus one change (experiments.md). argmax = model's own choice; head_all = choice by the
predicted total; bias = signed error of the drivable-area head on the policy's top 64 (0 = calibrated).

| Run | Epoch | argmax | train | FDE | DAC | EP | expected | head_all | head_all EP | filter τ=0.8 | DAC bias |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p1c_scores | 12 | 0.699 | 0.871 | 1.98 | 0.817 | 0.846 | 0.639 | 0.746 | 0.655 | 0.734 | −0.087 |
| p1d_scorelabels | 7 | 0.709 | 0.883 | 2.03 | 0.829 | 0.847 | 0.645 | 0.746 | 0.657 | 0.726 | −0.075 |
| | 10 | 0.713 | 0.899 | 2.00 | 0.836 | 0.844 | 0.664 | 0.749 | 0.663 | 0.728 | −0.083 |
| | 12 | **0.720** | 0.908 | 1.99 | 0.841 | 0.846 | 0.673 | 0.746 | 0.647 | 0.729 | −0.104 |
| | 15 | **0.720** | 0.916 | 1.98 | 0.842 | 0.848 | 0.678 | 0.749 | 0.665 | 0.727 | −0.080 |
| | 19 | 0.719 | 0.920 | 2.00 | 0.842 | 0.846 | 0.681 | 0.748 | 0.669 | 0.725 | −0.086 |
| p1d_total | 7 | 0.682 | 0.850 | 2.05 | 0.803 | 0.851 | 0.608 | 0.739 | 0.668 | 0.728 | −0.014 |
| | 10 | 0.698 | 0.863 | 1.96 | 0.819 | 0.848 | 0.630 | 0.753 | 0.698 | 0.736 | −0.010 |
| | 12 | 0.693 | 0.870 | 1.99 | 0.813 | 0.845 | 0.635 | 0.753 | 0.666 | 0.732 | −0.037 |
| | 15 | 0.698 | 0.876 | 1.96 | 0.817 | 0.850 | 0.643 | **0.755** | 0.691 | 0.735 | −0.019 |
| p1d_headw | 7 | 0.696 | 0.860 | 1.99 | 0.814 | 0.848 | 0.620 | 0.744 | 0.646 | 0.735 | −0.085 |
| | 10 | 0.705 | 0.864 | 1.97 | 0.825 | 0.844 | 0.638 | 0.746 | 0.667 | 0.736 | −0.063 |
| | 12 | 0.700 | 0.874 | 1.99 | 0.819 | 0.846 | 0.642 | 0.748 | 0.664 | 0.733 | −0.081 |
| | 15 | 0.699 | 0.876 | 1.97 | 0.819 | 0.847 | 0.650 | 0.748 | 0.671 | 0.732 | −0.071 |
| | 19 | 0.699 | 0.878 | 1.98 | 0.818 | 0.847 | 0.651 | 0.743 | 0.673 | 0.732 | −0.069 |

p1d_total epoch 19: argmax 0.696, train 0.876, FDE 1.96, DAC 0.816, EP 0.848, expected 0.644, head_all 0.753
(EP 0.696), filter τ=0.8 0.731, DAC bias −0.022 (job 14850096, completed with the file_system sharing fix).

5. **Score-aware labels improve the model's own choice by +0.02** (0.720 vs 0.699, ~7 standard errors), almost all
   through drivable area (0.817 → 0.842) with unchanged progress and FDE; expected_score rises most (0.68). Flat from
   epoch 12; the train score keeps rising (gap 0.19–0.20 vs 0.17), so it also fits the train set harder.
6. **The combined-score loss calibrates the heads:** drivable-area bias −0.087 → −0.01…−0.04, heads' choice 0.755
   with more progress (0.69–0.70 vs 0.65–0.66). It does not change the model's own choice.
7. **Fixed per-head weights do nothing measurable** (argmax within noise, bias unchanged).
8. The gains do not add: score-aware labels move the argmax towards what the heads already pick (gap 0.047 → 0.026);
   the ceiling of choosing by the heads stays at ~0.75. Phase 3 (features that can learn road edges) is the lever.

## Round 5: both flags in phase 1, first phase 3 runs (val, jobs 14863619–21, 2026-11-03)

`p1e_combo` = `p1c_scores` + score-aware labels + combined-score loss (phase 1). `p3_total` / `p3_scorelabels` = phase 3
(LLM LoRA r32 + vision LoRA r16) from `p1d_total` / `p1d_scorelabels` epoch 10, each keeping its loss (experiments.md,
Round 5). Phase 3 epochs count from 0 within the phase 3 run; "start" rows repeat the round 4 table. Columns as in
round 4.

| Run | Epoch | argmax | train | FDE | DAC | EP | expected | head_all | head_all EP | filter τ=0.8 | DAC bias |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p1e_combo | 7 | 0.708 | 0.885 | 2.03 | 0.831 | 0.847 | 0.645 | 0.751 | 0.687 | 0.737 | −0.016 |
| | 10 | 0.721 | 0.899 | 1.96 | 0.844 | 0.845 | 0.667 | 0.760 | 0.701 | **0.745** | −0.021 |
| | 12 | 0.719 | 0.911 | 1.98 | 0.840 | 0.849 | 0.669 | **0.762** | 0.694 | 0.743 | −0.037 |
| | 15 | 0.719 | 0.917 | 1.97 | 0.840 | 0.845 | 0.676 | 0.761 | 0.699 | 0.744 | −0.032 |
| | 19 | 0.722 | 0.921 | 1.98 | 0.845 | 0.847 | 0.681 | 0.761 | 0.701 | 0.742 | −0.040 |
| p3_total | start (p1d_total 10) | 0.698 | 0.863 | 1.96 | 0.819 | 0.848 | 0.630 | 0.753 | 0.698 | 0.736 | −0.010 |
| | 2 | 0.716 | 0.878 | 1.90 | 0.837 | 0.852 | 0.653 | 0.763 | 0.693 | 0.744 | −0.034 |
| | 4 | 0.718 | 0.876 | 1.86 | 0.838 | 0.847 | 0.659 | 0.763 | 0.710 | 0.741 | −0.025 |
| | 6 | 0.723 | 0.881 | 1.83 | 0.842 | 0.849 | 0.665 | **0.765** | 0.716 | 0.744 | −0.024 |
| | 8 | 0.721 | 0.884 | **1.81** | 0.840 | 0.851 | 0.668 | 0.764 | **0.724** | 0.741 | −0.021 |
| p3_scorelabels | start (p1d_scorelabels 10) | 0.713 | 0.899 | 2.00 | 0.836 | 0.844 | 0.664 | 0.749 | 0.663 | 0.728 | −0.083 |
| | 2 | 0.724 | 0.907 | 1.95 | 0.846 | 0.848 | 0.677 | 0.753 | 0.683 | 0.730 | −0.075 |
| | 4 | 0.729 | 0.918 | 1.91 | 0.852 | 0.850 | 0.683 | 0.752 | 0.683 | 0.731 | −0.081 |
| | 6 | 0.729 | 0.920 | 1.89 | 0.850 | 0.850 | 0.688 | 0.752 | 0.687 | 0.729 | −0.085 |
| | 8 | **0.735** | 0.923 | 1.89 | **0.858** | 0.848 | **0.696** | 0.753 | 0.685 | 0.729 | −0.093 |

9. **Phase 3 lifts the argmax by +0.022–0.025** (p3_total 0.698 → 0.723, p3_scorelabels 0.713 → 0.735) and FDE by
   6–7% (1.96 → 1.81, 2.00 → 1.89). As in phase 1, the gain is drivable area (+0.02) with unchanged progress (~0.85). `p3_scorelabels` epoch 8 is
   the best argmax checkpoint so far (0.735), `p3_total` epoch 6 the best heads choice (0.765).
10. **The heads gain little from phase 3:** head_all +0.012 (p3_total) and +0.004 (p3_scorelabels), half or less of the
    argmax gain, so the heads-vs-argmax gap shrinks (0.055 → 0.042 and 0.036 → 0.018). In p3_total the heads choose
    faster trajectories over training (EP 0.698 → 0.724); in p3_scorelabels the drivable-area head drifts more
    pessimistic (bias −0.083 → −0.093). The shared LoRA features improve the policy but the head losses (weight 1, BCE
    ~0.28 vs soft CE ~4–7) barely pull them: candidate for a higher score-loss weight in phase 3.
11. **No overfitting through phase 3 epoch 8:** val score flat or rising at every evaluated epoch, train/val gap
    0.158–0.163 (p3_total) and 0.183–0.191 (p3_scorelabels), no larger than at the start (0.165 / 0.185). Train ADE
    halves (0.46 → 0.24 for p3_total) while the train score rises only 0.863 → 0.884. Epoch 9 was not evaluated (W&B
    flat from 8 to 9).
12. **The two phase 1 flags combine** (`p1e_combo`): argmax as score-aware labels (0.721), heads above the combined
    loss alone (0.760–0.762 vs 0.755) and calibrated (bias −0.02…−0.04); the safety filter also reaches its best value
    (0.745). Epoch 10 is the phase 3 start: later epochs are equal on val and only fit train more (0.899 → 0.921).
13. The safety filter (τ = 0.8) and the combination rule still never beat head_all in any run: best λ (mostly 30) is
    0.003–0.012 below head_all (e.g. p3_total epoch 8: 0.761 vs 0.764).
14. Training-log score (lenient) − evaluation score (strict) = 0.027–0.031 at every phase 3 epoch, so the log ranks
    epochs correctly but is not comparable across tables.

## Round 6: phase 3 loss weighting (val, jobs 14870345–48, 2026-11-04)

Four phase 3 runs with the round 5 phase 3 settings; only the start checkpoint and the loss weights differ
(experiments.md, Round 6). Weights = soft CE / expected distance / sub-score BCE / combined-score BCE.
`p3_combo` and `p3_combo_balanced` start from `p1e_combo` epoch 10 (score-aware labels), `p3_balanced` and `p3_heads`
from `p1d_total` epoch 10 (plain labels). Epochs: 2, 4, 6, 8 and the saved best (the balanced run stopped after
epoch 5, `p3_heads` after epoch 8). Columns as in round 4; "start" rows repeat earlier tables.

| Run | Weights | Epoch | argmax | train | FDE | DAC | EP | expected | head_all | head_all EP | filter τ=0.8 | DAC bias |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| start `p1e_combo` | | 10 | 0.721 | 0.899 | 1.96 | 0.844 | 0.845 | 0.667 | 0.760 | 0.701 | 0.745 | −0.021 |
| p3_combo | 1/1/1/1 | 2 | 0.732 | 0.908 | 1.93 | 0.855 | 0.851 | 0.681 | 0.768 | 0.710 | 0.752 | −0.026 |
| | | 4 | 0.737 | 0.918 | 1.89 | 0.860 | 0.850 | 0.689 | 0.766 | 0.724 | 0.751 | −0.039 |
| | | 6 | 0.739 | 0.921 | 1.87 | 0.861 | 0.850 | 0.696 | 0.769 | 0.719 | 0.750 | −0.041 |
| | | 8 | 0.738 | 0.923 | 1.86 | 0.861 | 0.850 | 0.697 | 0.769 | 0.728 | 0.749 | −0.042 |
| p3_combo_balanced | 1/1/5/5 | 2 | 0.737 | 0.910 | 1.94 | 0.862 | 0.847 | 0.687 | 0.778 | 0.705 | 0.765 | −0.040 |
| | | 4 | 0.742 | 0.916 | 1.88 | 0.866 | 0.850 | 0.692 | 0.776 | 0.737 | 0.760 | −0.022 |
| | | 6 | 0.746 | 0.920 | 1.84 | 0.867 | 0.851 | 0.702 | 0.785 | 0.739 | 0.767 | −0.034 |
| | | 7 | **0.750** | 0.924 | 1.83 | **0.873** | 0.849 | 0.706 | 0.785 | 0.740 | 0.767 | −0.040 |
| | | 8 | **0.750** | 0.923 | 1.83 | 0.871 | 0.850 | **0.707** | 0.786 | 0.746 | 0.769 | −0.036 |
| start `p1d_total` | | 10 | 0.698 | 0.863 | 1.96 | 0.819 | 0.848 | 0.630 | 0.753 | 0.698 | 0.736 | −0.010 |
| p3_balanced | 1/1/5/5 | 2 | 0.730 | 0.880 | 1.88 | 0.852 | 0.847 | 0.667 | 0.781 | 0.698 | 0.761 | −0.037 |
| | | 4 | 0.729 | 0.885 | 1.84 | 0.847 | 0.850 | 0.667 | 0.781 | 0.733 | 0.758 | −0.019 |
| | | 5 | 0.727 | 0.881 | **1.82** | 0.845 | 0.850 | 0.666 | 0.788 | 0.733 | 0.760 | −0.018 |
| p3_heads | **0/0**/5/5 | 2 | 0.749 | 0.879 | 2.27 | 0.877 | 0.856 | 0.694 | 0.814 | 0.761 | 0.792 | −0.006 |
| | | 4 | 0.740 | 0.886 | 2.42 | 0.866 | 0.858 | 0.689 | 0.815 | 0.787 | 0.791 | −0.009 |
| | | 5 | 0.743 | 0.883 | 2.55 | 0.870 | 0.859 | 0.696 | **0.818** | 0.799 | **0.795** | −0.012 |
| | | 8 | 0.730 | 0.878 | 2.79 | 0.858 | 0.856 | 0.690 | 0.814 | **0.817** | 0.790 | −0.007 |

For comparison, round 5 at weight 1: p3_total epoch 6 argmax 0.723, head_all 0.765; p3_scorelabels epoch 8 argmax
0.735, head_all 0.753.

Paired differences on the same 18,179 val scenes (mean ± standard error of the per-scene difference):

| Comparison | Choice | Difference |
|---|---|---|
| p3_combo_balanced ep 8 − p3_combo ep 6 (weight 5 vs 1, score-aware labels) | argmax | +0.011 ± 0.002 |
| p3_combo_balanced ep 8 − p3_combo ep 8 | head_all | +0.016 ± 0.001 |
| p3_combo_balanced ep 8 − p3_scorelabels ep 8 (previous best argmax) | argmax | +0.015 ± 0.002 |
| p3_combo ep 6 − p3_scorelabels ep 8 (both flags vs labels only, weight 1) | argmax | +0.005 ± 0.002 |
| p3_balanced ep 5 − p3_total ep 6 (weight 5 vs 1, plain labels) | head_all | +0.023 ± 0.002 |
| p3_balanced ep 2 − p3_total ep 2 | argmax | +0.015 ± 0.002 |
| p3_heads ep 5 − p3_balanced ep 5 (score losses only vs weight 5) | head_all | +0.031 ± 0.002 |
| p3_heads ep 5 − p3_combo_balanced ep 8 | head_all | +0.032 ± 0.002 |
| p3_heads ep 5 head_all − p3_combo_balanced ep 8 argmax | best selector vs best policy | +0.068 ± 0.003 |
| p3_heads ep 2 − p1d_total ep 10 (its start; the policy head gets no gradient) | argmax | +0.051 ± 0.003 |

Per-head quality on val (BCE over all entries / MAE on the policy's top 64; the round-3 table above has p1c_scores):

| Checkpoint | NC | DAC | EP | TTC | LK |
|---|---|---|---|---|---|
| p3_total ep 6 (weight 1) | 0.329 / 0.081 | 0.261 / 0.247 | 0.390 / 0.094 | 0.345 / 0.104 | 0.438 / 0.174 |
| p3_combo ep 8 (weight 1) | 0.327 / 0.076 | 0.260 / 0.241 | 0.390 / 0.093 | 0.342 / 0.100 | 0.440 / 0.173 |
| p3_combo_balanced ep 8 (weight 5) | 0.313 / 0.070 | 0.232 / 0.223 | 0.385 / 0.089 | 0.327 / 0.089 | 0.425 / 0.162 |
| p3_balanced ep 5 (weight 5) | 0.312 / 0.075 | 0.230 / 0.230 | 0.385 / 0.089 | 0.326 / 0.096 | 0.422 / 0.168 |
| p3_heads ep 5 (score losses only) | **0.291 / 0.059** | **0.179 / 0.179** | **0.380 / 0.081** | **0.302 / 0.074** | **0.406 / 0.138** |

15. **Score-loss weight 5 improves both choices when the labels are score-aware.** `p3_combo_balanced` epoch 8 (or 7,
    identical) is the **best argmax model so far: 0.750** (+0.015 over p3_scorelabels, +0.011 over p3_combo, both
    paired ≥5 SE), with the best FDE of the score-label runs (1.83), the highest expected score (0.707) and heads 0.786.
    The gain is drivable area (0.861 → 0.871); progress unchanged.
16. **Without score-aware labels, weight 5 helps the heads (+0.023) more than the argmax.** `p3_balanced` reaches
    0.730 argmax at epoch 2 (+0.015 over p3_total at the same epoch) and then slides to 0.727, while its heads keep
    rising (0.781 → 0.788) and FDE keeps falling (1.82). It stopped after epoch 5 because the argmax monitor
    (lenient, peak at epoch 2) ran out of patience 3, so its heads were not at their end point.
17. **At weight 1 the two flags add little in phase 3:** `p3_combo` is +0.005 ± 0.002 over p3_scorelabels (argmax)
    and +0.004 over p3_total (heads, 0.769 vs 0.765). The weighting, not the flags, is what moved round 6.
18. **Score losses only (`p3_heads`) gives the best selector: head_all 0.818**, +0.031 over weight 5 and +0.068 over
    the best argmax model. Every head improves (DAC top-64 MAE 0.179 vs 0.22–0.25, nearly calibrated: bias −0.006…
    −0.012), and the heads pick faster trajectories as training goes on (EP of the choice 0.761 → 0.817) while keeping
    DAC ~0.97. The safety filter also reaches its best value (0.795). head_all is flat from epoch 2 (0.814–0.818);
    W&B val BCE bottoms at epoch 3 while the train heads keep improving (0.916 → 0.941), so later epochs only fit train.
19. **`p3_heads` gives up imitation:** FDE 1.96 → 2.27 (epoch 2) → 2.79 (epoch 8). Its argmax is not trained, but the
    LoRA features become score-aware, so the frozen policy head picks safer entries at first (0.698 → 0.749 at epoch 2,
    DAC 0.877) and then drifts down (0.730 at epoch 8) as the features move away from the imitation solution.
20. Combination rule and safety filter: best λ (30 in every run) is 0.003–0.012 below head_all; the filter τ = 0.8
    stays 0.013–0.028 below head_all. Neither is worth adding.
21. **No overfitting of the argmax through epoch 8** in the three runs with imitation loss: val flat or rising at every
    evaluated epoch, train/val gap 0.176–0.185 (p3_combo), 0.173–0.174 (p3_combo_balanced), 0.149–0.156 (p3_balanced).
22. **Lenient training log − strict evaluation**: argmax 0.027–0.031, heads 0.030–0.032 (p3_heads epoch 5: W&B
    0.850 vs 0.818). The training-log head score is therefore ~0.03 too high, like the argmax score.

## Round 7: auxiliary perception losses (val, 2026-11-05/06)

Three phase 3 runs with the aux BEV decoder (BEV semantic / drivable area / lane heading, weights 3 / 3 / 1;
experiments.md, "Auxiliary perception losses"). Each has a round 6 twin that differs only by the aux losses.
Jobs 14885931 (p3_aux), 14885932 (p3_heads_aux), 14891502–06 (p3_combo_balanced_aux, one epoch per job).
"train" = every 20th train log (heads score for the heads-only runs, argmax for the combo run).

| Run (twin) | Epoch | argmax | FDE | expected | head_all | head_topk | head MAE | train |
|---|---|---|---|---|---|---|---|---|
| **p3_aux** (p3_heads; CE/ED 0, from p1d_total) | 2 | 0.759 | 2.28 | 0.702 | 0.819 | 0.806 | 0.232 | 0.916 |
| | 4 | 0.749 | 2.59 | 0.698 | 0.824 | 0.809 | 0.218 | 0.928 |
| | **5** | 0.747 | 2.64 | 0.698 | **0.826** | 0.812 | 0.214 | 0.933 |
| | 6 | 0.740 | 2.75 | 0.694 | 0.822 | 0.806 | 0.211 | 0.936 |
| | 8 | 0.735 | 2.92 | 0.690 | 0.822 | 0.810 | 0.208 | 0.940 |
| **p3_heads_aux** (p3_heads ep 5–8; from p3_heads ep 5) | 0 | 0.736 | 2.63 | 0.688 | 0.821 | 0.805 | 0.221 | 0.934 |
| | 1 | 0.733 | 2.65 | 0.688 | 0.823 | 0.810 | 0.215 | 0.935 |
| | **2** | 0.737 | 2.74 | 0.696 | **0.824** | 0.809 | 0.210 | 0.940 |
| | 4 | 0.729 | 2.91 | 0.691 | 0.823 | 0.809 | 0.206 | 0.943 |
| **p3_combo_balanced_aux** (p3_combo_balanced; 1/1/5/5 + aux, from p1e_combo) | 4 | 0.752 | 1.83 | 0.704 | 0.787 | 0.777 | 0.258 | 0.916 |
| | 6 | 0.764 | 1.79 | 0.717 | 0.795 | 0.788 | 0.250 | 0.920 |
| | 7 | 0.764 | 1.79 | 0.719 | 0.797 | 0.790 | 0.250 | 0.923 |
| | 8 | 0.765 | **1.78** | 0.721 | 0.796 | 0.789 | 0.247 | 0.924 |
| | **9** | **0.766** | 1.79 | **0.722** | **0.797** | 0.791 | 0.246 | 0.925 |

Paired differences (18,179 val scenes, mean ± standard error):

| Comparison | argmax | head_all | FDE |
|---|---|---|---|
| p3_aux ep 5 − p3_heads ep 5 | +0.0042 ± 0.0023 | **+0.0082 ± 0.0014** | |
| p3_heads_aux ep 2 − p3_heads ep 5 | −0.0057 ± 0.0022 | +0.0053 ± 0.0014 | |
| **p3_combo_balanced_aux ep 9 − p3_combo_balanced ep 8** | **+0.0158 ± 0.0017** | **+0.0111 ± 0.0013** | −0.034 ± 0.008 |
| p3_combo_balanced_aux ep 9 − ep 8 | +0.0004 ± 0.0011 | +0.0012 ± 0.0007 | +0.008 ± 0.004 |
| p3_combo_balanced_aux ep 9 − p3_aux ep 5 | +0.0185 ± 0.0027 | −0.0295 ± 0.0018 | −0.845 ± 0.018 |

Findings:
23. **Aux losses help everywhere they were tried**: the score heads (+0.008, p3_aux), and with imitation on also the
    policy (+0.016 argmax, ~9 SE) and the heads (+0.011) of p3_combo_balanced, with a slightly better FDE. Best policy
    now: **p3_combo_balanced_aux ep 9, argmax 0.766** (was 0.750).
24. Joint beats staged: p3_aux (aux from phase 1) +0.008 vs p3_heads_aux (aux added to converged p3_heads) +0.005.
25. Still two specialists: the best heads are p3_aux's (0.826, but FDE 2.6 and an untrained policy), the best policy is
    p3_combo_balanced_aux's (heads 0.797). The imitation terms keep competing with the heads, as in round 6.
26. p3_combo_balanced_aux kept improving to its last epoch (ep 8 and 9 within noise); 10 epochs may be short for it.
27. The proxy ranking carried over to the official benchmarks (final_evaluation.md, fixed filter): aux helps on v1, v2
    and navhard for both selections (p3_aux − p3_heads +0.008 / +0.008 / +0.018; p3_combo_balanced_aux − p3_combo_balanced,
    argmax, +0.010 / +0.011 / +0.016), and p3_aux keeps the best heads (v1 0.889, navhard 0.361). The proxy cannot see
    two-frame comfort, which makes argmax the best v2 navtest choice (0.826) despite its lower proxy score.

## Selection rules combining policy and heads (p3_combo_balanced_aux ep 9, val; job 14900641, 2026-11-06)

Output `exp/checkpoint_eval_selection/` (separate folder, val only). Three families, all scored with the strict proxy
(no two-frame comfort); paired over 18,179 scenes. "changed" = share of scenes not on the argmax entry; EP = ego
progress of the chosen entry; "scored 0" = share of scenes whose chosen entry fails a multiplier.
A `comb_<λ>`: argmax of log p + λ · log predicted total. B `shortlist_<k>` (new column 2026-11-06): the most likely
of the k entries with the best predicted total. D `filter_<τ>`: the most likely entry among those whose predicted
multiplicative product is ≥ τ (from round 4).

| Rule | Score | vs argmax | vs heads | EP | changed | scored 0 |
|---|---|---|---|---|---|---|
| argmax | 0.766 | | −0.031 | 0.850 | 0% | 15.6% |
| heads (head_all) | 0.797 | +0.031 | | 0.759 | | 8.1% |
| A λ = 1 | 0.777 | +0.012 ± 0.001 | −0.020 ± 0.002 | 0.852 | 36% | |
| A λ = 3 | 0.784 | +0.018 ± 0.002 | −0.013 ± 0.002 | 0.850 | 45% | 13.4% |
| A λ = 10 | 0.791 | +0.025 ± 0.002 | −0.006 ± 0.002 | 0.843 | 60% | |
| A λ = 30 | 0.797 | +0.031 ± 0.002 | +0.000 ± 0.002 | 0.827 | 75% | |
| D τ = 0.7 | 0.778 | +0.012 ± 0.002 | −0.019 ± 0.002 | 0.797 | 32% | |
| D τ = 0.9 | 0.783 | +0.017 ± 0.002 | −0.014 ± 0.002 | 0.719 | 54% | 7.9% |
| B k = 4 | 0.797 | +0.031 ± 0.002 | −0.000 ± 0.001 | 0.764 | 99.5% | |
| B k = 16 | 0.802 | +0.037 ± 0.002 | +0.005 ± 0.001 | 0.771 | 98% | 8.2% |
| **B k = 64** | **0.804** | +0.038 ± 0.002 | **+0.007 ± 0.002** | 0.778 | 93% | 8.3% |
| B k = 256 | 0.801 | +0.035 ± 0.002 | +0.004 ± 0.002 | 0.789 | 80% | |
| B k = 1024 | 0.788 | +0.022 ± 0.002 | −0.009 ± 0.002 | 0.815 | 61% | |

Findings:
28. **B (heads shortlist, policy picks) is the only rule that beats the heads**: k = 64 +0.007 ± 0.002 (~4.5 SE), with
    higher ego progress than the heads (0.778 vs 0.759) and the same share of zero scenes (8.3% vs 8.1%). Larger k
    moves towards the policy (EP up, safety down); k = 4 equals the heads.
29. A (mixture) keeps the policy's progress (EP ≈ 0.85) but removes few failures (13.4% zero at λ = 3): a soft
    penalty does not veto entries the policy is confident about. It reaches the heads only at λ = 30.
30. D (safety threshold) removes the most failures (7.9% zero) but gets cautious fast (EP 0.72 at τ = 0.9) and stays
    below the heads.
31. "changed" is not two-frame comfort: B changes the argmax entry in 80–99% of scenes, but whether its choice is
    stable from frame to frame (EC) only the official runs show.
32. Official results (final_evaluation.md, "Selection rules"): B k = 64 / 256 and A λ = 10 on v1 navtest 0.873 / 0.872 /
    0.866, v2 navtest 0.838 / 0.846 / 0.847, navhard 0.334 / 0.311 / 0.246. B keeps navtest EC at 0.61–0.70 (heads
    0.17), so "changed" overstated the instability; A, worst of the three here, ties for best on v2 through EC (0.81).

## Round 8: decoupled policy training, phase 3 policy-only baseline, p1e_combo starts (val, 2026-11-07)

Runs (experiments.md, "Results of the policy-only and p1e_combo runs"): `p3_aux_policy` / `p3_heads_policy` train only
the policy head (541 K parameters) on the frozen p3_aux / p3_heads ep 5 (heads, LoRA, planner queries unchanged);
`p3_policy` is phase 3 with imitation + expected distance only (from p1c_ed05 ep 10, no score heads); `p3_aux_p1e` /
`p3_heads_p1e` repeat p3_aux / p3_heads from `p1e_combo` ep 10. Jobs 14911313–15 and, after the loading fix below,
14911770–72 (`exp/checkpoint_eval/`). Policy-only checkpoints hold only the policy head; the evaluation loads the run's
start checkpoint first (`base_checkpoints`); a first evaluation without that is invalid
(`invalid_policy_only_partial_load/`). Check: the heads of p3_aux_policy equal p3_aux's exactly (paired difference
0.0000), p3_heads_policy's p3_heads' (−0.0001 ± 0.0004).

| Run, epoch | argmax | FDE | heads | A λ = 10 | D τ = 0.9 | B k = 16 | B k = 64 | B k = 256 | argmax scored 0 | k = 64 scored 0 | EP heads / k = 64 | train argmax |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p3_aux ep 5 (stale phase 1 policy) | 0.747 | 2.64 | 0.826 | 0.807 | 0.807 | – | – | – | 17.7% | – | 0.807 / – | |
| **p3_aux_policy ep 1** | 0.788 | 1.97 | 0.826 | 0.823 | 0.818 | **0.830** | **0.830** | 0.828 | 13.1% | 7.1% | 0.807 / 0.819 | 0.903 |
| **p3_aux_policy ep 3** | **0.790** | **1.94** | 0.826 | 0.822 | 0.817 | **0.830** | **0.830** | 0.828 | 13.1% | 7.1% | 0.807 / 0.820 | 0.905 |
| p3_heads ep 5 | 0.743 | 2.55 | 0.818 | 0.797 | 0.801 | – | – | – | 18.3% | – | 0.799 / – | |
| p3_heads_policy ep 2 | 0.777 | 1.96 | 0.818 | 0.810 | 0.809 | 0.823 | 0.822 | 0.818 | 14.3% | 7.6% | 0.799 / 0.813 | 0.905 |
| p3_combo_balanced_aux ep 9 (round 7) | 0.766 | 1.79 | 0.797 | 0.791 | 0.783 | 0.802 | 0.804 | 0.801 | 15.6% | 8.3% | 0.759 / 0.778 | 0.925 |
| p3_policy ep 7 (no heads) | 0.711 | 1.87 | – | – | – | – | – | – | 21.6% | – | – | 0.882 |
| p3_aux_p1e ep 7 | 0.757 | 2.76 | 0.820 | 0.806 | 0.810 | 0.823 | 0.826 | 0.821 | 16.6% | | 0.815 / 0.823 | |
| p3_heads_p1e ep 4 | 0.746 | 2.40 | 0.812 | 0.791 | 0.794 | 0.814 | 0.815 | 0.810 | 17.9% | | 0.780 / 0.797 | |

Paired differences (18,179 val scenes):

| Comparison | Difference |
|---|---|
| argmax: p3_aux_policy ep 1 − p3_aux ep 5 (stale policy) | **+0.041 ± 0.002** |
| argmax: p3_aux_policy ep 1 − p3_combo_balanced_aux ep 9 | **+0.022 ± 0.002** |
| argmax: p3_aux_policy ep 1 − ep 3 | −0.002 ± 0.001 |
| argmax: p3_heads_policy ep 2 − p3_heads ep 5 | +0.034 ± 0.002 |
| argmax: p3_aux_policy ep 1 − p3_heads_policy ep 2 | +0.011 ± 0.002 |
| shortlist k = 64: p3_aux_policy ep 1 − p3_aux heads | +0.0035 ± 0.0014 |
| shortlist k = 64: p3_aux_policy ep 1 − p3_combo_balanced_aux shortlist k = 64 | **+0.026 ± 0.002** |
| shortlist k = 64: p3_heads_policy − p3_heads heads | +0.0038 ± 0.0014 |
| argmax: p3_policy ep 7 − p3_combo_balanced ep 8 | −0.039 ± 0.002 (FDE +0.04 ± 0.01) |
| heads: p3_aux_p1e ep 7 − p3_aux ep 5 | −0.006 ± 0.002 |
| heads: p3_heads_p1e ep 4 − p3_heads ep 5 | −0.006 ± 0.002 |

Findings:
33. **Decoupled training gives the best policy and the best selection.** A policy head trained on p3_aux's frozen
    features reaches argmax 0.788–0.790 (+0.041 over its stale policy, +0.022 over p3_combo_balanced_aux's jointly
    trained policy), FDE 1.94–1.97, and the shortlist reaches **0.830** (k = 16 / 64): above p3_aux's heads (+0.0035),
    with fewer zero scenes (7.1% vs 8.1%) and more progress (EP 0.82 vs 0.81). Features trained only for safety
    scores (and aux) carry enough for imitation; imitation in the LoRA hurts the heads more than it helps the policy.
34. Epochs 1 and 3 are equal (policy-only training converges in one epoch).
35. Aux helps the decoupled policy too (+0.011 argmax, +0.008 shortlist over p3_heads_policy).
36. **Score heads help the policy itself:** the phase 3 imitation-only baseline p3_policy is 0.039 below
    p3_combo_balanced (same recipe with score heads and score-aware labels) at a similar FDE.
37. The stronger phase 1 start (p1e_combo) gives 0.006 weaker heads for both p3_aux and p3_heads; the shortlist with
    its stale p1e policy (0.826 / 0.815) does not compensate.

## Shortlist headroom for RL (p3_aux_policy ep 3; job 14920952, 2026-11-07, `exp/checkpoint_eval/headroom/`)

How much a policy that only re-ranks the heads' shortlist could gain (the question for RL on the policy). New columns of
`eval_vision_llm_lora_checkpoints.py`, per shortlist size k: `_oracle` = best true score in the shortlist, `_top4_oracle`
= best of the policy's 4 most likely entries in it, `_expected` = E[score] under the policy renormalized to it,
`_uniform` = its mean. Val 18,179 scenes; train = every 20th train log (5,681 scenes). The shortlist values equal
round 8 (0.8299 / 0.8278).

| k | val: policy pick | expected | uniform | top-4 oracle | **oracle** | train: policy pick | **oracle** |
|---|---|---|---|---|---|---|---|
| 4 | 0.826 | 0.826 | 0.825 | 0.863 | 0.863 | 0.936 | 0.945 |
| 16 | 0.830 | 0.829 | 0.824 | 0.865 | 0.886 | 0.936 | 0.952 |
| 64 | 0.830 | 0.827 | 0.819 | 0.867 | 0.905 | 0.934 | 0.959 |
| 256 | 0.828 | 0.820 | 0.803 | 0.871 | **0.926** | 0.930 | 0.967 |
| 1024 | 0.815 | 0.792 | 0.737 | 0.871 | 0.952 | 0.921 | 0.977 |
| all 16,384 | 0.789 (argmax) | 0.717 | – | – | 0.982 | 0.904 | 0.987 |

Findings:
38. **Large headroom inside the shortlist on val:** k = 256 holds entries worth 0.926 against the policy's 0.828
    (+0.098); the best of the policy's own 4 favourites already gives 0.871 (+0.043). The oracle ignores two-frame
    comfort and a learned policy will not find the per-scene best, so this is a ceiling, not a target.
39. **The policy does not rank by score inside the shortlist:** at k = 4 / 16 its pick is barely above a random pick
    (0.826 vs 0.825, 0.830 vs 0.824). It ranks by imitation, which is what RL would change. A score-aware policy could
    also use a larger k (ceiling 0.93–0.95 at k = 256–1024), where today's policy loses.
40. **Little headroom on the training scenes:** k = 256 gap +0.037 on train vs +0.098 on val. The heads were trained on
    those scenes (head_all 0.933 train vs 0.826 val), so their shortlist is nearly right there. RL that learns only
    inside the heads' shortlist on training scenes would see a much easier problem than at test time; the RL candidate
    set should not be the training-time shortlist (full codebook, or a broader set), with the shortlist applied only at
    inference.

## RL on the policy head (RL-1 / RL-2, val; jobs 14924451/52, 2026-11-08, `exp/rl_policy/<model>/<run>/summary.csv`)

`scripts/training/rl_policy_head.py` on cached fused states (`exp/rl_features/`, jobs 14921155–60): only the policy
head trains (scene projection, trajectory bias, logit scale); reward = the proxy's codebook score of every entry
(full codebook, no EC). `exact` = −E_π[r] + β·KL(π‖π_start); `grpo` = 8 samples per scene, group-normalized advantage,
+ β·KL. 5 epochs, AdamW lr 1e-4, batch 256, dropout 0.1 on the fused state; ~1 s per epoch. Epoch 0 reproduces the
offline evaluation within the bf16 tie-breaking of the full model (p3_aux_policy argmax 0.7888 vs 0.7895, shortlist
k = 64 0.8295 vs 0.8299, k = 256 0.8269 vs 0.8278; the model's top-two logits differ by a median 0.06, below bf16
resolution at logit ~37, so its argmax is a rounding tie-break: agreement 62%).

| Model, run (epoch 5; ep 4 equal) | argmax | E_π[r] | k = 16 | k = 64 | k = 256 | k = 1024 | k = 256 scored 0 | KL to start | argmax changed |
|---|---|---|---|---|---|---|---|---|---|
| p3_aux_policy ep 3 (start) | 0.789 | 0.717 | 0.830 | 0.830 | 0.827 | 0.812 | 7.6% | 0 | 0 |
| exact β = 0.02 | **0.824** | 0.787 | 0.831 | **0.833** | **0.833** | **0.830** | 6.9% | 0.41 | 70% |
| exact β = 0.05 | 0.821 | 0.779 | 0.831 | 0.833 | 0.833 | 0.828 | 6.8% | 0.23 | 57% |
| exact β = 0.1 | 0.815 | 0.770 | 0.831 | 0.832 | 0.832 | 0.824 | 7.0% | 0.14 | 46% |
| exact β = 0.2 | 0.812 | 0.759 | 0.831 | 0.831 | 0.831 | 0.822 | 7.1% | 0.07 | 33% |
| grpo β = 0.05, G = 8 | 0.767 | 0.726 | 0.829 | 0.827 | 0.819 | 0.799 | 9.5% | 2.72 | 93% |
| p3_heads_policy ep 2 (start) | 0.779 | 0.708 | 0.823 | 0.822 | 0.817 | 0.805 | 8.4% | 0 | 0 |
| exact β = 0.02 | **0.810** | 0.775 | 0.824 | **0.825** | **0.825** | **0.821** | 7.6% | 0.40 | 68% |
| exact β = 0.2 | 0.798 | 0.749 | 0.823 | 0.823 | 0.820 | 0.814 | 8.0% | 0.07 | 33% |
| grpo β = 0.05, G = 8 | 0.752 | 0.714 | 0.821 | 0.818 | 0.811 | 0.791 | 10.1% | 2.82 | 94% |

Findings:
41. **RL makes the policy itself much better, the shortlist only a little.** Exact β = 0.02: argmax +0.035 (0.789 →
    0.824, now close to the shortlist), but shortlist k = 64 / 256 only +0.004 / +0.006 (ceiling 0.905 / 0.926); the
    gain grows with k (k = 1024 +0.018), so a wider shortlist stops hurting. Same for p3_heads_policy (+0.031 argmax,
    +0.003 / +0.008). Smaller β is better on every val metric; the runs converge in 3 epochs.
42. **The limit is the frozen features, not the objective.** The policy head is the same function class as each
    sub-score head (a linear projection of the same fused state against the same trajectory embeddings), so inside the
    heads' shortlist it cannot know more about which entry is better than the heads already did. The shortlist
    headroom needs features that change: the LoRA variants (RL-3 / RL-4).
43. **GRPO with this setting diverges from the start policy** (KL 2.7, 93% of the argmax picks changed) and gets worse
    on val (argmax −0.022) while better on train (shortlist k = 256 0.917 → 0.928): with group-normalized advantages
    the policy gradient has unit scale regardless of the reward differences, and β = 0.05 at lr 1e-4 does not hold it.
    Not a verdict on GRPO; a larger β or lower learning rate would be needed.
44. The argmax change rate (33–70%) and KL say the RL policy picks different trajectories in most scenes; whether that
    costs two-frame comfort only the official v2 navtest shows.

### Extended sweep on p3_aux_policy (jobs 14924652 GRPO / 14924653 exact, 2026-11-08)

Smaller β, higher learning rates, more epochs; GRPO with larger β and groups, and "Dr. GRPO" (`--advantage mean`:
advantage = r − group mean, no division by the group std). Theory, including the KL of the optimum per β:
[theoretical.md](theoretical.md). Last epoch unless noted; "best k = 256" = best epoch on val (selection on val, so
slightly optimistic).

| Run | epochs, lr | argmax | k = 64 | k = 256 | k = 1024 | k = 256 scored 0 | KL val / train | argmax changed | train argmax | best k = 256 (epoch) |
|---|---|---|---|---|---|---|---|---|---|---|
| start | – | 0.789 | 0.830 | 0.827 | 0.812 | 7.6% | 0 / 0 | 0 | 0.891 | – |
| exact β = 0.2 | 5, 1e-4 | 0.812 | 0.831 | 0.831 | 0.822 | 7.1% | 0.07 / 0.06 | 33% | 0.905 | 0.831 (5) |
| exact β = 0.02 | 5, 1e-4 | 0.824 | 0.833 | 0.833 | 0.830 | 6.9% | 0.41 / 0.45 | 70% | 0.917 | 0.833 (3) |
| exact β = 0.02 | 20, 1e-4 | 0.830 | 0.835 | 0.836 | 0.833 | 6.8% | 0.68 / 0.66 | 74% | 0.925 | 0.836 (12) |
| exact β = 0.02 | 10, 3e-4 | 0.829 | 0.835 | 0.835 | 0.833 | 6.9% | 0.78 / 0.73 | 75% | 0.927 | 0.836 (4) |
| exact β = 0.02 | 10, 1e-3 | 0.832 | 0.835 | 0.835 | 0.834 | 7.1% | 1.00 / 0.85 | 77% | 0.935 | 0.836 (3) |
| exact β = 0.01 | 10, 1e-4 | 0.831 | 0.835 | 0.835 | 0.834 | 6.8% | 0.79 / 0.85 | 80% | 0.924 | 0.835 (9) |
| exact β = 0.005 | 10, 1e-4 | 0.832 | 0.834 | 0.834 | 0.834 | 6.8% | 1.09 / 1.18 | 84% | 0.926 | 0.835 (8) |
| **exact β = 0.005** | 10, 3e-4 | **0.834** | **0.836** | **0.836** | **0.835** | 7.0% | 1.69 / 1.73 | 88% | 0.935 | **0.837 (7)** |
| exact β = 0 | 10, 1e-4 | 0.831 | 0.835 | 0.834 | 0.833 | 6.9% | 2.18 / 2.32 | 91% | 0.927 | 0.834 (4) |
| Dr. GRPO β = 0.02, G = 8 | 10, 1e-4 | 0.825 | 0.834 | 0.834 | 0.831 | 6.8% | 0.43 / 0.45 | 69% | 0.917 | 0.834 (9) |
| Dr. GRPO β = 0.02, G = 32 | 10, 1e-4 | 0.828 | 0.834 | 0.834 | 0.832 | 6.8% | 0.50 / 0.52 | 72% | 0.920 | 0.835 (8) |
| Dr. GRPO β = 0.05, G = 32 | 10, 1e-4 | 0.822 | 0.833 | 0.833 | 0.828 | 6.9% | 0.27 / 0.26 | 58% | 0.915 | 0.833 (3) |
| GRPO β = 0.05, G = 8 | 5, 1e-4 | 0.767 | 0.827 | 0.819 | 0.799 | 9.5% | 2.72 / 2.83 | 93% | 0.905 | 0.820 (1) |
| GRPO β = 0.1, G = 8 | 10, 1e-4 | 0.781 | 0.829 | 0.822 | 0.806 | 9.3% | 2.36 / 2.47 | 90% | 0.916 | 0.823 (2) |
| GRPO β = 0.2, G = 8 | 10, 1e-4 | 0.797 | 0.829 | 0.825 | 0.814 | 8.7% | 1.30 / 1.52 | 83% | 0.920 | 0.827 (1) |
| GRPO β = 0.2, G = 8 | 10, 3e-5 | 0.796 | 0.829 | 0.826 | 0.814 | 8.3% | 0.86 / 1.09 | 80% | 0.913 | 0.827 (5) |
| GRPO β = 0.2, G = 32 | 10, 1e-4 | 0.803 | 0.830 | 0.825 | 0.816 | 8.6% | 1.30 / 1.62 | 82% | 0.923 | 0.827 (1) |
| GRPO β = 0.2, G = 128 | 10, 1e-4 | 0.808 | 0.831 | 0.827 | 0.820 | 8.4% | 1.19 / 1.55 | 81% | 0.925 | 0.829 (1) |
| GRPO β = 0.5, G = 8 | 10, 1e-4 | 0.805 | 0.830 | 0.829 | 0.820 | 7.9% | 0.37 / 0.47 | 64% | 0.916 | 0.829 (7) |
| GRPO β = 0.5, G = 32 | 10, 1e-4 | 0.810 | 0.831 | 0.830 | 0.822 | 7.8% | 0.43 / 0.57 | 64% | 0.919 | 0.830 (3) |

Findings:
45. **Every exact variant plateaus at the same place:** val shortlist k = 64 / 256 ≈ 0.835–0.836 and argmax ≈ 0.83,
    whatever β (0 to 0.02), learning rate (1e-4 to 1e-3) or epochs (5 to 20). The best run (β = 0.005, lr 3e-4:
    k = 256 0.837 at epoch 7, argmax up to 0.835) is +0.010 / +0.046 over the start and only +0.003 over β = 0.02 after 5 epochs. The
    shortlist ceiling (k = 256 oracle 0.926) stays far away: the capacity limit of finding 42 (theoretical.md §6). On
    train the head also stays below the β = 0.02 optimum (argmax 0.935 at most vs 0.965).
46. **The learning rate and the epochs only change how fast the plateau is reached** (lr 1e-3: 3 epochs; lr 1e-4: ~12
    epochs at β = 0.02). Lower β buys almost nothing on val (+0.001) for much more KL (0.4 → 1.7) and more changed picks
    (70% → 88%), i.e. more risk for two-frame comfort. A moderate setting (β = 0.02, 10–20 epochs) is the safer
    candidate for the official run; β = 0.005 / lr 3e-4 the val-best one.
47. **Dr. GRPO matches the exact objective** (β = 0.02, G = 32: k = 256 0.834–0.835 vs 0.835–0.836 exact, similar KL),
    as expected from theory (same expected gradient, plus sampling noise; G = 32 slightly better than G = 8).
48. **Standard GRPO (std-normalized advantage) drifts and gets worse on val in every setting** (k = 256 0.819–0.830 vs
    0.827 start; argmax changed 64–93%, KL 0.4–2.7), while improving on train: it overfits. Larger β (0.5) and larger
    groups (G = 128) reduce the damage, a lower learning rate does not. The std division sets a per-scene effective
    β ≈ β · σ_group and amplifies noise between nearly equal entries (theoretical.md §5). For the sampled variant in
    stage 2 (reactive rewards), use Dr. GRPO.

## Train vs val for every evaluated checkpoint

Proxy score of the argmax entry and ADE/FDE (m) on val and on the train subset (every 20th train log), all in eval mode.
After epoch ~10–12 the val columns stay flat while the train columns keep improving (train ADE halves from epoch 10
to 19), so the gap grows: the head memorizes the train scenes without getting better on new ones. Val ADE/FDE do not
rise noticeably; the later epochs are not worse on val, only more fitted to train. This is why phase 3 starts from
epoch 10.

| Run | Epoch | val score | train score | gap | val ADE | val FDE | train ADE | train FDE |
|---|---|---|---|---|---|---|---|---|
| training_vision_llm_lora_agent | 3 | 0.612 | 0.789 | 0.177 | 1.056 | 2.589 | 0.790 | 1.888 |
| training_vision_llm_lora_agent | 5 | 0.630 | 0.820 | 0.190 | 1.010 | 2.454 | 0.670 | 1.557 |
| training_vision_llm_lora_agent | 7 | 0.642 | 0.843 | 0.201 | 0.996 | 2.404 | 0.551 | 1.250 |
| training_vision_llm_lora_agent | 10 | 0.646 | 0.869 | 0.222 | 0.997 | 2.387 | 0.344 | 0.700 |
| training_vision_llm_lora_agent | 12 | 0.645 | 0.874 | 0.229 | 0.999 | 2.383 | 0.274 | 0.524 |
| training_vision_llm_lora_agent | 15 | 0.651 | 0.883 | 0.231 | 0.986 | 2.350 | 0.219 | 0.384 |
| training_vision_llm_lora_agent | 17 | 0.650 | 0.885 | 0.235 | 0.985 | 2.346 | 0.203 | 0.350 |
| p1_trajemb | 3 | 0.636 | 0.805 | 0.169 | 0.997 | 2.473 | 0.748 | 1.809 |
| p1_trajemb | 5 | 0.650 | 0.824 | 0.174 | 0.918 | 2.239 | 0.680 | 1.615 |
| p1_trajemb | 7 | 0.663 | 0.833 | 0.170 | 0.912 | 2.218 | 0.645 | 1.514 |
| p1_trajemb | 10 | 0.673 | 0.849 | 0.176 | 0.854 | 2.065 | 0.529 | 1.210 |
| p1_trajemb | 12 | 0.673 | 0.859 | 0.186 | 0.858 | 2.071 | 0.478 | 1.069 |
| p1_trajemb | 15 | 0.676 | 0.870 | 0.193 | 0.865 | 2.081 | 0.422 | 0.921 |
| p1_trajemb | 17 | 0.676 | 0.869 | 0.194 | 0.876 | 2.099 | 0.399 | 0.849 |
| r2_trajemb_T2_q8_ed | 3 | 0.657 | 0.819 | 0.162 | 0.908 | 2.208 | 0.690 | 1.631 |
| r2_trajemb_T2_q8_ed | 5 | 0.656 | 0.829 | 0.173 | 0.889 | 2.150 | 0.629 | 1.469 |
| r2_trajemb_T2_q8_ed | 7 | 0.683 | 0.839 | 0.156 | 0.847 | 2.035 | 0.558 | 1.274 |
| r2_trajemb_T2_q8_ed | 9 | 0.684 | 0.859 | 0.175 | 0.827 | 1.990 | 0.487 | 1.074 |
| r2_trajemb_T2_q8_ed | 12 | 0.686 | 0.864 | 0.178 | 0.829 | 1.981 | 0.406 | 0.856 |
| r2_trajemb_T2_q8_ed | 15 | 0.692 | 0.869 | 0.176 | 0.840 | 1.999 | 0.349 | 0.708 |
| r2_trajemb_T2_q8_ed | 19 | 0.692 | 0.867 | 0.175 | 0.849 | 2.015 | 0.320 | 0.630 |
| p1c_base | 3 | 0.661 | 0.819 | 0.158 | 0.921 | 2.254 | 0.691 | 1.643 |
| p1c_base | 5 | 0.666 | 0.838 | 0.172 | 0.860 | 2.080 | 0.604 | 1.418 |
| p1c_base | 7 | 0.682 | 0.848 | 0.166 | 0.839 | 2.036 | 0.535 | 1.223 |
| p1c_base | 10 | 0.684 | 0.859 | 0.175 | 0.822 | 1.971 | 0.421 | 0.917 |
| p1c_base | 12 | 0.689 | 0.868 | 0.179 | 0.837 | 2.001 | 0.371 | 0.785 |
| p1c_base | 14 | 0.684 | 0.873 | 0.189 | 0.836 | 1.996 | 0.330 | 0.673 |
| p1c_base | 19 | 0.685 | 0.875 | 0.190 | 0.852 | 2.020 | 0.286 | 0.557 |
| p1c_ed05 | 5 | 0.664 | 0.832 | 0.168 | 0.868 | 2.101 | 0.622 | 1.455 |
| p1c_ed05 | 7 | 0.681 | 0.845 | 0.164 | 0.842 | 2.031 | 0.548 | 1.258 |
| p1c_ed05 | 10 | 0.695 | 0.863 | 0.168 | 0.823 | 1.976 | 0.428 | 0.941 |
| p1c_ed05 | 13 | 0.694 | 0.873 | 0.178 | 0.830 | 1.973 | 0.346 | 0.710 |
| p1c_ed05 | 19 | 0.693 | 0.880 | 0.187 | 0.838 | 1.980 | 0.283 | 0.543 |
| p1c_refine | 3 | 0.652 | 0.803 | 0.151 | 0.913 | 2.220 | 0.715 | 1.693 |
| p1c_refine | 5 | 0.655 | 0.824 | 0.169 | 0.854 | 2.076 | 0.644 | 1.501 |
| p1c_refine | 7 | 0.682 | 0.843 | 0.160 | 0.841 | 2.027 | 0.569 | 1.312 |
| p1c_refine | 10 | 0.681 | 0.853 | 0.172 | 0.817 | 1.986 | 0.464 | 1.037 |
| p1c_refine | 12 | 0.687 | 0.861 | 0.173 | 0.805 | 1.957 | 0.404 | 0.883 |
| p1c_refine | 15 | 0.687 | 0.873 | 0.186 | 0.812 | 1.972 | 0.333 | 0.702 |
| p1c_refine | 19 | 0.687 | 0.872 | 0.185 | 0.822 | 1.994 | 0.291 | 0.597 |
| p1c_scores | 3 | 0.666 | 0.821 | 0.156 | 0.918 | 2.233 | 0.692 | 1.638 |
| p1c_scores | 5 | 0.664 | 0.836 | 0.172 | 0.848 | 2.062 | 0.611 | 1.423 |
| p1c_scores | 7 | 0.690 | 0.854 | 0.165 | 0.837 | 2.020 | 0.542 | 1.233 |
| p1c_scores | 10 | 0.696 | 0.861 | 0.165 | 0.822 | 1.974 | 0.430 | 0.934 |
| p1c_scores | 12 | 0.699 | 0.871 | 0.171 | 0.828 | 1.983 | 0.377 | 0.789 |
| p1c_scores | 14 | 0.689 | 0.872 | 0.183 | 0.829 | 1.980 | 0.331 | 0.670 |
| p1c_scores | 19 | 0.693 | 0.873 | 0.180 | 0.833 | 1.983 | 0.283 | 0.542 |
| p1d_total | 7 | 0.682 | 0.850 | 0.168 | 0.848 | 2.045 | 0.570 | 1.303 |
| p1d_total | 10 | 0.698 | 0.863 | 0.165 | 0.816 | 1.959 | 0.456 | 1.012 |
| p1d_total | 12 | 0.693 | 0.870 | 0.177 | 0.828 | 1.985 | 0.408 | 0.870 |
| p1d_total | 15 | 0.698 | 0.876 | 0.178 | 0.817 | 1.957 | 0.342 | 0.695 |
| p1d_total | 19 | 0.696 | 0.876 | 0.180 | 0.824 | 1.961 | 0.308 | 0.607 |
| p1d_headw | 7 | 0.696 | 0.860 | 0.164 | 0.821 | 1.989 | 0.522 | 1.196 |
| p1d_headw | 10 | 0.705 | 0.864 | 0.160 | 0.821 | 1.970 | 0.427 | 0.931 |
| p1d_headw | 12 | 0.700 | 0.874 | 0.174 | 0.832 | 1.987 | 0.369 | 0.772 |
| p1d_headw | 15 | 0.699 | 0.876 | 0.177 | 0.826 | 1.965 | 0.316 | 0.629 |
| p1d_headw | 19 | 0.699 | 0.878 | 0.179 | 0.839 | 1.984 | 0.287 | 0.555 |
| p1d_scorelabels | 7 | 0.709 | 0.883 | 0.173 | 0.843 | 2.033 | 0.526 | 1.198 |
| p1d_scorelabels | 10 | 0.713 | 0.899 | 0.185 | 0.835 | 2.001 | 0.416 | 0.896 |
| p1d_scorelabels | 12 | 0.720 | 0.908 | 0.189 | 0.836 | 1.991 | 0.359 | 0.735 |
| p1d_scorelabels | 15 | 0.720 | 0.916 | 0.196 | 0.834 | 1.982 | 0.301 | 0.580 |
| p1d_scorelabels | 19 | 0.719 | 0.920 | 0.201 | 0.840 | 1.995 | 0.274 | 0.514 |
| p1e_combo | 7 | 0.708 | 0.885 | 0.177 | 0.843 | 2.032 | 0.531 | 1.206 |
| p1e_combo | 10 | 0.721 | 0.899 | 0.178 | 0.823 | 1.965 | 0.427 | 0.920 |
| p1e_combo | 12 | 0.719 | 0.911 | 0.193 | 0.832 | 1.983 | 0.373 | 0.771 |
| p1e_combo | 15 | 0.719 | 0.917 | 0.198 | 0.828 | 1.968 | 0.317 | 0.626 |
| p1e_combo | 19 | 0.722 | 0.921 | 0.199 | 0.836 | 1.978 | 0.290 | 0.551 |
| p3_total (phase 3) | 2 | 0.716 | 0.878 | 0.162 | 0.800 | 1.902 | 0.389 | 0.817 |
| p3_total (phase 3) | 4 | 0.718 | 0.876 | 0.158 | 0.784 | 1.860 | 0.322 | 0.642 |
| p3_total (phase 3) | 6 | 0.723 | 0.881 | 0.158 | 0.774 | 1.829 | 0.267 | 0.502 |
| p3_total (phase 3) | 8 | 0.721 | 0.884 | 0.163 | 0.767 | 1.813 | 0.237 | 0.430 |
| p3_scorelabels (phase 3) | 2 | 0.724 | 0.907 | 0.183 | 0.821 | 1.954 | 0.357 | 0.730 |
| p3_scorelabels (phase 3) | 4 | 0.729 | 0.918 | 0.189 | 0.801 | 1.909 | 0.302 | 0.594 |
| p3_scorelabels (phase 3) | 6 | 0.729 | 0.920 | 0.191 | 0.797 | 1.885 | 0.263 | 0.488 |
| p3_scorelabels (phase 3) | 8 | 0.735 | 0.923 | 0.188 | 0.795 | 1.887 | 0.244 | 0.441 |
| p3_combo (phase 3) | 2 | 0.732 | 0.908 | 0.176 | 0.812 | 1.933 | 0.370 | 0.763 |
| p3_combo (phase 3) | 4 | 0.737 | 0.918 | 0.180 | 0.796 | 1.894 | 0.311 | 0.618 |
| p3_combo (phase 3) | 6 | 0.739 | 0.921 | 0.182 | 0.790 | 1.869 | 0.267 | 0.501 |
| p3_combo (phase 3) | 8 | 0.738 | 0.923 | 0.185 | 0.788 | 1.861 | 0.244 | 0.440 |
| p3_combo_balanced (phase 3) | 2 | 0.737 | 0.910 | 0.173 | 0.818 | 1.942 | 0.377 | 0.782 |
| p3_combo_balanced (phase 3) | 4 | 0.742 | 0.916 | 0.174 | 0.789 | 1.878 | 0.319 | 0.636 |
| p3_combo_balanced (phase 3) | 6 | 0.746 | 0.920 | 0.174 | 0.781 | 1.841 | 0.273 | 0.508 |
| p3_combo_balanced (phase 3) | 7 | 0.750 | 0.924 | 0.174 | 0.775 | 1.830 | 0.256 | 0.469 |
| p3_combo_balanced (phase 3) | 8 | 0.750 | 0.923 | 0.173 | 0.774 | 1.826 | 0.248 | 0.447 |
| p3_balanced (phase 3) | 2 | 0.730 | 0.880 | 0.149 | 0.793 | 1.876 | 0.390 | 0.817 |
| p3_balanced (phase 3) | 4 | 0.729 | 0.885 | 0.156 | 0.775 | 1.838 | 0.318 | 0.633 |
| p3_balanced (phase 3) | 5 | 0.727 | 0.881 | 0.153 | 0.767 | 1.815 | 0.289 | 0.560 |
| p3_heads (phase 3, no imitation loss) | 2 | 0.749 | 0.879 | 0.131 | 0.994 | 2.271 | 0.726 | 1.537 |
| p3_heads (phase 3, no imitation loss) | 4 | 0.740 | 0.886 | 0.146 | 1.062 | 2.425 | 0.819 | 1.743 |
| p3_heads (phase 3, no imitation loss) | 5 | 0.743 | 0.883 | 0.140 | 1.120 | 2.548 | 0.874 | 1.853 |
| p3_heads (phase 3, no imitation loss) | 8 | 0.730 | 0.878 | 0.148 | 1.236 | 2.793 | 0.989 | 2.099 |

## Human-penalty filter: evaluation score vs training-log score

NAVSIM also scores the human trajectory of each original scene; a metric the human fails (e.g. "off road" because of
map errors, "collision" from noisy boxes) is reported as 1 for the prediction (`navsim/evaluate/pdm_score.py:170-201`).
That loop skips `multiplicative_metrics_prod` and `weighted_metrics`, and the final score is built from those
(`compute_final_scores` in `run_pdm_score.py`), so the **official total ignores the filter**; only the reported
sub-scores show it.

| Number | Sub-scores | Total | Relation to official |
|---|---|---|---|
| `sel_score` here (`codebook_scores_navtrain` `score`) | filtered | without filter | same behaviour (strict) |
| `val/pdm_scores/argmax_score` in training logs, score-aware labels | filtered (cache) | recomputed from filtered sub-scores | lenient |

Measured on p1d_scorelabels epoch 10, val, same chosen entries: strict 0.713, with the filter 0.741 (+0.028); 1,977
of 18,179 scenes (10.9%) differ, never in the other direction. Compare runs on `sel_score`; the training-log value
is only for choosing epochs within a run (it is ~0.03 too high).

**Update 2026-11-05:** the paragraph above describes this devkit before the upstream human-filter fix (commit
359c7f72), which is now ported into `navsim/evaluate/pdm_score.py`. With the fix the official total *is* built from the
filtered sub-scores, i.e. it behaves like the lenient training-log number, and `sel_score` here behaves like the
pre-fix (EPDMS*) total. Both remain proxies (navtrain val, precomputed log-replay scores, no two-frame comfort); they
rank checkpoints, and the strict `sel_score` stays the one to compare runs with (paired per scene).
Official numbers: [final_evaluation.md](final_evaluation.md).

## Findings

1. **Score follows FDE across architectures** (linear 0.65 < trajemb 4 queries 0.68 < 8 queries + ED or score heads
   0.69–0.70) but saturates earlier: +0.015–0.027 from epoch 5 to 7, small gains to epoch 10–15, flat after.
   Epochs 3–5 are clearly worse; after ~12 only the train/val gap grows (0.165 → 0.18–0.19).
2. **Drivable area is the main loss:** the argmax entry fails DAC in ~19–22% of val scenes (~10% on train); EP
   (~0.85) and LK (~0.91) follow. The val/train gap in score is mostly DAC and LK.
3. **Selecting by the predicted sub-scores is the largest gain found** (+0.047), with no retraining. Not yet in the
   agent: needs a selection flag in `compute_trajectory` (proposed, not applied) and an official EPDMS check.
4. `expected_score` rises until the last epoch while the argmax score is flat after ~12: the distribution keeps
   moving mass off zero-score entries.

## Open

- Epoch 12 of the head analysis repeats epoch 10: λ = 0.3/1/3/10/30 → 0.712/0.715/0.721/0.729/0.736, head_all
  0.747 (ego progress 0.655, DAC 0.930); DAC head top-64 MAE 0.287, lane keeping 0.194.
- Evaluate the safety filter on the p1d runs too, and try a per-head calibration (temperature/offset fitted on half
  of val) before filtering.

### Safety filter and head bias (p1c_scores epoch 12, val; job 14811610, `exp/checkpoint_eval_filter/full/`)

Rule: argmax of p among entries whose predicted NC·DAC·DDC·TLC ≥ τ (head_all if none passes).

| Rule | Score | Ego progress | Picks changed vs argmax |
|---|---|---|---|
| argmax | 0.700 | 0.846 | – |
| filter τ = 0.3 | 0.701 | 0.817 | 23% |
| filter τ = 0.5 | 0.712 | 0.783 | 38% |
| filter τ = 0.7 | 0.728 | 0.721 | 55% |
| filter τ = 0.8 | **0.734** | 0.670 | 66% |
| filter τ = 0.9 | 0.726 | 0.598 | 81% |
| head_all | **0.747** | 0.655 | – |

The filter does not beat head_all and loses progress the same way: the safety heads are **pessimistic about
plausible entries**. Signed error (predicted − true) on the policy's top 64: DAC −0.087, DDC −0.063, LK −0.050,
TTC −0.041, NC −0.039, EP +0.003, HC −0.006, TLC −0.002. Multiplying four under-estimated pass probabilities compounds
the bias, so a threshold removes most normal-speed entries (66% of picks change at τ = 0.8). The progress head is
unbiased; the loss of progress comes from the safety heads. Calibrating the heads (p1d_total trains the product
directly; post-hoc per-head calibration is untested) is the lever, more than the selection rule.
- Evaluate `p1d_total` / `p1d_headw` checkpoints the same way (argmax and head_all) once trained.
- Official EPDMS: done for p3_heads ep 5 (score_heads, navtest one-stage, 0.771). Still to run: p3_combo_balanced ep 8
  (argmax; also tells whether the argmax policy is more temporally consistent, EC), navhard two-stage.
- The evaluation jobs log that transformers falls back to the reference `chunk_gated_delta_rule` ("flash-linear-attention
  is not installed", although fla 0.5.2 is installed and `is_flash_linear_attention_available()` is true on the
  login node). Phase 3 training (p3_total, 2026-11-03) logs it too, on every rank, plus `causal_conv1d_fn` (`causal_conv1d` is not installed: its wheels need a newer glibc). Transformers resolves the fla kernel on the login node but falls back on the GPU nodes; the import error is swallowed (`transformers/integrations/hub_kernels.py:1010-1020`). Affects speed only (phase 3 runs at ~1.0-1.2 batches/s per GPU, batch 4); not pursued.
