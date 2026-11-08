# vision_llm_lora experiment log

Record of every training run and data artifact, so work can continue from here in a new session.
Update it when a run finishes or a new one starts. Metrics come from W&B (project `biroad-personal/vision_llm_project`);
"best" = epoch with the lowest **validation FDE**. Last updated 2026-11-04.
Official navtest / navhard scores of the final models: [final_evaluation.md](final_evaluation.md).
Headline (2026-11-06, official, fixed human filter): v1 navtest best **0.889** (p3_aux ep 5, heads), v2 navtest best
**0.847 / 0.846** (p3_combo_balanced_aux ep 9, mixture λ = 10 / heads shortlist k = 256), navhard best **0.361**
(p3_aux ep 5, heads). For p3_combo_balanced_aux the heads shortlist (k = 64: v1 0.873, v2 0.838, navhard 0.334) beats
both argmax and heads on all three benchmarks.

## Common setup (all runs below, unless a row says otherwise)

- Agent `vision_llm_lora` (Qwen3.5-4B, frozen backbone), agent YAML defaults: panorama strips + front zoom
  (`image_layout: strips`, `thumbnail: front_zoom`, 3 images, 208 image tokens), ego state as LLM tokens, command
  in the prompt, no late ego fusion, bf16 + sdpa.
- **Phase 1** (`training_phase=1`): only the head, planner queries and ego tokens train; LLM and vision tower frozen.
- Data: navtrain from `exp/training_cache` (103,288 scenes; train 85,109 in 978 logs, val 18,179 in 214 logs,
  split by log). Each epoch = 2,660 batches per GPU.
- 4 × A100-40GB (`ai` or `gpu` partition), batch 8 per GPU, gradient accumulation 4 (effective 128), 20 epochs,
  AdamW, lr_base 5e-4, 10% warmup then cosine to 5% of each group's peak, early-stopping patience 10,
  `phase1_lr_multiplier: 1.0` (head and ego at 5e-4, ego-token norm gain at 6e-5).
- Launch: `scripts/training/run_vision_llm_lora_agent_training_slurm.sh` with `EXPERIMENT_NAME`, `MONITOR`, and
  `++agent.config.*` overrides. Typical: `sbatch -p ai,gpu --cpus-per-task=10 --mem=156G` with `NUM_WORKERS=8`.
- Output: `exp/<EXPERIMENT_NAME>/<date>/checkpoints/{best,epochs}`. Runs up to round 4 saved ~9 GB per checkpoint
  (8.7 GB of it the frozen Qwen weights; trained weights 0.1 GB + optimizer 0.2 GB). From 2026-11-02 on
  `AgentLightningModule.on_save_checkpoint` leaves frozen parameters out (~0.3 GB per phase 1 checkpoint) and
  `on_load_checkpoint` refills them for a native resume; `initialize()` and the eval script read trained keys only. `exp/` totals ~2.2 TB of the 4.3 TB project quota.

## How to read the metrics

- **ADE / FDE** (m, validation): of the output trajectory (argmax codebook entry, or refined entry when the
  refinement head is on). These are the comparable metrics across runs.
- **val loss** is NOT comparable across runs: `label_temperature` shifts the loss floor (soft-label entropy 3.35 at
  T=1, 4.31 at T=2), and the expected-distance term is added to it in `_ed` runs. In every run val loss bottomed out
  around epoch 5–9 while val FDE kept improving, so it is a poor checkpoint criterion (see Findings).
- References on validation: codebook oracle ADE / FDE 0.21 / 0.35 m (nearest entry); constant velocity 2.73 / 6.59;
  constant acceleration 1.59 / 4.23.
- No official PDMS/EPDMS evaluation has been run yet. A proxy PDM score per epoch (navtrain val, precomputed
  non-reactive codebook scores) is in [checkpoint_evaluation.md](checkpoint_evaluation.md) and in the "Eval" columns
  below: best argmax score 0.750 (`p3_combo_balanced` epoch 8), best choice by the predicted sub-scores 0.818 (`p3_heads`
  epoch 5); best phase 1: 0.722 / 0.762 (`p1e_combo`).
- Offline evaluation metrics (the "Eval" columns; full column list in
  [checkpoint_evaluation.md](checkpoint_evaluation.md), Method):
  - **argmax** (`sel_score`): proxy PDM score of the codebook entry with the highest policy probability, i.e. the
    trajectory the agent outputs today. Mean over the 18,179 navtrain val scenes, standard error ~0.003; paired
    differences between two checkpoints on the same scenes are tighter (~0.002).
  - **heads** (`head_all`): proxy score of the entry with the highest *predicted* total score from the 8 sub-score
    heads, searched over all 16,384 entries. This is a different decision rule (pick by predicted safety/progress, not
    by imitation probability); not in the agent yet (needs a selection flag in `compute_trajectory`).
  - **expected**: Σ_k p_k · score_k, the score of the whole policy distribution; what an expected-reward RL stage
    maximizes.
  - **train**: argmax score on the train subset (every 20th train log); **gap** = train − val, the overfitting measure.
  - **DAC / EP** of the chosen entry: drivable-area compliance (the main failure: the argmax entry leaves the drivable area in ~13–22% of val scenes) and ego
    progress (how far it drives relative to the PDM-Closed reference; the heads trade this for safety).
  - **DAC bias**: signed error of the drivable-area head on the policy's top 64 (negative = too pessimistic).
  - **filter τ** / **comb λ**: two ways of mixing policy and heads (argmax among entries the heads call safe;
    argmax of log p + λ · log predicted total). Neither has beaten the heads alone.
  - **Strict vs lenient:** the evaluation script computes the total like the official NAVSIM total (strict). The
    `val/pdm_scores/{argmax,head}_score` series in W&B apply the human-penalty filter (lenient) and read ~0.03 higher;
    use them only to compare epochs within a run.
- Runs started after 2026-11-01 ~01:00 also log `<train|val>/loss_terms/{soft_ce,expected_ade,score_bce,refine_l1}`
  (unweighted, so loss terms compare across runs), `<train|val>/pdm_scores/<metric>_{bce,mae,acc}` and
  `<train|val>/refine/anchor_{ade,fde}`. The six `p1c_*` runs started before that and do not log them. Use the
  `_epoch` series (mean over the whole epoch and all GPUs) to compare runs; during training `train/<name>_mean` is the
  mean over the last logging interval (50 optimizer steps × accumulation), while the `_step` series is the single
  batch at each logging step, i.e. 1 of every 200 batches.

## Results (phase 1)

Settings column = overrides on top of the common setup.

The two "Eval" columns are the offline proxy PDM score on navtrain val (18,179 scenes, strict like the official total,
±0.003; [checkpoint_evaluation.md](checkpoint_evaluation.md)), best evaluated epoch in brackets: **argmax** = the
model's own choice (what the agent outputs today), **heads** = the entry with the best predicted total score from the
sub-score heads (`head_all`; only runs with score heads, not in the agent yet). "–" = not evaluated (checkpoints
deleted or not a candidate).

| Run | W&B | Settings | Best val FDE | ADE at best | Best epoch | Min val loss (epoch) | Epochs run | Eval argmax | Eval heads |
|---|---|---|---|---|---|---|---|---|---|
| `training_vision_llm_lora_agent` (baseline) | ym24ch5t | – (linear head 2048, 4 queries, T=1) | 2.349 | 0.986 | 15 | 6.042 (7) | 18 | 0.651 (15) | – |
| `p1_head1024` | axiuciqb | `head_hidden_dim=1024` | 2.320 | 0.975 | 16 | 5.987 (6) | 17 | – | – |
| `p1_dropout04` | uc3oqdf0 | `head_dropout=0.4` | 2.369 | 0.980 | 16 | 6.081 (8) | 19 | – | – |
| `p1_T2` | o78558h9 | `label_temperature=2.0` | 2.213 | 0.942 | 17 | 6.190 (7) | 18 | – | – |
| `p1_wd01` | z5zkxtg0 | `traj_weight_decay=0.1` | 2.361 | 0.990 | 16 | 6.046 (6) | 17 | – | – |
| `p1_q8` | mv1q4i6j | `num_planner_queries=8` | 2.297 | 0.966 | 16 | 6.036 (6) | 17 | – | – |
| `p1_lr03` | 596fl2kg | `phase1_lr_multiplier=0.3` (lr 1.5e-4) | 2.366 | 0.969 | 15 | 6.001 (13) | 20 | – | – |
| `p1_trajemb` | fcfg93tx | `head_type=trajectory_embedding`, `head_hidden_dim=1024` | 2.069 | 0.858 | 12 | 5.705 (7) | 18 | 0.676 (15) | – |
| `r2_trajemb_T2` | 77qc5cfr | trajemb + 1024 + `label_temperature=2.0` | 2.061 | 0.861 | 13 | 6.052 (7) | 20 | – | – |
| `r2_trajemb_T2_q8` | fk6cpdpi | r2_trajemb_T2 + `num_planner_queries=8` | 2.010 | 0.845 | 10 | 6.025 (5) | 20 | – | – |
| `r2_trajemb_T2_ed` | tekmyk1s | r2_trajemb_T2 + `expected_distance_loss_weight=1.0` | 2.014 | 0.837 | 15 | 7.296 (9) | 20 | – | – |
| `r2_trajemb_T2_q8_ed` | doinc3mg | r2_trajemb_T2 + 8 queries + expected distance 1.0 | 1.981 | 0.829 | 12 | 7.251 (7) | 20 | 0.692 (15) | – |
| `p1c_base` | cyebw2tt | trajemb + 1024 + 8 queries + ED 1.0, T=1 ("p1c base") | 1.971 | 0.822 | 10 | 6.836 (5) | 20 | 0.689 (12) | – |
| `p1c_ed05` | 816dovsf | p1c base + `expected_distance_loss_weight=0.5` | 1.973 | 0.830 | 13 | 6.282 (5) | 20 | 0.695 (10) | – |
| `p1c_ed2` | vroxlkw0 | p1c base + `expected_distance_loss_weight=2.0` | 1.989 | 0.826 | 10 | 7.998 (6) | 20 | – | – |
| `p1c_edonly` | f7pnkuha | p1c base + `trajectory_token_loss_weight=0.0` | 2.452 | 0.984 | 19 | 0.986 (19)¹ | 20 | – | – |
| `p1c_refine` | vd60ifha | p1c base + `refinement_loss_weight=1.0` | 1.958² | 0.805² | 12 | 7.252 (8) | 20 | 0.687 (12)³ | – |
| `p1c_scores` | u1wsfr3p | p1c base + `pdm_score_loss_weight=1.0` ("p1c_scores") | 1.974 | 0.821 | 10 | 7.110 (5) | 20 | 0.699 (12) | 0.746 (12) |
| `p1d_total` | 876ig2il | p1c_scores + `pdm_total_score_loss_weight=1.0` | 1.957 | 0.815 | 10 | 7.376 (6) | 20 | 0.698 (10, 15) | 0.755 (15) |
| `p1d_headw` | 9szkmofr | p1c_scores + `pdm_score_head_weights=[1,3,1,0.5,1,1,2,0.5]` | 1.958 | 0.821 | 13 | 7.093 (7) | 20 | 0.705 (10) | 0.748 (12, 15) |
| `p1d_scorelabels` | b2u2xlif | p1c_scores + `score_weighted_labels=true` | 1.983 | 0.835 | 15 | 7.025 (5)⁴ | 20 | 0.720 (12, 15) | 0.749 (10, 15) |
| **`p1e_combo`** | 2mcmltci | p1c_scores + `score_weighted_labels=true` + `pdm_total_score_loss_weight=1.0` | 1.969 | 0.824 | 10 | 7.287 (5)⁴ | 20 | **0.722** (19), 0.721 (10) | **0.762** (12), 0.760 (10) |

Not listed: W&B `7opvya75`, the same baseline cancelled after 42 min (restarted as ym24ch5t).
¹ expected ADE only (no soft CE in the loss). ² of the refined, off-codebook trajectory, not of the argmax entry.
³ of the argmax codebook entry (the refined trajectory has no precomputed score). ⁴ soft CE against score-weighted
labels, not comparable to the other runs.

**Best phase 1 model:** `p1e_combo` epoch 10 (eval argmax 0.721, heads 0.760, FDE 1.965; later epochs are not better
on val, only more fitted to train). By FDE alone round 3 and 4 are all within noise (1.957–1.989), so the eval score is
the deciding metric since round 4. The old best-model copy `exp/best_models/phase1_r2_trajemb_T2_q8_ed_ep12/` (Hydra
config + README) is superseded; no newer checkpoint has been copied there. Phase 3 results: Round 5 below.

### Checkpoints

- `r2_*` and `p1c_*` runs used `MONITOR=stabilize/val/fde_loss`, so `checkpoints/best/best-epoch=NN.ckpt` is the
  best-FDE epoch.
- `p1_*` runs and the baseline used the old default `val/loss`: their `best/best-epoch=NN-val_loss=0.00.ckpt` is the
  min-val-loss epoch (the `0.00` is a filename bug, fixed since). For the best-FDE epoch use
  `checkpoints/epochs/epoch-epoch=NN.ckpt` with NN from the table.
- Output folders: baseline `training_vision_llm_lora_agent/2026.10.27.23.47.48`, `p1_head1024` and `p1_dropout04`
  `2026.10.29.22.36.03`, `p1_T2` `2026.10.29.22.48.55`, `p1_wd01` `2026.10.30.08.31.03`, `p1_q8`
  `2026.10.30.10.00.02`, `p1_lr03` `2026.10.30.15.24.27`, `p1_trajemb` `2026.10.30.17.04.29`, `r2_trajemb_T2` and
  `r2_trajemb_T2_q8` `2026.10.31.00.18.16`, `r2_trajemb_T2_ed` `2026.10.31.01.33.10`, `r2_trajemb_T2_q8_ed`
  `2026.10.31.04.35.28`, `p1c_base` and `p1c_ed05` `2026.11.01.00.05.04`, `p1c_ed2` `2026.11.01.00.05.03`,
  `p1c_edonly` `2026.11.01.00.10.17`, `p1c_refine` and `p1c_scores` `2026.11.01.00.34.10`. The other dated folders
  next to these hold only `code/` and a log (no checkpoints).

## Findings

1. **The trajectory-embedding head is the main gain** (−12% FDE): shared MLP embedding of each codebook trajectory
   scored by dot product, instead of one weight row per entry; 11.3 M head parameters instead of 54.5 M; overfits far
   less (train/val gap ~0.5 vs 1.1).
2. **8 planner queries (−2.5%) and the expected-distance loss (−2.3%) add up** (−4% together on top of trajemb).
3. **Temperature 2** helps the linear head (−5.8%) but adds nothing to the trajemb head (2.069 → 2.061).
4. **Learning rate is not the limit:** lr × 0.3 changed nothing; val ADE stayed flat while the cosine lowered the lr
   tenfold. Head width 1024 (−1.2%), dropout 0.4 and weight decay 0.1 are within run-to-run noise.
5. **Validation loss is a bad checkpoint criterion:** it rises after epoch ~7 (the head becomes overconfident on its
   wrong samples) while val ADE/FDE keep improving; choosing the min-loss epoch costs 3–7% FDE. Since round 2,
   checkpoints and early stopping use val FDE.
6. Trajemb runs start to overfit in FDE too after epoch ~10–15 (e.g. 1.981 → 2.015 at epoch 20).
7. All runs see the same data order (Lightning's DistributedSampler, seed 0 + epoch); differences between runs also
   come from initialization, dropout and GPU non-determinism, so ±1–2% is within noise for single runs.
8. **Round 3 (`p1c_*`): no single change beats noise on FDE.** Except `p1c_edonly`, best val FDE spans 1.958–1.989.
   The mean over epochs 9–19 is steadier than the single best epoch (table below): ED 0.5 and the score heads are
   marginally best, ED 2.0 marginally worst; none differ by more than ~1.5%.
9. **Temperature 1 vs 2** (`p1c_base` vs `r2_trajemb_T2_q8_ed`): same FDE (best 1.971 vs 1.981, late mean 2.001 vs
   1.999). T=1 ends sharper (val entropy 4.05 vs 4.70 nats at epoch 19) and drifts up slightly more after epoch 10.
10. **The soft CE is needed** (`p1c_edonly`, +24% FDE): without it the val entropy collapses from 6.2 to ~0.4 nats
    within two epochs, FDE stalls at 4.2–4.5 m until epoch 5 and is still falling at epoch 19 (2.452). An
    expected-value objective alone gives a sparse, poorly conditioned signal over 16384 entries. This matters for an
    expected-reward (RL) stage: keep the CE or a KL anchor.
11. **The refiner lowers ADE consistently, FDE within noise:** late mean ADE 0.818 vs 0.830–0.839 for the others
    (−2%), FDE 1.989. Its output is off-codebook, so it is not directly comparable to argmax entries and has no
    precomputed PDM score; `refine/anchor_{ade,fde}` were not logged (the run predates that logging).
12. **The PDM sub-score heads cost no FDE** (1.974, late mean 1.985), but their accuracy is UNKNOWN: the run predates
    the `pdm_scores/*` logging, so it has to be measured on val from the checkpoint.

Mean over epochs 9–19 (val, W&B `_epoch` series):

| Run | Mean FDE | Mean ADE | Val FDE at epoch 19 |
|---|---|---|---|
| `p1c_ed05` | 1.981 | 0.833 | 1.980 |
| `p1c_scores` | 1.985 | 0.830 | 1.980 |
| `p1c_refine` | 1.989 | 0.818 | 1.991 |
| `r2_trajemb_T2_q8_ed` (T=2) | 1.999 | 0.839 | 2.015 |
| `p1c_base` | 2.001 | 0.839 | 2.020 |
| `p1c_ed2` | 2.009 | 0.838 | 2.014 |

## Round 3 setup (`p1c_*`, 2026-11-01, all finished, 20 epochs each)

Base for all: `head_type=trajectory_embedding head_hidden_dim=1024 num_planner_queries=8
expected_distance_loss_weight=1.0 label_temperature=1.0`, `MONITOR=stabilize/val/fde_loss`. One change each (see the
Results table). Jobs 14774321–14774324, 14774326 (`p1c_scores`), 14774672 (`p1c_refine`, resubmitted with 8 CPUs/GPU
and 120 GB to fit a free node). Optimizer parameters: 23.16 M (base), 23.97 M (+0.81 M refiner), 27.49 M (+4.33 M
sub-score heads); all ran at ~1.10 batches/s per GPU, so the score targets and heads do not slow training.

## Round 5 (started 2026-11-02 ~22:40, finished and evaluated 2026-11-03)

| Run | Job | Settings |
|---|---|---|
| `p1e_combo` | 14850668 | `p1c_scores` + `score_weighted_labels=true` + `pdm_total_score_loss_weight=1.0` (the two round-4 winners), 20 epochs; SLURM log `exp/logs/` |
| `p3_total` | 14852291 | **phase 3** from `p1d_total` epoch 10, keeps its loss (`pdm_total_score_loss_weight=1.0`) |
| `p3_scorelabels` | 14852292 | **phase 3** from `p1d_scorelabels` epoch 10, keeps its loss (`score_weighted_labels=true`) |

Phase 3 settings (both): `training_phase=3` (LLM + vision LoRA), same head flags as the start run, batch 4 × accum 8
× 4 GPUs (effective 128), 10 epochs, patience 3, `MONITOR=val/pdm_scores/argmax_score_gap` (1 − proxy score of the
argmax entry, logged in validation since 2026-11-02; computed with the human-penalty filter, so ~0.03 above the
evaluation script's `sel_score`, see checkpoint_evaluation.md), LR multipliers traj 0.10 (5e-5, incl. score heads), ego 0.12,
`lang_lr_multiplier=0.2` (1e-4), `vision_lr_multiplier=0.1` (5e-5), vision LoRA `vision_lora_r=16`, `vision_lora_alpha=32` (6.5 M = 1.8% of the vision tower instead of 26.1 M = 7.3% at the default r 64; LLM LoRA r 32 = 60.9 M = 1.4% unchanged; ~95 M trainable in total). First submitted with r 64 as jobs 14850918/19 and cancelled after 15 / 4 min (their exp/ folders hold no checkpoints). Epoch 10 chosen as the start (not the best epoch)
because the later epochs fit the train set more for the same val score. First runs with slim checkpoints. Phase 3
with both flags and with heavier score losses: Round 6.

Results from W&B (finished 2026-11-03; `val/pdm_scores/*` is lenient, ~0.03 above the evaluation script, see
checkpoint_evaluation.md; gap = train − val of the same metric). Phase 3 ran ~67 min per epoch, 11.2 h for 10 epochs;
slim checkpoints ~1.1 GB (phase 3) and ~0.3 GB (phase 1).

The last two columns are the evaluation script (strict, comparable to every other run; jobs 14863620/21, epochs 2, 4,
6, 8). "start" = the phase 1 checkpoint the run was initialised from, evaluated the same way.

| Run | Epoch | val FDE | val ADE | val argmax (lenient) | val head choice | train argmax | gap | **Eval argmax** | **Eval heads** |
|---|---|---|---|---|---|---|---|---|---|
| `p3_total` (W&B 5jyme0ri) | start (`p1d_total` ep 10) | 1.959 | 0.816 | | | | | 0.698 | 0.753 |
| | 0 | 1.964 | 0.817 | 0.728 | 0.781 | 0.872 | 0.144 | | |
| | 2 | 1.902 | 0.801 | 0.745 | 0.794 | 0.877 | 0.132 | 0.716 | 0.763 |
| | 4 | 1.862 | 0.784 | 0.746 | 0.793 | 0.886 | 0.140 | 0.718 | 0.763 |
| | 6 | 1.831 | 0.774 | 0.750 | 0.795 | 0.890 | 0.140 | **0.723** | **0.765** |
| | 8 | 1.815 | 0.768 | 0.750 | 0.795 | 0.894 | 0.144 | 0.721 | 0.764 |
| | 9 | 1.811 | 0.767 | 0.748 | 0.795 | 0.893 | 0.145 | – | – |
| `p3_scorelabels` (W&B 7hrq6xyq) | start (`p1d_scorelabels` ep 10) | 2.001 | 0.835 | | | | | 0.713 | 0.749 |
| | 0 | 2.022 | 0.846 | 0.740 | 0.778 | 0.915 | 0.175 | | |
| | 2 | 1.953 | 0.820 | 0.753 | 0.782 | 0.919 | 0.165 | 0.724 | 0.753 |
| | 4 | 1.915 | 0.803 | 0.760 | 0.781 | 0.928 | 0.169 | 0.729 | 0.752 |
| | 6 | 1.885 | 0.796 | 0.759 | 0.783 | 0.937 | 0.178 | 0.729 | 0.752 |
| | 8 | 1.887 | 0.795 | 0.763 | 0.783 | 0.941 | 0.178 | **0.735** | 0.753 |
| | 9 | 1.884 | 0.795 | 0.762 | 0.781 | 0.942 | 0.180 | – | – |
| `p1e_combo` (W&B 2mcmltci, phase 1) | 7 | 2.032 | 0.843 | – (started before the logging) | | | | 0.708 | 0.751 |
| | 10 | 1.965 | 0.823 | | | | | 0.721 | 0.760 |
| | 12 | 1.983 | 0.832 | | | | | 0.719 | **0.762** |
| | 15 | 1.968 | 0.828 | | | | | 0.719 | 0.761 |
| | 19 | 1.978 | 0.836 | | | | | **0.722** | 0.761 |

(FDE/ADE of the start and `p1e_combo` rows are from the evaluation script, which matches W&B to ±0.005.)

Observations (details and per-sub-score numbers in [checkpoint_evaluation.md](checkpoint_evaluation.md), Round 5):

1. **Phase 3 improves the model's own choice by ~+0.02–0.025** over its phase 1 start (p3_total 0.698 → 0.723,
   p3_scorelabels 0.713 → 0.735, ≥6 standard errors) and FDE by 6–7% (1.96 → 1.81, 2.00 → 1.89). `p3_scorelabels`
   epoch 8 is the **best argmax model so far (0.735)**, `p3_total` epoch 6 the best heads choice (0.765).
2. **The heads barely gain in phase 3:** p3_total heads 0.753 → 0.765 (+0.012), p3_scorelabels 0.749 → 0.753 (+0.004),
   while the argmax gains twice as much. The gap between heads and argmax shrinks (p3_total 0.055 → 0.042,
   p3_scorelabels 0.036 → 0.018). The drivable-area head stays pessimistic in p3_scorelabels (bias −0.083 → −0.093) and
   calibrated in p3_total (−0.010 → −0.021). This is the motivation for raising the score-loss weight in phase 3.
3. **No overfitting in phase 3 within 8 epochs:** val score and FDE flat or improving at every evaluated epoch; the
   train/val score gap stays at 0.158–0.163 (p3_total) and 0.183–0.191 (p3_scorelabels), i.e. no larger than at the
   start (0.165 / 0.185). Train ADE falls fast (0.46 → 0.24) but the train score hardly moves (p3_total
   0.863 → 0.884): the LoRA fits the human trajectory more closely without making the train choice safer. The rise
   in val imitation CE (from epoch ~2–3) is overconfidence, not a worse decision.
4. **The lenient training-log score is 0.027–0.031 above the evaluation script at every phase 3 epoch** (e.g.
   p3_total ep 6: 0.750 vs 0.723; p3_scorelabels ep 8: 0.763 vs 0.735), so it ranks epochs correctly.
5. **`p1e_combo`: the two flags add up in phase 1.** argmax 0.721 (= `p1d_scorelabels`, from the score-aware labels)
   and heads 0.760–0.762 (better than `p1d_total` 0.755, with the same calibrated drivable-area head, bias
   −0.02…−0.04). Flat after epoch 10 while the train score rises (0.899 → 0.921), so **epoch 10 is the phase 3
   start** (0.721 / 0.760, FDE 1.965).

## Round 6 (2026-11-03, finished and evaluated 2026-11-04): phase 3 loss weighting

Motivation: in round 5 the heads gained little in phase 3 (observation 2). At weight 1 the imitation terms push the
shared LoRA features roughly 10× harder than the head BCE (soft CE gradient up to ~2 per scene, head BCE averaged over
8 × 16384 entries ~0.2; estimate, not measured), so the score losses get weight **5** to bring both to the same order of
magnitude (5 × 0.28 ≈ 1.4 per score term vs soft CE ~4–7). Weight only affects the shared parameters; the head-only
parameters move at the same speed under Adam (their LR multiplier stays 0.10). Phase 3 settings as round 5 (LLM LoRA
r32, vision LoRA r16/α32, lang 0.2, vision 0.1, batch 4 × accum 8 × 4 GPUs, 10 epochs, patience 3).

| Run | Job | W&B | Start | Loss (CE / ED / sub-score BCE / combined BCE) | Labels | Monitor | Epochs run | Compare with |
|---|---|---|---|---|---|---|---|---|
| `p3_combo` | 14863792 | 6de5snzg | `p1e_combo` ep 10 | 1 / 1 / 1 / 1 | score-aware | argmax gap | 10 (11.1 h) | p3_total, p3_scorelabels (flags combined) |
| `p3_combo_balanced` | 14863793 | kukysvj9 | `p1e_combo` ep 10 | 1 / 1 / **5 / 5** | score-aware | argmax gap | 10 (11.2 h) | p3_combo (weighting) |
| `p3_balanced` | 14863794 | pmnt6pe4 | `p1d_total` ep 10 | 1 / 1 / **5 / 5** | plain | argmax gap | 6, early stop (6.6 h) | p3_total (weighting) |
| `p3_heads` | 14863795 | fy9idr4z | `p1d_total` ep 10 | **0 / 0** / 5 / 5 | (unused) | head gap | 9, early stop (10.0 h) | p3_total, p3_balanced (weight 1 → 5 → only heads) |

`p3_heads`: only the score losses train the LoRA, planner queries and heads (the soft CE stays in the graph with weight 0,
so DDP sees every parameter). Monitor `val/pdm_scores/head_score_gap` = 1 − proxy score of the heads' choice (added
2026-11-03, lenient like `argmax_score_gap`).

Results: W&B columns as in round 5 (lenient, ~0.03 above the evaluation); the last two columns are the evaluation
script (strict; jobs 14870345–48; per-sub-score numbers, paired tests and per-head quality in
[checkpoint_evaluation.md](checkpoint_evaluation.md), Round 6). Start rows: Round 5 table.

| Run | Epoch | val FDE | val ADE | val argmax (lenient) | val head choice | train argmax | gap | **Eval argmax** | **Eval heads** |
|---|---|---|---|---|---|---|---|---|---|
| `p3_combo` | 0 | 1.999 | 0.837 | 0.742 | 0.789 | 0.914 | 0.172 | | |
| | 2 | 1.934 | 0.812 | 0.761 | 0.799 | 0.917 | 0.156 | 0.732 | 0.768 |
| | 4 | 1.896 | 0.797 | 0.764 | 0.796 | 0.927 | 0.163 | 0.737 | 0.766 |
| | 6 | 1.867 | 0.789 | 0.770 | 0.801 | 0.936 | 0.166 | **0.739** | **0.769** |
| | 8 | 1.861 | 0.788 | 0.768 | 0.799 | 0.941 | 0.173 | 0.738 | **0.769** |
| | 9 | 1.859 | 0.787 | 0.768 | 0.799 | 0.942 | 0.174 | – | – |
| **`p3_combo_balanced`** | 0 | 1.945 | 0.817 | 0.752 | 0.795 | 0.915 | 0.163 | | |
| | 2 | 1.944 | 0.818 | 0.765 | 0.808 | 0.920 | 0.155 | 0.737 | 0.778 |
| | 4 | 1.882 | 0.790 | 0.772 | 0.806 | 0.930 | 0.158 | 0.742 | 0.776 |
| | 6 | 1.845 | 0.782 | 0.777 | 0.816 | 0.937 | 0.160 | 0.746 | 0.785 |
| | 7 (best, monitor) | 1.832 | 0.775 | 0.779 | 0.815 | 0.939 | 0.160 | **0.750** | 0.785 |
| | 8 | 1.831 | 0.775 | 0.777 | 0.817 | 0.941 | 0.164 | **0.750** | **0.786** |
| | 9 | 1.829 | 0.776 | 0.778 | 0.817 | 0.943 | 0.164 | – | – |
| `p3_balanced` | 0 | 1.917 | 0.799 | 0.730 | 0.789 | 0.873 | 0.144 | | |
| | 2 (best, monitor) | 1.878 | 0.794 | 0.758 | 0.812 | 0.882 | 0.124 | **0.730** | 0.781 |
| | 4 | 1.832 | 0.773 | 0.758 | 0.812 | 0.890 | 0.132 | 0.729 | 0.781 |
| | 5 (last) | 1.820 | 0.769 | 0.754 | 0.818 | 0.891 | 0.137 | 0.727 | **0.788** |
| `p3_heads` | 0 | 2.010 | 0.850 | 0.749 | 0.810 | 0.875 | 0.126 | | |
| | 2 | 2.275 | 0.995 | 0.777 | 0.845 | 0.885 | 0.108 | **0.749** | 0.814 |
| | 4 | 2.425 | 1.062 | 0.769 | 0.847 | 0.889 | 0.120 | 0.740 | 0.815 |
| | 5 (best, monitor) | 2.540 | 1.117 | 0.772 | 0.850 | 0.890 | 0.117 | 0.743 | **0.818** |
| | 8 (last) | 2.794 | 1.238 | 0.760 | 0.845 | 0.888 | 0.128 | 0.730 | 0.814 |

Observations:

1. **Weight 5 + score-aware labels (`p3_combo_balanced`) is the best policy so far:** argmax **0.750** at epoch 7–8
   (+0.015 ± 0.002 over the previous best p3_scorelabels, +0.011 ± 0.002 over p3_combo at weight 1; paired), heads
   0.786, FDE 1.83, highest expected score (0.707). The gain is drivable area (0.861 → 0.871). Still rising slowly at
   epoch 9 in W&B; no overfitting (gap flat 0.155–0.164).
2. **Weight 5 helps the heads in both starts:** +0.016 (p3_combo → p3_combo_balanced) and +0.023 (p3_total →
   p3_balanced, 0.765 → 0.788); every head's BCE and top-64 MAE improve, most the drivable-area head.
3. **Without score-aware labels the argmax gains only early:** p3_balanced 0.730 at epoch 2 (+0.015 over p3_total
   at the same epoch), then 0.727, while its heads and FDE keep improving. The argmax monitor stopped it after
   epoch 5 (lenient peak at epoch 2, patience 3), so its heads did not reach their end point.
4. **At weight 1, combining the two flags adds little in phase 3** (p3_combo vs best single-flag runs: argmax
   +0.005 ± 0.002, heads +0.004). The weighting is what moved this round.
5. **Score losses only (`p3_heads`) give the best selector: 0.818** (strict; W&B shows 0.850 lenient), +0.031 over
   weight 5 and **+0.068 over the best argmax model**. Its drivable-area head is nearly calibrated (bias −0.01) and its
   choice drives faster as training goes on (EP 0.76 → 0.82, DAC ~0.97). Flat from epoch 2 (0.814–0.818); val BCE
   bottoms at epoch 3 and the train head score keeps rising (0.896 → 0.956), so later epochs only fit train.
6. **…but it gives up imitation:** FDE 1.96 → 2.79 by epoch 8. The policy head gets no gradient, yet its argmax first
   rises (0.698 → 0.749 at epoch 2, DAC 0.877) because the LoRA features become score-aware, then drifts down (0.730).
   The imitation loss and the heads compete for the shared LoRA features: heads 0.765/0.769 (weight 1) → 0.786/0.788
   (weight 5) → 0.818 (no imitation).
7. Combination rule (best λ = 30) and safety filter stay below head_all in every run (0.003–0.012 and 0.013–0.028).

### Phase 3 decision before RL (proposal, 2026-11-04; not decided)

- **Policy checkpoint for RL: `p3_combo_balanced` epoch 8** (`exp/p3_combo_balanced/2026.11.03.23.40.54/checkpoints/epochs/epoch-epoch=08.ckpt`;
  epoch 7 = `best/` is equal). Best argmax (0.750), best expected score (0.707, the quantity an expected-reward stage
  optimizes), best FDE of the score-label runs, calibrated heads (0.786). Keep a CE / KL anchor in RL (Findings 10).
- **Best decision rule found: score-head selection** (`p3_heads` epoch 5 = `exp/p3_heads/2026.11.04.00.53.01/checkpoints/best/best-epoch=05.ckpt`, 0.818). It is a separate model (own LoRA);
  using it needs the selection flag in `compute_trajectory` (proposed earlier, not applied).
- Worth one more phase 3 run before RL: **`p3_combo` with score weights 20** (1 / 1 / 20 / 20, from `p1e_combo` ep 10,
  score-aware labels), to see whether one model can keep the argmax ≥ 0.750 and FDE ~1.83 while its heads move from
  0.786 towards 0.818. Monitor `head_score_gap` or patience ≥ 5 (the argmax monitor stopped p3_balanced while its heads
  were still improving). ~11 h on 4 GPUs. If it does not get the heads near 0.80, the two-model option stays
  (policy from p3_combo_balanced, selection from p3_heads), at twice the inference cost.
- Not worth continuing: plain labels (`p3_balanced`: heads like p3_combo_balanced, argmax 0.020 lower), weight 1 runs,
  `p3_heads` beyond epoch 5.
- **navtest PDM score (2026-11-04):** new agent flag `trajectory_selection` (`argmax` default | `score_heads`, used only in
  `compute_trajectory`; test `tests/test_vision_llm_lora_trajectory_selection.py`) and script
  `scripts/evaluation/run_vision_llm_lora_pdm_score_slurm.sh` (`run_pdm_score_one_stage.py`, navtest, `traffic_agents=
  non_reactive` = log replay as in v1, human-penalty filter on as in the official config (`HUMAN_FILTER=false` for raw
  sub-scores; the total is the same either way); mounts
  `images/test_sensor_blobs.sqfs` per job). It prints the one-stage EPDMS and a v1-style PDMS recomputed from the
  sub-scores (NC·DAC·DDC·(5 EP + 5 TTC + 2 HC)/12; approximation: v2 history comfort instead of v1 comfort, and whether
  v1 multiplies by DDC is UNVERIFIED). Job 14871654: `p3_heads` epoch 5 with `score_heads` → `exp/pdm_navtest/` (first submitted as 14871628 with the filter off, cancelled while pending).
- **navtest result, job 14871654 (finished 2026-11-05 01:07, 2 h 24 min, 12146 scenes, 0 failed;
  `exp/pdm_navtest/p3_heads_best_ep05_score_heads_non_reactive/2026.11.04.22.46.04/`): EPDMS 0.7713** (the script's
  score; strict, the filter does not touch it). Sub-score columns (lenient, human-penalty filter on): NC 0.969,
  DAC 0.978, DDC 0.994, TLC 0.998, EP 0.843, TTC 0.959, LK 0.932, HC 0.977, two-frame extended comfort (EC) 0.215
  (NaN in 2106 scenes without an adjacent frame, which drop EC from the weights). v1-style PDMS from these columns
  0.873 (× DDC) / 0.878 (without DDC); lenient, so an upper bound: EPDMS recomputed from the same columns is 0.805
  vs the strict 0.771, a 0.034 filter gap. Raw columns need a rerun with `HUMAN_FILTER=false` (same score).
  - **EC is the largest single loss:** EC = 0 in 64.9% of the scored scenes. EC compares the trajectory at a frame
    with the one 0.5 s earlier. Likely cause (not checked): `score_heads` picks the best predicted total independently per frame,
    and near-tied entries flip between frames. Recomputed from the columns with EC = 1 everywhere the EPDMS would be
    0.881 instead of 0.805, i.e. up to ~0.07 of the score. EC is not in the codebook-score targets (computed without
    extended comfort), so neither the heads nor the proxy see it. Candidate fixes (not implemented): select by the
    predicted total plus λ · log p_policy, or among the policy's top-k entries; compare with the argmax EC of
    p3_combo_balanced ep 8 (navtest run not done yet).
- **Evaluations submitted 2026-11-05 (~11:00):** proxy checkpoint eval of p3_aux (epochs 2, 4, 5, 6, 8; job 14885931)
  and p3_heads_aux (0, 1, 2, 4; 14885932); navtest one-stage non-reactive: p3_combo_balanced epoch 8 argmax (14885933),
  p3_aux epoch 5 score_heads (14885934); navhard two-stage (reactive IDM in both stages, `run_pdm_score.py`): p3_heads
  epoch 5 score_heads (14885935, `exp/pdm_navhard_two_stage/`). The PDM score script now takes `SPLIT=navtest|
  navhard_two_stage`; for navhard it mounts the test sensor archive for the real frames and reads the synthetic scenes and
  images from `$OPENSCENE_DATA_ROOT/navhard_two_stage`. (A first submission, 14885925–28, failed in 2 s: the Python env
  was not active at `sbatch` time; activate `envs/navsim_vlm` before submitting.)
- **Where the navtest score goes (p3_heads ep 5, job 14871654; gap to 1 = 0.229):** scenes scored 0 (7.7%; of
  these, reported NC < 1 in 40%, DAC < 1 in 29%, and 2.3% of all scenes are 0 with every reported multiplier at 1, i.e. a
  failure the human-penalty filter hides in the columns) cost 0.077; EC 0.076; EP 0.047 (median 0.92, 10th percentile
  0.53 in non-zero scenes: conservative); LK 0.008, TTC 0.003, HC 0.003; the rest (~0.015) is partial multipliers
  (NC 0.5) and filter effects. Approximate score without EC (EC rows reweighted to /14 as the proxy does; uses the
  lenient multipliers for EC = 1 rows): **0.837**, so the gap to the val proxy (0.818) is EC alone; without EC navtest
  is even a little higher than navtrain val.
- **p3_combo_balanced_aux (job 14885950, submitted 2026-11-05 ~12:30, `-p ai,gpu`):** p3_combo_balanced exactly (from
  p1e_combo ep 10, score-aware labels, CE 1 / ED 1 / score 5 / total 5, monitor argmax_score_gap, patience 3, 10 epochs)
  plus aux 3 / 3 / 1. Tests whether the aux losses help the imitation policy (argmax, FDE), which p3_aux / p3_heads_aux
  cannot show (CE and ED 0 there, so the policy head gets no gradient).
- **Reactive vs non-reactive codebook scores (job 14874518, 2026-11-05; `scripts/evaluation/compare_reactive_scores.py`,
  `exp/reactive_vs_nonreactive_navtest/`):** 256 navtest scenes × the 32 codebook entries closest to the human
  trajectory, each scored with log replay and with reactive IDM traffic (IDM rollout 1.14 s per entry).

  | metric | non-reactive | reactive | entries changed | scenes with a change |
  |---|---|---|---|---|
  | NC | 0.986 | 0.882 | 11.5% | 21.1% |
  | EP | 0.862 | 0.861 | 9.2% | 11.7% |
  | TTC | 0.982 | 0.878 | 11.2% | 20.3% |
  | DAC, DDC, TLC, LK, HC | – | identical | 0% | 0% |
  | score | 0.746 | 0.653 | 15.4% | 25.4% |

  Closest entry to the human 0.850 → 0.738, 5th 0.804 → 0.695, 20th 0.728 → 0.646; 255 entries higher, 1004 lower.
  The best entry (by score) differs in 11.3% of scenes; the non-reactive best scores 0.798 under reactive traffic vs
  0.829 for the reactive best. Example of the EP change: in scene 534431abbbd75d4f the PDM-Closed reference collides
  with a log-replay car (so EP is normalised to the entry itself, 1.0) but not with the reacting IDM car (EP 0.59).
  Why near-human entries collide so often under IDM traffic was not inspected (UNKNOWN; e.g. IDM agents ahead
  braking more than the log, the open-loop ego not reacting back).
  - **What the reactive drop is (2026-11-05 analysis of the shards):** score changes per entry: 84.6% none; 6.5% from a
    pass to 0, 3.3% −0.5…−0.99, 2.3% −0.1…−0.5; only 0.5% go up by > 0.1. NC 1 → 0 in 892 entries, 0 → 1 in 43. The
    ranking of the 32 entries within a scene is unchanged (median Spearman 1.0); 23 of 256 scenes (9%) lose all 32
    near-human entries. Inspected 4 of those 23 (0854af02, 2f23d31a, 620f2720, 053d94cd; script
    scratchpad `diag_collide.py`, not in the repo): all are `ACTIVE_FRONT_COLLISION`s of an entry 0.01–0.11 m ADE from
    the human with an IDM vehicle ahead that stops where the logged car drove on (0854af02: the logged car keeps 6 m/s,
    the IDM car brakes to a stop ~9 m further; 053d94cd: a car 9.5 m ahead, turning in the log, stands still under IDM
    from 0.5 s on). The open-loop ego plan cannot react, so it rear-ends it. These look like IDM behaviour rather than
    unsafe plans (4 scenes inspected; not checked for all 892 entries).
  - **Bug found and fixed first (2026-11-05):** the first run (14873852) reused one metric cache per scene for all
    candidates and both modes. Scoring with reactive tracks leaves state in the metric cache's observation, so every
    later scoring call on it differs, log-replay calls included (bisected on one scene: replay 0.0, reactive 1.0,
    replay 1.0; the IDM rollout alone changes nothing; the exact attribute was not located). All 41 changes of that run
    were in candidate 0. Fix: `copy.deepcopy(mc)` per scoring call; invalid shards in
    `exp/reactive_vs_nonreactive_navtest/invalid_shared_metric_cache/`. Repeated log-replay calls on one cache are
    identical, so the offline codebook scores (training targets, proxy evaluation) and the non-reactive navtest run are
    unaffected. [LIKELY, upstream code] the official reactive two-stage evaluation scores the human trajectory after the
    model's on the same metric cache for the human-penalty filter, so its reported sub-score columns may be affected
    (the total is not, it ignores the filter).
  - For RL: the non-reactive scores are the official one-stage metric and stay the RL-A reward. Reactive traffic
    changes the reward of ~15% of the near-human entries and the best entry in 11% of scenes, so an RL-B variant with
    reactive rewards optimises a measurably different target; at 1.14 s per entry, 8 entries per navtrain scene cost
    ~230 CPU-hours per round.

## Auxiliary perception losses (implemented 2026-11-04, trained and evaluated 2026-11-05)

Training-only BEV decoder (`navsim/agents/vision_llm_lora/vision_llm_lora_aux.py`, ~3.3 M parameters) on the LLM's
last-layer hidden states at the 208 image tokens (front strip, rear strip, front zoom; they come before the ego and
command text, so they carry no ego information): LayerNorm + Linear 2560 → 256 plus a learned position embedding per
image token; 20 × 16 learned BEV queries (one per 4 × 4 m cell) through 2 transformer decoder layers (self-attention,
cross-attention to the image tokens, MLP); 3 × (bilinear ×2, conv, GroupNorm, GELU) to 160 × 128 = the cached target
grid (0.5 m, x −16…64 m, y −32…32 m); one 1×1 conv per target. Gradient reaches the LLM and vision LoRA only through
the image-token states.

| Flag (default 0 = off) | Target (`aux_targets_navtrain.sqfs`) | Loss | Metric it serves |
|---|---|---|---|
| `aux_bev_semantic_loss_weight` | 7 classes: none, road, walkway, centerline, static objects, vehicles, pedestrians | CE over all cells | layout, NC/TTC (objects) |
| `aux_drivable_area_loss_weight` | PDM's drivable surface (roadblocks, intersections, car parks, connector lanes) | BCE | DAC |
| `aux_lane_heading_loss_weight` | travel direction of lanes, 36 bins | CE on lane cells only | DDC, lane keeping (which lane is ours) |

Other new fields: `aux_target_cache_path` (set by `AUX_TARGETS=1` in the training SLURM script, which mounts the
image per job), `aux_decoder_{dim,layers,heads}`, `aux_query_grid`, `aux_upsample_steps`, `aux_lr_multiplier` (own
optimizer group, default 1.0 × lr_base, the decoder starts from scratch). `CacheOnlyDataset` (shared code) now reads a
target builder's files from its `cache_root` if it sets one; no other builder sets it. Logged: `loss_terms/aux_*`
(unweighted), `aux/bev_miou`, `aux/drivable_iou`, `aux/lane_heading_acc`. Tests: `tests/test_vision_llm_lora_aux.py`
(+ optimizer test); full suite 98 passed.

Loss balance probe (job 14873514; first attempt 14872616 failed on a probe-script bug, `scripts/training/run_vision_llm_lora_loss_probe_slurm.sh`, output
`exp/loss_probe/`): p3_heads model, start checkpoint `p1d_total` ep 10 and trained `p3_heads` ep 5; decoder warm-up
600 batches with the rest frozen, then per-term loss and gradient norm on the LLM LoRA, vision LoRA and planner
queries over 40 batches (train scenes, batch 4). Unweighted loss / gradient norm on LLM LoRA / vision LoRA:

| Term | start (`p1d_total` ep 10, LoRA untrained) | trained (`p3_heads` ep 5) |
|---|---|---|
| soft CE | 4.68 / 44.7 / 61.7 | 6.80 / 90.5 / 73.7 (not trained in p3_heads) |
| expected distance | 0.69 / 12.4 / 15.6 | 1.12 / 21.8 / 18.3 |
| sub-score BCE | 0.235 / 2.30 / 3.76 | 0.145 / 0.64 / 0.63 |
| total-score BCE | 0.285 / 3.19 / 4.67 | 0.201 / 0.76 / 0.82 |
| aux BEV semantic | 0.98 / 0.93 / 4.85 | 0.99 / 1.12 / 2.85 |
| aux drivable area | 0.51 / 0.83 / 4.48 | 0.51 / 1.00 / 2.53 |
| aux lane heading | 2.48 / 2.40 / 13.9 | 2.41 / 4.12 / 10.1 |

Planner queries get exactly 0 from the aux terms (they reach the planner only through the image-token features).
Decoder warm-up on frozen features (600 batches): BEV CE 1.95 → 1.03 (mIoU 0.07), drivable BCE 0.69 → 0.52 (IoU 0.55),
heading CE 3.58 → 2.47 (55% within ±10°), the same on both checkpoints. Soft CE pushes the LoRA ~20× harder than
one score term (round 6 estimated ~10×). Aux terms act mostly on the vision LoRA (~5× their LLM-LoRA push).
Weights **3 / 3 / 1** (BEV / drivable / heading) balance the aux total to the two score terms at weight 5 on the vision
LoRA at the start (aux ≈ 42 vs scores ≈ 42; on the LLM LoRA aux ≈ 8 vs 27). Once the score losses are fit (p3_heads
ep 5) their gradients are ~4× smaller, so the same weights give aux ~3.6× (vision) / ~1.5× (LLM) the score push.

| Run | Job | Start | Losses | Epochs | Baseline |
|---|---|---|---|---|---|
| `p3_aux` | 14873887 | `p1d_total` ep 10 | CE/ED 0, score 5/5, aux 3/3/1 | 10, patience 3, head gap | p3_heads (identical except aux) |
| `p3_heads_aux` | 14873888 | `p3_heads` ep 5 | same | 5, patience 3, head gap | p3_heads ep 5–8 (more training without aux; LR schedule restarts here) |

Question: joint (perception and scores from the start) vs staged (aux added to a converged score model).

| Run | Job | Start | Losses | Epochs | Baseline |
|---|---|---|---|---|---|
| `p3_combo_balanced_aux` | 14885950 | `p1e_combo` ep 10 | p3_combo_balanced (score-aware labels, CE 1 / ED 1 / score 5 / total 5) + aux 3/3/1 | 10, patience 3, argmax gap | p3_combo_balanced (identical except aux) |

| `p3_aux_policy` | 14905303 (2026-11-06/07) | `p3_aux` ep 5 | **policy head only** (`train_policy_head_only=true`: `scene_proj`, `trajectory_bias`, `logit_scale`; heads, LoRA, planner queries, trajectory embedding frozen), CE 1 / ED 1 with score-aware labels, `traj_lr_multiplier=1.0` (5e-4), no aux | 4, patience 2, argmax gap, batch 8 × accum 4 | p3_aux (same heads); goal: a policy for the shortlist selection with p3_aux's heads unchanged |

| `p3_heads_policy` | 14906148 (2026-11-07) | `p3_heads` ep 5 | as p3_aux_policy | 4, patience 2, argmax gap | p3_aux_policy (aux ablation with the shortlist) |
| `p3_combo_balanced_aux_policy` | 14906393 (2026-11-07) | `p3_combo_balanced_aux` ep 9 | as p3_aux_policy (its own losses: score-aware labels, CE 1 / ED 1) | 4, patience 2, argmax gap | does extra policy-only training help a policy that was already trained jointly with its LoRA? (expected: little) |

p3_aux_policy started cleanly (job 14905303): 541 K trainable parameters (`scene_proj` 1024 → 512 = 524,800, plus the 16,384
trajectory biases and the logit scale), all LoRA / head / planner keys loaded, the 62 aux-decoder keys unused (no
decoder in this run), 0.92 it/s at batch 8 per GPU (phase 3 with LoRA training: 1.38 it/s at batch 4), ~48 min per
training epoch.

### Heads-only runs from the stronger phase 1 start (submitted 2026-11-07)

p3_heads and p3_aux started from `p1d_total` ep 10 for comparability (round 6 changed one thing at a time along the
plain-label chain p3_total → p3_balanced → p3_heads; p3_aux = p3_heads + aux), not because it was the better start.
`p1e_combo` ep 10 differs only by score-aware labels in phase 1 and is better on both sides (proxy policy 0.722 vs
0.698, heads 0.762 vs 0.755); its policy matters now that the shortlist selection uses it.

| Run | Job | Start | Losses | Twin (identical except the start) |
|---|---|---|---|---|
| `p3_aux_p1e` | 14906426 | `p1e_combo` ep 10 | CE/ED 0, score 5/5, aux 3/3/1 (`AUX_TARGETS=1`) | p3_aux |
| `p3_heads_p1e` | 14906427 | `p1e_combo` ep 10 | CE/ED 0, score 5/5 | p3_heads |

Both: 10 epochs, patience 3, monitor head gap, batch 4 × accum 8, `-p ai,gpu`.

### Results of the policy-only and p1e_combo runs (all finished 2026-11-07; W&B val, lenient)

| Run | Job, time | Epochs (best) | val heads score by epoch | val argmax by epoch | Start value |
|---|---|---|---|---|---|
| p3_aux_policy | 14905303, 3.5 h | 4 (ep 1) | 0.8585 every epoch (heads frozen) | 0.810 **0.820** 0.818 0.820 | p3_aux ep 5: argmax 0.775 |
| p3_heads_policy | 14906148, 3.5 h | 4 (ep 2) | 0.8497 every epoch | 0.798 0.806 **0.807** 0.806 | p3_heads ep 5: 0.772 |
| p3_combo_balanced_aux_policy | 14906393, 3.5 h | 4 (ep 3) | 0.8278 every epoch | 0.793 0.796 0.795 **0.797** | p3_combo_balanced_aux ep 9: 0.797 |
| p3_policy | 14906154, 10.7 h | 10 (ep 7) | – (no heads) | 0.717 0.725 0.727 0.735 0.735 0.738 0.734 **0.741** 0.738 0.738 | (p3_combo_balanced: 0.778 at ep 9) |
| p3_aux_p1e | 14906426, 12.0 h | 10 (ep 7) | 0.823 0.839 0.847 0.850 0.852 0.853 0.850 **0.853** 0.850 0.850 | 0.778 … 0.801 (ep 3) … 0.784 | (p3_aux: heads 0.859 at ep 5) |
| p3_heads_p1e | 14906427, 9.0 h | 8, early stop (ep 4) | 0.810 0.831 0.838 0.841 **0.844** 0.841 0.841 0.842 | 0.769 … 0.788 (ep 3) … 0.783 | (p3_heads: heads 0.850 at ep 5) |

p3_aux_p1e's aux metrics at the end match p3_aux (BEV mIoU 0.246, drivable IoU 0.841, heading acc 0.862).

**Evaluation pitfall (found 2026-11-07):** checkpoints keep only trained parameters, so the policy-only runs' checkpoints
hold just the policy head (5 keys, 8 MB). The first offline evaluation (jobs 14911310–12) loaded only those and left the
LoRA, heads and planner queries uninitialised (heads 0.13, FDE 15.8 m; moved to
`exp/checkpoint_eval/invalid_policy_only_partial_load/`). Fixed: the offline evaluation now loads the run's start
checkpoint first; full checkpoints for the official evaluation via `scripts/evaluation/merge_policy_only_checkpoint.py`
(`checkpoints/merged/epoch-epoch=NN.ckpt`; written for p3_aux_policy ep 1, 3 and p3_heads_policy ep 2). Re-run:
jobs 14911770–72. Training itself was unaffected (it starts from the full start checkpoint).

**Strict offline results (checkpoint_evaluation.md, Round 8):** p3_aux_policy ep 3 argmax **0.790** (best policy so far;
+0.041 over p3_aux's stale policy, +0.022 over p3_combo_balanced_aux), FDE 1.94, heads unchanged 0.826, **shortlist
k = 16 / 64: 0.830** (best selection so far, +0.0035 over its heads, +0.026 over p3_combo_balanced_aux's shortlist).
p3_heads_policy ep 2: argmax 0.777, shortlist 0.822. p3_policy ep 7: argmax 0.711 (−0.039 vs p3_combo_balanced).
p3_aux_p1e / p3_heads_p1e: heads −0.006 vs p3_aux / p3_heads. Full checkpoints for the official runs:
`p3_aux_policy/2026.11.06.23.42.11/checkpoints/merged/epoch-epoch=0{1,3}.ckpt`,
`p3_heads_policy/2026.11.07.00.04.51/checkpoints/merged/epoch-epoch=02.ckpt`.

Observations (lenient; strict offline evaluation of the best checkpoints: jobs 14911310–15):
1. **Decoupled policy training works:** training only the policy head (541 K parameters, frozen features) moves
   p3_aux's stale policy from 0.775 to 0.820 in one epoch, above p3_combo_balanced_aux's jointly trained policy
   (0.797), with p3_aux's heads unchanged. p3_heads: 0.772 → 0.807. Aux helps here too (+0.013).
2. A policy already trained jointly with its LoRA gains nothing (p3_combo_balanced_aux_policy: 0.797 → 0.797).
3. **The phase 3 imitation-only baseline is clearly weaker:** p3_policy 0.741 vs p3_combo_balanced 0.778 (same
   recipe plus score heads and score-aware labels).
4. **The stronger phase 1 start does not help the heads:** p3_aux_p1e 0.853 vs p3_aux 0.859, p3_heads_p1e 0.844 vs
   p3_heads 0.850; the p1d_total start (chosen for comparability) turned out to be the better one.

**Status at the end of 2026-11-07 / next steps.** Running or queued: p3_aux_policy (14905303), p3_heads_policy
(14906148), p3_policy (14906154), p3_combo_balanced_aux_policy (14906393), p3_aux_p1e (14906426), p3_heads_p1e
(14906427). When each finishes: offline evaluation (`run_vision_llm_lora_checkpoint_eval_slurm.sh`; the selection
columns comb/filter/shortlist come with it), then official v1 / v2 / navhard with `SELECTION=shortlist[_v1]`
(`selection_shortlist_size` 64 and 256) for runs with heads, argmax for p3_policy (`WORKERS=3`). For
p3_combo_balanced_aux_policy: if its first epoch's argmax falls below the start (proxy 0.766), rerun with the policy
LR it was trained with (`traj_lr_multiplier=0.1`). **RL fine-tuning starts next** (plan: RL-A, exact expected reward
on the precomputed log-replay codebook scores + KL to the start policy, then a GRPO variant; start model
p3_combo_balanced_aux ep 9 unless the runs above give a better policy+heads model; evaluate with the shortlist).

### Phase 3 policy-only baseline

| Run | Job | Start | Losses | Epochs | Purpose |
|---|---|---|---|---|---|
| `p3_policy` | 14906154 (2026-11-07) | `p1c_ed05` ep 10 (best phase 1 run without score heads: proxy argmax 0.695 vs p1c_base 0.689) | CE 1 / ED 1, plain labels, no score heads, no aux; `log_pdm_score_metrics=true` (loads the cached scores for the proxy metrics only; new flag, loss unchanged) | 10, patience 3, argmax gap, batch 4 × accum 8 | the phase 3 run without score heads: what the heads add on top of imitation |

### Results (all three finished; p3_aux 2026-11-05 10:24, p3_heads_aux 05:46, p3_combo_balanced_aux 23:56)

- `p3_aux` early-stopped after epoch 8 (best ep 5), `p3_heads_aux` ran its 5 epochs (best ep 1), `p3_combo_balanced_aux`
  ran all 10 (W&B checkpoint best ep 8 by the argmax gap). Output `exp/<run>/2026.11.05.*/` (the folder with
  `checkpoints/`; the twin folder one second earlier holds only a second rank's Hydra output).
- W&B val per epoch (lenient), heads / argmax / aux metrics at the last epoch:

| Run | head score by epoch | argmax by epoch | BEV mIoU | drivable IoU | heading acc |
|---|---|---|---|---|---|
| p3_heads (no aux) | 0.810 0.834 0.845 0.845 0.847 **0.850** 0.849 0.849 0.845 | 0.749 … 0.777 (ep 2) … 0.760 | – | – | – |
| p3_aux | 0.818 0.840 0.851 0.854 0.855 **0.859** 0.853 0.854 0.854 | 0.761 0.772 **0.789** 0.787 … 0.762 | 0.242 | 0.839 | 0.860 |
| p3_heads_aux | 0.853 **0.856** 0.855 0.850 0.854 | 0.763 0.763 0.767 0.758 0.757 | 0.199 | 0.814 | 0.830 |
| p3_combo_balanced (no aux) | 0.795 0.803 0.808 0.809 0.806 0.812 0.816 0.815 0.817 0.817 | 0.752 0.762 0.765 0.770 0.772 0.774 0.777 0.779 0.777 0.778 | – | – | – |
| p3_combo_balanced_aux | 0.796 0.808 0.815 0.818 0.818 0.824 0.827 0.828 0.827 **0.828** | 0.752 0.772 0.781 0.785 0.785 0.790 0.795 0.793 **0.797** 0.797 | 0.205 | 0.807 | 0.829 |

- Offline proxy (strict, val, paired over 18,179 scenes; `exp/checkpoint_eval/`): p3_aux ep 5 heads **0.826** vs
  p3_heads ep 5 0.818, paired **+0.0082 ± 0.0014**; argmax +0.0042 ± 0.0023 (untrained policy head, CE 0). p3_heads_aux
  best ep 2 heads 0.8235, +0.0053 ± 0.0014 vs p3_heads ep 5 (ep 1 +0.0047); its argmax drops (−0.006 to −0.010).
  p3_combo_balanced_aux (jobs 14891502–06): best ep 9, argmax **0.766** (+0.0158 ± 0.0017 vs p3_combo_balanced ep 8),
  heads 0.797 (+0.0111 ± 0.0013), FDE 1.79 (−0.034); checkpoint_evaluation.md, Round 7. Official runs of ep 9 (heads):
  navtest v1 `score_heads_v1` 14891553, navtest v2 `score_heads` 14891554, navhard 14891555.
  Official results (fixed filter): heads v1 0.863 / v2 0.786 / navhard 0.301; argmax v1 0.842 / **v2 0.826** /
  navhard 0.210 (+0.010 / +0.011 / +0.016 over p3_combo_balanced argmax). All official numbers: final_evaluation.md.
- navtest (official, one stage): p3_aux ep 5 score_heads EPDMS 0.779 pre-fix / 0.812 post-fix vs p3_heads 0.771 /
  0.804 ([final_evaluation.md](final_evaluation.md)).
- **Conclusions.** (1) Aux losses help the score heads: +0.008 on val (≈ 6 SE) and on navtest. (2) Joint training
  (p3_aux, from phase 1) beats staged (p3_heads_aux, aux added to a converged p3_heads): +0.008 vs +0.005 with fewer
  total epochs, so the extra stage is not worth it. (3) With imitation on, aux also helps the policy: p3_combo_balanced_aux
  is ahead of p3_combo_balanced at every epoch, argmax +0.019 and heads +0.011 at the last epoch on W&B (lenient; the
  strict paired numbers come from the offline evaluation). Its heads stay below p3_aux's (0.828 vs 0.859 on W&B): the
  imitation terms still compete with the heads, as in round 6.

## RL fine-tuning (started 2026-11-07, stage 1 on the policy head done 2026-11-08)

Theory (objective, optimal policy, KL bounds, exact vs GRPO gradients, capacity limit): [theoretical.md](theoretical.md).
Per-run tables and findings: [checkpoint_evaluation.md](checkpoint_evaluation.md) ("Shortlist headroom for RL", "RL on
the policy head", findings 38–48).

Plan: four runs. RL-1 / RL-2 = policy head only on p3_aux_policy ep 3 / p3_heads_policy ep 2 (heads, LoRA frozen; done
as a sweep); RL-3 / RL-4 = the same with the LoRA trainable and a frozen copy of the scorer adapter for the heads'
shortlist (not implemented yet). Reward = the offline proxy's codebook score of every entry (log replay, no EC, no
human filter); candidate set = the full codebook (the heads' train-time shortlist is too easy: headroom +0.037 on
train vs +0.098 on val).

| Step | Jobs | Output (under `exp/`) |
|---|---|---|
| Shortlist headroom (offline eval with oracle columns) | 14920952 | `checkpoint_eval/headroom/` |
| Fused-state cache (3 shards per model, ~30 min each) | 14921155–57 (p3_aux_policy ep 3), 14921158–60 (p3_heads_policy ep 2) | `rl_features/<model>/` |
| β sweep, both models (exact β 0.02–0.2, GRPO β 0.05) | 14924451 / 14924452 (first attempt 14921216/17 OOM in the reward loader, fixed) | `rl_policy/<model>/<run>/` |
| Extended sweep, p3_aux_policy (exact β 0–0.02, lr, epochs; GRPO / Dr. GRPO) | 14924652 / 14924653 | same |

Code: `navsim/agents/vision_llm_lora/vision_llm_lora_rl.py` (PolicyHead, `expected_reward_kl_loss`, `grpo_loss`,
`selection_metrics`; tests `tests/test_vision_llm_lora_rl.py`), `scripts/training/cache_policy_head_features.py`,
`scripts/training/rl_policy_head.py` (`--export-epoch N` writes a full checkpoint for the official evaluation),
`scripts/training/run_rl_policy_head_slurm.sh`. The agent's training loss is unchanged.

Results (val proxy, p3_aux_policy): the policy improves a lot (argmax 0.789 → 0.83), the shortlist a little
(k = 256 0.827 → 0.835–0.837), and all exact settings plateau there: the frozen fused state limits what a policy-head
run can learn inside the heads' shortlist (oracle 0.926). Dr. GRPO matches the exact objective; standard GRPO
(std-normalized advantages) overfits and gets worse on val in every setting. p3_heads_policy behaves the same
(argmax 0.779 → 0.810, k = 256 0.817 → 0.825 at β = 0.02).

Official (shortlist k = 256; jobs 14924996–14925001; [final_evaluation.md](final_evaluation.md), "RL on the policy
head"): exact β = 0.02 ep 12 v1 **0.899** / v2 **0.872** / navhard 0.382, β = 0.005 lr 3e-4 ep 7 0.898 / 0.869 /
0.384, vs p3_aux_policy 0.893 / 0.867 / 0.382. navtest v2 +0.005 ± 0.001 (β = 0.02); EC drops 0.751 → 0.741 / 0.728
(not in the reward), so the looser β is worse on v2 although it was val-best; navhard unchanged. RL-2 (p3_heads_policy)
was only run in the first sweep (same behaviour, ~0.01 lower), no official runs.

Re-evaluation 2026-11-08 (all models, split pipeline, batch 64; [final_evaluation.md](final_evaluation.md), "Results"):
RL β = 0.02 v1 0.899 / v2 0.872 / navhard 0.383 vs p3_aux_policy 0.892 / 0.867 / 0.386; navtest paired +0.0047 ± 0.0011
(v2), +0.0073 ± 0.0012 (v1). navtest numbers within ±0.002 of the combined runs, navhard within ±0.012.

Next: RL-3 (LoRA trainable, frozen scorer adapter) on p3_aux_policy with β = 0.02.

## Round 4 (finished 2026-11-02 ~15:00): score-head loss

Base = `p1c_scores` (trajemb 1024, 8 queries, ED 1.0, T=1, `pdm_score_loss_weight=1.0`), 20 epochs, MONITOR val FDE,
4 GPUs, batch 8 × accum 4. One change each, motivated by the head analysis in
[checkpoint_evaluation.md](checkpoint_evaluation.md) (the drivable-area head is worst on plausible candidates):

| Run | Job | Change | Eval result (val, best epoch; base `p1c_scores` 0.699 / 0.746) |
|---|---|---|---|
| `p1d_total` | 14810662 | `pdm_total_score_loss_weight=1.0`: BCE of the aggregated predicted total (product of NC·DAC·DDC·TLC × weighted mean of EP/TTC/LK/HC) vs the aggregated target; its gradient weights each head by its effect on the total | argmax 0.698 (unchanged), **heads 0.755** (best of round 4), drivable-area head calibrated (bias −0.087 → −0.01…−0.04), heads pick faster trajectories (EP 0.66 → 0.69–0.70) |
| `p1d_headw` | 14810663 | `pdm_score_head_weights=[1,3,1,0.5,1,1,2,0.5]` (NC, DAC, DDC, TLC, EP, TTC, LK, HC): DAC ×3, LK ×2, TLC/HC ×0.5 | argmax 0.705, heads 0.748: within noise of the base, bias unchanged; no effect |
| `p1d_scorelabels` | 14811648 | `score_weighted_labels=true`: soft labels exp(−d/T) × total PDM score of each entry, renormalized (plain labels where no entry scores); trains the policy itself towards safe entries near GT. Its soft CE / val loss is against different targets, so not comparable; logs `labels/score_mass` | **argmax 0.720** (+0.021, almost all drivable area 0.817 → 0.842; same FDE and progress), heads 0.749; fits train harder (gap 0.19–0.20) |

Both log `loss_terms/score_total_bce` and `pdm_scores/total_mae` (also when the weight is 0). Compare against
`p1c_scores` by the proxy score of argmax and of head selection (`head_all`), not by val loss.

## Data artifacts

| Artifact | Path (under `exp/`) | Notes |
|---|---|---|
| Training cache | `training_cache` | 103,288 scenes, `vision_llm_feature.gz` + `vision_llm_target.gz` per token (78 GB); verified, no truncated files |
| PDM score targets | `training_cache/<log>/<token>/vision_llm_pdm_scores.gz` | (8, 16384) float16 per scene, for `pdm_score_loss_weight > 0`; written by `scripts/training/cache_pdm_score_targets.py` |
| Aux perception targets | `$OPENSCENE_DATA_ROOT/images/aux_targets_navtrain.sqfs` (+ `_info.json`), i.e. `/project/nr_e2e_vlm/dataset/images/` | built 2026-11-03, 574 MB, 103,288 scenes of 1,192 logs, 0 missing, 0 errors (job array 14852058, 40 parts on `cpu`, pack job 14852059; `scripts/training/run_cache_aux_targets_slurm.sh`): one squashfs image, `<log>/<token>/vision_llm_aux_target.gz` per training-cache scene, from `VisionLLMLoRAAuxTargetBuilder` (annotations + maps, no sensors). Ego-frame grid x −16…64 m, y −32…32 m, 0.5 m (160 × 128): `bev_semantic_map` (TransFuser classes), `drivable_area_map` (PDM DAC layers), `lane_heading_map` (36 bins, 255 = none), `agent_states` (40 nearest vehicles/pedestrians/cyclists: x, y, heading, length, width, vx, vy) + `agent_classes`. ~8 KB per scene. Check on 298 scenes: ego on drivable area 100%, lane under ego straight 94%. Not read by training yet (needs a second cache root for the mounted image and an aux loss) |
| Codebook scores navtrain | `codebook_scores_navtrain` | one `.npz` per log (1,192), every codebook entry PDM-scored per scene, non-reactive (log replay), no extended comfort; 11.8 GB; 0 failed scenes; oracle (best entry) mean score 0.9825, 91.4% of scenes have a perfect entry, 71.9% of entries score 0 |
| Metric cache navtest | `metric_cache_navtest` | 12,146 scenes, 3.6 GB; `run_pdm_score.py` in this v2 devkit gives navtest EPDMS (one stage), not v1 PDMS (needs the v1 devkit; approximable from the sub-scores) |
| Metric cache navhard | `metric_cache_navhard_two_stage` | 5,912 scenes (real + synthetic), 1.3 GB, for NAVSIM v2 EPDMS |
| Best model copy | `best_models/phase1_r2_trajemb_T2_q8_ed_ep12` | checkpoint + Hydra config + README |
| RL fused-state caches | `rl_features/p3_aux_policy_ep03`, `rl_features/p3_heads_policy_ep02` | per model: `head.pt` (trajectory embeddings, policy parameters, source checkpoints) + 3 shards (token, log, split, fused state fp16 (1024), heads' top-1024 entries, model argmax) for 103,288 scenes; ~0.45 GB each; `scripts/training/cache_policy_head_features.py` |

Scripts: metric caching `scripts/evaluation/run_metric_caching_slurm.sh` (`SPLIT=navtest|navhard_two_stage`);
codebook scoring `scripts/evaluation/run_codebook_scoring_slurm.sh` (splits large logs into jobs of
`--scenes-per-job` scenes); offline checkpoint evaluation (proxy PDM score per epoch, score-head selection)
`scripts/evaluation/run_vision_llm_lora_checkpoint_eval_slurm.sh`, method and results in
[checkpoint_evaluation.md](checkpoint_evaluation.md); GPU memory probe `scripts/training/run_vision_llm_lora_batch_probe_slurm.sh`; cache
check `scripts/training/verify_training_cache.py`.

## GPU memory (A100-40GB, phase 1 unless noted)

Linear head, 4 queries: batch 2 / 4 / 8 → phase 1 18.6 / 25.5 / 39.3 GB (39.5 GB on 4 GPUs with accumulation,
stable over 600 steps), phase 2 22.1 / 30.3 / OOM, phase 3 24.6 / 35.0 / OOM. So phases 2–3 need batch 4 with
accumulation 8. The trajectory-embedding head is smaller than the linear one.

## Open items

- PDMS evaluation (navtest v1, navhard v2) of the best phase 1 checkpoint: not run yet.
- Sub-score selection at inference (combine predicted PDM sub-scores to pick the trajectory): not implemented; the
  `p1c_scores` run uses the heads only as an auxiliary loss.
- Not done yet: MBR decoding, vocabulary dropout / ensembles (GTRS ideas), ego dropout, image-input variants
  (previous frame, LiDAR BEV, higher resolution; each new image cache adds ~206k files, the project is at ~660k of
  1M), BEV / agent auxiliary losses (better in phase 2/3, when LoRA can change the features).
- Next (2026-11-04): phase 3 rounds 5–6 done; proposed: one more phase 3 run (`p3_combo` with score weights 20), the
  score-head selection flag in `compute_trajectory`, official EPDMS of the best checkpoints, then RL from
  `p3_combo_balanced` epoch 8 on the precomputed codebook scores (Round 6, "Phase 3 decision before RL").
- Disk: per-epoch checkpoints of finished runs (~18 × 8.8 GB each) can be deleted once the kept epochs are chosen.
