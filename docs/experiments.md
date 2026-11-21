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
  v1 multiplies by DDC is UNVERIFIED; since 2026-11-18 the official v1 numbers use the v1 comfort term,
  `scripts/evaluation/v1_pdms_official_comfort.py --from-run-config`, final_evaluation.md findings 51-52). Job 14871654: `p3_heads` epoch 5 with `score_heads` → `exp/pdm_navtest/` (first submitted as 14871628 with the filter off, cancelled while pending).
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

### RL-3 / RL-4: LoRA, vision LoRA, planner queries and policy head trainable (2026-11-08/09)

Config flags `rl_loss_weight`, `rl_beta`, `rl_reference_cache_path`, `rl_reference_head_path` (`vision_llm_lora_rl.py`,
`rl_loss_terms`; tests `tests/test_vision_llm_lora_rl3.py`); per-scene data from `scripts/training/cache_rl_reference.py`
(`exp/rl_reference/<model>/`). Same exact objective as RL-1; π_ref = the start model's frozen policy head on its cached
fused state; all imitation and score losses 0; phase 3, LR multipliers of lr_base 5e-4: LLM LoRA 0.2 and vision LoRA 0.1 (as p3_aux), trajectory head + planner queries 0.2 (new: p3_aux used the default 0.10, p3_aux_policy 1.0), ego tokens 0.12 (default); 4 GPUs, batch
4 × accum 8, val every half epoch, monitor `val/rl/shortlist_256_gap`. The scorer stays the start model: evaluate with
`SCORES_PRED=<start model predictions>`. Jobs and W&B ids: `exp/rl3_logs/rl3_runs_2026-11-08.txt`.
All 2-epoch jobs show FAILED / OUT_OF_MEMORY: Lightning's teardown copied every rank's model to host RAM after `fit`
(checkpoints and the last val were already written). Fixed in `run_training.py` (`KeepModelOnDeviceDDPStrategy`,
`tests/test_run_training_teardown.py`); 14929861 is the first job with it (COMPLETED).

Val proxy at the best `shortlist_256` validation (18,179 scenes; start = RL-1 epoch 0, same features and rewards):

| Start model | Run | best val | k = 256 | argmax | E_π[r] | argmax = 0 | KL | RL-1 best k = 256 at this β |
|---|---|---|---|---|---|---|---|---|
| p3_aux_policy ep 3 | start | | 0.8269 | 0.7888 | 0.7166 | 0.1315 | 0 | |
| | `rl3_b0.05` (W&B 9udwu19i) | 2 ep | 0.8426 | 0.8466 | 0.8187 | 0.0735 | 0.66 | 0.8331 |
| | `rl3_b0.02` (h3x58s9j) | 2 ep | 0.8466 | 0.8533 | 0.8345 | 0.0702 | 1.17 | 0.8357 |
| | `rl3_b0.005` (523dciyw) | 2 ep | **0.8506** | **0.8548** | **0.8455** | **0.0664** | 3.06 | 0.8366 |
| | `rl3_b0.02_ep5` (3wohi0k3) | 3 of 5 ep | 0.8467 | 0.8509 | 0.8338 | 0.0727 | 1.38 | |
| p3_heads_policy ep 2 | start | | 0.8170 | 0.7786 | 0.7077 | 0.1421 | 0 | |
| | `rl3_heads_b0.05` (yyxe9n1q) | 2 ep | 0.8327 | 0.8310 | 0.8045 | 0.0879 | 0.63 | 0.8230 |
| | `rl3_heads_b0.02` (ub82zval) | 2 ep | 0.8362 | 0.8361 | 0.8181 | 0.0854 | 1.12 | 0.8246 |
| | `rl3_heads_b0.005` (m8j8so68) | 2 ep | **0.8387** | **0.8403** | **0.8304** | **0.0832** | 2.53 | (not run) |

- RL-3 beats RL-1 at every β by +0.010 to +0.014 (k = 256), on both start models; the two models keep their ~0.01 gap.
- Lower β is better on val (0.05 < 0.02 < 0.005) on both models, unlike RL-1 (flat below 0.02). RL-1's val-best
  looser β was worse on navtest v2 officially (EC is not in the reward), so the official runs decide.
- The policy's argmax now matches or beats the k = 256 shortlist pick (start: 0.789 vs 0.827).
- 5 epochs instead of 2 (β = 0.02): no gain; k = 256 peaks at 3 ep (0.8467) and ends at 0.8453, argmax 0.8542 at 4.5 ep.

### Phase-1 scorer-only runs (2026-11-08)

`p1h_heads` (14929075, W&B w3rxso3k) and `p1h_heads_aux` (14929076, a4qcps8f, + aux 3/3/1): phase 1 from scratch,
only `pdm_score_loss_weight` 1 and `pdm_total_score_loss_weight` 1 (CE 0, ED 0), monitor `val/pdm_scores/head_score_gap`,
20 epochs with patience 5. Both early-stopped after epoch 13 (best epoch 8; FAILED status = the teardown OOM above).
Best val `head_score`: 0.8235 / 0.8249. In phase 1 the aux loss can only train the aux decoder (the image tokens
precede every trainable input in the causal LLM), so the 0.0014 difference is run-to-run noise. Phase-3 scorers for
reference: p3_heads 0.8497 (ep 5), p3_aux 0.8585 (ep 5). p1d_total has no val `head_score` in W&B (the metric was added
later); p3_heads after its first phase-3 epoch from p1d_total ep 10 was at 0.8102.

### Phase 3 from the scorer-only phase 1 (p3h, 2026-11-09/10)

Recipes of p3_heads / p3_aux, start = p1h best epoch 8 (`exp/p3h_logs/p3h_runs_2026-11-09.txt`); all three stopped
early (patience 3). Best val `head_score`:

| Run (job, W&B) | start | aux | best (epoch) | compare |
|---|---|---|---|---|
| `p3h_heads` (14933662, e67uguex) | p1h_heads | – | 0.8352 (4) | p3_heads 0.8497 |
| `p3h_aux` (14933663, 2ttxv032) | p1h_heads | 3/3/1 | 0.8498 (4) | p3_aux 0.8585 |
| `p3h_heads_aux` (14933664, iv88b7mt) | p1h_heads_aux | 3/3/1 | 0.8434 (5) | p3_aux 0.8585 |

Starting from a scorer-only phase 1 is worse by 0.009-0.015: the phase-1 head start (epoch 0: 0.824-0.837 vs
0.810-0.818) is gone after one epoch and the runs plateau lower. p1d_total (CE + scores in phase 1) stays the better
start. The two aux jobs show FAILED / OUT_OF_MEMORY after training (host memory crept to 93-94% of 30G over 9-11 h and
the teardown's optimizer move to CPU tipped it); `KeepModelOnDeviceDDPStrategy` now skips that move as well.

### Pairwise neighbour score loss (2026-11-10)

`pairwise_score_loss_weight` (vision_llm_lora_loss.py `pairwise_score_loss`; tests
`tests/test_vision_llm_lora_pairwise_score.py`). Both runs from p3_aux best epoch 5 with the p3_aux recipe for 5 more
epochs (fresh warmup + cosine); the control differs only in the pairwise weight (`exp/p3_pair_logs/`).

| Run (job) | pairwise weight | val head_score per epoch | best | val pair_acc (NC/DAC/DDC pairs near GT) |
|---|---|---|---|---|
| p3_aux (reference) | 0 | | 0.8585 (ep 5) | not measured |
| `p3_aux_ctrl` (14935958) | 0 | 0.8552 0.8592 0.8597 0.8555 0.8565 | 0.8597 (ep 2) | not logged |
| `p3_aux_pair` (14935930) | 1.0 | 0.8488 0.8536 0.8556 0.8494 0.8505 | 0.8556 (ep 2) | 0.913 → 0.925 |

The extra epochs alone add +0.0012 head_score (noise level); the pairwise loss costs −0.004 head_score against its
control while raising pair accuracy. Combined with the RL-3 β = 0.02 policy on navtrain val it is nevertheless the
best scorer for every selection rule (shortlist k = 256 +0.0012 vs argmax instead of −0.0061, mixture λ = 10 +0.0052
instead of +0.0007, local N 64 λ 0.2 positive for the first time; control in between): final_evaluation.md,
"Scorer swap on val". Official runs with it are the next step.

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

- v1 PDMS with the v1 comfort term (done 2026-11-18): all 61 own-scorer navtest v1 results recomputed (jobs
  15175655-15175716, `exp/v1_comfort_logs/`); final_evaluation.md section "v1 PDMS with the v1 comfort term for every
  model". Own-heads mixture_v1 with v1 comfort: RL-3 seed 1 0.9171, EC reward w 2 0.9118, w 4 0.9072, shortlist
  previous plan w 2 / w 4 / w 4 seed 1 0.9120 / 0.9077 / 0.9075 (done; the last four after a resubmission).
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
- Future work (2026-11-16): a 4:1 front row instead of 6:1, for more vertical view at the front. The 6:1 centre crop of
  the 4096 px wide L0 | F0 | R0 row keeps only 683 of 1080 rows (≈ 25° vertically, ±12° around the horizon; ground
  visible from ~6 m), while the rear row keeps 960 rows (≈ 36°), because L0/R0 lose 416 px per side and the row is
  narrower (findings F11). At 4:1 the front row keeps 1024 rows. Rear row unchanged (6:1, 768 × 128, 96 tokens).
  Horizontal resolution: F0 strip pixels / 72.3°.

  | Front row option | Height kept | Vertical view | Tokens | Horizontal resolution | Cost |
  |---|---|---|---|---|---|
  | 6:1 (current), 768 × 128 | 683 rows | ≈ 25° | 96 | 5.0 px/° | – |
  | 4:1, same width, 768 × 192 | 1024 rows | ≈ 38° | 144 | 5.0 px/° | +48 tokens (prompt 268 → 316) |
  | 4:1, 640 × 160 | 1024 rows | ≈ 38° | 100 | 4.1 px/° | ~same tokens, lower horizontal resolution |
  | 4:1, same height, 512 × 128 | 1024 rows | ≈ 38° | 64 | 3.3 px/° | −32 tokens, horizontal resolution −33% |

  Not a config switch: the cache stores 13 square tiles and `image_layout: strips` rebuilds each row from 6 of them, so
  a front row of another shape needs a builder change behind a new flag (default = current layout), a fresh
  `cache_path` (~206k files; pack it into squashfs) and retraining from phase 1. Vertical angles from the
  distortion-corrected calibration (fx 1545, cy 560, k1 −0.36), ±1°.

### Phase 1 with past inputs (p1hist, started 2026-11-10)

p1d_total recipe unchanged (trajemb 1024, 8 queries, ED 1.0, T 1, score 1 / total 1; monitor `stabilize/val/fde_loss`,
patience 10, 20 epochs; 4 GPUs batch 8 × accum 4), one change each; past inputs read from
`dataset/images/history_navtrain.sqfs` (`HISTORY_CACHE=1`, 16 squashfuse mounts read in turn: one mount serves
~11 past-frame files/s and held the first p1hist_prev (14949462) at 0.32 it/s; 8 mounts gave 42 files/s on the login node).
First submission 14949462-64 crashed: diagnostic `srun --overlap` steps wiped the jobs' /tmp (DataLoader sockets). 12 CPUs / 46G / 22 h, 4 workers per GPU
(`exp/p1hist_logs/p1hist_runs_2026-11-10.txt`). Baseline: `p1d_total` (14810662, best epoch 10).

| Run | Job | Change |
|---|---|---|
| `p1hist_prev` | 14949469 | `temporal_history_frame=1`: the t−0.5 s frame fills Qwen's second temporal slot (same 208 image tokens) |
| `p1hist_ego` | 14949470 | `ego_history_tokens=true`: 3 past poses (x, y, yaw; ego frame) as `<EGO_STATE>` tokens, dropout 0.5 |
| `p1hist_both` | 14949471 | both |
| `p1hist_none` | 14949472 | none: the same recipe on the current code (code drift and run-to-run noise vs p1d_total) |

Results (2026-11-11; W&B p1hist_prev unzxpa6o, p1hist_ego 42y9tq9c, p1hist_both 3y5px581, p1hist_none 70nk1gxk;
p1d_total 876ig2il). All ran 20 epochs (14.7-15.0 h); best checkpoint = best val FDE, as for p1d_total. val argmax / head
score = `val/pdm_scores/{argmax,head}_score` (not logged for p1d_total, so p1hist_none is the reference for them).

| Run | best epoch | val FDE (best) | argmax at best | head at best | mean ep 8-19: FDE / argmax / head |
|---|---|---|---|---|---|
| p1d_total (2026-11-01) | 10 | 1.957 | – | – | 1.973 / – / – |
| `p1hist_none` | 12 | 1.967 | 0.722 | 0.793 | 1.978 / 0.724 / 0.790 |
| **`p1hist_ego`** | 15 | **1.883** | **0.737** | **0.800** | **1.904 / 0.734 / 0.797** |
| `p1hist_prev` | 12 | 2.024 | 0.712 | 0.788 | 2.037 / 0.714 / 0.786 |
| `p1hist_both` | 12 | 1.994 | 0.719 | 0.790 | 2.010 / 0.722 / 0.787 |

- The ego history tokens help at every epoch from 2 on: FDE −0.074 m (−3.7%), argmax +0.010, head +0.007 (means
  over epochs 8-19 vs p1hist_none).
- The previous frame hurts in phase 1 (FDE +0.06 m, argmax −0.010) and cancels most of the ego gain in p1hist_both.
  In phase 1 the vision tower is frozen: its pretrained Conv3d patch embedding mixes the two frames into one temporal
  patch, which likely blurs the current frame where the scene moves, and only the heads can adapt (UNVERIFIED).
  Phase 3 trains the vision LoRA, so phase 1 does not settle whether the frame can help.
- p1hist_none reproduces p1d_total within noise (best FDE 1.967 vs 1.957, mean 1.978 vs 1.973): the code changes
  since 2026-11-01 do not alter the phase-1 recipe.

Phase 3 (submitted 2026-11-11): p3_aux recipe exactly (overrides of p3_aux 14873887: score 5/5, aux 3/3/1, CE/ED 0, LoRA
lang 0.2 / vision 0.1, vision LoRA r16/α32; 10 epochs, patience 3, monitor `val/pdm_scores/head_score_gap`; 4 GPUs batch
4 × accum 8; `AUX_TARGETS=1`, history runs also `HISTORY_CACHE=1`), from each run's best phase-1 checkpoint; 8 CPUs
(2 per task) / 32000M / 13 h, `gpu,ai` (first submission 14967646-49 asked for 32 CPUs and could not start). Baselines: p3_aux (from p1d_total) and p3hist_none.

| Run | Job | Start |
|---|---|---|
| `p3hist_none` | 14967704 | p1hist_none ep 12 |
| `p3hist_ego` | 14967705 | p1hist_ego ep 15 |
| `p3hist_prev` | 14967706 | p1hist_prev ep 12 |
| `p3hist_both` | 14967707 | p1hist_both ep 12 |

Results (2026-11-12; W&B none 0iqou1j4, ego cgbi9ymv, prev pimmr35e, both f7gcookv; p3_aux d8vbo5hz). All stopped by
patience 3 after 8-9 epochs (10.9-12.3 h). Best = lowest `val/pdm_scores/head_score_gap`.

| Run | best epoch | val head_score | val score BCE | vs p3hist_none |
|---|---|---|---|---|
| p3_aux (2026-11-05) | 5 | 0.8585 | 0.2392 | −0.0006 |
| `p3hist_none` | 5 | **0.8591** | 0.2394 | – |
| `p3hist_ego` | 5 | 0.8572 | **0.2389** | −0.0019 |
| `p3hist_prev` | 4 | 0.8511 | 0.2436 | −0.0080 |
| `p3hist_both` | 5 | 0.8522 | 0.2454 | −0.0069 |

- p3hist_none reproduces p3_aux (0.8591 vs 0.8585): run-to-run noise ~0.001, epoch-to-epoch ~0.002-0.003.
- The previous frame hurts the scorer also with the vision LoRA trained (−0.007 to −0.008, worse BCE at every epoch):
  under the p3_aux recipe (vision LR ×0.1, vision LoRA r16) phase 3 does not recover the phase-1 loss.
- The ego history tokens tie the baseline as a scorer (−0.0019, within epoch noise; BCE marginally best). Their phase-1
  gain was in the policy (val FDE −3.7%), which phase 3 does not train (CE/ED 0); the next stage (policy head) tests it.

Policy head on all four (submitted 2026-11-12): p3_aux_policy recipe exactly (overrides of 14905303: policy head only,
score-aware labels, CE 1 / ED 1, traj LR ×1.0, score 5/5 with heads frozen, no aux; 4 epochs, patience 2, monitor
`val/pdm_scores/argmax_score_gap`; batch 8 × accum 4), from each p3hist best checkpoint; 8 CPUs / 32000M / 5.5 h, `gpu,ai`.
Jobs: p3hist_none_policy 14999967, p3hist_ego_policy 14999968, p3hist_prev_policy 14999969, p3hist_both_policy 14999970.
Reference: p3_aux_policy (14905303) val argmax 0.820 (ep 1). Then RL-3.

Results (all finished 2026-11-13; per-epoch values from the W&B run files, val = navtrain val 18,179 scenes):

| Run | job, time | best ep | val argmax score, ep 0 / 1 / 2 / 3 | expected score (ep 3) | soft CE (ep 3) | expected ADE (ep 3) | head score (frozen) |
|---|---|---|---|---|---|---|---|
| `p3_aux_policy` (reference, from p1d_total) | 14905303, 3.5 h | 3 | 0.810 / **0.820** / 0.818 / 0.820 | 0.745 | 5.543 | 1.052 | 0.8585 |
| `p3hist_none_policy` | 14999967, 3.6 h | 2 | 0.807 / 0.812 / **0.813** / 0.813 | 0.744 | 5.654 | 1.042 | 0.8589 |
| `p3hist_ego_policy` | 15016982, 1.8 h | 2 | 0.814 / 0.815 / **0.818** / 0.817 | **0.748** | 5.622 | **1.015** | 0.8568 |
| `p3hist_prev_policy` | 14999969, 3.5 h | 2 | 0.799 / 0.804 / **0.804** / 0.803 | 0.736 | 5.662 | 1.020 | 0.8507 |
| `p3hist_both_policy` | 15016983, 2.2 h | 1 | 0.803 / **0.810** / 0.809 / 0.809 | 0.741 | 5.601 | 1.017 | 0.8529 |

(First attempts of ego/both, 14999968/70, ran out of GPU memory in step 1; see below. Expected score = Σ p·score over
the codebook under the policy; head score = total of the frozen sub-score heads' best entry, identical every epoch.)

- Matched comparison (same phase-1 recipe, p3hist_none as baseline): **ego history +0.004 argmax** (0.818 vs 0.813),
  expected ADE −2.6%, expected score +0.004; previous frame −0.009; both −0.004. Same ordering as phase 1 (ego helps
  the policy, the previous frame hurts) and phase 3 (scorer: ego tie, prev/both −0.007/−0.008).
- Noise: the two no-history runs differ by 0.007 argmax (p3_aux_policy 0.820 vs p3hist_none_policy 0.813; different
  phase-1 start, same head score), so the ego gain (+0.004) is within run-to-run spread; no history variant beats p3_aux_policy.
- As scorers nothing changes in this stage (heads frozen): ego ties the baseline, prev/both are worse.
- Not yet measured: shortlist / mixture proxies and the official evaluation of these runs (needs the history-cache
  override in `cache_policy_head_features.py` / `predict_vision_llm_lora_cache.py`, see "Open" below).

**OOM cause (CONFIRMED, reproduced on a tiny Qwen3.5 + PEFT model):** policy-only runs kept all LLM activations.
`gradient_checkpointing_enable()` and `enable_input_require_grads()` each register an input-embedding hook that sets
`requires_grad`, and transformers keeps only the last call's handles, so `_freeze_all_but_policy_head` removed one and
the other stayed; the freeze also turns gradient checkpointing off. Every policy-only run, p3_aux_policy included, ran at the GPU's
limit (sacct peak gpumem 39.5-40.4 GB) and at phase-1 speed (~0.9 it/s); the ego/both runs went just over. Results are
unaffected (only the policy head had trainable parameters; same forward values). Fix (2026-11-13): `VisionLLMLoRAModel.forward`
runs `_encode_inputs` and the LLM under `torch.no_grad()` when `train_policy_head_only`; test
`test_policy_only_forward_keeps_no_backbone_graph`. The resubmitted runs use the same recipe (batch 8 × accum 4).
With the fix (15016982/83, started 2026-11-13 12:39): peak GPU memory 11.3 GB (was ~40 GB), 2.0 it/s (was ~0.9).
Affected (slow, valid): the 7 `train_policy_head_only` runs (p3_aux_policy, p3_heads_policy, p3_combo_balanced_aux_policy,
p3hist_*_policy). Not affected: phases 1-3 and RL-3 (they need the backward pass), RL-1/RL-2 (cached head inputs).
Open: `cache_policy_head_features.py` and `predict_vision_llm_lora_cache.py` build the agent from the run config,
whose `history_cache_path` is the training job's mount; they need a history mount and an override before RL-3 / val
predictions of these runs. The official evaluation computes past frames from the raw images (no cache needed).

### RL-3 on the p3hist policy models (submitted 2026-11-13)

Question: does the ego history (or the previous frame) still help after RL-3, the step that gave the best models?
Recipe of `rl3_b0.005` exactly (exact objective, β = 0.005, LoRA + vision LoRA + planner queries + policy head, traj LR
×0.2, imitation and score losses 0, 2 epochs, batch 4 × accum 8, val every half epoch, monitor
`val/rl/shortlist_256_gap`), from each policy run's best epoch (none ep 2, ego ep 2, prev ep 2, both ep 1).
β = 0.005 (first submitted with 0.02, cancelled before starting): with a scorer, 0.005 is better on navtest v1 with
every scorer (mixture_v1 λ 10: own 0.9165 vs 0.9131, pairwise 0.9180 vs 0.9162, control 0.9176 vs 0.9142) and on
navhard (control k 256 0.419 vs 0.409, own mixture λ 3 0.374 vs 0.348, pairwise k 256 0.382 vs 0.364); 0.02 wins only
navtest v2 with the own / pairwise scorer (+0.002) and loses it with the control scorer (0.8953 vs 0.8965). The
argmax-only comparison (v2 0.895 vs 0.892) had pointed to 0.02.
`rl3_b0.005_p3hist_none_policy` is the matched baseline (p3hist_none_policy differs from p3_aux_policy by 0.007 val argmax).

Chain per model (`exp/rl3_logs/rl3_hist_runs_2026-11-13.txt`): merge the policy-only checkpoint (cpu) → fused-state
cache in 3 shards (1 GPU each; `run_rl_policy_head_slurm.sh` now takes `HISTORY_CACHE=1`, and
`cache_policy_head_features.py --history-cache-path` replaces the training job's temporary mounts) → RL-3, which builds
the RL reference in /dev/shm at job start (`RL_REFERENCE=<rl_features name>` in the training script; no 103k files on
/project, no squashfs packing). Jobs: none 15026637 / 38-40 / **15026641**, ego 15026642 / 43-45 / **15026646**,
prev 15026647 / 48-50 / **15026651**, both 15026652 / 53-55 / **15026656** (merge / cache / RL-3); the four RL-3 jobs
(β 0.02) were cancelled before starting and resubmitted with β 0.005: **15026878 / 79 / 80 / 81** (none / ego / prev / both).
Caches: 12 × 32-33 min, 103,288 scenes per model, ~20.5 scenes/s, peak GPU 22.5 GB. All four RL-3 jobs started
2026-11-13 20:13-20:16; the in-job RL reference (/dev/shm, 1.6 GB, 103,288 scenes, 0 skipped) took ~2 min; ~1.3-1.4 it/s.
Evaluation plan: argmax (no scorer); shortlist / mixture with each start model's own frozen scorer (as the first RL-3
evaluation) and with one common scorer (p3_aux_ctrl) for all four, which isolates the policy.

Results (all four `COMPLETED`, 2 h 29 min - 2 h 44 min; W&B val on navtrain val, `val/rl/*_epoch`; best checkpoint =
epoch 1 in every run). Each cell: argmax / shortlist k 256, at the four validations (every half epoch):

| Run | ep 0.5 | ep 1 | ep 1.5 | **ep 2 (final)** | E_π[r] final | KL final |
|---|---|---|---|---|---|---|
| `rl3_b0.005` (from p3_aux_policy, reference) | 0.8255 / 0.8405 | 0.8440 / 0.8461 | 0.8505 / 0.8480 | **0.8548 / 0.8506** | 0.8455 | 3.06 |
| `rl3_b0.005_p3hist_none_policy` | 0.8185 / 0.8358 | 0.8379 / 0.8453 | 0.8454 / 0.8460 | 0.8518 / 0.8478 | 0.8409 | 2.62 |
| `rl3_b0.005_p3hist_ego_policy` | 0.8233 / 0.8388 | 0.8377 / 0.8434 | 0.8444 / 0.8457 | 0.8513 / 0.8476 | 0.8425 | 2.57 |
| `rl3_b0.005_p3hist_prev_policy` | 0.8097 / 0.8289 | 0.8206 / 0.8340 | 0.8424 / 0.8380 | 0.8464 / 0.8405 | 0.8360 | 2.68 |
| `rl3_b0.005_p3hist_both_policy` | 0.8186 / 0.8355 | 0.8272 / 0.8386 | 0.8432 / 0.8437 | 0.8452 / 0.8440 | 0.8357 | 2.63 |

- RL-3 keeps the order of the starts: the matched baseline (none) and ego end level (0.8478 vs 0.8476 shortlist k 256,
  0.8518 vs 0.8513 argmax); the ego history's +0.004-0.005 argmax lead of the policy stage is gone after RL-3. prev and
  both stay behind (−0.007 / −0.004 shortlist), as before RL.
- All four are 0.003-0.010 below the p3_aux_policy reference, the same gap as their starts (0.813-0.818 vs 0.820 val
  argmax; the no-history runs differ by 0.007 from seed / phase-1 recipe alone).
- val has no EC and no reactive stage two, so the official navtest / navhard runs decide; on val, history does not help.

Official evaluation (2026-11-14; full tables and paired differences in final_evaluation.md, "Models with past inputs";
findings 33-35 there): **ego history is the only past input with a consistent gain**: RL-3 ego beats RL-3 none on navtest
v2 with every scorer (+0.0018 to +0.0052, up to 3 SE) and on navhard (control scorer 0.418 vs 0.384). prev / both give
nothing reliable. No history model beats the no-history reference rl3_b0.005 (navtest v2 0.8965 / v1 0.9180, navhard
0.419): RL-3 none, the matched baseline, is 0.005-0.0065 below it (run-to-run variation as large as the ego effect), and
RL-3 ego lands in between (navhard 0.418, navtest −0.002 to −0.003). One run per variant, so the ego gain is suggestive,
not conclusive.

### Seeds 2-5 of rl3_b0.005 (submitted 2026-11-17)

Goal: the seed SD of navhard / navtest with more degrees of freedom (6 seeds of rl3_b0.005 + 2 of ecshort w 4 = 6 df,
pooled) and a 6-seed mean ± SD for the thesis model (final_evaluation.md finding 48). `rl3_b0.005_s{2,3,4,5}`: recipe
identical to rl3_b0.005 / rl3_b0.005_s1, `++seed=N`. Evaluation of `epochs/epoch-epoch=01.ckpt` with the start model's
own sub-score heads (`p3_aux_policy_ep03`): navtest argmax, shortlist / shortlist_v1 k 256, mixture / mixture_v1 λ 10,
v1 PDMS with the real v1 comfort term for argmax / shortlist_v1 / mixture_v1; navhard argmax, shortlist k 256; EC rule
with the own scorer (argmax / shortlist / mixture base, n 16 / 64; navtest sequential, navhard; prediction folders
`<pred>_own_ec[seq]_*`, separate from the control-scorer `<pred>_ec[seq]_*`). The same own-scorer jobs that were missing
for seeds 0 and 1 were added, so all six seeds are evaluated identically. Script `exp/rl3_logs/submit_rl3_seeds_2to5.sh`,
job ids in `exp/rl3_logs/rl3_seeds_2to5_2026-11-17.txt` (trainings 15175909 / 36 / 62 / 88).

Status 2026-11-18: all four trainings completed (2 h 21-24 min; end-of-training val argmax 0.8542-0.8577, KL
2.64-3.15, entropy 3.08-3.12, i.e. seeds 1-5 within ±0.002 of each other on val). Predictions and EC selections all
completed. 27 of the 105 scoring jobs of this batch TIMEOUT (limits 30 / 50 min; completed jobs took 16-28 / 29-36 min;
all on six crowded nodes; final_evaluation.md finding 58). Resubmitted with 60 / 90 min
(`exp/rl3_logs/resubmit_timeouts_2026-11-18.sh`, jobs 15199720-15199772): all completed, every navtest result 12,146
valid scenes, no invalid navhard scenes. Results (6-seed mean ± SD, own heads): navtest v2 0.9056 ± 0.0014 (mixture
+ EC n 64), navtest v1 0.9169 ± 0.0010 (mixture_v1), navhard 0.438 ± 0.011 (shortlist + EC n 64); final_evaluation.md
section "Seeds of rl3_b0.005" (findings 56-57).

### Seed replicates of p3_aux and p3_heads (started 2026-11-18)

Question: is the aux-loss effect (p3_aux vs p3_heads, both from p1d_total ep 10) larger than the run-to-run spread,
at stage 2 (heads) and after stage 3 (policy head)? Seeds 1, 2 and 3 for both models (seed 3 added the same evening,
for 4 runs per model), so each seed gives a matched p3_aux / p3_heads pair (seed 0 = the existing runs). Work in `exp/seed_aux_heads/`.

1. Stage 2 (phase 3; `submit_stage2.sh`, log `stage2_runs_2026-11-18.txt`): `p3_aux_s{1,2}` = the overrides of p3_aux
   (job 14873887), `p3_heads_s{1,2}` = those of p3_heads (job 14863795), copied from the seed-0 runs'
   `code/hydra/overrides.yaml`, plus `++seed=N` (N = 1, 2, 3). Checked: the argument lists equal seed 0's, and the Hydra config
   composed with the current code differs from the seed-0 `config.yaml` only in `seed` and the per-job aux mount
   path. Not checkable: code defaults outside the YAML (no code snapshot in the run folders; later flags default to
   the old behaviour by convention). Environment as seed 0: batch 4 × accum 8 × 4 GPUs, 10 epochs, patience 3,
   monitor `val/pdm_scores/head_score_gap`, 8 loader workers, `AUX_TARGETS=1` for p3_aux (aux 3 / 3 / 1). Resources
   as seed 0, 40 CPUs / 156G / 16 h: the first submission with 8 CPUs / 32000M (15201792-95) was OOM-killed after
   3 min, while the 4 ranks loaded the full 9.4 GB p1d_total checkpoint (seed-0 peak 35-37 GB per rank; newer
   checkpoints hold only the trainable weights, 0.4-1.2 GB, which is why 32000M suffices for later runs). Jobs
   15202212 (aux s1), 13 (heads s1), 14 (aux s2), 15 (heads s2), 15202261 (aux s3), 62 (heads s3).
2. Stage 2 checkpoint choice as for seed 0: offline val evaluation (`eval_latest_run.sh` →
   `eval_vision_llm_lora_checkpoints.py`, val only), best `head_all`; all saved epochs instead of seed 0's subsets
   (p3_heads 2/4/5/8, p3_aux 2/4/5/6/8; both picked ep 5). Jobs 15202216-19 and 15202263/64 (afterok on the trainings). Never
   chosen on navtest / navhard.
3. Stage 3: `p3_aux_policy_sN` / `p3_heads_policy_sN` from the chosen stage 2 epoch, recipe of p3_aux_policy (policy
   head only, score-aware labels, CE 1 / ED 1, traj LR 1.0, 4 epochs, patience 2, batch 8 × accum 4, monitor
   `val/pdm_scores/argmax_score_gap`) + `++seed=N`; epoch by val argmax; merged checkpoints
   (`merge_policy_only_checkpoint.py`).
4. Val proxy of all 8 models (stage 2: head_all; stage 3: argmax, shortlist_256), then official: stage 2 score_heads
   (navtest v2, navhard) and score_heads_v1 (navtest v1); stage 3 argmax and shortlist k 256 (navtest v2),
   argmax and shortlist_v1 k 256 (v1), shortlist k 256 (navhard); v1 comfort for every v1 result. No EC rule, λ
   sweeps, reactive traffic or RL.
5. Analysis: per seed the paired aux − heads difference on every metric, mean ± SE over the 4 pairs, stage 2 seed SD
   of each model (4 runs each).

Status 2026-11-19: all 6 stage 2 trainings and their val evaluations COMPLETED without errors (trainings 8 h 55 min -
12 h 28 min; stopped after epoch 7 (heads s3), 8 (s2 both) or 9). Val head_all (18,179 scenes) per epoch, best in
brackets; the pick equals the training monitor's best epoch in every run:

| Run | head_all by epoch (0 … last) | chosen | head_all | head_topk | shortlist_256 | argmax |
|---|---|---|---|---|---|---|
| p3_aux (seed 0) | ep 2/4/5/6/8: 0.8191 / 0.8236 / 0.8264 / 0.8220 / 0.8217 | ep 5 | 0.8264 | 0.8119 | – | 0.7472 |
| p3_aux_s1 | 0.7907 0.8065 0.8168 0.8221 0.8243 0.8246 0.8263 0.8235 0.8233 0.8227 | ep 6 | 0.8263 | 0.8122 | 0.8218 | 0.7436 |
| p3_aux_s2 | 0.7878 0.8087 0.8173 0.8197 0.8232 0.8276 0.8257 0.8220 0.8222 | ep 5 | 0.8276 | 0.8132 | 0.8213 | 0.7362 |
| p3_aux_s3 | 0.7876 0.8083 0.8121 0.8183 0.8234 0.8232 0.8238 0.8234 0.8219 0.8201 | ep 6 | 0.8238 | 0.8121 | 0.8211 | 0.7363 |
| p3_heads (seed 0) | ep 2/4/5/8: 0.8141 / 0.8150 / 0.8182 / 0.8138 | ep 5 | 0.8182 | 0.8032 | – | 0.7430 |
| p3_heads_s1 | 0.7856 0.8003 0.8103 0.8145 0.8195 0.8179 0.8205 0.8156 0.8143 0.8135 | ep 6 | 0.8205 | 0.8060 | 0.8154 | 0.7364 |
| p3_heads_s2 | 0.7796 0.8025 0.8113 0.8108 0.8122 0.8198 0.8173 0.8132 0.8140 | ep 5 | 0.8198 | 0.8029 | 0.8142 | 0.7352 |
| p3_heads_s3 | 0.7835 0.8026 0.8059 0.8148 0.8194 0.8184 0.8160 0.8155 | ep 4 | 0.8194 | 0.8027 | 0.8150 | 0.7456 |

Val head_all aux − heads per seed: +0.0082 / +0.0058 / +0.0078 / +0.0044 (seeds 0-3), mean +0.0066 ± 0.0009 (SE over the
4 pairs); seed SD p3_aux 0.0016, p3_heads 0.0010.

**Stage 2 official evaluation (2026-11-19, all 30 jobs COMPLETED, 0 resubmissions).** `submit_stage2_official.sh`, log
`stage2_official_2026-11-19.txt`. Jobs:
- GPU predict 15236358 / 64 / 69 / 75 / 80 / 85, each 20-21 min (11 CPUs / 40G; MemEff ~100 % of 40G).
- CPU scoring 15236360-89:
  - navtest `score_heads` and `score_heads_v1`: 17.8-21.5 min (4 CPUs / 8000M, CPU efficiency 76 %);
  - navhard `score_heads`: 30.0-35.2 min (4 CPUs / 6G, 84 %).
- v1 comfort 15236363 / 68 / 74 / 79 / 84 / 90: 9.4-12.4 min.

0 failed scenes. The results and the paired analysis are in
[final_evaluation.md](final_evaluation.md), section "Seed replicates of p3_aux and p3_heads". Collector:
`exp/seed_aux_heads/collect_seed_study.py`.

**Stage 3 trainings (2026-11-19, all 6 COMPLETED).** `submit_stage3.sh`, log `stage3_runs_2026-11-19.txt`, jobs
15236401-06.
- Setup: 12 CPUs / 48000M, nodes cn02, cn03 and x1001 nodes.
- Config: the composed config differs from p3_aux_policy only in start checkpoint, experiment name and seed.
- Run length: all 4 epochs ran (patience 2 never triggered), 1 h 38 min - 1 h 41 min each, ~24 min per epoch. Seed 0
  took ~51 min per epoch (p3_aux_policy 2026-11-06, p3_heads_policy 2026-11-07) with the same trainer settings
  (`max_epochs 4`, full train / val, val every epoch). The cause of the 2× speed-up was not checked (UNKNOWN). The
  logs contain no errors, only the `fla` "Python 3.10 below 3.11" warning.
- Two folders: p3_aux_policy_s2 has two run folders (2026.11.19.17.30.31 with the checkpoints, 17.30.32 with only a
  log; the ranks started one second apart). The scripts take the folder with `checkpoints/epochs`.
- Usage: CPU efficiency 44-49 %. sacct "memory utilized" 49-57 GB, which is above the 46.9 GB limit, but no job was
  killed (the counter includes shared or cached pages).

Val evaluations 15236408-13 (16 CPUs / 64000M, 1 h 03-06 min for 4 epochs; CPU efficiency 8 %, a GPU-bound job). Val
(18,179 scenes) per epoch, argmax `sel_score` and shortlist_256 (k 256). head_all is constant within each run (frozen
heads) and equals the chosen stage 2 epoch's value, which confirms each run started from its own stage 2 checkpoint.

| Run (start) | argmax ep 0 / 1 / 2 / 3 | shortlist_256 ep 0 / 1 / 2 / 3 | chosen (val argmax) | training monitor best |
|---|---|---|---|---|
| p3_aux_policy (p3_aux ep 5) | – / 0.7879 / – / 0.7895 | – / 0.8283 / – / 0.8278 | ep 3 | ep 1 |
| p3_aux_policy_s1 (p3_aux_s1 ep 6) | 0.7823 / **0.7876** / 0.7865 / 0.7870 | 0.8281 / 0.8290 / 0.8300 / 0.8290 | ep 1 | ep 3 |
| p3_aux_policy_s2 (p3_aux_s2 ep 5) | 0.7814 / 0.7871 / **0.7880** / 0.7872 | 0.8289 / 0.8296 / 0.8309 / 0.8307 | ep 2 | ep 3 |
| p3_aux_policy_s3 (p3_aux_s3 ep 6) | 0.7817 / 0.7867 / **0.7899** / 0.7897 | 0.8289 / 0.8294 / 0.8295 / 0.8297 | ep 2 | ep 3 |
| p3_heads_policy (p3_heads ep 5) | – / – / 0.7772 / – | – / – / 0.8175 / – | ep 2 | ep 2 |
| p3_heads_policy_s1 (p3_heads_s1 ep 6) | 0.7689 / 0.7727 / 0.7729 / **0.7733** | 0.8190 / 0.8200 / 0.8197 / 0.8198 | ep 3 | ep 2 |
| p3_heads_policy_s2 (p3_heads_s2 ep 5) | 0.7735 / 0.7775 / **0.7787** / 0.7778 | 0.8203 / 0.8205 / 0.8217 / 0.8218 | ep 2 | ep 3 |
| p3_heads_policy_s3 (p3_heads_s3 ep 4) | 0.7696 / 0.7724 / 0.7719 / **0.7734** | 0.8192 / 0.8195 / 0.8199 / 0.8200 | ep 3 | ep 3 |

How the epochs were chosen:
- Epochs 1-3 differ by ≤ 0.001 in val argmax, against a per-run SE of 0.0024. The val pick and the training
  monitor's pick (lenient metric) agree in 2 of the 6 new runs.
- Seed 0 was evaluated on only a subset of epochs: p3_aux_policy on ep 1 and 3 (val picked ep 3), p3_heads_policy on
  ep 2 only (the monitor's pick). Its choice is therefore not a full val argmax over 4 epochs. Given the ≤ 0.001
  spread between epochs, this matters little.

Merged checkpoints (`merge_policy_only_checkpoint.py`, 2026-11-19 20:28-20:31, on the login node, ~1 min each):
`<run>/checkpoints/merged/epoch-epoch=NN.ckpt`.
- p3_aux runs: 686 keys from the stage 2 checkpoint, 5 replaced by the policy head.
- p3_heads runs: 624 keys, 5 replaced.

**Stage 3 official evaluation (submitted 2026-11-19 20:32).** `submit_stage3_official.sh`, log
`stage3_official_2026-11-19.txt`, jobs 15239330-84.
- GPU predict on the merged checkpoints, then navtest argmax (gives v1 and v2), shortlist k 256 and shortlist_v1
  k 256; navhard argmax and shortlist k 256; v1 comfort on the argmax and shortlist_v1 k 256 results.
- Seed 0's p3_heads_policy_ep02 had only shortlist k 64 results, so the same CPU jobs run on its saved predictions
  (15239378-84, no GPU job).
- Shortlist k 64 for the 6 new models, added at the user's request: navtest shortlist and shortlist_v1, navhard
  shortlist, and v1 comfort on shortlist_v1. These are CPU jobs on the same predictions (`PRED=… AFTER=<GPU job>`),
  15239483-506. Seed 0 already had k 64 for both models.
- Status: all 79 jobs COMPLETED, 0 resubmissions (finished 2026-11-19 ~21:20).
  - GPU predict: 17.9-19.1 min.
  - navtest scoring: 15.4-31.6 min.
  - navhard scoring: 29.2-44.6 min.
  - v1 comfort: 7.9-10.8 min.

  Results and paired analysis: [final_evaluation.md](final_evaluation.md), section "Seed replicates of p3_aux and
  p3_heads", findings 59-64.

### RL-3 on the seed 1-3 policy models: whole-pipeline replicates (submitted 2026-11-19 20:45)

**Question.** How much does the full pipeline (phase 3 → policy stage → RL-3) vary when every stage gets a different
seed? The 6 seeds of rl3_b0.005 all start from p3_aux_policy ep 3 (seed 0), so they measure only the RL-3 seed.

**Runs.** `rl3_b0.005_pipe_s{1,2,3}` use the rl3_b0.005 recipe (exact objective, β 0.005, 2 epochs, batch 4 × accum
8, monitor `val/rl/shortlist_256_gap`) with `++seed=N`. They start from the merged stage 3 checkpoints
p3_aux_policy_s1 ep 1, s2 ep 2 and s3 ep 2. Seed 0 of the pipeline is rl3_b0.005_ep01, which gives 4 full-pipeline
replicates.

**Config check.** The composed config differs from `rl3_b0.005_s2`'s only in:
- experiment name and seed;
- start checkpoint;
- `rl_reference_head_path` (the new model's fused-state cache);
- the per-job /dev/shm reference path.

**Chain per model.** Script `exp/rl3_logs/submit_rl3_pipeline.sh`, log `rl3_pipeline_runs_2026-11-19.txt`.
1. Fused-state cache `rl_features/p3_aux_policy_sN_epNN`: 3 shards, 1 GPU each, 8 CPUs / 32000M / 50 min, as for
   p3_aux_policy_ep03.
2. RL-3 training: 12 CPUs / 46000M / 4 h. The rl3_b0.005 seeds ran with 32 CPUs at 14 % CPU efficiency.
3. GPU predictions of epoch 1. These also wait for the start model's stage 3 official GPU job, whose predictions
   serve as the own scorer.
4. The evaluation chain of the rl3_b0.005 seeds (`submit_rl3_seeds_2to5.sh`), with the start model's own heads
   (`SCORES_PRED` = `p3_aux_policy_sN_epNN`):
   - navtest: argmax, shortlist k 256, shortlist_v1 k 256, mixture λ 10, mixture_v1 λ 10;
   - v1 comfort on the three v1 results;
   - navhard: argmax and shortlist k 256;
   - EC rule (own scorer, argmax / shortlist / mixture base, n 16 / 64; navtest sequential and navhard).

   Scoring limits are 60 / 90 min.

| Run | cache | training | predict | scoring / v1 comfort / EC jobs |
|---|---|---|---|---|
| rl3_b0.005_pipe_s1 | 15239395-97 | 15239398 | 15239399 | 15239400-23 |
| rl3_b0.005_pipe_s2 | 15239424-26 | 15239427 | 15239428 | 15239429-52 |
| rl3_b0.005_pipe_s3 | 15239453-55 | 15239456 | 15239457 | 15239458-81 |

**Status 2026-11-20: all 87 jobs completed (2 after a resubmission).**
- Durations:
  - caches 30-32 min;
  - trainings 2 h 22-24 min (cn02, x1001c5s2b0n1, x1001c0s0b0n0);
  - predictions 16-17 min;
  - navtest scoring 15-21 min, navhard 29-34 min;
  - EC selection 7-20 min;
  - v1 comfort 8-33 min.
- Run folders with checkpoints, used by the predictions:
  - `rl3_b0.005_pipe_s1/2026.11.19.21.20.45`;
  - `rl3_b0.005_pipe_s2/2026.11.19.21.19.06`;
  - `rl3_b0.005_pipe_s3/2026.11.19.21.18.58`.

  The extra empty timestamp folders next to them (s1: 1, s2: 2) are the known multi-rank artifact.
- End-of-training val (W&B summary, epoch 1):
  - argmax 0.8571 / 0.8576 / 0.8574;
  - shortlist 256 0.8515 / 0.8514 / 0.8502;
  - KL 2.63 / 2.79 / 2.65;
  - entropy 3.16 / 3.06 / 3.10.

  All are within the range of the 6 rl3_b0.005 seeds.
- Two pipe_s3 jobs hit their time limit on stalled nodes, not because of the code:
  - v1 comfort for mixture_v1 (15239467): empty log after 40 min on x1000c2s4b1n0, where pipe_s2's v1c jobs also took
    23-33 min instead of ~9;
  - navhard EC rule, argmax base n 64 (15239473): 48 min before scoring started on x1000c5s3b1n1, while its 11 sibling
    jobs took ~30 min in total.

  Both were resubmitted 2026-11-20 01:37 with the same settings, excluding the two nodes: 15247940 and 15247941 (logged
  in `rl3_pipeline_runs_2026-11-19.txt`). Both completed, in 14 and 32 min.
- Results: final_evaluation.md section "Whole-pipeline replicates of RL-3" (findings 65-68). The whole-pipeline spread
  is not larger than the RL-seed spread. Thesis numbers over the 4 pipeline runs:
  - navtest v2 0.9064 ± 0.0016 (mixture + EC n 64);
  - navtest v1 0.9183 ± 0.0005 (mixture_v1);
  - navhard 0.435 ± 0.007 (shortlist + EC n 64).

**Val history** (navtrain val, 18,179 scenes; 4 checks per run at val_check_interval 0.5; read from each run's W&B
binary `wandb/latest-run/*.wandb` by `exp/rl3_logs/dump_rl3_pipeline_full.py`; seed 0 = rl3_b0.005
`2026.11.08.23.46.54`). The shortlist values use the start model's cached top-1024 heads shortlist; "argmax zero" =
share of val scenes whose argmax scores 0; "changed" = share whose argmax differs from the start (reference) policy's argmax
(`navsim/agents/vision_llm_lora/vision_llm_lora_rl.py:130-155`).

| Run | step (epoch) | argmax | shortlist 256 | shortlist 64 | expected reward | KL | entropy | argmax zero | changed | val loss | expected ADE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 332 (0) | 0.8255 | 0.8405 | 0.8383 | 0.8132 | 3.0840 | 3.3921 | 0.0870 | 0.9483 | -0.7978 | 1.4277 |
| seed 0 | 664 (0) | 0.8440 | 0.8461 | 0.8414 | 0.8339 | 3.4493 | 3.1677 | 0.0668 | 0.9438 | -0.8166 | 1.4687 |
| seed 0 | 997 (1) | 0.8505 | 0.8480 | 0.8433 | 0.8413 | 3.0289 | 3.0947 | 0.0702 | 0.9439 | -0.8262 | 1.3368 |
| seed 0 | 1329 (1) | 0.8548 | 0.8506 | 0.8443 | 0.8455 | 3.0553 | 3.0766 | 0.0664 | 0.9403 | -0.8303 | 1.3501 |
| pipe s1 | 332 (0) | 0.8256 | 0.8424 | 0.8387 | 0.8130 | 3.3542 | 3.5423 | 0.0816 | 0.9571 | -0.7963 | 1.5576 |
| pipe s1 | 664 (0) | 0.8332 | 0.8431 | 0.8387 | 0.8207 | 2.7018 | 3.3693 | 0.0795 | 0.9447 | -0.8072 | 1.3232 |
| pipe s1 | 997 (1) | 0.8525 | 0.8513 | 0.8428 | 0.8442 | 2.8758 | 3.1638 | 0.0667 | 0.9498 | -0.8298 | 1.3148 |
| pipe s1 | 1329 (1) | 0.8571 | 0.8515 | 0.8431 | 0.8465 | 2.6289 | 3.1640 | 0.0673 | 0.9420 | -0.8333 | 1.2491 |
| pipe s2 | 332 (0) | 0.8130 | 0.8372 | 0.8371 | 0.8017 | 3.8409 | 3.5059 | 0.0898 | 0.9651 | -0.7825 | 1.5902 |
| pipe s2 | 664 (0) | 0.8388 | 0.8441 | 0.8406 | 0.8282 | 2.9533 | 3.2650 | 0.0774 | 0.9467 | -0.8134 | 1.3353 |
| pipe s2 | 997 (1) | 0.8528 | 0.8486 | 0.8429 | 0.8426 | 2.6686 | 3.1173 | 0.0680 | 0.9376 | -0.8293 | 1.2318 |
| pipe s2 | 1329 (1) | 0.8576 | 0.8514 | 0.8440 | 0.8464 | 2.7886 | 3.0642 | 0.0681 | 0.9410 | -0.8325 | 1.2476 |
| pipe s3 | 332 (0) | 0.8320 | 0.8413 | 0.8416 | 0.8152 | 2.5701 | 3.6841 | 0.0761 | 0.9383 | -0.8023 | 1.3728 |
| pipe s3 | 664 (0) | 0.8356 | 0.8444 | 0.8412 | 0.8222 | 2.8301 | 3.4028 | 0.0785 | 0.9498 | -0.8080 | 1.3177 |
| pipe s3 | 997 (1) | 0.8508 | 0.8484 | 0.8449 | 0.8410 | 2.7363 | 3.1300 | 0.0731 | 0.9446 | -0.8273 | 1.2701 |
| pipe s3 | 1329 (1) | 0.8574 | 0.8502 | 0.8452 | 0.8486 | 2.6545 | 3.1018 | 0.0661 | 0.9387 | -0.8353 | 1.2488 |

- Every run improves monotonically on val argmax (0.813-0.832 → 0.855-0.858) and expected reward (0.802-0.815 →
  0.846-0.849). The last check is the best on argmax in all four runs. The monitor `val/rl/shortlist_256_gap` is also
  lowest at the last check, so the epoch 1 checkpoint is the monitor's choice as well.
- KL to the reference ends at 2.63-3.06 and entropy at 3.06-3.16. The pipeline runs end with slightly lower KL than
  seed 0 (2.63-2.79 vs 3.06).

**Job accounting** (sacct; CPU efficiency = TotalCPU / (elapsed × CPUs)):

| Run | Job type | jobs | state | elapsed (min) | CPUs | CPU efficiency | peak RSS / request | nodes | start → end (2026-11-19/20) |
|---|---|---|---|---|---|---|---|---|---|
| pipe s1 | fused-state cache (3 shards) | 15239395-97 (3) | 3 COMPLETED | 29.8-32.0 | 8 | 15-16% | 24.9G/31.2G of 32000M | x1001c0s0b0n1, x1001c0s1b0n1 | 11-19T20:45 → 11-19T21:17 |
| pipe s1 | RL-3 training | 15239398 | 1 COMPLETED | 144.4 | 12 | 37% | 18.1G of 46000M | cn02 | 11-19T21:17 → 11-19T23:42 |
| pipe s1 | GPU predictions | 15239399 | 1 COMPLETED | 16.2 | 11 | 34% | 40.0G of 40G | x1001c0s0b0n0 | 11-19T23:42 → 11-19T23:58 |
| pipe s1 | navtest scoring | 15239400-22 (11) | 11 COMPLETED | 15.3-17.9 | 4 | 85-94% | 6.1G/6.4G/6.8G/7.8G of 8000M | x1000c2s2b1n1, x1000c5s5b0n0 | 11-19T23:58 → 11-20T00:34 |
| pipe s1 | v1 comfort | 15239407-09 (3) | 3 COMPLETED | 8.4-9.5 | 4 | 87-94% | 7.8G of 8000M | x1000c2s3b0n1, x1000c5s5b0n0 | 11-20T00:14 → 11-20T00:24 |
| pipe s1 | EC selection | 15239410-11 (2) | 2 COMPLETED | 7.0-18.0 | 9 | 68-87% | 11.5G/16.0G of 16G | x1000c5s5b0n0, x1000c6s0b0n0 | 11-19T23:58 → 11-20T00:16 |
| pipe s1 | navhard scoring | 15239405-23 (8) | 8 COMPLETED | 28.9-30.5 | 4 | 83-84% | 6.0G of 6G | x1000c5s5b0n0, x1000c6s0b0n0 | 11-19T23:58 → 11-20T00:34 |
| pipe s2 | fused-state cache (3 shards) | 15239424-26 (3) | 3 COMPLETED | 30.0-31.1 | 8 | 16% | 31.2G of 32000M | x1001c1s1b0n0, x1001c2s1b0n0, x1001c4s0b0n1 | 11-19T20:45 → 11-19T21:16 |
| pipe s2 | RL-3 training | 15239427 | 1 COMPLETED | 142.0 | 12 | 38% | 18.1G of 46000M | x1001c5s2b0n1 | 11-19T21:16 → 11-19T23:38 |
| pipe s2 | GPU predictions | 15239428 | 1 COMPLETED | 17.2 | 11 | 32% | 40.0G of 40G | x1001c0s0b0n0 | 11-19T23:38 → 11-19T23:56 |
| pipe s2 | navtest scoring | 15239429-51 (11) | 11 COMPLETED | 15.6-18.7 | 4 | 83-94% | 7.0G/7.5G/7.8G of 8000M | 4 nodes | 11-19T23:56 → 11-20T00:33 |
| pipe s2 | v1 comfort | 15239436-38 (3) | 3 COMPLETED | 12.4-33.0 | 4 | 55-81% | 7.8G of 8000M | x1000c2s4b1n0, x1000c7s4b0n0 | 11-20T00:13 → 11-20T00:47 |
| pipe s2 | EC selection | 15239439-40 (2) | 2 COMPLETED | 7.1-18.6 | 9 | 69-81% | 16.0G of 16G | x1000c2s3b1n1 | 11-19T23:56 → 11-20T00:14 |
| pipe s2 | navhard scoring | 15239434-52 (8) | 8 COMPLETED | 28.8-34.2 | 4 | 82-85% | 6.0G of 6G | 4 nodes | 11-19T23:56 → 11-20T00:37 |
| pipe s3 | fused-state cache (3 shards) | 15239453-55 (3) | 3 COMPLETED | 30.1-30.6 | 8 | 15-16% | 31.2G of 32000M | x1001c5s1b0n1, x1001c5s2b0n1 | 11-19T20:45 → 11-19T21:16 |
| pipe s3 | RL-3 training | 15239456 | 1 COMPLETED | 142.5 | 12 | 38% | 18.2G of 46000M | x1001c0s0b0n0 | 11-19T21:16 → 11-19T23:38 |
| pipe s3 | GPU predictions | 15239457 | 1 COMPLETED | 17.2 | 11 | 34% | 40.0G of 40G | x1001c0s0b0n0 | 11-19T23:38 → 11-19T23:56 |
| pipe s3 | navtest scoring | 15239458-80 (11) | 11 COMPLETED | 16.7-21.0 | 4 | 82-95% | 6.0G/6.1G/7.3G/7.8G of 8000M | 5 nodes | 11-19T23:56 → 11-20T00:37 |
| pipe s3 | v1 comfort | 15239465-67 + 15247940 (4) | 3 COMPLETED, 1 TIMEOUT | 11.1-13.7 (timeout 40.4) | 4 | 58-79% | 7.8G of 8000M | x1000c0s0b0n1, x1000c2s4b1n0, x1000c3s0b1n1 | 11-20T00:13 → 11-20T01:51 |
| pipe s3 | EC selection | 15239468-69 (2) | 2 COMPLETED | 7.1-20.4 | 9 | 70-83% | 16.0G of 16G | x1000c5s5b0n0 | 11-19T23:56 → 11-20T00:16 |
| pipe s3 | navhard scoring | 15239463-81 + 15247941 (9) | 8 COMPLETED, 1 TIMEOUT | 29.7-32.3 (timeout 90.0) | 4 | 40-88% | 6.0G of 6G | 5 nodes | 11-19T23:56 → 11-20T02:09 |

- Peak RSS equals the request for navhard scoring (6G), v1 comfort (8000M), EC selection (16G) and the GPU
  predictions (40G). All of these completed, so the cap did not kill them. Whether page cache is counted toward the
  cgroup RSS here is UNVERIFIED.
- The fused-state caches used 15-16% of 8 CPUs; 4 CPUs would do. The training used 37-38% of 12 CPUs, against 14% of
  32 for the rl3_b0.005 seeds.
- Seed 0's jobs are listed in the sections of rl3_b0.005 and its seeds.

### RL-3 on the seed 1-3 heads policy models: matched aux / heads pairs after RL-3 (submitted 2026-11-20 23:47)

**Question.** Does the aux-loss advantage (findings 59-64) survive RL-3? RL-3 evens out differences between start
models (finding 66), and the heads family had one RL-3 run only (rl3_heads_b0.005, seed 0; aux − heads after RL-3:
navtest v1 +0.012, v2 +0.014, navhard +0.017, a single pair).

**Runs.** `rl3_heads_b0.005_pipe_s{1,2,3}`, the same recipe as the aux pipeline runs (rl3_b0.005, β 0.005, `++seed=N`),
from the val-chosen stage 3 heads models: p3_heads_policy_s1 ep 3, s2 ep 2, s3 ep 3 (merged checkpoints). Each run's
own scorer is its start model's stage 3 predictions (`p3_heads_policy_sN_epNN`, already on disk, so the predict job
depends only on the training). Seed 0 of the heads pipeline is `rl3_heads_b0.005_ep01`: its composed overrides differ
from rl3_b0.005's only in the start checkpoint, the RL reference cache and the experiment name (checked 2026-11-20),
so 4 matched aux / heads pairs after RL-3 result (aux: rl3_b0.005_ep01 + rl3_b0.005_pipe_s1-s3).

**Chain.** `exp/rl3_logs/submit_rl3_pipeline.sh` (now with `SRC`, `EXP`, `LOG` and a "-" GPU id; defaults unchanged)
run as `SRC=p3_heads_policy EXP=rl3_heads_b0.005_pipe SPECS="1:3:- 2:2:- 3:3:-"`; log
`exp/rl3_logs/rl3_heads_pipeline_runs_2026-11-20.txt`. Same jobs and resources as the aux pipeline runs.

| Run | cache | training | predict | scoring / v1 comfort / EC jobs |
|---|---|---|---|---|
| rl3_heads_b0.005_pipe_s1 | 15282125-27 | 15282128 | 15282129 | 15282130-53 |
| rl3_heads_b0.005_pipe_s2 | 15282154-56 | 15282157 | 15282158 | 15282159-82 |
| rl3_heads_b0.005_pipe_s3 | 15282183-85 | 15282186 | 15282187 | 15282188-211 |

**Status 2026-11-21: all 87 jobs completed** (no timeouts).
- Durations:
  - caches 31-32 min;
  - trainings 2 h 31-32 min;
  - predictions 19 min;
  - navtest scoring 15-25 min, navhard 29-40 min;
  - EC selection 7-21 min;
  - v1 comfort 8-12 min.
- Run folders with checkpoints, used by the predictions:
  - `rl3_heads_b0.005_pipe_s1/2026.11.21.00.25.17`;
  - `rl3_heads_b0.005_pipe_s2/2026.11.21.00.21.42`;
  - `rl3_heads_b0.005_pipe_s3/2026.11.21.00.24.13`.

  The empty twin folders are the multi-rank artifact.
- Every navtest result has 12,146 valid scenes; no navhard scene is invalid.
- Results: final_evaluation.md section "aux vs heads after RL-3" (findings 71-73). aux > heads in 4/4 pairs on every
  navtest rule (+0.009 to +0.012) and on navhard argmax / argmax- and mixture-base EC rules; 3/4 on the navhard
  shortlist rules.

**Val history** (as for the aux pipeline runs; seed 0 = `rl3_heads_b0.005/2026.11.09.00.15.00`):

| Run | step (epoch) | argmax | shortlist 256 | shortlist 64 | expected reward | KL | entropy | argmax zero | changed | val loss | expected ADE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 332 (0) | 0.8082 | 0.8293 | 0.8282 | 0.7919 | 3.5434 | 3.5982 | 0.0972 | 0.9525 | -0.7742 | 1.5184 |
| seed 0 | 664 (0) | 0.8293 | 0.8382 | 0.8322 | 0.8175 | 2.5288 | 3.3432 | 0.0891 | 0.9346 | -0.8049 | 1.2622 |
| seed 0 | 997 (1) | 0.8372 | 0.8381 | 0.8316 | 0.8249 | 2.4150 | 3.2012 | 0.0876 | 0.9349 | -0.8128 | 1.1887 |
| seed 0 | 1329 (1) | 0.8403 | 0.8387 | 0.8327 | 0.8304 | 2.5310 | 3.1349 | 0.0832 | 0.9358 | -0.8178 | 1.2087 |
| pipe s1 | 332 (0) | 0.8169 | 0.8345 | 0.8346 | 0.8039 | 2.8619 | 3.6157 | 0.0953 | 0.9544 | -0.7896 | 1.3908 |
| pipe s1 | 664 (0) | 0.8351 | 0.8408 | 0.8354 | 0.8239 | 3.4624 | 3.3836 | 0.0704 | 0.9488 | -0.8066 | 1.4699 |
| pipe s1 | 997 (1) | 0.8409 | 0.8413 | 0.8373 | 0.8275 | 2.9295 | 3.1948 | 0.0812 | 0.9414 | -0.8129 | 1.3128 |
| pipe s1 | 1329 (1) | 0.8474 | 0.8444 | 0.8388 | 0.8366 | 2.9573 | 3.1451 | 0.0733 | 0.9399 | -0.8218 | 1.3189 |
| pipe s2 | 332 (0) | 0.8103 | 0.8302 | 0.8300 | 0.7963 | 3.3151 | 3.5741 | 0.1006 | 0.9543 | -0.7798 | 1.4660 |
| pipe s2 | 664 (0) | 0.8358 | 0.8398 | 0.8360 | 0.8258 | 3.0342 | 3.2504 | 0.0755 | 0.9451 | -0.8107 | 1.3595 |
| pipe s2 | 997 (1) | 0.8412 | 0.8415 | 0.8357 | 0.8296 | 2.8477 | 3.2035 | 0.0789 | 0.9427 | -0.8153 | 1.3131 |
| pipe s2 | 1329 (1) | 0.8448 | 0.8411 | 0.8357 | 0.8342 | 2.6293 | 3.1302 | 0.0773 | 0.9348 | -0.8211 | 1.2420 |
| pipe s3 | 332 (0) | 0.8070 | 0.8273 | 0.8278 | 0.7900 | 2.6202 | 3.5375 | 0.1094 | 0.9467 | -0.7769 | 1.3073 |
| pipe s3 | 664 (0) | 0.8305 | 0.8387 | 0.8340 | 0.8207 | 3.0046 | 3.2280 | 0.0808 | 0.9432 | -0.8057 | 1.3116 |
| pipe s3 | 997 (1) | 0.8328 | 0.8383 | 0.8332 | 0.8215 | 2.7078 | 3.1267 | 0.0900 | 0.9390 | -0.8079 | 1.2355 |
| pipe s3 | 1329 (1) | 0.8425 | 0.8408 | 0.8345 | 0.8316 | 2.6894 | 3.0792 | 0.0793 | 0.9336 | -0.8181 | 1.2236 |

- Val argmax ends at 0.8403-0.8474 (aux pipeline 0.8548-0.8576); the last check is the best in all four runs.
- KL ends at 2.53-2.96 and entropy at 3.08-3.15.

**Job accounting** (sacct; CPU efficiency = TotalCPU / (elapsed × CPUs)):

| Run | Job type | jobs | state | elapsed (min) | CPUs | CPU efficiency | peak RSS / request | start → end (2026-11-20/21) |
|---|---|---|---|---|---|---|---|---|
| heads s1 | fused-state cache (3 shards) | 15282125-27 (3) | 3 COMPLETED | 31.4-32.4 | 8 | 14-15% | 25.7G/31.2G of 32000M | 11-20T23:47 → 11-21T00:20 |
| heads s1 | RL-3 training | 15282128 | 1 COMPLETED | 150.8 | 12 | 36% | 17.1G of 46000M | 11-21T00:21 → 11-21T02:52 |
| heads s1 | GPU predictions | 15282129 | 1 COMPLETED | 18.8 | 11 | 32% | 40.0G of 40G | 11-21T02:52 → 11-21T03:10 |
| heads s1 | navtest scoring | 15282130-52 (11) | 11 COMPLETED | 16.6-21.4 | 4 | 75-93% | 7.2G/7.8G of 8000M | 11-21T03:11 → 11-21T03:53 |
| heads s1 | v1 comfort | 15282137-39 (3) | 3 COMPLETED | 7.9-11.5 | 4 | 79-90% | 7.8G of 8000M | 11-21T03:29 → 11-21T03:45 |
| heads s1 | EC selection | 15282140-41 (2) | 2 COMPLETED | 7.3-20.5 | 8 | 74-81% | 16.0G of 16G | 11-21T03:11 → 11-21T03:32 |
| heads s1 | navhard scoring | 15282135-53 (8) | 8 COMPLETED | 30.9-36.2 | 4 | 75-81% | 6.0G of 6G | 11-21T03:11 → 11-21T03:55 |
| heads s2 | fused-state cache (3 shards) | 15282154-56 (3) | 3 COMPLETED | 31.4 | 8 | 14-15% | 23.4G/25.6G/31.2G of 32000M | 11-20T23:47 → 11-21T00:19 |
| heads s2 | RL-3 training | 15282157 | 1 COMPLETED | 150.9 | 12 | 38% | 17.0G of 46000M | 11-21T00:19 → 11-21T02:50 |
| heads s2 | GPU predictions | 15282158 | 1 COMPLETED | 18.7 | 11 | 30% | 40.0G of 40G | 11-21T02:50 → 11-21T03:08 |
| heads s2 | navtest scoring | 15282159-81 (11) | 11 COMPLETED | 14.8-25.1 | 4 | 69-93% | 7.8G of 8000M | 11-21T03:09 → 11-21T03:47 |
| heads s2 | v1 comfort | 15282166-68 (3) | 3 COMPLETED | 8.8-11.3 | 4 | 78-87% | 7.8G of 8000M | 11-21T03:29 → 11-21T03:43 |
| heads s2 | EC selection | 15282169-70 (2) | 2 COMPLETED | 8.0-20.1 | 9 | 64-75% | 16.0G of 16G | 11-21T03:09 → 11-21T03:29 |
| heads s2 | navhard scoring | 15282164-82 (8) | 8 COMPLETED | 29.8-40.2 | 4 | 75-83% | 6.0G of 6G | 11-21T03:09 → 11-21T03:55 |
| heads s3 | fused-state cache (3 shards) | 15282183-85 (3) | 3 COMPLETED | 31.3-31.6 | 8 | 15-16% | 31.2G of 32000M | 11-20T23:47 → 11-21T00:19 |
| heads s3 | RL-3 training | 15282186 | 1 COMPLETED | 152.3 | 12 | 36% | 15.3G of 46000M | 11-21T00:19 → 11-21T02:52 |
| heads s3 | GPU predictions | 15282187 | 1 COMPLETED | 18.8 | 11 | 28% | 40.0G of 40G | 11-21T02:52 → 11-21T03:10 |
| heads s3 | navtest scoring | 15282188-10 (11) | 11 COMPLETED | 15.5-18.9 | 4 | 77-91% | 7.8G of 8000M | 11-21T03:11 → 11-21T03:49 |
| heads s3 | v1 comfort | 15282195-97 (3) | 3 COMPLETED | 10.3-10.7 | 4 | 75-80% | 7.8G of 8000M | 11-21T03:29 → 11-21T03:41 |
| heads s3 | EC selection | 15282198-99 (2) | 2 COMPLETED | 7.3-20.5 | 8 | 75-83% | 16.0G of 16G | 11-21T03:11 → 11-21T03:32 |
| heads s3 | navhard scoring | 15282193-11 (8) | 8 COMPLETED | 29.4-32.8 | 4 | 78-83% | 6.0G of 6G | 11-21T03:11 → 11-21T03:52 |
| heads seed 0 (own-scorer additions) | navtest scoring | 15282212-25 (7) | 7 COMPLETED | 17.2-21.0 | 4 | 77-85% | 6.6G/7.3G/7.8G of 8000M | 11-20T23:48 → 11-21T00:29 |
| heads seed 0 (own-scorer additions) | EC selection | 15282213-14 (2) | 2 COMPLETED | 8.1-20.5 | 8 | 68-80% | 16.0G of 16G | 11-20T23:48 → 11-21T00:09 |
| heads seed 0 (own-scorer additions) | navhard scoring | 15282216-26 (6) | 6 COMPLETED | 28.9-31.8 | 4 | 80-82% | 6.0G of 6G | 11-20T23:57 → 11-21T00:28 |

### Own-scorer evaluations that were missing (submitted 2026-11-20 23:47)

- `rl3_heads_b0.005_ep01` (heads seed 0) already had its own scorer (p3_heads_policy_ep02) for argmax, shortlist /
  shortlist_v1 k 256, mixture λ 3, mixture_v1 λ 10 and navhard. Added: navtest mixture λ 10 (15282212) and the EC rule
  with the own scorer (selection 15282213-14, scoring 15282215-26), so that it matches the aux runs' rule set.
- `rl3_ecshort4_b0.005_ep01` / `_s1_ep01` (EC in the reward, shortlist previous plan, weight 4). The navhard best
  0.456 (shortlist + EC rule n 64) and every other scorer-based number of the EC-reward models used the control scorer
  (p3_aux_ctrl ep 2). Only mixture_v1 λ 10 had the own scorer (p3_aux_policy_ep03). Added with the own scorer:
  - navtest shortlist k 256, shortlist_v1 k 256 (+ v1 comfort), mixture λ 10;
  - navhard shortlist k 256;
  - EC rule (3 bases, n 16 / 64; navtest sequential, navhard).

  Seed 0 jobs 15282227-45, seed 1 jobs 15282246-64.
- The other EC-reward models (ec2, ec4, ecshort2) still have only the control scorer.

Script `exp/rl3_logs/submit_own_scorer_evals.sh`, log `exp/rl3_logs/own_scorer_evals_2026-11-20.txt`. Output
folders: `<pred>_<rule>` and `<pred>_own_ec[seq]_*`, separate from the control-scorer folders `<pred>__ctrlscorer_*`
and `<pred>_ec[seq]_*`.

**Status 2026-11-21: all 53 jobs completed** (ecshort4 navtest scoring 17-35 min, navhard 29-33 min).
Results: final_evaluation.md "aux vs heads after RL-3" (heads seed 0) and "rl3_ecshort4 with its own scorer"
(finding 74). With the own scorer, ecshort4's navhard shortlist + EC n 64 is 0.453 / 0.450 (control 0.456 / 0.444).

| Run | Job type | jobs | state | elapsed (min) | CPUs | CPU efficiency | peak RSS / request | start → end (2026-11-20/21) |
|---|---|---|---|---|---|---|---|---|
| ecshort4 seed 0 (own scorer) | navtest scoring | 15282227-44 (9) | 9 COMPLETED | 16.9-35.1 | 4 | 68-86% | 6.3G/7.8G of 8000M | 11-20T23:48 → 11-21T00:29 |
| ecshort4 seed 0 (own scorer) | v1 comfort | 15282231 | 1 COMPLETED | 12.1 | 4 | 71% | 7.8G of 8000M | 11-21T00:05 → 11-21T00:17 |
| ecshort4 seed 0 (own scorer) | EC selection | 15282232-33 (2) | 2 COMPLETED | 7.5-19.7 | 8 | 71-86% | 16.0G of 16G | 11-20T23:48 → 11-21T00:08 |
| ecshort4 seed 0 (own scorer) | navhard scoring | 15282230-45 (7) | 7 COMPLETED | 28.7-29.9 | 4 | 75-84% | 6.0G of 6G | 11-20T23:48 → 11-21T00:26 |
| ecshort4 seed 1 (own scorer) | navtest scoring | 15282246-63 (9) | 9 COMPLETED | 17.6-19.5 | 4 | 80-86% | 6.3G/6.4G/7.8G of 8000M | 11-20T23:48 → 11-21T00:29 |
| ecshort4 seed 1 (own scorer) | v1 comfort | 15282250 | 1 COMPLETED | 11.2 | 4 | 74% | 7.8G of 8000M | 11-21T00:08 → 11-21T00:19 |
| ecshort4 seed 1 (own scorer) | EC selection | 15282251-52 (2) | 2 COMPLETED | 7.5-19.6 | 8 | 71-83% | 16.0G of 16G | 11-20T23:48 → 11-21T00:08 |
| ecshort4 seed 1 (own scorer) | navhard scoring | 15282249-64 (7) | 7 COMPLETED | 29.9-33.1 | 4 | 79-83% | 6.0G of 6G | 11-20T23:48 → 11-21T00:30 |

### Imitation-only phase 3 from p1d_total: p3_imitation (submitted 2026-11-20 23:47)

**Question.** Is training on scores better than imitation alone, with the start checkpoint held fixed? The existing
imitation-only run p3_policy started from p1c_ed05, while p3_heads / p3_aux start from p1d_total ep 10. So its gap to
the score-trained models (navtest v2 0.775 vs 0.815, final_evaluation.md "Decoupled policy head") mixes the loss with
the start checkpoint.

**Run.** `p3_imitation`: the p3_heads recipe (`seed_aux_heads/submit_stage2.sh` COMMON, seed 0, from p1d_total ep 10)
with p3_policy's losses:
- CE 1, ED 1;
- sub-score losses 0;
- plain labels, no aux;
- `log_pdm_score_metrics=true` (proxy metrics only, loss unchanged);
- monitor `val/pdm_scores/argmax_score_gap`.

Batch 4 × accum 8 × 4 GPUs, 10 epochs, patience 3; 40 CPUs / 156G / 16 h, as the stage 2 seeds. Training 15282265,
then the offline val evaluation of every epoch, 15282266 (`eval_latest_run.sh`, afterany).

**Next.** Official evaluation of the val-argmax epoch: argmax on navtest v2, navtest v1 (+ v1 comfort) and navhard.

**Caveat.** p1d_total's phase 1 already had the sub-score losses (weight 1), and they also trained the planner queries
that the policy head shares. Phase 3 is therefore matched in everything but the loss, while phase 1 is shared. Script
`exp/seed_aux_heads/submit_p3_imitation.sh`, log `p3_imitation_2026-11-20.txt`.

**Status 2026-11-21.**
- Training 15282265 completed: 10 h 56 min on x1001c5s1b0n0, all 10 epochs (no early stop). Run folder
  `p3_imitation/2026.11.20.23.49.38`.
- Offline val evaluation 15282266 completed: 2 h 38 min, peak GPU memory 22.5 GB.
- Val per epoch (18,179 scenes; `checkpoint_eval/summary_p3_imitation.csv`):

| Epoch | val argmax | ADE | FDE | expected score | entropy | NC | DAC | EP | TTC | LK |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.6884 | 0.812 | 1.942 | 0.619 | 4.68 | 0.966 | 0.810 | 0.849 | 0.955 | 0.918 |
| 1 | 0.7103 | 0.810 | 1.944 | 0.646 | 4.54 | 0.969 | 0.832 | 0.846 | 0.959 | 0.921 |
| 2 | 0.7076 | 0.800 | 1.900 | 0.647 | 4.46 | 0.967 | 0.829 | 0.853 | 0.958 | 0.924 |
| 3 | 0.7062 | 0.800 | 1.915 | 0.645 | 4.40 | 0.969 | 0.828 | 0.849 | 0.958 | 0.918 |
| 4 | 0.7155 | 0.789 | 1.872 | 0.656 | 4.24 | 0.970 | 0.834 | 0.850 | 0.961 | 0.928 |
| 5 | 0.7124 | 0.787 | 1.864 | 0.658 | 4.13 | 0.970 | 0.832 | 0.849 | 0.961 | 0.928 |
| 6 | 0.7168 | 0.779 | 1.844 | 0.664 | 4.05 | 0.971 | 0.834 | 0.850 | 0.962 | 0.929 |
| **7** | **0.7207** | 0.776 | 1.840 | 0.664 | 3.97 | 0.969 | 0.839 | 0.850 | 0.961 | 0.926 |
| 8 | 0.7202 | 0.772 | 1.826 | 0.666 | 3.91 | 0.971 | 0.838 | 0.850 | 0.962 | 0.927 |
| 9 | 0.7194 | 0.772 | 1.828 | 0.666 | 3.87 | 0.969 | 0.838 | 0.850 | 0.962 | 0.928 |

- Chosen: epoch 7, val argmax 0.7207. Epochs 8-9 are within the SE (≈ 0.003). ADE / FDE keep improving to epoch 9.
- Same val metric for comparison:
  - p3_policy (imitation from p1c_ed05) ep 7: 0.7109;
  - p3_heads ep 5 argmax: 0.7430. Its policy was not trained in phase 3, since CE is 0 there; its heads selection
    (head_all) is 0.8182;
  - p3_aux ep 5: argmax 0.7472, head_all 0.8264.
- Official evaluation of epoch 7 submitted 2026-11-21 16:21:
  - GPU predictions 15303412 (`predictions/p3_imitation_ep07`);
  - navtest argmax (v1 + v2) 15303413;
  - navhard argmax 15303414;
  - v1 comfort 15303415.
- Status 2026-11-21 17:10: all 4 jobs completed (predict 19 min, navtest 18 min, navhard 34 min, v1 comfort 11 min).
  Results:
  - navtest v2 0.7852 (DAC 0.878, EC 0.811, 15.3% of scenes scored 0);
  - navtest v1 (v1 comfort) 0.8041;
  - navhard 0.189 (stage one 0.586 / two 0.322).

  The score-trained models from the same start are +0.019 to +0.073 better on navtest v2 (paired SE 0.003) and
  +0.045 to +0.153 on navhard. Details: final_evaluation.md, "Imitation only vs training on scores" (finding 75).

### Official evaluation of stage 1: p1d_total ep 10 (submitted 2026-11-21 00:24)

**Why.** p1d_total ep 10 is the start of every phase 3 model (p3_heads, p3_aux and their seeds; p3_imitation). It had
only the offline val proxy (checkpoint_evaluation.md: argmax 0.698, head_all 0.753). Section 6.3 of the thesis needs
one official row per training stage.

**Jobs.**
- GPU predictions 15282853 (`predictions/p1d_total_ep10`). Submitted directly with 16 CPUs / 64000M instead of the
  script's 40G, because the checkpoint is a full 9.4 GB file and earlier predict jobs with 1.2 GB checkpoints already
  peaked at 40G.
- CPU scoring with `PRED=p1d_total_ep10 AFTER=15282853 SELECTIONS="argmax score_heads score_heads_v1"`
  (`submit_vision_llm_lora_eval.sh`):
  - navtest argmax (v1 + v2): 15282854;
  - navtest score_heads (v2): 15282855;
  - navtest score_heads_v1 (v1): 15282856;
  - navhard score_heads: 15282857;
  - navhard argmax: 15282858.
- v1 comfort on argmax (15282859) and score_heads_v1 (15282860).

**Setup.** Same selections, k and pipeline as the stage 2 official evaluation. Log
`exp/seed_aux_heads/stage1_official_2026-11-21.txt`.

**Status 2026-11-21 01:18: all 8 jobs completed.**
- Durations: predict 16 min (peak 61.2 GiB of 64000M); navtest scoring 19 min, navhard 31 min, v1 comfort 16 min.
- Results: navtest v2 argmax 0.7687 / heads 0.7558; v1 (v1 comfort) argmax 0.7881 / heads_v1 0.8119; navhard argmax
  0.170 / heads 0.264.
- Details and the one-row-per-stage table: final_evaluation.md, "Stage 1 official evaluation" (findings 69-70).

### RL-3 with the GRPO objective (submitted 2026-11-17)

Question: does the sampled objective (GRPO) match or beat the exact objective once the LoRA trains (RL-3)? On the
policy head alone (RL-1), Dr. GRPO matched the exact objective and standard GRPO drifted (checkpoint_evaluation.md
findings 47-48). New flags (defaults = the exact RL-3 runs): `rl_objective` exact | grpo, `rl_grpo_group_size` (32),
`rl_grpo_normalize_std` (true = standard GRPO, false = Dr. GRPO); `vision_llm_lora_rl.grpo_loss`, tests in
`tests/test_vision_llm_lora_rl3.py`. Six runs: `rl3_grpo_b{0.05,0.02,0.005}` and `rl3_drgrpo_b{0.05,0.02,0.005}`, G 32,
otherwise exactly `rl3_b<beta>` (start p3_aux_policy ep 3, same reward = cached per-frame PDM total without EC, same
KL to the start policy, LRs, 2 epochs, seed 0, 4 GPUs). Evaluation of `epochs/epoch-epoch=01.ckpt` with the start
model's own sub-score heads (`SCORES_PRED=p3_aux_policy_ep03`): navtest argmax, shortlist / shortlist_v1 k 256,
mixture / mixture_v1 λ 10, and v1 PDMS with the real v1 comfort term (`v1_pdms_official_comfort.py`) for argmax,
shortlist_v1 and mixture_v1; navhard argmax and shortlist k 256. Submission script `exp/rl3_logs/submit_rl3_grpo.sh`,
job ids in `exp/rl3_logs/rl3_grpo_runs_2026-11-17.txt` (trainings 15175565 / 77 / 91 / 603 / 615 / 631).

Status 2026-11-18: all six trainings completed (2 h 21-24 min). End-of-training val (18,179 scenes):

| Run | val argmax | E_π[r] | argmax = 0 | KL | entropy | train argmax |
|---|---|---|---|---|---|---|
| `rl3_grpo_b0.05` | 0.7832 | 0.7755 | 17.0% | 7.97 | 2.27 | 0.901 |
| `rl3_grpo_b0.02` | 0.7701 | 0.7628 | 18.3% | 10.36 | 2.02 | 0.886 |
| `rl3_grpo_b0.005` | 0.7616 | 0.7571 | 19.2% | 12.73 | 1.75 | 0.876 |
| `rl3_drgrpo_b0.05` | 0.8460 | 0.8182 | 7.3% | 0.68 | 4.06 | 0.920 |
| `rl3_drgrpo_b0.02` | 0.8505 | 0.8313 | 7.1% | 1.17 | 3.71 | 0.924 |
| `rl3_drgrpo_b0.005` | 0.8579 | 0.8463 | 6.2% | 2.87 | 3.15 | 0.926 |

Standard GRPO ends below the start model on val (argmax 0.7888) with 6-12× the KL of the exact run at the same β; its
train-val gap is larger (0.11-0.12 vs 0.07). Dr. GRPO equals the exact runs (0.8466 / 0.8533 / 0.8548, KL 0.66 / 1.17 /
3.06). 12 of the 42 scoring jobs TIMEOUT (GRPO β 0.05 navtest mixture / mixture_v1 and both navhard, GRPO β 0.02 navtest
argmax / shortlist / shortlist_v1, Dr. GRPO β 0.05 navtest shortlist_v1 / mixture / mixture_v1 and both navhard), and 5
v1-comfort jobs wait on them; resubmitted (`exp/rl3_logs/resubmit_timeouts_2026-11-18.sh`), all completed. Official
results: final_evaluation.md section "RL-3 with the GRPO objective" (findings 53-55): standard GRPO fails (navtest
argmax 0.775-0.806 vs ~0.892, navhard shortlist ~0.35 vs ~0.41); Dr. GRPO equals the exact objective within the
seed spread. The exact objective stays the RL-3 objective.

### RL-3 with two-frame extended comfort in the reward (submitted 2026-11-14)

Motivation: EC-aware selection on the saved predictions (final_evaluation.md, "EC-aware selection") gave navtest v2
+0.0077 to +0.0107 and navhard +0.022 at almost no cost; the question is whether the policy can learn it.

Recipe of `rl3_b0.005` exactly (from p3_aux_policy ep 3, exact objective, β = 0.005, LoRA + vision LoRA + planner
queries + policy head, traj LR ×0.2, other losses 0, 2 epochs, batch 4 × accum 8, val every half epoch, monitor
`val/rl/shortlist_256_gap`), plus `rl_ec_weight` (new config field, default 0 = off):
- **Previous plan:** the current model's argmax at t − 0.5 s, recomputed in every training and validation step: a
  no-grad forward in eval mode on the past frame's camera tiles (history cache, `vision_llm_history_t1`) and past
  velocity / acceleration / driving command (`vision_llm_ego_history`) (`VisionLLMLoRAAgent._ec_against_previous_plan`,
  called from `compute_loss`, so evaluation is unaffected). On-policy: the plan follows the model as it trains;
  no gradient through it.
- **EC on the GPU** (`navsim/agents/vision_llm_lora/ec_torch.py`): all 16,384 entries at t against the previous plan,
  each simulated from its own frame's logged velocity / acceleration with a torch port of PDMSimulator (LQR tracker +
  kinematic bicycle) and the official EC features (Savitzky-Golay filters as matrices). The tracker reference of each
  entry is scene-independent and precomputed once with the official numpy code
  (`scripts/training/precompute_ec_tracking.py` → `exp/codebook/16384_8step_ec_tracking.npz`). Tests
  (`tests/test_vision_llm_lora_ec_torch.py`): simulated states match the official simulator within 1e-6, EC decisions
  are identical (864 checks); at real log timestamps ≥ 99% agree (the official code truncates `int(time_s * 1e6)`, which
  moves some samples by 1 µs depending on the timestamp). End-to-end check on navtest pairs: job 15046015
  (`scripts/evaluation/validate_ec_torch.py`, against the official EC of rl3_b0.005 argmax).
- **Reward:** M · (5 EP + 5 TTC + 2 LK + 2 HC + w · EC) / (14 + w) = (14 r + w M EC)/(14 + w), with r the cached
  unfiltered codebook total and M = NC · DAC · DDC · TLC (new `rl_multiplier` in the RL reference, set to 0 where r = 0).
  w = 2 is the official EPDMS weighting. Approximation: the stored sub-scores carry the human-penalty filter, r does
  not; where the filter changed a sub-score (~2.3% of entries) M can be off, so the EC bonus there is off by at most
  w/(14 + w) · 0.5.
- New val metrics: `val/rl/ec_argmax` (EC of the argmax vs the previous plan), `ec_expected`, `ec_any` (share of scenes
  with at least one consistent entry), `argmax_no_ec` (argmax under the old reward, comparable to rl3_b0.005).
- Cost: one extra forward pass (no grad) per step plus the EC simulation; `HISTORY_CACHE=1`.

| Run | Job | rl_ec_weight | Start |
|---|---|---|---|
| `rl3_ec2_b0.005` | ~~15046017~~ → **15046364** | 2 (official) | 2026-11-14 |
| `rl3_ec4_b0.005` | ~~15046018~~ → **15046365** | 4 (double) | 2026-11-14 |

Validation job 15046015 (4 min): tracking precompute 52 s; on 10,040 navtest pairs the torch EC of rl3_b0.005's argmax
outputs agrees with the official EC in **100.00%** of pairs (mean 0.7799 both). First submission 15046017/18 ran at
~0.6 it/s (rl3_b0.005: ~1.35 it/s), so 2 epochs would have exceeded the 5 h limit (the limit cannot be raised by the
user); cancelled after ~25 min and resubmitted with 7 h and EC simulated only for the entries with M > 0 (EC does not
change the reward of the others, ~72% of all entries; exact, tested): the EC val metrics therefore count EC only on
entries with M > 0.

Log: `exp/rl3_logs/rl3_ec_runs_2026-11-14.txt`. Planned evaluation: argmax, argmax + EC rule, shortlist k 256 (control
scorer) and shortlist + EC rule, on navtest and navhard, against rl3_b0.005 (0.8920 / 0.365 argmax; 0.9027 / 0.441
with the EC rule).

First val (half epoch, step 332, navtrain val): `argmax_no_ec` 0.8116 (w 2) / 0.8066 (w 4) vs rl3_b0.005's argmax
0.8255 at the same step; `ec_argmax` 0.715 / 0.738 (EC and M > 0; at most 1 − argmax_zero ≈ 0.905); KL 3.78 / 3.60
vs 3.08. At this point the EC runs pay in the other sub-scores without much EC gain yet.
Epoch 1 (step 664): `argmax_no_ec` 0.8289 / 0.8199 vs 0.8440 (−0.015 / −0.024); `ec_argmax` 0.735 / 0.765 (EC among
entries with M > 0: 0.80 / 0.84, start ~0.78); argmax under the EC reward 0.8167 / 0.8066; KL 3.13 / 3.25 vs 3.45. The
EC gain stays small and the cost in the other sub-scores stays. rl3_b0.005's EC on val is not measured, so its value
under the EC reward is an estimate: ≈ (14 · 0.8440 + 2 · ~0.73) / 16 ≈ 0.83 (UNVERIFIED), above both EC runs.
Epoch 1.5 (step 997): `argmax_no_ec` 0.8408 / 0.8330 vs 0.8505 (−0.010 / −0.018: the w 2 gap narrows); EC among
M > 0 entries 0.81 / 0.84; argmax under the EC reward 0.8291 / 0.8192 (reference estimate ≈ 0.836, UNVERIFIED).
**Final (epoch 2, step 1329; both `COMPLETED` in 6 h 20 min; best = last = epoch 1 checkpoint):** `argmax_no_ec`
0.8500 (w 2) / 0.8422 (w 4) vs 0.8548 (−0.005 / −0.013); EC among M > 0 entries 0.82 / 0.85 (start ~0.78); argmax
under the EC reward 0.8387 / 0.8291 vs the reference estimate ≈ 0.839 (UNVERIFIED): w 2 about level on the official
weighting, w 4 below. `shortlist_256` (EC reward) 0.8231 / 0.8070; KL 2.95 / 3.52. A C++ TCPStore trace at the end of
the w 2 log is from process teardown after `Trainer.fit` finished (harmless).
**Official evaluation (2026-11-14; final_evaluation.md "RL-3 with EC in the reward: official evaluation", findings 36-38):** without a test-time rule, w 2 gains navtest v2 +0.002 with every rule (argmax 0.8943, mixture 0.8983, shortlist 0.8859; EC 0.78 → 0.85, EP 0.892 → 0.879) and navhard argmax 0.365 → 0.376, but navhard shortlist 0.419 → 0.415; w 4 is worse. With the EC rule both are below rl3_b0.005 (navtest 0.8996 / 0.8943 vs 0.9027; navhard 0.439 / 0.438 vs 0.441): the rule gives the EC gain anyway, the EP loss stays.
Shortlist previous plan, first val (step 332): `argmax_no_ec` 0.8081 (w 2) / 0.8060 (w 4) vs 0.8255; `ec_argmax`
0.653 / 0.686 (EC of the argmax at t against the shortlist pick at t − 0.5 s, so lower than with the argmax previous
plan); `shortlist_256` (EC reward) 0.8151 / 0.7994 vs 0.8102 / 0.7905 for the argmax-previous runs at the same step.
Epoch 1 (step 664): `argmax_no_ec` 0.8346 / 0.8254 vs 0.8440 (argmax-previous runs 0.8289 / 0.8199); epoch 1.5 (step
997): 0.8432 / 0.8370 vs 0.8505 (0.8408 / 0.8330).
**Final (epoch 2, step 1329; 15049985 / 15049986 `COMPLETED` in 6 h 22 min / 6 h 20 min; best = last = epoch 1
checkpoint):**

| Run (w) | `argmax_no_ec` | argmax (EC reward) | `ec_argmax` (EC among M > 0) | `shortlist_256` (EC reward) | KL |
|---|---|---|---|---|---|
| rl3_b0.005 (reference, no EC) | 0.8548 | – | – | – | – |
| rl3_ec2 (argmax previous) | 0.8500 | 0.8387 | 0.763 (0.82) | 0.8231 | 2.95 |
| **rl3_ecshort2** (shortlist previous) | **0.8517** | 0.8327 | 0.701 (0.75) | **0.8263** | 3.05 |
| rl3_ec4 (argmax previous) | 0.8422 | 0.8291 | 0.787 (0.85) | 0.8070 | 3.52 |
| rl3_ecshort4 (shortlist previous) | 0.8454 | 0.8182 | 0.725 (0.78) | 0.8108 | 3.48 |

`argmax_no_ec` (old reward, no EC) is directly comparable: the shortlist-previous runs lose less against the reference
(−0.003 / −0.009 vs −0.005 / −0.013). EC and the EC-reward columns are measured against each run's own previous plan
(argmax vs shortlist pick at t − 0.5 s), so they compare only within a previous-plan type: EC against the shortlist pick
is lower because that pick is often outside the policy's mode. `shortlist_256` uses the same start-model shortlist for
all runs, but its EC term also uses the run's own previous plan; it is the closest val proxy to the target
(shortlist + EC rule on navhard): +0.003 / +0.004 over the argmax-previous runs.
Official evaluation submitted 2026-11-14 21:22 (same chain as rl3_ec2/ec4, scratchpad `submit_rl3_ec_eval.sh`): predict
15058967 / 15058984, scoring 15058968-73 / 15058985-90, EC rule selection 15058974-75 / 15058991-92, EC rule scoring
15058976-83 / 15058993-15059000 (log `exp/rl3_logs/rl3_ec_runs_2026-11-14.txt`). Shortlist k 64 and navhard mixture
not run: on rl3_b0.005 they are well below shortlist k 256 on navhard (0.396 / 0.395 vs 0.419).
EC rule on the mixture base (λ 10; new `--bases mixture` of `ec_aware_selection.py`, default unchanged) for rl3_b0.005,
rl3_ec2/ec4 and rl3_ecshort2/4, navtest sequential + navhard: submitted 2026-11-14 21:38 (job IDs in
final_evaluation.md and the log). All completed.
**Official evaluation of the shortlist-previous runs and the mixture + EC rule (final_evaluation.md "Shortlist previous
plan and the EC rule on the mixture base", findings 39-41):** mixture + EC rule n 64 is the best navtest rule: rl3_b0.005
v2 **0.9079** (+0.0052 ± 0.0012 over argmax + rule 0.9027, same model; the mixture and rule gains add). ecshort w 4 with
shortlist + EC rule n 64 reaches navhard **0.456** (w 2 0.447; rl3_b0.005 0.441, ec2/ec4 0.439 / 0.438), but the
gain is in stage two (0.570 vs 0.549; stage one level): its stage-one plans end where the follow-up synthetic scenes go
better for either model, while both drive a given synthetic scene about equally well (swap of driving vs endpoint weights: 0.001-0.006 vs 0.015-0.022; finding 40); combined +0.015 ± 0.013 (bootstrap over scene groups), one seed per run.
Submission pickle: `exp/submission_navhard_two_stage/rl3_ecshort4_b0.005_ep01_ec_short256_n64/submission.pkl`. On navtest the ecshort runs behave
like the argmax-previous runs.
**Seed replicates (submitted 2026-11-15 00:31):** the navhard 0.456 vs 0.441 comparison rests on one training each, so
both runs are repeated with `++seed=1` (recipes otherwise identical; Hydra overrides diffed against the seed-0 runs):
`rl3_b0.005_s1` (job 15065648, 8 CPUs / 32000M / 4 h) and `rl3_ecshort4_b0.005_s1` (job 15065649, 7 h). Full official
evaluation queued behind each (predict after the training via `exp/rl3_logs/predict_latest_run.sh`, which picks the
run's date folder at start; scoring argmax / mixture / mixture_v1 / shortlist k 256, EC rule on argmax, short256 and
mix10 bases, n 16 / 64): rl3_b0.005_s1 predict 15065650, jobs 15065651-70; rl3_ecshort4_b0.005_s1 predict 15065671,
jobs 15065672-91. Log `exp/rl3_logs/rl3_seed_runs_2026-11-15.txt`; scratchpad `submit_seed_runs.sh`.
Both trainings `COMPLETED` (2 h 26 min / 6 h 26 min; `Global Seed set to 1` on all ranks; run folders
`rl3_b0.005_s1/2026.11.15.01.02.32`, `rl3_ecshort4_b0.005_s1/2026.11.15.01.02.34`). The first predict jobs failed in 2 s:
each DDP rank creates its own date folder a second apart and the wrapper took the newest (rank ≠ 0, no checkpoints);
fixed (it now takes the folder holding the checkpoint), the 40 stuck jobs cancelled and the chain resubmitted
2026-11-15 11:02: predict 15081432 / 15081453, jobs 15081433-52 / 15081454-73.
Validation (navtrain val, final step 1329), seed 0 → seed 1: rl3_b0.005 argmax 0.8548 → 0.8560, `shortlist_256` 0.8506
→ 0.8490, KL 3.06 → 2.84; rl3_ecshort4 `argmax_no_ec` 0.8454 → 0.8454, `ec_argmax` 0.725 → 0.725, `shortlist_256`
0.8108 → 0.8103, KL 3.48 → 3.53. On val the seed moves the final metrics by ≤ 0.002 (mid-training up to 0.016: rl3
argmax at step 664 0.8440 vs 0.8285).
**Official results (final_evaluation.md "Seed replicates", findings 42-45):** navtest reproduces within the paired SE
(rl3_b0.005 seed 1: argmax + EC rule 0.9026 vs 0.9027, mixture + EC rule 0.9085 vs 0.9079; ecshort w 4 below the
reference with every EC rule in both seeds). navhard does not: shortlist + EC rule n 64 rl3_b0.005 0.441 → 0.454,
ecshort w 4 0.456 → 0.444, so the same recipe moves ±0.012-0.013 between seeds and the ecshort gain (finding 40)
reverses (two-seed mean +0.002 ± 0.010). Conclusion: EC in the reward (either previous plan) gives no measurable
navhard gain and costs navtest when the EC rule is used; the thesis bests are rule results on rl3_b0.005 (navtest
0.908, navhard 0.449 ± ~0.01 over four runs).

**Previous plan from the shortlist (submitted 2026-11-14).** At a shortlist evaluation the plan output at t − 0.5 s is
the policy's best entry inside the scorer's top 256, not its argmax; on rl3_b0.005's saved predictions (control scorer,
k 256) the two differ in 55.8% of navtest and 67.7% of navhard scenes (argmax outside the shortlist 45.7% / 64.8%; where
they differ, endpoint distance median 1.40 m / 2.62 m). `rl_ec_previous_plan=shortlist` (config; default `argmax`)
takes the previous plan as that pick: the policy's argmax over `rl_previous_shortlist[:256]`, the control scorer's
top entries at t − 0.5 s by predicted total (not the evaluator's scores), ties in shortlist order as `select_trajectory`.
- Scorer cache: `scripts/training/cache_scorer_shortlists.py` (control scorer p3_aux_ctrl best ep 2 on frames t,
  t − 0.5, t − 1.0 s of every navtrain scene, past frames from the history cache; top 1024 per frame) →
  `exp/scorer_shortlists/p3_aux_ctrl_ep02/` (6 shards, jobs 15048738-43, `MODE=shortlist` of
  `run_rl_policy_head_slurm.sh`; smoke 15048720: 128 scenes, top-256 overlap with the p3_aux_policy heads at t 77%,
  between t and t − 0.5 s 55%). Log `exp/rl3_logs/scorer_shortlists_2026-11-14.txt`.
- RL cache: `cache_rl_reference.py --previous-shortlists` (training script `RL_PREVIOUS_SHORTLISTS=p3_aux_ctrl_ep02`)
  adds `rl_previous_shortlist` (256 int16) per scene. `rl_shortlist` (the val monitor) stays the start model's heads,
  so `shortlist_256` is comparable with rl3_b0.005 and the argmax EC runs.

| Run | Job | rl_ec_weight | Previous plan |
|---|---|---|---|
| `rl3_ecshort2_b0.005` | 15049985 (after the scorer cache; `COMPLETED`) | 2 | policy's best in the control scorer's top 256 at t − 0.5 s |
| `rl3_ecshort4_b0.005` | 15049986 (after the scorer cache; `COMPLETED`) | 4 | same |

### Storage cleanup (2026-11-10)

Per-epoch checkpoints of the 15 phase-1 runs from 2026-10-27 to 2026-11-02 (saved before checkpoints held only the
trainable parameters, ~9 GB each) were deleted: 2.75 TB. Kept: every `checkpoints/best/` and the epoch files other runs
start from (`p1c_ed05`, `p1d_scorelabels`, `p1d_total` `epochs/epoch-epoch=10.ckpt`). Per-epoch offline evaluations of
those runs (checkpoint_evaluation.md) can no longer be re-run except for the kept epochs.
