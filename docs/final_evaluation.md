# Final evaluation (official NAVSIM metrics)

Official PDM scores of the thesis models, from raw camera images through the NAVSIM devkit scripts. The model-selection
proxy (navtrain val, precomputed codebook scores) is in [checkpoint_evaluation.md](checkpoint_evaluation.md); runs and
their settings are in [experiments.md](experiments.md). Last updated 2026-11-18: v1 PDMS with the v1 comfort term in the main
tables (findings 51-52), RL-3 with GRPO / Dr. GRPO (53-55), 6 seeds of rl3_b0.005 (56-58).

**Current bests (2026-11-14), all from RL-3 β = 0.005 (p3_aux_policy ep 3 + RL-3, some with EC in the reward), human filter on:**

| Benchmark | Without EC-aware selection | With EC-aware selection (section "EC-aware selection") |
|---|---|---|
| navhard (leaderboard metric) | 0.419 (shortlist k 256, control scorer; leaderboard **41.94**); 0.425 with RL-3 + EC reward w 4, shortlist previous plan | 0.441 (same + EC, top 64; stage one 0.795 / two 0.549; not yet submitted); **0.456** with RL-3 + EC reward w 4, shortlist previous plan (stage one 0.792 / two 0.570; pickle ready); seed replicates: rl3_b0.005 0.454, ecshort w 4 0.444, so the level is 0.449 ± ~0.01 and the EC-reward gain did not replicate (finding 42); with the own heads, 6 seeds of rl3_b0.005: **0.438 ± 0.011** (finding 57) |
| navtest v2 EPDMS | 0.8983 (RL-3 + EC reward w 2, control scorer, mixture λ 10; 0.8965 without the EC reward) | **0.9079** (mixture λ 10 + EC, top 64, sequential; EC 0.917; finding 39); seed 1: 0.9085 (finding 43); own heads, 6 seeds: 0.9056 ± 0.0014 (finding 57) |
| navtest v1 PDMS (v1 comfort term) | 0.9193 (mixture_v1 λ 10, pairwise scorer; control 0.9189, own heads 0.9178; argmax 0.9143); own heads, 6 seeds: 0.9169 ± 0.0010 | – (EC is not in v1; the rule costs v1 −0.0004 to −0.002) |

Seed-averaged level of the thesis recipe (rl3_b0.005, 6 trainings, own heads; findings 56-57): navtest v2 0.9056 ±
0.0014 (mixture + EC n 64), navtest v1 0.9169 ± 0.0010 (mixture_v1), navhard 0.438 ± 0.011 (shortlist + EC n 64).
Objective: the exact RL objective; Dr. GRPO gives the same results, standard GRPO fails (findings 53-55).

The EC-aware rule needs the plan of the previous frame. The numbers here take it from the saved prediction of the
previous test scene; a leaderboard agent has to recompute it from the past frames in its own input. The models with
past inputs (section "Models with past inputs") do not beat these.

**Best models since 2026-11-10: p3_aux_policy ep 3 + RL-3 policy, sub-scores from a p3_aux scorer fine-tuned 5 more
epochs** (sections "RL-3", "Official eval with the pairwise scorer", "Control scorer"): navtest v1 **0.9193** (v1
comfort term, sections "v1 PDMS with the v1 comfort term"; β = 0.005, mixture_v1 λ = 10, pairwise scorer; control
scorer 0.9189, own heads 0.9178; HC version 0.9180), navtest v2 EPDMS **0.8978** (β = 0.02, mixture
λ = 10, pairwise scorer; +0.0025 ± 0.0008 over the control scorer), navhard **0.419** (β = 0.005, shortlist k = 256,
control scorer; a test-informed rule; own heads 0.412-0.413, pairwise scorer 0.382). Previous (2026-11-09, own heads):
v1 0.9165 (HC version; 0.9178 with v1 comfort), v2 0.8961, navhard 0.413. **Comparability (section "Calibration"):** navtest v2 here uses the fixed human
filter; papers report filter-off numbers, where our best is 0.864 (human 0.903). navhard filter-on = leaderboard scoring.

**Best model: p3_aux_policy ep 3 + RL-1 (exact β = 0.02, epoch 12) with `shortlist` k = 256** (p3_aux's LoRA and
sub-score heads, a policy head trained on top by imitation and then by RL on the codebook scores, everything else
frozen): v1 PDMS navtest **0.899**, v2 EPDMS navtest **0.872**, v2 EPDMS navhard **0.383** (stage one 0.741 / stage two
0.509). RL adds +0.007 (v1) / +0.005 (v2) on navtest over p3_aux_policy (0.892 / 0.867, significant) and nothing on
navhard (0.383 vs 0.386, noise). All numbers from the re-evaluation of 2026-11-08 (section "Results"); the earlier
combined runs (batch 1) agree within ±0.002 on navtest.

## Metrics

| Metric | Split, script | Traffic | Formula |
|---|---|---|---|
| **EPDMS (one stage)** | navtest, 12,146 scenes, `run_pdm_score_one_stage.py` | log replay (`traffic_agents=non_reactive`) | NC · DAC · DDC · TLC · (5 EP + 5 TTC + 2 LK + 2 HC + 2 EC) / 16; EC drops out of the weights in scenes without an adjacent frame (17% of navtest) |
| **v1 PDMS** | a navtest run (one stage), NC / DAC / EP / TTC from its columns, comfort re-simulated | log replay | NC · DAC · (5 EP + 5 TTC + 2 C) / 12 with the v1 comfort term C (since 2026-11-18, section "v1 PDMS with the v1 comfort term for every model"; before: v2 history comfort HC in place of C) |
| **EPDMS (two stage)** | navhard_two_stage, 450 real + 5,462 synthetic scenes, `run_pdm_score.py` | reactive IDM in both stages (always) | per scene group, stage one score × weighted stage two score, averaged |

### Formulas (per scene, then averaged over scenes)

**v2 EPDMS** (this devkit: `pdm_scorer.py`, weights from `config/pdm_scoring/scorer/pdm_scorer.yaml`; multipliers
`MultiMetricIndex` in `pdm_enums.py`; EC added per scene in `run_pdm_score_one_stage.py:compute_final_scores`):

    EPDMS = NC · DAC · DDC · TLC · (5·EP + 5·TTC + 2·LK + 2·HC + 2·EC) / 16
            with EC (and its weight) dropped, i.e. / 14, in scenes without an adjacent frame

**v1 PDMS** (NAVSIM v1 paper; the v1 code is not in this repo, so its exact implementation is UNVERIFIED):

    PDMS = NC · DAC · (5·EP + 5·TTC + 2·C) / 12

Differences: v1 has no lane keeping, no two-frame comfort, and no driving-direction or traffic-light multiplier;
its comfort C is checked on the predicted trajectory, while v2's history comfort HC also covers the past frames. Here
v1 is recomputed from the v2 columns with HC in place of C. Multipliers are 0 or 1 (NC is 0.5 for an at-fault collision
with a static object, DDC 0.5 for a short wrong-way stretch). EP is the ego's progress divided by the best progress
among the safe trajectories of the scene (itself and the PDM-Closed reference); 1 if that best progress is below 5 m.

NC no at-fault collision, DAC drivable area, DDC driving direction, TLC traffic light, EP ego progress, TTC time to
collision, LK lane keeping, HC history comfort, EC two-frame extended comfort (the plan at t vs the plan at t − 0.5 s,
on their 3.5 s overlap: RMS differences of acceleration ≤ 0.7 m/s², jerk ≤ 0.5 m/s³, yaw rate ≤ 0.1 rad/s, yaw
acceleration ≤ 0.1 rad/s²).

Caveats and decisions:
- **Human-penalty filter: all results below use upstream's fixed filter** (decision 2026-11-05). NAVSIM scores the
  human trajectory too; every metric the human fails is set to 1 for the model. This devkit originally applied that
  only to the reported columns, not to the score (pre-fix, "EPDMS*" in papers). Upstream fixed it on 2025-09-30
  (commit 359c7f72, "Fix human filter bug (Issue #151)"); the fix is ported into `navsim/evaluate/pdm_score.py` since
  2026-11-05 (tested on navtest scenes where the human runs a red light: the multiplier follows the column, 0 → 1).
  The filter acts on real scenes only (navtest; navhard stage one). It raises scores by ~+0.03 (navtest) and ~+0.02
  (navhard combined). Pre-fix numbers are kept at the end of this file.
- **v1 PDMS** is computed from the same filter-on (fixed) columns. Until 2026-11-17 it used v2 history comfort as the
  comfort term; the main tables now use the v1 comfort term (findings 51-52: +0.002 for policy picks, −0.010 for
  score-head picks); sections that still show the HC version say so. The v1
  devkit is not in this repo, so how close this is to the original v1 implementation (filter, comfort) is UNVERIFIED.
  v1 is higher than v2 for every model because v1 has fewer ways to lose points (no LK, EC, DDC, TLC).
- **Selection** (`trajectory_selection`): `argmax` = the policy's most probable codebook entry; `score_heads` = the
  entry with the best total predicted by the 8 sub-score heads, combined with the v2 formula without EC (used for v2);
  `score_heads_v1` = the same heads combined with the v1 formula (used for v1). With argmax one run gives both v1 and v2.
- Published reference numbers of other methods: not collected here; see the papers (navhard: EPDMS vs EPDMS*).

## RL-3: LoRA, vision LoRA and policy head trained by RL (2026-11-09)

Runs and val proxy: [experiments.md](experiments.md), "RL-3 / RL-4". Best val checkpoint of each 2-epoch run
(`checkpoints/best/best-epoch=01.ckpt`, = end of training), split pipeline (batch 64, bf16). The RL model has no
sub-score heads (`pdm_score_loss_weight` 0), so `shortlist` selections take the start model's saved sub-scores
(`SCORES_PRED` = p3_aux_policy_ep03 / p3_heads_policy_ep02), as the two-adapter model would. v1 = the `argmax` job's
v1 (selection does not depend on the weights) or the `shortlist_v1` job's; v2 = the `argmax` / `shortlist` job.
0 failed scenes everywhere. Jobs: GPU 14933665/77/83/89 and 14933701 (14933671 cancelled: node cn04 fed 3.9 scenes/s),
CPU 14933666-14933694 (list in `exp/rl3_logs/rl3_runs_2026-11-08.txt`).

| Start model | β | Selection | v1 PDMS navtest (v1 comfort) | v2 EPDMS navtest | v2 EC | navhard | navhard stage 1 / 2 |
|---|---|---|---|---|---|---|---|
| p3_aux_policy ep 3 (reference, re-evaluation) | – | argmax | 0.863 | 0.843 | 0.815 | 0.254 | 0.664 / 0.374 |
| | – | shortlist k = 256 | 0.892 | 0.867 | 0.756 | 0.386 | 0.737 / 0.517 |
| + RL-1 (reference) | 0.02 | shortlist k = 256 | 0.899 | 0.872 | 0.742 | 0.383 | 0.741 / 0.509 |
| + **RL-3** | 0.005 | argmax | **0.914** | 0.892 | 0.780 | 0.365 | 0.760 / 0.473 |
| | 0.005 | shortlist k = 256 | 0.911 | 0.883 | 0.730 | 0.412 | 0.764 / 0.533 |
| | 0.02 | argmax | **0.914** | **0.895** | 0.821 | 0.335 | 0.770 / 0.441 |
| | 0.02 | shortlist k = 256 | 0.909 | 0.881 | 0.752 | **0.413** | 0.760 / 0.532 |
| | 0.05 | argmax | 0.911 | 0.893 | 0.829 | 0.351 | 0.767 / 0.444 |
| | 0.05 | shortlist k = 256 | 0.906 | 0.880 | 0.756 | 0.405 | 0.746 / 0.532 |
| p3_heads_policy ep 2 (reference) | – | shortlist k = 64 | 0.885 | 0.853 | 0.668 | 0.359 | 0.722 / 0.483 |
| + **RL-3** | 0.005 | argmax | 0.902 | 0.878 | 0.775 | 0.359 | 0.756 / 0.470 |
| | 0.005 | shortlist k = 256 | 0.900 | 0.871 | 0.723 | 0.395 | 0.750 / 0.517 |
| | 0.02 | argmax | 0.903 | 0.882 | 0.815 | 0.316 | 0.736 / 0.433 |
| | 0.02 | shortlist k = 256 | 0.899 | 0.871 | 0.747 | 0.363 | 0.738 / 0.485 |
| | 0.05 | argmax | 0.901 | 0.881 | 0.826 | 0.308 | 0.748 / 0.418 |
| | 0.05 | shortlist k = 256 | 0.898 | 0.871 | 0.747 | 0.362 | 0.738 / 0.474 |

Findings (no paired tests yet):
1. RL-3 is far above RL-1: navtest v2 0.895 vs 0.872 (+0.023), v1 0.914 vs 0.899 (v1 comfort), navhard 0.413 vs 0.383.
2. navtest: the RL policy alone (`argmax`) beats every shortlist selection on v2 (+0.009 to +0.014), mostly through
   extended comfort (EC 0.78-0.83 vs 0.73-0.76): consecutive argmax picks are consistent, the scorer's re-ranking among
   near-ties is not. On v1 (no EC) argmax is also above shortlist_v1: +0.003 / +0.005 for β 0.005 / 0.02 (2.4 / 3.6 SE)
   with the v1 comfort term (finding 52; with HC +0.002 / +0.003).
3. navhard: the scorer is still needed. `shortlist` k = 256 beats `argmax` by +0.036 to +0.078, all in stage two
   (0.53 vs 0.44-0.47); stage one is equal. The best selection therefore depends on the benchmark.
4. β: on navtest v2 the order is 0.02 > 0.05 > 0.005 for argmax (EC falls with lower β), although 0.005 was val-best;
   v1 and navhard do not separate 0.005 and 0.02. p3_heads_policy follows the same pattern ~0.01-0.02 lower.

### Selection sweep on RL-3 β = 0.02 (2026-11-09)

Same saved predictions (`predictions/rl3_b0.02_ep01`, sub-scores from p3_aux_policy_ep03), CPU jobs 14934130-14934141,
0 failed scenes. Softer scorer filters than `shortlist` k = 256: larger k, or `mixture` (argmax of log p +
λ · log predicted total). navtest = v2 EPDMS only (no `_v1` runs).

| Selection | navtest v2 | navtest EC | navhard | navhard stage 1 / 2 |
|---|---|---|---|---|
| argmax (no filter) | 0.8950 | 0.821 | 0.335 | 0.770 / 0.441 |
| mixture λ = 1 | 0.8960 | 0.819 | 0.346 | 0.767 / 0.450 |
| mixture λ = 3 | **0.8961** | 0.818 | 0.348 | 0.768 / 0.449 |
| mixture λ = 10 | 0.8944 | 0.812 | 0.370 | 0.771 / 0.477 |
| shortlist k = 2048 | 0.8932 | 0.808 | 0.351 | 0.758 / 0.460 |
| shortlist k = 1024 | 0.8904 | 0.790 | 0.377 | 0.760 / 0.489 |
| shortlist k = 512 | 0.8871 | 0.778 | 0.397 | 0.761 / 0.510 |
| shortlist k = 256 | 0.8812 | 0.752 | **0.413** | 0.760 / 0.532 |

5. A clean trade-off, no setting wins both: the harder the scorer filters, the better navhard and the worse navtest
   (mostly through EC). mixture λ = 10 is the best compromise (0.894 / 0.370) and beats k = 2048 on both.
6. Stage one barely moves (0.758-0.771) under any selection; the whole navhard difference is in stage two (the
   synthetic scenes from shifted ego poses). The scorer pays off where the policy is out of distribution and costs
   comfort on familiar scenes. Candidates to get both: a guarded rule (argmax unless the scorer rates it clearly below
   the shortlist's best), stricter weights on the multiplicative sub-scores in the selection, test-time search (TOAD).

### Local search around the policy's choice (discrete TOAD; 2026-11-09)

`local` / `local_v1` (select_trajectory): among the N codebook entries nearest to the RL policy's argmax (ADE over
x, y; median spacing 0.25 m, 64th neighbour at 0.71 m), the one with the best predicted total − λ · ADE. Same saved
predictions and scorer as above; CPU jobs 14935522-14935545 (14935530 timed out at 40 min, rerun as 14935739). This is
a test-set grid (sensitivity analysis), not a tuned result. Reference argmax: v1 0.9138 (v1 comfort), v2 0.8950, navhard 0.335.

| N | λ | moved (navtest / navhard, 1000 scenes) | v1 navtest (v1 comfort) | v2 navtest | v2 EC | navhard | stage 1 / 2 |
|---|---|---|---|---|---|---|---|
| 64 | 0.02 | 68% / 85% | 0.8992 | 0.8702 | 0.768 | 0.331 | 0.718 / 0.458 |
| 64 | 0.05 | 42% / 65% | 0.9033 | 0.8771 | 0.788 | 0.331 | 0.733 / 0.452 |
| 64 | 0.1 | 21% / 42% | 0.9073 | 0.8837 | 0.804 | 0.332 | 0.743 / 0.454 |
| 64 | 0.2 | 9% / 20% | 0.9102 | 0.8892 | 0.813 | 0.329 | 0.759 / 0.435 |
| 128 | 0.02 | 69% / 87% | 0.8986 | 0.8678 | 0.757 | 0.330 | 0.721 / 0.455 |
| 128 | 0.05 | 43% / 66% | 0.9024 | 0.8743 | 0.780 | 0.329 | 0.728 / 0.451 |
| 128 | 0.1 | 22% / 44% | 0.9064 | 0.8831 | 0.801 | 0.334 | 0.747 / 0.453 |
| 128 | 0.2 | 9% / 21% | 0.9102 | 0.8889 | 0.812 | 0.333 | 0.758 / 0.440 |

7. Local search is worse than argmax everywhere and approaches it as λ grows (fewer moves); N does not matter.
   navtest sub-scores (N 64, λ 0.02 vs argmax): DAC 0.980 → 0.970, NC 0.981 → 0.976, LK 0.957 → 0.939, EC 0.821 →
   0.768, zero-score scenes 4.1% → 5.7%. The scorer's preferred neighbour is less safe than the policy's entry.
8. The scorer's logits vary strongly among 64 neighbours (std: DAC 1.6, HC 1.2, DDC 0.9) and the per-entry
   `score_bias` is negligible there (0.02-0.05). Correction (2026-11-10): this is not chance-level ordering. On
   neighbour pairs near the ground truth whose labels differ, a scorer trained with the pairwise loss orders 91-92%
   correctly (p3_aux_pair, val `pair_acc`; p3_aux's own value not measured yet). LIKELY mechanism (unverified): most
   neighbours of the policy's entry have nearly equal true scores, so the max over N noisy predictions picks the
   neighbour whose score is most overestimated (winner's curse), i.e. close to a random neighbour, and the policy's
   own entry is better than a random neighbour. The navtrain-val table below shows the same ordering (local worse than
   argmax at every N and λ). TOAD's continuous search maximises the same noisy estimate over more candidates.

### Selection rules chosen on navtrain val (held out; 2026-11-10)

The grids above were run on the test benchmarks. To choose a rule without them: model outputs on the 18,179 navtrain
val scenes from the training feature cache (`scripts/evaluation/predict_vision_llm_lora_cache.py`, jobs 14935879/80),
every rule applied with the same `select_trajectory` and graded by lookup in the codebook score table
(`scripts/evaluation/eval_selection_rules_val.py`, job 14935903; full table `exp/selection_val/rl3_b0.02_ep01.csv`).
Grade: per-frame EPDMS without EC and without human filter (v2 rules) or v1 PDMS from the same sub-scores (_v1 rules).
RL-3 β = 0.02 policy, p3_aux_policy sub-scores; ± = SE of the paired difference.

| Rule (v2 grade) | val | vs argmax | | Rule (v1 grade) | val | vs argmax |
|---|---|---|---|---|---|---|
| argmax | 0.8532 | | | argmax | 0.8946 | |
| shortlist k = 256 | 0.8471 | −0.0061 ± 0.0015 | | shortlist_v1 k = 256 | 0.8919 | −0.0027 ± 0.0014 |
| shortlist k = 2048 | 0.8540 | +0.0008 ± 0.0009 | | shortlist_v1 k = 2048 | 0.8958 | +0.0012 ± 0.0008 |
| **mixture λ = 3** | **0.8555** | **+0.0024 ± 0.0008** | | mixture_v1 λ = 3 | 0.8973 | +0.0027 ± 0.0007 |
| mixture λ = 10 | 0.8539 | +0.0007 ± 0.0011 | | **mixture_v1 λ = 10** | **0.8983** | **+0.0037 ± 0.0010** |
| local N 64, λ 0.2 (best local) | 0.8479 | −0.0053 ± 0.0009 | | local_v1 N 64, λ 0.2 | 0.8919 | −0.0027 ± 0.0008 |

9. Val picks mixture λ = 3 (v2) and mixture_v1 λ = 10 (v1), small but significant gains over argmax (~3 SE). On navtest
   the val-chosen v2 rule is the best v2 number measured (mixture λ = 3: 0.8961 vs argmax 0.8950); mixture_v1 λ = 10
   gives navtest v1 0.9131 for β = 0.02 (0.9165 for β = 0.005, next section). val cannot see navhard's stage two, where every rule val prefers (close to argmax) loses to the
   hard shortlist (0.348 vs 0.413), so for navhard the shortlist k remains a test-informed choice; k = 256 over k = 64
   is supported by RL-3's training-time val (shortlist_256 0.8466 vs shortlist_64 0.8407).

### Val-chosen rules on every RL-3 model (2026-11-10)

The two rules chosen on the p3_aux family's val (mixture λ = 3 for v2, mixture_v1 λ = 10 for v1) applied unchanged to
all six RL-3 models (CPU jobs 14944779-94; rl3_b0.02 mixture λ = 3 from 14934138/39; rl3_heads_b0.05 official runs
GPU 14944773, CPU 14944774-78). 0 failed scenes.

| Start model | β | navtest v1 (mixture_v1 λ 10, v1 comfort) | navtest v2 (mixture λ 3) | navhard (mixture λ 3) | for reference: navhard shortlist k = 256 |
|---|---|---|---|---|---|
| p3_aux_policy | 0.005 | **0.9178** | 0.8942 | **0.374** (0.767 / 0.484) | 0.412 |
| | 0.02 | 0.9151 | **0.8961** | 0.348 (0.768 / 0.449) | **0.413** |
| | 0.05 | 0.9133 | 0.8952 | 0.350 (0.751 / 0.450) | 0.405 |
| p3_heads_policy | 0.005 | 0.9058 | 0.8805 | 0.372 (0.754 / 0.489) | 0.395 |
| | 0.02 | 0.9049 | 0.8842 | 0.333 (0.742 / 0.449) | 0.363 |
| | 0.05 | 0.9021 | 0.8815 | 0.325 (0.743 / 0.432) | 0.362 |

10. With validation-chosen rules the best numbers are RL-3 p3_aux_policy β = 0.005 for navtest v1 (**0.9178** with v1
    comfort, HC version 0.9165, new best; argmax 0.9143) and β = 0.02 for navtest v2 (0.8961). On navhard the lower β keeps more of the shortlist's
    benefit under mixture (0.374 vs 0.348), but the hard shortlist stays clearly better (0.412-0.413).
11. The heads family's own val (`selection_val/rl3_heads_b0.02_ep01.csv`, jobs 14944795-97) agrees with the borrowed
    choice: mixture beats argmax (λ 3: +0.0026 ± 0.0009, λ 10: +0.0030 ± 0.0012, a tie), mixture_v1 λ 10 is the best v1
    rule (+0.0055), local search is worse than argmax.

### Scorer swap on val: pairwise loss vs control (2026-11-10)

RL-3 β = 0.02 policy combined with three scorers on navtrain val (`selection_val/rl3_b0.02_ep01[__scores_<scorer>].csv`,
jobs 14935903, 14944573, 14944575; scorer val predictions 14935880, 14944572, 14944574). vs argmax (0.8532 v2,
0.8946 v1), ± SE of the paired difference.

| Rule | p3_aux_policy scorer (current) | p3_aux_ctrl ep 2 (+5 epochs) | **p3_aux_pair ep 2 (+ pairwise loss)** |
|---|---|---|---|
| score_heads (heads alone) | −0.0267 | −0.0256 | −0.0296 |
| shortlist k = 64 | −0.0120 | −0.0120 | −0.0060 |
| shortlist k = 256 | −0.0061 ± 0.0015 | −0.0032 ± 0.0015 | **+0.0012 ± 0.0014** |
| shortlist k = 2048 | +0.0008 | +0.0018 | +0.0031 |
| mixture λ = 3 | +0.0024 ± 0.0008 | +0.0035 ± 0.0008 | +0.0049 ± 0.0008 |
| mixture λ = 10 | +0.0007 ± 0.0011 | +0.0030 ± 0.0011 | **+0.0052 ± 0.0010** |
| mixture_v1 λ = 10 (v1) | +0.0037 ± 0.0010 | +0.0050 ± 0.0010 | **+0.0073 ± 0.0010** |
| local N 64, λ 0.2 | −0.0053 ± 0.0009 | −0.0033 ± 0.0009 | **+0.0013 ± 0.0008** |

12. The pairwise loss makes the scorer a better partner for the policy although its own head_score is lower: every
    combined rule improves over the control (+0.0014 to +0.0046) and over the current scorer (+0.002 to +0.007);
    shortlist k = 256 turns from below argmax to above it, and local search turns positive for the first time. The
    extra epochs alone (control) explain about a third of the gain. Official navtest / navhard runs with this scorer
    need its navtest / navhard predictions (one GPU job) and CPU jobs with `SCORES_PRED=p3_aux_pair_ep02`.

### Official eval with the pairwise scorer (2026-11-10)

The three p3_aux_policy RL-3 policies (ep 1) with sub-scores from p3_aux_pair ep 2 (`SCORES_PRED=p3_aux_pair_ep02`; scorer
predictions GPU 14947367, 21 min; CPU 14947368-403, all 36 completed, 0 failed scenes; outputs
`pdm_navtest|pdm_navhard_two_stage/<pred>__pairscorer_*`). The policy predictions are the cached ones; only the scorer
is new. In brackets: the same rule with the start model's own heads (tables above; – = not run).

| β | v1 mixture_v1 λ 3 | v1 mixture_v1 λ 10 | v1 shortlist_v1 k 256 | v1 shortlist_v1 k 64 | v2 mixture λ 3 | v2 mixture λ 10 | v2 shortlist k 256 | v2 shortlist k 64 |
|---|---|---|---|---|---|---|---|---|
| 0.005 | 0.9162 | **0.9180** (0.9165) | 0.9120 | 0.9059 | 0.8946 (0.8942) | 0.8958 | 0.8877 | 0.8706 |
| 0.02 | 0.9148 | 0.9162 (0.9131) | 0.9100 | 0.9026 | 0.8968 (0.8961) | **0.8978** (0.8944) | 0.8873 (0.8812) | 0.8701 |
| 0.05 | 0.9118 | 0.9134 (0.9110) | 0.9090 | 0.9017 | 0.8954 (0.8952) | 0.8950 | 0.8863 | 0.8686 |

navhard two-stage EPDMS (stage one / stage two):

| β | mixture λ 3 | mixture λ 10 | shortlist k 256 | shortlist k 64 |
|---|---|---|---|---|
| 0.005 | 0.369 (0.374) | 0.378 | **0.382** (0.412) | 0.368 |
| 0.02 | 0.348 (0.348) | 0.352 (0.370) | 0.364 (**0.413**) | 0.372 |
| 0.05 | 0.349 (0.350) | 0.355 | 0.359 (0.405) | 0.366 |

13. The val result carries over to navtest: with the pairwise scorer, mixture_v1 λ = 10 gives a new best v1 0.9180
    (β = 0.005, +0.0015 over the own heads) and mixture λ = 10 a new best v2 0.8978 (β = 0.02, +0.0017 over the
    previous best 0.8961; +0.0034 over the same rule with the own heads). Per scene on navtest (paired, n = 12146): v2
    +0.0017 ± 0.0008 vs the previous best (1215 scenes better, 644 worse), +0.0034 ± 0.0009 vs mixture λ 10 with the
    own heads; v1 +0.0015 ± 0.0008 (1087 better, 377 worse). Small (2-4 SE) but not noise. The scorer alone is worse
    (val score_heads −0.0296 vs −0.0267): the pairwise term improves the ordering of close neighbours, which is all a
    policy-restricted rule needs, at a small cost to picking the best of all 16384 entries. The hard shortlist stays clearly below mixture on navtest (k 256: 0.886-0.888;
    k 64: 0.869-0.871).
14. On navhard the pairwise scorer is worse: shortlist k = 256 drops from 0.405-0.413 to 0.359-0.382, mostly in stage
    two (β = 0.02: stage one 0.750 vs 0.760, stage two 0.476 vs 0.532). Mixture changes little (±0.02). With this scorer k = 64 beats k = 256 for β 0.02 and 0.05
    (0.372 vs 0.364), the reverse of the own-heads result. The pairwise loss sharpens near-ground-truth ranking on
    navtrain-like scenes, which does not help (and slightly hurts) navhard's shifted start states. For navhard the
    own-heads k = 256 run (0.413) remains the best among these scorers (the control scorer reaches 0.419, finding 15).

### Control scorer: extra epochs without the pairwise loss (2026-11-10)

The same policies with sub-scores from p3_aux_ctrl ep 2 (`SCORES_PRED=p3_aux_ctrl_ep02`: the p3_aux_pair fine-tune with
`pairwise_score_loss_weight=0`; predictions GPU 14949220, CPU 14949221-230, all completed; outputs `<pred>__ctrlscorer_*`).
navhard: two-stage EPDMS (stage one / stage two).

| β | navtest v2 mixture λ 10 | navtest v1 mixture_v1 λ 10 | navhard mixture λ 10 | navhard shortlist k 256 | navhard shortlist k 64 |
|---|---|---|---|---|---|
| 0.005 | 0.8965 | 0.9176 | 0.395 (0.771 / 0.504) | **0.419** (0.774 / 0.534) | 0.396 (0.760 / 0.511) |
| 0.02 | 0.8953 | 0.9142 | 0.370 (0.779 / 0.479) | 0.409 (0.765 / 0.527) | 0.386 (0.748 / 0.507) |

Pairwise minus control on navtest, per scene (paired, n = 12146):

| Policy | v2 mixture λ 10 | v1 mixture_v1 λ 10 |
|---|---|---|
| β = 0.02 | **+0.0025 ± 0.0008** (1247 better, 488 worse) | **+0.0020 ± 0.0008** |
| β = 0.005 | −0.0007 ± 0.0007 | +0.0003 ± 0.0007 |

15. The navhard drop of the pairwise scorer comes from the pairwise loss, not from the extra epochs: the control keeps
    navhard at the own-heads level or above (k 256: 0.419 / 0.409 vs own 0.412 / 0.413; pairwise 0.382 / 0.364), and
    its stage two stays at 0.53 (pairwise 0.48-0.49). The loss's pairs all lie near the human trajectory of real
    scenes; navhard's stage two starts off that path, where the sharper near-GT ranking misleads the selection.
16. On navtest the pairwise loss helps only with the β = 0.02 policy (+0.002-0.0025, ~3 SE); with β = 0.005 it ties
    the control. The v1 best (0.9180) is therefore mostly the extra epochs (control 0.9176); the v2 best (0.8978,
    β = 0.02) needs the pairwise loss. A plausible reading: the β = 0.005 policy stays closer to its start and already
    concentrates on fewer candidates, leaving less near-neighbour choice for the scorer (UNVERIFIED).
17. navhard 0.419 is the best navhard number so far but +0.006 over 0.413 is likely within noise (no paired SE for the
    two-stage score computed), and k = 256 remains a rule chosen on test results.

### navtest with reactive traffic (2026-11-11)

Diagnostic, not a leaderboard number (official navtest v2 is non-reactive): RL-3 p3_aux_policy β = 0.02, argmax, the same
saved predictions scored with `TRAFFIC=reactive` (IDM agents react to the ego vehicle, as in navhard). Job 14949219
(4 CPUs, 2 h 30 min; reactive scoring is ~7x slower than non-reactive, the first two attempts 14945141 / 14947203 timed
out at 50 min / 1 h 30 min). 12146 / 12146 scenes valid.

| Traffic | EPDMS | v1 PDMS | NC | TTC | DAC | EP | LK | EC |
|---|---|---|---|---|---|---|---|---|
| non-reactive (official) | 0.8950 | 0.9117 | 0.9814 | 0.9756 | 0.9795 | 0.8891 | 0.9565 | 0.8208 |
| reactive | 0.8926 | 0.9095 | 0.9778 | 0.9769 | 0.9795 | 0.8916 | 0.9565 | 0.8208 |

18. Reactive traffic alone costs only −0.0025 ± 0.0015 EPDMS (per scene, paired; 11% of scenes change, 847 better / 527
    worse): slightly more at-fault collisions (−0.004), slightly better TTC and progress. navhard stage one is ~0.12
    below navtest (0.76-0.77 vs 0.895), so reactivity explains almost none of that gap; it comes from navhard's harder
    scene selection. Training or evaluating with reactive simulation is therefore not where the navhard gain lies.

### Calibration: human agent, human filter and published numbers (2026-11-12)

The logged human trajectory (`navsim.agents.human_agent.HumanAgent`) scored with exactly our navtest settings (same
balanced NAVSIM script and metric cache; `exp/human_agent_logs/`), with the human filter on (our default) and off.
Jobs: human 15000303 (on) / 15001861 (off); best navtest 15001862 (off); best navhard 15001863 (off).

| navtest | filter | EPDMS | v1 PDMS | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Human (ours) | on | 0.9451 | 0.9444 | 1.000 | 1.000 | 0.998 | 1.000 | 0.874 | 1.000 | 1.000 | 0.981 | 0.901 |
| Human (ours) | off | **0.9032** | 0.9442 | 1.000 | 1.000 | 0.997 | 0.974 | 0.874 | 1.000 | 0.874 | 0.981 | 0.901 |
| Human, NAVSIM v1 paper (Table 1) | – | – | **0.948** | 1.000 | 1.000 | – | – | 0.875 | 1.000 | – | 0.999 (v1 comfort) | – |
| Best v2 (RL-3 β 0.02 + pairwise scorer, mixture λ 10) | on | 0.8978 | 0.9162 | 0.981 | 0.985 | 0.992 | 0.999 | 0.890 | 0.975 | 0.956 | 0.983 | 0.815 |
| same | off | **0.8639** | 0.9162 | 0.981 | 0.985 | 0.992 | 0.976 | 0.890 | 0.975 | 0.848 | 0.983 | 0.815 |

| navhard (best: RL-3 β 0.005 + control scorer, shortlist k 256) | stage one | stage two | combined |
|---|---|---|---|
| filter on (= leaderboard scoring) | 0.774 | 0.534 | **0.419** |
| filter off | 0.724 | 0.534 | 0.395 |

19. Published navtest v2 numbers are filter-off: the human scores 0.9032 here, the value the papers report (90.3; e.g.
    ZTRS). The filter waives mostly lane keeping (human LK 0.874 unfiltered) and traffic lights. Paper-comparable navtest
    v2 for our best model is **0.864** (filter on 0.898); for comparison (filter-off convention, camera only): DriveSuprim
    ViT-L 87.1, ZTRS ViT-L 86.2 / V2-99 85.3, Hydra-MDP++ ViT-L 85.6 (ZTRS paper, Table 3).
20. v1 PDMS is unaffected by the filter (the human never fails NC / DAC / TTC). Our v1 uses v2 history comfort (human
    0.981) instead of v1 comfort (0.999), so our v1 is slightly conservative: the human gets 0.944 here vs the
    official 0.948; with 0.999 comfort the same formula gives 0.948.
21. The navhard leaderboard applies the filter (pseudo-simulation paper, Sec. 3.1), and our filter-on navhard matches it:
    p3_aux_policy shortlist k 256 scored **38.15** on the leaderboard vs 0.382 locally. The filter acts on stage one only
    (stage two is synthetic, without a human trajectory). Leaderboard snapshot 03/2026 (pseudo-simulation paper, Table
    2): PDM-Closed (privileged) 56.6, DrivoR 54.5, SimScale 53.2, GuideFlow 51.5, ZTRS 48.1, RAP 39.6, NavFormer 34.1,
    LTF 25.1. Our best (0.419 locally) scored **41.94** on the leaderboard (finding 22): above RAP, ~6 below ZTRS, ~13 below DrivoR. navhard is where we
    are clearly behind; on navtest we are level with the strong camera-only methods.
22. Submission pickle of the best navhard setup (no GPU, from the saved predictions:
    `scripts/submission/create_submission_from_predictions.py`, job 15002048):
    `exp/submission_navhard_two_stage/rl3_b0.005_ep01__ctrlscorer_shortlist_k256/submission.pkl` (450 + 5462
    trajectories, metadata as the previous submission). **Leaderboard: 41.936** (submitted 2026-11-13, shown as
    "anonym"), identical to the local 0.41936. Every per-metric value matches the local CSV summary rows to the 4
    digits shown, in particular the *weighted* stage-two values (e.g. LK 53.26 = 0.533, EC 59.67 = 0.597; the
    unweighted means would be 0.505 / 0.521), which confirms the corrected navhard table below. The leaderboard shows
    the combined score and the 18 sub-scores, not the stage totals (local: stage one 77.38, stage two 53.35).

    | Stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
    |---|---|---|---|---|---|---|---|---|---|
    | one | 95.78 | 96.00 | 99.78 | 99.78 | 76.83 | 95.33 | 93.56 | 96.89 | 52.44 |
    | two | 85.79 | 86.77 | 91.84 | 98.37 | 77.71 | 84.20 | 53.26 | 96.61 | 59.67 |

### v1 PDMS with the v1 comfort term (2026-11-12)

The v1 numbers above use v2 history comfort, which checks [human past states + simulated plan] and is stricter than
v1 comfort (simulated plan alone; same `ego_is_comfortable` thresholds). `scripts/evaluation/v1_pdms_official_comfort.py`
re-simulates each selected trajectory exactly as `navsim/evaluate/pdm_score.py` does and applies the check without the
history padding (upstream code only called). NC, DAC, EP, TTC from the same run. Jobs 15002417-21 (`exp/v1_comfort_logs/`,
per-token CSVs `*_v1_comfort.csv` next to the result CSVs).

| navtest | comfort: v1 / history | **v1 PDMS (v1 comfort)** | previous (history comfort) |
|---|---|---|---|
| Human (filter off; v1 has no filter) | 0.9958 / 0.9807 | **0.9468** | 0.9442 |
| Human, NAVSIM v1 paper (Table 1) | 0.999 | **0.948** | – |
| RL-3 β 0.005 + pairwise scorer, mixture_v1 λ 10 | 0.9914 / 0.9826 | **0.9193** | 0.9180 |
| RL-3 β 0.005 + control scorer, mixture_v1 λ 10 | 0.9912 / 0.9827 | 0.9189 | 0.9176 |
| RL-3 β 0.005 + own heads, mixture_v1 λ 10 | 0.9913 / 0.9826 | 0.9178 | 0.9165 |
| RL-3 β 0.005, argmax | 0.9915 / 0.9822 | 0.9143 | 0.9129 |

23. With v1 comfort the human reaches 0.9468 vs the official 0.948. The rest (~0.001) is comfort 0.9958 vs 0.999 and
    progress 0.8742 vs 0.875, most likely v1/v2 devkit differences in the simulation and the PDM-Closed progress
    reference (the v1 devkit is not in this repo; UNVERIFIED). Models gain +0.0013-0.0014 (their plans start smoothly
    from the current state, so the history padding failed them less often than the human). Corrected best navtest v1:
    **0.9193**; all v1 values elsewhere in this file are history-comfort values, ~0.0013 lower. (Superseded in part by
    finding 52: the shift is +0.002 for policy picks but −0.010 for score-head picks.)

### v1 PDMS with the v1 comfort term for every model (2026-11-18)

Every non-reactive navtest result that reports v1 (argmax runs and the `_v1` selections), recomputed with the v1
comfort term: `v1_pdms_official_comfort.py --from-run-config` rebuilds the agent (selection, k / λ, scorer) from the
run's own `code/hydra/config.yaml`, so each value is the same trajectory set as the original run, only the comfort
term changes. Own sub-score heads only (RL-3: the start model's heads, `SCORES_PRED`; pairwise / control scorer runs
were not recomputed, except the two in the table above). 61 CPU jobs 15175655-15175715, all completed; check job
15175716 reproduces the hand-configured value of the table above (0.9913 / 0.9178). List of runs:
`exp/v1_comfort_logs/todo_runs.txt`, per-token CSVs `*_v1_comfort.csv` next to each result CSV. Δ = v1 comfort − HC
version. Main tables of this file (RL-3, val-chosen rules, past inputs, local search, re-evaluation 2026-11-08) now show
these values; older sections (combined pipeline, selection rules 2026-11-06, ablation chain, RL-1) keep the HC values.

| Model | Selection | comfort pass (v1) | **v1 PDMS (v1 comfort)** | HC version | Δ |
|---|---|---|---|---|---|
| p3_policy ep 7 | argmax | 0.998 | **0.7944** | 0.7926 | +0.0018 |
| p3_combo_balanced ep 8 | argmax | 0.998 | **0.8327** | 0.8307 | +0.0020 |
| p3_combo_balanced_aux ep 9 | argmax | 0.998 | **0.8429** | 0.8407 | +0.0022 |
| | score_heads_v1 | 0.905 | **0.8523** | 0.8631 | −0.0108 |
| | shortlist_v1 k 64 | 0.939 | **0.8671** | 0.8729 | −0.0058 |
| | shortlist_v1 k 256 | 0.955 | **0.8688** | 0.8723 | −0.0035 |
| | mixture_v1 λ 10 | 0.998 | **0.8687** | 0.8666 | +0.0021 |
| p3_heads ep 5 | score_heads_v1 | 0.911 | **0.8713** | 0.8818 | −0.0105 |
| p3_heads_policy ep 2 | shortlist_v1 k 64 | 0.974 | **0.8850** | 0.8859 | −0.0009 |
| p3_aux ep 5 | score_heads_v1 | 0.915 | **0.8792** | 0.8894 | −0.0102 |
| p3_aux_policy ep 3 | argmax | 0.999 | **0.8627** | 0.8607 | +0.0020 |
| | shortlist_v1 k 64 | 0.975 | **0.8913** | 0.8922 | −0.0009 |
| | shortlist_v1 k 256 | 0.988 | **0.8924** | 0.8915 | +0.0009 |
| + RL-1 β 0.02 (ep 12) | shortlist_v1 k 256 | 0.985 | **0.8994** | 0.8988 | +0.0006 |
| + RL-1 β 0.005, lr 3e-4 (ep 7) | shortlist_v1 k 256 | 0.984 | **0.8983** | 0.8979 | +0.0004 |
| + RL-3 β 0.005 | argmax (2026-11-12) | 0.992 | **0.9143** | 0.9129 | +0.0014 |
| | shortlist_v1 k 256 | 0.981 | **0.9110** | 0.9113 | −0.0003 |
| | mixture_v1 λ 10 | 0.991 | **0.9178** | 0.9165 | +0.0013 |
| + RL-3 β 0.005, seed 1 | argmax | 0.993 | **0.9142** | 0.9126 | +0.0016 |
| + RL-3 β 0.02 | argmax | 0.997 | **0.9138** | 0.9117 | +0.0021 |
| | shortlist_v1 k 256 | 0.984 | **0.9088** | 0.9083 | +0.0005 |
| | mixture_v1 λ 10 | 0.997 | **0.9151** | 0.9131 | +0.0020 |
| | local_v1 N 64 / 128, λ 0.02-0.2 (8 runs) | 0.997 | 0.8986-0.9102 | 0.8964-0.9081 | +0.0021-0.0022 |
| + RL-3 β 0.05 | argmax | 0.999 | **0.9106** | 0.9083 | +0.0023 |
| | shortlist_v1 k 256 | 0.986 | **0.9063** | 0.9056 | +0.0007 |
| | mixture_v1 λ 10 | 0.999 | **0.9133** | 0.9110 | +0.0023 |
| + RL-3 β 0.005 + EC reward w 2 / w 4 | argmax | 0.995 / 0.994 | **0.9095** / 0.9051 | 0.9076 / 0.9032 | +0.0019 |
| + RL-3 β 0.005 + EC reward, shortlist previous plan, w 2 / w 4 | argmax | 0.993 / 0.994 | **0.9098** / 0.9047 | 0.9082 / 0.9029 | +0.0016 / +0.0018 |
| | w 4, seed 1, argmax | 0.994 | 0.9052 | 0.9035 | +0.0017 |
| + RL-3 β 0.005, seed 1 / EC reward w 2 (own heads, 2026-11-18) | mixture_v1 λ 10 | – | 0.9171 / 0.9118 | 0.9155 / 0.9100 | +0.0016 / +0.0018 |
| + RL-3 β 0.005 + EC reward w 4 / shortlist previous plan w 2 / w 4 / w 4 seed 1 (own heads) | mixture_v1 λ 10 | – | 0.9072 / 0.9120 / 0.9077 / 0.9075 | 0.9054 / 0.9104 / 0.9061 / 0.9057 | +0.0016-0.0019 |
| p3_heads_policy ep 2 + RL-3 β 0.005 | argmax | 0.994 | **0.9017** | 0.9001 | +0.0016 |
| | shortlist_v1 k 256 | 0.981 | 0.8996 | 0.8997 | −0.0001 |
| | mixture_v1 λ 10 | 0.993 | **0.9058** | 0.9042 | +0.0016 |
| p3_heads_policy ep 2 + RL-3 β 0.02 | argmax / shortlist_v1 k 256 / mixture_v1 λ 10 | 0.997 / 0.984 / 0.997 | 0.9028 / 0.8987 / **0.9049** | 0.9007 / 0.8982 / 0.9030 | +0.0021 / +0.0005 / +0.0019 |
| p3_heads_policy ep 2 + RL-3 β 0.05 | argmax / shortlist_v1 k 256 / mixture_v1 λ 10 | 0.998 / 0.985 / 0.997 | 0.9008 / 0.8982 / **0.9021** | 0.8986 / 0.8976 / 0.9000 | +0.0022 / +0.0006 / +0.0021 |
| p3hist_none / ego / prev / both policy | argmax | 0.999 | 0.8578 / 0.8603 / 0.8537 / 0.8527 | 0.8557 / 0.8583 / 0.8516 / 0.8507 | +0.0020-0.0021 |
| | mixture_v1 λ 10 | 0.999 | 0.8884 / **0.8899** / 0.8836 / 0.8811 | 0.8862 / 0.8876 / 0.8814 / 0.8790 | +0.0021-0.0023 |
| RL-3 p3hist_none / ego / prev / both | argmax | 0.993-0.994 | 0.9088 / 0.9113 / 0.9089 / 0.9076 | 0.9071 / 0.9097 / 0.9073 / 0.9060 | +0.0016-0.0017 |
| | mixture_v1 λ 10 | 0.992-0.994 | 0.9123 / **0.9139** / 0.9120 / 0.9096 | 0.9107 / 0.9123 / 0.9106 / 0.9080 | +0.0014-0.0016 |

Paired differences on navtest that change (12,146 scenes, mean ± SE; script
`exp/v1_comfort_logs/paired_v1c.py`, same per-token CSVs):

| Comparison | v1 (v1 comfort) | v1 (HC version) |
|---|---|---|
| p3_aux_policy shortlist_v1 k 256 − p3_aux score_heads_v1 | **+0.0132 ± 0.0019** | +0.0022 ± 0.0019 |
| p3_heads_policy shortlist_v1 k 64 − p3_heads score_heads_v1 | **+0.0136 ± 0.0017** | +0.0041 ± 0.0017 |
| p3_combo_balanced_aux shortlist_v1 k 64 − mixture_v1 λ 10 | −0.0017 ± 0.0021 (tie) | +0.0063 ± 0.0021 |
| p3_aux_policy shortlist_v1 k 256 − k 64 | +0.0011 ± 0.0014 (tie) | −0.0007 ± 0.0014 (tie) |
| RL-3 β 0.005 argmax − shortlist_v1 k 256 | **+0.0033 ± 0.0014** | +0.0016 ± 0.0014 |
| RL-3 β 0.02 argmax − shortlist_v1 k 256 | **+0.0050 ± 0.0014** | +0.0034 ± 0.0014 |
| RL-3 β 0.005 mixture_v1 λ 10 − argmax | +0.0035 ± 0.0010 | +0.0036 ± 0.0010 |
| RL-1 β 0.02 − p3_aux_policy (shortlist_v1 k 256) | +0.0070 ± 0.0012 | +0.0073 ± 0.0012 |

51. **v1 comfort is not uniformly more lenient than history comfort.** Policy picks (argmax, mixture, local) pass v1
    comfort in 99.2-99.9% of scenes (HC: 98.2-98.3%) and gain +0.0013 to +0.0023. Picks of the sub-score heads over all
    16,384 entries (`score_heads_v1`) pass only 90.5-91.5% and lose 0.010-0.011; shortlist picks lie in between
    (k 256: 98.1-98.8%, ±0.001; k 64 with the weaker p3_combo_balanced_aux scorer: 93.9%, −0.006). HC pads the plan
    with 1.5 s of human past states before the same `ego_is_comfortable` check, so it can only add failing time steps
    in principle; that it passes more score-head picks LIKELY comes from the derivative filter in
    `pdm_comfort_metrics.py` behaving differently at the plan start with and without the padded history (not verified
    per metric). The v1 comfort check on the simulated plan alone is what the v1 formula specifies, so these are the
    representative v1 numbers.
52. **Conclusions that change:** (a) the policy head now helps v1 clearly: p3_aux_policy shortlist over p3_aux's heads
    +0.013 (7 SE; HC version +0.002, a tie), the same for p3_heads (+0.014); (b) for p3_combo_balanced_aux the
    shortlist k 64 no longer beats mixture_v1 on v1 (tie); (c) for RL-3 the policy argmax beats shortlist_v1 on v1
    (+0.003 / +0.005, 2.4 / 3.6 SE), as it already did on v2. **Unchanged:** all RL gains (RL-1 +0.007, RL-3 over its
    start, mixture_v1 over argmax +0.0035), the ranking of the RL-3 models and of the past-input variants, and the best
    navtest v1: RL-3 β 0.005 mixture_v1 λ 10, **0.9178** with the own heads (0.9193 pairwise, 0.9189 control scorer).

### Models with past inputs: policy stage and RL-3 (2026-11-14)

The four p3hist policy models (policy head only, from p3hist_none / ego / prev / both; experiments.md "Phase 1 with past
inputs") and their RL-3 β = 0.005 models (recipe of rl3_b0.005), next to the no-history reference (p3_aux_policy and
rl3_b0.005 from it). Predictions: GPU 15031806/07/24/25/42/43/60/61 (24-26 min; prev / both read the past camera
frames, also those of navhard's synthetic scenes, all present); scoring: 72 CPU jobs (`exp/rl3_logs/rl3_hist_eval_2026-11-13.txt`),
all completed. Selections: navtest v1 = mixture_v1 λ 10, v2 = mixture λ 10 (argmax: one run), navhard = shortlist
k = 256. Scorers: argmax (none); own = the policy model's own heads (RL-3: its start policy model's heads,
`SCORES_PRED`); control = p3_aux_ctrl ep 2 (the same for all, isolates the policy); pairwise = p3_aux_pair ep 2
(navtest only, RL-3 only). v1 with the v1 comfort term in the argmax and own columns and for the reference row
(2026-11-18); with HC in the other control / pairwise cells (not recomputed; v1 comfort adds ~+0.0015 to policy-based picks). navhard: combined (stage one / stage two).

| Model | argmax: v1 / v2 / navhard | own scorer: v1 / v2 / navhard | control scorer: v1 / v2 / navhard | pairwise: v1 / v2 |
|---|---|---|---|---|
| p3_aux_policy (reference) | 0.8627 / 0.8433 / 0.254 | – | – | – |
| p3hist_none_policy | 0.8578 / 0.8386 / 0.258 | 0.8884 / 0.8661 / 0.375 | 0.8920 / 0.8709 / 0.373 | – |
| p3hist_ego_policy | 0.8603 / 0.8416 / 0.242 | 0.8899 / 0.8688 / 0.369 | 0.8939 / 0.8752 / 0.362 | – |
| p3hist_prev_policy | 0.8537 / 0.8346 / 0.236 | 0.8836 / 0.8615 / 0.374 | 0.8943 / 0.8751 / 0.365 | – |
| p3hist_both_policy | 0.8527 / 0.8337 / 0.227 | 0.8811 / 0.8624 / 0.339 | 0.8916 / 0.8730 / 0.361 | – |
| RL-3 from p3_aux_policy (rl3_b0.005, reference) | 0.9143 / 0.8920 / 0.365 | 0.9178 / – / 0.412 | **0.9189 / 0.8965 / 0.419** | **0.9193** / 0.8958 |
| RL-3 p3hist_none | 0.9088 / 0.8855 / 0.324 | 0.9123 / 0.8872 / 0.403 | 0.9130 / 0.8913 / 0.384 | 0.9130 / 0.8904 |
| RL-3 p3hist_ego | 0.9113 / 0.8897 / 0.355 | 0.9139 / 0.8924 / 0.408 | 0.9150 / 0.8932 / **0.418** | 0.9151 / 0.8943 |
| RL-3 p3hist_prev | 0.9089 / 0.8849 / 0.318 | 0.9120 / 0.8882 / 0.391 | 0.9147 / 0.8926 / 0.387 | 0.9155 / 0.8925 |
| RL-3 p3hist_both | 0.9076 / 0.8852 / 0.337 | 0.9096 / 0.8881 / 0.350 | 0.9132 / 0.8931 / 0.393 | 0.9140 / 0.8933 |

navhard stages (control scorer, k = 256): RL-3 none 0.752 / 0.509, ego 0.775 / 0.531, prev 0.764 / 0.507, both 0.770 /
0.506; reference 0.774 / 0.534. navtest EC: policy models 0.80-0.82, RL-3 models 0.77-0.79 (reference 0.78).

navtest v2, per scene (paired, n = 12,146), history variant minus none, same scorer:

| Scorer | ego − none (policy / RL-3) | prev − none (policy / RL-3) | both − none (policy / RL-3) | RL-3 none − reference | RL-3 ego − reference |
|---|---|---|---|---|---|
| argmax | +0.0030 ± 0.0025 / **+0.0042 ± 0.0018** | −0.0040 ± 0.0025 / −0.0006 ± 0.0019 | −0.0049 ± 0.0025 / −0.0003 ± 0.0019 | −0.0065 ± 0.0017 | −0.0023 ± 0.0018 |
| own | +0.0028 ± 0.0019 / **+0.0052 ± 0.0017** | −0.0045 ± 0.0020 / +0.0010 ± 0.0017 | −0.0037 ± 0.0021 / +0.0009 ± 0.0017 | – | – |
| control | +0.0043 ± 0.0017 / +0.0018 ± 0.0016 | +0.0041 ± 0.0017 / +0.0012 ± 0.0016 | +0.0021 ± 0.0017 / +0.0018 ± 0.0016 | −0.0052 ± 0.0015 | −0.0033 ± 0.0016 |
| pairwise | – / **+0.0039 ± 0.0016** | – / +0.0021 ± 0.0017 | – / +0.0029 ± 0.0017 | −0.0054 ± 0.0016 | −0.0015 ± 0.0016 |

33. **Ego history is the only past input that helps consistently:** RL-3 ego beats RL-3 none on navtest v2 with every
    scorer (+0.0018 to +0.0052; 2.3-3 SE with argmax, own and pairwise, 1.1 SE with control), and on navhard
    (control 0.418 vs 0.384, own 0.408 vs 0.403, argmax 0.355 vs 0.324). The previous camera frame (prev, both) gives
    nothing reliable: negative before RL-3 except with the control scorer, about zero after it, and lower navhard.
34. **But no history model beats the no-history reference.** RL-3 none, the matched baseline with the same recipe as the
    reference, is 0.005-0.0065 below it on navtest v2 (3-4 SE) and 0.035 below on navhard (control). That gap is
    run-to-run variation (another phase-1 / phase-3 run of the same recipe), and it is as large as the ego effect.
    RL-3 ego lands between the two (−0.0015 to −0.0033 vs the reference, 1-2 SE; navhard 0.418 vs 0.419). With one run
    per variant, "ego history +0.004" is suggestive but not conclusive; the reference remains the best model (v1 0.9193
    with the pairwise scorer (v1 comfort), v2 0.8965 with the control scorer, navhard 0.419; with EC-aware selection 0.9027 / 0.441).
35. The navtrain val ranking after RL-3 (ego = none, shortlist 0.8476 vs 0.8478) did not show the navtest ego gain;
    val has no EC and no DDC / TLC / LK, and its differences are of the same size as the noise.

### EC-aware selection (step 0 of the EC work; 2026-11-14)

No training: a selection rule on the saved predictions of RL-3 β = 0.005 (`scripts/evaluation/ec_aware_selection.py`).
For every scene with a previous frame (navtest: the scene starting 0.5 s earlier in the same log, paired as
`run_pdm_score_one_stage.py` does, 10,040 of 12,146 scenes; navhard: the (now, previous) pairs of the split, 2,956),
the previous plan is the base rule's choice at t − 0.5 s. At t, the candidates are taken in the policy's order (its
top N, or the top N by policy inside the control scorer's shortlist of 256) and the first one that passes two-frame
extended comfort against the previous plan is output; if none passes, the base choice stays. EC is computed as the
official scorer does (both plans simulated by `PDMSimulator` from their logged ego states, then the overlap and
`ego_is_two_frame_extended_comfort`). The chosen trajectories are written as prediction folders
(`predictions/rl3_b0.005_ep01_ec_<base>_n<N>/`) and scored by the official pipeline with argmax. Jobs: selection
15032113 (navtest, 8 CPUs, 23 min) / 15032114 (navhard, 7 min), scoring 15032116-23, navtest baseline shortlist k 256
with the control scorer 15032115. v1 here: NC · DAC · (5 EP + 5 TTC + 2 HC)/12 (HC instead of the v1 comfort term,
as the scoring log; EC is not in v1, so v1 shows only what the rule costs).

| Rule | changed (navtest / navhard) | navtest v1 | navtest v2 | EC | EP | NC | DAC | navhard (stage one / two) | EC stage one / two |
|---|---|---|---|---|---|---|---|---|---|
| argmax | – | 0.9129 | 0.8920 | 0.780 | 0.892 | 0.979 | 0.983 | 0.365 (0.760 / 0.473) | 0.609 / 0.598 |
| argmax + EC, top 16 | 1021 / 607 | 0.9125 | 0.8994 | 0.863 | 0.892 | 0.979 | 0.982 | 0.391 (0.778 / 0.494) | 0.791 / 0.811 |
| argmax + EC, top 64 | 1339 / 826 | 0.9111 | **0.8997** | **0.878** | 0.892 | 0.979 | 0.980 | 0.393 (0.780 / 0.497) | 0.858 / 0.880 |
| shortlist k 256, control scorer | – | 0.9069 | 0.8836 | 0.742 | 0.876 | 0.976 | 0.986 | 0.419 (0.774 / 0.534) | 0.524 / 0.597 |
| shortlist + EC, top 16 | 974 / 539 | 0.9059 | 0.8909 | 0.825 | 0.874 | 0.976 | 0.985 | 0.437 (0.791 / 0.546) | 0.702 / 0.773 |
| shortlist + EC, top 64 | 1224 / 683 | 0.9048 | 0.8913 | 0.838 | 0.873 | 0.976 | 0.985 | **0.441** (0.795 / 0.549) | 0.756 / 0.805 |
| argmax + EC, top 16, sequential | 1083 / – | 0.9124 | 0.9012 | 0.884 | 0.892 | 0.979 | 0.981 | – | – |
| argmax + EC, top 64, sequential | 1431 / – | 0.9108 | **0.9027** | **0.915** | 0.892 | 0.979 | 0.979 | – | – |
| shortlist + EC, top 16, sequential | 1030 / – | 0.9054 | 0.8923 | 0.844 | 0.874 | 0.976 | 0.985 | – | – |
| shortlist + EC, top 64, sequential | 1349 / – | 0.9045 | 0.8944 | 0.872 | 0.873 | 0.976 | 0.985 | – | – |

navtest v2, per scene against the base rule (paired, n = 12,146): argmax + EC top 16 +0.0074 ± 0.0006 (915 scenes
better, 191 worse), top 64 +0.0077 ± 0.0008 (1128 / 294); shortlist + EC top 16 +0.0073 ± 0.0005, top 64 +0.0078 ±
0.0006. argmax + EC top 64 against the previous best v2 (0.8978, RL-3 β 0.02 + pairwise scorer, mixture λ 10):
+0.0019 ± 0.0016.

Sequential variant (navtest only; selection job 15033074, 21 min; scoring 15033075-78): each log in time order, the
previous plan is the choice actually made at t − 0.5 s (EC changes included), so the rule checks exactly the pair the
official scorer compares (its predicted EC equals the official one: argmax top 64 0.9147 = 0.9147). Per scene against
the base rule: argmax top 16 +0.0092 ± 0.0007, top 64 **+0.0107 ± 0.0008**; shortlist top 16 +0.0087 ± 0.0005, top 64
+0.0108 ± 0.0006; against the non-sequential rule +0.0014 to +0.0031 (±0.0003-0.0006). argmax + EC top 64 sequential
against the previous best v2 (0.8978): **+0.0049 ± 0.0016**.

27. **Consistency with the previous plan pays off on both benchmarks, without training.** navtest v2 +0.0077 (EC
    0.780 → 0.878, near the human's 0.901); navhard +0.022 with the best rule (0.419 → 0.441, i.e. ~44.1 on the
    leaderboard), +0.028 with argmax. EC rises in both navhard stages (shortlist: 0.52 → 0.76, 0.60 → 0.81).
28. **It costs almost nothing elsewhere:** EP, NC, TTC, LK unchanged within ±0.001; DAC −0.001 to −0.003; v1 (no EC)
    −0.0004 to −0.002. Most of the gain is already in the top 16 (+0.0074 of +0.0077 on navtest).
29. Replication check: the EC of the unchanged base choices equals the official EC (navtest argmax 0.7799 = 0.7799;
    navhard shortlist 0.5217 vs the official unweighted 0.524 / 0.521 per stage).
30. **EC needs the simulation:** EC computed from the planned trajectories without the LQR / bicycle simulation agrees
    with the simulated EC on only 57-65% of candidates (navtest 57-58%, navhard 62-65%). EC inside training therefore
    needs the simulator (a torch / GPU port of `PDMSimulator`).
31. navtest caveat: a previous frame can itself be changed by the rule (chains of frames), and the official EC then
    compares with the changed choice, not the base choice used by the rule. The sequential variant (each log in time
    order, previous = the choice actually made) measures this; navhard is not affected (its previous frames are never
    changed). For a leaderboard agent the previous plan must come from its own input (recomputed from the past frames),
    not from another test scene's prediction; the base choice at t − 0.5 s can be recomputed that way.
32. **Using the plan actually output at t − 0.5 s is better still:** navtest v2 0.9027 (+0.0107 over argmax, +0.0049
    over the previous best 0.8978), EC 0.915, above the human's 0.901, at the same small cost (v1 −0.002, DAC −0.003).
    The chain matters: +0.003 over the non-sequential rule. A real agent knows its last output; under NAVSIM's
    per-scene evaluation it would have to rebuild the chain from its 1.5 s of past frames (3 steps), which is close to
    but not exactly the full chain. This is also the target RL with EC aims at: a policy whose consecutive argmax plans
    are consistent by themselves.

### RL-3 with EC in the reward: official evaluation (2026-11-14)

`rl3_ec2_b0.005` / `rl3_ec4_b0.005` (experiments.md, "RL-3 with two-frame extended comfort in the reward"): the
rl3_b0.005 recipe with EC in the reward at weight 2 (the official EPDMS weighting) / 4; previous plan = the current
model's argmax at t − 0.5 s. Final epoch-1 checkpoints. Predictions GPU 15053086 / 15053103 (`rl_ec_weight=0` at
inference); scoring and EC rule 15053087-102 / 15053104-119 (`exp/rl3_logs/rl3_ec_runs_2026-11-14.txt`), all
completed. Scorer for every non-argmax rule: control (p3_aux_ctrl ep 2). EC rule on navtest: sequential; on navhard:
previous = base choice (as in "EC-aware selection"). Paired navtest v2 differences per scene against rl3_b0.005 with
the same rule (n = 12,146). Script: scratchpad `ec_train_results.py`.

| Rule | Model | navtest v1 | navtest v2 | Δ v2 vs reference | EC | EP | navhard (stage one / two) | EC stage one / two |
|---|---|---|---|---|---|---|---|---|
| argmax | rl3_b0.005 | **0.9129** | 0.8920 | | 0.780 | **0.892** | 0.365 (0.760 / 0.473) | 0.609 / 0.598 |
| | EC w 2 | 0.9076 | **0.8943** | +0.0023 ± 0.0016 | 0.851 | 0.879 | 0.376 (0.759 / 0.487) | 0.742 / 0.705 |
| | EC w 4 | 0.9032 | 0.8913 | −0.0008 ± 0.0017 | **0.870** | 0.870 | **0.379** (0.751 / 0.493) | 0.751 / 0.779 |
| mixture λ 10 | rl3_b0.005 | 0.9173 | 0.8965 | | 0.780 | 0.888 | – | – |
| | EC w 2 | 0.9121 | **0.8983** | +0.0018 ± 0.0014 | 0.847 | 0.877 | – | – |
| | EC w 4 | 0.9048 | 0.8928 | −0.0036 ± 0.0015 | 0.863 | 0.868 | – | – |
| shortlist k 256 | rl3_b0.005 | 0.9069 | 0.8836 | | 0.742 | 0.876 | **0.419** (0.774 / 0.534) | 0.524 / 0.597 |
| | EC w 2 | 0.9043 | 0.8859 | +0.0023 ± 0.0011 | 0.784 | 0.868 | 0.415 (0.767 / 0.538) | 0.596 / 0.625 |
| | EC w 4 | 0.9015 | 0.8843 | +0.0007 ± 0.0013 | 0.800 | 0.863 | 0.411 (0.760 / 0.536) | 0.591 / 0.668 |
| argmax + EC rule n 64 | rl3_b0.005 | 0.9108 | **0.9027** | | 0.915 | 0.892 | 0.393 (0.780 / 0.497) | 0.858 / 0.880 |
| | EC w 2 | 0.9048 | 0.8996 | −0.0031 ± 0.0017 | 0.941 | 0.879 | 0.396 (0.772 / 0.506) | 0.916 / 0.917 |
| | EC w 4 | 0.8986 | 0.8943 | −0.0084 ± 0.0018 | 0.948 | 0.869 | 0.397 (0.758 / 0.508) | 0.929 / 0.932 |
| shortlist + EC rule n 64 | rl3_b0.005 | 0.9045 | 0.8944 | | 0.872 | 0.873 | **0.441** (0.795 / 0.549) | 0.756 / 0.805 |
| | EC w 2 | 0.9012 | 0.8929 | −0.0015 ± 0.0011 | 0.886 | 0.866 | 0.439 (0.779 / 0.561) | 0.782 / 0.805 |
| | EC w 4 | 0.8990 | 0.8910 | −0.0034 ± 0.0013 | 0.894 | 0.861 | 0.438 (0.786 / 0.550) | 0.804 / 0.811 |

mixture_v1 λ 10 (v1): 0.9176 / 0.9117 / 0.9060. EC rule n 16 (navtest v2 / navhard): argmax base 0.9012 / 0.391,
0.8989 / 0.397, 0.8944 / 0.394; shortlist base 0.8923 / 0.437, 0.8918 / 0.434, 0.8900 / 0.434 (reference / w 2 / w 4).

36. **EC in the reward makes the policy itself more consistent, at a cost in progress.** Weight 2 raises argmax EC on
    navtest from 0.780 to 0.851 (w 4: 0.870) and navhard EC in both stages (0.61 / 0.60 → 0.74 / 0.71), while EP falls
    0.892 → 0.879 (w 4: 0.870) and v1 (no EC) falls 0.005 (w 4: 0.010). Net on navtest v2 without a test-time rule:
    w 2 +0.002 with every rule (argmax +0.0023 ± 0.0016, mixture +0.0018 ± 0.0014, shortlist +0.0023 ± 0.0011, the last
    2 SE); w 4 about zero or negative. navhard: argmax +0.011 / +0.014, but the best rule (shortlist k 256) −0.004 /
    −0.008, all from stage one (0.774 → 0.767 / 0.760); stage two +0.002-0.004.
37. **The test-time EC rule beats EC training, and the two do not add up.** With the rule, both EC models are below
    the reference on navtest v2 (n 64: −0.0031 / −0.0084; n 16: −0.0023 / −0.0068) and level or slightly below on
    navhard (0.439 / 0.438 vs 0.441). The rule reaches about the same EC from either starting point (navtest 0.915 vs
    0.941 / 0.948), so the training's EC gain is largely redundant with it, while its EP loss (−0.013 / −0.022) stays.
    The rule knows the previous plan and pays only where a consistent plan is not already near the top; the trained
    policy cannot see its previous plan and becomes consistent by choosing more predictable, slower plans everywhere.
38. Best results unchanged: navtest v2 0.9027 (rl3_b0.005 + sequential EC rule, argmax base, n 64), navhard 0.441
    (rl3_b0.005, shortlist k 256 + EC rule n 64). A rule-free agent with EC training (w 2, mixture λ 10) reaches navtest
    v2 0.8983, the best without a test-time rule (previous 0.8978, β 0.02 + pairwise scorer). navhard stage two is the
    one place the EC model is better with the rule (0.561 vs 0.549), offset by stage one (0.779 vs 0.795).

### Shortlist previous plan and the EC rule on the mixture base (2026-11-14)

`rl3_ecshort2_b0.005` / `rl3_ecshort4_b0.005`: as rl3_ec2 / ec4, but the previous plan in the reward is the policy's best
in the control scorer's top 256 at t − 0.5 s (the plan the shortlist rule outputs there). Same chain as above:
predictions 15058967 / 15058984, scoring and EC rule 15058968-83 / 15058985-15059000. EC rule on the mixture base
(λ 10, control scorer; `ec_aware_selection.py --bases mixture`: candidates ranked by log p + 10 log predicted total, the
first entry is the mixture choice, tested in `tests/test_ec_aware_selection_rankings.py`) for rl3_b0.005 and all four
EC models, navtest sequential + navhard: selection 15059165-66, 71-72, 77-78, 83-84, 89-90, scoring 15059167-70, 73-76,
79-82, 85-88, 91-94 (scratchpad `submit_ec_mixture.sh`). All 64 jobs completed. Δ = paired navtest v2 difference per
scene against rl3_b0.005 with the same rule (n = 12,146); scripts: scratchpad `ec_train_results.py`, `paired_extra.py`.

| Rule | Model | navtest v1 | navtest v2 | Δ v2 vs reference | EC | EP | navhard (stage one / two) | EC stage one / two |
|---|---|---|---|---|---|---|---|---|
| argmax | rl3_b0.005 | 0.9129 | 0.8920 | | 0.780 | 0.892 | 0.365 (0.760 / 0.473) | 0.609 / 0.598 |
| | ecshort w 2 | 0.9082 | 0.8941 | +0.0021 ± 0.0016 | 0.846 | 0.883 | 0.379 (0.756 / 0.493) | 0.711 / 0.715 |
| | ecshort w 4 | 0.9029 | 0.8907 | −0.0013 ± 0.0017 | 0.862 | 0.873 | 0.400 (0.764 / 0.520) | 0.760 / 0.741 |
| mixture λ 10 | rl3_b0.005 | 0.9173 | 0.8965 | | 0.780 | 0.888 | – | – |
| | ecshort w 2 | 0.9097 | 0.8956 | −0.0009 ± 0.0015 | 0.844 | 0.881 | – | – |
| | ecshort w 4 | 0.9071 | 0.8944 | −0.0020 ± 0.0015 | 0.860 | 0.871 | – | – |
| shortlist k 256 | rl3_b0.005 | 0.9069 | 0.8836 | | 0.742 | 0.876 | 0.419 (0.774 / 0.534) | 0.524 / 0.597 |
| | ecshort w 2 | 0.9033 | 0.8847 | +0.0011 ± 0.0011 | 0.788 | 0.869 | 0.421 (0.760 / 0.546) | 0.604 / 0.625 |
| | ecshort w 4 | 0.9017 | 0.8850 | +0.0014 ± 0.0012 | 0.804 | 0.864 | 0.425 (0.770 / 0.546) | 0.618 / 0.626 |
| argmax + EC rule n 64 | rl3_b0.005 | 0.9108 | 0.9027 | | 0.915 | 0.892 | 0.393 (0.780 / 0.497) | 0.858 / 0.880 |
| | ecshort w 2 | 0.9045 | 0.8996 | −0.0031 ± 0.0017 | 0.938 | 0.883 | 0.396 (0.775 / 0.504) | 0.911 / 0.916 |
| | ecshort w 4 | 0.8993 | 0.8952 | −0.0075 ± 0.0017 | 0.946 | 0.872 | 0.413 (0.777 / 0.528) | 0.929 / 0.920 |
| shortlist + EC rule n 64 | rl3_b0.005 | 0.9045 | 0.8944 | | 0.872 | 0.873 | 0.441 (0.795 / 0.549) | 0.756 / 0.805 |
| | ecshort w 2 | 0.9009 | 0.8924 | −0.0021 ± 0.0012 | 0.888 | 0.867 | 0.447 (0.781 / 0.569) | 0.787 / 0.820 |
| | ecshort w 4 | 0.8988 | 0.8912 | −0.0033 ± 0.0012 | 0.897 | 0.862 | **0.456** (0.792 / 0.570) | 0.796 / 0.800 |
| **mixture + EC rule n 64** | rl3_b0.005 | 0.9155 | **0.9079** | | 0.917 | 0.889 | 0.435 (**0.800** / 0.535) | 0.867 / 0.869 |
| | EC w 2 | 0.9089 | 0.9046 | −0.0033 ± 0.0015 | 0.943 | 0.876 | 0.414 (0.789 / 0.525) | 0.933 / 0.912 |
| | EC w 4 | 0.9018 | 0.8981 | −0.0097 ± 0.0016 | 0.949 | 0.867 | 0.415 (0.779 / 0.523) | 0.929 / 0.930 |
| | ecshort w 2 | 0.9076 | 0.9028 | −0.0051 ± 0.0015 | 0.940 | 0.880 | 0.420 (0.792 / 0.525) | 0.920 / 0.913 |
| | ecshort w 4 | 0.9037 | 0.8996 | −0.0083 ± 0.0015 | 0.948 | 0.870 | 0.418 (0.784 / 0.530) | 0.916 / 0.937 |

mixture_v1 λ 10 (v1): ecshort 0.9113 / 0.9065. EC rule n 16 (navtest v2 / navhard), ecshort w 2 / w 4: argmax base
0.8985 / 0.393, 0.8953 / 0.410; shortlist base 0.8907 / 0.443, 0.8902 / 0.448. Mixture base n 16: rl3_b0.005
0.9055 / 0.426, EC w 2 0.9039 / 0.411, EC w 4 0.8972 / 0.416, ecshort w 2 0.9018 / 0.419, ecshort w 4 0.8988 / 0.414.

**How navhard aggregates (`run_pdm_score.py` `calculate_individual_mapping_scores`, `scene_aggregator.py`).** Stage one
has 450 scenes (225 groups of a frame and its previous frame). Each stage-two group score is a weighted mean of its
synthetic scenes with Gaussian weights exp(−d²/0.2) on the distance d between the end of the agent's stage-one plan and
each synthetic scene's start (σ ≈ 0.32 m), so in practice the synthetic scene that starts nearest to where the stage-one
plan ends; combined = mean over groups of stage one × stage two. The stage-one plan therefore decides which stage-two
scenes count, and the combined score rests on ~450 groups.
Why every frame comes in a pair: the two-frame extended comfort (EC, weight 2 of 16 in EPDMS) compares the plan at a
frame with the plan at the frame 0.5 s earlier, so each group holds a "now" frame and its "previous" frame, and the
stage-two synthetic scenes are also now / previous pairs (2,731 pairs = 5,462 frames; the synthetic now frames follow up
the real now frame, the synthetic previous frames the real previous frame). EC is computed once per pair, from the two
plans, and assigned to both frames (`SceneAggregator.aggregate_scores`); all other sub-scores are per frame. The
previous frame is therefore also a full stage-one scene with its own stage-two follow-ups: group score = (now × its
weighted stage two + previous × its weighted stage two) / 2. Every frame is predicted independently (the model does not
see its earlier plan; EC only compares the two outputs, which is what the EC rule exploits at test time). This is the
benchmark's structure, the same for every model with or without EC in training or selection: 450 + 5,462 = 5,912
predictions. Only v1 PDMS (no EC) would need single frames.

39. **Mixture + EC rule is the best navtest rule: v2 0.9079** for rl3_b0.005 (n 64; n 16 0.9055), +0.0052 ± 0.0012 over
    argmax + EC rule (0.9027) and +0.0114 ± 0.0007 over mixture alone, with EC 0.917 and EP 0.889 (argmax + rule 0.915 /
    0.892). The two gains add: mixture +0.0045 and rule +0.0107 over argmax predict +0.0152; measured +0.0159. This compares rules on
    the same model, so the paired SE is the whole uncertainty (no training-seed noise). On navhard the mixture base +
    rule gives the best stage one so far (0.800) but a weaker stage two (0.535 vs 0.549), 0.435 combined vs 0.441 with
    the shortlist base. With every EC-trained model the mixture + rule is below rl3_b0.005 (−0.003 to −0.010), as with
    the other bases (finding 37).
40. **The shortlist previous plan gives the highest navhard numbers: 0.456 (ecshort w 4) / 0.447 (w 2) with shortlist +
    EC rule n 64** vs 0.441 for rl3_b0.005 and 0.438 / 0.439 for the argmax-previous runs; also without the rule
    (shortlist k 256: 0.425 / 0.421 vs 0.419) and with argmax + rule (0.413 vs 0.393 for w 4). Per stage (official):

    | shortlist + EC rule n 64 | stage | score | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | rl3_b0.005 | one | 0.7945 | 0.958 | 0.956 | 0.996 | 0.998 | 0.765 | 0.956 | 0.929 | 0.969 | 0.756 |
    | ecshort w 4 | one | 0.7920 | 0.962 | 0.942 | 0.996 | 0.998 | 0.761 | 0.962 | 0.947 | 0.967 | 0.796 |
    | rl3_b0.005 | two | 0.5486 | 0.855 | 0.864 | 0.918 | 0.984 | 0.777 | 0.840 | 0.543 | 0.963 | 0.805 |
    | ecshort w 4 | two | **0.5696** | 0.868 | 0.870 | 0.927 | 0.988 | 0.756 | 0.861 | 0.562 | 0.962 | 0.800 |

    The gain is all in stage two (+0.021: NC +0.013, TTC +0.021, LK +0.019, DDC +0.009, EP −0.021); stage one is level
    (−0.003). Rebuilding the official aggregation from the trajectories (`exp/headroom/navhard_weighted.py`, ego poses
    from `exp/headroom/navhard_ego_states.py`; reproduces the official combined and stage scores of all four compared runs
    to 4 digits) separates the two ingredients of the stage-two score: (a) each model's per-scene scores on all 5,462
    synthetic scenes (its driving there) and (b) its stage-one endpoints, which set the weights (section "How navhard
    aggregates"). Recombined:

    | Driving in the synthetic scenes (a) | Weights from the stage-one endpoints of (b) | Stage two |
    |---|---|---|
    | rl3_b0.005 | rl3_b0.005 (official) | 0.5486 |
    | ecshort w 4 | rl3_b0.005 | 0.5475 |
    | rl3_b0.005 | ecshort w 4 | 0.5636 |
    | ecshort w 4 | ecshort w 4 (official) | 0.5696 |

    Changing who drives moves stage two by 0.001-0.006; changing whose endpoints set the weights moves it by 0.015-0.022.
    Both models drive a given synthetic scene about equally well; ecshort w 4's stage-one plans end next to synthetic
    scenes that go better for either model. It is not that its plans end closer to the synthetic starts: the nearest
    start is a median 1.03 m (rl3_b0.005) / 1.12 m (ecshort w 4) from the plan's end, within 0.5 m in 18% / 14% of the
    450 stage-one plans, and the normalized weights give the nearest scene ~70% of the weight either way (mean largest
    weight per group 0.69). The two models' endpoints differ by more than 0.1 m in 68% of stage-one frames, with almost
    the same plan length at 4 s (19.02 vs 19.13 m, paired −0.12 m), and these small shifts change which synthetic scene
    dominates. This is what navhard rewards by design (pseudo closed loop: the stage-one plan decides the state the car
    reaches), so it is a real benchmark gain, but it comes from the stage-one endpoints, not from better stage-two
    driving. Whether the training taught the model to aim for better positions or this is chance is UNKNOWN.
    Uncertainty (bootstrap over the 225 scene groups, paired): combined +0.0146 ± 0.0131, 95% [−0.011, +0.040], P(≤ 0)
    0.13; stage two +0.021 ± 0.014; ecshort w 2 +0.006 ± 0.013; ec w 4 −0.003 ± 0.014. Each training was run once, so
    seed noise comes on top. Treat 0.456 as the best measured navhard score, not a proven improvement over 0.441. **Update 2026-11-15: not
    replicated with seed 1 (finding 42).** On
    navtest the ecshort runs behave like the argmax-previous runs (w 2 +0.002 without a rule, below the reference with it).
    Submission pickle (job 15063317; trajectories identical to the scored ones, metadata as the 41.94 submission):
    `exp/submission_navhard_two_stage/rl3_ecshort4_b0.005_ep01_ec_short256_n64/submission.pkl`. Not submitted.
41. Bests after this round: navtest v2 **0.9079** (rl3_b0.005, mixture λ 10 + sequential EC rule n 64); navhard
    **0.456** (rl3_ecshort4_b0.005, shortlist k 256 + EC rule n 64; caveat in finding 40), 0.441 for rl3_b0.005 with the
    same rule; navtest v2 without a test-time rule 0.8983 (rl3_ec2_b0.005, mixture λ 10).

### Seed replicates (2026-11-15)

`rl3_b0.005_s1` and `rl3_ecshort4_b0.005_s1`: the two runs of the navhard comparison (finding 40) retrained with
`++seed=1`, recipes otherwise identical (experiments.md, "Seed replicates"). Predictions 15081432 / 15081453, scoring and
EC rule 15081433-52 / 15081454-73, all 42 completed. Δ v2 = paired navtest difference per scene against rl3_b0.005
seed 0 with the same rule. navhard as combined (stage one / two).

| Rule | Model | Seed | navtest v2 | Δ v2 | EC | EP | navhard |
|---|---|---|---|---|---|---|---|
| argmax | rl3_b0.005 | 0 / 1 | 0.8920 / 0.8925 | – / +0.0005 ± 0.0015 | 0.780 / 0.779 | 0.892 / 0.887 | 0.365 / 0.378 |
| | ecshort w 4 | 0 / 1 | 0.8907 / 0.8919 | −0.0013 / −0.0001 | 0.862 / 0.870 | 0.873 / 0.874 | 0.400 / 0.387 |
| mixture λ 10 | rl3_b0.005 | 0 / 1 | 0.8965 / 0.8957 | – / −0.0008 ± 0.0013 | | | – |
| | ecshort w 4 | 0 / 1 | 0.8944 / 0.8938 | −0.0020 / −0.0027 | | | – |
| shortlist k 256 | rl3_b0.005 | 0 / 1 | 0.8836 / 0.8836 | – / +0.0000 ± 0.0010 | | | 0.419 / 0.425 (0.782 / 0.539) |
| | ecshort w 4 | 0 / 1 | 0.8850 / 0.8837 | +0.0014 / +0.0001 | | | 0.425 / 0.421 |
| argmax + EC rule n 64 | rl3_b0.005 | 0 / 1 | 0.9027 / 0.9026 | – / −0.0001 ± 0.0016 | 0.915 / 0.918 | 0.892 / 0.887 | 0.393 / 0.410 |
| | ecshort w 4 | 0 / 1 | 0.8952 / 0.8940 | −0.0075 / −0.0087 | 0.946 / 0.948 | 0.872 / 0.873 | 0.413 / 0.400 |
| shortlist + EC rule n 64 | rl3_b0.005 | 0 / 1 | 0.8944 / 0.8951 | – / +0.0007 ± 0.0010 | | | 0.441 / **0.454** (0.809 / 0.559) |
| | ecshort w 4 | 0 / 1 | 0.8912 / 0.8902 | −0.0033 / −0.0042 | | | **0.456** / 0.444 (0.786 / 0.556) |
| mixture + EC rule n 64 | rl3_b0.005 | 0 / 1 | 0.9079 / **0.9085** | – / +0.0006 ± 0.0013 | 0.917 / 0.919 | 0.889 / 0.884 | 0.435 / 0.447 (**0.811** / 0.545) |
| | ecshort w 4 | 0 / 1 | 0.8996 / 0.8980 | −0.0083 / −0.0098 | 0.948 / 0.949 | 0.870 / 0.871 | 0.418 / 0.437 |

n 16 (navtest v2 / navhard), seed 1: rl3_b0.005 argmax base 0.9009 / 0.403, shortlist base 0.8932 / 0.447, mixture base
0.9058 / 0.444; ecshort w 4 0.8949 / 0.396, 0.8896 / 0.442, 0.8982 / 0.436. mixture_v1 λ 10 (v1): 0.9159 / 0.9062.

navhard shortlist + EC rule n 64, rebuilt aggregation with a paired bootstrap over the 225 scene groups
(`exp/headroom/navhard_weighted_seeds.py`, reproduces all four official scores):

| Comparison | Combined difference |
|---|---|
| rl3_b0.005 seed 1 − seed 0 (same recipe) | +0.0127 ± 0.0122 |
| ecshort w 4 seed 1 − seed 0 (same recipe) | −0.0118 ± 0.0083 |
| ecshort w 4 − rl3_b0.005, seed 0 | +0.0146 ± 0.0131 |
| ecshort w 4 − rl3_b0.005, seed 1 | −0.0099 ± 0.0113 |
| mean of the two seeds: ecshort w 4 (0.4498) − rl3_b0.005 (0.4475) | +0.0023 ± 0.0097 |

42. **The navhard gain of the shortlist previous plan (finding 40) does not replicate.** With seed 1 the order flips
    (ecshort w 4 0.444 vs rl3_b0.005 0.454); over the two seeds the difference is +0.002 ± 0.010. Retraining the same
    recipe with another seed moves navhard combined by ±0.012-0.013, as much as the difference that finding 40
    reported, and the stage-two weighting (finding 40) is where it moves (the stage-one endpoints change with the seed).
    So the EC reward with the shortlist previous plan has no measurable navhard effect, and differences between trained
    models on navhard below ~0.025 cannot be resolved with one seed each.
43. **navtest is reproducible across seeds.** rl3_b0.005 seed 1 matches seed 0 within the paired SE under every rule
    (|Δ| ≤ 0.0008): argmax 0.8925, mixture 0.8957, argmax + EC rule 0.9026, mixture + EC rule 0.9085 (seed 0: 0.9079).
    The finding that EC training costs navtest when the EC rule is used replicates (ecshort w 4 with argmax + rule
    −0.0075 / −0.0087, with mixture + rule −0.0083 / −0.0098), as does its EC / EP trade-off (EC 0.86-0.87, EP 0.87).
44. **Selection-rule effects replicate on navhard** (same model, larger than the seed noise): rl3_b0.005 seed 1 argmax
    0.378 → shortlist k 256 0.425 → shortlist + EC rule n 64 0.454 (seed 0: 0.365 → 0.419 → 0.441); mixture + EC rule
    gives the best stage one in both seeds (0.800 / 0.811).
45. **Bests after the seed replicates.** navtest v2: mixture λ 10 + sequential EC rule n 64 on rl3_b0.005, 0.9079 /
    0.9085 (seeds 0 / 1). navhard: shortlist k 256 + EC rule n 64, 0.441 / 0.454 for rl3_b0.005 and 0.456 / 0.444 for
    ecshort w 4: one recipe-independent level of 0.449 ± ~0.01 (mean of the four runs). The highest single run is
    0.456 (ecshort w 4 seed 0, pickle ready); choosing it among four runs on the public navhard split is a selection on
    the test set, so for the thesis report the mean and spread; for the leaderboard any of the four is legitimate as
    a single submission.
48. **Noise thresholds (from the two seeds).** navhard, two training runs with one seed each (different recipes, or
    the same recipe with another seed): the same recipe moved by +0.0127 (rl3_b0.005) and −0.0118 (ecshort w 4) under
    shortlist + EC rule n 64, so the SD of a single-seed difference is ≈ 0.012 (bootstrap over the 225 groups:
    0.008-0.012) and a difference needs ≳ 0.025 to be ~95% real; with the mean of 2 seeds per recipe ≈ 0.017. The seed
    shifts every rule of a model together (rl3_b0.005 seed 1 is +0.006 to +0.019 under all 8 rules), so it is one
    offset per trained model. Selection rules on the same model are compared paired on the same predictions and are
    much more precise: paired SE 0.002-0.005 for close shortlist-based rules (e.g. shortlist + EC n 64 vs shortlist
    +0.022 / +0.029, SE 0.004-0.005), up to 0.016 against argmax; shortlist base vs mixture base (both + EC n 64, −0.006,
    SE ≈ 0.013) is not resolved. navtest v2 is ~10× quieter: all seed-to-seed differences ≤ 0.0016 (SD ≈ 0.001,
    threshold ≈ 0.002-0.003). Caveat: the navhard seed SD rests on two replicate pairs, so the threshold itself is
    uncertain by a factor of ~2 or more. **Updated by findings 56-57 (6 seeds, 2026-11-18):** navhard SD 0.009-0.011
    (threshold ≈ 0.025 confirmed); navtest v2 SD 0.0010-0.0017 (threshold ≈ 0.004); shortlist base vs mixture base
    on navhard is now resolved (+0.014 ± 0.0035 for the shortlist base).

### Inference speed and GPU memory (2026-11-17)

`scripts/evaluation/benchmark_vision_llm_lora_inference.py` (job 15175731, `exp/benchmark/`; a first run, 15175729,
gave the policy rows within ~7%): one A100-SXM4-40GB, bf16 autocast, eval mode, real cached inputs (64 navtrain-val
scenes), 10 warm-up + 50 timed iterations per batch size. Policy = rl3_b0.005 ep 1; scorer = p3_aux_policy ep 3 (its
sub-score heads feed shortlist / mixture; same architecture, same timings within 1%).

| Batch | Latency per batch (p99) | Scenes/s | Vision tower | LLM | Peak allocated |
|---|---|---|---|---|---|
| 1 | 175 ms (194) | 5.7 | 38 ms | 134 ms | 9.2 GB |
| 2 | 178 ms (182) | 11.2 | 42 ms | 134 ms | 9.2 GB |
| 4 | 204 ms (205) | 19.6 | 57 ms | 145 ms | 9.4 GB |
| 8 | 379 ms (390) | 21.1 | 101 ms | 276 ms | 9.7 GB |
| 16 | 729-743 ms | 21.5-21.9 | 191-196 ms | 535-543 ms | 10.3 GB |
| 32 | 1.41 s | 22.7 | 371 ms | 1.03 s | 11.4 GB |
| 64 | 2.74 s | 23.3 | 729 ms | 2.01 s | 13.8 GB |

Weights 8.9 GB per model (CUDA context ~0.5 GB more in nvidia-smi). Selection over the 16,384 entries: argmax
0.007 ms, shortlist k 256 0.40 ms, mixture 0.32 ms. CPU input preparation per scene (median, one process, 50 navtest
scenes): reading + decoding the 6 camera JPEGs 174 ms, feature building (crop, resize, prompt) 20 ms.

49. **Real-time budget at batch 1 (one A100, sequential):** argmax ≈ 195 ms (feature building + policy; 5 Hz),
    shortlist / mixture ≈ 370 ms (+ scorer forward; 2.7 Hz); +174 ms when the inputs come from JPEG files instead of
    camera frames. Both fit NAVSIM's 2 Hz (500 ms; shortlist / mixture with JPEG decoding only if decoding overlaps the
    previous frame's GPU work), none fits 10 Hz (the LLM alone takes 134 ms). The LLM is ~77% of the forward, the
    vision tower ~22%. The EC rule's comfort check (≤ 64 candidates against the previous plan) was not timed.
50. **Batching:** batch 2 costs about the same as batch 1 (178 vs 175 ms) and throughput saturates at ~21-23 scenes/s
    from batch 8, so the GPU is under-used at batch 1. Policy and scorer share the frozen Qwen base and differ only in
    LoRA adapters and heads: one base with two adapters saves memory (~9 GB instead of ~18 GB) but not time, since
    different LoRA weights in every adapted layer still need two forward passes. Only running the two as one batch-2
    forward with a different adapter per sample could bring shortlist / mixture near ~180 ms (estimate from the
    batch-2 latency; mixed-adapter batching is not implemented or measured).
    Memory is no constraint: 9.2 GB at batch 1, 13.8 GB at batch 64.

### Choosing the selection rule on the warmup split (2026-11-17)

The public `warmup_two_stage` split (8 scene groups = 16 stage-one frames (now + previous, 0.5 s apart) and 204
synthetic frames in 102 now / previous pairs, i.e. 220 predictions = 3.7% of navhard's 5,912 (groups 3.6%, stage one
3.6%, stage two 3.7%; 12.8 synthetic pairs per group vs 12.1); meant for checking a submission before the leaderboard) is a subset of navhard: all 16 stage-one tokens, all 204 synthetic tokens and the 7
logs are in `navhard_two_stage`. So its scores can be read from the navhard runs, and a rule chosen on it has seen 8 of
the 225 navhard groups. `exp/headroom/rule_choice_subsets.py` rebuilds the per-group combined score as the official
aggregation does (rebuilt totals equal the official ones to 4 decimals) for 8 rules on rl3_b0.005 (EC rule: control
scorer, as all EC-rule runs).

| Rule | navhard seed 0 | warmup seed 0 | navhard seed 1 | warmup seed 1 |
|---|---|---|---|---|
| argmax | 0.365 | 0.358 | 0.378 | 0.389 |
| shortlist k 256 | 0.419 | 0.439 | 0.425 | 0.505 |
| EC rule n 16 / 64, argmax base | 0.391 / 0.394 | 0.388 / 0.428 | 0.403 / 0.410 | 0.425 / 0.426 |
| EC rule n 16 / 64, shortlist base | 0.437 / **0.441** | 0.446 / **0.452** | 0.447 / **0.454** | 0.518 / **0.525** |
| EC rule n 16 / 64, mixture base | 0.426 / 0.435 | 0.389 / 0.401 | 0.445 / 0.447 | 0.510 / 0.512 |

Random subsets of k groups (20,000 draws; seed 0 / seed 1): how often the subset's best rule is the navhard best, and
the mean loss on navhard when it is not.

| k groups | picks the navhard best | mean loss | P(loss > 0.01) |
|---|---|---|---|
| 8 | 36% / 35% | 0.011 / 0.011 | 26% / 17% |
| 16 | 41% / 44% | 0.008 / 0.008 | 19% / 10% |
| 32 | 45% / 50% | 0.005 / 0.005 | 10% / 4% |
| 112 (split half) | 65% / 69% | 0.002 / 0.002 | 0% / 0% |

46. **The warmup split picks the rule that is best on all of navhard, in both seeds: shortlist k 256 + EC rule n 64.**
    Protocol for the thesis: choose the rule on warmup (as the benchmark allows), report navhard. Without the 8 warmup
    groups (217 groups) the chosen rule scores 0.4407 / 0.4512 (seeds 0 / 1; all 225: 0.4411 / 0.4538), so the overlap
    does not inflate the result.
47. **8 groups are noisy for close rules.** A random 8-group subset picks the navhard-best rule only ~35% of the time;
    the misses are mostly the next rules (mixture base n 64: −0.006, shortlist base n 16: −0.004 to −0.007), so the mean
    loss is 0.011; the two most-picked rules are shortlist base n 64 and mixture base n 64 (64% / 62% of the picks
    together). The warmup agreement is therefore partly luck for the top-2 ordering
    (paired SE over 8 groups: 0.013-0.015 for shortlist base n 16, 0.064-0.073 for mixture base n 64); the gains of the
    EC rule and of the shortlist over argmax are established on all 225 groups (SE 0.002-0.016).

### RL-3 with the GRPO objective (2026-11-18)

Runs: experiments.md "RL-3 with the GRPO objective". Six runs `rl3_grpo_b{0.05,0.02,0.005}` (std-normalized group
advantage) and `rl3_drgrpo_b…` (no std division), G = 32, otherwise exactly the exact-objective `rl3_b<β>` (seed 0).
All six trainings completed (2 h 21-24 min each, as the exact runs). Evaluation with the start model's own heads
(`SCORES_PRED=p3_aux_policy_ep03`); all 42 scoring and 18 v1-comfort jobs completed (12 scoring jobs only after a
resubmission with longer limits, finding 58). Numbers: `exp/rl3_logs/collect_grpo_seeds.py`. Val = the last validation
(end of epoch 2, 18,179 navtrain-val scenes; W&B summaries), same columns as experiments.md "RL-3 / RL-4".

Val proxy (start p3_aux_policy ep 3: argmax 0.7888, E_π[r] 0.7166, argmax = 0 13.2%):

| β | Objective | val argmax | E_π[r] | argmax = 0 | KL to start | entropy |
|---|---|---|---|---|---|---|
| 0.05 | exact | 0.8466 | 0.8187 | 7.4% | 0.66 | – |
| | Dr. GRPO | 0.8460 | 0.8182 | 7.3% | 0.68 | 4.06 |
| | GRPO | 0.7832 | 0.7755 | 17.0% | 7.97 | 2.27 |
| 0.02 | exact | 0.8533 | 0.8345 | 7.0% | 1.17 | – |
| | Dr. GRPO | 0.8505 | 0.8313 | 7.1% | 1.17 | 3.71 |
| | GRPO | 0.7701 | 0.7628 | 18.3% | 10.36 | 2.02 |
| 0.005 | exact (seed 0; seeds 1-5: 0.8542-0.8577, KL 2.64-3.15) | 0.8548 | 0.8455 | 6.6% | 3.06 | 3.05-3.12 |
| | Dr. GRPO | 0.8579 | 0.8463 | 6.2% | 2.87 | 3.15 |
| | GRPO | 0.7616 | 0.7571 | 19.2% | 12.73 | 1.75 |

Official (navtest v2 EPDMS; v1 = v1 PDMS with the v1 comfort term; navhard combined (stage one / stage two)):

| β | Objective | navtest argmax (EC, scored 0) | shortlist k 256 | mixture λ 10 | v1 argmax / shortlist_v1 / mixture_v1 | navhard argmax | navhard shortlist k 256 |
|---|---|---|---|---|---|---|---|
| 0.05 | exact | 0.893 (0.829) | 0.880 | – | 0.9106 / 0.9063 / 0.9133 | 0.351 | 0.405 |
| | Dr. GRPO | 0.8902 (0.829, 4.3%) | 0.8802 | 0.8909 | 0.9084 / 0.9058 / 0.9112 | 0.334 (0.779 / 0.432) | 0.404 (0.754 / 0.528) |
| | GRPO | 0.8058 (0.635, 13.7%) | 0.8477 | 0.8267 | 0.8528 / 0.8883 / 0.8727 | 0.228 (0.626 / 0.355) | 0.348 (0.705 / 0.483) |
| 0.02 | exact | 0.8950 (0.821) | 0.881 | 0.8944 | 0.9138 / 0.9088 / 0.9151 | 0.335 | 0.413 |
| | Dr. GRPO | 0.8946 (0.815, 3.9%) | 0.8821 | 0.8934 | 0.9141 / 0.9074 / 0.9155 | 0.358 (0.782 / 0.454) | 0.400 (0.759 / 0.520) |
| | GRPO | 0.7836 (0.561) | 0.8373 | 0.8125 | 0.8387 / 0.8830 / 0.8663 | 0.242 (0.611 / 0.375) | 0.355 (0.702 / 0.494) |
| 0.005 | exact, mean ± SD of 6 seeds | 0.8918 ± 0.0017 (0.780) | 0.8816 ± 0.0011 | 0.8938 ± 0.0014 | 0.9141 / 0.9097 / 0.9169 (SD ≤ 0.001) | 0.370 ± 0.010 | 0.410 ± 0.009 (0.763 / 0.529) |
| | Dr. GRPO | 0.8924 (0.783, 3.9%) | 0.8809 | 0.8939 | 0.9139 / 0.9089 / 0.9159 | 0.350 (0.751 / 0.461) | 0.393 (0.755 / 0.516) |
| | GRPO | 0.7750 (0.488, 15.6%) | 0.8333 | 0.8026 | 0.8347 / 0.8849 / 0.8609 | 0.215 (0.565 / 0.365) | 0.349 (0.689 / 0.491) |

53. **Standard GRPO fails in RL-3; the std normalization is the cause.** At every β it leaves the start policy far
    behind (KL 8-13 vs 0.7-3 for the exact objective), its val argmax ends *below* the start (0.76-0.78 vs 0.789) with
    17-19% zero-score picks, and lower β makes it worse. Officially it loses 0.09-0.12 on navtest argmax (0.775-0.806
    vs 0.89), EC falls to 0.49-0.64 and 14-16% of scenes score 0; navhard argmax drops to 0.22-0.24. The scorer repairs
    part of it (shortlist k 256: 0.833-0.848 navtest, 0.35 navhard) but not all; v1 loses 0.02-0.08. LIKELY mechanism:
    dividing by the group's reward SD gives every group advantages of size ~1, so groups whose 32 samples have almost
    equal rewards (most navtrain scenes, rewards near 1) get their tiny, mostly noise differences blown up to the same
    weight as informative groups, and the KL term, not rescaled, becomes β/SD times weaker. This is the same failure as
    on the policy head alone (RL-1, checkpoint_evaluation.md finding 48), now larger because the LoRA can move further.
54. **Dr. GRPO reproduces the exact objective.** Val: within ±0.003 of the exact run at every β with the same KL
    (0.68 / 1.17 / 2.87 vs 0.66 / 1.17 / 3.06), as expected (its gradient is an unbiased estimate of the exact
    objective's, `test_rl_loss_terms_grpo`). navtest: β 0.005 lies inside the 6-seed range of the exact objective on
    argmax (0.8924 vs 0.8886-0.8934), shortlist (0.8809 vs 0.8801-0.8829), mixture (0.8939 vs 0.8916-0.8951) and v1
    argmax / shortlist_v1 (0.9139 / 0.9089); v1 mixture_v1 is 1 SD below (0.9159 vs 0.9169 ± 0.0010). β 0.02 and 0.05
    equal their exact runs within 0.003 on every navtest number.
55. **navhard: no reliable difference.** Dr. GRPO's shortlist k 256 (0.393 / 0.400 / 0.404 for β 0.005 / 0.02 / 0.05)
    is 0.001-0.017 below the exact runs (0.410 ± 0.009 over 6 seeds / 0.413 / 0.405), within the seed spread (the
    lowest exact seed is 0.397, finding 56); argmax likewise (0.334-0.358 vs 0.335-0.370). The exact objective stays the
    RL-3 objective of the thesis model: it is never worse and needs no sampling.

### Seeds of rl3_b0.005: 6 trainings, spread of every metric (2026-11-18)

Six trainings of the same recipe (seeds 0 and 1 from before; seeds 2-5 run 2026-11-17, 2 h 21-24 min each; val argmax
0.8542-0.8579, KL 2.64-3.15). Own heads (`p3_aux_policy_ep03`) for every scorer-based rule; all 105 scoring jobs of
this batch completed (27 after a resubmission, finding 58). `exp/rl3_logs/collect_grpo_seeds.py`.

| Metric (rl3_b0.005, own heads; 6 seeds) | mean | SD | range | seed 0 |
|---|---|---|---|---|
| navtest v2 argmax (EC 0.780 ± 0.009) | 0.8918 | 0.0017 | 0.8886-0.8934 | 0.8920 |
| navtest v2 shortlist k 256 | 0.8816 | 0.0011 | 0.8801-0.8829 | 0.8829 |
| navtest v2 mixture λ 10 | 0.8938 | 0.0014 | 0.8916-0.8951 | 0.8948 |
| navtest v2 EC rule (sequential), argmax base n 16 / n 64 | 0.9006 / 0.9021 | 0.0017 / 0.0013 | 0.8973-0.9029 | 0.9012 / 0.9027 |
| navtest v2 EC rule, shortlist base n 16 / n 64 | 0.8912 / 0.8929 | 0.0012 / 0.0010 | 0.8896-0.8941 | 0.8925 / 0.8941 |
| navtest v2 EC rule, mixture base n 16 / **n 64** | 0.9033 / **0.9056** | 0.0013 / 0.0014 | 0.9013-0.9077 | 0.9041 / 0.9062 |
| navtest v1 (v1 comfort) argmax | 0.9141 | 0.0006 | 0.9131-0.9150 | 0.9143 |
| navtest v1 (v1 comfort) shortlist_v1 k 256 | 0.9097 | 0.0008 | 0.9089-0.9110 | 0.9110 |
| navtest v1 (v1 comfort) **mixture_v1 λ 10** | **0.9169** | 0.0010 | 0.9151-0.9178 | 0.9178 |
| navhard argmax | 0.370 | 0.010 | 0.355-0.383 | 0.365 |
| navhard shortlist k 256 (stage one 0.763 ± 0.005, two 0.529 ± 0.011) | 0.410 | 0.009 | 0.397-0.420 | 0.412 |
| navhard EC rule, argmax base n 16 / n 64 | 0.396 / 0.402 | 0.011 / 0.011 | 0.377-0.411 | 0.391 / 0.393 |
| navhard EC rule, shortlist base n 16 / **n 64** | 0.433 / **0.438** | 0.011 / 0.011 | 0.418-0.455 | 0.434 / 0.434 |
| navhard EC rule, mixture base n 16 / n 64 | 0.419 / 0.425 | 0.014 / 0.012 | 0.395-0.440 | 0.417 / 0.422 |

Per seed (same runs; seeds 0 / 1 are `rl3_b0.005_ep01` / `rl3_b0.005_s1_ep01`, all files checked to come from the
own-scorer jobs; best seed in bold):

| Metric (own heads) | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
|---|---|---|---|---|---|---|
| navtest v2 argmax | 0.8920 | 0.8925 | 0.8886 | 0.8916 | 0.8931 | **0.8934** |
| navtest EC (argmax) | 0.780 | 0.779 | 0.763 | 0.786 | **0.787** | **0.787** |
| navtest v2 shortlist k 256 | **0.8829** | 0.8826 | 0.8801 | 0.8805 | 0.8816 | 0.8818 |
| navtest v2 mixture λ 10 | 0.8948 | 0.8945 | 0.8916 | 0.8925 | **0.8951** | 0.8943 |
| navtest v2 EC rule, argmax base n 16 | 0.9012 | 0.9009 | 0.8973 | 0.9009 | **0.9020** | 0.9015 |
| navtest v2 EC rule, argmax base n 64 | 0.9027 | 0.9026 | 0.8994 | 0.9028 | **0.9029** | 0.9025 |
| navtest v2 EC rule, shortlist base n 16 | **0.8925** | **0.8925** | 0.8901 | 0.8896 | 0.8909 | 0.8913 |
| navtest v2 EC rule, shortlist base n 64 | **0.8941** | 0.8938 | 0.8923 | 0.8916 | 0.8924 | 0.8932 |
| navtest v2 EC rule, mixture base n 16 | 0.9041 | **0.9049** | 0.9013 | 0.9023 | 0.9041 | 0.9031 |
| navtest v2 EC rule, mixture base n 64 | 0.9062 | **0.9077** | 0.9040 | 0.9042 | 0.9062 | 0.9056 |
| navtest v1 argmax | 0.9143 | 0.9142 | 0.9131 | 0.9140 | 0.9141 | **0.9150** |
| navtest v1 shortlist_v1 k 256 | **0.9110** | 0.9104 | 0.9095 | 0.9094 | 0.9093 | 0.9089 |
| navtest v1 mixture_v1 λ 10 | **0.9178** | 0.9171 | 0.9165 | 0.9151 | 0.9177 | 0.9171 |
| navhard argmax | 0.365 | 0.378 | 0.375 | 0.368 | **0.383** | 0.355 |
| navhard shortlist k 256 | 0.412 | **0.420** | 0.417 | 0.410 | 0.397 | 0.404 |
| (stage one / stage two) | 0.764 / 0.533 | 0.773 / 0.537 | 0.758 / 0.537 | 0.761 / 0.534 | 0.762 / 0.510 | 0.762 / 0.523 |
| navhard EC rule, argmax base n 16 | 0.391 | 0.403 | 0.401 | 0.396 | **0.407** | 0.377 |
| navhard EC rule, argmax base n 64 | 0.393 | 0.410 | **0.411** | 0.401 | 0.410 | 0.385 |
| navhard EC rule, shortlist base n 16 | 0.434 | **0.449** | 0.439 | 0.428 | 0.418 | 0.429 |
| navhard EC rule, shortlist base n 64 | 0.434 | **0.455** | 0.446 | 0.434 | 0.424 | 0.436 |
| (stage one / stage two) | 0.781 / 0.547 | 0.794 / 0.567 | 0.777 / 0.563 | 0.778 / 0.552 | 0.778 / 0.534 | 0.782 / 0.553 |
| navhard EC rule, mixture base n 16 | 0.417 | **0.435** | 0.432 | 0.421 | 0.412 | 0.395 |
| navhard EC rule, mixture base n 64 | 0.422 | **0.440** | 0.435 | 0.429 | 0.416 | 0.406 |
| (stage one / stage two) | 0.797 / 0.524 | 0.806 / 0.541 | 0.792 / 0.545 | 0.804 / 0.532 | 0.801 / 0.514 | 0.802 / 0.506 |

**The navhard 0.456 in context.** The earlier navhard bests (finding 45: 0.441 / 0.454 for rl3_b0.005 seeds 0 / 1,
0.456 for ecshort w 4 seed 0) used the control scorer (p3_aux_ctrl ep 2); the table above uses the own heads. On the two
seeds with both, the control scorer is +0.007 / −0.001 with shortlist + EC n 64 (own 0.434 / 0.455) and +0.007 / +0.005
with shortlist k 256 (own 0.412 / 0.420 vs 0.419 / 0.425). With the own heads the best of the six seeds is seed 1,
**0.455** (stage one 0.794 / two 0.567; the 0.456 run: 0.786 / 0.556), i.e. the same level is reached by the plain
recipe. It is the maximum of six runs: for six draws from 0.438 ± 0.011 the expected maximum is ≈ 0.452, so 0.455-0.456
is what a best-of-several pick gives, not a better recipe. Seeds 2-5 were not scored with the control scorer.

56. **Seed spread with 6 seeds** (replaces the 2-pair estimate of finding 48). navhard: SD 0.009-0.011 for the
    shortlist and EC rules (0.014 for the mixture base), so two single-seed runs differ by SD ≈ 0.013 and need
    ≳ 0.025 to be ~95% real (mean of 2 seeds each: ≈ 0.018) — finding 48's estimate holds. The navhard spread is in
    stage two (SD 0.011; stage one 0.005). navtest v2: SD 0.0010-0.0017 (argmax the largest, through EC 0.763-0.787),
    threshold for a single-seed difference ≈ 0.004, about twice finding 48's estimate; navtest v1 is the quietest
    (SD 0.0006-0.0010, threshold ≈ 0.002-0.003). Seed 0, the run all earlier comparisons used, is typical: within 1 SD
    everywhere except the navtest shortlist rules (v2 +1.2 SD, v1 +1.6 SD, ~+0.0013).
57. **The rule ranking holds in every seed (6 / 6):** navtest mixture + EC n 64 > mixture > argmax > shortlist; navhard
    EC rule on the shortlist base > shortlist > argmax, with n 64 ≥ n 16 (seed 0: tie). The two rules finding 48 could
    not separate are now resolved: on navhard the shortlist base beats the mixture base (both + EC n 64) in all 6 seeds,
    +0.014 ± 0.0035 (mean ± SE over seeds), and on navtest the mixture base beats the argmax base (both + EC n 64) in
    all 6, +0.0035. Expected thesis numbers of this recipe (6-seed mean ± SD): navtest v2 0.9056 ± 0.0014 (mixture + EC
    n 64), navtest v1 0.9169 ± 0.0010 (mixture_v1), navhard 0.438 ± 0.011 (shortlist + EC n 64), all with the own heads.
58. **Scoring time limits.** Completed scoring jobs took 16-28 min (navtest) and 29-36 min (navhard) with 4 CPUs; the
    30 / 50 min limits failed 43 jobs, all on six nodes (x1001c3s3b0n0, c4s0b0n0, c3s0b0n0, c0s0b0n0, c5s0b0n0,
    c6s0b0n1) where 6-10 of these jobs started at once; they were 35-99% through (navtest seed 5 argmax: 12,124 of
    12,146 scenes). Nothing failed in the code. Resubmitted with 60 / 90 min (`exp/rl3_logs/resubmit_timeouts_2026-11-18.sh`,
    jobs 15199720-15199772): all 53 completed, longest 37 min.

## Results (re-evaluation 2026-11-08: split pipeline, batch 64, bf16)

All models through the same pipeline (section "Split evaluation pipeline"): one GPU job per checkpoint saves the
policy and sub-score logits (batch 64, bf16), CPU jobs apply each selection and score with the unchanged NAVSIM
scripts. 0 failed scenes in every run (navtest 12,146 / navhard 5,912). Jobs: GPU 14926360 (p3_aux_policy) and
14927392-14927429, CPU 14926657/58, 14927215, 14927280-84, 14927393-14927432 (list:
`$NAVSIM_EXP_ROOT/predictions/logs/reevaluation_2026-11-08.txt`). Outputs:
`pdm_navtest/<model>_<selection>[_k<k>|_lam10]_non_reactive_split/`, `pdm_navhard_two_stage/<model>_<selection>…_reactive_split/`
(p3_aux_policy k = 256 navhard: `…_split_cpu4_fastload/`).

| Model | Aux | Selection | **v1 PDMS** navtest (v1 comfort) | **v2 EPDMS** navtest | **v2 EPDMS** navhard | navhard stage 1 / stage 2 | v2 EC | v2 scored 0 |
|---|---|---|---|---|---|---|---|---|
| p3_policy ep 7 | – | argmax | 0.794 | 0.775 | 0.189 | 0.569 / 0.317 | 0.804 | 16.3% |
| p3_heads ep 5 | – | heads | 0.871 | 0.804 | 0.342 | 0.700 / 0.480 | 0.217 | 5.3% |
| p3_aux ep 5 | ✓ | heads | 0.879 | 0.812 | 0.363 | 0.696 / 0.509 | 0.229 | 4.8% |
| p3_combo_balanced ep 8 | – | argmax | 0.833 | 0.814 | 0.189 | 0.611 / 0.321 | 0.815 | 12.1% |
| p3_combo_balanced_aux ep 9 | ✓ | heads | 0.852 | 0.787 | 0.300 | 0.656 / 0.450 | 0.172 | 5.5% |
| p3_combo_balanced_aux ep 9 | ✓ | argmax | 0.843 | 0.824 | 0.214 | 0.639 / 0.340 | 0.818 | 11.1% |
| p3_combo_balanced_aux ep 9 | ✓ | shortlist k = 64 | 0.867 | 0.838 | 0.322 | 0.675 / 0.467 | 0.608 | 5.3% |
| p3_combo_balanced_aux ep 9 | ✓ | shortlist k = 256 | 0.869 | 0.845 | 0.317 | 0.675 / 0.462 | 0.695 | 5.8% |
| p3_combo_balanced_aux ep 9 | ✓ | mixture λ = 10 | 0.869 | 0.847 | 0.250 | 0.662 / 0.386 | 0.808 | 8.2% |
| p3_heads_policy ep 2 | – | shortlist k = 64 | 0.885 | 0.853 | 0.359 | 0.722 / 0.483 | 0.668 | 5.4% |
| p3_aux_policy ep 3 | ✓ | argmax | 0.863 | 0.843 | 0.254 | 0.664 / 0.374 | 0.815 | 9.0% |
| p3_aux_policy ep 3 | ✓ | shortlist k = 64 | 0.891 | 0.861 | 0.375 | 0.723 / 0.504 | 0.684 | 4.8% |
| p3_aux_policy ep 3 | ✓ | shortlist k = 256 | 0.892 | 0.867 | **0.386** | 0.737 / 0.517 | 0.756 | 4.9% |
| **p3_aux_policy ep 3 + RL β = 0.02** | ✓ | **shortlist k = 256** | **0.899** | **0.872** | 0.383 | **0.741** / 0.509 | 0.742 | **4.5%** |
| p3_aux_policy ep 3 + RL β = 0.005 | ✓ | shortlist k = 256 | 0.898 | 0.870 | 0.385 | 0.741 / 0.503 | 0.727 | 4.8% |

v1 from the `_v1` variant of the selection (`score_heads_v1`, `shortlist_v1`, `mixture_v1`; argmax: the same run), v2
and navhard from the v2 variant. v1 with the v1 comfort term (2026-11-18); the heads rows lose 0.010-0.011 against the
HC version (0.882 / 0.889 / 0.863), so the policy head is now clearly better on v1 too (findings 51-52). The v1 column
of the paired table below is the HC version; the v1-comfort pairs are in section "v1 PDMS with the v1 comfort term for
every model".

Paired differences on navtest (12,146 scenes, mean ± standard error):

| Comparison | v2 | v1 (HC version) |
|---|---|---|
| RL β = 0.02 − p3_aux_policy (k = 256) | +0.0047 ± 0.0011 | +0.0073 ± 0.0012 |
| RL β = 0.005 − p3_aux_policy (k = 256) | +0.0025 ± 0.0014 | +0.0064 ± 0.0014 |
| p3_aux_policy k = 256 − k = 64 | +0.0067 ± 0.0014 | −0.0007 ± 0.0014 |
| p3_aux_policy k = 256 − p3_aux heads | +0.0552 ± 0.0018 | +0.0022 ± 0.0019 |
| p3_aux_policy k = 256 − p3_combo_balanced_aux k = 256 | +0.0225 ± 0.0020 | +0.0192 ± 0.0021 |
| p3_aux_policy k = 64 − p3_heads_policy k = 64 | +0.0079 ± 0.0017 | +0.0063 ± 0.0018 |
| p3_aux heads − p3_heads heads | +0.0078 ± 0.0016 | +0.0076 ± 0.0018 |
| p3_aux_policy argmax − p3_combo_balanced_aux argmax | +0.0192 ± 0.0026 | +0.0200 ± 0.0026 |
| p3_policy argmax − p3_combo_balanced_aux argmax | −0.0494 ± 0.0028 | −0.0482 ± 0.0028 |

Against the previous table (combined runs, batch 1, below): every navtest number within ±0.002 (no ranking changes), navhard
numbers within ±0.012 (largest: p3_combo_balanced_aux shortlist k = 64, 0.334 → 0.322; p3_aux_policy argmax 0.263 →
0.254). This is bf16 tie-breaking (section "Split evaluation pipeline"), which navhard amplifies: its score is a
product of two stages averaged over only 225 scene groups. Treat navhard differences below ~0.01 as ties: the three
p3_aux_policy k = 256 variants (0.383-0.386) are equal on navhard; RL's gain is a navtest result. The sub-score tables,
the selection-rule and ablation sections below still quote the previous runs; their conclusions hold for these numbers.

## Results (previous pipeline: combined runs, batch 1, fixed filter, 2026-11-07)

| Model | Aux | Selection | **v1 PDMS** navtest | **v2 EPDMS** navtest | **v2 EPDMS** navhard | navhard stage 1 / stage 2 |
|---|---|---|---|---|---|---|
| p3_policy ep 7 | – | argmax | 0.793 | 0.775 | 0.195 | 0.583 / 0.316 |
| p3_heads ep 5 | – | heads | 0.882 | 0.804 | 0.343 | 0.699 / 0.483 |
| p3_aux ep 5 | ✓ | heads | 0.889 | 0.812 | 0.361 | 0.692 / 0.509 |
| p3_combo_balanced ep 8 | – | argmax | 0.832 | 0.815 | 0.194 | 0.614 / 0.319 |
| p3_combo_balanced_aux ep 9 | ✓ | heads | 0.863 | 0.786 | 0.301 | 0.654 / 0.451 |
| p3_combo_balanced_aux ep 9 | ✓ | argmax | 0.842 | 0.826 | 0.210 | 0.625 / 0.328 |
| p3_combo_balanced_aux ep 9 | ✓ | shortlist k = 64 | 0.873 | 0.838 | 0.334 | 0.686 / 0.475 |
| p3_combo_balanced_aux ep 9 | ✓ | shortlist k = 256 | 0.872 | 0.846 | 0.311 | 0.677 / 0.456 |
| p3_combo_balanced_aux ep 9 | ✓ | mixture λ = 10 | 0.866 | 0.847 | 0.246 | 0.655 / 0.386 |
| p3_heads_policy ep 2 | – | shortlist k = 64 | 0.887 | 0.854 | 0.360 | 0.723 / 0.483 |
| p3_aux_policy ep 3 | ✓ | argmax | 0.861 | 0.844 | 0.263 | 0.676 / 0.389 |
| p3_aux_policy ep 3 | ✓ | shortlist k = 64 | 0.891 | 0.860 | 0.374 | 0.724 / 0.501 |
| p3_aux_policy ep 3 | ✓ | shortlist k = 256 | 0.893 | 0.867 | 0.382 | 0.740 / 0.511 |
| **p3_aux_policy ep 3 + RL β = 0.02** | ✓ | **shortlist k = 256** | **0.899** | **0.872** | 0.382 | 0.739 / 0.512 |
| p3_aux_policy ep 3 + RL β = 0.005 | ✓ | shortlist k = 256 | 0.898 | 0.869 | **0.384** | 0.736 / 0.508 |

Heads rows: v1 with `score_heads_v1`, v2 and navhard with `score_heads`; shortlist / mixture rows likewise with the
`_v1` variant for v1 (section "Selection rules" below). Argmax rows: v1 and v2 from the same run. navtest: 12,146
scenes, navhard: 5,912 (450 real + 5,462 synthetic); 0 failed scenes in every run. The `_policy` runs are described in
the section "Decoupled policy head" below; p3_policy is the phase 3 run with imitation only (no sub-score heads).

### navtest sub-scores

v2 runs (navtest one stage, the selection of the v2 column):

| Model, selection | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | scored 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| p3_heads, score_heads | 0.969 | 0.978 | 0.994 | 0.998 | 0.843 | 0.959 | 0.932 | 0.977 | 0.215 | 5.4% |
| p3_aux, score_heads | 0.972 | 0.981 | 0.994 | 0.998 | 0.847 | 0.962 | 0.934 | 0.979 | 0.229 | 4.8% |
| p3_combo_balanced, argmax | 0.968 | 0.909 | 0.980 | 0.997 | 0.873 | 0.958 | 0.945 | 0.983 | 0.814 | 12.1% |
| p3_combo_balanced_aux, score_heads | 0.976 | 0.970 | 0.994 | 0.999 | 0.806 | 0.961 | 0.925 | 0.975 | 0.171 | 5.5% |
| p3_combo_balanced_aux, argmax | 0.969 | 0.919 | 0.983 | 0.997 | 0.873 | 0.960 | 0.950 | 0.982 | 0.819 | 11.0% |
| p3_policy, argmax | 0.964 | 0.871 | 0.975 | 0.997 | 0.872 | 0.954 | 0.942 | 0.982 | 0.804 | 16.3% |
| p3_heads_policy, shortlist k = 64 | 0.972 | 0.976 | 0.993 | 0.998 | 0.852 | 0.964 | 0.940 | 0.979 | 0.670 | 5.4% |
| p3_aux_policy, argmax | 0.972 | 0.939 | 0.985 | 0.998 | 0.874 | 0.964 | 0.954 | 0.983 | 0.813 | 9.0% |
| p3_aux_policy, shortlist k = 64 | 0.973 | 0.979 | 0.994 | 0.998 | 0.856 | 0.967 | 0.939 | 0.980 | 0.682 | 4.9% |
| p3_aux_policy, shortlist k = 256 | 0.976 | 0.975 | 0.993 | 0.999 | 0.858 | 0.969 | 0.943 | 0.980 | 0.751 | 5.0% |
| p3_aux_policy + RL β = 0.02, k = 256 | 0.976 | 0.980 | 0.993 | 0.999 | 0.862 | 0.970 | 0.943 | 0.979 | 0.741 | 4.5% |
| p3_aux_policy + RL β = 0.005, k = 256 | 0.972 | 0.981 | 0.994 | 0.999 | 0.868 | 0.965 | 0.943 | 0.979 | 0.728 | 4.8% |

v1 runs with `score_heads_v1` (v1 / v2 of the same run):

| Model | NC | DAC | EP | TTC | HC | **v1** | v2 of this run | v1 with `score_heads` |
|---|---|---|---|---|---|---|---|---|
| p3_heads | 0.965 | 0.973 | 0.871 | 0.957 | 0.979 | **0.882** | 0.793 | 0.878 |
| p3_aux | 0.968 | 0.978 | 0.875 | 0.959 | 0.980 | **0.889** | 0.802 | 0.885 |
| p3_combo_balanced_aux | 0.972 | 0.961 | 0.840 | 0.958 | 0.978 | **0.863** | 0.778 | 0.860 |

v1 runs with `shortlist_v1` (v1 / v2 of the same run; last column: v1 of the run with `shortlist`, the v2 formula):

| Model, k | NC | DAC | EP | TTC | HC | **v1** | v2 of this run | v1 with `shortlist` |
|---|---|---|---|---|---|---|---|---|
| p3_heads_policy, k = 64 | 0.970 | 0.972 | 0.873 | 0.961 | 0.981 | **0.887** | 0.849 | 0.883 |
| p3_aux_policy, k = 64 | 0.969 | 0.976 | 0.876 | 0.963 | 0.981 | **0.891** | 0.853 | 0.890 |
| p3_aux_policy, k = 256 | 0.974 | 0.975 | 0.872 | 0.967 | 0.982 | **0.893** | 0.866 | 0.889 |
| p3_aux_policy + RL β = 0.02, k = 256 | 0.975 | 0.980 | 0.875 | 0.968 | 0.981 | **0.899** | 0.872 | 0.895 |
| p3_aux_policy + RL β = 0.005, k = 256 | 0.970 | 0.980 | 0.880 | 0.963 | 0.981 | **0.898** | 0.869 | 0.894 |

### navhard sub-scores (stage one / stage two)

Official values: the three summary rows at the end of each navhard CSV (`extended_pdm_score_stage_one`,
`extended_pdm_score_stage_two`, `extended_pdm_score_combined`), written by `run_pdm_score.py`. Stage two is weighted per
scene group by how close each synthetic start lies to the endpoint of the agent's stage-one plan (Gaussian kernel,
σ² = 0.1 m², `SceneAggregator.calculate_pseudo_closed_loop_weights`). All 83 navhard runs:
`exp/headroom/navhard_stage_summary.csv`. Filter on (leaderboard scoring) in every row.

| Model, selection | combined | stage one | stage two | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p3_heads, heads | 0.343 | 0.699 | 0.483 | 0.956 / 0.872 | 0.951 / 0.842 | 0.990 / 0.918 | 0.998 / 0.985 | 0.724 / 0.731 | 0.953 / 0.861 | 0.927 / 0.545 | 0.960 / 0.948 | 0.102 / 0.190 |
| p3_aux, heads | 0.361 | 0.692 | 0.509 | 0.947 / 0.885 | 0.944 / 0.875 | 0.992 / 0.922 | 0.996 / 0.984 | 0.729 / 0.731 | 0.949 / 0.872 | 0.936 / 0.542 | 0.964 / 0.955 | 0.120 / 0.160 |
| p3_combo_balanced, argmax | 0.194 | 0.614 | 0.319 | 0.927 / 0.762 | 0.751 / 0.623 | 0.968 / 0.777 | 0.993 / 0.984 | 0.842 / 0.847 | 0.909 / 0.747 | 0.911 / 0.453 | 0.976 / 0.954 | 0.707 / 0.619 |
| p3_combo_balanced_aux, heads | 0.301 | 0.654 | 0.451 | 0.949 / 0.874 | 0.907 / 0.798 | 0.990 / 0.927 | 0.998 / 0.983 | 0.686 / 0.697 | 0.956 / 0.861 | 0.904 / 0.533 | 0.951 / 0.946 | 0.151 / 0.160 |
| p3_combo_balanced_aux, argmax | 0.210 | 0.625 | 0.328 | 0.933 / 0.766 | 0.760 / 0.619 | 0.969 / 0.804 | 0.993 / 0.983 | 0.845 / 0.844 | 0.916 / 0.749 | 0.911 / 0.444 | 0.976 / 0.954 | 0.698 / 0.636 |
| p3_policy, argmax | 0.195 | 0.583 | 0.316 | 0.947 / 0.766 | 0.702 / 0.595 | 0.959 / 0.779 | 0.996 / 0.983 | 0.832 / 0.839 | 0.929 / 0.749 | 0.896 / 0.462 | 0.976 / 0.962 | 0.684 / 0.641 |
| p3_heads_policy, shortlist k = 64 | 0.360 | 0.723 | 0.483 | 0.962 / 0.871 | 0.913 / 0.809 | 0.991 / 0.904 | 1.000 / 0.986 | 0.745 / 0.748 | 0.960 / 0.855 | 0.922 / 0.530 | 0.964 / 0.949 | 0.467 / 0.505 |
| p3_aux_policy, argmax | 0.263 | 0.676 | 0.389 | 0.933 / 0.802 | 0.816 / 0.700 | 0.966 / 0.835 | 0.996 / 0.982 | 0.842 / 0.845 | 0.927 / 0.774 | 0.938 / 0.467 | 0.976 / 0.955 | 0.693 / 0.629 |
| p3_aux_policy, shortlist k = 64 | 0.374 | 0.724 | 0.501 | 0.944 / 0.866 | 0.927 / 0.839 | 0.993 / 0.915 | 0.998 / 0.982 | 0.754 / 0.748 | 0.947 / 0.855 | 0.924 / 0.536 | 0.964 / 0.951 | 0.471 / 0.513 |
| p3_aux_policy, shortlist k = 256 | 0.382 | 0.740 | 0.511 | 0.960 / 0.868 | 0.918 / 0.834 | 0.992 / 0.916 | 0.996 / 0.982 | 0.768 / 0.758 | 0.960 / 0.848 | 0.938 / 0.530 | 0.964 / 0.966 | 0.547 / 0.626 |
| p3_aux_policy + RL β = 0.02, k = 256 | 0.382 | 0.739 | 0.512 | 0.953 / 0.864 | 0.927 / 0.850 | 0.992 / 0.916 | 0.996 / 0.984 | 0.766 / 0.752 | 0.956 / 0.851 | 0.942 / 0.546 | 0.960 / 0.963 | 0.547 / 0.595 |
| p3_aux_policy + RL β = 0.005, k = 256 | 0.384 | 0.736 | 0.508 | 0.942 / 0.866 | 0.933 / 0.842 | 0.994 / 0.916 | 0.998 / 0.984 | 0.764 / 0.757 | 0.942 / 0.847 | 0.938 / 0.545 | 0.962 / 0.964 | 0.529 / 0.573 |
| RL-3 β = 0.02, argmax | 0.335 | 0.770 | 0.441 | 0.958 / 0.808 | 0.909 / 0.791 | 0.987 / 0.862 | 0.996 / 0.982 | 0.837 / 0.838 | 0.949 / 0.775 | 0.949 / 0.439 | 0.978 / 0.955 | 0.689 / 0.622 |
| RL-3 β = 0.02, shortlist k = 256 | 0.413 | 0.760 | 0.532 | 0.957 / 0.857 | 0.940 / 0.868 | 0.996 / 0.920 | 0.996 / 0.983 | 0.764 / 0.759 | 0.953 / 0.849 | 0.942 / 0.544 | 0.962 / 0.963 | 0.560 / 0.612 |
| RL-3 β = 0.005, argmax | 0.365 | 0.760 | 0.473 | 0.940 / 0.829 | 0.936 / 0.824 | 0.991 / 0.898 | 0.998 / 0.980 | 0.810 / 0.818 | 0.938 / 0.801 | 0.951 / 0.489 | 0.973 / 0.965 | 0.609 / 0.598 |
| RL-3 β = 0.005, shortlist k = 256 | 0.412 | 0.764 | 0.533 | 0.951 / 0.865 | 0.956 / 0.865 | 0.998 / 0.921 | 0.996 / 0.983 | 0.756 / 0.759 | 0.949 / 0.849 | 0.947 / 0.532 | 0.964 / 0.967 | 0.542 / 0.577 |
| **RL-3 β = 0.005 + control scorer, k = 256 (best)** | 0.419 | 0.774 | 0.533 | 0.958 / 0.858 | 0.960 / 0.868 | 0.998 / 0.918 | 0.998 / 0.984 | 0.768 / 0.777 | 0.953 / 0.842 | 0.936 / 0.533 | 0.969 / 0.966 | 0.524 / 0.597 |

Correction (2026-11-13): until today the stage-two sub-scores of this table (from the p3_policy row down) were
unweighted means over all synthetic scenes, not the official weighted values. Stage one, the stage totals and the
combined scores were already the official ones. The differences are up to 0.05 (e.g. p3_aux_policy k = 256: NC 0.884 →
0.868, LK 0.499 → 0.530, EC 0.572 → 0.626). The "scored 0" column was dropped because it was an unweighted share of
scenes; the headroom section below gives it for the best run.

Combined = per scene group, stage one × weighted stage two, averaged over the 225 groups
(`run_pdm_score.py:calculate_individual_mapping_scores`), so it is far below either stage.

### Headroom: what each sub-score costs (2026-11-13)

How much the final score would rise if one sub-score were perfect in every scene, all others unchanged
(`scripts/evaluation/score_headroom.py`). Scores are recomputed from the per-scene sub-scores of the CSV. Each run is
checked first: navtest recomputed EPDMS = the CSV score column; navhard rebuilt combined / stage one / stage two = the
CSV summary rows (0.419360 / 0.773813 / 0.533524, exact). The navhard stage-two weights are not in the CSV; they depend
only on the planned stage-one endpoint and the stage-two start positions, so the script rebuilds them by replaying the
run's selection rule on the saved predictions (start positions read once from the metric cache:
`exp/headroom/navhard_two_stage_start_points.csv`, CPU job 15031309, 2.5 min; weights per run:
`<run>/stage_two_weights.csv`). Gains overlap, so the rows do not add up. "Human level" = per scene the better of the
agent's and the human agent's value (same filter setting).

navtest, best v2 (RL-3 β = 0.02 + pairwise scorer, mixture λ 10):

| Metric | filter on: mean | failing scenes | + if perfect | + at human level | filter off: mean | + if perfect | + at human level |
|---|---|---|---|---|---|---|---|
| NC | 0.981 | 2.0% | +0.012 | +0.012 | 0.981 | +0.011 | +0.011 |
| DAC | 0.985 | 1.5% | +0.012 | +0.012 | 0.985 | +0.012 | +0.012 |
| DDC | 0.992 | 1.3% | +0.006 | +0.005 | 0.992 | +0.005 | +0.004 |
| TLC | 0.999 | 0.1% | +0.001 | +0.001 | 0.975 | +0.021 | +0.001 |
| EP | 0.890 | 60.0% (partial) | +0.034 | +0.006 | 0.890 | +0.034 | +0.006 |
| TTC | 0.975 | 2.5% | +0.002 | +0.002 | 0.975 | +0.002 | +0.002 |
| LK | 0.956 | 4.4% | +0.005 | +0.005 | 0.848 | +0.018 | +0.005 |
| HC | 0.983 | 1.7% | +0.002 | +0.000 | 0.983 | +0.002 | +0.000 |
| EC | 0.815 | 18.5% of scenes with EC | +0.018 | +0.010 | 0.815 | +0.017 | +0.009 |
| all multipliers | | | +0.032 | | | +0.052 | |
| all weighted | | | +0.061 | | | +0.072 | |

EPDMS 0.8978 (filter on) / 0.8639 (off); scenes scored 0: 3.6% / 5.9%.

navhard, best (RL-3 β = 0.005 + control scorer, shortlist k = 256; combined 0.419):

| Metric | stage one: mean / failing | stage two: mean / failing (unweighted share) | + combined if perfect in both stages | stage one only | stage two only |
|---|---|---|---|---|---|
| NC | 0.958 / 4.2% | 0.858 / 11.9% | +0.069 | +0.011 | +0.051 |
| DAC | 0.960 / 4.0% | 0.868 / 15.0% | **+0.080** | +0.015 | **+0.060** |
| DDC | 0.998 / 0.4% | 0.918 / 15.9% | +0.036 | +0.001 | +0.035 |
| TLC | 0.998 / 0.2% | 0.984 / 1.1% | +0.009 | +0.000 | +0.009 |
| EP | 0.768 / 70.9% (partial) | 0.777 / 59.3% (partial) | +0.094 | +0.040 | +0.048 |
| TTC | 0.953 / 4.7% | 0.842 / 13.6% | +0.006 | +0.001 | +0.005 |
| LK | 0.936 / 6.4% | 0.533 / 49.5% | +0.031 | +0.004 | +0.026 |
| HC | 0.969 / 3.1% | 0.966 / 5.0% | +0.003 | +0.002 | +0.001 |
| EC | 0.524 / 47.6% | 0.597 / 47.9% | +0.059 | +0.032 | +0.025 |
| all multipliers | | | +0.227 | | |
| all weighted | | | +0.206 | | |

Scenes scored 0 (unweighted): stage one 8.4%, stage two 29.9%. Combined if stage two were perfect: 0.774 (= stage one);
if stage one were perfect: 0.534 (= stage two).

24. **navtest is close to its ceiling under this metric.** Of the 0.102 lost, 0.036 is the 3.6% of scenes scored 0
    (NC / DAC / DDC failures). EP looks large (+0.034) but our progress is already above the human's (0.890 vs 0.874),
    so at human level it is worth +0.006; pushing it further trades safety. Realistic gains: EC (+0.010 at human level)
    and the zero-score scenes (+0.005-0.010). Filter off, the extra losses are TLC and LK where the human fails too
    (human level +0.001 / +0.005), i.e. mostly not recoverable.
25. **On navhard the loss is in stage two**: 0.53 vs 0.77, and 29.9% of the synthetic scenes score 0 against 8.4% of
    the real ones. The biggest levers are the stage-two multipliers (DAC +0.060, NC +0.051, DDC +0.035 from stage two
    alone; +0.227 for all multipliers together) and EC in both stages (+0.059; it fails in 48% of scenes in both,
    against 18.5% on navtest). Stage two starts from perturbed ego states with synthetic images, which the model never
    saw in training (navtrain only); LK failing in half of them suggests it does not steer back to the lane centre.
26. Reaching ZTRS (48.1 on the leaderboard, ~+0.06 combined) needs e.g. EC at ~0.8 in both stages, or about a third of
    the stage-two multiplier failures removed; DrivoR (54.5, ~+0.13) needs both.

### Checks (all passed)

- p3_aux `score_heads` v2 rerun with the fixed code: 0.8116, identical to the value rebuilt from the earlier filter-on
  CSV, with identical per-metric columns (the evaluation is deterministic and the rebuild exact).
- In every fixed-filter run the reported score equals the score rebuilt from its columns (max difference 3·10⁻¹⁶).
- navhard p3_heads: stage two identical to the pre-fix run, stage one 0.662 → 0.699, combined 0.326 → 0.343, as
  estimated beforehand (0.699 / ≈ 0.344).
- `score_heads_v1` raises v1 (+0.003 to +0.004 over `score_heads`) and lowers v2 (−0.009 to −0.011), as expected;
  `shortlist_v1` likewise (v1 +0.001 to +0.004, v2 −0.001 to −0.007).
- The 13 runs of 2026-11-07 (jobs 14913472–84): all `COMPLETED` with exit code 0 in 52–83 min; navtest 12,146 scenes,
  navhard 450 + 5,462 rows, 0 failed scenes; score = score rebuilt from the columns (max difference 3·10⁻¹⁶). The
  policy-only runs used the merged checkpoints (log: "Loaded 685 keys from checkpoint", the full set of trainable
  parameters, as for p3_aux).

### Findings

1. **Best model on all three benchmarks: p3_aux_policy with `shortlist` k = 256** (v1 0.893, v2 0.867, navhard
   0.382). Before it: v1 navtest p3_aux (heads) 0.889; v2 navtest p3_combo_balanced_aux with a combined selection,
   0.847 (mixture λ = 10) / 0.846 (shortlist k = 256); navhard p3_aux (heads) 0.361.
2. **The selection decides more than the model.** Argmax is temporally consistent (EC 0.82 vs 0.17–0.23 for the
   heads), which wins v2 on navtest. But it leaves the drivable area far more often (navtest DAC 0.91–0.92 vs
   0.97–0.98), and on navhard this dominates: DAC 0.75 / 0.62, 30% / 48–51% of the scenes scored 0, combined
   0.19–0.21 vs 0.30–0.36 with the heads. The heads' safety check matters most where the scenes are hard or synthetic.
3. **Aux losses help on all three benchmarks**, for both selections:
   p3_aux − p3_heads: v1 +0.008, v2 +0.008, navhard +0.018;
   p3_combo_balanced_aux − p3_combo_balanced (argmax): v1 +0.010, v2 +0.011, navhard +0.016.
4. p3_combo_balanced_aux's heads are weaker than p3_aux's on every benchmark (v1 −0.027, v2 −0.025, navhard −0.061),
   as on the val proxy (0.797 vs 0.826): imitation still competes with the heads.
5. navhard stage two is the weak part for every model (lane keeping 0.44–0.55, a third to half of the synthetic
   scenes scored 0): the models are trained only on real navtrain images.
6. Combining both selections works (next section): for p3_combo_balanced_aux the heads shortlist with the policy
   picking beats argmax and heads on all three benchmarks.
7. **Training the policy separately on top of the heads model beats training both jointly** (section "Decoupled
   policy head"): p3_aux_policy (shortlist) is above p3_combo_balanced_aux (shortlist) on every benchmark.

## Selection rules combining policy and heads (p3_combo_balanced_aux ep 9, 2026-11-06)

Chosen from the val proxy (checkpoint_evaluation.md, "Selection rules"). `shortlist` k: the policy's most likely entry
among the k entries with the best predicted total; `mixture` λ: argmax of log p + λ · log predicted total. v1 runs use
the `_v1` variants (v1 formula for the predicted total). All runs `WORKERS=3`, 0 failed scenes.

| Selection | v1 navtest | v2 navtest | navhard | navhard stage 1 / 2 | navtest scored 0 | navtest EC | navhard stage 1 scored 0 |
|---|---|---|---|---|---|---|---|
| argmax | 0.842 | 0.826 | 0.210 | 0.625 / 0.328 | 11.0% | 0.819 | 29.6% |
| heads | 0.863 | 0.786 | 0.301 | 0.654 / 0.451 | 5.5% | 0.171 | 15.1% |
| **shortlist k = 64** | **0.873** | 0.838 | **0.334** | 0.686 / 0.475 | 5.4% | 0.610 | 14.7% |
| shortlist k = 256 | 0.872 | **0.846** | 0.311 | 0.677 / 0.456 | 5.8% | 0.696 | 17.1% |
| mixture λ = 10 | 0.866 | **0.847** | 0.246 | 0.655 / 0.386 | 8.2% | 0.809 | 25.1% |

(navtest "scored 0" and EC from the v2 run of each selection.) Paired differences on navtest (12,146 scenes):

| Comparison | Benchmark | Difference |
|---|---|---|
| shortlist k = 256 − argmax | v2 | +0.020 ± 0.002 |
| shortlist k = 64 − heads | v2 | +0.051 ± 0.002 |
| mixture λ = 10 − shortlist k = 256 | v2 | +0.001 ± 0.002 (equal) |
| shortlist k = 256 − shortlist k = 64 | v2 | +0.008 ± 0.002 |
| shortlist_v1 k = 64 − heads (`score_heads_v1`) | v1 | +0.010 ± 0.002 |
| shortlist_v1 k = 64 − shortlist_v1 k = 256 | v1 | +0.001 ± 0.002 (equal) |
| shortlist_v1 k = 64 − mixture_v1 λ = 10 | v1 | +0.006 ± 0.002 |

Sub-scores, navtest v2 runs (NC, DAC, DDC, TLC, EP, TTC, LK, HC, EC): shortlist k = 64: 0.976 0.971 0.994 0.998 0.824
0.965 0.935 0.975 0.610; k = 256: 0.977 0.967 0.992 0.998 0.832 0.966 0.939 0.974 0.696; mixture λ = 10: 0.974 0.945
0.984 0.998 0.868 0.966 0.945 0.982 0.809. navhard stage one / two: shortlist k = 64 DAC 0.902 / 0.818, EP 0.715 /
0.708, LK 0.918 / 0.496, EC 0.391 / 0.404; k = 256 DAC 0.882 / 0.767, EC 0.467 / 0.508; mixture DAC 0.811 / 0.693,
EC 0.684 / 0.625.

Findings:
1. **The shortlist keeps most of the policy's consistency and the heads' safety.** On navtest v2, k = 256 has EC 0.70
   (argmax 0.82, heads 0.17) with the heads' failure rate (5.8% vs 5.5%; argmax 11.0%): +0.020 over argmax and
   +0.060 over the heads. The worry that the shortlist would flip between frames like the heads did not hold.
2. **k trades consistency for safety:** k = 256 is better on v2 navtest (+0.008, EC 0.70 vs 0.61), k = 64 on navhard
   (0.334 vs 0.311; fewer zero scenes in the hard real scenes, 14.7% vs 17.1%). On v1 (no EC) they are equal.
3. **The mixture is the most consistent combination** (EC 0.81, as argmax) and ties the shortlist on v2 navtest, but
   it removes too few failures for navhard (25% zero in stage one, 0.246) and v1 (−0.006).
4. **Best single rule for this model: shortlist k = 64** (best v1 0.873 and navhard 0.334, v2 0.838); k = 256 if v2
   navtest matters most. Neither reaches p3_aux's heads on v1 (0.889) and navhard (0.361), whose heads are stronger;
   p3_aux's policy head is untrained (CE 0), so the shortlist does not apply to it; p3_aux_policy (next section) trains
   that policy head afterwards and then beats all of these.
5. The val proxy ranked the rules on safety correctly (shortlist > mixture on failures), but it cannot see EC, which
   decides v2 navtest (the mixture was the worst rule on the proxy among these three and ties for best on v2).

## Decoupled policy head and ablation chain (2026-11-07)

`p3_heads_policy` and `p3_aux_policy` start from p3_heads ep 5 and p3_aux ep 5 and train only the policy head
(`train_policy_head_only=true`: scene projection, trajectory bias and logit scale; LoRA, sub-score heads and planner
queries frozen) with the p3_combo_balanced policy losses (score-weighted labels, CE 1, ED 1). The heads, and therefore
the `score_heads` selection, are those of the start model; the trained policy makes argmax and the shortlist usable on
these models. Checkpoints: the val proxy's best epoch (p3_aux_policy ep 3, p3_heads_policy ep 2), merged with the start
checkpoint (`scripts/evaluation/merge_policy_only_checkpoint.py`). `p3_policy` is the phase 3 run without sub-score
heads (imitation only, from p1c_ed05).

Ablation chain (navtest v1 / v2, navhard):

| Step | Model, selection | v1 | v2 | navhard |
|---|---|---|---|---|
| imitation only | p3_policy, argmax | 0.793 | 0.775 | 0.195 |
| + sub-score heads and score-weighted labels | p3_combo_balanced, argmax | 0.832 | 0.815 | 0.194 |
| + aux losses | p3_combo_balanced_aux, argmax | 0.842 | 0.826 | 0.210 |
| + shortlist selection | p3_combo_balanced_aux, shortlist k = 64 | 0.873 | 0.838 | 0.334 |
| heads-only training (no policy) | p3_aux, heads | 0.889 | 0.812 | 0.361 |
| + policy head trained afterwards | p3_aux_policy, argmax | 0.861 | 0.844 | 0.263 |
| + shortlist selection | p3_aux_policy, shortlist k = 64 | 0.891 | 0.860 | 0.374 |
| | p3_aux_policy, shortlist k = 256 | 0.893 | 0.867 | 0.382 |
| + RL on the policy head (β = 0.02) | **p3_aux_policy + RL, shortlist k = 256** | **0.899** | **0.872** | **0.382** |
| (without aux) | p3_heads, heads → p3_heads_policy, shortlist k = 64 | 0.882 → 0.887 | 0.804 → 0.854 | 0.343 → 0.360 |

Paired differences on navtest (12,146 scenes, mean ± standard error):

| Comparison | Benchmark | Difference |
|---|---|---|
| p3_aux_policy k = 256 − p3_combo_balanced_aux k = 256 | v2 | +0.021 ± 0.002 |
| p3_aux_policy k = 256 − p3_aux heads (`score_heads`) | v2 | +0.055 ± 0.002 |
| p3_aux_policy k = 256 − p3_aux_policy argmax | v2 | +0.023 ± 0.002 |
| p3_aux_policy k = 256 − k = 64 | v2 | +0.007 ± 0.001 |
| p3_aux_policy argmax − p3_combo_balanced_aux argmax | v2 | +0.019 ± 0.003 |
| p3_aux_policy k = 64 − p3_heads_policy k = 64 | v2 | +0.007 ± 0.002 |
| p3_aux_policy `shortlist_v1` k = 256 − p3_aux `score_heads_v1` | v1 | +0.004 ± 0.002 |
| p3_aux_policy `shortlist_v1` k = 256 − k = 64 | v1 | +0.002 ± 0.001 (equal) |
| p3_aux_policy `shortlist_v1` k = 64 − p3_combo_balanced_aux `shortlist_v1` k = 64 | v1 | +0.019 ± 0.002 |
| p3_heads_policy `shortlist_v1` k = 64 − p3_heads `score_heads_v1` | v1 | +0.005 ± 0.002 |
| p3_policy argmax − p3_combo_balanced_aux argmax | v2 / v1 | −0.051 ± 0.003 / −0.049 ± 0.003 |

Findings:
1. **The decoupled policy gives the heads model temporal consistency.** On v2 navtest, p3_aux_policy with the
   shortlist adds +0.055 over p3_aux's heads, almost all from two-frame comfort (EC 0.75 vs 0.23), with the same
   failure rate (5.0% vs 4.8% scored 0). p3_heads_policy: +0.050 over p3_heads.
2. **It also beats the jointly trained model:** +0.021 (v2) and +0.019 (v1) over p3_combo_balanced_aux with the same
   selection, and +0.071 / +0.040 on navhard (k = 256 / k = 64). Even its argmax is better (+0.019 v2, navhard 0.263
   vs 0.210), although only the policy head was trained on top of a LoRA trained for the heads alone.
3. **On v1 and navhard the gain is small but positive** (v1 +0.004 over p3_aux's heads; navhard +0.021, stage one
   0.692 → 0.740): the shortlist keeps the heads' safety and the policy's choice inside it is at least as good.
4. **k = 256 is best for this model on all three benchmarks** (for p3_combo_balanced_aux k = 64 was better on
   navhard). With a trained policy on top of stronger heads, a wider shortlist costs no safety here (navhard stage one
   scored 0: 12.7% vs 12.9%).
5. **Aux losses still help** with the decoupled policy: p3_aux_policy vs p3_heads_policy, k = 64: v1 +0.005, v2
   +0.007, navhard +0.014.
6. **Imitation only is clearly the weakest** (v1 0.793, v2 0.775): the sub-score heads and score-weighted labels add
   +0.04 on navtest. On navhard, p3_policy and p3_combo_balanced are equal (0.195 / 0.194); only the heads' selection
   (or aux) lifts navhard.

## RL on the policy head (RL-1, 2026-11-08)

p3_aux_policy ep 3 with its policy head fine-tuned by RL (exact objective: expected codebook score − β · KL to the
start policy, on the cached fused states; LoRA and sub-score heads unchanged, so the shortlist is the same). Theory:
[theoretical.md](theoretical.md); val proxy sweep: [checkpoint_evaluation.md](checkpoint_evaluation.md), findings
41–48. Two checkpoints: β = 0.02, 20 epochs, epoch 12 (val KL 0.65) and β = 0.005, lr 3e-4, epoch 7 (val-best proxy,
KL 1.7). Selection `shortlist` k = 256 (`shortlist_v1` for v1); 0 failed scenes in all six runs.

| Model | v1 | v2 | navhard | stage 1 / 2 | v2 EC | v2 DAC | v2 scored 0 |
|---|---|---|---|---|---|---|---|
| p3_aux_policy (start) | 0.893 | 0.867 | 0.382 | 0.740 / 0.511 | 0.751 | 0.975 | 5.0% |
| + RL β = 0.02 | **0.899** | **0.872** | 0.382 | 0.739 / 0.512 | 0.741 | 0.980 | **4.5%** |
| + RL β = 0.005 | 0.898 | 0.869 | 0.384 | 0.736 / 0.508 | 0.728 | 0.981 | 4.8% |

Paired differences on navtest (12,146 scenes, mean ± standard error):

| Comparison | Benchmark | Difference | Scenes with a different score |
|---|---|---|---|
| RL β = 0.02 − start | v2 | +0.0052 ± 0.0011 | 38% |
| RL β = 0.02 − start | v1 | +0.0061 ± 0.0011 | 35% |
| RL β = 0.005 − start | v2 | +0.0028 ± 0.0014 | 50% |
| RL β = 0.005 − start | v1 | +0.0045 ± 0.0014 | 46% |
| RL β = 0.005 − RL β = 0.02 | v2 | −0.0024 ± 0.0010 | 33% |

Findings:
1. **RL helps on navtest, as the proxy predicted.** β = 0.02: v2 +0.005 (≈ 5 SE), v1 +0.006; the val proxy gain
   for the shortlist was +0.008 to +0.010. The gain is in the multiplicative terms: DAC 0.975 → 0.980, scored 0
   5.0% → 4.5%, EP +0.004.
2. **The KL anchor matters for what the reward does not see.** Two-frame comfort (EC, not in the reward) drops
   0.751 → 0.741 at β = 0.02 and → 0.728 at β = 0.005. The looser β = 0.005 gains more on the reward's terms (EP
   0.868, DAC 0.981) but loses it again in EC and NC/TTC, and is below β = 0.02 on v2 (−0.002 ± 0.001), although it
   was the val-best on the proxy. The proxy cannot rank β because it has no EC.
3. **No gain on navhard** (0.382 / 0.384 vs 0.382; stage one 0.739 / 0.736 vs 0.740, within noise). The reward is
   log replay; navhard is reactive with synthetic stage two scenes, and stage two EC also drops (official weighted values 0.626 → 0.595 /
   0.573; the unweighted means first quoted here were 0.572 → 0.546 / 0.523). Gains on navhard need either reactive rewards or a better scene representation (RL-3, LoRA trainable).
4. Choice for later work: **β = 0.02** (best v1 and v2, smallest EC cost).

## Split evaluation pipeline (GPU outputs, CPU scoring; 2026-11-08)

The combined runs (`run_vision_llm_lora_pdm_score_slurm.sh`) used the GPU at 15-34% (HPC warning): each worker read
the images through one squashfuse mount, ran the model at batch 1 and then simulated on the CPU while the GPU waited.
Now one GPU job writes the model's raw outputs and CPU jobs score them with the unchanged NAVSIM scripts:

1. GPU (`scripts/evaluation/run_vision_llm_lora_predict_slurm.sh` → `predict_vision_llm_lora.py`): navtest and
   navhard_two_stage, batch 64, 30 loader processes over 16 squashfuse mounts of the image (one mount serves one read
   at a time: ~4 scenes/s; 16 mounts: ~20 scenes/s). Saves policy logits, sub-score logits (bf16 bit-exact) and the
   model's trajectory per scene to `predictions/<name>/<split>/` (3.4 GB navtest, 1.7 GB navhard). p3_aux_policy:
   19.5 min (setup 3.4 min, navtest 584 s, navhard 292 s), GPU 95-100% while running, 65% over the job, ~14 GB RAM.
2. CPU (`run_vision_llm_lora_score_predictions_slurm.sh`, `-p cpu`): `VisionLLMLoRAPrecomputedAgent` applies the
   selection rule (any of argmax / heads / shortlist k / mixture, v1 or v2) to the saved outputs and returns one
   trajectory per scene; scoring through `run_pdm_score_balanced.py` (NAVSIM script unchanged; one task per log, navtest
   logs split into 100-scene chunks, instead of one fixed chunk of logs per process). navtest: 5.5 min on 32 CPUs
   (efficiency 41-43%), 8 min on 16 CPUs with chunks (57%); default now 8 CPUs. navhard: 12 min on 32 CPUs.
3. `scripts/evaluation/submit_vision_llm_lora_eval.sh` submits both (CPU jobs with `afterok` on the GPU job);
   `PRED=<name>` reuses saved outputs for new selections without a GPU.

Same model, new pipeline (p3_aux_policy ep 3, shortlist k = 256; jobs 14926360, 14926657, 14926658, 14926363):

| | combined (batch 1) | split (batch 64) | new − old, paired |
|---|---|---|---|
| v1 navtest (`shortlist_v1`) | 0.8931 | 0.8915 | −0.0015 ± 0.0007 |
| v2 navtest (`shortlist`) | 0.8667 | 0.8674 | +0.0007 ± 0.0007 |
| navhard | 0.3815 | 0.3860 | (225 groups) |

The difference is bf16 tie-breaking, not the pipeline: the policy logits (~37) have a bf16 spacing of 0.25, batch 64
uses other matmul kernels than batch 1, and 17-18% of the scenes get a different (near-tied) trajectory. Rescoring the
same outputs with the chunked scheduler gives exactly the same v2 (0.8674). So official numbers carry about ±0.002
from bf16 ties; compare models only within one pipeline and treat navtest differences below ~0.003 as ties.

## Runs (fixed filter)

| Result | Job | Output (under `$NAVSIM_EXP_ROOT`) |
|---|---|---|
| navtest v2, p3_heads ep 5 score_heads | 14871654 (pre-fix run; fixed score rebuilt from its columns) | `pdm_navtest/p3_heads_best_ep05_score_heads_non_reactive/` |
| navtest v2, p3_aux ep 5 score_heads | 14891494 | `pdm_navtest/p3_aux_best_ep05_score_heads_non_reactive_filterfix/` |
| navtest v1 + v2, p3_combo_balanced ep 8 argmax | 14885933 (pre-fix run; fixed score rebuilt) | `pdm_navtest/p3_combo_balanced_ep08_argmax_non_reactive/` |
| navtest v1, p3_heads ep 5 score_heads_v1 | 14891501 | `pdm_navtest/p3_heads_best_ep05_score_heads_v1_non_reactive_filterfix/` |
| navtest v1, p3_aux ep 5 score_heads_v1 | 14891500 | `pdm_navtest/p3_aux_best_ep05_score_heads_v1_non_reactive_filterfix/` |
| navtest v1, p3_combo_balanced_aux ep 9 score_heads_v1 | 14891553 | `pdm_navtest/p3_combo_balanced_aux_ep09_score_heads_v1_non_reactive_filterfix/` |
| navtest v2, p3_combo_balanced_aux ep 9 score_heads | 14891554 | `pdm_navtest/p3_combo_balanced_aux_ep09_score_heads_non_reactive_filterfix/` |
| navtest v1 + v2, p3_combo_balanced_aux ep 9 argmax | 14891559 | `pdm_navtest/p3_combo_balanced_aux_ep09_argmax_non_reactive_filterfix/` |
| navhard, p3_heads ep 5 score_heads | 14891468 | `pdm_navhard_two_stage/p3_heads_best_ep05_score_heads_reactive_filterfix/` |
| navhard, p3_aux ep 5 score_heads | 14891469 | `pdm_navhard_two_stage/p3_aux_best_ep05_score_heads_reactive_filterfix/` |
| navhard, p3_combo_balanced ep 8 argmax | 14891470 | `pdm_navhard_two_stage/p3_combo_balanced_ep08_argmax_reactive_filterfix/` |
| navhard, p3_combo_balanced_aux ep 9 score_heads | 14891555 | `pdm_navhard_two_stage/p3_combo_balanced_aux_ep09_score_heads_reactive_filterfix/` |
| navhard, p3_combo_balanced_aux ep 9 argmax | 14891560 | `pdm_navhard_two_stage/p3_combo_balanced_aux_ep09_argmax_reactive_filterfix/` |
| p3_combo_balanced_aux ep 9, shortlist k = 64: v1 / v2 / navhard | 14903090 / 91 / 92 | `pdm_navtest/…_shortlist_v1_k64_…`, `pdm_navtest/…_shortlist_k64_…`, `pdm_navhard_two_stage/…_shortlist_k64_…` |
| p3_combo_balanced_aux ep 9, shortlist k = 256: v1 / v2 / navhard | 14903093 / 94 / 95 | same pattern with `k256` |
| p3_combo_balanced_aux ep 9, mixture λ = 10: v1 / v2 / navhard | 14903096 / 97 / 98 | same pattern with `mixture[_v1]_lam10` |
| p3_aux_policy ep 3, shortlist k = 64: v1 / v2 / navhard | 14913472 / 73 / 74 | `pdm_navtest/p3_aux_policy_ep03_shortlist[_v1]_k64_non_reactive_filterfix/`, `pdm_navhard_two_stage/p3_aux_policy_ep03_shortlist_k64_reactive_filterfix/` |
| p3_aux_policy ep 3, shortlist k = 256: v1 / v2 / navhard | 14913475 / 76 / 77 | same pattern with `k256` |
| p3_aux_policy ep 3, argmax: navtest (v1 + v2) / navhard | 14913478 / 79 | `…/p3_aux_policy_ep03_argmax_{non_reactive,reactive}_filterfix/` |
| p3_heads_policy ep 2, shortlist k = 64: v1 / v2 / navhard | 14913480 / 81 / 82 | `…/p3_heads_policy_ep02_shortlist[_v1]_k64_…_filterfix/` |
| p3_policy ep 7, argmax: navtest (v1 + v2) / navhard | 14913483 / 84 | `…/p3_policy_ep07_argmax_{non_reactive,reactive}_filterfix/` |
| p3_aux_policy + RL β = 0.02 ep 12, shortlist k = 256: v1 / v2 / navhard | 14924996 / 97 / 98 | `…/p3_aux_policy_rl_b0.02_ep12_shortlist[_v1]_k256_…_filterfix/` |
| p3_aux_policy + RL β = 0.005 lr 3e-4 ep 7, shortlist k = 256: v1 / v2 / navhard | 14924999 / 14925000 / 14925001 | `…/p3_aux_policy_rl_b0.005_lr3e-4_ep07_shortlist[_v1]_k256_…_filterfix/` |

Checkpoints: p3_heads `p3_heads/2026.11.04.00.53.01/checkpoints/best/best-epoch=05.ckpt`, p3_aux
`p3_aux/2026.11.05.00.00.51/checkpoints/best/best-epoch=05.ckpt`, p3_combo_balanced
`p3_combo_balanced/2026.11.03.23.40.54/checkpoints/epochs/epoch-epoch=08.ckpt`, p3_combo_balanced_aux
`p3_combo_balanced_aux/2026.11.05.12.26.07/checkpoints/epochs/epoch-epoch=09.ckpt` (best by the val proxy),
p3_aux_policy `p3_aux_policy/2026.11.06.23.42.11/checkpoints/merged/epoch-epoch=03.ckpt`, p3_heads_policy
`p3_heads_policy/2026.11.07.00.04.51/checkpoints/merged/epoch-epoch=02.ckpt`, p3_policy
`p3_policy/2026.11.07.00.04.52/checkpoints/epochs/epoch-epoch=07.ckpt`, RL-1
`rl_policy/p3_aux_policy_ep03/exact_beta0.02_epochs20/checkpoints/epoch-epoch=12.ckpt` and
`rl_policy/p3_aux_policy_ep03/exact_beta0.005_epochs10_lr3e-4/checkpoints/epoch-epoch=07.ckpt` (full checkpoints).
Policy-only runs need the merged checkpoint (their own checkpoints hold only the policy head).
Time per run: navtest ~2.4 h sequential or ~55–65 min with `WORKERS=3` (3 model copies on one A100-40GB, peak
32.4 GB); navhard ~2.1–2.3 h sequential or ~50 min with `WORKERS=3`. Command (`scripts/evaluation/run_vision_llm_lora_pdm_score_slurm.sh`;
submit with `bashrc` sourced and `envs/navsim_vlm` active):

```bash
WORKERS=3 RUN=p3_aux/2026.11.05.00.00.51 CKPT=checkpoints/best/best-epoch=05.ckpt SELECTION=score_heads_v1 \
    sbatch -p ai,gpu --cpus-per-task=16 --mem=120G scripts/evaluation/run_vision_llm_lora_pdm_score_slurm.sh   # navtest
SPLIT=navhard_two_stage RUN=... CKPT=... SELECTION=score_heads \
    sbatch -p ai,gpu scripts/evaluation/run_vision_llm_lora_pdm_score_slurm.sh                                  # navhard
```

## Pre-fix numbers (history)

Before the filter fix (runs of 2026-11-04/05, filter on, score not filtered = EPDMS*): navtest v2 p3_heads 0.771,
p3_aux 0.779, p3_combo_balanced 0.783; navhard p3_heads 0.326 (0.662 / 0.483). The "v1-style" numbers reported then
(0.878 / 0.885 / 0.832) used the same columns as now but the v2 selection for the heads. Two earlier observations
still hold: the gap between the val proxy and navtest is two-frame comfort (navtest without EC was 0.837 / 0.844 for
p3_heads / p3_aux vs 0.818 / 0.826 on the proxy), and EC = 0 in ~78% of the scenes with heads selection.

## To do

- Submission pickle of p3_aux_policy ep 3, shortlist k = 256, on navhard_two_stage (placeholder team name
  "Anonymous"; `scripts/submission/run_vision_llm_lora_create_submission_slurm.sh`): job 14916750, output
  `submission_navhard_two_stage/p3_aux_policy_ep03_shortlist_k256/`. Optional check: score it with
  `run_pdm_score_from_submission.py` (should reproduce 0.382). Which leaderboard accepts it: UNKNOWN (navhard has
  public ground truth; the 2025 leaderboards used `warmup_two_stage` and a private test split, not downloaded here).
- Optional: shortlist on other models with a trained policy; navhard / navtest for p3_heads_aux (staged aux).
