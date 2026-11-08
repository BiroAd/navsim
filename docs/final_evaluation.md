# Final evaluation (official NAVSIM metrics)

Official PDM scores of the thesis models, from raw camera images through the NAVSIM devkit scripts. The model-selection
proxy (navtrain val, precomputed codebook scores) is in [checkpoint_evaluation.md](checkpoint_evaluation.md); runs and
their settings are in [experiments.md](experiments.md). Last updated 2026-11-08.

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
| **v1 PDMS** | a navtest run (one stage), recomputed from its sub-score columns | log replay | NC · DAC · (5 EP + 5 TTC + 2 HC) / 12 (NAVSIM v1 formula with v2 history comfort) |
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
- **v1 PDMS** is computed from the same filter-on (fixed) columns, with v2 history comfort as the comfort term (a
  slightly stricter check: it also covers the transition from the past trajectory; effect ≤ ~0.003 here). The v1
  devkit is not in this repo, so how close this is to the original v1 implementation (filter, comfort) is UNVERIFIED.
  v1 is higher than v2 for every model because v1 has fewer ways to lose points (no LK, EC, DDC, TLC).
- **Selection** (`trajectory_selection`): `argmax` = the policy's most probable codebook entry; `score_heads` = the
  entry with the best total predicted by the 8 sub-score heads, combined with the v2 formula without EC (used for v2);
  `score_heads_v1` = the same heads combined with the v1 formula (used for v1). With argmax one run gives both v1 and v2.
- Published reference numbers of other methods: not collected here; see the papers (navhard: EPDMS vs EPDMS*).

## Results (re-evaluation 2026-11-08: split pipeline, batch 64, bf16)

All models through the same pipeline (section "Split evaluation pipeline"): one GPU job per checkpoint saves the
policy and sub-score logits (batch 64, bf16), CPU jobs apply each selection and score with the unchanged NAVSIM
scripts. 0 failed scenes in every run (navtest 12,146 / navhard 5,912). Jobs: GPU 14926360 (p3_aux_policy) and
14927392-14927429, CPU 14926657/58, 14927215, 14927280-84, 14927393-14927432 (list:
`$NAVSIM_EXP_ROOT/predictions/logs/reevaluation_2026-11-08.txt`). Outputs:
`pdm_navtest/<model>_<selection>[_k<k>|_lam10]_non_reactive_split/`, `pdm_navhard_two_stage/<model>_<selection>…_reactive_split/`
(p3_aux_policy k = 256 navhard: `…_split_cpu4_fastload/`).

| Model | Aux | Selection | **v1 PDMS** navtest | **v2 EPDMS** navtest | **v2 EPDMS** navhard | navhard stage 1 / stage 2 | v2 EC | v2 scored 0 |
|---|---|---|---|---|---|---|---|---|
| p3_policy ep 7 | – | argmax | 0.793 | 0.775 | 0.189 | 0.569 / 0.317 | 0.804 | 16.3% |
| p3_heads ep 5 | – | heads | 0.882 | 0.804 | 0.342 | 0.700 / 0.480 | 0.217 | 5.3% |
| p3_aux ep 5 | ✓ | heads | 0.889 | 0.812 | 0.363 | 0.696 / 0.509 | 0.229 | 4.8% |
| p3_combo_balanced ep 8 | – | argmax | 0.831 | 0.814 | 0.189 | 0.611 / 0.321 | 0.815 | 12.1% |
| p3_combo_balanced_aux ep 9 | ✓ | heads | 0.863 | 0.787 | 0.300 | 0.656 / 0.450 | 0.172 | 5.5% |
| p3_combo_balanced_aux ep 9 | ✓ | argmax | 0.841 | 0.824 | 0.214 | 0.639 / 0.340 | 0.818 | 11.1% |
| p3_combo_balanced_aux ep 9 | ✓ | shortlist k = 64 | 0.873 | 0.838 | 0.322 | 0.675 / 0.467 | 0.608 | 5.3% |
| p3_combo_balanced_aux ep 9 | ✓ | shortlist k = 256 | 0.872 | 0.845 | 0.317 | 0.675 / 0.462 | 0.695 | 5.8% |
| p3_combo_balanced_aux ep 9 | ✓ | mixture λ = 10 | 0.867 | 0.847 | 0.250 | 0.662 / 0.386 | 0.808 | 8.2% |
| p3_heads_policy ep 2 | – | shortlist k = 64 | 0.886 | 0.853 | 0.359 | 0.722 / 0.483 | 0.668 | 5.4% |
| p3_aux_policy ep 3 | ✓ | argmax | 0.861 | 0.843 | 0.254 | 0.664 / 0.374 | 0.815 | 9.0% |
| p3_aux_policy ep 3 | ✓ | shortlist k = 64 | 0.892 | 0.861 | 0.375 | 0.723 / 0.504 | 0.684 | 4.8% |
| p3_aux_policy ep 3 | ✓ | shortlist k = 256 | 0.892 | 0.867 | **0.386** | 0.737 / 0.517 | 0.756 | 4.9% |
| **p3_aux_policy ep 3 + RL β = 0.02** | ✓ | **shortlist k = 256** | **0.899** | **0.872** | 0.383 | **0.741** / 0.509 | 0.742 | **4.5%** |
| p3_aux_policy ep 3 + RL β = 0.005 | ✓ | shortlist k = 256 | 0.898 | 0.870 | 0.385 | 0.741 / 0.503 | 0.727 | 4.8% |

v1 from the `_v1` variant of the selection (`score_heads_v1`, `shortlist_v1`, `mixture_v1`; argmax: the same run), v2
and navhard from the v2 variant.

Paired differences on navtest (12,146 scenes, mean ± standard error):

| Comparison | v2 | v1 |
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

| Model, selection | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | scored 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| p3_heads, heads | 0.956 / 0.872 | 0.951 / 0.842 | 0.990 / 0.919 | 0.998 / 0.985 | 0.724 / 0.731 | 0.953 / 0.861 | 0.927 / 0.545 | 0.960 / 0.948 | 0.102 / 0.190 | 9.3% / 32.7% |
| p3_aux, heads | 0.947 / 0.885 | 0.944 / 0.875 | 0.992 / 0.922 | 0.996 / 0.984 | 0.729 / 0.731 | 0.949 / 0.872 | 0.936 / 0.542 | 0.964 / 0.955 | 0.120 / 0.160 | 11.1% / 29.7% |
| p3_combo_balanced, argmax | 0.927 / 0.762 | 0.751 / 0.623 | 0.968 / 0.777 | 0.993 / 0.984 | 0.842 / 0.847 | 0.909 / 0.747 | 0.911 / 0.453 | 0.976 / 0.954 | 0.707 / 0.619 | 30.7% / 50.6% |
| p3_combo_balanced_aux, heads | 0.949 / 0.874 | 0.907 / 0.798 | 0.990 / 0.927 | 0.998 / 0.983 | 0.686 / 0.697 | 0.956 / 0.861 | 0.904 / 0.533 | 0.951 / 0.946 | 0.151 / 0.160 | 15.1% / 33.9% |
| p3_combo_balanced_aux, argmax | 0.933 / 0.766 | 0.760 / 0.619 | 0.969 / 0.804 | 0.993 / 0.983 | 0.845 / 0.844 | 0.916 / 0.749 | 0.911 / 0.444 | 0.976 / 0.954 | 0.698 / 0.636 | 29.6% / 48.3% |
| p3_policy, argmax | 0.947 / 0.823 | 0.702 / 0.618 | 0.959 / 0.770 | 0.996 / 0.987 | 0.832 / 0.781 | 0.929 / 0.798 | 0.896 / 0.457 | 0.976 / 0.964 | 0.684 / 0.595 | 33.3% / 51.9% |
| p3_heads_policy, shortlist k = 64 | 0.962 / 0.883 | 0.913 / 0.810 | 0.991 / 0.883 | 1.000 / 0.988 | 0.745 / 0.755 | 0.960 / 0.867 | 0.922 / 0.499 | 0.964 / 0.943 | 0.467 / 0.461 | 12.4% / 33.5% |
| p3_aux_policy, argmax | 0.933 / 0.855 | 0.816 / 0.739 | 0.966 / 0.833 | 0.996 / 0.987 | 0.842 / 0.785 | 0.927 / 0.828 | 0.938 / 0.488 | 0.976 / 0.969 | 0.693 / 0.620 | 24.4% / 41.7% |
| p3_aux_policy, shortlist k = 64 | 0.944 / 0.889 | 0.927 / 0.836 | 0.993 / 0.889 | 0.998 / 0.988 | 0.754 / 0.762 | 0.947 / 0.871 | 0.924 / 0.507 | 0.964 / 0.944 | 0.471 / 0.480 | 12.9% / 30.8% |
| p3_aux_policy, shortlist k = 256 | 0.960 / 0.884 | 0.918 / 0.826 | 0.992 / 0.883 | 0.996 / 0.988 | 0.768 / 0.762 | 0.960 / 0.860 | 0.938 / 0.499 | 0.964 / 0.948 | 0.547 / 0.572 | 12.7% / 31.8% |
| p3_aux_policy + RL β = 0.02, k = 256 | 0.953 / 0.882 | 0.927 / 0.837 | 0.992 / 0.886 | 0.996 / 0.988 | 0.766 / 0.762 | 0.956 / 0.861 | 0.942 / 0.509 | 0.960 / 0.944 | 0.547 / 0.546 | 12.4% / 31.1% |
| p3_aux_policy + RL β = 0.005, k = 256 | 0.942 / 0.882 | 0.933 / 0.836 | 0.994 / 0.888 | 0.998 / 0.989 | 0.764 / 0.769 | 0.942 / 0.861 | 0.938 / 0.507 | 0.962 / 0.945 | 0.529 / 0.523 | 12.7% / 31.2% |

Combined = per scene group, stage one × weighted stage two, averaged over the 225 groups
(`run_pdm_score.py:calculate_individual_mapping_scores`), so it is far below either stage.

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
   log replay; navhard is reactive with synthetic stage two scenes, and stage two EC also drops (0.572 → 0.546 /
   0.523). Gains on navhard need either reactive rewards or a better scene representation (RL-3, LoRA trainable).
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
