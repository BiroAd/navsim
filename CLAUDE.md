# CLAUDE.md — NAVSIM VLM trajectory-planning thesis

Fork of the NAVSIM v2 devkit. Thesis model: `vision_llm_lora` agent (branch `vision_llm_third_phase`;
phase branches `vision_llm_{lora,first_stage,second_stage,third_phase}` on origin). Other agents are baselines.
Upstream NAVSIM code (simulation, PDM scoring, metrics, dataset loaders) is read-only.
One deliberate exception: `navsim/evaluate/pdm_score.py` carries upstream's human-filter fix (commit 359c7f72,
ported 2026-11-05; filter-on scores before that date are pre-fix EPDMS*, see docs/final_evaluation.md).
Open issues found during setup: [docs/vision_llm_lora_findings.md](docs/vision_llm_lora_findings.md).
All training runs, their settings and best metrics, data artifacts and open items: [docs/experiments.md](docs/experiments.md)
(read it first in a new session; update it when a run finishes or starts). Offline checkpoint evaluation (proxy PDM
score per epoch, score-head selection, how to run it): [docs/checkpoint_evaluation.md](docs/checkpoint_evaluation.md).
Official navtest/navhard scores: [docs/final_evaluation.md](docs/final_evaluation.md).
RL theory (objective, optimal policy, KL bounds): [docs/theoretical.md](docs/theoretical.md).

## Guardrails (apply to all work and every skill)
- Never run `sbatch`/`srun`/`salloc`/`scancel`, `pip`/`conda` installs, training, evaluation, dataset/metric
  caching, or HF model downloads. Write the command; the user runs it.
- Never change model architecture, `*_config.py`, Hydra YAML, optimizer, scheduler, or loss unless the user
  asks for that specific change. Propose a diff instead.
- Report a bug only with evidence (see Reporting). If the claim depends on library internals
  (transformers remote code, PEFT, Lightning), cite the installed source or mark it UNVERIFIED.
- Write UNKNOWN instead of guessing (model dims, dataset sizes, cluster paths, library versions).
- Never print, copy, or commit secrets (the repo contains one; see findings F1).

## Reporting findings (format shared by all skills)
`[CONFIRMED|LIKELY|UNVERIFIED] [high|med|low] path:line — claim`
then: code excerpt → failure mechanism → effect on trajectories/PDMS → suggested check or fix (not applied).
CONFIRMED = follows from repo code alone, or was reproduced. LIKELY = depends on unread library code.

## vision_llm_lora: verified data flow  (navsim/agents/vision_llm_lora/)
Many steps have a flag whose config default reproduces the original run and whose agent-YAML value is the fix
(see findings). "default → YAML" below. Checkpoints from before 2026-10-26 need the defaults (listed in the YAML).
1. Features: `VisionLLMLoRAFeatureBuilder` (subclass of `vision_llm/vision_llm_features.py`; cache name still
   `vision_llm_feature`). Current frame only, 12 tiles + a 13th image → `camera_feature [13,3,tile,tile]`.
   `camera_layout` original (crop `[316:-316, 512:-512]`, l0,f0,r0 over l2,b0,r2) → panorama (l0,f0,r0 over
   r1,b0,l1); `thumbnail` canvas → front_zoom; tile 112 → 128; `swap_red_blue` true → false;
   `image_normalization` imagenet → qwen (0.5/0.5). Also `velocity[2]`, `acceleration[2]` (ego frame, raw m/s, m/s²)
   and `driving_command` (argmax of a one-hot; 0 left, 1 straight, 2 right, 3 unknown, checked against heading
   change in the logs). Training reads cached features (`use_cache_without_dataset: true`), so builder changes
   need re-caching into a fresh `cache_path`.
2. Vision: `image_layout` (config default and YAML `strips`; `tiles` for runs before 2026-10-26): `strips`
   reassembles tiles 0–5 and 6–11 into the two canvas rows (bit-exact; the cached tiles are crops), so the tower
   sees front strip H×6W + rear strip H×6W + the 13th image; `tiles` feeds the 13 images separately. No re-caching.
   `_patchify` pads each image to a multiple of 32 px (112→128; none at 128), duplicates it ×2 along time, patches
   16², `patch_order` raster → merge_window; 2×2 merge → 96 + 96 + 16 (strips) or 13 × 16 (tiles) = 208 tokens. The
   tower returns `BaseModelOutputWithPooling`; the code uses `pooler_output` (merged, D). `image_tokens: qwen`
   requires `patch_order: merge_window` (checked in `__init__`).
3. Prompt: ChatML string, tokenized once and cached per (images, tokens/image), one row per command if
   `command_in_prompt`. `image_tokens` custom (`<img> <IMG_CONTEXT>×208 </img>` with a space token between
   placeholders; new tokens on identical untrained rows) → qwen (`<|vision_start|><|image_pad|>×16<|vision_end|>`
   per image). After the images: `Current velocity:<EGO_STATE>, acceleration:<EGO_STATE>.` if `ego_state_tokens`,
   `Navigation command: go left|go straight|go right|not given.` if `command_in_prompt` (both false → true).
   `qwen` also ends the prompt with Qwen's empty think block (`<think>\n\n</think>\n\n`). Image embeddings replace
   the image placeholders; 4 learned planner queries (+ position embedding) are appended.
   Prompt length: 452 tokens (custom) / 288 (qwen + ego + command, tiles) / 268 (same, strips: 3 image blocks).
   The prompt cache is keyed by the tokens per image. Positions: `custom` passes no `position_ids` (1-D); `qwen`
   passes multimodal RoPE `position_ids` (3, B, L) from `self.model.get_rope_index`, cached with the prompt; image
   tokens get (t, h, w) on their merged grid (4×24 per strip, 4×4 per tile), the planner queries continue the text
   positions (F14).
   Positions only act in the 8 full-attention layers, on 64 of 256 head dims.
4. LLM: `model.language_model` (Qwen3.5 text, D = 2560, 32 layers: 24 Gated DeltaNet + 8 full attention;
   optionally PEFT-wrapped) → `last_hidden_state[:, -4:]`.
5. Ego, late fusion (`late_ego_fusion` true → false; when false these modules are not created): vel/acc MLPs
   (raw inputs) + direction embedding → `EgoFusionMLP` → [B,D] × clamp(scale,0,1) → head. Ego in the LLM (if
   `ego_state_tokens`): `EgoStateTokens` scales (x, y) by 15 m/s / 2 m/s², adds sin/cos at π·2^k (k<4), one MLP
   per quantity, RMSNorm with gain initialized to the embedding RMS (0.0130), injected at the `<EGO_STATE>`
   placeholders. With the YAML values, ego state reaches the head only through the LLM. `n_ego_tokens` is unused.
6. Head: `TrajectoryClassifier` concat(4D, + D ego if `late_ego_fusion`) → 2048 → K logits, tanh soft-cap ±30 ×
   exp(clamp(logit_scale,±2)). The input size differs between the two settings, so checkpoints don't transfer
   (`load_state_dict(strict=False)` still raises on shape mismatches).
7. Output: argmax over K=16384 codebook entries → `codebook[id]` [8,3] (x, y, yaw; ego frame; 0.5–4.0 s).
   No autoregression; `teacher_forcing_*` arguments are ignored.
- Loss (`vision_llm_lora_loss.py`): `trajectory_token_loss_weight` × soft-label CE; labels = softmax(−d/T),
  d = Σ(Δx²+Δy²) + 2.5·Σwrap(Δyaw)², T = `label_temperature` (1.0; code fallback 0.7); optional terms, each 0 by
  default: `expected_distance_loss_weight` × Σ_k p_k·ADE_k, `pdm_score_loss_weight` × BCE of predicted per-candidate
  PDM sub-scores, `refinement_loss_weight` × L1 of the refined trajectory. ADE/FDE are metrics only (of the output
  trajectory: argmax entry, or refined). `trajectory_continuous_loss_weight` is unused. Returns 8 values: the 7 that
  `agent_lightning_module.py` unpacks plus a dict of extra metrics it logs as `<train|val>/<name>`: unweighted loss
  terms (`loss_terms/*`), per-sub-score BCE/MAE/accuracy (`pdm_scores/*`), anchor ADE/FDE (`refine/*`).
- Codebook: `codebook/16384_8step.npy` (16384,8,3), built by `codebook/codebook_builder.py` from `16384.npy`
  (40 steps). The loss keeps its own global codebook copy loaded from `config.codebook_path`.
- `total_training_steps` = batches per GPU per epoch (config default 2660 = ⌈85109 / (8 × 4)⌉; the SLURM script
  recomputes it from TRAIN_SAMPLES). navtrain = 103288 scenes in the training cache: train 85109 (978 logs), val
  18179 (214 logs), split by log via `config/training/default_train_val_test_log_split.yaml` (counts from the
  training log). The original 10639 = ⌈85109 / (2 × 4)⌉.

## Fine-tuning
- dtype: config default fp16 + `default_training.yaml` `precision: 16-mixed`; agent YAML bf16 + `sdpa`, and the
  SLURM script passes `precision=bf16-mixed`. 4/8-bit loading off.
- `training_phase` (1 default | 2 | 3) gates what trains; the whole Qwen backbone is always frozen first.
  1 = heads, ego encoders, planner queries (gradients still pass through the frozen LLM); 2 = + LLM LoRA;
  3 = + vision-tower LoRA (and gradients through the tower). Set per phase, e.g. `++agent.config.training_phase=2`.
  (Replaced `enable_lora`/`train_language_model`/`train_vision_encoder`/`enable_vision_lora`, findings F19.)
- PEFT wrappers: LLM `task_type=FEATURE_EXTRACTION`; vision no `task_type` (plain `PeftModel`; FEATURE_EXTRACTION
  would inject `attention_mask=None` and crash SDPA in the tower). Both keep `base_model.model.` key prefixes.
- LLM LoRA r32/α64/p0.1 on q,k,v,o,gate,up,down (default) → + in_proj_qkv, in_proj_z, out_proj (YAML, findings F18:
  q/k/v/o exist only in the 8 full-attention layers). Vision LoRA r64/α128 on attn.qkv, attn.proj, linear_fc1,
  linear_fc2 (the last two also match the merger).
- AdamW β(0.9,0.95), eps 1e-5, wd 0.01 (0 for bias/norm/lora_B). Name-based groups with
  LR = multiplier × lr_base (5e-4, agent YAML): traj .10, ego .12 (incl. `ego_state_tokens`), lang .08,
  vision .25, "other" = lang LR. `phase1_lr_multiplier` (default None → YAML 1.0) replaces the traj and ego
  multipliers in phase 1 only (head, planner queries, ego encoders/tokens at the full lr_base); the ego-token RMSNorm
  gain keeps the ego multiplier in its own group (it starts at the embedding RMS 0.013). Tested in
  `tests/test_vision_llm_lora_optimizer.py`. Each group decays to (lr_final / lr_base) × its own peak.
- LambdaLR: 10% linear warmup → cosine to lr_final (2.5e-5); total = total_training_steps × num_epochs // accum
  (now 2660×20//4 = 13300 optimizer steps; the original run used 10639×8//16); interval `step`, frequency = accum.
  Lightning 2.2.1 calls the step scheduler only on optimizer steps and fires on `(batch_idx+1) % accum == 0`, so it
  steps once per optimizer step (verified in source).
- Must be kept in sync by hand: `num_epochs`↔`max_epochs` (20), `gradient_accumulation_steps`↔`accumulate_grad_batches`
  (4). The SLURM script passes both sides; `default_training.yaml` still has 8 and 16 (shared with other agents).
- `default_training.yaml`: batch 2/GPU, ddp, 1 node, clip 5.0, seed 0, `early_stopping_patience: 3` on `val/loss`
  (the SLURM script passes `PATIENCE`, default 10).
  Default agent there is `ego_status_mlp_agent`, so always pass `agent=vision_llm_lora_agent`.
  Launch: `scripts/training/run_vision_llm_lora_agent_training_slurm.sh` (defaults: 4 GPUs, batch 8, accum 4 →
  effective 128, 20 epochs; it recomputes total_training_steps = ⌈85109 / (batch × GPUs)⌉ = 2660; phase settings as
  extra Hydra args). The old `nohup` scripts still exist.
- GPU memory per A100-40GB (`scripts/training/run_vision_llm_lora_batch_probe_slurm.sh`, 2026-10-27): batch 2/4/8 →
  phase 1 18.6/25.5/39.3 GB, phase 2 22.1/30.3/OOM, phase 3 24.6/35.0/OOM. Phase 1 batch 8 on 4 GPUs with accum
  peaked at 39.5 of 40.5 GB, flat over 600 steps, ~1.1 batches/s per GPU (~40 min per epoch). Phases 2 and 3 need
  `BATCH_SIZE=4 ACCUM=8`.
- Phase handoff: `agent.checkpoint_path` → `initialize()` loads only head/ego/`ego_state_tokens`/planner/`lora_` keys
  (strict=False; LoRA modules missing from the checkpoint keep lora_B = 0, i.e. no change). It leaves the model on
  CPU (Lightning places it per rank); `compute_trajectory` moves it to `self.device` on the first eval scene.
  `+ckpt_path=` resumes natively in Lightning (skips initialize, strict). `+wandb_id=` resumes the W&B run.
- Checkpoints hold only the parameters that were trainable (`agent_lightning_module.on_save_checkpoint` drops frozen
  ones). A policy-only run (`train_policy_head_only`) therefore saves only the policy head: the offline evaluation
  loads the run's start checkpoint first (`base_checkpoints` in `eval_vision_llm_lora_checkpoints.py`); for the official
  evaluation write full checkpoints with `scripts/evaluation/merge_policy_only_checkpoint.py` (→ `checkpoints/merged/`).
- Eval: `scripts/evaluation/run_vision_llm_lora.sh` → `run_pdm_score.py`, `warmup_two_stage`, sequential worker,
  autocast in `config.torch_dtype` (findings F17).

## Environment / HPC
- Repo: `/project/nr_e2e_vlm/navsim` (also reached as `/home/nr_e2ba/nr_e2e_vlm/navsim`). `bashrc` sets
  `OPENSCENE_DATA_ROOT=/project/nr_e2e_vlm/dataset` (navsim_logs: trainval 1310, test 147 logs; sensor_blobs, maps),
  `NAVSIM_EXP_ROOT=/project/nr_e2e_vlm/exp`, `HF_HOME=/project/nr_e2e_vlm/hf_cache` (Qwen/Qwen3.5-4B present).
  The agent YAML's `codebook_path` uses `NAVSIM_DEVKIT_ROOT`; checkpoint paths in the training/eval scripts still
  point to `/root/workdir/biroa/navsim` (another machine).
- Python env: `/project/nr_e2e_vlm/envs/navsim_vlm` (uv, Python 3.10; `scripts/setup/create_vlm_env.sh`,
  `requirements_vlm.txt`, lock in `requirements_vlm_lock.txt`). Installed: torch 2.8.0, transformers 5.20.2,
  peft 0.21.2, pytorch_lightning 2.2.1, bitsandbytes 0.50.2, flash-linear-attention 0.5.2.
  `requirements_new.txt` does not describe this env.
- SLURM + Lmod (cuda 11.4–13.2, miniconda/3, apptainer). Partitions: `gpu` (A100-SXM4-40GB ×3–4/node, 64 CPU,
  256 GB, 7 d), `ai` (8 GPU/node, type unreported), `test` (1 h), `cpu`. Login node `vn01` has no `python` on PATH;
  call the env's `bin/python` directly.

## Conventions
- black/isort at 120 columns, `.flake8`, pre-commit. Match surrounding style.
- Agent code goes in `navsim/agents/<name>/` with a config dataclass + `planning/script/config/common/agent/<name>.yaml`.
  New config fields default to current behavior. Don't hardcode hyperparameters or add another copy of a value.
- `vision_llm_lora` shares `vision_llm/` builders, `agent_lightning_module.py` and `run_training.py`. Changes
  there affect other agents and cached features, so say so in the change.
- Any change that alters loss, labels, data or schedule for an existing config needs a new flag. Thesis runs must
  stay comparable.

## Testing
- Tests live in `tests/` (`test_vision_llm_lora_{codebook,codebook_scoring,preprocessing,prompt,ego,lora_setup}.py`):
  CPU-only, no network, dataset or HF weights; tiny tensors and stubbed backbones (a tiny random Qwen3.5 text model
  is fine). Run from the repo root: `OMP_NUM_THREADS=4 /project/nr_e2e_vlm/envs/navsim_vlm/bin/python -m pytest
  tests/` (~2–3 min, mostly imports). Still untested: soft-label loss,
  `initialize()` key filtering (param groups and LRs: `test_vision_llm_lora_optimizer.py`).
- Checks that need the 4B model, a GPU or data go to the user as a command (e.g. `trainer.params.fast_dev_run=true`).
- When finishing a change, state what was run, what wasn't, and why.
