---
name: architecture-review
description: Review the vision_llm_lora model architecture — tensor shapes, data flow, gradient flow, and Qwen/PEFT integration. Use when changing or questioning vision_llm_lora_model.py, the shared vision_llm feature/target builders, prompt/token handling, or the heads.
argument-hint: "[component, file, or question; default = whole vision_llm_lora model]"
allowed-tools: Read, Grep, Glob, Bash(git diff:*), Bash(git log:*), Bash(git show:*)
---
Scope: $ARGUMENTS (default: `navsim/agents/vision_llm_lora/` plus the `vision_llm/` builders it reuses).
Start from the verified data flow in CLAUDE.md, and re-check it against the current code before relying on it.
Skip issues already listed in `docs/vision_llm_lora_findings.md` unless the code has changed.

1. **Shape table.** Trace one batch: stage | tensor | shape | dtype | device | path:line.
   camera_feature → pixel_values/grid_thw → visual(+merger) → vis_out → prompt_ids/img_pos → prompt_embeds
   (+planner queries) → last_hidden_state → planner slice → ego_repr → fused → logits [B,K] → trajectory [B,8,3].
   Use symbols (B, D, K, L) when a value is only known at runtime.
2. **Inputs.** Crop/stitch/tile arithmetic, camera order, effect of zero padding in normalized space,
   normalization vs the Qwen processor, `driving_command` dtype/range, ego frame and yaw wrapping of GT vs codebook,
   waypoint timing (0.5–4.0 s) vs codebook indices.
3. **Token integration.** Count of `<IMG_CONTEXT>` ids in the tokenized prompt == vis_out tokens per sample.
   Check whether added tokens tokenize as single ids, the effect of `resize_token_embeddings` (row init, tied
   weights), attention mask, and causal visibility of the appended planner queries.
4. **Backbone/PEFT integration.** Attribute paths (`model.visual`, `model.language_model`, `patch_embed.proj`,
   `merger.*`, `blocks`) before and after `get_peft_model`; what `visual(...)` returns; the attention_mask-stripping
   wrapper and the forward hooks. Read the installed transformers/PEFT source or mark UNVERIFIED.
5. **Gradient flow.** For the active flags, list which parameters should receive gradients, then confirm a
   differentiable path to each: `no_grad` around the frozen visual tower, `.detach()` in `TrajectoryClassifier`,
   gradient checkpointing + `enable_input_require_grads` with a frozen base, the non-differentiable argmax decode.
   Flag parameters that are created but unused (relevant under `strategy: ddp`).
6. **Numerics.** fp16 weights under 16-mixed, logit soft-cap/scale, softmax over K=16384, ego scale clamp, RMSNorm dtype.
7. **Train/inference parity.** `compute_trajectory` (eval mode, fp16 autocast, device), and whether the
   `initialize()` prefix filter matches the real parameter names (PEFT `base_model.model.`, `lora_A.default`).

Output: shape table → findings in CLAUDE.md format, most severe first → UNKNOWNs. When a claim needs a runtime
check, give the user a minimal snippet to run. Don't run it yourself.
