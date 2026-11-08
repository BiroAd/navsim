---
name: training-audit
description: Audit vision_llm_lora fine-tuning — effective config, trainable-parameter set, LoRA/PEFT settings, optimizer groups, LR schedule, precision, effective batch size, DDP, and phase-to-phase checkpoint handoff. Use before launching, resuming, or comparing a training phase.
argument-hint: "[phase name and/or planned Hydra overrides]"
allowed-tools: Read, Grep, Glob, Bash(git diff:*), Bash(git log:*), Bash(git show:*)
---
Resolve the effective config by reading files, not by running Hydra: `VisionLLMLoRAConfig` defaults →
`config/common/agent/vision_llm_lora_agent.yaml` → `config/training/default_training.yaml` →
`scripts/training/run_vision_llm_lora_agent_training.sh` → $ARGUMENTS.
Reuse `docs/vision_llm_lora_findings.md` for known issues; report only new ones or changes in status.

1. **Effective-config table:** param | value | source path:line, with overrides marked. Cover lr_base/final,
   group multipliers, LoRA r/α/dropout/targets/DoRA, `training_phase`, dtype/precision, batch size, accumulation,
   devices/nodes, epochs, total_training_steps, warmup, clip, seed, checkpoint_path vs ckpt_path.
2. **Trainable set.** Derive what trains from the freezing and PEFT code. Flag contradictions, e.g. a non-zero
   multiplier for a frozen group, a trainable non-LoRA backbone param (= full fine-tuning), or trainable fp16
   params under GradScaler. Check that the LoRA target names exist in the backbone (installed source or a printed
   module list, else UNVERIFIED). Confirm the name-based grouping in `get_optimizers` puts every trainable param
   in its intended group.
3. **Schedule math.** Optimizer steps/epoch = ceil(batches_per_rank / accum); then total and warmup steps.
   Compare with the scheduler's `total_steps`. Read the installed Lightning source to see how `interval: step` +
   `frequency` interact with accumulation, then state per-group peak LR, when warmup ends, and final LR.
   If the source isn't available, mark UNVERIFIED and say which logged metric (`train/lr_step`) would settle it.
4. **Batch:** effective = batch_size × accumulate_grad_batches × devices × num_nodes. Check that the duplicated
   values listed in CLAUDE.md agree.
5. **Precision:** weight dtype vs autocast vs GradScaler, fp32 heads/adapters, loss dtype, DoRA under fp16.
6. **Distributed:** ddp vs unused params, `sync_dist` on logged metrics, per-rank state (diagnostic lists,
   global codebook cache), seeding, device placement at model init.
7. **Phase handoff:** what `initialize()` keeps and what is re-initialized, resume semantics, W&B resume,
   early-stopping and checkpoint monitor keys.
8. **Memory/throughput** on A100-40GB only if it can be derived; otherwise UNKNOWN.

Output: config table → derived numbers (steps, LR curve, effective batch) → findings in CLAUDE.md format →
proposed changes as diffs (not applied).
