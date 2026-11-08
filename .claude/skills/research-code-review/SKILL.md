---
name: research-code-review
description: Independent review of a code change in this NAVSIM VLM repo for correctness, regressions, reproducibility, and missing tests. Use after making changes or before merging/launching a training phase.
argument-hint: "[git range | commit | empty = uncommitted changes]"
context: fork
allowed-tools: Read, Grep, Glob, Bash(git diff:*), Bash(git log:*), Bash(git show:*), Bash(git status:*)
---
Target: $ARGUMENTS (empty: `git diff HEAD` plus untracked files). Review independently. Derive behavior from the
code, not from commit messages, comments, or earlier conclusions. Don't edit files.

1. List changed files, tagged model / loss / features / training loop / config / eval / scripts / upstream NAVSIM.
2. For each hunk, check:
   - **Correctness:** shapes, dtypes, devices; ego frame and yaw wrapping; waypoint/codebook index alignment;
     train vs eval mode; `no_grad`/`detach`; in-place ops on tensors needed by autograd.
   - **Regressions:** grep the callers of changed symbols. Check shared code (`agent_lightning_module.py`,
     `run_training.py`, `vision_llm/` builders, `AbstractAgent`), renamed or removed Hydra keys, checkpoint-key
     compatibility with earlier phases, and training-cache invalidation.
   - **Reproducibility:** absolute paths, hyperparameters outside the config, newly duplicated values,
     settings that exist only in shell scripts, seeds and nondeterminism, large binaries.
   - **Silent experiment changes:** anything that changes loss, labels, data or schedule for an existing config
     without a flag (breaks comparability of thesis results).
   - **Hygiene:** secrets, prints in per-sample paths, commented-out code that replaces behavior.
3. **Tests:** for each behavior change, say whether a test covers it. Propose the missing tests (name + assertion)
   following CLAUDE.md § Testing. Run existing fast CPU tests only if a Python env is available.

Output: verdict (approve / approve with nits / changes requested) → findings in CLAUDE.md format → missing tests →
reproducibility notes. Note whether a change resolves or touches an entry in `docs/vision_llm_lora_findings.md`.
