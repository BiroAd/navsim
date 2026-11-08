# vision_llm_lora — findings from initial repository discovery

Recorded 2026-10-25 on branch `vision_llm_third_phase` @ `39ea18a`. Static reading only: nothing was run.
Format and confidence labels follow `CLAUDE.md` § Reporting. Nothing here has been fixed.
Update an entry's status when you resolve it, and delete it once the fix is merged.

## F1 `[CONFIRMED] [high]` navsim/planning/script/run_training.py:150 — WandB API key hardcoded
`wandb.login(key="...")`. The key is in git history and pushed to GitHub.
Fix: rotate the key, then read it from the `WANDB_API_KEY` environment variable instead.
Status: open

## F2 `[CONFIRMED] [high]` navsim/agents/vision_llm_lora/vision_llm_lora_config.py:54,55,66 — LoRA disabled as committed
`train_vision_encoder`, `train_language_model` and `enable_lora` are all `False`, and
`scripts/training/run_vision_llm_lora_agent_training.sh` passes no overrides. So only the heads, ego encoders and
planner queries train, while the `lang`/`vision` LR multipliers (0.08/0.25) apply to empty groups.
UNKNOWN: whether the third-phase runs used command-line overrides that were never committed.
Status: open (confirm what was intended)

## F3 `[CONFIRMED] [med]` navsim/agents/vision_llm_lora/vision_llm_lora_model.py:917-932, 1051 — ego state is not in the LLM sequence
The ego embedding goes only to `TrajectoryClassifier` (late fusion), so the planner queries can't condition on
speed or driving command. `config.n_ego_tokens` is unused. This may be by design; it's a thesis-relevant
architectural fact.
Status: addressed 2026-10-26 behind two flags (config defaults `false` = original; agent YAML sets both `true`),
late fusion kept in both cases. `command_in_prompt` writes "Navigation command: go left / go straight / go right /
not given." after the images. The one-hot order was checked on 250 trainval logs: mean heading change over the next
4 s is +20°, +0.2°, −22° for indices 0, 1, 2. All four phrasings are 7 tokens, so the prompt is cached per command
and needs no padding. `ego_state_tokens` adds "Current velocity:<EGO_STATE>, acceleration:<EGO_STATE>." The
placeholders are replaced by `EgoStateTokens`: (x, y) / fixed scale (15 m/s, 2 m/s², ~p99 of the same logs), sin/cos
at π·2^k (k < 4), MLP, then RMSNorm with a gain initialized to the embedding RMS (0.0130), one MLP per quantity.
The prompt grows from 267 to 284 tokens. `tests/test_vision_llm_lora_ego.py`.
`late_ego_fusion` (config default `true`; agent YAML `false`) removes the late path: the velocity/acceleration/
direction encoders and `EgoFusionMLP` are not created and the head reads only the 4 planner queries (input 4D
instead of 5D), so the model can't shortcut from ego state to trajectory past the LLM. `agent_lightning_module.py`
(shared): `grad/ego_mlp` and `grad_ratio/ego_mlp` follow `ego_fusion_mlp` if present, else `ego_state_tokens`;
with both present, `grad/ego_state_tokens` is logged as well.

## F4 `[CONFIRMED] [med]` codebook subsampling indices disagree
- `codebook/codebook_builder.py:45` → `[4, 9, …, 39]` (produced `16384_8step.npy`)
- `vision_llm_lora_model.py:497` and `vision_llm_lora_loss.py:21` → `[0, 5, …, 35]` for 40-step input

This only matters if `codebook_path` points to a raw 40-step file: model and loss would then use templates
shifted by 4 frames relative to the committed 8-step file.
Resolved 2026-10-26: in `16384.npy`, pose i is at t = (i + 1) · 0.1 s (straight entries: median x[0]/x[39] = 0.025,
x[4]/x[39] = 0.1255), so `[4, 9, …, 39]` = 0.5 … 4.0 s is correct, and `16384_8step.npy` equals `16384.npy[:, 4::5]`
exactly. Model and loss now share `vision_llm_lora_codebook.load_codebook` with those indices
(`tests/test_vision_llm_lora_codebook.py`). No effect on runs that use the 8-step file.
Status: fixed

## F5 `[CONFIRMED] [med]` navsim/planning/training/agent_lightning_module.py — shared module now specific to vision_llm_lora
`_step` unpacks a 7-tuple from `compute_loss`, and `on_before_optimizer_step` reads `model.traj_classifier` and
`model.ego_fusion_mlp` whenever the agent has `_vlm_model`. On this branch, transfuser, ego_mlp and vision_llm
training will fail.
Status: open

## F6 `[CONFIRMED] [low]` requirements_new.txt — listed environment cannot run this code
It lists `torch 2.2.2+cu118`, but the model uses `nn.RMSNorm` (added in torch 2.4). `torchaudio 2.5.1+cu121`
also mismatches. The actual training environment is UNKNOWN.
Status: superseded for this cluster (2026-10-26) by `requirements_vlm.txt` + `scripts/setup/create_vlm_env.sh`:
Python 3.10 via uv (transformers ≥ 5.2.0 is required for Qwen3.5 and needs Python ≥ 3.10), torch 2.8.0,
transformers 5.20.2, flash-linear-attention 0.5.2 (Triton; 0.6.0 needs Triton ≥ 3.5 despite its metadata, torch
2.8 ships 3.4) for Qwen3.5's 24 Gated DeltaNet layers (8 of 32 text
layers are full attention). Attention uses PyTorch SDPA: the flash-attn and causal-conv1d wheels need glibc 2.32
(checked with objdump) and Komondor's RHEL 8 has 2.28 (torch needs ≤ 2.28). The resolved set goes to
`requirements_vlm_lock.txt`. GPU check reached CUDA 12.8 / bf16 on an A100; the kernel checks are pending.

## F7 `[CONFIRMED] [med]` absolute paths from another machine
`codebook_path` (`vision_llm_lora_config.py:36`), `bashrc`, both run scripts, and the checkpoint paths all use
`/root/workdir/biroa/...`, which is inaccessible on this cluster. No conda env, HF cache, `exp/` dir or
`NAVSIM_*` variables were found here.
Status: partly resolved (2026-10-26). `bashrc` and `exp/` now point to `/project/nr_e2e_vlm`. The agent YAML sets
`codebook_path: ${oc.env:NAVSIM_DEVKIT_ROOT}/...`, which covers caching, training and eval. Overriding config
fields that are not in the agent YAML needs `++agent.config.<field>=...`. The checkpoint paths in the training and
eval scripts are unchanged. Still missing: the Python env (uv, `scripts/setup/create_vlm_env.sh`) and the HF
model cache.

## F8 `[CONFIRMED] [low]` navsim/planning/script/run_training.py:176 — best-checkpoint filename metric
`filename="best-{epoch:02d}-{val_loss:.2f}"`, but the logged key is `val/loss`, so filenames always read
`val_loss=0.00` (seen in the paths inside the run scripts). Monitoring itself uses `val/loss` and works.
Status: open

## F9 `[CONFIRMED] [high]` navsim/agents/vision_llm/vision_llm_features.py:103 — red and blue channels swapped
`Cameras.from_camera_dict` loads images with `np.array(Image.open(...))` (`navsim/common/dataclasses.py:81`), which
is RGB. The builder then applies `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` to that RGB array, so every tile and the
thumbnail reach the model as BGR, normalized with RGB ImageNet statistics.
Effect: the pretrained Qwen visual tower sees red and blue exchanged (red lights and brake lights appear blue).
Train and eval go through the same builder, so this is not a train/test mismatch, but a frozen or LoRA-adapted
encoder works off-distribution on colour cues.
Status: fixed 2026-10-26 behind `swap_red_blue` (config default `true` = original; agent YAML sets `false`), in
`VisionLLMLoRAFeatureBuilder`. Normalization is now selectable too: `image_normalization: qwen` uses
mean = std = 0.5 from Qwen3.5-4B's `preprocessor_config.json`; ImageNet statistics put inputs at roughly
[-2.1, 2.6] instead of [-1, 1]. Needs re-caching. The cache file name `vision_llm_feature` does not change, so
use a fresh `cache_path`.

## F10 `[CONFIRMED] [med]` vision_llm_lora_model.py:832-837 — 23% of every tile is zero padding
Tiles are 112×112, but the 2×2 merger needs a grid divisible by 32 px, so `_preprocess_images` pads each tile to
128×128 at the right and bottom. That is 15 of 64 patches per tile, and 7 of the 16 merged tokens per tile
contain padding (1 − (112/128)² = 23% of the area). Zero after normalization is the dataset mean colour, so
the 12 tiles of one continuous panorama are separated by artificial grey seams.
Status: fixed 2026-10-26: the LoRA builder now takes the tile size from `camera_width`/`camera_height` (default
112 = original; agent YAML sets 128, stitched 768×256). Same 16 tokens per tile, no padding. Needs re-caching.

## F11 `[CONFIRMED] [med]` vision_llm_features.py:78-79,102 — tight crop and 4× downscale
Cameras are 1920×1080 (checked on navtrain JPEGs). `img[316:-316, 512:-512]` keeps the central 896×448
(47% of the width, 41% of the height; the comment says 4:1, the crop is 2:1). The 2688×896 panorama is then
resized to 672×224, so each camera ends up 224×112 px. The 112×112 thumbnail also squeezes the 3:1 panorama to 1:1.
Measured from the trainval calibration (`cam_intrinsic`, `sensor2lidar_rotation`): every camera has 63.7° × 38.5°
FOV; the crop keeps 32.3° × 16.5°. The 6 cropped views cover 194° of 360°, leaving blind wedges of 23° between
F0 and L0/R0 (16°–39° off the heading on both sides), 53° beside the car (L0→L2, R0→R2) and 6° at the rear.
Vertically only ±8° around the optical axis remains, so e.g. a 5 m traffic light more than ~10° up is cut off.
Uncropped, the same cameras overlap at the front (F0/L0 −8.7°). TransFuser's crop (`transfuser_features.py:65-67`)
keeps the full F0 width and 36.9° of height.
Design decision rather than a bug; thesis-relevant.
Status: addressed 2026-10-26 behind `camera_layout` (config default `original`; agent YAML sets `panorama`):
top row L0|F0|R0 with TransFuser's horizontal crop, bottom row full-width R1|B0|L1 (side cameras instead of
L2/R2), each row centre-cropped to 6:1. Same 13 × 16 tokens. Coverage 194° → 332° (largest gap ≤ 6°), front vertical
FOV 16.5° → 24.9°, front resolution 7.9 → 5.3–5.7 px/°, rear 4.0 px/°. The agent's sensor config loads L1/R1 for
this layout. L1/R1 verified present for all 103,288 navtrain scenes and all 75,122 test-split frames.
The 13th image is now selectable with `thumbnail` (config default `canvas` = original squeezed canvas; agent YAML
sets `front_zoom`): F0 rows 290:674, cols 768:1152, i.e. 14.2° wide and about −5.6° to +8.5° around the horizon
(fx 1545, cy 560, ~1.5° downward pitch) at ~9 px/°, for lead vehicles and traffic lights ahead. This offsets the
panorama's lower front resolution at the same 208 tokens. `none` drops it (192 tokens). Needs re-caching.

## F12 `[CONFIRMED] [low]` vision_llm_features.py:112 — print on every sample
`print(f"Pixel values shape: ...")` runs once per scene: ~103k lines through Ray's `log_to_driver` during navtrain
caching, and once per scene during evaluation. Removing it does not change the features.
Status: fixed 2026-10-26 (print removed).

## F13 `[LIKELY] [high]` vision_llm_lora_model.py `_preprocess_images` — patches fed in raster order
The Qwen2-VL image processor (which Qwen3.5 uses, `image_processor_type: Qwen2VLImageProcessorFast`) emits
patches grouped by 2×2 merge window: `[grid_h/2, grid_w/2, 2, 2]`. The vision tower relies on that order: its
position ids are built in window order (`transformers/vision_utils.py` `get_vision_position_ids`,
`block_shape` + `transpose(1, 2)`), and `Qwen3_5VisionPatchMerger` concatenates every 4 consecutive patches.
`_preprocess_images` emitted row-major order, so each merged token fused a 1×4 horizontal strip instead of a 2×2
square, and patches received the rotary/absolute positions of other patches.
LIKELY rather than CONFIRMED: read from transformers `main` (2026-10), not the installed version (UNKNOWN). The
window order has been the Qwen2-VL convention since that model was released.
Status: fixed 2026-10-26 behind `patch_order` (config default `raster` = original; agent YAML sets
`merge_window`). Model-side only, so the cache is not affected. `tests/test_vision_llm_lora_preprocessing.py`
checks the new order against a copy of the processor's `patchify`.

## F14 `[CONFIRMED] [med]` vision_llm_lora_model.py:1044 — image tokens get 1-D positions in the LLM
`language_model(inputs_embeds=..., attention_mask=...)` is called without `position_ids`, so all 208 image tokens
get consecutive 1-D positions. Qwen3.5 was pretrained with multimodal RoPE, which gives image tokens 2-D
(height, width) positions (`get_rope_index` / `compute_3d_position_ids` in `modeling_qwen3_5.py`). The LLM
therefore sees the 13 images as one flat sequence of tokens.
Checked 2026-10-26 in transformers 5.20.2: positions only matter in the 8 full-attention layers (the Gated DeltaNet
layers take none), and there only on 64 of 256 head dims (`partial_rotary_factor` 0.25, `mrope_section` [11, 11, 10]
interleaved over t, h, w). `get_rope_index` gives an image's tokens t = s, h = s + row, w = s + col on the merged
4×4 grid and continues the text at s + 4.
Status: fixed 2026-10-26 as part of `image_tokens: qwen` (no separate flag; `custom` still passes no position_ids):
`_encode_inputs` calls `self.model.get_rope_index` once per cached prompt (`mm_token_type_ids` = image-pad
positions), appends text positions for the planner queries, and passes the (3, B, L) tensor as `position_ids`, as
`Qwen3_5Model.forward` does. The last prompt position drops from 287 to 131. The same setting appends Qwen's empty
think block (`<think>\n\n</think>\n\n`) after `assistant\n`, so the planner queries sit where a non-thinking
answer starts. Checked: explicit 1-D positions reproduce the default output and 2-D ones change it through the PEFT
wrapper (tiny random model). `tests/test_vision_llm_lora_ego.py`.

## F15 `[LIKELY] [med]` vision_llm_lora_model.py:134-136, 513-527 — new delimiter tokens instead of Qwen's own
The prompt wraps the image in newly added `<img>`/`</img>` tokens (and uses a new `<IMG_CONTEXT>` placeholder).
Their embedding rows come from `resize_token_embeddings` and never train (`_unfreeze_new_token_embeddings` is
commented out, line 365). Qwen3.5 has pretrained `<|vision_start|>` (248053), `<|vision_end|>` (248054) and
`<|image_pad|>` (248056) tokens in its config, and wraps each image separately; here all 13 images form one
block.
Checked 2026-10-26 on the installed tokenizer and weights: the tokenizer has 248,077 tokens, the embedding 248,320 rows,
so `resize_token_embeddings(248080)` shrinks the matrix and the new tokens get ids 248077–248079, i.e. existing padding
rows. Those rows are identical (norm 0.4447, same as e.g. row 248200), so `<img>` and `</img>` are indistinguishable.
Status: fixed 2026-10-26 behind `image_tokens` (config default `custom` = original; agent YAML sets `qwen`): each image
is wrapped in `<|vision_start|>` + 16 × `<|image_pad|>` + `<|vision_end|>`, nothing is added to the tokenizer, and
the embedding matrix is not resized. Model-side only, no re-caching. `tests/test_vision_llm_lora_prompt.py`.

## F16 `[CONFIRMED] [med]` vision_llm_lora_model.py `_build_prompt` — spaces between image placeholder tokens
`_build_prompt` joins the 208 `<IMG_CONTEXT>` tokens with `" "`. Reproduced 2026-10-26 with the installed Qwen3.5
tokenizer: the image block takes 420 tokens instead of 210, i.e. a `Ġ` token after every placeholder (209 in all),
and the full prompt is 452 tokens.
Status: fixed 2026-10-26 with F15 (`image_tokens: qwen`): no separators; the prompt is 267 tokens.

## F17 `[LIKELY] [med]` vision_llm_lora_agent.py:328 — evaluation autocasts to fp16 over bf16 weights
`compute_trajectory` ran under `torch.autocast(device_type="cuda", dtype=torch.float16)` regardless of
`config.torch_dtype`. With the agent YAML's `torch_dtype: bfloat16` and `bf16-mixed` training, evaluation computed
in a different precision than training, and fp16's smaller range can overflow Qwen activations to inf/NaN.
Status: fixed 2026-10-26: autocast uses `config.torch_dtype` (disabled for float32). Checkpoints with
`torch_dtype=float16` evaluate as before.

## F18 `[CONFIRMED] [high]` vision_llm_lora_config.py `llm_lora_target_modules` — LoRA misses 24 of 32 token mixers
Qwen3.5-4B is hybrid: 8 full-attention layers (`self_attn.{q,k,v,o}_proj`) and 24 Gated DeltaNet layers
(`linear_attn.in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`; transformers 5.20.2
`modeling_qwen3_5.py`). The default targets match only the former, so 75% of layers adapt their MLP but not how
they mix tokens, including the image tokens.
Status: fixed 2026-10-26 in the agent YAML (config default unchanged): targets add `in_proj_qkv`, `in_proj_z`,
`out_proj` (`in_proj_a`/`in_proj_b` output one value per head, too small for r = 32).
`tests/test_vision_llm_lora_prompt.py` checks that every layer's mixer and MLP get an adapter.

## F19 `[CONFIRMED] [med]` vision_llm_lora_model.py `__init__` / `_attach_lora` — phase flags and vision LoRA wrapping
Flags: nothing checked their combination. `train_vision_encoder` + `enable_vision_lora` without `enable_lora` skipped
`_attach_lora`, so the whole vision tower (333.5M params, counted from the safetensors) trained in full, without
gradient checkpointing, in the "other" optimizer group, with no error. `enable_vision_lora` without
`train_vision_encoder` silently gave no vision LoRA; `train_vision_encoder` alone, or `train_language_model` without
`enable_lora`, meant a full fine-tune.
Wrapping: the vision `LoraConfig` used `task_type=FEATURE_EXTRACTION`, whose forward (peft 0.21.2
`PeftModelForFeatureExtraction.forward`) adds `input_ids`/`attention_mask`/`inputs_embeds=None`. The tower passes
`attention_mask` on to SDPA next to its own, so it raised `TypeError ... multiple values for keyword argument
'attention_mask'` (reproduced on a tiny tower); a monkey-patch on every block's `forward` stripped it.
Status: fixed 2026-10-26. The four flags are replaced by one `training_phase` (1 | 2 | 3; default 1 = the
committed all-false setting), so the inconsistent combinations can't be expressed; `__init__` freezes the whole
backbone and adds LLM LoRA from phase 2, vision LoRA in phase 3. The vision `LoraConfig` has no `task_type`, so `get_peft_model`
returns a plain `PeftModel` that forwards only the given arguments, and the stripper is gone. State-dict keys are
identical to before (`base_model.model.blocks...`), so checkpoints still load. The LLM keeps FEATURE_EXTRACTION.
`tests/test_vision_llm_lora_lora_setup.py` (incl. phase-3 forward/backward through gradient checkpointing).

## F20 `[CONFIRMED] [med]` vision_llm_lora_model.py `_preprocess_images` — 12 separate tiles instead of the canvas rows
The 6×2 canvas was fed to Qwen's vision tower as 12 square images (InternVL-style `dynamic_preprocess`; InternVL
tiles because its ViT is fixed-size, Qwen's is native-resolution). The tower attends only within an image, so
nothing crossed tile seams, and each tile got its own 8×8 position grid (tower) / 4×4 M-RoPE grid (LLM).
Status: changed 2026-10-26 behind `image_layout` (`strips`, config default and YAML, per the user; `tiles` =
before). Tiles 0–5 and 6–11 are reassembled into the front and rear canvas rows (checked bit-exact against the
builder's canvas), one image each; a 13th image stays separate. Not one 768×256 image: the rear row is not below
the front row. Same 832 patches and 208 tokens; vision attention FLOPs ~5 → ~29 GFLOP per sample of ~530 total,
memory unchanged under SDPA (per-image calls without a mask). Model-side only, no re-caching. Prompt 288 → 268
tokens. `tests/test_vision_llm_lora_preprocessing.py` (strips vs the reference patchify of the canvas rows),
`tests/test_vision_llm_lora_ego.py` (4×24 M-RoPE grid per strip).

## To verify (UNVERIFIED, needs library source or a runtime check)
- ~~LR scheduler frequency~~: resolved 2026-10-26 from Lightning 2.2.1 `training_epoch_loop.py:345-375`. The step
  scheduler is only called on optimizer steps, and fires when `(batch_idx + 1) % frequency == 0`, so
  `frequency = accumulate_grad_batches` steps once per optimizer step (except a skipped step at a ragged epoch end).
- ~~Image normalization~~: resolved, Qwen3.5-4B uses mean = std = 0.5 (see F9). From the same `config.json`:
  text hidden size D = 2560, vision tower 24 layers × 1024, patch 16, temporal 2, merge 2.
- Whether `<img>`, `</img>` and `<IMG_CONTEXT>` already exist in the Qwen tokenizer, and how the new embedding rows
  are initialized by `resize_token_embeddings`.
- ~~fp16 base weights + `16-mixed` with full fine-tuning~~: moot since 2026-10-26; `training_phase` always freezes
  the backbone, so only fp32 heads and LoRA adapters train.
