"""
CPU-only checks for ego state inside the vision_llm_lora LLM sequence (docs/vision_llm_lora_findings.md F3).
No network, dataset or HF weights: `_encode_inputs` runs on a stub with a word-level fake tokenizer, a fake visual
tower and an 8-dim embedding.
Run: pytest tests/test_vision_llm_lora_ego.py
"""

import re
from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_model import (
    EgoFusionMLP,
    EgoStateTokens,
    TrajectoryClassifier,
    VisionLLMLoRAModel,
)

D = 8
SPECIAL = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>")


class _WordTokenizer:
    """One id per special token, word, punctuation mark or newline; ids are assigned on first use."""

    def __init__(self):
        self.vocab = {token: i for i, token in enumerate(SPECIAL)}
        self.special = list(SPECIAL)

    def __len__(self):
        return len(self.vocab)

    def add_tokens(self, tokens, special_tokens=False):
        for token in tokens:
            self.special.append(token)
            self.convert_tokens_to_ids(token)

    def convert_tokens_to_ids(self, token):
        return self.vocab.setdefault(token, len(self.vocab))

    def _ids(self, text):
        pattern = "(" + "|".join(re.escape(t) for t in self.special) + ")"
        tokens = []
        for piece in re.split(pattern, text):
            tokens += [piece] if piece in self.special else re.findall(r"\w+|[^\w\s]|\n", piece)
        return [self.convert_tokens_to_ids(t) for t in tokens]

    def __call__(self, texts, return_tensors, padding, add_special_tokens):
        rows = [self._ids(text) for text in texts]
        length = max(len(row) for row in rows)
        return {
            "input_ids": torch.tensor([row + [0] * (length - len(row)) for row in rows]),
            "attention_mask": torch.tensor([[1] * len(row) + [0] * (length - len(row)) for row in rows]),
        }


def _model(config):
    model = VisionLLMLoRAModel.__new__(VisionLLMLoRAModel)
    nn.Module.__init__(model)
    model.config = config
    model.tokenizer = _WordTokenizer()
    model.embedding = nn.Embedding(256, D)
    model.vis_tokens = None

    def visual(hidden_states, grid_thw):
        assert hidden_states.shape[0] == int(grid_thw.prod(-1).sum())
        model.vis_tokens = torch.randn(int(grid_thw.prod(-1).sum()) // 4, D)  # 2x2 merge
        return model.vis_tokens

    # Qwen3.5's real get_rope_index, bound to a stub that only carries the merge size it reads.
    rope = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    rope.get_vision_position_ids = MethodType(Qwen3_5Model.get_vision_position_ids, rope)
    model.model = SimpleNamespace(
        visual=visual,
        get_input_embeddings=lambda: model.embedding,
        get_rope_index=MethodType(Qwen3_5Model.get_rope_index, rope),
    )
    model._patch_size, model._temporal_size, model._patch_dim, model._merger_factor, model._merge_size = 16, 2, 1536, 4, 2
    if config.late_ego_fusion:
        model.velocity_encoder = nn.Linear(2, D)
        model.acceleration_encoder = nn.Linear(2, D)
        model.direction_embedding = nn.Embedding(4, D)
        model.ego_fusion_mlp = EgoFusionMLP(3 * D, 4, D)
    model.ego_state_tokens = EgoStateTokens(D, (15.0, 2.0), 4, embed_rms=0.5)
    model.num_planner_queries = 4
    model.planner_queries = nn.Parameter(torch.randn(4, D))
    model.query_pos_embed = nn.Parameter(torch.zeros(4, D))
    model._prompt_cache = {}
    model._prompt_template = "<|im_start|>user\nVisual context:\n<IMG>\n<EGO>Predict.<|im_end|>\n"
    model._register_special_tokens()
    return model


STRIPS = (96, 96, 16)  # prompt-cache key: tokens per image with the default image_layout="strips" + front zoom


def _features(commands):
    b = len(commands)
    return {
        "camera_feature": torch.randn(b, 13, 3, 128, 128),
        "velocity": torch.tensor([[5.0, 0.0], [6.0, 0.1], [12.0, -0.2]])[:b],
        "acceleration": torch.tensor([[0.5, 0.0], [-1.0, 0.3], [0.0, 0.0]])[:b],
        "driving_command": torch.tensor(commands),
    }


def test_each_sample_gets_its_command_text_and_ego_tokens():
    config = VisionLLMLoRAConfig(
        image_tokens="qwen", patch_order="merge_window", command_in_prompt=True, ego_state_tokens=True
    )
    model = _model(config)
    features = _features([0, 2, 3])

    embeds, mask, _, _, _ = model._encode_inputs(features)

    ids = model._prompt_cache[STRIPS][0][features["driving_command"]]
    words = {i: w for w, i in model.tokenizer.vocab.items()}
    assert [[words[int(i)] for i in row].count("left") for row in ids] == [1, 0, 0]
    assert [[words[int(i)] for i in row].count("right") for row in ids] == [0, 1, 0]
    assert [[words[int(i)] for i in row].count("given") for row in ids] == [0, 0, 1]

    expected_ego = model.ego_state_tokens(torch.stack([features["velocity"], features["acceleration"]], dim=1))
    ego_pos = ids == model.ego_state_id
    assert ego_pos.sum(dim=1).tolist() == [2, 2, 2]
    assert torch.allclose(embeds[:, : ids.shape[1]][ego_pos].view(3, 2, D), expected_ego)

    img_pos = ids == model.img_ctx_id
    assert torch.equal(embeds[:, : ids.shape[1]][img_pos], model.vis_tokens)

    assert embeds.shape == (3, ids.shape[1] + 4, D)
    assert bool(mask.all())


def test_qwen_tokens_get_multimodal_rope_positions():
    config = VisionLLMLoRAConfig(image_tokens="qwen", patch_order="merge_window", command_in_prompt=True)
    model = _model(config)

    _, _, _, pos, _ = model._encode_inputs(_features([0, 2]))

    ids = model._prompt_cache[STRIPS][0][torch.tensor([0, 2])]
    assert pos.shape == (3, 2, ids.shape[1] + 4)

    # Front strip: t constant, (h, w) walks its 4x24 merged grid row by row, offset by the position after
    # <|vision_start|>; <|vision_end|> follows at start + 24 (max(h, w) of the grid), not start + 96.
    img = (ids[0] == model.img_ctx_id).nonzero().squeeze(1)
    first = pos[:, 0, img[:96]]
    start = first[0, 0]
    assert torch.equal(first[0], torch.full((96,), int(start)))
    assert torch.equal(first[1] - start, torch.arange(4).repeat_interleave(24))
    assert torch.equal(first[2] - start, torch.arange(24).repeat(4))
    assert torch.equal(pos[:, 0, img[95] + 1], torch.full((3,), int(start) + 24))

    # Front zoom (last image): its own 4x4 grid.
    zoom = pos[:, 0, img[-16:]]
    assert torch.equal(zoom[1] - zoom[0, 0], torch.arange(4).repeat_interleave(4))
    assert torch.equal(zoom[2] - zoom[0, 0], torch.arange(4).repeat(4))

    # Planner queries: text positions right after the last prompt token, equal in t, h, w.
    queries = pos[:, :, -4:]
    assert torch.equal(queries, (pos[:, :, -5:-4] + 1 + torch.arange(4)))
    assert torch.equal(queries[0], queries[1]) and torch.equal(queries[0], queries[2])


def test_flags_off_keep_a_single_shared_prompt_without_ego_text():
    config = VisionLLMLoRAConfig(image_tokens="qwen", patch_order="merge_window")
    model = _model(config)

    embeds, _, _, _, _ = model._encode_inputs(_features([0, 2]))

    cached_ids = model._prompt_cache[STRIPS][0]
    words = {i: w for w, i in model.tokenizer.vocab.items()}
    assert cached_ids.shape[0] == 1
    assert "Navigation" not in [words[int(i)] for i in cached_ids[0]]
    assert "<EGO_STATE>" not in model.tokenizer.vocab


def test_without_late_fusion_the_head_sees_only_planner_queries():
    config = VisionLLMLoRAConfig(
        image_tokens="qwen",
        patch_order="merge_window",
        command_in_prompt=True,
        ego_state_tokens=True,
        late_ego_fusion=False,
    )
    model = _model(config)

    embeds, _, ego_repr, _, _ = model._encode_inputs(_features([0, 1]))

    assert ego_repr is None
    assert not any(hasattr(model, name) for name in ("velocity_encoder", "acceleration_encoder", "ego_fusion_mlp"))

    head = TrajectoryClassifier(D, num_trajectories=16, num_queries=4, bottleneck_dim=32, late_ego_fusion=False)
    assert head.query_fusion[0].in_features == 4 * D
    logits, *_ = head(SimpleNamespace(last_hidden_state=embeds), ego_repr)
    assert logits.shape == (2, 16)


def test_late_fusion_head_keeps_its_original_input_size():
    head = TrajectoryClassifier(D, num_trajectories=16, num_queries=4, bottleneck_dim=32)
    assert head.query_fusion[0].in_features == 5 * D


def test_ego_state_tokens_match_embedding_scale_and_train():
    tokens = EgoStateTokens(hidden_size=64, scales=(15.0, 2.0), n_frequencies=4, embed_rms=0.013)
    states = torch.tensor([[[5.0, 0.0], [0.5, 0.0]], [[6.0, 0.0], [0.5, 0.0]]])

    out = tokens(states)

    assert out.shape == (2, 2, 64)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 0.013), rtol=1e-3)
    assert not torch.allclose(out[0, 0], out[1, 0])  # 5 vs 6 m/s
    assert torch.allclose(out[0, 1], out[1, 1])  # same acceleration, separate MLP

    out.sum().backward()
    assert all(p.grad is not None for p in tokens.parameters())
