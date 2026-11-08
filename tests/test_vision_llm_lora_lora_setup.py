"""
CPU-only checks for the vision_llm_lora training-phase flags and LoRA wrapping (docs/vision_llm_lora_findings.md F19).
No network, dataset or HF weights: flag checks stop the constructor right after validation, and `_attach_lora`
runs on tiny random Qwen3.5 models.
Run: pytest tests/test_vision_llm_lora_lora_setup.py
"""

from types import SimpleNamespace

import pytest
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel, Qwen3_5VisionModel

from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_model import VisionLLMLoRAModel

PHASES = (1, 2, 3)


class _PassedValidation(Exception):
    pass


def _construct(monkeypatch, **flags):
    """Run VisionLLMLoRAModel.__init__ up to its first load (the codebook), which comes after the flag checks."""

    def stop(self, path):
        raise _PassedValidation

    monkeypatch.setattr(VisionLLMLoRAModel, "_init_codebook", stop)
    VisionLLMLoRAModel(TrajectorySampling(time_horizon=4, interval_length=0.5), VisionLLMLoRAConfig(**flags))


@pytest.mark.parametrize("phase", PHASES)
def test_valid_training_phases_pass(monkeypatch, phase):
    with pytest.raises(_PassedValidation):
        _construct(monkeypatch, training_phase=phase)


@pytest.mark.parametrize("phase", [0, 4, "3"])
def test_invalid_training_phase_raises(monkeypatch, phase):
    with pytest.raises(ValueError):
        _construct(monkeypatch, training_phase=phase)


def test_qwen_tokens_need_merge_window_order(monkeypatch):
    with pytest.raises(ValueError):
        _construct(monkeypatch, image_tokens="qwen", patch_order="raster")


def test_phase3_vision_lora_trains_through_checkpointing():
    torch.manual_seed(0)
    text = Qwen3_5TextModel(
        Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            full_attention_interval=4,
        )
    )
    visual = Qwen3_5VisionModel(
        Qwen3_5VisionConfig(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=24,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            num_position_embeddings=64,
        )
    )
    for p in list(text.parameters()) + list(visual.parameters()):  # as __init__ does before _attach_lora
        p.requires_grad_(False)
    stub = SimpleNamespace(model=SimpleNamespace(language_model=text, visual=visual))
    config = VisionLLMLoRAConfig(print_trainable_parameters=False, training_phase=3)

    VisionLLMLoRAModel._attach_lora(stub, config)

    trainable = [n for n, p in stub.model.visual.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)

    stub.model.visual.train()
    out = stub.model.visual(hidden_states=torch.randn(128, 1536), grid_thw=torch.tensor([[1, 8, 8]] * 2))
    out.pooler_output.sum().backward()

    lora_b = [p.grad for n, p in stub.model.visual.named_parameters() if "lora_B" in n]
    assert lora_b and all(g is not None for g in lora_b)
    assert any(g.abs().sum() > 0 for g in lora_b)
