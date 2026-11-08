"""
CPU-only checks for the vision_llm_lora prompt and LoRA targets (docs/vision_llm_lora_findings.md F15, F16, F18).
No network, dataset or HF weights: the prompt builder runs on a stub, and LoRA is attached to a tiny random
Qwen3.5 text model.
Run: pytest tests/test_vision_llm_lora_prompt.py
"""

from pathlib import Path
from types import SimpleNamespace

import yaml

from navsim.agents.vision_llm_lora.vision_llm_lora_model import VisionLLMLoRAModel

AGENT_YAML = Path(__file__).parents[1] / "navsim/planning/script/config/common/agent/vision_llm_lora_agent.yaml"


def _prompt(image_tokens, ego_state_tokens=False, command=None, n_images=13, n_tokens_per_image=16):
    stub = SimpleNamespace(
        config=SimpleNamespace(image_tokens=image_tokens, ego_state_tokens=ego_state_tokens),
        _prompt_template="ctx:\n<IMG>\n<EGO>end",
    )
    return VisionLLMLoRAModel._build_prompt(stub, (n_tokens_per_image,) * n_images, command)


def test_custom_image_tokens_reproduce_original_prompt():
    original = "<img> " + " ".join(["<IMG_CONTEXT>"] * 208) + " </img>"
    assert _prompt("custom") == f"ctx:\n{original}\nend"


THINK = "<think>\n\n</think>\n\n"


def test_qwen_image_tokens_wrap_each_image_without_spaces():
    image = "<|vision_start|>" + "<|image_pad|>" * 16 + "<|vision_end|>"
    assert _prompt("qwen") == "ctx:\n" + image * 13 + "\nend" + THINK


def test_qwen_blocks_follow_each_images_token_count():
    stub = SimpleNamespace(
        config=SimpleNamespace(image_tokens="qwen", ego_state_tokens=False), _prompt_template="<IMG><EGO>"
    )
    blocks = ["<|vision_start|>" + "<|image_pad|>" * n + "<|vision_end|>" for n in (96, 96, 16)]
    assert VisionLLMLoRAModel._build_prompt(stub, (96, 96, 16)) == "".join(blocks) + THINK


def test_ego_text_follows_the_images():
    image = "<|vision_start|>" + "<|image_pad|>" * 16 + "<|vision_end|>"
    ego = "Current velocity:<EGO_STATE>, acceleration:<EGO_STATE>.\nNavigation command: go right.\n"
    assert _prompt("qwen", ego_state_tokens=True, command=2) == "ctx:\n" + image * 13 + "\n" + ego + "end" + THINK
    assert _prompt("qwen", command=3).endswith("\nNavigation command: not given.\nend" + THINK)


def test_yaml_lora_targets_reach_every_qwen35_layer():
    from peft import LoraConfig, get_peft_model
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    targets = yaml.safe_load(AGENT_YAML.read_text())["config"]["llm_lora_target_modules"]
    config = Qwen3_5TextConfig(
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
    model = get_peft_model(Qwen3_5TextModel(config), LoraConfig(r=4, target_modules=targets))

    adapted = {name.split(".lora_A")[0] for name, _ in model.named_parameters() if ".lora_A." in name}
    for idx, layer_type in enumerate(config.layer_types):
        mixer = "self_attn" if layer_type == "full_attention" else "linear_attn"
        assert any(f"layers.{idx}.{mixer}." in name for name in adapted), (idx, layer_type)
        assert any(f"layers.{idx}.mlp." in name for name in adapted), idx
