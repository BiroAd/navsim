"""
CPU-only checks that AgentLightningModule saves only trainable parameters and can still restore a strict state dict.
Run: pytest tests/test_agent_lightning_checkpoint.py
"""

import torch
from torch import nn

from navsim.planning.training.agent_lightning_module import AgentLightningModule


class _Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)          # frozen, like the Qwen backbone
        self.alias = self.backbone               # same tensors under a second name, like vision_llm_lora's embedding
        self.head = nn.Linear(4, 2)              # trained
        self.register_buffer("codebook", torch.ones(3))
        for p in self.backbone.parameters():
            p.requires_grad_(False)


def _module() -> AgentLightningModule:
    return AgentLightningModule(_Agent())


def test_frozen_parameters_are_left_out_of_the_checkpoint():
    module = _module()
    checkpoint = {"state_dict": module.state_dict(keep_vars=True)}
    module.on_save_checkpoint(checkpoint)
    assert sorted(checkpoint["state_dict"]) == ["agent.codebook", "agent.head.bias", "agent.head.weight"]
    assert checkpoint["frozen_parameters_omitted"]


def test_omitted_parameters_are_refilled_so_a_strict_load_works():
    trained = _module()
    with torch.no_grad():
        trained.agent.head.weight.fill_(3.0)
    checkpoint = {"state_dict": dict(trained.state_dict())}
    trained.on_save_checkpoint(checkpoint)

    fresh = _module()
    fresh.on_load_checkpoint(checkpoint)
    fresh.load_state_dict(checkpoint["state_dict"], strict=True)
    assert torch.equal(fresh.agent.head.weight, torch.full((2, 4), 3.0))
    assert torch.equal(fresh.agent.backbone.weight, fresh.state_dict()["agent.backbone.weight"])


def test_agents_without_frozen_parameters_are_unchanged():
    module = _module()
    for p in module.parameters():
        p.requires_grad_(True)
    state = dict(module.state_dict())
    checkpoint = {"state_dict": dict(state)}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint["state_dict"].keys() == state.keys()
    assert "frozen_parameters_omitted" not in checkpoint
