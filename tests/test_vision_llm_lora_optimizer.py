"""
CPU-only checks for the vision_llm_lora optimizer parameter groups and their learning rates.
No network, dataset or HF weights: `get_optimizers` runs on a stub module whose parameter names follow the agent's.
Run: pytest tests/test_vision_llm_lora_optimizer.py
"""

import pytest
import torch
from torch import nn

from navsim.agents.vision_llm_lora.vision_llm_lora_agent import VisionLLMLoRAAgent
from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig

LR_BASE = 5e-4


class _StubAgent(nn.Module):
    """One parameter per optimizer group, named as in VisionLLMLoRAAgent."""

    get_optimizers = VisionLLMLoRAAgent.get_optimizers
    split_decay_params = VisionLLMLoRAAgent.split_decay_params

    def __init__(self, config: VisionLLMLoRAConfig):
        super().__init__()
        self._config = config
        self._lr_base = LR_BASE
        self._lr_final = 2.5e-5
        self.traj_classifier = nn.Linear(2, 2, bias=False)
        self.ego_state_tokens = nn.Module()
        self.ego_state_tokens.mlps = nn.Linear(2, 2, bias=False)
        self.ego_state_tokens.norm = nn.RMSNorm(2)
        self.lora_A = nn.Linear(2, 2, bias=False)

    def lrs(self):
        """Initial (peak) LR per parameter, read before the scheduler's warmup factor."""
        optimizer = self.get_optimizers()["optimizer"]
        lr_of = {id(p): group["initial_lr"] for group in optimizer.param_groups for p in group["params"]}
        return {name: lr_of[id(p)] for name, p in self.named_parameters()}


def test_without_phase1_multiplier_the_original_multipliers_apply():
    lrs = _StubAgent(VisionLLMLoRAConfig(training_phase=1)).lrs()
    assert lrs["traj_classifier.weight"] == pytest.approx(0.10 * LR_BASE)
    assert lrs["ego_state_tokens.mlps.weight"] == pytest.approx(0.12 * LR_BASE)
    assert lrs["ego_state_tokens.norm.weight"] == pytest.approx(0.12 * LR_BASE)
    assert lrs["lora_A.weight"] == pytest.approx(0.08 * LR_BASE)


def test_phase1_multiplier_raises_head_and_ego_but_not_the_ego_norm_gain():
    lrs = _StubAgent(VisionLLMLoRAConfig(training_phase=1, phase1_lr_multiplier=1.0)).lrs()
    assert lrs["traj_classifier.weight"] == pytest.approx(LR_BASE)
    assert lrs["ego_state_tokens.mlps.weight"] == pytest.approx(LR_BASE)
    assert lrs["ego_state_tokens.norm.weight"] == pytest.approx(0.12 * LR_BASE)


@pytest.mark.parametrize("phase", [2, 3])
def test_phase1_multiplier_is_ignored_in_later_phases(phase):
    lrs = _StubAgent(VisionLLMLoRAConfig(training_phase=phase, phase1_lr_multiplier=1.0)).lrs()
    assert lrs["traj_classifier.weight"] == pytest.approx(0.10 * LR_BASE)
    assert lrs["ego_state_tokens.mlps.weight"] == pytest.approx(0.12 * LR_BASE)


def test_traj_weight_decay_applies_to_the_head_only():
    agent = _StubAgent(VisionLLMLoRAConfig(traj_weight_decay=0.1))
    decay_of = {id(p): g["weight_decay"] for g in agent.get_optimizers()["optimizer"].param_groups for p in g["params"]}
    assert decay_of[id(agent.traj_classifier.weight)] == pytest.approx(0.1)
    assert decay_of[id(agent.ego_state_tokens.mlps.weight)] == pytest.approx(0.01)


def test_ego_norm_gain_has_no_weight_decay():
    optimizer = _StubAgent(VisionLLMLoRAConfig()).get_optimizers()["optimizer"]
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.dim() == 1:  # the RMSNorm gain is the only 1-D parameter
                assert group["weight_decay"] == 0.0


def test_aux_decoder_gets_its_own_lr():
    agent = _StubAgent(VisionLLMLoRAConfig(training_phase=3, aux_lr_multiplier=0.5))
    agent.aux_decoder = nn.Linear(2, 2, bias=False)
    lrs = agent.lrs()
    assert lrs["aux_decoder.weight"] == pytest.approx(0.5 * LR_BASE)
    assert lrs["traj_classifier.weight"] == pytest.approx(0.10 * LR_BASE)
