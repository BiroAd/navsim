"""
CPU half of the split official evaluation: an agent that replays the outputs of scripts/evaluation/
predict_vision_llm_lora.py. compute_trajectory looks the scene up by token and applies the same select_trajectory as
VisionLLMLoRAAgent, so one GPU prediction run serves every selection rule (argmax, score_heads, shortlist k, ...).
It needs no model, no GPU and no camera images; the NAVSIM scoring scripts run unchanged with it.

requires_scene is only used to learn the scene's token (scene.scene_metadata.initial_token, the current frame's token,
which is what the scoring scripts and the prediction run key scenes by); the scene's annotations are not read.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.vision_llm_lora.vision_llm_lora_agent import TRAJECTORY_SELECTIONS, select_trajectory
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

LOGIT_KEYS = ("trajectory_logits", "pdm_score_logits")


def load_predictions(predictions_dir: Path) -> Dict[str, object]:
    """tokens, row index per token, memory-mapped arrays and the codebook of one prediction folder."""
    meta = json.loads((predictions_dir / "meta.json").read_text())
    tokens: List[str] = json.loads((predictions_dir / "tokens.json").read_text())
    arrays = {p.stem: np.load(p, mmap_mode="r") for p in predictions_dir.glob("*.npy") if p.stem != "codebook"}
    return {
        "meta": meta,
        "row": {token: i for i, token in enumerate(tokens)},
        "arrays": arrays,
        "codebook": torch.from_numpy(np.load(predictions_dir / "codebook.npy")),
    }


def row_predictions(predictions: Dict[str, object], token: str) -> Dict[str, torch.Tensor]:
    """The model outputs of one scene with a batch dimension of 1, in the dtype the model produced."""
    i = predictions["row"][token]
    bf16 = predictions["meta"]["logits_dtype"] == "bfloat16"
    out = {}
    for key, array in predictions["arrays"].items():
        value = torch.from_numpy(np.array(array[i : i + 1]))
        if bf16 and key in LOGIT_KEYS:
            value = value.view(torch.bfloat16)
        out[key] = value
    return out


class VisionLLMLoRAPrecomputedAgent(AbstractAgent):
    """Replays saved vision_llm_lora outputs with a trajectory selection rule."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        predictions_path: str,
        trajectory_selection: str = "argmax",
        selection_shortlist_size: int = 64,
        selection_mixture_lambda: float = 10.0,
    ):
        super().__init__(trajectory_sampling, requires_scene=True)
        if trajectory_selection not in TRAJECTORY_SELECTIONS:
            raise ValueError(f"trajectory_selection must be one of {TRAJECTORY_SELECTIONS}, got {trajectory_selection!r}")
        self._predictions_path = Path(predictions_path)
        self._selection = trajectory_selection
        self._shortlist_size = selection_shortlist_size
        self._mixture_lambda = selection_mixture_lambda
        self._predictions: Optional[Dict[str, object]] = None

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        self._predictions = load_predictions(self._predictions_path)
        if self._selection != "argmax" and "pdm_score_logits" not in self._predictions["arrays"]:
            raise ValueError(f"{self._predictions_path}: no pdm_score_logits, selection {self._selection!r} impossible")

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return []

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return []

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        token = scene.scene_metadata.initial_token
        if token not in self._predictions["row"]:
            raise KeyError(f"{token}: not in {self._predictions_path}")
        trajectory = select_trajectory(
            row_predictions(self._predictions, token),
            self._selection,
            self._predictions["codebook"],
            shortlist_size=self._shortlist_size,
            mixture_lambda=self._mixture_lambda,
        )
        return Trajectory(trajectory.squeeze(0).float().numpy(), self._trajectory_sampling)
