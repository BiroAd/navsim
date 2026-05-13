from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig


def _build_codebook(config: VisionLLMLoRAConfig) -> torch.Tensor:
    if config.trajectory_codebook_path:
        codebook_np = np.load(config.trajectory_codebook_path)

        if codebook_np.ndim == 3:
            if codebook_np.shape[1] == 1 and codebook_np.shape[2] == 3:
                codebook_np = codebook_np[:, 0, :]
            else:
                raise ValueError(
                    "Loaded codebook has shape [K, T, 3], but current tokenizer expects [K, 3]. "
                    "Use --level point in offline KMeans export or extend model for trajectory-level tokens."
                )

        if codebook_np.ndim != 2 or codebook_np.shape[1] != 3:
            raise ValueError(
                f"Offline codebook must have shape [K, 3]. Got {codebook_np.shape}."
            )

        if config.trajectory_token_k != int(codebook_np.shape[0]):
            raise ValueError(
                "trajectory_token_k mismatch with offline codebook. "
                f"k={config.trajectory_token_k}, codebook={codebook_np.shape[0]}"
            )

        return torch.from_numpy(codebook_np.astype(np.float32))

    if config.trajectory_token_grid_x * config.trajectory_token_grid_y * config.trajectory_token_grid_heading != config.trajectory_token_k:
        raise ValueError(
            "trajectory_token_k must match grid_x * grid_y * grid_heading. "
            f"Got k={config.trajectory_token_k}, "
            f"grid={config.trajectory_token_grid_x}x{config.trajectory_token_grid_y}x{config.trajectory_token_grid_heading}."
        )

    grid_x = torch.linspace(config.trajectory_token_x_min, config.trajectory_token_x_max, config.trajectory_token_grid_x)
    grid_y = torch.linspace(config.trajectory_token_y_min, config.trajectory_token_y_max, config.trajectory_token_grid_y)
    grid_h = torch.linspace(
        config.trajectory_token_heading_min,
        config.trajectory_token_heading_max,
        config.trajectory_token_grid_heading,
    )

    mesh = torch.meshgrid(grid_x, grid_y, grid_h, indexing="ij")
    codebook = torch.stack(mesh, dim=-1).reshape(-1, 3)
    return codebook


class CoordinateKNNTokenizer(nn.Module):
    """Coordinate tokenizer using nearest-neighbor search over a fixed codebook."""

    def __init__(self, config: VisionLLMLoRAConfig):
        super().__init__()
        self._config = config
        self.register_buffer("codebook", _build_codebook(config), persistent=False)

    @property
    def num_tokens(self) -> int:
        return self.codebook.shape[0]

    def encode(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous coordinates to token ids with 1-NN lookup.
        :param coordinates: [B, T, 3]
        :return: token ids [B, T]
        """
        flat_coords = coordinates.reshape(-1, 3)
        distances = torch.cdist(flat_coords.float(), self.codebook.float())
        token_ids = torch.argmin(distances, dim=-1)
        return token_ids.view(coordinates.shape[0], coordinates.shape[1])

    def decode_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token ids to continuous coordinates via codebook lookup.
        :param token_ids: [B, T]
        :return: coordinates [B, T, 3]
        """
        return self.codebook[token_ids]

    def decode_logits(
        self,
        token_logits: torch.Tensor,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Soft decode logits into continuous coordinates by expectation over codebook.
        :param token_logits: [B, T, K]
        :return: coordinates [B, T, 3]
        """
        temp = self._config.trajectory_token_temperature if temperature is None else temperature
        temp = max(temp, 1e-6)

        logits = token_logits / temp
        codebook = self.codebook.to(token_logits.device, dtype=token_logits.dtype)

        if topk is not None and 0 < topk < logits.shape[-1]:
            top_vals, top_ids = torch.topk(logits, k=topk, dim=-1)
            probs = torch.softmax(top_vals, dim=-1)
            selected_coords = codebook[top_ids]
            return (probs.unsqueeze(-1) * selected_coords).sum(dim=-2)

        probs = torch.softmax(logits, dim=-1)
        return probs @ codebook
