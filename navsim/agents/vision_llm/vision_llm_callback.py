from typing import Any, Dict, Optional, Union

import wandb
import cv2
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import SemanticMapLayer
from PIL import ImageColor

from navsim.agents.vision_llm.vision_llm_config import VisionLLMConfig
from navsim.agents.vision_llm.vision_llm_features import BoundingBox2DIndex
from navsim.visualization.config import AGENT_CONFIG, MAP_LAYER_CONFIG


class VisionLLMCallback(pl.Callback):
    """Visualization Callback for TransFuser during training."""

    def __init__(
        self,
        config: VisionLLMConfig,
        num_plots: int = 3,
        num_rows: int = 2,
        num_columns: int = 2,
    ) -> None:
        """
        Initializes the visualization callback.
        :param config: global config dataclass of TransFuser
        :param num_plots: number of images tiles, defaults to 3
        :param num_rows: number of rows in image tile, defaults to 2
        :param num_columns: number of columns in image tile, defaults to 2
        """

        self._config = config

        self._num_plots = num_plots
        self._num_rows = num_rows
        self._num_columns = num_columns

    def on_validation_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_validation_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        '''device = lightning_module.device
        for idx_plot in range(self._num_plots):
            features, targets = next(iter(trainer.val_dataloaders))
            features, targets = dict_to_device(features, device), dict_to_device(targets, device)
            with torch.no_grad():
                predictions = lightning_module.agent.forward(features)

            features, targets, predictions = (
                dict_to_device(features, "cpu"),
                dict_to_device(targets, "cpu"),
                dict_to_device(predictions, "cpu"),
            )
            grid = self._visualize_model(features, targets, predictions)
            #trainer.logger.experiment.add_image(f"val_plot_{idx_plot}", grid, global_step=trainer.current_epoch)

            trainer.logger.experiment.log({
                f"train_plot_{idx_plot}": wandb.Image(grid.permute(1, 2, 0).numpy()),  # Convert to WandB Image
                "epoch": trainer.current_epoch  # Log the current epoch
            })'''

    def on_test_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_test_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_train_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_train_epoch_end(
        self, trainer: pl.Trainer, lightning_module: pl.LightningModule, unused: Optional[Any] = None
    ) -> None:
        """Inherited, see superclass."""

        '''device = lightning_module.device
        for idx_plot in range(self._num_plots):
            features, targets = next(iter(trainer.train_dataloader))
            features, targets = dict_to_device(features, device), dict_to_device(targets, device)
            with torch.no_grad():
                predictions = lightning_module.agent.forward(features)

            features, targets, predictions = (
                dict_to_device(features, "cpu"),
                dict_to_device(targets, "cpu"),
                dict_to_device(predictions, "cpu"),
            )
            grid = self._visualize_model(features, targets, predictions)
            #trainer.logger.experiment.add_image(f"train_plot_{idx_plot}", grid, global_step=trainer.current_epoch)
            trainer.logger.experiment.log({
                f"train_plot_{idx_plot}": wandb.Image(grid.permute(1, 2, 0).numpy()),  # Convert to WandB Image
                "epoch": trainer.current_epoch  # Log the current epoch
            })'''

    def _visualize_model(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Create tile of input-output visualizations for TransFuser.
        :param features: dictionary of feature names and tensors
        :param targets: dictionary of target names and tensors
        :param predictions: dictionary of target names and predicted tensors
        :return: image tiles as RGB tensors
        """
        camera = features["camera_feature"].permute(0, 2, 3, 1).numpy()
        bev = targets["bev_semantic_map"].numpy()

        pred_bev = predictions["bev_semantic_map"].argmax(1).numpy()

        plots = []
        for sample_idx in range(self._num_rows * self._num_columns):
            plot = np.zeros((256, 512, 3), dtype=np.uint8)
            plot[:128, :512] = (camera[sample_idx] * 255).astype(np.uint8)[::2, ::2]

            plot[128:, :256] = semantic_map_to_rgb(bev[sample_idx], self._config)
            plot[128:, 256:512] = semantic_map_to_rgb(pred_bev[sample_idx], self._config)

            plots.append(torch.tensor(plot).permute(2, 0, 1))

        return vutils.make_grid(plots, normalize=False, nrow=self._num_rows)


def dict_to_device(dict: Dict[str, torch.Tensor], device: Union[torch.device, str]) -> Dict[str, torch.Tensor]:
    """
    Helper function to move tensors from dictionary to device.
    :param dict: dictionary of names and tensors
    :param device: torch device to move tensors to
    :return: dictionary with tensors on specified device
    """
    for key in dict.keys():
        dict[key] = dict[key].to(device)
    return dict


def semantic_map_to_rgb(semantic_map: npt.NDArray[np.int64], config: VisionLLMConfig) -> npt.NDArray[np.uint8]:
    """
    Convert semantic map to RGB image.
    :param semantic_map: numpy array of segmentation map (multi-channel)
    :param config: global config dataclass of TransFuser
    :return: RGB image as numpy array
    """

    height, width = semantic_map.shape[:2]
    rgb_map = np.ones((height, width, 3), dtype=np.uint8) * 255

    for label in range(1, config.num_bev_classes):

        if config.bev_semantic_classes[label][0] == "linestring":
            hex_color = MAP_LAYER_CONFIG[SemanticMapLayer.BASELINE_PATHS]["line_color"]
        else:
            layer = config.bev_semantic_classes[label][-1][0]  # take color of first element
            hex_color = (
                AGENT_CONFIG[layer]["fill_color"]
                if layer in AGENT_CONFIG.keys()
                else MAP_LAYER_CONFIG[layer]["fill_color"]
            )

        rgb_map[semantic_map == label] = ImageColor.getcolor(hex_color, "RGB")
    return rgb_map[::-1, ::-1]