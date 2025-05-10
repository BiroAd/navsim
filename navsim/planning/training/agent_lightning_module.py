from typing import Dict, Tuple

import pytorch_lightning as pl
from torch import Tensor
import torch
from torch.amp import autocast
import math

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.training_step_outputs = []

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        #self.trainer.loggers[0].log_metrics({"step": self.global_step})
        #print(self.trainer.optimizers[0].param_groups[0]['lr'])


        features, targets = batch
        teacher_forcing_seq = None
        teacher_forcing_ratio = 1.0
        if logging_prefix == "train":
            '''progress = self.current_epoch / self.trainer.max_epochs
            #cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            cosine_decay = 1
            linear_decay = 1.0 - progress
            teacher_forcing_ratio = cosine_decay * linear_decay'''
            start = 0.7
            end = 0.1
            progress = min(self.current_epoch / (self.trainer.max_epochs - 1), 1.0)
            teacher_forcing_ratio = start + progress * (end - start)
            teacher_forcing_seq = targets["trajectory"].clone()

        prediction = self.agent.forward(features, teacher_forcing_seq, teacher_forcing_ratio=teacher_forcing_ratio)
        #loss, separate_losses = self.agent.compute_loss(features, targets, prediction)
        loss = self.agent.compute_loss(features, targets, prediction)
        if logging_prefix == "train":
            self.training_step_outputs.append(loss)
            self.log(f"{logging_prefix}/loss_mean", torch.stack(self.training_step_outputs).mean(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/tfr", teacher_forcing_ratio, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        if len(self.training_step_outputs) == 1600:
            self.training_step_outputs = []
        #pos_loss = separate_losses["pos_loss"]
        #theta_loss = separate_losses["theta_loss"]
        #self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log(f"{logging_prefix}/pos_loss", pos_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log(f"{logging_prefix}/theta_loss", theta_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log(f"{logging_prefix}/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr_step", lr, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
