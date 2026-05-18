from typing import Dict, Tuple

import pytorch_lightning as pl
from torch import Tensor
import torch
from torch.amp import autocast
import math

from navsim.agents.abstract_agent import AbstractAgent
import torchmetrics


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
        self.training_grad_lang = []
        self.training_grad_vision = []
        self.training_grad_traj = []
        self.training_grad_ego = []
        self.training_entropy = []
        #self.training_attn_weights = []
        self.training_similarity = []
        self.training_ade = []
        self.training_fde = []
        #self.train_loss_mean = torchmetrics.MeanMetric()

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
        loss, entropy, similarity, ade_loss, fde_loss = self.agent.compute_loss(features, targets, prediction)
        if logging_prefix == "train":
            self.training_step_outputs.append(loss.detach().cpu())
            self.training_entropy.append(entropy.detach().cpu())
            #self.training_attn_weights.append(attn_weights.detach().cpu())
            self.training_similarity.append(similarity.detach().cpu())
            self.training_ade.append(ade_loss.detach().cpu())
            self.training_fde.append(fde_loss.detach().cpu())
            #self.train_loss_mean(loss)
            self.log(f"{logging_prefix}/loss_mean", torch.stack(self.training_step_outputs).mean(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
            self.log(
                f"stabilize/{logging_prefix}/entropy_mean",
                torch.stack(self.training_entropy).mean(),
                on_step=True,
                on_epoch=False,            
                prog_bar=False,
                sync_dist=False,
            )
        
            #self.log(
            #    f"stabilize/{logging_prefix}/attn_weights_mean",
            #    torch.stack(self.training_attn_weights).mean(),
            #    on_step=True,
            #    on_epoch=False,            
            #    prog_bar=False,
            #    sync_dist=False,
            #)

            self.log(
                f"stabilize/{logging_prefix}/similarity_mean",
                torch.stack(self.training_similarity).mean(),
                on_step=True,
                on_epoch=False,            
                prog_bar=False,
                sync_dist=False,
            )
            
            self.log(
                f"stabilize/{logging_prefix}/ade_mean",
                torch.stack(self.training_ade).mean(),
                on_step=True,
                on_epoch=False,            
                prog_bar=False,
                sync_dist=False,
            )

            self.log(
                f"stabilize/{logging_prefix}/fde_mean",
                torch.stack(self.training_fde).mean(),
                on_step=True,
                on_epoch=False,            
                prog_bar=False,
                sync_dist=False,
            )
            #self.log(f"{logging_prefix}/loss_mean", self.train_loss_mean.compute(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            #self.train_loss_mean.reset()
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/tfr", teacher_forcing_ratio, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"stabilize/{logging_prefix}/ade_loss", ade_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"stabilize/{logging_prefix}/fde_loss", fde_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"stabilize/{logging_prefix}/entropy", entropy, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"stabilize/{logging_prefix}/similarity", similarity, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        #self.log(f"stabilize/{logging_prefix}/attn_weights", attn_weights, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)  
        #pos_loss = separate_losses["pos_loss"]
        #theta_loss = separate_losses["theta_loss"]
        #self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log(f"{logging_prefix}/pos_loss", pos_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log(f"{logging_prefix}/theta_loss", theta_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log(f"{logging_prefix}/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=True, prog_bar=True)


        if len(self.training_step_outputs) == 800:
            self.training_step_outputs = []
            self.training_entropy = []
            #self.training_attn_weights = []
            self.training_similarity = []
            self.training_ade = []
            self.training_fde = []

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


    def _grad_norm(self, module):
        """
        Computes total L2 gradient norm of a module.
        Only includes trainable parameters with gradients.
        """
        total_norm = 0.0

        for p in module.parameters():
            if p.requires_grad and p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm ** 0.5


    def _grad_norm_by_name(self, model, keyword):
        """
        Computes gradient norm for parameters whose name contains `keyword`.
        Useful for LoRA monitoring.
        """
        total_norm = 0.0

        for name, p in model.named_parameters():
            if keyword in name and p.requires_grad and p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm ** 0.5


    def _param_norm(self, module):
        """
        Computes parameter L2 norm.
        Useful for grad/weight ratio diagnostics.
        """
        total_norm = 0.0

        for p in module.parameters():
            if p.requires_grad:
                param_norm = p.detach().data.norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm ** 0.5

    def on_after_backward(self):
        """
        Logs gradient norms after backward pass.
        # Extremely useful for debugging optimization imbalance.
        # """

        if hasattr(self.agent, "model"):
            model = self.agent.model
        elif hasattr(self.agent, "_vlm_model"):
            model = self.agent._vlm_model
        else:
            return  # Skip gradient logging if model attribute is not found

        # --------------------------------------------------
        # Main trainable modules
        # --------------------------------------------------

        traj_head_grad = self._grad_norm(model.traj_classifier)
        self.training_grad_traj.append(traj_head_grad)
        ego_mlp_grad = self._grad_norm(model.ego_fusion_mlp)
        self.training_grad_ego.append(ego_mlp_grad)

        # --------------------------------------------------
        # LoRA gradient norms
        # --------------------------------------------------

        language_lora_grad = self._grad_norm_by_name(
            model,
            "language_model"
        )
        self.training_grad_lang.append(language_lora_grad)

        vision_lora_grad = self._grad_norm_by_name(
            model,
            "visual"
        )
        self.training_grad_vision.append(vision_lora_grad)

        # --------------------------------------------------
        # Parameter norms
        # --------------------------------------------------

        traj_head_param = self._param_norm(model.traj_classifier)
        ego_mlp_param = self._param_norm(model.ego_fusion_mlp)

        # --------------------------------------------------
        # Gradient / weight ratios
        # --------------------------------------------------

        traj_ratio = traj_head_grad / (traj_head_param + 1e-8)
        ego_ratio = ego_mlp_grad / (ego_mlp_param + 1e-8)

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        self.log(
            "grad/traj_head",
            torch.tensor(self.training_grad_traj, dtype=torch.float32).mean() if self.training_grad_traj else 0.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
        )

        self.log(
            "grad/ego_mlp",
            torch.tensor(self.training_grad_ego, dtype=torch.float32).mean() if self.training_grad_ego else 0.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
        )

        self.log(
            "grad/lang_lora",
            torch.tensor(self.training_grad_lang, dtype=torch.float32).mean() if self.training_grad_lang else 0.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
        )

        self.log(
            "grad/vision_lora",
            torch.tensor(self.training_grad_vision, dtype=torch.float32).mean() if self.training_grad_vision else 0.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
        )

        self.log(
            "grad_ratio/traj_head",
            traj_ratio,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
        )

        self.log(
            "grad_ratio/ego_mlp",
            ego_ratio,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
        )

        if len(self.training_grad_vision) == 800:
            self.training_grad_vision = []
            self.training_grad_lang = []
            self.training_grad_traj = []
            self.training_grad_ego = []