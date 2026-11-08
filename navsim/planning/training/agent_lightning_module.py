from typing import Dict, List, Tuple

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
        self.training_query_variance = []
        self.training_relative_variance = []
        self.training_extra: Dict[str, List[Tensor]] = {}  # extra loss metrics, same window as the lists above
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
            # max(.., 1): a 1-epoch run (RL-3 smoke test 14928925) divided by zero here; unchanged for >= 2 epochs
            progress = min(self.current_epoch / max(self.trainer.max_epochs - 1, 1), 1.0)
            teacher_forcing_ratio = start + progress * (end - start)
            teacher_forcing_seq = targets["trajectory"].clone()

        prediction = self.agent.forward(features, teacher_forcing_seq, teacher_forcing_ratio=teacher_forcing_ratio)
        #loss, separate_losses = self.agent.compute_loss(features, targets, prediction)
        # An optional 8th value is a dict of extra scalar metrics (e.g. separate loss terms), logged below.
        loss, entropy, similarity, ade_loss, fde_loss, query_variance, relative_variance, *extra = self.agent.compute_loss(features, targets, prediction)
        if logging_prefix == "train":
            self.training_step_outputs.append(loss.detach().cpu())
            self.training_entropy.append(entropy.detach().cpu())
            #self.training_attn_weights.append(attn_weights.detach().cpu())
            self.training_similarity.append(similarity.detach().cpu())
            self.training_ade.append(ade_loss.detach().cpu())
            self.training_fde.append(fde_loss.detach().cpu())
            self.training_query_variance.append(query_variance.detach().cpu())
            self.training_relative_variance.append(relative_variance.detach().cpu())
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

            self.log(
                f"stabilize/{logging_prefix}/query_variance_mean",
                torch.stack(self.training_query_variance).mean(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=False,
            )

            self.log(
                f"stabilize/{logging_prefix}/relative_variance_mean",
                torch.stack(self.training_relative_variance).mean(),
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
        self.log(f"stabilize/{logging_prefix}/query_variance", query_variance, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"stabilize/{logging_prefix}/relative_variance", relative_variance, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        for name, value in (extra[0] if extra else {}).items():
            self.log(f"{logging_prefix}/{name}", value, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            if logging_prefix == "train":  # windowed mean over the logging interval, as the *_mean metrics above
                self.training_extra.setdefault(name, []).append(value.detach().float().cpu())
                self.log(
                    f"train/{name}_mean",
                    torch.stack(self.training_extra[name]).mean(),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    sync_dist=False,
                )
        #self.log(f"stabilize/{logging_prefix}/attn_weights", attn_weights, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)  
        #pos_loss = separate_losses["pos_loss"]
        #theta_loss = separate_losses["theta_loss"]
        #self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log(f"{logging_prefix}/pos_loss", pos_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log(f"{logging_prefix}/theta_loss", theta_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log(f"{logging_prefix}/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=True, prog_bar=True)


        # One running-mean window = one logging interval: log_every_n_steps optimizer steps of accumulate_grad_batches
        # batches each.
        if len(self.training_step_outputs) >= self.trainer.log_every_n_steps * self.trainer.accumulate_grad_batches:
            self.training_step_outputs = []
            self.training_entropy = []
            #self.training_attn_weights = []
            self.training_similarity = []
            self.training_ade = []
            self.training_fde = []
            self.training_query_variance = []
            self.training_relative_variance = []
            self.training_extra = {}

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
        Computes total L2 gradient norm of a module safely in mixed precision.
        """
        total_norm = 0.0
        for p in module.parameters():
            if p.requires_grad and p.grad is not None:
                # Cast to float32 FIRST to prevent internal FP16 overflow
                param_norm = p.grad.detach().float().norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm ** 0.5


    def _grad_norm_by_name(self, model, keyword):
        """
        Computes gradient norm for parameters whose name contains `keyword`.
        """
        total_norm = 0.0
        for name, p in model.named_parameters():
            if keyword in name and p.requires_grad and p.grad is not None:
                # Cast to float32 FIRST to prevent internal FP16 overflow
                param_norm = p.grad.detach().float().norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm ** 0.5


    def _param_norm(self, module):
        """
        Computes parameter L2 norm.
        """
        total_norm = 0.0
        for p in module.parameters():
            if p.requires_grad:
                param_norm = p.detach().float().norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm ** 0.5

    def _frozen_parameter_names(self) -> List[str]:
        # remove_duplicate=False: a frozen tensor registered under two names (vision_llm_lora's `embedding` alias of the
        # LLM input embedding) must be left out under both.
        return [name for name, p in self.named_parameters(remove_duplicate=False) if not p.requires_grad]

    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        """
        Leave frozen parameters out of the checkpoint: they come back from the pretrained weights whenever the agent is
        built (for vision_llm_lora the Qwen backbone, ~96% of a 9 GB checkpoint). Agents that train every parameter
        are unaffected. agent.initialize() and the phase handoff read only trained keys, so they work unchanged.
        """
        frozen = set(self._frozen_parameter_names())
        if frozen:
            checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if k not in frozen}
            checkpoint["frozen_parameters_omitted"] = True

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        """
        Refill parameters left out by on_save_checkpoint from the freshly built model, so the strict load of a native
        resume (+ckpt_path=) works. Lightning calls this hook before loading the state dict
        (pytorch_lightning 2.2.1, trainer/connectors/checkpoint_connector.py, restore_model).
        """
        if checkpoint.get("frozen_parameters_omitted"):
            state = self.state_dict()
            for name in self._frozen_parameter_names():
                checkpoint["state_dict"].setdefault(name, state[name])

    def on_before_optimizer_step(self, optimizer):
        """
        Logs true, unscaled gradient norms before the optimizer updates weights.
        """
        if hasattr(self.agent, "model"):
            model = self.agent.model
        elif hasattr(self.agent, "_vlm_model"):
            model = self.agent._vlm_model
        else:
            return  

        # --------------------------------------------------
        # Main trainable modules & LoRA gradients
        # --------------------------------------------------
        # The ego_mlp metrics follow the module that encodes velocity and acceleration: ego_fusion_mlp (late fusion)
        # if present, else ego_state_tokens (late_ego_fusion=False). When both exist, ego_state_tokens is logged apart.
        ego_fusion_mlp = getattr(model, "ego_fusion_mlp", None)
        ego_state_tokens = getattr(model, "ego_state_tokens", None)
        ego_module = ego_fusion_mlp if ego_fusion_mlp is not None else ego_state_tokens

        traj_head_grad = self._grad_norm(model.traj_classifier)
        ego_mlp_grad = self._grad_norm(ego_module) if ego_module is not None else math.nan
        language_lora_grad = self._grad_norm_by_name(model, "language_model")
        vision_lora_grad = self._grad_norm_by_name(model, "visual")

        # --------------------------------------------------
        # SAFETY FILTERS: Prevent AMP Nan/Inf from poisoning the mean
        # --------------------------------------------------
        if not math.isnan(traj_head_grad) and not math.isinf(traj_head_grad):
            self.training_grad_traj.append(traj_head_grad)
            
        if not math.isnan(ego_mlp_grad) and not math.isinf(ego_mlp_grad):
            self.training_grad_ego.append(ego_mlp_grad)
            
        if not math.isnan(language_lora_grad) and not math.isinf(language_lora_grad):
            self.training_grad_lang.append(language_lora_grad)
            
        if not math.isnan(vision_lora_grad) and not math.isinf(vision_lora_grad):
            self.training_grad_vision.append(vision_lora_grad)

        # --------------------------------------------------
        # Parameter norms & Ratios
        # --------------------------------------------------
        traj_head_param = self._param_norm(model.traj_classifier)
        ego_mlp_param = self._param_norm(ego_module) if ego_module is not None else 0.0

        # Protect the ratio calculation
        traj_ratio = 0.0
        if not math.isnan(traj_head_grad) and not math.isinf(traj_head_grad):
            traj_ratio = traj_head_grad / (traj_head_param + 1e-8)
            
        ego_ratio = 0.0
        if not math.isnan(ego_mlp_grad) and not math.isinf(ego_mlp_grad):
            ego_ratio = ego_mlp_grad / (ego_mlp_param + 1e-8)

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        self.log("grad/traj_head", torch.tensor(self.training_grad_traj, dtype=torch.float32).mean() if self.training_grad_traj else 0.0, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        self.log("grad/ego_mlp", torch.tensor(self.training_grad_ego, dtype=torch.float32).mean() if self.training_grad_ego else 0.0, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        self.log("grad/lang_lora", torch.tensor(self.training_grad_lang, dtype=torch.float32).mean() if self.training_grad_lang else 0.0, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        self.log("grad/vision_lora", torch.tensor(self.training_grad_vision, dtype=torch.float32).mean() if self.training_grad_vision else 0.0, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)

        self.log("grad_ratio/traj_head", traj_ratio, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        self.log("grad_ratio/ego_mlp", ego_ratio, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        if ego_state_tokens is not None and ego_module is not ego_state_tokens:
            self.log("grad/ego_state_tokens", self._grad_norm(ego_state_tokens), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        if ego_state_tokens is not None:
            # Mean |gain| of the ego-token RMSNorm = the scale the ego tokens enter the LLM at (starts at the token
            # embedding RMS, ~0.013 for Qwen3.5-4B).
            ego_token_gain = ego_state_tokens.norm.weight.detach().abs().mean()
            self.log("param/ego_token_gain", ego_token_gain, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)

        # Clear lists to prevent memory leaks over long epochs
        if len(self.training_grad_traj) >= self.trainer.log_every_n_steps:  # appended once per optimizer step
            self.training_grad_vision = []
            self.training_grad_lang = []
            self.training_grad_traj = []
            self.training_grad_ego = []