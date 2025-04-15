from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.vision_llm.vision_llm_config import VisionLLMConfig
from navsim.common.enums import StateSE2Index
from navsim.agents.vision_llm.vision_llm_features import BoundingBox2DIndex

class VisionLLMModel(nn.Module):
    def __init__(self, trajectory_sampling: TrajectorySampling, config: VisionLLMConfig):
        super().__init__()

        # Load the model and tokenizer
        self.model = AutoModel.from_pretrained(
            config.architecture,
            torch_dtype=torch.float32,
            #device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(config.architecture, trust_remote_code=True)

        #print(f"Model: {self.model}")
        #print(f"Tokenizer: {self.tokenizer}")
        self.visual = self.model.vision_model
        self.language = self.model.language_model.model
        self.mlp = self.model.mlp1
        # a tokeneket beadjuk a pixel_values értékekkel együtt.

        hidden_dim = self.language.norm.weight.shape[0]

        self.agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=hidden_dim,
            d_model=config.tf_d_model,
        ).to(self.model.device)

        self.trajectory_head = TrajectoryHead(
            num_poses=trajectory_sampling.num_poses,
            d_ffn=config.tf_d_model,
            d_model=hidden_dim
        ).to(self.model.device)


        self.bev_semantic_head = BEVSemanticHead(
            d_ffn=hidden_dim,
            num_classes=config.num_bev_classes,
            height=config.bev_pixel_height // config.bev_down_sample_factor,
            width=config.bev_pixel_width // config.bev_down_sample_factor,
        ).to(self.model.device)

        self.model.gradient_checkpointing_enable()

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        '''vision_outputs = None
        if pixel_values is not None:
            vision_outputs = self.visual(pixel_values)'''
        
        prompt = features["prompt"]
        tokenized_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = tokenized_inputs.input_ids.to(self.model.device)
        attention_mask = tokenized_inputs.attention_mask.to(self.model.device)

        outputs = self.language.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
        )


        hidden_states = outputs.last_hidden_state
        hidden_states = hidden_states[:, -1, :]
        #print(f"Hidden state: {hidden_states.shape}")
        #last_hidden_state = hidden_states[:, -1, :]

        #bev_semantic_map = self.bev_semantic_head(last_hidden_state)

        #output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        output: Dict[str, torch.Tensor] = {}
        trajectory_outputs = self.trajectory_head(hidden_states)
        #print(f"Trajectory outputs: {trajectory_outputs['trajectory'].shape}")
        output.update(trajectory_outputs)

        #agent_outputs = self.agent_head(hidden_states)
        #output.update(agent_outputs)

        return output
    
class BEVSemanticHead(nn.Module):
    def __init__(self, d_ffn, num_classes=7, height=128, width=256):
        super().__init__()
        
        self.num_classes = num_classes
        self.height = height
        self.width = width
        
        # Feature transformation network
        self.feature_transform = nn.Sequential(
            nn.Linear(d_ffn, d_ffn),
            nn.LayerNorm(d_ffn),
            nn.GELU()
        )
        
        # Project to initial BEV features
        self.projection = nn.Linear(d_ffn, num_classes * height * width)
        
        # Refinement convolutions
        self.refine = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
    def forward(self, hidden_states):
        # Transform features
        features = self.feature_transform(hidden_states)
        
        # Project to BEV dimensions
        batch_size = hidden_states.size(0)
        projected = self.projection(features)
        
        # Reshape to [batch_size, num_classes, height, width]
        bev_features = projected.view(batch_size, self.num_classes, self.height, self.width)
        
        # Apply refinement
        bev_semantic_map = self.refine(bev_features)
        
        return bev_semantic_map

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}


class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, num_poses * StateSE2Index.size()),
        )

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        return {"trajectory": poses}