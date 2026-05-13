import warnings
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.vision_llm.vision_llm_config import VisionLLMConfig
from navsim.common.enums import StateSE2Index
from navsim.agents.vision_llm.vision_llm_features import BoundingBox2DIndex
from navsim.agents.vision_llm.conversation import get_conv_template

import time
from PIL import Image
from torch.amp import autocast

class VisionLLMModel(nn.Module):
    def __init__(self, trajectory_sampling: TrajectorySampling, config: VisionLLMConfig):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
        # Load the model and tokenizer
        self.model = AutoModel.from_pretrained(
            config.architecture,
            torch_dtype=torch.float32,
            device_map={"": self.device},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(config.architecture, trust_remote_code=True)

        self.ego_tokens_start = "<ego>"
        self.ego_tokens_end = "</ego>"
        self.ego_context_tokens = "<EGO_CONTEXT>"
        self.image_context_tokens = "<IMG_CONTEXT>"
        self.image_tokens_start = "<img>"
        self.image_tokens_end = "</img>"

        special_tokens_dict = {'additional_special_tokens': [self.ego_tokens_start, self.ego_tokens_end, self.ego_context_tokens, self.image_context_tokens, self.image_tokens_start, self.image_tokens_end]}
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)

        old_embeddings = self.model.get_input_embeddings()
        new_embeddings = torch.nn.Embedding(len(self.tokenizer), old_embeddings.embedding_dim)
        
        new_embeddings.weight.data[:old_embeddings.weight.size(0)] = old_embeddings.weight.data
        self.model.set_input_embeddings(new_embeddings)

        self.template = get_conv_template(self.model.template)

        '''self.template.system_message = (
            "You are an AI assistant for autonomous driving, specialized in trajectory planning. "
            "You receive six images covering a 360° view around the ego vehicle, arranged as: "
            "Top row: FRONT LEFT, FRONT, FRONT RIGHT; Bottom row: BACK LEFT, BACK, BACK RIGHT. "
            "You also receive the vehicle’s status: velocity, acceleration, and current driving command. "
            "Analyze the environment and dynamics to generate a safe, feasible trajectory. "
            "Adjust for obstacles, occlusions, or risks to ensure safety and efficiency."
        )'''

        '''self.template.system_message = (
            "You are an AI assistant specialized in trajectory planning for autonomous driving. "
            "You receive six camera images covering a full 360° view around the ego vehicle, organized as follows: "
            "Top row (left to right): FRONT LEFT, FRONT, FRONT RIGHT; "
            "Bottom row (left to right): BACK LEFT, BACK, BACK RIGHT. "
            "You also receive the ego vehicle’s status data, including velocity, acceleration, and the current driving command. "
            "Your task is to analyze the surroundings and vehicle dynamics to generate a set of safe, feasible trajectory waypoints for the ego vehicle. "
            "Adjust the trajectory as needed to avoid obstacles, handle occlusions, and maintain safe and efficient driving behavior."
        )'''

        self.template.system_message = (
            "You are an Autonomous Driving AI assistant. You receive an image that consists "
            "of six surrounding camera views. The layout is as follows: The first row contains three images: "
            "FRONT LEFT, FRONT, FRONT RIGHT. The second row contains three images: BACK LEFT, "
            "BACK, BACK RIGHT. Your task is to analyze these images and provide insights or actions based on "
            "the visual data."
        )

        '''self.template.system_message = (
            "You are an AI assistant for autonomous driving, focused on trajectory planning. "
            "You receive six 360° images around the vehicle: "
            "FRONT LEFT, FRONT, FRONT RIGHT (top row) and BACK LEFT, BACK, BACK RIGHT (bottom row), "
            "plus velocity, acceleration, and driving command. "
            "Generate a safe, feasible trajectory, accounting for obstacles, occlusions, and risks."
        )'''

        question = ("current statuses: <EGO>, current images: <image>.\nWhat should be the next trajectory waypoints for the ego vehicle?")

        self.template.append_message(self.template.roles[0], question)
        self.template.append_message(self.template.roles[1], None)
        print(f"Prompt template: {self.template}")
        

        self.visual = self.model.vision_model
        self.language = self.model.language_model.model
        self.mlp = self.model.mlp1

        self.hidden_dim = self.language.norm.weight.shape[0]

        self.token_type_embed = nn.Embedding(3, self.hidden_dim).to(self.model.device)

        '''self.encode_ego = EgoContinuousEncoder(input_dim=4, hidden_dim=self.hidden_dim).to(self.model.device)
        self.encode_direction = DirectionEmbedding(num_directions=4, hidden_dim=self.hidden_dim).to(self.model.device)

        self.ego_fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        ).to(self.model.device)'''

        self.velocity_encoder = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        ).to(self.model.device)
        
        self.acceleration_encoder = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        ).to(self.model.device)

        self.direction_embedding = nn.Embedding(4, self.hidden_dim).to(self.model.device)

        '''self.agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=self.hidden_dim,
            d_model=config.tf_d_model,
        ).to(self.model.device)'''

        '''self.trajectory_head = TrajectoryHead(
            num_poses=trajectory_sampling.num_poses,
            d_ffn=config.tf_d_model,
            d_model=hidden_dim
        ).to(self.model.device)'''

        self.trajectory_head = WaypointTransformerHead(
            vlm_dim=self.hidden_dim,
            traj_dim=3,
            num_waypoints=trajectory_sampling.num_poses,
            model_dim=self.hidden_dim,
            num_layers=4,
            nhead=4
        ).to(self.model.device)


        self.bev_semantic_head = BEVSemanticHead(
            d_ffn=self.hidden_dim,
            num_classes=config.num_bev_classes,
            height=config.bev_pixel_height // config.bev_down_sample_factor,
            width=config.bev_pixel_width // config.bev_down_sample_factor,
        ).to(self.model.device)

        self.N_EGO_TOKENS = 32
        self.image_tokens = 208

        self.ego_fusion_mlp = nn.Sequential(
            nn.Linear(3*self.hidden_dim, self.N_EGO_TOKENS *self.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(self.N_EGO_TOKENS * self.hidden_dim // 4, self.N_EGO_TOKENS * self.hidden_dim)
        ).to(self.model.device)


        self.visual.gradient_checkpointing_enable()
        self.language.gradient_checkpointing_enable()
        self.embedding = self.language.embed_tokens.to(self.model.device)

        '''self.raw_prompt_template = self.template.get_prompt()
        ego_tokens = "<ego>" + ("<EGO_CONTEXT>" * self.N_EGO_TOKENS) + "</ego>"
        img_tokens = "<img>" + ("<IMG_CONTEXT>" * self.image_tokens) + "</img>"
        self.raw_prompt_template = self.raw_prompt_template.replace("<EGO>", ego_tokens)
        query = self.raw_prompt_template.replace("<image>", img_tokens)

        tok = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True
        )

        self.register_buffer("input_ids", tok["input_ids"])
        self.register_buffer("attention_mask", tok["attention_mask"])

        ego_id = self.tokenizer.convert_tokens_to_ids("<EGO_CONTEXT>")
        img_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

        self.register_buffer("ego_positions", (self.input_ids == ego_id))
        self.register_buffer("img_positions", (self.input_ids == img_id))'''



    def forward(self, features: Dict[str, torch.Tensor], teacher_forcing_seq=None, teacher_forcing_ratio=1.0) -> Dict[str, torch.Tensor]:
            images = features["camera_feature"]

            #print out one image
            b, p, c, h, w = images.shape

            resized_images = images.view(b*p, c, h, w)

            feature = self.extract_feature(resized_images)
            feature = feature.view(b, -1, feature.shape[-1])

            velocity = features["velocity"]
            direction = features["driving_command"]
            acceleration = features["acceleration"]

            enc_velocity = self.velocity_encoder(velocity).unsqueeze(dim=1)
            enc_acceleration = self.acceleration_encoder(acceleration).unsqueeze(dim=1)
            enc_direction = self.direction_embedding(direction).unsqueeze(dim=1)

            ego_embeds = torch.cat([enc_velocity, enc_acceleration, enc_direction], dim=1)

            ego_embeds_flat = ego_embeds.view(b, -1)

            ego_embeds = self.ego_fusion_mlp(ego_embeds_flat)

            ego_embeds = ego_embeds.view(b, self.N_EGO_TOKENS, self.hidden_dim)
            
            batch_size = ego_embeds.shape[0]

            query = self.template.get_prompt()

            ego_tokens = self.ego_tokens_start + self.ego_context_tokens * ego_embeds.shape[1] + self.ego_tokens_end
            image_tokens = self.image_tokens_start + self.image_context_tokens * feature.shape[1] + self.image_tokens_end

            query = query.replace('<image>', image_tokens)
            query = query.replace('<EGO>', ego_tokens)

            tokenized_inputs = self.tokenizer(query, return_tensors="pt", padding=True).to(self.model.device)

            input_ids = tokenized_inputs['input_ids'].expand(batch_size, -1)

            attention_mask = tokenized_inputs['attention_mask'].expand(batch_size, -1)
            #input_ids = self.input_ids.to(self.model.device).expand(batch_size, -1)
            #attention_mask = self.attention_mask.to(self.model.device).expand(batch_size, -1)
            #ego_positions = self.ego_positions.to(self.model.device).expand(batch_size, -1)
            #image_positions = self.img_positions.to(self.model.device).expand(batch_size, -1)

            ego_positions = (input_ids == self.ego_context_tokens)
            image_positions = (input_ids == self.image_context_tokens)
            text_len = input_ids.shape[1]

            token_type_ids = torch.zeros((batch_size, text_len), dtype=torch.int32, device=input_ids.device)
            token_type_ids[ego_positions] = 1
            token_type_ids[image_positions] = 2
            token_type_embeds = self.token_type_embed(token_type_ids)

            #embedding = self.language.embed_tokens#.to(self.model.device)

            text_embeds = self.embedding(input_ids)
            text_embeds[ego_positions] = ego_embeds.reshape(-1, ego_embeds.shape[-1]).to(text_embeds.dtype)
            text_embeds[image_positions] = feature.reshape(-1, feature.shape[-1]).to(text_embeds.dtype)


            all_embeds = text_embeds + token_type_embeds

            outputs = self.language.forward(
                inputs_embeds=all_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
            )

            hidden_states = outputs.last_hidden_state
            #hidden_states = hidden_states[:, -1, :]
            #print(f"Hidden state: {hidden_states.shape}")
            #last_hidden_state = hidden_states[:, -1, :]

            #bev_semantic_map = self.bev_semantic_head(last_hidden_state)

            #output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

            output: Dict[str, torch.Tensor] = {}
            s0 = torch.zeros(batch_size, 3, device=self.model.device, dtype=torch.float32)
            trajectory_outputs = self.trajectory_head(s0, hidden_states, teacher_forcing_seq=teacher_forcing_seq, teacher_forcing_ratio=teacher_forcing_ratio)
            #print(f"Trajectory outputs: {trajectory_outputs['trajectory'].shape}")
            output.update(trajectory_outputs)

            #del token_type_ids, token_type_embeds, text_embeds, all_embeds, outputs, feature, hidden_states, ego_embeds, resized_images
            #del input_ids, attention_mask, image_positions, ego_positions
            #torch.cuda.empty_cache()

            #agent_outputs = self.agent_head(hidden_states)
            #output.update(agent_outputs)

            return output

    def extract_feature(self, pixel_values):
        if self.model.select_layer == -1:
            vit_embeds = self.visual(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.visual(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.model.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.model.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp(vit_embeds)
        return vit_embeds

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.model.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x
    
class EgoContinuousEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        out = self.fc2(x)
        out = self.norm(out + x)
        return self.act(out)
    
class DirectionEmbedding(nn.Module):
    def __init__(self, num_directions=4, hidden_dim=4096):
        super().__init__()
        self.embedding = nn.Embedding(num_directions, hidden_dim)

    def forward(self, direction_idx):
        return self.embedding(direction_idx)

class WaypointTransformerHead(nn.Module):
    def __init__(self, vlm_dim, traj_dim=3, num_waypoints=8, model_dim=512, num_layers=4, nhead=4):
        super().__init__()
        self.traj_dim = traj_dim
        self.num_waypoints = num_waypoints
        self.model_dim = model_dim

        # Project last known state into transformer input space
        self.input_proj = nn.Linear(traj_dim, model_dim)

        # Project VLM output into transformer memory space
        self.memory_proj = nn.Linear(vlm_dim, model_dim)

        # Positional embeddings for decoder input
        self.pos_emb = nn.Parameter(torch.empty(1, num_waypoints, model_dim))
        torch.nn.init.normal_(self.pos_emb)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=nhead, batch_first=True, dropout=0.2)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final output projection to (x, y, yaw)
        self.output_proj = nn.Linear(model_dim, traj_dim)

    def forward(self, s0, vlm_hidden_states, teacher_forcing_seq=None, teacher_forcing_ratio=1.0):
        """
        s0: initial ego vehicle state [B, 3]
        vlm_hidden_states: full VLM sequence [B, L, D]
        teacher_forcing_seq: [B, T, 3] for training, None for inference
        """
        B = s0.size(0)
        memory = self.memory_proj(vlm_hidden_states)  # [B, L, model_dim]

        if teacher_forcing_seq is not None:
            # Training mode
            teacher_forcing_seq = teacher_forcing_seq.to(s0.dtype)
            preds = []
            current = s0
            
            for t in range(self.num_waypoints):
                if t == 0:
                    inp = current
                else:
                    use_teacher = torch.rand(B, device = s0.device) < teacher_forcing_ratio
                    pred_for_inp = preds[-1].detach().squeeze(1)  # Remove the extra dimension
                    inp = torch.where(use_teacher.unsqueeze(1), 
                                    teacher_forcing_seq[:, t-1], 
                                    pred_for_inp).squeeze(1)
                if t == 0:
                    decoder_input = inp.unsqueeze(1)
                else:
                    decoder_input = torch.cat([decoder_input, inp.unsqueeze(1)], dim=1)

                x = self.input_proj(decoder_input) + self.pos_emb[:, :decoder_input.size(1)].to(decoder_input.dtype)
                T = decoder_input.size(1)
                causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
                causal_mask = causal_mask.masked_fill(causal_mask, float('-inf')).masked_fill(~causal_mask, float(0.0))

                out = self.decoder(tgt=x, memory=memory, tgt_mask=causal_mask)
                next_step = self.output_proj(out[:, -1:])
                preds.append(next_step)

            return {"trajectory": torch.cat(preds, dim=1)}  # [B, T, 3]
        else:
            # Inference (autoregressive generation)
            preds = []
            current = s0  # [B, 3]
            for t in range(self.num_waypoints):
                if t == 0:
                    decoder_input = current.unsqueeze(1)
                else:
                    decoder_input = torch.cat([decoder_input, preds[-1].squeeze(1).unsqueeze(1)], dim=1)
                    
                x = self.input_proj(decoder_input) + self.pos_emb[:, :decoder_input.size(1)].to(decoder_input.dtype)
                T = decoder_input.size(1)
                causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
                causal_mask = causal_mask.masked_fill(causal_mask, float('-inf')).masked_fill(~causal_mask, float(0.0))
                
                out = self.decoder(tgt=x, memory=memory, tgt_mask=causal_mask)
                next_step = self.output_proj(out[:, -1:])
                preds.append(next_step)
            
            return {"trajectory": torch.cat(preds, dim=1)}  # [B, T, 3]
    
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