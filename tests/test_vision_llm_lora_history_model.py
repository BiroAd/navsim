"""
Model side of the past inputs (config.temporal_history_frame, config.ego_history_tokens), CPU-only on stubs:
the previous frame fills the first temporal slot of every patch and the current frame the second (video order),
with the same token count; previous = current reproduces the still-image patches; the strips layout arranges both
frames alike; the prompt gets 3 more <EGO_STATE> placeholders; the agent lists the history builders and loads the
history camera frame only when enabled.
"""
from types import MethodType, SimpleNamespace

import torch

from navsim.agents.vision_llm_lora.vision_llm_lora_agent import VisionLLMLoRAAgent
from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_model import VisionLLMLoRAModel


def _stub(image_layout="strips"):
    stub = SimpleNamespace(_patch_size=16, _temporal_size=2, _patch_dim=1536, _merger_factor=4, _merge_size=2,
                           config=SimpleNamespace(patch_order="merge_window", image_layout=image_layout))
    stub._patchify = MethodType(VisionLLMLoRAModel._patchify, stub)
    stub._preprocess_images = MethodType(VisionLLMLoRAModel._preprocess_images, stub)
    return stub


def _slots(patches):
    """(..., 1536) -> (..., C=3, t=2, 256): the two temporal slots of every patch."""
    return patches.reshape(*patches.shape[:-1], 3, 2, 256)


def test_previous_frame_fills_the_first_temporal_slot():
    stub = _stub()
    torch.manual_seed(0)
    cur, prev = torch.randn(2, 13, 3, 128, 128), torch.randn(2, 13, 3, 128, 128)
    still, grid_h, grid_w = stub._patchify(cur)
    pair, gh2, gw2 = stub._patchify(cur, prev)
    assert pair.shape == still.shape and (gh2, gw2) == (grid_h, grid_w)
    torch.testing.assert_close(_slots(pair)[..., 1, :], _slots(still)[..., 1, :])  # current in slot 2
    torch.testing.assert_close(_slots(pair)[..., 0, :], _slots(stub._patchify(prev)[0])[..., 0, :])  # earlier in slot 1
    torch.testing.assert_close(stub._patchify(cur, cur.clone())[0], still)  # previous = current: the still image


def test_strips_arrange_both_frames_alike():
    stub = _stub("strips")
    torch.manual_seed(1)
    cur, prev = torch.randn(2, 13, 3, 128, 128), torch.randn(2, 13, 3, 128, 128)
    still = stub._preprocess_images(cur)
    pair = stub._preprocess_images(cur, prev)
    assert pair[0].shape == still[0].shape and torch.equal(pair[1], still[1]) and pair[2] == still[2] == (96, 96, 16)
    torch.testing.assert_close(_slots(pair[0])[:, :, 1], _slots(still[0])[:, :, 1])
    torch.testing.assert_close(_slots(pair[0])[:, :, 0], _slots(stub._preprocess_images(prev)[0])[:, :, 0])


def test_prompt_has_three_past_position_placeholders():
    stub = SimpleNamespace(config=SimpleNamespace(image_tokens="qwen", ego_state_tokens=True, ego_history_tokens=True,
                                                  ego_history_position_scales=(7.5, 15.0, 22.5)),
                           _prompt_template="<IMG>\n<EGO>end")
    prompt = VisionLLMLoRAModel._build_prompt(stub, (16,), 1)
    assert "Current velocity:<EGO_STATE>, acceleration:<EGO_STATE>.\nPast positions:<EGO_STATE><EGO_STATE><EGO_STATE>.\n" in prompt
    assert prompt.count("<EGO_STATE>") == 5


def test_agent_builders_and_sensor_config():
    def agent(**kwargs):
        stub = SimpleNamespace(_config=VisionLLMLoRAConfig(camera_layout="panorama", **kwargs))
        stub._ec_in_reward = lambda: VisionLLMLoRAAgent._ec_in_reward(stub)
        return stub

    off = agent()
    on = agent(temporal_history_frame=1, ego_history_tokens=True, history_cache_path="/mnt/h")
    names = lambda agent: [b.get_unique_name() for b in VisionLLMLoRAAgent.get_feature_builders(agent)]
    assert names(off) == ["vision_llm_feature"]
    assert names(on) == ["vision_llm_feature", "vision_llm_history_t1", "vision_llm_ego_history"]
    assert all(getattr(b, "cache_root", "/mnt/h") == "/mnt/h" for b in VisionLLMLoRAAgent.get_feature_builders(on)[1:])
    assert VisionLLMLoRAAgent.get_sensor_config(off).cam_f0 == [3]
    assert VisionLLMLoRAAgent.get_sensor_config(on).cam_f0 == [2, 3]


def test_ego_tokens_two_components_unchanged_and_yaw_scaled():
    from navsim.agents.vision_llm_lora.vision_llm_lora_model import EgoStateTokens, ego_history_inputs

    torch.manual_seed(2)
    two = EgoStateTokens(hidden_size=8, scales=(15.0, 2.0), n_frequencies=3, embed_rms=0.013)
    states = torch.randn(4, 2, 2) * 5
    # the original computation: one scale per quantity for both components
    x = states / torch.tensor([15.0, 2.0])[:, None]
    angles = x[..., None] * two.freqs
    feats = torch.cat([x, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)
    expected = two.norm(torch.stack([mlp(feats[:, i]) for i, mlp in enumerate(two.mlps)], dim=1))
    torch.testing.assert_close(two(states), expected)

    three = EgoStateTokens(hidden_size=8, scales=[(7.5, 7.5, 0.5), (15.0, 15.0, 0.5)], n_frequencies=3,
                           embed_rms=0.013, n_components=3)
    assert three.scales.tolist() == [[7.5, 7.5, 0.5], [15.0, 15.0, 0.5]]
    assert three.mlps[0][0].in_features == 3 * (1 + 2 * 3)
    assert three(torch.randn(4, 2, 3)).shape == (4, 2, 8)


def test_history_inputs_wrap_yaw():
    import math

    from navsim.agents.vision_llm_lora.vision_llm_lora_model import ego_history_inputs

    pose = torch.tensor([[[-5.0, 0.2, 0.1], [-10.0, 0.5, 2 * math.pi - 0.2], [-15.0, 1.0, -3 * math.pi]]])
    out = ego_history_inputs(pose, use_yaw=True)
    torch.testing.assert_close(out[0, :, :2], pose[0, :, :2])
    torch.testing.assert_close(out[0, :, 2], torch.tensor([0.1, -0.2, math.pi]), atol=1e-5, rtol=0)
    assert ego_history_inputs(pose, use_yaw=False).shape == (1, 3, 2)
