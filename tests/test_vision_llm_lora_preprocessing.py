"""
CPU-only checks for vision_llm_lora image preprocessing (docs/vision_llm_lora_findings.md F9, F10, F13).
No network, dataset or HF weights: the model method is called on a stub with the Qwen3.5 patch constants.
Run: pytest tests/test_vision_llm_lora_preprocessing.py
"""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from navsim.agents.vision_llm.vision_llm_features import VisionLLMFeatureBuilder
from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_features import VisionLLMLoRAFeatureBuilder
from navsim.agents.vision_llm_lora.vision_llm_lora_model import VisionLLMLoRAModel

CAMERAS = ("cam_l0", "cam_f0", "cam_r0", "cam_l1", "cam_r1", "cam_l2", "cam_b0", "cam_r2")


def _agent_input(make_image):
    """AgentInput stand-in: only cameras[-1].<cam>.image is read by the builders."""
    cams = SimpleNamespace(**{name: SimpleNamespace(image=make_image(name)) for name in CAMERAS})
    return SimpleNamespace(cameras=[cams])


def _model_stub(patch_order, image_layout="tiles"):
    """Qwen3.5-4B vision constants: patch 16, temporal 2, 2x2 merge, patch_dim 3*2*16*16."""
    stub = SimpleNamespace(
        _patch_size=16,
        _temporal_size=2,
        _patch_dim=1536,
        _merger_factor=4,
        _merge_size=2,
        config=SimpleNamespace(patch_order=patch_order, image_layout=image_layout),
    )
    stub._patchify = MethodType(VisionLLMLoRAModel._patchify, stub)
    return stub


def _hf_patchify(images, patch_size=16, merge_size=2, temporal_patch_size=2):
    """Reference layout, copied from transformers Qwen2VLImageProcessor.patchify (main, 2026-10)."""
    b, c, h, w = images.shape
    gh, gw = h // patch_size, w // patch_size
    p = images.reshape(b, c, gh // merge_size, merge_size, patch_size, gw // merge_size, merge_size, patch_size)
    p = p.permute(0, 2, 5, 3, 6, 1, 4, 7)
    p = p.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
    return p.reshape(b * gh * gw, c * temporal_patch_size * patch_size * patch_size)


def test_lora_builder_defaults_reproduce_original_features():
    rng = np.random.default_rng(0)
    agent_input = _agent_input(lambda _: rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8))
    config = VisionLLMLoRAConfig()

    original = VisionLLMFeatureBuilder(config)._get_camera_feature(agent_input)
    lora = VisionLLMLoRAFeatureBuilder(config)._get_camera_feature(agent_input)

    assert lora.shape == (13, 3, 112, 112)
    assert torch.equal(original, lora)


def test_lora_builder_fixed_preprocessing_keeps_rgb_and_qwen_range():
    red = np.zeros((1080, 1920, 3), dtype=np.uint8)
    red[..., 0] = 255
    config = VisionLLMLoRAConfig(
        camera_width=128, camera_height=128, swap_red_blue=False, image_normalization="qwen"
    )

    features = VisionLLMLoRAFeatureBuilder(config)._get_camera_feature(_agent_input(lambda _: red.copy()))

    assert features.shape == (13, 3, 128, 128)
    assert torch.allclose(features[:, 0], torch.ones(1), atol=1e-6)  # red stays in channel 0, mapped to +1
    assert torch.allclose(features[:, 2], -torch.ones(1), atol=1e-6)  # blue empty, mapped to -1


def test_panorama_layout_places_cameras():
    # Each camera gets its own grey level; tile centres must show the expected camera.
    levels = {name: 20 * (i + 1) for i, name in enumerate(CAMERAS)}
    config = VisionLLMLoRAConfig(
        camera_layout="panorama",
        thumbnail="front_zoom",
        camera_width=128,
        camera_height=128,
        swap_red_blue=False,
        image_normalization="qwen",
    )

    features = VisionLLMLoRAFeatureBuilder(config)._get_camera_feature(
        _agent_input(lambda name: np.full((1080, 1920, 3), levels[name], dtype=np.uint8))
    )

    # Front row (4096 source px -> 768): L0 0-204, F0 204-564, R0 564-768. Rear row: R1 | B0 | L1, 256 px each.
    # Image 12 is the front zoom.
    expected = {
        0: "cam_l0", 2: "cam_f0", 3: "cam_f0", 5: "cam_r0", 6: "cam_r1", 8: "cam_b0", 9: "cam_b0", 11: "cam_l1",
        12: "cam_f0",
    }
    assert features.shape == (13, 3, 128, 128)
    for tile, name in expected.items():
        value = (levels[name] / 255 - 0.5) / 0.5
        assert abs(features[tile, 0, 64, 64].item() - value) < 1e-3, (tile, name)


def test_front_zoom_is_sharper_than_canvas_and_none_drops_the_13th_image():
    # Horizontal sine grating on F0 with a 16-px period: it survives the 3x zoom resize but not the 15x canvas one.
    grating = (127 + 100 * np.sin(np.arange(1920) * 2 * np.pi / 16)).astype(np.uint8)
    f0 = np.broadcast_to(grating[None, :, None], (1080, 1920, 3)).copy()

    def make(name):
        # Other cameras at the grating's mean, so only the grating can create variation in the 13th image.
        return f0 if name == "cam_f0" else np.full((1080, 1920, 3), 127, dtype=np.uint8)

    base = dict(camera_layout="panorama", camera_width=128, camera_height=128, swap_red_blue=False)
    zoom = VisionLLMLoRAFeatureBuilder(VisionLLMLoRAConfig(thumbnail="front_zoom", **base))
    canvas = VisionLLMLoRAFeatureBuilder(VisionLLMLoRAConfig(thumbnail="canvas", **base))
    none = VisionLLMLoRAFeatureBuilder(VisionLLMLoRAConfig(thumbnail="none", **base))

    assert zoom._get_camera_feature(_agent_input(make))[12].std() > 4 * canvas._get_camera_feature(
        _agent_input(make)
    )[12].std()
    assert none._get_camera_feature(_agent_input(make)).shape == (12, 3, 128, 128)


def test_merge_window_order_matches_qwen_processor():
    images = torch.randn(1, 2, 3, 128, 128)  # (B, P, C, H, W)

    pixel_values, grid_thw, tokens_per_image = VisionLLMLoRAModel._preprocess_images(
        _model_stub("merge_window"), images
    )

    assert torch.equal(pixel_values, _hf_patchify(images.reshape(2, 3, 128, 128)))
    assert grid_thw.tolist() == [[1, 8, 8], [1, 8, 8]]
    assert tokens_per_image == (16, 16)


def test_strips_reassemble_the_canvas_rows():
    torch.manual_seed(0)
    canvas = torch.randn(2, 3, 256, 768)  # (B, C, 2 rows × 128, 6 tiles × 128)
    zoom = torch.randn(2, 1, 3, 128, 128)
    # Cached layout: 12 tiles row by row (as dynamic_preprocess crops them), then the 13th image.
    tiles = canvas.reshape(2, 3, 2, 128, 6, 128).permute(0, 2, 4, 1, 3, 5).reshape(2, 12, 3, 128, 128)
    images = torch.cat([tiles, zoom], dim=1)

    pixel_values, grid_thw, tokens_per_image = VisionLLMLoRAModel._preprocess_images(
        _model_stub("merge_window", "strips"), images
    )

    # Per sample: front row, rear row, zoom, each patchified as the Qwen processor would.
    expected = torch.cat(
        [
            torch.cat([_hf_patchify(canvas[b : b + 1, :, :128]), _hf_patchify(canvas[b : b + 1, :, 128:]),
                       _hf_patchify(zoom[b])])
            for b in range(2)
        ]
    )
    assert torch.equal(pixel_values, expected)
    assert grid_thw.tolist() == [[1, 8, 48], [1, 8, 48], [1, 8, 8]] * 2
    assert tokens_per_image == (96, 96, 16)


def test_strips_without_a_13th_image_and_with_too_few_tiles():
    _, grid_thw, tokens_per_image = VisionLLMLoRAModel._preprocess_images(
        _model_stub("merge_window", "strips"), torch.randn(1, 12, 3, 128, 128)
    )
    assert grid_thw.tolist() == [[1, 8, 48], [1, 8, 48]]
    assert tokens_per_image == (96, 96)

    with pytest.raises(ValueError):
        VisionLLMLoRAModel._preprocess_images(_model_stub("merge_window", "strips"), torch.randn(1, 2, 3, 128, 128))


def test_raster_order_is_a_permutation_of_merge_window_order():
    images = torch.randn(1, 1, 3, 128, 128)
    raster, _, _ = VisionLLMLoRAModel._preprocess_images(_model_stub("raster"), images)
    window, _, _ = VisionLLMLoRAModel._preprocess_images(_model_stub("merge_window"), images)

    grid = 8
    for row in range(grid):
        for col in range(grid):
            window_idx = ((row // 2) * (grid // 2) + col // 2) * 4 + (row % 2) * 2 + col % 2
            assert torch.equal(raster[row * grid + col], window[window_idx])


def test_112_tiles_are_padded_to_128():
    images = torch.randn(1, 1, 3, 112, 112)
    pixel_values, grid_thw, tokens_per_image = VisionLLMLoRAModel._preprocess_images(
        _model_stub("merge_window"), images
    )

    assert grid_thw.tolist() == [[1, 8, 8]]
    assert pixel_values.shape == (64, 1536)
    assert tokens_per_image == (16,)
