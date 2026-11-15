"""
History builders (VisionLLMLoRAHistoryFrameBuilder, VisionLLMLoRAEgoHistoryBuilder): a past frame's uint8 tiles,
normalized, equal the current-frame builder's features of that frame (lossless storage), frame k reads
cameras[-1 - k], one cache name per frame, and the ego history is the 3 past statuses, most recent first.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_features import (
    VisionLLMLoRAEgoHistoryBuilder,
    VisionLLMLoRAFeatureBuilder,
    VisionLLMLoRAHistoryFrameBuilder,
    normalize_uint8_camera,
)

CAMERAS = ("cam_l0", "cam_f0", "cam_r0", "cam_l1", "cam_r1", "cam_l2", "cam_b0", "cam_r2")


def _frames(seed=0, n=4):
    rng = np.random.default_rng(seed)
    return [SimpleNamespace(**{c: SimpleNamespace(image=rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8))
                               for c in CAMERAS}) for _ in range(n)]


@pytest.mark.parametrize("normalization", ["qwen", "imagenet"])
def test_history_frames_match_the_current_frame_builder(normalization):
    config = VisionLLMLoRAConfig(camera_layout="panorama", thumbnail="front_zoom", camera_width=128,
                                 camera_height=128, swap_red_blue=False, image_normalization=normalization)
    frames = _frames()
    current = VisionLLMLoRAFeatureBuilder(config)
    for k in (1, 2, 3):
        builder = VisionLLMLoRAHistoryFrameBuilder(config, k)
        assert builder.get_unique_name() == f"vision_llm_history_t{k}"
        tiles = builder.compute_features(SimpleNamespace(cameras=frames))[f"camera_history_t{k}"]
        assert tiles.dtype == torch.uint8 and tiles.shape == (13, 3, 128, 128)
        reference = current._get_camera_feature(SimpleNamespace(cameras=[frames[-1 - k]]))  # that frame as "current"
        torch.testing.assert_close(normalize_uint8_camera(tiles, normalization), reference, rtol=0, atol=1e-6)
    with pytest.raises(ValueError):
        VisionLLMLoRAHistoryFrameBuilder(config, 4)


def test_ego_history_is_most_recent_first():
    statuses = [SimpleNamespace(ego_pose=np.array([-3.0 + i, 0.1 * i, 0.01 * i]), ego_velocity=np.array([i, 0.0]),
                                ego_acceleration=np.array([0.0, i]), driving_command=np.eye(4)[i % 4])
                for i in range(4)]  # index 3 = current
    out = VisionLLMLoRAEgoHistoryBuilder().compute_features(SimpleNamespace(ego_statuses=statuses))
    assert out["ego_history_pose"].shape == (3, 3)
    torch.testing.assert_close(out["ego_history_pose"][:, 0], torch.tensor([-1.0, -2.0, -3.0]))  # t-0.5, t-1.0, t-1.5
    torch.testing.assert_close(out["ego_history_velocity"][:, 0], torch.tensor([2.0, 1.0, 0.0]))
    assert out["ego_history_command"].tolist() == [2, 1, 0]


def test_history_builders_rotate_over_mounts(tmp_path):
    """Several mounts joined by os.pathsep are used in turn by CacheOnlyDataset's lookup; one path stays one path."""
    import os
    from pathlib import Path

    from navsim.agents.vision_llm_lora.vision_llm_lora_features import VisionLLMLoRAEgoHistoryBuilder
    from navsim.planning.training.dataset import _builder_cache_file

    mounts = [str(tmp_path / f"m{i}") for i in range(3)]
    builder = VisionLLMLoRAEgoHistoryBuilder(os.pathsep.join(mounts))
    cache_path = tmp_path / "cache"
    token_path = cache_path / "log_a" / "tok1"
    seen = [_builder_cache_file(builder, cache_path, token_path) for _ in range(6)]
    assert {str(p.parents[2]) for p in seen} == set(mounts)  # every mount used
    assert all(p.relative_to(p.parents[2]) == Path("log_a/tok1/vision_llm_ego_history.gz") for p in seen)
    assert seen[0] != seen[1]  # consecutive reads go to different mounts

    single = VisionLLMLoRAEgoHistoryBuilder(mounts[0])
    assert {str(_builder_cache_file(single, cache_path, token_path).parents[2]) for _ in range(3)} == {mounts[0]}
    assert VisionLLMLoRAEgoHistoryBuilder(None).cache_root is None
    assert _builder_cache_file(VisionLLMLoRAEgoHistoryBuilder(None), cache_path, token_path) == (
        token_path / "vision_llm_ego_history.gz"
    )
