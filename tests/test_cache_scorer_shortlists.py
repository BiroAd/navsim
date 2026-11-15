"""Past-frame inputs of scripts/training/cache_scorer_shortlists.py (the same as the EC reward's previous plan)."""
import importlib.util
from pathlib import Path

import torch

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "training" / "cache_scorer_shortlists.py"
_spec = importlib.util.spec_from_file_location("cache_scorer_shortlists", _PATH)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)


def test_frame_features():
    B = 2
    features = {
        "camera_feature": torch.zeros(B, 13, 3, 4, 4),
        "camera_history_t1": torch.full((B, 13, 3, 4, 4), 255, dtype=torch.uint8),
        "camera_history_t2": torch.zeros((B, 13, 3, 4, 4), dtype=torch.uint8),
        "velocity": torch.tensor([[5.0, 0.0], [6.0, 0.0]]),
        "acceleration": torch.zeros(B, 2),
        "driving_command": torch.tensor([1, 1]),
        "ego_history_velocity": torch.arange(B * 3 * 2, dtype=torch.float32).view(B, 3, 2),
        "ego_history_acceleration": -torch.arange(B * 3 * 2, dtype=torch.float32).view(B, 3, 2),
        "ego_history_command": torch.tensor([[0, 2, 1], [2, 0, 3]]),
    }
    assert script.frame_features(features, 0, "qwen") is features
    f1, f2 = script.frame_features(features, 1, "qwen"), script.frame_features(features, 2, "qwen")
    torch.testing.assert_close(f1["camera_feature"], torch.ones(B, 13, 3, 4, 4))  # 255 -> (1 - 0.5) / 0.5
    torch.testing.assert_close(f2["camera_feature"], -torch.ones(B, 13, 3, 4, 4))
    torch.testing.assert_close(f2["velocity"], features["ego_history_velocity"][:, 1])
    torch.testing.assert_close(f1["acceleration"], features["ego_history_acceleration"][:, 0])
    assert f1["driving_command"].tolist() == [0, 2] and f2["driving_command"].tolist() == [2, 0]
    torch.testing.assert_close(features["camera_feature"], torch.zeros(B, 13, 3, 4, 4))  # input unchanged
