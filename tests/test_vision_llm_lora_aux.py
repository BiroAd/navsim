"""CPU-only checks of the auxiliary BEV decoder, its losses and the per-builder cache root of CacheOnlyDataset."""
from pathlib import Path

import pytest
import torch

from navsim.agents.vision_llm_lora.vision_llm_lora_aux import (
    AUX_HEADING_BINS,
    AUX_MAP_SHAPE,
    AuxBEVDecoder,
    aux_losses,
    aux_output_channels,
)
from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_features import NO_LANE_HEADING
from navsim.planning.training.dataset import _builder_cache_file

ALL_AUX = dict(aux_bev_semantic_loss_weight=1.0, aux_drivable_area_loss_weight=1.0, aux_lane_heading_loss_weight=1.0)


def test_no_weights_no_outputs():
    assert aux_output_channels(VisionLLMLoRAConfig()) == {}


def test_outputs_follow_the_weights():
    config = VisionLLMLoRAConfig(aux_drivable_area_loss_weight=2.0)
    assert aux_output_channels(config) == {"aux_drivable_area": 1}
    config = VisionLLMLoRAConfig(**ALL_AUX)
    assert aux_output_channels(config) == {
        "aux_bev_semantic": len(config.bev_semantic_classes) + 1,
        "aux_drivable_area": 1,
        "aux_lane_heading": AUX_HEADING_BINS,
    }


def test_default_decoder_output_matches_the_cached_maps():
    config = VisionLLMLoRAConfig(**ALL_AUX)
    decoder = AuxBEVDecoder(in_dim=16, outputs=aux_output_channels(config), dim=32, heads=4,
                            query_grid=config.aux_query_grid, upsample_steps=config.aux_upsample_steps)
    out = decoder(torch.randn(2, 208, 16))
    assert out["aux_bev_semantic"].shape == (2, 7, *AUX_MAP_SHAPE)
    assert out["aux_drivable_area"].shape == (2, 1, *AUX_MAP_SHAPE)
    assert out["aux_lane_heading"].shape == (2, AUX_HEADING_BINS, *AUX_MAP_SHAPE)


def _targets(batch=2, shape=(8, 4)):
    bev = torch.randint(0, 7, (batch, *shape), dtype=torch.uint8)
    drivable = torch.randint(0, 2, (batch, *shape), dtype=torch.uint8)
    heading = torch.randint(0, AUX_HEADING_BINS, (batch, *shape), dtype=torch.uint8)
    heading[:, :2] = NO_LANE_HEADING
    return {"bev_semantic_map": bev, "drivable_area_map": drivable, "lane_heading_map": heading}


def _perfect_predictions(targets, scale=50.0):
    one_hot = lambda t, n: torch.nn.functional.one_hot(t.long().clamp(max=n - 1), n).permute(0, 3, 1, 2).float()  # noqa
    return {
        "aux_bev_semantic": one_hot(targets["bev_semantic_map"], 7) * scale,
        "aux_drivable_area": (targets["drivable_area_map"].float()[:, None] * 2 - 1) * scale,
        "aux_lane_heading": one_hot(targets["lane_heading_map"], AUX_HEADING_BINS) * scale,
    }


def test_perfect_predictions_give_zero_loss_and_full_metrics():
    targets = _targets()
    loss, extra = aux_losses(_perfect_predictions(targets), targets, VisionLLMLoRAConfig(**ALL_AUX))
    assert loss.item() == pytest.approx(0.0, abs=1e-4)
    assert extra["aux/bev_miou"].item() == pytest.approx(1.0)
    assert extra["aux/drivable_iou"].item() == pytest.approx(1.0)
    assert extra["aux/lane_heading_acc"].item() == pytest.approx(1.0)


def test_losses_are_weighted_and_logged_unweighted():
    targets = _targets()
    predictions = {k: torch.zeros_like(v) for k, v in _perfect_predictions(targets).items()}
    _, extra = aux_losses(predictions, targets, VisionLLMLoRAConfig(**ALL_AUX))
    loss, _ = aux_losses(predictions, targets, VisionLLMLoRAConfig(**{k: 2.0 for k in ALL_AUX}))
    unweighted = sum(extra[f"loss_terms/{k}"] for k in ("aux_bev_semantic", "aux_drivable_area", "aux_lane_heading"))
    assert loss.item() == pytest.approx(2.0 * unweighted.item())
    # Uniform logits: ln(classes) for the classifications, ln 2 for the BCE.
    assert extra["loss_terms/aux_bev_semantic"].item() == pytest.approx(torch.log(torch.tensor(7.0)).item(), rel=1e-4)
    assert extra["loss_terms/aux_lane_heading"].item() == pytest.approx(
        torch.log(torch.tensor(float(AUX_HEADING_BINS))).item(), rel=1e-4
    )


def test_lane_heading_without_lane_cells_is_zero_not_nan():
    targets = _targets()
    targets["lane_heading_map"][:] = NO_LANE_HEADING
    config = VisionLLMLoRAConfig(aux_lane_heading_loss_weight=1.0)
    loss, _ = aux_losses({"aux_lane_heading": torch.randn(2, AUX_HEADING_BINS, 8, 4)}, targets, config)
    assert torch.isfinite(loss) and loss.item() == 0.0


def test_shape_mismatch_raises():
    targets = _targets()
    config = VisionLLMLoRAConfig(aux_drivable_area_loss_weight=1.0)
    with pytest.raises(ValueError):
        aux_losses({"aux_drivable_area": torch.zeros(2, 1, 4, 4)}, targets, config)


def test_builder_cache_root_redirects_the_file(tmp_path):
    class Builder:
        def get_unique_name(self):
            return "vision_llm_aux_target"

    cache, token_path = tmp_path / "cache", tmp_path / "cache" / "log_a" / "token_1"
    builder = Builder()
    assert _builder_cache_file(builder, cache, token_path) == token_path / "vision_llm_aux_target.gz"
    builder.cache_root = str(tmp_path / "aux")
    assert _builder_cache_file(builder, cache, token_path) == Path(tmp_path / "aux/log_a/token_1/vision_llm_aux_target.gz")
