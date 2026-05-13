from navsim.agents.vision_llm.vision_llm_features import (
    BoundingBox2DIndex,
    VisionLLMFeatureBuilder,
    VisionLLMTargetBuilder,
)


class VisionLLMLoRAFeatureBuilder(VisionLLMFeatureBuilder):
    """Thin wrapper to keep LoRA feature builder customization isolated."""


class VisionLLMLoRATargetBuilder(VisionLLMTargetBuilder):
    """Thin wrapper to keep LoRA target builder customization isolated."""
