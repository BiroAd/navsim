"""
Checks the vision_llm_lora environment on a GPU node: versions, CUDA, bf16, and that the fast paths Qwen3.5 uses
actually run here: PyTorch SDPA's flash kernel (full-attention layers, vision tower) and fla's Triton kernel for the
Gated DeltaNet layers, which compiles on this machine. No HF weights or network needed. Run inside a GPU allocation:
  srun -p gpu --gres=gpu:1 -c 8 -t 0:20:00 python scripts/setup/check_vlm_env.py
"""

import importlib
from importlib.metadata import PackageNotFoundError, version

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def main() -> None:
    for package in ("torch", "transformers", "peft", "accelerate", "pytorch-lightning", "fla-core", "triton",
                    "ray", "nuplan-devkit"):
        print(f"{package:18} {version(package)}")
    for package in ("flash-attn", "causal-conv1d"):
        try:
            print(f"{package:18} {version(package)}  <- installed but not importable on glibc 2.28; uninstall it")
        except PackageNotFoundError:
            print(f"{package:18} not installed (expected)")

    assert torch.cuda.is_available(), "no GPU visible"
    print(f"GPU: {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}, bf16 {torch.cuda.is_bf16_supported()}")

    importlib.import_module("transformers.models.qwen3_5")

    q = torch.randn(1, 16, 256, 128, device="cuda", dtype=torch.bfloat16)  # (B, heads, T, head_dim)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    print(f"SDPA flash kernel ok: {tuple(out.shape)}")

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    B, T, H, D = 1, 256, 4, 64
    k = F.normalize(torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16), dim=-1)
    v = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.float32))
    beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16)
    o, _ = chunk_gated_delta_rule(q=k, k=k, v=v, g=g, beta=beta)
    assert torch.isfinite(o).all()
    print(f"fla chunk_gated_delta_rule ok (Triton compiled here): {tuple(o.shape)}")
    print("all checks passed")


if __name__ == "__main__":
    main()
