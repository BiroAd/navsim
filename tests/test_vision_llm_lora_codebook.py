"""
CPU-only checks for codebook loading (docs/vision_llm_lora_findings.md F4).
Run: pytest tests/test_vision_llm_lora_codebook.py
"""

from pathlib import Path

import numpy as np

from navsim.agents.vision_llm_lora.vision_llm_lora_codebook import load_codebook

CODEBOOK_DIR = Path(__file__).resolve().parents[1] / "navsim/agents/vision_llm_lora/codebook"


def test_40_step_codebook_keeps_half_second_poses(tmp_path):
    # value = pose index, so the selected indices can be read back directly
    raw = np.broadcast_to(np.arange(40, dtype=np.float32)[None, :, None], (2, 40, 3)).copy()
    np.save(tmp_path / "raw.npy", raw)

    codebook = load_codebook(str(tmp_path / "raw.npy"))

    assert codebook.shape == (2, 8, 3)
    assert codebook[0, :, 0].tolist() == [4, 9, 14, 19, 24, 29, 34, 39]  # 0.5, 1.0, ..., 4.0 s


def test_8_step_codebook_is_used_as_is(tmp_path):
    raw = np.random.default_rng(0).standard_normal((3, 8, 3)).astype(np.float32)
    np.save(tmp_path / "eight.npy", raw)

    assert np.array_equal(load_codebook(str(tmp_path / "eight.npy")), raw)


def test_raw_repo_codebook_matches_committed_8_step_file():
    from_raw = load_codebook(str(CODEBOOK_DIR / "16384.npy"))
    committed = np.load(CODEBOOK_DIR / "16384_8step.npy")

    assert from_raw.shape == committed.shape == (16384, 8, 3)
    assert np.array_equal(from_raw, committed)
