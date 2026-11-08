"""Trajectory codebook loading shared by the vision_llm_lora model and loss."""

import numpy as np


def load_codebook(path: str, num_waypoints: int = 8) -> np.ndarray:
    """
    Load a (K, T, 3) codebook of (x, y, yaw) trajectories and subsample it to num_waypoints poses.
    Raw navsim codebooks hold 40 poses at 10 Hz with pose i at t = (i + 1) * 0.1 s, so for T = 40 and 8 waypoints
    this keeps indices [4, 9, ..., 39] = 0.5, 1.0, ..., 4.0 s, as codebook/codebook_builder.py does.
    :param path: .npy file of shape (K, T, 3)
    :param num_waypoints: poses per trajectory after subsampling
    :return: float32 array of shape (K, num_waypoints, 3)
    """
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Codebook must have shape (K, T, 3), got {arr.shape}.")

    num_steps = arr.shape[1]
    if num_steps < num_waypoints:
        raise ValueError(f"Codebook has {num_steps} timesteps but num_waypoints={num_waypoints}; cannot upsample.")

    step = num_steps // num_waypoints
    return arr[:, step - 1 : step * num_waypoints : step]
