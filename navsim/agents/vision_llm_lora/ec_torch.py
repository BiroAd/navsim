"""
Two-frame extended comfort (EC) of every codebook entry against a previous plan, batched in torch (GPU).

A port of what the official scorer does for EC (navsim/planning/simulation/planner/pdm_planner): each plan is tracked
by BatchLQRTracker and propagated by BatchKinematicBicycleModel (PDMSimulator, 40 steps of 0.1 s) from its own logged
ego state, then SceneAggregator._compute_two_frame_comfort compares the two simulated state sequences on their 3.5 s
overlap with ego_is_two_frame_extended_comfort (RMS difference of acceleration, jerk, yaw rate and yaw acceleration
against fixed thresholds).

Simplifications that are exact:
- Everything is done in each frame's own ego frame (rear axle at the origin, heading 0). The tracker and the motion model
  are invariant to a rigid transform, and the EC features (acceleration magnitude, jerk, yaw rate, yaw acceleration)
  too, so the global poses of the two frames are not needed.
- The tracker's reference (the interpolated poses and the velocity / curvature profiles it fits to them) depends only
  on the codebook entry: it is precomputed once with the official numpy functions (precompute_tracking).
- The Savitzky-Golay filters of the EC features are linear maps of a fixed-length signal (mode "interp"): one matrix
  each (built with scipy's savgol_filter on the identity).

The initial state is what NavSimScenario / the metric cache build from a frame's ego status: velocity and acceleration
(rear axle, ego frame) as logged, steering angle, steering rate, angular velocity and angular acceleration 0.
float64 throughout, as the official code.
"""
from typing import Dict, Optional

import numpy as np
import torch

# StateIndex (pdm_enums.py)
X, Y, HEADING, VX, VY, AX, AY, STEER, STEER_RATE, ANG_VEL, ANG_ACC = range(11)
NUM_STATES = 11
DT = 0.1  # proposal_sampling.interval_length
NUM_POSES = 40  # proposal_sampling.num_poses
OVERLAP_SHIFT = 5  # 0.5 s between the two frames / DT
# BatchLQRTracker defaults
Q_LONGITUDINAL, R_LONGITUDINAL = 10.0, 1.0
Q_LATERAL = (1.0, 10.0, 0.0)
R_LATERAL = 1.0
TRACKING_HORIZON = 10
JERK_PENALTY, CURVATURE_RATE_PENALTY = 1e-4, 1e-2
STOPPING_GAIN, STOPPING_VELOCITY = 0.5, 0.2
# BatchKinematicBicycleModel defaults
MAX_STEERING_ANGLE = np.pi / 3
ACCEL_TIME_CONSTANT, STEERING_TIME_CONSTANT = 0.2, 0.05
# pdm_comfort_metrics.py thresholds
EC_THRESHOLDS = {"acceleration": 0.7, "jerk": 0.5, "yaw_rate": 0.1, "yaw_accel": 0.1}


def precompute_tracking(codebook: np.ndarray) -> Dict[str, np.ndarray]:
    """
    The tracker reference of every codebook entry (K, 8, 3; x, y, heading at 0.5 ... 4.0 s, ego frame), with the official
    functions: transform_trajectory + get_trajectory_as_array from an ego state at the origin (poses only; velocity is
    ignored there), then get_velocity_curvature_profiles_with_derivatives_from_poses.
    Returns ref_poses (K, 41, 3), velocity_profile (K, 40), curvature_profile (K, 40), wheel_base, all float64.
    """
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.simulation.batch_lqr_utils import (
        get_velocity_curvature_profiles_with_derivatives_from_poses,
    )

    vehicle = get_pacifica_parameters()
    origin = EgoState.build_from_rear_axle(
        StateSE2(0.0, 0.0, 0.0),
        tire_steering_angle=0.0,
        vehicle_parameters=vehicle,
        time_point=TimePoint(0),
        rear_axle_velocity_2d=StateVector2D(0.0, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
    )
    traj_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    proposal_sampling = TrajectorySampling(num_poses=NUM_POSES, interval_length=DT)
    ref = np.stack([
        get_trajectory_as_array(
            transform_trajectory(Trajectory(np.asarray(entry, np.float32), traj_sampling), origin),
            proposal_sampling,
            origin.time_point,
        )[:, :3]
        for entry in codebook
    ])
    velocity, _, curvature, _ = get_velocity_curvature_profiles_with_derivatives_from_poses(
        discretization_time=DT, poses=ref, jerk_penalty=JERK_PENALTY, curvature_rate_penalty=CURVATURE_RATE_PENALTY
    )
    return {"ref_poses": ref, "velocity_profile": velocity, "curvature_profile": curvature,
            "wheel_base": np.float64(vehicle.wheel_base)}


def _savgol_matrix(n: int, window: int, poly: int, deriv: int, delta: float) -> np.ndarray:
    """F (n, n) with savgol_filter(y, ..., axis=-1) == y @ F.T (mode "interp", as scipy's default)."""
    from scipy.signal import savgol_filter

    return savgol_filter(np.eye(n), window_length=min(window, n), polyorder=poly, deriv=deriv, delta=delta, axis=0)


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    """normalize_angle (pdm_geometry_utils): arctan2(sin, cos)."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _principal_value(angle: torch.Tensor) -> torch.Tensor:
    """nuplan.common.geometry.compute.principal_value with min_ = -pi: (angle + pi) % 2pi - pi."""
    return torch.remainder(angle + np.pi, 2 * np.pi) - np.pi


class TorchExtendedComfort:
    """
    ec = TorchExtendedComfort(precompute_tracking(codebook), device)
    ec.against_previous(v_t, a_t, v_prev, a_prev, prev_idx) -> (S, K) bool: EC of every entry at t against entry
    prev_idx[s] at t - 0.5 s, each simulated from its own frame's velocity / acceleration (S, 2).
    """

    def __init__(self, tracking: Dict[str, np.ndarray], device: torch.device):
        f64 = dict(dtype=torch.float64, device=device)
        self.ref = torch.as_tensor(tracking["ref_poses"], **f64)  # (K, 41, 3)
        self.vel = torch.as_tensor(tracking["velocity_profile"], **f64)  # (K, 40)
        self.curv = torch.as_tensor(tracking["curvature_profile"], **f64)  # (K, 40)
        self.wheel_base = float(tracking["wheel_base"])
        self.device = device
        n = NUM_POSES + 1 - OVERLAP_SHIFT  # 36 overlap steps
        self.f_acc = torch.as_tensor(_savgol_matrix(n, 8, 2, 0, 1.0), **f64)
        self.f_jerk = torch.as_tensor(_savgol_matrix(n, 15, 2, 1, DT), **f64)
        self.f_yaw1 = torch.as_tensor(_savgol_matrix(n, 5, 2, 1, DT), **f64)
        self.f_yaw2 = torch.as_tensor(_savgol_matrix(n, 5, 2, 2, DT), **f64)
        # curvature reference over the tracking horizon at every iteration (as _compute_reference_velocity_and_...)
        last = self.vel.shape[1] - 1
        idx = torch.arange(NUM_POSES)[:, None] + torch.arange(TRACKING_HORIZON)[None]  # (40, 10)
        ref_idx = torch.clamp(torch.arange(NUM_POSES) + TRACKING_HORIZON, max=last)  # (40,)
        self.curv_idx = torch.minimum(idx, ref_idx[:, None]).to(device)  # beyond the reference: the reference value
        self.vref_idx = ref_idx.to(device)

    @torch.no_grad()
    def simulate(self, velocity: torch.Tensor, acceleration: torch.Tensor, entries: Optional[torch.Tensor] = None):
        """
        PDMSimulator.simulate_proposals in each frame's ego frame. velocity / acceleration (S, 2) per frame; entries
        (S, M) codebook ids (None: all K). Returns heading, acceleration x / y over the 41 steps, each (S, M, 41).
        """
        S = velocity.shape[0]
        if entries is None:
            entries = torch.arange(self.ref.shape[0], device=self.device).expand(S, -1)
        M = entries.shape[1]
        flat = entries.reshape(-1)
        ref, vel, curv = self.ref[flat], self.vel[flat], self.curv[flat]  # (N, 41, 3), (N, 40), (N, 40)
        N = flat.shape[0]
        s = torch.zeros(N, NUM_STATES, dtype=torch.float64, device=self.device)
        s[:, VX:VY + 1] = velocity.double().repeat_interleave(M, 0)
        s[:, AX:AY + 1] = acceleration.double().repeat_interleave(M, 0)
        heading, ax, ay = [s[:, HEADING]], [s[:, AX]], [s[:, AY]]
        q1, q2, _ = Q_LATERAL
        L = self.wheel_base
        steps = torch.arange(TRACKING_HORIZON, dtype=torch.float64, device=self.device)
        for i in range(NUM_POSES):  # current iteration index; produces state i + 1
            # ── BatchLQRTracker.track_trajectory ──
            r = ref[:, i]
            x_err, y_err = s[:, X] - r[:, 0], s[:, Y] - r[:, 1]
            lat_err = -x_err * torch.sin(r[:, 2]) + y_err * torch.cos(r[:, 2])
            head_err = _wrap(s[:, HEADING] - r[:, 2])
            v0 = s[:, VX]
            v_ref = vel[:, self.vref_idx[i]]
            kappa = curv.gather(1, self.curv_idx[i].expand(N, -1))  # (N, 10)
            stop = (v_ref <= STOPPING_VELOCITY) & (v0 <= STOPPING_VELOCITY)
            b_long = TRACKING_HORIZON * DT
            accel_lqr = -1.0 / (b_long * Q_LONGITUDINAL * b_long + R_LONGITUDINAL) * b_long * Q_LONGITUDINAL * (v0 - v_ref)
            accel = torch.where(stop, -STOPPING_GAIN * (v0 - v_ref), accel_lqr)
            v_prof = v0[:, None] + accel_lqr[:, None] * DT * steps  # (N, 10) linearization velocities
            # lateral: x_{j+1} = M_j x_j + in * u + aff_j, accumulated over the horizon (A, B, g as the official loop)
            A = torch.eye(3, dtype=torch.float64, device=self.device).expand(N, 3, 3).clone()
            B = torch.zeros(N, 3, dtype=torch.float64, device=self.device)
            g = torch.zeros(N, 3, dtype=torch.float64, device=self.device)
            for j in range(TRACKING_HORIZON):
                a01 = v_prof[:, j] * DT
                a12 = v_prof[:, j] * DT / L
                aff1 = -v_prof[:, j] * kappa[:, j] * DT
                # M_j = I + a01 e0 e1^T + a12 e1 e2^T
                A = A.clone()
                A_new0 = A[:, 0] + a01[:, None] * A[:, 1]
                A_new1 = A[:, 1] + a12[:, None] * A[:, 2]
                A = torch.stack([A_new0, A_new1, A[:, 2]], 1)
                B = torch.stack([B[:, 0] + a01 * B[:, 1], B[:, 1] + a12 * B[:, 2], B[:, 2] + DT], 1)
                g = torch.stack([g[:, 0] + a01 * g[:, 1], g[:, 1] + a12 * g[:, 2] + aff1, g[:, 2]], 1)
            x0 = torch.stack([lat_err, head_err, s[:, STEER]], 1)
            err = torch.einsum("nij,nj->ni", A, x0) + g
            err = torch.stack([err[:, 0], _wrap(err[:, 1]), _wrap(err[:, 2])], 1)
            btq = B * torch.tensor(Q_LATERAL, dtype=torch.float64, device=self.device)  # B^T Q (Q diagonal)
            steer_rate_lqr = -1.0 / ((btq * B).sum(1) + R_LATERAL) * (btq * err).sum(1)
            steer_rate_cmd = torch.where(stop, torch.zeros_like(steer_rate_lqr), steer_rate_lqr)
            # ── BatchKinematicBicycleModel.propagate_state ──
            steer = s[:, STEER]
            ideal_steer = DT * steer_rate_cmd + steer
            new_accel = DT / (DT + ACCEL_TIME_CONSTANT) * (accel - s[:, AX]) + s[:, AX]
            new_steer = DT / (DT + STEERING_TIME_CONSTANT) * (ideal_steer - steer) + steer
            new_steer_rate = (new_steer - steer) / DT
            out = s.clone()
            out[:, X] = s[:, X] + s[:, VX] * torch.cos(s[:, HEADING]) * DT
            out[:, Y] = s[:, Y] + s[:, VX] * torch.sin(s[:, HEADING]) * DT
            out[:, HEADING] = _principal_value(s[:, HEADING] + s[:, VX] * torch.tan(steer) / L * DT)
            out[:, VX] = s[:, VX] + new_accel * DT
            out[:, VY] = 0.0
            out[:, STEER] = torch.clamp(steer + new_steer_rate * DT, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
            out[:, ANG_VEL] = out[:, VX] * torch.tan(out[:, STEER]) / L
            out[:, AX] = new_accel
            out[:, AY] = 0.0
            out[:, ANG_ACC] = (out[:, ANG_VEL] - s[:, ANG_VEL]) / DT
            out[:, STEER_RATE] = new_steer_rate
            s = out
            heading.append(s[:, HEADING])
            ax.append(s[:, AX])
            ay.append(s[:, AY])
        shape = (S, M, NUM_POSES + 1)
        return (torch.stack(heading, 1).view(shape), torch.stack(ax, 1).view(shape), torch.stack(ay, 1).view(shape))

    def _features(self, heading: torch.Tensor, ax: torch.Tensor, ay: torch.Tensor) -> Dict[str, torch.Tensor]:
        """extract_features (pdm_comfort_metrics.py) on (..., 36) sequences."""
        acc = torch.round(torch.hypot(ax, ay) @ self.f_acc.T, decimals=8)
        jerk = torch.round(acc @ self.f_jerk.T, decimals=8)
        two_pi = 2 * np.pi
        adj = torch.zeros_like(heading)
        adj[..., 1:] = torch.cumsum(torch.round(torch.diff(heading, dim=-1) / two_pi), dim=-1)
        unwrapped = heading - two_pi * adj
        yaw_rate = torch.round(unwrapped @ self.f_yaw1.T, decimals=8)
        yaw_accel = torch.round(unwrapped @ self.f_yaw2.T, decimals=8)
        return {"acceleration": acc, "jerk": jerk, "yaw_rate": yaw_rate, "yaw_accel": yaw_accel}

    @torch.no_grad()
    def against_previous(
        self,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        prev_velocity: torch.Tensor,
        prev_acceleration: torch.Tensor,
        prev_entries: torch.Tensor,
        entries: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        EC (S, M) of the entries at t (all K if entries is None) against prev_entries (S,) at t - 0.5 s; the current
        sequence's first 36 states against the previous one's last 36 (_compute_two_frame_comfort, interval 0.5 s).
        """
        cur = self.simulate(velocity, acceleration, entries)
        prev = self.simulate(prev_velocity, prev_acceleration, prev_entries[:, None].long())
        fc = self._features(*(x[..., :-OVERLAP_SHIFT] for x in cur))
        fp = self._features(*(x[..., OVERLAP_SHIFT:] for x in prev))  # (S, 1, 36)
        ok = None
        for name, threshold in EC_THRESHOLDS.items():
            rms = (fc[name] - fp[name]).pow(2).mean(-1).sqrt()
            ok = rms <= threshold if ok is None else ok & (rms <= threshold)
        return ok
