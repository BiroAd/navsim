"""
Offline PDM scoring of every codebook trajectory in a scene, against the scene's metric cache.

Matches navsim/evaluate/pdm_score.py run once per codebook entry, but vectorized over the codebook:
- ego-frame poses -> interpolated global state arrays, as transform_trajectory + get_trajectory_as_array do,
- entries are simulated and scored in chunks, each chunk together with the PDM-Closed reference,
- ego progress is normalized per (reference, entry) pair, as when evaluation scores [reference, prediction].
  PDMScorer normalizes by the best proposal in the batch, so scoring all entries in one batch would not match.

Background traffic is log replay (non-reactive): the environment does not depend on the ego trajectory, so all
entries share one simulation. Official evaluation uses reactive IDM traffic, so scores can differ where traffic
would have reacted to the ego. The per-frame score excludes the two-frame extended comfort term, which needs
consecutive frames' predictions.
"""

from typing import Dict, List

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.geometry.compute import principal_value
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.enums import SceneFrameType
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)
from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents

MULTI_METRICS = {
    "no_at_fault_collisions": MultiMetricIndex.NO_COLLISION,
    "drivable_area_compliance": MultiMetricIndex.DRIVABLE_AREA,
    "driving_direction_compliance": MultiMetricIndex.DRIVING_DIRECTION,
    "traffic_light_compliance": MultiMetricIndex.TRAFFIC_LIGHT_COMPLIANCE,
}
WEIGHTED_METRICS = {
    "ego_progress": WeightedMetricIndex.PROGRESS,
    "time_to_collision_within_bound": WeightedMetricIndex.TTC,
    "lane_keeping": WeightedMetricIndex.LANE_KEEPING,
    "history_comfort": WeightedMetricIndex.HISTORY_COMFORT,
}


def _interp_linear(knot_us: npt.NDArray, values: npt.NDArray, query_us: npt.NDArray) -> npt.NDArray:
    """
    Linear interpolation along axis 1, written as scipy.interpolate.interp1d computes it.
    :param knot_us: [T] increasing knot times
    :param values: [K, T, ...] values at the knots
    :param query_us: [Q] query times within [knot_us[0], knot_us[-1]]
    :return: [K, Q, ...]
    """
    hi = np.clip(np.searchsorted(knot_us, query_us), 1, len(knot_us) - 1)
    lo = hi - 1
    shape = (1, len(query_us)) + (1,) * (values.ndim - 2)
    slope = (values[:, hi] - values[:, lo]) / (knot_us[hi] - knot_us[lo]).reshape(shape)
    return slope * (query_us - knot_us[lo]).reshape(shape) + values[:, lo]


def codebook_to_state_arrays(
    poses: npt.NDArray,
    initial_ego_state: EgoState,
    trajectory_sampling: TrajectorySampling,
    proposal_sampling: TrajectorySampling,
) -> npt.NDArray[np.float64]:
    """
    Vectorized transform_trajectory + get_trajectory_as_array (navsim/evaluate/pdm_score.py) for many trajectories.
    :param poses: [K, T, 3] (x, y, heading) in the ego frame at trajectory_sampling
    :param initial_ego_state: ego state the poses are relative to
    :param trajectory_sampling: sampling of the poses
    :param proposal_sampling: sampling of the returned state arrays
    :return: [K, proposal_sampling.num_poses + 1, StateIndex.size()]
    """
    poses = np.asarray(poses, dtype=np.float64)
    num_entries, num_poses, _ = poses.shape
    expected = trajectory_sampling.num_poses
    assert num_poses == expected, f"expected {expected} poses, got {num_poses}"

    # Relative -> absolute poses: origin_transform @ relative_transform (nuplan relative_to_absolute_poses)
    origin = initial_ego_state.rear_axle
    cos_o, sin_o = np.cos(origin.heading), np.sin(origin.heading)
    cos_r, sin_r = np.cos(poses[..., 2]), np.sin(poses[..., 2])
    abs_x = cos_o * poses[..., 0] - sin_o * poses[..., 1] + origin.x
    abs_y = sin_o * poses[..., 0] + cos_o * poses[..., 1] + origin.y
    abs_heading = np.arctan2(sin_o * cos_r + cos_o * sin_r, cos_o * cos_r - sin_o * sin_r)

    # Knots: the initial state, then one state per pose with zero velocity/acceleration/steering
    # (_get_fixed_timesteps and _se2_vel_acc_to_ego_state in nuplan's ml_planner transform_utils)
    time_point = initial_ego_state.time_point
    knot_s = np.arange(0.0, trajectory_sampling.time_horizon, trajectory_sampling.interval_length)
    knot_s = knot_s + trajectory_sampling.interval_length + time_point.time_s
    knot_us = np.array([time_point.time_us] + [int(t * 1e6) for t in knot_s.tolist()], dtype=np.float64)

    dynamic = initial_ego_state.dynamic_car_state
    linear = np.zeros((num_entries, num_poses + 1, 7), dtype=np.float64)  # x, y, vx, vy, ax, ay, steering
    linear[:, 0] = [
        origin.x,
        origin.y,
        dynamic.rear_axle_velocity_2d.x,
        dynamic.rear_axle_velocity_2d.y,
        dynamic.rear_axle_acceleration_2d.x,
        dynamic.rear_axle_acceleration_2d.y,
        initial_ego_state.tire_steering_angle,
    ]
    linear[:, 1:, 0] = abs_x
    linear[:, 1:, 1] = abs_y
    heading = np.concatenate([np.full((num_entries, 1), origin.heading), abs_heading], axis=1)

    # Query times as in get_trajectory_as_array
    query_s = np.arange(
        0.0,
        proposal_sampling.time_horizon + proposal_sampling.interval_length,
        proposal_sampling.interval_length,
    )
    query_s += time_point.time_s
    query_us = np.clip([int(t * 1e6) for t in query_s], knot_us[0], knot_us[-1]).astype(np.float64)

    linear_q = _interp_linear(knot_us, linear, query_us)
    heading_q = principal_value(_interp_linear(knot_us, np.unwrap(heading, axis=1), query_us))

    # Interpolated EgoStates are rebuilt without steering rate, angular velocity and angular acceleration (zeros)
    states = np.zeros((num_entries, len(query_us), StateIndex.size()), dtype=np.float64)
    states[..., StateIndex.X] = linear_q[..., 0]
    states[..., StateIndex.Y] = linear_q[..., 1]
    states[..., StateIndex.HEADING] = heading_q
    states[..., StateIndex.VELOCITY_X] = linear_q[..., 2]
    states[..., StateIndex.VELOCITY_Y] = linear_q[..., 3]
    states[..., StateIndex.ACCELERATION_X] = linear_q[..., 4]
    states[..., StateIndex.ACCELERATION_Y] = linear_q[..., 5]
    states[..., StateIndex.STEERING_ANGLE] = linear_q[..., 6]
    return states


def pairwise_scores(
    multi_metrics: npt.NDArray[np.float64],
    weighted_metrics: npt.NDArray[np.float64],
    progress_raw: npt.NDArray[np.float64],
    config: PDMScorerConfig,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Ego progress and score of proposals 1..N, each as if scored in a pair with proposal 0 (the PDM-Closed
    reference), following PDMScorer._aggregate_pdm_scores.
    :param multi_metrics: [len(MultiMetricIndex), N + 1]
    :param weighted_metrics: [len(WeightedMetricIndex), N + 1]; the progress row is ignored
    :param progress_raw: [N + 1] progress along the centerline [m]
    :param config: scorer config (weights and progress threshold)
    :return: "ego_progress" and "score", each [N]
    """
    multi_prod = multi_metrics.prod(axis=0)
    masked_progress = progress_raw * multi_prod
    norm = np.maximum(masked_progress[0], masked_progress[1:])
    above = norm > config.progress_distance_threshold
    ratio = np.divide(progress_raw[1:], norm, out=np.ones_like(norm), where=above)
    ego_progress = np.where(above, np.clip(ratio, 0.0, 1.0), 1.0)

    weighted = weighted_metrics[:, 1:].copy()
    weighted[WeightedMetricIndex.PROGRESS] = ego_progress
    mask = np.ones(len(WeightedMetricIndex), dtype=bool)
    mask[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] = False
    weights = config.weighted_metrics_array[mask]
    score = multi_prod[1:] * (weighted[mask] * weights[:, None]).sum(axis=0) / weights.sum()
    return {"ego_progress": ego_progress, "score": score}


def human_zero_metrics(
    metric_cache: MetricCache,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_agents_policy: LogReplayTrafficAgents,
) -> List[str]:
    """
    Metrics the human trajectory fails (scores 0). Evaluation reports these metrics as 1 for the prediction
    (human_penalty_filter in navsim/evaluate/pdm_score.py); its final score is computed without this filter.
    :return: names from MULTI_METRICS / WEIGHTED_METRICS
    """
    if not (scorer._config.human_penalty_filter and metric_cache.scene_type == SceneFrameType.ORIGINAL):
        return []
    initial_ego_state = metric_cache.ego_state
    human_trajectory = transform_trajectory(metric_cache.human_trajectory, initial_ego_state)
    human_states = get_trajectory_as_array(human_trajectory, simulator.proposal_sampling, initial_ego_state.time_point)
    simulated = simulator.simulate_proposals(human_states[None, ...], initial_ego_state)
    tracks = traffic_agents_policy.simulate_environment(simulated[0], metric_cache)
    result = scorer.score_proposals(
        simulated,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.map_parameters,
        tracks,
    )[0]
    return [name for name in {**MULTI_METRICS, **WEIGHTED_METRICS} if result[name].iloc[0] == 0]


def score_codebook(
    metric_cache: MetricCache,
    poses: npt.NDArray,
    trajectory_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_agents_policy: LogReplayTrafficAgents,
    chunk_size: int = 1024,
) -> Dict[str, npt.NDArray]:
    """
    PDM sub-scores and per-frame score of every codebook entry in one scene.
    :param metric_cache: the scene's metric cache
    :param poses: [K, T, 3] codebook in the ego frame
    :param trajectory_sampling: sampling of the codebook poses
    :param chunk_size: entries simulated and scored per batch (memory grows with it)
    :return: one [K] array per metric in MULTI_METRICS / WEIGHTED_METRICS, plus "score" and "progress_raw"
    """
    assert isinstance(traffic_agents_policy, LogReplayTrafficAgents), "only non-reactive traffic can be shared"
    proposal_sampling = simulator.proposal_sampling
    assert proposal_sampling == scorer.proposal_sampling, "simulator and scorer proposal sampling must match"

    initial_ego_state = metric_cache.ego_state
    reference = get_trajectory_as_array(metric_cache.trajectory, proposal_sampling, initial_ego_state.time_point)
    entries = codebook_to_state_arrays(poses, initial_ego_state, trajectory_sampling, proposal_sampling)

    results: Dict[str, List[npt.NDArray]] = {name: [] for name in [*MULTI_METRICS, *WEIGHTED_METRICS, "score"]}
    results["progress_raw"] = []
    tracks = None
    for start in range(0, len(entries), chunk_size):
        states = np.concatenate([reference[None], entries[start : start + chunk_size]], axis=0)
        simulated = simulator.simulate_proposals(states, initial_ego_state)
        if tracks is None:  # log replay ignores the ego states
            tracks = traffic_agents_policy.simulate_environment(simulated[0], metric_cache)
        scorer.score_proposals(
            simulated,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            metric_cache.map_parameters,
            tracks,
            metric_cache.past_human_trajectory,
        )
        # Raw arrays from the scorer; its batch-normalized progress row is replaced by pairwise_scores
        multi, weighted, progress_raw = scorer._multi_metrics, scorer._weighted_metrics, scorer._progress_raw
        pairwise = pairwise_scores(multi, weighted, progress_raw, scorer._config)
        for name, index in MULTI_METRICS.items():
            results[name].append(multi[index, 1:].copy())
        for name, index in WEIGHTED_METRICS.items():
            results[name].append(pairwise["ego_progress"] if name == "ego_progress" else weighted[index, 1:].copy())
        results["score"].append(pairwise["score"])
        results["progress_raw"].append(progress_raw[1:].copy())

    scores = {name: np.concatenate(chunks) for name, chunks in results.items()}
    for name in human_zero_metrics(metric_cache, simulator, scorer, traffic_agents_policy):
        scores[name] = np.ones_like(scores[name])
    return scores
