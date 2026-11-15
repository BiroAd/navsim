"""The torch EC port (ec_torch.py) against the official numpy simulator and comfort metric, on real codebook entries."""
from pathlib import Path

import numpy as np
import pytest
import torch

from navsim.agents.vision_llm_lora.ec_torch import TorchExtendedComfort, precompute_tracking

CODEBOOK = Path(__file__).resolve().parents[1] / "navsim/agents/vision_llm_lora/codebook/16384_8step.npy"


def _official(entries, codebook, velocity, acceleration, start_us=0):
    """PDMSimulator.simulate_proposals of codebook[entries] from an ego state at the origin, starting at start_us."""
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

    state = EgoState.build_from_rear_axle(
        StateSE2(0.0, 0.0, 0.0),
        tire_steering_angle=0.0,
        vehicle_parameters=get_pacifica_parameters(),
        time_point=TimePoint(start_us),
        rear_axle_velocity_2d=StateVector2D(*velocity),
        rear_axle_acceleration_2d=StateVector2D(*acceleration),
    )
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    traj_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    planned = np.stack([
        get_trajectory_as_array(transform_trajectory(Trajectory(codebook[i], traj_sampling), state), sampling,
                                state.time_point)
        for i in entries
    ])
    return PDMSimulator(sampling).simulate_proposals(planned, state)


@pytest.fixture(scope="module")
def setup():
    if not CODEBOOK.exists():
        pytest.skip("codebook not present")
    codebook = np.load(CODEBOOK).astype(np.float32)
    rng = np.random.default_rng(0)
    entries = np.sort(rng.choice(len(codebook), 96, replace=False))
    sub = codebook[entries]
    return sub, TorchExtendedComfort(precompute_tracking(sub), torch.device("cpu"))


@pytest.mark.parametrize("velocity,acceleration", [((0.0, 0.0), (0.0, 0.0)), ((0.1, 0.0), (0.3, 0.0)),
                                                   ((6.3, 0.2), (-1.1, 0.4)), ((14.0, -0.3), (0.8, -0.2))])
def test_simulation_matches_official(setup, velocity, acceleration):
    sub, ec = setup
    ref = _official(range(len(sub)), sub, velocity, acceleration)
    heading, ax, ay = ec.simulate(torch.tensor([velocity]), torch.tensor([acceleration]))
    np.testing.assert_allclose(heading[0].numpy(), ref[:, :, 2], atol=1e-6)
    np.testing.assert_allclose(ax[0].numpy(), ref[:, :, 5], atol=1e-6)
    np.testing.assert_allclose(ay[0].numpy(), ref[:, :, 6], atol=1e-6)


def test_ec_matches_official(setup):
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        ego_is_two_frame_extended_comfort,
    )

    sub, ec = setup
    cases = [((5.0, 0.1), (0.4, 0.0), (4.8, 0.0), (0.5, 0.1)), ((0.0, 0.0), (0.0, 0.0), (0.2, 0.0), (-0.4, 0.0)),
             ((11.0, 0.2), (-0.6, 0.1), (11.3, 0.1), (-0.5, 0.0))]
    total = 0
    for v, a, pv, pa in cases:
        for prev in (0, 17, 50):
            cur = _official(range(len(sub)), sub, v, a)
            prv = _official([prev], sub, pv, pa)[0]
            c = cur[:, :-5]
            expected = ego_is_two_frame_extended_comfort(c, np.broadcast_to(prv[None, 5:], c.shape), np.arange(36) * 0.1)
            got = ec.against_previous(torch.tensor([v]), torch.tensor([a]), torch.tensor([pv]), torch.tensor([pa]),
                                      torch.tensor([prev]))[0].numpy()
            np.testing.assert_array_equal(got, expected)
            total += expected.sum()
    assert 0 < total < len(cases) * 3 * len(sub)  # both outcomes occur


def test_ec_agreement_at_log_timestamps(setup):
    """
    At real log timestamps (~1.6e9 s) the official get_trajectory_as_array truncates int(time_s * 1e6), which moves
    some interpolation samples by 1 us depending on the timestamp; the port uses the start time 0. The EC decisions
    must still agree almost everywhere (they can differ only at a threshold).
    """
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        ego_is_two_frame_extended_comfort,
    )

    sub, ec = setup
    rng = np.random.default_rng(1)
    agree = n = 0
    for _ in range(6):
        v, pv = rng.uniform(0, 14), rng.uniform(0, 14)
        a, pa = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
        start = int(rng.uniform(1.6e15, 1.7e15))
        prev = int(rng.integers(len(sub)))
        cur = _official(range(len(sub)), sub, (v, 0.0), (a, 0.0), start)
        prv = _official([prev], sub, (pv, 0.0), (pa, 0.0), start - 500_000)[0]
        c = cur[:, :-5]
        expected = ego_is_two_frame_extended_comfort(c, np.broadcast_to(prv[None, 5:], c.shape), np.arange(36) * 0.1)
        got = ec.against_previous(torch.tensor([[v, 0.0]]), torch.tensor([[a, 0.0]]), torch.tensor([[pv, 0.0]]),
                                  torch.tensor([[pa, 0.0]]), torch.tensor([prev]))[0].numpy()
        agree += (got == expected).sum()
        n += len(sub)
    assert agree / n >= 0.99, agree / n
