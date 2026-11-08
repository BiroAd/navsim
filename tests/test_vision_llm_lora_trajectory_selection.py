"""Output selection of compute_trajectory (config.trajectory_selection): policy argmax vs best predicted total score."""
import pytest
import torch

from navsim.agents.vision_llm_lora.vision_llm_lora_agent import select_trajectory
from navsim.agents.vision_llm_lora.vision_llm_lora_loss import (
    PDM_SCORE_KEYS,
    V1_MULTIPLICATIVE_PDM_SCORE_KEYS,
    V1_WEIGHTED_PDM_SCORE_WEIGHTS,
    aggregate_pdm_scores,
)

K = 5


def _inputs():
    codebook = torch.arange(K * 8 * 3, dtype=torch.float32).reshape(K, 8, 3)
    logits = torch.full((2, len(PDM_SCORE_KEYS), K), -4.0)
    logits[0, :, 3] = 4.0  # scene 0: entry 3 scores best on every head
    logits[1, :, 1] = 4.0  # scene 1: entry 1
    predictions = {"trajectory": codebook[[0, 0]], "pdm_score_logits": logits}
    return predictions, codebook


def test_argmax_returns_the_model_output():
    predictions, codebook = _inputs()
    assert select_trajectory(predictions, "argmax", codebook) is predictions["trajectory"]


def test_score_heads_pick_the_entry_with_the_best_predicted_total():
    predictions, codebook = _inputs()
    selected = select_trajectory(predictions, "score_heads", codebook)
    assert torch.equal(selected, codebook[[3, 1]])


def test_score_heads_use_the_aggregated_total_not_a_single_head():
    # Entry 0 is best on 7 heads but fails drivable area (multiplicative); entry 2 is good everywhere.
    codebook = torch.randn(K, 8, 3)
    logits = torch.full((1, len(PDM_SCORE_KEYS), K), -4.0)
    logits[0, :, 0] = 6.0
    logits[0, PDM_SCORE_KEYS.index("drivable_area_compliance"), 0] = -6.0
    logits[0, :, 2] = 3.0
    total = aggregate_pdm_scores(logits.sigmoid())
    assert total.argmax(dim=-1).item() == 2
    selected = select_trajectory({"trajectory": codebook[[0]], "pdm_score_logits": logits}, "score_heads", codebook)
    assert torch.equal(selected, codebook[[2]])


def test_unknown_selection_raises():
    predictions, codebook = _inputs()
    with pytest.raises(ValueError):
        select_trajectory(predictions, "best", codebook)


def test_score_heads_v1_matches_the_v1_formula():
    sub = torch.rand(3, len(PDM_SCORE_KEYS), K)
    i = {k: n for n, k in enumerate(PDM_SCORE_KEYS)}
    expected = (
        sub[:, i["no_at_fault_collisions"]]
        * sub[:, i["drivable_area_compliance"]]
        * (
            5 * sub[:, i["ego_progress"]]
            + 5 * sub[:, i["time_to_collision_within_bound"]]
            + 2 * sub[:, i["history_comfort"]]
        )
        / 12
    )
    total = aggregate_pdm_scores(sub, V1_MULTIPLICATIVE_PDM_SCORE_KEYS, V1_WEIGHTED_PDM_SCORE_WEIGHTS)
    assert torch.allclose(total, expected)


def test_score_heads_v1_ignores_lane_keeping_and_the_extra_multipliers():
    # Entry 0 fails lane keeping and traffic light (v2 only), entry 1 is slightly worse on progress but passes both:
    # v2 selection picks 1, v1 selection picks 0.
    codebook = torch.randn(K, 8, 3)
    logits = torch.full((1, len(PDM_SCORE_KEYS), K), -6.0)
    logits[0, :, 0] = 6.0
    logits[0, PDM_SCORE_KEYS.index("lane_keeping"), 0] = -6.0
    logits[0, PDM_SCORE_KEYS.index("traffic_light_compliance"), 0] = -6.0
    logits[0, :, 1] = 6.0
    logits[0, PDM_SCORE_KEYS.index("ego_progress"), 1] = 1.0
    predictions = {"trajectory": codebook[[2]], "pdm_score_logits": logits}
    assert torch.equal(select_trajectory(predictions, "score_heads", codebook), codebook[[1]])
    assert torch.equal(select_trajectory(predictions, "score_heads_v1", codebook), codebook[[0]])


def _policy_and_heads(seed=0, batch=4, k=64):
    gen = torch.Generator().manual_seed(seed)
    codebook = torch.randn(k, 8, 3, generator=gen)
    policy_logits = torch.randn(batch, k, generator=gen) * 3
    score_logits = torch.randn(batch, len(PDM_SCORE_KEYS), k, generator=gen) * 3
    predictions = {
        "trajectory": codebook[policy_logits.argmax(-1)],
        "trajectory_logits": policy_logits,
        "pdm_score_logits": score_logits,
    }
    return predictions, codebook


def test_shortlist_of_one_is_the_heads_choice_and_of_all_is_argmax():
    predictions, codebook = _policy_and_heads()
    k = codebook.shape[0]
    heads = select_trajectory(predictions, "score_heads", codebook)
    assert torch.equal(select_trajectory(predictions, "shortlist", codebook, shortlist_size=1), heads)
    selected = select_trajectory(predictions, "shortlist", codebook, shortlist_size=k)
    assert torch.equal(selected, predictions["trajectory"])


def test_shortlist_picks_the_most_likely_of_the_best_predicted_entries():
    predictions, codebook = _policy_and_heads(seed=1)
    total = aggregate_pdm_scores(predictions["pdm_score_logits"].sigmoid())
    short = total.topk(8, dim=-1).indices
    expected = short.gather(1, predictions["trajectory_logits"].gather(1, short).argmax(-1, keepdim=True))[:, 0]
    selected = select_trajectory(predictions, "shortlist", codebook, shortlist_size=8)
    assert torch.equal(selected, codebook[expected])


def test_mixture_interpolates_between_argmax_and_the_heads():
    predictions, codebook = _policy_and_heads(seed=2)
    selected = select_trajectory(predictions, "mixture", codebook, mixture_lambda=0.0)
    assert torch.equal(selected, predictions["trajectory"])
    heads = select_trajectory(predictions, "score_heads", codebook)
    assert torch.equal(select_trajectory(predictions, "mixture", codebook, mixture_lambda=1e6), heads)


def test_v1_variants_rank_with_the_v1_formula():
    predictions, codebook = _policy_and_heads(seed=3)
    heads_v1 = select_trajectory(predictions, "score_heads_v1", codebook)
    assert torch.equal(select_trajectory(predictions, "shortlist_v1", codebook, shortlist_size=1), heads_v1)
    assert torch.equal(select_trajectory(predictions, "mixture_v1", codebook, mixture_lambda=1e6), heads_v1)
