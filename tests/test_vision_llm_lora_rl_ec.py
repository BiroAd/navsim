"""EC in the RL-3 reward (vision_llm_lora_rl.ec_reward, rl_loss_terms with ec)."""
import torch

from navsim.agents.vision_llm_lora import vision_llm_lora_rl as rl


def test_ec_reward_is_official_epdms():
    # one entry: M = 0.5, EP 0.8, TTC 1, LK 1, HC 1 -> r = 0.5 * (4 + 5 + 2 + 2) / 14
    r = torch.tensor([[0.5 * 13 / 14, 0.5 * 13 / 14, 0.0]])
    m = torch.tensor([[0.5, 0.5, 0.0]])
    ec = torch.tensor([[1.0, 0.0, 1.0]])
    out = rl.ec_reward(r, m, ec, 2.0)
    expected = torch.tensor([[0.5 * (13 + 2) / 16, 0.5 * 13 / 16, 0.0]])
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(rl.ec_reward(r, m, ec, 0.0), r)  # weight 0: unchanged


def test_rl_loss_terms_with_ec(monkeypatch):
    K = 6
    head = rl.PolicyHead(torch.randn(K, 4), in_dim=3)
    monkeypatch.setattr(rl, "reference_head", lambda path, device: head)
    targets = {
        "rl_reward": torch.rand(2, K),
        "rl_multiplier": torch.ones(2, K),
        "rl_reference_fused": torch.randn(2, 3),
        "rl_shortlist": torch.arange(K).repeat(2, 1),
    }
    logits = torch.randn(2, K, requires_grad=True)
    ec = torch.zeros(2, K)
    ec[:, 0] = 1.0
    loss_ec, extra = rl.rl_loss_terms(logits, targets, 0.01, "unused", shortlist_sizes=(4,), ec=ec, ec_weight=2.0)
    loss_plain, plain = rl.rl_loss_terms(logits, targets, 0.01, "unused", shortlist_sizes=(4,))
    assert {"rl/ec_argmax", "rl/ec_expected", "rl/ec_any", "rl/argmax_no_ec"} <= set(extra)
    assert "rl/ec_argmax" not in plain
    torch.testing.assert_close(extra["rl/argmax_no_ec"], plain["rl/argmax"])
    loss_ec.backward()
    # entry 0 gains reward from EC, so its gradient pushes its logit up more than without EC
    grad_ec = logits.grad.clone()
    logits.grad = None
    loss_plain.backward()
    assert (grad_ec[:, 0] < logits.grad[:, 0]).all()


def test_agent_ec_against_previous_plan(tmp_path):
    """The previous plan comes from a no-grad, eval-mode forward on the past frame's inputs; EC is (B, K)."""
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    from navsim.agents.vision_llm_lora.ec_torch import precompute_tracking
    from navsim.agents.vision_llm_lora.vision_llm_lora_agent import VisionLLMLoRAAgent

    codebook = np.load(Path(__file__).resolve().parents[1] / "navsim/agents/vision_llm_lora/codebook/16384_8step.npy")
    sub = codebook[:: 16384 // 32].astype(np.float32)  # 32 entries
    np.savez(tmp_path / "tracking.npz", **precompute_tracking(sub))
    seen = {}

    class FakeModel(torch.nn.Module):
        def forward(self, features, teacher_forcing_seq, ratio):
            seen.update(training=self.training, grad=torch.is_grad_enabled(), camera=features["camera_feature"],
                        velocity=features["velocity"], command=features["driving_command"])
            logits = torch.zeros(features["velocity"].shape[0], len(sub))
            logits[:, 3] = 1.0
            return {"trajectory_logits": logits}

    stub = SimpleNamespace(
        _vlm_model=FakeModel().train(),
        _config=SimpleNamespace(image_normalization="qwen", rl_ec_tracking_path=str(tmp_path / "tracking.npz")),
        _ec=None,
    )
    B = 2
    features = {
        "camera_feature": torch.zeros(B, 13, 3, 8, 8),
        "camera_history_t1": torch.full((B, 13, 3, 8, 8), 255, dtype=torch.uint8),
        "velocity": torch.tensor([[5.0, 0.0], [10.0, 0.1]]),
        "acceleration": torch.tensor([[0.2, 0.0], [-0.3, 0.0]]),
        "driving_command": torch.tensor([1, 1]),
        "ego_history_velocity": torch.tensor([[[4.9, 0.0]] * 3, [[10.1, 0.0]] * 3]),
        "ego_history_acceleration": torch.tensor([[[0.2, 0.0]] * 3, [[-0.2, 0.0]] * 3]),
        "ego_history_command": torch.tensor([[2, 1, 1], [0, 1, 1]]),
    }
    ec = VisionLLMLoRAAgent._ec_against_previous_plan(stub, features)
    assert ec.shape == (B, len(sub)) and ec.dtype == torch.float32
    assert seen["training"] is False and seen["grad"] is False and stub._vlm_model.training
    torch.testing.assert_close(seen["camera"], torch.ones(B, 13, 3, 8, 8))  # 255 -> (1 - 0.5) / 0.5
    torch.testing.assert_close(seen["velocity"], features["ego_history_velocity"][:, 0])
    assert seen["command"].tolist() == [2, 0]
    direct = stub._ec.against_previous(features["velocity"], features["acceleration"],
                                       features["ego_history_velocity"][:, 0], features["ego_history_acceleration"][:, 0],
                                       torch.tensor([3, 3]))  # previous plan = the fake model's argmax
    torch.testing.assert_close(ec, direct.float())
    assert set(ec.unique().tolist()) <= {0.0, 1.0}

    # only the needed entries are simulated; the others are 0, the needed ones unchanged
    needed = torch.zeros(B, len(sub), dtype=torch.bool)
    needed[0, ::3] = True
    needed[1, 5:9] = True
    stub._vlm_model.train()
    part = VisionLLMLoRAAgent._ec_against_previous_plan(stub, features, needed)
    torch.testing.assert_close(part, ec * needed)


def _agent_stub(tmp_path, logits_fn, K=32):
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    from navsim.agents.vision_llm_lora.ec_torch import precompute_tracking

    codebook = np.load(Path(__file__).resolve().parents[1] / "navsim/agents/vision_llm_lora/codebook/16384_8step.npy")
    np.savez(tmp_path / "tracking.npz", **precompute_tracking(codebook[:: 16384 // K].astype(np.float32)))

    class FakeModel(torch.nn.Module):
        def forward(self, features, teacher_forcing_seq, ratio):
            return {"trajectory_logits": logits_fn(features["velocity"].shape[0])}

    config = SimpleNamespace(image_normalization="qwen", rl_ec_tracking_path=str(tmp_path / "tracking.npz"),
                             rl_loss_weight=1.0, rl_ec_weight=2.0, history_cache_path="x",
                             rl_ec_previous_plan="argmax", rl_ec_previous_shortlist_size=3)
    return SimpleNamespace(_vlm_model=FakeModel().train(), _config=config, _ec=None)


def _past_features(B=2):
    return {
        "camera_feature": torch.zeros(B, 13, 3, 8, 8),
        "camera_history_t1": torch.zeros((B, 13, 3, 8, 8), dtype=torch.uint8),
        "velocity": torch.tensor([[5.0, 0.0], [10.0, 0.1]]),
        "acceleration": torch.tensor([[0.2, 0.0], [-0.3, 0.0]]),
        "driving_command": torch.tensor([1, 1]),
        "ego_history_velocity": torch.tensor([[[4.9, 0.0]] * 3, [[10.1, 0.0]] * 3]),
        "ego_history_acceleration": torch.tensor([[[0.2, 0.0]] * 3, [[-0.2, 0.0]] * 3]),
        "ego_history_command": torch.tensor([[1, 1, 1], [1, 1, 1]]),
    }


def test_previous_plan_from_shortlist(tmp_path):
    """rl_ec_previous_plan = shortlist: the most likely entry inside the scorer's shortlist (ties: shortlist order)."""
    from navsim.agents.vision_llm_lora.vision_llm_lora_agent import VisionLLMLoRAAgent, select_trajectory

    def logits_fn(B):
        logits = torch.zeros(B, 32)
        logits[0, 3], logits[0, 7], logits[1, 20] = 2.0, 1.0, 5.0  # scene 1: argmax 20 is outside its shortlist
        return logits

    stub = _agent_stub(tmp_path, logits_fn)
    features = _past_features()
    shortlist = torch.tensor([[5, 7, 3], [9, 10, 11]], dtype=torch.int16)
    ec = VisionLLMLoRAAgent._ec_against_previous_plan(stub, features, None, shortlist)
    previous = torch.tensor([3, 9])  # scene 0: 3 (best logit in the list); scene 1: all tied -> first, 9
    v, a = features["ego_history_velocity"][:, 0], features["ego_history_acceleration"][:, 0]
    direct = stub._ec.against_previous(features["velocity"], features["acceleration"], v, a, previous)
    torch.testing.assert_close(ec, direct.float())
    argmax_ec = VisionLLMLoRAAgent._ec_against_previous_plan(stub, features)
    torch.testing.assert_close(argmax_ec, stub._ec.against_previous(
        features["velocity"], features["acceleration"], v, a, torch.tensor([3, 20])).float())

    # the same pick as the evaluation's "shortlist" selection (scorer totals ranking exactly this shortlist)
    totals = torch.zeros(2, 8, 32)  # sub-score logits; make the shortlist the top 3 of the aggregated total
    for b in range(2):
        for rank, entry in enumerate(shortlist[b].long().tolist()):
            totals[b, :, entry] = 10.0 - rank
    codebook = torch.arange(32, dtype=torch.float32).view(32, 1, 1).expand(32, 8, 3)
    picked = select_trajectory({"pdm_score_logits": totals, "trajectory_logits": logits_fn(2)}, "shortlist", codebook,
                               shortlist_size=3)
    assert picked[:, 0, 0].long().tolist() == previous.tolist()


def test_compute_loss_passes_previous_shortlist(tmp_path, monkeypatch):
    from types import MethodType

    from navsim.agents.vision_llm_lora import vision_llm_lora_agent as agent_module

    seen = {}
    monkeypatch.setattr(agent_module, "vision_llm_lora_loss", lambda targets, predictions, config: predictions["rl_ec"])
    stub = _agent_stub(tmp_path, lambda B: torch.zeros(B, 32))
    stub._ec_in_reward = MethodType(agent_module.VisionLLMLoRAAgent._ec_in_reward, stub)

    def fake_ec(features, needed, previous_shortlist):
        seen.update(needed=needed, previous_shortlist=previous_shortlist)
        return torch.zeros(needed.shape)

    stub._ec_against_previous_plan = fake_ec
    targets = {"rl_multiplier": torch.tensor([[1.0, 0.0], [0.5, 1.0]]),
               "rl_previous_shortlist": torch.arange(10, dtype=torch.int16).repeat(2, 1)}
    agent_module.VisionLLMLoRAAgent.compute_loss(stub, {}, targets, {})
    assert seen["previous_shortlist"] is None  # default argmax: the shortlist is not used
    stub._config.rl_ec_previous_plan = "shortlist"
    agent_module.VisionLLMLoRAAgent.compute_loss(stub, {}, targets, {})
    assert seen["previous_shortlist"].tolist() == [[0, 1, 2]] * 2  # cut to rl_ec_previous_shortlist_size
    assert seen["needed"].tolist() == [[True, False], [True, True]]
    stub._config.rl_ec_previous_plan = "typo"
    import pytest

    with pytest.raises(ValueError):
        agent_module.VisionLLMLoRAAgent.compute_loss(stub, {}, targets, {})


def test_cache_rl_reference_previous_shortlist(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    import numpy as np

    from navsim.planning.training.dataset import load_feature_target_from_pickle

    path = Path(__file__).resolve().parents[1] / "scripts" / "training" / "cache_rl_reference.py"
    spec = importlib.util.spec_from_file_location("cache_rl_reference", path)
    script = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "cache_rl_reference", script)  # Pool workers pickle write_log by module name
    spec.loader.exec_module(script)
    K, tokens = 16, ["a", "b", "c"]
    features, scores, short = tmp_path / "features", tmp_path / "scores", tmp_path / "short"
    for d in (features, scores, short):
        d.mkdir()
    torch.save({"token": tokens, "log": ["log1"] * 3, "fused": torch.randn(3, 4).half(),
                "shortlist": torch.arange(8, dtype=torch.int16).repeat(3, 1)}, features / "shard_0_of_1.pt")
    sub = {k: np.ones((3, K), np.float32) for k in ("no_at_fault_collisions", "drivable_area_compliance",
                                                     "driving_direction_compliance", "traffic_light_compliance")}
    np.savez(scores / "log1.npz", tokens=np.array(tokens), score=np.full((3, K), 0.5, np.float32), **sub)
    # scorer cache: frames (0, 1, 2); "c" has none (skipped)
    shortlist = torch.stack([torch.stack([torch.arange(K).roll(f + 3 * i) for f in range(3)]) for i in range(2)])
    torch.save({"token": ["b", "a"], "log": ["log1"] * 2, "split": ["train"] * 2, "frames": [0, 1, 2],
                "shortlist": shortlist.short()}, short / "shard_0_of_1.pt")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--features-dir", str(features), "--scores-dir", str(scores), "--out-dir",
                                      str(out), "--workers", "1", "--previous-shortlists", str(short),
                                      "--previous-shortlist-size", "5"])
    script.main()
    a = load_feature_target_from_pickle(out / "log1" / "a" / "vision_llm_rl_reference.gz")
    b = load_feature_target_from_pickle(out / "log1" / "b" / "vision_llm_rl_reference.gz")
    assert not (out / "log1" / "c").exists()
    assert a["rl_previous_shortlist"].dtype == torch.int16
    assert a["rl_previous_shortlist"].tolist() == torch.arange(K).roll(1 + 3)[:5].tolist()  # token a = row 1, frame 1
    assert b["rl_previous_shortlist"].tolist() == torch.arange(K).roll(1)[:5].tolist()
    assert set(a) == {"rl_reward", "rl_multiplier", "rl_reference_fused", "rl_shortlist", "rl_previous_shortlist"}
