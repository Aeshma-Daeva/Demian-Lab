import torch

from development.demian_uncertainty_probe import encode_evidence, run_uncertainty_episode
from development.demian_uncertainty_world import AmbiguousEvidenceWorld


def test_encoder_is_stateless_and_has_disjoint_slots() -> None:
    observation = AmbiguousEvidenceWorld(hidden_target=1, evidence=[(1, 0.75)], delay_steps=2).observe()
    encoded = encode_evidence(observation, 16)
    assert encoded.shape == (1, 16)
    assert torch.count_nonzero(encoded) == 4


def test_episode_replay_is_target_leak_free_and_exact() -> None:
    public = AmbiguousEvidenceWorld(hidden_target=1, evidence=[(1, 0.75), (0, 0.75)], delay_steps=2).frames()
    left = run_uncertainty_episode(model_seed=94, hidden_size=16, public_frames=public)
    right = run_uncertainty_episode(model_seed=94, hidden_size=16, public_frames=public)
    assert left["state_hashes"] == right["state_hashes"]
    assert left["replay_exact"] is True


def test_episode_can_capture_pre_resolution_delay_state() -> None:
    public = AmbiguousEvidenceWorld(hidden_target=1, evidence=[(1, 0.75)], delay_steps=2).frames()
    trace = run_uncertainty_episode(
        model_seed=94,
        hidden_size=16,
        public_frames=public,
        checkpoint_phases=("delay",),
    )

    assert set(trace["checkpoints"]) == {"delay:1", "delay:2"}
