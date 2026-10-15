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


def test_route_capture_is_complete_and_does_not_change_the_episode() -> None:
    public = AmbiguousEvidenceWorld(hidden_target=1, evidence=[(1, 0.75)], delay_steps=2).frames()
    plain = run_uncertainty_episode(model_seed=94, hidden_size=16, public_frames=public)
    traced = run_uncertainty_episode(
        model_seed=94,
        hidden_size=16,
        public_frames=public,
        capture_route_tensors=True,
    )

    assert traced["state_hashes"] == plain["state_hashes"]
    assert len(traced["route_tensors"]) == len(public)
    assert "carrier_to_slow" in traced["route_tensors"][0]
    assert traced["route_tensors"][0]["carrier_to_slow"].shape == (1, 16)


def test_batched_public_episode_matches_each_single_history() -> None:
    """The relation screen batches histories only for speed, never semantics."""
    from development.demian_v1_gate_state import DemianV1GateState
    from development.demian_uncertainty_probe import encode_evidence

    histories = [
        AmbiguousEvidenceWorld(hidden_target=0, evidence=[(0, .95), (1, .5)], delay_steps=2).frames(),
        AmbiguousEvidenceWorld(hidden_target=0, evidence=[(1, .95), (0, .5)], delay_steps=2).frames(),
    ]
    torch.manual_seed(94)
    model = DemianV1GateState(16, trace_routes=True)
    state = tuple(value.repeat(2, 1) for value in model.initial_state(1, torch.device("cpu")))
    for tick in range(len(histories[0])):
        inputs = torch.cat([encode_evidence(frames[tick], 16) for frames in histories])
        state = model.step((state[0] + inputs, *state[1:]))
    for lane, frames in enumerate(histories):
        torch.manual_seed(94)
        one = DemianV1GateState(16, trace_routes=True)
        single_state = one.initial_state(1, torch.device("cpu"))
        for observation in frames:
            single_state = one.step((single_state[0] + encode_evidence(observation, 16), *single_state[1:]))
        assert torch.allclose(state[0][lane], single_state[0][0])
        assert torch.allclose(model.route_trace().values["carrier_to_slow"][lane], one.route_trace().values["carrier_to_slow"][0])
