from development.run_demian_uncertainty_discovery import run_protocol


def test_discovery_runner_keeps_model_seeds_and_public_episodes_separate() -> None:
    payload = run_protocol({
        "protocol_id": "test-uncertainty", "hidden_size": 16, "model_seeds": [94, 95],
        "episode_seeds": [1, 2], "delay_steps": [2], "post_resolution_steps": 1,
    })
    assert len(payload["cells"]) == 2 * 2 * 4
    assert payload["integrity"]["all_replay_exact"] is True
    assert payload["integrity"]["all_finite"] is True
