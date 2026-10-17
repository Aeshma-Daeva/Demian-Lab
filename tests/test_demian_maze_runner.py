from dataclasses import replace

import numpy as np
import pytest


def test_encoder_uses_only_public_observations():
    from development.demian_maze_world import MazeWorld, make_config
    from development.demian_maze_runner import encode_observation
    a,b = [MazeWorld(make_config(0,c,1)) for c in ("open","closed")]
    np.testing.assert_array_equal(encode_observation(a.observe()),encode_observation(b.observe()))
    a.step("east");b.step("east")
    assert not np.array_equal(encode_observation(a.observe()),encode_observation(b.observe()))
    assert encode_observation(a.observe()).shape == (49,)


def test_fitted_action_head_replays_and_cannot_be_refit_by_evaluation(tmp_path):
    from development.demian_maze_runner import fit_action_head, predict_action
    rng=np.random.default_rng(3); x=rng.normal(size=(12,20));labels=np.arange(12)%5
    fit=fit_action_head(x,labels); saved={k:v.copy() for k,v in fit.items()}
    evaluation=rng.normal(size=(4,20)); first=predict_action(fit,evaluation)
    predict_action(fit,evaluation+100)
    for k in fit:np.testing.assert_array_equal(fit[k],saved[k])
    np.savez_compressed(tmp_path/'fit.npz',**fit)
    with np.load(tmp_path/'fit.npz') as a:np.testing.assert_array_equal(first,predict_action(dict(a),evaluation))
    with pytest.raises(ValueError):fit_action_head(np.full((12,20),np.nan),labels)


def test_demian_world_snapshot_and_observation_replay_are_exact():
    from development.demian_maze_world import make_config
    from development.demian_maze_runner import MazeRunner, MazeEngine
    runner=MazeRunner(31,16,make_config(0,"closed",1))
    weight_hash=runner.engine.parameter_hash()
    first=runner.step("east"); snapshot=runner.snapshot()
    a=runner.step("west");runner.restore(snapshot);b=runner.step("west")
    np.testing.assert_array_equal(a.state,b.state)
    assert a.action == b.action and a.acknowledgement == b.acknowledgement
    engine=MazeEngine(31,16)
    np.testing.assert_array_equal(engine.observe(first.observation)[0],first.state)
    assert runner.engine.parameter_hash() == weight_hash
    assert runner.world.tick == 2


def test_incompatible_snapshot_and_invalid_action_fail_before_mutation():
    from development.demian_maze_world import make_config
    from development.demian_maze_runner import MazeRunner
    a=MazeRunner(31,16,make_config(0,"closed",1));b=MazeRunner(32,16,make_config(0,"closed",1))
    before=a.snapshot()
    with pytest.raises(ValueError):a.restore(b.snapshot())
    with pytest.raises(ValueError):a.step("teleport")
    np.testing.assert_array_equal(a.snapshot().model.state[0],before.model.state[0])
    assert a.world.tick == 0


def test_reset_state_features_are_independent_of_previous_frames():
    from development.demian_maze_world import MazeWorld, make_config
    from development.demian_maze_runner import MazeEngine
    engine=MazeEngine(31,16,reset_each_tick=True)
    world=MazeWorld(make_config(0,"closed",1)); observation=world.observe()
    expected=engine.observe(observation)[0]
    world.step("east");engine.observe(world.observe())
    np.testing.assert_array_equal(engine.observe(observation)[0],expected)


def test_negative_runtime_counter_and_nonfinite_fit_penalty_fail():
    from development.demian_maze_world import make_config
    from development.demian_maze_runner import MazeRunner,fit_action_head
    runner=MazeRunner(31,16,make_config(0,"open",1));before=runner.snapshot()
    invalid=replace(before,model=replace(before.model,step_index=-1))
    with pytest.raises(ValueError):runner.restore(invalid)
    assert runner.world.tick==0 and runner.engine.model._step_index==0
    with pytest.raises(ValueError):fit_action_head(np.eye(5),np.arange(5),alpha=np.nan)
