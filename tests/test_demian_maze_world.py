from dataclasses import replace

import pytest


def test_maze_motion_cost_and_exact_snapshot_replay():
    from development.demian_maze_world import MazeWorld, make_config
    world = MazeWorld(make_config(0, "closed", 11))
    assert world.position == (1, 3)
    world.step("east")
    assert world.observe().door_visible == 0
    snapshot = world.snapshot()
    first = world.step("east")
    assert first.blocked and world.energy == 94
    world.restore(snapshot)
    assert world.step("east") == first
    assert world.energy == 94


def test_hidden_door_and_future_tape_cannot_change_current_observation():
    from development.demian_maze_world import MazeWorld, make_config
    a, b = MazeWorld(make_config(0, "open", 11)), MazeWorld(make_config(0, "closed", 11))
    assert a.observe() == b.observe()
    a.step("east"); b.step("east")
    assert a.observe().door_visible == 1 and b.observe().door_visible == 0
    a.step("west"); b.step("west")
    assert a.observe() == b.observe()


def test_same_observation_can_require_different_history_conditioned_actions():
    from development.demian_maze_world import MazeWorld, MazeTeacher, make_config
    worlds = [MazeWorld(make_config(0, c, 11)) for c in ("open", "closed")]
    teachers = [MazeTeacher(w.config.layout) for w in worlds]
    for world, teacher in zip(worlds, teachers):
        teacher.observe(world.observe())
        world.step("east"); teacher.observe(world.observe())
        world.step("west"); teacher.observe(world.observe())
    assert worlds[0].observe() == worlds[1].observe()
    assert [t.action() for t in teachers] == ["east", "south"]


def test_closed_door_blocks_entry_but_allows_occupant_exit():
    from development.demian_maze_world import MazeWorld, make_config
    config = make_config(0, "open", 11)
    tape = (True, True)+tuple(False for _ in range(95))
    world = MazeWorld(replace(config, door_tape=tape))
    world.step("east"); world.step("east")
    assert world.position == config.layout.door
    assert world.observe().door_visible == 0
    assert world.step("east").moved


def test_invalid_action_and_snapshot_do_not_mutate_world():
    from development.demian_maze_world import MazeWorld, make_config
    world = MazeWorld(make_config(0,"open",1)); before = world.snapshot()
    with pytest.raises(ValueError): world.step("teleport")
    assert world.snapshot() == before
    with pytest.raises(ValueError): world.restore(replace(before,tick=99))
    assert world.snapshot() == before


def test_all_orientations_solvable_with_strictly_longer_detour():
    from development.demian_maze_world import make_config, shortest_path
    for orientation in range(8):
        layout = make_config(orientation,"open",1).layout
        assert len(shortest_path(layout, layout.start, True)) == 4
        assert len(shortest_path(layout, layout.start, False)) == 8


def test_energy_exhaustion_and_goal_are_terminal():
    from development.demian_maze_world import MazeWorld, make_config
    world = MazeWorld(make_config(0,"open",1))
    for _ in range(4): world.step("east")
    assert world.success and world.done and world.energy == 92
    with pytest.raises(RuntimeError): world.step("wait")
    world = MazeWorld(make_config(0,"closed",1))
    for _ in range(96): world.step("wait")
    assert world.done and not world.success and world.energy == 0


def test_teacher_has_no_future_or_hidden_door_access():
    from development.demian_maze_world import MazeWorld, MazeTeacher, make_config
    worlds=[MazeWorld(make_config(0,c,1)) for c in ("open","closed")]
    teachers=[MazeTeacher(w.config.layout) for w in worlds]
    for w,t in zip(worlds,teachers):t.observe(w.observe())
    assert [t.action() for t in teachers] == ["east","east"]


def test_malformed_acknowledgement_snapshot_is_rejected_atomically():
    from development.demian_maze_world import MazeWorld, make_config
    world=MazeWorld(make_config(0,"open",1));before=world.snapshot()
    with pytest.raises(ValueError):world.restore(replace(before,tick=1,acknowledgement=None))
    assert world.snapshot()==before


def test_teacher_waits_once_at_visible_closed_entry_then_takes_detour():
    from development.demian_maze_world import MazeWorld, MazeTeacher, make_config
    world=MazeWorld(make_config(0,"closed",1));teacher=MazeTeacher(world.config.layout)
    chosen=[]
    while not world.done:
        teacher.observe(world.observe());action=teacher.action();chosen.append(action);world.step(action)
    assert chosen[:3]==["east","wait","west"]
    assert chosen.count("wait")==1 and world.success
