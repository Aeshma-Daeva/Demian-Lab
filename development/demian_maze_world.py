"""Small deterministic partially observed maze; no controller or learning state."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal

import numpy as np

MazeAction = Literal["north", "south", "east", "west", "wait"]
ACTIONS = ("north", "south", "east", "west", "wait")
DELTAS = {"north": (0,-1), "south": (0,1), "east": (1,0), "west": (-1,0), "wait": (0,0)}


@dataclass(frozen=True)
class MazeLayout:
    orientation: int
    floor: tuple[tuple[int,int], ...]
    start: tuple[int,int]
    goal: tuple[int,int]
    door: tuple[int,int]


@dataclass(frozen=True)
class MazeConfig:
    layout: MazeLayout
    door_tape: tuple[bool, ...]
    budget: int = 96

    def __post_init__(self):
        floor = self.layout.floor
        if type(self.budget) is not int or self.budget < 1 or len(self.door_tape) < self.budget+1:
            raise ValueError("invalid budget or short door tape")
        if any(type(v) is not bool for v in self.door_tape):
            raise ValueError("door tape must contain booleans")
        if len(floor) != len(set(floor)) or any(not (0 <= x < 7 and 0 <= y < 7) for x,y in floor):
            raise ValueError("invalid floor coordinates")
        if len({self.layout.start,self.layout.goal,self.layout.door}) != 3:
            raise ValueError("start, goal and door must differ")
        if not {self.layout.start,self.layout.goal,self.layout.door}.issubset(set(floor)):
            raise ValueError("special cell outside floor")
        shortest_path(self.layout, self.layout.start, True)
        shortest_path(self.layout, self.layout.start, False)


@dataclass(frozen=True)
class MazeAcknowledgement:
    action: str
    moved: bool
    blocked: bool


@dataclass(frozen=True)
class MazeObservation:
    occupancy: tuple[int, ...]  # row-major3x3: wall0/floor1/open door2/closed door3
    door_visible: int  # -1 unseen,0closed,1open
    goal_bearing: tuple[int,int]
    energy: int
    acknowledgement: MazeAcknowledgement | None


@dataclass(frozen=True)
class MazeSnapshot:
    configuration_hash: str
    position: tuple[int,int]
    tick: int
    acknowledgement: MazeAcknowledgement | None


def shortest_path(layout, position, door_open):
    queue = deque([(position, [])]); seen = {position}
    while queue:
        point, path = queue.popleft()
        if point == layout.goal:
            return path
        for action in ACTIONS[:4]:
            dx,dy = DELTAS[action]; next_point = (point[0]+dx,point[1]+dy)
            if next_point in seen or next_point not in layout.floor or (next_point == layout.door and not door_open):
                continue
            seen.add(next_point); queue.append((next_point,path+[action]))
    raise ValueError("maze has no route to goal")


def make_config(orientation, condition, seed, budget=96):
    if type(orientation) is not int or not 0 <= orientation < 8:
        raise ValueError("orientation must be0..7")
    def transform(point):
        x,y = point
        if orientation >= 4: x = 6-x
        for _ in range(orientation % 4): x,y = 6-y,x
        return x,y
    floor = {(x,3) for x in range(1,6)} | {(1,4),(1,5),(2,5),(3,5),(4,5),(5,5),(5,4)}
    layout = MazeLayout(orientation,tuple(sorted(map(transform,floor))),transform((1,3)),transform((5,3)),transform((3,3)))
    rng = np.random.default_rng(seed)
    period = int(rng.choice([4,6,8])); phase = int(rng.integers(period))
    tape = [bool((t+phase)%period < period//2) for t in range(budget+1)]
    if condition in ("open","closed"):
        tape = [condition == "open"]*(budget+1)
    elif condition == "switch":
        # Decision-window schedule change, frozen before scientific collection.
        initial = not tape[1]
        for t in range(2,budget+1): tape[t] = initial if (t-2)%8 < 4 else not initial
    elif condition != "periodic":
        raise ValueError("unknown condition")
    return MazeConfig(layout,tuple(tape),budget)


class MazeWorld:
    def __init__(self, config):
        self.reset(config)

    def reset(self, config):
        self.config = config
        self.configuration_hash = hashlib.sha256(json.dumps(asdict(config),sort_keys=True).encode()).hexdigest()
        self.position = config.layout.start
        self.tick = 0
        self.acknowledgement = None

    @property
    def energy(self): return self.config.budget-self.tick

    @property
    def success(self): return self.position == self.config.layout.goal

    @property
    def done(self): return self.success or self.energy == 0

    def observe(self):
        x,y = self.position; layout = self.config.layout
        opened = self.config.door_tape[self.tick]
        cells = []
        visible = -1
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                p = (x+dx,y+dy)
                code = int(p in layout.floor)
                if p == layout.door:
                    code = 2 if opened else 3; visible = int(opened)
                cells.append(code)
        bearing = tuple(int(np.sign(g-p)) for g,p in zip(layout.goal,self.position))
        return MazeObservation(tuple(cells),visible,bearing,self.energy,self.acknowledgement)

    def step(self, action):
        if action not in ACTIONS:
            raise ValueError("invalid maze action")
        if self.done:
            raise RuntimeError("maze episode completed")
        dx,dy = DELTAS[action]; target = (self.position[0]+dx,self.position[1]+dy)
        blocked = action != "wait" and (target not in self.config.layout.floor or
                   (target == self.config.layout.door and not self.config.door_tape[self.tick]))
        moved = action != "wait" and not blocked
        if moved: self.position = target
        self.tick += 1
        self.acknowledgement = MazeAcknowledgement(action,moved,blocked)
        return self.acknowledgement

    def snapshot(self):
        return MazeSnapshot(self.configuration_hash,self.position,self.tick,self.acknowledgement)

    def restore(self, snapshot):
        ack=snapshot.acknowledgement
        if ((snapshot.tick==0 and ack is not None) or
            (snapshot.tick!=0 and (not isinstance(ack,MazeAcknowledgement) or ack.action not in ACTIONS or
             type(ack.moved) is not bool or type(ack.blocked) is not bool or (ack.moved and ack.blocked) or
             (ack.action=="wait" and (ack.moved or ack.blocked))))):
            raise ValueError("invalid acknowledgement snapshot")
        if (snapshot.configuration_hash != self.configuration_hash or type(snapshot.tick) is not int or
            not 0 <= snapshot.tick <= self.config.budget or snapshot.position not in self.config.layout.floor):
            raise ValueError("invalid or incompatible maze snapshot")
        self.position,self.tick,self.acknowledgement = snapshot.position,snapshot.tick,snapshot.acknowledgement


class MazeTeacher:
    """Static map plus public action/observation history; no dynamic-world access."""
    def __init__(self, layout):
        self.layout = layout
        self.position = layout.start
        self.door_belief = True
        self.last_energy = None
        self.last_visible = -1
        self.waited_for_closed = False

    def observe(self, observation):
        if self.last_energy == observation.energy: return
        if self.last_energy is not None:
            if observation.energy != self.last_energy-1 or observation.acknowledgement is None:
                raise ValueError("teacher observation history is discontinuous")
            if observation.acknowledgement.moved:
                dx,dy = DELTAS[observation.acknowledgement.action]
                self.position = self.position[0]+dx,self.position[1]+dy
            if observation.acknowledgement.action=="wait" and self.last_visible==0:
                self.waited_for_closed=True
        if observation.door_visible != -1:
            self.door_belief = bool(observation.door_visible)
            if observation.door_visible==1:self.waited_for_closed=False
        self.last_visible=observation.door_visible
        self.last_energy = observation.energy

    def action(self):
        open_path=shortest_path(self.layout,self.position,True)
        if self.last_visible==0 and not self.waited_for_closed and open_path:
            dx,dy=DELTAS[open_path[0]]
            if (self.position[0]+dx,self.position[1]+dy)==self.layout.door:return "wait"
        path = shortest_path(self.layout,self.position,self.door_belief)
        return path[0] if path else "wait"
