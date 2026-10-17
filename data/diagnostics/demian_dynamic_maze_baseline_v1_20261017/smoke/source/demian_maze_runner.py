"""Frozen Demian/public-maze boundary and separately fitted external action heads."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from development.demian_maze_world import ACTIONS, MazeWorld, MazeSnapshot, MazeAcknowledgement, MazeObservation
from development.demian_v1_gate_state import DemianV1GateState, capture_v1_runtime, restore_v1_runtime, V1RuntimeSnapshot


def encode_observation(observation):
    if (len(observation.occupancy) != 9 or any(v not in (0,1,2,3) for v in observation.occupancy) or
        observation.door_visible not in (-1,0,1) or not 0 <= observation.energy <= 96 or
        any(v not in (-1,0,1) for v in observation.goal_bearing)):
        raise ValueError("invalid public observation")
    x = np.zeros(49,dtype=np.float32)
    for i,code in enumerate(observation.occupancy): x[4*i+code] = 1
    x[36+observation.door_visible+1] = 1
    x[39:41] = observation.goal_bearing
    x[41] = observation.energy/96
    ack = observation.acknowledgement
    if ack is not None:
        x[42+ACTIONS.index(ack.action)] = 1
        x[47:49] = ack.moved,ack.blocked
    return x


def fit_action_head(features, labels, alpha=1.):
    x = np.asarray(features,dtype=np.float64); labels = np.asarray(labels)
    if x.ndim != 2 or not len(x) or labels.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError("invalid training features/labels")
    if labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= 5)) or alpha <= 0:
        raise ValueError("invalid action labels or ridge alpha")
    mean,scale = x.mean(0),x.std(0)
    scale = np.where(scale > 1e-12,scale,1.)
    z = (x-mean)/scale; y = np.eye(5)[labels]; offset = y.mean(0)
    if z.shape[1] > len(z):
        weights = z.T@np.linalg.solve(z@z.T+alpha*np.eye(len(z)),y-offset)
    else:
        weights = np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-offset))
    return {"mean":mean,"scale":scale,"weights":weights,"offset":offset}


def predict_action(fitted, features):
    x = np.asarray(features,dtype=np.float64)
    if not np.isfinite(x).all(): raise ValueError("nonfinite evaluation features")
    return ((x-fitted["mean"])/fitted["scale"])@fitted["weights"]+fitted["offset"]


class MazeEngine:
    def __init__(self, seed, hidden_size, reset_each_tick=False):
        torch.manual_seed(seed)
        self.model = DemianV1GateState(hidden_size,trace_routes=True).eval()
        self.hidden_size = hidden_size
        self.reset_each_tick = reset_each_tick
        self.state = self.model.initial_state(1,torch.device("cpu"))
        self.initial = capture_v1_runtime(self.model,self.state)
        self.weights_at_start = self.parameter_hash()
        self.route_ids = None

    def parameter_hash(self):
        digest = hashlib.sha256()
        for key,value in self.model.state_dict().items():
            digest.update(key.encode());digest.update(value.detach().numpy().tobytes())
        return digest.hexdigest()

    def reset(self):
        self.state = restore_v1_runtime(self.model,self.initial)

    @torch.no_grad()
    def observe(self, observation):
        if self.reset_each_tick:self.reset()
        public = encode_observation(observation)
        injection = np.zeros(self.hidden_size,dtype=np.float32)
        np.add.at(injection,np.arange(len(public))%self.hidden_size,.25*public)
        state = (self.state[0]+torch.from_numpy(injection)[None],*self.state[1:])
        self.state = self.model.step(state)
        vector = torch.cat(self.state,dim=1).numpy().reshape(-1).copy()
        if not np.isfinite(vector).all(): raise ValueError("nonfinite Demian state")
        trace = self.model.route_trace()
        ids = tuple(trace.values)
        if self.route_ids is not None and ids != self.route_ids: raise RuntimeError("route catalog changed")
        self.route_ids = ids
        routes = np.array([float(torch.linalg.vector_norm(v)) for v in trace.values.values()])
        channels = np.array([float(torch.linalg.vector_norm(v)) for v in self.state])
        return vector,public,routes,channels


@dataclass(frozen=True)
class MazeStepRecord:
    observation: MazeObservation
    action: str
    acknowledgement: MazeAcknowledgement
    position_before: tuple[int,int]
    position_after: tuple[int,int]
    state: np.ndarray
    public_features: np.ndarray
    route_norms: np.ndarray
    channel_norms: np.ndarray


@dataclass(frozen=True)
class MazeRuntime:
    fingerprint: str
    world: MazeSnapshot
    model: V1RuntimeSnapshot


class MazeRunner:
    def __init__(self, seed, hidden_size, config, head=None, reset_each_tick=False, observation_head=False):
        self.engine = MazeEngine(seed,hidden_size,reset_each_tick)
        self.world = MazeWorld(config)
        self.head = head
        self.observation_head = observation_head

    def fingerprint(self):
        digest = hashlib.sha256((self.engine.parameter_hash()+self.world.configuration_hash+
                   str((self.engine.reset_each_tick,self.observation_head))).encode())
        if self.head is not None:
            for key in sorted(self.head): digest.update(key.encode()+self.head[key].tobytes())
        return digest.hexdigest()

    def step(self, action=None):
        if action is not None and action not in ACTIONS: raise ValueError("invalid action")
        if self.world.done:raise RuntimeError("maze episode completed")
        if action is None and self.head is None:raise ValueError("action or fitted head required")
        observation = self.world.observe(); before = self.world.position
        vector,public,routes,channels = self.engine.observe(observation)
        if action is None:
            features = public if self.observation_head else vector
            action = ACTIONS[int(np.argmax(predict_action(self.head,features)))]
        ack = self.world.step(action)
        return MazeStepRecord(observation,action,ack,before,self.world.position,vector,public,routes,channels)

    def snapshot(self):
        return MazeRuntime(self.fingerprint(),self.world.snapshot(),capture_v1_runtime(self.engine.model,self.engine.state))

    def restore(self, snapshot):
        if snapshot.fingerprint != self.fingerprint(): raise ValueError("incompatible runner snapshot")
        if any(not torch.isfinite(v).all() for v in snapshot.model.state):raise ValueError("nonfinite snapshot")
        # Validate both halves before mutating either owner.
        trial_world = MazeWorld(self.world.config);trial_world.restore(snapshot.world)
        expected = [v.shape for v in self.engine.state]
        if [v.shape for v in snapshot.model.state] != expected:raise ValueError("invalid state shapes")
        self.engine.state = restore_v1_runtime(self.engine.model,snapshot.model)
        self.world.restore(snapshot.world)
