"""Frozen navigation baseline: external action fitting, matched controls, saved evidence."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from development.demian_maze_world import ACTIONS, DELTAS, MazeWorld, MazeTeacher, make_config
from development.demian_maze_runner import MazeEngine, MazeRunner, fit_action_head, predict_action
from development.run_demian_cross_seed_transfer import dump, sha


def episode_configs(protocol, training):
    layouts=protocol["training_orientations"] if training else protocol["evaluation_orientations"]
    configs=[]
    for orientation in (layouts if training else [None]):
        for ci,condition in enumerate(protocol["conditions"]):
            count=protocol["training_replicates_per_layout_condition"] if training else protocol["evaluation_episodes_per_condition"]
            for replicate in range(count):
                layout=orientation if training else layouts[replicate%len(layouts)]
                seed=(protocol["training_world_seed_base"]+100*layout+10*ci+replicate if training else
                      protocol["evaluation_world_seed_base"]+10*ci+replicate)
                configs.append((condition,seed,replicate,make_config(layout,condition,seed,protocol["budget"])))
    return configs


def collect_teacher(seed, width, configs, training):
    engines=[MazeEngine(seed,width),MazeEngine(seed,width,reset_each_tick=True)]
    features,reset,public,labels,actual,episodes,observations=[],[],[],[],[],[],[]
    for ei,(_,_,replicate,config) in enumerate(configs):
        world=MazeWorld(config);teacher=MazeTeacher(config.layout)
        for engine in engines:engine.reset()
        while not world.done:
            observation=world.observe();teacher.observe(observation);desired=teacher.action()
            state,x,_,_=engines[0].observe(observation)
            rstate,_,_,_=engines[1].observe(observation)
            action=desired
            if training and world.tick == 0:
                if replicate == 1:action="north"
                elif replicate == 2:action="wait"
            features.append(state);reset.append(rstate);public.append(x);labels.append(ACTIONS.index(desired))
            actual.append(ACTIONS.index(action));episodes.append(ei);observations.append(asdict(observation))
            world.step(action)
    for engine in engines:
        if engine.parameter_hash()!=engine.weights_at_start:raise RuntimeError("weights changed during teacher collection")
    arrays={"state":np.stack(features),"reset":np.stack(reset),"public":np.stack(public),
            "labels":np.array(labels),"actual_actions":np.array(actual),"episode_ids":np.array(episodes)}
    return arrays,observations,engines[0]


def memory_probe(seed,width,heads):
    rows=[];arrays={}
    for orientation in (0,1):
        states=[];reset_states=[];observations=[];labels=[]
        for condition in ("open","closed"):
            config=make_config(orientation,condition,1)
            world=MazeWorld(config);teacher=MazeTeacher(config.layout)
            engine=MazeEngine(seed,width);reset=MazeEngine(seed,width,True)
            teacher.observe(world.observe())
            direction=teacher.action();dx,dy=DELTAS[direction]
            reverse=next(a for a,d in DELTAS.items() if d==(-dx,-dy))
            for action in (direction,reverse):
                observation=world.observe();engine.observe(observation);reset.observe(observation)
                world.step(action);teacher.observe(world.observe())
            observation=world.observe();states.append(engine.observe(observation)[0]);reset_states.append(reset.observe(observation)[0])
            observations.append(observation);labels.append(ACTIONS.index(teacher.action()))
        if observations[0]!=observations[1] or labels[0]==labels[1]:raise RuntimeError("memory-demand probe invalid")
        np.testing.assert_array_equal(reset_states[0],reset_states[1])
        logits=predict_action(heads["recurrent"],np.stack(states))
        predicted=np.argmax(logits,axis=1)
        rows.append({"orientation":orientation,"same_current_observation":True,"teacher_actions":[ACTIONS[v] for v in labels],
                     "predicted_actions":[ACTIONS[v] for v in predicted],"both_correct":bool(np.array_equal(predicted,labels)),
                     "state_history_gap":float(np.linalg.norm(states[0]-states[1])),"reset_gap":0.})
        arrays[f"orientation_{orientation}_states"]=np.stack(states)
        arrays[f"orientation_{orientation}_logits"]=logits
    return rows,arrays


def evaluate_policy(seed,width,configs,policy,head):
    reset=policy in ("reset_matched","reset_acute")
    runner=MazeRunner(seed,width,configs[0][-1],head,reset,policy=="observation")
    states,public,routes,channels,actions,positions,episodes=[],[],[],[],[],[],[]
    rows,trace=[],[]
    for ei,(condition,world_seed,_,config) in enumerate(configs):
        runner.world.reset(config);runner.engine.reset()
        rng=np.random.default_rng(seed+world_seed)
        blocked=repeated=0;last_block=None;recovery=[];pending_block=None
        while not runner.world.done:
            record=runner.step(str(rng.choice(ACTIONS)) if policy=="random" else None)
            states.append(record.state);public.append(record.public_features);routes.append(record.route_norms)
            channels.append(record.channel_norms);actions.append(ACTIONS.index(record.action));positions.append(record.position_before);episodes.append(ei)
            if record.acknowledgement.blocked:
                blocked+=1
                key=(record.position_before,record.action)
                repeated+=int(key==last_block);last_block=key
                if pending_block is None:pending_block=runner.world.tick
            elif record.acknowledgement.moved:
                if pending_block is not None:recovery.append(runner.world.tick-pending_block);pending_block=None
                last_block=None
            trace.append({"episode":ei,"tick":runner.world.tick-1,"observation":asdict(record.observation),
                          "action":record.action,"acknowledgement":asdict(record.acknowledgement),
                          "position_before":record.position_before,"position_after":record.position_after})
        rows.append({"seed":seed,"policy":policy,"episode":ei,"condition":condition,"world_seed":world_seed,
                     "orientation":config.layout.orientation,"success":runner.world.success,"ticks":runner.world.tick,
                     "cost":runner.world.tick if runner.world.success else config.budget,"blocked":blocked,"repeated_blocked":repeated,
                     "recovery_ticks":recovery,"unrecovered_block":pending_block is not None,
                     "observed_schedule_change":bool(condition=="switch" and runner.world.tick>2)})
    if runner.engine.parameter_hash()!=runner.engine.weights_at_start:raise RuntimeError("weights changed during evaluation")
    arrays={"states":np.stack(states),"public":np.stack(public),"route_norms":np.stack(routes),
            "channel_norms":np.stack(channels),"actions":np.array(actions),"positions":np.array(positions),"episodes":np.array(episodes)}
    return rows,trace,arrays,runner.engine.route_ids


def execute(protocol,out,stage):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    width=16 if stage=="smoke" else protocol["hidden_size"]
    seeds=protocol["development_seeds"] if stage!="confirmation" else protocol["confirmation_seeds"]
    if stage=="smoke":seeds=seeds[:1]
    torch.set_num_threads(1)
    execution={"status":"running","stage":stage,"hidden_size":width,"source_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
               "python":sys.version,"numpy":np.__version__,"torch":str(torch.__version__),"threads":1}
    dump(out/"execution.json",execution);dump(out/"protocol.json",protocol)
    try:
        (out/"source").mkdir()
        for name in ("run_demian_maze_baseline.py","demian_maze_world.py","demian_maze_runner.py","demian_v1_gate_state.py","substrate_lab.py"):
            file=Path("development")/name;(out/"source"/name).write_bytes(file.read_bytes())
        train_configs=episode_configs(protocol,True);eval_configs=episode_configs(protocol,False)
        if stage=="smoke":train_configs=train_configs[:2];eval_configs=eval_configs[:2]
        dump(out/"worlds.json",{key:[{"condition":c,"seed":s,"replicate":r,"config":asdict(config)} for c,s,r,config in configs]
             for key,configs in (("training",train_configs),("evaluation",eval_configs))})
        all_rows=[];checks=[];probes=[]
        for seed in seeds:
            training,observations,engine=collect_teacher(seed,width,train_configs,True)
            np.savez_compressed(out/f"training_{seed}.npz",**training)
            dump(out/f"training_observations_{seed}.json",observations)
            np.savez_compressed(out/f"parameters_{seed}.npz",**{k:v.numpy() for k,v in engine.model.state_dict().items()},
                                **{f"initial_{i}":v.numpy() for i,v in enumerate(engine.initial.state)})
            heads={name:fit_action_head(training[feature],training["labels"]) for name,feature in
                   (("recurrent","state"),("reset_matched","reset"),("observation","public"))}
            fit_files=[]
            for name,head in heads.items():
                file=out/f"fit_{seed}_{name}.npz";np.savez_compressed(file,**head);fit_files.append(file)
            frozen={f.name:sha(f) for f in fit_files};dump(out/f"frozen_fits_{seed}.json",frozen)
            print(json.dumps({"stage":"heads_frozen","seed":seed,"training_rows":len(training["labels"])}),flush=True)
            # Evaluation begins only after fit artifacts and hashes exist.
            evaluation,eval_observations,_=collect_teacher(seed,width,eval_configs,False)
            np.savez_compressed(out/f"evaluation_teacher_{seed}.npz",**evaluation)
            dump(out/f"evaluation_observations_{seed}.json",eval_observations)
            for name,feature in (("recurrent","state"),("reset_matched","reset"),("observation","public")):
                checks.append({"seed":seed,"head":name,"training_accuracy":float(np.mean(np.argmax(predict_action(heads[name],training[feature]),axis=1)==training["labels"])),
                               "heldout_teacher_accuracy":float(np.mean(np.argmax(predict_action(heads[name],evaluation[feature]),axis=1)==evaluation["labels"]))})
            probe,arrays=memory_probe(seed,width,heads);probes.extend([{"seed":seed,**p} for p in probe])
            np.savez_compressed(out/f"memory_probe_{seed}.npz",**arrays)
            for policy in protocol["policies"]:
                head=heads.get("recurrent" if policy=="reset_acute" else policy)
                rows,trace,arrays,route_ids=evaluate_policy(seed,width,eval_configs,policy,head)
                all_rows.extend(rows)
                np.savez_compressed(out/f"trajectory_{seed}_{policy}.npz",**arrays)
                dump(out/f"trace_{seed}_{policy}.json",trace)
            dump(out/"route_ids.json",list(route_ids))
            if any(sha(out/f)!=value for f,value in frozen.items()):raise RuntimeError("frozen head changed")
            print(json.dumps({"stage":"seed_complete","seed":seed}),flush=True)
        dump(out/"episodes.json",all_rows);dump(out/"head_checks.json",checks);dump(out/"memory_probe.json",probes)
        aggregates=[];supported=[]
        for seed in seeds:
            by_policy={}
            for policy in protocol["policies"]:
                rows=[r for r in all_rows if r["seed"]==seed and r["policy"]==policy]
                by_policy[policy]={"success_rate":float(np.mean([r["success"] for r in rows])),"cost":float(np.mean([r["cost"] for r in rows]))}
                for condition in ["all",*protocol["conditions"]]:
                    subset=rows if condition=="all" else [r for r in rows if r["condition"]==condition]
                    if subset:aggregates.append({"seed":seed,"policy":policy,"condition":condition,
                         "success_rate":float(np.mean([r["success"] for r in subset])),"cost":float(np.mean([r["cost"] for r in subset]))})
            intact=by_policy["recurrent"]
            if all(intact["success_rate"]>=by_policy[c]["success_rate"]+.1 and intact["cost"]<=by_policy[c]["cost"] for c in ("reset_matched","observation")):
                supported.append(seed)
        dump(out/"aggregates.json",aggregates)
        required=3 if stage=="development" else 5
        summary={"protocol_id":protocol["protocol_id"],"stage":stage,"seed_count":len(seeds),"supporting_seeds":supported,
                 "gate_pass":len(supported)>=required if stage!="smoke" else None,"claim_boundary":protocol["claim_boundary"]}
        dump(out/"summary.json",summary)
        dump(out/"execution.json",{**execution,"status":"complete","wall_seconds":time.monotonic()-start})
        dump(out/"manifest.json",{"files":{str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file() and p.name!="manifest.json"}})
        print(json.dumps({"stage":"complete",**summary,"wall_seconds":time.monotonic()-start}),flush=True)
    except Exception as error:
        dump(out/"execution.json",{**execution,"status":"failed","error":f"{type(error).__name__}: {error}"})
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--stage",choices=("smoke","development","confirmation"),required=True)
    args=parser.parse_args();protocol=json.loads(args.protocol.read_text())
    execute(protocol,Path(protocol["artifact_root"])/args.stage,args.stage)


if __name__=="__main__":main()
