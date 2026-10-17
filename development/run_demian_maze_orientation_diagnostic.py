"""Frozen-head, paired-world orientation diagnosis after the original pilot."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from development.run_demian_maze_baseline import episode_configs, collect_teacher, evaluate_policy
from development.demian_maze_runner import predict_action
from development.run_demian_cross_seed_transfer import dump, sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--protocol",type=Path,required=True)
    args=parser.parse_args();p=json.loads(args.protocol.read_text())
    base=json.loads(Path(p["source_protocol"]).read_text());source=Path(p["source_artifact"])
    altered={**base,"evaluation_orientations":[p["orientation_mapping"][str(i)] for i in base["evaluation_orientations"]]}
    old_configs=episode_configs(base,False);configs=episode_configs(altered,False)
    assert all(a[-1].door_tape==b[-1].door_tape for a,b in zip(old_configs,configs,strict=True))
    torch.set_num_threads(1);out=Path(p["artifact_directory"]);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();execution={"status":"running","source_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}
    dump(out/"execution.json",execution);dump(out/"protocol.json",p);dump(out/"worlds.json",[asdict(c[-1]) for c in configs])
    try:
        rows=[];checks=[];source_hashes={}
        for seed in p["model_seeds"]:
            frozen=json.loads((source/f"frozen_fits_{seed}.json").read_text())
            heads={}
            for policy in p["policies"]:
                file=source/f"fit_{seed}_{policy}.npz"
                if sha(file)!=frozen[file.name]:raise ValueError("original fitted head hash mismatch")
                with np.load(file) as archive:heads[policy]=dict(archive)
                source_hashes[str(file)]=sha(file)
            data,observations,engine=collect_teacher(seed,base["hidden_size"],configs,False)
            with np.load(source/f"parameters_{seed}.npz") as saved:
                for key,value in engine.model.state_dict().items():np.testing.assert_array_equal(value.numpy(),saved[key])
            np.savez_compressed(out/f"teacher_{seed}.npz",**data)
            dump(out/f"teacher_observations_{seed}.json",observations)
            logits={}
            for policy,feature in (("recurrent","state"),("reset_matched","reset"),("observation","public")):
                pred=predict_action(heads[policy],data[feature]);logits[policy]=pred
                checks.append({"seed":seed,"policy":policy,"teacher_accuracy":float(np.mean(np.argmax(pred,axis=1)==data["labels"]))})
                metrics,trace,arrays,_=evaluate_policy(seed,base["hidden_size"],configs,policy,heads[policy])
                rows.extend(metrics);np.savez_compressed(out/f"trajectory_{seed}_{policy}.npz",**arrays)
                dump(out/f"trace_{seed}_{policy}.json",trace)
            np.savez_compressed(out/f"teacher_logits_{seed}.npz",**logits,labels=data["labels"])
            print(json.dumps({"seed":seed,"status":"complete"}),flush=True)
        dump(out/"episodes.json",rows);dump(out/"teacher_checks.json",checks)
        original=json.loads((source/"episodes.json").read_text());effects=[]
        for seed in p["model_seeds"]:
            for policy in p["policies"]:
                new=[r for r in rows if r["seed"]==seed and r["policy"]==policy]
                old=[r for r in original if r["seed"]==seed and r["policy"]==policy]
                effects.append({"seed":seed,"policy":policy,"familiar_success":float(np.mean([r["success"] for r in new])),
                                "heldout_success":float(np.mean([r["success"] for r in old])),
                                "familiar_cost":float(np.mean([r["cost"] for r in new])),"heldout_cost":float(np.mean([r["cost"] for r in old]))})
        dump(out/"summary.json",{"scope":p["scope"],"effects":effects,"interpretation":p["interpretation"]})
        dump(out/"source_hashes.json",source_hashes)
        dump(out/"execution.json",{**execution,"status":"complete","wall_seconds":time.monotonic()-start})
        dump(out/"manifest.json",{"files":{str(f.relative_to(out)):sha(f) for f in out.rglob("*") if f.is_file() and f.name!="manifest.json"}})
    except Exception as error:
        dump(out/"execution.json",{**execution,"status":"failed","error":f"{type(error).__name__}: {error}"});raise


if __name__=="__main__":main()
