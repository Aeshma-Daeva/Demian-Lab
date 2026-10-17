"""Prospective anchor-calibrated transfer; fit artifacts precede evaluation runs."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from development.cross_seed_reservoir_reference import collect_reservoir
from development.demian_cross_seed_transfer import (
    BITS, HISTORIES, anchor_features, feature_maps, fit_readouts, guarded_geometry,
    make_split, predict_readouts, r2, shape_diagnostics,
)
from development.demian_uncertainty_probe import encode_evidence
from development.demian_uncertainty_world import EvidenceObservation
from development.demian_v1_gate_state import DemianV1GateState, V1_CHANNELS
from development.run_demian_uncertainty_localization import fit_probe, predict_probe


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")


@torch.no_grad()
def collect_demian(seed, histories, reliabilities, delays, hidden_size=256):
    torch.manual_seed(seed)
    model = DemianV1GateState(hidden_size).eval()
    initial = model.initial_state(1, torch.device("cpu"))
    state = tuple(part.repeat(len(histories), 1) for part in initial)
    params = {key: value.detach().numpy().copy() for key, value in model.state_dict().items()}
    collected = []
    for tick in range(6+max(delays)):
        if tick < 6:
            frames = [EvidenceObservation(tick, "evidence", int(symbol), reliabilities[tick]) for symbol in histories[:, tick]]
        else:
            frames = [EvidenceObservation(tick, "delay", None, None)]*len(histories)
        encoded = torch.cat([encode_evidence(frame, hidden_size) for frame in frames])
        state = model.step((state[0]+encoded, *state[1:]))
        if tick-5 in delays:
            collected.append(torch.cat(state, dim=1).numpy().copy())
    for key, value in model.state_dict().items():
        if not np.array_equal(value.numpy(), params[key]):
            raise RuntimeError("Demian weights changed")
    states = np.stack(collected, axis=1)
    if not np.isfinite(states).all():
        raise ValueError("nonfinite Demian rollout")
    arrays = {**{f"weight_{k}": v for k, v in params.items()},
              **{f"initial_{k}": v.numpy() for k, v in zip(V1_CHANNELS, initial, strict=True)}}
    return states, {"seed": seed, "channel_widths": [int(v.shape[1]) for v in initial],
                    "channels": list(V1_CHANNELS), "parameter_count": sum(v.size for v in params.values()),
                    "frozen_weights_verified": True, **arrays}


def current_inputs(protocol):
    return np.stack([np.concatenate([
        encode_evidence(EvidenceObservation(5, "evidence", int(history[-1]), protocol["reliabilities"][-1])
                        if delay == 0 else EvidenceObservation(5+delay, "delay", None, None), protocol["hidden_size"]).numpy()
        for history in HISTORIES], axis=0) for delay in protocol["delays"]], axis=1)


def save_collection(out, model, seed, states, metadata):
    arrays = {key: value for key, value in metadata.items() if isinstance(value, np.ndarray)}
    info = {key: value for key, value in metadata.items() if key not in arrays}
    stem = out / f"{model}_{seed}"
    np.savez_compressed(f"{stem}_states.npz", states=states)
    np.savez_compressed(f"{stem}_parameters.npz", **arrays)
    dump(f"{stem}_metadata.json", info)


def load_collected(source, protocol):
    """Recover original observations and frozen fits, without collecting or fitting."""
    manifest = json.loads((source/"collection_manifest.json").read_text())
    for name, expected in manifest.items():
        path = source/name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"collection hash mismatch: {name}")
    if json.loads((source/"protocol.json").read_text()) != protocol:
        raise ValueError("recovery protocol mismatch")
    split = make_split(protocol["split_seed"])
    if json.loads((source/"split.json").read_text()) != split:
        raise ValueError("recovery split mismatch")
    corpus = {model: {} for model in ("demian", "reservoir", "current_input")}
    for model in corpus:
        for seed in protocol["train_seeds"]+protocol["evaluation_seeds"]:
            name = f"{model}_{seed}_states.npz"
            if name not in manifest:
                raise ValueError(f"unhashed recovery input: {name}")
            with np.load(source/name) as archive:
                states = archive["states"]
            if states.shape[:2] != (64, len(protocol["delays"])) or not np.isfinite(states).all():
                raise ValueError(f"invalid recovery states: {name}")
            corpus[model][seed] = states
    frozen = {}
    fits = json.loads((source/"frozen_fit_manifest.json").read_text())
    for model in corpus:
        for delay in protocol["delays"]:
            name = f"fit_{model}_{delay}.npz"
            if name not in manifest or fits.get(name) != sha(source/name):
                raise ValueError(f"frozen fit hash mismatch: {name}")
            with np.load(source/name) as archive:
                frozen[(model, delay)] = dict(archive)
    return corpus, frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path,
                        help="Analyze hash-verified collected states and frozen fits in a new directory")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.out.exists():
        raise FileExistsError("refusing to overwrite experiment")
    if set(protocol["train_seeds"]) & set(protocol["evaluation_seeds"]):
        raise ValueError("overlapping fitting/evaluation seeds")
    torch.set_num_threads(1)
    out = args.out
    out.mkdir(parents=True)
    dump(out/"protocol.json", protocol)
    split = make_split(protocol["split_seed"])
    dump(out/"split.json", split)
    execution = {"source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                               "python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
                               "threads": 1, "status": "running"}
    dump(out/"execution.json", execution)
    # Archive exact implementation; a dirty checkout cannot hide the executed source.
    sources = [Path(__file__), Path("development/demian_cross_seed_transfer.py"), Path("development/cross_seed_reservoir_reference.py"),
               Path("development/demian_uncertainty_probe.py"), Path("development/demian_uncertainty_world.py"),
               Path("development/demian_v1_gate_state.py"), Path("development/substrate_lab.py"),
               Path("development/run_demian_uncertainty_localization.py")]
    (out/"source").mkdir()
    for path in sources:
        (out/"source"/path.name).write_bytes(path.read_bytes())
    targets = BITS@np.log(np.array(protocol["reliabilities"])/(1-np.array(protocol["reliabilities"])))
    np.savez_compressed(out/"histories_targets.npz", histories=HISTORIES, targets=targets)
    corpus = {model: {} for model in ("demian", "reservoir", "current_input")}
    frozen_all = {}
    input_states = current_inputs(protocol)
    if args.resume_from:
        corpus, frozen_all = load_collected(args.resume_from, protocol)
        names = json.loads((args.resume_from/"collection_manifest.json").read_text())
        for name in names:
            if name.endswith(("_states.npz", "_parameters.npz", "_metadata.json")) or name.startswith("fit_") or name == "frozen_fit_manifest.json":
                shutil.copy2(args.resume_from/name, out/name)
        shutil.copytree(args.resume_from/"source", out/"collection_source")
        dump(out/"recovery.json", {"source_directory": str(args.resume_from.resolve()),
             "collection_manifest_sha256": sha(args.resume_from/"collection_manifest.json"),
             "collection_execution": json.loads((args.resume_from/"execution.json").read_text()),
             "change": "Convert precision_floor scalar to Python float for JSON export; replay saved observations and frozen fits.",
             "recollected": False, "refitted_primary_readouts": False})
    for phase in (() if args.resume_from else ("train", "evaluation")):
        for seed in protocol[f"{phase}_seeds"]:
            for model in corpus:
                if model == "demian":
                    states, metadata = collect_demian(seed, HISTORIES, protocol["reliabilities"], protocol["delays"], protocol["hidden_size"])
                elif model == "reservoir":
                    states, metadata = collect_reservoir(seed, HISTORIES, protocol["reliabilities"], protocol["delays"])
                else:
                    states, metadata = input_states.copy(), {"seed": seed, "model": "current_public_input", "recurrent": False}
                corpus[model][seed] = states
                save_collection(out, model, seed, states, metadata)
            print(json.dumps({"stage": phase, "seed": seed, "status": "collected"}), flush=True)
        if phase == "train":
            fit_paths = []
            for model, cells in corpus.items():
                for ti, delay in enumerate(protocol["delays"]):
                    frozen, info = fit_readouts(np.stack([cells[s][:, ti] for s in protocol["train_seeds"]]), targets, split,
                                               projection_seed=protocol["projection_seed"], shuffle_seed=protocol["label_shuffle_seed"], alpha=protocol["ridge_alpha"])
                    frozen_all[(model, delay)] = frozen
                    path = out/f"fit_{model}_{delay}.npz"
                    np.savez_compressed(path, **frozen, **info)
                    fit_paths.append(path)
            dump(out/"frozen_fit_manifest.json", {p.name: sha(p) for p in fit_paths})
            print(json.dumps({"stage": "fit_frozen", "files": len(fit_paths)}), flush=True)
    evaluation = split["evaluation"]
    fitting = split["fit"]
    y = targets[evaluation]
    permutations = [np.random.default_rng(protocol["anchor_permutation_seed"]+i).permutation(16)
                    for i in range(protocol["anchor_permutation_count"])]
    latest = BITS[:, -1]*np.log(protocol["reliabilities"][-1]/(1-protocol["reliabilities"][-1]))
    repeats = len(protocol["train_seeds"])
    public_fit = fit_probe(np.tile(BITS[fitting], (repeats, 1)), np.tile(targets[fitting], repeats), alpha=1)
    last_fit = fit_probe(np.tile(BITS[fitting, -1:], (repeats, 1)), np.tile(targets[fitting], repeats), alpha=1)
    public_pred = predict_probe(public_fit, BITS)[evaluation]
    last_pred = predict_probe(last_fit, BITS[:, -1:])[evaluation]
    metrics, shapes, geometry = [], [], []
    prediction_arrays = {"public_bits": public_pred, "latest": latest[evaluation], "last_bit": last_pred, "target": y}
    for model, cells in corpus.items():
        for ti, delay in enumerate(protocol["delays"]):
            frozen = frozen_all[(model, delay)]
            residuals = {}
            for seed in protocol["evaluation_seeds"]:
                x = cells[seed][:, ti]
                preds = predict_readouts(frozen, x, split)
                maps, calibration = feature_maps(x, split, frozen["projection"])
                fit = {key: frozen[f"anchor_{key}"] for key in ("mean", "scale", "weights")}
                perm_scores = np.stack([predict_probe(fit, maps["anchor"][:, perm])[evaluation] for perm in permutations])
                within = fit_probe(maps["anchor"][fitting], targets[fitting], alpha=1)
                preds["within_seed"] = predict_probe(within, maps["anchor"])
                scores = {name: r2(y, pred[evaluation]) for name, pred in preds.items()}
                scores.update(latest=r2(y, latest[evaluation]), public_bits=r2(y, public_pred), last_bit=r2(y, last_pred),
                              constant=r2(y, np.repeat(targets[fitting].mean(), len(y))),
                              permutation_q95=float(np.quantile([r2(y, p) for p in perm_scores], .95)))
                passed = (not calibration["precision_limited"] and scores["anchor"] >= .1 and
                          all(scores["anchor"] >= scores[c]+.1 for c in ("latest", "shuffle", "permutation_q95")))
                metrics.append({"model": model, "seed": seed, "delay": delay, **scores,
                                "precision_limited": calibration["precision_limited"], "anchor_scatter": calibration["scatter"],
                                "precision_floor": calibration["precision_floor"], "gate_pass": bool(passed)})
                key = f"{model}_{seed}_{delay}"
                for name, pred in preds.items():
                    prediction_arrays[f"{key}_{name}"] = pred[evaluation]
                prediction_arrays[f"{key}_permutations"] = perm_scores
                # Label-free target-seed calibration and separate diagnostic fits are retained explicitly.
                np.savez_compressed(out/f"calibration_{key}.npz", **calibration,
                                    **{f"within_{k}": v for k, v in within.items()})
                desc, residual = shape_diagnostics(x, split)
                residuals[seed] = residual
                shapes.append({"model": model, "seed": seed, "delay": delay, **desc})
            for a, b in itertools.combinations(protocol["evaluation_seeds"], 2):
                direct = guarded_geometry(cells[a][:, ti], cells[b][:, ti])
                residual = guarded_geometry(residuals[a][evaluation], residuals[b][evaluation],
                                            cells[a][evaluation, ti], cells[b][evaluation, ti])
                geometry.append({"model": model, "delay": delay, "seed_a": a, "seed_b": b, **direct,
                                 "residual": residual})
    if args.resume_from:
        with np.load(args.resume_from/"predictions.npz") as original:
            if set(original.files) != set(prediction_arrays) or any(
                    not np.array_equal(original[k], prediction_arrays[k]) for k in prediction_arrays):
                raise RuntimeError("recovered predictions differ from original execution")
        dump(out/"replay_verification.json", {"prediction_arrays": len(prediction_arrays),
             "exact_match": True, "original_sha256": sha(args.resume_from/"predictions.npz")})
    np.savez_compressed(out/"predictions.npz", **prediction_arrays)
    dump(out/"metrics.json", metrics)
    dump(out/"geometry.json", geometry)
    dump(out/"shape.json", shapes)
    gates = {}
    for model in corpus:
        counts = {str(d): sum(m["gate_pass"] for m in metrics if m["model"] == model and m["delay"] == d)
                  for d in protocol["primary_delays"]}
        intersection = [s for s in protocol["evaluation_seeds"] if all(next(m["gate_pass"] for m in metrics
                        if m["model"] == model and m["seed"] == s and m["delay"] == d) for d in protocol["primary_delays"])]
        gates[model] = {"passing_seeds_by_delay": counts, "overall_pass": all(n >= 5 for n in counts.values()),
                        "passing_seed_intersection": intersection}
    dump(out/"summary.json", {"protocol_id": protocol["protocol_id"], "status": "complete", "gates": gates,
                              "claim_boundary": protocol["claim_boundary"]})
    before = json.loads((out/"frozen_fit_manifest.json").read_text())
    if any(sha(out/name) != value for name, value in before.items()):
        raise RuntimeError("frozen fit changed after evaluation")
    dump(out/"execution.json", {**execution, "status": "complete"})
    dump(out/"manifest.json", {"protocol_sha256": sha(out/"protocol.json"), "files": {
        str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}})
    print(json.dumps({"stage": "complete", "gates": gates}), flush=True)


if __name__ == "__main__":
    main()
