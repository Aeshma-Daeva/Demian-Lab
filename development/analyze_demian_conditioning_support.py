"""Post-result descriptive audit of frozen whitened feature energy; no fitting."""
import argparse
import json
from pathlib import Path

import numpy as np

from development.run_demian_cross_seed_transfer import dump, sha
from development.run_demian_distance_conditioning import maps_for


def energy(x, fit):
    z = (x-fit["white_mean"])@fit["white_matrix"]
    return float(np.mean(np.sum(z*z, axis=1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    out = args.artifact
    protocol = json.loads((out/"protocol.json").read_text())
    split = json.loads((out/"split.json").read_text())
    rows = []
    sources = {}
    for model in ("demian", "reservoir"):
        for ti, delay in enumerate(protocol["delays"]):
            for variant in ("contrast_white", "affine_contrast_white"):
                fit_path = out/f"fit_{model}_{delay}_{variant}_true.npz"
                with np.load(fit_path) as file:
                    fit = dict(file)
                sources[str(fit_path)] = sha(fit_path)
                training = []
                for seed in protocol["train_seeds"]:
                    path = Path(protocol["training_artifact"])/f"{model}_{seed}_states.npz"
                    with np.load(path) as file:
                        training.append(maps_for(file["states"][:, ti], split)[0][variant][split["fit"]])
                    sources[str(path)] = sha(path)
                train_energy = energy(np.concatenate(training), fit)
                for seed in protocol["evaluation_seeds"]:
                    path = out/f"{model}_{seed}_states.npz"
                    with np.load(path) as file:
                        features = maps_for(file["states"][:, ti], split)[0][variant][split["evaluation"]]
                    sources[str(path)] = sha(path)
                    value = energy(features, fit)
                    rows.append({"model": model, "delay": delay, "variant": variant, "seed": seed,
                                 "training_energy": train_energy, "evaluation_energy": value,
                                 "energy_ratio": value/train_energy if train_energy else None})
    dump(out/"support_audit.json", {"scope": "Post-result descriptive transformed-feature energy; no refitting or new success gate.",
         "definition": "Mean squared norm after frozen training mean and whitening; ratio to pooled fitting-history energy.",
         "rows": rows, "source_hashes": sources})


if __name__ == "__main__":
    main()
