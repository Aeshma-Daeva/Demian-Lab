"""Run a frozen family of matched-target cue-pair permutations.

The runner only coordinates existing state collection and frozen-readout
tracking.  It never alters Demian, its encoder, or its weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from development import run_demian_uncertainty_decoder_tracking as tracking
from development import run_demian_uncertainty_localization as matrix


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_decomposition_protocol(protocol):
    train = protocol["train_order"]
    conditions = protocol["conditions"]
    if not any(condition["id"] == "identity" for condition in conditions):
        raise ValueError("an identity replay condition is required")
    if len({condition["id"] for condition in conditions}) != len(conditions):
        raise ValueError("condition identifiers must be unique")
    for condition in conditions:
        evaluation = condition["evaluation_order"]
        permutation = condition["evaluation_symbol_permutation"]
        if len(evaluation) != len(train) or sorted(permutation) != list(range(len(train))):
            raise ValueError("condition must be an equal-length cue-pair permutation")
        if any(evaluation[position] != train[source] for position, source in enumerate(permutation)):
            raise ValueError("conditions must preserve evidence pairs")
    identity = next(condition for condition in conditions if condition["id"] == "identity")
    if identity["evaluation_order"] != train or identity["evaluation_symbol_permutation"] != list(range(len(train))):
        raise ValueError("identity condition must replay the base cue-pair order")


def materialize_condition_protocol(protocol, condition):
    """Create the one-condition matrix protocol consumed by the existing runner."""
    return {
        key: value for key, value in protocol.items()
        if key not in {"conditions", "artifact_directory", "confirmation", "discovery_rule"}
    } | {
        "protocol_id": f"{protocol['protocol_id']}-{condition['id']}",
        "condition": condition["id"],
        "evaluation_order": condition["evaluation_order"],
        "evaluation_symbol_permutation": condition["evaluation_symbol_permutation"],
        "matched_target_rule": protocol.get("target_rule", "cue-pair permutation"),
        **({"frozen_reference_checkpoint": protocol["frozen_reference_checkpoint"]}
           if "frozen_reference_checkpoint" in protocol else {}),
    }


def run_protocol(protocol, out):
    validate_decomposition_protocol(protocol)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    out.mkdir(parents=True)
    outputs = []
    for condition in protocol["conditions"]:
        condition_protocol = materialize_condition_protocol(protocol, condition)
        condition_out = out / condition["id"]
        matrix_out = condition_out / "matrix"
        matrix_payload = matrix.run_protocol(condition_protocol)
        matrix.write_artifacts(matrix_out, condition_protocol, matrix_payload)
        tracking_protocol = {
            "protocol_id": f"{condition_protocol['protocol_id']}-tracking",
            "claim_boundary": protocol["claim_boundary"],
            "checkpoints": [point["id"] for point in protocol["checkpoints"]],
            "projection_dimension": protocol["projection_dimension"],
            "projection_seeds": protocol["projection_seeds"],
            "ridge_alpha": protocol["ridge_alpha"],
            "include_dynamic_observers": False,
            "batches": [{
                "id": condition["id"], "artifact_directory": str(matrix_out),
                "source_manifest_sha256": _sha(matrix_out / "manifest.json"),
                "model_seeds": protocol["model_seeds"],
            }],
        }
        tracking_out = condition_out / "tracking"
        tracking_payload = tracking.run_protocol(tracking_protocol)
        tracking.write_artifacts(tracking_out, tracking_protocol, tracking_payload)
        outputs.append({"condition": condition["id"], "matrix_manifest_sha256": _sha(matrix_out / "manifest.json"),
                        "tracking_manifest_sha256": _sha(tracking_out / "manifest.json")})
    (out / "summary.json").write_text(json.dumps({"experiment": protocol["protocol_id"], "conditions": outputs}, indent=2, sort_keys=True) + "\n")
    (out / "manifest.json").write_text(json.dumps({"protocol": protocol, "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "file_sha256": {path.relative_to(out).as_posix(): _sha(path) for path in out.rglob("*") if path.is_file()}}, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    run_protocol(protocol, args.out)
    print(json.dumps({"out": str(args.out), "conditions": len(protocol["conditions"])}))


if __name__ == "__main__":
    main()
