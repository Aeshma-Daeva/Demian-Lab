"""Audit immutable Phase 0 archives, without requiring current source to equal historical source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from development.regime_validation import audit_development, sha256


def audit_campaign(campaign: Path) -> dict:
    raw = (campaign / "protocol.json").read_bytes()
    protocol = json.loads(raw)
    original_bytes = (campaign / "decision.json").read_bytes()
    original = json.loads(original_bytes)
    hashes = original["implementation_hashes"]
    development = audit_development(
        [campaign / f"development_h{h}" for h in protocol["horizons"]],
        raw,
        hashes,
        test_verification=original["test_verification"],
    )
    confirmation = audit_development(
        [campaign / f"confirmation_h{h}" for h in protocol["horizons"]],
        raw,
        hashes,
        test_verification=original["test_verification"],
        panel="confirmation",
        decision_sha256=sha256(original_bytes),
    )
    if (
        not original["approved"]
        or not development["approved"]
        or development["evidence_hashes"] != original["evidence_hashes"]
    ):
        confirmation["approved"] = False
        confirmation["reasons"].append("original development decision is invalid or stale")
    log = (campaign / "verification.log").read_bytes()
    if sha256(log) != original["test_verification"]["output_sha256"]:
        confirmation["approved"] = False
        confirmation["reasons"].append("original test log digest mismatch")
    confirmation["campaign"] = str(campaign.resolve())
    confirmation["original_decision_sha256"] = sha256(original_bytes)
    return confirmation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Fresh audit JSON; never overwrite")
    args = parser.parse_args()
    result = audit_campaign(args.campaign)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("approved", "records_checked", "reasons", "authorized_stage")}))
    if not result["approved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
