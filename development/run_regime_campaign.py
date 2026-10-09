"""Frozen Phase 0 development → audit → confirmation; at most two CPU workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from development.regime_validation import ROOT, audit_development, sha256, snapshot_sources, source_hashes


def _save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _run_stage(
    out: Path, protocol: dict, *, mode: str, protocol_path: Path, decision: Path | None = None
) -> list[Path]:
    pending = list(protocol["horizons"])
    active, logs, directories = {}, [], []
    try:
        while pending or active:
            while pending and len(active) < 2:
                horizon = pending.pop(0)
                directory = out / f"{mode}_h{horizon}"
                directories.append(directory)
                command = [
                    sys.executable,
                    "-m",
                    "development.run_regime_characterization",
                    "--mode",
                    mode,
                    "--protocol",
                    str(protocol_path),
                    "--steps",
                    str(horizon),
                    "--out",
                    str(directory),
                ]
                if decision is not None:
                    command += ["--decision", str(decision)]
                log = (out / f"{mode}_h{horizon}.log").open("x")
                logs.append(log)
                active[horizon] = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                print(f"started {mode} horizon={horizon}", flush=True)
            for horizon, process in list(active.items()):
                code = process.poll()
                if code is not None:
                    del active[horizon]
                    if code != 0:
                        raise RuntimeError(f"{mode} horizon={horizon} failed with exit {code}; inspect its log")
                    print(f"completed {mode} horizon={horizon}", flush=True)
            if active:
                time.sleep(1)
    finally:
        for process in active.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in logs:
            log.close()
    return directories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/REGIME_MATCHED_PROTOCOL.json")
    parser.add_argument("--out", type=Path, required=True, help="Fresh campaign directory")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    protocol = json.loads(raw)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    frozen = args.out / "protocol.json"
    frozen.write_bytes(raw)
    hashes = snapshot_sources(args.out)
    status = {
        "stage": "verifying",
        "protocol_sha256": sha256(raw),
        "implementation_hashes": hashes,
        "validation_only": args.validation_only,
        "maximum_workers": 2,
        "later_phases_authorized": False,
    }
    _save(args.out / "status.json", status)
    try:
        command = [sys.executable, "-m", "pytest", "tests/test_regime_characterization.py", "-q"]
        verification = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (args.out / "verification.log").write_text(verification.stdout + verification.stderr)
        proof = {
            "command": command,
            "returncode": verification.returncode,
            "implementation_hashes": hashes,
            "output_sha256": sha256((verification.stdout + verification.stderr).encode()),
        }
        if verification.returncode != 0 or source_hashes() != hashes:
            raise RuntimeError("instrument verification failed or code changed")
        status["stage"] = "development_running"
        _save(args.out / "status.json", status)
        directories = _run_stage(args.out, protocol, mode="development", protocol_path=frozen)
        decision = audit_development(directories, raw, hashes, test_verification=proof)
        decision_path = args.out / "decision.json"
        _save(decision_path, decision)
        if not decision["approved"]:
            raise RuntimeError("development gate failed: " + "; ".join(decision["reasons"][:5]))
        status["stage"] = "development_validated"
        status["decision_sha256"] = sha256(decision_path.read_bytes())
        _save(args.out / "status.json", status)
        if args.validation_only:
            return
        status["stage"] = "confirmation_running"
        _save(args.out / "status.json", status)
        directories = _run_stage(args.out, protocol, mode="confirmation", protocol_path=frozen, decision=decision_path)
        # Completion is data collection only; interpretation and Phase 0b still require a decision.
        status["stage"] = "confirmation_collected"
        status["confirmation_dirs"] = [str(p) for p in directories]
        _save(args.out / "status.json", status)
    except BaseException as exc:
        status.update(stage="failed", reason=str(exc))
        _save(args.out / "status.json", status)
        raise


if __name__ == "__main__":
    main()
