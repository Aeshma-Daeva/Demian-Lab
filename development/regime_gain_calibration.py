"""Select a development candidate, never infer architectural equivalence."""

import itertools
import math
from statistics import mean


def select_gain(records: list[dict], targets: dict[int, float], protocol: dict) -> dict:
    expected = set(itertools.product(protocol["grid"], protocol["seeds"], protocol["directions"], protocol["horizons"]))
    cells = {(r["multiplier"], r["seed"], r["direction_seed"], r["horizon"]): r for r in records}
    candidates = []
    if set(cells) == expected and len(records) == len(expected):
        for gain in protocol["grid"]:
            rows = [r for r in records if r["multiplier"] == gain]
            valid = all(r["passed"] and r["rate"] is not None and math.isfinite(r["rate"]) for r in rows)
            means = {}
            if valid:
                for horizon in protocol["horizons"]:
                    means[horizon] = mean(r["rate"] for r in rows if r["horizon"] == horizon)
                for seed in protocol["seeds"]:
                    seed_means = []
                    for horizon in protocol["horizons"]:
                        rates = [r["rate"] for r in rows if r["seed"] == seed and r["horizon"] == horizon]
                        valid &= max(rates) - min(rates) <= protocol["direction_tolerance"]
                        seed_means.append(mean(rates))
                    valid &= max(seed_means) - min(seed_means) <= protocol["horizon_tolerance"]
            gap = max(abs(means[h] - targets[h]) for h in protocol["horizons"]) if means else None
            candidates.append(
                {"multiplier": gain, "valid": bool(valid), "horizon_means": means, "maximum_target_gap": gap}
            )
    eligible = [c for c in candidates if c["valid"] and c["maximum_target_gap"] <= protocol["matching_tolerance"]]
    selected = min(eligible, key=lambda c: (c["maximum_target_gap"], c["multiplier"])) if eligible else None
    return {
        "status": "development_candidate" if selected else "no_valid_match",
        "selected_multiplier": selected["multiplier"] if selected else None,
        "candidates": candidates,
        "selection_uses_confirmation_seeds": False,
        "fresh_confirmation_required": True,
        "architecture_equivalence_established": False,
    }
