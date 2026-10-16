import numpy as np

from development.run_demian_frozen_readout_delayed_cue import fit_and_score, latest_seen


def test_latest_seen_uses_last_presented_prefix_not_future_history():
    history = np.asarray([2, 1, 0, 2])
    assert latest_seen(history, 2) == 1


def test_heldout_labels_do_not_change_frozen_fit():
    protocol = {"model_seeds": [7], "train_history_ids": [0, 1], "symbols": [0, 1, 2],
                "delays": [2], "projection_seed": 3, "ridge_alpha": 1.0}
    rows, states = [], []
    for history in range(4):
        for cue in range(3):
            rows.append({"seed": 7, "delay": 2, "cue": cue, "id": history, "latest": history % 3})
            states.append([cue, history, 1.0])
    _, frozen, _ = fit_and_score(np.asarray(states), rows, protocol)
    altered = [dict(row) for row in rows]
    for row in altered:
        if row["id"] not in protocol["train_history_ids"]:
            row["cue"] = (row["cue"] + 1) % 3
    _, altered_frozen, _ = fit_and_score(np.asarray(states), altered, protocol)
    for model in ("demian", "latest"):
        for field in ("mean", "scale", "weights"):
            assert np.array_equal(frozen["fits"]["7"][model][field], altered_frozen["fits"]["7"][model][field])
