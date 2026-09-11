"""Contract tests for the matched v10 context-updating cells."""

from pathlib import Path
import sys
import inspect

import numpy as np
import numpyro.handlers as handlers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper import import_dataset_hier
import modelSpecification as ms


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/production_model_input.csv"


def _arguments(n_rows=16):
    data = import_dataset_hier(file_path=DATA, state_encoding="target_match")
    return {
        "states": data["states_train"][:n_rows],
        "empirical": None,
        "participant_idx": data["participant_idx"][:n_rows],
        "n_participants": data["n_participants"],
        "sufficient_dim": data["sufficient_dim"][:n_rows],
        "has_one_word_solution": data["has_one_word_solution"][:n_rows],
        "is_sharp": data["sharpness_idx"][:n_rows],
        "is_colour_sufficient": data["is_colour_sufficient"][:n_rows],
    }


def _trace(model, args, seed=11, conditions=None):
    seeded = handlers.seed(model, seed)
    if conditions:
        seeded = handlers.condition(seeded, data=conditions)
    return handlers.trace(seeded).get_trace(**args)


def _sample_values(trace):
    return {
        name: site["value"]
        for name, site in trace.items()
        if site["type"] == "sample"
    }


def _probs(trace):
    return np.asarray(trace["obs"]["fn"].probs)


def test_updating_registry_has_three_matched_cells():
    assert set(ms.V10_ORDER_UPD_MODELS) == {"G-UPD-HO", "K-UPD-HO", "I-UPD-HO"}
    assert set(ms.V10_ORDER_MODELS) == {"G-HO", "K-HO", "I-HO"}


def test_updating_kappa_endpoints_match_global_and_incremental():
    args = _arguments()
    k_trace = _trace(ms.V10_ORDER_UPD_MODELS["K-UPD-HO"], args)
    common = _sample_values(k_trace)

    k_at_zero = _trace(
        ms.V10_ORDER_UPD_MODELS["K-UPD-HO"], args,
        conditions={**common, "kappa": np.asarray(0.0)},
    )
    g_trace = _trace(
        ms.V10_ORDER_UPD_MODELS["G-UPD-HO"], args,
        conditions={**common, "kappa": np.asarray(0.0)},
    )
    k_at_one = _trace(
        ms.V10_ORDER_UPD_MODELS["K-UPD-HO"], args,
        conditions={**common, "kappa": np.asarray(1.0)},
    )
    i_trace = _trace(
        ms.V10_ORDER_UPD_MODELS["I-UPD-HO"], args,
        conditions={**common, "kappa": np.asarray(1.0)},
    )

    assert np.max(np.abs(_probs(k_at_zero) - _probs(g_trace))) < 1e-10
    assert np.max(np.abs(_probs(k_at_one) - _probs(i_trace))) < 1e-10


def test_fixed_and_updating_cells_share_parameter_contract_and_change_forward_map():
    args = _arguments()
    fixed = _trace(ms.V10_ORDER_MODELS["K-HO"], args)
    updating = _trace(
        ms.V10_ORDER_UPD_MODELS["K-UPD-HO"], args,
        conditions=_sample_values(fixed),
    )

    fixed_samples = {
        name for name, site in fixed.items() if site["type"] == "sample"
    }
    updating_samples = {
        name for name, site in updating.items() if site["type"] == "sample"
    }
    assert fixed_samples == updating_samples
    fixed_closure = inspect.getclosurevars(ms.V10_ORDER_MODELS["K-HO"]).nonlocals
    updating_closure = inspect.getclosurevars(
        ms.V10_ORDER_UPD_MODELS["K-UPD-HO"]
    ).nonlocals
    assert fixed_closure["recursive"] is False
    assert updating_closure["recursive"] is True
    for name in ("variant", "participant_hierarchy", "prefix_mode", "form_spec", "order_source", "policy_light"):
        assert fixed_closure[name] == updating_closure[name]


if __name__ == "__main__":
    for test in (
        test_updating_registry_has_three_matched_cells,
        test_updating_kappa_endpoints_match_global_and_incremental,
        test_fixed_and_updating_cells_share_parameter_contract_and_change_forward_map,
    ):
        test()
        print(f"PASS {test.__name__}")
