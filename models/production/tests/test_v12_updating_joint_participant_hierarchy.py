"""Contract tests for the participant-specific updating K-HKO cell."""

from pathlib import Path
import inspect
import sys

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


def _trace(model, args, seed=17, conditions=None):
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


def _shared_conditions(source_trace, target_trace):
    target_samples = {
        name for name, site in target_trace.items() if site["type"] == "sample"
    }
    return {
        name: value
        for name, value in _sample_values(source_trace).items()
        if name in target_samples
    }


def _probs(trace):
    return np.asarray(trace["obs"]["fn"].probs)


def test_registry_contains_only_the_new_updating_joint_cell():
    assert set(ms.V12_UPDATING_JOINT_PARTICIPANT_MODELS) == {"K-UPD-HKO"}


def test_khko_and_updating_khko_have_identical_sample_contracts():
    args = _arguments()
    fixed = _trace(ms.V11_JOINT_PARTICIPANT_MODELS["K-HKO"], args)
    updating = _trace(
        ms.V12_UPDATING_JOINT_PARTICIPANT_MODELS["K-UPD-HKO"],
        args,
        conditions=_shared_conditions(fixed, _trace(
            ms.V12_UPDATING_JOINT_PARTICIPANT_MODELS["K-UPD-HKO"], args, seed=23
        )),
    )
    fixed_samples = {
        name for name, site in fixed.items() if site["type"] == "sample"
    }
    updating_samples = {
        name for name, site in updating.items() if site["type"] == "sample"
    }
    assert fixed_samples == updating_samples

    fixed_closure = inspect.getclosurevars(
        ms.V11_JOINT_PARTICIPANT_MODELS["K-HKO"]
    ).nonlocals
    updating_closure = inspect.getclosurevars(
        ms.V12_UPDATING_JOINT_PARTICIPANT_MODELS["K-UPD-HKO"]
    ).nonlocals
    assert fixed_closure["recursive"] is False
    assert updating_closure["recursive"] is True
    for name in (
        "variant", "participant_hierarchy", "prefix_mode", "form_spec",
        "order_source", "policy_light",
    ):
        assert fixed_closure[name] == updating_closure[name]


def test_participant_kappa_endpoints_match_updating_global_and_incremental():
    args = _arguments()
    updating_model = ms.V12_UPDATING_JOINT_PARTICIPANT_MODELS["K-UPD-HKO"]
    baseline = _trace(updating_model, args)
    endpoint_conditions = _sample_values(baseline)
    endpoint_conditions.update({
        "kappa_mean": np.asarray(0.0),
        "tau_kappa": np.asarray(0.0),
        "z_kappa": np.zeros_like(endpoint_conditions["z_kappa"]),
    })
    k_at_zero = _trace(updating_model, args, conditions=endpoint_conditions)
    global_trace = _trace(
        ms.V10_ORDER_UPD_MODELS["G-UPD-HO"],
        args,
        conditions=_shared_conditions(k_at_zero, _trace(
            ms.V10_ORDER_UPD_MODELS["G-UPD-HO"], args, seed=29
        )),
    )
    endpoint_conditions.update({"kappa_mean": np.asarray(1.0)})
    k_at_one = _trace(updating_model, args, conditions=endpoint_conditions)
    incremental_trace = _trace(
        ms.V10_ORDER_UPD_MODELS["I-UPD-HO"],
        args,
        conditions=_shared_conditions(k_at_one, _trace(
            ms.V10_ORDER_UPD_MODELS["I-UPD-HO"], args, seed=31
        )),
    )
    assert np.max(np.abs(_probs(k_at_zero) - _probs(global_trace))) < 1e-10
    assert np.max(np.abs(_probs(k_at_one) - _probs(incremental_trace))) < 1e-10


if __name__ == "__main__":
    for test in (
        test_registry_contains_only_the_new_updating_joint_cell,
        test_khko_and_updating_khko_have_identical_sample_contracts,
        test_participant_kappa_endpoints_match_updating_global_and_incremental,
    ):
        test()
        print(f"PASS {test.__name__}")
