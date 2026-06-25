import os
import sys
import tempfile
import inspect
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd
import jax.numpy as jnp


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "models" / "production"))

import helper  # noqa: E402
import modelSpecification as ms  # noqa: E402
import run_inference as ri  # noqa: E402
from semantic_variant_forward_audit import comparison_class_fallback_for_label  # noqa: E402


def test_target_match_encoding_default_and_canonical_opt_in_for_colour():
    canonical = helper.import_dataset(state_encoding="canonical")
    target_match = helper.import_dataset(state_encoding="target_match")

    df = canonical["df"].reset_index(drop=True)
    red_target_idx = int(df.index[df["color_A"].eq("red")][0])

    canonical_states = np.asarray(canonical["states_train"])[red_target_idx]
    target_match_states = np.asarray(target_match["states_train"])[red_target_idx]

    assert canonical_states[0, 1] == 0.0
    assert target_match_states[0, 1] == 1.0
    assert np.all(
        target_match_states[:, 1]
        == (df.loc[red_target_idx, [f"color_{x}" for x in "ABCDEF"]].to_numpy() == "red")
    )


def test_target_match_encoding_handles_noncanonical_form_targets():
    cols = [
        "id", "item", "conditions", "list", "annotation", "trials",
        *[f"size_{x}" for x in "ABCDEF"],
        *[f"color_{x}" for x in "ABCDEF"],
        *[f"form_{x}" for x in "ABCDEF"],
        "sharpness", "combination", "relevant_property",
    ]
    row = [
        "p1", "i1", "ercf", "l1", "C", 1,
        6, 5, 4, 3, 2, 1,
        "blue", "red", "red", "blue", "red", "blue",
        "square", "circle", "square", "circle", "square", "circle",
        "sharp", "cf", "first",
    ]
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        pd.DataFrame([row], columns=cols).to_csv(path, index=False)
        canonical = helper.import_dataset(path, state_encoding="canonical")
        target_match = helper.import_dataset(path, state_encoding="target_match")

        canonical_states = np.asarray(canonical["states_train"])[0]
        target_match_states = np.asarray(target_match["states_train"])[0]

        assert canonical_states[0, 2] == 0.0
        assert target_match_states[0, 2] == 1.0
        assert np.all(target_match_states[:, 2] == np.array([1, 0, 1, 0, 1, 0]))
    finally:
        path.unlink(missing_ok=True)


def test_comparison_class_listener_changes_size_threshold_for_dc():
    states = jnp.asarray(
        [
            [8.0, 1.0, 1.0],
            [6.0, 1.0, 1.0],
            [20.0, 0.0, 1.0],
            [18.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    dc_idx = ms.UTTERANCE_LABELS.index("DC")
    cd_idx = ms.UTTERANCE_LABELS.index("CD")
    utterances = jnp.asarray([ms.utterance_list[dc_idx], ms.utterance_list[cd_idx]])

    static = np.asarray(
        ms.incremental_semantics_jax_frozen(
            states,
            color_sem=0.59,
            form_sem=0.50,
            k=0.50,
            wf=0.6856,
            utterances=utterances,
        )
    )
    comparison_class = np.asarray(
        ms.incremental_semantics_jax_comparison_class(
            states,
            color_sem=0.59,
            form_sem=0.50,
            k=0.50,
            wf=0.6856,
            utterances=utterances,
        )
    )

    assert not np.allclose(comparison_class[0], static[0], atol=1e-4)
    assert comparison_class[0, 0] > static[0, 0]
    assert np.allclose(comparison_class[1], static[1], atol=1e-4)


def test_comparison_class_principled_speaker_simplex_and_new_models_register():
    states = jnp.asarray(
        [
            [8.0, 1.0, 1.0],
            [6.0, 1.0, 1.0],
            [20.0, 0.0, 1.0],
            [18.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    kwargs = dict(
        sufficient_dim=jnp.int32(1),
        has_one_word_solution=jnp.float32(0.0),
        is_sharp=jnp.float32(1.0),
        alpha=jnp.float32(3.0),
        beta_order=jnp.float32(1.0),
        lambda_salience=jnp.float32(0.8),
        rho_salience_stop=jnp.float32(0.75),
        gamma_uncertainty_len=jnp.float32(0.0),
        color_semval=0.59,
        form_semval=0.50,
        k=0.50,
        wf=0.6856,
        epsilon=0.01,
        size_context_mode="comparison_class",
    )
    updating = np.asarray(ms.incremental_speaker_principled(states, recursive=True, **kwargs))
    fixed = np.asarray(ms.incremental_speaker_principled(states, recursive=False, **kwargs))

    assert np.all(updating >= 0.0)
    assert np.allclose(updating.sum(), 1.0, atol=1e-4)
    assert np.allclose(fixed.sum(), 1.0, atol=1e-4)
    assert not np.allclose(updating, fixed, atol=1e-4)

    for key in (
        "principled_salience_stop_regularized_tmcc_2x2_inc_rec",
        "principled_salience_stop_regularized_tmcc_2x2_inc_static",
        "principled_salience_stop_regularized_tmcc_2x2_glob_rec",
        "principled_salience_stop_regularized_tmcc_2x2_glob_static",
        "principled_salience_stop_regularized_tmcc_2x2_glob_rec_fixedeps",
        "principled_salience_stop_regularized_tmcc_2x2_glob_static_fixedeps",
    ):
        assert key in ri.HIER_MODELS

    default_encoding = inspect.signature(ri.run_inference_hier).parameters["state_encoding"].default
    assert default_encoding == "target_match"


def test_comparison_class_fallback_flag_detects_empty_class():
    no_target_colour_states = np.asarray(
        [
            [8.0, 0.0, 1.0],
            [6.0, 0.0, 1.0],
            [5.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert comparison_class_fallback_for_label(no_target_colour_states, "DC")
    assert not comparison_class_fallback_for_label(no_target_colour_states, "D")


if __name__ == "__main__":
    test_target_match_encoding_default_and_canonical_opt_in_for_colour()
    test_target_match_encoding_handles_noncanonical_form_targets()
    test_comparison_class_listener_changes_size_threshold_for_dc()
    test_comparison_class_principled_speaker_simplex_and_new_models_register()
    test_comparison_class_fallback_flag_detects_empty_class()
    print("PASS semantic variant tests")
