import os
import sys
from pathlib import Path

import numpy as np


os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "models" / "slider"))

import jax.numpy as jnp  # noqa: E402
import modelSpecification as ms  # noqa: E402


def make_listener_arrays(size=0.25, color=0.75, dc=0.55, cd=0.55):
    l1 = jnp.array(
        [
            [size, 0.15, 0.10],
            [color, 0.20, 0.10],
        ],
        dtype=jnp.float32,
    )
    l2 = jnp.array(
        [
            [dc, 0.20, 0.10],
            [cd, 0.20, 0.10],
        ],
        dtype=jnp.float32,
    )
    return l1, l2


def test_usefulness_adjusted_order_bias_drops_for_color_advantage():
    l1, _ = make_listener_arrays(size=0.25, color=0.75)

    adjusted = ms._usefulness_adjusted_order_bias(l1, bias=2.0, usefulness_order_scale=4.0)

    assert float(adjusted) == 0.0


def test_planned_usefulness_order_returns_valid_probability():
    l1, l2 = make_listener_arrays()

    pred = ms._planned_usefulness_order_from_listeners(
        l1,
        l2,
        alpha=1.0,
        bias=2.0,
        usefulness_order_scale=4.0,
    )

    assert np.isfinite(float(pred))
    assert 0.0 <= float(pred) <= 1.0


def test_vectorized_planned_usefulness_order_returns_simplex_component():
    l1, l2 = make_listener_arrays()
    pred = ms.jitted_planned_usefulness_order_speaker_fast(
        jnp.stack([l1, l1]),
        jnp.stack([l2, l2]),
        1.0,
        2.0,
        4.0,
    )

    assert pred.shape == (2,)
    assert np.all(np.isfinite(np.asarray(pred)))
    assert np.all((np.asarray(pred) >= 0.0) & (np.asarray(pred) <= 1.0))


if __name__ == "__main__":
    test_usefulness_adjusted_order_bias_drops_for_color_advantage()
    test_planned_usefulness_order_returns_valid_probability()
    test_vectorized_planned_usefulness_order_returns_simplex_component()
    print("PASS planned usefulness-order speaker tests")
