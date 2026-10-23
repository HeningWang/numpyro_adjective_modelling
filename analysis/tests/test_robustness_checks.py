"""Minimal numerical check for participant bootstrap and oracle selection."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robustness_utils import bootstrap_totals
from participant_architecture_checks import participant_architecture


def test_bootstrap_and_participant_oracle():
    # With two participants, 2*w@[0,10] is Uniform(0,20).
    result = bootstrap_totals([0., 10.])
    assert result['estimate_elpd'] == 10
    assert abs(result['posterior_mean']-10) < .05
    assert abs(result['credible_lower_95']-.5) < .03
    assert abs(result['credible_upper_95']-19.5) < .03
    assert result['probability_positive'] == 1
    negative = bootstrap_totals([-2., -2.], draws=100)
    np.testing.assert_allclose([negative['posterior_mean'], negative['credible_lower_95'],
                               negative['credible_upper_95']], -4)
    assert negative['probability_positive'] == 0
    # Select the oracle endpoint after summing each participant's trials.
    rows = pd.DataFrame({'id': [1, 1, 2, 2], 'g': [3, 0, 0, 0],
                         'i': [0, 2, 2, 2], 'k': [2, 2, 3, 3], 'ki': [4, 3, 3, 3]})
    table = participant_architecture(rows, ('g', 'i', 'k', 'ki'))
    np.testing.assert_equal(table.elpd_oracle.values, [3, 4])
    np.testing.assert_equal(table.Ki_exceeds_both.values, [True, True])
    np.testing.assert_allclose(table.se_Ki_minus_G.iloc[0], 2)
