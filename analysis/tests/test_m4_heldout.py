"""The one M4 test: shared latent effects are integrated after multiplying trials."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_m4_heldout import joint_log_score


def test_joint_heldout_score_matches_analytic_mixture():
    # Two successes, one participant-level coin p~{.1,.9}, equally weighted.
    # E[p**2]=.41; multiplying trial-marginal probabilities would give .25.
    log_likelihood = np.log(np.array([[[.1, .1], [.9, .9]]]))
    np.testing.assert_allclose(joint_log_score(log_likelihood), np.log(.41), atol=1e-12)
    # Add a second posterior draw with a known p=.5: all four components equally weighted.
    values = np.concatenate([log_likelihood, np.log(np.full((1, 2, 2), .5))])
    np.testing.assert_allclose(joint_log_score(values), np.log((.01+.81+.25+.25)/4), atol=1e-12)
