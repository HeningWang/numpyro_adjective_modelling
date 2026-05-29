import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from models.production.principled_features import (
    LM_PRIOR_15,
    UTTERANCE_LABELS,
    order_only_lm_residual,
    size_uncertainty_excess,
    visual_salience_features,
)


def test_order_only_lm_residual_centers_within_unordered_content():
    scores = order_only_lm_residual(UTTERANCE_LABELS, LM_PRIOR_15)

    by_content = {}
    for label, score in zip(UTTERANCE_LABELS, scores):
        by_content.setdefault("".join(sorted(label)), []).append(score)

    for content_scores in by_content.values():
        assert abs(float(np.mean(content_scores))) < 1e-7

    idx = {label: i for i, label in enumerate(UTTERANCE_LABELS)}
    assert scores[idx["DC"]] > scores[idx["CD"]]
    assert scores[idx["DF"]] > scores[idx["FD"]]
    assert scores[idx["CF"]] > scores[idx["FC"]]
    assert scores[idx["DFC"]] > scores[idx["FCD"]]
    assert scores[idx["D"]] == 0.0


def test_visual_salience_is_soft_contextual_and_color_biased():
    color_distinctive = np.array([
        [9.0, 1.0, 1.0],
        [8.0, 0.0, 1.0],
        [7.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [5.0, 0.0, 1.0],
        [4.0, 0.0, 0.0],
    ])
    size_distinctive = np.array([
        [10.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [2.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 1.0],
        [2.0, 0.0, 0.0],
    ])

    color_sal = visual_salience_features(color_distinctive, is_sharp=1.0)
    size_sal = visual_salience_features(size_distinctive, is_sharp=1.0)

    assert color_sal.shape == (3,)
    assert abs(float(color_sal.mean())) < 1e-7
    assert color_sal[1] > color_sal[0]
    assert size_sal[0] > color_sal[0]
    assert size_sal[1] > size_sal[2]


def test_size_uncertainty_excess_responds_to_blur_reliability():
    states = np.array([
        [10.0, 1.0, 1.0],
        [9.0, 0.0, 1.0],
        [8.0, 0.0, 0.0],
        [7.0, 0.0, 0.0],
        [6.0, 0.0, 1.0],
        [5.0, 0.0, 0.0],
    ])

    sharp_unc = size_uncertainty_excess(states, is_sharp=1.0)
    blur_unc = size_uncertainty_excess(states, is_sharp=0.0)

    assert sharp_unc == 0.0
    assert blur_unc > 0.0


if __name__ == "__main__":
    test_order_only_lm_residual_centers_within_unordered_content()
    test_visual_salience_is_soft_contextual_and_color_biased()
    test_size_uncertainty_excess_responds_to_blur_reliability()
