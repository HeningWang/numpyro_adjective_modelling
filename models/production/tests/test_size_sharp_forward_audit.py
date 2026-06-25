import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from models.production.size_sharp_forward_audit import build_gate_decision


def test_size_sharp_gate_requires_target_gain_without_second_worsening():
    rows = pd.DataFrame({
        "variant": ["sizesharp_a", "sizesharp_b"],
        "first_sharp_DF_abs_residual_reduction": [0.06, 0.06],
        "first_sharp_D_abs_residual_reduction": [0.05, 0.05],
        "second_sharp_CF_abs_residual_worsening": [0.01, 0.02],
        "condition_rmse_delta": [0.0005, 0.0005],
    })

    gate = build_gate_decision(rows)

    assert list(gate["full_inference_gate"]) == ["pass", "fail"]
    assert "target residuals pass" in gate.iloc[0]["gate_reason"]
    assert "second/sharp CF" in gate.iloc[1]["gate_reason"]


if __name__ == "__main__":
    test_size_sharp_gate_requires_target_gain_without_second_worsening()
    print("PASS size-sharp forward audit tests")
