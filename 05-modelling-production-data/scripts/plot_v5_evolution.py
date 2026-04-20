"""Fit-metric evolution across v1 -> v5 (F2). dc subset N = 3196."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = Path("figures/v5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CSP palette
GLAUCOUS = "#7581B3"
GLAUCOUS_DARK = "#505D91"
SHIMMER = "#C65353"
ACCENT_1 = "#4092A2"  # teal
ACCENT_2 = "#EEB14F"  # orange
ACCENT_3 = "#5C7457"  # fern
INDEPENDENCE = "#575463"

stages = ["baseline", "ext-v1", "v5a", "v5b", "v5 (F1)", "v5 (C1)", "v5 (F2)"]
elpd   = [-7157,    -6387,   -6265,  -6381,  -5933,     -5323,     -5294]
r2     = [ 0.346,    0.596,   None,   None,   0.672,     0.908,     0.929]  # R² on emp>=.02
l1     = [ 3.17,     3.17,    None,   None,   2.70,      1.34,      1.22]
eps    = [ None,     0.22,    None,   None,   0.16,      0.10,      0.10]
lamC   = [ None,     None,    3.4,    None,   5.0,       3.06,      3.09]
mu_nc  = [ None,     None,    None,   None,   None,     -5.07,     -5.08]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
x = np.arange(len(stages))

# Panel 1: LOO ELPD
ax = axes[0]
ax.plot(x, elpd, "o-", color=GLAUCOUS_DARK, lw=2.5, markersize=9, zorder=3)
for i, v in enumerate(elpd):
    ax.text(i, v + 60, f"{v:.0f}", ha="center", fontsize=9, color=INDEPENDENCE)
ax.set_xticks(x)
ax.set_xticklabels(stages, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("ELPD LOO", fontsize=11, color=INDEPENDENCE)
ax.set_title("Predictive performance", fontsize=12, color=GLAUCOUS_DARK, weight="bold")
ax.grid(True, alpha=0.3)
ax.set_ylim(min(elpd) - 200, max(elpd) + 200)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

# Panel 2: R² and L1 residual (dual axis)
ax = axes[1]
r2_x = [i for i, v in enumerate(r2) if v is not None]
r2_y = [v for v in r2 if v is not None]
l1_x = [i for i, v in enumerate(l1) if v is not None]
l1_y = [v for v in l1 if v is not None]

ax.plot(r2_x, r2_y, "o-", color=GLAUCOUS_DARK, lw=2.5, markersize=9, label="R² (emp≥.02)", zorder=3)
for xi, yi in zip(r2_x, r2_y):
    ax.text(xi, yi + 0.03, f"{yi:.3f}", ha="center", fontsize=8, color=GLAUCOUS_DARK)
ax.set_ylabel("R²  (emp ≥ .02)", color=GLAUCOUS_DARK, fontsize=11)
ax.tick_params(axis="y", labelcolor=GLAUCOUS_DARK)
ax.set_xticks(x)
ax.set_xticklabels(stages, rotation=30, ha="right", fontsize=9)
ax.set_ylim(0.0, 1.05)
ax.set_title("Group-level fit", fontsize=12, color=GLAUCOUS_DARK, weight="bold")
ax.grid(True, alpha=0.3)

ax2 = ax.twinx()
ax2.plot(l1_x, l1_y, "s--", color=SHIMMER, lw=2.0, markersize=8, label="L1 residual", zorder=2)
for xi, yi in zip(l1_x, l1_y):
    ax2.text(xi, yi + 0.12, f"{yi:.2f}", ha="center", fontsize=8, color=SHIMMER)
ax2.set_ylabel("L1 residual  (sum)", color=SHIMMER, fontsize=11)
ax2.tick_params(axis="y", labelcolor=SHIMMER)
ax2.set_ylim(0, 4)
for spine in ("top",):
    ax.spines[spine].set_visible(False)
    ax2.spines[spine].set_visible(False)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right", frameon=False)

# Panel 3: Lapse ε + key new parameters
ax = axes[2]
eps_x = [i for i, v in enumerate(eps) if v is not None]
eps_y = [v for v in eps if v is not None]
ax.plot(eps_x, eps_y, "o-", color=INDEPENDENCE, lw=2.5, markersize=9, label="ε (lapse)", zorder=3)
for xi, yi in zip(eps_x, eps_y):
    ax.text(xi, yi + 0.02, f"{yi:.2f}", ha="center", fontsize=8, color=INDEPENDENCE)
ax.set_xticks(x)
ax.set_xticklabels(stages, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("lapse ε", fontsize=11, color=INDEPENDENCE)
ax.set_ylim(0, 0.3)
ax.set_title("Residual noise absorbed", fontsize=12, color=GLAUCOUS_DARK, weight="bold")
ax.grid(True, alpha=0.3)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

fig.suptitle("Extension v5 — Fit evolution across mechanism additions (dc subset, N = 3196)",
             fontsize=13, color=GLAUCOUS_DARK, weight="bold", y=1.02)
fig.tight_layout()
for fmt in ("pdf", "png"):
    out = OUT_DIR / f"v5_fit_evolution.{fmt}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)

# --- Second figure: parameter posterior evolution for mechanism params -------
mechs = {
    "λ_C":         [None, None, 3.4,  None, 5.0, 3.06, 3.09],
    "γ_1":         [None, None, None, 1.9, 2.56, 1.99, 2.32],
    "γ_2":         [None, None, None, 0.8, 0.92, 1.42, 1.58],
    "δγ_1":        [None, None, None, None, -1.89, -2.10, -2.16],
    "δγ_2":        [None, None, None, None, -3.51, -0.66, -0.58],
    "η_1":         [None, None, None, None, None, None, -0.61],
    "η_2":         [None, None, None, None, None, None, -0.34],
    "μ_noncanon":  [None, None, None, None, None, -5.07, -5.08],
}
# log β is a pre-existing parameter (not a v5 mechanism) but its posterior shifts
# sharply once μ_noncanon is introduced. Plotted with a dashed line to signal
# that it is sampled across all stages, not newly added.
log_beta = [None, 2.05, None, None, 2.05, -0.18, -0.17]
colors = [GLAUCOUS_DARK, GLAUCOUS, SHIMMER, ACCENT_1, ACCENT_2, ACCENT_3, INDEPENDENCE, "#8e44ad"]
LOG_BETA_COLOR = "#7F4A2F"

fig, ax = plt.subplots(figsize=(11, 4.5))
for (name, vals), c in zip(mechs.items(), colors):
    xs = [i for i, v in enumerate(vals) if v is not None]
    ys = [v for v in vals if v is not None]
    if not xs:
        continue
    ax.plot(xs, ys, "o-", color=c, lw=2, markersize=8, label=name)
    # label the final value
    ax.annotate(name, xy=(xs[-1] + 0.05, ys[-1]), fontsize=9, color=c,
                va="center", ha="left")

# log β: pre-existing parameter, dashed to distinguish from v5 mechanisms
lb_xs = [i for i, v in enumerate(log_beta) if v is not None]
lb_ys = [v for v in log_beta if v is not None]
ax.plot(lb_xs, lb_ys, "s--", color=LOG_BETA_COLOR, lw=2, markersize=7,
        label="log β")
ax.annotate("log β", xy=(lb_xs[-1] + 0.05, lb_ys[-1]), fontsize=9,
            color=LOG_BETA_COLOR, va="center", ha="left")

ax.axhline(0, color=INDEPENDENCE, lw=0.5, linestyle=":")
ax.set_xticks(x)
ax.set_xticklabels(stages, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("posterior mean", fontsize=11, color=INDEPENDENCE)
ax.set_title("Mechanism parameter posteriors across stages",
             fontsize=13, color=GLAUCOUS_DARK, weight="bold")
ax.grid(True, alpha=0.3)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

fig.tight_layout()
for fmt in ("pdf", "png"):
    out = OUT_DIR / f"v5_param_evolution.{fmt}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)
