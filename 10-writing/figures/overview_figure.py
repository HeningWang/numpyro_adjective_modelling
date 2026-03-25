"""
Overview figure for the adjective ordering paper.
Hand-drawn (xkcd) style, showing:
  - Top-left:    Visual scene with objects + referential task
  - Top-center:  RSA model architecture (listener ↔ speaker)
  - Top-right:   Two experiments (slider + production)
  - Bottom:      2×2 model comparison + key finding
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── Pastel palette ─────────────────────────────────────────────────────────────
C_BLUE   = "#A8D8EA"
C_ORANGE = "#FFCBA4"
C_GREEN  = "#B5EAD7"
C_PINK   = "#FFDAC1"
C_PURPLE = "#D5AAFF"
C_YELLOW = "#FFFACD"
C_GRAY   = "#E8E8E8"
C_WHITE  = "#FFFFFF"
C_LGRAY  = "#F5F5F5"

SKETCH = dict(scale=4, length=80, randomness=0.3)
SKETCH_LIGHT = dict(scale=2, length=60, randomness=0.3)

def rbox(ax, x, y, w, h, fc, ec="#333333", lw=1.8):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    box.set_sketch_params(**SKETCH)
    ax.add_patch(box)

def arr(ax, x0, y0, x1, y1, color="#333", lw=1.8, style="-|>",
        cs="arc3,rad=0.0"):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, connectionstyle=cs,
        linewidth=lw, color=color, zorder=4, mutation_scale=14,
        shrinkA=2, shrinkB=2)
    a.set_sketch_params(**SKETCH_LIGHT)
    ax.add_patch(a)

def draw_obj(ax, cx, cy, r, color, form="circle", target=False):
    if form == "circle":
        c = Circle((cx, cy), r, facecolor=color, edgecolor="#333", linewidth=1.3, zorder=3)
        c.set_sketch_params(**SKETCH_LIGHT)
        ax.add_patch(c)
    elif form == "square":
        s = FancyBboxPatch(
            (cx - r, cy - r), 2 * r, 2 * r, boxstyle="round,pad=0.003",
            facecolor=color, edgecolor="#333", linewidth=1.3, zorder=3)
        s.set_sketch_params(**SKETCH_LIGHT)
        ax.add_patch(s)
    elif form == "triangle":
        tri = plt.Polygon(
            [(cx, cy + r), (cx - r * 0.9, cy - r * 0.65),
             (cx + r * 0.9, cy - r * 0.65)],
            facecolor=color, edgecolor="#333", linewidth=1.3, zorder=3)
        tri.set_sketch_params(**SKETCH_LIGHT)
        ax.add_patch(tri)
    if target:
        pad = 0.009
        f = FancyBboxPatch(
            (cx - r - pad, cy - r - pad), 2 * (r + pad), 2 * (r + pad),
            boxstyle="round,pad=0.004", facecolor="none",
            edgecolor="#D32F2F", linewidth=2.8, zorder=5, linestyle=(0, (4, 3)))
        f.set_sketch_params(scale=3, length=60, randomness=0.3)
        ax.add_patch(f)


# ══════════════════════════════════════════════════════════════════════════════
with plt.xkcd(scale=1.0, length=100, randomness=2):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(C_WHITE)

    # ═══════════════════════════════════════════════════════════════════════
    # A  VISUAL SCENE  (top-left)
    # ═══════════════════════════════════════════════════════════════════════
    Ax, Ay, Aw, Ah = 0.015, 0.42, 0.20, 0.55

    rbox(ax, Ax, Ay, Aw, Ah, C_LGRAY, ec="#999", lw=1.2)
    ax.text(Ax + Aw / 2, Ay + Ah - 0.03, "Visual Scene",
            fontsize=14, ha="center", fontweight="bold", color="#333", zorder=5)

    # 6 objects
    objs = [
        (Ax + 0.050, Ay + Ah - 0.13, 0.035, "#42A5F5", "circle",   True),
        (Ax + 0.150, Ay + Ah - 0.13, 0.020, "#42A5F5", "square",   False),
        (Ax + 0.050, Ay + Ah - 0.26, 0.024, "#EF5350", "circle",   False),
        (Ax + 0.150, Ay + Ah - 0.26, 0.028, "#EF5350", "triangle", False),
        (Ax + 0.050, Ay + Ah - 0.38, 0.018, "#66BB6A", "square",   False),
        (Ax + 0.150, Ay + Ah - 0.38, 0.026, "#66BB6A", "circle",   False),
    ]
    for o in objs:
        draw_obj(ax, *o)

    ax.text(Ax + Aw / 2, Ay + 0.065, '"the big blue sticker"',
            fontsize=9.5, ha="center", style="italic", color="#444", zorder=5)
    ax.text(Ax + Aw / 2, Ay + 0.025, "target = dashed frame",
            fontsize=7.5, ha="center", color="#888", zorder=5)

    # Context info below scene
    ax.text(Ax + Aw / 2, Ay - 0.04,
            "3 referential contexts × 2 size discriminability",
            fontsize=8, ha="center", color="#666")

    # ═══════════════════════════════════════════════════════════════════════
    # B  RSA MODEL  (top-center)
    # ═══════════════════════════════════════════════════════════════════════
    Bx, By, Bw, Bh = 0.25, 0.42, 0.44, 0.55

    rbox(ax, Bx, By, Bw, Bh, "#FAFAFA", ec="#999", lw=1.2)
    ax.text(Bx + Bw / 2, By + Bh - 0.03, "RSA Model",
            fontsize=14, ha="center", fontweight="bold", color="#333", zorder=5)

    # -- Compositional Semantics --
    sx, sy, sw, sh = Bx + 0.045, By + Bh - 0.19, 0.17, 0.10
    rbox(ax, sx, sy, sw, sh, C_GREEN)
    ax.text(sx + sw / 2, sy + sh / 2 + 0.012,
            "Compositional", fontsize=10, ha="center", fontweight="bold", zorder=5)
    ax.text(sx + sw / 2, sy + sh / 2 - 0.018,
            "Semantics", fontsize=10, ha="center", fontweight="bold", zorder=5)

    # Static / recursive labels
    ax.text(sx + sw + 0.02, sy + sh / 2 + 0.015,
            "static:  θ fixed from scene prior", fontsize=8, color="#555",
            style="italic", zorder=5)
    ax.text(sx + sw + 0.02, sy + sh / 2 - 0.018,
            "recursive:  θ updates with listener", fontsize=8, color="#555",
            style="italic", zorder=5)

    # -- Literal Listener --
    lx, ly, lw_, lh = Bx + 0.045, By + 0.12, 0.15, 0.10
    rbox(ax, lx, ly, lw_, lh, C_BLUE)
    ax.text(lx + lw_ / 2, ly + lh / 2 + 0.012,
            "Literal Listener", fontsize=10, ha="center", fontweight="bold", zorder=5)
    ax.text(lx + lw_ / 2, ly + lh / 2 - 0.018,
            "L₀(o | u)", fontsize=10, ha="center", zorder=5)

    # -- Pragmatic Speaker --
    px, py, pw, ph = Bx + 0.25, By + 0.12, 0.15, 0.10
    rbox(ax, px, py, pw, ph, C_ORANGE)
    ax.text(px + pw / 2, py + ph / 2 + 0.012,
            "Pragmatic Speaker", fontsize=10, ha="center", fontweight="bold", zorder=5)
    ax.text(px + pw / 2, py + ph / 2 - 0.018,
            "S₁(u | o*)", fontsize=10, ha="center", zorder=5)

    # Arrow: Semantics → Listener
    arr(ax, sx + sw / 2, sy, sx + sw / 2, ly + lh + 0.008)

    # Arrow: Listener → Speaker (forward)
    arr(ax, lx + lw_ + 0.005, ly + lh / 2,
        px - 0.005, py + ph / 2)

    # Incremental loop (curved above)
    arr(ax, px + pw / 2 - 0.02, py + ph + 0.005,
        lx + lw_ / 2 + 0.02, ly + lh + 0.005,
        color="#E65100", lw=2.2, cs="arc3,rad=-0.45")
    ax.text(Bx + Bw / 2 - 0.01, ly + lh + 0.075,
            "word-by-word (incremental)", fontsize=9,
            ha="center", color="#E65100", fontweight="bold",
            style="italic", zorder=5)

    # Global label (straight below)
    mid_y = ly - 0.015
    arr(ax, lx + lw_ / 2 + 0.04, mid_y,
        px + pw / 2 - 0.04, mid_y,
        color="#1565C0", lw=1.5, style="-|>")
    ax.text(Bx + Bw / 2 - 0.01, mid_y - 0.03,
            "full utterance (global)", fontsize=9,
            ha="center", color="#1565C0", fontweight="bold",
            style="italic", zorder=5)

    # Parameters
    ax.text(px + pw / 2, py - 0.035,
            "α rationality  ·  β LM-prior / b subj-bias",
            fontsize=7.5, ha="center", color="#777", zorder=5)

    # ═══════════════════════════════════════════════════════════════════════
    # C  EXPERIMENTS  (top-right)
    # ═══════════════════════════════════════════════════════════════════════
    Cx, Cy, Cw, Ch = 0.73, 0.42, 0.255, 0.55

    # Exp 1
    e1x, e1y, e1w, e1h = Cx + 0.005, Cy + Ch - 0.25, Cw - 0.01, 0.22
    rbox(ax, e1x, e1y, e1w, e1h, C_PINK, ec="#C0392B", lw=1.5)
    ax.text(e1x + e1w / 2, e1y + e1h - 0.03,
            "Exp 1: Slider Rating", fontsize=11, ha="center",
            fontweight="bold", color="#333", zorder=5)
    ax.text(e1x + e1w / 2, e1y + e1h - 0.075,
            '"big blue"  vs  "blue big"', fontsize=9.5, ha="center",
            style="italic", color="#555", zorder=5)
    ax.text(e1x + e1w / 2, e1y + e1h - 0.115,
            "continuous preference (0–1)", fontsize=8.5, ha="center",
            color="#777", zorder=5)
    ax.text(e1x + e1w / 2, e1y + e1h - 0.155,
            "ZOIB likelihood", fontsize=8, ha="center",
            color="#999", zorder=5)
    # Mini slider graphic
    sl_y = e1y + 0.02
    ax.plot([e1x + 0.04, e1x + e1w - 0.04], [sl_y, sl_y],
            color="#999", linewidth=2, zorder=5)
    ax.plot(e1x + e1w * 0.55, sl_y, 'o', color="#D32F2F",
            markersize=8, zorder=6)
    ax.text(e1x + 0.02, sl_y - 0.015, "0", fontsize=6.5, color="#999",
            ha="center", zorder=5)
    ax.text(e1x + e1w - 0.02, sl_y - 0.015, "1", fontsize=6.5, color="#999",
            ha="center", zorder=5)

    # Exp 2
    e2x, e2y, e2w, e2h = Cx + 0.005, Cy + 0.03, Cw - 0.01, 0.22
    rbox(ax, e2x, e2y, e2w, e2h, C_PURPLE, ec="#7B1FA2", lw=1.5)
    ax.text(e2x + e2w / 2, e2y + e2h - 0.03,
            "Exp 2: Free Production", fontsize=11, ha="center",
            fontweight="bold", color="#333", zorder=5)
    ax.text(e2x + e2w / 2, e2y + e2h - 0.075,
            "S · SC · CS · SCF · CSF · ...", fontsize=9.5, ha="center",
            style="italic", color="#555", zorder=5)
    ax.text(e2x + e2w / 2, e2y + e2h - 0.115,
            "15 utterance types", fontsize=8.5, ha="center",
            color="#777", zorder=5)
    ax.text(e2x + e2w / 2, e2y + e2h - 0.155,
            "categorical likelihood", fontsize=8, ha="center",
            color="#999", zorder=5)
    # Mini bar chart
    bar_y0 = e2y + 0.02
    bar_xs = [e2x + 0.03 + i * 0.025 for i in range(8)]
    bar_hs = [0.045, 0.035, 0.02, 0.01, 0.03, 0.008, 0.015, 0.005]
    bar_cs = ["#7E57C2"] * 8
    for bx_, bh_ in zip(bar_xs, bar_hs):
        r = FancyBboxPatch(
            (bx_, bar_y0), 0.018, bh_, boxstyle="round,pad=0.002",
            facecolor="#9575CD", edgecolor="#5E35B1", linewidth=0.8, zorder=5)
        r.set_sketch_params(scale=2, length=40, randomness=0.5)
        ax.add_patch(r)

    # ═══════════════════════════════════════════════════════════════════════
    # D  2×2 MODEL COMPARISON  (bottom-left)
    # ═══════════════════════════════════════════════════════════════════════
    Dx, Dy, Dw, Dh = 0.08, 0.01, 0.52, 0.35

    rbox(ax, Dx, Dy, Dw, Dh, "#FAFAFA", ec="#888", lw=1.5)
    ax.text(Dx + Dw / 2, Dy + Dh - 0.03,
            "2 × 2  Model Comparison  (Bayesian PSIS-LOO)",
            fontsize=12, ha="center", fontweight="bold", color="#333", zorder=5)

    # Column headers
    c1x = Dx + 0.20
    c2x = Dx + 0.40
    rhx = Dx + 0.065
    ax.text(c1x, Dy + Dh - 0.075, "Static θ", fontsize=11,
            ha="center", fontweight="bold", color="#555", zorder=5)
    ax.text(c2x, Dy + Dh - 0.075, "Recursive θ", fontsize=11,
            ha="center", fontweight="bold", color="#555", zorder=5)

    # Row headers
    ax.text(rhx, Dy + 0.19, "Global\nSpeaker", fontsize=10,
            ha="center", va="center", fontweight="bold", color="#555",
            linespacing=1.3, zorder=5)
    ax.text(rhx, Dy + 0.065, "Incremental\nSpeaker", fontsize=10,
            ha="center", va="center", fontweight="bold", color="#555",
            linespacing=1.3, zorder=5)

    # Grid cells
    cw, ch_ = 0.15, 0.09
    cells = [
        (c1x, Dy + 0.19, C_GRAY,    "S_G × θ_fix",   "",       "#777", 1.2),
        (c2x, Dy + 0.19, C_GRAY,    "S_G × θ_rec",   "",       "#777", 1.2),
        (c1x, Dy + 0.065, C_YELLOW, "S_I × θ_fix",   "better", "#777", 1.5),
        (c2x, Dy + 0.065, "#A5D6A7","S_I × θ_rec",   "best ★", "#2E7D32", 2.5),
    ]
    for (cx_, cy_, fc_, lbl, sub, ec_, lw_) in cells:
        rbox(ax, cx_ - cw / 2, cy_ - ch_ / 2, cw, ch_, fc_, ec=ec_, lw=lw_)
        ax.text(cx_, cy_ + 0.012, lbl, fontsize=10, ha="center", va="center",
                fontweight="bold" if "★" in sub else "normal", zorder=5)
        if sub:
            ax.text(cx_, cy_ - 0.023, sub, fontsize=8.5, ha="center",
                    va="center",
                    color="#2E7D32" if "★" in sub else "#888",
                    fontweight="bold" if "★" in sub else "normal",
                    style="italic", zorder=5)

    # ═══════════════════════════════════════════════════════════════════════
    # E  KEY FINDINGS  (bottom-right)
    # ═══════════════════════════════════════════════════════════════════════
    Kx, Ky, Kw, Kh = 0.64, 0.01, 0.345, 0.35

    rbox(ax, Kx, Ky, Kw, Kh, "#E8F5E9", ec="#2E7D32", lw=2.0)
    ax.text(Kx + Kw / 2, Ky + Kh - 0.03, "Key Findings",
            fontsize=13, ha="center", fontweight="bold", color="#2E7D32", zorder=5)

    findings = [
        ("Q2.", "Incremental > Global  (both datasets)", "#333", False),
        ("Q3.", "Recursive semantics helps, but only", "#333", False),
        ("",    "   when paired with incremental speaker", "#555", True),
        ("",    "   → synergistic interaction", "#1B5E20", True),
        ("Q1.", "Composition alone yields baseline", "#333", False),
        ("",    "   ordering effect  (simulation)", "#555", True),
    ]
    for i, (q, txt, col, ital) in enumerate(findings):
        yy = Ky + Kh - 0.09 - i * 0.042
        if q:
            ax.text(Kx + 0.02, yy, q, fontsize=9.5, ha="left",
                    fontweight="bold", color="#1565C0", zorder=5)
        xoff = 0.065 if q else 0.065
        ax.text(Kx + xoff, yy, txt, fontsize=9,
                ha="left", color=col,
                style="italic" if ital else "normal", zorder=5)

    # ═══════════════════════════════════════════════════════════════════════
    #  CONNECTING ARROWS
    # ═══════════════════════════════════════════════════════════════════════

    # Scene → Model
    arr(ax, Ax + Aw + 0.005, Ay + Ah / 2,
        Bx - 0.005, By + Bh / 2, lw=2.5)

    # Model → Exp 1
    arr(ax, Bx + Bw + 0.005, By + Bh * 0.7,
        Cx - 0.005, e1y + e1h / 2, lw=2.0)

    # Model → Exp 2
    arr(ax, Bx + Bw + 0.005, By + Bh * 0.3,
        Cx - 0.005, e2y + e2h / 2, lw=2.0)

    # Model ↓ 2×2
    arr(ax, Bx + Bw / 2, By - 0.005,
        Dx + Dw / 2, Dy + Dh + 0.005, lw=2.0, color="#888")

    # ═══════════════════════════════════════════════════════════════════════
    out = "/Users/heningwang/Documents/GitHub/numpyro_adjective_modelling/10-writing/figures/"
    fig.savefig(out + "overview_figure.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out + "overview_figure.png", bbox_inches="tight", dpi=200)
    print("Saved overview_figure.pdf and overview_figure.png")
