"""
Generate an Excalidraw overview figure for the adjective ordering paper.

Layout (wide, two rows):
  Row 1: [Visual Scene] ---> [RSA Model (L→R speaker, R→L listener)] ---> [Experiments]
  Row 2: [2×2 Model Comparison]  [Research Questions & Findings]
"""

import json
import random

random.seed(42)
_idx = 0
_el_counter = 0

def uid():
    global _idx; _idx += 1
    return f"el_{_idx:04d}"

def index_str():
    global _el_counter; _el_counter += 1
    return f"a{_el_counter}"

def base(id_, type_, x, y, w, h, **kw):
    return {
        "id": id_, "type": type_,
        "x": x, "y": y, "width": w, "height": h, "angle": 0,
        "strokeColor": kw.get("strokeColor", "#1e1e1e"),
        "backgroundColor": kw.get("backgroundColor", "transparent"),
        "fillStyle": kw.get("fillStyle", "solid"),
        "strokeWidth": kw.get("strokeWidth", 2),
        "strokeStyle": kw.get("strokeStyle", "solid"),
        "roughness": kw.get("roughness", 1),
        "opacity": kw.get("opacity", 100),
        "seed": random.randint(1, 2**31 - 1),
        "version": 1, "versionNonce": 0,
        "index": index_str(),
        "isDeleted": False,
        "groupIds": kw.get("groupIds", []),
        "frameId": None,
        "boundElements": kw.get("boundElements", None),
        "updated": 1700000000000,
        "link": None, "locked": False,
    }

def rect(x, y, w, h, bg="transparent", **kw):
    el = base(uid(), "rectangle", x, y, w, h, backgroundColor=bg, **kw)
    el["roundness"] = kw.get("roundness", {"type": 3})
    return el

def ellipse(x, y, w, h, bg="transparent", **kw):
    el = base(uid(), "ellipse", x, y, w, h, backgroundColor=bg, **kw)
    el["roundness"] = None
    return el

def diamond(x, y, w, h, bg="transparent", **kw):
    el = base(uid(), "diamond", x, y, w, h, backgroundColor=bg, **kw)
    el["roundness"] = {"type": 2}
    return el

def txt(x, y, s, fs=24, family=5, align="center", valign="middle",
        container=None, **kw):
    lines = s.split("\n")
    lh = 1.25
    ew = max(len(l) for l in lines) * fs * 0.55
    eh = len(lines) * fs * lh
    el = base(uid(), "text", x, y, ew, eh, **kw)
    el["roundness"] = None
    el.update({"text": s, "fontSize": fs, "fontFamily": family,
               "textAlign": align, "verticalAlign": valign,
               "containerId": container, "originalText": s,
               "autoResize": True, "lineHeight": lh})
    return el

def arr(x, y, dx, dy, pts=None, start_id=None, end_id=None, **kw):
    el = base(uid(), "arrow", x, y, abs(dx), abs(dy), **kw)
    el["roundness"] = {"type": 2}
    el["points"] = pts if pts else [[0, 0], [dx, dy]]
    el["startBinding"] = {"elementId": start_id,
        "fixedPoint": kw.get("sFP", [1.0, 0.5]), "mode": "inside"} if start_id else None
    el["endBinding"] = {"elementId": end_id,
        "fixedPoint": kw.get("eFP", [0.0, 0.5]), "mode": "inside"} if end_id else None
    el["startArrowhead"] = kw.get("sHead", None)
    el["endArrowhead"] = kw.get("eHead", "arrow")
    el["elbowed"] = False
    return el

def line_el(x, y, dx, dy, **kw):
    el = base(uid(), "line", x, y, abs(dx), abs(dy), **kw)
    el["roundness"] = {"type": 2}
    el["points"] = [[0, 0], [dx, dy]]
    el.update({"startBinding": None, "endBinding": None,
               "startArrowhead": None, "endArrowhead": None})
    return el

def labeled_box(x, y, w, h, label, bg, fs=24, sc="#1e1e1e", sw=2, **kw):
    r = rect(x, y, w, h, bg=bg, strokeColor=sc, strokeWidth=sw, **kw)
    t = txt(x, y, label, fs=fs, container=r["id"], strokeColor=sc)
    r["boundElements"] = [{"id": t["id"], "type": "text"}]
    return r, t

def bind_arrow(r, a):
    if r.get("boundElements") is None:
        r["boundElements"] = []
    r["boundElements"].append({"id": a["id"], "type": "arrow"})

E = []  # all elements

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
R1Y = 40       # row 1 top
R2Y = 620      # row 2 top

# ═══════════════════════════════════════════════════════════════════════════════
# A.  VISUAL SCENE  (top-left)
# ═══════════════════════════════════════════════════════════════════════════════
Ax, Ay, Aw, Ah = 30, R1Y, 340, 500
E.append(rect(Ax, Ay, Aw, Ah, bg="#f8f9fa", strokeColor="#ced4da"))
E.append(txt(Ax + 60, Ay + 12, "Visual Scene", fs=32, strokeColor="#343a40"))

# 6 objects: big blue circle (target), small blue square, red circle, red triangle, green sq, green circle
shapes = [
    ("ellipse",   Ax+70,  Ay+80,  80, 80, "#a5d8ff", "#228be6", True),
    ("rectangle", Ax+200, Ay+95,  50, 50, "#a5d8ff", "#228be6", False),
    ("ellipse",   Ax+70,  Ay+195, 55, 55, "#ffc9c9", "#e03131", False),
    ("diamond",   Ax+200, Ay+190, 60, 60, "#ffc9c9", "#e03131", False),
    ("rectangle", Ax+70,  Ay+295, 40, 40, "#b2f2bb", "#2f9e44", False),
    ("ellipse",   Ax+200, Ay+290, 55, 55, "#b2f2bb", "#2f9e44", False),
]
for (tp, sx, sy, sw, sh, sbg, ssc, tgt) in shapes:
    el = base(uid(), tp, sx, sy, sw, sh,
              backgroundColor=sbg, strokeColor=ssc, fillStyle="solid")
    el["roundness"] = {"type": 3} if tp in ("rectangle","diamond") else None
    E.append(el)
    if tgt:
        fr = base(uid(), "rectangle", sx-12, sy-12, sw+24, sh+24,
                  strokeColor="#e03131", strokeWidth=3, strokeStyle="dashed")
        fr["roundness"] = {"type": 3}
        E.append(fr)

E.append(txt(Ax+40, Ay+380, '"the big blue sticker"', fs=24,
             strokeColor="#343a40"))
E.append(txt(Ax+20, Ay+420, "target = dashed red frame", fs=18,
             strokeColor="#868e96"))
E.append(txt(Ax+10, Ay+455, "3 contexts × 2 discriminability", fs=18,
             strokeColor="#868e96"))

# ═══════════════════════════════════════════════════════════════════════════════
# B.  RSA MODEL  (top-center)  — the core of the figure
# ═══════════════════════════════════════════════════════════════════════════════
Bx, By, Bw, Bh = 440, R1Y, 660, 500
E.append(rect(Bx, By, Bw, Bh, bg="#f8f9fa", strokeColor="#ced4da"))
E.append(txt(Bx+200, By+10, "RSA Model", fs=32, strokeColor="#343a40"))

# --- Compositional Semantics (top of model panel) ---
sem_r, sem_t = labeled_box(Bx+30, By+60, 280, 80,
    "Compositional Semantics", "#b2f2bb", fs=24, sc="#2f9e44")
E.extend([sem_r, sem_t])

# --- Right-to-left listener direction label ---
E.append(txt(Bx+340, By+65, "Listener reads right → left\n(noun outward)",
             fs=20, strokeColor="#1971c2", align="left"))

# --- Two semantics regimes side by side ---
stat_r, stat_t = labeled_box(Bx+30, By+165, 280, 70,
    'Static:  θ from full scene\n"big" = same meaning always',
    "#e9ecef", fs=18, sc="#868e96")
E.extend([stat_r, stat_t])

rec_r, rec_t = labeled_box(Bx+340, By+165, 290, 70,
    'Recursive:  θ from listener posterior\n"big" depends on prior words',
    "#d3f9d8", fs=18, sc="#2f9e44")
E.extend([rec_r, rec_t])

# Arrow: Semantics → Static / Recursive
a1 = arr(Bx+170, By+140, 0, 20, sFP=[0.5,1.0], eFP=[0.5,0.0],
         start_id=sem_r["id"], end_id=stat_r["id"])
bind_arrow(sem_r, a1); bind_arrow(stat_r, a1); E.append(a1)

a2 = arr(Bx+310, By+140, 175, 20,
         start_id=sem_r["id"], end_id=rec_r["id"],
         sFP=[1.0,1.0], eFP=[0.5,0.0])
bind_arrow(sem_r, a2); bind_arrow(rec_r, a2); E.append(a2)

# --- Literal Listener ---
ll_r, ll_t = labeled_box(Bx+40, By+290, 240, 80,
    "Literal Listener  L₀\nP(o | u)", "#a5d8ff", fs=24, sc="#1971c2")
E.extend([ll_r, ll_t])

# Arrow: semantics zone → listener
a_to_ll = arr(Bx+170, By+235, 0, 50, strokeColor="#868e96")
E.append(a_to_ll)

# --- Pragmatic Speaker ---
sp_r, sp_t = labeled_box(Bx+370, By+290, 250, 80,
    "Pragmatic Speaker  S₁\nS(u | o*)", "#ffd8a8", fs=24, sc="#e8590c")
E.extend([sp_r, sp_t])

# Arrow: Listener → Speaker  (forward pass)
a_ll_sp = arr(Bx+280, By+330, 85, 0,
              start_id=ll_r["id"], end_id=sp_r["id"],
              sFP=[1.0,0.5], eFP=[0.0,0.5], strokeWidth=3)
bind_arrow(ll_r, a_ll_sp); bind_arrow(sp_r, a_ll_sp); E.append(a_ll_sp)

# --- Incremental loop arrow (curved, orange, above) ---
inc_arr = arr(Bx+495, By+285, -230, -30,
              start_id=sp_r["id"], end_id=ll_r["id"],
              sFP=[0.5,0.0], eFP=[0.5,0.0],
              strokeColor="#e8590c", strokeWidth=3)
inc_arr["points"] = [[0, 0], [-115, -50], [-230, 0]]
bind_arrow(sp_r, inc_arr); bind_arrow(ll_r, inc_arr); E.append(inc_arr)

E.append(txt(Bx+210, By+248, "incremental: choose word-by-word  →",
             fs=20, strokeColor="#e8590c"))

# --- Global label (below boxes) ---
E.append(txt(Bx+200, By+385,
    "global: evaluate full utterances at once",
    fs=20, strokeColor="#1971c2"))

# --- Speaker produces left-to-right ---
E.append(txt(Bx+340, By+105, "Speaker produces left → right\n(word by word)",
             fs=20, strokeColor="#e8590c", align="left"))

# --- Parameters ---
E.append(txt(Bx+100, By+430,
    "α = rationality    β = LM-prior weight    b = subjectivity bias",
    fs=18, strokeColor="#868e96"))

# --- Example: "big blue sticker" with arrows showing directions ---
ex_y = By + 460
E.append(txt(Bx+70, ex_y, 'Example:   "big   blue   sticker"',
             fs=20, strokeColor="#555555"))
# Left→Right arrow (speaker)
E.append(arr(Bx+85, ex_y - 8, 180, 0, strokeColor="#e8590c",
             strokeWidth=2, eHead="arrow"))
E.append(txt(Bx+105, ex_y - 30, "speaker →", fs=14, strokeColor="#e8590c"))
# Right→Left arrow (listener)
E.append(arr(Bx+355, ex_y - 8, -180, 0, strokeColor="#1971c2",
             strokeWidth=2, eHead="arrow"))
E.append(txt(Bx+270, ex_y - 30, "← listener", fs=14, strokeColor="#1971c2"))


# ═══════════════════════════════════════════════════════════════════════════════
# C.  EXPERIMENTS  (top-right)
# ═══════════════════════════════════════════════════════════════════════════════
Cx, Cy = 1170, R1Y

# --- Exp 1: Slider Rating ---
e1_r, e1_t = labeled_box(Cx, Cy, 380, 220,
    'Exp 1: Slider Rating\n\n"big blue"  vs  "blue big"\n\nrate preference  0 ———— 1',
    "#fff5f5", fs=22, sc="#e03131")
E.extend([e1_r, e1_t])

# --- Exp 2: Free Production ---
e2_r, e2_t = labeled_box(Cx, Cy + 260, 380, 240,
    'Exp 2: Free Production\n\nDescribe the target freely:\n"big" · "big blue" · "blue big"\n"big blue round" · ...\n\n→ 15 possible utterance forms',
    "#f3f0ff", fs=22, sc="#6741d9")
E.extend([e2_r, e2_t])

# ── Arrows Scene → Model → Experiments ────────────────────────────────────
a_sm = arr(Ax+Aw+5, Ay+Ah/2, 65, 0, strokeWidth=3)
E.append(a_sm)
E.append(txt(Ax+Aw+10, Ay+Ah/2-30, "encode", fs=16, strokeColor="#868e96"))

a_me1 = arr(Bx+Bw+5, Cy+110, 65, 0, strokeWidth=3)
E.append(a_me1)
a_me2 = arr(Bx+Bw+5, Cy+380, 65, 0, strokeWidth=3)
E.append(a_me2)
E.append(txt(Bx+Bw+8, Cy+240, "predict", fs=16, strokeColor="#868e96"))


# ═══════════════════════════════════════════════════════════════════════════════
# D.  2×2 MODEL COMPARISON  (bottom-left)
# ═══════════════════════════════════════════════════════════════════════════════
Dx, Dy, Dw, Dh = 30, R2Y, 720, 360
E.append(rect(Dx, Dy, Dw, Dh, bg="#f8f9fa", strokeColor="#ced4da"))
E.append(txt(Dx+100, Dy+12,
    "2 × 2  Model Comparison  (PSIS-LOO)", fs=28, strokeColor="#343a40"))

# Column headers
c1x, c2x = Dx+250, Dx+510
E.append(txt(c1x-20, Dy+60, "Static Semantics\nθ fixed from scene", fs=20,
             strokeColor="#555555"))
E.append(txt(c2x-20, Dy+60, "Recursive Semantics\nθ updates with posterior", fs=20,
             strokeColor="#2f9e44"))

# Row headers
E.append(txt(Dx+20, Dy+140, "Global Speaker\nevaluate full\nutterance", fs=20,
             strokeColor="#555555", align="left"))
E.append(txt(Dx+20, Dy+260, "Incremental Speaker\nchoose word\nby word", fs=20,
             strokeColor="#555555", align="left"))

# Cells
cw, ch = 200, 80
cells = [
    (c1x, Dy+135, "#e9ecef", "Global × Static",    "#868e96", 2, ""),
    (c2x, Dy+135, "#e9ecef", "Global × Recursive",  "#868e96", 2, ""),
    (c1x, Dy+250, "#ffec99", "Incr. × Static",      "#f08c00", 2, ""),
    (c2x, Dy+250, "#b2f2bb", "Incr. × Recursive ★", "#2f9e44", 3, "best"),
]
for (cx_, cy_, bg_, lbl_, sc_, sw_, note_) in cells:
    cr, ct = labeled_box(cx_, cy_, cw, ch, lbl_, bg_, fs=20, sc=sc_, sw=sw_)
    E.extend([cr, ct])

# "best fit" callout
E.append(txt(Dx+570, Dy+310, "← best fit (production)", fs=18,
             strokeColor="#2f9e44"))


# ═══════════════════════════════════════════════════════════════════════════════
# E.  RESEARCH QUESTIONS + KEY FINDINGS  (bottom-right)
# ═══════════════════════════════════════════════════════════════════════════════
Kx, Ky, Kw, Kh = 810, R2Y, 740, 360
E.append(rect(Kx, Ky, Kw, Kh, bg="#ebfbee", strokeColor="#2f9e44", strokeWidth=3))
E.append(txt(Kx+190, Ky+10,
    "Research Questions & Findings", fs=28, strokeColor="#2f9e44"))

lines = [
    ("Q1", "Can ordering arise from composition alone,",
           "without incremental production?",
           "Yes — simulation shows recursive composition",
           "yields a baseline size-first preference"),
    ("Q2", "Does incremental (word-by-word) production",
           "add value over global evaluation?",
           "Yes — incremental speaker wins in both",
           "slider and production datasets"),
    ("Q3", "Does recursive semantics help further?",
           "Is it contingent on speaker architecture?",
           "Yes, but only with an incremental speaker",
           "→  interaction: the two are synergistic"),
]

y0 = Ky + 65
for i, (q, q1, q2, f1, f2) in enumerate(lines):
    yy = y0 + i * 95
    # Question label
    E.append(txt(Kx+15, yy, q, fs=24, strokeColor="#228be6", align="left"))
    # Question text
    E.append(txt(Kx+60, yy-5, q1, fs=20, strokeColor="#343a40", align="left"))
    E.append(txt(Kx+60, yy+20, q2, fs=20, strokeColor="#343a40", align="left"))
    # Finding (indented, green)
    E.append(txt(Kx+80, yy+48, "→ " + f1, fs=20,
                 strokeColor="#2f9e44", align="left"))
    E.append(txt(Kx+100, yy+73, f2, fs=18,
                 strokeColor="#2f9e44", align="left"))


# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════════════════
doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "python-generator",
    "elements": E,
    "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
    "files": {},
}

out = "/Users/heningwang/Documents/GitHub/numpyro_adjective_modelling/10-writing/figures/overview_figure.excalidraw"
with open(out, "w") as f:
    json.dump(doc, f, indent=2)

print(f"Saved {out}")
print(f"Total elements: {len(E)}")
