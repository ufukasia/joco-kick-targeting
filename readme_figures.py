# -*- coding: utf-8 -*-
"""README figures (SVG, light + dark) computed from the stored run records.

    py -3.13 readme_figures.py      -> release_meta/docs/*.svg  (copied into the repo by make_release.py)
"""
import json
from html import escape
from pathlib import Path

import numpy as np

import v2_analysis as A

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "release_meta" / "docs"
R = A.OUT_ROOT

THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", blue="#2a78d6", orange="#eb6834", gray="#b5b3ab"),
    "dark": dict(bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", blue="#3987e5", orange="#d95926", gray="#6b6a65"),
}
FONT = "system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
EXC, RND, GNN = "excess_global@0.06:0.5", "random_global", "gnn_global@0.04:0.5"
NAMES = {
    "excess_global@0.06:0.5": "Excess-length proxy (untrained)",
    "knn_global@0.04:0.75": "kNN rank (untrained)",
    "gnn_global@0.04:0.5": "GNN trained on LKH-3 tours",
    "gain_global": "MLP trained on kick gain",
    "cheap_global": "MLP trained on edge membership",
    "linprior_global": "Static linear prior",
    "onlinewarm_global": "Online learner (in-run)",
}


class Svg:
    def __init__(self, w, h, t):
        self.w, self.h, self.t, self.el = w, h, t, []

    def text(self, x, y, s, size=13, fill="ink", anchor="start", weight=400):
        self.el.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
                       f'fill="{self.t[fill]}" text-anchor="{anchor}">{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke="grid", width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.el.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                       f'stroke="{self.t[stroke]}" stroke-width="{width}"{d}/>')

    def rect(self, x, y, w, h, fill, rx=0):
        self.el.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" '
                       f'rx="{rx}" fill="{self.t[fill]}"/>')

    def dot(self, x, y, r, fill):
        self.el.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{self.t[fill]}" '
                       f'stroke="{self.t["bg"]}" stroke-width="2"/>')

    def save(self, path, title):
        body = "\n".join(self.el)
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}" role="img">\n<title>{escape(title)}</title>\n'
            f'<rect width="100%" height="100%" rx="8" fill="{self.t["bg"]}"/>\n{body}\n</svg>\n', encoding="utf-8")


def fmt(v):
    return f"{v:+.2f}".replace("+", "+").replace("-", "−")


def ci_rows(svg, rows, x0, x1, y0, step, lo, hi, label_w, ticks):
    """rows: (label, mean, ci_lo, ci_hi, color, note). Horizontal dot + 95% CI."""
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    yb = y0 + step * len(rows)
    for tv in ticks:
        svg.line(X(tv), y0 - 6, X(tv), yb - step / 2 + 8, "grid")
        svg.text(X(tv), yb - step / 2 + 24, fmt(tv) if tv else "0", 11, "muted", "middle")
    svg.line(X(0), y0 - 10, X(0), yb - step / 2 + 8, "axis", 1.5)
    for i, (lab, m, a, b, col, note) in enumerate(rows):
        y = y0 + i * step
        svg.text(x0 - 14, y + 4.5, lab, 13, "ink" if col != "gray" else "ink2", "end", 600 if col != "gray" else 400)
        svg.line(X(a), y, X(b), y, col, 2)
        svg.dot(X(m), y, 6, col)
        svg.text(x1 + 12, y + 4.5, f"{fmt(m)}", 13, "ink", "start", 600 if col != "gray" else 400)
        if note:
            svg.text(x1 + 60, y + 4.5, note, 11, "muted")
    return X, yb


def fig_hero(t):
    rows = []
    for s, lab in NAMES.items():
        r = A.paired("holdout2_10s", s, RND)
        rows.append((lab, r["mean"], r["ci95"][0], r["ci95"][1], r["wins_first"]))
    rows.sort(key=lambda r: r[1])
    svg = Svg(860, 450, t)
    svg.text(24, 38, "An untrained edge-length signal matches trained models — learning online hurts", 19, "ink", weight=700)
    svg.text(24, 62, "Final tour gap vs random kicks, 54 held-out TSP instances × 10 seeds, 10 s each "
                     "(mean, 95% bootstrap CI). Left of 0 = better.", 13, "ink2")
    out = []
    for lab, m, a, b, w in rows:
        col = "blue" if "Excess" in lab else ("orange" if "Online" in lab else "gray")
        out.append((lab, m, a, b, col, f"better on {w}/54"))
    X, yb = ci_rows(svg, out, 300, 640, 108, 38, -0.3, 0.3, 270, [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    svg.text(X(-0.15), yb + 30, "← lower gap than random", 12, "blue", "middle", 600)
    svg.text(X(0.15), yb + 30, "higher gap than random →", 12, "orange", "middle", 600)
    svg.text(24, 436, "Preregistered (PROTOCOL_v2, SHA-256 + OpenTimestamps): proxy < random, proxy non-inferior to GNN "
                      "and gain-MLP (margin 0.05), online > static — all four confirmed.", 11, "muted")
    return svg


def fig_per_instance(t):
    data, _ = A.per_instance("holdout2_10s")
    d = np.sort(np.array([data[k][EXC] - data[k][GNN] for k in data]))
    n, wins = len(d), int((d < 0).sum())
    svg = Svg(860, 330, t)
    svg.text(24, 38, f"The proxy gives a lower gap than the GNN on {wins} of {n} instances", 19, "ink", weight=700)
    svg.text(24, 62, "Per-instance difference in final gap, excess-length proxy − GNN (percentage points), sorted. "
                     "Each bar = one instance (10 seeds).", 13, "ink2")
    x0, x1, yt, yb = 70, 836, 90, 290
    lo, hi = min(-1.0, d.min()), max(0.3, d.max())
    Y = lambda v: yt + (hi - v) / (hi - lo) * (yb - yt)
    for tv in [-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2]:
        if lo <= tv <= hi:
            svg.line(x0, Y(tv), x1, Y(tv), "grid")
            svg.text(x0 - 8, Y(tv) + 4, fmt(tv) if tv else "0", 11, "muted", "end")
    bw = (x1 - x0) / n
    for i, v in enumerate(d):
        col = "blue" if v < 0 else "orange"
        y, h = (Y(0), Y(v) - Y(0)) if v < 0 else (Y(v), Y(0) - Y(v))
        svg.rect(x0 + i * bw + 1, y, bw - 2, h, col, 1.5)
    svg.line(x0, Y(0), x1, Y(0), "axis", 1.5)
    svg.text(x0 + 0.35 * (x1 - x0), Y(-0.5), f"■ proxy better ({wins})", 12, "blue", weight=600)
    svg.text(x1, Y(hi) + 12, f"GNN better ({n - wins}) ■", 12, "orange", "end", 600)
    return svg


def fig_length(t):
    a = json.loads((R / "holdout2_length_10s_analysis.json").read_text(encoding="utf-8"))["pairs"]
    L, E = "length_global@0.06:0.5", EXC
    spec = [(f"{L} - {RND}", "Raw edge length − random", "blue"),
            (f"{E} - {RND}", "Excess-length proxy − random", "blue"),
            (f"{L} - {E}", "Raw length − excess proxy", "gray")]
    svg = Svg(860, 270, t)
    svg.text(24, 38, "Even simpler: plain edge length does as well as the proxy", 19, "ink", weight=700)
    svg.text(24, 62, "Exploratory follow-up on the same 54 instances (one session, paired): target the longest tour "
                     "edges, no local-scale correction.", 13, "ink2")
    rows = []
    for k, lab, col in spec:
        r = a[k]
        rows.append((lab, r["mean"], r["ci95"][0], r["ci95"][1], col,
                     f"p = {r['p_wilcoxon']:.1g}" if r["p_wilcoxon"] < 0.01 else f"p = {r['p_wilcoxon']:.2f} (no difference)"))
    ci_rows(svg, rows, 300, 640, 108, 38, -0.3, 0.1, 270, [-0.3, -0.2, -0.1, 0, 0.1])
    return svg


def fig_online(t):
    m = json.loads((R / "online_reward_mechanism.json").read_text(encoding="utf-8"))["comparison_online_minus_static"]
    data, _ = A.per_instance("holdout2_10s")
    g_on = np.mean([data[k]["onlinewarm_global"] for k in data])
    g_st = np.mean([data[k]["linprior_global"] for k in data])
    panels = [
        ("Kick penalty (−reward)", "what the learner minimises", -m["mean_r"]["mean_static"], -m["mean_r"]["mean_online"], "{:.1f}"),
        ("Neutral kicks", "undone by local search", 100 * m["frac_neutral"]["mean_static"], 100 * m["frac_neutral"]["mean_online"], "{:.0f}%"),
        ("Final gap (%)", "what we care about", g_st, g_on, "{:.2f}"),
    ]
    svg = Svg(860, 300, t)
    svg.text(24, 38, "Why online learning hurts: it learns to make harmless kicks", 19, "ink", weight=700)
    svg.text(24, 62, "Static prior vs the same scorer updated during the run. The learner cuts its penalty by picking kicks "
                     "that local search simply undoes.", 13, "ink2")
    pw, x = 262, 24
    for title, sub, vs, vo, f in panels:
        svg.text(x, 100, title, 14, "ink", weight=700)
        svg.text(x, 118, sub, 12, "muted")
        vmax = max(abs(vs), abs(vo))
        for j, (lab, v, col) in enumerate([("static", vs, "blue"), ("online", vo, "orange")]):
            y = 146 + j * 50
            w = (pw - 90) * abs(v) / vmax
            svg.text(x, y + 16, lab, 12, "ink2")
            svg.rect(x + 50, y, w, 24, col, 4)
            svg.text(x + 50 + w + 6, y + 17, f.format(v), 13, "ink", weight=600)
        x += pw + 24
    svg.text(24, 262, "Penalty and neutral share: 12 development instances × 3 seeds (online higher on 12/12). "
                      "Final gap: 54 held-out instances, preregistered test V1,", 11, "muted")
    svg.text(24, 278, "+0.291 points, p = 1.6e−10. The learner's pool drifts to short edges (cos(w₀, w_final) = −0.62).", 11, "muted")
    return svg


def fig_budget(t):
    v = json.loads((R / "v4_robustness_analysis.json").read_text(encoding="utf-8"))["S2_60s"]
    rows = []
    for other, lab0, col in [(RND, "Proxy − random", "blue"), (GNN, "Proxy − GNN", "blue")]:
        for cp in ("10.0", "30.0", "60.0"):
            r = next(c for c in v[cp]["contrasts"] if c["pair"] == f"{EXC} - {other}")
            rows.append((f"{lab0}, {cp[:-2]} s", r["mean"], r["ci95"][0], r["ci95"][1], col,
                         f"{r['wins_first']}/54 · p_Holm {r['p_holm']:.1g}"))
    svg = Svg(860, 392, t)
    svg.text(24, 38, "With a 6× longer budget the advantage narrows but stays below zero", 19, "ink", weight=700)
    svg.text(24, 62, "Exploratory 60 s session (54 held-out instances × 5 seeds); 10 s and 30 s are checkpoints of the same runs. "
                     "Left of 0 = proxy better.", 13, "ink2")
    ci_rows(svg, rows, 300, 640, 108, 38, -0.35, 0.05, 270, [-0.3, -0.2, -0.1, 0])
    svg.text(24, 380, "Registered before running (PROTOCOL_v3_exploratory.json, SHA-256 + OpenTimestamps). "
                      "Does not change the confirmatory V1–V4 decisions.", 11, "muted")
    return svg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figs = {"hero": (fig_hero, "Kick targeting vs random"), "per_instance": (fig_per_instance, "Proxy vs GNN per instance"),
            "length_arm": (fig_length, "Raw edge length arm"), "online": (fig_online, "Online learning mechanism"),
            "budget": (fig_budget, "Advantage at longer budgets")}
    for name, (fn, title) in figs.items():
        for mode, t in THEMES.items():
            fn(t).save(OUT / f"{name}_{mode}.svg", title)
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
