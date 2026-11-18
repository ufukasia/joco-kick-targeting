#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figür veri dosyalarını (figures/data/*.dat) gerçek sonuç JSON'larından türetir.

Yeniden üretmek için:  python figures/scripts/make_fig_data.py
Salt okunur; yeni deney BAŞLATMAZ.
"""
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
ROOT = WS.parents[1]
RESULTS = ROOT / "results"
DATADIR = WS / "figures" / "data"
DATADIR.mkdir(parents=True, exist_ok=True)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FOREST_SRC = "diversification_synthetic_100_summary.json"
FOCUSED_SRC = {
    "synthetic": "focused_synthetic_100_summary.json",
    "tsplib": "focused_tsplib_9x10_summary.json",
    "world": "focused_world_10x10_summary.json",
}
KICK_SRC = ["focused_tsplib_9x10_summary.json", "focused_world_10x10_summary.json"]
KICK_METHODS = ["random", "suspicious", "suspicious_reactive", "focused_bandit"]


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def write_dat(ad, header, rows):
    # pgfplots ilk satiri baslik (sutun adlari) olarak okur; yorum satiri koyma.
    lines = [" ".join(header)]
    for r in rows:
        lines.append(" ".join(str(x) for x in r))
    (DATADIR / ad).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"yazildi: figures/data/{ad} ({len(rows)} satir)")


def fig_synthetic_forest():
    d = load(FOREST_SRC)
    comps = [c for c in d["comparisons"] if c["opponent"] == "random"]
    comps.sort(key=lambda c: c["mean_quality_difference"])  # artan; pgfplots y=sirali kategorik
    rows = []
    per_class = {}
    for y, c in enumerate(comps):
        lo, hi = c["bootstrap_95_ci"]
        cls = c.get("classification") or "inconclusive"
        key = {"significant_benefit": "benefit", "significant_harm": "harm"}.get(cls, "inconclusive")
        rows.append((c["method"], f"{c['mean_quality_difference']:.5f}",
                     f"{lo:.5f}", f"{hi:.5f}", cls, c["method"] == "suspicious"))
        per_class.setdefault(key, []).append((y, f"{c['mean_quality_difference']:.5f}",
                                              f"{lo:.5f}", f"{hi:.5f}"))
    for key, lst in per_class.items():
        write_dat(f"fig_synthetic_forest_{key}.dat", ["y", "mean_diff", "ci_lo", "ci_hi"], lst)
    write_dat("fig_synthetic_forest.dat",
              ["method", "mean_diff", "ci_lo", "ci_hi", "classification", "is_proposed"], rows)


def fig_focused_ci():
    for suite, fname in FOCUSED_SRC.items():
        d = load(fname)
        comps = [c for c in d["comparisons"] if c["opponent"] == "random"]
        order = {m: i for i, m in enumerate(
            [m for m in d["quality_ranking"] if m != "random"])}
        comps.sort(key=lambda c: order.get(c["method"], 99))
        rows = []
        for c in comps:
            lo, hi = c["bootstrap_95_ci"]
            key = {"suspicious": "susp", "suspicious_reactive": "sreac", "focused_bandit": "fband"}.get(c["method"], c["method"])
            y = order[c["method"]]
            rows.append((key, f"{c['mean_quality_difference']:.5f}",
                         f"{lo:.5f}", f"{hi:.5f}", c.get("classification") or "inconclusive"))
            # method basina ayri tek-satir dosya: scatter/classes yerine duz \addplot
            write_dat(f"fig_focused_ci_{suite}_{key}.dat",
                      ["y", "mean_diff", "ci_lo", "ci_hi"],
                      [(y, f"{c['mean_quality_difference']:.5f}", f"{lo:.5f}", f"{hi:.5f}")])
        write_dat(f"fig_focused_ci_{suite}.dat",
                  ["method", "mean_diff", "ci_lo", "ci_hi", "classification"], rows)


def fig_kick_activity():
    kick_rows, ela_rows = [], []
    for fname in KICK_SRC:
        d = load(fname)
        suite = d["suite"]
        for p in d["per_instance"]:
            kick_rows.append([suite, p["instance"], p["n"]] +
                             [f"{p[m]['kicks']:.2f}" for m in KICK_METHODS])
            ela_rows.append([suite, p["instance"], p["n"]] +
                            [f"{p[m]['elapsed']:.2f}" for m in KICK_METHODS])
    write_dat("fig_kick_activity_kicks.dat",
              ["suite", "instance", "n"] + KICK_METHODS, kick_rows)
    write_dat("fig_kick_activity_elapsed.dat",
              ["suite", "instance", "n"] + KICK_METHODS, ela_rows)
    # turetilmis sayim (kanit idleri evidence.json'da; notlar notes/derivations.md'de)
    zero = sum(1 for r in kick_rows if all(float(x) == 0.0 for x in r[3:]))
    print(f"sifir-kick problem sayisi (4 yontemin tumunde): {zero} / {len(kick_rows)}")


def main():
    fig_synthetic_forest()
    fig_focused_ci()
    fig_kick_activity()


if __name__ == "__main__":
    main()
