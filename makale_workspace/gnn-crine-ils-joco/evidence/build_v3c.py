# -*- coding: utf-8 -*-
"""Adds exploratory length-arm records (review Ö4) from holdout2_length_10s_analysis.json."""
import json
from pathlib import Path
W = Path(__file__).resolve().parents[1]; ROOT = W.parents[1]
SRC = "results/kick_targeting/holdout2_length_10s_analysis.json"
a = json.loads((ROOT / SRC).read_text(encoding="utf-8"))
ev = json.loads((W / "evidence/evidence.json").read_text(encoding="utf-8"))
ev["kayitlar"] = [r for r in ev["kayitlar"] if r.get("ek", {}).get("alt_uretici") != "build_v3c"]
nid = max(int(r["id"][1:]) for r in ev["kayitlar"] if r["id"].startswith("E")) + 1
def add(metrik, yontem, deger, birim, konum, ek):
    global nid
    ev["kayitlar"].append({"id": f"E{nid}", "metrik": metrik, "yontem": yontem, "veri_seti": "holdout2 (54 problem x 10 tohum, 10 sn)",
        "deger": float(deger), "birim": birim, "kaynak": "yazilim", "kaynak_dosya": SRC, "kaynak_konum": konum, "atif": None,
        "durum": "gecerli", "ek": {"uretici": "build_v3", "anahtar": f"v3c.{metrik}.{yontem}", "alt_uretici": "build_v3c", "asama": 3, "not_": "kesifsel, protokol sonrasi (hakem O4)", **ek}})
    nid += 1
names = {"length_global@0.06:0.5 - random_global": "uzunluk-rastgele", "length_global@0.06:0.5 - excess_global@0.06:0.5": "uzunluk-fazla",
         "excess_global@0.06:0.5 - random_global": "fazla-rastgele"}
for pk, lab in names.items():
    r = a["pairs"][pk]
    add(f"uzunluk_kolu_{lab}_ortalama_fark", pk, r["mean"], "puan", f"pairs['{pk}']['mean']",
        {"p_wilcoxon": r["p_wilcoxon"], "ci95": r["ci95"], "hl": r["hl"], "median": r["median"], "wins_first": r["wins_first"],
         "ties": r["ties"], "wins_second": r["wins_second"], "n_inst": r["n_inst"]})
    for s, v in r["by_stratum"].items():
        add(f"uzunluk_kolu_{lab}_{s}", pk, v["mean"], "puan", f"pairs['{pk}']['by_stratum']['{s}']", {"n": v["n"], "wins_first": v["wins_first"]})
for s, c in a["counters"].items():
    add("uzunluk_kolu_vurus_hizi", s, c["kicks_per_s"], "vurus/sn", f"counters['{s}']['kicks_per_s']", {"accept_rate": c["accept_rate"]})
(W / "evidence/evidence.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
print("added up to", f"E{nid-1}", "total", len(ev["kayitlar"]))
