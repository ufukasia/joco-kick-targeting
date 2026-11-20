# -*- coding: utf-8 -*-
"""Kanit defterine: holdout2'den Asama 1'de kullanilmis 3 TSPLIB problemi cikarildiginda V1-V4
(kesifsel duyarlilik; yeni kosu yok). Idempotent: ek.anahtar ayniysa kayit yeniden eklenmez.

Kaynak: results/kick_targeting/holdout2_10s_confirm_no_tsplib.json
        (results/kick_targeting/analysis_sensitivity_no_tsplib.py ile uretilir)

    py -3.13 evidence/scripts/build_sens_tsplib.py
"""
import json
from pathlib import Path

W = Path(__file__).resolve().parents[2]
ROOT = W.parents[1]
EVF = W / "evidence" / "evidence.json"
SRC = "results/kick_targeting/holdout2_10s_confirm_no_tsplib.json"

db = json.loads(EVF.read_text(encoding="utf-8"))
recs = db["kayitlar"]
have = {(r.get("ek") or {}).get("anahtar"): r["id"] for r in recs}
nxt = max(int(r["id"][1:]) for r in recs if r["id"].startswith("E")) + 1
ids = {}
for h in json.loads((ROOT / SRC).read_text(encoding="utf-8")):
    key = f"sens.no_tsplib.{h['id']}"
    if key in have:
        ids[h["id"]] = have[key]
        continue
    rid = f"E{nxt}"
    nxt += 1
    recs.append({
        "id": rid, "metrik": "dogrulayici_fark_ortalama_51", "yontem": f"{h['a']} - {h['b']}",
        "veri_seti": "holdout2_10s, TSPLIB3 haric (51 problem; kesifsel duyarlilik)", "deger": h["mean"],
        "birim": "puan", "kaynak": "yazilim", "kaynak_dosya": SRC, "kaynak_konum": f"[id={h['id']}].mean",
        "atif": None, "durum": "gecerli",
        "ek": {"uretici": "build_sens_tsplib", "anahtar": key, "p": h["p"], "test": h["test"],
               "rejected": h["rejected"], "n_inst": h["n_inst"]}})
    ids[h["id"]] = rid
EVF.write_text(json.dumps(db, ensure_ascii=False, indent=1), encoding="utf-8")
print(ids)
