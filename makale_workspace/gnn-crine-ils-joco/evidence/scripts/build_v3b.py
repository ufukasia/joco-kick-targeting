# -*- coding: utf-8 -*-
"""Asama 3 ek kanitlar (hakem turu 1): cevrimici odul mekanizmasi, HL tahminleri, zaman damgasi ilerleyisi.

Kaynak (yazilim koku = tsplib/):
  results/kick_targeting/online_reward_mechanism.json  (gelistirme: vlsi_primary 12 problem x 3 tohum; kesifsel)
  results/kick_targeting/holdout2_10s_hl.json           (V1-V4 Hodges-Lehmann + kesin GA + bootstrap GA)
  results/kick_targeting/PROTOCOL_DEVIATIONS.log        (12:51 ots stamp, 15:01 ots upgrade @ 384/540)
  evidence/logs/test_online_reward.log                  (pytest ciktisi: test_online_reward.py, 5 passed)
Cikti: evidence.json kayitlari (build_v3 Defter'i; ek.alt_uretici = build_v3b; anahtar 'v3b.*', idempotent),
       evidence/tables/v3_online_reward.tex

    py -3.13 evidence/scripts/build_v3b.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v3 as B  # noqa: E402  (Defter, bicim yardimcilari)

KT, WS, ROOT = B.KT, B.WS, B.ROOT
ORM = "results/kick_targeting/online_reward_mechanism.json"
HL = "results/kick_targeting/holdout2_10s_hl.json"
LOG = "results/kick_targeting/PROTOCOL_DEVIATIONS.log"


def main():
    D = B.Defter(B.EVID)
    ek0 = {"alt_uretici": "build_v3b"}
    # ---- (1) cevrimici odul mekanizmasi
    o = json.loads((ROOT / ORM).read_text(encoding="utf-8"))
    C = o["comparison_online_minus_static"]
    ids = {}
    for k, c in C.items():
        ids[k] = D.ekle(f"v3b.reward.{k}", f"cevrimici_odul_{k}", "onlinewarm vs linprior (online - static)",
                        "vlsi_primary dev (12 problem x 3 tohum)", c["mean_online"], "", ORM,
                        f"comparison_online_minus_static['{k}']", **ek0, mean_static=c["mean_static"],
                        mean_diff=c["mean_diff"], online_higher=c["online_higher_count"], n=c["n"],
                        p_wilcoxon=c["p_wilcoxon"], not_="kesifsel, gelistirme setinde (holdout2 degil)")
    R = o["records"]
    for a, lab in (("onlinewarm", "cevrimici"), ("linprior", "sabit")):
        kk = np.array([r["arms"][a]["kicks"] for r in R], float)
        ac = np.array([r["arms"][a]["accepted"] for r in R], float)
        ids[f"run.{a}"] = D.ekle(f"v3b.reward.run.{a}", "vurus_per_kosum", a, "vlsi_primary dev (12 x 3)",
                                 float(kk.mean()), "vurus", ORM, "records[*].arms[a].kicks ortalamasi (36 kosum)",
                                 **ek0, accepted_mean=float(ac.mean()), n_kosum=len(R))
    ids["meta"] = D.ekle("v3b.reward.meta", "odul_geri_kazanim_hatasi", "online_targeting.py", "vlsi_primary dev",
                         o["max_rel_err_sum_r_vs_delta_b_bias_over_3"], "goreli", ORM,
                         "max_rel_err_sum_r_vs_delta_b_bias_over_3", **ek0, budget_s=o["budget_s"], seeds=o["seeds"],
                         n_instances=len(o["instances"]), neutral_tol=o["neutral_tol_r"], windows_s=o["windows_s"],
                         arms=o["arms"])
    # yazilim dogrulama notu
    lg = (WS / "evidence" / "logs" / "test_online_reward.log").read_text(encoding="utf-8")
    m = re.search(r"(\d+) passed", lg)
    ids["test"] = D.ekle("v3b.test.online_reward", "birim_testi_gecen", "test_online_reward.py", "-",
                         int(m.group(1)), "test", "evidence/logs/test_online_reward.log", "pytest ozet satiri",
                         **ek0, failed=len(re.findall("FAILED", lg)),
                         testler=re.findall(r"PASSED \S+::(\S+)", lg),
                         sha256_test=re.search(r"test_online_reward.py sha256 (\w+)", lg).group(1),
                         sha256_online_targeting=re.search(r"online_targeting.py sha256 (\w+)", lg).group(1),
                         tur="yazilim dogrulama notu (sayisal sonuc degil)")
    # ---- (2) HL tahminleri
    h = json.loads((ROOT / HL).read_text(encoding="utf-8"))
    for x in h["hypotheses"]:
        ids[f"hl.{x['id']}"] = D.ekle(f"v3b.hl.{x['id']}", "hodges_lehmann", f"{x['a']} - {x['b']}", "holdout2",
                                      x["hl_pseudomedian"], "yuzde_puani", HL, f"hypotheses[id={x['id']}]", **ek0,
                                      **{k: v for k, v in x.items() if k not in ("a", "b", "hl_pseudomedian")},
                                      yontem_aciklama=h["method"])
    # ---- (3) zaman damgasi ilerleyisi
    log = (ROOT / LOG).read_text(encoding="utf-8")
    l1 = re.search(r"^\[2026-11-17 12:51\].*$", log, re.M).group(0)
    l2 = re.search(r"^\[2026-11-17 15:01\].*$", log, re.M).group(0)
    prog = re.search(r"holdout2 progress (\d+)/(\d+)", l2)
    blk = int(re.search(r"BitcoinBlockHeaderAttestation\((\d+)\)", l2).group(1))
    ids["ots"] = D.ekle("v3b.ots.progress", "ots_upgrade_aninda_holdout2_ilerleyisi", "OpenTimestamps", "holdout2",
                        int(prog.group(1)), "gorev", LOG, "[2026-11-17 12:51] ve [2026-11-17 15:01] satirlari", **ek0,
                        toplam_gorev=int(prog.group(2)), bitcoin_blok=blk, stamp_zamani="2026-11-17 12:51",
                        upgrade_zamani="2026-11-17 15:01", holdout2_baslangic="2026-11-17 12:52:01 (config.json)",
                        satir_stamp=l1[:300], satir_upgrade=l2,
                        not_="damga holdout2 kosusundan once alindi; Bitcoin onayi kosu %71'deyken (384/540) geldi, "
                             "sonuclar incelenmeden")
    D.kaydet(json.loads(B.EVID.read_text(encoding="utf-8")).get("deney_meta_v3", {}))
    tab_reward(C, R)
    print({k: v for k, v in ids.items() if not k.startswith(("frac_neutral@", "mean_r@"))})
    allid = sorted(set(ids.values()), key=lambda s: int(s[1:]))
    print(f"build_v3b kayitlari: {allid[0]}..{allid[-1]} ({len(allid)})")


def tab_reward(C, R):
    tr, trp = B.tr, B.trp
    def row(lab, k, sc=1.0, d=3):
        c = C[k]
        return (f"{lab} & {tr(sc * c['mean_online'], d)} & {tr(sc * c['mean_static'], d)} & "
                f"{c['online_higher_count']}/{c['n']} & {trp(c['p_wilcoxon'])} \\\\")
    kk = {a: np.mean([r["arms"][a]["kicks"] for r in R]) for a in ("onlinewarm", "linprior")}
    ac = {a: np.mean([r["arms"][a]["accepted"] for r in R]) for a in ("onlinewarm", "linprior")}
    rows = [row("Vuruş başı ödül", "mean_r", d=2),
            row("Nötr vuruş [\\%]", "frac_neutral", 100, 1),
            row("Üç kesik kenar geri gelir [\\%]", "frac_all3_cut_edges_restored", 100, 1),
            row("İyileştiren vuruş [\\%]", "frac_improving", 100, 2),
            row("Kötüleştiren vuruş [\\%]", "frac_worsening", 100, 1),
            row("Anlık hasar (yerel arama öncesi)", "mean_kick_damage_preLS", 1, 1),
            row("Ödül | iyileştiren", "mean_r_given_improving", 1, 2),
            row("Havuz kenar uzunluğu / medyan", "pool_len_ratio_mean", 1, 2),
            "\\midrule",
            row("Son gap [\\%]", "gap_pct", 1, 3),
            f"Vuruş / koşu & {kk['onlinewarm']:,.0f} & {kk['linprior']:,.0f} & -- & -- \\\\".replace(",", "."),
            f"Kabul / koşu & {tr(ac['onlinewarm'], 1)} & {tr(ac['linprior'], 1)} & -- & -- \\\\"]
    body = (
        "\\begin{table}[!htbp]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{Çevrimiçi ve sabit doğrusal skorun vuruş düzeyi ödülü (geliştirme, 12 VLSI problemi $\\times$ 3 tohum, "
        "10~sn; keşifsel). Ç$>$S: çevrimiçi değeri büyük problem sayısı.}\n"
        "\\label{tab:v3-online-reward}\n"
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccc}\n\\toprule\n"
        "Büyüklük & Çevrimiçi & Sabit & Ç$>$S & Wilcoxon $p$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    B.write_tab("v3_online_reward.tex", body)


if __name__ == "__main__":
    main()
