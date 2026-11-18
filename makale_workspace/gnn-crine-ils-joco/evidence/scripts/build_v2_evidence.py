# -*- coding: utf-8 -*-
"""Asama 2 (PROTOCOL v1, hizli ILS kampanyasi) kanit cikarimi -- metindeki her sayi icin kayit.

Kaynak (yazilim koku = tsplib/): results/kick_targeting/*_summary.json, mechanism_summary.json,
exploratory_summary.json, cheap_summary.json, gain_summary.json, pool_composition_summary.json,
cheap_mechanism_summary.json, lkh_baseline/*.json, lkh_ref/*.json, vlsi_primary_10s/*.json (ham kosumlar),
models/cheap_scorer.json, PROTOCOL.json/.sha256, PROTOCOL_DEVIATIONS.log, data_synth_fast/catalog.json.

Her kayit `ek.anahtar` ile idempotent (ID sabit, silme yok, `durum: terk` korunur; uretici=build_v2_evidence).
Her kayit icin sections/*.tex'teki metin degeri (`ek.metin`) ve kaynaga gore uyum (`ek.metin_uyumlu`)
tutulur; uyumsuzluklar notes/kanit-v3-ozet.md "Asama 2" bolumune otomatik yazilir.

    py -3.13 evidence/scripts/build_v2_evidence.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WS = Path(__file__).resolve().parents[2]
ROOT = WS.parents[1]
KT = ROOT / "results" / "kick_targeting"
EVID = WS / "evidence" / "evidence.json"
OZET = WS / "notes" / "kanit-v3-ozet.md"
URETICI = "build_v2_evidence"


def L(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- defter
class Defter:
    def __init__(self):
        self.d = json.loads(EVID.read_text(encoding="utf-8"))
        self.k = self.d["kayitlar"]
        self.by_key = {r["ek"]["anahtar"]: r for r in self.k
                       if isinstance(r.get("ek"), dict) and r["ek"].get("uretici") == URETICI}
        self.next = max(int(r["id"][1:]) for r in self.k if re.fullmatch(r"E\d+", r["id"])) + 1
        self.rows = []      # (id, anahtar, metin bilgisi) eslestirme tablosu icin

    def ekle(self, anahtar, metrik, yontem, veri_seti, deger, birim, kaynak_dosya, kaynak_konum, **ek):
        ek = {"uretici": URETICI, "anahtar": anahtar, "asama": 2, **ek}
        if isinstance(deger, (np.floating, np.integer)):
            deger = float(deger)
        rec = {"metrik": metrik, "yontem": yontem, "veri_seti": veri_seti, "deger": deger, "birim": birim,
               "kaynak": "yazilim", "kaynak_dosya": kaynak_dosya, "kaynak_konum": kaynak_konum, "atif": None}
        old = self.by_key.get(anahtar)
        if old is not None:
            durum, par = old.get("durum", "gecerli"), old.get("paradigma")
            old.update(rec)
            old["ek"] = ek
            old["durum"] = durum
            if par:
                old["paradigma"] = par
            return old
        new = {"id": f"E{self.next:03d}", **rec, "durum": "gecerli", "ek": ek}
        self.next += 1
        self.k.append(new)
        self.by_key[anahtar] = new
        return new

    def kaydet(self):
        EVID.write_text(json.dumps(self.d, ensure_ascii=False, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- metin denetimi
def uyum(deger, metin, tur="r", olcek=1.0):
    """metin: Python sayi yazimi (str). tur: r (metnin ondalik basamagina yuvarla), lt/le/ge, sig1 (1 anlamli basamak)."""
    v = float(deger) * olcek
    x = float(metin)
    if tur == "yaklasik":            # metinde "yaklasik / ~ / ucte biri" gibi niteleme: %12 goreli tolerans
        return abs(v - x) <= 0.12 * abs(x)
    if tur == "lt":
        return v < x
    if tur == "le":
        return v <= x + 1e-12
    if tur == "ge":
        return v >= x - 1e-12
    if tur == "sig1":
        e = math.floor(math.log10(abs(v)))
        return round(v / 10 ** e) * 10 ** e == x or abs(round(v / 10 ** e) - x / 10 ** e) < 1e-9
    d = len(metin.split(".")[1]) if "." in metin else 0
    return abs(round(v, d) - x) < 10 ** (-d - 6) or abs(v - x) <= 0.5 * 10 ** (-d) + 1e-12


CHECKS = []   # (anahtar, alan, metin, tur, olcek, bolum, cumle)


def chk(anahtar, alan, metin, bolum, cumle, tur="r", olcek=1.0):
    CHECKS.append((anahtar, alan, metin, tur, olcek, bolum, cumle))


# ----------------------------------------------------------------------------- yardimcilar
def comp(D, anahtar, c, veri_seti, kaynak_dosya, konum, **ek):
    """Esli karsilastirma kaydi: deger = ortalama fark; diger istatistikler ek'te."""
    a, b = c.get("a"), c.get("b")
    md = c.get("mean_diff", c.get("mean"))
    e = {k: v for k, v in c.items() if k not in ("a", "b")}
    if "bootstrap95" in e:
        e["ci_lo"], e["ci_hi"] = e["bootstrap95"]
    e.update(ek)
    return D.ekle(anahtar, "fark_ortalama", f"{a} - {b}" if a else anahtar.split(".")[-1], veri_seti, md,
                  "yuzde_puani", kaynak_dosya, konum, **e)


def scal(D, anahtar, metrik, yontem, veri_seti, deger, birim, kaynak_dosya, konum, **ek):
    return D.ekle(anahtar, metrik, yontem, veri_seti, deger, birim, kaynak_dosya, konum, **ek)


def per_run(tag):
    A = defaultdict(lambda: defaultdict(list))
    for f in sorted((KT / tag).glob("*__s*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for s, r in d["runs"].items():
            A[s]["acc"].append(r["accepted"] / max(r["kicks"], 1))
            A[s]["refr"].append(r.get("refreshes", 0))
            A[s]["stale"].append(r.get("stale_cuts", 0) / max(r.get("guided_cuts", 0), 1))
    return {s: {k: float(np.mean(v)) for k, v in m.items()} for s, m in A.items()}


def wil(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(stats.wilcoxon(d).pvalue), int((d > 0).sum()), len(d)


# ============================================================================= cikarim
def main():
    D = Defter()
    PS = "results/kick_targeting/vlsi_primary_10s_summary.json"
    P = L(PS)
    B = "bulgular.tex"
    # ---- protokol / kurulum
    sha = (KT / "PROTOCOL.sha256").read_text(encoding="utf-8").split()[0]
    assert sha == hashlib.sha256((KT / "PROTOCOL.json").read_bytes()).hexdigest()
    scal(D, "v2.protokol.sha256", "protokol_sha256", "PROTOCOL v1", "vlsi_primary", sha, "",
         "results/kick_targeting/PROTOCOL.sha256", "satir 1")
    chk("v2.protokol.sha256", "deger8", "cce83992", "deneysel_kurulum.tex", "SHA-256 özeti (cce83992…)", tur="str")
    proto = L("results/kick_targeting/PROTOCOL.json")
    scal(D, "v2.protokol.tost_margin", "esdeger_sinir", "PROTOCOL v1", "vlsi_primary", 0.10, "yuzde_puani",
         "results/kick_targeting/PROTOCOL.json", "equivalence (margin +-0.10)", metin_kaynak=proto["equivalence"][:80])
    chk("v2.protokol.tost_margin", "deger", "0.10", "deneysel_kurulum.tex", "±0,10 gap puanı sınırıyla TOST")
    scal(D, "v2.protokol.workers_planned", "isci_sayisi", "PROTOCOL v1", "vlsi_primary", proto["workers"], "adet",
         "results/kick_targeting/PROTOCOL.json", "workers")
    chk("v2.protokol.workers_planned", "deger", "8", "deneysel_kurulum.tex", "8 yerine 4--10 paralel işçi")
    log = (KT / "PROTOCOL_DEVIATIONS.log").read_text(encoding="utf-8")
    for k, pat in (("kill1", r"after (\d+)/380 tasks"), ("kill2", r"again by memory-pressure reaper at (\d+)/380")):
        v = int(re.search(pat, log).group(1))
        scal(D, f"v2.protokol.{k}", "sapma_gorev_no", "PROTOCOL v1", "vlsi_primary", v, "gorev",
             "results/kick_targeting/PROTOCOL_DEVIATIONS.log", f"regex '{pat}'")
    chk("v2.protokol.kill1", "deger", "160", "deneysel_kurulum.tex", "160. ve 298. görevde durduruldu")
    chk("v2.protokol.kill2", "deger", "298", "deneysel_kurulum.tex", "160. ve 298. görevde durduruldu")
    scal(D, "v2.protokol.max_workers", "isci_sayisi_maks", "PROTOCOL v1", "vlsi_primary", 10, "adet",
         "results/kick_targeting/PROTOCOL_DEVIATIONS.log", "00:44:09 NOTE: 'up to 10 workers'",
         alt=4, not_="4 (yeniden baslatma) ile 10 (artakalan zincirle birlikte) arasi")
    chk("v2.protokol.max_workers", "deger", "10", "deneysel_kurulum.tex", "4--10 paralel işçi")

    # ---- setler
    syncat = L("data_synth_fast/catalog.json")["instances"]
    for tag, lab in (("vlsi_primary_10s", "VLSI-38"), ("vlsi_old_10s", "VLSI-20"), ("tsplib72_10s", "TSPLIB-72"),
                     ("synth_holdout_1s", "Sentetik-100"), ("synth_1000_10s", "Sentetik-1000")):
        s = L(f"results/kick_targeting/{tag}_summary.json")
        c = L(f"results/kick_targeting/{tag}/config.json")
        ns = [r["n"] for r in s["per_instance"].values()]
        scal(D, f"v2.set.{tag}", "set_problem_sayisi", lab, tag, s["n_instances"], "adet",
             f"results/kick_targeting/{tag}_summary.json", "n_instances", n_min=min(ns), n_max=max(ns),
             seeds=c["seeds"], budget=c["budget"], start_gap_mean=s["start_gap_mean"])
    chk("v2.set.vlsi_primary_10s", "deger", "38", "ozet.tex/bulgular.tex", "38 VLSI problemi")
    chk("v2.set.vlsi_primary_10s", "n_min", "1376", "ozet.tex/bulgular.tex", "n=1.376--3.954")
    chk("v2.set.vlsi_primary_10s", "n_max", "3954", "ozet.tex/bulgular.tex", "n=1.376--3.954")
    chk("v2.set.vlsi_primary_10s", "seeds", "10", "ozet.tex", "10 tohum")
    chk("v2.set.vlsi_primary_10s", "budget", "10", "ozet.tex", "10 sn")
    chk("v2.set.vlsi_primary_10s", "start_gap_mean", "9.18", B, "ortalama başlangıç gap'i %9,18")
    chk("v2.set.vlsi_old_10s", "deger", "20", "bulgular_ikincil.tex", "20 problem, 131--3.293 nokta")
    chk("v2.set.vlsi_old_10s", "n_min", "131", "bulgular_ikincil.tex", "131--3.293 nokta")
    chk("v2.set.vlsi_old_10s", "n_max", "3293", "bulgular_ikincil.tex", "131--3.293 nokta")
    chk("v2.set.tsplib72_10s", "n_min", "51", "bulgular_ikincil_ek.tex", "72 problem, 51--2.392 nokta")
    chk("v2.set.tsplib72_10s", "n_max", "2392", "bulgular_ikincil_ek.tex", "72 problem, 51--2.392 nokta")
    chk("v2.set.synth_1000_10s", "deger", "40", "bulgular_ikincil.tex", "Sentetik n=1.000 (40 problem)")
    chk("v2.set.synth_holdout_1s", "deger", "100", "bulgular_ikincil_ek.tex", "100 problem, n∈[80,160], 1 sn")
    chk("v2.set.synth_holdout_1s", "budget", "1", "bulgular_ikincil_ek.tex", "1 sn")
    chk("v2.set.tsplib72_10s", "seeds", "5", "deneysel_kurulum.tex", "diğerlerinde 5 tohum")
    chk("v2.set.vlsi_old_10s", "seeds", "10", "deneysel_kurulum.tex", "VLSI setlerinde 10 tohum")
    tl = sorted({1})
    scal(D, "v2.set.synth_ref_time", "referans_lkh_suresi", "LKH-3", "sentetik setler", 30.0, "sn",
         "make_synth_suite.py", "satir 65: run_lkh(p, len(xs), 30.0, 1); data_synth_fast/catalog.json ref",
         n_ref=len(syncat), uyari="evidence/tables/v2_sets.tex eskiden 'LKH-3 (60 sn)' yaziyordu; duzeltildi")
    chk("v2.set.synth_ref_time", "deger", "30", "deneysel_kurulum.tex", "LKH-3 varsayılan, 30 sn")

    # ---- birincil: H1-H4, S1-S3
    for h, c in P["primary"].items():
        comp(D, f"v2.primary.{h}", c, "vlsi_primary", PS, f"primary.{h}", hipotez=h)
    for h, c in P["secondary"].items():
        comp(D, f"v2.primary.{h.split('_')[0]}", c, "vlsi_primary", PS, f"secondary.{h}", hipotez=h)
    chk("v2.primary.H1", "deger", "-0.052", "ozet.tex/bulgular.tex", "GNN−rastgele −0,052 puan")
    chk("v2.primary.H1", "ci_lo", "-0.077", B, "GA [−0,077; −0,029]")
    chk("v2.primary.H1", "ci_hi", "-0.029", B, "GA [−0,077; −0,029]")
    chk("v2.primary.H1", "a_better", "29", "ozet.tex/bulgular.tex", "29/38")
    chk("v2.primary.H1", "holm_p", "0.0003", "ozet.tex/bulgular.tex", "p_Holm=0,0003")
    chk("v2.primary.H1", "tost_p_margin_0.10", "0.0002", B, "TOST p=0,0002")
    chk("v2.primary.H3", "deger", "0.048", "ozet.tex/bulgular.tex", "GNN−kNN +0,048 puan")
    chk("v2.primary.H3", "ci_lo", "0.019", B, "GA [0,019; 0,076]")
    chk("v2.primary.H3", "ci_hi", "0.076", B, "GA [0,019; 0,076]")
    chk("v2.primary.H3", "b_better", "29", "ozet.tex/bulgular.tex", "kNN 29/38 önde")
    chk("v2.primary.H3", "holm_p", "0.0047", "ozet.tex/bulgular.tex", "p_Holm=0,0047")
    chk("v2.primary.H2", "deger", "0.017", B, "H2 +0,017 puan")
    chk("v2.primary.H2", "holm_p", "0.41", B, "H2 p_Holm=0,41")
    chk("v2.primary.H4", "deger", "0.011", B, "H4 +0,011 puan")
    chk("v2.primary.H4", "holm_p", "0.41", B, "H4 p_Holm=0,41")
    chk("v2.primary.H2", "tost_p_margin_0.10", "1e-4", B, "TOST p<10^-4 (H2)", tur="lt")
    chk("v2.primary.H4", "tost_p_margin_0.10", "1e-4", B, "TOST p<10^-4 (H4)", tur="lt")
    chk("v2.primary.S1", "deger", "-0.100", B, "kNN−rastgele −0,100 puan")
    chk("v2.primary.S1", "a_better", "34", B, "34/38")
    chk("v2.primary.S3", "deger", "0.362", B, "yerel−global +0,362 puan")
    chk("v2.primary.S3", "b_better", "37", B, "37/38")
    # gorece etki
    rel = -P["primary"]["H1"]["mean_diff"] / P["mean_gap"]["random_global"]
    scal(D, "v2.primary.H1_rel", "goreli_etki", "gnn_global vs random_global", "vlsi_primary", rel, "oran", PS,
         "-primary.H1.mean_diff / mean_gap.random_global")
    chk("v2.primary.H1_rel", "deger", "6", "bulgular.tex/tartisma.tex", "rastgele gap'inin yaklaşık %6'sı", olcek=100, tur="yaklasik")
    # ortalama gap'ler ve aralik
    for s, v in P["mean_gap"].items():
        scal(D, f"v2.primary.gap.{s}", "ortalama_gap", s, "vlsi_primary", v, "yuzde", PS, f"mean_gap.{s}")
    chk("v2.primary.gap.knn_global", "deger", "0.76", B, "altı strateji %0,76--1,24 (min)")
    chk("v2.primary.gap.gnn_local", "deger", "1.24", B, "altı strateji %0,76--1,24 (maks)")
    chk("v2.primary.gap.knn_global", "deger", "0.76", "tartisma_lkh.tex", "en iyi ILS 10 sn'de %0,76")
    # operasyonel
    for s, o in P["operational"].items():
        scal(D, f"v2.primary.kps.{s}", "vurus_per_sn", s, "vlsi_primary", o["kicks_per_s_mean"], "vurus/sn", PS,
             f"operational.{s}.kicks_per_s_mean", model_share=o["model_time_frac_mean"],
             stale_pooled=o["stale_cut_rate"])
    chk("v2.primary.kps.random_global", "deger", "7388", B, "rastgele--global saniyede 7.388")
    chk("v2.primary.kps.random_local", "deger", "12474", B, "rastgele--yerel 12.474")
    chk("v2.primary.kps.gnn_global", "deger", "4038", B, "GNN 4.038 kick/sn")
    chk("v2.primary.kps.knn_global", "deger", "4072", B, "kNN 4.072 kick/sn")
    chk("v2.primary.kps.gnn_global", "model_share", "6.0", B, "GNN model payı %6,0", olcek=100)
    chk("v2.primary.kps.knn_global", "model_share", "1.2", B, "kNN model payı %1,2", olcek=100)
    pr = per_run("vlsi_primary_10s")
    for s in ("random_global", "gnn_global", "knn_global"):
        scal(D, f"v2.primary.run.{s}", "kabul_orani_kosum_ort", s, "vlsi_primary", pr[s]["acc"], "oran",
             "results/kick_targeting/vlsi_primary_10s/*__s*.json", "kosum basina accepted/kicks ortalamasi (380 kosum)",
             refreshes=pr[s]["refr"], stale_kosum_ort=pr[s]["stale"])
    chk("v2.primary.run.gnn_global", "deger", "0.41", B, "kabul GNN %0,41", olcek=100)
    chk("v2.primary.run.knn_global", "deger", "0.39", B, "kabul kNN %0,39", olcek=100)
    chk("v2.primary.run.random_global", "deger", "0.22", B, "kabul rastgele %0,22", olcek=100)
    chk("v2.primary.run.knn_global", "refreshes", "83", "bulgular_mekanizma_neden.tex", "kNN havuzu 83 kez tazelenir")
    chk("v2.primary.run.gnn_global", "refreshes", "20", "bulgular_mekanizma_neden.tex", "GNN havuzu 20 kez")
    chk("v2.primary.run.gnn_global", "stale_kosum_ort", "4.9", "bulgular_mekanizma_neden.tex", "GNN bayat %4,9", olcek=100)
    chk("v2.primary.run.knn_global", "stale_kosum_ort", "0.2", "bulgular_mekanizma_neden.tex", "kNN bayat %0,2", olcek=100)
    # anytime + strata
    a01 = P["anytime"]["0.1"]["H"]
    for h in ("H1", "H3"):
        comp(D, f"v2.primary.any0.1.{h}", a01[h], "vlsi_primary@0.1s", PS, f"anytime['0.1'].H.{h}")
    chk("v2.primary.any0.1.H1", "deger", "0.68", B, "0,1 sn'de GNN−rastgele +0,68")
    chk("v2.primary.any0.1.H3", "deger", "0.59", B, "0,1 sn'de GNN−kNN +0,59")
    pmax = max(P["anytime"][cp]["H"]["H3"]["wilcoxon_p"] for cp in P["anytime"])
    scal(D, "v2.primary.H3_anytime_pmax", "maks_p", "gnn_global vs knn_global", "vlsi_primary", pmax, "", PS,
         "max over anytime[*].H.H3.wilcoxon_p")
    chk("v2.primary.H3_anytime_pmax", "deger", "0.006", B, "her kontrol noktasında ham p≤0,006", tur="le")
    for st in ("n<2500", "n>=2500"):
        for h in ("H1", "H3"):
            comp(D, f"v2.primary.strata.{st}.{h}", P["strata"][st][h], f"vlsi_primary/{st}", PS, f"strata['{st}'].{h}")
    chk("v2.primary.strata.n<2500.H1", "a_better", "13", B, "n<2.500: GNN rastgeleyi 13/18")
    chk("v2.primary.strata.n<2500.H3", "b_better", "15", B, "n<2.500: kNN GNN'yi 15/18")
    chk("v2.primary.strata.n>=2500.H1", "a_better", "16", B, "n≥2.500: 16/20")
    chk("v2.primary.strata.n>=2500.H3", "b_better", "14", B, "n≥2.500: 14/20")

    # ---- ikincil setler
    for tag in ("vlsi_old_10s", "synth_1000_10s", "tsplib72_10s", "synth_holdout_1s"):
        s = L(f"results/kick_targeting/{tag}_summary.json")
        src = f"results/kick_targeting/{tag}_summary.json"
        for h, c in list(s["primary"].items()) + list(s["secondary"].items()):
            comp(D, f"v2.{tag}.{h.split('_')[0]}", c, tag, src, f"{'primary' if h.startswith('H') else 'secondary'}.{h}")
        for g, v in s["mean_gap"].items():
            scal(D, f"v2.{tag}.gap.{g}", "ortalama_gap", g, tag, v, "yuzde", src, f"mean_gap.{g}")
    I = "bulgular_ikincil.tex"
    chk("v2.vlsi_old_10s.H1", "deger", "-0.014", I, "GNN−global −0,014 puan")
    chk("v2.vlsi_old_10s.H1", "a_better", "10", I, "10/1/9")
    chk("v2.vlsi_old_10s.H1", "holm_p", "1", I, "p_Holm=1")
    chk("v2.vlsi_old_10s.H1", "tost_p_margin_0.10", "0.0003", I, "TOST p=0,0003")
    chk("v2.vlsi_old_10s.H3", "deger", "0.043", I, "kNN−GNN +0,043")
    chk("v2.vlsi_old_10s.H3", "b_better", "15", I, "kNN 15/20 önde")
    chk("v2.vlsi_old_10s.H3", "holm_p", "0.029", I, "p_Holm=0,029")
    chk("v2.vlsi_old_10s.S1", "deger", "-0.057", I, "kNN−rastgele −0,057")
    chk("v2.vlsi_old_10s.S1", "a_better", "13", I, "13/1/6")
    chk("v2.vlsi_old_10s.S1", "wilcoxon_p", "0.023", I, "ham p=0,023")
    chk("v2.synth_1000_10s.H1", "deger", "-0.019", I, "GNN−global −0,019")
    chk("v2.synth_1000_10s.H1", "tost_p_margin_0.10", "0.0001", I, "TOST p=0,0001")
    chk("v2.synth_1000_10s.H2", "deger", "0.146", I, "GNN−yerel +0,146")
    chk("v2.synth_1000_10s.H2", "a_better", "12", I, "12/28")
    chk("v2.synth_1000_10s.H2", "holm_p", "0.002", I, "p_Holm=0,002")
    chk("v2.synth_1000_10s.H3", "tost_p_margin_0.10", "0.0001", I, "kNN–GNN TOST p=0,0001")
    E = "bulgular_ikincil_ek.tex"
    chk("v2.tsplib72_10s.H1", "deger", "0.008", E, "GNN−rastgele +0,008")
    chk("v2.tsplib72_10s.H3", "deger", "0.003", E, "GNN−kNN +0,003")
    chk("v2.tsplib72_10s.S1", "deger", "0.005", E, "kNN−rastgele +0,005")
    for h in ("H1", "H3", "S1"):
        chk(f"v2.tsplib72_10s.{h}", "tost_p_margin_0.10", "1e-4", E, f"TOST p<10^-4 ({h})", tur="lt")
    chk("v2.tsplib72_10s.H2", "deger", "0.058", E, "segment GNN−rastgele +0,058")
    chk("v2.tsplib72_10s.H2", "a_better", "14", E, "14/26/32")
    chk("v2.tsplib72_10s.H2", "ties", "26", E, "14/26/32")
    chk("v2.tsplib72_10s.H2", "holm_p", "0.016", E, "p_Holm=0,016")
    # tsplib n>=1000 alt kume (problem bazli yeniden hesap)
    T = L("results/kick_targeting/tsplib72_10s_summary.json")["per_instance"]
    big = [r for r in T.values() if r["n"] >= 1000]
    d = np.array([r["knn_global"] - r["random_global"] for r in big])
    d2 = np.array([r["knn_global"] - r["gnn_global"] for r in big])
    scal(D, "v2.tsplib72_10s.big.S1", "fark_ortalama", "knn_global - random_global", "tsplib72/n>=1000", float(d.mean()),
         "yuzde_puani", "results/kick_targeting/tsplib72_10s_summary.json", "per_instance (n>=1000) yeniden hesap",
         n_instances=len(big), a_better=int((d < 0).sum()), wilcoxon_p=float(stats.wilcoxon(d).pvalue),
         knn_beats_gnn=int((d2 < 0).sum()))
    chk("v2.tsplib72_10s.big.S1", "n_instances", "18", E, "n≥1.000 olan 18 problem")
    chk("v2.tsplib72_10s.big.S1", "a_better", "14", E, "kNN rastgeleyi 14/18")
    chk("v2.tsplib72_10s.big.S1", "deger", "-0.015", E, "−0,015 puan")
    chk("v2.tsplib72_10s.big.S1", "wilcoxon_p", "0.18", E, "p=0,18")
    chk("v2.tsplib72_10s.big.S1", "knn_beats_gnn", "12", E, "GNN'yi 12/18")
    # tsplib esitlikler (uc global strateji ayni)
    eq = sum(1 for r in T.values() if r["knn_global"] == r["random_global"] == r["gnn_global"])
    scal(D, "v2.tsplib72_10s.ties_all3", "esit_problem_sayisi", "random/gnn/knn global", "tsplib72", eq, "adet",
         "results/kick_targeting/tsplib72_10s_summary.json", "per_instance: uc global strateji birebir ayni",
         oran=eq / len(T))
    chk("v2.tsplib72_10s.ties_all3", "oran", "0.33", E, "problemlerin yaklaşık üçte biri aynı sonuca ulaşır", tur="yaklasik")
    H = L("results/kick_targeting/synth_holdout_1s_summary.json")
    chk("v2.synth_holdout_1s.H1", "deger", "0.017", E, "GNN−global +0,017")
    chk("v2.synth_holdout_1s.H1", "a_better", "9", E, "9/70/21")
    chk("v2.synth_holdout_1s.H1", "ties", "70", E, "9/70/21 (~%70 eşit)")
    chk("v2.synth_holdout_1s.H1", "holm_p", "0.042", E, "p_Holm=0,042")
    chk("v2.synth_holdout_1s.H1", "tost_p_margin_0.10", "1e-4", E, "TOST p<10^-4", tur="lt")
    gaps = list(H["mean_gap"].values())
    scal(D, "v2.synth_holdout_1s.gap_range", "ortalama_gap_aralik", "6 strateji", "synth_holdout_1s", min(gaps),
         "yuzde", "results/kick_targeting/synth_holdout_1s_summary.json", "min/max mean_gap", maks=max(gaps))
    chk("v2.synth_holdout_1s.gap_range", "deger", "0.02", E, "ortalama %0,02--0,05 gap (min)")
    chk("v2.synth_holdout_1s.gap_range", "maks", "0.05", E, "ortalama %0,02--0,05 gap (maks)")

    # ---- mekanizma (AUC)
    MS = "results/kick_targeting/mechanism_summary.json"
    M = L(MS)
    for st in ("start_local_opt", "ils_2s"):
        s = M["summary"][st]
        scal(D, f"v2.mech.{st}", "auc_gnn", "gnn", f"vlsi58/{st}", s["auc_gnn"], "AUC", MS, f"summary.{st}",
             **{k: v for k, v in s.items() if not isinstance(v, dict)})
    rows = [r for r in M["rows"] if r.get("state", "ils_2s") == "ils_2s" or "state" not in r]
    rows = [r for r in M["rows"] if r.get("state") == "ils_2s"]
    pa, na, nn = wil([r["auc_gnn"] for r in rows], [r["auc_knn"] for r in rows])
    pp, np_, _ = wil([r["prec_gnn"] for r in rows], [r["prec_knn"] for r in rows])
    scal(D, "v2.mech.ils_2s.gnn_vs_knn", "auc_gnn_buyuk_problem_sayisi", "gnn vs knn", "vlsi58/ils_2s", na, "adet", MS,
         "rows[state=ils_2s]: auc_gnn>auc_knn; Wilcoxon", n=nn, p_auc=pa, prec_wins=np_, p_prec=pp)
    m = "bulgular_mekanizma.tex"
    chk("v2.mech.ils_2s", "n_instances", "58", m, "58 VLSI problemi")
    chk("v2.mech.ils_2s", "base_rate_wrong", "15.3", m, "yanlış kenar oranı %15,3", olcek=100)
    chk("v2.mech.ils_2s", "auc_gnn", "0.854", m, "GNN AUC 0,854")
    chk("v2.mech.ils_2s", "auc_knn", "0.821", m, "kNN AUC 0,821")
    chk("v2.mech.ils_2s", "prec_gnn", "64.6", m, "kesinlik GNN %64,6", olcek=100)
    chk("v2.mech.ils_2s", "prec_knn", "60.7", m, "kesinlik kNN %60,7", olcek=100)
    chk("v2.mech.ils_2s", "spearman_gnn_knn", "0.80", m, "Spearman ρ=0,80")
    chk("v2.mech.ils_2s", "pool_overlap_gnn_knn", "62", m, "havuzlar ~%62 paylaşır", olcek=100)
    chk("v2.mech.ils_2s.gnn_vs_knn", "deger", "57", "ozet.tex/bulgular_mekanizma.tex", "58 problemin 57'sinde daha yüksek AUC")
    chk("v2.mech.ils_2s.gnn_vs_knn", "p_auc", "5e-11", m, "Wilcoxon p=5×10^-11", tur="sig1")
    chk("v2.mech.ils_2s.gnn_vs_knn", "prec_wins", "48", m, "kesinlik 48/58")
    chk("v2.mech.ils_2s.gnn_vs_knn", "p_prec", "3e-6", m, "p=3×10^-6", tur="sig1")
    refs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in glob.glob(str(KT / "lkh_ref" / "*.json"))]
    scal(D, "v2.mech.lkh_ref_gap", "referans_gap_ortalama", "LKH-3 60 sn", "vlsi58", float(np.mean([r["gap_pct"] for r in refs])),
         "yuzde", "results/kick_targeting/lkh_ref/*.json", "mean(gap_pct)", n=len(refs),
         time_limit=sorted({r["time_limit"] for r in refs}))
    chk("v2.mech.lkh_ref_gap", "deger", "0.03", m, "LKH-3 referans turu ortalama gap %0,03")
    chk("v2.mech.lkh_ref_gap", "n", "58", "deneysel_kurulum.tex", "58 VLSI problemi için LKH-3 (60 sn)")

    # ---- kesifsel: kahin, taze, 60 sn
    XS = "results/kick_targeting/exploratory_summary.json"
    X = L(XS)
    N = "bulgular_mekanizma_neden.tex"
    comp(D, "v2.expl.oracle-knn", {"a": "oracle_global", "b": "knn_global", **X["oracle"]["pairs"]["oracle_global-knn_global"]},
         "vlsi_primary/oracle", XS, "oracle.pairs.oracle_global-knn_global")
    for k in ("oracle-gnn_global", "oracle-random_global"):
        a, b = k.split("-")
        comp(D, f"v2.expl.{k}", {"a": a, "b": b, **X["oracle_vs_primary_seeds012"][k]}, "vlsi_primary/oracle", XS,
             f"oracle_vs_primary_seeds012.{k}")
    scal(D, "v2.expl.oracle_ops", "model_payi", "oracle_global", "vlsi_primary/oracle", X["oracle_ops"]["model_share"],
         "oran", XS, "oracle_ops.model_share", **X["oracle_ops"])
    pm = X["oracle_vs_primary_seeds012"]["primary_mean_gap"]
    orc = -X["oracle_vs_primary_seeds012"]["oracle-random_global"]["mean_diff"]
    scal(D, "v2.expl.share_knn", "kahin_kazancinin_payi", "knn_global", "vlsi_primary seeds0-2",
         (pm["random_global"] - pm["knn_global"]) / orc, "oran", XS,
         "(primary_mean_gap.random - .knn) / -(oracle-random_global.mean_diff)")
    scal(D, "v2.expl.share_gnn", "kahin_kazancinin_payi", "gnn_global", "vlsi_primary seeds0-2",
         (pm["random_global"] - pm["gnn_global"]) / orc, "oran", XS,
         "(primary_mean_gap.random - .gnn) / -(oracle-random_global.mean_diff)")
    chk("v2.expl.oracle_ops", "deger", "0.3", N, "kâhin maliyeti %0,3", olcek=100)
    chk("v2.expl.oracle-knn", "deger", "-0.049", N, "kâhin−kNN −0,049")
    chk("v2.expl.oracle-knn", "ci_lo", "-0.094", N, "GA [−0,094; −0,006]")
    chk("v2.expl.oracle-knn", "ci_hi", "-0.006", N, "GA [−0,094; −0,006]")
    chk("v2.expl.oracle-knn", "a_better", "23", N, "23/38")
    chk("v2.expl.oracle-knn", "wilcoxon_p", "0.053", N, "p=0,053")
    chk("v2.expl.oracle-random_global", "deger", "-0.127", N, "rastgeleye kazanç 0,127")
    chk("v2.expl.oracle-gnn_global", "deger", "-0.089", N, "kâhin−GNN −0,089")
    chk("v2.expl.oracle-gnn_global", "a_better", "28", N, "28/38")
    chk("v2.expl.oracle-gnn_global", "wilcoxon_p", "0.002", N, "p=0,002")
    chk("v2.expl.share_knn", "deger", "0.8", N, "yaklaşık beşte dördü (kNN)", tur="yaklasik")
    chk("v2.expl.share_gnn", "deger", "0.3", N, "yaklaşık üçte biri (GNN)", tur="yaklasik")
    fo = X["fresh_ops"]["gnn_global"]
    scal(D, "v2.expl.fresh_ops.gnn", "model_payi", "gnn_global (taze)", "vlsi_primary/fresh", fo["model_share"], "oran",
         XS, "fresh_ops.gnn_global", **fo)
    chk("v2.expl.fresh_ops.gnn", "stale_share", "0.1", N, "taze GNN bayat %0,1", olcek=100)
    chk("v2.expl.fresh_ops.gnn", "deger", "32", N, "model payı %32", olcek=100)
    for k in ("gnn_global-random_global", "gnn_global-knn_global", "knn_global-random_global"):
        a, b = k.split("-")
        comp(D, f"v2.expl.fresh.{k}", {"a": a, "b": b, **X["fresh_10s"]["pairs"][k]}, "vlsi_primary/fresh", XS,
             f"fresh_10s.pairs.{k}")
        comp(D, f"v2.expl.b60.{k}", {"a": a, "b": b, **X["budget_60s"]["pairs"][k]}, "vlsi_primary/60s", XS,
             f"budget_60s.pairs.{k}")
    chk("v2.expl.fresh.gnn_global-random_global", "deger", "0.064", N, "taze GNN rastgeleden 0,064 kötü")
    chk("v2.expl.fresh.gnn_global-random_global", "wilcoxon_p", "0.041", N, "p=0,041")
    chk("v2.expl.fresh.gnn_global-knn_global", "deger", "0.114", N, "kNN'den 0,114 kötü")
    chk("v2.expl.fresh.gnn_global-knn_global", "wilcoxon_p", "0.0005", N, "p=0,0005")
    K = "bulgular_kesifsel.tex"
    for g, v in X["budget_60s"]["mean_gap"].items():
        scal(D, f"v2.expl.b60.gap.{g}", "ortalama_gap", g, "vlsi_primary/60s", v, "yuzde", XS, f"budget_60s.mean_gap.{g}")
    chk("v2.expl.b60.gap.random_global", "deger", "0.621", K, "60 sn rastgele %0,621")
    chk("v2.expl.b60.gap.gnn_global", "deger", "0.590", K, "GNN %0,590")
    chk("v2.expl.b60.gap.knn_global", "deger", "0.565", K, "kNN %0,565")
    chk("v2.expl.b60.knn_global-random_global", "deger", "-0.057", K, "kNN−rastgele −0,057")
    chk("v2.expl.b60.knn_global-random_global", "a_better", "25", K, "25/38")
    chk("v2.expl.b60.knn_global-random_global", "wilcoxon_p", "0.017", K, "p=0,017")
    chk("v2.expl.b60.gnn_global-random_global", "deger", "-0.031", K, "GNN−rastgele −0,031")
    chk("v2.expl.b60.gnn_global-random_global", "wilcoxon_p", "0.19", K, "p=0,19")
    chk("v2.expl.b60.gnn_global-knn_global", "deger", "0.026", K, "GNN−kNN +0,026")
    chk("v2.expl.b60.gnn_global-knn_global", "wilcoxon_p", "0.35", K, "p=0,35")
    an = X["budget_60s"]["anytime"]
    order_ok = [cp for cp in an if an[cp]["knn_global"] < an[cp]["gnn_global"] < an[cp]["random_global"]]
    scal(D, "v2.expl.b60.order", "siralama_korunan_kontrol_noktasi_sayisi", "knn<gnn<random", "vlsi_primary/60s",
         len(order_ok), "adet", XS, "budget_60s.anytime: knn<gnn<random", noktalar=order_ok,
         ilk_nokta=min(float(c) for c in order_ok))
    chk("v2.expl.b60.order", "ilk_nokta", "1.0", K, "sıralama 1 sn'den 60 sn'ye korunur")

    # ---- ucuz skor
    CS = "results/kick_targeting/cheap_summary.json"
    C = L(CS)
    pc = C["primary_cheap_10s"]
    for k in ("cheap_global-knn_global", "cheap_global-gnn_global"):
        a, b = k.split("-")
        comp(D, f"v2.cheap.{k}", {"a": a, "b": b, **pc["pairs"][k]}, "vlsi_primary/cheap", CS, f"primary_cheap_10s.pairs.{k}")
    scal(D, "v2.cheap.ops", "tazeleme_sayisi", "cheap_global", "vlsi_primary/cheap", pc["ops"]["cheap_global"]["refreshes"],
         "adet", CS, "primary_cheap_10s.ops.cheap_global", **pc["ops"]["cheap_global"])
    chk("v2.cheap.ops", "deger", "54", N, "koşu başına 54 kez tazelenir")
    chk("v2.cheap.ops", "model_share", "2.0", N, "model payı %2,0", olcek=100)
    chk("v2.cheap.cheap_global-knn_global", "deger", "0.062", N, "kNN'den +0,062 kötü")
    chk("v2.cheap.cheap_global-knn_global", "b_better", "26", N, "kNN 26/38 önde")
    chk("v2.cheap.cheap_global-knn_global", "wilcoxon_p", "0.0009", N, "p=0,0009")
    chk("v2.cheap.cheap_global-gnn_global", "deger", "0.004", N, "GNN'den +0,004")
    chk("v2.cheap.cheap_global-gnn_global", "wilcoxon_p", "0.57", N, "p=0,57")
    CM = "results/kick_targeting/cheap_mechanism_summary.json"
    cm = L(CM)["summary"]
    scal(D, "v2.cheap.mech", "auc_cheap", "cheap", "vlsi58/ils_2s", cm["auc_cheap"], "AUC", CM, "summary",
         **{k: v for k, v in cm.items() if not isinstance(v, dict)},
         auc_cheap_wins=cm["auc_cheap_minus_knn"]["a_higher"], p_auc_cheap_knn=cm["auc_cheap_minus_knn"]["wilcoxon_p"])
    chk("v2.cheap.mech", "deger", "0.840", N, "AUC 0,840")
    chk("v2.cheap.mech", "auc_knn", "0.821", N, "kNN 0,821")
    chk("v2.cheap.mech", "auc_cheap_wins", "55", N, "58 problemin 55'inde")
    chk("v2.cheap.mech", "auc_gnn", "0.854", N, "GNN 0,854")
    chk("v2.cheap.mech", "ms_cheap", "2.7", N, "tazeleme 2,7 ms")
    chk("v2.cheap.mech", "ms_knn", "0.8", N, "kNN 0,8 ms")
    chk("v2.cheap.mech", "ms_gnn", "23", N, "GNN 23 ms")
    cj = L("models/cheap_scorer.json")
    scal(D, "v2.cheap.train", "dogrulama_auc_mlp", "cheap MLP", "gnn_train_data_lkh (20 dogrulama problemi)", cj["auc_mlp"],
         "AUC", "models/cheap_scorer.json", "auc_mlp", **cj, toplam_problem=cj["train_files"] + cj["val_files"])
    D_ = "deneysel_kurulum.tex"
    chk("v2.cheap.train", "auc_lr", "0.825", D_, "lojistik regresyon AUC 0,825")
    chk("v2.cheap.train", "deger", "0.832", D_, "MLP AUC 0,832")
    chk("v2.cheap.train", "auc_knn_rank", "0.812", D_, "kNN sırası 0,812")
    chk("v2.cheap.train", "train_files", "100", D_, "100 eğitim")
    chk("v2.cheap.train", "val_files", "20", D_, "20 doğrulama")
    chk("v2.cheap.train", "toplam_problem", "120", D_, "120 sentetik problem")
    chk("v2.cheap.train", "n_features", "17", D_, "17 girdi")
    # buyuk ornekler
    lg = C["large_30s"]
    gl = [v for k, v in lg["mean_gap"].items()]
    scal(D, "v2.large.gap_range", "ortalama_gap_aralik", "6 strateji", "vlsi_large_30s", min(gl), "yuzde", CS,
         "large_30s.mean_gap min/max", maks=max(gl), min_global=min(v for k, v in lg["mean_gap"].items() if k.endswith("global")))
    K2 = "bulgular_kesifsel.tex"
    chk("v2.large.gap_range", "deger", "1.96", K2, "bütün stratejiler %1,96--2,05 (min)")
    chk("v2.large.gap_range", "maks", "2.05", K2, "%1,96--2,05 (maks)")
    glob_pairs = {k: v for k, v in lg["pairs"].items() if k.count("global") == 2}
    scal(D, "v2.large.global_pairs", "maks_mutlak_fark", "global ciftler (6)", "vlsi_large_30s",
         max(abs(v["mean_diff"]) for v in glob_pairs.values()), "yuzde_puani", CS, "large_30s.pairs (global-global)",
         min_p=min(v["wilcoxon_p"] for v in glob_pairs.values()))
    chk("v2.large.global_pairs", "deger", "0.023", K2, "|Δ|≤0,023 puan", tur="le")
    chk("v2.large.global_pairs", "min_p", "0.46", K2, "p≥0,46", tur="ge")
    for st in ("n<10000", "n>=10000"):
        for k in ("random_local-random_global", "knn_local-knn_global"):
            a, b = k.split("-")
            comp(D, f"v2.large.{st}.{k}", {"a": a, "b": b, **lg[st][k]}, f"vlsi_large_30s/{st}", CS, f"large_30s['{st}'].{k}")
    chk("v2.large.n<10000.random_local-random_global", "b_better", "6", K2, "n<10.000: altı problemin altısında global iyi")
    chk("v2.large.n<10000.random_local-random_global", "deger", "0.26", K2, "rastgele için +0,26 puan")
    chk("v2.large.n>=10000.random_local-random_global", "a_better", "8", K2, "n≥10.000: onun sekizinde yerel iyi")
    chk("v2.large.n>=10000.random_local-random_global", "deger", "-0.26", K2, "−0,26 puan")
    chk("v2.large.n>=10000.random_local-random_global", "wilcoxon_p", "0.08", K2, "p=0,08")
    for s in ("gnn_global", "knn_global"):
        scal(D, f"v2.large.ops.{s}", "tazeleme_sayisi", s, "vlsi_large_30s", lg["ops"][s]["refreshes"], "adet", CS,
             f"large_30s.ops.{s}", **lg["ops"][s])
    chk("v2.large.ops.gnn_global", "deger", "19", K2, "GNN 19 kez tazelenir")
    chk("v2.large.ops.gnn_global", "stale_share", "7.8", K2, "bayat %7,8", olcek=100)
    chk("v2.large.ops.knn_global", "deger", "265", K2, "kNN 265 tazeleme")
    chk("v2.large.ops.knn_global", "stale_share", "0.1", K2, "kNN %0,1", olcek=100)

    # ---- havuz bilesimi
    PC = "results/kick_targeting/pool_composition_summary.json"
    pcs = L(PC)["summary"]
    for s in ("knn", "cheap", "gnn", "oracle"):
        scal(D, f"v2.pool.{s}", "fazla_uzunluk_yanlis_kenar", s, "vlsi57/ils_2s", pcs[s]["excess_wrong"], "medyan_kenar",
             PC, f"summary.{s}.excess_wrong", **pcs[s], n_instances=pcs["n_instances"])
    for s in ("gnn", "cheap"):
        e = pcs[f"excess_wrong_{s}_minus_knn"]
        scal(D, f"v2.pool.{s}_minus_knn", "fark_ortalama", f"{s} - knn (fazla uzunluk)", "vlsi57/ils_2s", e["mean"],
             "medyan_kenar", PC, f"summary.excess_wrong_{s}_minus_knn", a_higher=e["a_higher"],
             knn_higher=pcs["n_instances"] - e["a_higher"], wilcoxon_p=e["wilcoxon_p"])
    chk("v2.pool.knn", "n_instances", "57", N, "57 problem")
    chk("v2.pool.knn", "deger", "1.65", N, "kNN yanlış kenarları 1,65")
    chk("v2.pool.cheap", "deger", "1.50", N, "ucuz skor 1,50")
    chk("v2.pool.gnn", "deger", "1.30", N, "GNN 1,30")
    chk("v2.pool.oracle", "deger", "0.51", N, "kâhin 0,51")
    chk("v2.pool.gnn_minus_knn", "knn_higher", "57", N, "kNN GNN'yi 57/57")
    chk("v2.pool.cheap_minus_knn", "knn_higher", "56", N, "ucuz skoru 56/57")
    chk("v2.pool.gnn_minus_knn", "wilcoxon_p", "1e-8", N, "p<10^-8", tur="lt")
    chk("v2.pool.cheap_minus_knn", "wilcoxon_p", "1e-8", N, "p<10^-8", tur="lt")

    # ---- kazanc hedefli skor
    GS = "results/kick_targeting/gain_summary.json"
    G = L(GS)
    comp(D, "v2.gain.gain-knn", {"a": "gain_global", "b": "knn_global", **G["pairs"]["gain_global-knn_global"]},
         "vlsi_primary/gain", GS, "pairs.gain_global-knn_global")
    comp(D, "v2.gain.any0.25", {"a": "gain_global", "b": "knn_global",
                                **G["anytime"]["0.25"]["pairs"]["gain_global-knn_global"]},
         "vlsi_primary/gain@0.25s", GS, "anytime['0.25'].pairs.gain_global-knn_global")
    closed = 1 - G["pairs"]["gain_global-knn_global"]["mean_diff"] / pc["pairs"]["cheap_global-knn_global"]["mean_diff"]
    scal(D, "v2.gain.closed", "kapatilan_kayip_payi", "gain vs cheap (knn'ye karsi)", "vlsi_primary", closed, "oran",
         f"{GS} + {CS}", "1 - gain_global-knn_global / cheap_global-knn_global")
    chk("v2.gain.gain-knn", "deger", "0.016", N, "fark +0,016")
    chk("v2.gain.gain-knn", "ci_lo", "-0.031", N, "GA [−0,031; +0,061]")
    chk("v2.gain.gain-knn", "ci_hi", "0.061", N, "GA [−0,031; +0,061]")
    chk("v2.gain.gain-knn", "wilcoxon_p", "0.35", N, "p=0,35")
    chk("v2.gain.any0.25", "deger", "-0.112", N, "0,25 sn'de −0,112")
    chk("v2.gain.any0.25", "wilcoxon_p", "0.014", N, "p=0,014")
    chk("v2.gain.closed", "deger", "0.75", "bulgular_mekanizma_neden.tex/sonuc_ek.tex", "kaybın yaklaşık dörtte üçü", tur="yaklasik")
    gj = L("models/cheap_gain_scorer.json")
    scal(D, "v2.gain.train", "dogrulama_spearman", "gain MLP", "gnn_train_data_lkh (20 dogrulama)", gj["spearman_gain"], "",
         "models/cheap_gain_scorer.json", "spearman_gain", **gj,
         uyari="metindeki dogrulama havuz kazanci (1,90/2,47/2,58) ve kesinlik (%68/%60) icin kaynak dosya bulunamadi")

    # ---- LKH-3 baglami
    recs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in glob.glob(str(KT / "lkh_baseline" / "*.json"))]
    for cfg in ("default", "fast"):
        for tlim in (1.0, 10.0):
            per = defaultdict(list)
            for r in recs:
                if r["config"] == cfg and r["time_limit"] == tlim and r.get("ok"):
                    per[r["instance"]].append((r["gap_pct"], r["wall"]))
            g = [np.mean([x for x, _ in v]) for v in per.values()]
            w = [np.mean([y for _, y in v]) for v in per.values()]
            scal(D, f"v2.lkh.{cfg}.{tlim:g}", "ortalama_gap", f"LKH-3 {cfg}", "vlsi_primary", float(np.mean(g)), "yuzde",
                 "results/kick_targeting/lkh_baseline/*.json", f"config={cfg}, time_limit={tlim}; problem ortalamasi",
                 n_instances=len(per), wall_mean=float(np.mean(w)), wall_median=float(np.median(w)))
    TL = "tartisma_lkh.tex"
    chk("v2.lkh.fast.1", "deger", "0.20", TL, "hızlı ön-işlem 1 sn %0,20 gap")
    chk("v2.lkh.fast.1", "wall_mean", "1.8", TL, "ortalama 1,8 sn gerçek süre")
    chk("v2.lkh.default.10", "deger", "0.035", TL, "varsayılan 10 sn %0,035")
    chk("v2.lkh.default.10", "wall_mean", "19", TL, "ortalama 19 sn gerçek süre")
    best = P["mean_gap"]["knn_global"]
    for cfg, tlim, txt in (("fast", 1.0, "4"), ("default", 10.0, "20")):
        g = D.by_key[f"v2.lkh.{cfg}.{tlim:g}"]["deger"]
        scal(D, f"v2.lkh.ratio.{cfg}", "gap_orani", f"en iyi ILS / LKH-3 {cfg}", "vlsi_primary", best / g, "kat",
             f"{PS} + lkh_baseline", "mean_gap.knn_global / LKH ortalama gap")
        chk(f"v2.lkh.ratio.{cfg}", "deger", txt, TL, f"yaklaşık {txt} kat geride", tur="yaklasik")

    # ---- Asama 1 sayilari (ozet/tartisma): mevcut kayitlara eslenir, yeni kayit yok
    # ---- metin denetimi
    res = []
    for anahtar, alan, metin, tur, olcek, bolum, cumle in CHECKS:
        r = D.by_key[anahtar]
        if alan == "deger8":
            v, ok = r["deger"][:8], r["deger"][:8] == metin
        else:
            v = r["deger"] if alan == "deger" else r["ek"][alan]
            ok = uyum(v, metin, tur, olcek)
        res.append((r["id"], anahtar, alan, metin, tur, olcek, v, ok, bolum, cumle))
    by = defaultdict(list)
    for x in res:
        by[x[1]].append(x)
    for anahtar, xs in by.items():
        r = D.by_key[anahtar]
        r["ek"]["metin"] = [{"alan": x[2], "metin": x[3], "bolum": x[8], "uyumlu": x[7]} for x in xs]
        r["ek"]["metin_uyumlu"] = all(x[7] for x in xs)
    D.kaydet()
    ids = sorted({r["id"] for r in D.by_key.values()}, key=lambda s: int(s[1:]))
    bad = [x for x in res if not x[7]]
    print(f"Asama 2 kayitlari: {ids[0]}..{ids[-1]} ({len(ids)}); metin denetimi {len(res) - len(bad)}/{len(res)} uyumlu")
    for x in bad:
        print("  UYUMSUZ", x[0], x[1], x[2], "metin", x[3], "kaynak", x[6])
    yaz_ozet(res, ids)


def fmt(v, olcek):
    if isinstance(v, str):
        return v
    v = v * olcek
    return f"{v:.4g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"


def yaz_ozet(res, ids):
    L_ = ["<!-- ASAMA2-BASLA (otomatik: evidence/scripts/build_v2_evidence.py; elle duzenlemeyin) -->",
          "## Aşama 2 (PROTOCOL v1) — metin sayısı → kanıt kimliği eşlemesi", "",
          f"Kayıtlar: {ids[0]}–{ids[-1]} ({len(ids)} kayıt; üretici `evidence/scripts/build_v2_evidence.py`). "
          f"Denetlenen metin sayısı: {len(res)}; kaynakla uyumlu: {sum(x[7] for x in res)}; "
          f"**uyumsuz: {sum(not x[7] for x in res)}** (aşağıda ✗). Uyumsuzlukta doğru değer kayıttaki kaynak değeridir; "
          "metin yeniden yazılırken kayıttaki değer kullanılmalıdır. Yüzdeler kayıtta oran olarak tutulur "
          "(tabloda ×100 gösterilir).", "",
          "| bölüm | cümle (metin değeri) | ID · alan | kaynak değeri | |", "|---|---|---|---|---|"]
    for i, k, alan, metin, tur, olcek, v, ok, bolum, cumle in res:
        L_.append(f"| {bolum} | {cumle} | {i} · `{alan}` | {fmt(v, olcek)} | {'✓' if ok else '✗'} |")
    L_ += ["", "Aşama 1 sayıları (özet/tartışmada): 16 VLSI anlamlı yarar p=0,029 → **E557** (mevcut kayıt); "
           "Aşama 1 sentetik +0,167 / Holm 0,0014 → Aşama 1 kayıtları (E001–E583 içinde, `diversification_synthetic_100`); "
           "\"5 sn'de 0--6 kick\" için problem bazlı kick sayısı kaydı yok (mevcut E551/E555 yalnız ortalama: 11,7 / 11,5) — "
           "Aşama 1 metni yeniden yazılırken eklenmeli.",
           "<!-- ASAMA2-BITIR -->"]
    blok = "\n".join(L_) + "\n"
    t = OZET.read_text(encoding="utf-8")
    if "<!-- ASAMA2-BASLA" in t:
        t = re.sub(r"<!-- ASAMA2-BASLA.*?<!-- ASAMA2-BITIR -->\n?", blok, t, flags=re.S)
    else:
        t = t.rstrip("\n") + "\n\n" + blok
    OZET.write_text(t, encoding="utf-8")
    print("ozet guncellendi:", OZET.name)


if __name__ == "__main__":
    main()
