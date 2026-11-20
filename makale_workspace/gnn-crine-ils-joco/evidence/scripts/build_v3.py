# -*- coding: utf-8 -*-
"""Asama 3 (PROTOCOL_v2) kanit, tablo ve figur verisi ureticisi.

Kaynak (yazilim koku = tsplib/):
  results/kick_targeting/PROTOCOL_v2.json (+ .sha256, .ots), PROTOCOL_DEVIATIONS.log
  results/kick_targeting/holdout2_10s/*.json            (dogrulayici, 54 problem x 10 tohum x 8 strateji)
  results/kick_targeting/holdout2_10s_confirm.json      (V1-V4, v2_analysis.py confirm)
  results/kick_targeting/holdout2_10s_mechanism.json    (analysis_online_mechanism.py)
  results/kick_targeting/dev_*_10s/*.json               (gelistirme / kesifsel)
  data_holdout2/catalog.json                            (katmanlar)
Istatistik: v2_analysis.per_instance / v2_analysis.paired (degistirilmeden ice aktarilir; hash-kayitli).

Cikti:
  evidence/evidence.json        (E584+ kayitlari eklenir/guncellenir; ek.anahtar ile idempotent; silme yok)
  evidence/tables/v3_*.tex      (tam genislik booktabs, figure-style.md A4)
  figures/data/v3_*.dat, figures/data/manset*.{dat,csv}
  notes/vitrin-adaylari.md + evidence/vitrin.json (vitrin_sec.py; V... kayitlari evidence.json'a birlesir)

    py -3.13 evidence/scripts/build_v3.py
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from scipy import stats

WS = Path(__file__).resolve().parents[2]
ROOT = WS.parents[1]
KT = ROOT / "results" / "kick_targeting"
TAB = WS / "evidence" / "tables"
DAT = WS / "figures" / "data"
EVID = WS / "evidence" / "evidence.json"
SKILL_SCRIPTS = (ROOT.parents[1] / "yzmakaleuretimskill" / ".claude" / "skills" / "makale-uret" / "scripts")
TAB.mkdir(parents=True, exist_ok=True)
DAT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
import v2_analysis as A  # noqa: E402  (hash-kayitli; yalniz ice aktarilir)

TAG = "holdout2_10s"
SRC_H2 = "results/kick_targeting/holdout2_10s/*__s*.json"
R, KNN, EXC, GNN, GAIN, CHEAP, LIN, ONL = ("random_global", "knn_global@0.04:0.75", "excess_global@0.06:0.5",
                                          "gnn_global@0.04:0.5", "gain_global", "cheap_global",
                                          "linprior_global", "onlinewarm_global")
S8 = [R, KNN, EXC, GNN, GAIN, CHEAP, LIN, ONL]
TR = {R: "Rastgele", KNN: "kNN (0,04/0,75)", EXC: "Fazla-uzunluk (0,06/0,5)", GNN: "GNN (0,04/0,5)",
      GAIN: "Kazanç-MLP", CHEAP: "Üyelik-MLP", LIN: "Sabit doğrusal", ONL: "Çevrimiçi (ılık)",
      "knn_global": "kNN", "excess_global": "Fazla-uzunluk", "gain_global": "Kazanç-MLP",
      "linprior_global": "Sabit doğrusal", "onlinewarm_global": "Çevrimiçi (ılık)",
      "online_global": "Çevrimiçi (soğuk)", "random_global": "Rastgele", "cheap_global": "Üyelik-MLP"}
EN = {R: "Random", KNN: "kNN", EXC: "Excess", GNN: "GNN", GAIN: "Gain-MLP", CHEAP: "Member-MLP",
      LIN: "Static-linear", ONL: "Online"}
SHORT = {R: "random", KNN: "knn", EXC: "excess", GNN: "gnn", GAIN: "gain", CHEAP: "cheap", LIN: "linprior",
         ONL: "onlinewarm"}
STRATA = [("all", "Tümü", None), ("real", "Gerçek", {"vlsi6", "national3", "tsplib3"}),
          ("uniform", "Uniform", {"synth_uniform"}), ("clustered", "Kümeli", {"synth_clustered"}),
          ("mixed", "Karışık", {"synth_mixed"})]
BOOT, BSEED = 10000, 0


# ----------------------------------------------------------------------------- bicim yardimcilari
def tr(x, d=3):
    x = round(float(x), d) + 0.0
    return f"{x:.{d}f}".replace("-", "$-$").replace(".", "{,}")


def trs(x, d=3):
    x = round(float(x), d) + 0.0
    return ("+" if x > 0 else "") + tr(x, d)


def trp(p):
    if p >= 1e-3:
        return tr(p, 3)
    e = int(math.floor(math.log10(p)))
    m = p / 10 ** e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return f"${tr(m, 1)}\\times10^{{{e}}}$"


def ci(c):
    return f"[{trs(c[0])}; {trs(c[1])}]"


def wtl(r):
    return f"{r['wins_first']}/{r['ties']}/{r['wins_second']}"


def renk_fark(txt, r, alpha=0.05):
    """S8/kural 7: yalniz anlamli (p<alpha ve GA sifiri kesmiyor) farklar renklenir; negatif = A iyi."""
    sig = r["p_wilcoxon"] < alpha and (r["ci95"][1] < 0 or r["ci95"][0] > 0)
    if not sig:
        return txt
    return f"\\iyi{{{txt}}}" if r["mean"] < 0 else f"\\kotu{{{txt}}}"


def renk_sutun(vals, yon):
    """yon: 'dusuk' | 'yuksek' -> (en iyi indeks, en kotu indeks)."""
    arr = np.array(vals, dtype=float)
    if yon == "dusuk":
        return int(np.nanargmin(arr)), int(np.nanargmax(arr))
    return int(np.nanargmax(arr)), int(np.nanargmin(arr))


def write_tab(name, body):
    (TAB / name).write_text("% OTOMATIK URETIM -- evidence/scripts/build_v3.py (elle duzenleme)\n" + body,
                            encoding="utf-8")
    print("tablo:", name)


def write_dat(name, header, rows, sep=" "):
    lines = [sep.join(header)] + [sep.join(str(v) for v in r) for r in rows]
    (DAT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("veri :", name)


def f5(x):
    return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.5f}"


# ----------------------------------------------------------------------------- kanit defteri
class Defter:
    def __init__(self, path):
        self.path = path
        self.d = json.loads(path.read_text(encoding="utf-8"))
        self.k = self.d["kayitlar"]
        self.by_key = {r["ek"]["anahtar"]: r for r in self.k
                       if isinstance(r.get("ek"), dict) and r["ek"].get("uretici") == "build_v3"}
        nums = [int(r["id"][1:]) for r in self.k if r["id"].startswith("E") and r["id"][1:].isdigit()]
        self.next = max(nums) + 1
        self.first_new = None
        self.touched = []

    def ekle(self, anahtar, metrik, yontem, veri_seti, deger, birim, kaynak_dosya, kaynak_konum, **ek):
        ek = {"uretici": "build_v3", "anahtar": anahtar, "asama": 3, **ek}
        rec = {"metrik": metrik, "yontem": yontem, "veri_seti": veri_seti,
               "deger": (float(deger) if isinstance(deger, (int, float, np.floating, np.integer))
                         and not isinstance(deger, bool) else deger),
               "birim": birim, "kaynak": "yazilim", "kaynak_dosya": kaynak_dosya, "kaynak_konum": kaynak_konum,
               "atif": None}
        old = self.by_key.get(anahtar)
        if old is not None:
            durum = old.get("durum", "gecerli")
            par = old.get("paradigma")
            old.update(rec)
            old["ek"] = ek
            old["durum"] = durum          # sefin koydugu "terk" korunur
            if par:
                old["paradigma"] = par
            self.touched.append(old["id"])
            return old["id"]
        rid = f"E{self.next:03d}"
        self.next += 1
        self.first_new = self.first_new or rid
        new = {"id": rid, **rec, "durum": "gecerli", "ek": ek}
        self.k.append(new)
        self.by_key[anahtar] = new
        self.touched.append(rid)
        return rid

    def vitrin_birlestir(self, vpath):
        if not vpath.exists():
            return []
        v = json.loads(vpath.read_text(encoding="utf-8"))
        ids = []
        index = {r["id"]: i for i, r in enumerate(self.k)}
        for r in v["kayitlar"]:
            r = dict(r)
            r["ek"] = {"uretici": "vitrin_sec.py", "asama": 3, "vitrin_meta": v["vitrin_meta"]}
            if r["id"] in index:
                old = self.k[index[r["id"]]]
                r["durum"] = old.get("durum", r.get("durum", "gecerli"))
                self.k[index[r["id"]]] = r
            else:
                self.k.append(r)
            ids.append(r["id"])
        return ids

    def kaydet(self, meta):
        self.d["deney_meta_v3"] = meta
        self.path.write_text(json.dumps(self.d, ensure_ascii=False, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- istatistik
def paired_keys(data, keys, a, b):
    """v2_analysis.paired ile ozdes hesap, problem alt kumesi icin (ayni bootstrap tohumu ve sayisi)."""
    d = np.array([data[k][a] - data[k][b] for k in keys])
    rng = np.random.default_rng(BSEED)
    bs = d[rng.integers(len(d), size=(BOOT, len(d)))].mean(1)
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return {"pair": f"{a} - {b}", "n_inst": len(d), "mean": float(d.mean()), "median": float(np.median(d)),
            "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "wins_first": int((d < 0).sum()), "ties": int((d == 0).sum()), "wins_second": int((d > 0).sum()),
            "sd": float(d.std(ddof=1)), "p_wilcoxon": float(p)}


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj.tolist()


def load_runs(tag):
    """(instance, seed) -> json; ham sayaclar icin."""
    out = {}
    for f in sorted((KT / tag).glob("*__s*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[(d["instance"], d["seed"])] = d
    return out


# ============================================================================= ana akis
def main():
    D = Defter(EVID)
    proto = json.loads((KT / "PROTOCOL_v2.json").read_text(encoding="utf-8"))
    sha_file = (KT / "PROTOCOL_v2.sha256").read_text(encoding="utf-8").split()[0]
    sha_now = hashlib.sha256((KT / "PROTOCOL_v2.json").read_bytes()).hexdigest()
    assert sha_file == sha_now, "PROTOCOL_v2.json hash degismis!"
    for f, h in proto["code_and_data_sha256"].items():
        p = ROOT / f
        if p.exists() and f.endswith(".py"):
            assert hashlib.sha256(p.read_bytes()).hexdigest() == h, f"hash uyusmazligi: {f}"
    cat = json.loads((ROOT / "data_holdout2" / "catalog.json").read_text(encoding="utf-8"))["instances"]
    data, ns = A.per_instance(TAG)
    keys_all = list(data)
    assert len(keys_all) == 54 and all(len(data[k]) == 8 for k in keys_all)
    runs = load_runs(TAG)
    assert len(runs) == 540, len(runs)
    strat_keys = {sk: ([k for k in keys_all if cat[k]["stratum"] in st] if st else keys_all)
                  for sk, _, st in STRATA}

    # ---- protokol meta kayitlari
    D.ekle("v3.protokol.sha256", "protokol_sha256", "PROTOCOL_v2", "holdout2", sha_now, "",
           "results/kick_targeting/PROTOCOL_v2.sha256", "satir 1", not_="SHA-256 dondurulmus protokol")
    D.ekle("v3.protokol.bitcoin_blok", "zaman_damgasi_blok", "OpenTimestamps", "holdout2", 976258, "blok",
           "results/kick_targeting/PROTOCOL_DEVIATIONS.log",
           "[2026-11-17 15:01] satiri: BitcoinBlockHeaderAttestation(976258)",
           not_="PROTOCOL_v2.json 'external_timestamp' alani OSF diyor; OSF yerine OTS kullanildi (sapma kaydi 12:51)")
    D.ekle("v3.protokol.n_inst", "problem_sayisi", "holdout2", "holdout2", proto["holdout_set"]["n_instances"],
           "adet", "results/kick_targeting/PROTOCOL_v2.json", "holdout_set.n_instances")
    D.ekle("v3.protokol.seeds", "tohum_sayisi", "holdout2", "holdout2", proto["seeds"], "adet",
           "results/kick_targeting/PROTOCOL_v2.json", "seeds")
    D.ekle("v3.protokol.budget", "butce", "holdout2", "holdout2", proto["budget_seconds"], "sn",
           "results/kick_targeting/PROTOCOL_v2.json", "budget_seconds")
    D.ekle("v3.protokol.margin", "kotu_degil_siniri", "V3,V4", "holdout2", 0.05, "yuzde_puani",
           "results/kick_targeting/PROTOCOL_v2.json", "confirmatory_hypotheses[V3,V4].margin")
    D.ekle("v3.protokol.n_runs", "kosum_sayisi", "holdout2", "holdout2", len(runs) * 8, "adet", SRC_H2,
           "540 dosya x 8 strateji (runs alani)")
    for st in ("vlsi6", "national3", "tsplib3", "synth_uniform", "synth_clustered", "synth_mixed"):
        D.ekle(f"v3.protokol.katman.{st}", "katman_problem_sayisi", st, "holdout2",
               sum(1 for c in cat.values() if c["stratum"] == st), "adet", "data_holdout2/catalog.json",
               f"instances[*].stratum == {st}")
    nlist = [cat[k]["n"] for k in keys_all]
    D.ekle("v3.protokol.n_min", "problem_boyutu_min", "holdout2", "holdout2", min(nlist), "dugum",
           "data_holdout2/catalog.json", "min(instances[*].n)")
    D.ekle("v3.protokol.n_max", "problem_boyutu_max", "holdout2", "holdout2", max(nlist), "dugum",
           "data_holdout2/catalog.json", "max(instances[*].n)")

    # ---- (a) dogrulayici V1-V4
    conf = json.loads((KT / "holdout2_10s_confirm.json").read_text(encoding="utf-8"))
    hyp = {h["id"]: h for h in proto["confirmatory_hypotheses"]}
    conf_rows = []
    for c in conf:
        r = A.paired(TAG, c["a"], c["b"])
        assert abs(r["mean"] - c["mean"]) < 1e-12, (c["id"], r["mean"], c["mean"])
        h = hyp[c["id"]]
        rid = D.ekle(f"v3.confirm.{c['id']}", "dogrulayici_fark_ortalama", f"{c['a']} - {c['b']}", "holdout2",
                     c["mean"], "yuzde_puani", "results/kick_targeting/holdout2_10s_confirm.json",
                     f"[id={c['id']}].mean", hipotez=c["id"], test=c["test"], margin=h.get("margin"),
                     p=c["p"], rejected=c["rejected"], tested=c["tested"], n_inst=c["n_inst"],
                     ci95=r["ci95"], wins_a=r["wins_first"], ties=r["ties"], wins_b=r["wins_second"],
                     p_wilcoxon_iki_yonlu=r["p_wilcoxon"], median=r["median"],
                     ci_wtl_kaynak=f"v2_analysis.paired('{TAG}', a, b) <- {SRC_H2}",
                     not_="on-kayitli, sabit sirali test (PROTOCOL_v2)")
        conf_rows.append((c, r, h, rid))

    # ikincil (kesifsel) -- protokolde adi gecen 5 cift, Holm
    SEC = [(KNN, R), (EXC, KNN), (GNN, KNN), (GAIN, CHEAP), (EXC, GNN)]
    sec_res = [A.paired(TAG, a, b) for a, b in SEC]
    sec_holm = holm([r["p_wilcoxon"] for r in sec_res])
    sec_rows = []
    for (a, b), r, ph in zip(SEC, sec_res, sec_holm):
        rid = D.ekle(f"v3.secondary.{SHORT[a]}-{SHORT[b]}", "ikincil_fark_ortalama", f"{a} - {b}", "holdout2",
                     r["mean"], "yuzde_puani", SRC_H2, f"v2_analysis.paired('{TAG}','{a}','{b}')",
                     ci95=r["ci95"], wins_a=r["wins_first"], ties=r["ties"], wins_b=r["wins_second"],
                     p_raw=r["p_wilcoxon"], p_holm=ph, holm_aile="PROTOCOL_v2.secondary_exploratory[0] (5 cift)",
                     median=r["median"], sd=r["sd"], not_="kesifsel (on-kayitli ikincil aile, Holm/5)")
        sec_rows.append((a, b, r, ph, rid))
    tab_confirm(conf_rows, sec_rows)

    # ---- (b) 8 strateji x katman
    strat_res = {}
    for sk, slab, _ in STRATA:
        keys = strat_keys[sk]
        for s in S8:
            g = [data[k][s] for k in keys]
            r = None if s == R else paired_keys(data, keys, s, R)
            if sk == "all" and r is not None:
                r_ref = A.paired(TAG, s, R)
                assert all(abs(r[x] - r_ref[x]) < 1e-12 for x in ("mean", "p_wilcoxon")) and r["ci95"] == r_ref["ci95"]
            ek = dict(n_inst=len(keys), median_gap=float(np.median(g)), katman=sk)
            if r is not None:
                ek.update(fark_rastgele=r["mean"], ci95_fark=r["ci95"], wins=r["wins_first"], ties=r["ties"],
                          losses=r["wins_second"], p_wilcoxon=r["p_wilcoxon"])
            rid = D.ekle(f"v3.h2.gap.{sk}.{SHORT[s]}", "ortalama_gap", s, f"holdout2/{sk}", float(np.mean(g)),
                         "yuzde", SRC_H2, f"v2_analysis.per_instance('{TAG}') katman={sk}; fark: rastgele ile eslenik",
                         not_="kesifsel (V2 disindaki karsilastirmalar on-kayitli degil)", **ek)
            strat_res[(sk, s)] = (float(np.mean(g)), r, rid)
    # fazla-uzunluk vs tum strateji (her katman) -- kazanma oranlari
    for sk, _, _ in STRATA:
        for b in S8:
            if b == EXC:
                continue
            r = paired_keys(data, strat_keys[sk], EXC, b)
            D.ekle(f"v3.h2.excess_vs.{sk}.{SHORT[b]}", "fark_ortalama", f"{EXC} - {b}", f"holdout2/{sk}",
                   r["mean"], "yuzde_puani", SRC_H2, f"eslenik fark, katman={sk}", ci95=r["ci95"],
                   wins=r["wins_first"], ties=r["ties"], losses=r["wins_second"], p_wilcoxon=r["p_wilcoxon"],
                   n_inst=r["n_inst"], not_="kesifsel")
    tab_strategies(strat_res)

    # ---- anytime (protokol ikincil listesi)
    cps = sorted(next(iter(runs.values()))["runs"][R]["trace_gap_pct"], key=float)
    any_rows = []
    for cp in cps:
        dcp, _ = A.per_instance(TAG, float(cp) if cp != "10.0" else 10.0)
        row = [cp]
        for s in S8:
            row.append(f5(float(np.mean([dcp[k][s] for k in dcp]))))
        for b in (R, GNN):
            r = paired_keys(dcp, list(dcp), EXC, b)
            D.ekle(f"v3.h2.anytime.{cp}.excess-{SHORT[b]}", "fark_ortalama_anytime", f"{EXC} - {b}", "holdout2",
                   r["mean"], "yuzde_puani", SRC_H2, f"runs[*].trace_gap_pct['{cp}']", checkpoint_s=float(cp),
                   ci95=r["ci95"], wins=r["wins_first"], losses=r["wins_second"], p_wilcoxon=r["p_wilcoxon"],
                   not_="kesifsel")
            row += [f5(r["mean"]), f5(r["ci95"][0]), f5(r["ci95"][1])]
        any_rows.append(row)
    write_dat("v3_anytime.dat", ["t"] + [SHORT[s] for s in S8] +
              ["d_rand", "lo_rand", "hi_rand", "d_gnn", "lo_gnn", "hi_gnn"], any_rows)

    # ---- (c) mekanizma / verimlilik
    mech = json.loads((KT / "holdout2_10s_mechanism.json").read_text(encoding="utf-8"))
    raw = defaultdict(lambda: defaultdict(list))
    per_inst_acc = defaultdict(lambda: defaultdict(list))
    for (inst, seed), d in runs.items():
        for s, r in d["runs"].items():
            raw[s]["kicks_per_s"].append(r["kicks"] / r["elapsed"])
            raw[s]["accept_rate"].append(r["accepted"] / max(r["kicks"], 1))
            raw[s]["model_share"].append(r["model_time"] / r["elapsed"])
            raw[s]["stale_share"].append(r["stale_cuts"] / max(r["guided_cuts"], 1))
            per_inst_acc[inst][s].append(r["accepted"] / max(r["kicks"], 1))
    mech_vals = {}
    for s in S8:
        for m in ("kicks_per_s", "accept_rate", "model_share", "stale_share"):
            v_json = mech["counters"][s][m]
            v_raw = float(np.mean(raw[s][m]))
            assert abs(v_json - v_raw) < 1e-9 * max(1, abs(v_raw)), (s, m, v_json, v_raw)
            birim = "vurus/sn" if m == "kicks_per_s" else "oran"
            rid = D.ekle(f"v3.mech.{m}.{SHORT[s]}", f"mekanizma_{m}", s, "holdout2", v_json, birim,
                         "results/kick_targeting/holdout2_10s_mechanism.json", f"counters['{s}'].{m}",
                         tanim={"kicks_per_s": "kicks/elapsed", "accept_rate": "accepted/kicks",
                                "model_share": "model_time/elapsed",
                                "stale_share": "stale_cuts/guided_cuts"}[m],
                         dogrulama="ham kosum dosyalarindan yeniden hesaplandi (540x)", not_="kesifsel, post-hoc")
            mech_vals[(s, m)] = (v_json, rid)
    dr = mech["drift"]
    drift_keys = ["rel_len_all", "rel_len_pool_prior", "rel_len_pool_excess", "rel_len_pool_learned",
                  "overlap_prior_excess", "overlap_learned_excess", "overlap_learned_prior", "cos_w0_wfinal"]
    drift_stats = {}
    for k in drift_keys:
        v = np.array([r[k] for r in dr])
        assert abs(v.mean() - mech["drift_mean"][k]) < 1e-12
        q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
        extra = {}
        if k == "cos_w0_wfinal":
            extra["negatif_sayisi"] = int((v < 0).sum())
        if k == "rel_len_pool_learned":
            extra["tur_ortalamasinin_altinda"] = int(sum(r["rel_len_pool_learned"] < r["rel_len_all"] for r in dr))
        rid = D.ekle(f"v3.drift.{k}", f"cevrimici_kayma_{k}", ONL, "holdout2", float(v.mean()), "oran",
                     "results/kick_targeting/holdout2_10s_mechanism.json", f"drift_mean.{k} (drift[*], 54 problem, tohum 0)",
                     median=float(med), q1=float(q1), q3=float(q3), min=float(v.min()), max=float(v.max()),
                     not_="kesifsel, post-hoc; goreli uzunluk = d(u,v) / (uc noktalarin 5. en yakin komsu mesafesi ort.)",
                     **extra)
        drift_stats[k] = (float(v.mean()), float(med), float(q1), float(q3), rid)
    tab_mechanism(mech_vals, drift_stats, strat_res)
    # cevrimici kayma figuru verisi
    rows = []
    for r in sorted(dr, key=lambda r: r["n"]):
        acc = per_inst_acc[r["instance"]]
        rows.append([r["instance"], r["n"], f5(r["rel_len_all"]), f5(r["rel_len_pool_prior"]),
                     f5(r["rel_len_pool_excess"]), f5(r["rel_len_pool_learned"]), f5(r["cos_w0_wfinal"]),
                     f5(100 * np.mean(acc[LIN])), f5(100 * np.mean(acc[ONL])), f5(100 * np.mean(acc[EXC])),
                     f5(100 * np.mean(acc[R]))])
    write_dat("v3_online_drift.dat", ["instance", "n", "rel_all", "rel_prior", "rel_excess", "rel_learned", "cos",
                                      "acc_linprior", "acc_online", "acc_excess", "acc_random"], rows)
    n_acc_lower = sum(np.mean(per_inst_acc[i][ONL]) < np.mean(per_inst_acc[i][LIN]) for i in per_inst_acc)
    D.ekle("v3.drift.acc_online_lt_linprior", "kabul_orani_dusen_problem_sayisi", f"{ONL} vs {LIN}", "holdout2",
           int(n_acc_lower), "adet", SRC_H2, "problem bazinda ortalama accepted/kicks karsilastirmasi (10 tohum)",
           n_inst=54, not_="kesifsel")

    # ---- (d) gelistirme + havuz kurali izgarasi
    dev_rows = tab_development(D)

    # ---- HERO: problem bazinda tablo + vitrin_sec
    hero(D, data, cat, keys_all)
    write_drift_tex()
    write_grid_tex()

    meta = {"tarih": "2026-11-17", "donanim": proto["machine"], "kosul": "holdout2, 10 sn, 10 tohum, 8 strateji",
            "protokol_sha256": sha_now, "uretici": "evidence/scripts/build_v3.py",
            "istatistik": "problem birim; tohum ici ortalama; Wilcoxon isaretli sira; bootstrap 10000 (tohum 0)"}
    D.kaydet(meta)
    ids = sorted(set(D.touched), key=lambda x: int(x[1:]))
    print(f"kanit kayitlari (build_v3): {ids[0]}..{ids[-1]} ({len(ids)} kayit); ilk yeni: {D.first_new}")


# ----------------------------------------------------------------------------- tablolar
def tab_confirm(conf_rows, sec_rows):
    TEST = {"superiority": "üstünlük", "noninferiority": "kötü-değil"}
    TR = {**globals()["TR"], KNN: "kNN", EXC: "Fazla-uzunluk", GNN: "GNN", ONL: "Çevrimiçi"}
    rows = []
    for c, r, h, rid in conf_rows:
        karar = "$H_0$ ret" if c["rejected"] else ("ret yok" if c["tested"] else "test edilmedi")
        rows.append(f"{c['id']} & {TR[c['a']]} $-$ {TR[c['b']]} & {TEST[c['test']]} & "
                    f"{renk_fark(trs(r['mean']), r)} & {ci(r['ci95'])} & {wtl(r)} & {trp(c['p'])} & {karar} \\\\"
                    f" % {rid}")
    rows.append("\\midrule")
    rows.append("\\multicolumn{8}{l}{\\textit{İkincil, keşifsel (protokolde adı geçen 5 çift; $p$ Holm düzeltmeli)}} \\\\")
    for a, b, r, ph, rid in sec_rows:
        karar = "anlamlı" if ph < 0.05 else "anlamsız"
        rr = dict(r, p_wilcoxon=ph)
        rows.append(f"-- & {TR[a]} $-$ {TR[b]} & üstünlük & {renk_fark(trs(r['mean']), rr)} & {ci(r['ci95'])} & "
                    f"{wtl(r)} & {trp(ph)} & {karar} \\\\ % {rid}")
    body = (
        "\\begin{table}[!htbp]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{Doğrulayıcı test (PROTOCOL\\_v2, holdout2: 54 problem, 10 tohum, 10~sn). $\\Delta$: A$-$B gap farkı "
        "[puan], negatif A lehine; kötü-değil sınırı 0{,}05. Renk: anlamlı fark. Ö/E/G: A'nın önde, eşit ve geride olduğu problem sayısı.}\n"
        "\\label{tab:v3-confirm}\n"
        "\\begin{tabularx}{\\textwidth}{@{}l>{\\raggedright\\arraybackslash}Xlccccl@{}}\n\\toprule\n"
        "ID & Karşılaştırma (A$-$B) & Test & $\\Delta$ & \\%95 GA & Ö/E/G & $p$ & Karar \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n")
    write_tab("v3_confirm.tex", body)


def tab_strategies(res):
    rows = []
    for sk, slab, _ in STRATA:
        gaps = [res[(sk, s)][0] for s in S8]
        best, worst = renk_sutun(gaps, "dusuk")
        nk = res[(sk, EXC)][1]["n_inst"]
        comp = [s for s in S8 if res[(sk, s)][1] is not None]
        padj = dict(zip(comp, holm([res[(sk, s)][1]["p_wilcoxon"] for s in comp])))  # O6: katman ici Holm
        ref_rows, kat_rows = [], []
        for i, s in enumerate(S8):
            g, r, rid = res[(sk, s)]
            gt = tr(g)
            gt = f"\\iyi{{{gt}}}" if i == best else (f"\\kotu{{{gt}}}" if i == worst else gt)
            if r is None:
                # katman başlığı referans satırıyla birleşik (sn-jnl metin yüksekliğine sığması için)
                ref_rows.append(f"\\textit{{{slab} ({nk} problem)}}: {TR[s]} & {gt} & \\multicolumn{{4}}{{c}}{{referans}} \\\\ % {rid}")
            else:
                kat_rows.append(f"\\quad {TR[s]} & {gt} & {renk_fark(trs(r['mean']), {**r, 'p_wilcoxon': padj[s]})} & {ci(r['ci95'])} & {wtl(r)} & "
                                f"{trp(r['p_wilcoxon'])} \\\\ % {rid}")
        rows.extend(ref_rows + kat_rows)
        rows.append("\\addlinespace")
    body = (
        "\\begin{table}[!htbp]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{Holdout2'de sekiz strateji, katman bazında: ortalama gap ve rastgeleye eşlenik fark $\\Delta$. "
        "Renk: gap sütununda katmanın en iyi/en kötüsü, $\\Delta$'da katman içi Holm düzeltmeli $p<0{,}05$; $p$ sütunu düzeltmesizdir. Ö/E/G: stratejinin rastgeleden önde, eşit ve geride olduğu problem sayısı.}\n"
        "\\label{tab:v3-strategies}\n"
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccc@{}}\n\\toprule\n"
        "Strateji & Gap [\\%] & $\\Delta$ [puan] & \\%95 GA & Ö/E/G & $p$ \\\\\n\\midrule\n"
        + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    write_tab("v3_strategies.tex", body)


def tab_mechanism(mv, ds, strat_res):
    cols = [("kicks_per_s", None, lambda v: f"{v:,.0f}".replace(",", ".")), ("accept_rate", None, lambda v: tr(100 * v, 2)),
            ("model_share", "dusuk", lambda v: tr(100 * v, 2)), ("stale_share", "dusuk", lambda v: tr(100 * v, 2))]
    cells = {s: {} for s in S8}
    for m, yon, fmt in cols:
        vals = [mv[(s, m)][0] for s in S8]
        if yon:
            sub = [v if s != R else np.nan for s, v in zip(S8, vals)]   # rastgele: model yok (referans, notr)
            b, w = renk_sutun(sub, yon)
        for i, s in enumerate(S8):
            t = fmt(vals[i])
            if yon and s != R:
                t = f"\\iyi{{{t}}}" if i == b else (f"\\kotu{{{t}}}" if i == w else t)
            cells[s][m] = t
    gaps = [strat_res[("all", s)][0] for s in S8]
    gb, gw = renk_sutun(gaps, "dusuk")
    rows = []
    for i, s in enumerate(S8):
        gt = tr(gaps[i])
        gt = f"\\iyi{{{gt}}}" if i == gb else (f"\\kotu{{{gt}}}" if i == gw else gt)
        ids = ",".join(mv[(s, m)][1] for m, _, _ in cols)
        rows.append(f"{TR[s]} & {cells[s]['kicks_per_s']} & {cells[s]['accept_rate']} & {cells[s]['model_share']} & "
                    f"{cells[s]['stale_share']} & {gt} \\\\ % {ids}")
    rows.append("\\midrule")
    rows.append("\\multicolumn{6}{l}{\\textit{Çevrimiçi ağırlık kayması (54 problem, tohum 0): ortalama / medyan / "
                "[Q1; Q3]}} \\\\")
    lab = {"rel_len_all": "Uzunluk: tüm tur kenarları",
           "rel_len_pool_prior": "Uzunluk: ön-bilgi havuzu",
           "rel_len_pool_excess": "Uzunluk: fazla-uz. havuzu",
           "rel_len_pool_learned": "Uzunluk: öğrenilmiş havuz",
           "overlap_prior_excess": "Örtüşme: ön-bilgi ∩ fazla-uz.",
           "overlap_learned_excess": "Örtüşme: öğrenilmiş ∩ fazla-uz.",
           "cos_w0_wfinal": "$\\cos(w_0, w_{\\text{son}})$"}
    for k, l in lab.items():
        mean, med, q1, q3, rid = ds[k]
        l = l.replace("∩", "$\\cap$")
        rows.append(f"\\quad {l} & {tr(mean)} & {tr(med)} & \\multicolumn{{2}}{{c}}{{[{tr(q1)}; {tr(q3)}]}} & \\\\ % {rid}")
    body = (
        "\\begin{table}[!htbp]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{Mekanizma ve verimlilik, holdout2 (540 koşu/strateji). Model: skor süresinin duvar-saatine oranı; "
        "Bayat: turda artık olmayan havuz kenarına düşen kesimler. Renk: sütunun en iyi/en kötüsü.}\n"
        "\\label{tab:v3-mechanism}\n"
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccc}\n\\toprule\n"
        "Strateji & Vuruş/sn & Kabul [\\%] & Model [\\%] & Bayat [\\%] & Gap [\\%] \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    write_tab("v3_mechanism.tex", body)


DEV = [("dev_vlsi_primary_10s", "VLSI-38", "knn_global", "random_global"),
       ("dev_vlsi_primary_10s", "VLSI-38", "excess_global", "random_global"),
       ("dev_vlsi_primary_10s", "VLSI-38", "excess_global", "knn_global"),
       ("dev_vlsi_primary_10s", "VLSI-38", "gain_global", "knn_global"),
       ("dev_vlsi_primary_10s", "VLSI-38", "onlinewarm_global", "linprior_global"),
       ("dev_excess_synth1000_10s", "Sent-1000", "excess_global", "random_global"),
       ("dev_excess_synth1000_10s", "Sent-1000", "excess_global", "knn_global"),
       ("dev_online_synth1000_10s", "Sent-1000", "onlinewarm_global", "excess_global"),
       ("dev_excess_vlsi_old_10s", "VLSI-20", "knn_global", "random_global"),
       ("dev_online_vlsi_old_10s", "VLSI-20", "online_global", "excess_global")]
GRID_TAGS = [("dev_grid_vlsi_primary_10s", "vlsi_primary"), ("dev_grid_synth1000_10s", "synth_1000")]
FRACS = ["0.02", "0.04", "0.06", "0.10"]
PROBS = ["0.5", "0.75", "0.9"]
SIG = [("knn_global", "kNN"), ("excess_global", "Fazla-uzunluk"), ("gnn_global", "GNN")]
SIG_EN = {"knn_global": "kNN", "excess_global": "Excess", "gnn_global": "GNN"}


def tab_development(D):
    rows = []
    for tag, setlab, a, b in DEV:
        r = A.paired(tag, a, b)
        rid = D.ekle(f"v3.dev.{tag}.{a}-{b}", "gelistirme_fark_ortalama", f"{a} - {b}", tag, r["mean"],
                     "yuzde_puani", f"results/kick_targeting/{tag}/*__s*.json", f"v2_analysis.paired('{tag}','{a}','{b}')",
                     ci95=r["ci95"], wins_a=r["wins_first"], ties=r["ties"], wins_b=r["wins_second"],
                     p_wilcoxon=r["p_wilcoxon"], n_inst=r["n_inst"], sd=r["sd"],
                     not_="gelistirme / kesifsel; yontem ve ayar secimi icin kullanildi, dogrulayici degil")
        rows.append(f"{TR[a]} $-$ {TR[b]} & {setlab} ({r['n_inst']}) & {renk_fark(trs(r['mean']), r)} & "
                    f"{ci(r['ci95'])} & {wtl(r)} & {trp(r['p_wilcoxon'])} \\\\ % {rid}")
    # izgara
    obj, objd = {}, {}
    per_set = {}
    for tag, setk in GRID_TAGS:
        data, _ = A.per_instance(tag)
        per_set[setk] = data
    pooled_keys = [(setk, k) for setk, data in per_set.items() for k in data]
    rand_mean = {setk: float(np.mean([d[k]["random_global"] for k in d])) for setk, d in per_set.items()}
    for setk in per_set:
        D.ekle(f"v3.grid.{setk}.random", "izgara_ortalama_gap", "random_global", f"dev_grid/{setk}",
               rand_mean[setk], "yuzde", f"results/kick_targeting/dev_grid_{'vlsi_primary' if setk=='vlsi_primary' else 'synth1000'}_10s/*__s*.json",
               "v2_analysis.per_instance; problem ortalamasi", n_inst=len(per_set[setk]), not_="kesifsel ayar")
    grid_dat = []
    choice = {}
    for sig, _ in SIG:
        for fr in FRACS:
            for pr in PROBS:
                s = f"{sig}@{fr}:{pr}"
                ms = {setk: float(np.mean([d[k][s] for k in d])) for setk, d in per_set.items()}
                o = float(np.mean(list(ms.values())))
                od = float(np.mean([ms[x] - rand_mean[x] for x in ms]))
                obj[s], objd[s] = o, od
        cand = [f"{sig}@{fr}:{pr}" for fr in FRACS for pr in PROBS]
        best = min(cand, key=lambda s: obj[s])
        default = f"{sig}@0.06:0.75"
        chosen = default if obj[default] - obj[best] <= 0.01 else best
        diffs = np.array([per_set[sk][k][chosen] - per_set[sk][k][default] for sk, k in pooled_keys])
        pch = float(stats.wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0
        choice[sig] = (chosen, best, default, pch)
        fr02 = [objd[f"{sig}@0.02:{pr}"] for pr in PROBS]
        rest = [objd[f"{sig}@{fr}:{pr}"] for fr in FRACS[1:] for pr in PROBS]
        D.ekle(f"v3.grid.choice.{sig}", "izgara_secim", sig, "dev_grid (vlsi_primary + synth_1000)",
               chosen.split("@")[1], "frac:prob", "results/kick_targeting/dev_grid_*_10s/*__s*.json",
               "on-yazili kural: iki setin esit agirlikli ortalama gap'i min; varsayilan 0.06:0.75 0.01 icindeyse korunur",
               en_iyi=best, varsayilan=default, amac_secilen=obj[chosen], amac_varsayilan=obj[default],
               p_secilen_vs_varsayilan=pch, p_birim="78 problem (38+40) birlesik Wilcoxon",
               protokolle_uyumlu=(chosen.split("@")[1] == {"knn_global": "0.04:0.75", "excess_global": "0.06:0.5",
                                                            "gnn_global": "0.04:0.5"}[sig]),
               frac002_ort_fark=float(np.mean(fr02)), diger_ort_fark=float(np.mean(rest)), not_="kesifsel ayar")
    for sig, _ in SIG:
        for fr in FRACS:
            for pr in PROBS:
                s = f"{sig}@{fr}:{pr}"
                rid = D.ekle(f"v3.grid.cell.{s}", "izgara_fark_rastgele", s, "dev_grid (vlsi_primary + synth_1000)",
                             objd[s], "yuzde_puani", "results/kick_targeting/dev_grid_*_10s/*__s*.json",
                             "iki setin esit agirlikli ortalamasi: mean_k gap(s) - mean_k gap(random)",
                             amac_gap=obj[s], not_="kesifsel ayar (3 tohum)")
                grid_dat.append((sig, fr, pr, objd[s], rid))
    # tablo bolum B: satir = sinyal x olasilik, sutun = havuz orani
    rows.append("\\midrule")
    rows.append("\\multicolumn{6}{l}{\\textit{Havuz kuralı ızgarası: rastgeleye fark [puan], iki setin eşit ağırlıklı "
                "ortalaması (3 tohum)}} \\\\")
    rows.append("Sinyal & Olasılık & $f{=}0{,}02$ & $0{,}04$ & $0{,}06$ & $0{,}10$ \\\\")
    rows.append("\\cmidrule(lr){1-6}")
    for sig, slab in SIG:
        vals_all = [objd[f"{sig}@{fr}:{pr}"] for pr in PROBS for fr in FRACS]
        worst = max(vals_all)
        for pr in PROBS:
            cells = []
            for fr in FRACS:
                s = f"{sig}@{fr}:{pr}"
                t = trs(objd[s])
                if s == choice[sig][0]:
                    t = f"\\iyi{{{t}}}"
                elif objd[s] == worst:
                    t = f"\\kotu{{{t}}}"
                cells.append(t)
            rows.append(f"{slab if pr == '0.5' else ''} & {tr(float(pr), 2)} & " + " & ".join(cells) + " \\\\")
    body = (
        "\\begin{table}[!htbp]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{Geliştirme koşuları (keşifsel; holdout2'den önce, seçim amaçlı). Üst: eşlenik farklar, "
        "5 tohum, 10~sn. Alt: havuz kuralı ızgarası; renk: seçilen / en kötü hücre.}\n"
        "\\label{tab:v3-development}\n"
        "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}Xlcccc@{}}\n\\toprule\n"
        "Karşılaştırma (A$-$B) & Set (n) & $\\Delta$ [puan] & \\%95 GA & Ö/E/G & $p$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n")
    write_tab("v3_development.tex", body)
    # figur verisi: her sinyal icin frac x prob
    out = []
    for sig, _ in SIG:
        for fr in FRACS:
            out.append([sig.split("_")[0], fr] + [f5(objd[f"{sig}@{fr}:{pr}"]) for pr in PROBS] +
                       [int(choice[sig][0] == f"{sig}@{fr}:{pr}") for pr in PROBS])
    write_dat("v3_pool_grid.dat", ["signal", "frac", "p050", "p075", "p090", "sel050", "sel075", "sel090"], out)
    for sig, _ in SIG:
        sub = [r for r in out if r[0] == sig.split("_")[0]]
        write_dat(f"v3_pool_grid_{sig.split('_')[0]}.dat",
                  ["frac", "p050", "p075", "p090"], [[r[1], r[2], r[3], r[4]] for r in sub])
        ch = choice[sig][0].split("@")[1].split(":")
        write_dat(f"v3_pool_grid_{sig.split('_')[0]}_sel.dat", ["frac", "d"], [[ch[0], f5(objd[choice[sig][0]])]])
    print("izgara secimleri:", {k: (v[0], v[1], round(v[3], 3)) for k, v in choice.items()})
    return rows


# ----------------------------------------------------------------------------- HERO
def hero(D, data, cat, keys):
    grp = {"vlsi6": "real", "national3": "real", "tsplib3": "real", "synth_uniform": "uniform",
           "synth_clustered": "clustered", "synth_mixed": "mixed"}
    csvp = DAT / "manset_birim.csv"
    lines = ["birim,yontem,deger,grup"]
    for k in keys:
        for s, lab in ((EXC, "Excess"), (GNN, "GNN"), (R, "Random")):
            lines.append(f"{k},{lab},{data[k][s]:.6f},{grp[cat[k]['stratum']]}")
    csvp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("veri :", csvp.name)
    vs = SKILL_SCRIPTS / "vitrin_sec.py"
    if vs.exists():
        cmd = [sys.executable, str(vs), "--csv", str(csvp), "--biz", "Excess", "--rakip", "GNN", "--yon", "dusuk",
               "--metrik", "gap", "--birim", "%", "--birim-adi", "problem", "--veri-seti", "holdout2",
               "--n", "3", "--ws", str(WS), "--evidence"]
        rc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        print(rc.stdout.strip())
        print("vitrin_sec rc =", rc.returncode)
    vids = D.vitrin_birlestir(WS / "evidence" / "vitrin.json")
    vj = json.loads((WS / "evidence" / "vitrin.json").read_text(encoding="utf-8"))
    picks = []
    for r in vj["kayitlar"]:
        v = r.get("vitrin")
        if v and v["kategori"] == "A" and r["yontem"] == "Excess":
            picks.append(v["birim"])
    # siralanmis fark verisi (iki panel)
    info = {}
    for b, tagb in ((GNN, "gnn"), (R, "random")):
        d = sorted(((data[k][EXC] - data[k][b], k) for k in keys))
        rows = []
        for i, (x, k) in enumerate(d, 1):
            rows.append([i, f5(x), f5(x) if x < 0 else "nan", f5(x) if x >= 0 else "nan", k, cat[k]["n"],
                         grp[cat[k]["stratum"]], f5(data[k][EXC]), f5(data[k][b])])
        write_dat(f"manset_diff_{tagb}.dat", ["rank", "d", "win", "loss", "instance", "n", "group", "excess",
                                               "other"], rows)
        pins = []
        if tagb == "gnn":
            lab_rows = []
            for i, (x, k) in enumerate(d, 1):
                if k in picks:
                    lab = (f"{k.split('_')[0]} ({grp[cat[k]['stratum']]}): "
                           f"Excess {data[k][EXC]:.2f}\\%, GNN {data[k][b]:.2f}\\%")
                    lab_rows.append([i, f5(x), lab])
                    pins.append((i, x, lab))
            write_dat("manset_vitrin.csv", ["rank", "d", "label"], lab_rows, sep=",")
        r = paired_keys(data, keys, EXC, b)
        rid = D.ekle(f"v3.hero.winrate.{tagb}", "kazanma_orani", f"{EXC} vs {b}", "holdout2",
                     r["wins_first"] / r["n_inst"], "oran", SRC_H2, "problem bazinda ortalama gap farki < 0",
                     wins=r["wins_first"], ties=r["ties"], losses=r["wins_second"], n_inst=r["n_inst"],
                     median_fark=r["median"], not_="manset; kesifsel ustunluk (V3/V4 yalniz kotu-degil)")
        worst = None
        if tagb == "gnn":   # en belirgin karsi-ornek (vitrin_sec kategori C'nin basi) gizlenmez
            x, k = d[-1]
            worst = (len(d), x, f"counter-example {k.split('_')[0]} ({grp[cat[k]['stratum']]}): "
                                f"Excess {data[k][EXC]:.2f}\\%, GNN {data[k][b]:.2f}\\%")
        info[tagb] = dict(w=r["wins_first"], l=r["wins_second"], n=r["n_inst"], lo=min(x for x, _ in d), hi=max(x for x, _ in d),
                          pins=pins, rid=rid, worst=worst)
    write_manset_tex(info, vj["vitrin_meta"])
    print("vitrin A secimleri:", picks, "| V kayitlari:", vids)


# ----------------------------------------------------------------------------- figur kaynaklari (uretilen)
def write_fig(name, body):
    (WS / "figures" / name).write_text(body, encoding="utf-8")
    print("figur:", name)


def write_manset_tex(info, vmeta):
    """Manset: problem bazinda Excess-GNN (a) ve Excess-Random (b) farki, sirali cubuk; A-vitrin pinleri."""
    P = []
    for j, (tagb, comp, panel) in enumerate((("gnn", "GNN", "a"), ("random", "Random", "b"))):
        I = info[tagb]
        ymin = math.floor((I["lo"] - 0.12) * 10) / 10
        ymax = math.ceil((I["hi"] + 0.12) * 10) / 10
        pins = ""
        # vitrin etiketleri: sag-alt bos bolgede (x=21), en derin cubuk en altta; ok noktaya gider
        span = I["hi"] - I["lo"]
        for q, (i, x, lab) in enumerate(sorted(I["pins"], key=lambda t: t[1])):
            ylev = I["lo"] + 0.06 * span + q * 0.16 * span
            pins += (f"\\node[circle, fill=black, inner sep=1.1pt] (v{q}) at (axis cs:{i},{x:.5f}) {{}};\n"
                     f"\\node[anchor=west, font=\\scriptsize, fill=white, inner sep=1pt, draw=black!30] (l{q}) "
                     f"at (axis cs:19,{ylev:.4f}) {{{lab}}};\n"
                     f"\\draw[black!70, thin, -{{Latex[length=1.5mm]}}] (l{q}.west) -- (v{q});\n")
        if I.get("worst"):
            i, x, lab = I["worst"]
            pins += (f"\\node[anchor=east, font=\\scriptsize, fill=white, inner sep=1pt, draw=black!30, text=kotuRenk] "
                     f"(c0) at (axis cs:51.5,{0.45 * x:.4f}) {{{lab}}};\n"
                     f"\\draw[black!70, thin, -{{Latex[length=1.5mm]}}] (c0.east) -- (axis cs:{i - 0.4},{0.45 * x:.4f});\n")
        P.append(
            f"\\nextgroupplot[ymin={ymin}, ymax={ymax},\n"
            f"  ylabel={{$\\Delta$ gap, Excess $-$ {comp} [pt]}}, title={{({panel}) Excess vs.\\ {comp}}}]\n"
            f"\\addplot[ybar, bar width=0.72, fill=iyiRenk, draw=none] table[x=rank, y=win, "
            f"unbounded coords=jump] {{figures/data/manset_diff_{tagb}.dat}};\n"
            f"\\addplot[ybar, bar width=0.72, fill=kotuRenk, draw=none] table[x=rank, y=loss, unbounded coords=jump] "
            f"{{figures/data/manset_diff_{tagb}.dat}};\n"
            f"\\draw[black!60] (axis cs:0,0) -- (axis cs:55,0);\n"
            f"\\node[anchor=north west, font=\\small, fill=white, inner sep=2pt, align=left] at "
            f"(axis description cs:0.01,0.97) {{\\textcolor{{iyiRenk}}{{\\textbf{{Excess lower gap: {I['w']}/{I['n']}}}}}"
            f"\\quad \\textcolor{{kotuRenk}}{{\\textit{{{comp} lower gap: {I['l']}/{I['n']}}}}}}};\n"
            + pins)
    body = (
        "% MANSET (hero) -- OTOMATIK URETIM: evidence/scripts/build_v3.py (figures/scripts/manset.py)\n"
        "% Veri: figures/data/manset_diff_gnn.dat, manset_diff_random.dat, manset_vitrin.csv (vitrin_sec.py, kategori A)\n"
        f"% Kanit: {info['gnn']['rid']} (vs GNN), {info['random']['rid']} (vs Random); vitrin: evidence/vitrin.json\n"
        f"% vitrin_sec kazanma orani (gurultu |d|>{vmeta['gurultu']:.4f} pt): {100 * vmeta['kazanma_orani']:.1f}%\n"
        "\\begin{tikzpicture}\n"
        "\\begin{groupplot}[group style={group size=1 by 2, vertical sep=1.25cm},\n"
        "  width=\\textwidth, height=5.4cm, xmin=0, xmax=55, xtick=\\empty,\n"
        "  xlabel={Holdout2 instances, sorted by $\\Delta$ (54)}, title style={font=\\small, yshift=-1ex},\n"
        "  tick label style={font=\\scriptsize}, label style={font=\\small},\n"
        "  y tick label style={/pgf/number format/fixed, /pgf/number format/precision=1},\n"
        "  ymajorgrids, grid style={black!10}]\n"
        + "".join(P) +
        "\\end{groupplot}\n\\end{tikzpicture}\n")
    write_fig("manset.tex", body)


def write_drift_tex():
    body = r"""% Cevrimici ogrenme kaymasi -- OTOMATIK URETIM: evidence/scripts/build_v3.py
% Veri: figures/data/v3_online_drift.dat (holdout2_10s_mechanism.json drift[*] + ham kosumlardan kabul orani)
\begin{tikzpicture}
\begin{groupplot}[group style={group size=1 by 2, vertical sep=1.9cm},
  width=\textwidth, height=5.0cm, xmode=log, xmin=1450, xmax=10500,
  xtick={1500,2000,3000,4000,6000,8000,10000}, xticklabels={1500,2000,3000,4000,6000,8000,10000},
  xlabel={Instance size $n$ (log scale)}, tick label style={font=\scriptsize}, label style={font=\small},
  title style={font=\small, yshift=-1ex},
  ymajorgrids, grid style={black!10},
  legend style={font=\scriptsize, draw=black!40, fill=white, legend columns=2}]
\nextgroupplot[title={(a) Which edges does the kick pool target?}, ylabel={Relative edge length},
  ymin=0, ymax=3.9, legend style={at={(0.99,0.97)}, anchor=north east}]
\addplot[only marks, mark=square*, mark size=1.4pt, black!75] table[x=n, y=rel_prior] {figures/data/v3_online_drift.dat};
\addlegendentry{Prior weights $w_0$ (static)}
\addplot[only marks, mark=*, mark size=1.4pt, iyiRenk] table[x=n, y=rel_excess] {figures/data/v3_online_drift.dat};
\addlegendentry{Excess proxy}
\addplot[only marks, mark=triangle*, mark size=1.7pt, kotuRenk] table[x=n, y=rel_learned] {figures/data/v3_online_drift.dat};
\addlegendentry{Learned weights $w_{\mathrm{final}}$}
\addplot[only marks, mark=x, mark size=1.4pt, black!40] table[x=n, y=rel_all] {figures/data/v3_online_drift.dat};
\addlegendentry{All tour edges}
\nextgroupplot[title={(b) Kick acceptance rate}, ylabel={Accepted kicks [\%] (log)}, ymode=log, ymin=0.1, ymax=40,
  log ticks with fixed point, legend style={at={(0.01,0.97)}, anchor=north west}]
\addplot[only marks, mark=square*, mark size=1.4pt, black!75] table[x=n, y=acc_linprior] {figures/data/v3_online_drift.dat};
\addlegendentry{Static-linear}
\addplot[only marks, mark=*, mark size=1.4pt, iyiRenk] table[x=n, y=acc_excess] {figures/data/v3_online_drift.dat};
\addlegendentry{Excess}
\addplot[only marks, mark=triangle*, mark size=1.7pt, kotuRenk] table[x=n, y=acc_online] {figures/data/v3_online_drift.dat};
\addlegendentry{Online (warm)}
\addplot[only marks, mark=x, mark size=1.4pt, black!40] table[x=n, y=acc_random] {figures/data/v3_online_drift.dat};
\addlegendentry{Random}
\end{groupplot}
\end{tikzpicture}
"""
    write_fig("fig_v3_online_drift.tex", body)


def write_grid_tex():
    panels = []
    for j, (sig, lab) in enumerate((("knn", "kNN"), ("excess", "Excess"), ("gnn", "GNN"))):
        yl = "ylabel={$\\Delta$ gap vs.\\ Random [pt]}, " if j == 0 else "yticklabels={}, "
        leg = ("legend to name=gridleg, legend columns=4, legend style={font=\\scriptsize, draw=black!40}, "
               if j == 0 else "")
        panels.append(
            f"\\nextgroupplot[title={{{lab}}}, {yl}{leg}]\n"
            f"\\addplot[black!45, dashed, forget plot] coordinates {{(0.01,0) (0.11,0)}};\n"
            f"\\addplot[thick, iyiRenk, mark=*, mark size=1.4pt] table[x=frac, y=p050] {{figures/data/v3_pool_grid_{sig}.dat}};\n"
            + ("\\addlegendentry{$p=0.5$}\n" if j == 0 else "")
            + f"\\addplot[thick, orange!85!black, dashed, mark=triangle*, mark size=1.6pt] table[x=frac, y=p075] "
              f"{{figures/data/v3_pool_grid_{sig}.dat}};\n"
            + ("\\addlegendentry{$p=0.75$}\n" if j == 0 else "")
            + f"\\addplot[thick, kotuRenk, densely dotted, mark=square*, mark size=1.3pt] table[x=frac, y=p090] "
              f"{{figures/data/v3_pool_grid_{sig}.dat}};\n"
            + ("\\addlegendentry{$p=0.9$}\n" if j == 0 else "")
            + f"\\addplot[only marks, mark=star, mark size=4pt, very thick, black] table[x=frac, y=d] "
              f"{{figures/data/v3_pool_grid_{sig}_sel.dat}};\n"
            + ("\\addlegendentry{selected rule}\n" if j == 0 else "")
            + "\\node[anchor=south east, font=\\scriptsize, fill=white, inner sep=1pt, text=black!60] at "
              "(axis cs:0.105,0) {Random};\n")
    body = (
        "% Havuz kurali izgarasi -- OTOMATIK URETIM: evidence/scripts/build_v3.py\n"
        "% Veri: figures/data/v3_pool_grid_{knn,excess,gnn}[_sel].dat (dev_grid_vlsi_primary_10s + dev_grid_synth1000_10s)\n"
        "\\begin{tikzpicture}\n"
        "\\begin{groupplot}[group style={group size=3 by 1, horizontal sep=0.35cm},\n"
        "  width=0.385\\textwidth, height=5.0cm, xmin=0.01, xmax=0.11, ymin=-0.17, ymax=0.11,\n"
        "  xtick={0.02,0.04,0.06,0.10}, xticklabels={0.02,0.04,0.06,0.10},\n"
        "  xlabel={Pool fraction $f$}, tick label style={font=\\scriptsize}, label style={font=\\small},\n"
        "  title style={font=\\small, yshift=-1ex},\n"
        "  y tick label style={/pgf/number format/fixed, /pgf/number format/precision=2},\n"
        "  ymajorgrids, grid style={black!10}]\n"
        + "".join(panels) +
        "\\end{groupplot}\n"
        "\\node[anchor=north] at ([yshift=-0.9cm]group c2r1.south) {\\pgfplotslegendfromname{gridleg}};\n"
        "\\end{tikzpicture}\n")
    write_fig("fig_v3_pool_grid.tex", body)


if __name__ == "__main__":
    main()
