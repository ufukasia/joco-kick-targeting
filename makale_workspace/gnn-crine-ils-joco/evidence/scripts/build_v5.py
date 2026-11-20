# -*- coding: utf-8 -*-
"""Kesifsel hakem blogu B1-B6 (PROTOCOL_v4_exploratory.json) -> kanit kayitlari ve tablo.

Girdi : results/kick_targeting/v5_analysis.json (analysis_v5.py, bir kez)
Cikti : evidence/evidence.json  (ek.uretici = build_v5, ek.anahtar ile idempotent)
        evidence/v5_ids.json, evidence/tables/v5_reviewer.tex
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v3 import tr, trs, trp, ci  # noqa: E402

W = Path(__file__).resolve().parents[2]
ROOT = W.parents[1]
SRC = "results/kick_targeting/v5_analysis.json"
A = json.loads(Path(os.environ.get("V5_ANALYSIS", ROOT / SRC)).read_text(encoding="utf-8"))
OUT = Path(os.environ.get("V5_OUT", W / "evidence"))  # test: geçici klasör
EVP = W / "evidence" / "evidence.json"
ev = json.loads(EVP.read_text(encoding="utf-8"))
SEL = A["selected_rules"]


def _mine(r):
    return isinstance(r.get("ek"), dict) and r["ek"].get("uretici") == "build_v5"


old = {r["ek"]["anahtar"]: r for r in ev["kayitlar"] if _mine(r)}
keep = [r for r in ev["kayitlar"] if not _mine(r)]
nid = max(int(r["id"][1:]) for r in ev["kayitlar"] if r["id"].startswith("E")) + 1
new, IDS = [], {}


def add(anahtar, metrik, yontem, veri, deger, birim, konum, **ek):
    global nid
    if anahtar in old:
        rid = old[anahtar]["id"]
    else:
        rid, nid = f"E{nid}", nid + 1
    new.append({"id": rid, "metrik": metrik, "yontem": yontem, "veri_seti": veri, "deger": float(deger), "birim": birim,
                "kaynak": "yazilim", "kaynak_dosya": SRC, "kaynak_konum": konum, "atif": None, "durum": "gecerli",
                "ek": {"uretici": "build_v5", "anahtar": anahtar, "asama": 3,
                       "not_": "kesifsel, protokol sonrasi (PROTOCOL_v4_exploratory.json)", **ek}})
    IDS[anahtar] = rid
    return rid


def add_pair(key, veri, r, konum):
    add(key, "fark_ortalama", r["pair"], veri, r["mean"], "puan", konum,
        p_wilcoxon=r["p_wilcoxon"], p_holm=r.get("p_holm"), ci95=r["ci95"], hl=r["hl"], median=r["median"],
        wins_first=r["wins_first"], ties=r["ties"], wins_second=r["wins_second"], n_inst=r["n_inst"],
        by_stratum=r["by_stratum"])


gsel, nsel = SEL["gain"]["selected"], SEL["neurolkh"]["selected"]


def fq(rule):
    f_, p_ = rule.split(":")
    return f"({tr(float(f_), 2)}; {tr(float(p_), 2)})"


def fp(rule):
    f_, p_ = rule.split(":")
    return f"$f={tr(float(f_), 2)}$, $p={tr(float(p_), 2)}$"


NAME = {
    "random_global": "rastgele", "excess_global@0.06:0.5": "fazla-uzunluk", "gnn_global@0.04:0.5": "GNN",
    "knn_global@0.04:0.75": "kNN sırası", "gain_global": "kazanç-MLP (varsayılan)",
    f"gain_global@{gsel}": "kazanç-MLP (ayarlı)",
    f"neurolkh_global@{nsel}": "NeuroLKH (süre sayılır)", f"neurolkhfree_global@{nsel}": "NeuroLKH (süre sayılmaz)",
    "linprior_global": "doğrusal ön-bilgi", "onlinewarm_global": "çevrimiçi, işaretli ödül",
    "onlinepos_global": "çevrimiçi, $\\max(0,r)$", "onlinesucc_global": "çevrimiçi, $\\mathbb{I}[\\text{kabul}]$",
    "lkh_default": "LKH-3 varsayılan", "lkh_ils_random": "LKH-3 ILS, rastgele vuruş",
    "lkh_ils_excess": "LKH-3 ILS, hedefli vuruş",
}
if gsel == "0.06:0.75":
    NAME["gain_global"] = "kazanç-MLP"

for sig in ("gain", "neurolkh"):
    add(f"v5.sel.{sig}", "secilen_kural_ortalama_gap", f"{sig}@{SEL[sig]['selected']}", "vlsi_primary + synth_1000",
        SEL[sig]["grid_mean_gap"][SEL[sig]["selected"]], "%", f"selected_rules['{sig}']",
        selected=SEL[sig]["selected"], best=SEL[sig]["best"],
        default_gap=SEL[sig]["grid_mean_gap"]["0.06:0.75"])

FAM = [("S2_equal_time", "S2", "holdout2 (54 problem x 10 tohum, 10 sn, esit sure)"),
       ("S1_equal_kicks", "S1", "holdout2 (54 problem x 5 tohum, 20 000 vurus)"),
       ("S3_credit", "S3", "holdout2 (54 problem x 10 tohum, 10 sn)"),
       ("S4_single", "S4", "holdout2 (54 problem x 2 tohum, 10 sn, tek isci)"),
       ("S5_lkh", "S5", "holdout2 (54 problem x 5 tohum, LKH-3, 10 sn)")]
rows_tab = []
for key, short, veri in FAM:
    fam = A.get(key, {})
    for i, r in enumerate(fam.get("contrasts", [])):
        a, b = r["pair"].split(" - ")
        add_pair(f"v5.{short}.{a}-{b}", veri, r, f"{key}['contrasts'][{i}]")
        nm = dict(NAME, **({f"neurolkh_global@{nsel}": "NeuroLKH"} if short == "S1" else {}))  # S1: model süresi hiç sayılmaz
        lab = f"{nm.get(a, a)} $-$ {nm.get(b, b)}"
        rows_tab.append((short, lab[0].upper() + lab[1:], r))
    for s, c in A.get(f"{short}_counters", {}).items():
        add(f"v5.{short}.gap.{s}", "ortalama_gap", s, veri, c["gap_mean"], "%", f"{short}_counters['{s}']['gap_mean']")
        if "kicks_per_s" in c:
            add(f"v5.{short}.kps.{s}", "vurus_hizi", s, veri, c["kicks_per_s"], "vurus/sn",
                f"{short}_counters['{s}']['kicks_per_s']", accept_rate=c["accept_rate"],
                model_share=c["model_share"], elapsed_mean=c["elapsed_mean"])
        if "trials_mean" in c:
            add(f"v5.{short}.trials.{s}", "deneme_sayisi", s, veri, c["trials_mean"], "deneme",
                f"{short}_counters['{s}']['trials_mean']")
oe = A.get("S2_order", {}).get("spearman_position_vs_relative_rate")
if oe:
    add("v5.S2.order_rho", "sira_etkisi_spearman", "konum vs goreli vurus hizi", FAM[0][2], oe["rho"], "rho",
        "S2_order['spearman_position_vs_relative_rate']", p=oe["p"],
        by_position=A["S2_order"]["relative_kick_rate_by_position"])

ev["kayitlar"] = keep + sorted(new, key=lambda r: int(r["id"][1:]))
(OUT / "evidence.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / "v5_ids.json").write_text(json.dumps(IDS, indent=1, ensure_ascii=False), encoding="utf-8")
print("build_v5:", len(new), "kayit")

# ---- tablo
GRP = {"S2": f"(a) Eşit süre --- 10~sn, 10 tohum; ayarlı $(f,p)$: kazanç-MLP {fq(gsel)}, NeuroLKH {fq(nsel)}",
       "S1": "(b) Eşit vuruş sayısı --- 20.000 vuruş, 5 tohum, model süresi sayılmaz",
       "S3": "(c) Çevrimiçi öğrenmede kredi tanımı --- 10~sn, 10 tohum",
       "S4": "(d) Tek işçi, rastgele kol sırası --- 10~sn, 2 tohum",
       "S5": "(e) LKH-3.0.13 içinde vuruş hedefleme --- 10~sn, 5 tohum"}


def cell(r):
    sig = r["p_holm"] < 0.05 and (r["ci95"][1] < 0 or r["ci95"][0] > 0)
    t = trs(r["mean"])
    return (f"\\iyi{{{t}}}" if r["mean"] < 0 else f"\\kotu{{{t}}}") if sig else t


lines = []
for g in ("S2", "S1", "S3", "S4", "S5"):
    rr = [x for x in rows_tab if x[0] == g]
    if not rr:
        continue
    if lines:
        lines.append("\\midrule")
    lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{GRP[g]}}}}} \\\\")
    for _, lab, r in rr:
        lines.append(f"{lab} & {cell(r)} & {ci(r['ci95'])} & {r['wins_first']}/{r['n_inst']} & {trp(r['p_holm'])} \\\\")
HEAD = (r"\begin{table}[!htbp]" "\n" r"\centering\small\setlength{\tabcolsep}{3pt}" "\n"
        r"\caption{Hakem odaklı keşifsel bloklar (holdout2; protokol koşulardan önce kaydedildi). Fark [puan]: negatif değer ilk terim lehine. Önde: ilk terimin önde olduğu problem sayısı. $p_{\mathrm{Holm}}$: blok içinde Holm düzeltmeli Wilcoxon. Renk: $p_{\mathrm{Holm}}<0{,}05$ ve GA sıfırı dışlar.}" "\n"
        r"\label{tab:v5-reviewer}" "\n"
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc}" "\n" r"\toprule" "\n"
        r"Karşılaştırma & Fark & \%95 GA & Önde & $p_{\mathrm{Holm}}$ \\" "\n" r"\midrule" "\n")
TAIL = "\n" r"\bottomrule" "\n" r"\end{tabular*}" "\n" r"\end{table}" "\n"
if rows_tab:
    (OUT / "tables" / "v5_reviewer.tex").write_text(
        "% OTOMATIK URETIM -- evidence/scripts/build_v5.py (elle duzenleme)\n" + HEAD + "\n".join(lines) + TAIL,
        encoding="utf-8")
    print("tablo: v5_reviewer.tex")
