# -*- coding: utf-8 -*-
"""Kesifsel saglamlik blogu (PROTOCOL_v3_exploratory.json) -> kanit kayitlari, tablo, figur verisi.

Girdi : results/kick_targeting/v4_robustness_analysis.json (analysis_v4_robustness.py)
Cikti : evidence/evidence.json  (ek.uretici = build_v4, ek.anahtar ile idempotent: var olan anahtarin id'si korunur)
        evidence/tables/v4_robustness.tex, figures/data/v4_structure.dat, figures/fig_v4_structure.tex
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v3 import tr, trs, trp, ci  # noqa: E402

W = Path(__file__).resolve().parents[2]
ROOT = W.parents[1]
SRC = "results/kick_targeting/v4_robustness_analysis.json"
A = json.loads((ROOT / SRC).read_text(encoding="utf-8"))
EVP = W / "evidence" / "evidence.json"
ev = json.loads(EVP.read_text(encoding="utf-8"))


def _mine(r):
    return isinstance(r.get("ek"), dict) and r["ek"].get("uretici") == "build_v4"


old = {r["ek"]["anahtar"]: r for r in ev["kayitlar"] if _mine(r)}
keep = [r for r in ev["kayitlar"] if not _mine(r)]
nid = max(int(r["id"][1:]) for r in ev["kayitlar"] if r["id"].startswith("E")) + 1
new, IDS = [], {}

R, E, G = "random_global", "excess_global@0.06:0.5", "gnn_global@0.04:0.5"
RL, EL, LL = "random_local", "excess_local@0.06:0.5", "length_local@0.06:0.5"


def add(anahtar, metrik, yontem, veri, deger, birim, konum, **ek):
    global nid
    if anahtar in old:
        rid = old[anahtar]["id"]
    else:
        rid, nid = f"E{nid}", nid + 1
    new.append({"id": rid, "metrik": metrik, "yontem": yontem, "veri_seti": veri, "deger": float(deger), "birim": birim,
                "kaynak": "yazilim", "kaynak_dosya": SRC, "kaynak_konum": konum, "atif": None, "durum": "gecerli",
                "ek": {"uretici": "build_v4", "anahtar": anahtar, "asama": 3,
                       "not_": "kesifsel, protokol sonrasi (PROTOCOL_v3_exploratory.json)", **ek}})
    IDS[anahtar] = rid
    return rid


def add_pair(key, veri, r, konum):
    add(key, "fark_ortalama", r["pair"], veri, r["mean"], "puan", konum,
        p_wilcoxon=r["p_wilcoxon"], p_holm=r.get("p_holm"), ci95=r["ci95"], hl=r["hl"], median=r["median"],
        wins_first=r["wins_first"], ties=r["ties"], wins_second=r["wins_second"], n_inst=r["n_inst"],
        by_stratum=r["by_stratum"])


rows_tab = []  # (grup, etiket, r)
V1 = "holdout2 (54 problem x 10 tohum, 10 sn)"
LAB1 = {(E, RL): "Fazla-uzunluk (global) $-$ rastgele yerel", (EL, RL): "Fazla-uzunluk (yerel) $-$ rastgele yerel",
        (LL, RL): "Ham uzunluk (yerel) $-$ rastgele yerel", (EL, E): "Fazla-uzunluk: yerel $-$ global",
        (RL, R): "Rastgele: yerel $-$ global"}
s1 = A["S1_local"]
if s1.get("contrasts"):
    for i, r in enumerate(s1["contrasts"]):
        a, b = r["pair"].split(" - ")
        add_pair(f"v4.S1.{a}-{b}", V1, r, f"S1_local['contrasts'][{i}]")
        rows_tab.append(("S1", LAB1[(a, b)], r))
    for s, m in s1["mean_gap"].items():
        add(f"v4.S1.gap.{s}", "ortalama_gap", s, V1, m, "%", f"S1_local['mean_gap']['{s}']")
    for s, c in s1["counters"].items():
        add(f"v4.S1.kps.{s}", "vurus_hizi", s, V1, c["kicks_per_s"], "vurus/sn", f"S1_local['counters']['{s}']",
            accept_rate=c["accept_rate"])

V2 = "holdout2 (54 problem x 5 tohum, 60 sn)"
s2 = A["S2_60s"]
for cp in ["10.0", "20.0", "30.0", "60.0"]:
    f = s2.get(cp, {})
    if not f.get("contrasts"):
        continue
    for i, r in enumerate(f["contrasts"]):
        a, b = r["pair"].split(" - ")
        add_pair(f"v4.S2.{cp}.{a}-{b}", V2, r, f"S2_60s['{cp}']['contrasts'][{i}]")
        if cp in ("10.0", "30.0", "60.0"):
            lab = ("Fazla-uzunluk $-$ rastgele" if b == R else "Fazla-uzunluk $-$ GNN") + f", {cp[:-2]}~sn"
            rows_tab.append(("S2", lab, r))
    for s, m in f["mean_gap"].items():
        add(f"v4.S2.{cp}.gap.{s}", "ortalama_gap", s, V2, m, "%", f"S2_60s['{cp}']['mean_gap']['{s}']")
if "noninferiority_excess_vs_gnn_margin0.05_tost_p" in s2:
    add("v4.S2.tost", "kotu_degil_tost_p", f"{E} vs {G}", V2, s2["noninferiority_excess_vs_gnn_margin0.05_tost_p"],
        "p", "S2_60s['noninferiority_excess_vs_gnn_margin0.05_tost_p']")
    for s, c in s2["counters"].items():
        add(f"v4.S2.kps.{s}", "vurus_hizi", s, V2, c["kicks_per_s"], "vurus/sn", f"S2_60s['counters']['{s}']",
            accept_rate=c["accept_rate"])

s3 = A["S3_rule"]
if s3.get("contrasts"):
    for i, r in enumerate(s3["contrasts"]):
        a, b = r["pair"].split(" - ")
        add_pair(f"v4.S3.{a}-{b}", V1, r, f"S3_rule['contrasts'][{i}]")
        f_, p_ = a.split("@")[1].split(":")
        lab = f"$f={tr(float(f_), 2)}$, $p={tr(float(p_), 2)}$ $-$ " + ("rastgele" if b == R else "seçilen kural")
        rows_tab.append(("S3", lab, r))
    for s, m in s3["mean_gap"].items():
        add(f"v4.S3.gap.{s}", "ortalama_gap", s, V1, m, "%", f"S3_rule['mean_gap']['{s}']")

VT = "holdout2_10s (PROTOCOL_v2 dogrulayici kosu, iz noktalari)"
for cp, d in A["T_anytime"].items():
    for s, r in d.items():
        add(f"v4.T.{cp}.{s}", "anytime_fark_rastgele", f"{s} - {R}", VT, r["mean"], "puan", f"T_anytime['{cp}']['{s}']",
            p_wilcoxon=r["p_wilcoxon"], ci95=r["ci95"], wins_first=r["wins_first"], checkpoint_s=float(cp))
ST = A["T_structure"]
for k in ("spearman_cv_vs_excess_minus_random", "spearman_cv_vs_excess_minus_gnn", "spearman_logn_vs_excess_minus_random"):
    add(f"v4.U.{k}", k, "Spearman rho", "holdout2_10s + koordinatlar", ST[k]["rho"], "rho", f"T_structure['{k}']",
        p=ST[k]["p"])
for s in ["synth_uniform", "synth_mixed", "synth_clustered", "vlsi6", "national3", "tsplib3"]:
    x = [r["cv_nn"] for r in ST["rows"] if r["stratum"] == s]
    add(f"v4.U.cv_mean.{s}", "cv_nn_katman_ortalama", s, "holdout2", sum(x) / len(x), "oran",
        f"T_structure['rows'][stratum=={s}]", min=min(x), max=max(x), n=len(x))

ev["kayitlar"] = keep + sorted(new, key=lambda r: int(r["id"][1:]))
EVP.write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
(W / "evidence" / "v4_ids.json").write_text(json.dumps(IDS, indent=1), encoding="utf-8")
print("build_v4:", len(new), "kayit")

# ---- tablo
GRP = {"S1": "(a) Klasik yerel (segment) vuruş --- 10~sn, 10 tohum",
       "S2": "(b) Uzun bütçe --- 60~sn, 5 tohum (aynı koşuların ara noktaları)",
       "S3": "(c) Havuz kuralı holdout2'de --- 10~sn, 10 tohum"}


def cell(r):
    sig = r["p_holm"] < 0.05 and (r["ci95"][1] < 0 or r["ci95"][0] > 0)
    t = trs(r["mean"])
    return (f"\\iyi{{{t}}}" if r["mean"] < 0 else f"\\kotu{{{t}}}") if sig else t


lines = []
for g in ("S1", "S2", "S3"):
    rr = [x for x in rows_tab if x[0] == g]
    if not rr:
        continue
    if lines:
        lines.append("\\midrule")
    lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{GRP[g]}}}}} \\\\")
    for _, lab, r in rr:
        lines.append(f"{lab} & {cell(r)} & {ci(r['ci95'])} & {r['wins_first']}/{r['n_inst']} & {trp(r['p_holm'])} \\\\")
HEAD = (r"\begin{table}[!htbp]" "\n" r"\centering\small\setlength{\tabcolsep}{3pt}" "\n"
        r"\caption{Keşifsel sağlamlık blokları (holdout2, protokol sonrası; PROTOCOL\_v3\_exploratory koşulardan önce "
        r"kaydedildi). Fark: problem başı ortalama gap farkı [puan], negatif değer ilk terim lehinedir. Önde: ilk terimin "
        r"daha düşük gap verdiği problem sayısı. $p_{\mathrm{Holm}}$: blok (b'de her zaman noktası) içinde Holm düzeltmeli "
        r"Wilcoxon; renk $p_{\mathrm{Holm}}<0{,}05$ ve GA'nın sıfırı dışladığını gösterir.}" "\n"
        r"\label{tab:v4-robustness}" "\n"
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc}" "\n" r"\toprule" "\n"
        r"Karşılaştırma & Fark & \%95 GA & Önde & $p_{\mathrm{Holm}}$ \\" "\n" r"\midrule" "\n")
TAIL = "\n" r"\bottomrule" "\n" r"\end{tabular*}" "\n" r"\end{table}" "\n"
if rows_tab:
    (W / "evidence" / "tables" / "v4_robustness.tex").write_text(
        "% OTOMATIK URETIM -- evidence/scripts/build_v4.py (elle duzenleme)\n" + HEAD + "\n".join(lines) + TAIL,
        encoding="utf-8")
    print("tablo: v4_robustness.tex")

# ---- yapi figuru
GRUP = {"synth_uniform": "uni", "synth_mixed": "mix", "synth_clustered": "clu", "vlsi6": "real", "national3": "real",
        "tsplib3": "real"}
dat = ["cv d_rand d_gnn grp n"] + [f"{r['cv_nn']:.5f} {r['excess_minus_random']:.5f} {r['excess_minus_gnn']:.5f} "
                                   f"{GRUP[r['stratum']]} {r['n']}" for r in ST["rows"]]
(W / "figures" / "data" / "v4_structure.dat").write_text("\n".join(dat) + "\n", encoding="utf-8")
rho, p = ST["spearman_cv_vs_excess_minus_random"]["rho"], ST["spearman_cv_vs_excess_minus_random"]["p"]
mant, ex = f"{p:.1e}".split("e")
ptxt = f"{mant}\\times10^{{{int(ex)}}}"
PLOTS = [("uni", "mark=x, mark size=1.8pt, black!45", "Uniform"),
         ("mix", "mark=square*, mark size=1.3pt, black!70", "Mixed"),
         ("clu", "mark=*, mark size=1.5pt, iyiRenk", "Clustered"),
         ("real", "mark=triangle*, mark size=1.9pt, kotuRenk", "Real")]
fig = [r"% Problem yapisi ve hedefleme kazanci -- OTOMATIK URETIM: evidence/scripts/build_v4.py",
       r"% Veri: figures/data/v4_structure.dat (v4_robustness_analysis.json T_structure.rows)",
       r"\begin{tikzpicture}",
       r"\begin{axis}[width=\textwidth, height=5.6cm, xmode=log, log basis x=10, xmin=0.25, xmax=2.5, ymax=0.9,",
       r"  xtick={0.3,0.5,0.75,1,1.5,2}, xticklabels={0.3,0.5,0.75,1,1.5,2},",
       r"  xlabel={Clustering index: CV of nearest-neighbour distance (log scale)},",
       r"  ylabel={Excess $-$ random [gap points]}, tick label style={font=\scriptsize}, label style={font=\small},",
       r"  ymajorgrids, grid style={black!10},",
       r"  legend style={font=\scriptsize, draw=black!40, fill=white, at={(0.01,0.03)}, anchor=south west, legend columns=4}]",
       r"\addplot[black!50, thin, forget plot] coordinates {(0.25,0) (2.5,0)};"]
for g, style, lab in PLOTS:
    fig.append(rf"\addplot[only marks, {style}] table[x=cv, y=d_rand, restrict expr to domain={{\thisrow{{g{g}}}}}{{1:1}}] "
               r"{figures/data/v4_structure.dat};")
    fig.append(rf"\addlegendentry{{{lab}}}")
fig += [rf"\node[anchor=north west, font=\scriptsize, fill=white, inner sep=2pt] at (rel axis cs:0.01,0.97) "
        rf"{{Spearman $\rho={rho:.2f}$, $p={ptxt}$}};",
        r"\end{axis}", r"\end{tikzpicture}", ""]
# pgfplots filtresi icin grup bayrak sutunlari (guni, gmix, gclu, greal)
dat = ["cv d_rand d_gnn grp n guni gmix gclu greal"] + [
    f"{r['cv_nn']:.5f} {r['excess_minus_random']:.5f} {r['excess_minus_gnn']:.5f} {GRUP[r['stratum']]} {r['n']} "
    + " ".join("1" if GRUP[r["stratum"]] == g else "0" for g in ("uni", "mix", "clu", "real")) for r in ST["rows"]]
(W / "figures" / "data" / "v4_structure.dat").write_text("\n".join(dat) + "\n", encoding="utf-8")
(W / "figures" / "fig_v4_structure.tex").write_text("\n".join(fig), encoding="utf-8")
print("figur: fig_v4_structure.tex")
