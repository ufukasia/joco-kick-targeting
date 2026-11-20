# -*- coding: utf-8 -*-
"""v2 tablolari ve figur verileri (hizli ILS kampanyasi).

Kaynak: tsplib/results/kick_targeting/*_summary.json, mechanism_summary.json,
lkh_baseline/*.json, results/waterloo_vlsi_pooled_8seed.json
Cikti : evidence/tables/v2_*.tex, figures/data/v2_*.dat
    py -3.13 evidence/scripts/build_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
ROOT = WS.parents[1]
KT = ROOT / "results" / "kick_targeting"
TAB = WS / "evidence" / "tables"
DAT = WS / "figures" / "data"
TAB.mkdir(parents=True, exist_ok=True)
DAT.mkdir(parents=True, exist_ok=True)

LABEL = {"random_global": "Rastgele--global", "gnn_global": "GNN--global",
         "knn_global": "kNN--global", "random_local": "Rastgele--yerel",
         "gnn_local": "GNN--yerel", "knn_local": "kNN--yerel"}
ORDER = ["random_global", "gnn_global", "knn_global", "random_local", "gnn_local", "knn_local"]
HYP = {"H1": "GNN$-$Rastgele (global)", "H2": "GNN$-$Rastgele (yerel)",
       "H3": "GNN$-$kNN (global)", "H4": "GNN$-$kNN (yerel)",
       "S1_knn_global_vs_random_global": "kNN$-$Rastgele (global)",
       "S2_knn_local_vs_random_local": "kNN$-$Rastgele (yerel)",
       "S3_random_local_vs_random_global": "Yerel$-$global (rastgele)"}
SUITE_NAME = {"vlsi_primary_10s": "VLSI-38 (birincil)", "vlsi_old_10s": "VLSI-20 (önceki)",
              "tsplib72_10s": "TSPLIB-72", "synth_holdout_1s": "Sentetik-100 (1 sn)",
              "synth_1000_10s": "Sentetik-1000"}


def tr(x, d=3):
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$").replace(".", "{,}")


def trp(p):
    if p < 1e-4:
        return "$<10^{-4}$"
    return tr(p, 4)


def trs(x, d=3):
    x = round(x, d) + 0.0
    return ("+" if x > 0 else "") + tr(x, d)


def load(tag):
    f = KT / f"{tag}_summary.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def write(name, body):
    (TAB / name).write_text("% OTOMATIK URETIM -- evidence/scripts/build_v2.py\n" + body, encoding="utf-8")
    print("yazildi", name)


def tab_primary_strategies(o):
    rows = []
    for s in ORDER:
        op = o["operational"][s]
        stale = "---" if op["stale_cut_rate"] is None else tr(100 * op["stale_cut_rate"], 1)
        rows.append(f"{LABEL[s]} & {tr(o['mean_gap'][s])} & {tr(o['median_gap'][s])} & "
                    f"{int(round(op['kicks_per_s_mean'])):d} & {int(round(op['accepted_mean'])):d} & "
                    f"{tr(100 * op['model_time_frac_mean'], 1)} & {stale} \\\\")
    body = (
        "\\begin{table}[!htbp]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n"
        f"\\caption{{Birincil set ({o['n_instances']} VLSI problemi, 10 tohum, 10~sn): gap ve operasyonel sayaçlar. "
        "Model: skor süresi payı; Bayat: turda olmayan havuz kenarına düşen kesimler.}\n"
        "\\label{tab:v2-primary-strategies}\n"
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc}\n\\toprule\n"
        "Strateji & Ort. gap [\\%] & Medyan & Kick/sn & Kabul & Model [\\%] & Bayat [\\%] \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    write("v2_primary_strategies.tex", body)


def hyp_row(h, c, holm=True):
    lo, hi = c["bootstrap95"]
    hp = trp(c["holm_p"]) if holm and "holm_p" in c else "---"
    return (f"{HYP[h]} & {trs(c['mean_diff'])} & [{trs(lo)}; {trs(hi)}] & "
            f"{c['a_better']}/{c['ties']}/{c['b_better']} & {trp(c['wilcoxon_p'])} & {hp} & "
            f"{trp(c['tost_p_margin_0.10'])} \\\\")


def tab_primary_tests(o):
    rows = [hyp_row(h, c) for h, c in o["primary"].items()]
    rows.append("\\midrule")
    rows += [hyp_row(h, c, holm=False) for h, c in o["secondary"].items()]
    body = (
        "\\begin{table}[!htbp]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{Birincil sette ön-kayıtlı hipotezler (H1--H4, tek Holm ailesi) ve ikincil karşılaştırmalar. "
        "$\\Delta$: A$-$B gap farkı [puan], negatif A lehine; TOST sınırı $\\pm0{,}10$. Ö/E/G: A'nın önde, eşit ve geride olduğu problem sayısı.}\n"
        "\\label{tab:v2-primary-tests}\n"
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc}\n\\toprule\n"
        "Karşılaştırma (A$-$B) & $\\Delta$ & \\%95 GA & Ö/E/G & Wilcoxon $p$ & Holm-$p$ & TOST $p$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    write("v2_primary_tests.tex", body)


def tab_suites(tags):
    rows = []
    for tag in tags:
        o = load(tag)
        if o is None:
            continue
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{SUITE_NAME[tag]}: {o['n_instances']} problem; "
                    f"rastgele--global gap {tr(o['mean_gap']['random_global'])}\\%}}}} \\\\")
        for h in ("H1", "H2", "H3", "H4"):
            rows.append(hyp_row(h, o["primary"][h]))
        rows.append(hyp_row("S1_knn_global_vs_random_global", o["secondary"]["S1_knn_global_vs_random_global"], holm=False))
        rows.append("\\addlinespace")
    body = (
        "\\begin{table}[!htbp]\n\\centering\\small\\setlength{\\tabcolsep}{2.5pt}\n"
        "\\caption{İkincil setlerde aynı karşılaştırmalar. Holm düzeltmesi her set içinde H1--H4 ailesine uygulanmıştır; "
        "gap sentetik setlerde LKH-3 referansına, gerçek setlerde bilinen optimuma/BKS'ye göredir.}\n"
        "\\label{tab:v2-suites}\n"
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc}\n\\toprule\n"
        "Karşılaştırma (A$-$B) & $\\Delta$ & \\%95 GA & Ö/E/G & Wilcoxon $p$ & Holm-$p$ & TOST $p$ \\\\\n\\midrule\n"
        + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    write("v2_suites.tex", body)


def dat_anytime(o, name):
    cps = sorted(o["anytime"], key=float)
    lines = ["t " + " ".join(ORDER)]
    for cp in cps:
        g = o["anytime"][cp]["mean_gap"]
        lines.append(f"{cp} " + " ".join(f"{g[s]:.5f}" for s in ORDER))
    (DAT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    # differences at checkpoints with p
    lines = ["t d_h1 p_h1 d_h3 p_h3 d_s1 p_s1"]
    for cp in cps:
        H = o["anytime"][cp]["H"]
        lines.append(f"{cp} {H['H1']['mean_diff']:.5f} {H['H1']['wilcoxon_p']:.5f} "
                     f"{H['H3']['mean_diff']:.5f} {H['H3']['wilcoxon_p']:.5f} "
                     f"{H['S1_knn_global_vs_random_global']['mean_diff']:.5f} "
                     f"{H['S1_knn_global_vs_random_global']['wilcoxon_p']:.5f}")
    (DAT / name.replace(".dat", "_diff.dat")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def dat_per_instance(o, name):
    lines = ["n gnn_rand knn_rand gnn_knn"]
    for i, r in o["per_instance"].items():
        lines.append(f"{r['n']} {r['gnn_global'] - r['random_global']:.5f} "
                     f"{r['knn_global'] - r['random_global']:.5f} {r['gnn_global'] - r['knn_global']:.5f}")
    (DAT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def tab_mechanism():
    f = KT / "mechanism_summary.json"
    if not f.exists():
        return
    m = json.loads(f.read_text(encoding="utf-8"))
    S = m["summary"]
    rows = []
    for state, lab in (("start_local_opt", "Başlangıç turu"), ("ils_2s", "2 sn ILS turu")):
        s = S[state]
        rows.append(f"{lab} ({s['n_instances']}) & {tr(100 * s['base_rate_wrong'], 1)} & "
                    f"{tr(s['auc_gnn'])} & {tr(s['auc_knn'])} & {tr(s['auc_rellen'])} & {tr(s['auc_length'])} & "
                    f"{tr(100 * s['prec_gnn'], 1)} & {tr(100 * s['prec_knn'], 1)} & {tr(s['spearman_gnn_knn'], 2)} & "
                    f"{tr(100 * s['pool_overlap_gnn_knn'], 0)} \\\\")
    body = (
        "\\begin{table}[!htbp]\n\\centering\\small\\setlength{\\tabcolsep}{2.5pt}\n"
        "\\caption{LKH-3 referans turunda bulunmayan (``yanlış'') tur kenarlarını ayırt etme, VLSI ortalamaları. "
        "Taban: yanlış kenar oranı; Kesinlik: havuzdaki yanlış oranı; $\\rho$: GNN--kNN Spearman.}\n"
        "\\label{tab:v2-mechanism}\n"
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccccc}\n\\toprule\n"
        " & Taban & \\multicolumn{4}{c}{AUC} & \\multicolumn{2}{c}{Kesinlik [\\%]} & & Örtüşme \\\\\n"
        "\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}\n"
        "Tur durumu & [\\%] & GNN & kNN & Göreli uz. & Uzunluk & GNN & kNN & $\\rho$ & [\\%] \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    write("v2_mechanism.tex", body)
    lines = ["n auc_gnn auc_knn prec_gnn prec_knn base"]
    for r in m["rows"]:
        if r["state"] == "ils_2s":
            lines.append(f"{r['n']} {r['auc_gnn']:.4f} {r['auc_knn']:.4f} {r['prec_gnn']:.4f} "
                         f"{r['prec_knn']:.4f} {r['base_rate_wrong']:.4f}")
    (DAT / "v2_mechanism_ils.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tab_lkh(prim):
    d = KT / "lkh_baseline"
    if not d.exists():
        return
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in d.glob("*.json")]
    rows = []
    for cfg, lab in (("default", "LKH-3 varsayılan"), ("fast", "LKH-3 hızlı ön-işlem")):
        for tl in (1.0, 10.0):
            R = [r for r in recs if r["config"] == cfg and r["time_limit"] == tl]
            ok = [r for r in R if r.get("ok")]
            if not R:
                continue
            per = {}
            for r in ok:
                per.setdefault(r["instance"], []).append((r["gap_pct"], r["wall"]))
            gaps = [np.mean([g for g, _ in v]) for v in per.values()]
            walls = [np.mean([w for _, w in v]) for v in per.values()]
            rows.append(f"{lab} & {tl:g} & {len(per)} & {tr(np.mean(gaps))} & {tr(np.median(walls), 1)} & "
                        f"{tr(np.mean(walls), 1)} & {tr(np.max(walls), 1)} \\\\")
    if prim is not None:
        for s in ("random_global", "gnn_global", "knn_global"):
            for cp in ("1.0", "10.0"):
                g = prim["anytime"][cp]["mean_gap"][s]
                rows.append(f"ILS {LABEL[s]} & {float(cp):g} & {prim['n_instances']} & {tr(g)} & "
                            f"{tr(float(cp), 1)} & {tr(float(cp), 1)} & {tr(float(cp), 1)} \\\\")
    body = (
        "\\begin{table}[!htbp]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{Bağlam: birincil VLSI setinde LKH-3 (3 tohum) ve hızlı ILS. LKH-3 bütçesi \\texttt{TIME\\_LIMIT}; "
        "süre sütunları ön-işlem dahil gerçek süre [sn].}\n"
        "\\label{tab:v2-lkh}\n"
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc}\n\\toprule\n"
        "Yöntem & Bütçe [sn] & Problem & Ort. gap [\\%] & Medyan & Ort. & Maks. \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    write("v2_lkh.tex", body)


def tab_old_reanalysis():
    f = ROOT / "results" / "waterloo_vlsi_pooled_8seed.json"
    o = json.loads(f.read_text(encoding="utf-8"))
    rows = []
    for k, lab in (("five_seed", "5 tohum (önceki taslağın ana sonucu)"), ("three_seed", "3 tohum (ayrı koşu)"),
                   ("pooled_eight_seed", "8 tohum birleşik"), ("pooled8_n_le_436", "\\quad $n\\le436$ (8 problem)"),
                   ("pooled8_n_gt_436", "\\quad $n>436$ (8 problem)")):
        s = o[k]
        rows.append(f"{lab} & {tr(s['mean'])} & [{tr(s['bootstrap95'][0])}; {tr(s['bootstrap95'][1])}] & "
                    f"{s['wins']}/{s['losses']} & {trp(s['wilcoxon_p'])} \\\\")
    body = (
        "\\begin{table}[!htbp]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n"
        "\\caption{Önceki CRINE-ILS motorunda 16 Waterloo VLSI problemi (5~sn): suspicious$-$random farkı "
        "($-$gap puanı; pozitif suspicious lehine). Boyut ayrımı sonradan yapılmıştır (keşifsel).}\n"
        "\\label{tab:v2-old-reanalysis}\n"
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccc}\n\\toprule\n"
        "Koşu & Ort. fark & \\%95 GA & G/M & Wilcoxon $p$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    write("v2_old_reanalysis.tex", body)


def tab_sets():
    rows = []
    for tag in SUITE_NAME:
        cfg = KT / tag / "config.json"
        if not cfg.exists():
            continue
        c = json.loads(cfg.read_text(encoding="utf-8"))
        o = load(tag)
        ns = [r["n"] for r in o["per_instance"].values()] if o else []
        ref = "LKH-3 (30 sn)" if tag.startswith("synth") else "Optimum/BKS"   # make_synth_suite.py: TIME_LIMIT 30
        role = "Ön-kayıtlı birincil" if tag == "vlsi_primary_10s" else "İkincil"
        rows.append(f"{SUITE_NAME[tag]} & {len(c['instances'])} & {min(ns) if ns else '--'}--{max(ns) if ns else '--'} & "
                    f"{c['seeds']} & {c['budget']:g} & {ref} & {role} \\\\")
    body = (
        "\\begin{table}[!htbp]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{Hızlı ILS kampanyasının deney setleri. Her (problem, tohum) için altı strateji aynı başlangıç turundan "
        "ve aynı kick rastgele sayı tohumundan başlar.}\n\\label{tab:v2-sets}\n"
        "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}Xcccccl@{}}\n\\toprule\n"
        "Set & Problem & $n$ & Tohum & Bütçe [sn] & Referans & Rol \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n")
    write("v2_sets.tex", body)


def tab_exploratory():
    f = KT / "exploratory_summary.json"
    if not f.exists():
        return
    S = json.loads(f.read_text(encoding="utf-8"))

    def row(lab, c):
        lo, hi = c["bootstrap95"]
        return (f"{lab} & {trs(c['mean_diff'])} & [{trs(lo)}; {trs(hi)}] & "
                f"{c['a_better']}/{c['ties']}/{c['b_better']} & {trp(c['wilcoxon_p'])} \\\\")

    rows = []
    o = S["oracle_vs_primary_seeds012"]
    rows.append("\\multicolumn{5}{l}{\\textit{Kâhin hedefleme (LKH referansında olmayan kenarlar), 10 sn, 3 tohum}} \\\\")
    rows.append(row("Kâhin$-$kNN (aynı oturum)", S["oracle"]["pairs"]["oracle_global-knn_global"]))
    rows.append(row("Kâhin$-$GNN (birincil, tohum 0--2)", o["oracle-gnn_global"]))
    rows.append(row("Kâhin$-$Rastgele (birincil, tohum 0--2)", o["oracle-random_global"]))
    rows.append("\\addlinespace")
    rows.append("\\multicolumn{5}{l}{\\textit{Taze GNN (model süresi sınırı yok), 10 sn, 5 tohum}} \\\\")
    fr = S["fresh_10s"]["pairs"]
    rows.append(row("GNN$-$Rastgele", fr["gnn_global-random_global"]))
    rows.append(row("GNN$-$kNN", fr["gnn_global-knn_global"]))
    rows.append(row("kNN$-$Rastgele", fr["knn_global-random_global"]))
    rows.append(row("Taze GNN$-$sınırlı GNN (birincil, tohum 0--4)", S["fresh_vs_capped_gnn"]))
    cf = KT / "cheap_summary.json"
    if cf.exists():
        C = json.loads(cf.read_text(encoding="utf-8")).get("primary_cheap_10s")
        if C:
            rows.append("\\addlinespace")
            rows.append("\\multicolumn{5}{l}{\\textit{Ucuz öğrenilmiş skor (MLP, $O(1)$ özellik), 10 sn, 5 tohum, aynı oturum}} \\\\")
            cp = C["pairs"]
            rows.append(row("Ucuz$-$kNN", cp["cheap_global-knn_global"]))
            rows.append(row("Ucuz$-$GNN", cp["cheap_global-gnn_global"]))
            rows.append(row("Ucuz$-$Rastgele", cp["cheap_global-random_global"]))
            rows.append(row("GNN$-$kNN", cp["gnn_global-knn_global"]))
    gf = KT / "gain_summary.json"
    if gf.exists():
        G = json.loads(gf.read_text(encoding="utf-8"))["pairs"]
        rows.append("\\addlinespace")
        rows.append("\\multicolumn{5}{l}{\\textit{Kazanç hedefli ucuz skor (yanlışlık $\\times$ fazla uzunluk), 10 sn, 5 tohum}} \\\\")
        rows.append(row("Kazanç$-$kNN", G["gain_global-knn_global"]))
        rows.append(row("Kazanç$-$Rastgele", G["gain_global-random_global"]))
        rows.append(row("kNN$-$Rastgele", G["knn_global-random_global"]))
    rows.append("\\addlinespace")
    rows.append("\\multicolumn{5}{l}{\\textit{60 sn bütçe, 3 tohum}} \\\\")
    b6 = S["budget_60s"]["pairs"]
    rows.append(row("GNN$-$Rastgele", b6["gnn_global-random_global"]))
    rows.append(row("GNN$-$kNN", b6["gnn_global-knn_global"]))
    rows.append(row("kNN$-$Rastgele", b6["knn_global-random_global"]))
    body = (
        "\\begin{table}[!htbp]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{Keşifsel analizler (ön-kayıtlı değil, çokluk düzeltmesi yok), birincil 38 VLSI problemi, global "
        "geometri. $\\Delta$: A$-$B gap farkı (puan; negatif A lehine).}\n"
        "\\label{tab:v2-exploratory}\n"
        "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}Xcccc@{}}\n\\toprule\n"
        "Karşılaştırma (A$-$B) & $\\Delta$ & \\%95 GA & Ö/E/G & Wilcoxon $p$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n")
    write("v2_exploratory.tex", body)


def main():
    tab_exploratory()
    prim = load("vlsi_primary_10s")
    if prim:
        tab_primary_strategies(prim)
        tab_primary_tests(prim)
        dat_anytime(prim, "v2_anytime_primary.dat")
        dat_per_instance(prim, "v2_per_instance_primary.dat")
    tab_suites(["vlsi_old_10s", "tsplib72_10s", "synth_holdout_1s", "synth_1000_10s"])
    tab_mechanism()
    tab_lkh(prim)
    tab_old_reanalysis()
    tab_sets()


if __name__ == "__main__":
    main()
