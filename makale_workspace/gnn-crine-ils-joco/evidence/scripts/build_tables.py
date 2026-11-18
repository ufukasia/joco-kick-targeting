#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""LaTeX tablo üretici — hücreler yalnızca results/*_summary.json kayıtlarından.

figure-style.md §A4 iskeleti: tek sütun (svjour3) → `table` + `tabular*{/textwidth}` +
@{\extracolsep{\fill}}, caption üstte, \small, booktabs, birimler başlıkta.
PRE-ATTENTIVE KODLAMA (kural 7): classification alanına göre OTOMATIK —
  significant_benefit → \iyi{} (mavi kalın), significant_harm → \kotu{} (kırmızı eğik),
  diğerleri düz siyah; baseline (random) satırı nötr. Renk kararı elle değil, veriden.

Yeniden üretmek için:  python evidence/scripts/build_tables.py
"""
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
ROOT = WS.parents[1]
RESULTS = ROOT / "results"
OUTDIR = WS / "evidence" / "tables"
OUTDIR.mkdir(parents=True, exist_ok=True)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEN_SUITES = [
    ("val40", "diversification_validation_40_summary.json", "Val-40"),
    ("s100", "diversification_synthetic_100_summary.json", "Sent-100"),
    ("tsplib", "diversification_tsplib_9x10_summary.json", "TSPLIB-9"),
    ("world", "diversification_world_10x10_summary.json", "World-9"),
]
FOCUSED_SUITES = [
    ("s100", "focused_synthetic_100_summary.json", "Sent-100"),
    ("tsplib", "focused_tsplib_9x10_summary.json", "TSPLIB-9"),
    ("world", "focused_world_10x10_summary.json", "World-10"),
]
LEGACY_ROWS = [
    ("entropy_holdout_100_t1_summary.json", "Sent-100 $\\times$ 3, 1 sn"),
    ("entropy_fusion_holdout_100_summary.json", "Sent-100 $\\times$ 3, 1 sn (füzyon)"),
    ("entropy_tsplib_9x10_summary.json", "TSPLIB-9 $\\times$ 10, 5 sn"),
    ("entropy_fusion_tsplib_9x10_summary.json", "TSPLIB-9 $\\times$ 10, 5 sn (füzyon)"),
    ("entropy_fusion_validation_40_summary.json", "Val-40 $\\times$ 3, 0,5 sn (füzyon)"),
    ("elite_entropy_validation_40_summary.json", "Val-40 $\\times$ 3, 0,5 sn (elit)"),
]
METHOD_TR = {
    "random": "Random", "suspicious": "Suspicious", "entropy": "Entropy", "hybrid": "Hybrid",
    "elite_entropy": "Elite-entropy", "long_edge": "Long-edge", "backbone": "Backbone",
    "reactive": "Reactive", "bandit": "Bandit", "adaptive_portfolio": "Adaptive-port.",
    "suspicious_reactive": "Susp.-reactive", "focused_bandit": "Focused-bandit",
    "crine": "CRINE-only", "hybrid25": "Hybrid25", "hybrid50": "Hybrid50", "hybrid75": "Hybrid75",
    "geo_fusion": "Geo-fusion", "confidence_gate": "Confidence-gate",
    "confidence_reactive": "Confidence-react.", "focused_adaptive": "Focused-adaptive",
}


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def tr_decimal(value):
    return value.replace(".", ",")


def fmt_q(v):
    return tr_decimal(f"{v:.2f}")


def fmt_d(v):
    return tr_decimal(f"{v:+.2f}")


def fmt_p(p):
    if p is None:
        return "--"
    if p >= 0.9995:
        return "1"
    if p >= 0.0001:
        return tr_decimal(f"{p:.4f}".rstrip("0").rstrip("."))
    mant, ex = f"{p:.1e}".split("e")
    return f"${tr_decimal(mant)}\\times10^{{{int(ex):+d}}}$"


def cls_wrap(value, classification):
    """Kural 7: anlamlı farklar renklenir; GA sıfırı kesiyorsa/ belirsizse nötr."""
    if classification == "significant_benefit":
        return f"\\iyi{{{value}}}"
    if classification == "significant_harm":
        return f"\\kotu{{{value}}}"
    return value


def tablo_govde(suites):
    """Her küme için 3 sütun (kalite, Δ, p)."""
    spec = "@{\\extracolsep{\\fill}}l" + "ccc" * len(suites)
    lines = []
    head_grp = " & ".join(f"\\multicolumn{{3}}{{c}}{{{label}}} " for _, _, label in suites)
    lines.append(f"\\begin{{tabular*}}{{\\textwidth}}{{{spec}}}")
    lines.append("\\toprule")
    lines.append(f"Strateji & {head_grp} \\\\")
    lines.append("\\cmidrule(lr){2-4}" + "".join(f"\\cmidrule(lr){{{i*3+2}-{i*3+4}}}" for i in range(1, len(suites))))
    sub = " & ".join(["Kalite & $\\Delta$R & $p$"] * len(suites))
    lines.append(f" & {sub} \\\\")
    lines.append("\\midrule")
    return lines


def tab_ten_strategy():
    """On-strateji sonuçlarını sentetik ve gerçek kümeler için iki tablo üretir."""
    data = {k: load(f) for k, f, _ in TEN_SUITES}
    head = [
        "% OTOMATIK URETIM — evidence/scripts/build_tables.py",
        "% Kaynaklar: " + ", ".join(f"results/{f}" for _, f, _ in TEN_SUITES),
    ]
    block1 = tab_ten_strategy_block(
        data, TEN_SUITES[:2],
        "On strateji ve CRINE-only: sentetik kümede kalite, random farkı ve Holm-$p$ (CRINE: ham $p$). Mavi: yararlı; kırmızı: zararlı; düz siyah: Random referansı.",
        "tab:ten-strategy")
    block2 = tab_ten_strategy_block(
        data, TEN_SUITES[2:],
        "On strateji ve ikincil CRINE-only: gerçek kümelerde ortalama $-$BKS gap, random'a farkı ve Holm-$p$ (CRINE: ham $p$). Renk kuralı Tablo~\\ref{tab:ten-strategy} ile aynıdır.",
        "tab:ten-strategy-real")
    return "\n".join(head + block1 + block2) + "\n"


def tab_ten_strategy_block(data, suites, caption, label):
    """10 strateji + CRINE-only satirlari; suites alt kumesi uzerinden."""
    lines = [
        "\\begin{table}[!htbp]",
        " \\centering\\small",
        f" \\caption{{{caption}}}",
        f" \\label{{{label}}}",
    ]
    lines += tablo_govde(suites)
    comp = {k: {c["method"]: c for c in data[k]["comparisons"]} for k, _, _ in suites}
    order = data["s100"]["quality_ranking"]
    for m in order:
        row = [METHOD_TR[m]]
        for k, _, _ in suites:
            d = data[k]
            q = fmt_q(d["aggregate"][m]["mean_quality"])
            if m == "random":
                row += [q, "---", "---"]
            else:
                c = comp[k][m]
                row += [q, cls_wrap(fmt_d(c["mean_quality_difference"]), c.get("classification")), fmt_p(c.get("p_holm"))]
        lines.append(" & ".join(row) + " \\\\")
        if m == order[-1]:
            lines.append("\\midrule")
    # CRINE-only ikincil referans (Holm ailesi ve Friedman disinda; anlamlilik ham p'ye gore)
    row = ["CRINE-only$^{\\dagger}$"]
    for k, _, _ in suites:
        d = data[k]
        sec = d["secondary_crine_comparison"]
        d_crine = cls_wrap(fmt_d(sec["mean_quality_difference"]),
                           "significant_harm" if sec["p_raw"] < 0.05 else None)
        row += [fmt_q(d["aggregate"]["crine"]["mean_quality"]), d_crine, fmt_p(sec["p_raw"])]
    lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\end{table}")
    return lines


def tab_focused_random():
    data = {k: load(f) for k, f, _ in FOCUSED_SUITES}
    lines = [
        "% OTOMATIK URETIM — evidence/scripts/build_tables.py",
        "% Kaynaklar: " + ", ".join(f"results/{f}" for _, f, _ in FOCUSED_SUITES),
        "\\begin{table}[!htbp]",
        " \\centering\\small",
        " \\setlength{\\tabcolsep}{4pt}% tek tablo svjour3'te 6.66 pt tastigindan yerel daraltma",
        " \\caption{Focused yöntemler: üç kümede ortalama kalite, random'a farkı ve Holm-$p$; birimler Tablo~\\ref{tab:ten-strategy} ile aynı. Mavi kalın: anlamlı yararlı.}",
        " \\label{tab:focused-random}",
    ]
    lines += tablo_govde(FOCUSED_SUITES)
    comp = {k: {c["method"]: c for c in d["comparisons"]} for k, d in data.items()}
    order = data["s100"]["quality_ranking"]
    for m in order:
        row = [METHOD_TR[m]]
        for k, _, _ in FOCUSED_SUITES:
            d = data[k]
            q = fmt_q(d["aggregate"][m]["mean_quality"])
            if m == "random":
                row += [q, "---", "---"]
            else:
                c = comp[k][m]
                row += [q, cls_wrap(fmt_d(c["mean_quality_difference"]), c.get("classification")), fmt_p(c.get("p_holm"))]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def tab_focused_suspicious():
    data = {k: load(f) for k, f, _ in FOCUSED_SUITES}
    spec = "@{\\extracolsep{\\fill}}l" + "cc" * len(FOCUSED_SUITES)
    lines = [
        "% OTOMATIK URETIM — evidence/scripts/build_tables.py",
        "% Kaynaklar: " + ", ".join(f"results/{f}" for _, f, _ in FOCUSED_SUITES) + " → comparisons_vs_suspicious",
        "\\begin{table}[!htbp]",
        " \\centering\\small",
        " \\caption{Seçilmiş varyantların suspicious ile doğrudan karşılaştırması: üç kümede fark ve Holm-$p$; nötr değerlerde anlamlı fark yoktur.}",
        " \\label{tab:focused-suspicious}",
        f"\\begin{{tabular*}}{{\\textwidth}}{{{spec}}}",
        "\\toprule",
        "Varyant & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{label}}} " for _, _, label in FOCUSED_SUITES) + "\\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        " & " + " & ".join(["$\\Delta$S & $p$"] * len(FOCUSED_SUITES)) + " \\\\",
        "\\midrule",
    ]
    vs = {k: {c["method"]: c for c in d["comparisons_vs_suspicious"]} for k, d in data.items()}
    for m in ["suspicious_reactive", "focused_bandit"]:
        row = [METHOD_TR[m]]
        for k, _, _ in FOCUSED_SUITES:
            c = vs[k][m]
            row += [cls_wrap(fmt_d(c["mean_quality_difference"]), c.get("classification")), fmt_p(c.get("p_holm"))]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def tab_entropy_legacy():
    methods = ["random", "suspicious", "entropy", "hybrid25", "hybrid50", "hybrid75", "elite_entropy"]
    spec = "@{\\extracolsep{\\fill}}>{\\raggedright\\arraybackslash}X" + "c" * len(methods)
    lines = [
        "% OTOMATIK URETIM — evidence/scripts/build_tables.py",
        "% Kaynaklar: " + ", ".join(f"results/{f}" for f, _ in LEGACY_ROWS),
        "\\begin{table}[!htbp]",
        " \\centering\\small",
        " \\setlength{\\tabcolsep}{4pt}",
        " \\caption{Eski iskelet deneylerinde ortalama kalite (sentetik: iyileşme; gerçek: $-$gap); yeni CRINE-ILS ile birleştirilmez.}",
        " \\label{tab:entropy-legacy}",
        f"\\begin{{tabularx}}{{\\textwidth}}{{{spec}}}",
        "\\toprule",
        "Deney & " + " & ".join(METHOD_TR[m] for m in methods) + " \\\\",
        "\\midrule",
    ]
    for f, label in LEGACY_ROWS:
        agg = load(f)["aggregate"]
        row = [label]
        for m in methods:
            row.append(fmt_q(agg[m]["mean_quality"]) if m in agg else "--")
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def tab_protocol():
    """Deney kümeleri tablosu — sayılar ham config'lerden (JSON yolu satır sonu yorumda)."""
    rows = [
        ("On strateji, validation (sentetik)", 40, 3, "0,5", 314159, "tamam", "diversification_validation_40.json"),
        ("On strateji, bağımsız sentetik", 100, 3, "1", 271828, "tamam", "diversification_synthetic_100.json"),
        ("On strateji, standart TSPLIB", 9, 10, "5", 161803, "tamam", "diversification_tsplib_9x10.json"),
        ("On strateji, ilk dış World seti", 9, 10, "5", 141421, "tamam (gr9882 durduruldu)", "diversification_world_10x10.json"),
        ("Focused, validation (sentetik)", 40, 3, "0,5", 57721, "tamam", "focused_validation_40.json"),
        ("Focused, bağımsız sentetik", 100, 3, "1", 7447, "tamam", "focused_synthetic_100.json"),
        ("Focused, standart TSPLIB", 9, 10, "5", 8899, "tamam", "focused_tsplib_9x10.json"),
        ("Focused, yeni World", 10, 10, "5", 10301, "tamam", "focused_world_10x10.json"),
        ("Bağımsız Waterloo VLSI", 16, 5, "5", 62003, "tamam", "waterloo_vlsi_current_16x5.json"),
        ("TSPLIB ölçek katmanları", 72, 5, "5", 91009, "tamam; 8 parça", "tsplib_scale_72x5.json"),
        ("LKH-3 bütçe–performans", 19, 10, "1 ve 5", 42577,
         "tamam" if completed_lkh() else "sürüyor", "lkh_comparison_19x10.json"),
        ("Focused, World kick-aktiflik ön-uçuşu", 2, 1, "30", 10301,
         "tamam; ho14473 kick=0", "focused_world_30s_preflight.json"),
        ("Focused, uzun-bütçe ön-uçuşu", 2, 1, "120", 10301,
         "tamam; ho14473 kick=0", "focused_world_120s_preflight.json"),
    ]
    lines = [
        "% OTOMATIK URETIM — evidence/scripts/build_tables.py (protokol gercekleri ham config'lerden)",
        "\\begin{table}[!htbp]",
        " \\centering\\small",
        " \\caption{Deney kümeleri ve protokol: problem sayısı, tohum/problem, nominal yöntem bütçesi ve deney tohumu.}",
        " \\label{tab:protocol}",
        "\\begin{tabularx}{\\textwidth}{@{\\extracolsep{\\fill}} >{\\raggedright\\arraybackslash}X cccc >{\\raggedright\\arraybackslash}X}",
        "\\toprule",
        "Deney & Problemler & Tohum/problem & Bütçe [sn] & Tohum & Durum \\\\",
        "\\midrule",
    ]
    for label, nprob, seeds, budget, seed, status, src in rows:
        lines.append(f"{label} & {nprob} & {seeds} & {budget} & {seed} & {status} \\\\% results/{src}" if src.endswith(".json")
                     else f"{label} & {nprob} & {seeds} & {budget} & {seed} & {status} \\\\% {src}")
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def completed_lkh():
    raw_path = RESULTS / "lkh_comparison_19x10.json"
    summary_path = RESULTS / "lkh_comparison_19x10_summary.json"
    if not raw_path.exists() or not summary_path.exists():
        return False
    raw = load("lkh_comparison_19x10.json")
    keys = {(r["suite"], r["instance"], r["budget"], r["seed"], r["method"])
            for r in raw["records"]}
    expected = 19 * raw["config"]["seeds"] * len(raw["config"]["budgets"]) * 2
    return (len(raw["records"]) == expected == len(keys)
            and all(r["cost"] is not None for r in raw["records"]))


def completed_world30():
    raw_path = RESULTS / "focused_world_30s_10x10.json"
    summary_path = RESULTS / "focused_world_30s_10x10_summary.json"
    if not raw_path.exists() or not summary_path.exists():
        return False
    raw = load("focused_world_30s_10x10.json")
    return (len(raw["records"]) == 10
            and all(len(r["runs"]) == 4 * raw["config"]["seeds"] for r in raw["records"]))


def tab_lkh_comparison():
    head = ["% OTOMATIK URETIM — evidence/scripts/build_tables.py",
            "% Kaynak: results/lkh_comparison_19x10_summary.json"]
    if not completed_lkh():
        return "\n".join(head + ["% Koşu eksiksiz değil; sonuç tablosu üretilmedi."]) + "\n"
    s = load("lkh_comparison_19x10_summary.json")
    lines = head + [
        "\\begin{table}[!htbp]",
        " \\centering\\small",
        " \\setlength{\\tabcolsep}{3pt}",
        " \\caption{Suspicious ve LKH-3: eş nominal bütçede ortalama $-$BKS gap, gerçek toplam süre ve LKH-3$-$suspicious kalite farkı. Holm-$p$ dört küme--bütçe Wilcoxon testi üzerinden düzeltilmiştir; GA problem örnekleri üzerinden bootstrap \\%95 aralığıdır.}",
        " \\label{tab:lkh-comparison}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccc}",
        "\\toprule",
        "Küme & Bütçe & \\multicolumn{2}{c}{$-$Gap} & \\multicolumn{2}{c}{Süre [sn]} & $\\Delta$ & Holm-$p$ \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        " & [sn] & LKH-3 & Susp. & LKH-3 & Susp. & LKH$-$S & \\\\",
        "\\midrule",
    ]
    for a in s["analyses"]:
        agg = a["aggregate"]
        c = a["lkh3_minus_suspicious"]
        label = ("TSPLIB" if a["suite"] == "tsplib" else "World") + f"-{a['n_complete_instances']}"
        row = [label, tr_decimal(f"{a['budget']:g}"),
               fmt_q(agg["lkh3"]["mean_quality"]), fmt_q(agg["suspicious"]["mean_quality"]),
               fmt_q(agg["lkh3"]["mean_elapsed_s"]), fmt_q(agg["suspicious"]["mean_elapsed_s"]),
               cls_wrap(fmt_d(c["mean_quality_difference"]), c["classification"]),
               fmt_p(c["p_holm"])]
        lines.append(" & ".join(row) + " \\\\")
        lines.append(f"% {label} {a['budget']:g}s bootstrap95={c['bootstrap_95_ci']}")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def tab_world30():
    name = "focused_world_30s_10x10"
    head = ["% OTOMATIK URETIM — evidence/scripts/build_tables.py",
            f"% Kaynak: results/{name}_summary.json"]
    if not completed_world30():
        return "\n".join(head + ["% Koşu eksiksiz değil; sonuç tablosu üretilmedi."]) + "\n"
    s = load(f"{name}_summary.json")
    random_comp = {c["method"]: c for c in s["comparisons"]}
    susp_comp = {c["method"]: c for c in s["comparisons_vs_suspicious"]}
    lines = head + [
        "\\begin{table}[!htbp]",
        " \\centering\\small",
        " \\setlength{\\tabcolsep}{3pt}",
        " \\caption{30 saniyelik World koşusunda ortalama kalite, gerçek süre, kick etkinliği ve Holm düzeltmeli farklar. $\\Delta$R: random'a; $\\Delta$S: suspicious'e fark.}",
        " \\label{tab:world30}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccc}",
        "\\toprule",
        "Yöntem & $-$Gap & Süre & Kick & Kabul & $\\Delta$R & $p_R$ & $\\Delta$S ($p_S$) \\\\",
        "\\midrule",
    ]
    for m in s["quality_ranking"]:
        agg = s["aggregate"][m]
        if m == "random":
            dr = pr = ds = "---"
        else:
            cr = random_comp[m]
            dr = cls_wrap(fmt_d(cr["mean_quality_difference"]), cr.get("classification"))
            pr = fmt_p(cr["p_holm"])
            if m == "suspicious":
                ds = "---"
            else:
                cs = susp_comp[m]
                ds = f"{cls_wrap(fmt_d(cs['mean_quality_difference']), cs.get('classification'))} ({fmt_p(cs['p_holm'])})"
        row = [METHOD_TR[m], fmt_q(agg["mean_quality"]), fmt_q(agg["mean_elapsed_s"]),
               fmt_q(agg["mean_kicks"]), fmt_q(agg["mean_accepted_kicks"]), dr, pr, ds]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def tab_vlsi_primary():
    s = load("waterloo_vlsi_current_16x5_summary.json")
    c = s["comparisons"][0]
    lines = [
        "% OTOMATIK URETIM — evidence/scripts/build_tables.py",
        "% Kaynak: results/waterloo_vlsi_current_16x5_summary.json",
        "\\begin{table}[!htbp]",
        " \\centering\\small",
        " \\caption{Bağımsız 16 Waterloo VLSI problemi ve beş tohumda suspicious--random karşılaştırması. $-$Gap ve farklar puan; süre gerçekleşen ortalama saniyedir.}",
        " \\label{tab:vlsi-primary}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc}",
        "\\toprule",
        "Yöntem & $-$Gap & Süre & Kick & Kabul & $\\Delta$R & Wilcoxon $p$ \\\\",
        "\\midrule",
    ]
    for method in ("random", "suspicious"):
        agg = s["aggregate"][method]
        if method == "random":
            delta = p = "---"
        else:
            delta = cls_wrap(fmt_d(c["mean_quality_difference"]), c["classification"])
            p = cls_wrap(fmt_p(c["p_holm"]), c["classification"])
        row = [METHOD_TR[method], fmt_q(agg["mean_quality"]),
               fmt_q(agg["mean_elapsed_s"]), fmt_q(agg["mean_kicks"]),
               fmt_q(agg["mean_accepted_kicks"]), delta, p]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def tab_tsplib_scale():
    s = load("tsplib_scale_72x5_analysis.json")
    labels = {
        "n_040_120": "42--120", "n_121_249": "121--249",
        "n_250_499": "250--499", "n_500_999": "500--999",
        "n_1000_2500": "1.000--2.500",
    }
    lines = [
        "% OTOMATIK URETIM — evidence/scripts/build_tables.py",
        "% Kaynak: results/tsplib_scale_72x5_analysis.json",
        "\\begin{table}[!htbp]",
        " \\centering\\small\\setlength{\\tabcolsep}{3pt}",
        " \\caption{72 TSPLIB problemi ve beş tohumda boyut katmanları. $\\Delta$: suspicious--random negatif-gap farkı; G/B/M: galibiyet/beraberlik/mağlubiyet; beş test tek Holm ailesindedir.}",
        " \\label{tab:tsplib-scale}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccc}",
        "\\toprule",
        "Nokta & $n$ & $\\Delta$ & G/B/M & Ham $p$ & Holm-$p$ & Etki & Kick/0 \\\\",
        "\\midrule",
    ]
    for c in s["strata"]:
        wtl = f"{c['wins']}/{c['ties']}/{c['losses']}"
        activity = f"{c['kick_active_both']}/{c['zero_kick_both']}"
        row = [labels[c["stratum"]], str(c["n_instances"]),
               fmt_d(c["mean_quality_difference"]), wtl, fmt_p(c["p_raw"]),
               fmt_p(c["p_holm"]), fmt_d(c["rank_biserial"]), activity]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def tab_circuit_regimes():
    s = load("tsplib_circuit_regimes.json")
    labels = {
        "pcb_drilling_all": "Devre-delme, tümü",
        "pcb_drilling_n_le_999": "Devre-delme, $\\leq999$",
        "pcb_drilling_n_ge_1000": "Devre-delme, $\\geq1000$",
        "rattled_grid": "Rattled grid",
    }
    lines = [
        "% OTOMATIK URETIM — evidence/scripts/build_tables.py",
        "% Kaynak: results/tsplib_circuit_regimes.json",
        "\\begin{table}[!htbp]",
        " \\centering\\small\\setlength{\\tabcolsep}{3pt}",
        " \\caption{Sonradan seçilmiş TSPLIB devre ve ızgara alt-kümelerinde keşifsel suspicious--random farkları. Dört test Holm düzeltmelidir; sonuçlar bağımsız doğrulama değildir. Kick/0: iki yöntemde kick-aktif / iki yöntemde sıfır-kick problem sayısı.}",
        " \\label{tab:circuit-regimes}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc}",
        "\\toprule",
        "Rejim & $n$ & $\\Delta$ & G/B/M & Ham $p$ & Holm-$p$ & Kick/0 \\\\",
        "\\midrule",
    ]
    for c in s["comparisons"]:
        row = [labels[c["regime"]], str(c["n_instances"]),
               fmt_d(c["mean_quality_difference"]),
               f"{c['wins']}/{c['ties']}/{c['losses']}", fmt_p(c["p_raw"]),
               fmt_p(c["p_holm"]), f"{c['kick_active_both']}/{c['zero_kick_both']}"]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    uretimler = {
        "tab_protocol.tex": tab_protocol(),
        "tab_ten_strategy.tex": tab_ten_strategy(),
        "tab_focused_random.tex": tab_focused_random(),
        "tab_focused_suspicious.tex": tab_focused_suspicious(),
        "tab_entropy_legacy.tex": tab_entropy_legacy(),
        "tab_lkh_comparison.tex": tab_lkh_comparison(),
        "tab_world30.tex": tab_world30(),
        "tab_vlsi_primary.tex": tab_vlsi_primary(),
        "tab_tsplib_scale.tex": tab_tsplib_scale(),
        "tab_circuit_regimes.tex": tab_circuit_regimes(),
    }
    for ad, icerik in uretimler.items():
        (OUTDIR / ad).write_text(icerik, encoding="utf-8")
        print(f"yazildi: evidence/tables/{ad}")


if __name__ == "__main__":
    main()
