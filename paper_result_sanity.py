# -*- coding: utf-8 -*-
"""Independent mechanical and inference checks for completed article experiments."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_STEMS = (
    "diversification_synthetic_100",
    "diversification_tsplib_9x10",
    "diversification_world_10x10",
    "focused_synthetic_100",
    "focused_tsplib_9x10",
    "focused_world_10x10",
)


def finding(group: str, rule: str, measured: str, expected: str,
            source: str, action: str, severity: str = "RED") -> dict:
    return {"severity": severity, "group": group, "rule": rule,
            "measured": measured, "expected": expected,
            "source": source, "action": action}


def check_experiment(raw_path: Path, summary_path: Path) -> list[dict]:
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    issues = []
    seeds = payload["config"]["seeds"]
    source = str(raw_path.relative_to(HERE))
    for record in payload["records"]:
        name = str(record["instance"])
        runs = record["runs"]
        by_method = defaultdict(list)
        by_seed = defaultdict(list)
        for run in runs:
            by_method[run["method"]].append(run)
            by_seed[run["seed"]].append(run)
            run_source = f"{source}:{name}/seed={run['seed']}/{run['method']}"
            numeric = [run.get(key) for key in ("cost", "initial_cost", "quality", "elapsed")]
            if any(value is None or not math.isfinite(value) for value in numeric):
                issues.append(finding("S", "finite-metrics", str(numeric),
                                      "Sonlu maliyet, başlangıç, kalite ve süre", run_source,
                                      "Sayısal üretim yolunu düzelt; sonucu makaleden çıkar."))
                continue
            if run["cost"] < 0 or run["initial_cost"] < 0 or run["elapsed"] < 0:
                issues.append(finding("S", "nonnegative-metrics", str(numeric),
                                      "Maliyet ve süre negatif olamaz", run_source,
                                      "Maliyet/süre ölçümünü düzelt."))
            if run["cost"] > run["initial_cost"] + 1e-7:
                issues.append(finding("S", "incumbent-guarantee",
                                      f"cost={run['cost']} initial={run['initial_cost']}",
                                      "Nihai incumbent başlangıçtan kötü olamaz", run_source,
                                      "Kabul kapısını ve kayıt eşleşmesini düzelt."))
            kicks = run.get("kicks", 0)
            accepted = run.get("accepted_kicks", 0)
            if kicks < 0 or accepted < 0 or accepted > kicks:
                issues.append(finding("S", "kick-counts", f"kicks={kicks}, accepted={accepted}",
                                      "0 <= accepted_kicks <= kicks", run_source,
                                      "Kick sayaçlarını düzelt."))
            operator_counts = run.get("operator_counts")
            if operator_counts is not None:
                used = {name for name, count in operator_counts.items() if count > 0}
                if sum(operator_counts.values()) != kicks:
                    issues.append(finding("I", "operator-count-total",
                                          f"sum={sum(operator_counts.values())}, kicks={kicks}",
                                          "Operatör sayaçlarının toplamı kick sayısına eşit olmalı",
                                          run_source, "Sayaç ve kick kayıtlarını eşleştir."))
                allowed = {
                    "reactive": {"random"},
                    "suspicious_reactive": {"suspicious"},
                    "bandit": {"random", "suspicious", "entropy", "long_edge", "backbone"},
                    "adaptive_portfolio": {"random", "suspicious", "entropy", "long_edge", "backbone"},
                    "confidence_gate": {"random", "suspicious"},
                    "confidence_reactive": {"random", "suspicious"},
                    "focused_bandit": {"random", "suspicious", "long_edge"},
                    "focused_adaptive": {"random", "suspicious", "long_edge"},
                }.get(run["method"], {run["method"]})
                if not used <= allowed:
                    issues.append(finding("I", "operator-contract", str(sorted(used)),
                                          f"İzin verilen operatörler: {sorted(allowed)}", run_source,
                                          "Yöntem–operatör eşleşmesini düzelt."))
            bks = record.get("bks")
            expected_quality = (run["improvement_pct"] if bks is None
                                else -100.0 * (run["cost"] - bks) / bks)
            if abs(run["quality"] - expected_quality) > 1e-7:
                issues.append(finding("S", "quality-identity",
                                      f"recorded={run['quality']} recomputed={expected_quality}",
                                      "Kalite, maliyet ve BKS/başlangıçtan yeniden hesapla aynı olmalı",
                                      run_source, "Kalite dönüşümünü düzelt."))
            if bks is not None and run["cost"] < bks - 1e-7:
                issues.append(finding("I", "below-bks", f"cost={run['cost']} BKS={bks}",
                                      "BKS proven optimum ise beklenmez; upper bound ise açıklanabilir",
                                      run_source,
                                      "BKS katalog durumunu doğrula ve metinde açıkla.", "WARN"))
        for method, method_runs in by_method.items():
            counts = defaultdict(int)
            for run in method_runs:
                counts[run["seed"]] += 1
            if len(method_runs) != seeds or any(count != 1 for count in counts.values()):
                issues.append(finding("I", "paired-seed-counts",
                                      f"{name}/{method}: {dict(counts)}",
                                      f"Her yöntemde {seeds} benzersiz tohum", source,
                                      "Eksik/tekrarlı koşuyu tamamla; ikili dropna kullanma."))
        for seed, seed_runs in by_seed.items():
            initials = {round(run["initial_cost"], 7) for run in seed_runs}
            if len(initials) > 1:
                issues.append(finding("I", "paired-initial-tour",
                                      f"{name}/seed={seed}: {sorted(initials)}",
                                      "Bütün kollar aynı başlangıç turunu paylaşmalı", source,
                                      "Başlangıç turu eşleşmesini düzelt."))
        if record.get("bks") is not None and by_method:
            if all(all(run.get("kicks", 0) == 0 for run in method_runs)
                   for method_runs in by_method.values()):
                issues.append(finding("D", "kick-inactive-instance", name,
                                      "Kick stratejisi farkını ölçmek için en az bir kick gerekli",
                                      source,
                                      "Bu örneği kick etkisi kanıtı sayma; uzun bütçe analiziyle birlikte sınırlılık yaz.",
                                      "WARN"))
    if summary.get("n_complete_instances") != len(payload["records"]):
        issues.append(finding("I", "summary-completeness",
                              f"summary={summary.get('n_complete_instances')} raw={len(payload['records'])}",
                              "Özet ve ham kayıt problem sayıları aynı olmalı", str(summary_path.relative_to(HERE)),
                              "Özeti ham kayıttan yeniden üret."))
    for comparison in summary.get("comparisons", []):
        significant = comparison["p_holm"] < 0.05
        difference = comparison["mean_quality_difference"]
        expected_class = ("significant_benefit" if significant and difference > 0
                          else "significant_harm" if significant and difference < 0
                          else "inconclusive")
        if comparison["classification"] != expected_class:
            issues.append(finding("I", "classification-contract",
                                  comparison["classification"], expected_class,
                                  str(summary_path.relative_to(HERE)),
                                  "Sınıflandırmayı Holm p ve fark işaretinden yeniden üret."))
    return issues


def check_lkh_experiment(raw_path: Path, summary_path: Path) -> list[dict]:
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    issues = []
    seeds = payload["config"]["seeds"]
    budgets = payload["config"]["budgets"]
    source = str(raw_path.relative_to(HERE))
    records = payload["records"]
    keys = [(r["suite"], r["instance"], r["budget"], r["seed"], r["method"])
            for r in records]
    if len(keys) != len(set(keys)):
        issues.append(finding("I", "unique-run-keys", f"{len(keys) - len(set(keys))} yinelenen kayıt",
                              "Her problem/bütçe/tohum/yöntem anahtarı benzersiz olmalı",
                              source, "Yinelenen kayıtları tekilleştir ve özeti yeniden üret."))
    names = list(dict.fromkeys((r["suite"], r["instance"]) for r in records))
    for suite, name in names:
        for budget in budgets:
            selected = [r for r in records if r["suite"] == suite
                        and r["instance"] == name and r["budget"] == budget]
            for method in ("suspicious", "lkh3"):
                runs = [r for r in selected if r["method"] == method]
                run_seeds = [r["seed"] for r in runs]
                run_source = f"{source}:{name}/budget={budget}/{method}"
                if len(runs) != seeds or len(set(run_seeds)) != seeds:
                    issues.append(finding("I", "paired-seed-counts",
                                          f"{len(runs)} kayıt, {len(set(run_seeds))} benzersiz tohum",
                                          f"Her iki yöntemde {seeds} benzersiz tohum", run_source,
                                          "Eksik/tekrarlı koşuyu tamamla; kalite testinden sessizce çıkarma."))
                for run in runs:
                    numeric = [run.get("cost"), run.get("quality"), run.get("elapsed")]
                    if any(value is None or not math.isfinite(value) for value in numeric):
                        issues.append(finding("S", "finite-metrics", str(numeric),
                                              "Geçerli tur, kalite ve toplam süre sonlu olmalı",
                                              run_source, "Koşuyu kurtar; kurtarılamayan kaydı açıkça raporla."))
                        continue
                    if run["cost"] < 0 or run["elapsed"] < 0:
                        issues.append(finding("S", "nonnegative-metrics", str(numeric),
                                              "Maliyet ve süre negatif olamaz", run_source,
                                              "Maliyet/süre ölçümünü düzelt."))
                    expected_quality = -100.0 * (run["cost"] - run["bks"]) / run["bks"]
                    if abs(run["quality"] - expected_quality) > 1e-7:
                        issues.append(finding("S", "quality-identity",
                                              f"recorded={run['quality']} recomputed={expected_quality}",
                                              "Kalite, maliyet ve BKS'den yeniden hesapla aynı olmalı",
                                              run_source, "Kalite dönüşümünü düzelt."))
                    if run["cost"] < run["bks"] - 1e-7:
                        issues.append(finding("I", "below-bks",
                                              f"cost={run['cost']} BKS={run['bks']}",
                                              "BKS proven optimum ise beklenmez; upper bound ise açıklanabilir",
                                              run_source, "BKS katalog durumunu doğrula ve metinde açıkla.", "WARN"))
                    if method == "suspicious":
                        kicks = run.get("kicks", 0)
                        accepted = run.get("accepted_kicks", 0)
                        if kicks < 0 or accepted < 0 or accepted > kicks:
                            issues.append(finding("S", "kick-counts",
                                                  f"kicks={kicks}, accepted={accepted}",
                                                  "0 <= accepted_kicks <= kicks", run_source,
                                                  "Kick sayaçlarını düzelt."))
                        history = run.get("history", [])
                        if history and run["cost"] > history[0]["cost"] + 1e-7:
                            issues.append(finding("S", "incumbent-guarantee",
                                                  f"cost={run['cost']} initial={history[0]['cost']}",
                                                  "Nihai incumbent başlangıçtan kötü olamaz", run_source,
                                                  "Kabul kapısını ve kayıt eşleşmesini düzelt."))
    analyses = {(a["suite"], a["budget"]): a for a in summary["analyses"]}
    for suite, name in names:
        for budget in budgets:
            analysis = analyses.get((suite, budget))
            if analysis is None:
                issues.append(finding("I", "summary-analysis", f"{suite}/{budget}",
                                      "Her küme ve bütçe için özet analizi", source,
                                      "Özeti ham kayıttan yeniden üret."))
                continue
            complete = name in {row["instance"] for row in analysis["per_instance"]}
            incomplete = name in analysis["incomplete_instances"]
            if complete == incomplete:
                issues.append(finding("I", "summary-completeness",
                                      f"{name}: complete={complete}, incomplete={incomplete}",
                                      "Problem tam ya da eksik listelerden yalnız birinde olmalı",
                                      str(summary_path.relative_to(HERE)),
                                      "Özeti ham kayıttan yeniden üret."))
    for analysis in summary["analyses"]:
        comparison = analysis.get("lkh3_minus_suspicious")
        if comparison is None:
            continue
        p_holm = comparison.get("p_holm")
        if p_holm is None:
            issues.append(finding("I", "lkh-holm-correction", "p_holm=None",
                                  "Küme-bütçe karşılaştırmalarında Holm düzeltmesi",
                                  str(summary_path.relative_to(HERE)),
                                  "Özeti güncel summarize_lkh ile yeniden üret."))
            continue
        difference = comparison["mean_quality_difference"]
        expected_class = ("significant_benefit" if p_holm < 0.05 and difference > 0
                          else "significant_harm" if p_holm < 0.05 and difference < 0
                          else "inconclusive")
        if comparison.get("classification") != expected_class:
            issues.append(finding("I", "classification-contract",
                                  str(comparison.get("classification")), expected_class,
                                  str(summary_path.relative_to(HERE)),
                                  "Sınıflandırmayı Holm p ve fark işaretinden yeniden üret."))
    return issues


def write_report(report_path: Path, issues: list[dict], checked: list[str]) -> None:
    red = sum(issue["severity"] == "RED" for issue in issues)
    warn = len(issues) - red
    lines = ["# Makale Sonuç-Tahlili Raporu", "",
             f"Denetlenen deneyler: {', '.join(checked)}", "",
             f"Durum: {red} KIRMIZI, {warn} SARI", "",
             "Kırmızı bulgular makaleye girişi bloklar. Sarı bulgular mekanizma veya sınırlılık açıklaması gerektirir.", ""]
    if not issues:
        lines.append("Bütün mekanik ve istatistik-sunum denetimleri temiz.")
    for issue in issues:
        lines.extend([
            f"## {issue['severity']} — {issue['group']}/{issue['rule']}", "",
            f"- Ölçülen: {issue['measured']}",
            f"- Beklenen: {issue['expected']}",
            f"- Kaynak: {issue['source']}",
            f"- Yapılacak: {issue['action']}", "",
        ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems", default=",".join(DEFAULT_STEMS))
    parser.add_argument("--report", default="makale_workspace/gnn-crine-ils-joco/evidence/SANITY_ALERT.md")
    args = parser.parse_args()
    issues = []
    checked = []
    for stem in args.stems.split(","):
        raw = HERE / "results" / f"{stem}.json"
        summary = HERE / "results" / f"{stem}_summary.json"
        if not raw.exists() or not summary.exists():
            raise FileNotFoundError(f"completed experiment required: {stem}")
        checker = check_lkh_experiment if stem.startswith("lkh_comparison") else check_experiment
        issues.extend(checker(raw, summary))
        checked.append(stem)
    report = Path(args.report)
    if not report.is_absolute():
        report = HERE / report
    write_report(report, issues, checked)
    red = sum(issue["severity"] == "RED" for issue in issues)
    print(json.dumps({"checked": checked, "red": red,
                      "warnings": len(issues) - red, "report": str(report)}, ensure_ascii=False))
    return 2 if red else 0


if __name__ == "__main__":
    raise SystemExit(main())
