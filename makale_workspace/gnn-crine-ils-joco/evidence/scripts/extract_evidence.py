#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kanıt çıkarımı: tamamlanmış results/*_summary.json dosyalarından evidence/evidence.json üretir.

Yalnızca TAMAMLANMIŞ deneyler okunur (manifest, 2026-11-02):
  - Yeni CRINE-ILS iskeleti: diversification_{validation_40,synthetic_100,tsplib_9x10,world_10x10},
    focused_{validation_40,synthetic_100,tsplib_9x10,world_10x10}
  - Eski iskelet (ayrı raporlanır, yeni iskeletle BİRLEŞTİRİLMEZ):
    entropy_holdout_100_t1, entropy_fusion_holdout_100, entropy_tsplib_9x10,
    entropy_fusion_tsplib_9x10, entropy_fusion_validation_40, elite_entropy_validation_40
LKH-3 sonuçları eksiksiz 760 kayıtla, 30/120 saniyelik World ön-uçuşları ise yalnız
operasyonel ulaşılabilirlik notuyla eklenir; ho14473 için kick yoktur ve kalite çıkarımı yapılmaz.
Son blokta 16 problemlik bağımsız Waterloo VLSI doğrulaması, 72×5 TSPLIB ölçek deneyi,
beş boyut katmanı ve dört sonradan-seçilmiş devre/ızgara karşılaştırması bulunur.
KULLANILMAZ: entropy_holdout_100.json (yarım).

Her kayıt: sayı + kaynak_dosya + kaynak_konum (JSON yolu). Yeniden üretmek için:
  python evidence/scripts/extract_evidence.py
"""
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]          # makale_workspace/gnn-crine-ils-joco
ROOT = WS.parents[1]                               # proje kökü (tsplib)
RESULTS = ROOT / "results"
OUT = WS / "evidence" / "evidence.json"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# yeni iskelet: (küme etiketi, özet dosyası, ham dosya, takma ad)
NEW_SUITES = [
    ("validation_synthetic_40", "diversification_validation_40_summary.json", "diversification_validation_40.json"),
    ("holdout_synthetic_100", "diversification_synthetic_100_summary.json", "diversification_synthetic_100.json"),
    ("tsplib_9", "diversification_tsplib_9x10_summary.json", "diversification_tsplib_9x10.json"),
    ("world_9", "diversification_world_10x10_summary.json", "diversification_world_10x10.json"),
    ("focused_validation_synthetic_40", "focused_validation_40_summary.json", "focused_validation_40.json"),
    ("focused_holdout_synthetic_100", "focused_synthetic_100_summary.json", "focused_synthetic_100.json"),
    ("focused_tsplib_9", "focused_tsplib_9x10_summary.json", "focused_tsplib_9x10.json"),
    ("focused_world_10", "focused_world_10x10_summary.json", "focused_world_10x10.json"),
]

LEGACY_SUITES = [
    ("legacy_entropy_holdout_100", "entropy_holdout_100_t1_summary.json"),
    ("legacy_entropy_fusion_holdout_100", "entropy_fusion_holdout_100_summary.json"),
    ("legacy_entropy_tsplib_9", "entropy_tsplib_9x10_summary.json"),
    ("legacy_entropy_fusion_tsplib_9", "entropy_fusion_tsplib_9x10_summary.json"),
    ("legacy_entropy_fusion_validation_40", "entropy_fusion_validation_40_summary.json"),
    ("legacy_elite_entropy_validation_40", "elite_entropy_validation_40_summary.json"),
]

METRIC_UNIT = {
    "improvement_pct": "yuzde",
    "negative_gap_pct": "yuzde",
}


def load(name):
    p = RESULTS / name
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    kayitlar = []
    n = 0

    def add(metrik, yontem, veri_seti, deger, birim, kaynak_dosya, konum, **ek):
        nonlocal n
        n += 1
        rec = {
            "id": f"E{n:03d}",
            "metrik": metrik,
            "yontem": yontem,
            "veri_seti": veri_seti,
            "deger": deger,
            "birim": birim,
            "kaynak": "yazilim",
            "kaynak_dosya": kaynak_dosya,
            "kaynak_konum": konum,
            "atif": None,
            "durum": "gecerli",
        }
        if ek:
            rec["ek"] = ek
        kayitlar.append(rec)
        return rec

    def temiz_ek(kw):
        """Kaynaginda olmayan (None) alanlari ekleme; mevcut alanlar korunur."""
        return {k: v for k, v in kw.items() if v is not None}

    # ---------- yeni iskelet ----------
    for suite, summ_name, raw_name in NEW_SUITES:
        s = load(summ_name)
        summ_path = f"results/{summ_name}"
        raw_path = f"results/{raw_name}"
        metric = s["metric"]
        birim = METRIC_UNIT.get(metric, "")

        for method, agg in sorted(s["aggregate"].items()):
            for field, label in [("mean_quality", "ortalama_kalite"), ("mean_elapsed_s", "ortalama_sure"),
                                 ("mean_kicks", "ortalama_kick"), ("mean_accepted_kicks", "ortalama_kabul_edilen_kick")]:
                add(label, method, suite, agg[field], "sn" if "sure" in label else ("adet" if "kick" in label else birim),
                    summ_path, f"aggregate.{method}.{field}", iskelet="yeni")

        fr = s["friedman"]
        add("friedman_chi2", "omnibus", suite, fr["statistic"], "", summ_path, "friedman.statistic", iskelet="yeni")
        add("friedman_p", "omnibus", suite, fr["p_value"], "", summ_path, "friedman.p_value", iskelet="yeni")
        add("kendalls_w", "omnibus", suite, fr["kendalls_w"], "", summ_path, "friedman.kendalls_w", iskelet="yeni")

        for c in s.get("comparisons", []):
            add("kalite_farki_ortalama", c["method"], suite, c["mean_quality_difference"], "yuzde_puani",
                summ_path, f"comparisons[{c['comparison']}].mean_quality_difference",
                **temiz_ek(dict(opponent=c["opponent"], p_raw=c["p_raw"], p_holm=c.get("p_holm"),
                                significant_0_05=c.get("significant_0_05"), classification=c.get("classification"),
                                wins=c.get("wins"), ties=c.get("ties"), losses=c.get("losses"),
                                rank_biserial=c.get("rank_biserial"), bootstrap_95_ci=c.get("bootstrap_95_ci"),
                                median_quality_difference=c.get("median_quality_difference"))), iskelet="yeni")

        for c in s.get("comparisons_vs_suspicious", []) or []:
            add("kalite_farki_ortalama", c["method"], suite, c["mean_quality_difference"], "yuzde_puani",
                summ_path, f"comparisons_vs_suspicious[{c['comparison']}].mean_quality_difference",
                **temiz_ek(dict(opponent=c["opponent"], p_raw=c["p_raw"], p_holm=c.get("p_holm"),
                                significant_0_05=c.get("significant_0_05"), classification=c.get("classification"),
                                wins=c.get("wins"), ties=c.get("ties"), losses=c.get("losses"),
                                rank_biserial=c.get("rank_biserial"), bootstrap_95_ci=c.get("bootstrap_95_ci"),
                                median_quality_difference=c.get("median_quality_difference"))), iskelet="yeni")

        sec = s.get("secondary_crine_comparison")
        if sec:
            add("kalite_farki_ortalama", sec["method"], suite, sec["mean_quality_difference"], "yuzde_puani",
                summ_path, "secondary_crine_comparison.mean_quality_difference",
                **temiz_ek(dict(opponent=sec["opponent"], p_raw=sec["p_raw"], p_holm=sec.get("p_holm"),
                                classification="ikincil_karsilastirma",
                                wins=sec.get("wins"), ties=sec.get("ties"), losses=sec.get("losses"),
                                rank_biserial=sec.get("rank_biserial"), bootstrap_95_ci=sec.get("bootstrap_95_ci"),
                                median_quality_difference=sec.get("median_quality_difference"))),
                not_="CRINE-only erken durur; eş-bütçeli değil", iskelet="yeni")

        # protokol gerçekleri ham config'den (izlenebilirlik)
        raw = load(raw_name)
        cfg = raw["config"]
        add("protokol_problem_sayisi", "deney", suite, len(s["per_instance"]), "adet", raw_path, "config (instances/datasets) + per_instance uzunluğu", iskelet="yeni")
        add("protokol_tohum_sayisi", "deney", suite, cfg["seeds"], "adet", raw_path, "config.seeds", iskelet="yeni")
        add("protokol_yontem_butcesi", "deney", suite, cfg["method_time"], "sn", raw_path, "config.method_time", iskelet="yeni")
        add("protokol_deney_tohumu", "deney", suite, cfg["seed"], "", raw_path, "config.seed", iskelet="yeni")

    # ---------- eski iskelet (ayrı rapor) ----------
    for suite, summ_name in LEGACY_SUITES:
        s = load(summ_name)
        summ_path = f"results/{summ_name}"
        birim = METRIC_UNIT.get(s["metric"], "")
        for method, agg in sorted(s["aggregate"].items()):
            add("ortalama_kalite", method, suite, agg["mean_quality"], birim,
                summ_path, f"aggregate.{method}.mean_quality", iskelet="eski",
                not_="yeni CRINE-ILS ile birlestirilmez")
            add("ortalama_sure", method, suite, agg["mean_elapsed_s"], "sn",
                summ_path, f"aggregate.{method}.mean_elapsed_s", iskelet="eski",
                not_="yeni CRINE-ILS ile birlestirilmez")
        for c in s.get("comparisons", []):
            add("kalite_farki_ortalama", c.get("method", c["comparison"].split("_minus_")[0]), suite,
                c["mean_quality_difference"], "yuzde_puani",
                summ_path, f"comparisons[{c['comparison']}].mean_quality_difference",
                **temiz_ek(dict(opponent=c["comparison"].split("_minus_")[-1], p_raw=c["p_raw"],
                                p_holm=c.get("p_holm"), significant_0_05=c.get("significant_0_05"),
                                wins=c.get("wins"), ties=c.get("ties"), losses=c.get("losses"),
                                rank_biserial=c.get("rank_biserial"), bootstrap_95_ci=c.get("bootstrap_95_ci"),
                                median_quality_difference=c.get("median_quality_difference"))),
                iskelet="eski", not_="yeni CRINE-ILS ile birlestirilmez")

    # ---------- LKH-3: protokol her zaman, sonuçlar yalnız eksiksiz koşuda ----------
    lkh_path = "results/lkh_comparison_19x10.json"
    lkh = load("lkh_comparison_19x10.json")
    lcfg = lkh["config"]
    lkh_keys = {(r["suite"], r["instance"], r["budget"], r["seed"], r["method"])
                for r in lkh["records"]}
    lkh_expected = 19 * lcfg["seeds"] * len(lcfg["budgets"]) * 2
    lkh_complete = (len(lkh["records"]) == lkh_expected == len(lkh_keys)
                    and all(r["cost"] is not None for r in lkh["records"]))
    lkh_note = "tamamlandi" if lkh_complete else "kosu suruyor; SONUC SAYISI ALINMADI"
    add("protokol_problem_sayisi", "lkh_comparison", "lkh_19x10", 19, "adet",
        lkh_path, "config ve benzersiz koşu anahtarları", iskelet="yeni", not_=lkh_note)
    add("protokol_tohum_sayisi", "lkh_comparison", "lkh_19x10", lcfg["seeds"], "adet",
        lkh_path, "config.seeds", iskelet="yeni", not_=lkh_note)
    add("protokol_yontem_butcesi", "lkh_comparison", "lkh_19x10", "1.0 ve 5.0", "sn",
        lkh_path, "config.budgets", iskelet="yeni", not_=lkh_note)
    add("protokol_deney_tohumu", "lkh_comparison", "lkh_19x10", lcfg["seed"], "",
        lkh_path, "config.seed", iskelet="yeni", not_=lkh_note)

    # ---------- 30 sn World ön-uçuşu: ho14473 kick=0; tam koşu yapılmadı ----------
    world30_name = "focused_world_30s_preflight"
    world30_path = f"results/{world30_name}.json"
    world30_summary_path = f"results/{world30_name}_summary.json"
    world30_preflight_complete = False
    if (RESULTS / f"{world30_name}.json").exists() and (RESULTS / f"{world30_name}_summary.json").exists():
        world30 = load(f"{world30_name}.json")
        world30_preflight_complete = (len(world30["records"]) == 2
                            and all(len(r["runs"]) == 4 * world30["config"]["seeds"]
                                    for r in world30["records"]))
    world30_source = world30_path
    world30_note = "on_ucus tamam; ho14473 kick=0; tam 30 sn kosu yapilmadi; kalite cikarimi yok"
    add("protokol_problem_sayisi", "focused_world_30s_preflight", world30_name, 2, "adet",
        world30_source, "records", iskelet="yeni", not_=world30_note)
    add("protokol_tohum_sayisi", "focused_world_30s_preflight", world30_name, 1, "adet",
        world30_source, "config.seeds", iskelet="yeni", not_=world30_note)
    add("protokol_yontem_butcesi", "focused_world_30s_preflight", world30_name, 30.0, "sn",
        world30_source, "config.method_time", iskelet="yeni", not_=world30_note)
    add("protokol_deney_tohumu", "focused_world_30s_preflight", world30_name, 10301, "",
        world30_source, "config.seed", iskelet="yeni", not_=world30_note)

    # Yeni sonuçlar mevcut E001--E452 kimliklerini değiştirmemek için sona eklenir.
    if lkh_complete:
        summ_name = "lkh_comparison_19x10_summary.json"
        s = load(summ_name)
        summ_path = f"results/{summ_name}"
        for analysis in s["analyses"]:
            suite = f"lkh_{analysis['suite']}_{analysis['budget']:g}s"
            for method, agg in analysis["aggregate"].items():
                for field, label, unit in [("mean_quality", "ortalama_kalite", "yuzde"),
                                           ("mean_elapsed_s", "ortalama_sure", "sn")]:
                    add(label, method, suite, agg[field], unit, summ_path,
                        f"analyses[{suite}].aggregate.{method}.{field}", iskelet="yeni")
                for target, value in agg["target_gap_success"].items():
                    add("hedef_gap_basarisi", method, suite, value, "oran", summ_path,
                        f"analyses[{suite}].aggregate.{method}.target_gap_success.{target}",
                        hedef_gap_yuzde=float(target), iskelet="yeni")
            c = analysis["lkh3_minus_suspicious"]
            add("kalite_farki_ortalama", "lkh3", suite, c["mean_quality_difference"], "yuzde_puani",
                summ_path, f"analyses[{suite}].lkh3_minus_suspicious.mean_quality_difference",
                opponent="suspicious", p_raw=c["p_raw"], p_holm=c["p_holm"],
                wins=c["wins"], ties=c["ties"], losses=c["losses"],
                rank_biserial=c["rank_biserial"], bootstrap_95_ci=c["bootstrap_95_ci"],
                median_quality_difference=c["median_quality_difference"], iskelet="yeni")

    if world30_preflight_complete:
        summ_name = f"{world30_name}_summary.json"
        s = load(summ_name)
        summ_path = f"results/{summ_name}"
        for method, agg in sorted(s["aggregate"].items()):
            for field, label, unit in [("mean_quality", "ortalama_kalite", "yuzde"),
                                       ("mean_elapsed_s", "ortalama_sure", "sn"),
                                       ("mean_kicks", "ortalama_kick", "adet"),
                                       ("mean_accepted_kicks", "ortalama_kabul_edilen_kick", "adet")]:
                add(label, method, world30_name, agg[field], unit, summ_path,
                    f"aggregate.{method}.{field}", iskelet="yeni")
        fr = s["friedman"]
        for field, label in [("statistic", "friedman_chi2"), ("p_value", "friedman_p"),
                             ("kendalls_w", "kendalls_w")]:
            add(label, "omnibus", world30_name, fr[field], "", summ_path,
                f"friedman.{field}", iskelet="yeni")
        for group in ("comparisons", "comparisons_vs_suspicious"):
            for c in s[group]:
                add("kalite_farki_ortalama", c["method"], world30_name,
                    c["mean_quality_difference"], "yuzde_puani", summ_path,
                    f"{group}[{c['comparison']}].mean_quality_difference",
                    opponent=c["opponent"], p_raw=c["p_raw"], p_holm=c.get("p_holm"),
                    significant_0_05=c.get("significant_0_05"), classification=c.get("classification"),
                    wins=c["wins"], ties=c["ties"], losses=c["losses"],
                    rank_biserial=c["rank_biserial"], bootstrap_95_ci=c["bootstrap_95_ci"],
                    median_quality_difference=c["median_quality_difference"], iskelet="yeni")

    # 120 sn ön-uçuş: yalnız operasyonel ulaşılabilirlik; kalite çıkarımı yok.
    world120_name = "focused_world_120s_preflight"
    world120 = load(f"{world120_name}.json")
    world120_summary = load(f"{world120_name}_summary.json")
    world120_path = f"results/{world120_name}.json"
    world120_summ_path = f"results/{world120_name}_summary.json"
    note120 = "on_ucus; ho14473 kick=0; kalite cikarimi yok"
    for metric, value, unit, location in [
        ("protokol_problem_sayisi", 2, "adet", "records"),
        ("protokol_tohum_sayisi", 1, "adet", "config.seeds"),
        ("protokol_yontem_butcesi", 120.0, "sn", "config.method_time"),
        ("protokol_deney_tohumu", 10301, "", "config.seed"),
    ]:
        add(metric, "focused_world_120s_preflight", world120_name, value, unit,
            world120_path, location, iskelet="yeni", not_=note120)
    for method, agg in sorted(world120_summary["aggregate"].items()):
        for field, label, unit in [("mean_quality", "ortalama_kalite", "yuzde"),
                                   ("mean_elapsed_s", "ortalama_sure", "sn"),
                                   ("mean_kicks", "ortalama_kick", "adet"),
                                   ("mean_accepted_kicks", "ortalama_kabul_edilen_kick", "adet")]:
            add(label, method, world120_name, agg[field], unit, world120_summ_path,
                f"aggregate.{method}.{field}", iskelet="yeni", not_=note120)
    fr = world120_summary["friedman"]
    for field, label in [("statistic", "friedman_chi2"), ("p_value", "friedman_p"),
                         ("kendalls_w", "kendalls_w")]:
        add(label, "omnibus", world120_name, fr[field], "", world120_summ_path,
            f"friedman.{field}", iskelet="yeni", not_=note120)
    for group in ("comparisons", "comparisons_vs_suspicious"):
        for c in world120_summary[group]:
            add("kalite_farki_ortalama", c["method"], world120_name,
                c["mean_quality_difference"], "yuzde_puani", world120_summ_path,
                f"{group}[{c['comparison']}].mean_quality_difference",
                opponent=c["opponent"], p_raw=c["p_raw"], p_holm=c.get("p_holm"),
                significant_0_05=c.get("significant_0_05"), classification=c.get("classification"),
                wins=c["wins"], ties=c["ties"], losses=c["losses"],
                rank_biserial=c["rank_biserial"], bootstrap_95_ci=c["bootstrap_95_ci"],
                median_quality_difference=c["median_quality_difference"], iskelet="yeni", not_=note120)

    for suite, stem in (("waterloo_vlsi_16", "waterloo_vlsi_current_16x5"),
                        ("tsplib_scale_72", "tsplib_scale_72x5")):
        summ_name = f"{stem}_summary.json"
        raw_name = f"{stem}.json"
        s = load(summ_name)
        raw = load(raw_name)
        summ_path = f"results/{summ_name}"
        raw_path = f"results/{raw_name}"
        for method, agg in sorted(s["aggregate"].items()):
            for field, label, unit in (("mean_quality", "ortalama_kalite", "yuzde"),
                                       ("mean_elapsed_s", "ortalama_sure", "sn"),
                                       ("mean_kicks", "ortalama_kick", "adet"),
                                       ("mean_accepted_kicks", "ortalama_kabul_edilen_kick", "adet")):
                add(label, method, suite, agg[field], unit, summ_path,
                    f"aggregate.{method}.{field}", iskelet="yeni")
        for c in s["comparisons"]:
            add("kalite_farki_ortalama", c["method"], suite,
                c["mean_quality_difference"], "yuzde_puani", summ_path,
                f"comparisons[{c['comparison']}].mean_quality_difference",
                opponent=c["opponent"], p_raw=c["p_raw"], p_holm=c["p_holm"],
                significant_0_05=c["significant_0_05"], classification=c["classification"],
                wins=c["wins"], ties=c["ties"], losses=c["losses"],
                rank_biserial=c["rank_biserial"], bootstrap_95_ci=c["bootstrap_95_ci"],
                median_quality_difference=c["median_quality_difference"], iskelet="yeni")
        cfg = raw["config"]
        for metric, value, unit, location in (
                ("protokol_problem_sayisi", len(s["per_instance"]), "adet", "records"),
                ("protokol_tohum_sayisi", cfg["seeds"], "adet", "config.seeds"),
                ("protokol_yontem_butcesi", cfg["method_time"], "sn", "config.method_time"),
                ("protokol_deney_tohumu", cfg["seed"], "", "config.seed")):
            add(metric, "deney", suite, value, unit, raw_path, location, iskelet="yeni")

    scale = load("tsplib_scale_72x5_analysis.json")
    scale_path = "results/tsplib_scale_72x5_analysis.json"
    for c in scale["strata"]:
        add("kalite_farki_ortalama", "suspicious", f"tsplib_scale_{c['stratum']}",
            c["mean_quality_difference"], "yuzde_puani", scale_path,
            f"strata[{c['stratum']}].mean_quality_difference",
            opponent="random", p_raw=c["p_raw"], p_holm=c["p_holm"],
            significant_0_05=c["significant_0_05"], wins=c["wins"], ties=c["ties"],
            losses=c["losses"], rank_biserial=c["rank_biserial"],
            bootstrap_95_ci=c["bootstrap_95_ci"], n_instances=c["n_instances"],
            kick_active_both=c["kick_active_both"], zero_kick_both=c["zero_kick_both"],
            iskelet="yeni", not_="bes katman tek Holm ailesi")

    circuits = load("tsplib_circuit_regimes.json")
    circuit_path = "results/tsplib_circuit_regimes.json"
    for c in circuits["comparisons"]:
        add("kalite_farki_ortalama", "suspicious", c["regime"],
            c["mean_quality_difference"], "yuzde_puani", circuit_path,
            f"comparisons[{c['regime']}].mean_quality_difference",
            opponent="random", p_raw=c["p_raw"], p_holm=c["p_holm"],
            significant_0_05=c["significant_0_05"], wins=c["wins"], ties=c["ties"],
            losses=c["losses"], rank_biserial=c["rank_biserial"],
            bootstrap_95_ci=c["bootstrap_95_ci"], n_instances=c["n_instances"],
            kick_active_both=c["kick_active_both"], zero_kick_both=c["zero_kick_both"],
            iskelet="yeni", not_="sonradan secilmis alt-kume; kesifsel; bagimsiz dogrulama degil")

    out = {
        "deney_meta": {
            "donanim": "Egitim: RTX 4060 Ti (LKH-GNN) / CPU (CRINE-GNN); cikarsim CPU/NumPy (evidence/software_manifest.md, 'Hesaplama ve Kick-Aktiflik Sinirliliklari')",
            "tarih": "2026-11-02",
            "tekrar_sayisi": None,
            "tekrar_notu": "Istatistik birimi problem ornegidir; tohumlar ornek icinde ortalanir (summary cikti semasi, inference_unit='instance').",
            "kosul": "Nominal yontem butceleri 0.5/1/5/30/120 sn; kalite: sentetikte baslangictan yuzde iyilesme, gercek problemlerde negatif BKS gap'i (buyuk daha iyi). 30/120 sn World yalniz on_ucus; ho14473 kick=0.",
            "dislanan": ([lkh_path + " — kosu suruyor; sayi alinmadi"] if not lkh_complete else [])
                        + ["results/entropy_holdout_100.json — yarim dosya"]
                        + ([world30_path + " — henuz tamamlanmadi"] if not world30_preflight_complete else []),
        },
        "kayitlar": kayitlar,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"yazildi: {OUT} ({n} kayit)")


if __name__ == "__main__":
    main()
