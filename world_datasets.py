# -*- coding: utf-8 -*-
"""Waterloo Üniversitesi Ülke (National TSP) koleksiyonu kataloğu.

Kaynak: https://www.math.uwaterloo.ca/tsp/world/countries.html
Her örnek bir ülkenin GERÇEK yerleşim/şehir noktalarıdır (National Imagery
and Mapping Agency + GEOnet verisi) -- VLSI'nin (mikroçip, düzgün-ızgara
eğilimli) tam karşıtı olan doğal/kümeli coğrafi dağılım sınıfı. Snake-Grid
ailesinin genelleme iddiası için üçüncü veri sınıfı: klasik TSPLIB (karma),
VLSI (çip yerleşimi), Ülke (coğrafya).

Dosya biçimi: düz .tsp, EDGE_WEIGHT_TYPE: EUC_2D (koordinatlar ondalık
derece x1000; Waterloo mesafeleri Öklid olarak tanımlar -- GEO DEĞİLDİR,
dosya başlığı ne diyorsa o kullanılır, motor bunu otomatik okur).

CATALOG: name -> (n, best_known, status)
  status "opt" = optimalliği kanıtlanmış tur uzunluğu
  status "ub"  = bilinen en iyi tur (üst sınır; countries sayfasında
                 "within x% of optimal" / "sample tour" diye işaretliler)
Değerler countries/summary sayfalarından (2026-07 itibarıyla).

İndirme kalıbı: https://www.math.uwaterloo.ca/tsp/world/{name}.tsp
"""
from __future__ import annotations

URL_PATTERN = "https://www.math.uwaterloo.ca/tsp/world/{name}.tsp"

# name -> (n, bks, status)   -- ülke adı yorumda
CATALOG: dict[str, tuple[int, int, str]] = {
    "wi29":     (29,     27603,   "opt"),   # Bati Sahra
    "dj38":     (38,     6656,    "opt"),   # Cibuti
    "qa194":    (194,    9352,    "opt"),   # Katar
    "uy734":    (734,    79114,   "opt"),   # Uruguay
    "zi929":    (929,    95345,   "opt"),   # Zimbabve
    "lu980":    (980,    11340,   "opt"),   # Luksemburg
    "rw1621":   (1621,   26051,   "opt"),   # Ruanda
    "mu1979":   (1979,   86891,   "opt"),   # Umman
    "nu3496":   (3496,   96132,   "opt"),   # Nikaragua
    "ca4663":   (4663,   1290319, "opt"),   # Kanada
    "tz6117":   (6117,   394718,  "ub"),    # Tanzanya (~%0.027)
    "eg7146":   (7146,   172386,  "ub"),    # Misir (~%0.021)
    "ym7663":   (7663,   238314,  "ub"),    # Yemen (~%0.030)
    "pm8079":   (8079,   114855,  "ub"),    # Panama (ornek tur)
    "ei8246":   (8246,   206171,  "ub"),    # Irlanda (~%0.021)
    "ar9152":   (9152,   837479,  "ub"),    # Arjantin (~%0.012)
    "ja9847":   (9847,   491924,  "opt"),   # Japonya
    "gr9882":   (9882,   300899,  "opt"),   # Yunanistan
    "kz9976":   (9976,   1061881, "ub"),    # Kazakistan (~%0.028)
    "fi10639":  (10639,  520527,  "opt"),   # Finlandiya
    "mo14185":  (14185,  427377,  "ub"),    # Fas (~%0.031)
    "ho14473":  (14473,  177092,  "ub"),    # Honduras (~%0.093)
    "it16862":  (16862,  557315,  "opt"),   # Italya
    "vm22775":  (22775,  569288,  "opt"),   # Vietnam
    "sw24978":  (24978,  855597,  "opt"),   # Isvec
    "bm33708":  (33708,  959289,  "ub"),    # Burma/Myanmar (~%0.031)
    "ch71009":  (71009,  4566506, "ub"),    # Cin (~%0.024)
}

BKS: dict[str, int] = {name: v[1] for name, v in CATALOG.items()}


def url_for(name: str) -> str:
    if name not in CATALOG:
        raise KeyError(f"Ülke kataloğunda yok: {name}")
    return URL_PATTERN.format(name=name)
