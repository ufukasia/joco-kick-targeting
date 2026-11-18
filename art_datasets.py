# -*- coding: utf-8 -*-
"""Waterloo Üniversitesi Sanat (TSP Art) koleksiyonu kataloğu.

Kaynak: https://www.math.uwaterloo.ca/tsp/data/art/  (Robert Bosch'un
"continuous line drawing" nokta-sanat örnekleri). Koleksiyonun amacı DEV
ölçek: 100.000 - 200.000 nokta; hiçbiri optimal çözülmemiştir (mona-lisa100K
için ödüllü açık problem). Nokta dağılımı yarıton (stippling) desenidir --
yoğunluk resmin tonuna göre değişir: ne VLSI'nin ızgara düzeni ne ülkelerin
coğrafi kümeleri; üç koleksiyonun en "yumuşak yoğunluk-gradyanlı" sınıfı.

CATALOG: name -> (n, best_known, status)  -- tümü "ub" (bilinen en iyi tur,
2026-07 itibarıyla; geçilmesi yeni dünya rekoru demektir).

Dosya biçimi: düz .tsp, EDGE_WEIGHT_TYPE: EUC_2D.
İndirme kalıbı: https://www.math.uwaterloo.ca/tsp/data/art/{name}.tsp
İSTİSNA: mona-lisa100K sitede /tsp/data/ml/ altında barındırılıyor (art/
altındaki istek 403 döner) -- URL_OVERRIDES'ta ayrıca tutulur.
"""
from __future__ import annotations

URL_PATTERN = "https://www.math.uwaterloo.ca/tsp/data/art/{name}.tsp"
URL_OVERRIDES = {
    "mona-lisa100K": "https://www.math.uwaterloo.ca/tsp/data/ml/mona-lisa100K.tsp",
}

CATALOG: dict[str, tuple[int, int, str]] = {
    "mona-lisa100K": (100000, 5757191, "ub"),
    "vangogh120K":   (120000, 6543609, "ub"),
    "venus140K":     (140000, 6810665, "ub"),
    "pareja160K":    (160000, 7619953, "ub"),
    "courbet180K":   (180000, 7888731, "ub"),
    "earring200K":   (200000, 8171677, "ub"),
}

BKS: dict[str, int] = {name: v[1] for name, v in CATALOG.items()}


def url_for(name: str) -> str:
    if name not in CATALOG:
        raise KeyError(f"Sanat kataloğunda yok: {name}")
    return URL_OVERRIDES.get(name) or URL_PATTERN.format(name=name)
