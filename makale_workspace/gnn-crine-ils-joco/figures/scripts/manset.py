# -*- coding: utf-8 -*-
"""Manset (hero) ureticisi -- figure-style.md A0.

Manset verisi ve kaynagi evidence/scripts/build_v3.py icindeki hero() / write_manset_tex() tarafindan
uretilir (vitrin_sec.py secimi + kanit kayitlari ayni gecisle tutarli kalsin diye). Bu dosya yalnizca
o ureticiyi calistirir:

    py -3.13 figures/scripts/manset.py      (== py -3.13 evidence/scripts/build_v3.py)

Cikti: figures/manset.tex, figures/data/manset_diff_{gnn,random}.dat, figures/data/manset_vitrin.csv,
       figures/data/manset_birim.csv, notes/vitrin-adaylari.md, evidence/vitrin.json
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[2] / "evidence" / "scripts" / "build_v3.py"), run_name="__main__")
