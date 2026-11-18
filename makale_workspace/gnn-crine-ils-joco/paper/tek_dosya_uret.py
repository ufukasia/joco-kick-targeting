#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tek_dosya_uret.py — paper/main.tex'ten Springer'in istediği tek .tex dosyasını (paper/main_tek.tex) üretir.

- Satırda tek başına duran her \\input{...} (bölümler, tablolar, figürler) özyinelemeli olarak gömülür.
- Kanıt izi yorumları (`% E…`) ve diğer yorumlar taşınmaz: yalnız yorumdan oluşan satır silinir,
  satır sonu yorumu `%` karakterine indirgenir (TeX anlamı değişmez; boşluk/paragraf davranışı korunur).
- Kaynak dosyalar (main.tex, sections/, evidence/tables/, figures/) değişmez; main_tek.tex her gönderimden
  önce yeniden üretilir.

Kullanım (paper/ içinden):  py tek_dosya_uret.py   →  xelatex main_tek; bibtex main_tek; xelatex ×2
"""
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
YOL = [PAPER.parent, PAPER]                     # main.tex'teki \input@path ile aynı: {../}{./}
INPUT = re.compile(r"^\s*\\input\{([^}]+)\}\s*$")
YORUM = re.compile(r"(?<!\\)%.*$")


def bul(ad: str, taban: Path) -> Path:
    for kok in [taban] + YOL:
        for aday in (kok / ad, kok / (ad + ".tex")):
            if aday.is_file():
                return aday.resolve()
    raise FileNotFoundError(ad)


def gom(dosya: Path, derinlik: int = 0) -> list[str]:
    if derinlik > 5:
        raise RuntimeError("\\input derinliği çok fazla: " + str(dosya))
    cikti = []
    for satir in dosya.read_text(encoding="utf-8").splitlines():
        m = INPUT.match(satir)
        if m:
            cikti += gom(bul(m.group(1), PAPER), derinlik + 1)
            continue
        if derinlik == 0 and satir.startswith("%%"):
            cikti.append(satir)                 # şablonun kendi başlık yorumları
            continue
        if satir.lstrip().startswith("%"):
            continue                            # yalnız yorum satırı
        cikti.append(YORUM.sub("%", satir))
    return cikti


def main() -> int:
    hedef = PAPER / (sys.argv[1] if len(sys.argv) > 1 else "main_tek.tex")
    satirlar = gom(PAPER / "main.tex")
    hedef.write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    print(f"yazıldı: {hedef} ({len(satirlar)} satır)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
