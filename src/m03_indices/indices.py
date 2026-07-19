"""Hiperspektral küpten 13 spektral bitki indeksi.

İndeks Tablosu
--------------
- ``NDVI``   — Bitki sağlığı (Rouse 1974): ``(R800-R670)/(R800+R670)``
- ``GNDVI``  — Klorofil hassasiyeti (Gitelson 1996): ``(R800-R550)/(R800+R550)``
- ``ARI``    — Antosiyanin/flavonol (Gitelson 2001): ``1/R550 - 1/R700``
- ``RVSI``   — Kırmızı kenar stresi (Merton 1999): ``(R714+R752)/2 - R733``
- ``ZTM``    — Klorofil oranı (Zarco-Tejada 2001): ``R750/R710``
- ``CRI``    — Karotenoid (Gitelson 2002): ``1/R510 - 1/R550``
- ``PRI``    — Fotokimyasal (Gamon 1997): ``(R531-R570)/(R531+R570)``
- ``mARI``   — NIR-norm. antosiyanin (Gitelson 2006): ``(1/R550-1/R700)*R800``
- ``SIPI``   — Yapı-bağımsız pigment (Peñuelas 1995): ``(R800-R445)/(R800-R680)``
- ``WBI``    — Su bandı (Peñuelas 1997): ``R900/R970``
- ``REP``    — Kırmızı kenar konumu (Guyot-Baret 1988)
- ``BES``    — Mavi kenar eğimi (özgün): ``(R450-R400)/50``
- ``FLAVI``  — Flavonoid (Cerovic 2012 uyarlama): ``R400/(R420*R700)``

Ayrıca Force-A Dualex prensibiyle uyumlu **NBI pipeline**: FRI, NDRE, NBI haritaları.

Tüm fonksiyonlar (lines, samples) 2B harita döndürür ve sıfıra-bölme korumalıdır.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.core.logging_setup import get as get_logger
from src.core.spectral_utils import find_band

log = get_logger("m03_indices")

# Geriye-uyum (eski projede ``get_band`` adı kullanılıyordu)
get_band = find_band


def _band(data: np.ndarray, wavelengths: Sequence[float], target_nm: float) -> np.ndarray:
    return data[:, :, find_band(wavelengths, target_nm)].astype(np.float64)


def _safe_inv(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return np.where(x > eps, 1.0 / x, 0.0)


def _finalize(arr: np.ndarray, default: float = 0.0) -> np.ndarray:
    return np.where(np.isfinite(arr), arr, default)


# ---------- 13 klasik indeks ---------------------------------------------------


def calc_ndvi(data, wavelengths):
    r800, r670 = _band(data, wavelengths, 800), _band(data, wavelengths, 670)
    den = r800 + r670
    return _finalize(np.where(den != 0, (r800 - r670) / den, 0.0))


def calc_gndvi(data, wavelengths):
    r800, r550 = _band(data, wavelengths, 800), _band(data, wavelengths, 550)
    den = r800 + r550
    return _finalize(np.where(den != 0, (r800 - r550) / den, 0.0))


def calc_ari(data, wavelengths):
    r550, r700 = _band(data, wavelengths, 550), _band(data, wavelengths, 700)
    return _finalize(_safe_inv(r550) - _safe_inv(r700))


def calc_rvsi(data, wavelengths):
    r714 = _band(data, wavelengths, 714)
    r752 = _band(data, wavelengths, 752)
    r733 = _band(data, wavelengths, 733)
    return _finalize((r714 + r752) / 2.0 - r733)


def calc_ztm(data, wavelengths):
    r750, r710 = _band(data, wavelengths, 750), _band(data, wavelengths, 710)
    return _finalize(np.where(r710 > 1e-10, r750 / r710, 0.0))


def calc_cri(data, wavelengths):
    r510, r550 = _band(data, wavelengths, 510), _band(data, wavelengths, 550)
    return _finalize(_safe_inv(r510) - _safe_inv(r550))


def calc_pri(data, wavelengths):
    r531, r570 = _band(data, wavelengths, 531), _band(data, wavelengths, 570)
    den = r531 + r570
    return _finalize(np.where(den != 0, (r531 - r570) / den, 0.0))


def calc_mari(data, wavelengths):
    r550 = _band(data, wavelengths, 550)
    r700 = _band(data, wavelengths, 700)
    r800 = _band(data, wavelengths, 800)
    return _finalize((_safe_inv(r550) - _safe_inv(r700)) * r800)


def calc_sipi(data, wavelengths):
    r800 = _band(data, wavelengths, 800)
    r445 = _band(data, wavelengths, 445)
    r680 = _band(data, wavelengths, 680)
    den = r800 - r680
    return _finalize(np.where(np.abs(den) > 1e-10, (r800 - r445) / den, 0.0))


def calc_wbi(data, wavelengths):
    r900 = _band(data, wavelengths, 900)
    r970 = _band(data, wavelengths, 970)
    return _finalize(np.where(r970 > 1e-10, r900 / r970, 0.0))


def calc_rep(data, wavelengths):
    r700 = _band(data, wavelengths, 700)
    r710 = _band(data, wavelengths, 710)
    r720 = _band(data, wavelengths, 720)
    r740 = _band(data, wavelengths, 740)
    num = (r700 + r740) / 2.0 - r710
    den = r720 - r710
    rep = 700.0 + 40.0 * np.where(np.abs(den) > 1e-10, num / den, 0.0)
    return _finalize(rep, default=700.0)


def calc_bes(data, wavelengths):
    r400 = _band(data, wavelengths, 400)
    r450 = _band(data, wavelengths, 450)
    return _finalize((r450 - r400) / 50.0)


def calc_flavi(data, wavelengths):
    r400 = _band(data, wavelengths, 400)
    r420 = _band(data, wavelengths, 420)
    r700 = _band(data, wavelengths, 700)
    den = r420 * r700
    return _finalize(np.where(den > 1e-10, r400 / den, 0.0))


# ---------- NBI odaklı pipeline -----------------------------------------------

NBI_TARGET_NMS: tuple[float, ...] = (410.0, 460.0, 670.0, 700.0, 800.0)


def slice_target_bands(
    data: np.ndarray, wavelengths: Sequence[float], targets: Sequence[float] | None = None
) -> dict[str, np.ndarray]:
    """3B küpten hedef dalga boylarına en yakın 2B bantları çıkar."""
    if targets is None:
        targets = NBI_TARGET_NMS
    layers: dict[str, np.ndarray] = {}
    for nm in targets:
        layers[f"R{int(round(nm))}"] = _band(data, wavelengths, nm)
    return layers


def calc_fri(R410: np.ndarray, R460: np.ndarray) -> np.ndarray:
    """FRI = 1/R410 - 1/R460 (Merzlyak 2005 türü flavonoid proxy)."""
    return _finalize(_safe_inv(R410) - _safe_inv(R460))


def calc_ndre(R700: np.ndarray, R800: np.ndarray) -> np.ndarray:
    """NDRE = (R800-R700)/(R800+R700) (klorofil/red-edge proxy)."""
    den = R800 + R700
    return _finalize(np.where(den > 1e-10, (R800 - R700) / den, 0.0))


def calc_nbi(R410, R460, R700, R800) -> np.ndarray:
    """NBI = NDRE/FRI (Force-A Dualex Chl/Flav prensibiyle uyumlu)."""
    fri = calc_fri(R410, R460)
    ndre = calc_ndre(R700, R800)
    return _finalize(np.where(np.abs(fri) > 1e-10, ndre / fri, 0.0))


def calc_nbi_maps(data: np.ndarray, wavelengths: Sequence[float]) -> dict[str, np.ndarray]:
    """5 hedef bant + FRI/NDRE/NBI 2B haritalarını dict olarak döndür.

    Hesaplamalar sıfıra-bölme/geçersiz-değer üretebilir (sonuç np.where ile
    zaten 0.0'a sabitleniyor); RuntimeWarning'leri ``np.errstate`` ile sustur.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        bands = slice_target_bands(data, wavelengths)
        R410, R460, R700, R800 = bands["R410"], bands["R460"], bands["R700"], bands["R800"]
        return {
            **bands,
            "FRI": calc_fri(R410, R460),
            "NDRE": calc_ndre(R700, R800),
            "NBI": calc_nbi(R410, R460, R700, R800),
        }


# ---------- Toplu hesaplama ---------------------------------------------------

INDEX_FUNCS = {
    "NDVI": calc_ndvi,
    "GNDVI": calc_gndvi,
    "ARI": calc_ari,
    "RVSI": calc_rvsi,
    "ZTM": calc_ztm,
    "CRI": calc_cri,
    "PRI": calc_pri,
    "mARI": calc_mari,
    "SIPI": calc_sipi,
    "WBI": calc_wbi,
    "REP": calc_rep,
    "BES": calc_bes,
    "FLAVI": calc_flavi,
}


def calc_all_indices(
    data: np.ndarray,
    wavelengths: Sequence[float],
    enabled: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Tüm (veya seçili) indeksleri tek seferde hesapla.

    Parameters
    ----------
    data
        ``(lines, samples, bands)`` hiperspektral küp.
    wavelengths
        Bant başına dalga boyu (nm).
    enabled
        Sadece bu indeksleri hesapla. ``None`` ise hepsi.
    """
    keys = enabled if enabled is not None else INDEX_FUNCS.keys()
    log.debug("İndeksler hesaplanıyor: %s", list(keys))
    out: dict[str, np.ndarray] = {}
    # Bireysel calc_* fonksiyonları zaten np.where ile sıfıra-bölme sonuçlarını
    # 0.0'a sabitliyor; RuntimeWarning'leri burada (toplu çağrı kapsamında)
    # bastırıyoruz ki konsol/dosya logları sade kalsın.
    with np.errstate(divide="ignore", invalid="ignore"):
        for name in keys:
            if name not in INDEX_FUNCS:
                raise ValueError(f"Bilinmeyen indeks: {name!r}. Geçerli: {list(INDEX_FUNCS)}")
            out[name] = INDEX_FUNCS[name](data, wavelengths)
    return out
