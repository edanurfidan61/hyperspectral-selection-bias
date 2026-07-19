"""NBI + flavonol odaklı genişletilmiş özellik çıkarımı.

Pipeline
--------
Maskelenmiş ham küp →
    A) **Mevcut 52 özellik** (5 ham bant + 8 türetilmiş katman × 4 istatistik)
    B) **SNV-mean spektrumu**       (yaprak-içi ortalama, SNV uygulanmış, B değer)
    C) **SG-1. türev SNV-mean**     (B değer; flavonol bandı bilgisi)
    D) **Continuum removal Band-I** (410-510 nm) → min depth + alan-altı (2 skaler)

Toplam: 52 + 2B + 2  (B = bant sayısı; sensöre göre değişir, ~204).

Cerovic-Dualex sensörünün biyofizik gerekçesiyle uyumlu: flavonol UV-mavi
(350-450 nm) emer, klorofil red-edge (670-700 nm) emer; oran/çarpım NBI verir.
Tam spektrum + türev + continuum-removal birlikte flavonol Band-I derinliğini
çok daha sadık temsil eder (Curran 1989, Kokaly 2001).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.m02_preprocessing.spectral import snv, sg_first_derivative
from src.m03_indices.indices import (
    INDEX_FUNCS,
    NBI_TARGET_NMS,
    calc_all_indices,
    calc_nbi_maps,
)

# Temel katmanlar: 5 ham bant + 3 birincil indeks
_BASE_LAYERS: list[str] = [f"R{int(round(nm))}" for nm in NBI_TARGET_NMS] + ["FRI", "NDRE", "NBI"]

# Flavonol-odaklı türetilmiş katmanlar
_DERIVED_LAYERS: list[str] = [
    "logR460_R410",
    "R670_R460",
    "R800_R460",
    "FRI_x_NDRE",
    "logNBI",
]

LAYER_NAMES: list[str] = _BASE_LAYERS + _DERIVED_LAYERS
STAT_NAMES: list[str] = ["mean", "std", "p25", "p75"]

# 13 klasik vejetasyon indeksi (NDVI/GNDVI/ARI/.../FLAVI). Her biri 4 istatistikle
# özetlenir → 13 × 4 = 52 ek özellik. Ayrıntı: src/m03_indices/indices.py.
INDEX_NAMES: list[str] = list(INDEX_FUNCS.keys())

# Flavonol Band-I için continuum-removal aralığı (nm)
_BAND_I_NM = (410.0, 510.0)

# argmax(d1) feature'ları için spektral pencereler (nm)
# - "red"  : klorofilin red-edge inflection noktası (REP'in türev karşılığı)
# - "blue" : flavonoid/karotenoid blue-edge inflection
# - "all"  : global argmax (peak slope wavelength)
_EDGE_WINDOWS: dict[str, tuple[float, float]] = {
    "red": (680.0, 760.0),
    "blue": (430.0, 510.0),
}
_EDGE_FEATURES: list[str] = ["red", "blue", "all"]


def _argmax_derivative_features(
    der_spec: np.ndarray,
    wavelengths: Sequence[float],
) -> tuple[list[float], list[str]]:
    """Yaprak-ortalama 1. türev spektrumundan argmax wl + value skalerleri.

    Her bölge (red/blue/all) için 2 özellik üretir:
      - ``d1_<region>_peak_wl``    → |d1|'in maksimum olduğu dalga boyu (nm)
      - ``d1_<region>_peak_value`` → o noktadaki d1 değeri (işaretli)

    Toplam 6 skaler. Bu feature'lar SVNS-türev kıvrımının lokasyonunu özetler;
    REP gibi formül-bazlı yaklaşıklamalardan farklı olarak doğrudan argmax.
    """
    wl = np.asarray(wavelengths, dtype=np.float64)
    abs_d1 = np.abs(der_spec)

    feats: list[float] = []
    names: list[str] = []
    for region in _EDGE_FEATURES:
        if region == "all":
            mask = np.ones_like(wl, dtype=bool)
        else:
            lo, hi = _EDGE_WINDOWS[region]
            mask = (wl >= lo) & (wl <= hi)
        if not mask.any() or not np.any(np.isfinite(abs_d1[mask])):
            feats.extend([0.0, 0.0])
        else:
            sub_idx = np.where(mask)[0]
            local_argmax = int(np.nanargmax(abs_d1[sub_idx]))
            i = sub_idx[local_argmax]
            feats.extend([float(wl[i]), float(der_spec[i])])
        names.extend([f"d1_{region}_peak_wl", f"d1_{region}_peak_value"])
    return feats, names


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    # GÖREV 8: a/b sıfıra-bölme uyarısını sustur; sonuç np.where ile yakalı.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > eps, a / b, 0.0)


def _safe_log(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a > eps, np.log(a + eps), 0.0)


def _build_derived_maps(maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    R410, R460, R670, R800 = maps["R410"], maps["R460"], maps["R670"], maps["R800"]
    FRI, NDRE, NBI = maps["FRI"], maps["NDRE"], maps["NBI"]
    derived = {
        "logR460_R410": _safe_log(_safe_div(R460, R410)),
        "R670_R460": _safe_div(R670, R460),
        "R800_R460": _safe_div(R800, R460),
        "FRI_x_NDRE": FRI * NDRE,
        "logNBI": _safe_log(np.maximum(NBI, 0.0) + 1.0),
    }
    return {k: np.where(np.isfinite(v), v, 0.0) for k, v in derived.items()}


def _summarize(arr_2d: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float]:
    vals = arr_2d[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(vals)),
        float(np.std(vals)),
        float(np.percentile(vals, 25)),
        float(np.percentile(vals, 75)),
    )


def _leaf_mean_spectrum(masked_cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Yaprak pikselleri üzerinden bant başına ortalama spektrum."""
    finite = np.isfinite(masked_cube)
    valid = finite & mask[..., None]
    safe = np.where(valid, masked_cube, 0.0)
    counts = valid.sum(axis=(0, 1)).astype(np.float64)
    sums = safe.sum(axis=(0, 1))
    mean = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return np.where(np.isfinite(mean), mean, 0.0).astype(np.float64)


def _continuum_band_i(spec: np.ndarray, wavelengths: Sequence[float]) -> tuple[float, float]:
    """410-510 nm aralığında doğrusal-taban continuum-removal: min derinlik + alan."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    lo, hi = _BAND_I_NM
    idx = np.where((wl >= lo) & (wl <= hi))[0]
    if idx.size < 3:
        return 0.0, 0.0
    sub = spec[idx]
    if not np.all(np.isfinite(sub)) or sub[0] <= 0 or sub[-1] <= 0:
        sub = np.where(np.isfinite(sub) & (sub > 0), sub, 1e-9)
    baseline = np.linspace(sub[0], sub[-1], num=sub.size)
    baseline = np.where(baseline > 1e-9, baseline, 1e-9)
    cr = sub / baseline
    depth = float(1.0 - np.min(cr))
    # NumPy 2.0+ uyumu: np.trapz kaldırıldı → np.trapezoid; eski sürümlere fallback.
    _trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    area = float(_trapezoid(np.maximum(1.0 - cr, 0.0), wl[idx]))
    return depth, area


def extract_features(
    masked_cube: np.ndarray,
    wavelengths: Sequence[float],
    *,
    snv_enabled: bool = True,
    savgol_enabled: bool = True,
) -> np.ndarray:
    """Maskelenmiş hiperspektral küpten genişletilmiş özellik vektörü.

    Çıktı boyutu: ``52 + 13*4 + 6 + 2*B + 2 = 112 + 2B``  (B = bant sayısı).

    Bileşenler
    ----------
    A) 52 sabit (5 ham bant + 8 türetilmiş katman) × 4 istatistik
    B) 52 indeks (13 klasik vejetasyon indeksi × 4 istatistik) — NDVI/GNDVI/...
    C) Yaprak-ortalama SNV spektrumu (B değer)
    D) SG-1. türev SNV spektrumu (B değer)
    E) Continuum-removal Band-I (depth + area, 2 skaler)
    F) Türev tepe noktaları: red/blue/all argmax(|d1|) → (wl, value) (6 skaler)

    GÖREV 6 — Ablation flag'leri:
        ``snv_enabled=False`` → SNV uygulanmaz, ortalama spektrum ham döner
        (kolon adları korunur, sadece içerik değişir).
        ``savgol_enabled=False`` → 1. türev hesaplanmaz, kolonlar 0 döner.
    """
    finite_voxel = np.isfinite(masked_cube)
    mask = finite_voxel.any(axis=2)
    raw_cube = np.where(finite_voxel, masked_cube, 0.0)

    # ---- A) Mevcut 52 özellik ------------------------------------------------
    base_maps = calc_nbi_maps(raw_cube, wavelengths)
    derived_maps = _build_derived_maps(base_maps)
    all_maps = {**base_maps, **derived_maps}
    fixed_feats: list[float] = []
    for name in LAYER_NAMES:
        fixed_feats.extend(_summarize(all_maps[name], mask))

    # ---- B) 13 klasik vejetasyon indeksi (52 özellik) ------------------------
    index_maps = calc_all_indices(raw_cube, wavelengths, enabled=INDEX_NAMES)
    index_feats: list[float] = []
    for name in INDEX_NAMES:
        index_feats.extend(_summarize(index_maps[name], mask))

    # ---- C/D) Tam spektrum SNV + 1. türev -----------------------------------
    mean_spec = _leaf_mean_spectrum(masked_cube, mask)  # (B,)
    # GÖREV 6: ablation için flag'ler — SNV/SG kapalıyken bile kolon sayısı
    # değişmesin diye ham ortalama / sıfır vektör ile doldurulur.
    if snv_enabled:
        snv_spec = snv(mean_spec.reshape(1, -1)).ravel()
    else:
        snv_spec = mean_spec.copy()
    if savgol_enabled:
        der_spec = sg_first_derivative(snv_spec.reshape(1, -1)).ravel()
    else:
        der_spec = np.zeros_like(snv_spec)
    snv_spec = np.where(np.isfinite(snv_spec), snv_spec, 0.0)
    der_spec = np.where(np.isfinite(der_spec), der_spec, 0.0)

    # ---- E) Continuum removal Band-I ----------------------------------------
    depth, area = _continuum_band_i(mean_spec, wavelengths)

    # ---- F) argmax(d1) tepe noktaları (red/blue/all) ------------------------
    edge_feats, _ = _argmax_derivative_features(der_spec, wavelengths)

    feats = np.concatenate([
        np.asarray(fixed_feats, dtype=np.float64),
        np.asarray(index_feats, dtype=np.float64),
        snv_spec.astype(np.float64),
        der_spec.astype(np.float64),
        np.asarray([depth, area], dtype=np.float64),
        np.asarray(edge_feats, dtype=np.float64),
    ])
    return np.where(np.isfinite(feats), feats, 0.0)


def get_feature_names(wavelengths: Sequence[float] | None = None) -> list[str]:
    """``extract_features`` çıktısıyla aynı sırada özellik isimleri.

    ``wavelengths`` verilmezse sadece sabit 52 özellik adı döner (geriye-uyum).
    """
    fixed = [f"{layer}_{stat}" for layer in LAYER_NAMES for stat in STAT_NAMES]
    index_feats = [f"{idx}_{stat}" for idx in INDEX_NAMES for stat in STAT_NAMES]
    edge_names: list[str] = []
    for region in _EDGE_FEATURES:
        edge_names.extend([f"d1_{region}_peak_wl", f"d1_{region}_peak_value"])
    if wavelengths is None:
        return fixed + index_feats + edge_names
    snv_names = [f"snv_R{int(round(w))}" for w in wavelengths]
    der_names = [f"d1snv_R{int(round(w))}" for w in wavelengths]
    cont_names = ["cont_BandI_depth", "cont_BandI_area"]
    return fixed + index_feats + snv_names + der_names + cont_names + edge_names
