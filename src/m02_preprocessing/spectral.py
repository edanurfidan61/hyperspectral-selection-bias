"""Yansıma spektrumlarına uygulanan ön işleme yöntemleri.

Tipik pipeline sırası: ham veri → segmentasyon → bu modülün yöntemleri →
özellik çıkarımı → model.

Yöntemler
---------
- ``savitzky_golay``   : Polinom tabanlı yumuşatma (Savitzky & Golay, 1964)
- ``snv``              : Standard Normal Variate (Barnes et al., 1989)
- ``msc``              : Multiplicative Scatter Correction (Geladi et al., 1985)
- ``first_derivative`` : Basit np.diff türev
- ``sg_first_derivative``: Savitzky-Golay tabanlı 1. türev (önerilen)
- ``continuum_removal``: Konveks zarf veya doğrusal taban çıkarma
- ``apply_pipeline``   : Yukarıdakileri zincirler ("sg", "snv", "d1", "msc", "continuum")

Tüm fonksiyonlar 1B veya (N, bands) 2B array kabul eder, aynı şekilde döndürür.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.signal import savgol_filter

from src.core.logging_setup import get as get_logger

log = get_logger("m02_preprocessing.spectral")


def _ensure_2d(spectra: np.ndarray) -> tuple[np.ndarray, bool]:
    single = spectra.ndim == 1
    if single:
        spectra = spectra.reshape(1, -1)
    return spectra, single


def savitzky_golay(
    spectra: np.ndarray, window_length: int = 11, polyorder: int = 2, deriv: int = 0
) -> np.ndarray:
    """Bant ekseni boyunca Savitzky-Golay filtresi uygular.

    ``window_length`` tek sayı olmalı ve ``polyorder``'dan büyük olmalı.
    """
    spectra, single = _ensure_2d(spectra)
    result = savgol_filter(
        spectra, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1
    )
    return result.ravel() if single else result


def snv(spectra: np.ndarray) -> np.ndarray:
    """Her piksel için ``(x - mean) / std`` normalizasyonu — saçılma düzeltir."""
    spectra, single = _ensure_2d(spectra)
    means = np.mean(spectra, axis=1, keepdims=True)
    stds = np.std(spectra, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    result = (spectra - means) / stds
    return result.ravel() if single else result


def first_derivative(spectra: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """``np.diff`` ile basit 1. türev. Çıktı 1 bant daha kısadır."""
    spectra, single = _ensure_2d(spectra)
    result = np.diff(spectra, axis=1) / spacing
    return result.ravel() if single else result


def sg_first_derivative(
    spectra: np.ndarray, window_length: int = 11, polyorder: int = 2
) -> np.ndarray:
    """Savitzky-Golay tabanlı 1. türev — bant sayısını korur, daha kararlıdır."""
    return savitzky_golay(spectra, window_length=window_length, polyorder=polyorder, deriv=1)


def msc(spectra: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Multiplicative Scatter Correction — saçılma kaynaklı çarpımsal etkileri giderir."""
    spectra, single = _ensure_2d(spectra)
    if reference is None:
        reference = np.mean(spectra, axis=0)
    result = np.zeros_like(spectra, dtype=np.float64)
    for i in range(spectra.shape[0]):
        b, a = np.polyfit(reference, spectra[i, :], 1)
        if abs(b) < 1e-10:
            result[i, :] = spectra[i, :]
        else:
            result[i, :] = (spectra[i, :] - a) / b
    return result.ravel() if single else result


def continuum_removal(spectra: np.ndarray, method: str = "convex_hull") -> np.ndarray:
    """Taban çizgisini çıkararak emilim özelliklerini vurgular.

    ``method``:
      - ``"convex_hull"``: yaklaşık konveks zarf
      - ``"linear"``     : başlangıç-bitiş arası doğrusal taban
    """
    spectra, single = _ensure_2d(spectra)
    result = np.zeros_like(spectra, dtype=np.float64)
    for i, spec in enumerate(spectra):
        if method == "convex_hull":
            continuum = np.maximum.accumulate(spec[::-1])[::-1]
        elif method == "linear":
            continuum = np.linspace(spec[0], spec[-1], len(spec))
        else:
            raise ValueError(f"Bilinmeyen continuum yöntemi: {method!r}")
        continuum = np.where(continuum == 0, 1e-9, continuum)
        result[i] = spec / continuum
    return result.ravel() if single else result


_STEP_FUNCS = {
    "sg": lambda x: savitzky_golay(x),
    "snv": lambda x: snv(x),
    "d1": lambda x: sg_first_derivative(x),
    "msc": lambda x: msc(x),
    "continuum": lambda x: continuum_removal(x),
}


def apply_pipeline(spectra: np.ndarray, steps: Iterable[str]) -> np.ndarray:
    """Sırayla birden fazla ön işleme adımı uygular.

    Geçerli adımlar: ``sg``, ``snv``, ``d1``, ``msc``, ``continuum``.
    """
    result = spectra.astype(np.float64, copy=True)
    for step in steps:
        key = step.lower().strip()
        if key not in _STEP_FUNCS:
            raise ValueError(
                f"Bilinmeyen ön işleme adımı: {step!r}. "
                f"Geçerli: {sorted(_STEP_FUNCS)}"
            )
        result = _STEP_FUNCS[key](result)
        log.debug("ön işleme: %s uygulandı", key)
    return result
