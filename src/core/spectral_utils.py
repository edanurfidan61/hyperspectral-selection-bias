"""Dalga boyu / bant erişimi için ortak yardımcılar.

Eski projede ``find_band`` ve ``get_band`` fonksiyonları hem
``module_2/indices.py`` hem ``module_1/visualize.py`` içinde duplike edilmişti.
Burada tek nüsha tutulur — her modül oradan import eder.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def find_band(wavelengths: Sequence[float], target_nm: float) -> int:
    """Hedef dalga boyuna en yakın bant indeksini döndür.

    Parameters
    ----------
    wavelengths
        Mevcut dalga boyları (nm).
    target_nm
        İstenen dalga boyu (nm).

    Returns
    -------
    int
        ``wavelengths`` içinde hedefe en yakın bantın indeksi.
    """
    arr = np.asarray(wavelengths, dtype=float)
    return int(np.argmin(np.abs(arr - target_nm)))


def get_band(cube: np.ndarray, wavelengths: Sequence[float], target_nm: float) -> np.ndarray:
    """Küpten hedef dalga boyuna en yakın 2B bant düzlemini al.

    Parameters
    ----------
    cube
        ``(H, W, C)`` şeklinde hiperspektral küp.
    wavelengths
        Bant başına dalga boyu listesi.
    target_nm
        İstenen dalga boyu.

    Returns
    -------
    np.ndarray
        ``(H, W)`` şeklinde 2B bant.
    """
    idx = find_band(wavelengths, target_nm)
    return cube[..., idx]


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Sıfıra-bölme korumalı element-wise bölme (indeks formüllerinde kullanılır)."""
    denom = np.where(np.abs(denominator) < eps, eps, denominator)
    return numerator / denom
