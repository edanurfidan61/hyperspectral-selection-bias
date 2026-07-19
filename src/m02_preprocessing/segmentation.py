"""Hiperspektral küpten yaprak / arka plan ayrımı (maskeleme).

Ana yöntem ``segment_hybrid`` (NIR-tabanlı v5):
  1. NIR yansıma filtresi (Otsu adaptif eşik)
  2. Beyaz disk eleme (yüksek yansıma + düşük NDVI + düz spektral profil)
  3. Sap/klips temizliği (morfolojik açma + küçük bölge filtresi)
  4. Tamamen kapalı delikleri doldurma

Convex hull **uygulanmaz** — yaprak lobları arası boşluklarda flavonol/indeks
ortalamalarını bozardı.

Alternatif yöntemler (karşılaştırma için): ``segment_ndvi``, ``segment_kmeans``,
``segment_pca``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    binary_opening,
)
from scipy.ndimage import label as ndimage_label

from src.core.logging_setup import get as get_logger
from src.core.spectral_utils import find_band, safe_divide

log = get_logger("m02_preprocessing.segmentation")


def _otsu_threshold(values: np.ndarray) -> float:
    """1B değerler için Otsu eşiğini hesaplar (bimodal yaprak/arka plan ayrımı)."""
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return 0.3

    n_bins = 256
    hist, bin_edges = np.histogram(values, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return 0.3

    cumsum_w = np.cumsum(hist)
    cumsum_wm = np.cumsum(hist * bin_centers)
    total_mean = cumsum_wm[-1]

    best_threshold = bin_centers[0]
    best_variance = 0.0
    for i in range(1, n_bins):
        w0 = cumsum_w[i]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        mean0 = cumsum_wm[i] / w0
        mean1 = (total_mean - cumsum_wm[i]) / w1
        variance = w0 * w1 * (mean0 - mean1) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = bin_centers[i]
    return float(best_threshold)


def _morphological_cleanup(mask: np.ndarray, struct_size: int = 3) -> np.ndarray:
    """closing → fill_holes → opening (yumuşak temizlik, ince kenarları korur)."""
    structure = np.ones((struct_size, struct_size), dtype=bool)
    closed = binary_closing(mask, structure=structure, iterations=2)
    filled = binary_fill_holes(closed)
    cleaned = binary_opening(filled, structure=structure, iterations=1)
    return cleaned.astype(bool)


def _remove_small_regions(mask: np.ndarray, min_area_ratio: float = 0.005) -> np.ndarray:
    labeled, n_labels = ndimage_label(mask)
    if n_labels == 0:
        return mask
    min_area = mask.size * min_area_ratio
    cleaned = np.zeros_like(mask)
    for i in range(1, n_labels + 1):
        region = labeled == i
        if np.sum(region) >= min_area:
            cleaned[region] = True
    return cleaned


def _detect_white_reference(data: np.ndarray, wavelengths) -> np.ndarray:
    """Beyaz referans diski tespit et: yüksek yansıma + düşük NDVI + düz profil."""
    mean_ref = np.mean(data, axis=2)
    high_ref = mean_ref > 0.60

    r800 = data[:, :, find_band(wavelengths, 800.0)].astype(np.float64)
    r670 = data[:, :, find_band(wavelengths, 670.0)].astype(np.float64)
    ndvi = np.where(r800 + r670 > 0.01, safe_divide(r800 - r670, r800 + r670), 0.0)
    low_ndvi = ndvi < 0.4

    spectral_std = np.std(data, axis=2)
    relative_std = np.where(mean_ref > 0.01, spectral_std / mean_ref, 1.0)
    flat_spectrum = relative_std < 0.20

    disk_mask = high_ref & low_ndvi & flat_spectrum
    if np.sum(disk_mask) > 0:
        structure = np.ones((3, 3), dtype=bool)
        disk_mask = binary_closing(disk_mask, structure=structure)
        disk_mask = binary_fill_holes(disk_mask)
        disk_mask = binary_dilation(disk_mask, structure=structure, iterations=2)
    log.debug("Disk pikselleri: %d (%.1f%%)", int(np.sum(disk_mask)),
              100 * np.sum(disk_mask) / disk_mask.size)
    return disk_mask.astype(bool)


def segment_ndvi(
    data: np.ndarray, wavelengths, threshold: float | None = None
) -> np.ndarray:
    """NDVI tabanlı maske; ``threshold=None`` ise Otsu adaptif eşik kullanılır."""
    r800 = data[:, :, find_band(wavelengths, 800.0)].astype(np.float64)
    r670 = data[:, :, find_band(wavelengths, 670.0)].astype(np.float64)
    ndvi = np.where(r800 + r670 > 0.02, safe_divide(r800 - r670, r800 + r670), 0.0)

    if threshold is None:
        threshold = max(_otsu_threshold(ndvi.ravel()), 0.15)
        log.debug("NDVI Otsu eşik: %.3f", threshold)
    else:
        log.debug("NDVI sabit eşik: %.3f", threshold)

    mask = _morphological_cleanup(ndvi > threshold)
    log.debug("NDVI yaprak oranı: %.1f%%", 100 * np.sum(mask) / mask.size)
    return mask


def segment_kmeans(
    data: np.ndarray, wavelengths, n_clusters: int = 3, max_iter: int = 50
) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    lines, samples, bands = data.shape
    pixels = data.reshape(-1, bands).astype(np.float32)

    km = MiniBatchKMeans(
        n_clusters=n_clusters, max_iter=max_iter, batch_size=1024, random_state=42, n_init=3
    )
    labels_2d = km.fit_predict(pixels).reshape(lines, samples)

    r800 = data[:, :, find_band(wavelengths, 800.0)].astype(np.float64)
    r670 = data[:, :, find_band(wavelengths, 670.0)].astype(np.float64)
    ndvi = np.where(r800 + r670 != 0, safe_divide(r800 - r670, r800 + r670), 0.0)

    best_cluster, best_ndvi = -1, -float("inf")
    for c in range(n_clusters):
        cm = labels_2d == c
        if np.sum(cm) == 0:
            continue
        cn = ndvi[cm].mean()
        if cn > best_ndvi:
            best_ndvi, best_cluster = cn, c

    return _morphological_cleanup(labels_2d == best_cluster)


def segment_pca(
    data: np.ndarray, wavelengths, n_components: int = 5, n_clusters: int = 3
) -> np.ndarray:
    """PCA + K-means; arka plan kümesini hariç tutarak diğer tüm kümeleri yaprak sayar."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    lines, samples, bands = data.shape
    pixels = data.reshape(-1, bands).astype(np.float32)

    pca = PCA(n_components=n_components, random_state=42)
    pixels_pca = pca.fit_transform(pixels)
    log.debug("PCA varyans: %.1f%%", 100 * np.sum(pca.explained_variance_ratio_))

    km = MiniBatchKMeans(
        n_clusters=n_clusters, max_iter=50, batch_size=1024, random_state=42, n_init=3
    )
    labels_2d = km.fit_predict(pixels_pca).reshape(lines, samples)

    r800 = data[:, :, find_band(wavelengths, 800.0)].astype(np.float64)
    cluster_nir = {}
    for c in range(n_clusters):
        cm = labels_2d == c
        if np.sum(cm) > 0:
            cluster_nir[c] = r800[cm].mean()
    bg_cluster = min(cluster_nir, key=cluster_nir.get)
    return _morphological_cleanup(labels_2d != bg_cluster)


def segment_hybrid(data: np.ndarray, wavelengths) -> np.ndarray:
    """Hibrit v5 — NIR-tabanlı (önerilen). Bkz. modül docstring'i."""
    r800 = data[:, :, find_band(wavelengths, 800.0)].astype(np.float64)
    r750 = data[:, :, find_band(wavelengths, 750.0)].astype(np.float64)
    nir_mean = (r800 + r750) / 2.0

    nir_threshold = float(np.clip(_otsu_threshold(nir_mean.ravel()), 0.05, 0.30))
    log.debug("Hibrit NIR eşik: %.3f", nir_threshold)

    nir_mask = nir_mean > nir_threshold
    disk_mask = _detect_white_reference(data, wavelengths)
    no_disk = nir_mask & (~disk_mask)

    cleaned = _morphological_cleanup(no_disk, struct_size=3)
    cleaned = _remove_small_regions(cleaned, min_area_ratio=0.005)
    final_mask = binary_fill_holes(cleaned).astype(bool)

    log.debug("Hibrit yaprak oranı: %.1f%%", 100 * np.sum(final_mask) / final_mask.size)
    return final_mask


_METHOD_FUNCS = {
    "ndvi": segment_ndvi,
    "kmeans": segment_kmeans,
    "pca": segment_pca,
    "hybrid": segment_hybrid,
}


def best_mask(data: np.ndarray, wavelengths, method: str = "hybrid") -> np.ndarray:
    """Konfig tarafından seçilebilen genel maskeleme arayüzü."""
    key = method.lower()
    if key not in _METHOD_FUNCS:
        raise ValueError(f"Bilinmeyen yöntem: {method!r}. Geçerli: {sorted(_METHOD_FUNCS)}")
    return _METHOD_FUNCS[key](data, wavelengths)
