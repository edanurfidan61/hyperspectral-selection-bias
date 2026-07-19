"""ENVI formatındaki hiperspektral görüntü dosyalarını okumak için yardımcı modül.

ENVI iki dosyadan oluşur:
  - .hdr: Metin tabanlı başlık (lines, samples, bands, dalga boyları)
  - .dat: Binary piksel verisi (BIL/BSQ/BIP interleave)

Ryckewaert veri setindeki bazı yaprakların (2020-09-10_* serisi) .hdr
dosyalarında lines/samples yanlış yazılmıştır. Bu durumda gerçek boyutlar
.dat dosya boyutundan tekrar hesaplanır.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.core.logging_setup import get as get_logger

log = get_logger("m01_io.envi_loader")

_ENVI_DTYPE_MAP: dict[int, type] = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
}


def parse_hdr(hdr_path: str | Path) -> dict[str, Any]:
    """ENVI .hdr dosyasını ayrıştırıp metadata sözlüğü döndürür."""
    hdr_path = Path(hdr_path)
    if not hdr_path.exists():
        raise FileNotFoundError(f".hdr dosyası bulunamadı: {hdr_path}")

    metadata: dict[str, Any] = {}
    content = hdr_path.read_text(encoding="utf-8", errors="ignore")

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith(";") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().lower()
        value = value.strip()

        if key == "lines":
            metadata["lines"] = int(value)
        elif key == "samples":
            metadata["samples"] = int(value)
        elif key == "bands":
            metadata["bands"] = int(value)
        elif key == "data type":
            metadata["data_type"] = int(value)
        elif key == "interleave":
            metadata["interleave"] = value.lower()

    wavelength_match = re.search(
        r"wavelength\s*=\s*\{([^}]+)\}", content, re.IGNORECASE | re.DOTALL
    )
    if wavelength_match:
        raw_wl = wavelength_match.group(1)
        metadata["wavelengths"] = [
            float(w.strip()) for w in re.split(r"[,\s]+", raw_wl) if w.strip()
        ]
    else:
        metadata["wavelengths"] = []
        log.warning(".hdr dosyasında wavelength bilgisi yok: %s", hdr_path)

    return metadata


def envi_dtype_to_numpy(envi_code: int) -> type:
    """ENVI veri tipi kodunu numpy dtype'a çevirir."""
    if envi_code not in _ENVI_DTYPE_MAP:
        log.warning("Bilinmeyen ENVI veri tipi (%s); float32 varsayılıyor.", envi_code)
        return np.float32
    return _ENVI_DTYPE_MAP[envi_code]


def _resolve_shape_mismatch(
    actual_total: int, lines: int, samples: int, bands: int
) -> tuple[int, int]:
    """.hdr boyutları .dat ile uyuşmuyorsa gerçek (lines, samples) tahmin et."""
    if actual_total % bands != 0:
        raise ValueError(
            f".dat boyutu ({actual_total}) bant sayısına ({bands}) tam bölünemiyor."
        )
    total_pixels = actual_total // bands
    side = int(np.sqrt(total_pixels))
    if side * side == total_pixels:
        return side, side
    for candidate in range(side, 0, -1):
        if total_pixels % candidate == 0:
            return candidate, total_pixels // candidate
    raise ValueError(f"Boyut çözülemedi: {total_pixels} piksel için uygun kombinasyon yok.")


def load_dat(dat_path: str | Path, metadata: dict[str, Any]) -> np.ndarray:
    """ENVI .dat binary dosyasını ``(lines, samples, bands)`` float32 array olarak yükler."""
    dat_path = Path(dat_path)
    if not dat_path.exists():
        raise FileNotFoundError(f".dat dosyası bulunamadı: {dat_path}")

    lines = metadata["lines"]
    samples = metadata["samples"]
    bands = metadata["bands"]
    interleave = metadata.get("interleave", "bil")
    dtype = envi_dtype_to_numpy(metadata.get("data_type", 4))

    raw_data = np.fromfile(dat_path, dtype=dtype)
    expected_total = lines * samples * bands
    actual_total = raw_data.size

    if actual_total != expected_total:
        log.warning(
            ".hdr/.dat boyut uyuşmazlığı: hdr=%d×%d×%d=%d, dat=%d → düzeltiliyor",
            lines, samples, bands, expected_total, actual_total,
        )
        lines, samples = _resolve_shape_mismatch(actual_total, lines, samples, bands)
        metadata["lines"] = lines
        metadata["samples"] = samples
        log.info("Düzeltilmiş boyut: %d×%d×%d", lines, samples, bands)

    if interleave == "bil":
        data = raw_data.reshape((lines, bands, samples)).transpose(0, 2, 1)
    elif interleave == "bsq":
        data = raw_data.reshape((bands, lines, samples)).transpose(1, 2, 0)
    elif interleave == "bip":
        data = raw_data.reshape((lines, samples, bands))
    else:
        raise ValueError(f"Bilinmeyen interleave formatı: {interleave!r}")

    return data.astype(np.float32)


def load_envi(
    hdr_path: str | Path, dat_path: str | Path | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """``.hdr`` ve ``.dat`` dosyalarını birlikte yükler.

    ``dat_path`` verilmezse ``.hdr`` ile aynı isimli ``.dat`` aranır.
    """
    hdr_path = Path(hdr_path)
    if dat_path is None:
        dat_path = hdr_path.with_suffix(".dat")
    log.debug("ENVI yükleniyor: %s", hdr_path.name)
    metadata = parse_hdr(hdr_path)
    data = load_dat(dat_path, metadata)
    return data, metadata
