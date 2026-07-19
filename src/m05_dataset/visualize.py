"""03_visualization aşaması: RGB sentezi + indeks haritaları + spektral profiller.

Her yaprak için ``outputs/03_visualization/<leaf_id>/`` altına dosyalar yazılır:
  - ``<leaf>_rgb.png``
  - ``<leaf>_mask_overlay.png``
  - ``<leaf>_spectral_profile.png``
  - ``<leaf>_<INDEX>_map.png`` (her etkin indeks için)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core import paths
from src.core.logging_setup import get as get_logger
from src.core.spectral_utils import find_band
from src.m01_io.envi_loader import load_envi
from src.m02_preprocessing.segmentation import best_mask
from src.m03_indices.indices import calc_all_indices
from src.m05_dataset.builder import find_leaf_folders

log = get_logger("m05_dataset.visualize")


def make_rgb(data: np.ndarray, wavelengths: Sequence[float]) -> np.ndarray:
    """670 / 550 / 450 nm bantlarından gerçek-renkli RGB üret."""
    red = data[:, :, find_band(wavelengths, 670)].astype(np.float64)
    green = data[:, :, find_band(wavelengths, 550)].astype(np.float64)
    blue = data[:, :, find_band(wavelengths, 450)].astype(np.float64)

    def normalize(ch: np.ndarray) -> np.ndarray:
        c_min, c_max = ch.min(), ch.max()
        return np.zeros_like(ch) if c_max - c_min == 0 else (ch - c_min) / (c_max - c_min)

    rgb = np.stack([normalize(red), normalize(green), normalize(blue)], axis=2)
    return (rgb * 255).astype(np.uint8)


def plot_rgb(rgb: np.ndarray, save_path: Path, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mask_overlay(rgb: np.ndarray, mask: np.ndarray, save_path: Path, title: str) -> None:
    overlay = rgb.copy()
    overlay[~mask, 0] = np.clip(overlay[~mask, 0].astype(int) + 100, 0, 255).astype(np.uint8)
    overlay[~mask, 1] = (overlay[~mask, 1] * 0.3).astype(np.uint8)
    overlay[~mask, 2] = (overlay[~mask, 2] * 0.3).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Orijinal RGB", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(title, fontsize=11)
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_index_map(
    index_array: np.ndarray, mask: np.ndarray, save_path: Path,
    title: str, cmap: str = "RdYlGn",
) -> None:
    display = index_array.astype(np.float64).copy()
    display[~mask] = np.nan
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(display, cmap=cmap)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("İndeks değeri")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_profile(
    data: np.ndarray, mask: np.ndarray, wavelengths: Sequence[float],
    save_path: Path, title: str,
) -> None:
    leaf_pixels = data[mask]
    mean_spectrum = np.mean(leaf_pixels, axis=0)
    std_spectrum = np.std(leaf_pixels, axis=0)
    wl = np.asarray(wavelengths)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(wl, mean_spectrum, color="darkblue", linewidth=1.5, label="Ortalama")
    ax.fill_between(wl, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum,
                    alpha=0.3, color="gray", label="± 1 std")
    ax.set_xlabel("Dalga boyu (nm)")
    ax.set_ylabel("Yansıma")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_one_leaf(
    folder_name: str, hdr_path: Path, out_root: Path,
    enabled_indices: Sequence[str] | None = None,
    seg_method: str = "hybrid",
) -> None:
    """Tek yaprak için tüm görselleri üret."""
    leaf_dir = out_root / folder_name
    leaf_dir.mkdir(parents=True, exist_ok=True)

    data, meta = load_envi(hdr_path)
    wavelengths = meta["wavelengths"]
    mask = best_mask(data, wavelengths, method=seg_method)
    rgb = make_rgb(data, wavelengths)

    plot_rgb(rgb, leaf_dir / f"{folder_name}_rgb.png", f"{folder_name} — RGB")
    plot_mask_overlay(rgb, mask, leaf_dir / f"{folder_name}_mask_overlay.png",
                      f"{folder_name} — maske")
    plot_spectral_profile(data, mask, wavelengths,
                          leaf_dir / f"{folder_name}_spectral_profile.png",
                          f"{folder_name} — ortalama spektrum")

    indices = calc_all_indices(data, wavelengths, enabled=enabled_indices)
    for name, arr in indices.items():
        plot_index_map(arr, mask, leaf_dir / f"{folder_name}_{name.lower()}_map.png",
                       f"{folder_name} — {name}")


def run(cfg, force: bool = False, max_leaves: int | None = None) -> None:
    """03_visualization aşaması — tüm yapraklar için görselleri üret."""
    t0 = time.time()
    out_dir = paths.stage_dir("03_visualization")
    data_dir = (paths.ROOT / cfg.get("data.raw_path")).resolve()
    seg_method = cfg.get("segmentation.method", "hybrid")
    enabled = cfg.get("indices.enabled")

    leaf_folders = find_leaf_folders(data_dir)
    if max_leaves is not None:
        leaf_folders = leaf_folders[:max_leaves]
    log.info("Görselleştirme başladı: %d yaprak", len(leaf_folders))

    success = 0
    for folder_name, hdr_path in leaf_folders:
        leaf_dir = out_dir / folder_name
        if not force and leaf_dir.exists() and any(leaf_dir.iterdir()):
            log.debug("cache var, atlanıyor: %s", folder_name)
            success += 1
            continue
        try:
            visualize_one_leaf(folder_name, hdr_path, out_dir,
                               enabled_indices=enabled, seg_method=seg_method)
            success += 1
        except Exception as exc:
            log.error("Görsel üretimi başarısız (%s): %s", folder_name, exc)

    paths.write_source_marker(out_dir, producer="src/m05_dataset/visualize.py",
                              config_source=cfg.source)
    log.info("03_visualization tamamlandı: %d/%d, süre=%.1fs",
             success, len(leaf_folders), time.time() - t0)
