"""Veri seti görselleştirme — REGISTRY'deki tüm setleri otomatik kapsar.

Tamamen ``HSIDataset`` arayüzü üzerinden çalışır (özel-durum yok), böylece
``dataset_registry.REGISTRY``'ye eklenen her yeni veri seti otomatik kapsanır.
Seçim-yanlılığı makalesi için her veri setinin betimsel görsellerini üretir:

    - feature_profile.png    : öznitelik ekseni boyunca ortalama ± std spektrum/profil
    - target_distributions.png : her hedefin histogramı
    - group_structure.png    : grup başına örnek sayısı (CV grup yapısı)

Ayrıca tüm setler için tek bir ``dataset_overview.csv`` özet tablosu yazılır.

Çalıştırma:
    python -m src.m05_dataset.dataset_viz
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from src.core import paths
from src.core.logging_setup import get as get_logger
from src.m01_io import dataset_registry as registry
from src.m01_io.dataset_registry import HSIDataset

log = get_logger("m05_dataset.dataset_viz")


def _stage_root() -> Path:
    return paths.stage_dir("01d_dataset_viz")


# ---------------------------------------------------------------------------
# Tekil figürler (hepsi yalnızca HSIDataset kullanır)
# ---------------------------------------------------------------------------
def plot_feature_profile(ds: HSIDataset, out_dir: Path) -> Path:
    """Öznitelik ekseni boyunca ortalama ± std.

    deep_patato gibi bant-temelli setlerde gerçek bir spektrumdur; karışık
    öznitelikli setlerde (ryckewaert) "öznitelik profili" olarak okunur.
    x ekseni: varsa dalga boyları, yoksa öznitelik indeksi.
    """
    X = ds.X
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    x = ds.wavelengths if ds.wavelengths is not None and len(ds.wavelengths) == X.shape[1] \
        else np.arange(X.shape[1])
    xlabel = "Dalga boyu (nm)" if ds.wavelengths is not None else "Öznitelik indeksi"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, mean, color="steelblue", lw=1.5, label="ortalama")
    ax.fill_between(x, mean - std, mean + std, alpha=0.25, color="steelblue",
                    label="±1 std")
    ax.set_xlabel(xlabel); ax.set_ylabel("Değer")
    ax.set_title(f"{ds.name} — öznitelik profili (n={X.shape[0]}, p={X.shape[1]})",
                 fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    p = out_dir / "feature_profile.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def plot_target_distributions(ds: HSIDataset, out_dir: Path) -> Path:
    """Her hedef için histogram (geçerli/sonlu değerler)."""
    tnames = list(ds.targets)
    n = len(tnames)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.2 * nrow),
                             squeeze=False)
    for i, tname in enumerate(tnames):
        ax = axes[i // ncol][i % ncol]
        v = np.asarray(ds.targets[tname], dtype=float)
        v = v[np.isfinite(v)]
        ax.hist(v, bins=min(30, max(5, len(v) // 5)), color="indianred", alpha=0.8)
        ax.set_title(f"{tname} (n={len(v)}, μ={v.mean():.2f})", fontsize=10)
        ax.grid(True, alpha=0.3)
    # boş eksenleri gizle
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{ds.name} — hedef dağılımları", fontweight="bold")
    plt.tight_layout()
    p = out_dir / "target_distributions.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def plot_group_structure(ds: HSIDataset, out_dir: Path) -> Path:
    """Grup başına örnek sayısı — CV için grup dengesini gösterir."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if ds.groups is None:
        ax.text(0.5, 0.5, "Grup tanımlı değil (groups=None)",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        vals, counts = np.unique(np.asarray(ds.groups), return_counts=True)
        order = np.argsort(counts)[::-1]
        counts = counts[order]
        ax.bar(np.arange(len(counts)), counts, color="seagreen")
        ax.set_xlabel(f"Grup ({len(vals)} benzersiz)")
        ax.set_ylabel("Örnek sayısı")
        ax.set_title(f"{ds.name} — grup yapısı "
                     f"(ort {counts.mean():.1f}, max {counts.max()}, min {counts.min()})",
                     fontweight="bold")
    plt.tight_layout()
    p = out_dir / "group_structure.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Orkestratör
# ---------------------------------------------------------------------------
def visualize_dataset(ds: HSIDataset, out_root: Path | None = None) -> dict[str, Path]:
    """Tek bir HSIDataset için tüm figürleri üret."""
    out_root = out_root or _stage_root()
    out_dir = out_root / ds.name
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = {
        "feature_profile": plot_feature_profile(ds, out_dir),
        "target_distributions": plot_target_distributions(ds, out_dir),
        "group_structure": plot_group_structure(ds, out_dir),
    }
    log.info("%s görselleştirildi → %s", ds.name, out_dir)
    return produced


def _overview_row(ds: HSIDataset) -> dict:
    n_groups = None if ds.groups is None else len(np.unique(np.asarray(ds.groups)))
    return {
        "dataset": ds.name,
        "n_samples": ds.X.shape[0],
        "n_features": ds.X.shape[1],
        "n_groups": n_groups,
        "targets": ";".join(ds.targets),
        "has_wavelengths": ds.wavelengths is not None,
    }


def run(cfg=None, datasets: list[str] | None = None) -> Path:
    """Pipeline aşaması: REGISTRY'deki (veya verilen) tüm setleri görselleştir.

    Yüklenemeyen setler (ham verisi yoksa) atlanır; üretilen overview tablosu
    yalnızca başarıyla yüklenenleri içerir.
    """
    out_root = _stage_root()
    names = datasets or list(registry.REGISTRY)
    rows = []
    for name in names:
        try:
            ds = registry.load(name)
        except Exception as exc:
            log.warning("%s yüklenemedi, atlanıyor: %s", name, exc)
            continue
        visualize_dataset(ds, out_root)
        rows.append(_overview_row(ds))

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_root / "dataset_overview.csv", index=False, encoding="utf-8")
        log.info("dataset_overview.csv yazıldı (%d set) → %s", len(rows), out_root)
    return out_root


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s | %(message)s",
                         datefmt="%H:%M:%S")
    out = run()
    print("Çıktı kökü:", out)
