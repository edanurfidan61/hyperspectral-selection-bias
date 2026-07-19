"""GÖREV 9 — GA dalga boyu önem görselleştirmesi.

Genetik algoritmanın seçtiği feature listesinden (``ga_best_features.txt``)
hangi spektral bantların ön plana çıktığını gösterir. Yaklaşım: ham seçim
histogramı yerine, sınıf-bazlı ortalama SNV spektrumu ÜZERİNE seçilen
dalga boylarını dikey vurgu olarak basıyoruz — "hangi dalga boyu, sınıflar
arasında neye karşılık geliyor" bir bakışta görünür (Sumak raporu stili).

Üretilen görseller (her GA klasörü için):
1. ``wavelength_spectrum.png`` — sınıf-bazlı SNV ortalama spektrumları +
   seçilen dalga boyları dikey vurgu. Spektral pencereler (VIS/Red-Edge/NIR)
   arka planda renkli.
2. ``top_bands_lollipop.png`` — en çok seçilen (model bazında: o modelin
   tüm seçimleri) ilk 30 dalga boyu lollipop grafiği.
3. ``feature_category_pie.png`` — seçilen feature'ların kategori dağılımı
   (snv / 1.türev / sabit bant / indeks / kontinyum / kenar).

Konsensüs (``run(cfg)``):
- ``_consensus_spectrum.png`` — sınıf spektrumları + dalga boyları renkleri
  KONSENSÜS derecesine göre (7/7 koyu kırmızı, 4/7 turuncu, 1-2/7 gri).
- ``_consensus_wavelengths.txt`` — okunaklı tablo (eski).
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.core import paths
from src.core.logging_setup import get as get_logger

log = get_logger("m04_features.ga_wavelength_viz")

# Stres sınıfı etiketleri (y_stress kodları için)
_STRESS_NAMES = {0: "Sağlıklı", 1: "Flavescence dorée (FD)",
                 2: "Diğer biyotik stres", 3: "Abiyotik / diğer"}
_STRESS_COLORS = {0: "#2ca02c", 1: "#d62728", 2: "#ff7f0e", 3: "#9467bd"}

# Feature ismi → dalga boyu çıkartma regex'leri
_RE_SNV = re.compile(r"^(snv|d1snv)_R(\d+)$")
_RE_FIXED_BAND = re.compile(r"^R(\d+)_(mean|std|p25|p75)$")
_RE_INDEX_PREFIX = re.compile(
    r"^(NDVI|GNDVI|ARI|CRI|PRI|ZTM|SIPI|FLAVI|RVSI|REP|BES|WBI|mARI|FRI|NDRE|NBI)",
)
_RE_CONT = re.compile(r"^cont_")
_RE_EDGE = re.compile(r"^d1_(red|blue|all)_")

# Spektral pencereler (nm) — birbirinden net ayrılan kontrastlı renkler
_REGIONS = [
    ("VIS",       400, 700, "#B2DFB0"),     # mint yeşil — görünür
    ("Red-Edge",  700, 740, "#E8B7E5"),     # leylak — kırmızı kenar
    ("NIR",       740, 1000, "#AED6F1"),    # açık mavi — yakın-IR
]


def _extract_wavelength(name: str) -> tuple[int | None, str]:
    """Feature isminden (wavelength_nm, category) çıkar.

    category ∈ {"snv", "d1snv", "fixed_band", "index", "continuum", "edge", "other"}.
    """
    m = _RE_SNV.match(name)
    if m:
        return int(m.group(2)), "snv" if m.group(1) == "snv" else "d1snv"
    m = _RE_FIXED_BAND.match(name)
    if m:
        return int(m.group(1)), "fixed_band"
    if _RE_INDEX_PREFIX.match(name):
        return None, "index"
    if _RE_CONT.match(name):
        return None, "continuum"
    if _RE_EDGE.match(name):
        return None, "edge"
    return None, "other"


def _read_selected_features(ga_dir: Path) -> list[str]:
    """``ga_best_features.txt`` dosyasından seçili feature isimlerini oku."""
    fpath = ga_dir / "ga_best_features.txt"
    if not fpath.exists():
        return []
    return [
        line.strip()
        for line in fpath.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _plot_wavelength_histogram(
    selected: list[str], out_path: Path, *, title: str,
) -> None:
    """SNV + d1snv + fixed_band bantlarının dalga boyu histogramını çiz."""
    cats: dict[str, list[int]] = {"snv": [], "d1snv": [], "fixed_band": []}
    for name in selected:
        wl, cat = _extract_wavelength(name)
        if wl is not None and cat in cats:
            cats[cat].append(wl)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    # Spektral pencere arka planı
    for label, lo, hi, color in _REGIONS:
        ax.axvspan(lo, hi, color=color, alpha=0.25, label=label)

    bins = np.arange(400, 1010, 10)
    colors = {"snv": "#1f77b4", "d1snv": "#d62728", "fixed_band": "#2ca02c"}
    labels = {"snv": "SNV spektrum", "d1snv": "1. türev", "fixed_band": "Sabit bant"}
    for key, wls in cats.items():
        if wls:
            ax.hist(wls, bins=bins, color=colors[key], alpha=0.7,
                    label=f"{labels[key]} (n={len(wls)})", edgecolor="white")

    ax.set_xlim(400, 1000)
    ax.set_xlabel("Dalga boyu (nm)")
    ax.set_ylabel("Seçim sayısı")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sınıf-bazlı spektrum + seçili bantlar (Sumak raporu stili)
# ---------------------------------------------------------------------------
def _load_class_spectra() -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, int]] | None:
    """``outputs/01_dataset``'ten sınıf-bazlı ortalama SNV spektrumu üret.

    Returns
    -------
    (wavelengths, class_mean_dict, class_count_dict)
        ``class_mean_dict[c]`` = (B,) ortalama SNV spektrumu
    """
    ds = paths.OUTPUTS_DIR / "01_dataset"
    try:
        X = np.load(ds / "X.npy")
        y = np.load(ds / "y_stress.npy")
        fn = json.loads((ds / "feature_names.json").read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Sınıf spektrumu yüklenemedi (%s); overlay devre dışı", exc)
        return None

    snv_idx, wls = [], []
    for i, name in enumerate(fn):
        m = re.match(r"^snv_R(\d+)$", name)
        if m:
            snv_idx.append(i)
            wls.append(int(m.group(1)))
    if not snv_idx:
        return None
    snv_idx = np.asarray(snv_idx)
    wavelengths = np.asarray(wls)
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    X_snv = X[:, snv_idx][:, order]

    class_mean: dict[int, np.ndarray] = {}
    class_count: dict[int, int] = {}
    for cls in np.unique(y):
        mask = y == cls
        if mask.sum() == 0:
            continue
        class_mean[int(cls)] = np.nanmean(X_snv[mask], axis=0)
        class_count[int(cls)] = int(mask.sum())
    return wavelengths, class_mean, class_count


def _draw_spectral_regions(ax) -> None:
    """Arka plana VIS/Red-Edge/NIR bant pencerelerini çiz."""
    for label, lo, hi, color in _REGIONS:
        ax.axvspan(lo, hi, color=color, alpha=0.18, label=label)


_CAT_COLORS = {"snv": "#1f77b4", "d1snv": "#d62728", "fixed_band": "#2ca02c"}
_CAT_LABELS = {"snv": "SNV spektrum", "d1snv": "1. türev", "fixed_band": "Sabit bant"}


def _bands_by_category(selected: list[str]) -> dict[str, list[int]]:
    """Seçilenleri (snv, d1snv, fixed_band) kategorilerine ayır."""
    out: dict[str, list[int]] = {"snv": [], "d1snv": [], "fixed_band": []}
    for name in selected:
        wl, cat = _extract_wavelength(name)
        if wl is not None and cat in out:
            out[cat].append(wl)
    return {k: sorted(v) for k, v in out.items()}


def _plot_class_spectra_only(
    out_path: Path, *,
    selected: list[str],
    spec_data: tuple[np.ndarray, dict[int, np.ndarray], dict[int, int]] | None,
    title: str,
) -> None:
    """GRAFİK A — Sadece sınıf-bazlı SNV ortalama spektrumları.

    Seçilen bantlar arka planda çok hafif dikey ışın olarak gösterilir
    (etiket YOK; sayısal değerler ayrı grafikte). Sumak raporu stili.
    """
    bands = _bands_by_category(selected)

    fig, ax = plt.subplots(1, 1, figsize=(13, 5.2))
    _draw_spectral_regions(ax)

    if spec_data is not None:
        wavelengths, class_mean, class_count = spec_data
        for cls in sorted(class_mean):
            label = f"{_STRESS_NAMES.get(cls, f'sınıf {cls}')} (n={class_count[cls]})"
            color = _STRESS_COLORS.get(cls, "gray")
            ax.plot(wavelengths, class_mean[cls], color=color,
                    linewidth=2.2, label=label, zorder=4)
        ax.set_ylabel("SNV-normalize ortalama yansıma", fontsize=10)
    else:
        ax.text(0.5, 0.5, "01_dataset/X.npy yüklenemedi",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, alpha=0.5)

    # GA seçili bantları arkada çok hafif vurgu (sadece bağlamı bozmasın)
    for cat, wls in bands.items():
        for wl in wls:
            ax.axvline(wl, color=_CAT_COLORS[cat], alpha=0.10,
                       linewidth=0.6, zorder=1)

    ax.set_xlim(400, 1000)
    ax.set_xlabel("Dalga boyu (nm)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.92)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_selected_bands_labeled(
    out_path: Path, *,
    selected: list[str],
    title: str,
) -> None:
    """GRAFİK B — Seçilen tüm dalga boyları yan-yana, her birinin sayısal etiketi.

    Lollipop: x ekseni dalga boyu (400-1000 nm), y ekseni kategori seviyesi.
    Her bantın hemen üstüne 90° çapraz olarak nm değeri yazılır.
    Spektral pencereler (VIS/Red-Edge/NIR) arka planda renklendirilir.
    """
    bands = _bands_by_category(selected)
    total = sum(len(v) for v in bands.values())
    if total == 0:
        return

    # Geniş figür: bant sayısı arttıkça okunabilirlik için y-yüksekliği biraz büyür.
    fig_h = max(4.5, 4.5 + min(2.0, total / 80))
    fig, ax = plt.subplots(1, 1, figsize=(15, fig_h))
    _draw_spectral_regions(ax)

    # Üç kategori için üç seviye — overlap olmasın
    levels = {"snv": 2, "d1snv": 1, "fixed_band": 0}
    for cat in ("snv", "d1snv", "fixed_band"):
        wls = bands[cat]
        if not wls:
            continue
        y = levels[cat]
        ax.vlines(wls, y, y + 0.55, colors=_CAT_COLORS[cat],
                  alpha=0.85, linewidth=1.4)
        ax.scatter(wls, [y + 0.55] * len(wls), s=22,
                   color=_CAT_COLORS[cat], zorder=5,
                   label=f"{_CAT_LABELS[cat]} (n={len(wls)})")
        # Her bant için sayısal etiket — 90° çapraz, lollipop tepesinin üstüne
        for wl in wls:
            ax.text(wl, y + 0.62, f"{wl}", rotation=90,
                    ha="center", va="bottom", fontsize=6.5,
                    color=_CAT_COLORS[cat])

    ax.set_xlim(400, 1000)
    ax.set_ylim(-0.25, 3.4)
    ax.set_yticks([0.4, 1.4, 2.4])
    ax.set_yticklabels(["Sabit bant", "1. türev", "SNV"], fontsize=9)
    ax.set_xlabel("Dalga boyu (nm)", fontsize=10)
    ax.set_title(f"{title}\n(toplam {total} dalga boyu)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=3, framealpha=0.92)
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_top_bands_lollipop(
    out_path: Path, *, selected: list[str], title: str, top_n: int = 30,
) -> None:
    """En çok seçilen ilk N dalga boyu lollipop grafiği.

    Tek model için: her dalga boyunun sayım=1 (zaten seçilmiş), kategorisine
    göre renklendirilmiş ve sıralanmış olarak gösterilir.
    """
    rows: list[tuple[int, str]] = []
    for name in selected:
        wl, cat = _extract_wavelength(name)
        if wl is not None and cat in ("snv", "d1snv", "fixed_band"):
            rows.append((wl, cat))
    if not rows:
        return
    rows.sort(key=lambda r: r[0])
    rows = rows[:top_n]
    cat_colors = {"snv": "#1f77b4", "d1snv": "#d62728", "fixed_band": "#2ca02c"}

    fig, ax = plt.subplots(1, 1, figsize=(11, max(3, 0.2 * len(rows) + 1.5)))
    y_pos = np.arange(len(rows))
    for i, (wl, cat) in enumerate(rows):
        ax.hlines(i, 0, wl, colors=cat_colors[cat], alpha=0.6, linewidth=1.2)
        ax.scatter(wl, i, s=60, color=cat_colors[cat], zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{wl} nm ({cat})" for wl, cat in rows], fontsize=8)
    ax.set_xlim(380, 1010)
    ax.set_xlabel("Dalga boyu (nm)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_category_pie(selected: list[str], out_path: Path, *, title: str) -> None:
    """Seçilen feature'ların kategori dağılımı (pie)."""
    cats = Counter()
    for name in selected:
        _, cat = _extract_wavelength(name)
        cats[cat] += 1
    if not cats:
        return
    labels = list(cats.keys())
    sizes = list(cats.values())
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.pie(sizes, labels=[f"{l}\n(n={s})" for l, s in zip(labels, sizes)],
           autopct="%.0f%%", startangle=90)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_for(ga_dir: Path) -> None:
    """Tek bir GA klasörü için spektrum-overlay + lollipop + pie üret.

    ga_feature_selection._save_outputs sonunda otomatik çağrılır.
    """
    ga_dir = Path(ga_dir)
    selected = _read_selected_features(ga_dir)
    if not selected:
        log.warning("GA wavelength viz atlandı (boş): %s", ga_dir.name)
        return
    title_base = ga_dir.name
    spec_data = _load_class_spectra()

    # GRAFİK A — Sadece sınıf spektrumları (üst üste)
    _plot_class_spectra_only(
        ga_dir / "spectra.png",
        selected=selected, spec_data=spec_data,
        title=f"Sınıf-bazlı SNV spektrumu — {title_base}",
    )
    # GRAFİK B — Seçilen tüm bantlar yan yana, sayısal etiketli
    _plot_selected_bands_labeled(
        ga_dir / "selected_bands.png",
        selected=selected,
        title=f"GA seçili dalga boyları — {title_base}",
    )
    # Kategori pie (indeks/spektrum/türev oranı)
    _plot_category_pie(
        selected, ga_dir / "feature_category_pie.png",
        title=f"GA feature kategori dağılımı — {title_base}",
    )

    # Eski format dosya adları varsa temizle
    for old_name in ("wavelength_importance.png", "wavelength_spectrum.png",
                     "top_bands_lollipop.png"):
        p = ga_dir / old_name
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass
    log.info("GA viz yazıldı: %s/(spectra + selected_bands + feature_category_pie).png",
             ga_dir.name)


def run(cfg=None) -> Path:
    """Tüm GA çalışmalarını tara + her birine viz üret + konsensüs plot.

    main.py'nin pipeline'ında 12_ga_feature_selection AŞAMASI sonrası
    çağrılır (ya da ga_feature_selection._save_outputs sonunda otomatik).
    """
    base = paths.OUTPUTS_DIR / "12_ga_feature_selection"
    if not base.exists():
        log.warning("12_ga_feature_selection dizini yok, viz atlandı")
        return base

    ga_dirs = [d for d in base.iterdir() if d.is_dir() and not d.name.startswith("_")]
    log.info("GA wavelength viz: %d GA çalışması bulundu", len(ga_dirs))

    # Her model için ayrı görsel
    all_selections: dict[str, list[str]] = {}
    for ga_dir in ga_dirs:
        plot_for(ga_dir)
        sel = _read_selected_features(ga_dir)
        if sel:
            all_selections[ga_dir.name] = sel

    if not all_selections:
        return base

    # ----- KONSENSÜS — tüm modellerin seçimlerinin birleşimi --------------
    # Her dalga boyu kaç modelde seçildi?
    wl_per_model: dict[int, set[str]] = {}
    for model_label, sel in all_selections.items():
        seen_wls: set[int] = set()
        for name in sel:
            wl, cat = _extract_wavelength(name)
            if wl is not None and cat in ("snv", "d1snv", "fixed_band"):
                seen_wls.add(wl)
        for wl in seen_wls:
            wl_per_model.setdefault(wl, set()).add(model_label)

    if not wl_per_model:
        return base

    wls_sorted = sorted(wl_per_model.keys())
    counts = [len(wl_per_model[wl]) for wl in wls_sorted]
    n_models = len(all_selections)
    spec_data = _load_class_spectra()

    # 1/7 gri → 7/7 koyu kırmızı renk skalası
    cmap = LinearSegmentedColormap.from_list(
        "consensus", ["#cccccc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"], N=256,
    )

    # ---- GRAFİK A — Sadece sınıf spektrumları (sade) -----------------------
    fig, ax = plt.subplots(1, 1, figsize=(13, 5.2))
    _draw_spectral_regions(ax)
    if spec_data is not None:
        wavelengths, class_mean, class_count = spec_data
        for cls in sorted(class_mean):
            label = f"{_STRESS_NAMES.get(cls, f'sınıf {cls}')} (n={class_count[cls]})"
            color = _STRESS_COLORS.get(cls, "gray")
            ax.plot(wavelengths, class_mean[cls], color=color,
                    linewidth=2.2, label=label, zorder=4)
        ax.set_ylabel("SNV-normalize ortalama yansıma", fontsize=10)
    # Güçlü konsensüs (>=60%) bantları arkada hafif kırmızı vurgu
    threshold_strong = max(2, int(round(n_models * 0.6)))
    for wl, n in zip(wls_sorted, counts):
        if n >= threshold_strong:
            ax.axvline(wl, color="#cc0000", alpha=0.10 + 0.35 * (n / n_models),
                       linewidth=0.6, zorder=1)
    ax.set_xlim(400, 1000)
    ax.set_xlabel("Dalga boyu (nm)", fontsize=10)
    ax.set_title(f"Sınıf-bazlı SNV spektrumu — {n_models} model konsensüsü",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.92)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(base / "_consensus_spectra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- GRAFİK B — Konsensüs lollipop, her bantın sayısal etiketi -------
    # Çok kalabalık olmasın diye SADECE konsensüs >= 2 olanları etiketle.
    fig_h = max(5.0, 5.0 + min(2.0, len(wls_sorted) / 100))
    fig, ax = plt.subplots(1, 1, figsize=(16, fig_h))
    _draw_spectral_regions(ax)
    for wl, n in zip(wls_sorted, counts):
        color = cmap(n / n_models)
        ax.vlines(wl, 0, n, colors=color, alpha=0.9, linewidth=2.0)
        ax.scatter(wl, n, s=28, color=color, zorder=5)
        # Sayısal etiket — sadece konsensüs >= 2 (kalabalık olmasın)
        if n >= 2:
            ax.text(wl, n + 0.18, f"{wl}", rotation=90,
                    ha="center", va="bottom", fontsize=6.5, color="#222")

    ax.set_xlim(400, 1000)
    ax.set_ylim(0, n_models + 1.2)
    ax.set_xlabel("Dalga boyu (nm)", fontsize=10)
    ax.set_ylabel(f"Bu bandı seçen model sayısı (max={n_models})", fontsize=10)
    ax.set_yticks(range(0, n_models + 1))
    ax.set_title(
        f"GA konsensüs — dalga boyu bazında seçim derecesi  ({n_models} model)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.25)

    # Renk skalası açıklaması
    from matplotlib.lines import Line2D
    legend_n = sorted({1, max(2, n_models // 3), max(2, n_models // 2), n_models})
    legend_handles = [
        Line2D([0], [0], color=cmap(n / n_models), lw=4, label=f"{n}/{n_models} model")
        for n in legend_n
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=9, framealpha=0.92, ncol=len(legend_n))

    plt.tight_layout()
    fig.savefig(base / "_consensus_bands.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    out_path = base / "_consensus_bands.png"

    # Konsensüs özet metni
    top_consensus = sorted(
        ((wl, len(wl_per_model[wl])) for wl in wl_per_model),
        key=lambda x: (-x[1], x[0]),
    )[:25]
    txt = [
        f"GA Konsensüs — {n_models} model",
        "=" * 50,
        f"Toplam ortak seçilen dalga boyu: {len(wls_sorted)}",
        "",
        "En çok seçilen 25 bant (wavelength_nm — kaç model):",
    ]
    for wl, n in top_consensus:
        bar = "█" * n
        txt.append(f"  {wl:>4} nm  {bar} ({n}/{n_models})")
    (base / "_consensus_wavelengths.txt").write_text(
        "\n".join(txt) + "\n", encoding="utf-8-sig",
    )

    log.info("Konsensüs viz: _consensus_spectra.png + _consensus_bands.png (+ .txt)")
    # Eski adlandırmadaki dosyalar varsa temizle — yeni format iki ayrı dosya.
    for old in ("_consensus_wavelengths.png", "_consensus_spectrum.png"):
        p = base / old
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass
    return base
