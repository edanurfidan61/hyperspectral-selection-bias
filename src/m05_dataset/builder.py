"""Yaprak seti üzerinde dataset oluşturma orkestrasyonu.

Eski projedeki 300 satırlık ``build_dataset`` fonksiyonu burada 4 alt fonksiyona
bölünmüştür:

1. :func:`find_leaf_folders` — Data dizinindeki yaprak klasörlerini ve .hdr yollarını listeler.
2. :func:`process_one_leaf`  — Tek bir yaprağı yükler, maskeler, özellik vektörü çıkarır.
3. :func:`assemble_features` — Tüm yaprakları işleyip X matrisi ve metadata toplar.
4. :func:`build`             — Yukarıdakileri + ground truth eşleştirme + kaydetme.

Pipeline çağrı şekli (``main.py`` veya ``scripts/run_pipeline.py``)::

    from src.m05_dataset.builder import build
    build(cfg, force=False)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core import paths
from src.core.logging_setup import get as get_logger
from src.m01_io.envi_loader import load_envi
from src.m02_preprocessing.segmentation import best_mask
from src.m04_features.extraction import extract_features, get_feature_names
from src.m05_dataset.ground_truth import (
    assign_stress_labels_from_ground_truth,
    load_ground_truth,
    match_ground_truth,
)

log = get_logger("m05_dataset.builder")


def find_leaf_folders(data_dir: str | Path, recursive: bool = True) -> list[tuple[str, Path]]:
    """Yaprak klasörlerini bul.

    Desteklenen yapıların çoğunu kapsamak için iki mod bulunur:
    - non-recursive (varsayılan eski davranış): ``<data_dir>/<leaf>/results/*.hdr``
    - recursive: ``**/results/*.hdr`` şeklinde tüm alt dizinleri ara ve
      her "leaf" klasörü için ilk bulunan .hdr dosyasını seç.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Veri seti dizini yok: {data_dir}")

    out: list[tuple[str, Path]] = []
    if recursive:
        # find all results/*.hdr anywhere under data_dir and pick one per leaf dir
        hdr_paths = sorted(data_dir.glob("**/results/*.hdr"))
        leaf_map: dict[Path, Path] = {}
        for hdr in hdr_paths:
            # leaf dir is parent of 'results' folder
            leaf_dir = hdr.parent.parent
            if leaf_dir in leaf_map:
                continue
            leaf_map[leaf_dir] = hdr

        for leaf_dir, hdr in sorted(leaf_map.items(), key=lambda x: str(x[0])):
            if not hdr.exists():
                log.warning("results/ içinde .hdr yok, atlanıyor: %s", leaf_dir.name)
                continue
            out.append((leaf_dir.name, hdr))
        log.info("Yaprak klasörü tarandı (recursive): %d adet", len(out))
        return out

    # fallback: old non-recursive behavior
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        results_dir = folder / "results"
        if not results_dir.is_dir():
            continue
        hdrs = sorted(results_dir.glob("*.hdr"))
        if not hdrs:
            log.warning("results/ içinde .hdr yok, atlanıyor: %s", folder.name)
            continue
        out.append((folder.name, hdrs[0]))
    log.info("Yaprak klasörü tarandı: %d adet", len(out))
    return out


def process_one_leaf(
    hdr_path: Path,
    seg_method: str = "hybrid",
    fallback_method: str = "ndvi",
    *,
    snv_enabled: bool = True,
    savgol_enabled: bool = True,
) -> Optional[tuple[np.ndarray, int, np.ndarray]]:
    """Bir yaprağı yükle, maskele, genişletilmiş özellik vektörü çıkar.

    GÖREV 6 — ``snv_enabled``/``savgol_enabled`` ablation için
    ``extract_features``'a aktarılır.

    Returns
    -------
    (feature_vector, leaf_pixel_count, wavelengths) | None
    """
    data, meta = load_envi(hdr_path)
    wavelengths = np.asarray(meta["wavelengths"], dtype=np.float64)

    mask = best_mask(data, wavelengths, method=seg_method)
    if int(np.sum(mask)) == 0 and seg_method != fallback_method:
        log.debug("%s ile maske boş, %s deneniyor", seg_method, fallback_method)
        mask = best_mask(data, wavelengths, method=fallback_method)
    leaf_count = int(np.sum(mask))
    if leaf_count == 0:
        return None

    masked_cube = data.astype(np.float64, copy=True)
    masked_cube[~mask] = np.nan
    feat = extract_features(
        masked_cube, wavelengths,
        snv_enabled=snv_enabled, savgol_enabled=savgol_enabled,
    )
    return feat, leaf_count, wavelengths


def assemble_features(
    leaf_folders: list[tuple[str, Path]],
    seg_method: str = "hybrid",
    *,
    snv_enabled: bool = True,
    savgol_enabled: bool = True,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray | None]:
    """Tüm yaprakları işle, X matrisi + folder isim listesi + atlananlar + wavelengths.

    GÖREV 6 — ``snv_enabled``/``savgol_enabled`` her yaprağa aktarılır
    (PREPROCESSING ablation için).
    """
    rows: list[np.ndarray] = []
    kept: list[str] = []
    skipped: list[str] = []
    wavelengths: np.ndarray | None = None
    expected_dim: int | None = None

    for folder_name, hdr_path in tqdm(leaf_folders, desc="Yapraklar", unit="yaprak"):
        try:
            result = process_one_leaf(
                hdr_path, seg_method=seg_method,
                snv_enabled=snv_enabled, savgol_enabled=savgol_enabled,
            )
        except Exception as exc:
            log.error("HATA: %s işlenemedi → %s", folder_name, exc)
            skipped.append(folder_name)
            continue

        if result is None:
            log.warning("Maske boş, atlanıyor: %s", folder_name)
            skipped.append(folder_name)
            continue

        feat, leaf_count, wls = result
        if wavelengths is None:
            wavelengths = wls
            expected_dim = feat.shape[0]
        elif feat.shape[0] != expected_dim:
            log.warning(
                "%s: beklenen %d özellik, alınan %d — atlanıyor",
                folder_name, expected_dim, feat.shape[0],
            )
            skipped.append(folder_name)
            continue

        rows.append(feat)
        kept.append(folder_name)

    X = (
        np.asarray(rows, dtype=np.float64)
        if rows else np.empty((0, expected_dim or 0))
    )
    log.info("Özellik çıkarımı: %d başarılı, %d atlandı, X=%s",
             len(kept), len(skipped), X.shape)
    return X, kept, skipped, wavelengths


def _build_groups(
    folder_names: list[str], gt_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Her örnek için group_leaf ve group_plot anahtarlarını üret.

    - ``group_leaf``: yaprak başına benzersiz id (folder_name). Bir yaprağın
      ileride birden fazla satır vermesi durumunda (ör. per-piksel) leakage
      önlenir; mevcut yapıda satır=yaprak olduğu için random split'e eşdeğer.
    - ``group_plot``: ``variety|plotLocation|collectionDate``. Aynı bağ ve
      tarihten gelen yaprakların hep aynı tarafta (train ya da test) kalmasını
      garantiler — daha sıkı leakage koruması.
    """
    gt = gt_df.copy()
    if "filename_lower" not in gt.columns:
        gt["filename_lower"] = (
            gt["filename"].astype(str).str.strip().str.lower()
        )

    groups_leaf = np.asarray(folder_names, dtype=object)
    groups_plot = np.full(len(folder_names), "unknown", dtype=object)

    for i, folder in enumerate(folder_names):
        key = str(folder).strip().lower()
        rows = gt[gt["filename_lower"] == key]
        if rows.empty:
            rows = gt[gt["filename_lower"].str.contains(key, na=False, regex=False)]
        if rows.empty:
            continue
        r = rows.iloc[0]
        variety = str(r.get("variety", "")).strip() or "unknown"
        plot = str(r.get("plotLocation", "")).strip() or "unknown"
        date = str(r.get("collectionDate", "")).strip() or "unknown"
        groups_plot[i] = f"{variety}|{plot}|{date}"

    n_leaf = len(set(groups_leaf.tolist()))
    n_plot = len(set(groups_plot.tolist()))
    log.info("Groups: leaf=%d unique, plot=%d unique (%d satır)",
             n_leaf, n_plot, len(folder_names))
    return groups_leaf, groups_plot


def _save_outputs(
    out_dir: Path,
    X: np.ndarray,
    y_chl: np.ndarray,
    y_flav: np.ndarray,
    y_nbi: np.ndarray,
    y_stress: np.ndarray,
    folder_names: list[str],
    varieties: list[str],
    feature_names: list[str],
    groups_leaf: np.ndarray | None = None,
    groups_plot: np.ndarray | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y_chl.npy", y_chl)
    np.save(out_dir / "y_flav.npy", y_flav)
    np.save(out_dir / "y_nbi.npy", y_nbi)
    np.save(out_dir / "y_stress.npy", y_stress)
    if groups_leaf is not None:
        np.save(out_dir / "groups_leaf.npy", np.asarray(groups_leaf, dtype=object))
    if groups_plot is not None:
        np.save(out_dir / "groups_plot.npy", np.asarray(groups_plot, dtype=object))
    (out_dir / "feature_names.json").write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    df = pd.DataFrame(X, columns=feature_names)
    df.insert(0, "filename", folder_names)
    df.insert(1, "variety", varieties)
    df.insert(2, "Chl", y_chl)
    df.insert(3, "Flav", y_flav)
    df.insert(4, "NBI", y_nbi)
    df.insert(5, "stress_label", y_stress)
    if groups_leaf is not None:
        df.insert(6, "group_leaf", groups_leaf)
    if groups_plot is not None:
        df.insert(7, "group_plot", groups_plot)
    df.to_csv(out_dir / "dataset_full.csv", index=False, encoding="utf-8")
    log.info("Dataset kaydedildi: %s", out_dir)


def _outputs_already_exist(out_dir: Path) -> bool:
    return all((out_dir / f).exists() for f in ("X.npy", "y_chl.npy", "y_flav.npy",
                                                "y_nbi.npy", "y_stress.npy",
                                                "feature_names.json"))


def build(cfg, force: bool = False) -> dict[str, np.ndarray]:
    """Pipeline'ın 01_dataset aşaması — ana orkestrasyon.

    Çıktılar ``outputs/01_dataset/`` altına yazılır. ``force=False`` iken
    çıktılar varsa yeniden hesaplama atlanır.

    Returns
    -------
    dict
        ``{"X", "y_chl", "y_flav", "y_nbi", "y_stress", "folder_names",
        "varieties", "feature_names"}``
    """
    t0 = time.time()
    out_dir = paths.stage_dir("01_dataset")

    if not force and _outputs_already_exist(out_dir):
        log.info("01_dataset cache mevcut, force=False → yükleniyor")
        return load(out_dir)

    data_dir = (paths.ROOT / cfg.get("data.raw_path")).resolve()
    tab_path = (paths.ROOT / cfg.get("data.ground_truth")).resolve()
    seg_method = cfg.get("segmentation.method", "hybrid")
    # GÖREV 6: ablation flag'leri — config'ten preprocessing aç/kapa
    snv_enabled = bool(cfg.get("preprocessing.snv.enabled", True))
    savgol_enabled = int(cfg.get("preprocessing.savgol.window_length", 11)) > 1

    log.info("Dataset oluşturma başladı | data=%s | gt=%s | seg=%s | snv=%s sg=%s",
             data_dir, tab_path, seg_method, snv_enabled, savgol_enabled)

    gt_df = load_ground_truth(tab_path)
    leaf_folders = find_leaf_folders(data_dir)
    if not leaf_folders:
        raise RuntimeError(f"Yaprak klasörü bulunamadı: {data_dir}")

    X, folder_names, _skipped, wavelengths = assemble_features(
        leaf_folders, seg_method=seg_method,
        snv_enabled=snv_enabled, savgol_enabled=savgol_enabled,
    )
    if X.shape[0] == 0:
        raise RuntimeError("Hiçbir yaprak başarıyla işlenemedi.")

    y_chl, y_flav, y_nbi, varieties, _unmatched = match_ground_truth(folder_names, gt_df)

    # Try to assign stress labels from explicit 'symptom' column; if missing
    # classes appear (ValueError), fall back to flavonol-based labeling when
    # possible. This avoids failing the entire pipeline when GT mapping is
    # imperfect.
    try:
        y_stress = assign_stress_labels_from_ground_truth(gt_df, folder_names)
    except ValueError as exc:
        log.warning("Stres etiketlemede hata: %s — fallback uygulanıyor (flavonol).", exc)
        from src.m05_dataset.ground_truth import assign_stress_labels_from_flavonol

        if np.any(np.isfinite(y_flav)):
            y_stress = assign_stress_labels_from_flavonol(y_flav)
        else:
            # As a last resort, mark everything as 'unknown' (class 3)
            log.warning("Flavonol verisi yok; tüm örnekler şiddetli (3) olarak etiketleniyor")
            y_stress = np.full(len(folder_names), 3, dtype=int)
    feature_names = get_feature_names(wavelengths)

    # Group split için yaprak/bağ-tarih grup anahtarları
    groups_leaf, groups_plot = _build_groups(folder_names, gt_df)

    _save_outputs(out_dir, X, y_chl, y_flav, y_nbi, y_stress,
                  folder_names, varieties, feature_names,
                  groups_leaf=groups_leaf, groups_plot=groups_plot)
    paths.write_source_marker(out_dir, producer="src/m05_dataset/builder.py",
                              config_source=cfg.source)

    log.info("01_dataset tamamlandı: X=%s, süre=%.1fs", X.shape, time.time() - t0)
    return {
        "X": X, "y_chl": y_chl, "y_flav": y_flav, "y_nbi": y_nbi, "y_stress": y_stress,
        "folder_names": folder_names, "varieties": varieties, "feature_names": feature_names,
        "groups_leaf": groups_leaf, "groups_plot": groups_plot,
    }


def load(out_dir: Path | None = None) -> dict[str, np.ndarray]:
    """``outputs/01_dataset/`` altından kaydedilmiş dataset'i yükle."""
    out_dir = Path(out_dir) if out_dir else paths.stage_dir("01_dataset")
    X = np.load(out_dir / "X.npy")
    y_chl = np.load(out_dir / "y_chl.npy")
    y_flav = np.load(out_dir / "y_flav.npy")
    y_nbi = np.load(out_dir / "y_nbi.npy")
    y_stress = np.load(out_dir / "y_stress.npy")
    feature_names = json.loads((out_dir / "feature_names.json").read_text(encoding="utf-8"))
    csv_path = out_dir / "dataset_full.csv"
    folder_names: list[str] = []
    varieties: list[str] = []
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        folder_names = df["filename"].astype(str).tolist()
        varieties = df.get("variety", pd.Series([""] * len(df))).astype(str).tolist()
    groups_leaf = (
        np.load(out_dir / "groups_leaf.npy", allow_pickle=True)
        if (out_dir / "groups_leaf.npy").exists() else None
    )
    groups_plot = (
        np.load(out_dir / "groups_plot.npy", allow_pickle=True)
        if (out_dir / "groups_plot.npy").exists() else None
    )
    return {
        "X": X, "y_chl": y_chl, "y_flav": y_flav, "y_nbi": y_nbi, "y_stress": y_stress,
        "folder_names": folder_names, "varieties": varieties, "feature_names": feature_names,
        "groups_leaf": groups_leaf, "groups_plot": groups_plot,
    }
