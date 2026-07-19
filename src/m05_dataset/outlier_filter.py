"""Dataset düzeyinde outlier tespiti ve filtreleme.

İki yöntem aynı anda uygulanır ve birleşik bir maske üretilir:

1. **IsolationForest** — X özellik uzayında çok-değişkenli outlier (yapısal
   anomali). Hiperspektral feature'lar yüksek korelasyonlu ve yüksek boyutlu;
   IF bu durumda Mahalanobis'ten daha gürbüzdür.
2. **IQR** — Her regresyon hedefi (Chl/Flav/NBI) üzerinde tek-değişkenli
   uç-değer kontrolü (Q1 - 1.5*IQR, Q3 + 1.5*IQR dışı). Etiket gürültüsü ve
   lab-ölçüm hatalarını yakalar.

Çıktı: ``outputs/01b_outliers/``
    - outlier_mask.npy        (n,) bool — True = TUT, False = at
    - outlier_report.csv      sample bazında detay (yöntem, skor, atılma sebebi)
    - outlier_summary.txt     türkçe özet

Davranış (config: ``outliers.action``):
    - ``"warn"``  → sadece raporla, dataset'i değiştirme (varsayılan, güvenli)
    - ``"drop"``  → X_clean.npy / y_*_clean.npy / groups_*_clean.npy kaydet
    - ``"none"``  → tamamen atla (tespit yapma)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.core import paths
from src.core.logging_setup import get as get_logger

log = get_logger("m05_dataset.outlier_filter")

_TARGETS = (("Chl", "y_chl.npy"), ("Flav", "y_flav.npy"), ("NBI", "y_nbi.npy"))


def _iqr_outliers(values: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Tek-değişkenli IQR outlier maskesi (True = outlier)."""
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=bool)
    q1, q3 = np.percentile(values[finite], [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    out = np.zeros_like(values, dtype=bool)
    out[finite] = (values[finite] < lo) | (values[finite] > hi)
    return out


def _isolation_forest_outliers(
    X: np.ndarray, contamination: float, random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """IsolationForest ile multivariate outlier; (mask, anomaly_score) döndür.

    ``mask`` True = outlier (sklearn -1), score düşükse anomali.
    """
    if X.shape[0] < 10:
        log.warning("IsolationForest için n=%d çok küçük; atlanıyor", X.shape[0])
        return np.zeros(X.shape[0], dtype=bool), np.zeros(X.shape[0])
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    pred = iso.fit_predict(X)  # -1 = outlier, 1 = inlier
    score = iso.score_samples(X)  # büyük = normal
    return pred == -1, score


def run(cfg=None, force: bool = False) -> Path:
    """01b_outliers aşamasını çalıştır."""
    action = "warn"
    contamination = 0.05
    iqr_factor = 1.5
    seed = 42
    if cfg is not None:
        action = str(cfg.get("outliers.action", action)).lower()
        contamination = float(cfg.get("outliers.contamination", contamination))
        iqr_factor = float(cfg.get("outliers.iqr_factor", iqr_factor))
        seed = int(cfg.get("models.random_state", seed))

    out_dir = paths.stage_dir("01b_outliers")

    if action == "none":
        log.info("outliers.action=none → outlier tespiti atlanıyor")
        return out_dir

    ds_dir = paths.OUTPUTS_DIR / "01_dataset"
    if not (ds_dir / "X.npy").exists():
        raise FileNotFoundError(
            f"01_dataset çıktıları yok: {ds_dir}. Önce dataset aşamasını çalıştır."
        )

    X = np.load(ds_dir / "X.npy")
    n = X.shape[0]
    log.info("Outlier tespiti: n=%d, action=%s, contamination=%.3f, iqr=%.2f",
             n, action, contamination, iqr_factor)

    # 1) IsolationForest (multivariate) — NaN'ı 0'a çevir, IF buna toleranslı
    X_safe = np.where(np.isfinite(X), X, 0.0)
    iso_mask, iso_score = _isolation_forest_outliers(X_safe, contamination, seed)
    log.info("IsolationForest: %d outlier (%.1f%%)",
             int(iso_mask.sum()), 100 * iso_mask.mean())

    # 2) IQR (her hedef için ayrı)
    iqr_masks: dict[str, np.ndarray] = {}
    for col_name, fname in _TARGETS:
        path = ds_dir / fname
        if not path.exists():
            continue
        y = np.load(path)
        m = _iqr_outliers(y, factor=iqr_factor)
        iqr_masks[col_name] = m
        log.info("IQR %s: %d outlier (%.1f%%)",
                 col_name, int(m.sum()), 100 * m.mean())

    # 3) Birleşik outlier maskesi (TRUE = outlier)
    combined = iso_mask.copy()
    for m in iqr_masks.values():
        combined |= m
    keep_mask = ~combined  # TRUE = TUT
    n_drop = int(combined.sum())
    log.info("Birleşik outlier: %d/%d (%.1f%%) → tutulan %d",
             n_drop, n, 100 * n_drop / max(n, 1), int(keep_mask.sum()))

    # ---- Rapor ----
    folder_names: list[str] = []
    csv_path = ds_dir / "dataset_full.csv"
    if csv_path.exists():
        folder_names = pd.read_csv(csv_path, usecols=["filename"])["filename"].astype(str).tolist()
    if not folder_names:
        folder_names = [f"sample_{i}" for i in range(n)]

    rows = []
    for i in range(n):
        reasons = []
        if iso_mask[i]:
            reasons.append("isolation_forest")
        for col, m in iqr_masks.items():
            if m[i]:
                reasons.append(f"iqr_{col.lower()}")
        rows.append({
            "index": i,
            "filename": folder_names[i] if i < len(folder_names) else f"sample_{i}",
            "is_outlier": bool(combined[i]),
            "iso_score": float(iso_score[i]),
            "reason": ",".join(reasons) if reasons else "",
        })
    report_df = pd.DataFrame(rows)
    report_df.to_csv(out_dir / "outlier_report.csv", index=False, encoding="utf-8")

    np.save(out_dir / "outlier_mask.npy", keep_mask)

    # ---- Türkçe özet ----
    summary = [
        "Outlier Tespit Özeti",
        "=" * 50,
        f"Toplam örnek            : {n}",
        f"IsolationForest (cont={contamination:.3f}): {int(iso_mask.sum())} outlier",
    ]
    for col, m in iqr_masks.items():
        summary.append(f"IQR {col:<5} (k={iqr_factor})           : {int(m.sum())} outlier")
    summary += [
        "-" * 50,
        f"Birleşik outlier        : {n_drop} (%.1f%%)" % (100 * n_drop / max(n, 1)),
        f"Korunan (clean) örnek   : {int(keep_mask.sum())}",
        f"action                  : {action}",
        "",
        "outlier_mask.npy: True=tut, False=at",
    ]
    if action == "drop" and n_drop > 0:
        summary.append("→ Filtrelenmiş X_clean / y_*_clean / groups_*_clean kaydedildi.")
    (out_dir / "outlier_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8-sig",
    )

    # ---- action=drop → filtreli arrays kaydet ----
    if action == "drop" and n_drop > 0:
        clean_dir = ds_dir  # aynı klasöre _clean.npy olarak yaz
        np.save(clean_dir / "X_clean.npy", X[keep_mask])
        for _, fname in _TARGETS:
            p = ds_dir / fname
            if p.exists():
                arr = np.load(p)
                np.save(clean_dir / fname.replace(".npy", "_clean.npy"), arr[keep_mask])
        for gf in ("groups_leaf.npy", "groups_plot.npy", "y_stress.npy"):
            p = ds_dir / gf
            if p.exists():
                arr = np.load(p, allow_pickle=True)
                np.save(clean_dir / gf.replace(".npy", "_clean.npy"), arr[keep_mask])
        log.info("Clean dataset kaydedildi: %s/*_clean.npy", clean_dir)

    paths.write_source_marker(
        out_dir,
        producer="src/m05_dataset/outlier_filter.py",
        config_source=cfg.source if cfg is not None else None,
    )
    log.info("Outlier raporu yazıldı: %s", out_dir)
    return out_dir
