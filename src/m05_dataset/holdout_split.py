"""Train/Val/Test üçlü split — held-out test seti üretici aşaması.

Mevcut pipeline her aşamada k-fold CV kullanıyor; bu k-fold'lar **aynı veri**
üzerinde çalıştığı için "final" performansı hiperparametre seçiminden
ayrıştırmıyor. Bu aşama dataset'in **%test_size kadarını tek seferlik kilitler**:

    full_dataset
      ├── trainval (CV burada yapılır — model seçimi, GA/RFE/SHAP, vb.)
      └── test     (yalnızca son raporlama için, hiçbir aşama bunu görmez)

Group/stratified-aware:
    - groups varsa  → GroupShuffleSplit (aynı yaprak/bağ asla iki tarafta olmaz)
    - sınıflandırma → StratifiedShuffleSplit (sınıf dağılımı korunur)
    - regresyon     → quantile-bin tabanlı stratified split (continuous binning)

Çıktılar (``outputs/01c_holdout/``):
    - holdout_indices.json   {"trainval": [...], "test": [...], "config": {...}}
    - holdout_indices.npz    np kaydı (yükleme hızı için)
    - holdout_summary.txt    Türkçe özet (boyutlar, sınıf/grup dağılımı)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sklearn.model_selection import StratifiedGroupKFold

from src.core import paths
from src.core.cv import bin_continuous, load_groups, make_holdout_split
from src.core.logging_setup import get as get_logger

log = get_logger("m05_dataset.holdout_split")


# ---------------------------------------------------------------------------
# GÖREV 4: Stratified + Grouped holdout (StratifiedGroupKFold tek-fold sim.)
#         + 5 seed araması (KL divergence + grup overlap puanlaması)
# ---------------------------------------------------------------------------
def _kl_divergence(train_y: np.ndarray, test_y: np.ndarray, eps: float = 1e-9) -> float:
    """KL(P_train || P_test) — sınıf dağılımları arasında bilgi kazancı.

    Küçük değer = train ve test sınıf dağılımları yakın (iyi).
    Laplace ε ile sıfır olasılıkları sıkıştırılır (log(0) yasaklı).
    """
    labels = np.unique(np.concatenate([train_y, test_y]))
    p = np.array([(train_y == c).sum() for c in labels], dtype=np.float64)
    q = np.array([(test_y == c).sum() for c in labels], dtype=np.float64)
    p = (p + eps) / (p.sum() + eps * len(labels))
    q = (q + eps) / (q.sum() + eps * len(labels))
    return float(np.sum(p * np.log(p / q)))


def _group_leakage(
    groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
) -> int:
    """Train ve test arasında ortak grup sayısı (0 = ideal)."""
    return len(set(groups[train_idx].tolist()) & set(groups[test_idx].tolist()))


def _stratified_grouped_holdout(
    y_strat: np.ndarray,
    groups: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """``StratifiedGroupKFold`` ile tek-fold holdout simülasyonu.

    ``test_size``'a en yakın test oranı veren fold'u seçer. Bu, hem grup
    leakage'ını engeller (aynı yaprak iki tarafta olmaz) hem de sınıf
    dağılımını trainval/test arasında benzer tutar.

    Returns
    -------
    train_idx, test_idx, achieved_test_size
    """
    n_splits = max(3, min(10, int(round(1.0 / max(test_size, 1e-3)))))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n = len(y_strat)
    best = None
    best_diff = float("inf")
    for tr, te in skf.split(np.zeros((n, 1)), y_strat, groups=groups):
        frac = len(te) / n
        diff = abs(frac - test_size)
        if diff < best_diff:
            best, best_diff = (np.asarray(tr), np.asarray(te), frac), diff
    assert best is not None
    return best


def _search_best_seed(
    y_strat: np.ndarray,
    groups: np.ndarray,
    *,
    test_size: float,
    seeds: list[int],
) -> tuple[int, list[dict]]:
    """Birden çok seed dener, KL ve grup overlap'ı raporlar; en iyiyi seçer.

    Seçim kuralı: grup overlap = 0 olanlar arasında en küçük KL'li seed.
    Hiçbiri grup-temiz değilse en küçük (KL + 100·overlap) seçilir.
    """
    rows: list[dict] = []
    for s in seeds:
        tr, te, frac = _stratified_grouped_holdout(
            y_strat, groups, test_size=test_size, random_state=s,
        )
        kl = _kl_divergence(y_strat[tr], y_strat[te])
        leak = _group_leakage(groups, tr, te)
        rows.append({
            "seed": s, "kl": kl, "group_overlap": leak,
            "test_frac": frac, "train_n": int(tr.size), "test_n": int(te.size),
        })
    clean = [r for r in rows if r["group_overlap"] == 0]
    pool = clean if clean else rows
    best = min(pool, key=lambda r: r["kl"] + 100.0 * r["group_overlap"])
    return best["seed"], rows


def _stratify_target_for_split(
    y: np.ndarray, task: str, n_bins: int = 5,
) -> np.ndarray | None:
    """Stratification için kategorik bir array döndür (regresyonda binle)."""
    if task == "classification":
        return y.astype(np.int64, copy=False)
    binned = bin_continuous(y, n_bins=n_bins, strategy="quantile")
    return binned if (binned >= 0).all() else None


def run(cfg=None, force: bool = False) -> Path:
    """01c_holdout aşaması — train+val / test indekslerini bir kez kilitle."""
    test_size = 0.20
    seed = 42
    group_key = "leaf"
    stratify_target = "stress"   # "stress" | "flavonol" | "chlorophyll" | "nbi" | "none"
    n_bins = 5

    if cfg is not None:
        test_size = float(cfg.get("models.test_size", test_size))
        seed = int(cfg.get("models.random_state", seed))
        group_key = str(cfg.get("cv.group_key", group_key))
        stratify_target = str(cfg.get("holdout.stratify_target", stratify_target))
        n_bins = int(cfg.get("models.regression_n_bins", n_bins))

    out_dir = paths.stage_dir("01c_holdout")

    ds_dir = paths.OUTPUTS_DIR / "01_dataset"
    if not (ds_dir / "X.npy").exists():
        raise FileNotFoundError(
            f"01_dataset çıktıları yok: {ds_dir}. Önce dataset aşamasını çalıştır."
        )

    # Cache: force=False ise mevcut split'i tekrar üretme
    idx_path = out_dir / "holdout_indices.npz"
    if idx_path.exists() and not force:
        log.info("01c_holdout cache mevcut, force=False → atlanıyor")
        return out_dir

    X = np.load(ds_dir / "X.npy")
    n = X.shape[0]
    groups = load_groups(ds_dir, key=group_key)

    # Stratification hedefi
    strat = None
    task_for_split = "regression"
    target_path_map = {
        "stress": "y_stress.npy",
        "flavonol": "y_flav.npy",
        "chlorophyll": "y_chl.npy",
        "nbi": "y_nbi.npy",
    }
    if stratify_target in target_path_map:
        ypath = ds_dir / target_path_map[stratify_target]
        if ypath.exists():
            y_strat = np.load(ypath)
            if stratify_target == "stress":
                task_for_split = "classification"
                strat = y_strat
            else:
                # Regresyon hedefini binle → stratification için kategorik
                task_for_split = "classification"
                strat = _stratify_target_for_split(y_strat, "regression", n_bins=n_bins)
                if strat is None:
                    log.warning("Stratify hedefi binlenemedi (%s); stratify atlanıyor",
                                stratify_target)

    log.info("Holdout: n=%d, test_size=%.2f, group_key=%s (groups=%s), stratify=%s",
             n, test_size, group_key,
             "yok" if groups is None else f"{len(set(groups.tolist()))} unique",
             stratify_target)

    # GÖREV 4: seed araması — group + stratify birlikte (StratifiedGroupKFold)
    # 5 farklı seed dener, KL divergence + grup overlap puanlar, en iyiyi seçer.
    seed_search_enabled = bool(cfg.get("holdout.seed_search.enabled", False)) if cfg else False
    seed_search_seeds: list[int] = list(
        cfg.get("holdout.seed_search.seeds", [42, 1, 7, 21, 123]) if cfg else [42, 1, 7, 21, 123]
    )
    seed_rows: list[dict] | None = None  # seed_selection.txt için

    if (
        seed_search_enabled
        and groups is not None
        and strat is not None
    ):
        chosen_seed, seed_rows = _search_best_seed(
            strat, groups,
            test_size=test_size, seeds=seed_search_seeds,
        )
        log.info("Seed araması: %d aday → seçilen seed=%d", len(seed_search_seeds), chosen_seed)
        train_idx, test_idx, _frac = _stratified_grouped_holdout(
            strat, groups, test_size=test_size, random_state=chosen_seed,
        )
        seed = chosen_seed
        split_method = f"StratifiedGroupKFold(group={group_key}, stratify={stratify_target})"

    # Group split öncelikli; yoksa stratified hedefi kullan
    elif groups is not None:
        train_idx, test_idx = make_holdout_split(
            X, np.zeros(n), groups=groups,
            test_size=test_size, random_state=seed, task="regression",
        )
        split_method = f"GroupShuffleSplit(group={group_key})"
    elif strat is not None:
        train_idx, test_idx = make_holdout_split(
            X, strat, groups=None,
            test_size=test_size, random_state=seed, task="classification",
        )
        split_method = f"StratifiedShuffleSplit(target={stratify_target})"
    else:
        train_idx, test_idx = make_holdout_split(
            X, np.zeros(n), groups=None,
            test_size=test_size, random_state=seed, task="regression",
        )
        split_method = "ShuffleSplit"

    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    # ---- kayıt ----
    np.savez(idx_path, trainval=train_idx, test=test_idx)
    (out_dir / "holdout_indices.json").write_text(
        json.dumps({
            "trainval": train_idx.tolist(),
            "test": test_idx.tolist(),
            "config": {
                "test_size": test_size,
                "random_state": seed,
                "group_key": group_key,
                "stratify_target": stratify_target,
                "split_method": split_method,
                "n_total": int(n),
                "n_trainval": int(train_idx.size),
                "n_test": int(test_idx.size),
            },
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- özet ----
    summary = [
        "Train/Val/Test Holdout Özeti",
        "=" * 50,
        f"Toplam örnek         : {n}",
        f"Trainval (CV burada) : {train_idx.size} ({train_idx.size/n*100:.1f}%)",
        f"Test (held-out)      : {test_idx.size} ({test_idx.size/n*100:.1f}%)",
        f"Yöntem               : {split_method}",
        f"Random state         : {seed}",
        "",
        "NOT: Test seti hiçbir CV/feature-selection/HPO aşamasında",
        "kullanılmamalıdır. Yalnızca final raporlama için açın.",
    ]

    # Sınıflandırma dağılımı (varsa)
    y_stress_path = ds_dir / "y_stress.npy"
    if y_stress_path.exists():
        ys = np.load(y_stress_path)
        summary += ["", "Sınıf dağılımı (y_stress):"]
        for c in sorted(np.unique(ys)):
            n_tr = int((ys[train_idx] == c).sum())
            n_te = int((ys[test_idx] == c).sum())
            summary.append(f"  sınıf {int(c)}: trainval={n_tr:3d}, test={n_te:3d}")

    if groups is not None:
        gtr = set(groups[train_idx].tolist())
        gte = set(groups[test_idx].tolist())
        leak = gtr & gte
        summary += [
            "",
            f"Group leakage kontrolü: {'YOK ✓' if not leak else 'VAR ✗ (' + str(len(leak)) + ' grup)'}",
        ]

    (out_dir / "holdout_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8-sig",
    )

    # GÖREV 4: seed arama raporu
    if seed_rows is not None:
        lines = [
            f"Holdout seed araması ({len(seed_rows)} aday)",
            "=" * 60,
            f"Hedef test_size: {test_size:.2f}",
            f"Aday seedler   : {seed_search_seeds}",
            "",
            f"{'Seed':>6}  {'KL(train||test)':>16}  {'Grup_overlap':>13}  {'Test_frac':>10}  {'n_tr':>5}/{'n_te':>5}",
            "-" * 60,
        ]
        for r in seed_rows:
            lines.append(
                f"{r['seed']:>6}  {r['kl']:>16.4f}  {r['group_overlap']:>13d}  "
                f"{r['test_frac']:>10.3f}  {r['train_n']:>5d}/{r['test_n']:>5d}"
            )
        lines += [
            "-" * 60,
            f"Seçilen seed: {seed}  (kural: grup_overlap=0 olanlar arasından en küçük KL)",
        ]
        (out_dir / "seed_selection.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8-sig",
        )

    log.info("Holdout kaydedildi: trainval=%d, test=%d, yöntem=%s",
             train_idx.size, test_idx.size, split_method)

    paths.write_source_marker(
        out_dir,
        producer="src/m05_dataset/holdout_split.py",
        config_source=cfg.source if cfg is not None else None,
    )
    return out_dir


def load(out_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Önceden üretilmiş holdout indekslerini yükle."""
    out_dir = Path(out_dir) if out_dir else paths.stage_dir("01c_holdout")
    npz = np.load(out_dir / "holdout_indices.npz")
    return npz["trainval"], npz["test"]
