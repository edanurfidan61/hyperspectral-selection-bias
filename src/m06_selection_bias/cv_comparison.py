"""Üç-pipeline CV karşılaştırması — iki sızıntı kaynağını ayrıştırma (Görev 3).

Aynı (dataset, hedef, model, selector) için üç doğrulama pipeline'ı koşulup iki
ayrı leakage kaynağı katman katman soyulur:

| Pipeline | Bölme            | Feature selection      | İçindeki sızıntı   |
|----------|------------------|------------------------|--------------------|
| P1       | Random KFold     | CV DIŞINDA (tüm veride)| grup + seçim       |
| P2       | GroupKFold       | CV DIŞINDA (tüm veride)| sadece seçim       |
| P3       | Nested GroupKFold| CV İÇİNDE (fold-train) | yok (dürüst)       |

Beklenti: ``P1 ≥ P2 ≥ P3``.
    * ``group_leakage   = P1 − P2`` → grup sızıntısının katkısı.
    * ``selection_leakage = P2 − P3`` → seçim sızıntısının katkısı (asıl bulgu).

Dürüstlük notu: Ryckewaert'te her örnek kendi grubudur (n_groups == n_samples),
ayıracak grup olmadığı için ``P1 ≈ P2`` BEKLENİR — bu bir kusur değil. CSV'deki
``n_groups`` kolonu bunu yorumlanabilir kılar. Gerçek tarla gruplu setlerde
(ör. deep_patato) ``P1 > P2`` farkı belirginleşir.

Çalıştırma (smoke):
    python -m src.m06_selection_bias.cv_comparison
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

from src.core.logging_setup import get as get_logger
from src.m04_features.ga_feature_selection import _build_regressor
from src.m06_selection_bias.nested_ga import (
    DEFAULT_GA_CFG,
    _finite_mask,
    _mae,
    _rmse,
    nested_evaluation,
)

log = get_logger("m06_selection_bias.cv_comparison")


def _r2_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """R²/RMSE/MAE üçlüsü — Görev 2'deki yardımcıların yeniden kullanımı."""
    from sklearn.metrics import r2_score

    return float(r2_score(y_true, y_pred)), _rmse(y_true, y_pred), _mae(y_true, y_pred)


def run_cv_comparison(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    model: str,
    selector_fn,
    k: int | None,
    n_splits: int = 5,
    seed: int = 42,
    ga_cfg: dict | None = None,
    *,
    dataset: str = "?",
    target: str = "?",
    selector_name: str = "?",
) -> dict:
    """Üç pipeline'ı hesapla ve iki leakage bileşenini döndür.

    P1 ve P2, *aynı* dış-seçim maskesini paylaşır (selector_fn TÜM veride bir kez
    çağrılır); böylece aralarındaki tek fark bölme stratejisidir (random vs group)
    ve P1−P2 saf grup sızıntısını ölçer. P3 için mevcut :func:`nested_evaluation`
    çağrılır — seçim her dış-fold'un train'inde yapılır (dürüst).

    Tüm rastgelelik (KFold shuffle, GroupKFold sırası, selector) ``seed``e bağlıdır;
    çoklu-seed runner ile uyumludur.

    Returns
    -------
    dict
        p1_r2/p2_r2/p3_r2 (+ rmse/mae), group_leakage, selection_leakage,
        n_groups/n_samples/n_features ve kimlik alanları.
    """
    cfg = {**DEFAULT_GA_CFG, **(ga_cfg or {}), "seed": int(seed)}
    sel_kwargs = dict(cfg)

    # NaN hedefleri at — nested_evaluation ile aynı temizlik
    fin = _finite_mask(y)
    X, y = X[fin], y[fin]
    g = groups[fin] if groups is not None else None
    n_groups = None if g is None else len(set(g.tolist()))

    # --- P1 & P2: dış-seçim (tüm veride bir kez), maske sabit ---------------
    mask = selector_fn(X, y, k, groups=g, model=model, **sel_kwargs)
    Xs = X[:, mask]
    reg = _build_regressor(model, int(mask.sum()), Xs.shape[0])

    # P1 — random KFold (grup yok sayılır)
    cv1 = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    p1_pred = np.asarray(cross_val_predict(reg, Xs, y, cv=cv1, n_jobs=1)).ravel()
    p1_r2, p1_rmse, p1_mae = _r2_rmse_mae(y, p1_pred)

    # P2 — GroupKFold (groups kullanılarak). Grup yoksa ya da grup sayısı
    # n_splits'ten azsa group-bölme tanımsız → P2'yi raporlanamaz (NaN) bırak.
    if g is not None and n_groups is not None and n_groups >= n_splits:
        cv2 = GroupKFold(n_splits=n_splits)
        p2_pred = np.asarray(
            cross_val_predict(reg, Xs, y, cv=cv2, groups=g, n_jobs=1)
        ).ravel()
        p2_r2, p2_rmse, p2_mae = _r2_rmse_mae(y, p2_pred)
    else:
        log.warning("[%s/%s/%s/%s] P2 atlandı (grup yok ya da n_groups<%d)",
                    dataset, target, model, selector_name, n_splits)
        p2_r2 = p2_rmse = p2_mae = float("nan")

    # --- P3: nested (seçim fold-train'de) ----------------------------------
    nres = nested_evaluation(
        X, y, g, model, selector_fn, n_outer=n_splits, ga_cfg=cfg, k=k,
        dataset=dataset, target=target, selector_name=selector_name,
    )
    p3_r2, p3_rmse, p3_mae = nres.nested_r2, nres.nested_rmse, nres.nested_mae

    group_leakage = p1_r2 - p2_r2
    selection_leakage = p2_r2 - p3_r2

    out = {
        "dataset": dataset, "target": target, "model": model,
        "selector": selector_name, "k": k,
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "n_groups": n_groups, "n_splits": n_splits, "seed": int(seed),
        "p1_r2": p1_r2, "p2_r2": p2_r2, "p3_r2": p3_r2,
        "p1_rmse": p1_rmse, "p2_rmse": p2_rmse, "p3_rmse": p3_rmse,
        "p1_mae": p1_mae, "p2_mae": p2_mae, "p3_mae": p3_mae,
        "group_leakage": group_leakage,
        "selection_leakage": selection_leakage,
    }
    log.info(
        "[%s/%s/%s/%s] P1=%.3f P2=%.3f P3=%.3f | grup=%.3f seçim=%.3f (n_grp=%s)",
        dataset, target, model, selector_name, p1_r2, p2_r2, p3_r2,
        group_leakage, selection_leakage, n_groups,
    )
    return out


def run_cv_comparison_multi(
    datasets: list[str] | None = None,
    combos: dict[str, list[tuple[str, str]]] | None = None,
    n_splits: int = 5,
    seed: int = 42,
    ga_cfg: dict | None = None,
    include_lambrusco: bool = False,
    selectors: list[str] | None = None,
    k: int = 20,
) -> list[dict]:
    """Config'deki tüm (dataset, hedef, model, selector) için cv_comparison koştur.

    :func:`multi_dataset.iter_combos` ile aynı kombinasyon/atlama mantığını paylaşır.
    """
    from src.m04_features.selectors import get_selector
    from src.m06_selection_bias.multi_dataset import iter_combos

    rows: list[dict] = []
    for name, ds, target, model, selector in iter_combos(
        datasets=datasets, combos=combos, include_lambrusco=include_lambrusco,
        selectors=selectors,
    ):
        log.info("=== CV-CMP %s / %s / %s / %s ===", name, target, model, selector)
        rows.append(run_cv_comparison(
            ds.X, ds.target(target), ds.groups, model=model,
            selector_fn=get_selector(selector), k=k, n_splits=n_splits,
            seed=seed, ga_cfg=ga_cfg, dataset=name, target=target,
            selector_name=selector,
        ))
    return rows


def write_summary_csv(rows: list[dict], out_dir) -> "Path":  # type: ignore[name-defined]
    """cv_comparison_summary.csv yaz."""
    from pathlib import Path

    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    path = out_dir / "cv_comparison_summary.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("CV karşılaştırma özeti yazıldı: %s (%d satır)", path, len(df))
    return path


def run(cfg=None):
    """Pipeline aşaması 17d_cv_comparison: tüm kombinasyonlarda üç-pipeline + rapor."""
    from src.core import paths
    from src.m06_selection_bias import report
    from src.m06_selection_bias.multi_dataset import DEFAULT_SELECTORS, DEFAULT_K
    from src.m06_selection_bias.nested_ga import ga_cfg_from

    out_dir = paths.stage_dir("17d_cv_comparison")
    ga_cfg = ga_cfg_from(cfg)
    n_splits = int(cfg.get("cv.n_splits", 5)) if cfg is not None else 5
    seed = int(cfg.get("models.random_state", 42)) if cfg is not None else 42
    include_lambrusco = bool(cfg.get("selection_bias.include_lambrusco", False)) \
        if cfg is not None else False
    selectors = (cfg.get("selection_bias.selectors", None) if cfg is not None else None) \
        or DEFAULT_SELECTORS
    k = int(cfg.get("selection_bias.k", DEFAULT_K)) if cfg is not None else DEFAULT_K

    rows = run_cv_comparison_multi(
        n_splits=n_splits, seed=seed, ga_cfg=ga_cfg,
        include_lambrusco=include_lambrusco, selectors=list(selectors), k=k,
    )
    write_summary_csv(rows, out_dir)
    report.fig_cv_comparison(rows, out_dir)
    log.info("17d_cv_comparison tamamlandı (%d satır) → %s", len(rows), out_dir)
    return out_dir


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s | %(message)s",
                         datefmt="%H:%M:%S")
    from src.m01_io.dataset_registry import load_ryckewaert
    from src.m04_features.selectors import get_selector

    ds = load_ryckewaert()
    # Smoke: ryckewaert flavonol / pls / ga, hafif GA ayarı
    res = run_cv_comparison(
        ds.X, ds.target("flavonol"), ds.groups, model="pls",
        selector_fn=get_selector("ga"), k=20, n_splits=5, seed=42,
        ga_cfg={"pop": 20, "ngen": 8, "n_jobs": -1},
        dataset=ds.name, target="flavonol", selector_name="ga",
    )
    print("\n" + "=" * 64)
    print(f"P1 (random + dış seçim) R² = {res['p1_r2']:.3f}")
    print(f"P2 (group  + dış seçim) R² = {res['p2_r2']:.3f}")
    print(f"P3 (nested group)       R² = {res['p3_r2']:.3f}")
    print(f"  grup sizintisi  (P1-P2) = {res['group_leakage']:.3f}")
    print(f"  secim sizintisi (P2-P3) = {res['selection_leakage']:.3f}")
    print(f"  n_groups = {res['n_groups']}  (n_samples = {res['n_samples']})")
