"""Nested-CV GA değerlendirmesi — seçim-yanlılığının çekirdek ölçümü.

İki yol karşılaştırılır:

apparent (biased)
    GA, TÜM veri üzerinde 5-fold CV R²'yi optimize eder; rapor edilen skor da
    aynı veri üzerindeki CV R²'dir. Öznitelik seçimi doğrulama fold'unu
    "görmüştür" → iyimser/şişmiş.

nested
    Dış GroupKFold. Her dış-fold'da GA SADECE dış-train üzerinde koşar, seçilen
    maske dış-train'e fit edilir, dış-test'te R² ölçülür. Öznitelik seçimi dış
    doğrulamayı hiç görmez → dürüst tahmin.

bias = apparent - nested.

RMSE agregasyonu — R² ile BİREBİR aynı mantık
--------------------------------------------
Nested tarafında R², fold başına ölçülen R²'lerin ORTALAMASIDIR
(``np.mean(fold_scores)``). Dolayısıyla nested RMSE de aynı biçimde
FOLD-ORTALAMASI olarak hesaplanır: her dış-fold'da ayrı ``sqrt(MSE)`` bulunur,
sonra fold'lar boyunca ortalaması alınır. Std ise fold-RMSE'lerin std'idir
(``nested_r2_std`` ile birebir tutarlı). Havuzlanmış (pooled) tek bir RMSE
KULLANILMAZ — çünkü nested R² da havuzlanmış değil, fold-ortalamasıdır.

Apparent tarafında R², tüm 5-fold out-of-fold tahminleri tek havuzda toplanıp
``r2_score(y, y_pred)`` ile ölçülür (pooled). Dolayısıyla apparent RMSE de aynı
havuzlanmış tahminlerden tek ``sqrt(MSE)`` olarak hesaplanır — apparent R² ile
birebir aynı agregasyon.

Çalıştırma (smoke / Adım 3 gösterimi):
    python -m src.m06_selection_bias.nested_ga
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict

from src.core.cv import make_cv_splitter
from src.core.logging_setup import get as get_logger
from src.m04_features.ga_feature_selection import _build_regressor, run_ga_core

log = get_logger("m06_selection_bias.nested_ga")


#: GA hiperparametre varsayılanları (tez ayarları). ga_cfg ile override edilir.
DEFAULT_GA_CFG: dict = {
    "pop": 150,
    "ngen": 150,
    "seed": 42,
    "n_jobs": -1,
}


def ga_cfg_from(cfg) -> dict:
    """cfg'den GA hiperparametrelerini çıkar (yoksa DEFAULT_GA_CFG)."""
    if cfg is None:
        return dict(DEFAULT_GA_CFG)
    return {
        "pop": int(cfg.get("ga.pop", DEFAULT_GA_CFG["pop"])),
        "ngen": int(cfg.get("ga.ngen", DEFAULT_GA_CFG["ngen"])),
        "seed": int(cfg.get("models.random_state", DEFAULT_GA_CFG["seed"])),
        "n_jobs": int(cfg.get("ga.n_jobs", DEFAULT_GA_CFG["n_jobs"])),
    }


@dataclass
class NestedResult:
    """Bir (dataset, target, model, selector) için apparent vs nested sonucu."""

    dataset: str
    target: str
    model: str
    # apparent_r2 ARTIK FOLD-ORTALAMASI — nested_r2 ile BİREBİR aynı agregasyon.
    # Eskiden pooled'dı; asimetri apparent-nested farkına karışıyordu. Pooled
    # değer atılmadı, apparent_r2_pooled'da saklanıyor (karşılaştırma için).
    apparent_r2: float
    nested_r2: float
    nested_r2_std: float
    bias: float
    # apparent fold-mean'e geçince artık sıfır OLMAYABİLİR (fold'lar arası
    # değişkenlik girer) — bu beklenen. Eski pickle geriye-uyumu için varsayılan 0.
    apparent_r2_std: float = 0.0
    # Çoklu metrik: RMSE/MAE (aynı tahminlerden, ek model eğitimi yok).
    # RMSE/MAE'de bias yönü R²'nin tersi: apparent daha DÜŞÜK hata gösterir, bu
    # yüzden bias_rmse/bias_mae = nested - apparent (pozitif = apparent iyimser).
    #
    # AGREGASYON — apparent ve nested artık BİREBİR aynı (ikisi de fold-ort):
    #   * apparent_rmse : FOLD-ORTALAMASI (apparent_r2 fold-ort olduğu için).
    #   * nested_rmse   : FOLD-ORTALAMASI (nested_r2 fold-ort olduğu için).
    #   * *_std         : ilgili fold metriklerinin std'i.
    # Eski pickle'larda bu alanlar olmayabilir; None geriye-uyumu korur.
    apparent_rmse: float | None = None
    apparent_rmse_std: float | None = None
    nested_rmse: float | None = None
    nested_rmse_std: float | None = None
    bias_rmse: float | None = None
    apparent_mae: float | None = None
    apparent_mae_std: float | None = None
    nested_mae: float | None = None
    nested_mae_std: float | None = None
    bias_mae: float | None = None
    # POOLED karşılaştırma değerleri (eski agregasyon) — atılmadı, ayrı tutulur.
    # apparent kolu için: tüm 5-fold out-of-fold tahminlerini havuzla, tek metrik.
    apparent_r2_pooled: float | None = None
    apparent_rmse_pooled: float | None = None
    apparent_mae_pooled: float | None = None
    # nested kolu için pooled R² zaten nested_pooled_r2'de; RMSE/MAE pooled da tutulur.
    nested_rmse_pooled: float | None = None
    nested_mae_pooled: float | None = None
    bias_pooled: float | None = None  # apparent_r2_pooled - nested_pooled_r2
    # Hangi seçici aile kullanıldı (ga|rfe|lasso|mutual_info). Eski pickle'larda
    # bu alan olmayabilir; varsayılan "ga" geriye-uyumu korur.
    selector: str = "ga"
    # Sabit-k seçicilerde (rfe/lasso/mutual_info) hedef öznitelik sayısı; GA'da None.
    k: int | None = None
    fold_scores: list[float] = field(default_factory=list)
    # Fold başına nested RMSE/MAE — makale ekine "fold ort ± std" olarak girer.
    fold_rmse: list[float] = field(default_factory=list)
    fold_mae: list[float] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    n_samples: int = 0
    n_features: int = 0
    n_groups: int | None = None
    # Fold-dışı (out-of-fold) havuzlanmış tahminler — bootstrap CI için saklanır
    nested_y_true: np.ndarray | None = None
    nested_y_pred: np.ndarray | None = None
    apparent_y_true: np.ndarray | None = None
    apparent_y_pred: np.ndarray | None = None
    # Havuzlanmış tahminler üzerinden %95 bootstrap GA: {r2, lo, hi, se, n_boot}
    nested_ci: dict | None = None
    apparent_ci: dict | None = None
    # Havuzlanmış R² (fold ortalamasından hafif farklı; CI bunun etrafında)
    nested_pooled_r2: float | None = None

    def summary(self) -> str:
        ci = ""
        if self.nested_ci is not None and self.apparent_ci is not None:
            ci = (f"  | %95 GA: apparent[{self.apparent_ci['lo']:.3f},"
                  f"{self.apparent_ci['hi']:.3f}] "
                  f"nested[{self.nested_ci['lo']:.3f},{self.nested_ci['hi']:.3f}]")
        extra = ""
        if self.nested_rmse is not None:
            extra = (f"  | RMSE app={self.apparent_rmse:.3f} "
                     f"nes={self.nested_rmse:.3f} bias={self.bias_rmse:+.3f}"
                     f"  MAE app={self.apparent_mae:.3f} "
                     f"nes={self.nested_mae:.3f} bias={self.bias_mae:+.3f}")
        return (
            f"[{self.dataset}/{self.target}/{self.model}/{self.selector}] "
            f"apparent={self.apparent_r2:.3f}  nested={self.nested_r2:.3f}"
            f"±{self.nested_r2_std:.3f}  bias={self.bias:+.3f}  "
            f"(n={self.n_samples}, p={self.n_features}, "
            f"folds={[round(s, 3) for s in self.fold_scores]})" + ci + extra
        )


def _finite_mask(y: np.ndarray) -> np.ndarray:
    return np.isfinite(y)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE — sklearn sürümünden bağımsız (sqrt(MSE))."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _apparent_with_selector(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    model: str,
    selector_fn,
    k: int | None,
    sel_kwargs: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Biased yol (genel): tüm veride seç → maske → aynı veride 5-fold CV.

    Seçim TÜM veriyi (dolayısıyla doğrulama fold'larını) gördüğü için iyimser.

    Nested kolla BİREBİR aynı agregasyon için apparent skorları da fold başına
    hesaplanıp ORTALANIR. Havuzlanmış (pooled) değer de ayrıca döndürülür
    (bootstrap CI + karşılaştırma için); atılmaz.

    Returns
    -------
    (mask, y_pred, metrics)
        y_pred : 5-fold CV fold-dışı havuzlanmış tahminler (pooled; CI için).
        metrics: {
            "r2", "r2_std", "rmse", "rmse_std", "mae", "mae_std",   # FOLD-ORT
            "r2_pooled", "rmse_pooled", "mae_pooled",                # POOLED
            "fold_r2", "fold_rmse", "fold_mae",
        }
    """
    mask = selector_fn(X, y, k, groups=groups, model=model, **sel_kwargs)
    Xs = X[:, mask]
    cv = make_cv_splitter(
        n_splits=5, task="regression", groups=groups, random_state=seed,
    )
    reg = _build_regressor(model, Xs.shape[1], Xs.shape[0])
    if groups is not None:
        y_pred = cross_val_predict(reg, Xs, y, cv=cv, n_jobs=1, groups=groups)
    else:
        y_pred = cross_val_predict(reg, Xs, y, cv=cv, n_jobs=1)
    y_pred = np.asarray(y_pred).ravel()

    # Fold başına metrik: aynı cv split'ini yeniden üret (deterministik), her
    # fold'un test dilimini havuzlanmış tahminlerden çıkar. Bu, nested kolun
    # fold-ortalaması mantığıyla BİREBİR aynıdır.
    splits = list(cv.split(Xs, y, groups) if groups is not None else cv.split(Xs, y))
    fold_r2 = [float(r2_score(y[te], y_pred[te])) for _, te in splits]
    fold_rmse = [_rmse(y[te], y_pred[te]) for _, te in splits]
    fold_mae = [_mae(y[te], y_pred[te]) for _, te in splits]

    metrics = {
        "r2": float(np.mean(fold_r2)), "r2_std": float(np.std(fold_r2)),
        "rmse": float(np.mean(fold_rmse)), "rmse_std": float(np.std(fold_rmse)),
        "mae": float(np.mean(fold_mae)), "mae_std": float(np.std(fold_mae)),
        "r2_pooled": float(r2_score(y, y_pred)),
        "rmse_pooled": _rmse(y, y_pred), "mae_pooled": _mae(y, y_pred),
        "fold_r2": fold_r2, "fold_rmse": fold_rmse, "fold_mae": fold_mae,
    }
    return mask, y_pred, metrics


def apparent_r2(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    model: str,
    ga_cfg: dict | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Biased yol (GA, geriye-uyumlu): tüm veride GA → maske → aynı veride CV R².

    İnce sarıcı: :func:`_apparent_with_selector` + GA seçici. Geriye-uyum için
    (apparent_r2_foldmean, mask, y_pred) döndürür; ayrıntılı metrikler için
    doğrudan :func:`_apparent_with_selector` çağrılmalıdır.
    """
    from src.m04_features.selectors import select_ga

    cfg = {**DEFAULT_GA_CFG, **(ga_cfg or {})}
    mask, y_pred, metrics = _apparent_with_selector(
        X, y, groups, model, select_ga, None, dict(cfg), cfg["seed"],
    )
    return metrics["r2"], mask, y_pred


def nested_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    model: str,
    selector_fn,
    n_outer: int = 5,
    ga_cfg: dict | None = None,
    k: int | None = None,
    *,
    dataset: str = "?",
    target: str = "?",
    selector_name: str = "?",
) -> NestedResult:
    """Nested-CV (genel): dış fold'larda seçim SADECE dış-train'de yapılır.

    GA'ya özgü değildir — ``selector_fn`` herhangi bir seçici olabilir
    (bkz. :mod:`src.m04_features.selectors`). Her seçici ortak imzayla çağrılır:
    ``selector_fn(X, y, k, groups=..., model=..., seed=..., pop=..., ngen=..., n_jobs=...)``
    ve fazla argümanları ``**_`` ile yutar.

    Parameters
    ----------
    X, y : ndarray
        Özellik matrisi ve hedef. NaN içeren hedef satırları otomatik atılır.
    groups : ndarray | None
        Dış GroupKFold için grup etiketleri. None ise standart KFold.
    model : str
        Tahmin/fitness regresörü (ga_feature_selection MODEL_CHOICES).
    selector_fn : callable
        ``select_features`` imzalı seçici (selectors.SELECTORS değerlerinden).
    n_outer : int
        Dış fold sayısı.
    ga_cfg : dict | None
        Seçici hiperparametreleri (pop, ngen, seed, n_jobs). GA bunları kullanır;
        diğer seçiciler yalnızca ``seed``i kullanır, gerisini yutar. DEFAULT_GA_CFG
        ile birleşir. ``seed`` ayrıca dış/iç CV ve bootstrap'a akar (çoklu-seed uyumu).
    k : int | None
        Sabit-k seçicilerde (rfe/lasso/mutual_info) hedef öznitelik sayısı.
        GA bunu yok sayar.

    Returns
    -------
    NestedResult
    """
    cfg = {**DEFAULT_GA_CFG, **(ga_cfg or {})}
    sel_kwargs = dict(cfg)  # seed, pop, ngen, n_jobs → seçiciye iletilir
    seed = cfg["seed"]

    # NaN hedefleri at — dış split ve seçim temiz veride çalışsın
    fin = _finite_mask(y)
    X, y = X[fin], y[fin]
    g = groups[fin] if groups is not None else None

    n_groups = None if g is None else len(set(g.tolist()))

    outer = make_cv_splitter(
        n_splits=n_outer, task="regression", groups=g, random_state=seed,
    )
    split_iter = outer.split(X, y, g) if g is not None else outer.split(X, y)

    fold_scores: list[float] = []
    # Fold başına RMSE/MAE — nested R² ile BİREBİR aynı agregasyon (fold-ort).
    fold_rmse: list[float] = []
    fold_mae: list[float] = []
    masks: list[np.ndarray] = []
    # Fold-dışı tahminleri havuzla (her test örneği tam bir kez tahmin edilir).
    # Bu havuz yalnız bootstrap CI ve apparent-benzeri pooled raporlama içindir;
    # nested R²/RMSE/MAE fold-ortalamasından gelir (havuzdan DEĞİL).
    pooled_true = np.full(X.shape[0], np.nan, dtype=float)
    pooled_pred = np.full(X.shape[0], np.nan, dtype=float)
    for fold_i, (tr, te) in enumerate(split_iter, start=1):
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]
        gtr = g[tr] if g is not None else None

        # Seçim SADECE dış-train'de — dış-test hiç görülmez
        mask = selector_fn(Xtr, ytr, k, groups=gtr, model=model, **sel_kwargs)

        reg = _build_regressor(model, int(mask.sum()), Xtr.shape[0])
        reg.fit(Xtr[:, mask], ytr)
        pred = np.asarray(reg.predict(Xte[:, mask])).ravel()
        fold_r2 = float(r2_score(yte, pred))
        # RMSE/MAE bu fold'un kendi tahminlerinden — R² ile aynı fold sınırları.
        fold_rmse_i = _rmse(yte, pred)
        fold_mae_i = _mae(yte, pred)

        pooled_true[te] = yte
        pooled_pred[te] = pred
        fold_scores.append(fold_r2)
        fold_rmse.append(fold_rmse_i)
        fold_mae.append(fold_mae_i)
        masks.append(mask)
        log.info("Dış fold %d/%d [%s]: nested R²=%.3f RMSE=%.3f (seçilen=%d)",
                 fold_i, n_outer, selector_name, fold_r2, fold_rmse_i,
                 int(mask.sum()))

    # Nested skorlar: fold-ortalaması (R², RMSE, MAE hepsi aynı mantık).
    nested = float(np.mean(fold_scores))
    nested_std = float(np.std(fold_scores))
    nested_rmse = float(np.mean(fold_rmse))
    nested_rmse_std = float(np.std(fold_rmse))
    nested_mae = float(np.mean(fold_mae))
    nested_mae_std = float(np.std(fold_mae))

    # Apparent kolu — artık FOLD-ORTALAMASI metrikleri döner (nested ile birebir).
    # Pooled değerler de metrics içinde; atılmaz, ayrı sütunlarda saklanır.
    _, app_pred, app_m = _apparent_with_selector(
        X, y, g, model, selector_fn, k, sel_kwargs, seed,
    )
    app = app_m["r2"]                 # fold-ortalaması apparent R²
    apparent_r2_std = app_m["r2_std"]
    apparent_rmse = app_m["rmse"]     # fold-ortalaması apparent RMSE
    apparent_rmse_std = app_m["rmse_std"]
    apparent_mae = app_m["mae"]
    apparent_mae_std = app_m["mae_std"]

    # Havuzlanmış tahminler üzerinden %95 bootstrap güven aralıkları (pooled).
    from src.core.metrics_ci import bootstrap_r2_ci
    nested_ci = bootstrap_r2_ci(pooled_true, pooled_pred, seed=seed)
    apparent_ci = bootstrap_r2_ci(y, app_pred, seed=seed)

    # POOLED karşılaştırma değerleri (eski agregasyon) — ayrı sütunlarda.
    pm = _finite_mask(pooled_pred) & _finite_mask(pooled_true)
    nested_rmse_pooled = _rmse(pooled_true[pm], pooled_pred[pm])
    nested_mae_pooled = _mae(pooled_true[pm], pooled_pred[pm])
    nested_pooled_r2 = float(nested_ci["r2"])

    res = NestedResult(
        dataset=dataset, target=target, model=model, selector=selector_name,
        k=k, apparent_r2=app, apparent_r2_std=apparent_r2_std,
        nested_r2=nested, nested_r2_std=nested_std,
        bias=app - nested, fold_scores=fold_scores,
        fold_rmse=fold_rmse, fold_mae=fold_mae,
        masks=masks,
        apparent_rmse=apparent_rmse, apparent_rmse_std=apparent_rmse_std,
        nested_rmse=nested_rmse, nested_rmse_std=nested_rmse_std,
        bias_rmse=nested_rmse - apparent_rmse,
        apparent_mae=apparent_mae, apparent_mae_std=apparent_mae_std,
        nested_mae=nested_mae, nested_mae_std=nested_mae_std,
        bias_mae=nested_mae - apparent_mae,
        # POOLED (karşılaştırma; atılmadı)
        apparent_r2_pooled=app_m["r2_pooled"],
        apparent_rmse_pooled=app_m["rmse_pooled"],
        apparent_mae_pooled=app_m["mae_pooled"],
        nested_rmse_pooled=nested_rmse_pooled,
        nested_mae_pooled=nested_mae_pooled,
        bias_pooled=app_m["r2_pooled"] - nested_pooled_r2,
        n_samples=X.shape[0], n_features=X.shape[1], n_groups=n_groups,
        nested_y_true=pooled_true, nested_y_pred=pooled_pred,
        apparent_y_true=np.asarray(y, dtype=float), apparent_y_pred=app_pred,
        nested_ci=nested_ci, apparent_ci=apparent_ci,
        nested_pooled_r2=nested_pooled_r2,
    )
    log.info(res.summary())
    return res


def nested_ga_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    model: str,
    n_outer: int = 5,
    ga_cfg: dict | None = None,
    *,
    dataset: str = "?",
    target: str = "?",
) -> NestedResult:
    """Nested-CV GA (geriye-uyumlu sarıcı).

    Mevcut çağrı yerlerini bozmamak için korunur; :func:`nested_evaluation`'ı
    GA seçici ile çağırır. Davranış birebir aynıdır.
    """
    from src.m04_features.selectors import select_ga

    return nested_evaluation(
        X, y, groups, model, select_ga, n_outer=n_outer, ga_cfg=ga_cfg, k=None,
        dataset=dataset, target=target, selector_name="ga",
    )


def run(cfg=None):
    """Pipeline aşaması 17_selection_bias: Ryckewaert ana örneği + rapor.

    nested-vs-apparent'i ana kombinasyonlarda koşar, sonuçları pickle'lar ve
    özet tablo + figürleri üretir.
    """
    import pickle

    from src.core import paths
    from src.m01_io.dataset_registry import load_ryckewaert
    from src.m06_selection_bias import report

    out_dir = paths.stage_dir("17_selection_bias")
    ga_cfg = ga_cfg_from(cfg)
    n_outer = int(cfg.get("cv.n_splits", 5)) if cfg is not None else 5

    ds = load_ryckewaert()
    combos = [("flavonol", "pls"), ("flavonol", "lightgbm")]
    results = []
    for target, model in combos:
        if target not in ds.targets:
            continue
        results.append(nested_ga_evaluation(
            ds.X, ds.target(target), ds.groups, model=model, n_outer=n_outer,
            ga_cfg=ga_cfg, dataset=ds.name, target=target,
        ))
    (out_dir / "results.pkl").write_bytes(pickle.dumps(results))
    report.generate_all(results, out_dir=out_dir)
    log.info("17_selection_bias tamamlandı → %s", out_dir)
    return out_dir


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s | %(message)s",
                         datefmt="%H:%M:%S")

    from src.m01_io.dataset_registry import load_ryckewaert

    ds = load_ryckewaert()
    y = ds.target("flavonol")
    print(f"Ryckewaert flavonol+PLS — X={ds.X.shape}, "
          f"groups={'yok' if ds.groups is None else len(set(ds.groups.tolist()))}")

    res = nested_ga_evaluation(
        ds.X, y, ds.groups, model="pls", n_outer=5,
        dataset=ds.name, target="flavonol",
    )
    print("\n" + "=" * 60)
    print(res.summary())
    print(f"\nBeklenti: apparent≈0.64, nested≈0.28")
    print(f"Ölçülen : apparent={res.apparent_r2:.3f}, "
          f"nested={res.nested_r2:.3f}, bias={res.bias:+.3f}")
