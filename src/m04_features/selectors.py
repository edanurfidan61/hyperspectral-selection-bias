"""Özellik seçici ailesi — "sorun GA değil, CV-dışı seçimin kendisi" tezi için.

Tek bir ortak imza altında dört farklı aileden seçici toplar. Hepsi aynı
``k`` (seçilen öznitelik sayısı) ile çalışacak şekilde tasarlandı ki adil
karşılaştırma yapılabilsin; tek istisna GA'dır (doğası gereği değişken sayıda
öznitelik seçer, ``k`` yalnızca ipucu olarak kabul edilir ve yok sayılır).

Ortak imza
----------
    select_features(X, y, k, **kwargs) -> mask  (bool ndarray, len == n_features)

Aileler
-------
wrapper   : :func:`select_rfe`         — RFE (Ridge estimator)
embedded  : :func:`select_lasso`       — LassoCV, k-tavanlı |katsayı| seçimi
filter    : :func:`select_mutual_info` — mutual_info_regression
(arama)   : :func:`select_ga`          — mevcut ``run_ga_core`` sarıcısı

Her seçici fazladan keyword argümanları (groups, model, seed, pop, ngen, ...)
``**_`` ile sessizce yutar; böylece nested döngü hepsini aynı çağrıyla
besleyebilir.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _clip_k(k: int | None, n_features: int, default: int = 20) -> int:
    """k'yı [1, n_features] aralığına kırp (None ise default)."""
    if k is None:
        k = default
    return max(1, min(int(k), int(n_features)))


def _topk_mask(scores: np.ndarray, k: int, n_features: int) -> np.ndarray:
    """En yüksek k skora sahip indeksler için bool maske üret."""
    idx = np.argsort(scores)[::-1][:k]
    mask = np.zeros(n_features, dtype=bool)
    mask[idx] = True
    return mask


# ---------------------------------------------------------------------------
# wrapper ailesi — RFE (Recursive Feature Elimination)
# ---------------------------------------------------------------------------
def select_rfe(X: np.ndarray, y: np.ndarray, k: int = 20, **kwargs) -> np.ndarray:
    """RFE ile k öznitelik seç (sarmalayıcı/wrapper ailesi).

    Estimator olarak Ridge kullanılır (coef_ → öznitelik önemi, hızlı ve
    yüksek-korelasyonlu hiperspektral bantlarda kararlı). ``step=0.1`` ile her
    turda kalan özniteliklerin %10'u elenir → büyük p'de makul süre.
    """
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import Ridge

    k = _clip_k(k, X.shape[1])
    sel = RFE(estimator=Ridge(alpha=1.0), n_features_to_select=k, step=0.1)
    sel.fit(X, y)
    return np.asarray(sel.support_, dtype=bool)


# ---------------------------------------------------------------------------
# embedded ailesi — LassoCV
# ---------------------------------------------------------------------------
def select_lasso(X: np.ndarray, y: np.ndarray, k: int = 20, *,
                 seed: int = 42, **kwargs) -> np.ndarray:
    """LassoCV ile k öznitelik seç (gömülü/embedded ailesi).

    LassoCV alpha'yı CV ile seçer; k bir TAVAN'dır, taban yoktur. Sıfır-olmayan
    sayısı k'dan fazlaysa en büyük |katsayı|lı k tanesi; az ise tümü tutulur
    (top-k'ya DOLDURULMAZ → k'dan az öznitelik dönebilir).
    Hiç sıfır-olmayan yoksa en büyük |katsayı|lı k'ya düşülür (her zaman maske üret).
    Lasso ölçek-duyarlı → StandardScaler şart.
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    n_features = X.shape[1]
    k = _clip_k(k, n_features)
    Xs = StandardScaler().fit_transform(X)
    lasso = LassoCV(cv=5, random_state=int(seed), max_iter=5000, n_jobs=1)
    # Yüksek-korelasyonlu hiperspektral bantlarda LassoCV koordinat-inişi sık sık
    # tolerans altına inmeden tavan iterasyona çarpar → zararsız ConvergenceWarning
    # seli. Seçim (en büyük |katsayı|lı k) bundan etkilenmez; logu temiz tutmak için sustur.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lasso.fit(Xs, y)
    coef = np.abs(np.asarray(lasso.coef_, dtype=float))

    nonzero = coef > 0
    n_nz = int(nonzero.sum())
    if 0 < n_nz <= k:
        mask = np.zeros(n_features, dtype=bool)
        mask[nonzero] = True
        return mask
    # n_nz > k  ya da  n_nz == 0 → en büyük |katsayı|lı k
    return _topk_mask(coef, k, n_features)


# ---------------------------------------------------------------------------
# filter ailesi — mutual information
# ---------------------------------------------------------------------------
def select_mutual_info(X: np.ndarray, y: np.ndarray, k: int = 20, *,
                       seed: int = 42, **kwargs) -> np.ndarray:
    """mutual_info_regression ile en yüksek k öznitelik (filtre ailesi).

    Model-bağımsız, doğrusal-olmayan ilişkileri de yakalar. random_state ile
    kNN tabanlı MI tahmini seed'e bağlanır (çoklu-seed uyumu).
    """
    from sklearn.feature_selection import mutual_info_regression

    k = _clip_k(k, X.shape[1])
    mi = mutual_info_regression(X, y, random_state=int(seed))
    return _topk_mask(np.asarray(mi, dtype=float), k, X.shape[1])


# ---------------------------------------------------------------------------
# arama ailesi — GA (mevcut run_ga_core'un ince sarıcısı)
# ---------------------------------------------------------------------------
def select_ga(X: np.ndarray, y: np.ndarray, k: int | None = None, *,
              groups: np.ndarray | None = None, model: str = "pls",
              seed: int = 42, pop: int = 150, ngen: int = 150,
              n_jobs: int = -1, **kwargs) -> np.ndarray:
    """Mevcut GA'yı ortak imzaya saran ince wrapper.

    ``k`` yok sayılır — GA seçilen öznitelik sayısını kendi optimize eder
    (sabit k'ya zorlamak mevcut nested iskeletinin davranışını değiştirirdi).
    ``run_ga_core`` dosya yazmaz/çizmez; nested-CV'de dış-fold başına güvenle çağrılır.
    """
    from src.m04_features.ga_feature_selection import run_ga_core

    return run_ga_core(
        X, y, groups, model=model,
        pop=int(pop), ngen=int(ngen), seed=int(seed), n_jobs=int(n_jobs),
    )


#: İsim → seçici fonksiyon. Config/CLI ``selectors`` listesi bu anahtarları kullanır.
SELECTORS: dict[str, Callable[..., np.ndarray]] = {
    "ga": select_ga,
    "rfe": select_rfe,
    "lasso": select_lasso,
    "mutual_info": select_mutual_info,
}

#: Seçici ailesi etiketleri (rapor/tablo için).
SELECTOR_FAMILY: dict[str, str] = {
    "ga": "search",
    "rfe": "wrapper",
    "lasso": "embedded",
    "mutual_info": "filter",
}


def get_selector(name: str) -> Callable[..., np.ndarray]:
    """İsimden seçici fonksiyonu getir (bilinmeyen ad → ValueError)."""
    try:
        return SELECTORS[name]
    except KeyError:
        raise ValueError(
            f"Bilinmeyen seçici: {name!r}. Seçenekler: {list(SELECTORS)}"
        ) from None


def select_features(X: np.ndarray, y: np.ndarray, k: int = 20, *,
                    selector: str = "ga", **kwargs) -> np.ndarray:
    """Ortak giriş noktası: ``selector`` adına göre uygun seçiciyi çağır."""
    return get_selector(selector)(X, y, k, **kwargs)
