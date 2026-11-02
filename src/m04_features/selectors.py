"""Özellik seçici ailesi — ortak imza, Revizyon 1 genişletmesi (plan §2.1).

Tek bir ortak imza altında farklı ailelerden seçiciler. Sabit-k seçiciler aynı
``k`` ile çağrılır (adil karşılaştırma); GA ``k``'yı yok sayar.

Ortak imza
----------
    select_xxx(X, y, k, *, return_info=False, **kwargs)
        -> mask                     (return_info=False; bool ndarray, len == n_features)
        -> (mask, info: dict)       (return_info=True)

Fazladan keyword argümanları (groups, model, seed, pop, ngen, n_jobs, ...) ``**_``
ile yutulur; nested döngü hepsini aynı çağrıyla besleyebilir.

Aileler
-------
wrapper   : :func:`select_rfe`         — RFE (Ridge; ``scale=True`` → önce StandardScaler)
embedded  : :func:`select_lasso`       — LassoCV + k-tavanlı |katsayı| seçimi
filter    : :func:`select_mutual_info` — mutual_info_regression (``n_neighbors``)
(arama)   : :func:`select_ga`          — ``run_ga_core`` sarıcısı
kontrol   : :func:`select_random`      — rastgele k öznitelik (R3-67)
kontrol   : :func:`select_none`        — tüm öznitelikler (seçimsiz)

LASSO iç CV (R3-33)
-------------------
``inner="group"`` (varsayılan, Revizyon 1): LassoCV'nin α seçimi için iç bölme
``GroupKFold(5)`` (groups verildiyse) ya da ``KFold(5, shuffle=True, random_state=seed)``.
``inner="legacy"``: makaledeki davranış BİREBİR — ``LassoCV(cv=5)`` (= düz, karışık
olmayan KFold, grup bilgisi yok), ``max_iter=5000``, ConvergenceWarning susturulur.

GERİYE UYUM: ``SELECTORS["lasso"]`` (m06 hattının kullandığı kayıt) ``inner="legacy"``
sürümüne bağlıdır → ``run_selection_bias.py`` yayımlanan sayıları aynen üretir.
Yeni deneyler ``select_lasso`` (varsayılan ``inner="group"``) ya da
``SELECTORS["lasso_group"]`` kullanır.
"""

from __future__ import annotations

import warnings
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


def _ret(mask: np.ndarray, info: dict, return_info: bool):
    """Ortak dönüş: yalnız maske ya da (maske, info)."""
    info.setdefault("n_selected", int(mask.sum()))
    return (mask, info) if return_info else mask


# ---------------------------------------------------------------------------
# wrapper ailesi — RFE (Recursive Feature Elimination)
# ---------------------------------------------------------------------------
def select_rfe(X: np.ndarray, y: np.ndarray, k: int = 20, *,
               alpha: float = 1.0, step: float = 0.1, scale: bool = False,
               return_info: bool = False, **kwargs):
    """RFE ile k öznitelik seç (sarmalayıcı/wrapper ailesi).

    Estimator: ``Ridge(alpha)``; her turda kalan özniteliklerin ``step`` oranı elenir.
    ``scale=False`` (makale davranışı): Ridge ölçeklemesiz X görür → sıralama sütun
    ölçeğine bağlıdır (R3-26). ``scale=True``: StandardScaler YALNIZ verilen X'e fit
    edilir (nested'de dış-train) — sızıntı yok. Deterministik (seed kullanmaz).
    """
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import Ridge

    k = _clip_k(k, X.shape[1])
    Xf = X
    if scale:
        from sklearn.preprocessing import StandardScaler
        Xf = StandardScaler().fit_transform(X)
    sel = RFE(estimator=Ridge(alpha=float(alpha)), n_features_to_select=k, step=step)
    sel.fit(Xf, y)
    mask = np.asarray(sel.support_, dtype=bool)
    n_steps = int(np.max(sel.ranking_))  # eleme turu sayısı + 1 (R3-39 bütçe)
    return _ret(mask, {"selector": "rfe", "alpha": float(alpha), "step": step,
                       "scale": bool(scale), "n_elimination_rounds": n_steps - 1}, return_info)


# ---------------------------------------------------------------------------
# embedded ailesi — LassoCV
# ---------------------------------------------------------------------------
def _lasso_inner_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None,
                    seed: int, n_splits: int = 5):
    """inner="group" için LassoCV'ye verilecek iç bölme listesi."""
    from sklearn.model_selection import GroupKFold, KFold

    if groups is not None:
        g = np.asarray(groups)
        n_g = len(np.unique(g))
        if n_g < n_splits:
            raise ValueError(f"iç GroupKFold için grup sayısı {n_g} < {n_splits}")
        return list(GroupKFold(n_splits=n_splits).split(X, y, g)), "GroupKFold(5)"
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    return list(cv.split(X, y)), f"KFold(5, shuffle=True, random_state={int(seed)})"


def select_lasso(X: np.ndarray, y: np.ndarray, k: int | None = 20, *,
                 groups: np.ndarray | None = None, seed: int = 42,
                 inner: str = "group", max_iter: int | None = None, tol: float = 1e-4,
                 return_info: bool = False, return_coef: bool = False, **kwargs):
    """LassoCV ile en çok k öznitelik seç (gömülü/embedded ailesi).

    α LassoCV ile seçilir; ardından k bir TAVAN'dır, taban yoktur:
      * 0 < sıfır-olmayan ≤ k → tüm sıfır-olmayanlar (k'ya DOLDURULMAZ)
      * sıfır-olmayan > k     → |katsayı| en büyük k  (``truncated=True``, R3-35)
      * sıfır-olmayan = 0     → |katsayı| en büyük k  (``filled_from_zero=True``)
    ``k=None`` → kesme yok: tüm sıfır-olmayanlar (E3 "kesmesiz" ekseni).
    Lasso ölçek-duyarlı → StandardScaler YALNIZ verilen X'e fit edilir.

    Parameters
    ----------
    inner : {"group", "legacy"}
        ``"group"``: iç CV GroupKFold(5) (groups varsa) / KFold(5, shuffle, seed).
        Yakınsama uyarıları SAYILIR (``info["n_convergence_warnings"]``), gizlenmez
        bilgi olarak raporlanır.
        ``"legacy"``: makaledeki çağrı birebir (``cv=5`` düz KFold, ``max_iter=5000``,
        uyarılar susturulur). ``random_state`` her iki modda da yalnız
        ``selection='random'`` iken etkilidir (varsayılan 'cyclic' → seed'den bağımsız).
    max_iter : int | None
        None → legacy'de 5000, group'ta 20000.
    return_coef : bool
        True → ``info["abs_coef"]`` (|katsayı| listesi; varsayılan kapalı, E3 için).
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    if inner not in ("group", "legacy"):
        raise ValueError(f"inner={inner!r}; 'group' ya da 'legacy' olmalı")
    n_features = X.shape[1]
    k_cap = None if k is None else _clip_k(k, n_features)
    Xs = StandardScaler().fit_transform(X)

    n_warn = 0
    if inner == "legacy":
        # Makale davranışı — DEĞİŞTİRME (geriye uyum testi bu satıra bağlı).
        lasso = LassoCV(cv=5, random_state=int(seed), max_iter=5000, n_jobs=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            lasso.fit(Xs, y)
        inner_desc = "KFold(5, shuffle=False) [LassoCV cv=5]"
    else:
        cv, inner_desc = _lasso_inner_cv(Xs, y, groups, seed)
        lasso = LassoCV(cv=cv, random_state=int(seed), n_jobs=1,
                        max_iter=int(max_iter or 20000), tol=float(tol))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            lasso.fit(Xs, y)
        n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in caught)

    coef = np.abs(np.asarray(lasso.coef_, dtype=float))
    mask, truncated, filled = lasso_mask_from_abs_coef(coef, k_cap)
    info = {"selector": "lasso", "inner": inner, "inner_cv": inner_desc,
            "alpha": float(lasso.alpha_), "n_nonzero": int((coef > 0).sum()), "k_cap": k_cap,
            "truncated": bool(truncated), "filled_from_zero": bool(filled),
            "max_iter": int(lasso.max_iter), "tol": float(lasso.tol),
            "n_iter": int(np.max(np.atleast_1d(lasso.n_iter_))),
            "n_convergence_warnings": int(n_warn),
            "n_alphas": int(len(lasso.alphas_)), "n_inner_folds": 5}
    if return_coef:
        # E3 k ekseni: LassoCV k'dan bağımsızdır; |katsayı| saklanırsa her k için
        # maske lasso_mask_from_abs_coef ile yeniden fit etmeden türetilir.
        info["abs_coef"] = coef.tolist()
    return _ret(mask, info, return_info)


def lasso_mask_from_abs_coef(coef: np.ndarray, k_cap: int | None) -> tuple[np.ndarray, bool, bool]:
    """select_lasso'nun k-tavan kuralı (tek kaynak): (maske, truncated, filled_from_zero).

    * 0 < sıfır-olmayan ≤ k → tüm sıfır-olmayanlar
    * sıfır-olmayan > k     → |katsayı| en büyük k  (truncated)
    * sıfır-olmayan = 0     → |katsayı| en büyük k  (filled_from_zero; k None ise 1)
    ``k_cap=None`` → kesme yok.
    """
    coef = np.abs(np.asarray(coef, dtype=float))
    n_features = coef.shape[0]
    nonzero = coef > 0
    n_nz = int(nonzero.sum())
    truncated = k_cap is not None and n_nz > k_cap
    filled = n_nz == 0
    if filled:
        mask = _topk_mask(coef, k_cap or 1, n_features)
    elif truncated:
        mask = _topk_mask(coef, k_cap, n_features)
    else:
        mask = nonzero.copy()
    return mask, bool(truncated), bool(filled)


def select_lasso_legacy(X: np.ndarray, y: np.ndarray, k: int | None = 20, **kwargs):
    """Makaledeki LASSO (``inner="legacy"``) — ``SELECTORS["lasso"]`` buna bağlı."""
    kwargs.pop("inner", None)
    return select_lasso(X, y, k, inner="legacy", **kwargs)


# ---------------------------------------------------------------------------
# filter ailesi — mutual information
# ---------------------------------------------------------------------------
def select_mutual_info(X: np.ndarray, y: np.ndarray, k: int = 20, *,
                       seed: int = 42, n_neighbors: int = 3,
                       return_info: bool = False, **kwargs):
    """mutual_info_regression ile en yüksek k öznitelik (filtre ailesi).

    kNN tabanlı MI tahmini; ``random_state=seed`` tahminciye eklenen küçük gürültüyü
    belirler (çoklu-seed değişkenliği buradan gelir). ``n_neighbors=3`` = makale.
    """
    from sklearn.feature_selection import mutual_info_regression

    k = _clip_k(k, X.shape[1])
    mi = mutual_info_regression(X, y, random_state=int(seed), n_neighbors=int(n_neighbors))
    mask = _topk_mask(np.asarray(mi, dtype=float), k, X.shape[1])
    return _ret(mask, {"selector": "mutual_info", "n_neighbors": int(n_neighbors),
                       "seed": int(seed)}, return_info)


# ---------------------------------------------------------------------------
# kontrol kolları — rastgele k ve seçimsiz
# ---------------------------------------------------------------------------
def select_random(X: np.ndarray, y: np.ndarray, k: int = 20, *, seed: int = 42,
                  return_info: bool = False, **kwargs):
    """Rastgele k öznitelik (y'ye bakmaz) — seçimsiz sonlu-örneklem tabanı (R3-67)."""
    k = _clip_k(k, X.shape[1])
    rng = np.random.default_rng(int(seed))
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[rng.choice(X.shape[1], size=k, replace=False)] = True
    return _ret(mask, {"selector": "random_k", "seed": int(seed)}, return_info)


def select_none(X: np.ndarray, y: np.ndarray, k: int | None = None, *,
                return_info: bool = False, **kwargs):
    """Seçim yok — tüm öznitelikler (seçimsiz kontrol kolu)."""
    return _ret(np.ones(X.shape[1], dtype=bool), {"selector": "none"}, return_info)


# ---------------------------------------------------------------------------
# arama ailesi — GA (mevcut run_ga_core'un ince sarıcısı)
# ---------------------------------------------------------------------------
def select_ga(X: np.ndarray, y: np.ndarray, k: int | None = None, *,
              groups: np.ndarray | None = None, model: str = "pls",
              seed: int = 42, pop: int = 150, ngen: int = 150,
              n_jobs: int = -1, return_info: bool = False, **kwargs):
    """Mevcut GA'yı ortak imzaya saran ince wrapper.

    ``k`` yok sayılır — GA seçilen öznitelik sayısını kendi optimize eder.
    ``run_ga_core`` dosya yazmaz; nested-CV'de dış-fold başına güvenle çağrılır.
    GA'nın iç fitness'ı yalnız verilen (X, y, groups) üzerinde hesaplanır.
    """
    from src.m04_features.ga_feature_selection import run_ga_core

    mask = run_ga_core(
        X, y, groups, model=model,
        pop=int(pop), ngen=int(ngen), seed=int(seed), n_jobs=int(n_jobs),
    )
    return _ret(np.asarray(mask, dtype=bool),
                {"selector": "ga", "pop": int(pop), "ngen": int(ngen), "seed": int(seed),
                 "model": model}, return_info)


#: İsim → seçici. m06 hattı (``get_selector``) bu kaydı kullanır; ``"lasso"``
#: GERİYE UYUM için legacy LASSO'dur (yayımlanan sayılar).
SELECTORS: dict[str, Callable[..., np.ndarray]] = {
    "ga": select_ga,
    "rfe": select_rfe,
    "lasso": select_lasso_legacy,
    "lasso_group": select_lasso,
    "mutual_info": select_mutual_info,
    "random_k": select_random,
    "none": select_none,
}

#: Seçici ailesi etiketleri (rapor/tablo için).
SELECTOR_FAMILY: dict[str, str] = {
    "ga": "search",
    "rfe": "wrapper",
    "lasso": "embedded",
    "lasso_group": "embedded",
    "mutual_info": "filter",
    "random_k": "control",
    "none": "control",
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
                    selector: str = "ga", **kwargs):
    """Ortak giriş noktası: ``selector`` adına göre uygun seçiciyi çağır."""
    return get_selector(selector)(X, y, k, **kwargs)


if __name__ == "__main__":
    # Duman testi (sentetik; gerçek veri gerekmez): her seçici maske + info döndürmeli.
    rng = np.random.default_rng(0)
    n, p = 60, 40
    X = rng.normal(size=(n, p))
    y = X[:, :3] @ np.array([2.0, -1.0, 1.0]) + rng.normal(scale=0.5, size=n)
    g = np.repeat(np.arange(12), 5)
    for name, fn, kw in [
        ("rfe", select_rfe, {}), ("rfe_scaled", select_rfe, {"scale": True}),
        ("lasso_group", select_lasso, {"groups": g}),
        ("lasso_nogroup", select_lasso, {}),
        ("lasso_legacy", select_lasso, {"inner": "legacy"}),
        ("lasso_uncapped", select_lasso, {"groups": g, "k": None}),
        ("mi", select_mutual_info, {"n_neighbors": 5}),
        ("random_k", select_random, {"seed": 1}), ("none", select_none, {}),
    ]:
        kk = kw.pop("k", 10)
        kw.setdefault("seed", 0)
        m, info = fn(X, y, kk, return_info=True, **kw)
        assert m.dtype == bool and m.shape == (p,)
        print(f"{name:>15}: n_sel={int(m.sum()):>2}  info={ {k2: v for k2, v in info.items() if k2 in ('inner_cv', 'alpha', 'n_nonzero', 'truncated', 'n_convergence_warnings', 'n_elimination_rounds')} }")
    assert np.array_equal(SELECTORS["lasso"](X, y, 10, seed=0), select_lasso(X, y, 10, inner="legacy"))
    print("selectors.py duman testi OK")
