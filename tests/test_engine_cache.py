"""Maske önbelleği ve eşli grup bootstrap testleri (E1/E2 altyapısı).

* Önbellekten okunan maske, hesaplanan maskeyle aynı sonucu verir.
* PLS ve LightGBM aynı maskeyi paylaşır (ikinci model yalnız önbellek isabeti).
* Seed'den bağımsız seçicide (RFE) tüm-veri maskesi tekrarlar arasında paylaşılır;
  seed'e bağlı seçicide (MI) paylaşılmaz.
* Grup bootstrap: aynı tahminlerde Δ = 0; ağırlıklı R² açık birleştirmeye eşit.
"""

from __future__ import annotations

import numpy as np

from src.core.splits import make_outer_splits
from src.m04_features import selectors as S
from src.m07_revision import common as C
from src.m07_revision.engine import MaskCache, evaluate_arms


def _data(seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(15), 4)
    X = rng.normal(size=(len(g), 12))
    y = X[:, :3].sum(axis=1) + 0.5 * rng.normal(size=len(g))
    return X, y, g


def _run(X, y, g, cache, model="pls", repeat=0, sel=S.select_rfe, name="rfe"):
    sp = make_outer_splits(len(y), g, "group", 5, repeat=repeat)
    return evaluate_arms(X, y, g, sp, model, sel, arms=("P2_apparent", "P3_nested"), k=4,
                         repeat=repeat, selector_name=name, mask_cache=cache,
                         cache_context={"dataset": "syn", "target": "y"})


def test_cache_hit_reproduces_results(tmp_path):
    X, y, g = _data()
    c = MaskCache(tmp_path)
    a = _run(X, y, g, c)
    assert c.hits == 0 and c.misses == 6
    b = _run(X, y, g, c)
    assert c.hits == 6
    assert np.allclose(a.summary[["r2_foldmean", "r2_pooled"]], b.summary[["r2_foldmean", "r2_pooled"]])
    assert (a.folds["mask"] == b.folds["mask"]).all()


def test_models_share_masks(tmp_path):
    X, y, g = _data()
    c = MaskCache(tmp_path)
    p = _run(X, y, g, c, model="pls")
    m0 = c.misses
    lg = _run(X, y, g, c, model="lightgbm")
    assert c.misses == m0, "LightGBM yeni seçim hesapladı"
    assert (p.folds["mask"].to_numpy() == lg.folds["mask"].to_numpy()).all()


def test_seed_free_full_mask_shared_across_repeats(tmp_path):
    X, y, g = _data()
    c = MaskCache(tmp_path)
    _run(X, y, g, c, repeat=0)
    h0 = c.hits
    _run(X, y, g, c, repeat=1)
    assert c.hits == h0 + 1, "RFE tüm-veri maskesi tekrar 1'de önbellekten gelmeli"
    c2 = MaskCache(tmp_path / "mi")
    _run(X, y, g, c2, repeat=0, sel=S.select_mutual_info, name="mutual_info")
    _run(X, y, g, c2, repeat=1, sel=S.select_mutual_info, name="mutual_info")
    assert c2.hits == 0, "MI seed'e bağlı — tekrarlar arası paylaşılmamalı"


def test_group_bootstrap_paired():
    rng = np.random.default_rng(1)
    y = rng.normal(size=30)
    p = y + rng.normal(scale=0.5, size=30)
    g = np.repeat(np.arange(10), 3)
    b = C.group_bootstrap_paired(y, p, p, g, B=200, seed=0)
    assert b["delta_r2_pooled"] == 0 and b["delta_r2_lo"] == 0 and b["delta_r2_hi"] == 0
    # Ağırlıklı R² == grupları açıkça birleştirerek hesaplanan R²
    w = np.zeros((1, 10))
    w[0, 0], w[0, 3], w[0, 7] = 2, 1, 1          # grup 0 iki kez çekildi
    W = w[:, np.repeat(np.arange(10), 3)]
    idx = np.concatenate([np.flatnonzero(g == k) for k in (0, 0, 3, 7)])
    explicit = 1 - ((y[idx] - p[idx]) ** 2).sum() / ((y[idx] - y[idx].mean()) ** 2).sum()
    assert np.isclose(C._weighted_r2(y, p, W)[0], explicit)
    assert C.group_bootstrap_paired(y, p, y, g, B=50, seed=3) == \
        C.group_bootstrap_paired(y, p, y, g, B=50, seed=3)
