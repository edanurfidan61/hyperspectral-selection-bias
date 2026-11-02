"""E6 GA birim testleri (sentetik, küçük bütçe)."""

from __future__ import annotations

import numpy as np
import pytest

from src.m07_revision.ga import Fitness, run_ga


def _data(seed=0, n_groups=15, per=4, p=25):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    X = rng.normal(size=(len(g), p))
    y = X[:, :3].sum(axis=1) + rng.normal(size=len(g))
    return X, y, g


@pytest.mark.parametrize("seed", [0, 7])
def test_revision_ga_equals_paper_ga(seed):
    """Revizyon GA (seri, önbellekli) = makaledeki run_ga_core (aynı maske)."""
    from src.m04_features.ga_feature_selection import run_ga_core
    X, y, g = _data(seed)
    ref = run_ga_core(X, y, g, model="pls", pop=14, ngen=3, seed=seed, n_jobs=1)
    m, _, _ = run_ga(X, y, g, pop=14, ngen=3, seed=seed)
    assert np.array_equal(ref, m)


def test_candidate_cache_invariance():
    X, y, g = _data(1)
    m1, i1, a1 = run_ga(X, y, g, pop=14, ngen=4, seed=3, candidate_cache=True)
    m2, i2, a2 = run_ga(X, y, g, pop=14, ngen=4, seed=3, candidate_cache=False)
    assert np.array_equal(m1, m2)
    assert i1["inner_fitness"] == i2["inner_fitness"]
    assert np.array_equal(a1["fitness_by_gen"], a2["fitness_by_gen"])
    assert i1["n_evaluations_computed"] <= i2["n_evaluations_computed"] == i1["n_evaluations_requested"]


def test_fitness_matches_paper_evaluate_individual():
    from src.m04_features import ga_feature_selection as G
    X, y, g = _data(2)
    G._init_worker(X, y, "pls", 0, g)
    f = Fitness(X, y, g, "pls", 0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        m = rng.random(X.shape[1]) < 0.4
        assert f(m) == G._evaluate_individual(m.astype(int).tolist())[0]
    assert f(np.zeros(X.shape[1], dtype=bool)) == 0.0                 # MIN_FEATURES cezası


def test_fixed_k_and_unclipped():
    X, y, g = _data(3)
    m, info, arch = run_ga(X, y, g, pop=12, ngen=3, seed=1, representation="fixed_k", k=6)
    assert m.sum() == 6 and (arch["size_by_gen"] == 6).all()
    f = Fitness(X, y, g, "pls", 0, clip=False)
    assert f(np.zeros(X.shape[1], dtype=bool)) == float("-inf")
    # kırpmasız: saf gürültü hedefte negatif R² kırpılmaz
    yn = np.random.default_rng(9).normal(size=len(y))
    fn = Fitness(X, yn, g, "pls", 0, clip=False)
    assert fn(np.ones(X.shape[1], dtype=bool)) < 0


def test_phase_a_key_matches_engine(tmp_path):
    """Aşama A'nın yazdığı anahtar, evaluate_arms'ın aradığı anahtarla aynı olmalı."""
    from src.core.splits import make_outer_splits
    from src.m07_revision.e6_ga import _cached_only, sel_kw_of, sel_seed
    from src.m07_revision.engine import MaskCache, evaluate_arms

    X, y, g = _data(4)
    spec = {"variant": "Gt", "pop": 8, "ngen": 2, "representation": "binary", "clip": True, "k": None,
            "seed_mode": "derived", "base_seed": 0}
    splits = make_outer_splits(len(y), g, "group", 5, 2)
    cache = MaskCache(tmp_path)
    ctx = {"dataset": "syn", "target": "y"}
    for f in [-1, 0, 1, 2, 3, 4]:
        rows = np.arange(len(y)) if f < 0 else splits[f][0]
        s = sel_seed(spec, 2, f)
        m, info, _ = run_ga(X[rows], y[rows], g[rows], pop=8, ngen=2, seed=s)
        cache.put(MaskCache.make_key(ctx, "ga", sel_kw_of(spec), None, rows, s, "pls"), m, info, {})
    res = evaluate_arms(X, y, g, splits, "pls", _cached_only, arms=("P2_apparent", "P3_nested"),
                        sel_kw=sel_kw_of(spec), k=None, repeat=2, base_seed=0, seed_mode="derived",
                        selector_name="ga", mask_cache=MaskCache(tmp_path), cache_context=ctx)
    assert set(res.summary.arm) == {"P2_apparent", "P3_nested"}      # _cached_only hiç çağrılmadı
