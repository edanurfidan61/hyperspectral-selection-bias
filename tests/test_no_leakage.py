"""Otomatik sızıntı testleri (R3-41, plan §2.4).

"Casus" (spy) sarmalayıcılar seçicinin / LassoCV'nin / GA fitness'ının gördüğü
satırları kaydeder. P3 (nested) kolunda hiçbir fold'da dış-test satırı bunlara
girmemelidir. Ek olarak: dış bölme geriye uyumu, eşli tasarım (kollar aynı
bölmeyi paylaşır), seed türetimi, seçici bilgi alanları ve (veri varsa) X ile
Dualex hedefleri arasında |r| = 1 sütun olmaması (R3-19).

Çalıştırma::

    .venv\\Scripts\\python.exe -m pytest -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.core.splits import derive_seed, make_outer_splits, splits_signature
from src.m04_features import selectors as S
from src.m07_revision.engine import ARMS, evaluate_arms

ROOT = Path(__file__).resolve().parents[1]
DS_DIR = ROOT / "outputs" / "01_dataset"


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def _data(n_groups: int = 15, per_group: int = 4, p: int = 12, seed: int = 0):
    """Gruplu sentetik veri; satırlar benzersiz (casus satır kimliğini bayttan çözer)."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per_group)
    X = rng.normal(size=(len(g), p))
    y = X[:, :3].sum(axis=1) + 0.5 * rng.normal(size=len(g))
    return X, y, g


class RowSpy:
    """Seçiciyi sarmalar; her çağrıda gördüğü satırların ORİJİNAL indekslerini kaydeder."""

    def __init__(self, fn, X_full: np.ndarray):
        self.fn = fn
        self.lookup = {row.tobytes(): i for i, row in enumerate(np.ascontiguousarray(X_full))}
        self.calls: list[set[int]] = []

    def rows_of(self, Xsub: np.ndarray) -> set[int]:
        return {self.lookup[r.tobytes()] for r in np.ascontiguousarray(Xsub)}

    def __call__(self, X, y, k, **kw):
        self.calls.append(self.rows_of(X))
        return self.fn(X, y, k, **kw)


def _p3_calls_are_train_only(spy: RowSpy, splits, first_call: int = 0):
    """P3: çağrı f'nin satırları == fold f'nin train'i ve test'le kesişmez."""
    for f, (tr, te) in enumerate(splits):
        seen = spy.calls[first_call + f]
        assert seen == set(tr.tolist()), f"fold {f}: seçici train dışı satır gördü"
        assert not (seen & set(te.tolist())), f"fold {f}: TEST satırı seçiciye girdi"


# ---------------------------------------------------------------------------
# P3'te seçici test satırı görmez (RFE, LASSO-group, MI, random)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,fn,kw", [
    ("rfe", S.select_rfe, {}),
    ("lasso_group", S.select_lasso, {"inner": "group"}),
    ("lasso_legacy", S.select_lasso, {"inner": "legacy"}),
    ("mutual_info", S.select_mutual_info, {}),
])
def test_nested_selector_sees_only_outer_train(name, fn, kw):
    X, y, g = _data()
    splits = make_outer_splits(len(y), g, "group", 5, repeat=3)
    spy = RowSpy(fn, X)
    evaluate_arms(X, y, g, splits, "pls", spy, arms=("P3_nested",), sel_kw=kw, k=5,
                  repeat=3, selector_name=name)
    assert len(spy.calls) == len(splits)
    _p3_calls_are_train_only(spy, splits)


def test_apparent_selector_sees_all_rows_once():
    """Kontrol: P2 casusu TÜM satırları görür (casusun çalıştığını kanıtlar)."""
    X, y, g = _data()
    splits = make_outer_splits(len(y), g, "group", 5, repeat=0)
    spy = RowSpy(S.select_rfe, X)
    evaluate_arms(X, y, g, splits, "pls", spy, arms=("P1_random", "P2_apparent"), k=5)
    assert len(spy.calls) == 1 and spy.calls[0] == set(range(len(y)))


def test_size_matched_leaky_group_count_and_flag():
    """size_matched: seçim alt kümesi dış-train ile aynı grup sayısında; test satırı sayısı kaydedilir."""
    X, y, g = _data()
    splits = make_outer_splits(len(y), g, "group", 5, repeat=1)
    spy = RowSpy(S.select_rfe, X)
    res = evaluate_arms(X, y, g, splits, "pls", spy, arms=("size_matched_leaky",), k=5, repeat=1)
    for f, (tr, te) in enumerate(splits):
        seen = np.array(sorted(spy.calls[f]))
        assert len(np.unique(g[seen])) == len(np.unique(g[tr]))
        info = json.loads(res.folds.iloc[f]["selector_info"])
        assert info["n_test_rows_in_selection"] == len(set(seen) & set(te.tolist()))


# ---------------------------------------------------------------------------
# LassoCV iç CV'si yalnız dış-train görür ve (inner="group") grupları bölmez
# ---------------------------------------------------------------------------
def test_lassocv_inner_cv_train_only_and_group_disjoint(monkeypatch):
    from sklearn.linear_model import LassoCV

    calls: list[dict] = []
    orig_fit = LassoCV.fit

    def spy_fit(self, X, y, *a, **kw):
        calls.append({"n": X.shape[0], "cv": self.cv})
        return orig_fit(self, X, y, *a, **kw)

    monkeypatch.setattr(LassoCV, "fit", spy_fit)
    X, y, g = _data(n_groups=20)
    splits = make_outer_splits(len(y), g, "group", 5, repeat=2)
    evaluate_arms(X, y, g, splits, "pls", S.select_lasso, arms=("P3_nested",),
                  sel_kw={"inner": "group"}, k=5, repeat=2)
    assert len(calls) == len(splits)
    for (tr, _te), c in zip(splits, calls):
        assert c["n"] == len(tr), "LassoCV dış-train dışında satır gördü"
        g_tr = g[tr]
        for itr, ite in c["cv"]:  # iç bölme dış-train içindeki konumlara göre
            assert not (set(g_tr[itr]) & set(g_tr[ite])), "iç CV aynı grubu böldü"


def test_lasso_legacy_inner_cv_is_plain_kfold():
    """Belgeleme: legacy LassoCV(cv=5) → düz KFold (grup yok) — R3-33'ün kaynağı."""
    from sklearn.model_selection import check_cv
    cv = check_cv(5, np.zeros(10), classifier=False)
    assert type(cv).__name__ == "KFold" and cv.shuffle is False


# ---------------------------------------------------------------------------
# GA fitness yalnız dış-train görür
# ---------------------------------------------------------------------------
def test_ga_fitness_sees_only_outer_train(monkeypatch):
    from src.m04_features import ga_feature_selection as gfs

    seen: list[set[int]] = []
    X, y, g = _data(n_groups=10, per_group=4, p=10)
    lookup = {r.tobytes(): i for i, r in enumerate(np.ascontiguousarray(X))}
    orig = gfs._init_worker

    def spy_init(Xw, yw, model, seed, groups=None):
        seen.append({lookup[r.tobytes()] for r in np.ascontiguousarray(Xw)})
        return orig(Xw, yw, model, seed, groups)

    monkeypatch.setattr(gfs, "_init_worker", spy_init)
    splits = make_outer_splits(len(y), g, "group", 5, repeat=0)
    evaluate_arms(X, y, g, splits, "pls", S.select_ga, arms=("P3_nested",),
                  sel_kw={"pop": 6, "ngen": 1, "n_jobs": 1}, k=None)
    assert len(seen) == len(splits)
    for (tr, te), s in zip(splits, seen):
        assert s == set(tr.tolist()) and not (s & set(te.tolist()))


# ---------------------------------------------------------------------------
# Bölme: geriye uyum, tekrar, eşli tasarım, seed
# ---------------------------------------------------------------------------
def test_outer_split_backcompat_with_make_cv_splitter():
    from src.core.cv import make_cv_splitter
    _, _, g = _data(n_groups=24, per_group=3)
    n = len(g)
    old = [te for _, te in make_cv_splitter(n_splits=5, task="regression", groups=g)
           .split(np.zeros(n), np.zeros(n), g)]
    new = [te for _, te in make_outer_splits(n, g, "group", 5, repeat=0, shuffle=False)]
    assert all(np.array_equal(a, b) for a, b in zip(old, new))


def test_group_splits_are_group_disjoint_and_repeats_differ():
    _, _, g = _data(n_groups=24, per_group=3)
    sigs = set()
    for r in range(4):
        sp = make_outer_splits(len(g), g, "group", 5, repeat=r)
        for tr, te in sp:
            assert not (set(g[tr]) & set(g[te]))
        assert sorted(np.concatenate([te for _, te in sp]).tolist()) == list(range(len(g)))
        sigs.add(splits_signature(sp))
    assert len(sigs) == 4
    assert len(make_outer_splits(len(g), g, "logo")) == 24


def test_arms_share_the_same_outer_split():
    X, y, g = _data()
    splits = make_outer_splits(len(y), g, "group", 5, repeat=5)
    res = evaluate_arms(X, y, g, splits, "pls", S.select_rfe,
                        arms=("P2_apparent", "P3_nested", "no_selection", "random_k"), k=5, repeat=5)
    p = res.predictions
    for f, (_tr, te) in enumerate(splits):
        for arm in ("P2_apparent", "P3_nested", "no_selection", "random_k"):
            got = sorted(p[(p.arm == arm) & (p.fold == f)].test_idx.tolist())
            assert got == sorted(te.tolist())
    # Her kolda her örnek tam bir kez tahmin edilir (pooled OOF geçerli)
    assert (p.groupby("arm").test_idx.nunique() == len(y)).all()


def test_derive_seed_deterministic_and_distinct():
    a = derive_seed(0, 1, 2, "lasso")
    assert a == derive_seed(0, 1, 2, "lasso")
    assert len({derive_seed(0, r, f, s) for r in range(3) for f in range(-1, 5)
                for s in ("rfe", "lasso", "mutual_info")}) == 3 * 6 * 3


def test_all_arms_run_and_contrast_signs():
    X, y, g = _data()
    splits = make_outer_splits(len(y), g, "group", 5, repeat=0)
    res = evaluate_arms(X, y, g, splits, "pls", S.select_rfe, arms=ARMS, k=5)
    assert set(res.summary.arm) == set(ARMS)
    s = res.summary.set_index("arm")
    c = res.contrasts
    assert np.isclose(c["apparent_nested_difference_r2_foldmean"],
                      s.at["P2_apparent", "r2_foldmean"] - s.at["P3_nested", "r2_foldmean"])
    # RMSE işareti: pozitif = apparent daha iyimser (R3-58)
    assert np.isclose(c["apparent_nested_difference_rmse_foldmean"],
                      s.at["P3_nested", "rmse_foldmean"] - s.at["P2_apparent", "rmse_foldmean"])


# ---------------------------------------------------------------------------
# Seçici sözleşmeleri
# ---------------------------------------------------------------------------
def test_selector_contracts():
    X, y, g = _data(n_groups=20, p=30)
    m, info = S.select_lasso(X, y, 2, groups=g, return_info=True)
    assert m.sum() <= 2 and info["inner"] == "group" and "alpha" in info
    assert info["truncated"] == (info["n_nonzero"] > 2)
    m_unc, info_unc = S.select_lasso(X, y, None, groups=g, return_info=True)
    assert info_unc["k_cap"] is None and m_unc.sum() == max(info_unc["n_nonzero"], 1)
    assert np.array_equal(S.select_random(X, y, 7, seed=3), S.select_random(X, y, 7, seed=3))
    assert S.select_random(X, y, 7, seed=3).sum() == 7
    assert S.select_none(X, y).all()
    assert S.select_rfe(X, y, 5, scale=True).sum() == 5
    assert S.select_mutual_info(X, y, 5, n_neighbors=5).sum() == 5


def test_registry_lasso_is_legacy_for_m06():
    """m06 hattı (get_selector('lasso')) yayımlanan legacy davranışı korur."""
    X, y, _ = _data(p=20)
    assert np.array_equal(S.get_selector("lasso")(X, y, 5, seed=0),
                          S.select_lasso(X, y, 5, inner="legacy"))


# ---------------------------------------------------------------------------
# R3-19: X içinde hedef sütunu yok (veri varsa)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (DS_DIR / "X.npy").exists(), reason="01_dataset yok (veri commitlenmez)")
def test_no_target_column_in_X():
    X = np.load(DS_DIR / "X.npy")
    for t in ("y_flav.npy", "y_chl.npy", "y_nbi.npy"):
        y = np.load(DS_DIR / t)
        sd = X.std(axis=0)
        r = np.array([np.corrcoef(X[:, j], y)[0, 1] if sd[j] > 0 else 0.0 for j in range(X.shape[1])])
        assert np.nanmax(np.abs(r)) < 0.999, f"{t}: |r|≈1 sütun var"


@pytest.mark.skipif(not (DS_DIR / "feature_names.json").exists(), reason="01_dataset yok")
def test_feature_names_have_no_dualex_targets():
    import re
    fn = json.loads((DS_DIR / "feature_names.json").read_text(encoding="utf-8"))
    bad = [n for n in fn if re.search(r"dualex|^(flav|chl|nbi)$|nbi_dualex", n, re.I)]
    assert not bad, bad
