"""E3 / E4 / E5 birim testleri (hızlı, sentetik; gerçek veri gerekenler veri yoksa atlanır)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.m04_features import selectors as S

ROOT = Path(__file__).resolve().parents[1]
HAVE_DATA = (ROOT / "outputs" / "01_dataset" / "X.npy").exists()


def _data(seed=0, n_groups=20, per=4, p=30):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    X = rng.normal(size=(len(g), p))
    y = X[:, :3].sum(axis=1) + 0.5 * rng.normal(size=len(g))
    return X, y, g


# ---------------------------------------------------------------------------
# E3
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [2, 5, 10, None])
@pytest.mark.parametrize("inner", ["group", "legacy"])
def test_lasso_coef_cache_matches_select_lasso(tmp_path, k, inner):
    from src.m07_revision.e3_sensitivity import make_lasso_coef_cached
    X, y, g = _data()
    sel = make_lasso_coef_cached(tmp_path)
    ref, rinfo = S.select_lasso(X, y, k, groups=g, inner=inner, return_info=True)
    got, ginfo = sel(X, y, k, groups=g, inner=inner, return_info=True)
    again, ainfo = sel(X, y, k, groups=g, inner=inner, return_info=True)
    assert np.array_equal(ref, got) and np.array_equal(ref, again)
    assert ainfo["coef_cache"] == "hit"
    for key in ("truncated", "filled_from_zero", "k_cap", "alpha", "n_nonzero"):
        assert ginfo[key] == rinfo[key], key


def test_lasso_mask_rule_single_source():
    coef = np.array([0.0, 0.5, 0.1, 0.0, 0.9])
    m, tr, fl = S.lasso_mask_from_abs_coef(coef, 2)
    assert m.tolist() == [False, True, False, False, True] and tr and not fl
    m, tr, fl = S.lasso_mask_from_abs_coef(coef, 5)
    assert m.sum() == 3 and not tr
    m, tr, fl = S.lasso_mask_from_abs_coef(np.zeros(4), 2)
    assert m.sum() == 2 and fl


def test_apply_x_variant():
    from src.m07_revision.e3_sensitivity import apply_x_variant
    names = ["NDVI_mean", "snv_R400", "snv_R500", "snv_R990", "d1snv_R410", "d1snv_R700"]
    data = {"X": np.arange(12.0).reshape(2, 6), "feature_names": names}
    X, n = apply_x_variant(data, {"spectral_block_keep_nm": [420, 980]})
    assert n == ["NDVI_mean", "snv_R500", "d1snv_R700"]
    X, n = apply_x_variant(data, {"representation": "snv_spectrum_only"})
    assert n == ["snv_R400", "snv_R500", "snv_R990"]
    X, n = apply_x_variant(data, {"drop_first": 1, "drop_last": 2})
    assert n == names[1:4] and X.shape == (2, 3)
    X, n = apply_x_variant(data, None)
    assert X is data["X"]


def test_e3_cells_cover_config_and_grouping_first():
    from src.m07_revision.common import load_config
    from src.m07_revision.e3_sensitivity import build_cells, build_tasks
    cfg = load_config()
    axes = {c["axis"] for c in build_cells(cfg)}
    assert axes == {"grouping", "k", "band_trim", "rfe", "mutual_info", "lasso", "representation",
                    "n_splits", "pls_components"}
    tasks = build_tasks(cfg, "full")
    first_non = next(i for i, t in enumerate(tasks) if not t["priority"])
    assert all(t["priority"] for t in tasks[:first_non])
    assert not any(t["priority"] for t in tasks[first_non:])
    # LASSO varsayılan imzası k, uncapped, inner=group, n_splits=5, pls_components'i tek görevde toplar
    t = next(t for t in tasks if t["selector"] == "lasso" and t["dataset"] == "ryckewaert"
             and "k=5" in t["cells"])
    assert {"k=5", "k=10", "k=20", "k=40", "lasso=uncapped", "lasso=inner=group", "n_splits=5"} <= set(t["cells"])


def test_pls_inner_cv():
    from src.m07_revision.models import PLSInnerCV
    X, y, g = _data(p=12)
    m = PLSInnerCV().fit(X, y, groups=g)
    assert 1 <= m.n_components <= 10 and m.predict(X).shape[0] == len(y)
    assert set(m.inner_scores_) <= set(range(1, 11))


# ---------------------------------------------------------------------------
# E4
# ---------------------------------------------------------------------------
def test_null_iid_identical_to_paper_generator():
    from src.m06_selection_bias.synthetic_null import make_null_data
    from src.m07_revision.e4_simulation import null_data
    X, y = null_data("iid_normal", 3)
    X0, y0, _ = make_null_data(204, 520, seed=1003)
    assert np.array_equal(X, X0) and np.array_equal(y, y0)


@pytest.mark.skipif(not HAVE_DATA, reason="LW kovaryansı için 01_dataset gerekir")
def test_lw_generators_share_y_and_cov():
    from src.m07_revision.e4_simulation import lw_factor, null_data
    _, y_iid = null_data("iid_normal", 0)
    X_lw, y_lw = null_data("ledoit_wolf_grapevine", 0)
    X_t, y_t = null_data("t3_ledoit_wolf", 0)
    assert np.array_equal(y_iid, y_lw) and np.array_equal(y_iid, y_t)
    assert X_lw.shape == X_t.shape == (204, 520)
    assert np.isfinite(lw_factor()["shrinkage"])


def test_analytic_baselines_sane():
    from src.m07_revision.e4_simulation import analytic_baselines
    a = analytic_baselines(204, 520, 20)
    assert 0 < a["analytic_topk_sum_r2"] < 1
    assert a["analytic_nosel_cv_r2_plan"] == pytest.approx(-20 / 184)
    assert analytic_baselines(204, 520, 40)["analytic_topk_sum_r2"] > a["analytic_topk_sum_r2"]


# ---------------------------------------------------------------------------
# E5
# ---------------------------------------------------------------------------
def test_field_block_permutation_moves_whole_blocks():
    from src.m07_revision.e5_permutation import permute_y
    g = np.repeat(np.arange(24), 3)
    y = np.arange(72, dtype=float)
    yp, used = permute_y(y, "field_block", 0, {"groups": g, "dataset": "deep_patato",
                                               "target": "chlorophyll"}, 0)
    assert used.startswith("field_block")
    assert sorted(yp) == sorted(y)
    blocks = {tuple(y[g == u]) for u in np.unique(g)}
    assert {tuple(yp[g == u]) for u in np.unique(g)} == blocks      # bloklar bütün halde
    assert not np.array_equal(yp, y)
    yp2, _ = permute_y(y, "field_block", 0, {"groups": g}, 0)
    assert np.array_equal(yp, yp2)                                    # deterministik


def test_unequal_blocks_fall_back_within_field():
    from src.m07_revision.e5_permutation import permute_y
    g = np.array([0, 0, 0, 1, 1, 2])
    y = np.arange(6, dtype=float)
    yp, used = permute_y(y, "field_block", 1, {"groups": g}, 0)
    assert used.startswith("within_field")
    for u in np.unique(g):
        assert sorted(yp[g == u]) == sorted(y[g == u])


def test_free_permutation_is_permutation():
    from src.m07_revision.e5_permutation import permute_y
    y = np.arange(50, dtype=float)
    yp, used = permute_y(y, "free", 3, {}, 0)
    assert used == "free" and sorted(yp) == sorted(y)


@pytest.mark.skipif(not HAVE_DATA, reason="plot grupları için 01_dataset gerekir")
def test_within_plot_permutation_stays_in_plot():
    from src.m07_revision import common as C
    from src.m07_revision.e5_permutation import permute_y
    d = C.load_combo("ryckewaert", "flavonol")
    plots = C.load_combo("ryckewaert", "flavonol", group_level="plot")["groups"]
    yp, used = permute_y(d["y"], "within_plot", 0, d, 0)
    for pl in np.unique(plots):
        assert sorted(yp[plots == pl]) == sorted(d["y"][plots == pl])


def test_e5_task_counts():
    from src.m07_revision.common import load_config
    from src.m07_revision.e5_permutation import build_tasks
    cfg = load_config()
    B = cfg["E5_permutation"]["B"]
    tasks = build_tasks(cfg, "full")
    # ryck: 2 hedef × 3 seçici × (1 + 2 şema × B); potato: 3 × (1 + B)
    assert len(tasks) == 2 * 3 * (1 + 2 * B) + 3 * (1 + B)
