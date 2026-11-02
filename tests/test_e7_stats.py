"""E7 — istatistik yardımcıları (hızlı; model koşusu yok)."""

from __future__ import annotations

import json
from itertools import permutations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.m07_revision import e7_stats_export as E7
from src.m07_revision.common import load_config


def test_exact_unit_permutation_matches_brute_force():
    """Paylaşılan birimli tam permütasyon = elle sayılan tüm atamalar."""
    rng = np.random.default_rng(3)
    x_unit = rng.normal(size=5)
    unit_of_row = np.array([0, 1, 2, 3, 4, 0, 2])          # iki birim ikişer satırda (paylaşılan maske)
    y = rng.normal(size=len(unit_of_row))
    r = E7.exact_unit_permutation_spearman(x_unit, y, unit_of_row)
    obs = spearmanr(x_unit[unit_of_row], y).statistic
    rhos = np.array([spearmanr(x_unit[list(p)][unit_of_row], y).statistic for p in permutations(range(5))])
    assert r["n_perm"] == 120
    assert r["rho"] == pytest.approx(obs)
    assert r["p_perm_two_sided"] == pytest.approx(np.mean(np.abs(rhos) >= abs(obs) - 1e-12))
    assert r["p_perm_less"] == pytest.approx(np.mean(rhos <= obs + 1e-12))


def test_exact_permutation_one_row_per_unit_is_classic_exact_test():
    """Birim = satır iken klasik tam permütasyon; mükemmel ters sıra → p_less = 1/n!."""
    x = np.arange(6, dtype=float)
    y = -x
    r = E7.exact_unit_permutation_spearman(x, y, np.arange(6))
    assert r["rho"] == pytest.approx(-1.0)
    assert r["p_perm_less"] == pytest.approx(1 / 720)
    assert r["p_perm_two_sided"] == pytest.approx(2 / 720)


def test_overlap_candidates_distinguish_lasso_case():
    """|A| = 17, |N| = 13 (ryckewaert/chlorophyll LASSO tipi): adaylar farklı değer vermeli."""
    i, a, n, u = 8.0, 17.0, 13.0, 22.0
    vals = {k: fn(i, a, n, u) for k, (_, fn) in E7.OVERLAP_CANDIDATES.items()}
    assert vals["inter_over_kA"] == pytest.approx(8 / 17)
    assert vals["inter_over_max"] == pytest.approx(8 / 17)        # burada |A| = max → aynı
    assert vals["inter_over_kN"] == pytest.approx(8 / 13)
    assert vals["jaccard"] == pytest.approx(8 / 22)
    assert vals["inter_over_k"] == pytest.approx(8 / 20)


def test_seed_wilcoxon_reproduces_outputs3():
    """Fold-mean seed-Wilcoxon p ve BH p'leri outputs3 stats_summary ile birebir; aile = 8."""
    df = E7.seed_wilcoxon(load_config())
    fm = df[df.aggregation == "foldmean"]
    assert len(fm) == 16
    assert (fm.reproduces_outputs3.astype(str) == "True").all()
    assert int(fm.bh_family_size.max()) == 8
    assert fm[~fm.stochastic_selector].p_unadjusted.isna().all()      # LASSO/RFE: tanımsız
    hl = fm[(fm.dataset == "ryckewaert") & (fm.target == "flavonol") & (fm.model == "pls") & (fm.selector == "ga")]
    assert hl.difference_mean.iloc[0] == pytest.approx(0.346, abs=5e-4)
    assert "cliffs_delta" not in df.columns


def test_verify_expected_detects_mismatch(tmp_path):
    pd.DataFrame({"k": ["a", "b"], "v": [1.0, 2.0]}).to_csv(tmp_path / "t.csv", index=False)
    spec = {"checks": [
        {"id": "ok", "file": "t.csv", "where": {"k": "a"}, "column": "v", "expected": 1.0,
         "tolerance": 1e-9, "source": "x"},
        {"id": "bad", "file": "t.csv", "where": {"k": "b"}, "column": "v", "expected": 2.1,
         "tolerance": 1e-3, "source": "x"},
        {"id": "missing", "file": "t.csv", "where": {"k": "zzz"}, "column": "v", "expected": 1.0,
         "tolerance": 1e-3, "source": "x"},
    ]}
    p = tmp_path / "ev.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    r = E7.verify_expected(p, root=tmp_path).set_index("id")
    assert r.at["ok", "ok"] and not r.at["bad", "ok"] and not r.at["missing", "ok"]
    assert "0 satır" in r.at["missing", "error"]


def test_analysis_roles_cover_primary_and_ids_unique():
    ar = E7.analysis_roles()
    assert ar.id.is_unique
    assert (ar.role.str.startswith("primary")).sum() >= 4
    banned = ("biased", "honest", "unbiased")
    text = " ".join(ar.astype(str).sum(axis=1)).lower()
    assert not any(b in text.split() for b in banned)
