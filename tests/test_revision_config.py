"""config/revision.yaml — plan Bölüm 3 ile tutarlılık ve seçici varsayılanlarıyla uyum.

Önceden belirleme (§0.3): grid koşulardan önce sabitlenir; bu test, config'in
plandaki sayıları taşıdığını ve kodun varsayılanlarıyla çelişmediğini doğrular.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from src.m04_features import selectors as S
from src.m07_revision.engine import ARMS

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "revision.yaml")
                     .read_text(encoding="utf-8"))


def test_plan_numbers():
    assert CFG["E1_repeated"]["repeats"] == 20
    assert CFG["E1_repeated"]["uncertainty"]["group_bootstrap"]["B"] == 2000
    assert CFG["E2_mechanism"]["repeats"] == 20
    assert CFG["E3_sensitivity"]["repeats"] == 10
    ax = CFG["E3_sensitivity"]["axes"]
    assert ax["k"] == [5, 10, 20, 40] and ax["n_splits"] == [3, 5, 10]
    assert ax["rfe"]["alpha"] == [0.1, 1.0, 10.0] and ax["rfe"]["step"] == [0.05, 0.1, 0.2]
    assert ax["mutual_info"]["n_neighbors"] == [3, 5, 10]
    assert ax["lasso"]["inner"] == ["legacy", "group"]
    sn = CFG["E4_simulation"]["s_null"]
    assert (sn["n"], sn["p"], sn["k"], sn["repeats"]) == (204, 520, 20, 200)
    g = CFG["E4_simulation"]["r3_65_grid"]
    assert g["n"] == [50, 100, 204] and g["p"] == [150, 520] and g["k"] == [5, 20, 40]
    ss = CFG["E4_simulation"]["s_signal"]
    assert ss["population_r2"] == [0.3, 0.6] and ss["n"] == [72, 204] and ss["n_test"] == 5000
    assert ss["repeats"] == 100 and ss["n_true_features"] == 10
    assert CFG["E5_permutation"]["B"] == 200
    v = CFG["E6_ga"]["variants"]
    assert v["G1"]["repeats"] == 10 and v["G2"]["repeats"] == 5 and v["G3"]["repeats"] == 5
    assert v["G0"]["expected"] == {"apparent_r2_foldmean": 0.615, "nested_r2_foldmean": 0.269,
                                   "difference": 0.346}


def test_arms_known_to_engine():
    for exp in ("E1_repeated", "E2_mechanism", "E3_sensitivity"):
        assert set(CFG[exp]["arms"]) <= set(ARMS), exp


def test_selector_defaults_match_code():
    d = CFG["common"]["selectors"]
    sig = inspect.signature(S.select_lasso).parameters
    assert sig["inner"].default == d["lasso"]["inner"] == "group"
    assert sig["tol"].default == d["lasso"]["tol"]
    sig = inspect.signature(S.select_rfe).parameters
    assert (sig["alpha"].default, sig["step"].default, sig["scale"].default) == \
        (d["rfe"]["alpha"], d["rfe"]["step"], d["rfe"]["scale"])
    assert inspect.signature(S.select_mutual_info).parameters["n_neighbors"].default == \
        d["mutual_info"]["n_neighbors"]
