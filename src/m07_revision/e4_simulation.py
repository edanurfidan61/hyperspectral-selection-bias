"""E4 — Simülasyon (plan §3 E4) → R3-2,16,64,65,66,67,68.

Üç parça (``config/revision.yaml`` ``E4_simulation``):

S-null (y bağımsız)
    Üreteçler: iid N(0,1); grapevine-gerçekçi Ledoit–Wolf kovaryansı (standardize 520
    sütun); t₃ (aynı kovaryans). n=204, p=520, k=20. Seçiciler: Pearson top-k (makale),
    RFE, LASSO, MI, random_k. Kollar: P2 (apparent), P3 (nested), random_k → R3-67
    ayrıştırması Δ = (apparent − random_k) + (random_k − nested). Her tekrar eşli:
    aynı X, y, fold'lar tüm kollarda; seed'ler kaydedilir; eşli fark %95 CI (R3-68).

    KARŞILAŞTIRILABİLİRLİK (outputs3/data/null.pkl): tekrar i'nin verisi
    ``synthetic_null.make_null_data(204, 520, seed=1000+i)`` ile AYNI RNG akışıdır
    (iid için birebir aynı X ve y). LW ve t₃ üreteçleri aynı standart normal Z ve
    aynı y'yi kullanır (X = Z Lᵀ) → üreteçler arası da eşli. Ana hücrelerde bölme
    config'e göre ``KFold(5, shuffle=True, random_state=1000+i)``. Ek doğrulama
    görevi (``legacy_check``) makaledeki bölmeyle (her örnek kendi grubu →
    shuffle'sız GroupKFold, pooled R²) null.pkl'nin 200 değerini yeniden üretir.

R3-65 ızgarası (yalnız Pearson, iid)
    n ∈ {50,100,204} × p ∈ {150,520} × k ∈ {5,20,40}, 200 tekrar + analitik tabanlar:
    yarı-normal sıra istatistiklerinden beklenen top-k Σr² (r ~ N(0, 1/n)) ve seçimsiz
    CV R² beklentisi (−k/(n−k); ayrıca n_train ile). Analitik değerler birinci derece
    yaklaşımdır (PLS ≠ OLS); etiketlerinde öyle yazar.

S-signal (bilinen gerçek; R3-2, R3-16)
    X ~ N(0, Σ_LW); β: rastgele s=10 öznitelikte N(0,1); σ² popülasyon R² = ρ verecek
    şekilde. n ∈ {72, 204}; her tekrarda bağımsız test kümesi (N_test=5000). Ölçüm
    hatası ekseni: gözlenen y'ye ek gürültü (popülasyon var(y)'nin %0/10/20'si) —
    eğitim ve test y'sine. Ortak rastgele sayılar: tekrar i için Z, β, ε, gürültü bir kez
    çekilir; hücreler (n, ρ, gürültü) bunları ölçekler/keser → hücreler arası eşli.
    θ_n: tüm n'de seçilip eğitilen pipeline'ın test R²'si (P2 maskesi);
    θ_0.8n: her dış-train'de seçilip eğitilen pipeline'ların test R² ortalaması (P3
    maskeleri). Her iki θ hem gözlenen (gürültülü) hem gerçek (gürültüsüz) test y'sine
    karşı. Rapor: E[apparent − θ_n], E[nested − θ_0.8n], E[nested − θ_n],
    corr(Δ, apparent − θ_n).

Tekrar sayısı kuralı (config ``s_null.reduced_repeats_rule``): LASSO/MI için 100'e
düşme kararı zaman ölçümünden SONRA, tam koşudan ÖNCE ``--reduced-repeats`` ile
verilir ve ``repeat_decision.json``'a yazılır.

Çıktılar (``outputs_rev1/E4_simulation/``): s_null_summary.csv, s_null_by_repeat.csv,
null_legacy_reproduction.csv, grid_summary.csv, s_signal_summary.csv,
s_signal_by_repeat.csv, run_config.json (+ repeat_decision.json).

Çalıştırma (proje kökünden)::

    .venv\\Scripts\\python.exe -m src.m07_revision.e4_simulation --quick
    .venv\\Scripts\\python.exe -m src.m07_revision.e4_simulation --timing
    .venv\\Scripts\\python.exe -m src.m07_revision.e4_simulation [--reduced-repeats lasso,mutual_info]
    .venv\\Scripts\\python.exe -m src.m07_revision.e4_simulation --check
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.splits import make_outer_splits, splits_signature
from src.m07_revision import common as C
from src.m07_revision.engine import _fit_predict, _metrics, evaluate_arms, resolve_selector

EXP = "E4_simulation"
NULL_BASE_SEED = 1000           # makaledeki sentetik null ile aynı taban (seed = 1000 + i)
COST_ORDER = {"lasso": 0, "mutual_info": 1, "rfe": 2, "pearson_topk": 3, "random_k": 4}


# ---------------------------------------------------------------------------
# Seçiciler
# ---------------------------------------------------------------------------
def select_pearson(X, y, k=20, *, return_info=False, **kw):
    """Makaledeki filtre: |Pearson r| en yüksek k (synthetic_null._topk_corr_idx, birebir)."""
    from src.m06_selection_bias.synthetic_null import _topk_corr_idx

    mask = np.zeros(X.shape[1], dtype=bool)
    mask[_topk_corr_idx(X, y, int(k))] = True
    info = {"selector": "pearson_topk", "n_selected": int(mask.sum())}
    return (mask, info) if return_info else mask


def selector_fn(name: str):
    return select_pearson if name == "pearson_topk" else resolve_selector(name)


def sel_kwargs(cfg: dict, name: str) -> dict:
    if name in ("pearson_topk", "random_k"):
        return {}
    return C.selector_kwargs(cfg, name)


# ---------------------------------------------------------------------------
# Veri üreteçleri
# ---------------------------------------------------------------------------
_LW: dict = {}


def lw_factor() -> dict:
    """Grapevine standardize 520 sütunun Ledoit–Wolf kovaryansı ve Cholesky çarpanı (süreç başına bir kez)."""
    if "L" not in _LW:
        from sklearn.covariance import LedoitWolf

        X = C.load_combo("ryckewaert", "flavonol")["X"]
        sd = X.std(axis=0)
        Z = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
        lw = LedoitWolf().fit(Z)
        S = lw.covariance_
        _LW.update({"S": S, "L": np.linalg.cholesky(S), "shrinkage": float(lw.shrinkage_),
                    "p": S.shape[0]})
    return _LW


def null_data(generator: str, i: int, n: int = 204, p: int = 520) -> tuple[np.ndarray, np.ndarray]:
    """S-null verisi. iid: make_null_data(n, p, seed=1000+i) ile birebir aynı."""
    from src.m06_selection_bias.synthetic_null import make_null_data

    seed = NULL_BASE_SEED + int(i)
    if generator == "iid_normal":
        X, y, _ = make_null_data(n, p, seed=seed)
        return X, y
    # Aynı akış: önce Z (n×p), sonra y — make_null_data ile özdeş sıra
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    lw = lw_factor()
    if p != lw["p"]:
        raise ValueError(f"LW üreteci p={lw['p']} gerektirir (istenen {p})")
    X = Z @ lw["L"].T
    if generator == "ledoit_wolf_grapevine":
        return X, y
    if generator == "t3_ledoit_wolf":
        df = 3
        W = np.random.default_rng(np.random.SeedSequence([seed, df])).chisquare(df, size=n)
        # Çok değişkenli t: X / sqrt(W/df); kovaryansı Σ olsun diye sqrt((df−2)/df) ile ölçekle
        return X / np.sqrt(W / df)[:, None] * np.sqrt((df - 2) / df), y
    raise ValueError(f"Bilinmeyen üreteç {generator!r}")


def signal_draws(i: int, n_max: int, n_test: int, s: int) -> dict:
    """S-signal ortak rastgele sayıları (tekrar i): hücreler bunları ölçekler/keser."""
    lw = lw_factor()
    p = lw["p"]
    rng = np.random.default_rng(np.random.SeedSequence([NULL_BASE_SEED + int(i), 7]))
    support = rng.choice(p, size=s, replace=False)
    beta = rng.standard_normal(s)
    Xtr = rng.standard_normal((n_max, p)) @ lw["L"].T
    Xte = rng.standard_normal((n_test, p)) @ lw["L"].T
    return {"support": support, "beta": beta, "Xtr": Xtr, "Xte": Xte,
            "e_tr": rng.standard_normal(n_max), "e_te": rng.standard_normal(n_test),
            "m_tr": rng.standard_normal(n_max), "m_te": rng.standard_normal(n_test),
            "var_signal": float(beta @ lw["S"][np.ix_(support, support)] @ beta)}


def signal_cell(d: dict, n: int, rho: float, noise: float) -> dict:
    """Hücre verisi: ilk n eğitim satırı; σ² = var_signal(1−ρ)/ρ; ölçüm gürültüsü noise·var(y)."""
    sigma = np.sqrt(d["var_signal"] * (1 - rho) / rho)
    var_y = d["var_signal"] / rho
    tau = np.sqrt(noise * var_y)
    f_tr = d["Xtr"][:n][:, d["support"]] @ d["beta"]
    f_te = d["Xte"][:, d["support"]] @ d["beta"]
    y_tr_true = f_tr + sigma * d["e_tr"][:n]
    y_te_true = f_te + sigma * d["e_te"]
    return {"X": d["Xtr"][:n], "y": y_tr_true + tau * d["m_tr"][:n], "y_true": y_tr_true,
            "X_test": d["Xte"], "y_test_obs": y_te_true + tau * d["m_te"], "y_test_true": y_te_true}


# ---------------------------------------------------------------------------
# Analitik tabanlar (R3-65)
# ---------------------------------------------------------------------------
def analytic_baselines(n: int, p: int, k: int, n_splits: int = 5) -> dict:
    """Birinci derece yaklaşımlar (null, iid):

    * top-k |r| sıra istatistikleri: r ~ N(0, 1/n) → |r|_(i) ≈ Φ⁻¹((1+u_i)/2)/√n,
      u_i = 1 − (i − 0.375)/(p + 0.25). S = Σ_{i≤k} |r|²_(i) ≈ seçilen k özniteliğin
      (yaklaşık dik varsayımıyla) örneklem-içi R²'si.
    * seçimsiz CV R² ≈ −k/(n−k) (plan) ve n_train = n(K−1)/K ile −k/(n_train−k−1).
    * apparent CV R² ≈ S + (seçimsiz CV R²|n_train) — seçim kazancı + tahmin cezası
      (OLS sezgisi; PLS ≤10 bileşen kullanır).
    """
    from scipy.stats import norm

    i = np.arange(1, k + 1)
    u = 1 - (i - 0.375) / (p + 0.25)
    q = norm.ppf((1 + u) / 2) / np.sqrt(n)
    S = float((q ** 2).sum())
    n_tr = n * (n_splits - 1) / n_splits
    nosel_plan = -k / (n - k) if n > k else np.nan
    nosel_tr = -k / (n_tr - k - 1) if n_tr - k - 1 > 0 else np.nan
    return {"analytic_topk_sum_r2": S, "analytic_nosel_cv_r2_plan": nosel_plan,
            "analytic_nosel_cv_r2_ntrain": nosel_tr,
            "analytic_apparent_cv_r2_approx": S + nosel_tr if np.isfinite(nosel_tr) else np.nan}


# ---------------------------------------------------------------------------
# Görevler
# ---------------------------------------------------------------------------
def repeats_for(cfg: dict, part: str, selector: str, decision: dict | None) -> int:
    e = cfg[EXP]
    if part == "s_null":
        rule = e["s_null"]["reduced_repeats_rule"]
        if decision and selector in decision.get("reduced_selectors", []) and selector in rule["selectors"]:
            return int(rule["reduced_repeats"])
        return int(e["s_null"]["repeats"])
    if part == "s_signal":
        return int(e["s_signal"]["repeats"])
    return int(e["r3_65_grid"]["repeats"])


def build_tasks(cfg: dict, mode: str, decision: dict | None, parts: set) -> list[dict]:
    e = cfg[EXP]
    tasks: list[dict] = []
    sn, gr, ss = e["s_null"], e["r3_65_grid"], e["s_signal"]
    quick, timing = mode == "quick", mode == "timing"

    if "s_null" in parts:
        gens = sn["generators"][:1] + sn["generators"][1:2] if quick else sn["generators"]
        sels = ["pearson_topk", "rfe", "random_k"] if quick else sn["selectors"]
        for g in gens:
            for s in sels:
                R = 2 if quick else 1 if timing else repeats_for(cfg, "s_null", s, decision)
                for i in range(R):
                    tasks.append({"part": "s_null", "generator": g, "selector": s, "repeat": i,
                                  "p2_only": timing and s == "lasso"})
        tasks.append({"part": "legacy_check", "selector": "pearson_topk",
                      "n_repeats": 3 if quick else 5 if timing else int(sn["repeats"])})
    if "grid" in parts:
        for n in gr["n"]:
            for p in gr["p"]:
                for k in gr["k"]:
                    if quick and not (n == 50 and p == 150):
                        continue
                    tasks.append({"part": "grid", "n": int(n), "p": int(p), "k": int(k),
                                  "selector": "pearson_topk",
                                  "n_repeats": 3 if quick else 2 if timing else int(gr["repeats"])})
    if "s_signal" in parts:
        sels = ["pearson_topk", "random_k"] if quick else ss["selectors"]
        for n in ss["n"]:
            for rho in ss["population_r2"]:
                for noise in ss["measurement_noise_extra_var"]:
                    if (quick or timing) and not (rho == ss["population_r2"][-1] and noise == 0):
                        continue
                    for s in sels:
                        R = 2 if quick else 1 if timing else int(ss["repeats"])
                        for i in range(R):
                            tasks.append({"part": "s_signal", "n": int(n), "rho": float(rho),
                                          "noise": float(noise), "selector": s, "repeat": i,
                                          "p2_only": timing and s == "lasso"})
    tasks.sort(key=lambda t: (COST_ORDER.get(t["selector"], 9), t.get("repeat", 0)))
    return tasks


def task_id(t: dict) -> str:
    if t["part"] == "s_null":
        return f"s_null/{t['generator']}__{t['selector']}__r{t['repeat']:03d}"
    if t["part"] == "legacy_check":
        return f"legacy_check/iid_pearson__R{t['n_repeats']}"
    if t["part"] == "grid":
        return f"grid/n{t['n']}_p{t['p']}_k{t['k']}__R{t['n_repeats']}"
    return (f"s_signal/n{t['n']}_rho{t['rho']}_noise{t['noise']}__{t['selector']}"
            f"__r{t['repeat']:03d}")


def task_path(out_dir: Path, t: dict) -> Path:
    return out_dir / "_checkpoints" / f"{task_id(t)}{'__p2only' if t.get('p2_only') else ''}.pkl"


def _label(t: dict) -> str:
    return task_id(t)


def run_task(t: dict, cfg: dict, dirs: dict) -> dict:
    from threadpoolctl import threadpool_limits

    p = task_path(dirs["out"], t)
    if p.exists():
        return {"label": _label(t), "skipped": True, "seconds": 0.0}
    t0 = time.time()
    with threadpool_limits(limits=1):
        fn = {"s_null": _task_s_null, "legacy_check": _task_legacy, "grid": _task_grid,
              "s_signal": _task_s_signal}[t["part"]]
        out = fn(t, cfg)
    out["task"] = t
    out["seconds"] = round(time.time() - t0, 2)
    C.save_pickle(p, out)
    return {"label": _label(t), "skipped": False, "seconds": out["seconds"], "task": t}


def _arms_for(selector: str, p2_only: bool) -> tuple:
    if p2_only:
        return ("P2_apparent",)
    return ("P2_apparent", "P3_nested") if selector == "random_k" else \
        ("P2_apparent", "P3_nested", "random_k")


def _row(res, prefix: str = "") -> dict:
    s = res.summary.set_index("arm")
    out = {}
    for arm in s.index:
        for q in ("r2_foldmean", "r2_pooled", "rmse_foldmean"):
            out[f"{prefix}{arm}.{q}"] = float(s.at[arm, q])
    out.update({f"{prefix}{k}": v for k, v in res.contrasts.items()
                if k.endswith(("r2_foldmean", "r2_pooled"))})
    return out


def _task_s_null(t: dict, cfg: dict) -> dict:
    e = cfg[EXP]
    sn = e["s_null"]
    i = t["repeat"]
    X, y = null_data(t["generator"], i, int(sn["n"]), int(sn["p"]))
    splits = make_outer_splits(len(y), None, "random", int(e["split"]["n_splits"]), NULL_BASE_SEED + i)
    res = evaluate_arms(X, y, None, splits, e["model"], selector_fn(t["selector"]),
                        arms=_arms_for(t["selector"], t.get("p2_only", False)),
                        sel_kw=sel_kwargs(cfg, t["selector"]), k=int(sn["k"]), repeat=i,
                        base_seed=cfg["common"]["base_seed"], seed_mode="derived",
                        selector_name=t["selector"])
    sel_secs = [json.loads(s).get("selection_seconds") for s in res.folds["selector_info"]]
    return {"row": {"generator": t["generator"], "selector": t["selector"], "repeat": i,
                    "data_seed": NULL_BASE_SEED + i, "split_random_state": NULL_BASE_SEED + i,
                    "splits_signature": splits_signature(splits), **_row(res)},
            "selection_seconds": [s for s in sel_secs if s is not None]}


def _task_legacy(t: dict, cfg: dict) -> dict:
    """null.pkl yeniden üretimi: makaledeki bölme (her örnek kendi grubu, shuffle'sız GroupKFold)."""
    sn = cfg[EXP]["s_null"]
    pkl = C.ROOT / "outputs3" / "data" / "null.pkl"
    ref = pickle.loads(pkl.read_bytes()) if pkl.exists() else None
    rows = []
    for i in range(int(t["n_repeats"])):
        X, y = null_data("iid_normal", i, int(sn["n"]), int(sn["p"]))
        g = np.arange(len(y))
        splits = make_outer_splits(len(y), g, "group", 5, repeat=0, shuffle=False)
        res = evaluate_arms(X, y, g, splits, "pls", select_pearson, arms=("P2_apparent", "P3_nested"),
                            k=int(sn["k"]), repeat=i, base_seed=0, seed_mode="fixed",
                            selector_name="pearson_topk")
        s = res.summary.set_index("arm")
        r = {"repeat": i, "data_seed": NULL_BASE_SEED + i,
             "engine_apparent_pooled": float(s.at["P2_apparent", "r2_pooled"]),
             "engine_nested_pooled": float(s.at["P3_nested", "r2_pooled"]),
             "engine_apparent_foldmean": float(s.at["P2_apparent", "r2_foldmean"]),
             "engine_nested_foldmean": float(s.at["P3_nested", "r2_foldmean"])}
        if ref is not None and i < len(ref.apparent):
            r.update({"pkl_apparent": float(ref.apparent[i]), "pkl_nested": float(ref.nested[i])})
            r["absdiff_apparent"] = abs(r["pkl_apparent"] - r["engine_apparent_pooled"])
            r["absdiff_nested"] = abs(r["pkl_nested"] - r["engine_nested_pooled"])
        rows.append(r)
    return {"rows": rows, "reference": str(pkl.relative_to(C.ROOT).as_posix()) if ref else None}


def _task_grid(t: dict, cfg: dict) -> dict:
    from src.m06_selection_bias.synthetic_null import make_null_data

    e = cfg[EXP]
    rows = []
    for i in range(int(t["n_repeats"])):
        X, y, _ = make_null_data(t["n"], t["p"], seed=NULL_BASE_SEED + i)
        splits = make_outer_splits(t["n"], None, "random", int(e["split"]["n_splits"]), NULL_BASE_SEED + i)
        res = evaluate_arms(X, y, None, splits, e["model"], select_pearson,
                            arms=("P2_apparent", "P3_nested", "random_k"), k=t["k"], repeat=i,
                            base_seed=cfg["common"]["base_seed"], seed_mode="derived",
                            selector_name="pearson_topk")
        rows.append({"n": t["n"], "p": t["p"], "k": t["k"], "repeat": i, **_row(res)})
    return {"rows": rows}


def _task_s_signal(t: dict, cfg: dict) -> dict:
    e = cfg[EXP]
    ss = e["s_signal"]
    i = t["repeat"]
    d = signal_draws(i, max(int(v) for v in ss["n"]), int(ss["n_test"]), int(ss["n_true_features"]))
    c = signal_cell(d, t["n"], t["rho"], t["noise"])
    X, y = c["X"], c["y"]
    splits = make_outer_splits(len(y), None, "random", int(e["split"]["n_splits"]), NULL_BASE_SEED + i)
    p2_only = t.get("p2_only", False)
    res = evaluate_arms(X, y, None, splits, e["model"], selector_fn(t["selector"]),
                        arms=("P2_apparent",) if p2_only else ("P2_apparent", "P3_nested"),
                        sel_kw=sel_kwargs(cfg, t["selector"]), k=int(ss["k"]), repeat=i,
                        base_seed=cfg["common"]["base_seed"], seed_mode="derived",
                        selector_name=t["selector"])
    row = {"n": t["n"], "rho": t["rho"], "noise": t["noise"], "selector": t["selector"],
           "repeat": i, **_row(res)}
    if not p2_only:
        f = res.folds
        Xall = np.vstack([X, c["X_test"]])
        n = len(y)
        te_idx = np.arange(n, n + len(c["X_test"]))
        yall = np.concatenate([y, np.zeros(len(c["X_test"]))])

        def theta(mask, tr):
            pred, _, _ = _fit_predict(e["model"], None, Xall, yall, tr, te_idx, mask)
            return _metrics(c["y_test_obs"], pred)["r2"], _metrics(c["y_test_true"], pred)["r2"]

        m_full = C.bits_to_mask(f[f.arm == "P2_apparent"].iloc[0]["mask"])
        row["theta_n_obs"], row["theta_n_true"] = theta(m_full, np.arange(n))
        th = [theta(C.bits_to_mask(r["mask"]), np.asarray(tr))
              for (_, r), (tr, _te) in zip(f[f.arm == "P3_nested"].sort_values("fold").iterrows(), splits)]
        row["theta_08n_obs"] = float(np.mean([a for a, _ in th]))
        row["theta_08n_true"] = float(np.mean([b for _, b in th]))
        app, nes = row["P2_apparent.r2_foldmean"], row["P3_nested.r2_foldmean"]
        row.update({"app_minus_theta_n": app - row["theta_n_obs"],
                    "nes_minus_theta_08n": nes - row["theta_08n_obs"],
                    "nes_minus_theta_n": nes - row["theta_n_obs"],
                    "delta": app - nes})
    return {"row": row}


# ---------------------------------------------------------------------------
# Toplama
# ---------------------------------------------------------------------------
def _paired_ci(v: np.ndarray, ci: float = 0.95) -> dict:
    from scipy.stats import t as tdist
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    n = len(v)
    if n < 2:
        return {"n": n, "mean": float(v.mean()) if n else np.nan, "se": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "p2_5": np.nan, "p97_5": np.nan}
    se = v.std(ddof=1) / np.sqrt(n)
    h = tdist.ppf(0.5 + ci / 2, n - 1) * se
    return {"n": n, "mean": float(v.mean()), "se": float(se), "ci_lo": float(v.mean() - h),
            "ci_hi": float(v.mean() + h), "p2_5": float(np.quantile(v, 0.025)),
            "p97_5": float(np.quantile(v, 0.975))}


def aggregate(out_dir: Path, cfg: dict) -> None:
    ck = out_dir / "_checkpoints"
    load = lambda sub: [pickle.loads(f.read_bytes()) for f in sorted((ck / sub).glob("*.pkl"))  # noqa: E731
                        if not f.name.endswith("__p2only.pkl")]

    sn = [d["row"] for d in load("s_null")]
    if sn:
        df = pd.DataFrame(sn).sort_values(["generator", "selector", "repeat"])
        df.to_csv(out_dir / "s_null_by_repeat.csv", index=False)
        out = []
        for (g, s), x in df.groupby(["generator", "selector"], sort=False):
            r = {"generator": g, "selector": s, "n_repeats": len(x),
                 "apparent_r2_foldmean_mean": x["P2_apparent.r2_foldmean"].mean(),
                 "nested_r2_foldmean_mean": x["P3_nested.r2_foldmean"].mean(),
                 "apparent_r2_pooled_mean": x["P2_apparent.r2_pooled"].mean(),
                 "nested_r2_pooled_mean": x["P3_nested.r2_pooled"].mean()}
            for tag, col in (("delta_foldmean", "apparent_nested_difference_r2_foldmean"),
                             ("delta_pooled", "apparent_nested_difference_r2_pooled"),
                             ("app_minus_random_k", "apparent_vs_random_k_r2_foldmean"),
                             ("random_k_minus_nested", "random_k_vs_nested_r2_foldmean")):
                if col in x:
                    r.update({f"{tag}_{k}": v for k, v in _paired_ci(x[col].to_numpy()).items()
                              if k != "n"})
            if "random_k.r2_foldmean" in x:
                r["random_k_r2_foldmean_mean"] = x["random_k.r2_foldmean"].mean()
            out.append(r)
        pd.DataFrame(out).to_csv(out_dir / "s_null_summary.csv", index=False)

    leg = load("legacy_check")
    if leg:
        lg = pd.DataFrame(leg[-1]["rows"])
        lg.to_csv(out_dir / "null_legacy_reproduction.csv", index=False)

    gr = [r for d in load("grid") for r in d["rows"]]
    if gr:
        df = pd.DataFrame(gr)
        out = []
        for (n, p, k), x in df.groupby(["n", "p", "k"]):
            r = {"n": n, "p": p, "k": k, "n_repeats": len(x),
                 "apparent_r2_foldmean_mean": x["P2_apparent.r2_foldmean"].mean(),
                 "nested_r2_foldmean_mean": x["P3_nested.r2_foldmean"].mean(),
                 "random_k_r2_foldmean_mean": x["random_k.r2_foldmean"].mean(),
                 "apparent_r2_pooled_mean": x["P2_apparent.r2_pooled"].mean(),
                 "nested_r2_pooled_mean": x["P3_nested.r2_pooled"].mean(),
                 **{f"delta_foldmean_{kk}": v for kk, v in
                    _paired_ci(x["apparent_nested_difference_r2_foldmean"].to_numpy()).items() if kk != "n"},
                 **analytic_baselines(int(n), int(p), int(k), int(cfg[EXP]["split"]["n_splits"]))}
            out.append(r)
        pd.DataFrame(out).to_csv(out_dir / "grid_summary.csv", index=False)

    ss = [d["row"] for d in load("s_signal")]
    if ss:
        df = pd.DataFrame(ss).sort_values(["n", "rho", "noise", "selector", "repeat"])
        df.to_csv(out_dir / "s_signal_by_repeat.csv", index=False)
        out = []
        for key, x in df.groupby(["n", "rho", "noise", "selector"]):
            r = dict(zip(["n", "rho", "noise", "selector"], key))
            r["n_repeats"] = len(x)
            for col in ("P2_apparent.r2_foldmean", "P3_nested.r2_foldmean", "theta_n_obs",
                        "theta_08n_obs", "theta_n_true", "theta_08n_true"):
                if col in x:
                    r[f"{col}_mean"] = x[col].mean()
            for col in ("app_minus_theta_n", "nes_minus_theta_08n", "nes_minus_theta_n", "delta"):
                if col in x:
                    r.update({f"E[{col}]_{kk}": v for kk, v in _paired_ci(x[col].to_numpy()).items()
                              if kk in ("mean", "ci_lo", "ci_hi")})
            if {"delta", "app_minus_theta_n"} <= set(x) and len(x) > 2:
                r["corr_delta_vs_app_minus_theta_n"] = float(np.corrcoef(x["delta"], x["app_minus_theta_n"])[0, 1])
            out.append(r)
        pd.DataFrame(out).to_csv(out_dir / "s_signal_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Zaman tahmini
# ---------------------------------------------------------------------------
def estimate(results: list[dict], cfg: dict, n_jobs: int, decision: dict | None, parts: set) -> dict:
    """R=1 (LASSO: yalnız tüm-veri seçimi) ölçümünden tam koşu süresi.

    Görev türü başına ölçülen süre × tam koşudaki görev sayısı. LASSO görevleri:
    ölçülen tek seçim × (n_splits + 1) (P2 + P3 fold'ları; random_k kolu ucuz).
    """
    e = cfg[EXP]
    ns = int(e["split"]["n_splits"])
    meas: dict = {}
    for r in results:
        t = r.get("task")
        if not t or r.get("skipped"):
            continue
        key = (t["part"], t.get("generator"), t.get("n"), t["selector"])
        sec = r["seconds"]
        if t["part"] in ("grid", "legacy_check"):
            sec = sec / max(1, t["n_repeats"])
        if t.get("p2_only"):
            sec = sec * (ns + 1)
        meas.setdefault(key, []).append(sec)
    per = {k: float(np.mean(v)) for k, v in meas.items()}
    durations, cpu_by_part = [], {}
    for t in build_tasks(cfg, "full", decision, parts):
        key = (t["part"], t.get("generator"), t.get("n"), t["selector"])
        sec = per.get(key)
        if sec is None:            # s_signal: ρ/gürültü hücreleri ölçülen (n, seçici) ile aynı maliyet
            sec = next((v for k, v in per.items() if k[0] == t["part"] and k[2] == t.get("n")
                        and k[3] == t["selector"]), 0.0)
        if t["part"] in ("grid", "legacy_check"):
            sec = sec * t["n_repeats"]
        durations.append(sec)
        cpu_by_part[t["part"]] = cpu_by_part.get(t["part"], 0.0) + sec
    est = C.lpt_makespan(durations, n_jobs)
    return {"n_jobs": n_jobs, "per_task_seconds": {"|".join(map(str, k)): round(v, 2) for k, v in per.items()},
            "cpu_hours_by_part": {k: round(v / 3600, 2) for k, v in cpu_by_part.items()},
            "total_cpu_hours": round(sum(durations) / 3600, 2),
            "estimate_wall_hours_LPT": round(est / 3600, 2),
            "reduced_repeats_decision": decision}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _decision_path() -> Path:
    return C.ROOT / "outputs_rev1" / EXP / "repeat_decision.json"


def check(cfg: dict) -> int:
    dirs = C.experiment_dirs(EXP, "full")
    dp = _decision_path()
    decision = json.loads(dp.read_text(encoding="utf-8")) if dp.exists() else None
    tasks = build_tasks(cfg, "full", decision, {"s_null", "grid", "s_signal"})
    ok = C.check_tasks(EXP, tasks, lambda t: task_path(dirs["out"], t), dirs["done"])
    print("Tekrar kararı:", decision)
    lg = dirs["out"] / "null_legacy_reproduction.csv"
    if lg.exists():
        d = pd.read_csv(lg)
        mx = max(d["absdiff_apparent"].max(), d["absdiff_nested"].max())
        print(f"null.pkl yeniden üretimi: {len(d)} tekrar, max |fark| = {mx:.2e}")
        ok &= mx < 1e-9
    sp = dirs["out"] / "s_null_summary.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        print("\nS-null (Δ fold-mean, eşli %95 CI):")
        print(s[["generator", "selector", "n_repeats", "delta_foldmean_mean", "delta_foldmean_ci_lo",
                 "delta_foldmean_ci_hi"]].round(3).to_string(index=False))
    print("\nSONUÇ:", "TAMAM" if ok else "EKSİK / SORUN VAR")
    return 0 if ok else 1


def main() -> None:
    ap = C.base_argparser("E4 — simülasyon")
    ap.add_argument("--parts", type=str, default="s_null,grid,s_signal",
                    help="koşulacak parçalar (virgülle)")
    ap.add_argument("--reduced-repeats", type=str, default="",
                    help="config kuralı: bu seçiciler (lasso,mutual_info) S-null'da reduced_repeats "
                         "ile koşar; karar repeat_decision.json'a yazılır")
    args = ap.parse_args()
    cfg = C.load_config()
    if args.check:
        sys.exit(check(cfg))
    mode = "quick" if args.quick else "timing" if args.timing else "full"
    dirs = C.experiment_dirs(EXP, mode)
    parts = {p.strip() for p in args.parts.split(",") if p.strip()}
    if (mode == "timing" or (mode == "quick" and args.fresh)) and not args.aggregate_only \
            and dirs["out"].exists():
        shutil.rmtree(dirs["out"])
    log = C.setup_logging(dirs["log"])

    decision = None
    if mode == "full":
        dp = _decision_path()
        red = [s.strip() for s in args.reduced_repeats.split(",") if s.strip()]
        allowed = cfg[EXP]["s_null"]["reduced_repeats_rule"]["selectors"]
        if red:
            bad = set(red) - set(allowed)
            if bad:
                ap.error(f"--reduced-repeats yalnız {allowed} için (config kuralı); verilen: {sorted(bad)}")
            decision = {"reduced_selectors": red,
                        "reduced_repeats": cfg[EXP]["s_null"]["reduced_repeats_rule"]["reduced_repeats"],
                        "rule": cfg[EXP]["s_null"]["reduced_repeats_rule"],
                        "decided": datetime.now().isoformat(timespec="seconds"),
                        "basis": "outputs_rev1/_timing/E4_simulation_timing.json"}
            if dp.exists():
                old = json.loads(dp.read_text(encoding="utf-8"))
                if old.get("reduced_selectors") != red:
                    ap.error(f"Önceki karar {old.get('reduced_selectors')} ile çelişiyor; "
                             "karar koşu başladıktan sonra değiştirilemez (DEVIATIONS.md)")
            dp.parent.mkdir(parents=True, exist_ok=True)
            dp.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        elif dp.exists():
            decision = json.loads(dp.read_text(encoding="utf-8"))
    if args.aggregate_only:
        aggregate(dirs["out"], cfg)
        log.info("E4 tabloları yeniden üretildi → %s", dirs["out"])
        return
    tasks = build_tasks(cfg, mode, decision, parts)
    n_jobs = args.n_jobs or C.default_n_jobs()
    import functools

    from src.m07_revision.e4_simulation import run_task as entry
    C.run_experiment(
        name=EXP, mode=mode, tasks=tasks, entry=functools.partial(entry, cfg=cfg, dirs=dirs),
        aggregate=lambda: aggregate(dirs["out"], cfg), dirs=dirs, n_jobs=n_jobs,
        payload={"experiment": EXP, "parts": sorted(parts), "reduced_repeats_decision": decision,
                 "seeds": {"data": "1000 + i (iid = make_null_data birebir; LW/t3 aynı Z,y; "
                                   "s_signal SeedSequence([1000+i, 7]))",
                           "split": "KFold(5, shuffle=True, random_state=1000+i)",
                           "selector": "derive_seed(base_seed, i, fold, selector)"},
                 "config_blocks": {"common": cfg["common"], EXP: cfg[EXP]}},
        log=log, label=_label,
        estimate=lambda res: estimate(res, cfg, n_jobs, decision, parts))


if __name__ == "__main__":
    main()
