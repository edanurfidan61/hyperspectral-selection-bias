"""E2 — Mekanizma ayrıştırma (plan §3 E2) → R3-1,7,43,46,47,67,73,77.

E1 ile AYNI bölmeler (aynı random_state) ve AYNI maske önbelleği → P2/P3 maskeleri
yeniden hesaplanmaz. Yalnız PLS.

* Boyut-eşli ayrıştırma (R3-46):
  Δ_toplam = (P2 − size_matched) [örneklem boyutu bileşeni]
           + (size_matched − P3) [eşit boyutta sızıntı bileşeni].
* LASSO 2×2 (R3-7): {α: tüm veri, dış-train} × {destek: tüm veri, dış-train}.
  α'lar LassoCV'den (inner=group; önbellekteki P2/P3 seçimlerinin ``alpha``'sı),
  destek ``Lasso(alpha)`` ile (StandardScaler yalnız destek verisine fit), sonra
  aynı k-tavan kuralı; model dış-train'e fit, dış-test'te değerlendirilir.
  Köşegen hücreler P2/P3 maskelerini yeniden üretmeli (``matches_arm_mask``).
* Tahmin kararlılığı (R3-77): kollar arası OOF tahmin korelasyonu ve ortalama
  mutlak fark; fold maskeleri ile apparent maske arasında Jaccard; aynı dış-train'de
  fit edilen P2 ve P3 PLS katsayılarının ortak öznitelikler üzerinde kosinüs benzerliği.
* Bileşen sayısı dağılımı kol bazında (R3-47).
* Negatif farklar (R3-73): config'teki kombinasyon/seçiciler için fold düzeyi Δ,
  maske Jaccard'ı, bileşen sayıları yan yana.

Çıktılar (``outputs_rev1/E2_mechanism/``): decomposition.csv, lasso_2x2.csv,
stability.csv, components_by_arm.csv, negative_differences.csv,
fold_metrics.csv.gz, samples/folds.csv.gz, run_config.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.splits import splits_signature
from src.m04_features.selectors import SELECTOR_FAMILY
from src.m07_revision import common as C
from src.m07_revision.engine import (ArmsResult, MaskCache, _fit_predict, _metrics,
                                     evaluate_arms, folds_long, resolve_selector)

EXP = "E2_mechanism"
STABILITY_PAIRS = [("P2_apparent", "P3_nested"), ("P2_apparent", "size_matched_leaky"),
                   ("size_matched_leaky", "P3_nested"), ("P3_nested", "no_selection")]


def task_path(out_dir: Path, t: dict) -> Path:
    return out_dir / "_checkpoints" / "tasks" / f"{t['dataset']}__{t['target']}__{t['selector']}__r{t['repeat']:02d}.pkl"


def lasso_support(X: np.ndarray, y: np.ndarray, alpha: float, k: int | None,
                  max_iter: int, tol: float) -> np.ndarray:
    """Sabit α ile Lasso desteği + select_lasso ile AYNI k-tavan kuralı."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        coef = np.abs(Lasso(alpha=alpha, max_iter=max_iter, tol=tol).fit(Xs, y).coef_)
    nz = coef > 0
    if k is not None and (nz.sum() > k or nz.sum() == 0):
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[np.argsort(coef)[::-1][:k]] = True
        return mask
    return nz


def _lasso_2x2(data, splits, res: ArmsResult, kw: dict, k: int) -> list[dict]:
    X, y = data["X"], data["y"]
    f = res.folds
    p2 = f[f.arm == "P2_apparent"].sort_values("fold")
    p3 = f[f.arm == "P3_nested"].sort_values("fold")
    a_all = json.loads(p2.iloc[0].selector_info)["alpha"]
    m_p2 = C.bits_to_mask(p2.iloc[0]["mask"])
    rows = []
    cells = [("all_data", "all_data"), ("all_data", "outer_train"),
             ("outer_train", "all_data"), ("outer_train", "outer_train")]
    for a_src, s_src in cells:
        fold_r2, preds, match, nsel = [], np.full(len(y), np.nan), [], []
        for fi, (tr, te) in enumerate(splits):
            a_tr = json.loads(p3.iloc[fi].selector_info)["alpha"]
            alpha = a_all if a_src == "all_data" else a_tr
            rows_sup = np.arange(len(y)) if s_src == "all_data" else tr
            mask = lasso_support(X[rows_sup], y[rows_sup], alpha, k, kw["max_iter"], kw["tol"])
            pred, _, _ = _fit_predict("pls", None, X, y, tr, te, mask)
            preds[te] = pred
            fold_r2.append(_metrics(y[te], pred)["r2"])
            nsel.append(int(mask.sum()))
            if (a_src, s_src) == ("all_data", "all_data"):
                match.append(bool(np.array_equal(mask, m_p2)))
            elif (a_src, s_src) == ("outer_train", "outer_train"):
                match.append(bool(np.array_equal(mask, C.bits_to_mask(p3.iloc[fi]["mask"]))))
        rows.append({"alpha_source": a_src, "support_source": s_src,
                     "r2_foldmean": float(np.mean(fold_r2)), "r2_pooled": _metrics(y, preds)["r2"],
                     "n_selected_mean": float(np.mean(nsel)),
                     "matches_arm_mask": (float(np.mean(match)) if match else np.nan)})
    return rows


def _stability(res: ArmsResult) -> dict:
    pr, f = res.predictions, res.folds
    out = {}
    piv = pr.pivot_table(index="test_idx", columns="arm", values="y_pred")
    for a, b in STABILITY_PAIRS:
        if a in piv and b in piv:
            out[f"pred_corr__{a}__{b}"] = float(np.corrcoef(piv[a], piv[b])[0, 1])
            out[f"pred_meanabsdiff__{a}__{b}"] = float(np.mean(np.abs(piv[a] - piv[b])))
    m_app = C.bits_to_mask(f[f.arm == "P2_apparent"].iloc[0]["mask"])
    for arm in ("P3_nested", "size_matched_leaky"):
        ms = [C.bits_to_mask(s) for s in f[f.arm == arm].sort_values("fold")["mask"]]
        out[f"jaccard_vs_apparent__{arm}"] = float(np.mean([C.jaccard(m, m_app) for m in ms]))
        out[f"jaccard_pairwise_folds__{arm}"] = float(np.mean(
            [C.jaccard(ms[i], ms[j]) for i in range(len(ms)) for j in range(i + 1, len(ms))]))
    # PLS katsayı benzerliği: aynı dış-train'de P2 ve P3 modelleri, ortak öznitelikler
    cos, shared = [], []
    p2 = f[f.arm == "P2_apparent"].sort_values("fold")
    p3 = f[f.arm == "P3_nested"].sort_values("fold")
    for (_, r2), (_, r3) in zip(p2.iterrows(), p3.iterrows()):
        m2, m3 = C.bits_to_mask(r2["mask"]), C.bits_to_mask(r3["mask"])
        c2 = dict(zip(np.flatnonzero(m2), json.loads(r2["coef"])))
        c3 = dict(zip(np.flatnonzero(m3), json.loads(r3["coef"])))
        common = sorted(set(c2) & set(c3))
        shared.append(len(common))
        if len(common) >= 2:
            v2, v3 = np.array([c2[i] for i in common]), np.array([c3[i] for i in common])
            den = np.linalg.norm(v2) * np.linalg.norm(v3)
            cos.append(float(v2 @ v3 / den) if den > 0 else np.nan)
    out["pls_coef_cosine_shared__P2_P3"] = float(np.nanmean(cos)) if cos else np.nan
    out["n_shared_features_mean__P2_P3"] = float(np.mean(shared))
    return out


def run_task(t: dict, cfg: dict, out_dir: Path, mask_root: Path) -> dict:
    p = task_path(out_dir, t)
    if p.exists():
        return {"skipped": True, "seconds": 0.0}
    t0 = time.time()
    e = cfg[EXP]
    data = C.load_combo(t["dataset"], t["target"])
    splits = C.outer_splits(data, t["repeat"], e["split"]["n_splits"])
    kw = C.selector_kwargs(cfg, t["selector"])
    cache = MaskCache(mask_root)
    res = evaluate_arms(data["X"], data["y"], data["groups"], splits, "pls",
                        resolve_selector(t["selector"]), arms=tuple(e["arms"]), sel_kw=kw,
                        k=cfg["common"]["k"], repeat=t["repeat"], base_seed=cfg["common"]["base_seed"],
                        seed_mode="derived", selector_name=t["selector"],
                        sample_ids=data["sample_ids"], group_ids=data["group_ids"],
                        mask_cache=cache, cache_context={"dataset": t["dataset"], "target": t["target"]},
                        record_coef=True)
    out = {"task": t, "splits_signature": splits_signature(splits), "summary": res.summary,
           "contrasts": res.contrasts, "folds": res.folds, "predictions": res.predictions,
           "stability": _stability(res), "lasso_2x2": None}
    if t["selector"] == "lasso":
        out["lasso_2x2"] = _lasso_2x2(data, splits, res, kw, cfg["common"]["k"])
    out["seconds"] = round(time.time() - t0, 2)
    out["cache"] = {"hits": cache.hits, "misses": cache.misses}
    C.save_pickle(p, out)
    return {"skipped": False, "seconds": out["seconds"]}


def aggregate(out_dir: Path, cfg: dict) -> dict:
    import pickle
    tasks = [pickle.loads(f.read_bytes()) for f in sorted((out_dir / "_checkpoints" / "tasks").glob("*.pkl"))]
    if not tasks:
        raise RuntimeError(f"{out_dir}: görev checkpoint'i yok")
    dec, l22, stab, folds, samples = [], [], [], [], []
    for T in tasks:
        t = T["task"]
        ident = {"dataset": t["dataset"], "target": t["target"], "selector": t["selector"],
                 "family": SELECTOR_FAMILY.get(t["selector"], t["selector"]), "model": "pls",
                 "repeat": t["repeat"]}
        s = T["summary"].set_index("arm")
        c = T["contrasts"]
        dec.append({**ident,
                    **{f"{arm}.r2_foldmean": s.at[arm, "r2_foldmean"] for arm in s.index},
                    "total_apparent_nested_difference": c["apparent_nested_difference_r2_foldmean"],
                    "sample_size_component": c["sample_size_component_r2_foldmean"],
                    "equal_size_leakage_component": c["equal_size_leakage_component_r2_foldmean"],
                    "apparent_vs_random_k": c.get("apparent_vs_random_k_r2_foldmean"),
                    "random_k_vs_nested": c.get("random_k_vs_nested_r2_foldmean")})
        for r in T["lasso_2x2"] or []:
            l22.append({**ident, **r})
        stab.append({**ident, **T["stability"]})
        folds.append(T["folds"].drop(columns=["mask"]).assign(**ident))
        samples.append(folds_long(ArmsResult(T["summary"], T["contrasts"], T["folds"],
                                             T["predictions"])).assign(**ident))

    keys = ["dataset", "target", "selector", "family", "model"]

    def with_summary(df: pd.DataFrame, value_cols: list[str], extra_keys: list[str] = ()) -> pd.DataFrame:
        k = keys + list(extra_keys)
        summ = []
        for gk, gdf in df.groupby(k, sort=False):
            for col in value_cols:
                summ.append({**dict(zip(k, gk)), "quantity": col, **C.describe(gdf[col])})
        return pd.DataFrame(summ)

    dec = pd.DataFrame(dec)
    vcols = [c for c in dec.columns if c not in keys + ["repeat"]]
    with_summary(dec, vcols).to_csv(out_dir / "decomposition.csv", index=False)
    dec.to_csv(out_dir / "decomposition_by_repeat.csv", index=False)
    if l22:
        l22 = pd.DataFrame(l22)
        l22.to_csv(out_dir / "lasso_2x2_by_repeat.csv", index=False)
        with_summary(l22, ["r2_foldmean", "r2_pooled", "n_selected_mean", "matches_arm_mask"],
                     ["alpha_source", "support_source"]).to_csv(out_dir / "lasso_2x2.csv", index=False)
    stab = pd.DataFrame(stab)
    stab.to_csv(out_dir / "stability_by_repeat.csv", index=False)
    with_summary(stab, [c for c in stab.columns if c not in keys + ["repeat"]]) \
        .to_csv(out_dir / "stability.csv", index=False)

    folds = pd.concat(folds, ignore_index=True)
    folds.to_csv(out_dir / "fold_metrics.csv.gz", index=False, compression="gzip")
    f = folds.assign(n_components=pd.to_numeric(folds.n_components, errors="coerce"))
    comp = f.groupby(["dataset", "target", "selector", "arm"]).n_components \
        .apply(lambda v: pd.Series({**C.describe(v), **{f"count_{int(c)}": int((v == c).sum())
                                                        for c in sorted(v.dropna().unique())}})) \
        .unstack().reset_index()
    comp.to_csv(out_dir / "components_by_arm.csv", index=False)

    _negative_differences(folds, tasks, cfg).to_csv(out_dir / "negative_differences.csv", index=False)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    pd.concat(samples, ignore_index=True).to_csv(out_dir / "samples" / "folds.csv.gz",
                                                 index=False, compression="gzip")
    return {"n_tasks": len(tasks), "decomposition": dec}


def _negative_differences(folds: pd.DataFrame, tasks: list, cfg: dict) -> pd.DataFrame:
    """R3-73: config'teki (kombinasyon, seçici) için fold düzeyi Δ + maske kararlılığı + bileşen."""
    nd = cfg[EXP]["negative_differences"]
    want = {(c["dataset"], c["target"]) for c in nd["combos"]}
    sels = set(nd["selectors"])
    rows = []
    for T in tasks:
        t = T["task"]
        if (t["dataset"], t["target"]) not in want or t["selector"] not in sels:
            continue
        f = T["folds"]
        p2 = f[f.arm == "P2_apparent"].set_index("fold")
        p3 = f[f.arm == "P3_nested"].set_index("fold")
        m_app = C.bits_to_mask(p2.iloc[0]["mask"])
        for fold in p3.index:
            rows.append({"dataset": t["dataset"], "target": t["target"], "selector": t["selector"],
                         "repeat": t["repeat"], "fold": fold,
                         "r2_apparent": p2.at[fold, "r2"], "r2_nested": p3.at[fold, "r2"],
                         "fold_delta_r2": p2.at[fold, "r2"] - p3.at[fold, "r2"],
                         "jaccard_nested_vs_apparent_mask": C.jaccard(C.bits_to_mask(p3.at[fold, "mask"]), m_app),
                         "n_components_apparent": p2.at[fold, "n_components"],
                         "n_components_nested": p3.at[fold, "n_components"],
                         "n_test": p3.at[fold, "n_test"], "n_test_groups": p3.at[fold, "n_test_groups"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.m07_revision.run_e1e2 import main
    main(experiments=("E2",))
