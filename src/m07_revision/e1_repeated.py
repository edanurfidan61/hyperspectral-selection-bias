"""E1 — Tekrarlı eşli grup-CV (plan §3 E1) → R3-4,5,42,52,55,57,62,79,80,85.

Görev birimi: (repeat, dataset, target, selector). Her görevde seçici maskeleri
BİR KEZ hesaplanır (disk önbelleği, ``outputs_rev1/_checkpoints/masks``) ve PLS
ile LightGBM aynı maskeleri kullanır. Kollar: P1, P2, P3, no_selection, random_k.
Bölme: ``GroupKFold(5, shuffle=True, random_state=repeat)`` — E2 ve E6-G1 ile ortak
(:func:`src.m07_revision.common.outer_splits`).

Potato ek: LeaveOneGroupOut (24 Field) yalnız repeat 0'da (deterministik bölme);
test fold'u 3 örnek → yalnız pooled R² raporlanır.

Belirsizlik:
  (a) tekrarlar arası dağılım — ortalama, medyan, 2.5–97.5 persentil;
  (b) eşli grup bootstrap %95 CI (B=2000) her tekrarın pooled OOF tahminlerinde,
      apparent ve nested AYNI indekslerle; tekrarlar boyunca medyan CI.
      Varsayım: fit edilmiş modeller ve maskeler sabit (koşullu bootstrap).

Çıktılar (``outputs_rev1/E1_repeated/``):
  summary_by_combo.csv     kombinasyon × seçici × model × nicelik: tekrarlar arası özet
  by_family.csv            veri–hedef × seçici ailesi × model: medyan Δ (R3-5)
  delta_ci.csv             tekrar başına bootstrap CI + tekrarlar boyunca medyan CI (R3-62)
  lasso_truncation.csv     LASSO k-tavanı kesme / sıfırdan doldurma sayıları (R3-35)
  components.csv           PLS bileşen sayısı dağılımı, kol bazında (R3-47)
  logo_potato.csv          potato LOGO pooled R² (yalnız repeat 0)
  fold_metrics.csv.gz      (tekrar, fold, kol) metrikleri — örnek kimliği yok
  splits_signatures.csv    tekrar başına bölme imzası (E2/E6 eşliğini doğrulamak için)
  samples/folds.csv.gz     örnek düzeyi fold kaydı (git-ignored)
  run_config.json

Çalıştırma: ortak koşucu üzerinden (E1+E2 aynı maskeleri paylaşır)::

    python -m src.m07_revision.run_e1e2 [--quick | --timing | --check]
    python -m src.m07_revision.e1_repeated [--quick]     # yalnız E1
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.splits import derive_seed, make_outer_splits, splits_signature
from src.m04_features.selectors import SELECTOR_FAMILY
from src.m07_revision import common as C
from src.m07_revision.engine import MaskCache, evaluate_arms, folds_long, resolve_selector

EXP = "E1_repeated"


def task_path(out_dir: Path, t: dict) -> Path:
    return out_dir / "_checkpoints" / "tasks" / f"{t['dataset']}__{t['target']}__{t['selector']}__r{t['repeat']:02d}.pkl"


def run_task(t: dict, cfg: dict, out_dir: Path, mask_root: Path) -> dict:
    """Tek (repeat, dataset, target, selector) görevi — E1 tüm modeller (+ potato LOGO r0)."""
    p = task_path(out_dir, t)
    if p.exists():
        return {"skipped": True, "seconds": 0.0}
    t0 = time.time()
    e = cfg[EXP]
    data = C.load_combo(t["dataset"], t["target"])
    splits = C.outer_splits(data, t["repeat"], e["split"]["n_splits"])
    sel = resolve_selector(t["selector"])
    kw = C.selector_kwargs(cfg, t["selector"])
    cache = MaskCache(mask_root)
    ctx = {"dataset": t["dataset"], "target": t["target"]}
    base = cfg["common"]["base_seed"]
    B = e["uncertainty"]["group_bootstrap"]["B"]
    out = {"task": t, "splits_signature": splits_signature(splits), "models": {}, "logo": {}}
    for model in t["models"]:
        res = evaluate_arms(data["X"], data["y"], data["groups"], splits, model, sel,
                            arms=tuple(e["arms"]), sel_kw=kw, k=cfg["common"]["k"],
                            repeat=t["repeat"], base_seed=base, seed_mode="derived",
                            selector_name=t["selector"], sample_ids=data["sample_ids"],
                            group_ids=data["group_ids"], mask_cache=cache, cache_context=ctx)
        pr = res.predictions
        a = pr[pr.arm == "P2_apparent"].sort_values("test_idx")
        b = pr[pr.arm == "P3_nested"].sort_values("test_idx")
        boot = C.group_bootstrap_paired(a.y_true.to_numpy(), a.y_pred.to_numpy(), b.y_pred.to_numpy(),
                                        a.group_id.to_numpy(), B,
                                        derive_seed(base, t["repeat"], -1, "bootstrap"))
        out["models"][model] = {"summary": res.summary, "contrasts": res.contrasts,
                                "folds": res.folds, "predictions": res.predictions,
                                "bootstrap": boot, "meta": res.meta}
        # Potato LOGO — deterministik bölme, yalnız repeat 0
        if (t["dataset"] == "deep_patato" and t["repeat"] == 0 and e["potato_logo"]["enabled"]):
            lsp = make_outer_splits(data["n"], data["groups"], "logo")
            lres = evaluate_arms(data["X"], data["y"], data["groups"], lsp, model, sel,
                                 arms=("P2_apparent", "P3_nested", "no_selection", "random_k"),
                                 sel_kw=kw, k=cfg["common"]["k"], repeat=0, base_seed=base,
                                 selector_name=t["selector"], sample_ids=data["sample_ids"],
                                 group_ids=data["group_ids"], mask_cache=cache, cache_context=ctx)
            out["logo"][model] = {"summary": lres.summary, "contrasts": lres.contrasts,
                                  "folds": lres.folds, "predictions": lres.predictions}
    out["seconds"] = round(time.time() - t0, 2)
    out["cache"] = {"hits": cache.hits, "misses": cache.misses}
    C.save_pickle(p, out)
    return {"skipped": False, "seconds": out["seconds"]}


# ---------------------------------------------------------------------------
# Toplama
# ---------------------------------------------------------------------------
def _load_tasks(out_dir: Path) -> list[dict]:
    import pickle
    return [pickle.loads(f.read_bytes()) for f in sorted((out_dir / "_checkpoints" / "tasks").glob("*.pkl"))]


def aggregate(out_dir: Path, cfg: dict) -> dict:
    tasks = _load_tasks(out_dir)
    if not tasks:
        raise RuntimeError(f"{out_dir}: görev checkpoint'i yok")
    long_rows, ci_rows, fold_frames, sample_frames, logo_rows, sig_rows = [], [], [], [], [], []
    for T in tasks:
        t = T["task"]
        ident = {"dataset": t["dataset"], "target": t["target"], "selector": t["selector"],
                 "family": SELECTOR_FAMILY.get(t["selector"], t["selector"])}
        sig_rows.append({**ident, "repeat": t["repeat"], "splits_signature": T["splits_signature"]})
        for model, R in T["models"].items():
            base = {**ident, "model": model, "repeat": t["repeat"]}
            for _, s in R["summary"].iterrows():
                for q in ("r2_foldmean", "r2_pooled", "rmse_foldmean", "rmse_pooled",
                          "mae_foldmean", "mae_pooled", "n_selected_mean", "n_components_mean"):
                    long_rows.append({**base, "quantity": f"{s['arm']}.{q}", "value": s[q]})
            for q, v in R["contrasts"].items():
                long_rows.append({**base, "quantity": q, "value": v})
            ci_rows.append({**base, **R["bootstrap"]})
            fd = R["folds"].drop(columns=["mask"]).assign(**base)
            fold_frames.append(fd)
            sample_frames.append(folds_long(_as_res(R)).assign(**base))
        for model, L in T.get("logo", {}).items():
            s = L["summary"].set_index("arm")
            for arm in s.index:
                logo_rows.append({**ident, "model": model, "arm": arm, "n_folds": int(s.at[arm, "n_folds"]),
                                  "r2_pooled": s.at[arm, "r2_pooled"], "rmse_pooled": s.at[arm, "rmse_pooled"],
                                  "mae_pooled": s.at[arm, "mae_pooled"]})
            logo_rows.append({**ident, "model": model, "arm": "apparent_nested_difference",
                              "r2_pooled": L["contrasts"]["apparent_nested_difference_r2_pooled"],
                              "rmse_pooled": L["contrasts"]["apparent_nested_difference_rmse_pooled"]})

    long = pd.DataFrame(long_rows)
    keys = ["dataset", "target", "selector", "family", "model", "quantity"]
    summ = (long.groupby(keys, sort=False)["value"].apply(lambda v: pd.Series(C.describe(v)))
            .unstack().reset_index())
    summ["n"] = summ["n"].astype(int)
    summ.insert(5, "split", "GroupKFold(5, shuffle=True, random_state=repeat)")
    summ.to_csv(out_dir / "summary_by_combo.csv", index=False)

    d = long[long.quantity == "apparent_nested_difference_r2_foldmean"]
    fam = (d.groupby(["dataset", "target", "family", "model"])["value"]
           .apply(lambda v: pd.Series(C.describe(v))).unstack().reset_index())
    fam["n"] = fam["n"].astype(int)
    fam.insert(4, "quantity", "apparent_nested_difference_r2_foldmean")
    fam.to_csv(out_dir / "by_family.csv", index=False)

    ci = pd.DataFrame(ci_rows).sort_values(["dataset", "target", "selector", "model", "repeat"])
    med = (ci.groupby(["dataset", "target", "selector", "family", "model"])
           [["delta_r2_pooled", "delta_r2_lo", "delta_r2_hi", "delta_rmse_pooled", "delta_rmse_lo",
             "delta_rmse_hi"]].median().reset_index().assign(repeat="median_over_repeats"))
    ci_out = pd.concat([ci, med], ignore_index=True)
    for col in ("B", "n_groups", "seed"):
        ci_out[col] = ci_out[col].astype("Int64")
    ci_out["note"] = ("paired group bootstrap on pooled OOF predictions; fitted models and masks "
                      "held fixed (conditional)")
    ci_out.to_csv(out_dir / "delta_ci.csv", index=False)

    folds = pd.concat(fold_frames, ignore_index=True)
    folds.to_csv(out_dir / "fold_metrics.csv.gz", index=False, compression="gzip")
    _lasso_truncation(folds).to_csv(out_dir / "lasso_truncation.csv", index=False)
    _components(folds).to_csv(out_dir / "components.csv", index=False)
    if logo_rows:
        pd.DataFrame(logo_rows).to_csv(out_dir / "logo_potato.csv", index=False)
    pd.DataFrame(sig_rows).drop_duplicates(["dataset", "target", "repeat", "splits_signature"]) \
        .sort_values(["dataset", "target", "repeat"]).to_csv(out_dir / "splits_signatures.csv", index=False)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    pd.concat(sample_frames, ignore_index=True).to_csv(out_dir / "samples" / "folds.csv.gz",
                                                       index=False, compression="gzip")
    return {"n_tasks": len(tasks), "summary": summ, "by_family": fam, "delta_ci": ci_out}


def _as_res(R: dict):
    from src.m07_revision.engine import ArmsResult
    return ArmsResult(summary=R["summary"], contrasts=R["contrasts"], folds=R["folds"],
                      predictions=R["predictions"])


def _lasso_truncation(folds: pd.DataFrame) -> pd.DataFrame:
    """LASSO seçimleri: k tavanında kesilme / sıfırdan doldurma (R3-35).

    Model'den bağımsız (aynı maske) → yalnız ilk model; P2 maskesi tüm fold'larda
    aynı olduğundan P2 için yalnız fold 0 sayılır.
    """
    f = folds[(folds.selector == "lasso")]
    if f.empty:
        return pd.DataFrame()
    f = f[f.model == f.model.iloc[0]]
    f = f[(f.arm == "P3_nested") | ((f.arm == "P2_apparent") & (f.fold == 0))]
    info = f.selector_info.map(json.loads)
    f = f.assign(truncated=info.map(lambda d: d.get("truncated")),
                 filled=info.map(lambda d: d.get("filled_from_zero")),
                 n_nonzero=info.map(lambda d: d.get("n_nonzero")),
                 alpha=info.map(lambda d: d.get("alpha")),
                 conv=info.map(lambda d: d.get("n_convergence_warnings", 0)))
    g = f.groupby(["dataset", "target", "arm"])
    return pd.DataFrame({
        "n_selections": g.size(), "n_truncated": g.truncated.sum(),
        "frac_truncated": g.truncated.mean(), "n_filled_from_zero": g.filled.sum(),
        "n_nonzero_median": g.n_nonzero.median(), "n_nonzero_min": g.n_nonzero.min(),
        "n_nonzero_max": g.n_nonzero.max(), "alpha_median": g.alpha.median(),
        "n_convergence_warnings": g.conv.sum(),
    }).reset_index().assign(k_cap=20, note="P2: one all-data selection per repeat (fold 0 row)")


def _components(folds: pd.DataFrame) -> pd.DataFrame:
    f = folds[folds.model == "pls"].copy()
    f["n_components"] = pd.to_numeric(f.n_components, errors="coerce")
    tab = (f.groupby(["dataset", "target", "selector", "arm", "n_components"]).size()
           .rename("count").reset_index())
    tot = tab.groupby(["dataset", "target", "selector", "arm"])["count"].transform("sum")
    return tab.assign(fraction=tab["count"] / tot)


if __name__ == "__main__":
    from src.m07_revision.run_e1e2 import main
    main(experiments=("E1",))
