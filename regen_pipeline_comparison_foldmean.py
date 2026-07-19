"""Tablo 5 (P1/P2/P3 ayrıştırması) FOLD-MEAN + 12-seed yeniden üretimi.

Neden
-----
Tablo 5 eskiden tek seed (42) ve apparent kolu POOLED idi. Tablo 4 fold-mean'e
geçti; iki tablo aynı agregasyonda olsun diye Tablo 5 de fold-mean'e çekilir ve
12 seed üzerinden mean±std verilir.

Üç pipeline (hepsi fold-ortalaması R²)
    P1 = random KFold(shuffle=True, random_state=seed) + apparent maske
         (dış seçim; grup + seçim sızıntısı birlikte)
    P2 = GroupKFold + apparent maske (dış seçim; sadece seçim sızıntısı)
    P3 = nested GroupKFold (seçim fold-train'de; dürüst)
    group_leak = P1 − P2   |   sel_leak = P2 − P3

Ne yeniden hesaplanır, ne türetilir?
    * P2 = checkpoint'teki apparent havuz tahminleri (apparent_y_pred) —
      deterministik GroupKFold split'i yeniden dilimlenip FOLD-ORT alınır.
      YENİDEN FIT YOK. (Tablo 4'ün mean_apparent'ıyla BİREBİR aynı; smoke bunu
      doğrular.)
    * P3 = checkpoint'teki nested fold R²'leri (fold_scores) FOLD-ORT. YENİDEN
      FIT YOK.
    * P1 = random KFold checkpoint'te SAKLI DEĞİL. Apparent-maskeli veri üzerinde
      KFold(shuffle=True, random_state=seed) ile cross_val_predict YENİDEN
      koşulur. Bu bir GA koşusu DEĞİL — sadece PLS/LightGBM'i karışık bölmede
      fit etmek (ucuz, deterministik). Apparent maske deterministik seçiciyle
      tek kez üretilir (seed'den bağımsız); P1'in seed bağımlılığı yalnız KFold
      shuffle'ından gelir, dolayısıyla P1 seed'e göre DEĞİŞİR.

Kapsam
    12 deterministik kombinasyon (rfe/lasso/mutual_info × 4 konfig). GA'nın 4
    satırı CSV'de NaN (GA apparent maskesi saklı değil + stokastik).

Çıktı: outputs3/tables_foldmean/pipeline_comparison.csv

Çalıştırma:
    python regen_pipeline_comparison_foldmean.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

from src.core.cv import make_cv_splitter
from src.m01_io import dataset_registry as registry
from src.m04_features.ga_feature_selection import _build_regressor
from src.m04_features.selectors import get_selector

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "outputs" / "17_selection_bias" / "_checkpoints" / "pop150_ngen150_k20_no5"
NEW_TABLES = ROOT / "outputs3" / "tables_foldmean"
SEEDS = list(range(12))
DET = {"rfe", "lasso", "mutual_info"}
K = 20
N_SPLITS = 5
SMOKE_ATOL = 1e-6

_DS: dict[str, object] = {}


def _load(name: str):
    if name not in _DS:
        _DS[name] = registry.load(name)
    return _DS[name]


def _finite(dataset: str, target: str):
    """NaN-hedef filtresi sonrası (X, y, groups) — checkpoint ile aynı temizlik."""
    ds = _load(dataset)
    y = ds.target(target)
    fin = np.isfinite(y)
    return ds.X[fin], y[fin], (ds.groups[fin] if ds.groups is not None else None)


def _foldmean_r2(y_true, y_pred, splits) -> float:
    """Havuzlanmış tahminleri fold başına dilimle, fold-ort R² döndür."""
    return float(np.mean([r2_score(y_true[te], y_pred[te]) for _, te in splits]))


def _combos_from_checkpoints():
    """(dataset,target,model,selector) -> {seed: NestedResult} sözlüğü."""
    out: dict[tuple, dict] = {}
    for ck in sorted(CKPT_DIR.glob("seed*.pkl")):
        r = pickle.loads(ck.read_bytes())
        seed = int(ck.name[len("seed"):].split("_", 1)[0])
        key = (r.dataset, r.target, r.model, getattr(r, "selector", "ga"))
        out.setdefault(key, {})[seed] = r
    return out


def _p1_foldmean(X, y, g, model, mask, seed) -> float:
    """P1: apparent-maskeli veride random KFold(shuffle,seed) fold-ort R²."""
    Xs = X[:, mask]
    reg = _build_regressor(model, int(mask.sum()), Xs.shape[0])
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=int(seed))
    pred = np.asarray(cross_val_predict(reg, Xs, y, cv=cv, n_jobs=1)).ravel()
    splits = list(cv.split(Xs, y))
    return _foldmean_r2(y, pred, splits)


def build() -> tuple[pd.DataFrame, list[str]]:
    combos = _combos_from_checkpoints()
    smoke_msgs: list[str] = []
    rows = []

    # Tablo 4 fold-mean mean_apparent — P2 smoke karşılaştırması için.
    t4 = pd.read_csv(NEW_TABLES / "selection_bias_multiseed_raw.csv")
    t4g = t4.groupby(["dataset", "target", "model", "selector"], sort=False)["apparent_r2"].mean()

    for key in sorted(combos):
        dataset, target, model, selector = key
        n_seeds = len(combos[key])
        if selector not in DET:
            # GA — NaN satırı.
            rows.append(dict(
                dataset=dataset, target=target, model=model, selector=selector,
                n=n_seeds, n_groups=combos[key][next(iter(combos[key]))].n_groups,
                p1_mean=np.nan, p1_std=np.nan, p2_mean=np.nan, p2_std=np.nan,
                p3_mean=np.nan, p3_std=np.nan,
                group_leak_mean=np.nan, group_leak_std=np.nan,
                sel_leak_mean=np.nan, sel_leak_std=np.nan,
            ))
            continue

        X, y, g = _finite(dataset, target)
        n_groups = None if g is None else len(set(g.tolist()))

        # Apparent maske: deterministik seçici, tüm veride tek kez (seed'den bağımsız).
        mask = get_selector(selector)(X, y, K, groups=g, model=model, seed=42)

        p1s, p2s, p3s = [], [], []
        for seed in SEEDS:
            r = combos[key].get(seed)
            if r is None:
                continue
            # P2: checkpoint apparent havuz tahminleri, GroupKFold fold-ort.
            ay = np.asarray(r.apparent_y_true, float)
            ap = np.asarray(r.apparent_y_pred, float)
            gk = make_cv_splitter(n_splits=N_SPLITS, task="regression",
                                  groups=g, random_state=seed)
            gsp = list(gk.split(np.zeros((len(ay), 1)), ay, g) if g is not None
                       else gk.split(np.zeros((len(ay), 1)), ay))
            p2 = _foldmean_r2(ay, ap, gsp)
            # P3: checkpoint nested fold_scores fold-ort.
            p3 = float(np.mean(r.fold_scores))
            # P1: yeniden fit (random KFold).
            p1 = _p1_foldmean(X, y, g, model, mask, seed)
            p1s.append(p1); p2s.append(p2); p3s.append(p3)

        p1s, p2s, p3s = map(np.asarray, (p1s, p2s, p3s))
        group_leak = p1s - p2s
        sel_leak = p2s - p3s

        # SMOKE: P2 fold-mean ortalaması == Tablo 4 mean_apparent.
        t4_val = float(t4g.loc[(dataset, target, model, selector)])
        p2_mean = float(np.mean(p2s))
        if abs(p2_mean - t4_val) > SMOKE_ATOL:
            raise RuntimeError(
                f"SMOKE BASARISIZ: P2 fold-mean ({p2_mean:.6f}) != Tablo 4 "
                f"mean_apparent ({t4_val:.6f}) @ {key}. DUR.")
        smoke_msgs.append(f"  {dataset}/{target}/{model}/{selector}: "
                          f"P2={p2_mean:.4f} == T4_app={t4_val:.4f}")

        rows.append(dict(
            dataset=dataset, target=target, model=model, selector=selector,
            n=len(p2s), n_groups=n_groups,
            p1_mean=float(np.mean(p1s)), p1_std=float(np.std(p1s, ddof=1) if len(p1s) > 1 else 0.0),
            p2_mean=p2_mean, p2_std=float(np.std(p2s, ddof=1) if len(p2s) > 1 else 0.0),
            p3_mean=float(np.mean(p3s)), p3_std=float(np.std(p3s, ddof=1) if len(p3s) > 1 else 0.0),
            group_leak_mean=float(np.mean(group_leak)),
            group_leak_std=float(np.std(group_leak, ddof=1) if len(group_leak) > 1 else 0.0),
            sel_leak_mean=float(np.mean(sel_leak)),
            sel_leak_std=float(np.std(sel_leak, ddof=1) if len(sel_leak) > 1 else 0.0),
        ))

    return pd.DataFrame(rows), smoke_msgs


def main() -> None:
    import logging
    logging.disable(logging.INFO)  # seçici/nested logları sustur

    df, smoke_msgs = build()
    print("[SMOKE] P2 fold-mean == Tablo 4 mean_apparent (12 det. kombinasyon):")
    for m in smoke_msgs:
        print(m)
    print("[SMOKE] GECTI - OK\n")

    NEW_TABLES.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "target", "model", "selector", "n", "n_groups",
            "p1_mean", "p1_std", "p2_mean", "p2_std", "p3_mean", "p3_std",
            "group_leak_mean", "group_leak_std", "sel_leak_mean", "sel_leak_std"]
    p = NEW_TABLES / "pipeline_comparison.csv"
    df[cols].to_csv(p, index=False, encoding="utf-8")
    with open(p, "a", encoding="utf-8") as f:
        f.write("# NOT: GA satirlari NaN. GA apparent maskesi checkpoint lerde\n")
        f.write("# saklanmiyor ve stokastik oldugu icin yeniden uretilemez.\n")
        f.write("# P2/P3 checkpointten (yeniden fit YOK); P1 random-KFold ile\n")
        f.write("# yeniden fit edildi (GA degil). Hepsi fold-ortalamasi R2, 12 seed.\n")
    print(f"[YAZ] {p}\n")

    _report(df)


def _report(df: pd.DataFrame) -> None:
    d = df[df.selector.isin(DET)].copy()
    print("=" * 78)
    print("RAPOR")
    print("=" * 78)

    # (1) deep_patato/chlorophyll/pls/rfe sel_leak
    r = d[(d.dataset == "deep_patato") & (d.selector == "rfe")].iloc[0]
    print("\n(1) deep_patato/chlorophyll/pls/rfe sel_leak (fold-mean, 12 seed):")
    print(f"    sel_leak = {r.sel_leak_mean:+.4f} ± {r.sel_leak_std:.4f}")
    print(f"    (P2={r.p2_mean:.4f}  P3={r.p3_mean:.4f})")
    print("    Kıyas: pooled tek-seed -0.006; Tablo 4 fold-mean bias -0.062.")

    # (2) sel_leak pozitif kaç
    npos = int((d.sel_leak_mean > 0).sum())
    print(f"\n(2) sel_leak pozitif: {npos}/12")
    print("    Negatif olanlar:")
    for _, x in d[d.sel_leak_mean <= 0].iterrows():
        print(f"      {x.dataset}/{x.target}/{x.model}/{x.selector}: {x.sel_leak_mean:+.4f}")

    # (3) group_leak ryckewaert (her yaprak kendi grubu) + lightgbm
    print("\n(3) group_leak — ryckewaert (n_groups==n_samples ise ~0 beklenir):")
    ry = d[d.dataset == "ryckewaert"]
    for _, x in ry.iterrows():
        lo = x.group_leak_mean - x.group_leak_std
        hi = x.group_leak_mean + x.group_leak_std
        zero = "SIFIR İÇİNDE" if lo <= 0 <= hi else "sıfır DIŞINDA"
        print(f"    {x.target}/{x.model}/{x.selector}: "
              f"{x.group_leak_mean:+.4f} ± {x.group_leak_std:.4f}  "
              f"[{lo:+.4f},{hi:+.4f}] {zero}  (n_grp={x.n_groups})")
    print("    LightGBM satırları (eski tek-seed -0.058/-0.052/-0.023):")
    for _, x in ry[ry.model == "lightgbm"].iterrows():
        sign = "negatif" if x.group_leak_mean < 0 else "pozitif"
        print(f"      {x.selector}: {x.group_leak_mean:+.4f} ({sign})")

    # (4) deep_patato group_leak üç seçici
    print("\n(4) deep_patato group_leak (eski tek-seed +0.099/+0.026/-0.038):")
    dp = d[d.dataset == "deep_patato"].sort_values("selector")
    for _, x in dp.iterrows():
        print(f"    {x.selector}: {x.group_leak_mean:+.4f} ± {x.group_leak_std:.4f} (n_grp={x.n_groups})")
    signs = set(np.sign(dp.group_leak_mean.round(4)))
    print(f"    → Aynı yönde mi? {'EVET' if len(signs) == 1 else 'HAYIR (karışık)'}")


if __name__ == "__main__":
    main()
