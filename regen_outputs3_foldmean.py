"""outputs3 FOLD-MEAN apparent yeniden üretimi — YENİ KOŞUM YOK.

Amaç
----
apparent kolunu da nested ile BİREBİR aynı agregasyona (fold-ortalaması)
çevirmek. Eskiden apparent_r2 pooled'dı, nested_r2 fold-mean'di; bu asimetri
apparent-nested farkına karışıyor ve iki kombinasyonda R² bias ile RMSE bias'ın
işaretini ters düşürüyordu.

Yeni koşum gerekmez
    Her checkpoint (NestedResult) apparent 5-fold out-of-fold havuzlanmış
    tahminlerini (``apparent_y_true``/``apparent_y_pred``) saklar. apparent CV
    split'i seed + gruplara göre DETERMİNİSTİKtir (make_cv_splitter, n_splits=5,
    shuffle yok). Split'i yeniden üretip havuzlanmış tahminleri fold başına
    dilimleyerek fold-ortalaması apparent R²/RMSE'yi GA'sız hesaplarız.
    Doğrulama: yeniden hesaplanan POOLED apparent R² saklı ``apparent_r2``
    (eski pooled değer) ile birebir eşleşmelidir.

Agregasyon
    * apparent_r2   = fold başına R², sonra fold-ORTALAMASI (nested ile aynı)
    * apparent_rmse = fold başına sqrt(MSE), sonra fold-ORTALAMASI
    * *_std         = ilgili fold metriklerinin std'i
    * apparent_r2_pooled / apparent_rmse_pooled = eski pooled değerler (ATILMAZ)
    * nested tarafı zaten fold-mean; regen_outputs3_rmse.py ile aynı.

Çıktı (YENİ klasör; tables_rmse/ ve _pre_rmse_backup/ dokunulmaz)
    outputs3/tables_foldmean/  → raw + multiseed + stats_summary

Çalıştırma:
    python regen_outputs3_foldmean.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.core.cv import make_cv_splitter
from src.m01_io import dataset_registry as registry
from src.m06_selection_bias import stats_tests

# --- Yollar ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "outputs" / "17_selection_bias" / "_checkpoints" / "pop150_ngen150_k20_no5"
NEW_TABLES = ROOT / "outputs3" / "tables_foldmean"

R2_ATOL = 1e-6  # pooled apparent R² doğrulama toleransı

# Smoke hedefleri.
SMOKE_KEY = ("ryckewaert", "flavonol", "pls", "ga")
# Pooled apparent (eski) = 0.619; fold-mean nested = 0.269 (değişmez).
SMOKE_APPARENT_POOLED = 0.619
SMOKE_NESTED = 0.269
SMOKE_ATOL = 0.001


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


_DS_CACHE: dict[str, object] = {}


def _finite_groups(dataset: str, target: str):
    """NaN-hedef filtresi sonrası (y, groups) — GroupKFold(5) bölünmesini
    yeniden üretmek için (her iki kol da aynı bölünmeyi kullanır)."""
    if dataset not in _DS_CACHE:
        _DS_CACHE[dataset] = registry.load(dataset)
    ds = _DS_CACHE[dataset]
    y = ds.target(target)
    fin = np.isfinite(y)
    y = y[fin]
    g = ds.groups[fin] if ds.groups is not None else None
    return y, g


def _fold_slice_r2(yt, yp, splits):
    return [float(r2_score(yt[te], yp[te])) for _, te in splits]


def _fold_metrics(r, seed: int, n_outer: int = 5) -> dict:
    """Saklı tahminlerden nested ve apparent fold-ortalaması metrikleri.

    Her iki kol da AYNI partisyonu kullanır: make_cv_splitter(n_splits=5,
    task="regression", groups=g) → GroupKFold(5). nested_ga.py'de apparent
    (satır 204-206) ve nested dış döngü (satır 311-313) bu yardımcıyı aynı
    filtrelenmiş ``g`` ile çağırır; GroupKFold shuffle/random_state almaz
    (core/cv.py:181), dolayısıyla bölünme deterministiktir ve seed'den
    bağımsızdır. Yani "iç" ve "dış" ayrı partisyonlar DEĞİLDİR — burada da
    aynı splitter yeniden üretilip saklı tahminler fold başına dilimlenir.
    Doğrulama: her iki kolun POOLED R²'si saklı değerle eşleşmeli.
    """
    y_ref, g = _finite_groups(r.dataset, r.target)

    # --- NESTED (fold-ort; GroupKFold(5) — apparent ile aynı bölünme) ---
    nyt = np.asarray(r.nested_y_true, float)
    nyp = np.asarray(r.nested_y_pred, float)
    outer = make_cv_splitter(n_splits=n_outer, task="regression", groups=g,
                             random_state=seed)
    nsp = list(outer.split(np.zeros((len(nyt), 1)), nyt, g) if g is not None
               else outer.split(np.zeros((len(nyt), 1)), nyt))
    n_fold_r2 = _fold_slice_r2(nyt, nyp, nsp)
    n_fold_rmse = [_rmse(nyt[te], nyp[te]) for _, te in nsp]
    n_fold_mae = [_mae(nyt[te], nyp[te]) for _, te in nsp]
    # Doğrulama: yeniden dilimlenen fold R² == saklı fold_scores.
    if r.fold_scores:
        a = np.sort(n_fold_r2); b = np.sort(np.asarray(r.fold_scores, float))
        if a.shape == b.shape and not np.allclose(a, b, atol=R2_ATOL):
            raise RuntimeError(
                f"NESTED fold R² eşleşmedi ({r.dataset}/{r.target}/{r.model}/"
                f"{r.selector} seed={seed}). DUR.")

    # --- APPARENT (fold-ort; nested ile AYNI GroupKFold(5) bölünmesi,
    #     havuzlanmış out-of-fold tahminlerden dilimlenir) ---
    ayt = np.asarray(r.apparent_y_true, float)
    ayp = np.asarray(r.apparent_y_pred, float)
    inner = make_cv_splitter(n_splits=5, task="regression", groups=g,
                             random_state=seed)
    asp = list(inner.split(np.zeros((len(ayt), 1)), ayt, g) if g is not None
               else inner.split(np.zeros((len(ayt), 1)), ayt))
    a_fold_r2 = _fold_slice_r2(ayt, ayp, asp)
    a_fold_rmse = [_rmse(ayt[te], ayp[te]) for _, te in asp]
    a_fold_mae = [_mae(ayt[te], ayp[te]) for _, te in asp]
    # Doğrulama: yeniden hesaplanan POOLED apparent R² == saklı apparent_r2 (eski pooled).
    app_pooled = float(r2_score(ayt, ayp))
    if not np.isclose(app_pooled, r.apparent_r2, atol=R2_ATOL):
        raise RuntimeError(
            f"APPARENT pooled R² eşleşmedi ({r.dataset}/{r.target}/{r.model}/"
            f"{r.selector} seed={seed}): yeniden={app_pooled:.6f} "
            f"saklı={r.apparent_r2:.6f}. Split yeniden üretimi tutarsız — DUR.")

    return {
        # NESTED (fold-ort) — regen_outputs3_rmse ile aynı
        "nested_r2": float(np.mean(n_fold_r2)),
        "nested_rmse": float(np.mean(n_fold_rmse)),
        "nested_rmse_std": float(np.std(n_fold_rmse)),
        "nested_mae": float(np.mean(n_fold_mae)),
        "nested_rmse_pooled": _rmse(nyt, nyp),
        # APPARENT (fold-ort) — YENİ
        "apparent_r2": float(np.mean(a_fold_r2)),
        "apparent_r2_std": float(np.std(a_fold_r2)),
        "apparent_rmse": float(np.mean(a_fold_rmse)),
        "apparent_rmse_std": float(np.std(a_fold_rmse)),
        "apparent_mae": float(np.mean(a_fold_mae)),
        # APPARENT pooled (eski; ATILMAZ)
        "apparent_r2_pooled": app_pooled,
        "apparent_rmse_pooled": _rmse(ayt, ayp),
        "apparent_mae_pooled": _mae(ayt, ayp),
    }


def build_raw() -> pd.DataFrame:
    ckpts = sorted(CKPT_DIR.glob("seed*.pkl"))
    if not ckpts:
        raise FileNotFoundError(f"Checkpoint yok: {CKPT_DIR}")
    rows = []
    for ck in ckpts:
        r = pickle.loads(ck.read_bytes())
        seed = int(ck.name[len("seed"):].split("_", 1)[0])
        m = _fold_metrics(r, seed=seed)
        rows.append({
            "seed": seed, "dataset": r.dataset, "target": r.target,
            "model": r.model, "selector": getattr(r, "selector", "ga"),
            # R² — apparent ARTIK fold-mean, nested fold-mean (checkpoint fold_scores'tan).
            "apparent_r2": m["apparent_r2"], "apparent_r2_std": m["apparent_r2_std"],
            "nested_r2": m["nested_r2"],
            "bias": m["apparent_r2"] - m["nested_r2"],
            # RMSE — ikisi de fold-mean
            "apparent_rmse": m["apparent_rmse"], "apparent_rmse_std": m["apparent_rmse_std"],
            "nested_rmse": m["nested_rmse"], "nested_rmse_std": m["nested_rmse_std"],
            "bias_rmse": m["nested_rmse"] - m["apparent_rmse"],
            # POOLED (karşılaştırma; atılmaz)
            "apparent_r2_pooled": m["apparent_r2_pooled"],
            "apparent_rmse_pooled": m["apparent_rmse_pooled"],
            "nested_rmse_pooled": m["nested_rmse_pooled"],
            "bias_pooled": m["apparent_r2_pooled"] - m["nested_r2"],
        })
    return pd.DataFrame(rows).sort_values(
        ["dataset", "target", "model", "selector", "seed"]).reset_index(drop=True)


def smoke(raw: pd.DataFrame) -> None:
    d, t, mo, s = SMOKE_KEY
    sub = raw[(raw.dataset == d) & (raw.target == t)
              & (raw.model == mo) & (raw.selector == s)]
    if len(sub) != 12:
        raise RuntimeError(f"SMOKE: {SMOKE_KEY} 12 seed beklenirken {len(sub)}.")
    app_pool = float(sub.apparent_r2_pooled.mean())
    nes = float(sub.nested_r2.mean())
    print(f"[SMOKE] {SMOKE_KEY}: apparent_pooled={app_pool:.3f} "
          f"(hedef {SMOKE_APPARENT_POOLED}), nested={nes:.3f} (hedef {SMOKE_NESTED})")
    if not (abs(app_pool - SMOKE_APPARENT_POOLED) <= SMOKE_ATOL
            and abs(nes - SMOKE_NESTED) <= SMOKE_ATOL):
        raise RuntimeError("SMOKE BASARISIZ — pooled R² Tablo 4 ile eslesmedi. DUR.")
    print(f"[SMOKE] apparent_foldmean={float(sub.apparent_r2.mean()):.3f} "
          f"bias_foldmean={float(sub.bias.mean()):.3f}")
    print("[SMOKE] GECTI - OK")


def main() -> None:
    raw = build_raw()
    smoke(raw)

    NEW_TABLES.mkdir(parents=True, exist_ok=True)
    raw_path = NEW_TABLES / "selection_bias_multiseed_raw.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8")
    print(f"[YAZ] {raw_path} ({len(raw)} satir)")

    # 12-seed özet (fold-mean R²/RMSE + pooled karşılaştırma).
    agg = raw.groupby(["dataset", "target", "model", "selector"],
                      as_index=False, sort=False).agg(
        n_seeds=("seed", "nunique"),
        apparent_mean=("apparent_r2", "mean"), apparent_std=("apparent_r2", "std"),
        nested_mean=("nested_r2", "mean"), nested_std=("nested_r2", "std"),
        bias_mean=("bias", "mean"), bias_std=("bias", "std"),
        apparent_pooled_mean=("apparent_r2_pooled", "mean"),
        bias_pooled_mean=("bias_pooled", "mean"),
        apparent_rmse_mean=("apparent_rmse", "mean"),
        nested_rmse_mean=("nested_rmse", "mean"),
    )
    agg.to_csv(NEW_TABLES / "selection_bias_multiseed.csv", index=False, encoding="utf-8")

    # stats_summary — fold-mean R² üzerinde Wilcoxon/FDR/Cliff + RMSE sütunları.
    stats = stats_tests.compute_stats(raw)
    rmse_agg = raw.groupby(stats_tests.GROUP_KEYS, sort=False).agg(
        mean_apparent_rmse=("apparent_rmse", "mean"),
        std_apparent_rmse=("apparent_rmse", "std"),
        mean_nested_rmse=("nested_rmse", "mean"),
        std_nested_rmse=("nested_rmse", "std"),
        mean_apparent_pooled=("apparent_r2_pooled", "mean"),
    ).reset_index()
    stats = stats.merge(rmse_agg, on=stats_tests.GROUP_KEYS, how="left")
    summary = stats_tests.summarize(stats)
    stats_tests._write_with_summary(stats, summary, NEW_TABLES / "stats_summary.csv")
    print(f"[YAZ] stats_summary.csv ({len(stats)} kombinasyon)")
    print(f"[OZET] {summary['text']}")

    _report(raw, stats)


def _report(raw: pd.DataFrame, stats: pd.DataFrame) -> None:
    """Dört soruya net cevap (konsola)."""
    print("\n" + "=" * 78)
    print("RAPOR")
    print("=" * 78)

    keys = ["dataset", "target", "model", "selector"]
    g = raw.groupby(keys, sort=False)
    # Kombinasyon başına fold-mean ve pooled apparent ortalamaları.
    tbl = g.agg(
        app_fold=("apparent_r2", "mean"),
        app_pool=("apparent_r2_pooled", "mean"),
        nested=("nested_r2", "mean"),
        bias_fold=("bias", "mean"),
        bias_pool=("bias_pooled", "mean"),
        r2_bias=("bias", "mean"),
        rmse_bias=("bias_rmse", "mean"),
    ).reset_index()

    # (a) ryckewaert/flavonol/pls/ga
    row = tbl[(tbl.dataset == "ryckewaert") & (tbl.target == "flavonol")
              & (tbl.model == "pls") & (tbl.selector == "ga")].iloc[0]
    print("\n(a) ryckewaert/flavonol/pls/ga bias:")
    print(f"    pooled apparent bias (eski) = {row.bias_pool:+.4f}  (0.349 referans)")
    print(f"    fold-mean apparent bias     = {row.bias_fold:+.4f}")
    print(f"    0.349'dan sapma             = {row.bias_fold - 0.349:+.4f} "
          f"(|Δ|={abs(row.bias_fold - 0.349):.4f})")

    # (b) R² bias ile RMSE bias işaret uyumu (fold-mean)
    same = int((np.sign(tbl.r2_bias) == np.sign(tbl.rmse_bias)).sum())
    print(f"\n(b) R² bias ile RMSE bias işaret uyumu (fold-mean): {same}/16")
    mism = tbl[np.sign(tbl.r2_bias) != np.sign(tbl.rmse_bias)]
    if len(mism):
        print("    Uyuşmayanlar:")
        for _, r in mism.iterrows():
            print(f"      {r.dataset}/{r.target}/{r.model}/{r.selector}: "
                  f"R²bias={r.r2_bias:+.3f} RMSEbias={r.rmse_bias:+.3f}")
    else:
        print("    (hepsi uyuşuyor)")

    # (c) pooled vs fold-mean apparent R² farkı
    tbl["app_diff"] = tbl.app_pool - tbl.app_fold
    tc = tbl.sort_values("app_diff", key=lambda s: s.abs(), ascending=False)
    print("\n(c) apparent R²: pooled vs fold-mean (fark = pooled − foldmean):")
    print(f"    {'kombinasyon':<48} {'pooled':>8} {'foldmean':>9} {'fark':>8}")
    for _, r in tc.iterrows():
        name = f"{r.dataset}/{r.target}/{r.model}/{r.selector}"
        print(f"    {name:<48} {r.app_pool:>8.4f} {r.app_fold:>9.4f} {r.app_diff:>+8.4f}")
    big = tc.iloc[0]
    print(f"    → En büyük sapma: {big.dataset}/{big.target}/{big.model}/"
          f"{big.selector} = {big.app_diff:+.4f}")

    # (d) Wilcoxon/FDR/Cliff
    n_sig = int(stats["significant_fdr"].sum())
    print(f"\n(d) FDR-anlamlı kombinasyon: {n_sig}/16")
    print("    (fold-mean apparent R² üzerinde Wilcoxon signed-rank + BH-FDR)")


if __name__ == "__main__":
    main()
