"""outputs3 FULL-POOLED agregasyon tablosu — YENİ KOŞUM YOK.

Amaç
----
Makale Bölüm 4.6, agregasyon seçiminin bias'a etkisini tartışırken FULL-POOLED
sayıları alıntılıyor (+0.341 ana konfigürasyon, 16'nın 13'ünde pozitif, pooled
apparent 15/16'da fold-mean'i aşıyor, medyan +0.011). Bu sayılar şimdiye dek
hiçbir tabloda saklanmıyordu — checkpoint'lerden anlık hesaplanmışlardı. Repo
makalenin alıntıladığı sayıyı üretemiyordu: tekrarlanabilirlik açığı. Bu script
o açığı kapatır.

ÜÇ AGREGASYON — karıştırma
    Makale "pooled" kelimesini İKİ farklı şey için kullanıyor; ayrımı koru:

    1. FULL-POOLED  (bu script)          → outputs3/tables_pooled/
       apparent VE nested, ikisi de havuzlanmış out-of-fold tahminlerden tek R².
       4.6'nın "+0.341 / 13-of-16 / medyan +0.011" sayıları BUNDAN gelir.

    2. MIXED        (regen_outputs3.py)  → outputs3/tables/
       apparent pooled, nested fold-mean — ASİMETRİK (tarihsel artefakt).
       4.6'nın "would report +0.047 for Potato/chl/MI" örneği BUNDAN gelir.
       Geçersiz değil; alıntılanan referans koşu, silinmez.

    3. FOLD-MEAN    (regen_outputs3_foldmean.py) → outputs3/tables_foldmean/
       apparent VE nested, ikisi de fold-ortalaması — SİMETRİK, OTORİTE.
       Table 4 ve tüm headline sayılar BUNDAN gelir.

Yeni koşum gerekmez
    Her checkpoint (NestedResult) her iki kolun havuzlanmış out-of-fold
    tahminlerini saklar (``apparent_y_true``/``apparent_y_pred``,
    ``nested_y_true``/``nested_y_pred``). Full-pooled R² bunlardan doğrudan
    hesaplanır — fold bölünmesini yeniden üretmeye bile gerek yok, çünkü havuz
    zaten tüm out-of-fold tahminleri içerir.
    Doğrulama: yeniden hesaplanan pooled apparent R², saklı ``apparent_r2``
    (tarihsel pooled değer) ile eşleşmelidir.

İSTATİSTİK YOK — bilerek
    Bu tablo yalnız agregasyon karşılaştırması içindir. Wilcoxon/FDR yalnız
    otorite (fold-mean) koşuda anlamlıdır; burada tekrarlamak makaleye üçüncü
    bir p-değeri seti sokar ve hangisinin alıntılandığı karışır. Dejenerasyon
    sorunu (deterministik seçicilerde sabit fark → test tanımsız) bu agregasyonda
    da aynen geçerlidir; bkz. stats_tests.is_degenerate.

Çıktı (YENİ klasör; diğer tables_* dizinleri dokunulmaz)
    outputs3/tables_pooled/
        selection_bias_pooled_raw.csv  → per-seed × kombinasyon
        selection_bias_pooled.csv      → 12-seed özet + fold-mean karşılaştırma
        aggregation_comparison.csv     → 4.6'nın toplu iddiaları (tek satır)

Çalıştırma:
    python regen_outputs3_pooled.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# --- Yollar ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "outputs" / "17_selection_bias" / "_checkpoints" / "pop150_ngen150_k20_no5"
NEW_TABLES = ROOT / "outputs3" / "tables_pooled"
FOLDMEAN_RAW = ROOT / "outputs3" / "tables_foldmean" / "selection_bias_multiseed_raw.csv"

GROUP_KEYS = ["dataset", "target", "model", "selector"]

R2_ATOL = 1e-6  # saklı pooled apparent R² doğrulama toleransı

# Smoke hedefleri — makale 4.6'nın alıntıladığı sayılar. Bunlar tutmazsa
# script makalenin dayandığı sayıyı üretmiyor demektir; DUR.
SMOKE_KEY = ("ryckewaert", "flavonol", "pls", "ga")
SMOKE_BIAS_POOLED = 0.341     # "+0.341 pooled against +0.346 by fold mean"
SMOKE_N_POSITIVE = 13         # "positive in thirteen of the sixteen under pooling"
SMOKE_N_APP_EXCEEDS = 15      # "pooled apparent exceeds fold mean in fifteen of sixteen"
SMOKE_MEDIAN_GAP = 0.011      # "median +0.011"
SMOKE_ATOL = 0.001


def build_raw() -> pd.DataFrame:
    """Checkpoint'lerden per-seed full-pooled metrikleri çıkar."""
    ckpts = sorted(CKPT_DIR.glob("seed*.pkl"))
    if not ckpts:
        raise FileNotFoundError(
            f"Checkpoint yok: {CKPT_DIR}\n"
            "Bu script mevcut checkpoint'lerden regen eder; önce ana koşu gerekir."
        )
    rows = []
    for ck in ckpts:
        r = pickle.loads(ck.read_bytes())
        seed = int(ck.name[len("seed"):].split("_", 1)[0])

        ayt = np.asarray(r.apparent_y_true, float)
        ayp = np.asarray(r.apparent_y_pred, float)
        nyt = np.asarray(r.nested_y_true, float)
        nyp = np.asarray(r.nested_y_pred, float)

        app_pooled = float(r2_score(ayt, ayp))
        nes_pooled = float(r2_score(nyt, nyp))

        # Doğrulama: saklı apparent_r2 tarihsel POOLED değerdir; eşleşmeli.
        if not np.isclose(app_pooled, r.apparent_r2, atol=R2_ATOL):
            raise RuntimeError(
                f"APPARENT pooled R² eşleşmedi ({r.dataset}/{r.target}/{r.model}/"
                f"{r.selector} seed={seed}): yeniden={app_pooled:.6f} "
                f"saklı={r.apparent_r2:.6f}. Checkpoint tutarsız — DUR."
            )

        rows.append({
            "seed": seed, "dataset": r.dataset, "target": r.target,
            "model": r.model, "selector": getattr(r, "selector", "ga"),
            "apparent_r2_pooled": app_pooled,
            "nested_r2_pooled": nes_pooled,
            "bias_pooled": app_pooled - nes_pooled,
        })
    return pd.DataFrame(rows).sort_values(
        [*GROUP_KEYS, "seed"]).reset_index(drop=True)


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """12-seed özet + otorite (fold-mean) koşuyla yan yana karşılaştırma."""
    agg = raw.groupby(GROUP_KEYS, as_index=False, sort=False).agg(
        n_seeds=("seed", "nunique"),
        apparent_pooled_mean=("apparent_r2_pooled", "mean"),
        apparent_pooled_std=("apparent_r2_pooled", "std"),
        nested_pooled_mean=("nested_r2_pooled", "mean"),
        nested_pooled_std=("nested_r2_pooled", "std"),
        bias_pooled_mean=("bias_pooled", "mean"),
        bias_pooled_std=("bias_pooled", "std"),
    )

    if not FOLDMEAN_RAW.exists():
        raise FileNotFoundError(
            f"Fold-mean raw tablosu yok: {FOLDMEAN_RAW}\n"
            "Önce: python regen_outputs3_foldmean.py"
        )
    fm = pd.read_csv(FOLDMEAN_RAW)
    fm_agg = fm.groupby(GROUP_KEYS, as_index=False, sort=False).agg(
        apparent_foldmean_mean=("apparent_r2", "mean"),
        nested_foldmean_mean=("nested_r2", "mean"),
        bias_foldmean_mean=("bias", "mean"),
    )

    out = agg.merge(fm_agg, on=GROUP_KEYS, how="left")
    # 4.6'nın iki karşılaştırma ekseni.
    out["bias_delta_pooled_minus_foldmean"] = (
        out.bias_pooled_mean - out.bias_foldmean_mean)
    out["apparent_delta_pooled_minus_foldmean"] = (
        out.apparent_pooled_mean - out.apparent_foldmean_mean)
    out["sign_flips_vs_foldmean"] = (
        np.sign(out.bias_pooled_mean) != np.sign(out.bias_foldmean_mean))
    return out


def build_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """4.6'nın toplu iddiaları — tek satır, alıntılanabilir."""
    n = len(summary)
    return pd.DataFrame([{
        "n_combos": n,
        "n_positive_bias_pooled": int((summary.bias_pooled_mean > 0).sum()),
        "n_positive_bias_foldmean": int((summary.bias_foldmean_mean > 0).sum()),
        "n_apparent_pooled_exceeds_foldmean": int(
            (summary.apparent_delta_pooled_minus_foldmean > 0).sum()),
        "median_apparent_delta": float(
            summary.apparent_delta_pooled_minus_foldmean.median()),
        "n_sign_flips": int(summary.sign_flips_vs_foldmean.sum()),
        "sign_flip_combos": "; ".join(
            f"{r.dataset}/{r.target}/{r.model}/{r.selector}"
            for _, r in summary[summary.sign_flips_vs_foldmean].iterrows()),
    }])


def smoke(summary: pd.DataFrame, comp: pd.DataFrame) -> None:
    """Makale 4.6'nın alıntıladığı dört sayıyı doğrula."""
    d, t, mo, s = SMOKE_KEY
    row = summary[(summary.dataset == d) & (summary.target == t)
                  & (summary.model == mo) & (summary.selector == s)]
    if len(row) != 1:
        raise RuntimeError(f"SMOKE: {SMOKE_KEY} bulunamadı.")
    bias = float(row.iloc[0].bias_pooled_mean)
    c = comp.iloc[0]

    print(f"[SMOKE] {'/'.join(SMOKE_KEY)} bias_pooled = {bias:+.4f} "
          f"(hedef {SMOKE_BIAS_POOLED:+.3f})")
    print(f"[SMOKE] pozitif bias (pooled)     = {c.n_positive_bias_pooled}/16 "
          f"(hedef {SMOKE_N_POSITIVE})")
    print(f"[SMOKE] pooled apparent > foldmean = "
          f"{c.n_apparent_pooled_exceeds_foldmean}/16 (hedef {SMOKE_N_APP_EXCEEDS})")
    print(f"[SMOKE] medyan apparent farki      = {c.median_apparent_delta:+.4f} "
          f"(hedef {SMOKE_MEDIAN_GAP:+.3f})")

    fails = []
    if abs(bias - SMOKE_BIAS_POOLED) > SMOKE_ATOL:
        fails.append(f"bias_pooled {bias:+.4f} != {SMOKE_BIAS_POOLED:+.3f}")
    if int(c.n_positive_bias_pooled) != SMOKE_N_POSITIVE:
        fails.append(f"pozitif {c.n_positive_bias_pooled} != {SMOKE_N_POSITIVE}")
    if int(c.n_apparent_pooled_exceeds_foldmean) != SMOKE_N_APP_EXCEEDS:
        fails.append(
            f"apparent>foldmean {c.n_apparent_pooled_exceeds_foldmean} "
            f"!= {SMOKE_N_APP_EXCEEDS}")
    if abs(float(c.median_apparent_delta) - SMOKE_MEDIAN_GAP) > SMOKE_ATOL:
        fails.append(
            f"medyan {c.median_apparent_delta:+.4f} != {SMOKE_MEDIAN_GAP:+.3f}")
    if fails:
        raise RuntimeError(
            "SMOKE BASARISIZ — makale 4.6 ile eslesmedi. DUR.\n  " + "\n  ".join(fails))
    print("[SMOKE] GECTI - OK")


def main() -> None:
    raw = build_raw()
    summary = build_summary(raw)
    comp = build_comparison(summary)
    smoke(summary, comp)

    NEW_TABLES.mkdir(parents=True, exist_ok=True)
    raw.to_csv(NEW_TABLES / "selection_bias_pooled_raw.csv",
               index=False, encoding="utf-8")
    print(f"[YAZ] selection_bias_pooled_raw.csv ({len(raw)} satir)")
    summary.to_csv(NEW_TABLES / "selection_bias_pooled.csv",
                   index=False, encoding="utf-8")
    print(f"[YAZ] selection_bias_pooled.csv ({len(summary)} kombinasyon)")
    comp.to_csv(NEW_TABLES / "aggregation_comparison.csv",
                index=False, encoding="utf-8")
    print("[YAZ] aggregation_comparison.csv")

    _report(summary, comp)


def _report(summary: pd.DataFrame, comp: pd.DataFrame) -> None:
    print("\n" + "=" * 84)
    print("FULL-POOLED vs FOLD-MEAN (makale 4.6)")
    print("=" * 84)
    print(f"{'kombinasyon':<46} {'pooled':>9} {'foldmean':>9} {'fark':>8}")
    for _, r in summary.iterrows():
        name = f"{r.dataset}/{r.target}/{r.model}/{r.selector}"
        flip = "  <-- ISARET DONUYOR" if r.sign_flips_vs_foldmean else ""
        print(f"{name:<46} {r.bias_pooled_mean:>+9.4f} "
              f"{r.bias_foldmean_mean:>+9.4f} "
              f"{r.bias_delta_pooled_minus_foldmean:>+8.4f}{flip}")
    c = comp.iloc[0]
    print()
    print(f"Pozitif bias: pooled {c.n_positive_bias_pooled}/16, "
          f"foldmean {c.n_positive_bias_foldmean}/16")
    print(f"Isaret donen kombinasyon: {c.n_sign_flips}")
    if c.n_sign_flips:
        print(f"  {c.sign_flip_combos}")
    print(f"Pooled apparent > foldmean apparent: "
          f"{c.n_apparent_pooled_exceeds_foldmean}/16, "
          f"medyan {c.median_apparent_delta:+.4f}")
    print()
    print("NOT: Otorite sayilar fold-mean'dir (outputs3/tables_foldmean/).")
    print("     Bu tablo yalniz 4.6'nin agregasyon karsilastirmasi icindir.")


if __name__ == "__main__":
    main()
