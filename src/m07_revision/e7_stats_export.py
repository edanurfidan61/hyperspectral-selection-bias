"""E7 — İstatistik tabloları + ek materyal dışa aktarımı  → R3-39,51,53,54,59,61,75,76,81,83,86

Bu modül YENİ model koşusu yapmaz; E0–E6'nın diskteki özet çıktılarını (ve makalenin
otoriter ``outputs3/tables_foldmean`` tablolarını) birincil / ikincil tablolara,
ek materyale ve belgelere dönüştürür. İki küçük hesap istisnası vardır, ikisi de
diskteki kayıtlardan:

* **GA × PLS eşli grup bootstrap'ı** (E6-G1): ``outputs_rev1/E6_ga/samples/folds.csv.gz``
  (git-ignored örnek düzeyi tahminler) üzerinde, E1 ile AYNI fonksiyon
  (:func:`common.group_bootstrap_paired`) ve AYNI seed türetimi
  (``derive_seed(base, repeat, −1, "bootstrap")``). Böylece GA satırı E1 satırlarıyla
  aynı belirsizlik tanımını taşır.
* **Eski maske örtüşmesi formülünün yeniden kurulumu** (R3-76): ``mask_overlap.csv``'yi
  üreten betik yayında yok. Orijinal bölmede (shuffle'sız GroupKFold(5), sabit seed)
  deterministik seçicilerin tüm-veri maskesi A ve fold maskeleri N_f yeniden hesaplanır
  (``MaskCache``; ilk koşu LASSO yüzünden birkaç dk, sonra anında) ve aday formüller
  tabloyla karşılaştırılır. Eşleşen formül ``metrics_definitions.md``'ye yazılır.

Çıktılar (``outputs_rev1/E7_stats_export/``)
--------------------------------------------
Birincil (R3-61 ``analysis_roles.csv`` → role = primary):
  ``primary_table.csv`` (+ ``primary_table_display.csv``): veri–hedef × model × seçici;
  apparent / nested / Δ fold-mean (birincil) VE pooled R² (potato için birlikte birincil —
  test fold'u ≈ 14 örnek, fold-mean R² çok oynak), tekrarlar arası 2.5–97.5 persentil,
  eşli grup bootstrap CI'ının tekrarlar boyunca medyanı, ΔRMSE, P1−P2 karşıtlığı.
  GA satırları E6-G1'den (R = 10, E1'in ilk 10 bölmesiyle AYNI).
  ``primary_by_family.csv``: seçici ailesi bazında medyan Δ (R3-5).
İkincil / duyarlılık / keşifsel:
  ``mechanism_table.csv`` (E2 ayrıştırma), ``lasso_2x2_table.csv``, ``stability_table.csv``,
  ``sensitivity_table.csv`` (E3), ``simulation_null_table.csv`` / ``simulation_grid_table.csv``
  / ``simulation_signal_table.csv`` (E4), ``permutation_table.csv`` (E5; D2 işaretli),
  ``ga_variant_table.csv`` + ``ga_adaptive_overfitting.csv`` (E6), ``compute_budget.csv``
  (R3-39).
Eski 12-seed (outputs3) ek tablolar:
  ``seed_wilcoxon.csv`` (R3-51/53/54): düzeltilmemiş + BH p; aile = 8 stokastik
  kombinasyon (GA × 4 + MI × 4); Cliff's δ YOK, yerine eşli fark özeti
  (ortalama, medyan, Q1, Q3, IQR); 0.02 eşiği YOK.
  ``spearman_overlap.csv`` (R3-75): 12 satır (eski) / 9 benzersiz seçim; paylaşılan
  maskeleri koruyan TAM permütasyon p'si (9! = 362 880 atama).
  ``legacy_overlap_reconstruction.csv`` (R3-76).
Belgeler: ``analysis_roles.csv``, ``metrics_definitions.md``, ``reporting_checklist.md``.
Ek materyal: ``supplementary/TableSxx_*.csv`` + ``supplementary/INDEX.md`` +
``supplementary/supplementary_index.csv``.
Kök: ``expected_values.json`` — ``--freeze-expected`` ile dondurulur;
``reproduce_revision.py`` doğrular (R3-81).

Terminoloji (§0.8): apparent / nested / "apparent–nested difference"; işaret:
pozitif = apparent daha iyimser (R²: P2 − P3; RMSE: P3 − P2).

Çalıştırma::

    python -m src.m07_revision.e7_stats_export                  # tam (outputs_rev1/E7_stats_export)
    python -m src.m07_revision.e7_stats_export --quick          # duman testi → outputs_rev1/_quick/
    python -m src.m07_revision.e7_stats_export --check          # çıktılar + yeniden üretim kontrolleri
    python -m src.m07_revision.e7_stats_export --freeze-expected  # expected_values.json'u yaz
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

from src.core.splits import derive_seed, make_outer_splits
from src.m07_revision import common as C

EXP = "E7_stats_export"
ROOT = C.ROOT
OUT3 = ROOT / "outputs3" / "tables_foldmean"
OUT3_POOLED = ROOT / "outputs3" / "tables_pooled"
REV1 = ROOT / "outputs_rev1"
EXPECTED_PATH = ROOT / "expected_values.json"

#: Tablolarda satır sırası (makale sırası: arama → sarmalayıcı → gömülü → filtre)
SELECTOR_ORDER = ["ga", "rfe", "lasso", "mutual_info"]
MODEL_ORDER = ["pls", "lightgbm"]
FAMILY = {"ga": "search", "rfe": "wrapper", "lasso": "embedded", "mutual_info": "filter"}
#: Seed boyunca değişen (stokastik) seçiciler → seed-Wilcoxon tanımlı (config family)
STOCHASTIC = {"ga", "mutual_info"}

NOTE_GRAPE = ("fold-partitioning contrast (P1−P2) ≡ 0 by construction: every leaf is its own group, "
              "so GroupKFold(shuffle=True, random_state=r) gives the same partition as "
              "KFold(shuffle=True, random_state=r)")
NOTE_POTATO = ("outer test folds ≈ 14 samples (72/5): fold-mean R² is volatile; pooled R² is reported "
               "alongside as a co-primary quantity")
NOTE_GA = ("E6-G1: full GA budget (pop 150, ngen 150), R = 10 repeats on the same outer splits as "
           "E1 repeats 0–9")


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def _order(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Satırları config combo sırası → model → seçici sırasına diz."""
    combos = [(c["dataset"], c["target"]) for c in cfg["common"]["combos"]]
    out = df.copy()
    out["_c"] = [combos.index((d, t)) if (d, t) in combos else 99
                 for d, t in zip(out["dataset"], out["target"])]
    sort = ["_c"]
    if "model" in out:
        out["_m"] = out["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(9)
        sort.append("_m")
    if "selector" in out:
        out["_s"] = out["selector"].map({s: i for i, s in enumerate(SELECTOR_ORDER)}).fillna(9)
        sort.append("_s")
    return out.sort_values(sort, kind="stable").drop(columns=[c for c in ("_c", "_m", "_s")
                                                             if c in out]).reset_index(drop=True)


def _wide(df: pd.DataFrame, keys: list[str], quantities: list[str], stats: list[str],
          qcol: str = "quantity", rename: dict | None = None) -> pd.DataFrame:
    """Uzun özet (keys × quantity × {mean, median, …}) → geniş tablo ``<q>_<stat>``."""
    d = df[df[qcol].isin(quantities)]
    w = d.pivot_table(index=keys, columns=qcol, values=stats, aggfunc="first")
    w.columns = [f"{(rename or {}).get(q, q)}_{s}" for s, q in w.columns]
    cols = [f"{(rename or {}).get(q, q)}_{s}" for q in quantities for s in stats
            if f"{(rename or {}).get(q, q)}_{s}" in w.columns]
    return w[cols].reset_index()


def _fmt(v, nd: int = 3) -> str:
    return "" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _fmt_ci(m, lo, hi, nd: int = 3) -> str:
    if m is None or not np.isfinite(m):
        return ""
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# 1) Eski 12-seed: seed-Wilcoxon + BH (R3-51, 53, 54)
# ---------------------------------------------------------------------------
def seed_wilcoxon(cfg: dict) -> pd.DataFrame:
    """outputs3 12-seed eşli (apparent, nested) çiftleri → Wilcoxon + BH + eşli fark özeti.

    * Test yalnız STOKASTİK seçicilerde (GA, MI) tanımlı: LASSO/RFE deterministik →
      fark seed'ler boyunca sabit (etkin n = 1) → p tanımsız (NaN), fark yine raporlanır.
    * BH ailesi = 8 stokastik kombinasyon (GA × 4 + MI × 4), aggregation başına ayrı.
    * Cliff's δ YERİNE eşli fark özeti: ortalama, medyan, Q1, Q3, IQR, min, max,
      pozitif seed sayısı. Pratik eşik (0.02) YOK.
    * Fold-mean satırları outputs3/tables_foldmean/stats_summary.csv p_value / p_value_fdr
      ile aynı olmalı (``reproduces_outputs3`` sütunu; --check bunu doğrular).
    """
    from src.m06_selection_bias.stats_tests import is_degenerate

    raw = pd.read_csv(OUT3 / "selection_bias_multiseed_raw.csv")
    raw["nested_r2_pooled"] = raw["apparent_r2_pooled"] - raw["bias_pooled"]
    ref = pd.read_csv(OUT3 / "stats_summary.csv", comment="#").dropna(subset=["dataset"])
    ref = ref.set_index(["dataset", "target", "model", "selector"])
    wcfg = cfg["E7_stats_export"]["seed_wilcoxon"]
    seeds = set(wcfg["seeds"])
    raw = raw[raw.seed.isin(seeds)]
    rows = []
    for agg, a_col, n_col in (("foldmean", "apparent_r2", "nested_r2"),
                              ("pooled", "apparent_r2_pooled", "nested_r2_pooled")):
        for (ds, tg, md, sel), g in raw.groupby(["dataset", "target", "model", "selector"], sort=False):
            g = g.sort_values("seed")
            a, n = g[a_col].to_numpy(float), g[n_col].to_numpy(float)
            d = a - n
            stoch = sel in STOCHASTIC
            degen = bool(is_degenerate(d))
            stat = p = np.nan
            if stoch and not degen:
                try:
                    st = wilcoxon(a, n)                 # stats_tests._wilcoxon_safe ile aynı çağrı
                    stat, p = float(st.statistic), float(st.pvalue)
                except ValueError:
                    pass
            q1, q3 = np.quantile(d, [0.25, 0.75])
            rows.append({
                "aggregation": agg, "dataset": ds, "target": tg, "model": md, "selector": sel,
                "family": FAMILY[sel], "n_seeds": len(g), "seeds": ",".join(map(str, g.seed)),
                "stochastic_selector": stoch, "n_unique_difference": int(np.unique(d).size),
                "test_defined": bool(stoch and not degen and np.isfinite(p)),
                "apparent_r2_mean": float(a.mean()), "nested_r2_mean": float(n.mean()),
                "difference_mean": float(d.mean()), "difference_median": float(np.median(d)),
                "difference_q1": float(q1), "difference_q3": float(q3), "difference_iqr": float(q3 - q1),
                "difference_min": float(d.min()), "difference_max": float(d.max()),
                "n_seeds_positive": int((d > 0).sum()),
                "wilcoxon_statistic": stat, "p_unadjusted": p,
            })
    df = pd.DataFrame(rows)
    df["p_bh"] = np.nan
    df["bh_family"] = ""
    df["bh_family_size"] = 0
    for agg in ("foldmean", "pooled"):
        fam = (df.aggregation == agg) & df.stochastic_selector
        ok = fam & df.p_unadjusted.notna()
        if ok.any():
            df.loc[ok, "p_bh"] = multipletests(df.loc[ok, "p_unadjusted"], method="fdr_bh")[1]
        df.loc[fam, "bh_family"] = f"{wcfg['family']} ({agg})"
        df.loc[fam, "bh_family_size"] = int(ok.sum())
    df["note"] = np.where(df.stochastic_selector, "",
                          "deterministic selector: difference constant across seeds (effective n = 1); "
                          "seed-based test undefined")
    # outputs3 ile eşleşme (yalnız fold-mean; outputs3 pooled p hesaplamamıştı)
    p3, q3_ = [], []
    for _, r in df.iterrows():
        k = (r.dataset, r.target, r.model, r.selector)
        if r.aggregation == "foldmean" and k in ref.index:
            p3.append(float(ref.at[k, "p_value"]))
            q3_.append(float(ref.at[k, "p_value_fdr"]))
        else:
            p3.append(np.nan)
            q3_.append(np.nan)
    df["p_outputs3"], df["p_bh_outputs3"] = p3, q3_
    both = df.p_unadjusted.notna() | df.p_outputs3.notna()
    df["reproduces_outputs3"] = np.where(
        df.aggregation != "foldmean", "",
        np.where(both, (np.isclose(df.p_unadjusted, df.p_outputs3, rtol=0, atol=1e-12)
                        & np.isclose(df.p_bh, df.p_bh_outputs3, rtol=0, atol=1e-12)).astype(str),
                 "True"))
    return _order(df.sort_values("aggregation", kind="stable"), cfg).sort_values(
        "aggregation", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2) Spearman: maske örtüşmesi ↔ apparent–nested difference (R3-75)
# ---------------------------------------------------------------------------
def exact_unit_permutation_spearman(x_unit: np.ndarray, y_rows: np.ndarray,
                                    unit_of_row: np.ndarray) -> dict:
    """Paylaşılan maskeleri KORUYAN tam permütasyon testi.

    x (örtüşme) bir SEÇİMİN özelliğidir; aynı maskeyi paylaşan satırlar (ör. ryckewaert/
    flavonol'da PLS ve LightGBM, LASSO/RFE/MI maskeleri model-bağımsız) aynı x'i taşır.
    Satırları bağımsız permütelemek bu bağımlılığı yok sayar. Burada x değerleri
    BİRİMLER (benzersiz seçimler) arasında permüte edilir; her satır kendi biriminin
    permüte x'ini alır. U! atamanın HEPSİ sayılır (U = 9 → 362 880), Monte Carlo yok.

    Dönüş: rho, p_two_sided = P(|ρ*| ≥ |ρ_obs|), p_less = P(ρ* ≤ ρ_obs), n_perm.
    """
    x_unit = np.asarray(x_unit, float)
    y_rows = np.asarray(y_rows, float)
    unit_of_row = np.asarray(unit_of_row, int)
    U = len(x_unit)
    obs = float(spearmanr(x_unit[unit_of_row], y_rows).statistic)
    perms = np.array(list(permutations(range(U))), dtype=np.int16)          # U! × U
    Xp = x_unit[perms][:, unit_of_row]                                       # U! × R
    rx = rankdata(Xp, axis=1)
    ry = rankdata(y_rows)
    rx = rx - rx.mean(axis=1, keepdims=True)
    ry = ry - ry.mean()
    rho = (rx @ ry) / (np.sqrt((rx ** 2).sum(axis=1)) * np.sqrt((ry ** 2).sum()))
    eps = 1e-12
    return {"rho": obs, "p_perm_two_sided": float(np.mean(np.abs(rho) >= abs(obs) - eps)),
            "p_perm_less": float(np.mean(rho <= obs + eps)), "n_perm": int(len(perms))}


def spearman_overlap(cfg: dict, quick: bool = False) -> pd.DataFrame:
    """Örtüşme (overlap fraction / Jaccard) ile apparent–nested difference arasında Spearman.

    Kaynaklar ve sürümler:
    * ``outputs3_rows12`` — makaledeki 12 satır (GA hariç 16 − 4): ryckewaert/flavonol
      PLS ve LightGBM aynı maskeyi paylaştığından yalnız 9 BENZERSİZ seçim vardır.
      p: birimler (9 seçim) arası tam permütasyon.
    * ``outputs3_unique9_pls`` — benzersiz seçim başına bir satır (PLS Δ).
    * ``outputs3_unique9_model_mean`` — benzersiz seçim başına Δ'nın modeller arası ortalaması.
    * ``rev1_E2_unique9`` — Revizyon E2 (R = 20 tekrarlı bölme, PLS): tekrarlar boyunca
      medyan Jaccard(N_f, A) ve medyan Δ fold-mean (yalnız Jaccard; overlap fraction E2'de yok).
    ``p_asymptotic`` (scipy, satırları bağımsız sayar) yalnız karşılaştırma içindir.
    """
    ov = pd.read_csv(OUT3 / "mask_overlap.csv", comment="#").dropna(subset=["mean_jaccard"])
    ms = pd.read_csv(OUT3 / "selection_bias_multiseed.csv")
    d = ov.merge(ms[["dataset", "target", "model", "selector", "bias_mean"]],
                 on=["dataset", "target", "model", "selector"], how="left")
    d["unit"] = d.dataset + "/" + d.target + "/" + d.selector
    units = list(dict.fromkeys(d.unit))
    rows = []

    def add(source, version, x_name, x_unit, y_rows, unit_of_row, n_units, extra=""):
        r = exact_unit_permutation_spearman(x_unit, y_rows, unit_of_row)
        asy = spearmanr(np.asarray(x_unit)[unit_of_row], y_rows)
        rows.append({"source": source, "version": version, "x": x_name,
                     "y": "apparent_nested_difference_r2_foldmean", "n_rows": len(y_rows),
                     "n_unique_selections": n_units, **r, "p_asymptotic": float(asy.pvalue),
                     "permutation": "exact; x permuted across unique selections (shared masks kept)",
                     "note": extra})

    for x_name, col in (("overlap_fraction", "mean_overlap_frac"), ("jaccard", "mean_jaccard")):
        xu = d.groupby("unit", sort=False)[col].first().reindex(units).to_numpy()
        # Aynı birimde x gerçekten aynı mı? (maske paylaşımı varsayımının kontrolü)
        if (d.groupby("unit")[col].nunique() > 1).any():
            raise RuntimeError(f"{col}: aynı seçimde farklı örtüşme — maske paylaşımı varsayımı bozuk")
        uor = d.unit.map({u: i for i, u in enumerate(units)}).to_numpy()
        add("outputs3", "rows12", x_name, xu, d.bias_mean.to_numpy(), uor, len(units),
            "paper analysis: 12 rows, 9 unique selections (ryckewaert/flavonol PLS and LightGBM share masks)")
        p = d[d.model == "pls"].set_index("unit").reindex(units)
        add("outputs3", "unique9_pls", x_name, xu, p.bias_mean.to_numpy(), np.arange(len(units)),
            len(units), "one row per unique selection; PLS difference")
        mm = d.groupby("unit", sort=False).bias_mean.mean().reindex(units)
        add("outputs3", "unique9_model_mean", x_name, xu, mm.to_numpy(), np.arange(len(units)),
            len(units), "one row per unique selection; difference averaged over models")

    # Revizyon E2: tekrarlı bölmeler (PLS) — 9 benzersiz seçim
    st = pd.read_csv(REV1 / "E2_mechanism" / "stability_by_repeat.csv")
    de = pd.read_csv(REV1 / "E2_mechanism" / "decomposition_by_repeat.csv")
    k = ["dataset", "target", "selector", "repeat"]
    m = st[k + ["jaccard_vs_apparent__P3_nested"]].merge(
        de[k + ["total_apparent_nested_difference"]], on=k)
    g = m.groupby(["dataset", "target", "selector"]).median(numeric_only=True).reset_index()
    g = _order(g, cfg)
    add("rev1_E2", "unique9_median_over_repeats", "jaccard",
        g.jaccard_vs_apparent__P3_nested.to_numpy(), g.total_apparent_nested_difference.to_numpy(),
        np.arange(len(g)), len(g),
        f"E2 PLS, R = {int(m.repeat.nunique())}: median over repeats of mean_f J(N_f, A) and of Δ")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3) Eski örtüşme formülünün yeniden kurulumu (R3-76)
# ---------------------------------------------------------------------------
OVERLAP_CANDIDATES = {
    "inter_over_kA": ("|A ∩ N_f| / |A|", lambda i, a, n, u: i / a),
    "inter_over_k": ("|A ∩ N_f| / k (k = 20)", lambda i, a, n, u: i / 20.0),
    "inter_over_kN": ("|A ∩ N_f| / |N_f|", lambda i, a, n, u: i / n),
    "inter_over_max": ("|A ∩ N_f| / max(|A|, |N_f|)", lambda i, a, n, u: i / np.maximum(a, n)),
    "inter_over_min": ("|A ∩ N_f| / min(|A|, |N_f|)", lambda i, a, n, u: i / np.minimum(a, n)),
    "jaccard": ("|A ∩ N_f| / |A ∪ N_f|", lambda i, a, n, u: i / u),
}


def legacy_overlap_reconstruction(cfg: dict, quick: bool, mask_root: Path,
                                  log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """outputs3 ``mask_overlap.csv``'yi orijinal bölmede yeniden hesapla, formülü belirle.

    Orijinal hat: dış bölme shuffle'sız GroupKFold(5); seçici seed'i sabit
    (``seed_mode="fixed"``); LASSO ``inner="legacy"`` (LassoCV(cv=5)). MI seed'e bağlı →
    seed 0–11 ayrı ayrı ve ortalama; RFE/LASSO deterministik (seed 0).
    Dönüş: (fold düzeyi ham sayımlar, aday formül × seçim karşılaştırması).
    """
    from src.m07_revision.engine import MaskCache, resolve_selector, selector_uses_seed

    cache = MaskCache(mask_root)
    ref = pd.read_csv(OUT3 / "mask_overlap.csv", comment="#").dropna(subset=["mean_jaccard"])
    cases = [("rfe", {}, [0]), ("lasso", {"inner": "legacy"}, [0]),
             ("mutual_info", {}, list(range(12)))]
    if quick:
        cases = [("rfe", {}, [0]), ("mutual_info", {}, [0, 1])]
    k = int(cfg["common"]["k"])
    raw = []
    for c in cfg["common"]["combos"]:
        ds, tg = c["dataset"], c["target"]
        d = C.load_combo(ds, tg)
        sp = make_outer_splits(d["n"], d["groups"], "group", 5, repeat=0, shuffle=False)
        for sel, kw, seeds in cases:
            fn = resolve_selector(sel)
            for s in seeds:
                t0 = time.time()

                def mask_of(rows):
                    key = MaskCache.make_key({"dataset": ds, "target": tg}, sel, kw, k, rows,
                                             s if selector_uses_seed(sel, kw, True) else None, None)
                    hit = cache.get(key)
                    if hit is not None:
                        return hit[0]
                    m, info = fn(d["X"][rows], d["y"][rows], k, groups=d["groups"][rows], model="pls",
                                 seed=s, return_info=True, **kw)
                    m = np.asarray(m, bool)
                    cache.put(key, m, dict(info), {"experiment": EXP, "purpose": "legacy_overlap",
                                                   "seed": s})
                    return m

                A = mask_of(np.arange(d["n"]))
                for f, (tr, _) in enumerate(sp):
                    N = mask_of(np.asarray(tr))
                    raw.append({"dataset": ds, "target": tg, "selector": sel, "seed": s, "fold": f,
                                "k_apparent": int(A.sum()), "k_nested": int(N.sum()),
                                "intersection": int((A & N).sum()), "union": int((A | N).sum())})
                log(f"  örtüşme yeniden kurulumu: {ds}/{tg}/{sel} seed {s} [{time.time() - t0:.1f} s]")
    raw = pd.DataFrame(raw)
    i, a, n, u = (raw[c].to_numpy(float) for c in ("intersection", "k_apparent", "k_nested", "union"))
    for name, (_, fn) in OVERLAP_CANDIDATES.items():
        raw[name] = fn(i, a, n, u)
    comp = []
    # Seed kümeleri: "seed0" (tek seçim) ve "seeds0-11_mean" (MI'de 12 seed ortalaması;
    # RFE/LASSO deterministik → iki kümede aynı değer). Hangisinin outputs3'ü verdiği de bulgudur.
    seed_sets = {"seed0": [0], "seeds0-11_mean": list(range(12))}
    for (ds, tg, sel), g in raw.groupby(["dataset", "target", "selector"], sort=False):
        r = ref[(ref.dataset == ds) & (ref.target == tg) & (ref.selector == sel) & (ref.model == "pls")]
        if r.empty:
            continue
        r = r.iloc[0]
        for set_name, ss in seed_sets.items():
            gs = g[g.seed.isin(ss)] if sel == "mutual_info" else g
            if sel == "mutual_info" and gs.seed.nunique() < len(ss):
                continue                                   # --quick: eksik seed kümesi
            for name, (formula, _) in OVERLAP_CANDIDATES.items():
                v = float(gs[name].mean())                 # fold'lar (ve seed kümesi) üzerinden ortalama
                for target_col in ("mean_overlap_frac", "mean_jaccard"):
                    comp.append({"dataset": ds, "target": tg, "selector": sel, "seed_set": set_name,
                                 "seeds": ",".join(map(str, sorted(gs.seed.unique()))),
                                 "candidate": name, "formula": formula, "outputs3_column": target_col,
                                 "reconstructed": v, "outputs3": float(r[target_col]),
                                 "absdiff": abs(v - float(r[target_col])),
                                 "k_apparent_mean": float(gs.k_apparent.mean()),
                                 "k_nested_mean": float(gs.k_nested.mean()),
                                 "outputs3_k_apparent": float(r.mean_apparent_k),
                                 "outputs3_k_nested": float(r.mean_nested_k)})
    comp = pd.DataFrame(comp)
    return raw, comp


def overlap_verdict(comp: pd.DataFrame, tol: float = 1e-9) -> dict:
    """Her outputs3 sütunu için TÜM seçimlerde eşleşen (aday formül, MI seed kümesi) çiftleri."""
    out = {}
    for col, g in comp.groupby("outputs3_column"):
        res = {}
        for set_name, gs in g.groupby("seed_set"):
            ok = gs.groupby("candidate").absdiff.max()
            res[set_name] = {"matching": sorted(ok[ok <= tol].index.tolist()),
                             "max_absdiff_by_candidate": {k: float(v) for k, v in ok.items()}}
        matching = sorted({c for v in res.values() for c in v["matching"]})
        out[col] = {"matching": matching,
                    "matching_seed_sets": sorted(s for s, v in res.items() if v["matching"]),
                    "by_seed_set": res,
                    "n_selections": int(g[["dataset", "target", "selector"]].drop_duplicates().shape[0])}
    return out


# ---------------------------------------------------------------------------
# 4) Birincil tablo (E1 + E6-G1)
# ---------------------------------------------------------------------------
Q_PRIMARY = {
    "P2_apparent.r2_foldmean": "apparent_r2_foldmean",
    "P3_nested.r2_foldmean": "nested_r2_foldmean",
    "apparent_nested_difference_r2_foldmean": "delta_r2_foldmean",
    "P2_apparent.r2_pooled": "apparent_r2_pooled",
    "P3_nested.r2_pooled": "nested_r2_pooled",
    "apparent_nested_difference_r2_pooled": "delta_r2_pooled",
    "apparent_nested_difference_rmse_foldmean": "delta_rmse_foldmean",
    "fold_partitioning_contrast_r2_foldmean": "fold_partitioning_contrast_r2_foldmean",
    "P1_random.r2_foldmean": "P1_random_r2_foldmean",
    "no_selection.r2_foldmean": "no_selection_r2_foldmean",
    "random_k.r2_foldmean": "random_k_r2_foldmean",
    "P3_nested.n_selected_mean": "nested_n_selected",
}
STATS = ["mean", "median", "p2_5", "p97_5"]


def ga_bootstrap(cfg: dict, e6_dir: Path, B: int) -> pd.DataFrame:
    """E6-G1 örnek düzeyi tahminlerinden eşli grup bootstrap (E1 ile aynı tanım ve seed)."""
    f = e6_dir / "samples" / "folds.csv.gz"
    if not f.exists():
        return pd.DataFrame()
    s = pd.read_csv(f, usecols=["variant", "dataset", "target", "repeat", "arm", "test_idx", "group_id",
                                "y_true", "y_pred"], dtype={"group_id": str})
    s = s[(s.variant == "G1") & s.arm.isin(["P2_apparent", "P3_nested"])]
    base = cfg["common"]["base_seed"]
    rows = []
    for (ds, tg, r), g in s.groupby(["dataset", "target", "repeat"]):
        a = g[g.arm == "P2_apparent"].sort_values("test_idx")
        b = g[g.arm == "P3_nested"].sort_values("test_idx")
        if not np.array_equal(a.test_idx.to_numpy(), b.test_idx.to_numpy()):
            raise RuntimeError(f"GA bootstrap: {ds}/{tg}/r{r} P2/P3 örnekleri eşleşmiyor")
        bo = C.group_bootstrap_paired(a.y_true.to_numpy(), a.y_pred.to_numpy(), b.y_pred.to_numpy(),
                                      a.group_id.to_numpy(), B, derive_seed(base, int(r), -1, "bootstrap"))
        rows.append({"dataset": ds, "target": tg, "selector": "ga", "family": "search", "model": "pls",
                     "repeat": int(r), **bo})
    return pd.DataFrame(rows)


def _per_repeat_delta_e1(e1_dir: Path) -> pd.DataFrame:
    """E1 fold kayıtlarından tekrar başına Δ fold-mean (pozitif tekrar sayısı için)."""
    f = pd.read_csv(e1_dir / "fold_metrics.csv.gz",
                    usecols=["repeat", "fold", "arm", "r2", "dataset", "target", "selector", "model"])
    f = f[f.arm.isin(["P2_apparent", "P3_nested"])]
    m = f.groupby(["dataset", "target", "selector", "model", "repeat", "arm"]).r2.mean().unstack("arm")
    return (m["P2_apparent"] - m["P3_nested"]).rename("delta").reset_index()


def primary_table(cfg: dict, quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Birincil tablo: E1 (RFE/LASSO/MI × PLS/LightGBM, R = 20) + E6-G1 (GA × PLS, R = 10).

    Dönüş: (primary, display, ga_bootstrap_by_repeat).
    """
    e1 = REV1 / "E1_repeated"
    e6 = REV1 / "E6_ga"
    keys = ["dataset", "target", "selector", "family", "model"]
    s1 = pd.read_csv(e1 / "summary_by_combo.csv")
    w1 = _wide(s1, keys, list(Q_PRIMARY), STATS, rename=Q_PRIMARY)
    n1 = s1[s1.quantity == "apparent_nested_difference_r2_foldmean"].set_index(keys)["n"]
    w1["n_repeats"] = w1.set_index(keys).index.map(n1).astype(int)
    w1["source"] = "E1_repeated"
    pr = _per_repeat_delta_e1(e1)
    npos = pr.assign(pos=pr.delta > 0).groupby(["dataset", "target", "selector", "model"]).pos.sum()
    w1["n_repeats_delta_positive"] = [int(npos.get((d, t, s, m), 0)) for d, t, s, m in
                                      zip(w1.dataset, w1.target, w1.selector, w1.model)]
    # Kontrol: fold kayıtlarından Δ ortalaması özetle aynı mı?
    chk = pr.groupby(["dataset", "target", "selector", "model"]).delta.mean()
    for _, r in w1.iterrows():
        v = chk[(r.dataset, r.target, r.selector, r.model)]
        if abs(v - r.delta_r2_foldmean_mean) > 1e-10:
            raise RuntimeError(f"E1 Δ tutarsız: {r.dataset}/{r.target}/{r.selector}/{r.model}")
    ci = pd.read_csv(e1 / "delta_ci.csv")
    ci_med = ci[ci["repeat"].astype(str) == "median_over_repeats"].set_index(keys)
    B1 = int(ci["B"].dropna().iloc[0])

    # GA × PLS (E6-G1): tekrar başına değerler → aynı istatistikler
    s6 = pd.read_csv(e6 / "summary_by_combo.csv")
    s6 = s6[s6.variant == "G1"]
    long6 = (s6.groupby(keys + ["quantity"]).value.apply(lambda v: pd.Series(C.describe(v)))
             .unstack().reset_index())
    w6 = _wide(long6, keys, [q for q in Q_PRIMARY if q in set(long6.quantity)], STATS, rename=Q_PRIMARY)
    d6 = s6[s6.quantity == "apparent_nested_difference_r2_foldmean"]
    w6["n_repeats"] = int(d6.repeat.nunique())
    w6["n_repeats_delta_positive"] = [int((d6[(d6.dataset == d) & (d6.target == t)].value > 0).sum())
                                      for d, t in zip(w6.dataset, w6.target)]
    w6["source"] = "E6_ga.G1"
    B = 200 if quick else B1
    gb = ga_bootstrap(cfg, e6, B)
    prim = pd.concat([w1, w6], ignore_index=True)
    lo, hi, bB = [], [], []
    for _, r in prim.iterrows():
        k = (r.dataset, r.target, r.selector, r.family, r.model)
        if r.selector == "ga":
            g = gb[(gb.dataset == r.dataset) & (gb.target == r.target)] if not gb.empty else gb
            lo.append(float(g.delta_r2_lo.median()) if len(g) else np.nan)
            hi.append(float(g.delta_r2_hi.median()) if len(g) else np.nan)
            bB.append(B if len(g) else np.nan)
        else:
            lo.append(float(ci_med.at[k, "delta_r2_lo"]))
            hi.append(float(ci_med.at[k, "delta_r2_hi"]))
            bB.append(B1)
    prim["boot_delta_r2_pooled_lo_median"] = lo
    prim["boot_delta_r2_pooled_hi_median"] = hi
    prim["boot_B"] = bB
    # Potato LOGO (24 Field; yalnız pooled; repeat 0) — E1 seçicileri
    logo = pd.read_csv(e1 / "logo_potato.csv")
    lg = logo[logo.arm == "apparent_nested_difference"].set_index(["dataset", "target", "selector", "model"])
    prim["logo_delta_r2_pooled"] = [
        float(lg.at[(d, t, s, m), "r2_pooled"]) if (d, t, s, m) in lg.index else np.nan
        for d, t, s, m in zip(prim.dataset, prim.target, prim.selector, prim.model)]
    notes = []
    for _, r in prim.iterrows():
        n = [NOTE_GRAPE if r.dataset == "ryckewaert" else NOTE_POTATO]
        if r.selector == "ga":
            n.append(NOTE_GA)
        notes.append(" | ".join(n))
    prim["note"] = notes
    lead = keys + ["source", "n_repeats", "n_repeats_delta_positive"]
    rest = [c for c in prim.columns if c not in lead and c != "note"]
    first = ["apparent_r2_foldmean_mean", "nested_r2_foldmean_mean", "delta_r2_foldmean_mean",
             "delta_r2_foldmean_median", "delta_r2_foldmean_p2_5", "delta_r2_foldmean_p97_5",
             "apparent_r2_pooled_mean", "nested_r2_pooled_mean", "delta_r2_pooled_median",
             "boot_delta_r2_pooled_lo_median", "boot_delta_r2_pooled_hi_median", "boot_B",
             "logo_delta_r2_pooled", "delta_rmse_foldmean_median",
             "fold_partitioning_contrast_r2_foldmean_median"]
    prim = prim[lead + first + [c for c in rest if c not in first] + ["note"]]
    prim = _order(prim, cfg)

    disp = pd.DataFrame({
        "dataset": prim.dataset, "target": prim.target, "model": prim.model, "selector": prim.selector,
        "R": prim.n_repeats,
        "apparent R² (fold-mean)": prim.apparent_r2_foldmean_mean.map(_fmt),
        "nested R² (fold-mean)": prim.nested_r2_foldmean_mean.map(_fmt),
        "Δ fold-mean: median [2.5%, 97.5%] over repeats": [
            _fmt_ci(m, lo_, hi_) for m, lo_, hi_ in zip(prim.delta_r2_foldmean_median,
                                                       prim.delta_r2_foldmean_p2_5,
                                                       prim.delta_r2_foldmean_p97_5)],
        "repeats with Δ > 0": [f"{p}/{n}" for p, n in zip(prim.n_repeats_delta_positive, prim.n_repeats)],
        "apparent R² (pooled)": prim.apparent_r2_pooled_mean.map(_fmt),
        "nested R² (pooled)": prim.nested_r2_pooled_mean.map(_fmt),
        "Δ pooled: median [group-bootstrap 95% CI, median over repeats]": [
            _fmt_ci(m, lo_, hi_) for m, lo_, hi_ in zip(prim.delta_r2_pooled_median,
                                                       prim.boot_delta_r2_pooled_lo_median,
                                                       prim.boot_delta_r2_pooled_hi_median)],
        "note": prim.note,
    })
    return prim, disp, gb


def family_table(cfg: dict, prim: pd.DataFrame) -> pd.DataFrame:
    """Seçici ailesi bazında medyan Δ (R3-5): E1 by_family + GA (E6-G1)."""
    fam = pd.read_csv(REV1 / "E1_repeated" / "by_family.csv").assign(source="E1_repeated")
    s6 = pd.read_csv(REV1 / "E6_ga" / "summary_by_combo.csv")
    d6 = s6[(s6.variant == "G1") & (s6.quantity == "apparent_nested_difference_r2_foldmean")]
    ga = (d6.groupby(["dataset", "target", "family", "model"]).value
          .apply(lambda v: pd.Series(C.describe(v))).unstack().reset_index()
          .assign(quantity="apparent_nested_difference_r2_foldmean", source="E6_ga.G1"))
    out = pd.concat([fam, ga], ignore_index=True)
    out["n"] = out["n"].astype(int)
    out["_f"] = out.family.map({f: i for i, f in enumerate(["search", "wrapper", "embedded", "filter"])})
    out = _order(out.sort_values("_f", kind="stable"), cfg).drop(columns="_f")
    return out


# ---------------------------------------------------------------------------
# 5) İkincil / duyarlılık / keşifsel tablolar
# ---------------------------------------------------------------------------
def mechanism_tables(cfg: dict) -> dict[str, pd.DataFrame]:
    e2 = REV1 / "E2_mechanism"
    keys = ["dataset", "target", "selector", "family", "model"]
    dec = pd.read_csv(e2 / "decomposition.csv")
    qs = ["total_apparent_nested_difference", "sample_size_component", "equal_size_leakage_component",
          "apparent_vs_random_k", "random_k_vs_nested", "P2_apparent.r2_foldmean",
          "size_matched_leaky.r2_foldmean", "P3_nested.r2_foldmean", "random_k.r2_foldmean",
          "no_selection.r2_foldmean"]
    mech = _order(_wide(dec, keys, qs, ["median", "p2_5", "p97_5"]), cfg)
    mech.insert(5, "n_repeats", int(dec["n"].max()))
    l22 = pd.read_csv(e2 / "lasso_2x2.csv")
    l22 = l22[l22.quantity == "r2_foldmean"]
    l22 = l22[keys + ["alpha_source", "support_source", "n", "mean", "median", "p2_5", "p97_5"]]
    st = pd.read_csv(e2 / "stability.csv")
    stq = ["jaccard_vs_apparent__P3_nested", "jaccard_pairwise_folds__P3_nested",
           "pred_corr__P2_apparent__P3_nested", "pred_meanabsdiff__P2_apparent__P3_nested",
           "pls_coef_cosine_shared__P2_P3", "n_shared_features_mean__P2_P3"]
    stab = _order(_wide(st, keys, stq, ["median", "p2_5", "p97_5"]), cfg)
    neg = pd.read_csv(e2 / "negative_differences.csv")
    return {"mechanism_table": mech, "lasso_2x2_table": _order(l22, cfg), "stability_table": stab,
            "negative_differences": neg}


def sensitivity_table(cfg: dict) -> pd.DataFrame:
    s = pd.read_csv(REV1 / "E3_sensitivity" / "sensitivity.csv")
    cols = ["axis", "value", "dataset", "target", "selector", "family", "model", "n_splits", "group_level",
            "is_default_cell", "n_repeats", "apparent_mean", "nested_mean", "delta_mean", "delta_median",
            "delta_p2_5", "delta_p97_5", "delta_pooled_mean", "delta_minus_default_mean",
            "n_features_used", "n_selected_nested_mean", "n_components_nested_mean"]
    return s[[c for c in cols if c in s.columns]]


def simulation_tables() -> dict[str, pd.DataFrame]:
    e4 = REV1 / "E4_simulation"
    nul = pd.read_csv(e4 / "s_null_summary.csv")
    nul_cols = ["generator", "selector", "n_repeats", "apparent_r2_foldmean_mean", "nested_r2_foldmean_mean",
                "random_k_r2_foldmean_mean", "delta_foldmean_mean", "delta_foldmean_ci_lo",
                "delta_foldmean_ci_hi", "delta_pooled_mean", "delta_pooled_ci_lo", "delta_pooled_ci_hi",
                "app_minus_random_k_mean", "app_minus_random_k_ci_lo", "app_minus_random_k_ci_hi",
                "random_k_minus_nested_mean", "random_k_minus_nested_ci_lo", "random_k_minus_nested_ci_hi"]
    nul = nul[nul_cols].assign(deviation=np.where(nul.selector.isin(["lasso", "mutual_info"]),
                                                  "DEVIATIONS.md D1 (prespecified reduced-repeats rule)", ""))
    sig = pd.read_csv(e4 / "s_signal_summary.csv")
    grid = pd.read_csv(e4 / "grid_summary.csv")
    return {"simulation_null_table": nul, "simulation_grid_table": grid, "simulation_signal_table": sig}


def permutation_table(cfg: dict) -> pd.DataFrame:
    p = pd.read_csv(REV1 / "E5_permutation" / "permutation_summary.csv")
    cols = ["dataset", "target", "selector", "scheme", "scheme_used", "B", "B_planned", "B_deviation",
            "statistic", "observed_nested", "observed_apparent", "observed_difference", "perm_nested_mean",
            "perm_nested_sd", "perm_nested_p95", "perm_apparent_mean", "p_nested", "p_apparent",
            "p_formula", "lasso_convergence_warnings_total"]
    p = p[cols].copy()
    p["p_min_attainable"] = 1.0 / (p["B"] + 1)
    p["B_deviation"] = p["B_deviation"].fillna("")
    p["_sc"] = p.scheme.map({"free": 0, "within_plot": 1, "field_block": 2}).fillna(9)
    return _order(p.sort_values("_sc", kind="stable"), cfg).drop(columns="_sc")


def ga_tables(cfg: dict) -> dict[str, pd.DataFrame]:
    e6 = REV1 / "E6_ga"
    var = pd.read_csv(e6 / "summary_by_variant.csv")
    ao = pd.read_csv(e6 / "adaptive_overfitting.csv")
    aog = (ao.groupby(["variant", "dataset", "target", "arm"]).inner_minus_outer
           .apply(lambda v: pd.Series(C.describe(v))).unstack().reset_index())
    aog["n"] = aog["n"].astype(int)
    aog.insert(4, "quantity", "inner_fitness_minus_outer_r2")
    g0 = pd.read_csv(e6 / "g0_reproduction.csv")
    return {"ga_variant_table": var, "ga_adaptive_overfitting": aog, "ga_g0_reproduction": g0}


def compute_budget(cfg: dict) -> pd.DataFrame:
    """Seçim başına model fit sayısı (R3-39).

    GA: değerlendirilen benzersiz aday × iç fold (``model_fits_computed``; önbelleksiz karşılığı
    ``model_fits_without_cache``). RFE: eleme turu + 1 (Ridge fit). LASSO: α grid × iç fold +
    1 (son fit). MI: 0 model fit (öznitelik başına kNN MI tahmini). Kaynak: E1 fold kayıtları
    (``selector_info``) ve E6 ``ga_runs.csv``; seçim başına medyan [min, max].
    """
    f = pd.read_csv(REV1 / "E1_repeated" / "fold_metrics.csv.gz",
                    usecols=["arm", "fold", "dataset", "target", "selector", "model", "selector_info"])
    f = f[(f.model == "pls") & ((f.arm == "P3_nested") | ((f.arm == "P2_apparent") & (f.fold == 0)))]
    info = f.selector_info.map(json.loads)
    rows = []
    for (ds, tg, sel), idx in f.groupby(["dataset", "target", "selector"]).groups.items():
        inf = info.loc[idx]
        if sel == "rfe":
            fits = inf.map(lambda d: d["n_elimination_rounds"] + 1)
            desc = "Ridge fits = elimination rounds + 1"
        elif sel == "lasso":
            fits = inf.map(lambda d: d["n_alphas"] * d["n_inner_folds"] + 1)
            desc = "coordinate-descent path fits = n_alphas × inner folds + 1 refit"
        else:
            fits = inf.map(lambda d: 0)
            desc = "no model fit (k-NN mutual-information estimate per feature)"
        secs = inf.map(lambda d: d.get("selection_seconds", np.nan)).astype(float)
        rows.append({"dataset": ds, "target": tg, "selector": sel, "source": "E1_repeated",
                     "n_selections": len(idx), "model_fits_per_selection_median": float(fits.median()),
                     "model_fits_min": float(fits.min()), "model_fits_max": float(fits.max()),
                     "model_fits_without_cache_median": float(fits.median()),
                     "selection_seconds_median": float(np.nanmedian(secs)), "definition": desc})
    g = pd.read_csv(REV1 / "E6_ga" / "ga_runs.csv")
    g = g[g.variant == "G1"]
    for (ds, tg), gg in g.groupby(["dataset", "target"]):
        rows.append({"dataset": ds, "target": tg, "selector": "ga", "source": "E6_ga.G1",
                     "n_selections": len(gg),
                     "model_fits_per_selection_median": float(gg.model_fits_computed.median()),
                     "model_fits_min": float(gg.model_fits_computed.min()),
                     "model_fits_max": float(gg.model_fits_computed.max()),
                     "model_fits_without_cache_median": float(gg.model_fits_without_cache.median()),
                     "selection_seconds_median": float(gg.ga_seconds.median()),
                     "definition": "PLS fits = unique candidates evaluated × 5 inner folds "
                                   "(candidate cache; without cache = pop × (ngen+1) × 5)"})
    return _order(pd.DataFrame(rows), cfg)


# ---------------------------------------------------------------------------
# 6) Belgeler: analiz rolleri, metrik tanımları, raporlama kontrol listesi
# ---------------------------------------------------------------------------
#: (id, analiz, rol, hakem yorumları, önceden belirlendiği yer, istatistik, tablo, kaynak, sapma)
ANALYSIS_ROLES = [
    ("A01", "Repeated paired grouped CV, PLS: apparent–nested difference (fold-mean R²)", "primary",
     "R3-4,5,52,55,57,62", "config/revision.yaml E1_repeated", "median and 2.5–97.5 percentile over R = 20 repeats",
     "primary_table.csv", "E1_repeated/summary_by_combo.csv", ""),
    ("A02", "Pooled out-of-fold R² (both arms pooled)", "primary (co-reported; co-primary for potato)",
     "R3-58; user note 2026-10-31", "config/revision.yaml common.report_aggregations",
     "mean / median over repeats", "primary_table.csv", "E1_repeated/summary_by_combo.csv", ""),
    ("A03", "Paired group bootstrap 95% CI of Δ pooled R²", "primary (uncertainty)", "R3-62",
     "config/revision.yaml E1_repeated.uncertainty.group_bootstrap",
     "conditional percentile CI (B = 2000), median over repeats", "primary_table.csv",
     "E1_repeated/delta_ci.csv; GA: E6_ga/samples (E7)", ""),
    ("A04", "GA × PLS, full budget, repeated splits (E6-G1)", "primary (GA row)", "R3-44,78",
     "config/revision.yaml E6_ga.variants.G1", "as A01, R = 10", "primary_table.csv",
     "E6_ga/summary_by_combo.csv", ""),
    ("A05", "Selector-family summary", "primary (summary)", "R3-5", "E1_repeated.summaries",
     "median Δ per dataset–target × family", "primary_by_family.csv", "E1_repeated/by_family.csv", ""),
    ("A06", "LightGBM arm (factorial selector × model)", "secondary", "R3-79,80", "E1_repeated.models",
     "as A01", "primary_table.csv (model = lightgbm)", "E1_repeated/summary_by_combo.csv", ""),
    ("A07", "Size-matched decomposition of Δ", "secondary (mechanism)", "R3-1,43,46,67", "E2_mechanism.decomposition",
     "median [2.5, 97.5] over repeats", "mechanism_table.csv", "E2_mechanism/decomposition.csv", ""),
    ("A08", "LASSO 2×2 (α source × support source)", "secondary (mechanism)", "R3-7", "E2_mechanism",
     "median fold-mean R²", "lasso_2x2_table.csv", "E2_mechanism/lasso_2x2.csv", ""),
    ("A09", "Prediction / mask / coefficient stability", "exploratory", "R3-77,47", "E2_mechanism",
     "median over repeats", "stability_table.csv", "E2_mechanism/stability.csv", ""),
    ("A10", "Negative fold-level differences (potato RFE/MI)", "exploratory", "R3-73", "E2_mechanism",
     "fold-level table", "supplementary", "E2_mechanism/negative_differences.csv", ""),
    ("A11", "One-at-a-time sensitivity grid", "sensitivity", "R3-9,15,22,23,32,33,35,36,50,84,85",
     "config/revision.yaml E3_sensitivity", "median Δ [2.5, 97.5], R = 10", "sensitivity_table.csv",
     "E3_sensitivity/sensitivity.csv", ""),
    ("A12", "Simulation S-null (y independent of X)", "secondary (simulation)", "R3-64,66,67,68",
     "config/revision.yaml E4_simulation.s_null", "paired mean difference, 95% CI", "simulation_null_table.csv",
     "E4_simulation/s_null_summary.csv", "D1"),
    ("A13", "Simulation R3-65 grid + analytic baselines", "secondary (simulation)", "R3-65", "E4_simulation.grid",
     "mean difference, CI", "simulation_grid_table.csv", "E4_simulation/grid_summary.csv", ""),
    ("A14", "Simulation S-signal (known truth θ_n, θ_0.8n)", "secondary (simulation)", "R3-2,16",
     "E4_simulation.s_signal", "E[apparent − θ_n], E[nested − θ_0.8n], corr", "simulation_signal_table.csv",
     "E4_simulation/s_signal_summary.csv", ""),
    ("A15", "Permutation test of nested R² (and apparent R²)", "secondary", "R3-34,70,71,72",
     "config/revision.yaml E5_permutation", "p = (1 + #{perm ≥ obs}) / (B + 1), one-sided",
     "permutation_table.csv", "E5_permutation/permutation_summary.csv", "D2"),
    ("A16", "GA G0: original split, seed 0 (reproduction of the published run)", "reproducibility check",
     "R3-81", "E6_ga.variants.G0", "absolute difference to outputs3 seed 0", "ga_g0_reproduction",
     "E6_ga/g0_reproduction.csv", ""),
    ("A17", "GA variants: fixed cardinality (G2), unclipped fitness (G3)", "sensitivity", "R3-27,28,74",
     "E6_ga.variants.G2/G3", "median Δ, R = 5", "ga_variant_table.csv", "E6_ga/summary_by_variant.csv", ""),
    ("A18", "GA adaptive overfitting (inner fitness − outer R²)", "exploratory", "R3-48", "E6_ga",
     "median over folds and repeats", "ga_adaptive_overfitting.csv", "E6_ga/adaptive_overfitting.csv", ""),
    ("A19", "Potato leave-one-field-out (24 folds, pooled R² only)", "sensitivity", "R3-11,13",
     "E1_repeated.potato_logo", "pooled Δ", "primary_table.csv (logo_delta_r2_pooled)",
     "E1_repeated/logo_potato.csv", ""),
    ("A20", "Seed-based Wilcoxon on the 12 published seeds (+ BH)", "legacy (supplementary)", "R3-51,53,54",
     "E7_stats_export.seed_wilcoxon", "Wilcoxon signed-rank; BH over 8 stochastic combinations",
     "seed_wilcoxon.csv", "outputs3/tables_foldmean/selection_bias_multiseed_raw.csv", ""),
    ("A21", "Spearman: mask overlap vs apparent–nested difference", "exploratory", "R3-75",
     "E7_stats_export.spearman", "Spearman ρ; exact mask-preserving permutation p", "spearman_overlap.csv",
     "outputs3/tables_foldmean/mask_overlap.csv; E2_mechanism", ""),
    ("A22", "Computational budget per selection", "descriptive", "R3-39", "E6_ga (budget table)",
     "model fits per selection", "compute_budget.csv", "E1 fold records; E6_ga/ga_runs.csv", ""),
    ("A23", "Data / feature / group / split audit", "descriptive", "R3-9,10,11,13,18–22,24,35,37,71",
     "E0_audit", "—", "supplementary", "E0_audit/*", ""),
]


def analysis_roles() -> pd.DataFrame:
    return pd.DataFrame(ANALYSIS_ROLES, columns=["id", "analysis", "role", "reviewer_comments",
                                                 "prespecified_in", "statistic", "table", "source",
                                                 "deviation"])


def metrics_definitions(verdict: dict | None, comp: pd.DataFrame | None) -> str:
    """``metrics_definitions.md`` metni (R3-76, R3-58, R3-71). Örtüşme formülü yeniden kurulumdan."""
    ov_line = "_(reconstruction not performed in this run)_"
    jac_line = ov_line
    mi_line = ("MI depends on the seed; the seed from which the outputs3 MI rows originate is identified "
               "by the reconstruction (`legacy_overlap_reconstruction.csv`, `seed_set`).")
    if verdict:
        def say(col):
            v = verdict.get(col, {})
            m = v.get("matching", [])
            if not m:
                return ("**No matching candidate formula** — see `legacy_overlap_reconstruction.csv`.")
            forms = "; ".join(f"`{OVERLAP_CANDIDATES[x][0]}`" for x in m)
            return (f"{forms}, averaged over folds — identical to the outputs3 values for all "
                    f"{v['n_selections']} selections (|difference| ≤ 1e-9; MI seed set: "
                    f"{', '.join(v['matching_seed_sets'])}).")
        ov_line, jac_line = say("mean_overlap_frac"), say("mean_jaccard")
        sets = set(verdict.get("mean_jaccard", {}).get("matching_seed_sets", []))
        if sets == {"seed0"}:
            mi_line = ("MI depends on the seed. The outputs3 MI rows were computed from the mask of **a single "
                       "selection (seed 0)** (the legacy note lists MI together with the deterministic selectors "
                       "as \"selected once on all data\"); the 12-seed mean does NOT reproduce the outputs3 "
                       "values. The revision E2 definition (below) uses each repeat's own seed.")
    lasso_note = ""
    if comp is not None and not comp.empty:
        lz = comp[(comp.selector == "lasso") & (comp.candidate == "jaccard")
                  & (comp.outputs3_column == "mean_jaccard") & (comp.seed_set == "seed0")]
        if len(lz):
            lasso_note = "; ".join(f"{r.dataset}/{r.target}: |A| = {r.k_apparent_mean:.0f}, "
                                   f"mean |N_f| = {r.k_nested_mean:.1f}" for r in lz.itertuples())
    return f"""# metrics_definitions.md — revision-1 (E7)

This file is generated by `src/m07_revision/e7_stats_export.py`; do not edit it by hand.
Terminology: **apparent** (P2), **nested** (P3), **apparent–nested difference** (Δ),
**selection-scope contrast** (P2 − P3 = Δ), **fold-partitioning contrast** (P1 − P2).

## 1. Arms (same outer split, paired design)

| Arm | Selection data | Model fit / test |
|---|---|---|
| P1_random | all data (single mask) | KFold(5, shuffle=True, random_state=r), ignoring group information |
| P2_apparent | all data (single mask) | outer GroupKFold(5, shuffle=True, random_state=r); in each fold, fit on the outer-train set and predict the outer-test set |
| P3_nested | outer-train set only | same outer folds |
| size_matched_leaky | in each fold, a random draw from all data with the same number of groups as the outer-train set (test groups may be included) | same outer folds |
| random_k / no_selection | k random features / all features | same outer folds |

## 2. Metrics and sign convention (R3-58)

- **Fold-mean R² (primary):** θ̂ = (1/K) Σ_k R²(y_{{T_k}}, ŷ_{{T_k}}), K = 5 outer folds.
- **Pooled R²:** all outer-test predictions are concatenated and a single R² is computed (same folds; the same
  aggregation in both arms). For potato, a test fold contains ≈ 14 samples, so fold-mean R² is highly variable;
  pooled R² is therefore reported alongside it in the primary table.
- **Δ (apparent–nested difference):** Δ_R² = R²(P2) − R²(P3); Δ_RMSE = RMSE(P3) − RMSE(P2);
  Δ_MAE = MAE(P3) − MAE(P2). **Positive = apparent more optimistic** (for all metrics).
- **Fold-partitioning contrast:** R²(P1) − R²(P2). For grapevine, each leaf is a separate group, so
  GroupKFold(shuffle=True, random_state=r) and KFold(shuffle=True, random_state=r) produce the same split
  → this contrast is **0 by construction** (not an error).
- **Between-repeat summary:** mean, median and 2.5–97.5th percentiles over R repeats (E1: 20; E6-G1: 10)
  (split variability; NOT a confidence interval).
- **Paired group-bootstrap CI:** in each repeat, the pooled outer-test predictions are resampled by group
  (leaf / Field) with replacement (B = 2000), using the SAME indices in both arms; 2.5–97.5th percentiles of
  Δ pooled R²; the table reports the median lower / upper bound across repeats. Conditional: the fitted models
  and masks are treated as fixed; selection and fitting uncertainty are not included in this interval.
  Seed: derive_seed(0, r, −1, "bootstrap").

## 3. Mask overlap (R3-76)

Notation: A = mask selected on all data (apparent); N_f = mask selected on the outer-train set of outer fold f
(nested); |·| = number of selected features. Folds are compared **individually** and the results are averaged
over the folds (K = 5); fold masks are not merged (no union or intersection is taken).

- **Jaccard:** J_f = |A ∩ N_f| / |A ∪ N_f|; reported value = (1/K) Σ_f J_f.
- **Overlap fraction (outputs3 `mean_overlap_frac`):** {ov_line}
- **outputs3 `mean_jaccard`:** {jac_line}
- **LASSO with k < 20:** LASSO uses k as an UPPER LIMIT (if fewer than 20 coefficients are non-zero, fewer
  features are selected; truncation/padding in `E1_repeated/lasso_truncation.csv`). In this case the
  denominators are the actual mask sizes, not the nominal k = 20: |A| for the overlap fraction and |A ∪ N_f|
  for Jaccard.
  {lasso_note}
- {mi_line}
- GA rows are empty in outputs3 (the apparent mask was not stored in the checkpoints); the revision E6 archive
  (`E6_ga/archive/*.npz`) now stores the GA masks.
- **Revision (E2):** `jaccard_vs_apparent__P3_nested` = (1/K) Σ_f J(N_f, A) (per repeat);
  `jaccard_pairwise_folds__P3_nested` = mean pairwise Jaccard index of the fold masks (K(K−1)/2 pairs).

## 4. Tests

- **Permutation (E5, R3-71):** p = (1 + #{{b : T_b ≥ T_obs}}) / (B + 1), one-sided; T = nested fold-mean R²
  (and also apparent). Grapevine: unrestricted and restricted within plot; potato: equal-size Field block
  permutation. ryckewaert × LASSO: B = 50 (DEVIATIONS.md D2), smallest attainable p = 1/51.
- **Seed-Wilcoxon (legacy 12 seeds, R3-51/53/54):** paired Wilcoxon signed-rank test (scipy default:
  exact distribution if n ≤ 50 and there are no ties), apparent vs nested. Defined only for selectors whose
  selection depends on the seed (stochastic): family = 8 stochastic combinations (GA × 4 + MI × 4);
  Benjamini–Hochberg over these 8 tests (fold-mean and pooled as separate families). Instead of Cliff's δ,
  the effect size is summarised by the paired difference (mean, median, Q1, Q3, IQR). No practical
  threshold (0.02) is used.
- **Spearman (R3-75):** ρ(overlap, Δ). In the 12-row legacy analysis, the PLS and LightGBM rows of
  ryckewaert/flavonol share the same LASSO/RFE/MI mask → 9 unique selections. p: the overlap values are
  permuted across the 9 selections (rows sharing a mask move together), and all 9! = 362 880 assignments
  are enumerated; two-sided p = P(|ρ*| ≥ |ρ_obs|).

## 5. Compute budget (R3-39)

Model fits per selection: GA = unique candidates evaluated × 5 inner folds (the candidate cache does not
alter the RNG stream); RFE = elimination rounds + 1 Ridge fit; LASSO = α grid (100) × 5 inner folds + 1;
MI = 0 (per-feature k-NN MI estimate). `compute_budget.csv`.

## 6. Table S01 columns (feature audit)

Source: `E0_audit/feature_audit.csv` (generated by `src/m07_revision/audit.py`).

- `col_std`: standard deviation (ddof=0) of the feature over all samples.
- n_unique_values counts distinct feature values; n_finite_observations reports the sample-level finite count
  (204 for all 520 grapevine features).
- `r_chlorophyll`, `r_flavonol`, `r_nbi`: Pearson correlation of the feature with the Dualex target
  (empty if `col_std` = 0).
- `transform` = `reflectance-ratio transform`: the five reflectance-ratio layers of block A
  (logR460_R410, R670_R460, R800_R460, FRI_x_NDRE, logNBI; see `E0_audit/flavonol_transforms.md`).
"""


REPORTING_CHECKLIST = [
    ("Data", "Datasets, targets, n, p; raw-data DOIs", "README.md (Datasets); E0_audit/groups_summary.csv"),
    ("Data", "Grouping unit (grapevine leaf, potato Field) and group sizes", "E0_audit/groups_summary.csv; fold_composition.csv"),
    ("Data", "Target distributions by variety / date / plot", "E0_audit/target_distributions.csv"),
    ("Features", "Feature construction, sample-local preprocessing, equations (SG per band, REP-type index)",
     "E0_audit/feature_audit.csv; savgol_delta.md; continuum_removal.md; flavonol_transforms.md"),
    ("Features", "No target-derived feature (|r| = 1 / Dualex names) — automated test", "tests/test_no_leakage.py"),
    ("Splits", "Outer split scheme, number of folds, repeats, random_state per repeat", "config/revision.yaml; E1_repeated/splits_signatures.csv"),
    ("Splits", "Seeds: derivation and where each seed is used", "E0_audit/seed_propagation.md; src/core/splits.py derive_seed"),
    ("Selection", "Feature selection inside the outer loop (nested) vs on all data (apparent); leakage test",
     "metrics_definitions.md §1; tests/test_no_leakage.py"),
    ("Selection", "Selector settings (k, LASSO inner CV, RFE α/step, MI neighbours, GA pop/ngen/fitness)", "config/revision.yaml common.selectors"),
    ("Models", "PLS components rule, LightGBM settings; inner-CV PLS tuning sensitivity", "config/revision.yaml; sensitivity_table.csv (pls_components)"),
    ("Metrics", "Fold-mean vs pooled aggregation; sign convention (positive = apparent more optimistic)", "metrics_definitions.md §2"),
    ("Uncertainty", "Between-repeat spread and paired group-bootstrap CI (conditional; assumption stated)", "primary_table.csv; metrics_definitions.md §2"),
    ("Uncertainty", "Small-fold caveat for potato (pooled R² co-primary)", "primary_table.csv note"),
    ("Testing", "Permutation test: statistic, B, scheme, p formula, minimum attainable p", "permutation_table.csv"),
    ("Testing", "Multiple comparisons: family definition and BH correction", "seed_wilcoxon.csv; metrics_definitions.md §4"),
    ("Simulation", "Generators, n/p/k, repeats, known-truth targets", "simulation_*_table.csv; config/revision.yaml E4_simulation"),
    ("Prespecification", "Grid frozen before runs; analysis roles (primary vs exploratory)", "config/revision.yaml (meta.frozen_before_runs); analysis_roles.csv"),
    ("Prespecification", "Deviations from the prespecified plan, with basis and approval", "outputs_rev1/DEVIATIONS.md"),
    ("Computation", "Compute budget per selector", "compute_budget.csv"),
    ("Reproducibility", "run_config.json per experiment (commit SHA, command, versions, seeds)", "outputs_rev1/*/run_config.json; outputs_rev1/env.txt"),
    ("Reproducibility", "One-command verification of reported numbers", "reproduce_revision.py; expected_values.json"),
    ("Reproducibility", "Sample-level fold records (sample id, group id, y, ŷ, mask) available on request / regenerable",
     "outputs_rev1/*/samples/ (git-ignored; regenerated by the experiment modules)"),
    ("Language", "Terminology: apparent / nested / apparent–nested difference (no biased/honest/unbiased)", "README.md; all rev1 outputs"),
]


def reporting_checklist() -> str:
    lines = ["# reporting_checklist.md — revision-1 (R3-86)", "",
             "Generated by `src/m07_revision/e7_stats_export.py`. For each item, its location in the outputs is given.", "",
             "| # | Area | Item | Where |", "|---|---|---|---|"]
    for i, (area, item, where) in enumerate(REPORTING_CHECKLIST, 1):
        lines.append(f"| {i} | {area} | {item} | {where} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 7) Ek materyal dışa aktarımı
# ---------------------------------------------------------------------------
#: (Tablo no, dosya adı, başlık, tablo anahtarı ya da kaynak CSV, analiz id)
SUPPLEMENTARY = [
    ("S01", "feature_audit", "Feature audit: block, transform, source wavelengths, equation", "E0_audit/feature_audit.csv", "A23"),
    ("S02", "groups_and_folds", "Groups and original-split fold composition", "E0_audit/fold_composition.csv", "A23"),
    ("S03", "primary_full", "Primary table, all quantities (E1 + E6-G1)", "primary_table", "A01–A06"),
    ("S04", "group_bootstrap_by_repeat", "Paired group bootstrap CI per repeat (E1 + GA)", "bootstrap_by_repeat", "A03"),
    ("S05", "selector_family", "Apparent–nested difference by selector family", "primary_by_family", "A05"),
    ("S06", "potato_logo", "Potato leave-one-field-out (pooled R²)", "E1_repeated/logo_potato.csv", "A19"),
    ("S07", "lasso_truncation", "LASSO k-cap truncation and convergence warnings", "E1_repeated/lasso_truncation.csv", "A01"),
    ("S08", "pls_components", "PLS component distribution by arm", "E1_repeated/components.csv", "A09"),
    ("S09", "mechanism", "Size-matched decomposition of the apparent–nested difference", "mechanism_table", "A07"),
    ("S10", "lasso_2x2", "LASSO α-source × support-source", "lasso_2x2_table", "A08"),
    ("S11", "stability", "Prediction, mask and coefficient stability", "stability_table", "A09"),
    ("S12", "negative_differences", "Fold-level negative differences", "negative_differences", "A10"),
    ("S13", "sensitivity", "One-at-a-time sensitivity grid", "sensitivity_table", "A11"),
    ("S14", "simulation_null", "Simulation S-null", "simulation_null_table", "A12"),
    ("S15", "simulation_grid", "Simulation n × p × k grid and analytic baselines", "simulation_grid_table", "A13"),
    ("S16", "simulation_signal", "Simulation S-signal (known truth)", "simulation_signal_table", "A14"),
    ("S17", "permutation", "Permutation tests", "permutation_table", "A15"),
    ("S18", "ga_g0_reproduction", "GA G0 reproduction of the published seed-0 run", "ga_g0_reproduction", "A16"),
    ("S19", "ga_variants", "GA variants G0–G3", "ga_variant_table", "A17"),
    ("S20", "ga_adaptive_overfitting", "GA inner fitness minus outer R²", "ga_adaptive_overfitting", "A18"),
    ("S21", "seed_wilcoxon", "Seed-based Wilcoxon on the 12 published seeds (unadjusted + BH)", "seed_wilcoxon", "A20"),
    ("S22", "spearman_overlap", "Spearman: mask overlap vs apparent–nested difference", "spearman_overlap", "A21"),
    ("S23", "legacy_overlap_reconstruction", "Reconstruction of the published overlap formula", "legacy_overlap_reconstruction", "A21"),
    ("S24", "compute_budget", "Model fits per selection", "compute_budget", "A22"),
    ("S25", "analysis_roles", "Analysis roles (primary vs exploratory)", "analysis_roles", "—"),
]


def export_supplementary(out: Path, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sup = out / "supplementary"
    if sup.exists():
        shutil.rmtree(sup)
    sup.mkdir(parents=True)
    idx = []
    for tid, slug, title, src, aid in SUPPLEMENTARY:
        if src in tables:
            df = tables[src]
            origin = f"{_rel(out)}/{src}.csv"
        else:
            p = REV1 / src
            if not p.exists():
                continue
            df = pd.read_csv(p)
            origin = _rel(p)
        name = f"Table{tid}_{slug}.csv"
        df.to_csv(sup / name, index=False)
        idx.append({"table": f"Table {tid}", "file": name, "title": title, "n_rows": len(df),
                    "n_columns": df.shape[1], "source": origin, "analysis_id": aid})
    ix = pd.DataFrame(idx)
    ix.to_csv(sup / "supplementary_index.csv", index=False)
    md = ["# Supplementary tables — revision-1", "",
          "Generated by `python -m src.m07_revision.e7_stats_export`. Column definitions: "
          "`../metrics_definitions.md`; roles: `../analysis_roles.csv`.", "",
          "| Table | File | Title | Rows | Source |", "|---|---|---|---|---|"]
    md += [f"| {r.table} | `{r.file}` | {r.title} | {r.n_rows} | `{r.source}` |" for r in ix.itertuples()]
    (sup / "INDEX.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return ix


# ---------------------------------------------------------------------------
# 8) Beklenen değerler (R3-81) — reproduce_revision.py doğrular
# ---------------------------------------------------------------------------
def _ev(id_, desc, file, where, column, value, tol, source):
    return {"id": id_, "description": desc, "file": file, "where": where, "column": column,
            # Tam hassasiyet (1e-9 toleranslı p kontrolleri yuvarlamaya dayanmasın)
            "expected": None if value is None or not np.isfinite(value) else float(value),
            "tolerance": tol, "source": source}


def build_expected(out: Path) -> dict:
    """Makale (outputs3) başlık sayıları + revizyon birincil/ikincil sayılar.

    ``source="manuscript"`` satırları makalede yazan değerlerdir (elle girilmiş, 3 ondalık
    → tolerans 5e-4). ``source="revision-1 run"`` satırları 2026-10-31/11-01 koşularından
    DONDURULUR (``--freeze-expected``); revize metin bu sayıları kullanacak. Yeniden üretim
    (temiz ortam) bu değerleri tolerans içinde vermelidir.
    """
    checks = []
    f3 = "outputs3/tables_foldmean/selection_bias_multiseed.csv"
    w = {"dataset": "ryckewaert", "target": "flavonol", "model": "pls", "selector": "ga"}
    checks += [
        _ev("M1", "headline apparent R² (fold-mean, 12-seed mean)", f3, w, "apparent_mean", 0.615, 5e-4, "manuscript"),
        _ev("M2", "headline nested R² (fold-mean, 12-seed mean)", f3, w, "nested_mean", 0.269, 5e-4, "manuscript"),
        _ev("M3", "headline apparent–nested difference", f3, w, "bias_mean", 0.346, 5e-4, "manuscript"),
        _ev("M4", "headline nested SD over seeds", f3, w, "nested_std", 0.026, 5e-4, "manuscript"),
        _ev("M5", "pooled apparent R² (mixed reference)", "outputs3/tables/selection_bias_multiseed.csv", w,
            "apparent_mean", 0.619, 5e-4, "manuscript"),
        _ev("M6", "fully pooled difference (Section 4.6)", "outputs3/tables_pooled/selection_bias_pooled.csv", w,
            "bias_pooled_mean", 0.341, 5e-4, "manuscript"),
    ]
    rel = _rel(out)
    prim = pd.read_csv(out / "primary_table.csv")
    for r in prim.itertuples():
        wh = {"dataset": r.dataset, "target": r.target, "model": r.model, "selector": r.selector}
        tag = f"{r.dataset}/{r.target}/{r.model}/{r.selector}"
        for col in ("apparent_r2_foldmean_mean", "nested_r2_foldmean_mean", "delta_r2_foldmean_median",
                    "delta_r2_pooled_median", "boot_delta_r2_pooled_lo_median", "boot_delta_r2_pooled_hi_median"):
            checks.append(_ev(f"P:{tag}:{col}", f"primary {tag} {col}", f"{rel}/primary_table.csv", wh, col,
                              getattr(r, col), 5e-4, "revision-1 run"))
    g0 = pd.read_csv(REV1 / "E6_ga" / "g0_reproduction.csv")
    for r in g0.itertuples():
        checks.append(_ev(f"G0:{r.quantity}", f"GA G0 seed 0 {r.quantity} (vs outputs3 seed 0)",
                          "outputs_rev1/E6_ga/g0_reproduction.csv", {"quantity": r.quantity}, "rev1_G0",
                          r.outputs3_seed0, 5e-4, "outputs3 seed 0"))
    perm = pd.read_csv(out / "permutation_table.csv")
    for r in perm.itertuples():
        wh = {"dataset": r.dataset, "target": r.target, "selector": r.selector, "scheme": r.scheme}
        checks.append(_ev(f"E5:{r.dataset}/{r.target}/{r.selector}/{r.scheme}:p_nested",
                          "permutation p (nested)", f"{rel}/permutation_table.csv", wh, "p_nested",
                          r.p_nested, 1e-9, "revision-1 run"))
    nul = pd.read_csv(out / "simulation_null_table.csv")
    for r in nul.itertuples():
        wh = {"generator": r.generator, "selector": r.selector}
        checks.append(_ev(f"E4:{r.generator}/{r.selector}:delta", "S-null mean Δ fold-mean",
                          f"{rel}/simulation_null_table.csv", wh, "delta_foldmean_mean", r.delta_foldmean_mean,
                          5e-4, "revision-1 run"))
    sw = pd.read_csv(out / "seed_wilcoxon.csv")
    sw = sw[(sw.aggregation == "foldmean") & sw.p_bh.notna()]
    for r in sw.itertuples():
        wh = {"aggregation": "foldmean", "dataset": r.dataset, "target": r.target, "model": r.model,
              "selector": r.selector}
        checks.append(_ev(f"W:{r.dataset}/{r.target}/{r.model}/{r.selector}:p_bh", "seed-Wilcoxon BH p",
                          f"{rel}/seed_wilcoxon.csv", wh, "p_bh", r.p_bh, 1e-9, "outputs3 stats_summary"))
    return {"generated": datetime.now().isoformat(timespec="seconds"),
            "generator": "python -m src.m07_revision.e7_stats_export --freeze-expected",
            "note": ("manuscript rows are typed from the paper (3 decimals); revision-1 rows are frozen "
                     "from the 2026-10-31/11-01 runs and must match the revised text"),
            "checks": checks}


def verify_expected(path: Path = EXPECTED_PATH, root: Path = ROOT) -> pd.DataFrame:
    """expected_values.json'daki her kontrolü diskteki CSV'ye karşı doğrula."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    cache: dict[str, pd.DataFrame] = {}
    for c in spec["checks"]:
        f = root / c["file"]
        try:
            if c["file"] not in cache:
                cache[c["file"]] = pd.read_csv(f, comment="#")
            df = cache[c["file"]]
            m = np.ones(len(df), bool)
            for k, v in c["where"].items():
                m &= df[k].astype(str).to_numpy() == str(v)
            sel = df.loc[m, c["column"]]
            if len(sel) != 1:
                raise ValueError(f"{len(sel)} satır eşleşti")
            got = float(sel.iloc[0])
            exp = c["expected"]
            ok = (np.isnan(got) if exp is None else abs(got - exp) <= c["tolerance"] + 1e-12)
            rows.append({**{k: c[k] for k in ("id", "source", "file", "column")}, "expected": exp,
                         "got": got, "tolerance": c["tolerance"], "ok": bool(ok), "error": ""})
        except Exception as e:                                     # eksik dosya / satır → BAŞARISIZ
            rows.append({**{k: c[k] for k in ("id", "source", "file", "column")}, "expected": c["expected"],
                         "got": np.nan, "tolerance": c["tolerance"], "ok": False,
                         "error": f"{type(e).__name__}: {e}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def run(mode: str, log) -> dict:
    cfg = C.load_config()
    dirs = C.experiment_dirs(EXP, mode)
    out = dirs["out"]
    out.mkdir(parents=True, exist_ok=True)
    quick = mode == "quick"
    git = C.git_state()
    t0 = time.time()
    tables: dict[str, pd.DataFrame] = {}

    log.info("E7: birincil tablo (E1 + E6-G1; GA bootstrap B=%s)", 200 if quick else "E1 B")
    prim, disp, gb = primary_table(cfg, quick)
    tables["primary_table"] = prim
    tables["primary_table_display"] = disp
    ci = pd.read_csv(REV1 / "E1_repeated" / "delta_ci.csv")
    ci = ci[ci["repeat"].astype(str) != "median_over_repeats"].assign(source="E1_repeated")
    tables["bootstrap_by_repeat"] = pd.concat(
        [ci, gb.assign(source="E6_ga.G1 (E7)", note=ci["note"].iloc[0])], ignore_index=True)
    tables["primary_by_family"] = family_table(cfg, prim)

    log.info("E7: ikincil tablolar (E2–E6) + hesaplama bütçesi")
    tables.update(mechanism_tables(cfg))
    tables["sensitivity_table"] = sensitivity_table(cfg)
    tables.update(simulation_tables())
    tables["permutation_table"] = permutation_table(cfg)
    tables.update(ga_tables(cfg))
    tables["compute_budget"] = compute_budget(cfg)

    log.info("E7: eski 12-seed tablolar (seed-Wilcoxon + BH, Spearman, örtüşme yeniden kurulumu)")
    tables["seed_wilcoxon"] = seed_wilcoxon(cfg)
    tables["spearman_overlap"] = spearman_overlap(cfg, quick)
    raw, comp = legacy_overlap_reconstruction(cfg, quick, dirs["masks"], log=log.info)
    tables["legacy_overlap_reconstruction"] = comp
    verdict = overlap_verdict(comp)
    tables["analysis_roles"] = analysis_roles()

    for name, df in tables.items():
        if name in ("bootstrap_by_repeat", "negative_differences"):
            continue                                   # yalnız ek materyalde (büyük / ham)
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "metrics_definitions.md").write_text(metrics_definitions(verdict, comp), encoding="utf-8")
    (out / "reporting_checklist.md").write_text(reporting_checklist(), encoding="utf-8")
    ix = export_supplementary(out, tables)
    (out / "overlap_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    C.write_run_config(out, {"experiment": EXP, "mode": mode,
                             "inputs": ["outputs3/tables_foldmean", "outputs_rev1/E0_audit … E6_ga"],
                             "ga_bootstrap": {"B": 200 if quick else int(prim.boot_B.dropna().max()),
                                              "seed": "derive_seed(base_seed, repeat, -1, 'bootstrap')"},
                             "seconds": round(time.time() - t0, 1)}, git)
    log.info("E7 bitti: %d tablo, %d ek tablo → %s [%.1f s]", len(tables), len(ix), _rel(out),
             time.time() - t0)
    return {"out": out, "tables": tables, "verdict": verdict, "supplementary": ix}


def check(mode: str = "full") -> int:
    """Çıktılar var mı + iç tutarlılık / yeniden üretim kontrolleri."""
    out = C.experiment_dirs(EXP, mode)["out"]
    need = ["primary_table.csv", "primary_table_display.csv", "primary_by_family.csv", "seed_wilcoxon.csv",
            "spearman_overlap.csv", "legacy_overlap_reconstruction.csv", "permutation_table.csv",
            "analysis_roles.csv", "metrics_definitions.md", "reporting_checklist.md",
            "supplementary/INDEX.md", "supplementary/supplementary_index.csv", "run_config.json"]
    ok = True
    miss = [n for n in need if not (out / n).exists()]
    print(f"E7_stats_export: {len(need) - len(miss)}/{len(need)} çıktı" + (f" | EKSİK: {miss}" if miss else ""))
    ok &= not miss
    if miss:
        print("SONUÇ: EKSİK")
        return 1
    sw = pd.read_csv(out / "seed_wilcoxon.csv")
    rep = sw[sw.aggregation == "foldmean"].reproduces_outputs3.astype(str)
    print(f"seed-Wilcoxon ↔ outputs3 stats_summary (p, BH p): {int((rep == 'True').sum())}/{len(rep)} satır aynı; "
          f"BH ailesi = {int(sw[sw.aggregation == 'foldmean'].bh_family_size.max())} test")
    ok &= bool((rep == "True").all())
    v = json.loads((out / "overlap_verdict.json").read_text(encoding="utf-8"))
    for col, d in v.items():
        print(f"outputs3 {col}: eşleşen formül(ler) {d['matching'] or 'YOK'} ({d['n_selections']} seçim)")
    ok &= all(d["matching"] for d in v.values()) or mode == "quick"
    prim = pd.read_csv(out / "primary_table.csv")
    print(f"birincil tablo: {len(prim)} satır | GA bootstrap CI dolu: "
          f"{int(prim[prim.selector == 'ga'].boot_delta_r2_pooled_lo_median.notna().sum())}/"
          f"{int((prim.selector == 'ga').sum())}")
    grape = prim[prim.dataset == "ryckewaert"].fold_partitioning_contrast_r2_foldmean_median.abs().max()
    print(f"grapevine P1−P2 max |medyan| = {grape:.2e} (tanım gereği 0)")
    ok &= bool(grape < 1e-12)
    show = prim[prim.model == "pls"][["dataset", "target", "selector", "n_repeats", "apparent_r2_foldmean_mean",
                                      "nested_r2_foldmean_mean", "delta_r2_foldmean_median",
                                      "delta_r2_pooled_median", "boot_delta_r2_pooled_lo_median",
                                      "boot_delta_r2_pooled_hi_median"]]
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(show.round(3).to_string(index=False))
    sp = pd.read_csv(out / "spearman_overlap.csv")
    print(sp[["source", "version", "x", "n_rows", "n_unique_selections", "rho", "p_perm_two_sided",
              "p_asymptotic"]].round(4).to_string(index=False))
    if EXPECTED_PATH.exists() and mode == "full":
        r = verify_expected()
        print(f"expected_values.json: {int(r.ok.sum())}/{len(r)} kontrol tuttu")
        ok &= bool(r.ok.all())
    print("SONUÇ:", "TAMAM" if ok else "BAŞARISIZ")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="E7 — istatistik tabloları + ek materyal")
    mx = ap.add_mutually_exclusive_group()
    mx.add_argument("--quick", action="store_true", help="duman testi → outputs_rev1/_quick/ (bootstrap B=200)")
    mx.add_argument("--check", action="store_true", help="çıktılar + yeniden üretim kontrolleri")
    mx.add_argument("--freeze-expected", action="store_true",
                    help="expected_values.json'u mevcut tam çıktılardan yaz (kökte)")
    args = ap.parse_args()
    if args.check:
        sys.exit(check("full"))
    if args.freeze_expected:
        out = C.experiment_dirs(EXP, "full")["out"]
        spec = build_expected(out)
        EXPECTED_PATH.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        r = verify_expected()
        print(f"expected_values.json yazıldı: {len(spec['checks'])} kontrol; doğrulama {int(r.ok.sum())}/{len(r)}")
        if not r.ok.all():
            print(r[~r.ok].to_string(index=False))
        sys.exit(0 if r.ok.all() else 1)
    mode = "quick" if args.quick else "full"
    dirs = C.experiment_dirs(EXP, mode)
    log = C.setup_logging(dirs["log"])
    run(mode, log)
    if mode == "quick":
        sys.exit(check("quick"))


if __name__ == "__main__":
    main()
