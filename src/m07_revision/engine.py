"""Eşli kol değerlendirme motoru — Revizyon 1 çekirdeği (plan §2.3).

Tek fonksiyon :func:`evaluate_arms`. Aynı dış bölme (``splits``) TÜM kollarda
paylaşılır (EŞLİ tasarım); kollar yalnız *seçimin hangi veriyi gördüğü* ve
*bölme şeması* bakımından farklıdır.

Kollar
------
==================== ======================================= ===========================
Kol                  Seçim verisi                            Model fit / test
==================== ======================================= ===========================
``P1_random``        tüm veri, tek maske (P2 ile aynı)       rastgele KFold (gruplar yok)
``P2_apparent``      tüm veri, tek maske                     aynı grup bölmesi
``P3_nested``        yalnız dış-train                        dış-train fit, dış-test tahmin
``size_matched_leaky`` tüm veriden rastgele, dış-train ile    aynı fold
                     AYNI sayıda grup (test grupları         (R3-46)
                     dahil olabilir)
``no_selection``     — (tüm öznitelikler)                    aynı fold
``random_k``         rastgele k öznitelik (y'ye bakmaz)      aynı fold (R3-67)
==================== ======================================= ===========================

Estimand
--------
Dış bölme :math:`\\{T_1, …, T_K\\}`, :math:`D` tüm veri, :math:`S(\\cdot)` seçici,
:math:`\\hat f_{S, A}` seçilen öznitelikler ve :math:`A` eğitim verisiyle fit edilmiş model.

* nested (P3), fold-mean (birincil)::

      θ̂_nested = (1/K) Σ_k R²( y_{T_k}, f̂_{S(D \\ T_k), D \\ T_k}(X_{T_k}) )

* apparent (P2), fold-mean: aynı ifade, ancak seçim tüm veride: :math:`S(D)` →
  ``f̂_{S(D), D \\ T_k}``.
* pooled sürüm: tüm fold-dışı (OOF) tahminler birleştirilip tek R² (her örnek
  tam bir kez tahmin edilir).
* apparent–nested difference: ``Δ = θ̂_apparent − θ̂_nested`` (R² için);
  RMSE/MAE'de ``Δ = RMSE_nested − RMSE_apparent``. Her ikisinde de
  **pozitif = apparent daha iyimser** (R3-58).
* selection-scope contrast = P2 − P3; fold-partitioning contrast = P1 − P2.

Seed politikası
---------------
``seed_mode="derived"`` (Revizyon 1): seçici seed'i
``derive_seed(base_seed, repeat, fold, selector_name)``; tüm-veri seçimi fold=-1.
``seed_mode="fixed"`` (geriye uyum): her seçici çağrısı ``base_seed`` alır — makaledeki
``nested_evaluation`` davranışı. Kontrol kollarının (random_k, size_matched) RNG'si
her iki modda da türetilir.

Çıktılar (:class:`ArmsResult`)
-----------------------------
``summary``     kol başına metrikler (fold-mean + pooled; R², RMSE, MAE)
``contrasts``   farklar (dict)
``folds``       (kol, fold) başına satır — örnek kimliği İÇERMEZ (özet tablo)
``predictions`` (kol, fold, test örneği) başına satır — örnek düzeyi → ``samples/``
:func:`folds_long` ikisini plan §2.3'teki tek tabloya (``folds.csv.gz``) birleştirir.

Çalıştırma (duman testi, sentetik veri)::

    python -m src.m07_revision.engine
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.core.splits import derive_seed, make_outer_splits, splits_signature

ARMS: tuple[str, ...] = ("P1_random", "P2_apparent", "P3_nested",
                         "size_matched_leaky", "no_selection", "random_k")
DEFAULT_ARMS: tuple[str, ...] = ("P1_random", "P2_apparent", "P3_nested",
                                 "no_selection", "random_k")


# ---------------------------------------------------------------------------
# Seçici çözümü (Revizyon deneyleri için; m06 kaydından FARKLI)
# ---------------------------------------------------------------------------
def resolve_selector(name: str) -> Callable:
    """Config'teki seçici adı → fonksiyon.

    DİKKAT: buradaki ``"lasso"`` YENİ ``select_lasso``'dur (``inner`` sel_kw ile
    verilir; varsayılan "group"). m06 hattının ``SELECTORS["lasso"]`` kaydı ise
    legacy'dir. Geriye uyum için ``sel_kw={"inner": "legacy"}`` verin.
    """
    from src.m04_features import selectors as s

    table = {"rfe": s.select_rfe, "lasso": s.select_lasso, "mutual_info": s.select_mutual_info,
             "ga": s.select_ga, "random_k": s.select_random, "none": s.select_none}
    if name not in table:
        raise ValueError(f"Bilinmeyen seçici {name!r}; seçenekler {list(table)}")
    return table[name]


# ---------------------------------------------------------------------------
# Model + metrik yardımcıları
# ---------------------------------------------------------------------------
def build_model(model: str, n_sel: int, n_train: int, model_kw: dict | None = None):
    """Regresör üret. ``model_kw`` boşsa makaledeki ``_build_regressor`` aynen kullanılır.

    PLS: ``n_components = min(10, n_sel, n_train − 1)`` (makale). ``model_kw``
    ``{"n_components": c}`` verirse ``min(c, n_sel, n_train − 1)``.
    """
    from src.m04_features.ga_feature_selection import _build_regressor

    model_kw = dict(model_kw or {})
    if model == "pls_tuned":
        from src.m07_revision.models import PLSInnerCV
        return PLSInnerCV(**model_kw)
    if model == "pls" and "n_components" in model_kw:
        from sklearn.cross_decomposition import PLSRegression
        c = max(1, min(int(model_kw["n_components"]), n_sel, n_train - 1))
        return PLSRegression(n_components=c)
    return _build_regressor(model, n_sel, n_train)


def _fit_predict(model: str, model_kw: dict | None, X: np.ndarray, y: np.ndarray,
                 tr: np.ndarray, te: np.ndarray, mask: np.ndarray,
                 groups_tr: np.ndarray | None = None,
                 ) -> tuple[np.ndarray, int | None, np.ndarray | None]:
    """Seçilen sütunlarla dış-train'e fit, dış-test'i tahmin et.

    Dönüş: (tahmin, PLS bileşen sayısı | None, PLS regresyon katsayıları | None).
    Katsayılar seçili sütunlar sırasındadır (orijinal X ölçeğinde; R3-77 için).
    ``groups_tr`` yalnız ``model="pls_tuned"`` (iç GroupKFold ile bileşen ayarı,
    R3-50) tarafından kullanılır; diğer modellerde yok sayılır.
    """
    reg = build_model(model, int(mask.sum()), len(tr), model_kw)
    if model == "pls_tuned":
        reg.fit(X[np.ix_(tr, np.flatnonzero(mask))], y[tr], groups=groups_tr)
    else:
        reg.fit(X[np.ix_(tr, np.flatnonzero(mask))], y[tr])
    pred = np.asarray(reg.predict(X[np.ix_(te, np.flatnonzero(mask))])).ravel()
    n_comp = getattr(reg, "n_components", None)
    coef = getattr(reg, "coef_", None) if model in ("pls", "pls_tuned") else None
    return (pred, (int(n_comp) if n_comp is not None else None),
            None if coef is None else np.asarray(coef, dtype=float).ravel())


# ---------------------------------------------------------------------------
# Maske önbelleği
# ---------------------------------------------------------------------------
def selector_uses_seed(selector_name: str, sel_kw: dict, has_groups: bool) -> bool:
    """Seçicinin maskesi seed'e bağlı mı? (Önbellek anahtarına seed girsin mi?)

    E0 denetimi: RFE deterministik; LassoCV ``random_state``'i yalnız
    ``selection='random'``'da kullanır → LASSO'da seed yalnız iç CV
    ``KFold(shuffle=True, random_state=seed)`` (inner="group" + groups YOK) iken
    etkilidir. MI, random_k, GA seed'e bağlıdır. Bilinmeyen seçici → güvenli taraf (True).
    """
    if selector_name == "rfe":
        return False
    if selector_name == "lasso":
        return sel_kw.get("inner", "group") == "group" and not has_groups
    return True


class MaskCache:
    """Disk tabanlı seçim önbelleği (``<root>/<xx>/<sha1>.json``).

    Anahtar İÇERİK tabanlıdır: (bağlam = dataset/target, seçici adı, sel_kw, k,
    seçicinin gördüğü satırların özeti, seed [yalnız maskeyi etkiliyorsa], model
    [yalnız GA]). Böylece aynı maske PLS ve LightGBM'de, E1 ve E2'de ve tüm-veri
    seçiminde tekrarlar arasında BİR KEZ hesaplanır; bölme tanımı değişirse
    satırlar değişeceği için bayat maske kullanılamaz. Okunabilirlik için dosyaya
    (repeat, fold, arm) meta-verisi de yazılır (anahtarın parçası değildir).
    Yazım atomiktir (tmp → replace); paralel süreçler aynı anahtarı yazarsa içerik
    aynıdır.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(context: dict, selector_name: str, sel_kw: dict, k, rows: np.ndarray,
                 seed: int | None, model: str | None) -> dict:
        import hashlib
        rows_hash = hashlib.sha1(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()
        return {**context, "selector": selector_name,
                "sel_kw": {kk: sel_kw[kk] for kk in sorted(sel_kw)}, "k": k,
                "rows_sha1": rows_hash, "n_rows": int(len(rows)), "seed": seed, "model": model}

    def _path(self, key: dict) -> Path:
        import hashlib
        h = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()
        return self.root / h[:2] / f"{h}.json"

    def get(self, key: dict):
        p = self._path(key)
        if not p.exists():
            self.misses += 1
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        self.hits += 1
        return np.array([c == "1" for c in d["mask"]], dtype=bool), d["info"]

    def put(self, key: dict, mask: np.ndarray, info: dict, meta: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{np.random.randint(1 << 30)}.tmp")
        tmp.write_text(json.dumps({"key": key, "meta": meta, "mask": _bits(mask), "info": info},
                                  default=str), encoding="utf-8")
        tmp.replace(p)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"r2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p))}


def _bits(mask: np.ndarray) -> str:
    return "".join("1" if b else "0" for b in mask)


# ---------------------------------------------------------------------------
# Sonuç tipi
# ---------------------------------------------------------------------------
@dataclass
class ArmsResult:
    """:func:`evaluate_arms` çıktısı."""

    summary: pd.DataFrame
    contrasts: dict
    folds: pd.DataFrame
    predictions: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def arm(self, name: str) -> pd.Series:
        return self.summary.set_index("arm").loc[name]


# ---------------------------------------------------------------------------
# Çekirdek
# ---------------------------------------------------------------------------
def evaluate_arms(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    model: str,
    selector: Callable,
    arms: Sequence[str] = DEFAULT_ARMS,
    sel_kw: dict | None = None,
    model_kw: dict | None = None,
    *,
    k: int | None = 20,
    repeat: int = 0,
    base_seed: int = 0,
    seed_mode: str = "derived",
    selector_name: str = "selector",
    random_splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    sample_ids: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
    mask_cache: MaskCache | None = None,
    cache_context: dict | None = None,
    record_coef: bool = False,
) -> ArmsResult:
    """Kolları AYNI dış bölmede değerlendir (eşli tasarım). Estimand: modül docstring'i.

    Parameters
    ----------
    X, y : ndarray
        Öznitelik matrisi ve hedef. y sonlu olmalı (NaN'ler önceden atılmış).
    groups : ndarray | None
        Seçiciye iletilen grup etiketleri (LASSO iç GroupKFold, GA iç CV) ve
        ``size_matched_leaky`` alt kümelemesi için.
    splits : list[(train_idx, test_idx)]
        Dış bölme (:func:`src.core.splits.make_outer_splits`). P2/P3/kontrol
        kollarının hepsi bunu kullanır.
    model : str
        ``"pls"`` | ``"lightgbm"`` | … (``ga_feature_selection._build_regressor``).
    selector : callable
        Ortak imzalı seçici (``return_info=True`` destekli).
    arms : sıra
        :data:`ARMS` alt kümesi.
    sel_kw, model_kw : dict
        Seçiciye / modele ek argümanlar (ör. ``{"inner": "group"}``, GA pop/ngen).
    k : int | None
        Sabit-k seçicilerin k'sı. ``random_k`` kolu için de kullanılır; None ise
        aynı fold'daki P3 maskesinin boyutu (GA gibi değişken-k seçicilerde).
    repeat, base_seed, seed_mode
        Seed politikası (modül docstring'i).
    random_splits : list | None
        P1 bölmesi. None → ``make_outer_splits(n, None, "random", K, repeat)``.
    sample_ids, group_ids : ndarray | None
        Tahmin kayıtlarına yazılacak kimlikler (None → indeks / groups).
    mask_cache, cache_context
        :class:`MaskCache` ve bağlam (ör. ``{"dataset": ..., "target": ...}``).
        Verilirse her seçim önce önbellekte aranır.
    record_coef : bool
        PLS katsayılarını fold kaydına (``coef`` sütunu, JSON liste) yaz (R3-77).
    """
    t0 = time.time()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("y sonlu olmalı — NaN satırları önceden atın")
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"Bilinmeyen kol(lar): {sorted(unknown)}")
    if seed_mode not in ("derived", "fixed"):
        raise ValueError("seed_mode 'derived' ya da 'fixed' olmalı")
    n = len(y)
    sel_kw = dict(sel_kw or {})
    g = None if groups is None else np.asarray(groups)
    sample_ids = np.arange(n) if sample_ids is None else np.asarray(sample_ids)
    group_ids = (g if g is not None else np.arange(n)) if group_ids is None else np.asarray(group_ids)
    K = len(splits)
    sig = splits_signature(splits)

    def sel_seed(fold: int) -> int:
        return int(base_seed) if seed_mode == "fixed" else derive_seed(base_seed, repeat, fold, selector_name)

    uses_seed = selector_uses_seed(selector_name, sel_kw, g is not None)

    def run_selector(rows: np.ndarray, fold: int, arm: str) -> tuple[np.ndarray, dict, int]:
        rows = np.asarray(rows)
        s = sel_seed(fold)
        key = None
        if mask_cache is not None:
            key = MaskCache.make_key(dict(cache_context or {}), selector_name, sel_kw, k, rows,
                                     s if uses_seed else None,
                                     model if selector_name == "ga" else None)
            hit = mask_cache.get(key)
            if hit is not None:
                mask, info = hit
                return mask, {**info, "cache": "hit"}, s
        gs = None if g is None else g[rows]
        t_sel = time.time()
        out = selector(X[rows], y[rows], k, groups=gs, model=model, seed=s,
                       return_info=True, **sel_kw)
        mask, info = out if isinstance(out, tuple) else (out, {})
        mask, info = np.asarray(mask, dtype=bool), dict(info)
        info["selection_seconds"] = round(time.time() - t_sel, 3)
        if mask_cache is not None:
            mask_cache.put(key, mask, info, {"repeat": repeat, "fold": fold, "arm": arm,
                                             "splits_signature": sig})
        return mask, {**info, "cache": "miss" if mask_cache is not None else "off"}, s

    fold_rows: list[dict] = []
    pred_rows: list[dict] = []

    def record(arm: str, f: int, tr, te, mask, info, s, fp, sel_rows: int, scheme: str):
        pred, n_comp, coef = fp
        m = _metrics(y[te], pred)
        fold_rows.append({
            "repeat": repeat, "fold": f, "arm": arm, "split_scheme": scheme,
            "n_train": len(tr), "n_test": len(te),
            "n_test_groups": int(len(np.unique(group_ids[te]))),
            "n_selection_rows": sel_rows, "n_selected": int(mask.sum()),
            "n_components": n_comp, "selector_seed": s,
            "selector_info": json.dumps(info, ensure_ascii=False, default=str),
            "mask": _bits(mask), **m,
            **({"coef": json.dumps([round(float(c), 10) for c in coef])}
               if record_coef and coef is not None else {}),
        })
        for i, p in zip(te, pred):
            pred_rows.append({"repeat": repeat, "fold": f, "arm": arm, "test_idx": int(i),
                              "sample_id": sample_ids[i], "group_id": group_ids[i],
                              "y_true": float(y[i]), "y_pred": float(p)})

    # --- P1 / P2: tüm veride TEK seçim ------------------------------------
    if "P1_random" in arms or "P2_apparent" in arms:
        all_rows = np.arange(n)
        full_mask, full_info, full_seed = run_selector(all_rows, -1, "all_data")
        if "P2_apparent" in arms:
            for f, (tr, te) in enumerate(splits):
                fp = _fit_predict(model, model_kw, X, y, tr, te, full_mask, None if g is None else g[tr])
                record("P2_apparent", f, tr, te, full_mask, full_info, full_seed, fp, n, "outer")
        if "P1_random" in arms:
            rs = random_splits or make_outer_splits(n, None, "random", K, repeat)
            for f, (tr, te) in enumerate(rs):
                fp = _fit_predict(model, model_kw, X, y, tr, te, full_mask, None if g is None else g[tr])
                record("P1_random", f, tr, te, full_mask, full_info, full_seed, fp, n, "random")

    # --- P3: seçim yalnız dış-train'de ------------------------------------
    p3_nsel: dict[int, int] = {}
    if "P3_nested" in arms:
        for f, (tr, te) in enumerate(splits):
            mask, info, s = run_selector(np.asarray(tr), f, "P3_nested")
            p3_nsel[f] = int(mask.sum())
            fp = _fit_predict(model, model_kw, X, y, tr, te, mask, None if g is None else g[tr])
            record("P3_nested", f, tr, te, mask, info, s, fp, len(tr), "outer")

    # --- size_matched_leaky (R3-46) ----------------------------------------
    if "size_matched_leaky" in arms:
        gg = g if g is not None else np.arange(n)
        uniq = np.unique(gg)
        for f, (tr, te) in enumerate(splits):
            n_g = len(np.unique(gg[tr]))
            rng = np.random.default_rng(derive_seed(base_seed, repeat, f, "size_matched_leaky"))
            chosen = rng.choice(uniq, size=n_g, replace=False)
            rows = np.flatnonzero(np.isin(gg, chosen))
            mask, info, s = run_selector(rows, f, "size_matched_leaky")
            info["n_test_rows_in_selection"] = int(np.isin(rows, te).sum())
            fp = _fit_predict(model, model_kw, X, y, tr, te, mask, None if g is None else g[tr])
            record("size_matched_leaky", f, tr, te, mask, info, s, fp, len(rows), "outer")

    # --- no_selection --------------------------------------------------------
    if "no_selection" in arms:
        mask = np.ones(X.shape[1], dtype=bool)
        for f, (tr, te) in enumerate(splits):
            fp = _fit_predict(model, model_kw, X, y, tr, te, mask, None if g is None else g[tr])
            record("no_selection", f, tr, te, mask, {"selector": "none"}, None, fp, 0, "outer")

    # --- random_k (R3-67) ----------------------------------------------------
    if "random_k" in arms:
        from src.m04_features.selectors import select_random
        for f, (tr, te) in enumerate(splits):
            kk = k if k is not None else p3_nsel.get(f, 20)
            s = derive_seed(base_seed, repeat, f, "random_k")
            mask, info = select_random(X[tr], y[tr], kk, seed=s, return_info=True)
            fp = _fit_predict(model, model_kw, X, y, tr, te, mask, None if g is None else g[tr])
            record("random_k", f, tr, te, mask, info, s, fp, 0, "outer")

    folds = pd.DataFrame(fold_rows)
    preds = pd.DataFrame(pred_rows)
    summary = summarize(folds, preds)
    contrasts = compute_contrasts(summary)
    meta = {"selector": selector_name, "model": model, "k": k, "repeat": repeat,
            "base_seed": base_seed, "seed_mode": seed_mode, "n": n, "p": X.shape[1],
            "n_splits": K, "splits_signature": sig, "arms": list(arms),
            "sel_kw": {kk: v for kk, v in sel_kw.items()}, "model_kw": dict(model_kw or {}),
            "elapsed_s": round(time.time() - t0, 3)}
    return ArmsResult(summary=summary, contrasts=contrasts, folds=folds, predictions=preds, meta=meta)


def summarize(folds: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Kol başına fold-mean (birincil) + pooled OOF metrikleri."""
    rows = []
    for arm, fd in folds.groupby("arm", sort=False):
        pd_ = preds[preds.arm == arm]
        pooled = _metrics(pd_["y_true"].to_numpy(), pd_["y_pred"].to_numpy())
        rows.append({
            "arm": arm, "n_folds": len(fd),
            "r2_foldmean": float(fd["r2"].mean()), "r2_fold_std": float(fd["r2"].std(ddof=0)),
            "rmse_foldmean": float(fd["rmse"].mean()), "mae_foldmean": float(fd["mae"].mean()),
            "r2_pooled": pooled["r2"], "rmse_pooled": pooled["rmse"], "mae_pooled": pooled["mae"],
            "n_selected_mean": float(fd["n_selected"].mean()),
            "n_components_mean": float(pd.to_numeric(fd["n_components"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows)


def compute_contrasts(summary: pd.DataFrame) -> dict:
    """Plan terminolojisiyle farklar. Pozitif = ilk kol daha iyimser."""
    s = summary.set_index("arm")
    out: dict = {}

    def diff(a: str, b: str, tag: str) -> None:
        if a in s.index and b in s.index:
            for agg in ("foldmean", "pooled"):
                out[f"{tag}_r2_{agg}"] = float(s.at[a, f"r2_{agg}"] - s.at[b, f"r2_{agg}"])
                out[f"{tag}_rmse_{agg}"] = float(s.at[b, f"rmse_{agg}"] - s.at[a, f"rmse_{agg}"])
                out[f"{tag}_mae_{agg}"] = float(s.at[b, f"mae_{agg}"] - s.at[a, f"mae_{agg}"])

    diff("P2_apparent", "P3_nested", "apparent_nested_difference")  # = selection-scope contrast
    diff("P1_random", "P2_apparent", "fold_partitioning_contrast")
    diff("P2_apparent", "size_matched_leaky", "sample_size_component")
    diff("size_matched_leaky", "P3_nested", "equal_size_leakage_component")
    diff("P2_apparent", "random_k", "apparent_vs_random_k")
    diff("random_k", "P3_nested", "random_k_vs_nested")
    return out


def folds_long(res: ArmsResult) -> pd.DataFrame:
    """Plan §2.3 fold kaydı: tahmin satırları + fold düzeyi alanlar (örnek düzeyi → samples/)."""
    keep = ["repeat", "fold", "arm", "mask", "n_selected", "n_components", "selector_info",
            "n_train", "n_test", "n_test_groups"]
    return res.predictions.merge(res.folds[keep], on=["repeat", "fold", "arm"], how="left")


def checkpointed(path: Path, fn: Callable[[], object]):
    """Plan §0.5: sonuç diskteyse yükle, değilse hesapla + atomik yaz."""
    path = Path(path)
    if path.exists():
        return pickle.loads(path.read_bytes())
    obj = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(pickle.dumps(obj))
    tmp.replace(path)
    return obj


if __name__ == "__main__":
    # Duman testi (sentetik): tüm kollar çalışmalı; P3 seçimi test satırı görmemeli.
    from src.m04_features.selectors import select_rfe

    rng = np.random.default_rng(0)
    n, p = 80, 30
    g = np.repeat(np.arange(20), 4)
    X = rng.normal(size=(n, p))
    y = X[:, :4].sum(axis=1) + rng.normal(scale=1.0, size=n)
    sp = make_outer_splits(n, g, "group", 5, repeat=1)
    res = evaluate_arms(X, y, g, sp, "pls", select_rfe, ARMS, k=5, repeat=1,
                        base_seed=0, selector_name="rfe")
    print(res.summary[["arm", "r2_foldmean", "r2_pooled", "rmse_foldmean", "n_selected_mean"]]
          .to_string(index=False))
    print({k2: round(v, 3) for k2, v in res.contrasts.items() if k2.endswith("r2_foldmean")})
    assert set(res.summary.arm) == set(ARMS)
    assert len(res.predictions) == n * len(ARMS)
    print("engine.py duman testi OK  |  splits", res.meta["splits_signature"],
          f"| {res.meta['elapsed_s']} s")
