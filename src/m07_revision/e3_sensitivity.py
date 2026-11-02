"""E3 — Önceden belirlenmiş duyarlılık ızgarası (plan §3 E3)
→ R3-15,22,23,32,33,35,36,50,84,85,9.

Her eksen TEK TEK değişir (tam faktöriyel değil); diğer her şey
``config/revision.yaml`` ``common`` varsayılanında. R = 10 (r = 0..9; E1'in ilk 10
bölmesiyle aynı ``random_state``). Kollar: P2 (apparent) ve P3 (nested) — iki kol
AYNI dış bölmeyi paylaşır. Model: PLS (``pls_components`` ekseninde iç CV ile
ayarlı PLS).

Eksenler (config ``E3_sensitivity.axes``)
----------------------------------------
grouping        ryckewaert plot grupları (10 plot; 5 fold → fold başına 2 plot) — R3-9.
                ÖNCELİK: kuyrukta İLK sırada koşar.
k               {5, 10, 20, 40} — üç seçicide.
band_trim       ryckewaert: snv_R*/d1snv_R* sütunlarında 420–980 nm dışı çıkar;
                deep_patato: ilk/son 10 kanal çıkar.
rfe             α × step (3×3) + ölçekli RFE (scale=True).
mutual_info     n_neighbors {3, 5, 10}.
lasso           kesmesiz (k tavanı yok); inner ∈ {legacy, group} (R3-33).
representation  ryckewaert yalnız SNV spektrumu (204 sütun) — R3-15.
n_splits        dış fold sayısı {3, 5, 10}.
pls_components  PLS bileşeni nested iç GroupKFold(5) ile 1..10 (R3-50).

Verimlilik
----------
* Maske önbelleği E1 ile ORTAK (``outputs_rev1/_checkpoints/masks``): değiştirilmemiş
  veri + leaf grupları için bağlam E1 ile aynıdır → varsayılan hücreler (k=20, n_splits=5,
  ...) E1'in maskelerini yeniden kullanır. Değiştirilmiş veri (bant kırpma, temsil,
  plot grupları) için bağlama veri özeti (``fp``) eklenir → çakışma olmaz.
* LASSO: LassoCV fit'i k'dan bağımsızdır. |katsayı| önbelleği
  (``outputs_rev1/_checkpoints/lasso_coef``) ile aynı fit imzasını paylaşan hücreler
  (x varyantı, grup düzeyi, inner, n_splits) TEK görevde sırayla koşar → her LassoCV
  fit'i bir kez yapılır (paralel görevler aynı fit'i iki kez hesaplamaz).

Çıktılar (``outputs_rev1/E3_sensitivity/``)
------------------------------------------
sensitivity.csv            eksen, değer, kombinasyon, seçici → Δ (fold-mean R²) ortalama,
                           medyan, 2.5–97.5 persentil; P2/P3 ortalamaları; varsayılan hücreye
                           göre eşli fark (aynı tekrarlar).
sensitivity_by_repeat.csv  hücre × tekrar ham değerleri (örnek kimliği yok).
run_config.json

Çalıştırma (proje kökünden)::

    .venv\\Scripts\\python.exe -m src.m07_revision.e3_sensitivity --quick
    .venv\\Scripts\\python.exe -m src.m07_revision.e3_sensitivity --timing
    .venv\\Scripts\\python.exe -m src.m07_revision.e3_sensitivity
    .venv\\Scripts\\python.exe -m src.m07_revision.e3_sensitivity --check
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.splits import make_outer_splits, splits_signature
from src.m04_features.selectors import SELECTOR_FAMILY
from src.m07_revision import common as C
from src.m07_revision.engine import MaskCache, evaluate_arms, resolve_selector

EXP = "E3_sensitivity"
ARMS = ("P2_apparent", "P3_nested")
COST_ORDER = {"lasso": 0, "mutual_info": 1, "rfe": 2}


# ---------------------------------------------------------------------------
# Hücreler (config'ten; koşudan önce sabit)
# ---------------------------------------------------------------------------
def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", s)


def build_cells(cfg: dict) -> list[dict]:
    """Config eksenlerinden hücre listesi. Her hücre: eksen, değer ve varsayılandan fark."""
    e = cfg[EXP]
    ax = e["axes"]
    sels = list(e["selectors"])
    cells: list[dict] = []

    def add(axis, value, *, selectors=None, datasets=None, k="default", sel_override=None,
            x_variant=None, group_level=None, n_splits=None, model="pls", model_kw=None):
        cells.append({"axis": axis, "value": str(value), "cell": _slug(f"{axis}={value}"),
                      "selectors": list(selectors or sels),
                      "datasets": None if datasets is None else sorted(datasets),
                      "k": k, "sel_override": sel_override or {},
                      "x_variant": x_variant or {}, "group_level": group_level or {},
                      "n_splits": int(n_splits or cfg["common"]["n_splits"]),
                      "model": model, "model_kw": model_kw or {}})

    # 1) R3-9 — plot gruplaması (ÖNCELİKLİ)
    for ds, levels in (ax.get("grouping") or {}).items():
        for lv in levels:
            add("grouping", lv, datasets={ds}, group_level={ds: lv})
    # 2) k
    for kv in ax["k"]:
        add("k", kv, k=int(kv))
    # 3) bant kırpma
    bt = ax["band_trim"]
    add("band_trim", "trimmed", x_variant={ds: dict(spec) for ds, spec in bt.items()})
    # 4) RFE α × step + ölçekli
    for a in ax["rfe"]["alpha"]:
        for s in ax["rfe"]["step"]:
            add("rfe", f"alpha={a},step={s}", selectors=["rfe"],
                sel_override={"rfe": {"alpha": float(a), "step": float(s)}})
    for sc in ax["rfe"].get("scale", []):
        add("rfe", f"scale={sc}", selectors=["rfe"], sel_override={"rfe": {"scale": bool(sc)}})
    # 5) MI n_neighbors
    for nn in ax["mutual_info"]["n_neighbors"]:
        add("mutual_info", f"n_neighbors={nn}", selectors=["mutual_info"],
            sel_override={"mutual_info": {"n_neighbors": int(nn)}})
    # 6) LASSO: kesmesiz + inner
    if ax["lasso"].get("uncapped"):
        add("lasso", "uncapped", selectors=["lasso"], k=None)
    for inner in ax["lasso"]["inner"]:
        add("lasso", f"inner={inner}", selectors=["lasso"], sel_override={"lasso": {"inner": inner}})
    # 7) temsil
    for ds, vals in (ax.get("representation") or {}).items():
        for v in vals:
            add("representation", v, datasets={ds}, x_variant={ds: {"representation": v}})
    # 8) dış fold sayısı
    for ns in ax["n_splits"]:
        add("n_splits", ns, n_splits=int(ns))
    # 9) PLS bileşeni iç CV ile
    pc = ax["pls_components"]
    add("pls_components", pc["tuned_by"], model="pls_tuned",
        model_kw={"grid": [int(c) for c in pc["grid"]], "n_inner": 5})
    return cells


def cell_by_id(cfg: dict) -> dict:
    return {c["cell"]: c for c in build_cells(cfg)}


def _k_of(cell: dict, cfg: dict):
    return cfg["common"]["k"] if cell["k"] == "default" else cell["k"]


def _sel_kw(cell: dict, cfg: dict, selector: str) -> dict:
    return C.selector_kwargs(cfg, selector, cell["sel_override"].get(selector))


def _x_variant_id(cell: dict, dataset: str) -> str:
    spec = cell["x_variant"].get(dataset)
    return "full" if not spec else _slug(json.dumps(spec, sort_keys=True))


def _group_level(cell: dict, dataset: str) -> str | None:
    return cell["group_level"].get(dataset)


def fit_signature(cell: dict, cfg: dict, dataset: str, selector: str) -> str:
    """Aynı seçici fit'lerini paylaşan hücreler aynı imzayı alır (LASSO için k hariç)."""
    if selector != "lasso":
        return cell["cell"]
    inner = _sel_kw(cell, cfg, "lasso").get("inner", "group")
    return _slug(f"lasso|x={_x_variant_id(cell, dataset)}|g={_group_level(cell, dataset) or 'default'}"
                 f"|inner={inner}|ns={cell['n_splits']}")


# ---------------------------------------------------------------------------
# Veri varyantları
# ---------------------------------------------------------------------------
def apply_x_variant(data: dict, spec: dict | None) -> tuple[np.ndarray, list[str]]:
    """Sütun alt kümesi: bant kırpma / yalnız SNV. Satırlar değişmez."""
    X, names = data["X"], list(data["feature_names"])
    if not spec:
        return X, names
    keep = np.ones(len(names), dtype=bool)
    if "keep_nm" in spec or "spectral_block_keep_nm" in spec:
        lo, hi = spec.get("keep_nm") or spec.get("spectral_block_keep_nm")
        for j, n in enumerate(names):
            m = re.fullmatch(r"(?:snv|d1snv)_R(\d+)", n)
            if m and not (lo <= int(m.group(1)) <= hi):
                keep[j] = False
    if "drop_first" in spec or "drop_last" in spec:
        a, b = int(spec.get("drop_first", 0)), int(spec.get("drop_last", 0))
        keep[:a] = False
        if b:
            keep[len(names) - b:] = False
    if spec.get("representation") == "snv_spectrum_only":
        keep &= np.array([n.startswith("snv_R") for n in names])
    if not keep.any():
        raise ValueError(f"x varyantı tüm sütunları çıkardı: {spec}")
    return X[:, keep], [n for n, k in zip(names, keep) if k]


# ---------------------------------------------------------------------------
# LASSO |katsayı| önbelleği
# ---------------------------------------------------------------------------
def make_lasso_coef_cached(root: Path):
    """select_lasso ile AYNI maskeleri veren, LassoCV fit'ini k'dan bağımsız önbellekleyen seçici.

    Anahtar: veri özeti (X, y, groups) + inner + max_iter + tol + seed (yalnız iç CV
    seed'e bağlıysa: inner="group" ve groups yok). Maske k-tavan kuralı
    ``selectors.lasso_mask_from_abs_coef`` ile türetilir (select_lasso ile tek kaynak).
    """
    from src.m04_features.selectors import _clip_k, lasso_mask_from_abs_coef, select_lasso

    root = Path(root)

    def select(X, y, k=20, *, groups=None, seed=42, inner="group", max_iter=None, tol=1e-4,
               return_info=False, **kw):
        uses_seed = inner == "group" and groups is None
        key = {"fp": C.data_fingerprint(X, y, groups), "inner": inner, "max_iter": max_iter,
               "tol": tol, "seed": int(seed) if uses_seed else None}
        h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()
        p = root / h[:2] / f"{h}.json"
        status = "hit"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            info, coef = d["info"], np.asarray(d["abs_coef"], dtype=float)
        else:
            status = "miss"
            _, info = select_lasso(X, y, None, groups=groups, seed=seed, inner=inner,
                                   max_iter=max_iter, tol=tol, return_info=True, return_coef=True)
            coef = np.asarray(info.pop("abs_coef"), dtype=float)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(f".{np.random.randint(1 << 30)}.tmp")
            tmp.write_text(json.dumps({"key": key, "info": info, "abs_coef": coef.tolist()}),
                           encoding="utf-8")
            tmp.replace(p)
        k_cap = None if k is None else _clip_k(k, X.shape[1])
        mask, trunc, filled = lasso_mask_from_abs_coef(coef, k_cap)
        info = {**info, "k_cap": k_cap, "truncated": trunc, "filled_from_zero": filled,
                "n_selected": int(mask.sum()), "coef_cache": status}
        return (mask, info) if return_info else mask

    return select


# ---------------------------------------------------------------------------
# Görevler
# ---------------------------------------------------------------------------
def build_tasks(cfg: dict, mode: str) -> list[dict]:
    e = cfg[EXP]
    R = e["repeats"]
    combos = cfg["common"]["combos"]
    if mode == "timing":
        R = 1             # tek tekrar ölçülür; estimate() × R ile ölçekler
    if mode == "quick":
        R = 1
        combos = [{"dataset": "ryckewaert", "target": "flavonol"},
                  {"dataset": "deep_patato", "target": "chlorophyll"}]
    groups: dict = {}
    for cell in build_cells(cfg):
        for c in combos:
            if cell["datasets"] is not None and c["dataset"] not in cell["datasets"]:
                continue
            for s in cell["selectors"]:
                if s not in e["selectors"]:
                    continue
                if mode in ("quick", "timing") and c["dataset"] == "ryckewaert" and s == "lasso":
                    continue      # pahalı; zaman tahmini E1 LassoCV çağrı sürelerinden (estimate)
                for r in range(R):
                    sig = fit_signature(cell, cfg, c["dataset"], s)
                    key = (c["dataset"], c["target"], s, r, sig)
                    groups.setdefault(key, []).append(cell["cell"])
    tasks = [{"dataset": d, "target": t, "selector": s, "repeat": r, "signature": sig,
              "cells": cells, "priority": int(any(cl.startswith("grouping") for cl in cells))}
             for (d, t, s, r, sig), cells in groups.items()]
    # Plot gruplaması (R3-9) İLK; sonra pahalı seçici önce (yük dengesi)
    tasks.sort(key=lambda t: (-t["priority"], COST_ORDER.get(t["selector"], 9),
                              t["dataset"] != "ryckewaert", t["repeat"], t["signature"]))
    return tasks


def cell_path(out_dir: Path, dataset: str, target: str, selector: str, cell: str, repeat: int) -> Path:
    return out_dir / "_checkpoints" / "cells" / f"{cell}__{dataset}__{target}__{selector}__r{repeat:02d}.pkl"


def _label(t: dict) -> str:
    return (f"r{t['repeat']:02d} {t['dataset']}/{t['target']}/{t['selector']} "
            f"[{','.join(t['cells'])}]")


def run_task(t: dict, cfg: dict, dirs: dict) -> dict:
    """Bir fit-imzası grubundaki hücreleri sırayla koş (her hücre kendi checkpoint'i)."""
    from threadpoolctl import threadpool_limits

    t0 = time.time()
    out = {"label": _label(t), "skipped": True, "seconds": 0.0}
    cells = cell_by_id(cfg)
    with threadpool_limits(limits=1):
        for cid in t["cells"]:
            p = cell_path(dirs["out"], t["dataset"], t["target"], t["selector"], cid, t["repeat"])
            if p.exists():
                continue
            out["skipped"] = False
            C.save_pickle(p, _run_cell(cells[cid], t, cfg, dirs))
    out["seconds"] = round(time.time() - t0, 2)
    return out


def _run_cell(cell: dict, t: dict, cfg: dict, dirs: dict) -> dict:
    tc = time.time()
    ds, tg, sel_name, r = t["dataset"], t["target"], t["selector"], t["repeat"]
    level = _group_level(cell, ds)
    data = C.load_combo(ds, tg, group_level=level)
    X, names = apply_x_variant(data, cell["x_variant"].get(ds))
    y, g = data["y"], data["groups"]
    splits = make_outer_splits(len(y), g, "group", cell["n_splits"], r, shuffle=True)
    canonical = (X is data["X"]) and level is None
    ctx = {"dataset": ds, "target": tg}
    if not canonical:      # değişmiş veri/grup → bağlama veri özeti (E1 anahtarlarıyla çakışmaz)
        ctx.update({"x_variant": _x_variant_id(cell, ds), "group_level": level or "default",
                    "fp": C.data_fingerprint(X, y, g)})
    sel_fn = (make_lasso_coef_cached(dirs["lasso_coef"]) if sel_name == "lasso"
              else resolve_selector(sel_name))
    cache = MaskCache(dirs["masks"])
    res = evaluate_arms(X, y, g, splits, cell["model"], sel_fn, arms=ARMS,
                        sel_kw=_sel_kw(cell, cfg, sel_name), model_kw=cell["model_kw"] or None,
                        k=_k_of(cell, cfg), repeat=r, base_seed=cfg["common"]["base_seed"],
                        seed_mode="derived", selector_name=sel_name,
                        sample_ids=data["sample_ids"], group_ids=data["group_ids"],
                        mask_cache=cache, cache_context=ctx)
    return {"task": {k: t[k] for k in ("dataset", "target", "selector", "repeat")},
            "cell": cell, "n_features_used": X.shape[1], "n_groups": int(len(np.unique(g))),
            "splits_signature": splits_signature(splits), "summary": res.summary,
            "contrasts": res.contrasts, "folds": res.folds,
            "seconds": round(time.time() - tc, 2), "cache": {"hits": cache.hits, "misses": cache.misses}}


# ---------------------------------------------------------------------------
# Toplama
# ---------------------------------------------------------------------------
def _default_cell_id(cfg: dict) -> str:
    return _slug(f"k={cfg['common']['k']}")


def aggregate(out_dir: Path, cfg: dict) -> pd.DataFrame:
    files = sorted((out_dir / "_checkpoints" / "cells").glob("*.pkl"))
    if not files:
        raise RuntimeError(f"{out_dir}: hücre checkpoint'i yok")
    rows = []
    for f in files:
        d = pickle.loads(f.read_bytes())
        t, cell = d["task"], d["cell"]
        s = d["summary"].set_index("arm")
        c = d["contrasts"]
        rows.append({
            "axis": cell["axis"], "value": cell["value"], "cell": cell["cell"],
            "dataset": t["dataset"], "target": t["target"], "selector": t["selector"],
            "family": SELECTOR_FAMILY.get(t["selector"], t["selector"]), "model": cell["model"],
            "repeat": t["repeat"], "n_splits": cell["n_splits"],
            "group_level": _group_level(cell, t["dataset"]) or "default",
            "n_features_used": d["n_features_used"], "n_groups": d["n_groups"],
            "splits_signature": d["splits_signature"],
            "apparent_r2_foldmean": s.at["P2_apparent", "r2_foldmean"],
            "nested_r2_foldmean": s.at["P3_nested", "r2_foldmean"],
            "delta_r2_foldmean": c["apparent_nested_difference_r2_foldmean"],
            "delta_r2_pooled": c["apparent_nested_difference_r2_pooled"],
            "delta_rmse_foldmean": c["apparent_nested_difference_rmse_foldmean"],
            "n_selected_apparent": s.at["P2_apparent", "n_selected_mean"],
            "n_selected_nested_mean": s.at["P3_nested", "n_selected_mean"],
            "n_components_nested_mean": s.at["P3_nested", "n_components_mean"],
        })
    by = pd.DataFrame(rows).sort_values(["axis", "value", "dataset", "target", "selector", "repeat"])
    by.to_csv(out_dir / "sensitivity_by_repeat.csv", index=False)

    # Varsayılan hücreye (k = common.k; diğer her şey varsayılan) göre eşli fark
    dflt = by[by.cell == _default_cell_id(cfg)][["dataset", "target", "selector", "repeat",
                                                  "delta_r2_foldmean"]]
    dflt = dflt.rename(columns={"delta_r2_foldmean": "delta_default"})
    m = by.merge(dflt, on=["dataset", "target", "selector", "repeat"], how="left")
    m["delta_minus_default"] = m["delta_r2_foldmean"] - m["delta_default"]

    keys = ["axis", "value", "dataset", "target", "selector", "family", "model", "n_splits",
            "group_level"]
    out = []
    for gk, gdf in m.groupby(keys, sort=False):
        dsc = C.describe(gdf["delta_r2_foldmean"])
        out.append({**dict(zip(keys, gk)), "n_repeats": dsc["n"],
                    "delta_mean": dsc["mean"], "delta_median": dsc["median"],
                    "delta_p2_5": dsc["p2_5"], "delta_p97_5": dsc["p97_5"],
                    "apparent_mean": gdf["apparent_r2_foldmean"].mean(),
                    "nested_mean": gdf["nested_r2_foldmean"].mean(),
                    "delta_pooled_mean": gdf["delta_r2_pooled"].mean(),
                    "delta_minus_default_mean": gdf["delta_minus_default"].mean(),
                    "n_features_used": int(gdf["n_features_used"].iloc[0]),
                    "n_selected_nested_mean": gdf["n_selected_nested_mean"].mean(),
                    "n_components_nested_mean": gdf["n_components_nested_mean"].mean()})
    summ = pd.DataFrame(out)
    summ.insert(len(keys), "is_default_cell", summ["value"].eq(str(cfg["common"]["k"])) & summ["axis"].eq("k"))
    summ.to_csv(out_dir / "sensitivity.csv", index=False)
    return summ


# ---------------------------------------------------------------------------
# Zaman tahmini (timing): ryck LASSO hariç ölçülür; ryck LASSO LassoCV çağrı sayısı ×
# E1'de 14-yollu çekişmede ölçülen çağrı süresinden tahmin edilir.
# ---------------------------------------------------------------------------
def _lasso_call_seconds() -> dict:
    """E1 maske önbelleğinden ryck LASSO çağrı süresi medyanı (yoksa çekişme ölçümü değerleri)."""
    out = {"flavonol": 122.0, "chlorophyll": 19.4}       # 2026-10-31 14-yollu ölçüm
    root = C.ROOT / "outputs_rev1" / "_checkpoints" / "masks"
    acc: dict = {}
    if root.exists():
        for f in root.glob("*/*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            k = d.get("key", {})
            if k.get("selector") == "lasso" and k.get("dataset") == "ryckewaert":
                sec = d.get("info", {}).get("selection_seconds")
                if sec:
                    acc.setdefault(k.get("target"), []).append(float(sec))
    for tg, v in acc.items():
        out[tg] = float(np.median(v))
    return out


def estimate(results: list[dict], cfg: dict, n_jobs: int) -> dict:
    R = cfg[EXP]["repeats"]
    measured = [r["seconds"] for r in results if not r.get("skipped")]
    full_tasks = [t for t in build_tasks(cfg, "full")
                  if t["dataset"] == "ryckewaert" and t["selector"] == "lasso"]
    per_call = _lasso_call_seconds()
    lasso_secs = []
    for t in full_tasks:
        cell = cell_by_id(cfg)[t["cells"][0]]
        fits = cell["n_splits"] + 1                      # P3 fold'ları + tüm-veri seçimi
        lasso_secs.append(fits * per_call.get(t["target"], 122.0))
    durations = measured * R + lasso_secs
    est = C.lpt_makespan(durations, n_jobs)
    return {"R": R, "n_jobs": n_jobs, "measured_tasks_R1": len(measured),
            "measured_cpu_seconds_R1": round(sum(measured), 1),
            "ryck_lasso_tasks": len(lasso_secs), "ryck_lasso_call_seconds": per_call,
            "ryck_lasso_cpu_hours": round(sum(lasso_secs) / 3600, 2),
            "total_cpu_hours": round(sum(durations) / 3600, 2),
            "estimate_wall_hours_LPT": round(est / 3600, 2),
            "note": ("Muhafazakâr: varsayılan imzalı ryck LASSO maskeleri (k=20) E1 önbelleğinden "
                     "gelir; tahmin bunları yeniden hesaplanmış sayar.")}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _done_path(dirs: dict, axes: set | None) -> Path:
    """Eksen filtreli koşunun DONE dosyası AYRI (``--axes grouping`` tam koşuyu 'bitti' göstermesin)."""
    if not axes:
        return dirs["done"]
    return dirs["done"].with_name(f"{EXP}.axes-{'-'.join(sorted(axes))}.DONE.json")


def check(cfg: dict, axes: set | None = None) -> int:
    dirs = C.experiment_dirs(EXP, "full")
    dirs["done"] = _done_path(dirs, axes)
    tasks = build_tasks(cfg, "full")
    cid_axis = {c["cell"]: c["axis"] for c in build_cells(cfg)}
    want = [(t["dataset"], t["target"], t["selector"], cid, t["repeat"]) for t in tasks for cid in t["cells"]
            if not axes or cid_axis[cid] in axes]
    have = sum(cell_path(dirs["out"], *w).exists() for w in want)
    print(f"{EXP}{' ' + str(sorted(axes)) if axes else ''}: {have}/{len(want)} hücre checkpoint'i | "
          f"DONE: {dirs['done'].exists()}")
    ok = have == len(want) and dirs["done"].exists()
    sp = dirs["out"] / "sensitivity.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        print("\nR3-9 plot gruplaması (Δ fold-mean R²):")
        print(s[s.axis == "grouping"][["dataset", "target", "selector", "n_repeats", "delta_median",
                                       "delta_p2_5", "delta_p97_5", "delta_minus_default_mean"]]
              .round(3).to_string(index=False))
    # Tutarlılık: varsayılan hücre E1'in ilk 10 tekrarının PLS Δ'sını birebir vermeli
    e1 = C.ROOT / "outputs_rev1" / "E1_repeated" / "_checkpoints" / "tasks"
    byr = dirs["out"] / "sensitivity_by_repeat.csv"
    if e1.exists() and byr.exists():
        b = pd.read_csv(byr)
        b = b[b.cell == _default_cell_id(cfg)]
        diffs = []
        for _, r in b.iterrows():
            f = e1 / f"{r.dataset}__{r.target}__{r.selector}__r{int(r['repeat']):02d}.pkl"
            if f.exists():
                d = pickle.loads(f.read_bytes())
                diffs.append(abs(d["models"]["pls"]["contrasts"]["apparent_nested_difference_r2_foldmean"]
                                 - r.delta_r2_foldmean))
        if diffs:
            print(f"\nVarsayılan hücre ↔ E1 (PLS, r<10): {len(diffs)} karşılaştırma, max |fark| = {max(diffs):.2e}")
            ok &= max(diffs) < 1e-9
    print("\nSONUÇ:", "TAMAM" if ok else "EKSİK / SORUN VAR")
    return 0 if ok else 1


def main() -> None:
    ap = C.base_argparser("E3 — duyarlılık ızgarası")
    ap.add_argument("--axes", type=str, default="",
                    help="yalnız bu eksenler (virgülle; ör. grouping). Boş = hepsi.")
    args = ap.parse_args()
    cfg = C.load_config()
    axes = {a.strip() for a in args.axes.split(",") if a.strip()}
    if args.check:
        sys.exit(check(cfg, axes))
    mode = "quick" if args.quick else "timing" if args.timing else "full"
    dirs = C.experiment_dirs(EXP, mode)
    dirs["done"] = _done_path(dirs, axes)
    if (mode == "timing" or (mode == "quick" and args.fresh)) and not args.aggregate_only:
        for d in (dirs["out"], dirs["masks"], dirs["lasso_coef"]):
            if mode == "timing" and d.exists():
                shutil.rmtree(d)
        if mode == "quick" and dirs["out"].exists():
            shutil.rmtree(dirs["out"])
    log = C.setup_logging(dirs["log"])
    if args.aggregate_only:
        aggregate(dirs["out"], cfg)
        log.info("E3 tabloları yeniden üretildi → %s", dirs["out"])
        return
    tasks = build_tasks(cfg, mode)
    if axes:
        cid = cell_by_id(cfg)
        tasks = [dict(t, cells=[c for c in t["cells"] if cid[c]["axis"] in axes]) for t in tasks]
        tasks = [t for t in tasks if t["cells"]]
    n_jobs = args.n_jobs or C.default_n_jobs()
    import functools

    from src.m07_revision.e3_sensitivity import run_task as entry
    C.run_experiment(
        name=EXP, mode=mode, tasks=tasks,
        entry=functools.partial(entry, cfg=cfg, dirs=dirs),
        aggregate=lambda: aggregate(dirs["out"], cfg), dirs=dirs, n_jobs=n_jobs,
        payload={"experiment": EXP, "repeats": cfg[EXP]["repeats"], "axes_filter": sorted(axes),
                 "cells": build_cells(cfg), "selectors": cfg[EXP]["selectors"],
                 "split": "GroupKFold(n_splits, shuffle=True, random_state=repeat); r=0..9 = E1 first 10",
                 "seeds": {"base_seed": cfg["common"]["base_seed"], "seed_mode": "derived"},
                 "config_blocks": {"common": cfg["common"], EXP: cfg[EXP]},
                 "mask_cache": "outputs_rev1/_checkpoints/masks (shared with E1)",
                 "lasso_coef_cache": "outputs_rev1/_checkpoints/lasso_coef"},
        log=log, label=_label,
        estimate=lambda res: estimate(res, cfg, n_jobs))


if __name__ == "__main__":
    main()
