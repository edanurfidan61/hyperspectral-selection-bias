"""E5 — Permütasyon testi (plan §3 E5) → R3-34, 70, 72.

* B = 200 permütasyon; p = (1 + b) / (B + 1), b = #{perm ≥ observed}; tek yönlü.
  İstatistik: nested (P3) fold-mean R². Apparent (P2) için de aynı p bilgi amaçlı.
* Bölme: makaledeki permütasyonla aynı — shuffle'sız GroupKFold(5) (repeat 0);
  X ve gruplar SABİT, yalnız y karıştırılır. Seçici seed'leri permütasyonlar arasında
  aynı (engine repeat=0) → fark yalnız y'den gelir.
* Seçiciler: RFE, LASSO (inner=group, max_iter=20000, tol=1e-3), MI; model PLS.
  LASSO yakınsama uyarıları SAYILIR (bastırılmaz; seçim başına kaydedilir).
* Şemalar (config ``E5_permutation.schemes``):
    ryckewaert  free         — her leaf ayrı grup → serbest permütasyon
                within_plot  — plot grupları (10) içinde kısıtlı permütasyon
    deep_patato field_block  — Field başına örnek sayısı eşitse (24 × 3) Field
                               etiketleri karıştırılır, bloklar bütün olarak yer
                               değiştirir (blok içi sıra korunur). Eşit değilse Field
                               içinde permütasyona düşülür. Kullanılan şema
                               ``scheme_used`` sütununa yazılır.
* Permütasyon seed'i: ``derive_seed(base_seed, b, -1, "perm:<şema>")``.

Görev birimi: (dataset, target, selector, scheme, b); b = −1 gözlenen (permütasyonsuz)
değerdir ve kombinasyon × seçici başına BİR kez hesaplanır (iki ryck şeması paylaşır).

Çıktılar (``outputs_rev1/E5_permutation/``): permutation_summary.csv,
permutation_null.csv.gz (permütasyon başına değerler; örnek kimliği yok),
lasso_convergence.csv, run_config.json.

Çalıştırma (proje kökünden)::

    .venv\\Scripts\\python.exe -m src.m07_revision.e5_permutation --quick
    .venv\\Scripts\\python.exe -m src.m07_revision.e5_permutation --timing
    .venv\\Scripts\\python.exe -m src.m07_revision.e5_permutation
    .venv\\Scripts\\python.exe -m src.m07_revision.e5_permutation --check
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.splits import derive_seed, make_outer_splits, splits_signature
from src.m07_revision import common as C
from src.m07_revision.engine import evaluate_arms, resolve_selector

EXP = "E5_permutation"
ARMS = ("P2_apparent", "P3_nested")
COST_ORDER = {"lasso": 0, "mutual_info": 1, "rfe": 2}


# ---------------------------------------------------------------------------
# Permütasyon şemaları
# ---------------------------------------------------------------------------
def permute_y(y: np.ndarray, scheme: str, b: int, data: dict, base_seed: int) -> tuple[np.ndarray, str]:
    """y'nin permütasyonu ve gerçekte kullanılan şema."""
    rng = np.random.default_rng(derive_seed(base_seed, b, -1, f"perm:{scheme}"))
    n = len(y)
    if scheme == "free":
        return y[rng.permutation(n)], "free"
    if scheme == "within_plot":
        plots = C.load_combo(data["dataset"], data["target"], group_level="plot")["groups"]
        if len(plots) != n:
            raise ValueError("plot grupları y ile hizalı değil")
        out = y.copy()
        for pl in np.unique(plots):
            idx = np.flatnonzero(plots == pl)
            out[idx] = y[rng.permutation(idx)]
        return out, "within_plot"
    if scheme == "field_block":
        g = np.asarray(data["groups"])
        uniq, counts = np.unique(g, return_counts=True)
        out = y.copy()
        if np.all(counts == counts[0]):
            # Blokları bütün olarak değiştir: Field u_j'nin satırları ← Field π(u_j)'nin y'leri
            rows = {u: np.flatnonzero(g == u) for u in uniq}     # satır sırası korunur
            perm = rng.permutation(uniq)
            for u, v in zip(uniq, perm):
                out[rows[u]] = y[rows[v]]
            return out, f"field_block(equal_size={int(counts[0])})"
        for u in uniq:                                           # eşit değil → Field içinde
            idx = np.flatnonzero(g == u)
            out[idx] = y[rng.permutation(idx)]
        return out, "within_field(unequal_blocks)"
    raise ValueError(f"Bilinmeyen şema {scheme!r}")


# ---------------------------------------------------------------------------
# Görevler
# ---------------------------------------------------------------------------
def _sel_kw(cfg: dict, selector: str) -> dict:
    e = cfg[EXP]
    if selector == "lasso":
        lk = e["lasso"]
        return C.selector_kwargs(cfg, "lasso", {"inner": lk["inner"], "max_iter": int(lk["max_iter"]),
                                                "tol": float(lk["tol"])})
    return C.selector_kwargs(cfg, selector)


def build_tasks(cfg: dict, mode: str, exclude: set | None = None, only: set | None = None,
                n_perm: int | None = None) -> list[dict]:
    """Görev listesi. ``only`` / ``exclude``: "dataset/seçici" çiftleri. ``n_perm``: B yerine
    (DEVIATIONS.md D2 — ryck LASSO B=50). Permütasyon seed'i b'den türetildiği için b=0..n_perm−1
    B=200 akışının İLK n_perm permütasyonudur → sonradan bayraksız koşu 200'e tamamlar.
    """
    e = cfg[EXP]
    B = int(e["B"]) if n_perm is None else int(n_perm)
    combos = cfg["common"]["combos"]
    sels = list(e["selectors"])
    if mode == "quick":
        B = 2
        combos = [{"dataset": "ryckewaert", "target": "flavonol"},
                  {"dataset": "deep_patato", "target": "chlorophyll"}]
    elif mode == "timing":
        B = 2
    tasks = []
    for c in combos:
        for s in sels:
            if f"{c['dataset']}/{s}" in (exclude or set()):
                continue                     # --exclude (ör. ryckewaert/lasso ayrı koşu)
            if only and f"{c['dataset']}/{s}" not in only:
                continue                     # --only (ör. yalnız ryckewaert/lasso)
            ryck_lasso = c["dataset"] == "ryckewaert" and s == "lasso"
            if mode == "quick" and ryck_lasso:
                continue                     # pahalı; duman testinde potato LASSO yeterli
            base = {"dataset": c["dataset"], "target": c["target"], "selector": s}
            tasks.append({**base, "scheme": "observed", "b": -1,
                          "p2_only": mode == "timing" and ryck_lasso})
            if mode == "timing" and ryck_lasso:
                # Tek seçim çağrısı ölçülür (× n_splits+1). PERMÜTE y ile de ölçülür: LassoCV
                # permüte y'de gözlenen y'ye göre çok daha yavaş yakınsar (2026-10-31: ×5–×30).
                for sch in e["schemes"][c["dataset"]]:
                    tasks.append({**base, "scheme": sch, "b": 0, "p2_only": True})
                continue
            for sch in e["schemes"][c["dataset"]]:
                for b in range(B):
                    tasks.append({**base, "scheme": sch, "b": b, "p2_only": False})
    tasks.sort(key=lambda t: (COST_ORDER.get(t["selector"], 9), t["dataset"] != "ryckewaert",
                              t["b"], t["scheme"]))
    return tasks


def task_path(out_dir: Path, t: dict) -> Path:
    tag = "observed" if t["b"] < 0 else f"{t['scheme']}__b{t['b']:03d}"
    suffix = "__p2only" if t.get("p2_only") else ""
    return (out_dir / "_checkpoints" / f"{t['dataset']}__{t['target']}__{t['selector']}"
            / f"{tag}{suffix}.pkl")


def _label(t: dict) -> str:
    return f"{t['dataset']}/{t['target']}/{t['selector']} {t['scheme']} b={t['b']}"


def run_task(t: dict, cfg: dict, dirs: dict) -> dict:
    from threadpoolctl import threadpool_limits

    p = task_path(dirs["out"], t)
    if p.exists():
        return {"label": _label(t), "skipped": True, "seconds": 0.0}
    t0 = time.time()
    with threadpool_limits(limits=1):
        data = C.load_combo(t["dataset"], t["target"])
        y, used = (data["y"], "observed") if t["b"] < 0 else \
            permute_y(data["y"], t["scheme"], t["b"], data, cfg["common"]["base_seed"])
        sp = cfg[EXP]["split"]
        splits = make_outer_splits(data["n"], data["groups"], sp["scheme"], int(sp["n_splits"]),
                                   int(sp["repeat"]), shuffle=bool(sp["shuffle"]))
        res = evaluate_arms(data["X"], y, data["groups"], splits, cfg[EXP]["model"],
                            resolve_selector(t["selector"]),
                            arms=("P2_apparent",) if t.get("p2_only") else ARMS,
                            sel_kw=_sel_kw(cfg, t["selector"]), k=cfg["common"]["k"], repeat=0,
                            base_seed=cfg["common"]["base_seed"], seed_mode="derived",
                            selector_name=t["selector"])
    f = res.folds
    # Seçim başına yakınsama uyarısı: P2 tek seçim (fold 0 satırı), P3 fold başına bir seçim
    sel_rows = f[(f.arm == "P3_nested") | ((f.arm == "P2_apparent") & (f.fold == 0))]
    warns = [json.loads(s).get("n_convergence_warnings", 0) or 0 for s in sel_rows["selector_info"]]
    secs = [json.loads(s).get("selection_seconds") for s in sel_rows["selector_info"]]
    s = res.summary.set_index("arm")
    out = {"task": t, "scheme_used": used, "perm_seed": None if t["b"] < 0 else
           derive_seed(cfg["common"]["base_seed"], t["b"], -1, f"perm:{t['scheme']}"),
           "splits_signature": splits_signature(splits),
           "apparent_r2_foldmean": float(s.at["P2_apparent", "r2_foldmean"]),
           "nested_r2_foldmean": float(s.at["P3_nested", "r2_foldmean"]) if "P3_nested" in s.index else None,
           "nested_r2_pooled": float(s.at["P3_nested", "r2_pooled"]) if "P3_nested" in s.index else None,
           "n_selections": int(len(sel_rows)), "convergence_warnings": [int(w) for w in warns],
           "selection_seconds": [x for x in secs if x is not None],
           "seconds": round(time.time() - t0, 2)}
    C.save_pickle(p, out)
    return {"label": _label(t), "skipped": False, "seconds": out["seconds"], "task": t}


# ---------------------------------------------------------------------------
# Toplama
# ---------------------------------------------------------------------------
def aggregate(out_dir: Path, cfg: dict) -> None:
    files = [f for f in sorted((out_dir / "_checkpoints").glob("*/*.pkl")) if "__p2only" not in f.name]
    if not files:
        raise RuntimeError(f"{out_dir}: checkpoint yok")
    recs = [pickle.loads(f.read_bytes()) for f in files]
    rows = [{**{k: r["task"][k] for k in ("dataset", "target", "selector", "scheme", "b")},
             "scheme_used": r["scheme_used"], "perm_seed": r["perm_seed"],
             "apparent_r2_foldmean": r["apparent_r2_foldmean"],
             "nested_r2_foldmean": r["nested_r2_foldmean"], "nested_r2_pooled": r["nested_r2_pooled"],
             "n_selections": r["n_selections"], "n_convergence_warnings": int(sum(r["convergence_warnings"])),
             "n_selections_with_warning": int(sum(w > 0 for w in r["convergence_warnings"]))}
            for r in recs]
    df = pd.DataFrame(rows)
    perm = df[df.b >= 0].sort_values(["dataset", "target", "selector", "scheme", "b"])
    perm.to_csv(out_dir / "permutation_null.csv.gz", index=False, compression="gzip")
    obs = df[df.b < 0].set_index(["dataset", "target", "selector"])

    out = []
    for (ds, tg, sel, sch), x in perm.groupby(["dataset", "target", "selector", "scheme"]):
        if (ds, tg, sel) not in obs.index:
            continue
        o = obs.loc[(ds, tg, sel)]
        B = len(x)
        on, oa = float(o["nested_r2_foldmean"]), float(o["apparent_r2_foldmean"])
        out.append({
            "dataset": ds, "target": tg, "selector": sel, "scheme": sch,
            "scheme_used": ";".join(sorted(set(x.scheme_used))), "B": B, "B_planned": int(cfg[EXP]["B"]),
            "B_deviation": ("" if B == int(cfg[EXP]["B"]) else
                            "DEVIATIONS.md D2" if (ds, sel) == ("ryckewaert", "lasso") else "B != B_planned"),
            "statistic": "nested_r2_foldmean", "observed_nested": on, "observed_apparent": oa,
            "observed_difference": oa - on,
            "perm_nested_mean": x.nested_r2_foldmean.mean(), "perm_nested_sd": x.nested_r2_foldmean.std(ddof=1),
            "perm_nested_p95": x.nested_r2_foldmean.quantile(0.95),
            "perm_apparent_mean": x.apparent_r2_foldmean.mean(),
            "p_nested": (1 + int((x.nested_r2_foldmean >= on).sum())) / (B + 1),
            "p_apparent": (1 + int((x.apparent_r2_foldmean >= oa).sum())) / (B + 1),
            "p_formula": "(1 + #{perm >= observed}) / (B + 1), one-sided",
            "lasso_convergence_warnings_total": int(x.n_convergence_warnings.sum()
                                                    + o["n_convergence_warnings"]),
        })
    pd.DataFrame(out).to_csv(out_dir / "permutation_summary.csv", index=False)

    las = df[df.selector == "lasso"]
    if not las.empty:
        lk = cfg[EXP]["lasso"]
        conv = las.groupby(["dataset", "target", "scheme"]).agg(
            n_evaluations=("b", "size"), n_selections=("n_selections", "sum"),
            n_convergence_warnings=("n_convergence_warnings", "sum"),
            n_selections_with_warning=("n_selections_with_warning", "sum")).reset_index()
        conv["frac_selections_with_warning"] = conv.n_selections_with_warning / conv.n_selections
        conv["mean_warnings_per_selection"] = conv.n_convergence_warnings / conv.n_selections
        conv["tol"], conv["max_iter"], conv["inner"] = float(lk["tol"]), int(lk["max_iter"]), lk["inner"]
        conv.to_csv(out_dir / "lasso_convergence.csv", index=False)


# ---------------------------------------------------------------------------
# Zaman tahmini
# ---------------------------------------------------------------------------
def estimate(results: list[dict], cfg: dict, n_jobs: int) -> dict:
    """(dataset, target, selector) başına ölçülen değerlendirme süresi × tam koşudaki görev sayısı.

    ryck LASSO: yalnız tüm-veri seçimi ölçülür (p2_only) → × (n_splits + 1).
    """
    ns = int(cfg[EXP]["split"]["n_splits"])
    meas: dict = {}
    for r in results:
        t = r.get("task")
        if not t or r.get("skipped"):
            continue
        sec = r["seconds"] * ((ns + 1) if t.get("p2_only") else 1)
        meas.setdefault((t["dataset"], t["target"], t["selector"], t["scheme"]), []).append(sec)
    per = {k: float(np.mean(v)) for k, v in meas.items()}

    def cost(t):
        # Önce aynı şema (permüte y maliyeti), yoksa kombinasyonun herhangi bir permütasyon ölçümü
        k4 = (t["dataset"], t["target"], t["selector"], t["scheme"])
        if k4 in per:
            return per[k4]
        alt = [v for k, v in per.items() if k[:3] == k4[:3] and k[3] != "observed"]
        return float(np.mean(alt)) if alt else per.get(k4[:3] + ("observed",), 0.0)

    full = build_tasks(cfg, "full")
    durations = [cost(t) for t in full]
    est = C.lpt_makespan(durations, n_jobs)
    by = {}
    for t, d in zip(full, durations):
        k = f"{t['dataset']}/{t['selector']}"
        by[k] = by.get(k, 0.0) + d
    rl = [d for t, d in zip(full, durations) if t["dataset"] == "ryckewaert" and t["selector"] == "lasso"]
    rest = [d for t, d in zip(full, durations) if not (t["dataset"] == "ryckewaert" and t["selector"] == "lasso")]
    return {"n_jobs": n_jobs, "B": int(cfg[EXP]["B"]), "n_tasks_full": len(durations),
            "per_evaluation_seconds": {"/".join(k): round(v, 1) for k, v in per.items()},
            "cpu_hours_by_dataset_selector": {k: round(v / 3600, 2) for k, v in by.items()},
            "total_cpu_hours": round(sum(durations) / 3600, 2),
            "estimate_wall_hours_LPT": round(est / 3600, 2),
            "wall_hours_LPT_without_ryck_lasso": round(C.lpt_makespan(rest, n_jobs) / 3600, 2),
            "wall_hours_LPT_ryck_lasso_only": round(C.lpt_makespan(rl, n_jobs) / 3600, 2),
            "note": "ölçüm düşük/orta yükte; 14 süreçte LassoCV ≈ ×1.5–1.9 yavaşlar (E1)"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _run_tag(exclude: set | None, only: set | None = None, n_perm: int | None = None) -> str:
    """Filtreli koşuların DONE dosyası ayrı (ör. E5_permutation.only-ryckewaert_lasso.B50.DONE.json)."""
    j = lambda xs: "-".join(sorted(x.replace("/", "_") for x in xs))  # noqa: E731
    return ((f".excl-{j(exclude)}" if exclude else "") + (f".only-{j(only)}" if only else "")
            + (f".B{int(n_perm)}" if n_perm is not None else ""))


def check(cfg: dict, exclude: set | None = None, only: set | None = None, n_perm: int | None = None) -> int:
    dirs = C.experiment_dirs(EXP, "full")
    dirs["done"] = dirs["done"].with_name(f"{EXP}{_run_tag(exclude, only, n_perm)}.DONE.json")
    tasks = build_tasks(cfg, "full", exclude, only, n_perm)
    ok = C.check_tasks(EXP, tasks, lambda t: task_path(dirs["out"], t), dirs["done"])
    sp = dirs["out"] / "permutation_summary.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        print(s[["dataset", "target", "selector", "scheme", "scheme_used", "B", "B_planned", "observed_nested",
                 "perm_nested_mean", "p_nested", "p_apparent", "lasso_convergence_warnings_total"]]
              .round(4).to_string(index=False))
    print("\nSONUÇ:", "TAMAM" if ok else "EKSİK / SORUN VAR")
    return 0 if ok else 1


def main() -> None:
    ap = C.base_argparser("E5 — permütasyon testi")
    ap.add_argument("--exclude", type=str, default="",
                    help="atlanacak dataset/seçici çiftleri (virgülle), ör. ryckewaert/lasso. "
                         "Atlanan çift daha sonra --exclude olmadan koşulunca checkpoint'ten devam eder.")
    ap.add_argument("--only", type=str, default="",
                    help="yalnız bu dataset/seçici çiftleri (virgülle), ör. ryckewaert/lasso")
    ap.add_argument("--n-perm", type=int, default=None,
                    help="B yerine permütasyon sayısı (yalnız tam koşu; DEVIATIONS.md D2). "
                         "b=0..n−1 = B=200 akışının ilk n'i")
    args = ap.parse_args()
    cfg = C.load_config()
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    if args.n_perm is not None and args.n_perm < 1:
        ap.error("--n-perm >= 1 olmalı")
    if args.check:
        sys.exit(check(cfg, exclude, only, args.n_perm))
    mode = "quick" if args.quick else "timing" if args.timing else "full"
    n_perm = args.n_perm if mode == "full" else None
    dirs = C.experiment_dirs(EXP, mode)
    dirs["done"] = dirs["done"].with_name(f"{EXP}{_run_tag(exclude, only, n_perm)}.DONE.json")
    if (mode == "timing" or (mode == "quick" and args.fresh)) and not args.aggregate_only \
            and dirs["out"].exists():
        shutil.rmtree(dirs["out"])
    log = C.setup_logging(dirs["log"])
    if args.aggregate_only:
        aggregate(dirs["out"], cfg)
        log.info("E5 tabloları yeniden üretildi → %s", dirs["out"])
        return
    tasks = build_tasks(cfg, mode, exclude, only, n_perm)
    n_jobs = args.n_jobs or C.default_n_jobs()
    import functools

    from src.m07_revision.e5_permutation import run_task as entry
    C.run_experiment(
        name=EXP, mode=mode, tasks=tasks, entry=functools.partial(entry, cfg=cfg, dirs=dirs),
        aggregate=lambda: aggregate(dirs["out"], cfg), dirs=dirs, n_jobs=n_jobs,
        payload={"experiment": EXP, "B": cfg[EXP]["B"], "exclude": sorted(exclude), "only": sorted(only),
                 "n_perm_override": n_perm,
                 "seeds": {"permutation": "derive_seed(base_seed, b, -1, 'perm:<scheme>')",
                           "selector": "derive_seed(base_seed, 0, fold, selector) — same for all b"},
                 "config_blocks": {"common": cfg["common"], EXP: cfg[EXP]}},
        log=log, label=_label, estimate=lambda res: estimate(res, cfg, n_jobs))


if __name__ == "__main__":
    main()
