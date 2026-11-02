"""E6 — GA (plan §3 E6) → R3-27,28,29,39,44,48,74,78.

Varyantlar (``config/revision.yaml`` ``E6_ga.variants``)
------------------------------------------------------
G0  orijinal bölme (repeat=0, shuffle=False), seed 0 (``seed_mode="fixed"``),
    ryck/flavonol/PLS, P2 + P3. outputs3 seed-0 satırı ile karşılaştırılır
    (``g0_reproduction.csv``). NOT: makale başlığı (0.615 / 0.269 / +0.346) 12 seed'in
    ORTALAMASIDIR; seed 0 tek başına outputs3'te 0.611 / 0.290 / +0.321'dir.
G1  tekrarlı bölme (r = 0..9, E1 ile aynı ``random_state``), tam bütçe, 3 PLS
    kombinasyonu, P1 + P2 + P3 (R3-44). Seed: ``derive_seed(base, r, fold, "ga")``.
G2  kardinalite-sabit GA (k = 20; R3-27, R3-74), r = 0..4, 3 kombinasyon, P2 + P3.
G3  kırpmasız fitness (R3-28), r = 0..4, ryck/flavonol, P2 + P3.
G4  (opsiyonel, "süre kalırsa") bu modülde YOK.

İki aşama
---------
A) Her GA koşusu (varyant, kombinasyon, tekrar, fold | tüm veri) AYRI bir görevdir
   (tek süreçte seri; 14 görev paralel). Maske, engine'in kullanacağı AYNI anahtarla
   ``MaskCache``'e yazılır; arşiv ``E6_ga/archive/<run>.npz`` (R3-78, R3-29).
B) (varyant, kombinasyon, tekrar) başına ``evaluate_arms`` önbellekten (yalnız
   isabet) çalışır — GA yeniden koşmaz. Aşama B ana süreçte, saniyeler sürer.

Çıktılar (``outputs_rev1/E6_ga/``)
----------------------------------
summary_by_combo.csv      E1 ile aynı uzun biçim (variant sütunu ek) → E1 tablolarına eklenir
summary_by_variant.csv    varyant × kombinasyon: Δ (apparent − nested) ortalama/medyan/persentil,
                          fold-partitioning contrast (P1 − P2; G1)
g0_reproduction.csv       G0 ↔ outputs3 seed 0
adaptive_overfitting.csv  seçilen maskenin iç fitness'ı − dış-test R² (R3-48)
ga_runs.csv               GA koşusu başına: iç fitness, kardinalite, değerlendirme sayıları,
                          süre; hesaplama bütçesi (R3-39: fit = benzersiz aday × 5 iç fold)
logbook.csv.gz            nesil başına en iyi/ortalama/std fitness, kardinalite dağılımı,
                          benzersiz aday sayısı
fold_metrics.csv.gz       fold düzeyi (maske/örnek kimliği yok)
cache_invariance.csv      (quick/timing) aday önbelleği açık/kapalı → aynı maske/fitness
samples/folds.csv.gz      örnek düzeyi tahminler (git-ignored)
archive/*.npz             GA arşivleri (git-ignored; *.npz kuralı)

Çalıştırma (proje kökünden)::

    .venv\\Scripts\\python.exe -m src.m07_revision.e6_ga --quick
    .venv\\Scripts\\python.exe -m src.m07_revision.e6_ga --timing          # G0 tam bütçe + önbelleksiz kopya
    .venv\\Scripts\\python.exe -m src.m07_revision.e6_ga --variants G0,G1
    .venv\\Scripts\\python.exe -m src.m07_revision.e6_ga --variants G2,G3
    .venv\\Scripts\\python.exe -m src.m07_revision.e6_ga --check --variants G0,G1
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
from src.m07_revision.engine import MaskCache, evaluate_arms, folds_long

EXP = "E6_ga"
VARIANT_ORDER = ("G0", "G1", "G2", "G3")
REF_CSV = C.ROOT / "outputs3" / "tables_foldmean" / "selection_bias_multiseed_raw.csv"


# ---------------------------------------------------------------------------
# Varyant tanımları (config → somut ayarlar)
# ---------------------------------------------------------------------------
def variant_spec(cfg: dict, name: str, mode: str) -> dict:
    e = cfg[EXP]
    v = e["variants"][name]
    budget = dict(v.get("budget") or e["timing_first"]["budget"])
    combos = cfg["common"]["combos"] if v.get("combos") == "all" else list(v["combos"])
    fixed = name == "G0"
    repeats = 1 if fixed else int(v["repeats"])
    if mode == "quick":
        budget = {"pop": 16, "ngen": 3}
        repeats = 1
        combos = [{"dataset": "ryckewaert", "target": "flavonol"}]
    return {
        "variant": name, "combos": combos, "repeats": repeats, "arms": tuple(v["arms"]),
        "pop": int(budget["pop"]), "ngen": int(budget["ngen"]),
        "representation": "fixed_k" if v.get("individual") == "20_indices" else "binary",
        "k": int(v["k"]) if v.get("k") else None,
        "clip": bool(v.get("clip_fitness_at_zero", True)),
        "seed_mode": "fixed" if fixed else "derived",
        "base_seed": int(v["seed"]) if fixed else int(cfg["common"]["base_seed"]),
        "shuffle": bool(v.get("split", {}).get("shuffle", True)),
        "n_splits": int(v.get("split", {}).get("n_splits", cfg["common"]["n_splits"])),
    }


def sel_kw_of(spec: dict) -> dict:
    """Engine'e (ve MaskCache anahtarına) giden GA ayarları — varyantlar çakışmaz."""
    return {"variant": spec["variant"], "pop": spec["pop"], "ngen": spec["ngen"],
            "representation": spec["representation"], "clip": spec["clip"]}


def splits_of(spec: dict, data: dict, repeat: int) -> list:
    return make_outer_splits(data["n"], data["groups"], "group", spec["n_splits"], repeat,
                             shuffle=spec["shuffle"])


def sel_seed(spec: dict, repeat: int, fold: int) -> int:
    """engine.evaluate_arms.sel_seed ile AYNI (seçici adı "ga")."""
    if spec["seed_mode"] == "fixed":
        return int(spec["base_seed"])
    return derive_seed(spec["base_seed"], repeat, fold, "ga")


def run_id(t: dict) -> str:
    f = "all" if t["fold"] < 0 else f"f{t['fold']}"
    tail = "" if t.get("candidate_cache", True) else "__nocache"
    return f"{t['variant']}__{t['dataset']}__{t['target']}__r{t['repeat']:02d}__{f}{tail}"


# ---------------------------------------------------------------------------
# Görevler (Aşama A)
# ---------------------------------------------------------------------------
def build_tasks(cfg: dict, mode: str, variants: list[str]) -> list[dict]:
    tasks = []
    for vi, name in enumerate(variants):
        spec = variant_spec(cfg, name, mode)
        for c in spec["combos"]:
            for r in range(spec["repeats"]):
                folds = [-1] if any(a in spec["arms"] for a in ("P1_random", "P2_apparent")) else []
                if "P3_nested" in spec["arms"]:
                    folds += list(range(spec["n_splits"]))
                for f in folds:
                    tasks.append({"variant": name, "dataset": c["dataset"], "target": c["target"],
                                  "repeat": r, "fold": f, "candidate_cache": True, "order": vi})
    # Önbellek değişmezliği (quick/timing): G0 tüm-veri koşusunun önbelleksiz kopyası
    if mode in ("quick", "timing") and "G0" in variants:
        tasks.append({"variant": "G0", "dataset": "ryckewaert", "target": "flavonol", "repeat": 0,
                      "fold": -1, "candidate_cache": False, "order": -1})
    # Önbelleksiz kopya (en uzun) önce; sonra varyant sırası (G0, G1 önce), ryck önce
    tasks.sort(key=lambda t: (t["order"], t["dataset"] != "ryckewaert", t["target"], t["repeat"], t["fold"]))
    return tasks


def _label(t: dict) -> str:
    return run_id(t)


def ga_run_path(out: Path, t: dict) -> Path:
    return out / "_checkpoints" / "ga_runs" / f"{run_id(t)}.pkl"


def run_task(t: dict, cfg: dict, dirs: dict, mode: str) -> dict:
    """Tek GA koşusu → MaskCache + arşiv + kayıt."""
    from threadpoolctl import threadpool_limits

    from src.m07_revision.ga import run_ga, save_archive

    p = ga_run_path(dirs["out"], t)
    if p.exists():
        return {"label": _label(t), "skipped": True, "seconds": 0.0, "task": t}
    spec = variant_spec(cfg, t["variant"], mode)
    data = C.load_combo(t["dataset"], t["target"])
    n = data["n"]
    splits = splits_of(spec, data, t["repeat"])
    rows = np.arange(n) if t["fold"] < 0 else np.asarray(splits[t["fold"]][0])
    seed = sel_seed(spec, t["repeat"], t["fold"])
    kw = sel_kw_of(spec)
    key = MaskCache.make_key({"dataset": t["dataset"], "target": t["target"]}, "ga", kw, spec["k"],
                             rows, seed, "pls")
    t0 = time.time()
    with threadpool_limits(limits=1):
        mask, info, arch = run_ga(data["X"][rows], data["y"][rows], data["groups"][rows], model="pls",
                                  pop=spec["pop"], ngen=spec["ngen"], seed=seed,
                                  representation=spec["representation"], k=spec["k"], clip=spec["clip"],
                                  candidate_cache=t["candidate_cache"])
    arch_path = dirs["out"] / "archive" / f"{run_id(t)}.npz"
    info = {**info, "variant": t["variant"], "archive": arch_path.relative_to(C.ROOT).as_posix()}
    save_archive(arch_path, arch, info)
    if t["candidate_cache"]:          # önbelleksiz kopya engine önbelleğine yazılmaz
        info_c = dict(info)
        info_c["selection_seconds"] = info["ga_seconds"]
        MaskCache(dirs["masks"]).put(key, mask, info_c, {"repeat": t["repeat"], "fold": t["fold"],
                                                         "arm": "all_data" if t["fold"] < 0 else "P3_nested",
                                                         "splits_signature": splits_signature(splits)})
    from src.m07_revision.ga import generation_stats, mask_sha1
    rec = {"task": t, "spec": spec, "seed": seed, "n_rows": int(len(rows)), "info": info,
           "mask_sha1": mask_sha1(mask), "mask": mask, "logbook": generation_stats(arch),
           "seconds": round(time.time() - t0, 2)}
    C.save_pickle(p, rec)
    return {"label": _label(t), "skipped": False, "seconds": rec["seconds"], "task": t}


# ---------------------------------------------------------------------------
# Aşama B — önbellekten kol değerlendirmesi
# ---------------------------------------------------------------------------
def _cached_only(*_a, **_k):
    raise RuntimeError("GA maskesi önbellekte yok — Aşama A tamamlanmadı")


def eval_path(out: Path, variant: str, ds: str, tg: str, r: int) -> Path:
    return out / "_checkpoints" / "eval" / f"{variant}__{ds}__{tg}__r{r:02d}.pkl"


def phase_b(cfg: dict, dirs: dict, mode: str, variants: list[str], log) -> int:
    done = 0
    for name in variants:
        spec = variant_spec(cfg, name, mode)
        for c in spec["combos"]:
            data = C.load_combo(c["dataset"], c["target"])
            for r in range(spec["repeats"]):
                p = eval_path(dirs["out"], name, c["dataset"], c["target"], r)
                if p.exists():
                    continue
                splits = splits_of(spec, data, r)
                cache = MaskCache(dirs["masks"])
                res = evaluate_arms(data["X"], data["y"], data["groups"], splits, "pls", _cached_only,
                                    arms=spec["arms"], sel_kw=sel_kw_of(spec), k=spec["k"], repeat=r,
                                    base_seed=spec["base_seed"], seed_mode=spec["seed_mode"],
                                    selector_name="ga", sample_ids=data["sample_ids"],
                                    group_ids=data["group_ids"], mask_cache=cache,
                                    cache_context={"dataset": c["dataset"], "target": c["target"]})
                C.save_pickle(p, {"variant": name, "dataset": c["dataset"], "target": c["target"],
                                  "repeat": r, "spec": spec, "splits_signature": splits_signature(splits),
                                  "summary": res.summary, "contrasts": res.contrasts, "folds": res.folds,
                                  "predictions": res.predictions, "meta": res.meta,
                                  "cache": {"hits": cache.hits, "misses": cache.misses}})
                done += 1
    log.info("Aşama B: %d (varyant, kombinasyon, tekrar) değerlendirildi", done)
    return done


# ---------------------------------------------------------------------------
# Toplama
# ---------------------------------------------------------------------------
def _load(glob_dir: Path) -> list[dict]:
    return [pickle.loads(f.read_bytes()) for f in sorted(glob_dir.glob("*.pkl"))]


def _ref_seed0() -> pd.Series | None:
    if not REF_CSV.exists():
        return None
    r = pd.read_csv(REF_CSV)
    r = r[(r.seed == 0) & (r.dataset == "ryckewaert") & (r.target == "flavonol") & (r.model == "pls")
          & (r.selector == "ga")]
    return None if r.empty else r.iloc[0]


def aggregate(out: Path, cfg: dict, mode: str) -> None:
    runs = _load(out / "_checkpoints" / "ga_runs")
    evals = _load(out / "_checkpoints" / "eval")
    if not runs:
        raise RuntimeError(f"{out}: GA koşusu yok")

    # --- ga_runs.csv + logbook + hesaplama bütçesi (R3-39) --------------------
    run_rows, lb = [], []
    for R in runs:
        t, i = R["task"], R["info"]
        rid = run_id(t)
        run_rows.append({"run_id": rid, "variant": t["variant"], "dataset": t["dataset"],
                         "target": t["target"], "repeat": t["repeat"], "fold": t["fold"],
                         "selection_scope": "all_data" if t["fold"] < 0 else "outer_train",
                         "candidate_cache": t["candidate_cache"], "seed": R["seed"], "n_rows": R["n_rows"],
                         "representation": i["representation"], "clip": i["clip"], "pop": i["pop"],
                         "ngen": i["ngen"], "inner_fitness": i["inner_fitness"],
                         "n_selected": i["n_selected"], "mask_sha1": R["mask_sha1"],
                         "n_evaluations_requested": i["n_evaluations_requested"],
                         "n_unique_evaluated": i["n_unique_evaluated"],
                         "n_evaluations_computed": i["n_evaluations_computed"],
                         "model_fits_computed": i["n_evaluations_computed"] * i["inner_folds"],
                         "model_fits_without_cache": i["n_evaluations_requested"] * i["inner_folds"],
                         "ga_seconds": i["ga_seconds"]})
        st = pd.DataFrame(R["logbook"])
        st.insert(0, "run_id", rid)
        lb.append(st)
    runs_df = pd.DataFrame(run_rows).sort_values(["variant", "dataset", "target", "repeat", "fold"])
    runs_df.to_csv(out / "ga_runs.csv", index=False)
    pd.concat(lb).to_csv(out / "logbook.csv.gz", index=False, compression="gzip")

    # --- önbellek değişmezliği ------------------------------------------------
    nc = [R for R in runs if not R["task"]["candidate_cache"]]
    if nc:
        rows = []
        for R in nc:
            twin = next((Q for Q in runs if Q["task"]["candidate_cache"] and
                         run_id(Q["task"]) == run_id({**R["task"], "candidate_cache": True})), None)
            if twin is None:
                continue
            rows.append({"run_id": run_id(twin["task"]), "same_mask": bool(np.array_equal(R["mask"], twin["mask"])),
                         "inner_fitness_cache_on": twin["info"]["inner_fitness"],
                         "inner_fitness_cache_off": R["info"]["inner_fitness"],
                         "evals_cache_on": twin["info"]["n_evaluations_computed"],
                         "evals_cache_off": R["info"]["n_evaluations_computed"],
                         "seconds_cache_on": twin["info"]["ga_seconds"],
                         "seconds_cache_off": R["info"]["ga_seconds"]})
        pd.DataFrame(rows).to_csv(out / "cache_invariance.csv", index=False)

    if not evals:
        return
    # --- E1 biçiminde uzun özet + fold tablosu + örnek düzeyi --------------------
    long_rows, fold_frames, sample_frames, ao_rows = [], [], [], []
    inner = {(R["task"]["variant"], R["task"]["dataset"], R["task"]["target"], R["task"]["repeat"],
              R["task"]["fold"]): R["info"]["inner_fitness"] for R in runs if R["task"]["candidate_cache"]}
    for E in evals:
        base = {"variant": E["variant"], "dataset": E["dataset"], "target": E["target"], "selector": "ga",
                "family": "search", "model": "pls", "repeat": E["repeat"]}
        for _, s in E["summary"].iterrows():
            for q in ("r2_foldmean", "r2_pooled", "rmse_foldmean", "rmse_pooled", "mae_foldmean",
                      "mae_pooled", "n_selected_mean", "n_components_mean"):
                long_rows.append({**base, "quantity": f"{s['arm']}.{q}", "value": s[q]})
        for q, v in E["contrasts"].items():
            long_rows.append({**base, "quantity": q, "value": v})
        fd = E["folds"].drop(columns=["mask"]).assign(**base)
        fold_frames.append(fd)
        from src.m07_revision.engine import ArmsResult
        sample_frames.append(folds_long(ArmsResult(E["summary"], E["contrasts"], E["folds"],
                                                   E["predictions"])).assign(**base))
        # R3-48 adaptif aşırı uyum: iç fitness − dış-test R²
        f = E["folds"]
        app = E["summary"].set_index("arm")
        if "P2_apparent" in app.index:
            fi = inner.get((E["variant"], E["dataset"], E["target"], E["repeat"], -1))
            ao_rows.append({**base, "arm": "P2_apparent", "fold": -1, "inner_fitness": fi,
                            "outer_r2": app.at["P2_apparent", "r2_foldmean"],
                            "inner_minus_outer": None if fi is None else fi - app.at["P2_apparent", "r2_foldmean"]})
        for _, row in f[f.arm == "P3_nested"].iterrows():
            fi = inner.get((E["variant"], E["dataset"], E["target"], E["repeat"], int(row.fold)))
            ao_rows.append({**base, "arm": "P3_nested", "fold": int(row.fold), "inner_fitness": fi,
                            "outer_r2": row.r2, "inner_minus_outer": None if fi is None else fi - row.r2})
    long = pd.DataFrame(long_rows)
    long.to_csv(out / "summary_by_combo.csv", index=False)
    pd.concat(fold_frames).to_csv(out / "fold_metrics.csv.gz", index=False, compression="gzip")
    sd = out / "samples"
    sd.mkdir(parents=True, exist_ok=True)
    pd.concat(sample_frames).to_csv(sd / "folds.csv.gz", index=False, compression="gzip")
    pd.DataFrame(ao_rows).to_csv(out / "adaptive_overfitting.csv", index=False)

    # --- varyant özeti --------------------------------------------------------
    wide = long.pivot_table(index=["variant", "dataset", "target", "repeat"], columns="quantity",
                            values="value").reset_index()
    out_rows = []
    for (v, ds, tg), x in wide.groupby(["variant", "dataset", "target"]):
        d = C.describe(x["apparent_nested_difference_r2_foldmean"])
        row = {"variant": v, "dataset": ds, "target": tg, "n_repeats": d["n"],
               "apparent_r2_foldmean_mean": x["P2_apparent.r2_foldmean"].mean(),
               "nested_r2_foldmean_mean": x["P3_nested.r2_foldmean"].mean(),
               "delta_r2_foldmean_mean": d["mean"], "delta_r2_foldmean_median": d["median"],
               "delta_r2_foldmean_p2_5": d["p2_5"], "delta_r2_foldmean_p97_5": d["p97_5"],
               "delta_r2_pooled_mean": x["apparent_nested_difference_r2_pooled"].mean(),
               "n_selected_apparent_mean": x["P2_apparent.n_selected_mean"].mean(),
               "n_selected_nested_mean": x["P3_nested.n_selected_mean"].mean()}
        if "fold_partitioning_contrast_r2_foldmean" in x:
            row["fold_partitioning_contrast_r2_foldmean_mean"] = x["fold_partitioning_contrast_r2_foldmean"].mean()
            row["P1_random_r2_foldmean_mean"] = x["P1_random.r2_foldmean"].mean()
        out_rows.append(row)
    pd.DataFrame(out_rows).to_csv(out / "summary_by_variant.csv", index=False)

    # --- G0 ↔ outputs3 seed 0 -------------------------------------------------
    g0 = [E for E in evals if E["variant"] == "G0"]
    ref = _ref_seed0()
    if g0 and ref is not None:
        E = g0[0]
        s = E["summary"].set_index("arm")
        paper_budget = E["spec"]["pop"] == 150 and E["spec"]["ngen"] == 150
        rows = []
        for q, new, old in (("apparent_r2_foldmean", s.at["P2_apparent", "r2_foldmean"], ref.apparent_r2),
                            ("nested_r2_foldmean", s.at["P3_nested", "r2_foldmean"], ref.nested_r2),
                            ("apparent_nested_difference_r2_foldmean",
                             E["contrasts"]["apparent_nested_difference_r2_foldmean"], ref.bias),
                            ("apparent_r2_pooled", s.at["P2_apparent", "r2_pooled"], ref.apparent_r2_pooled)):
            rows.append({"quantity": q, "rev1_G0": float(new), "outputs3_seed0": float(old),
                         "absdiff": abs(float(new) - float(old)),
                         "match_3dp": round(float(new), 3) == round(float(old), 3),
                         "comparable": paper_budget, "pop": E["spec"]["pop"], "ngen": E["spec"]["ngen"]})
        pd.DataFrame(rows).to_csv(out / "g0_reproduction.csv", index=False)


# ---------------------------------------------------------------------------
# Zaman tahmini (timing: G0 tam bütçe ölçülür → G1/G2/G3)
# ---------------------------------------------------------------------------
def _eval_ms(dataset: str, target: str, n_eval: int = 25) -> float:
    """Kombinasyon başına tek fitness değerlendirmesinin süresi (ms; %50 yoğunluklu rastgele maske)."""
    from src.m07_revision.ga import Fitness
    d = C.load_combo(dataset, target)
    fit = Fitness(d["X"], d["y"], d["groups"], "pls", 0, use_cache=False)
    rng = np.random.default_rng(0)
    ms = [rng.random(d["X"].shape[1]) < 0.5 for _ in range(n_eval)]
    t0 = time.time()
    for m in ms:
        fit(m)
    return (time.time() - t0) / n_eval * 1000


def estimate(results: list[dict], cfg: dict, n_jobs: int, dirs: dict) -> dict:
    runs = [pickle.loads(f.read_bytes()) for f in (dirs["out"] / "_checkpoints" / "ga_runs").glob("*.pkl")]
    on = [R for R in runs if R["task"]["candidate_cache"]]
    off = [R for R in runs if not R["task"]["candidate_cache"]]
    sec_on = float(np.mean([R["info"]["ga_seconds"] for R in on]))
    ms = {f"{c['dataset']}/{c['target']}": _eval_ms(c["dataset"], c["target"]) for c in cfg["common"]["combos"]}
    ref = ms["ryckewaert/flavonol"]
    per_variant = {}
    durations = []
    for name in ("G1", "G2", "G3"):
        spec = variant_spec(cfg, name, "full")
        d = [sec_on * ms[f"{t['dataset']}/{t['target']}"] / ref
             for t in build_tasks(cfg, "full", [name])]
        durations += d
        per_variant[name] = {"n_ga_runs": len(d), "cpu_hours": round(sum(d) / 3600, 2),
                             "wall_hours_alone_LPT": round(C.lpt_makespan(d, n_jobs) / 3600, 2),
                             "repeats": spec["repeats"], "combos": len(spec["combos"])}
    g1 = [sec_on * ms[f"{t['dataset']}/{t['target']}"] / ref for t in build_tasks(cfg, "full", ["G0", "G1"])]
    return {"n_jobs": n_jobs, "measured_G0_runs": len(on), "G0_run_seconds_mean_cache_on": round(sec_on, 1),
            "G0_all_data_seconds_cache_off": off[0]["info"]["ga_seconds"] if off else None,
            "G0_all_data_evals_cache_on_vs_off": None if not off else
            [next(R["info"]["n_evaluations_computed"] for R in on if R["task"]["fold"] < 0),
             off[0]["info"]["n_evaluations_computed"]],
            "eval_ms_by_combo": {k: round(v, 2) for k, v in ms.items()},
            "per_variant": per_variant,
            "G0+G1_wall_hours_LPT": round(C.lpt_makespan(g1, n_jobs) / 3600, 2),
            "G2+G3_wall_hours_LPT": round(C.lpt_makespan(
                [x for name in ("G2", "G3") for x in
                 [sec_on * ms[f"{t['dataset']}/{t['target']}"] / ref for t in build_tasks(cfg, "full", [name])]],
                n_jobs) / 3600, 2),
            "note": (f"G0 koşuları {len(runs)} süreç paralelken ölçüldü; tam koşuda {n_jobs} süreç (8 fiziksel "
                     "çekirdek) → süreç başına yavaşlama beklenir (E1'de LassoCV ≈ ×1.9).")}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _dirs(mode: str, variants: list[str]) -> dict:
    d = C.experiment_dirs(EXP, mode)
    tag = "-".join(variants)
    d["done"] = d["done"].with_name(f"{EXP}.{tag}.DONE.json")
    return d


def check(cfg: dict, variants: list[str]) -> int:
    dirs = _dirs("full", variants)
    tasks = build_tasks(cfg, "full", variants)
    have = sum(ga_run_path(dirs["out"], t).exists() for t in tasks)
    n_eval = 0
    want_eval = 0
    for name in variants:
        spec = variant_spec(cfg, name, "full")
        for c in spec["combos"]:
            for r in range(spec["repeats"]):
                want_eval += 1
                n_eval += eval_path(dirs["out"], name, c["dataset"], c["target"], r).exists()
    print(f"{EXP} {variants}: GA koşusu {have}/{len(tasks)} | değerlendirme {n_eval}/{want_eval} | "
          f"DONE: {dirs['done'].exists()}")
    ok = have == len(tasks) and n_eval == want_eval and dirs["done"].exists()
    g0 = dirs["out"] / "g0_reproduction.csv"
    if "G0" in variants and g0.exists():
        g = pd.read_csv(g0)
        print("\nG0 ↔ outputs3 seed 0:")
        print(g.round(4).to_string(index=False))
        ok &= bool(g.match_3dp.all())
    sv = dirs["out"] / "summary_by_variant.csv"
    if sv.exists():
        s = pd.read_csv(sv)
        s = s[s.variant.isin(variants)]
        print("\nVaryant özeti (Δ = apparent − nested, fold-mean R²):")
        print(s[["variant", "dataset", "target", "n_repeats", "apparent_r2_foldmean_mean",
                 "nested_r2_foldmean_mean", "delta_r2_foldmean_median", "delta_r2_foldmean_p2_5",
                 "delta_r2_foldmean_p97_5"]].round(3).to_string(index=False))
    print("\nSONUÇ:", "TAMAM" if ok else "EKSİK / SORUN VAR")
    return 0 if ok else 1


def main() -> None:
    ap = C.base_argparser("E6 — GA")
    ap.add_argument("--variants", type=str, default="G0,G1,G2,G3",
                    help="koşulacak varyantlar (virgülle; G0,G1,G2,G3)")
    args = ap.parse_args()
    cfg = C.load_config()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = set(variants) - set(VARIANT_ORDER)
    if bad:
        ap.error(f"bilinmeyen varyant(lar) {sorted(bad)}; seçenekler {VARIANT_ORDER}")
    if args.check:
        sys.exit(check(cfg, variants))
    mode = "quick" if args.quick else "timing" if args.timing else "full"
    if mode == "timing":
        variants = ["G0"]                          # plan: önce zaman ölçümü (G0 tam bütçe)
    dirs = _dirs(mode, variants)
    if (mode == "timing" or (mode == "quick" and args.fresh)) and not args.aggregate_only:
        # Yalnız _timing/ ya da _quick/ altı (tam koşunun önbelleğine dokunulmaz)
        for d in (dirs["out"], dirs["masks"]):
            if d.exists():
                shutil.rmtree(d)
    log = C.setup_logging(dirs["log"])
    if args.aggregate_only:
        phase_b(cfg, dirs, mode, variants, log)
        aggregate(dirs["out"], cfg, mode)
        log.info("E6 tabloları yeniden üretildi → %s", dirs["out"])
        return
    tasks = build_tasks(cfg, mode, variants)
    n_jobs = args.n_jobs or C.default_n_jobs()
    import functools

    from src.m07_revision.e6_ga import run_task as entry

    def _agg():
        phase_b(cfg, dirs, mode, variants, log)
        aggregate(dirs["out"], cfg, mode)

    C.run_experiment(
        name=EXP, mode=mode, tasks=tasks,
        entry=functools.partial(entry, cfg=cfg, dirs=dirs, mode=mode),
        aggregate=_agg, dirs=dirs, n_jobs=n_jobs,
        payload={"experiment": EXP, "variants": variants,
                 "specs": {v: variant_spec(cfg, v, mode) for v in variants},
                 "ga_core": "src/m07_revision/ga.py (= ga_feature_selection._run_ga döngüsü; seri, aday önbellekli)",
                 "seeds": {"G0": "fixed 0", "G1-G3": "derive_seed(base_seed, repeat, fold, 'ga'); fold=-1 tüm veri"},
                 "split": "G0: GroupKFold(5) shuffle=False; G1-G3: GroupKFold(5, shuffle=True, random_state=repeat) = E1",
                 "mask_cache": "outputs_rev1/_checkpoints/masks (selector='ga', sel_kw variant'lı)",
                 "config_blocks": {"common": cfg["common"], EXP: cfg[EXP]}},
        log=log, label=_label,
        estimate=lambda res: estimate(res, cfg, n_jobs, dirs))


if __name__ == "__main__":
    main()
