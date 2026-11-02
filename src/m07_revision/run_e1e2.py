"""E1 + E2 ortak koşucusu — aynı bölmeler, aynı maske önbelleği, tek paralel kuyruk.

Görev birimi (repeat, dataset, target, selector): bir işçi süreçte önce E1
(tüm modeller), sonra E2 (PLS) koşar; seçici maskeleri disk önbelleğinden
(``<root>/_checkpoints/masks``) paylaşılır → her maske bir kez hesaplanır.
Görevler ``joblib`` (loky) ile paralel; ``n_jobs = cpu_count − 2``; seçicilerin
içi ``n_jobs=1``; BLAS iş parçacıkları 1 (paket ``__init__`` + threadpoolctl).

Kipler
------
(varsayılan)    tam koşu: R = config (E1/E2: 20), 3 kombinasyon, PLS + LightGBM (E1)
--quick         duman testi: R=2, ryck/flavonol, PLS  → outputs_rev1/_quick/
                (checkpoint'ten devam eder; ``--fresh`` ile sıfırdan)
--timing        zaman ölçümü: R=1 (repeat 0), tam kombinasyon seti, taze önbellek
                → outputs_rev1/_timing/ ; R=20 için LPT süre tahmini yazar
--check         tam koşu bitti mi? görev sayıları, E1/E2 bölme eşliği, özet
--aggregate-only yalnız checkpoint'lerden tabloları yeniden üret

Devam etme: her görev bitince kendi checkpoint'ini yazar; aynı komut yeniden
çalıştırılınca bitmiş görevler ATLANIR, maskeler önbellekten okunur.
Bitiş: log'da "ALL DONE" satırı ve ``<root>/logs/run_e1e2.DONE.json``.

Çalıştırma (proje kökünden)::

    .venv\\Scripts\\python.exe -m src.m07_revision.run_e1e2 --quick
    .venv\\Scripts\\python.exe -m src.m07_revision.run_e1e2 --timing
    .venv\\Scripts\\python.exe -m src.m07_revision.run_e1e2
    .venv\\Scripts\\python.exe -m src.m07_revision.run_e1e2 --check
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.m07_revision import common as C
from src.m07_revision import e1_repeated as E1
from src.m07_revision import e2_mechanism as E2

ROOT = C.ROOT
COST_ORDER = {"lasso": 0, "mutual_info": 1, "rfe": 2}   # pahalı görevler önce (yük dengesi)


def _dirs(mode: str) -> dict:
    base = ROOT / "outputs_rev1" / {"full": "", "quick": "_quick", "timing": "_timing"}[mode]
    return {"root": base, "E1": base / "E1_repeated", "E2": base / "E2_mechanism",
            "masks": base / "_checkpoints" / "masks", "log": base / "logs" / "run_e1e2.log",
            "done": base / "logs" / "run_e1e2.DONE.json"}


def build_tasks(cfg: dict, mode: str, repeats: int | None, skip: tuple = ()) -> list[dict]:
    combos = cfg["common"]["combos"]
    selectors = cfg["E1_repeated"]["selectors"]
    models = cfg["E1_repeated"]["models"]
    R = cfg["E1_repeated"]["repeats"]
    if mode == "quick":
        combos, models, R = [{"dataset": "ryckewaert", "target": "flavonol"}], ["pls"], 2
    elif mode == "timing":
        R = 1
    if repeats is not None:
        R = repeats
    tasks = [{"repeat": r, "dataset": c["dataset"], "target": c["target"], "selector": s,
              "models": list(models)}
             for r in range(R) for c in combos for s in selectors]
    # --skip "dataset/selector" (yalnız zaman ölçümünde maliyeti ayrıca ölçülen görevler için)
    tasks = [t for t in tasks if f"{t['dataset']}/{t['selector']}" not in set(skip)]
    tasks.sort(key=lambda t: (COST_ORDER.get(t["selector"], 9), t["dataset"] != "ryckewaert",
                              t["repeat"]))
    return tasks


def _label(t: dict) -> str:
    return f"r{t['repeat']:02d} {t['dataset']}/{t['target']}/{t['selector']}"


def _task_entry(t: dict, cfg: dict, dirs: dict, experiments: tuple) -> dict:
    """İşçi süreçte: E1 sonra E2 (aynı maske önbelleği). threadpool 1 iş parçacığı."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        out = {"label": _label(t), "task": t, "skipped": True, "seconds": 0.0}
        t0 = time.time()
        for name, mod in (("E1", E1), ("E2", E2)):
            if name in experiments:
                r = mod.run_task(t, cfg, dirs[name], dirs["masks"])
                out[f"{name}_seconds"] = r["seconds"]
                out["skipped"] &= r["skipped"]
        out["seconds"] = round(time.time() - t0, 2)
        return out


def _start_run_config(dirs: dict, exp: str, cfg: dict, tasks: list, mode: str, n_jobs: int,
                      git: dict, log) -> None:
    out = dirs[exp]
    rc = out / "run_config.json"
    entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), **git,
             "command": " ".join([Path(sys.executable).name, *sys.argv])}
    if rc.exists():
        body = json.loads(rc.read_text(encoding="utf-8"))
        if body.get("git_commit") != git["git_commit"]:
            log.warning("UYARI: %s ilk koşu commit'i %s, şimdiki %s — kod koşu ortasında değişti!",
                        exp, body.get("git_commit"), git["git_commit"])
        body.setdefault("resumes", []).append(entry)
        rc.write_text(json.dumps(body, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return
    block = "E1_repeated" if exp == "E1" else "E2_mechanism"
    C.write_run_config(out, {
        "experiment": block, "mode": mode, "n_jobs": n_jobs,
        "repeats": sorted({t["repeat"] for t in tasks}),
        "combos": sorted({(t["dataset"], t["target"]) for t in tasks}),
        "selectors": sorted({t["selector"] for t in tasks}),
        "models": tasks[0]["models"] if exp == "E1" else ["pls"],
        "seeds": {"base_seed": cfg["common"]["base_seed"], "seed_mode": "derived",
                  "rule": "derive_seed(base_seed, repeat, fold, selector); fold=-1 all-data; "
                          "bootstrap: derive_seed(base_seed, repeat, -1, 'bootstrap')"},
        "split": "GroupKFold(5, shuffle=True, random_state=repeat) — shared by E1/E2/E6-G1",
        "config_blocks": {"common": cfg["common"], block: cfg[block]},
        "mask_cache": str(dirs["masks"].relative_to(ROOT).as_posix()),
    }, git)


def aggregate(dirs: dict, cfg: dict, experiments: tuple, log) -> None:
    if "E1" in experiments:
        r = E1.aggregate(dirs["E1"], cfg)
        log.info("E1 tabloları yazıldı (%d görev) → %s", r["n_tasks"], dirs["E1"])
    if "E2" in experiments:
        r = E2.aggregate(dirs["E2"], cfg)
        log.info("E2 tabloları yazıldı (%d görev) → %s", r["n_tasks"], dirs["E2"])


def check(cfg: dict, experiments: tuple) -> int:
    """Tam koşu bitti mi? 0 = tamam, 1 = eksik/sorun."""
    dirs = _dirs("full")
    tasks = build_tasks(cfg, "full", None)
    ok = True
    print(f"Beklenen görev: {len(tasks)}")
    for exp, mod in (("E1", E1), ("E2", E2)):
        if exp not in experiments:
            continue
        have = sum(mod.task_path(dirs[exp], t).exists() for t in tasks)
        print(f"  {exp}: {have}/{len(tasks)} görev checkpoint'i")
        ok &= have == len(tasks)
    print("  DONE işaretçisi:", dirs["done"].exists())
    ok &= dirs["done"].exists()
    s1, s2 = dirs["E1"] / "splits_signatures.csv", dirs["E2"] / "decomposition_by_repeat.csv"
    if s1.exists():
        sig = pd.read_csv(s1)
        n_sig = sig.groupby(["dataset", "repeat"]).splits_signature.nunique().max()
        print("  Bölme imzası (dataset, repeat) başına tekil:", n_sig == 1)
        ok &= n_sig == 1
    if s1.exists() and "E2" in experiments:
        import pickle
        bad = 0
        for t in tasks[:: max(1, len(tasks) // 15)]:
            p1, p2 = E1.task_path(dirs["E1"], t), E2.task_path(dirs["E2"], t)
            if p1.exists() and p2.exists():
                bad += pickle.loads(p1.read_bytes())["splits_signature"] != \
                    pickle.loads(p2.read_bytes())["splits_signature"]
        print("  E1/E2 aynı bölme (örneklem):", bad == 0)
        ok &= bad == 0
    bf = dirs["E1"] / "by_family.csv"
    if bf.exists():
        d = pd.read_csv(bf)
        print("\nE1 apparent–nested difference (fold-mean R²), tekrarlar arası:")
        print(d[["dataset", "target", "family", "model", "n", "median", "p2_5", "p97_5"]]
              .round(3).to_string(index=False))
    dc = dirs["E2"] / "decomposition.csv"
    if dc.exists():
        d = pd.read_csv(dc)
        d = d[d.quantity.isin(["total_apparent_nested_difference", "sample_size_component",
                               "equal_size_leakage_component"])]
        print("\nE2 ayrıştırma (medyan):")
        print(d.pivot_table(index=["dataset", "target", "selector"], columns="quantity",
                            values="median").round(3).to_string())
    print("\nSONUÇ:", "TAMAM" if ok else "EKSİK / SORUN VAR")
    return 0 if ok else 1


def main(experiments: tuple = ("E1", "E2")) -> None:
    ap = argparse.ArgumentParser(description="E1 + E2 ortak koşucusu")
    mx = ap.add_mutually_exclusive_group()
    mx.add_argument("--quick", action="store_true")
    mx.add_argument("--timing", action="store_true")
    mx.add_argument("--check", action="store_true")
    mx.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--repeats", type=int, default=None, help="tekrar sayısını override et")
    ap.add_argument("--n-jobs", type=int, default=None, help="varsayılan cpu_count-2")
    ap.add_argument("--fresh", action="store_true",
                    help="--quick: önceki duman testi çıktılarını silip sıfırdan başla")
    ap.add_argument("--skip", type=str, default="",
                    help="atlanacak 'dataset/selector' listesi (virgülle); yalnız --timing ile")
    args = ap.parse_args()
    cfg = C.load_config()
    if args.check:
        sys.exit(check(cfg, experiments))

    mode = "quick" if args.quick else "timing" if args.timing else "full"
    dirs = _dirs(mode)
    if ((mode == "timing" or (mode == "quick" and args.fresh))
            and not args.aggregate_only and dirs["root"].exists()):
        shutil.rmtree(dirs["root"])          # taze önbellek: gerçek maliyeti ölç / temiz duman testi
    log = C.setup_logging(dirs["log"])
    skip = tuple(x.strip() for x in args.skip.split(",") if x.strip())
    if skip and mode != "timing":
        ap.error("--skip yalnız --timing ile kullanılabilir (tam koşu eksik kalmasın)")
    tasks = build_tasks(cfg, mode, args.repeats, skip)
    n_jobs = args.n_jobs or C.default_n_jobs()
    if args.aggregate_only:
        aggregate(dirs, cfg, experiments, log)
        return

    git = C.git_state()      # koşu BAŞINDA (kendi çıktıları "kirli" sayılmasın)
    log.info("=" * 70)
    log.info("run_e1e2 başladı | mod=%s | deneyler=%s | commit=%s | kirli dosya=%d",
             mode, ",".join(experiments), git["git_commit"][:7], len(git["git_dirty_files"]))
    if git["git_dirty_files"] and mode == "full":
        log.warning("UYARI: çalışma ağacı temiz değil: %s", git["git_dirty_files"][:5])
    for exp in experiments:
        _start_run_config(dirs, exp, cfg, tasks, mode, n_jobs, git, log)

    if dirs["done"].exists():
        dirs["done"].unlink()                # eski bitiş işaretçisi (devam eden koşu)
    # İşçiye modül yoluyla içe aktarılan fonksiyon gönderilir (``-m`` ile __main__ olan
    # kopya değil) → loky güvenle pickle'lar.
    import functools

    from src.m07_revision.run_e1e2 import _task_entry as entry
    t0 = time.time()
    res = C.run_tasks(tasks, functools.partial(entry, cfg=cfg, dirs=dirs, experiments=experiments),
                      n_jobs, log, _label)
    aggregate(dirs, cfg, experiments, log)
    wall = time.time() - t0

    if mode == "timing":
        dur = [r["seconds"] for r in res if not r["skipped"]]
        R_full = cfg["E1_repeated"]["repeats"]
        est = C.lpt_makespan(dur * R_full, n_jobs)
        per = pd.DataFrame([{"task": r["label"], "seconds": r["seconds"],
                             "E1_seconds": r.get("E1_seconds"), "E2_seconds": r.get("E2_seconds")}
                            for r in res]).sort_values("seconds", ascending=False)
        timing = {"wall_seconds_R1": round(wall, 1), "n_jobs": n_jobs, "R_full": R_full,
                  "cpu_seconds_R1": round(sum(dur), 1),
                  "estimate_R_full_seconds_LPT": round(est, 1),
                  "estimate_R_full_hms": str(timedelta(seconds=int(est))),
                  "note": ("R=1 görev süreleri × R; LPT zamanlama. Muhafazakâr: tüm-veri maskeleri "
                           "tam koşuda tekrarlar arası önbellekten gelir, potato LOGO yalnız r0.")}
        (dirs["root"] / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        per.to_csv(dirs["root"] / "timing_tasks.csv", index=False)
        log.info("ZAMAN: R=1 duvar %.1fs, CPU %.1fs → R=%d tahmini %s (LPT, n_jobs=%d)",
                 wall, sum(dur), R_full, timing["estimate_R_full_hms"], n_jobs)

    dirs["done"].write_text(json.dumps({"finished": datetime.now().isoformat(timespec="seconds"),
                                        "mode": mode, "n_tasks": len(tasks),
                                        "wall_seconds": round(wall, 1)}, indent=2), encoding="utf-8")
    log.info("ALL DONE | mod=%s | %d görev | duvar %s", mode, len(tasks),
             str(timedelta(seconds=int(wall))))


if __name__ == "__main__":
    main()
