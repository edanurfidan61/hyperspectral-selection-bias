"""E1–E7 ortak altyapısı: config, veri, bölmeler, görev koşucusu, bootstrap, kayıt.

* :func:`load_config`      — ``config/revision.yaml`` (önceden belirlenmiş grid).
* :func:`load_combo`       — (dataset, target) → X, y, groups, sample_ids (NaN'siz).
* :func:`outer_splits`     — E1/E2 (ve E6-G1) için ORTAK dış bölme: aynı
  ``(dataset, repeat)`` → aynı ``GroupKFold(shuffle=True, random_state=repeat)``.
* :func:`run_tasks`        — joblib/loky ile paralel görevler; her görev kendi
  checkpoint dosyasını yazar, var olanlar atlanır (yarıda kalırsa aynı komut devam
  eder). Ebeveyn süreç her tamamlanan görevi ETA ile log'a yazar.
* :func:`group_bootstrap_paired` — eşli grup bootstrap (aynı indeksler iki kolda).
* :func:`write_run_config` — commit SHA (koşu BAŞINDA), komut, sürümler, seed'ler.

BLAS iş parçacıkları paket ``__init__``'inde 1'e sabitlenir.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.core import paths
from src.core.splits import make_outer_splits, splits_signature

ROOT = paths.ROOT
CONFIG_PATH = ROOT / "config" / "revision.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: Path = CONFIG_PATH) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def default_n_jobs() -> int:
    """Kullanıcı kararı (E1/E2): ``cpu_count − 2`` (en az 1)."""
    return max(1, (os.cpu_count() or 2) - 2)


def selector_kwargs(cfg: dict, name: str, override: dict | None = None) -> dict:
    """``common.selectors`` varsayılanları + deney override'ı → seçici argümanları."""
    kw = dict(cfg["common"]["selectors"].get(name, {}) or {})
    kw.update(override or {})
    # Seçici fonksiyonu tanımadığı config anahtarlarını almasın
    for drop in ("model_fitness", "min_features", "clip_fitness_at_zero"):
        kw.pop(drop, None)
    return kw


# ---------------------------------------------------------------------------
# Veri
# ---------------------------------------------------------------------------
_COMBO_CACHE: dict = {}


def load_combo(dataset: str, target: str, group_level: str | None = None) -> dict:
    """(dataset, target) için X, y, groups, sample_ids, group_ids (y sonlu satırlar).

    ryckewaert: groups = leaf (varsayılan) ya da ``group_level="plot"``.
    deep_patato: groups = Field; sample_ids = CHESS_FOLDS.CSV ``Sample_Index``
    (npz sırası CSV ile aynıysa — E0'da doğrulandı; değilse satır indeksi).
    """
    key = (dataset, target, group_level)
    if key in _COMBO_CACHE:
        return _COMBO_CACHE[key]
    from src.m01_io import dataset_registry as reg

    if dataset == "ryckewaert":
        ds = reg.load_ryckewaert(group_key=group_level or "leaf")
        sample_ids = np.asarray(reg.load_ryckewaert(group_key="leaf").groups).astype(str)
    elif dataset == "deep_patato":
        ds = reg.load_deep_patato()
        sample_ids = np.arange(ds.X.shape[0]).astype(str)
        chess = ROOT / "data" / "deep_patato" / "measurements" / "CHESS_FOLDS.CSV"
        if chess.exists():
            c = pd.read_csv(chess)
            if (len(c) == ds.X.shape[0]
                    and np.array_equal(c["Field"].to_numpy(int), np.asarray(ds.groups).astype(int))):
                sample_ids = c["Sample_Index"].to_numpy(int).astype(str)
    else:
        raise ValueError(f"Bilinmeyen dataset {dataset!r}")
    y = np.asarray(ds.target(target), dtype=float)
    fin = np.isfinite(y)
    g = np.asarray(ds.groups)[fin]
    out = {"dataset": dataset, "target": target, "X": np.asarray(ds.X, dtype=float)[fin],
           "y": y[fin], "groups": g, "group_ids": g.astype(str),
           "sample_ids": sample_ids[fin], "n": int(fin.sum()),
           "feature_names": list(ds.feature_names)}
    _COMBO_CACHE[key] = out
    return out


def outer_splits(data: dict, repeat: int, n_splits: int = 5, scheme: str = "group",
                 shuffle: bool = True) -> list:
    """E1/E2/E6-G1 ORTAK dış bölme (aynı random_state = repeat)."""
    return make_outer_splits(data["n"], data["groups"], scheme, n_splits, repeat, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Eşli grup bootstrap (E1 belirsizlik (b))
# ---------------------------------------------------------------------------
def _weighted_r2(y: np.ndarray, p: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Satır ağırlıklı R² (W: B×n tekrar sayıları) — grupları birleştirmeye eşdeğer."""
    sw = W.sum(axis=1)
    ybar = (W * y).sum(axis=1) / sw
    ss_res = (W * (y - p) ** 2).sum(axis=1)
    ss_tot = (W * (y[None, :] - ybar[:, None]) ** 2).sum(axis=1)
    return 1.0 - ss_res / ss_tot


def group_bootstrap_paired(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, groups: np.ndarray,
                           B: int, seed: int, ci: float = 0.95) -> dict:
    """Δ = metrik(a) − metrik(b) için eşli grup bootstrap CI.

    Gruplar yerine koyarak yeniden örneklenir; AYNI örnekleme iki kolda da
    kullanılır (eşli). R² farkı (a − b) ve RMSE farkı (b − a; pozitif = a daha
    iyimser). Koşullu: fit edilmiş modeller/maskeler sabit.
    """
    y, p_a, p_b = (np.asarray(v, dtype=float) for v in (y, p_a, p_b))
    _, ginv = np.unique(np.asarray(groups), return_inverse=True)
    n_g = ginv.max() + 1
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, n_g, size=(B, n_g))
    counts = np.zeros((B, n_g))
    np.add.at(counts, (np.repeat(np.arange(B), n_g), draws.ravel()), 1.0)
    W = counts[:, ginv]                                   # B × n satır ağırlıkları
    d_r2 = _weighted_r2(y, p_a, W) - _weighted_r2(y, p_b, W)
    rm = lambda p: np.sqrt((W * (y - p) ** 2).sum(axis=1) / W.sum(axis=1))  # noqa: E731
    d_rmse = rm(p_b) - rm(p_a)
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2

    def r2(p):
        return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    def rmse(p):
        return float(np.sqrt(((y - p) ** 2).mean()))

    return {"B": int(B), "n_groups": int(n_g), "seed": int(seed),
            "delta_r2_pooled": float(r2(p_a) - r2(p_b)),
            "delta_r2_lo": float(np.quantile(d_r2, lo_q)), "delta_r2_hi": float(np.quantile(d_r2, hi_q)),
            "delta_r2_se": float(np.std(d_r2, ddof=1)),
            "delta_rmse_pooled": rmse(p_b) - rmse(p_a),
            "delta_rmse_lo": float(np.quantile(d_rmse, lo_q)),
            "delta_rmse_hi": float(np.quantile(d_rmse, hi_q))}


# ---------------------------------------------------------------------------
# Özet istatistik
# ---------------------------------------------------------------------------
def describe(values: Iterable[float]) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "sd": np.nan,
                "p2_5": np.nan, "p97_5": np.nan, "min": np.nan, "max": np.nan}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "sd": float(v.std(ddof=1)) if v.size > 1 else np.nan,
            "p2_5": float(np.quantile(v, 0.025)), "p97_5": float(np.quantile(v, 0.975)),
            "min": float(v.min()), "max": float(v.max())}


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else np.nan


def bits_to_mask(s: str) -> np.ndarray:
    return np.frombuffer(s.encode(), dtype=np.uint8) == ord("1")


# ---------------------------------------------------------------------------
# Log + run_config
# ---------------------------------------------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("rev1")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    # Windows konsolu (cp1252) "→", "ş" gibi karakterlerde UnicodeEncodeError verir;
    # log dosyası zaten UTF-8 — konsolu da UTF-8'e al, basılamayanı değiştir.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(fh)
    lg.addHandler(sh)
    lg.propagate = False
    return lg


def git_state() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"git_commit": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_files": git("status", "--porcelain").splitlines()}


def write_run_config(out_dir: Path, payload: dict, git: dict) -> None:
    pk = {}
    for p in ("numpy", "scipy", "scikit-learn", "pandas", "lightgbm", "joblib", "threadpoolctl"):
        try:
            pk[p] = importlib_metadata.version(p)
        except importlib_metadata.PackageNotFoundError:
            pk[p] = None
    out_dir.mkdir(parents=True, exist_ok=True)
    body = {"timestamp": datetime.now().isoformat(timespec="seconds"), **git,
            "command": " ".join([Path(sys.executable).name, *sys.argv]),
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "packages": pk,
            "thread_env": {v: os.environ.get(v) for v in
                           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            **payload}
    (out_dir / "run_config.json").write_text(json.dumps(body, indent=2, ensure_ascii=False,
                                                        default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Görev koşucusu (checkpoint + paralel + ETA)
# ---------------------------------------------------------------------------
def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_bytes(pickle.dumps(obj))
    tmp.replace(path)


def run_tasks(tasks: list[dict], fn: Callable[[dict], dict], n_jobs: int,
              log: logging.Logger, label: Callable[[dict], str]) -> list[dict]:
    """Görevleri paralel koş. ``fn(task)`` kendi checkpoint'ini yazar ve
    ``{"seconds": ..., "skipped": bool}`` döndürür. Ebeveyn tamamlananları ETA ile loglar.
    """
    from joblib import Parallel, delayed

    total = len(tasks)
    log.info("Görev sayısı: %d | n_jobs=%d", total, n_jobs)
    t0 = time.time()
    done, results = 0, []
    if total == 0:
        return results
    gen = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered",
                   batch_size=1)(delayed(fn)(t) for t in tasks)
    for res in gen:
        done += 1
        results.append(res)
        el = time.time() - t0
        eta = el / done * (total - done)
        log.info("[%d/%d] %s %s %.1fs | geçen %s | kalan ~%s", done, total,
                 "ATLANDI(checkpoint)" if res.get("skipped") else "bitti",
                 res.get("label", "?"), res.get("seconds", 0.0),
                 str(timedelta(seconds=int(el))), str(timedelta(seconds=int(eta))))
    return results


def lpt_makespan(durations: list[float], n_workers: int) -> float:
    """En-uzun-önce (LPT) zamanlamayla duvar saati tahmini."""
    loads = [0.0] * max(1, n_workers)
    for d in sorted(durations, reverse=True):
        i = int(np.argmin(loads))
        loads[i] += d
    return max(loads)


# ---------------------------------------------------------------------------
# Genel deney koşucusu (E3, E4, E5) — run_e1e2 ile aynı sözleşme
# ---------------------------------------------------------------------------
def experiment_dirs(exp_dirname: str, mode: str) -> dict:
    """Mod başına dizinler. full → outputs_rev1/, quick → _quick/, timing → _timing/."""
    base = ROOT / "outputs_rev1" / {"full": "", "quick": "_quick", "timing": "_timing"}[mode]
    return {"root": base, "out": base / exp_dirname,
            "log": base / "logs" / f"{exp_dirname}.log",
            "done": base / "logs" / f"{exp_dirname}.DONE.json",
            "masks": base / "_checkpoints" / "masks",
            "lasso_coef": base / "_checkpoints" / "lasso_coef"}


def start_run_config(out_dir: Path, payload: dict, git: dict, log: logging.Logger) -> None:
    """İlk koşuda run_config.json yaz; devam eden koşuda ``resumes`` listesine ekle."""
    rc = out_dir / "run_config.json"
    entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), **git,
             "command": " ".join([Path(sys.executable).name, *sys.argv])}
    if rc.exists():
        body = json.loads(rc.read_text(encoding="utf-8"))
        if body.get("git_commit") != git["git_commit"]:
            log.warning("UYARI: ilk koşu commit'i %s, şimdiki %s — kod koşu ortasında değişti!",
                        body.get("git_commit"), git["git_commit"])
        body.setdefault("resumes", []).append(entry)
        rc.write_text(json.dumps(body, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return
    write_run_config(out_dir, payload, git)


def run_experiment(*, name: str, mode: str, tasks: list[dict], entry: Callable[[dict], dict],
                   aggregate: Callable[[], None], dirs: dict, n_jobs: int, payload: dict,
                   log: logging.Logger, label: Callable[[dict], str],
                   estimate: Callable[[list[dict]], dict] | None = None) -> None:
    """Görevleri koş → tabloları üret → DONE işaretçisi. Yarıda kalırsa aynı çağrı devam eder.

    ``entry(task)`` kendi checkpoint'ini yazar ve ``{"label", "seconds", "skipped"}``
    döndürür (modül düzeyi fonksiyon olmalı; loky pickle'lar).
    ``estimate(results)`` yalnız ``mode == "timing"`` iken çağrılır; dönen dict
    ``<root>/<name>_timing.json`` olarak yazılır ve log'a basılır.
    """
    git = git_state()                     # koşu BAŞINDA
    log.info("=" * 70)
    log.info("%s başladı | mod=%s | görev=%d | commit=%s | kirli dosya=%d", name, mode, len(tasks),
             (git["git_commit"] or "?")[:7], len(git["git_dirty_files"]))
    if git["git_dirty_files"] and mode == "full":
        log.warning("UYARI: çalışma ağacı temiz değil: %s", git["git_dirty_files"][:5])
    start_run_config(dirs["out"], {**payload, "mode": mode, "n_jobs": n_jobs, "n_tasks": len(tasks)},
                     git, log)
    if dirs["done"].exists():
        dirs["done"].unlink()
    t0 = time.time()
    res = run_tasks(tasks, entry, n_jobs, log, label)
    aggregate()
    wall = time.time() - t0
    if mode == "timing" and estimate is not None:
        est = estimate(res)
        (dirs["root"] / f"{name}_timing.json").write_text(json.dumps(est, indent=2, default=str),
                                                          encoding="utf-8")
        log.info("ZAMAN [%s]: %s", name, json.dumps(est, default=str))
    dirs["done"].parent.mkdir(parents=True, exist_ok=True)
    dirs["done"].write_text(json.dumps({"finished": datetime.now().isoformat(timespec="seconds"),
                                        "mode": mode, "n_tasks": len(tasks),
                                        "wall_seconds": round(wall, 1)}, indent=2), encoding="utf-8")
    log.info("ALL DONE | %s | mod=%s | %d görev | duvar %s", name, mode, len(tasks),
             str(timedelta(seconds=int(wall))))


def check_tasks(name: str, tasks: list[dict], path_of: Callable[[dict], Path], done: Path) -> bool:
    """--check: beklenen görev checkpoint'leri + DONE işaretçisi."""
    have = sum(path_of(t).exists() for t in tasks)
    print(f"{name}: {have}/{len(tasks)} görev checkpoint'i | DONE işaretçisi: {done.exists()}")
    return have == len(tasks) and done.exists()


def data_fingerprint(*arrays) -> str:
    """X/y/groups içerik özeti — maske önbelleği bağlamı için (değişen veri ≠ aynı anahtar)."""
    import hashlib
    h = hashlib.sha1()
    for a in arrays:
        if a is None:
            h.update(b"<none>")
            continue
        a = np.asarray(a)
        if a.dtype == object:
            a = a.astype(str)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def base_argparser(description: str):
    """E3/E4/E5 ortak CLI."""
    import argparse
    ap = argparse.ArgumentParser(description=description)
    mx = ap.add_mutually_exclusive_group()
    mx.add_argument("--quick", action="store_true", help="duman testi → outputs_rev1/_quick/")
    mx.add_argument("--timing", action="store_true", help="kısa zaman ölçümü → outputs_rev1/_timing/")
    mx.add_argument("--check", action="store_true", help="tam koşu bitti mi?")
    mx.add_argument("--aggregate-only", action="store_true", help="yalnız tabloları yeniden üret")
    ap.add_argument("--n-jobs", type=int, default=None, help="varsayılan cpu_count-2")
    ap.add_argument("--fresh", action="store_true", help="--quick: önceki çıktıları silip sıfırdan")
    return ap
