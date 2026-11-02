"""Geriye uyum kontrolü — yeni motor makaledeki sayıları üretiyor mu?

``revision-1`` motoru (:func:`src.m07_revision.engine.evaluate_arms`) orijinal
bölme (``repeat=0, shuffle=False`` → shuffle'sız GroupKFold(5)) ve sabit seed 0
(``seed_mode="fixed"``) ile koşulur; ryckewaert / flavonol / PLS için RFE,
LASSO (``inner="legacy"``) ve MI'nin apparent (P2, fold-mean), nested (P3,
fold-mean) ve farkı ``outputs3/tables_foldmean/selection_bias_multiseed_raw.csv``
seed-0 satırlarıyla **3 ondalığa kadar** karşılaştırılır.

Çıktı: ``outputs_rev1/backcompat/`` → ``backcompat.csv`` + ``run_config.json``.
Herhangi bir değer 3 ondalıkta tutmazsa çıkış kodu 1 (dur).

Çalıştırma::

    python -m src.m07_revision.check_backcompat            # RFE + LASSO + MI
    python -m src.m07_revision.check_backcompat --quick    # yalnız RFE + MI (LASSO ~dakikalar)
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import paths
from src.core.splits import make_outer_splits, splits_signature
from src.m07_revision.engine import evaluate_arms, resolve_selector

REF_CSV = paths.ROOT / "outputs3" / "tables_foldmean" / "selection_bias_multiseed_raw.csv"
OUT_DIR = paths.ROOT / "outputs_rev1" / "backcompat"

#: (config adı, outputs3 selector adı, sel_kw)
CASES = [
    ("rfe", "rfe", {}),
    ("lasso", "lasso", {"inner": "legacy"}),
    ("mutual_info", "mutual_info", {}),
]


def run(quick: bool = False, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    from src.m01_io.dataset_registry import load_ryckewaert

    t0 = time.time()
    # Ağaç durumu koşu BAŞINDA okunur (kendi çıktı klasörü "kirli" sayılmasın).
    git = lambda *a: subprocess.run(["git", *a], cwd=paths.ROOT, capture_output=True,  # noqa: E731
                                    text=True).stdout.strip()
    git_commit, git_dirty = git("rev-parse", "HEAD"), git("status", "--porcelain").splitlines()
    ds = load_ryckewaert()
    X, y, g = ds.X, ds.target("flavonol"), ds.groups
    fin = np.isfinite(y)
    X, y, g = X[fin], y[fin], g[fin]
    splits = make_outer_splits(len(y), g, "group", 5, repeat=0, shuffle=False)

    ref = pd.read_csv(REF_CSV)
    ref = ref[(ref.seed == 0) & (ref.dataset == "ryckewaert") & (ref.target == "flavonol")
              & (ref.model == "pls")].set_index("selector")

    rows = []
    for name, ref_name, kw in CASES:
        if quick and name == "lasso":
            continue
        t = time.time()
        res = evaluate_arms(X, y, g, splits, "pls", resolve_selector(name),
                            arms=("P2_apparent", "P3_nested"), sel_kw=kw, k=20,
                            repeat=0, base_seed=0, seed_mode="fixed", selector_name=name)
        app = float(res.arm("P2_apparent")["r2_foldmean"])
        nes = float(res.arm("P3_nested")["r2_foldmean"])
        r = ref.loc[ref_name]
        row = {"selector": name, "sel_kw": json.dumps(kw),
               "apparent_new": app, "apparent_ref": float(r.apparent_r2),
               "nested_new": nes, "nested_ref": float(r.nested_r2),
               "diff_new": app - nes, "diff_ref": float(r.bias),
               "apparent_pooled_new": float(res.arm("P2_apparent")["r2_pooled"]),
               "apparent_pooled_ref": float(r.apparent_r2_pooled),
               "elapsed_s": round(time.time() - t, 1)}
        for q in ("apparent", "nested", "diff", "apparent_pooled"):
            row[f"{q}_match3"] = round(row[f"{q}_new"], 3) == round(row[f"{q}_ref"], 3)
            row[f"{q}_absdiff"] = abs(row[f"{q}_new"] - row[f"{q}_ref"])
        rows.append(row)
        print(f"{name:>12}: apparent {app:.3f} (ref {r.apparent_r2:.3f})  nested {nes:.3f} "
              f"(ref {r.nested_r2:.3f})  diff {app - nes:+.3f} (ref {r.bias:+.3f})  "
              f"[{row['elapsed_s']} s]", flush=True)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "backcompat.csv", index=False)
    pk = {}
    for p in ("numpy", "scipy", "scikit-learn", "pandas", "lightgbm"):
        pk[p] = importlib_metadata.version(p)
    (out_dir / "run_config.json").write_text(json.dumps({
        "experiment": "backcompat", "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit, "git_dirty_files": git_dirty,
        "command": " ".join([Path(sys.executable).name, "-m", "src.m07_revision.check_backcompat", *sys.argv[1:]]),
        "python": sys.version, "platform": platform.platform(), "packages": pk,
        "split": {"scheme": "group", "n_splits": 5, "repeat": 0, "shuffle": False,
                  "signature": splits_signature(splits)},
        "seeds": {"seed_mode": "fixed", "base_seed": 0}, "k": 20,
        "reference": str(REF_CSV.relative_to(paths.ROOT).as_posix()),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="LASSO'yu atla (yavaş)")
    args = ap.parse_args()
    df = run(quick=args.quick)
    cols = [c for c in df.columns if c.endswith("_match3") and not c.startswith("apparent_pooled")]
    ok = bool(df[cols].all().all())
    print("GERİYE UYUM (3 ondalık):", "GEÇTİ" if ok else "BAŞARISIZ — DUR")
    sys.exit(0 if ok else 1)
