"""Sentetik-null deneyini YÜKSEK tekrarla (varsayılan 200), checkpoint'li koş.

Yalnız bu deneyi çalıştırır — 24 saatlik pipeline'ı tetiklemez. synthetic_null.py
HİÇ değiştirilmez; mevcut ``run_filter_null_experiment`` her tekrar için
``n_repeats=1`` ile çağrılır (saf fonksiyon, GA yok).

Determinizm
-----------
Tekrar ``i`` için efektif seed = ``base_seed + i``. Bunu ``base_seed`` argümanını
``base_seed + i`` yaparak ve ``n_repeats=1`` çağırarak sağlarız: fonksiyon içinde
tek tekrar ``seed = base_seed + 0`` kullanır → ``base_seed + i``. Bu, orijinal
30-tekrarlık döngüyle birebir aynı seed dizisidir (tekrar i → seed 1000+i), yani
checkpoint'ten devam ederken sonuçlar değişmez.

Checkpoint
----------
Her tekrar bitince ``_checkpoints/null_progress.pkl`` güncellenir: tamamlanmış
tekrar indekslerini ve (apparent, nested) değerlerini tutar. Yeniden başlatınca bu
dosya okunur ve yalnız EKSİK tekrarlar koşulur. 200 tekrar birikince final
``null_n200.pkl`` yazılır.

Salt-okunur kabul edilenler: outputs2/, outputs3/, eski null.pkl. Onlara dokunulmaz.

Çıktı yeri (Revizyon 1)
-----------------------
Çıktılar doğrudan ``outputs_rev1/null_n200/`` altına yazılır (eskiden
``outputs/17_selection_bias/`` + elle kopya). Checkpoint'ler
``outputs_rev1/null_n200/_checkpoints/`` altında (git-ignored). Her koşu
``run_config.json`` yazar (commit SHA, komut, sürümler, seed tabanı, tarih).

Kullanım
--------
    python run_null_n200.py --n-repeats 5      # checkpoint testi
    python run_null_n200.py --n-repeats 200     # asıl koşu
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import platform
import sys
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

from src.m06_selection_bias.synthetic_null import (
    NULL_FILTER_K,
    NullResult,
    run_filter_null_experiment,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs_rev1" / "null_n200"
CKPT_DIR = OUT_DIR / "_checkpoints"
CKPT_PATH = CKPT_DIR / "null_progress.pkl"

BASE_SEED = 1000   # orijinal run_filter_null_experiment ile aynı taban
K = NULL_FILTER_K  # 20

log = logging.getLogger("run_null_n200")


def _load_checkpoint() -> dict[int, tuple[float, float]]:
    """Tamamlanmış tekrarları {i: (apparent, nested)} olarak yükle (yoksa boş)."""
    if not CKPT_PATH.exists():
        return {}
    try:
        data = pickle.loads(CKPT_PATH.read_bytes())
        done = data.get("done", {})
        log.info("Checkpoint bulundu: %d tekrar zaten tamam.", len(done))
        return done
    except Exception as exc:  # bozuksa sıfırdan ama eski dosyayı silme
        log.warning("Checkpoint okunamadı (%s); sıfırdan başlanıyor.", exc)
        return {}


def _save_checkpoint(done: dict[int, tuple[float, float]], n_repeats: int) -> None:
    """Atomik-ish yaz: önce .tmp'e, sonra rename."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"done": done, "n_repeats_target": n_repeats,
               "base_seed": BASE_SEED, "k": K}
    tmp = CKPT_PATH.with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps(payload))
    tmp.replace(CKPT_PATH)


def _write_run_config(n: int, out_name: str) -> None:
    """Koşu kaydı (plan §0.4): commit SHA, komut, sürümler, seed tabanı, tarih."""
    from src.core import paths

    pkgs = {}
    for p in ("numpy", "scipy", "scikit-learn", "pandas"):
        try:
            pkgs[p] = importlib_metadata.version(p)
        except importlib_metadata.PackageNotFoundError:
            pkgs[p] = None
    payload = {
        "experiment": "null_n200",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": paths._git_sha(),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "python": sys.version, "platform": platform.platform(), "packages": pkgs,
        "n_repeats": n, "base_seed": BASE_SEED, "seeds": f"{BASE_SEED}..{BASE_SEED + n - 1}",
        "k": K, "method": "filter", "out_name": out_name,
    }
    (OUT_DIR / "run_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_one(i: int) -> tuple[float, float]:
    """Tek tekrar (indeks i) → efektif seed = BASE_SEED + i."""
    res = run_filter_null_experiment(
        n_repeats=1, k=K, base_seed=BASE_SEED + i,
    )
    return res.apparent[0], res.nested[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-repeats", type=int, default=200)
    ap.add_argument("--out-name", type=str, default=None,
                    help="Final pkl adı (varsayılan null_n<N>.pkl)")
    args = ap.parse_args()
    n = args.n_repeats
    out_name = args.out_name or f"null_n{n}.pkl"

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_run_config(n, out_name)

    done = _load_checkpoint()
    # Hedef küçüldüyse fazlalıkları yok say (sadece 0..n-1 ile ilgileniriz)
    remaining = [i for i in range(n) if i not in done]
    log.info("Hedef %d tekrar; %d tamam, %d kaldı.", n, len(done), len(remaining))

    for i in remaining:
        app, nes = _run_one(i)
        done[i] = (app, nes)
        _save_checkpoint(done, n)
        log.info("Tekrar %d/%d tamam (seed=%d): apparent=%+.3f nested=%+.3f "
                 "[checkpoint yazıldı: %d/%d]",
                 i + 1, n, BASE_SEED + i, app, nes, len(done), n)

    # Final NullResult'i indeks sırasına göre kur
    res = NullResult(n_repeats=n, n_samples=204, n_features=520,
                     method="filter", k=K)
    for i in range(n):
        app, nes = done[i]
        res.apparent.append(app)
        res.nested.append(nes)

    out_path = OUT_DIR / out_name
    out_path.write_bytes(pickle.dumps(res))
    (OUT_DIR / out_name.replace(".pkl", "_summary.txt")).write_text(
        res.summary(), encoding="utf-8")
    log.info("\n%s", res.summary())
    log.info("Final yazıldı → %s", out_path)


if __name__ == "__main__":
    main()
