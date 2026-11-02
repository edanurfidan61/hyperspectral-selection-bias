"""reproduce_revision.py — revision-1 (CHEMOLAB-D-26-00975) sayılarını yeniden üret ve doğrula (R3-81).

Üç düzey (en hızlıdan en uzuna):

``--verify`` (varsayılan, saniyeler)
    ``expected_values.json``'daki her kontrolü diskteki tablolara karşı doğrular:
    makalenin başlık sayıları (0.615 / 0.269 / +0.346, outputs3) ve revizyon tablolarının
    birincil / ikincil sayıları (outputs_rev1). Biri tolerans dışındaysa çıkış kodu 1.

``--aggregate`` (dakikalar; checkpoint'ler gerekir)
    Model koşmadan, her deneyin checkpoint'lerinden TÜM ``outputs_rev1`` tablolarını yeniden
    yazar (E1/E2, E3, E4, E5, E6, E7), sonra ``--verify``. Öncesi/sonrası İÇERİK karşılaştırması
    yapılır (``.gz`` açılarak): değişen dosyalar listelenir. İçeriği aynı kalan ``.gz``
    dosyalarının ÖNCEKİ baytları geri yazılır (gzip başlığındaki zaman damgası git'i kirletmesin).
    ``--steps E6,E7`` ile alt küme.

``--full`` (≈ 15–20 sa; temiz ortam: veri + ``pip install -r requirements.txt``)
    Tüm deneyleri sırayla koşturacak komutları yazdırır; ``--execute`` ile gerçekten koşturur
    (ilk hata durumunda durur). Her deney checkpoint'lidir: yarıda kalırsa aynı komut devam eder.
    Uzun koşuyu tek komutla gözetimsiz yürütmek için: ``tools/run_overnight.ps1``.

Çalıştırma::

    python reproduce_revision.py                       # = --verify
    python reproduce_revision.py --aggregate
    python reproduce_revision.py --full                # yalnız komut listesi
    python reproduce_revision.py --full --execute      # UZUN
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

#: --aggregate adımları (ad, modül argümanları). Sıra önemli: E7 en sonda (diğerlerini okur).
AGGREGATE_STEPS = [
    ("E1E2", ["-m", "src.m07_revision.run_e1e2", "--aggregate-only"]),
    ("E3", ["-m", "src.m07_revision.e3_sensitivity", "--aggregate-only"]),
    ("E4", ["-m", "src.m07_revision.e4_simulation", "--aggregate-only"]),
    ("E5", ["-m", "src.m07_revision.e5_permutation", "--aggregate-only"]),
    ("E6", ["-m", "src.m07_revision.e6_ga", "--aggregate-only"]),
    ("E7", ["-m", "src.m07_revision.e7_stats_export"]),
]

#: --full komutları (ad, argümanlar, tahmini süre). Sıra = gerçekte koşulan sıra
#: (E1/E2 önce; E3–E6 gece zinciri tools/run_overnight.py ile aynı; E7 en sonda).
FULL_STEPS = [
    ("E0 audit", ["-m", "src.m07_revision.audit"], "dakikalar"),
    ("backcompat (outputs3 seed 0)", ["-m", "src.m07_revision.check_backcompat"], "≈ 5 dk"),
    ("E1 + E2 (R = 20)", ["-m", "src.m07_revision.run_e1e2"], "≈ 1 sa"),
    ("E3 grouping axis", ["-m", "src.m07_revision.e3_sensitivity", "--axes", "grouping"], "≈ 15 dk"),
    ("E6 G0 + G1", ["-m", "src.m07_revision.e6_ga", "--variants", "G0,G1"], "≈ 1–1.5 sa"),
    ("E3 all axes", ["-m", "src.m07_revision.e3_sensitivity"], "≈ 1–1.5 sa"),
    ("E6 G2 + G3", ["-m", "src.m07_revision.e6_ga", "--variants", "G2,G3"], "≈ 1 sa"),
    ("E5 (without ryckewaert/lasso)", ["-m", "src.m07_revision.e5_permutation", "--exclude", "ryckewaert/lasso"],
     "≈ 1 sa"),
    ("E4 (reduced-repeats rule, DEVIATIONS D1)",
     ["-m", "src.m07_revision.e4_simulation", "--reduced-repeats", "lasso,mutual_info"], "≈ 3–5 sa"),
    ("E5 ryckewaert/lasso, B = 50 (DEVIATIONS D2)",
     ["-m", "src.m07_revision.e5_permutation", "--only", "ryckewaert/lasso", "--n-perm", "50"], "≈ 4–7 sa"),
    ("E7 tables + supplementary", ["-m", "src.m07_revision.e7_stats_export"], "< 1 dk (+ ilk koşuda ≈ 4 dk)"),
]

#: --aggregate'in içerik karşılaştırmasına giren (git'te izlenen) tablo uzantıları
TABLE_SUFFIXES = (".csv", ".gz", ".json", ".md")


def _python() -> str:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv) if venv.exists() else sys.executable


def verify() -> int:
    from src.m07_revision.e7_stats_export import EXPECTED_PATH, verify_expected

    if not EXPECTED_PATH.exists():
        print(f"{EXPECTED_PATH.name} yok — önce: python -m src.m07_revision.e7_stats_export --freeze-expected")
        return 1
    r = verify_expected()
    by = r.groupby("source").ok.agg(["sum", "count"])
    print("expected_values.json doğrulaması:")
    for s, row in by.iterrows():
        print(f"  {s:<24} {int(row['sum'])}/{int(row['count'])}")
    bad = r[~r.ok]
    if len(bad):
        import pandas as pd
        with pd.option_context("display.width", 250, "display.max_colwidth", 80):
            print("\nTUTMAYAN KONTROLLER:")
            print(bad[["id", "file", "column", "expected", "got", "tolerance", "error"]].to_string(index=False))
    head = r[r.id.isin(["M1", "M2", "M3"])]
    if len(head):
        print("\nBaşlık (outputs3, fold-mean, 12 seed): " + " | ".join(
            f"{i} {g:.3f} (beklenen {e:.3f})" for i, g, e in zip(head.id, head.got, head.expected)))
    print("\nSONUÇ:", "TAMAM" if r.ok.all() else "BAŞARISIZ")
    return 0 if r.ok.all() else 1


def _tracked_tables() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "outputs_rev1"], cwd=ROOT, capture_output=True, text=True).stdout
    return [ROOT / p for p in out.splitlines() if p.endswith(TABLE_SUFFIXES) and not p.endswith("run_config.json")]


def _content_sha(p: Path) -> str | None:
    if not p.exists():
        return None
    b = p.read_bytes()
    if p.suffix == ".gz":
        b = gzip.decompress(b)
    return hashlib.sha1(b).hexdigest()


def aggregate(steps: set[str] | None) -> int:
    files = _tracked_tables()
    before = {p: (p.read_bytes() if p.exists() else None, _content_sha(p)) for p in files}
    py = _python()
    for name, args in AGGREGATE_STEPS:
        if steps and name not in steps:
            continue
        t0 = time.time()
        print(f"\n=== {name}: {' '.join(args)}", flush=True)
        rc = subprocess.run([py, *args], cwd=ROOT).returncode
        print(f"=== {name}: çıkış {rc} [{time.time() - t0:.0f} s]", flush=True)
        if rc != 0:
            print("SONUÇ: BAŞARISIZ (toplama adımı hata verdi)")
            return 1
    changed, restored = [], 0
    for p, (raw, sha) in before.items():
        now = _content_sha(p)
        if now != sha:
            changed.append(p.relative_to(ROOT).as_posix())
        elif raw is not None and p.suffix == ".gz" and p.read_bytes() != raw:
            p.write_bytes(raw)                      # içerik aynı: yalnız gzip başlığı farklı → eski bayt
            restored += 1
    print(f"\nİçerik karşılaştırması: {len(files)} izlenen tablo | değişen {len(changed)} | "
          f"gzip başlığı geri alınan {restored}")
    for c in changed:
        print(f"  DEĞİŞTİ: {c}")
    rc = verify()
    if changed:
        print("NOT: değişen dosyalar için `git diff --stat outputs_rev1` (zaman damgası sütunları "
              "dışında fark beklenmez).")
    return rc


def full(execute: bool) -> int:
    py = _python()
    print("revision-1 tam yeniden üretim — komutlar (sırayla):\n")
    for i, (name, args, dur) in enumerate(FULL_STEPS, 1):
        print(f"{i:>2}. {name:<45} [{dur}]\n    {Path(py).name} {' '.join(args)}")
    print(f"{len(FULL_STEPS) + 1:>2}. doğrulama\n    {Path(py).name} reproduce_revision.py --verify")
    if not execute:
        print("\n(Yalnız liste. Koşturmak için --execute; gözetimsiz: tools/run_overnight.ps1.)")
        return 0
    for name, args, _ in FULL_STEPS:
        t0 = time.time()
        print(f"\n=== {name}", flush=True)
        rc = subprocess.run([py, *args], cwd=ROOT).returncode
        print(f"=== {name}: çıkış {rc} [{(time.time() - t0) / 60:.1f} dk]", flush=True)
        if rc != 0:
            print("DURDU: adım hata verdi; aynı komut checkpoint'ten devam eder.")
            return rc
    return verify()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mx = ap.add_mutually_exclusive_group()
    mx.add_argument("--verify", action="store_true", help="(varsayılan) expected_values.json doğrulaması")
    mx.add_argument("--aggregate", action="store_true", help="checkpoint'lerden tüm tabloları yeniden üret")
    mx.add_argument("--full", action="store_true", help="tam yeniden üretim komutları (--execute ile koş)")
    ap.add_argument("--execute", action="store_true", help="--full: komutları gerçekten koştur (UZUN)")
    ap.add_argument("--steps", type=str, default="",
                    help=f"--aggregate alt kümesi (virgülle): {','.join(n for n, _ in AGGREGATE_STEPS)}")
    a = ap.parse_args()
    if a.execute and not a.full:
        ap.error("--execute yalnız --full ile")
    if a.aggregate:
        steps = {s.strip() for s in a.steps.split(",") if s.strip()} or None
        bad = (steps or set()) - {n for n, _ in AGGREGATE_STEPS}
        if bad:
            ap.error(f"bilinmeyen adım(lar): {sorted(bad)}")
        sys.exit(aggregate(steps))
    if a.full:
        sys.exit(full(a.execute))
    sys.exit(verify())


if __name__ == "__main__":
    main()
