"""Ryckewaert (DOI 10.57745/WW7TY7) yerel kopyalarını resmi Dataverse MD5'leriyle doğrular.

Kapsam (409 dosya): her yaprak için results/REFLECTANCE_*.dat + .hdr (204 × 2) ve description-2.tab.
Resmi checksum'lar Dataverse API'sinden (latestVersion.files[].dataFile.checksum) alınır; algoritma
dosya başına deponun bildirdiğidir (bu depoda tümü MD5).

Not: description-2.tab Dataverse'te "ingest" edilmiş tablodur; checksum yüklenen ORİJİNAL dosyaya
(description-2.csv, 19 363 bayt) aittir. Yerel kopya ?format=original ile indirildiği için doğrudan
karşılaştırılabilir (sunucunun türettiği 19 681 baytlık .tab ile DEĞİL).

Kullanım:
    python tools/verify_dataset_md5.py                    # A + B, manifest API'den, ~2 × 40.6 GB okur
    python tools/verify_dataset_md5.py --quick            # ilk 3 yaprak + açıklama dosyası → outputs_rev1/_quick/
    python tools/verify_dataset_md5.py --manifest dataset.json --copy A <raw> <description-2.tab>
    python tools/verify_dataset_md5.py --from-results A=result_A.csv B=result_B.csv   # hash'lemeden birleştir
    python tools/verify_dataset_md5.py --selftest         # ağsız, sentetik dosyalarla duman testi

Dosyalar yalnızca "rb" ile açılır; hiçbir veri dosyası yazılmaz/değiştirilmez.
Çıkış kodu: 0 = tüm kopyalarda tüm dosyalar eşleşti, 1 = eşleşmeyen/eksik var.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOI = "doi:10.57745/WW7TY7"
API = "https://entrepot.recherche.data.gouv.fr/api/datasets/:persistentId/?persistentId=" + DOI
OUT = ROOT / "outputs_rev1" / "E0_audit" / "data_integrity_md5.csv"
OUT_QUICK = ROOT / "outputs_rev1" / "_quick" / "data_integrity_md5.csv"
DESC = "description-2.tab"
QUICK_LEAVES = 3
# Varsayılan kopyalar: A = orijinal çalışma deposu (salt okunur), B = OneDrive'daki indirme kopyası.
# Kullanıcı adı koda gömülmesin diye B ev dizininden türetilir.
DEFAULT_COPIES = {
    "A": (Path(r"C:\selection_bias_paper\data\ryckewaert\raw"),
          Path(r"C:\selection_bias_paper\data\ryckewaert\metadata") / DESC),
    "B": (Path.home() / "OneDrive" / "ryckewaert_dataset" / "Dataset" / "Data",
          Path.home() / "OneDrive" / "ryckewaert_dataset" / "Dataset" / DESC),
}


# ── Manifest ────────────────────────────────────────────────────────────────
def load_manifest(path: Path | None) -> dict:
    """Yerel JSON verilmişse onu, yoksa API yanıtını döndürür (yalnız GET)."""
    if path is not None:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    with urllib.request.urlopen(API, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def official_records(meta: dict) -> tuple[list[dict], dict]:
    """Kapsamdaki dosyalar (depo yoluna göre sıralı) + sürüm bilgisi."""
    v = meta["data"]["latestVersion"]
    recs = []
    for f in v["files"]:
        df, label = f["dataFile"], f.get("directoryLabel") or ""
        name = df["filename"]
        is_refl = (name.startswith("REFLECTANCE_") and name.rsplit(".", 1)[-1] in ("dat", "hdr")
                   and label.endswith("/results"))
        if not (is_refl or name == DESC):
            continue
        ck = df.get("checksum") or {}
        recs.append(dict(file=f"{label}/{name}" if label else name, name=name,
                         leaf=label.split("/")[2] if is_refl else None,
                         # ingest edilmiş tabloda checksum orijinale ait → boyut da orijinalin
                         size_official=df.get("originalFileSize") or df.get("filesize"),
                         algorithm=ck.get("type"),
                         md5_official=ck.get("value")))
    recs.sort(key=lambda r: r["file"])
    algs = {r["algorithm"] for r in recs}
    if algs != {"MD5"}:                                   # başka algoritma → bu betik yetkisiz
        raise SystemExit(f"Beklenmeyen checksum algoritması: {algs}")
    info = dict(version=f"{v.get('versionNumber')}.{v.get('versionMinorNumber')}",
                state=v.get("versionState"), release=v.get("releaseTime"),
                n_files_deposit=len(v["files"]))
    return recs, info


# ── Hash ────────────────────────────────────────────────────────────────────
def md5_file(p: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:                             # salt okuma
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def check_copy(recs: list[dict], data_root: Path, desc_path: Path, log=print) -> dict[str, str]:
    """Her depo yolu için 'match' | 'mismatch' | 'missing'."""
    out, t0 = {}, time.time()
    for i, r in enumerate(recs, 1):
        p = desc_path if r["name"] == DESC else Path(data_root) / r["leaf"] / "results" / r["name"]
        if not p.is_file():
            out[r["file"]] = "missing"
        else:
            out[r["file"]] = "match" if md5_file(p) == r["md5_official"] else "mismatch"
        if i % 50 == 0 or i == len(recs):
            log(f"  {i}/{len(recs)}  {time.time() - t0:.0f} s")
    return out


def statuses_from_results(path: Path, recs: list[dict]) -> dict[str, str]:
    """Önceki koşunun dosya-başı çıktısından (sütunlar: name, status[, official]) durum tablosu."""
    by_name = {r["name"]: r for r in recs}
    st = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        r = by_name.get(row["name"])
        if r is None:                                     # kapsam dışı (ör. .png) → atla
            continue
        if row.get("official") and row["official"] != r["md5_official"]:
            raise SystemExit(f"{path}: {row['name']} için resmi MD5 manifestle uyuşmuyor")
        st[r["file"]] = row["status"]
    return st


# ── Özet ────────────────────────────────────────────────────────────────────
def write_summary(recs: list[dict], statuses: dict[str, dict[str, str]], out: Path) -> dict:
    labels = list(statuses)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["file", "size_official", "md5_official"] + [f"match_{k}" for k in labels])
        for r in recs:
            w.writerow([r["file"], r["size_official"], r["md5_official"]]
                       + [statuses[k].get(r["file"], "missing") for k in labels])
    counts = {}
    for k in labels:
        vals = [statuses[k].get(r["file"], "missing") for r in recs]
        counts[k] = {s: vals.count(s) for s in ("match", "mismatch", "missing")}
    return counts


def report(counts: dict, recs: list[dict], statuses: dict) -> bool:
    ok = True
    for k, c in counts.items():
        print(f"{k}: eşleşen {c['match']} | eşleşmeyen {c['mismatch']} | eksik {c['missing']}  (/{len(recs)})")
        bad = [r["file"] for r in recs if statuses[k].get(r["file"], "missing") != "match"]
        for b in bad[:50]:
            print(f"    {statuses[k].get(b, 'missing'):8s} {b}")
        ok &= not bad
    print("SONUÇ:", "TAMAM" if ok else "UYUŞMAZLIK VAR")
    return ok


# ── Duman testi ─────────────────────────────────────────────────────────────
def selftest() -> None:
    """Sentetik manifest + dosyalar: match / mismatch / missing ayrımı ve kapsam filtresi."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        payload = {"L1.dat": b"abc", "L1.hdr": b"hdr1", "L2.dat": b"xyz", "L2.hdr": b"hdr2"}
        files = []
        for leaf in ("L1", "L2"):
            for ext in ("dat", "hdr", "png"):
                name = f"REFLECTANCE_{leaf}.{ext}"
                body = payload.get(f"{leaf}.{ext}", b"png")
                files.append({"directoryLabel": f"Dataset/Data/{leaf}/results",
                              "dataFile": {"filename": name, "filesize": len(body),
                                           "checksum": {"type": "MD5", "value": hashlib.md5(body).hexdigest()}}})
        files.append({"directoryLabel": "Dataset", "dataFile": {
            "filename": DESC, "filesize": 3, "checksum": {"type": "MD5", "value": hashlib.md5(b"tab").hexdigest()}}})
        meta = {"data": {"latestVersion": {"versionNumber": 1, "versionMinorNumber": 0, "files": files}}}
        recs, info = official_records(meta)
        assert len(recs) == 5 and info["version"] == "1.0"          # png kapsam dışı
        root = td / "Data"
        for leaf in ("L1", "L2"):
            (root / leaf / "results").mkdir(parents=True)
        (root / "L1" / "results" / "REFLECTANCE_L1.dat").write_bytes(b"abc")       # match
        (root / "L1" / "results" / "REFLECTANCE_L1.hdr").write_bytes(b"hdr1")      # match
        (root / "L2" / "results" / "REFLECTANCE_L2.dat").write_bytes(b"xyZ")       # mismatch
        (td / DESC).write_bytes(b"tab")                                            # match; L2.hdr eksik
        st = check_copy(recs, root, td / DESC, log=lambda *_: None)
        assert st["Dataset/Data/L1/results/REFLECTANCE_L1.dat"] == "match"
        assert st["Dataset/Data/L2/results/REFLECTANCE_L2.dat"] == "mismatch"
        assert st["Dataset/Data/L2/results/REFLECTANCE_L2.hdr"] == "missing"
        assert st[f"Dataset/{DESC}"] == "match"
        counts = write_summary(recs, {"A": st}, td / "s.csv")
        assert counts["A"] == {"match": 3, "mismatch": 1, "missing": 1}
        rows = list(csv.DictReader(open(td / "s.csv", encoding="utf-8")))
        assert [r["file"] for r in rows] == sorted(r["file"] for r in rows) and "match_A" in rows[0]
    print("selftest: TAMAM")


# ── Ana akış ────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, help="Yerel Dataverse JSON (yoksa API'den çekilir)")
    ap.add_argument("--copy", nargs=3, action="append", metavar=("LABEL", "DATA_ROOT", "DESC_PATH"),
                    help="Doğrulanacak kopya (tekrarlanabilir); varsayılan A ve B")
    ap.add_argument("--from-results", nargs="+", metavar="LABEL=CSV",
                    help="Hash'lemeden, önceki dosya-başı sonuçlardan birleştir")
    ap.add_argument("--out", type=Path, help=f"Özet CSV (varsayılan {OUT.relative_to(ROOT)})")
    ap.add_argument("--quick", action="store_true", help=f"İlk {QUICK_LEAVES} yaprak + açıklama dosyası")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0

    recs, info = official_records(load_manifest(a.manifest))
    print(f"{DOI} sürüm {info['version']} ({info['state']}, {info['release']}); "
          f"depoda {info['n_files_deposit']} dosya, kapsam {len(recs)}")
    if a.quick:
        leaves = sorted({r["leaf"] for r in recs if r["leaf"]})[:QUICK_LEAVES]
        recs = [r for r in recs if r["leaf"] in leaves or r["name"] == DESC]
    out = a.out or (OUT_QUICK if a.quick else OUT)

    statuses: dict[str, dict[str, str]] = {}
    if a.from_results:
        for spec in a.from_results:
            label, path = spec.split("=", 1)
            statuses[label] = statuses_from_results(Path(path), recs)
    else:
        copies = {c[0]: (Path(c[1]), Path(c[2])) for c in a.copy} if a.copy else DEFAULT_COPIES
        for label, (data_root, desc) in copies.items():
            print(f"[{label}] {data_root}")
            statuses[label] = check_copy(recs, data_root, desc)

    counts = write_summary(recs, statuses, out)
    print(f"yazıldı: {out}")
    return 0 if report(counts, recs, statuses) else 1


if __name__ == "__main__":
    sys.exit(main())
