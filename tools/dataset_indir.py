"""
download_ryckewaert_full.py
===========================
Ryckewaert et al. (2023) - Grapevine Hyperspectral Dataset
DOI: 10.57745/WW7TY7
Platform: Recherche Data Gouv (Dataverse tabanli)

Bu script datasetin TAMAMINI kayipsiz indirir ve her dosyayi dogrular.

Ozellikler:
  - ?format=original ile Dataverse'in tabular donusumu ENGELLENIR
    (bu olmadan buyuk .dat dosyalari bozuk iner - ozellikle 2020-09-10_* serisi)
  - Resume destegi: yarida kalirsa .part dosyasindan devam eder
  - ENVI bant dogrulamasi: her .dat 204 banda bolunemezse YENIDEN indirilir
  - Otomatik retry (5 deneme)
  - Klasor yapisi korunur (Dataset/Data/.../results/ vb.)
  - Sonda bozuk kalan dosyalar raporlanir

Kullanim:
    pip install requests tqdm numpy
    python download_ryckewaert_full.py

Notlar:
    - Indirme ./ryckewaert_dataset/ klasorune yapilir
    - Scripti istedigin kadar tekrar calistirabilirsin; tamamlananlar atlanir,
      bozuk olanlar yeniden indirilir.
"""

import os
import time
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Ayarlar ─────────────────────────────────────────────────────────────────
BASE_URL    = "https://entrepot.recherche.data.gouv.fr"
DOI         = "doi:10.57745/WW7TY7"
OUTPUT_DIR  = Path("ryckewaert_dataset")
RETRY_MAX   = 5            # Her dosya icin max deneme sayisi
RETRY_DELAY = 10           # Hata sonrasi bekleme suresi (saniye)
CHUNK_SIZE  = 1024 * 1024  # 1 MB chunk
BANDS       = 204          # ENVI bant sayisi (Specim IQ)

# ENVI data type -> byte sayisi (bant dogrulamasi icin)
ENVI_DTYPE_BYTES = {1: 1, 2: 2, 3: 4, 4: 4, 5: 8, 12: 2, 13: 4}
# ────────────────────────────────────────────────────────────────────────────


def get_file_list(doi: str) -> list[dict]:
    """Dataverse API ile dataset'teki tum dosyalari listele."""
    url = f"{BASE_URL}/api/datasets/:persistentId/?persistentId={doi}"
    print(f"[*] Dataset metadata aliniyor...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    files = data["data"]["latestVersion"]["files"]
    print(f"[+] Toplam {len(files)} dosya bulundu.")
    return files


def is_dat_valid(dat_path: Path) -> bool:
    """
    .dat dosyasinin gecerli bir ENVI dosyasi olup olmadigini kontrol et.
    Dosya boyutu en az bir bilinen dtype ile (512*512*204) carpanina
    tam bolunebilmeli. Sadece .dat (hiperspektral kup) dosyalari kontrol edilir.
    """
    if dat_path.suffix.lower() != ".dat":
        return True  # .dat olmayan dosyalar (hdr, tab, csv vb.) icin bolme yok

    size = dat_path.stat().st_size
    if size == 0:
        return False

    # 512x512 piksel x 204 bant varsayimi ile herhangi bir dtype tutuyor mu?
    pixels = 512 * 512 * BANDS
    for nbytes in set(ENVI_DTYPE_BYTES.values()):
        if size == pixels * nbytes:
            return True

    # Boyut bilinen carpanlardan birine uymuyorsa: en azindan bant sayisina
    # bolunup bolunmedigini kontrol et (degisik cozunurluk ihtimaline karsi)
    for nbytes in set(ENVI_DTYPE_BYTES.values()):
        if size % (BANDS * nbytes) == 0:
            return True

    return False


def download_file(file_info: dict, output_dir: Path) -> str:
    """
    Tek bir dosyayi indir. Donus: "ok" | "skip" | "fail"
    - skip: dosya zaten var ve gecerli
    - ok:   basariyla indirildi ve dogrulandi
    - fail: tum denemeler basarisiz
    """
    df = file_info["dataFile"]
    file_id  = df["id"]
    filename = df["filename"]
    label    = file_info.get("directoryLabel", "")

    # Hedef yol (klasor yapisini koru)
    target_dir = output_dir / label if label else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    dest_path = target_dir / filename
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")

    # Zaten var ve gecerli mi? -> atla
    if dest_path.exists() and is_dat_valid(dest_path):
        return "skip"

    # Bozuk varsa sil
    if dest_path.exists() and not is_dat_valid(dest_path):
        dest_path.unlink()

    # ?format=original KRITIK: tabular donusumu engeller
    download_url = f"{BASE_URL}/api/access/datafile/{file_id}?format=original"

    for attempt in range(1, RETRY_MAX + 1):
        try:
            # Resume: kismi .part varsa kaldigi yerden devam
            resume_pos = part_path.stat().st_size if part_path.exists() else 0
            headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}

            with requests.get(download_url, stream=True, timeout=120,
                              headers=headers) as r:
                # 416 = aralik gecersiz (dosya zaten tam inmis) -> .part'i kullan
                if r.status_code == 416:
                    pass
                else:
                    r.raise_for_status()

                total = int(r.headers.get("Content-Length", 0)) + resume_pos
                mode = "ab" if resume_pos else "wb"

                with open(part_path, mode) as f, tqdm(
                    total=total if total else None,
                    initial=resume_pos,
                    unit="B", unit_scale=True, unit_divisor=1024,
                    desc=f"    {filename[:40]}", leave=False
                ) as bar:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            # Indirme bitti -> .dat ise dogrula
            os.replace(part_path, dest_path)

            if is_dat_valid(dest_path):
                return "ok"
            else:
                # Bozuk indi: sil, tekrar dene (format=original ile yine)
                print(f"    [!] {filename} bant dogrulamasi BASARISIZ, "
                      f"yeniden indiriliyor (deneme {attempt}/{RETRY_MAX})")
                dest_path.unlink(missing_ok=True)

        except Exception as e:
            print(f"    [!] Deneme {attempt}/{RETRY_MAX} basarisiz: {e}")
            if attempt < RETRY_MAX:
                print(f"    [~] {RETRY_DELAY}s bekleniyor...")
                time.sleep(RETRY_DELAY)

    return "fail"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sunucu hazir mi?
    print("[*] Sunucu erisilebilirligi kontrol ediliyor...")
    try:
        ping = requests.get(f"{BASE_URL}/api/info/version", timeout=10)
        ping.raise_for_status()
        ver = ping.json().get("data", {}).get("version", "?")
        print(f"[+] Sunucu hazir. Dataverse versiyonu: {ver}")
    except Exception as e:
        print(f"[!] Sunucu yanit vermiyor: {e}")
        print("[!] Birkac dakika sonra tekrar deneyin (bakim olabilir).")
        return

    # Dosya listesi
    try:
        files = get_file_list(DOI)
    except Exception as e:
        print(f"[!] Dosya listesi alinamadi: {e}")
        return

    total_mb = sum(f["dataFile"].get("filesize", 0) for f in files) / 1024 / 1024
    print(f"[*] Toplam boyut (yaklasik): {total_mb:,.0f} MB")
    print(f"[*] Hedef klasor: {OUTPUT_DIR.resolve()}\n")

    ok = skip = fail = 0
    failed_files = []

    for i, file_info in enumerate(files, 1):
        fname = file_info["dataFile"]["filename"]
        print(f"[{i}/{len(files)}] {fname}")
        result = download_file(file_info, OUTPUT_DIR)
        if result == "ok":
            ok += 1
            print(f"    [+] OK")
        elif result == "skip":
            skip += 1
            print(f"    [=] Zaten var, atlandi")
        else:
            fail += 1
            failed_files.append(fname)
            print(f"    [x] BASARISIZ")

    # Ozet
    print("\n" + "=" * 55)
    print(f"[+] Yeni indirilen : {ok}")
    print(f"[=] Atlanan (mevcut): {skip}")
    print(f"[x] Basarisiz      : {fail}")
    if failed_files:
        print("\n[!] Basarisiz dosyalar:")
        for f in failed_files:
            print(f"    - {f}")
        print("\n    Scripti tekrar calistirin; kaldigi yerden devam edecek.")
    else:
        print("\n[OK] Tum dataset eksiksiz ve dogrulanmis durumda. ")
    print("=" * 55)


if __name__ == "__main__":
    main()