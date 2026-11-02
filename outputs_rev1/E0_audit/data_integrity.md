# Ryckewaert verisinin bütünlüğü — resmi MD5 doğrulaması

| Öğe | Değer |
|---|---|
| Depo | Recherche Data Gouv (Dataverse), DOI [10.57745/WW7TY7](https://doi.org/10.57745/WW7TY7) |
| Sürüm | 1.0, RELEASED (yayın 2023-06-13); depoda toplam 2889 dosya |
| Resmi checksum kaynağı | `GET /api/datasets/:persistentId/?persistentId=doi:10.57745/WW7TY7` → `latestVersion.files[].dataFile.checksum` (dosya başına, tümü MD5) |
| Kontrol tarihi | 2026-11-01 |
| Kapsam | **409 dosya**: 204 yaprak × `results/REFLECTANCE_*.dat` + `results/REFLECTANCE_*.hdr`, ve `description-2.tab` |
| Sonuç | **A: 409/409 eşleşme** · **B: 409/409 eşleşme** — eşleşmeyen 0, eksik 0 |
| Dosya başına tablo | [`data_integrity_md5.csv`](data_integrity_md5.csv) (`file`, `size_official`, `md5_official`, `match_A`, `match_B`) |
| Yeniden üretim | `python tools/verify_dataset_md5.py` (manifest API'den; iki kopya × 40.6 GB okunur, ≈ 4 dk) |

## Kopyalar
- **A** — orijinal çalışma deposundaki kopya (`C:\selection_bias_paper\data\ryckewaert\raw\<yaprak>\results\`,
  açıklama dosyası `…\ryckewaert\metadata\description-2.tab`); orijinal deponun `config/default.yaml`
  `raw_path` / `ground_truth` ayarlarının gösterdiği, `outputs3/` hattının okuduğu kopya.
- **B** — OneDrive'daki indirme kopyası (`%USERPROFILE%\OneDrive\ryckewaert_dataset\Dataset\Data\`).
  Kontrol anında 409 dosyanın tamamı yerelde mevcuttu (hiçbiri "yalnızca bulutta" değildi).

İki kopya bağımsız olarak hash'lendi; dosyalar yalnız okuma kipinde açıldı.

## Notlar
- `description-2.tab` Dataverse'te ingest edilmiş bir tablodur. Resmi MD5, yüklenen **orijinal**
  dosyaya (`description-2.csv`, 19 363 bayt) aittir; yerel kopyalar da 19 363 bayttır
  (`?format=original` ile indirilmiş) ve bu MD5 ile eşleşir. Sunucunun türettiği 19 681 baytlık
  `.tab` sürümü karşılaştırmaya girmez.
- Ek bilgi (CSV'de yok): `results/REFLECTANCE_*.png` dosyaları da her iki kopyada 204/204 eşleşti.
- İndirme betiği `tools/dataset_indir.py` yalnızca `.dat` boyutunun 512 × 512 × 204 × (veri tipi baytı)
  ile bölünebilirliğini denetler; checksum doğrulaması yapmaz. B'deki 4 adet boş `.validated` işaret dosyası
  (`2020-09-10_003`–`_006`) bilinen bir betiğe bağlanamadı ve doğrulama kanıtı sayılmaz; bütünlüğün kanıtı
  bu belgedeki MD5 karşılaştırmasıdır.
