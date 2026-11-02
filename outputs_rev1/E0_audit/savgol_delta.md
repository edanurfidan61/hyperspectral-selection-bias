# Savitzky–Golay `delta` ve bant aralığı

## Grapevine (Ryckewaert) — `d1snv_R*` (204 öznitelik)
| Öğe | Değer | Kaynak |
|---|---|---|
| Fonksiyon | `sg_first_derivative(window_length=11, polyorder=2)` → `savitzky_golay(..., deriv=1)` | `src/m02_preprocessing/spectral.py:69` |
| `savgol_filter` çağrısı | `savgol_filter(spectra, window_length, polyorder, deriv, axis=1)` | `src/m02_preprocessing/spectral.py:46` |
| `delta` | **verilmiyor → scipy varsayılanı `delta=1.0`** (türev bant-indeksi başına; nm başına DEĞİL) | aynı satır |
| `mode` | verilmiyor → varsayılan `'interp'` (kenar 5 bantta pencereye polinom uydurma) | aynı satır |
| `config/default.yaml` `preprocessing.savgol.deriv` | `0` yazıyor ama d1 için **okunmuyor**; `deriv=1` kodda sabit | `config/default.yaml:55` |
| Girdi | yaprak-ortalama spektrumun SNV'si (`snv_R*`) | `src/m04_features/extraction.py:221` |

### Bant aralığı (Specim IQ, 204 bant, ilk yaprağın .hdr dosyası)
- Aralık: 397.32 – 1003.58 nm
- Δλ: min 2.880, medyan 2.990, ortalama 2.987, max 3.090 nm
- nm başına türeve dönüşüm: `dR/dλ ≈ d1snv / Δλ` (≈ d1snv / 2.987)

### Doğrulama (kayıtlı X.npy üzerinden, config'e güvenmeden)
- `snv_R*` sütunlarından `savgol_filter(11, 2, deriv=1)` (delta=1) ile yeniden hesap → kayıtlı `d1snv_R*` ile
  max |fark| = **4.02e-16**  (delta = ortalama Δλ ile: 1.28e-01) → **delta = 1 doğrulandı**.
- corr(snv, d1snv) = -0.007, std(snv)/std(d1snv) = 28.9 → deriv = 1 (deriv = 0 olsaydı korelasyon ≈ +1 olurdu).
- SNV örnek-yerel: satır ortalaması max |·| = 1.0e-15, satır std − 1 max |·| = 5.6e-16.

## Potato (deep_patato) — 150 kanal
- `_preprocess_spectra`: `savitzky_golay(window=11, polyorder=2)` → **deriv=0 (yumuşatma)**, ardından SNV
  (`src/m01_io/dataset_registry.py:171`). `delta` deriv=0'da etkisizdir.
- Kayıtlı önbellek doğrulaması: satır ortalaması max |·| = 6.6e-16, satır std − 1 max |·| = 2.2e-16
  → SNV uygulanmış, örnek-yerel. Kanallar "ham" değildir (R3-22: makaledeki "raw channels" ifadesi düzeltilmeli).
- Potato kanal dalga boyları önbellekte yok (öznitelik adları `band_000`…`band_149`).
