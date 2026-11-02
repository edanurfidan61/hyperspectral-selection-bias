# Flavonol-türevli 5 dönüşüm (R3-19)

Kaynak: `src/m04_features/extraction.py:112`; NBI tanımı `src/m03_indices/indices.py:168`.
Bant seçimi `find_band` ile (hedef nm'ye en yakın bant; Specim IQ, 204 bant).
Gerçek nm değerleri ilk yaprağın ENVI başlığından (`data/ryckewaert/raw_hdr/2020-09-10_003/results/REFLECTANCE_2020-09-10_003.hdr`).

| dönüşüm | denklem (piksel başına) | hedef nm | gerçek nm (bant indeksi) | max |r| Flav (4 istatistik) | max |r| NBI (4 istatistik) |
|---|---|---|---|---|---|
| logR460_R410 | log(R460/R410 + 1e-10)  (oran<=1e-10 ise 0; |R410|<=1e-10 ise oran=0) | 460, 410 | 460.96 (#22), 408.85 (#4) | 0.285 | 0.255 |
| R670_R460 | R670/R460  (|R460|<=1e-10 ise 0) | 670, 460 | 669.77 (#93), 460.96 (#22) | 0.017 | 0.485 |
| R800_R460 | R800/R460  (|R460|<=1e-10 ise 0) | 800, 460 | 798.77 (#136), 460.96 (#22) | 0.290 | 0.125 |
| FRI_x_NDRE | FRI * NDRE = (1/R410 - 1/R460) * (R800-R700)/(R800+R700) | 410, 460, 700, 800 | 408.85 (#4), 460.96 (#22), 699.60 (#103), 798.77 (#136) | 0.206 | 0.298 |
| logNBI | log(max(NBI,0) + 1 + 1e-10) | 410, 460, 700, 800 | 408.85 (#4), 460.96 (#22), 699.60 (#103), 798.77 (#136) | 0.199 | 0.280 |

## Hesaplama sırası
1. Ham yansıma küpünden (SNV/SG UYGULANMADAN) R410, R460, R670, R700, R800 piksel haritaları.
2. FRI, NDRE, NBI piksel haritaları (`calc_nbi_maps`), sonra yukarıdaki 5 dönüşüm.
3. Sonlu olmayan piksel değerleri 0'a çekilir (`np.where(np.isfinite(v), v, 0.0)`).
4. Her harita yaprak pikselleri üzerinde mean / std / p25 / p75 ile özetlenir → 5 × 4 = 20 öznitelik.

## Dualex ile ilişki
- Hiçbir dönüşüm Dualex ölçümünü (hedef `Flav`, `Chl`, `NBI`) girdi olarak kullanmaz;
  hepsi yalnız yansıma bantlarından hesaplanır. Öznitelik adındaki `NBI`, spektral
  `NDRE/FRI` oranıdır; Dualex NBI'si (= Chl/Flav) DEĞİLDİR.
- Hedeflerle en büyük mutlak korelasyon tabloda; |r| = 1 olan sütun yok
  (tüm 520 sütun için: bkz. `section1_verification.md`).
