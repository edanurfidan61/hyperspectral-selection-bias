# Five reflectance-ratio transforms (R3-19)

Source: `src/m04_features/extraction.py:112`; NBI definition: `src/m03_indices/indices.py:168`.
Bands are selected with `find_band` (nearest band to the target wavelength; Specim IQ, 204 bands).
Actual wavelengths are taken from the ENVI header of the first leaf (`data/ryckewaert/raw_hdr/2020-09-10_003/results/REFLECTANCE_2020-09-10_003.hdr`).
In `feature_audit.csv` and Table S01 these 20 features carry the transform label
`reflectance-ratio transform`.

| transform | equation (per pixel) | target nm | actual nm (band index) | max |r| Flav (4 statistics) | max |r| NBI (4 statistics) |
|---|---|---|---|---|---|
| logR460_R410 | log(R460/R410 + 1e-10)  (set to 0 if ratio<=1e-10; ratio=0 if |R410|<=1e-10) | 460, 410 | 460.96 (#22), 408.85 (#4) | 0.285 | 0.255 |
| R670_R460 | R670/R460  (set to 0 if |R460|<=1e-10) | 670, 460 | 669.77 (#93), 460.96 (#22) | 0.017 | 0.485 |
| R800_R460 | R800/R460  (set to 0 if |R460|<=1e-10) | 800, 460 | 798.77 (#136), 460.96 (#22) | 0.290 | 0.125 |
| FRI_x_NDRE | FRI * NDRE = (1/R410 - 1/R460) * (R800-R700)/(R800+R700) | 410, 460, 700, 800 | 408.85 (#4), 460.96 (#22), 699.60 (#103), 798.77 (#136) | 0.206 | 0.298 |
| logNBI | log(max(NBI,0) + 1 + 1e-10) | 410, 460, 700, 800 | 408.85 (#4), 460.96 (#22), 699.60 (#103), 798.77 (#136) | 0.199 | 0.280 |

## Order of computation
1. Pixel-wise R410, R460, R670, R700 and R800 maps from the raw reflectance cube (no SNV or SG applied).
2. Pixel-wise FRI, NDRE and NBI maps (`calc_nbi_maps`), followed by the five transforms above.
3. Non-finite pixel values are set to 0 (`np.where(np.isfinite(v), v, 0.0)`).
4. Each map is summarised over the leaf pixels by mean / std / p25 / p75 → 5 × 4 = 20 features.

## Relation to Dualex
- No transform uses a Dualex measurement (targets `Flav`, `Chl`, `NBI`) as input;
  all are computed from reflectance bands only. `NBI` in the feature names denotes the
  spectral `NDRE/FRI` ratio; it is NOT the Dualex NBI (= Chl/Flav).
- The largest absolute correlation with each target is given in the table; no column has |r| = 1
  (for all 520 columns, see `section1_verification.md`).
