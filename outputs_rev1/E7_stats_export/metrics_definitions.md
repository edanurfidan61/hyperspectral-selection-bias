# metrics_definitions.md — revision-1 (E7)

Bu dosya `src/m07_revision/e7_stats_export.py` tarafından üretilir; elle düzenlemeyin.
Terminoloji: **apparent** (P2), **nested** (P3), **apparent–nested difference** (Δ),
**selection-scope contrast** (P2 − P3 = Δ), **fold-partitioning contrast** (P1 − P2).

## 1. Kollar (aynı dış bölme, eşli tasarım)

| Kol | Seçim verisi | Model fit / test |
|---|---|---|
| P1_random | tüm veri (tek maske) | grup bilgisi yok sayılan KFold(5, shuffle=True, random_state=r) |
| P2_apparent | tüm veri (tek maske) | dış GroupKFold(5, shuffle=True, random_state=r); her fold'da dış-train fit, dış-test tahmin |
| P3_nested | yalnız dış-train | aynı dış fold'lar |
| size_matched_leaky | her fold'da tüm veriden rastgele, dış-train ile aynı sayıda grup (test grupları dahil olabilir) | aynı dış fold'lar |
| random_k / no_selection | rastgele k öznitelik / tüm öznitelikler | aynı dış fold'lar |

## 2. Metrikler ve işaret (R3-58)

- **Fold-mean R² (birincil):** θ̂ = (1/K) Σ_k R²(y_{T_k}, ŷ_{T_k}), K = 5 dış fold.
- **Pooled R²:** tüm dış-test tahminleri birleştirilip tek R² (aynı fold'lar; iki kolda da aynı toplama).
  Potato'da test fold'u ≈ 14 örnek olduğundan fold-mean R² çok oynaktır; pooled R² birincil tabloda
  birlikte raporlanır.
- **Δ (apparent–nested difference):** Δ_R² = R²(P2) − R²(P3); Δ_RMSE = RMSE(P3) − RMSE(P2);
  Δ_MAE = MAE(P3) − MAE(P2). **Pozitif = apparent daha iyimser** (tüm metriklerde).
- **Fold-partitioning contrast:** R²(P1) − R²(P2). Grapevine'de her leaf ayrı grup olduğundan
  GroupKFold(shuffle=True, random_state=r) ile KFold(shuffle=True, random_state=r) aynı bölmeyi verir
  → bu karşıtlık **tanım gereği 0**'dır (hata değil).
- **Tekrarlar arası özet:** R tekrar (E1: 20; E6-G1: 10) üzerinden ortalama, medyan, 2.5–97.5 persentil
  (bölme değişkenliği; güven aralığı DEĞİL).
- **Eşli grup bootstrap CI:** her tekrarın pooled dış-test tahminlerinde gruplar (leaf / Field) yerine
  koyarak yeniden örneklenir (B = 2000), AYNI indeksler iki kolda; Δ pooled R²'nin %2.5–97.5 persentilleri;
  tabloda tekrarlar boyunca medyan alt / üst sınır. Koşullu: fit edilmiş modeller ve maskeler sabit
  kabul edilir; seçim ve fit belirsizliği bu aralığa dahil değildir. Seed: derive_seed(0, r, −1, "bootstrap").

## 3. Maske örtüşmesi (R3-76)

Notasyon: A = tüm veride seçilen (apparent) maske; N_f = f. dış fold'un dış-train'inde seçilen (nested)
maske; |·| = seçili öznitelik sayısı. Fold'lar **ayrı ayrı** karşılaştırılır ve fold'lar (K = 5) üzerinden
ortalama alınır; fold maskeleri birleştirilmez (union/intersection alınmaz).

- **Jaccard:** J_f = |A ∩ N_f| / |A ∪ N_f|; rapor = (1/K) Σ_f J_f.
- **Overlap fraction (outputs3 `mean_overlap_frac`):** `|A ∩ N_f| / |A|`, fold'lar üzerinden ortalama — outputs3 değerleriyle 9 seçimin hepsinde birebir aynı (|fark| ≤ 1e-9; MI seed kümesi: seed0).
- **outputs3 `mean_jaccard`:** `|A ∩ N_f| / |A ∪ N_f|`, fold'lar üzerinden ortalama — outputs3 değerleriyle 9 seçimin hepsinde birebir aynı (|fark| ≤ 1e-9; MI seed kümesi: seed0).
- **LASSO'da k < 20:** LASSO k'yı TAVAN olarak kullanır (sıfır olmayan katsayı < 20 ise daha az öznitelik
  seçilir; kırpma/doldurma `E1_repeated/lasso_truncation.csv`). Bu durumda paydalar gerçek maske
  boyutlarıdır, nominal k = 20 değil: overlap fraction'da |A|, Jaccard'da |A ∪ N_f|.
  ryckewaert/flavonol: |A| = 20, ortalama |N_f| = 20.0; ryckewaert/chlorophyll: |A| = 17, ortalama |N_f| = 13.4; deep_patato/chlorophyll: |A| = 20, ortalama |N_f| = 19.0
- MI seed'e bağlıdır. outputs3 MI satırları **tek bir seçimin (seed 0)** maskesinden hesaplanmıştır (eski not MI'yi deterministik seçicilerle birlikte "tüm veride tek kez seçildi" diye anar); 12 seed ortalaması outputs3 değerlerini VERMEZ. Revizyon E2 tanımı (aşağıda) her tekrarın kendi seed'ini kullanır.
- GA satırları outputs3'te boştur (apparent maske checkpoint'lerde saklanmamıştı); Revizyon E6 arşivi
  (`E6_ga/archive/*.npz`) GA maskelerini artık saklar.
- **Revizyon (E2):** `jaccard_vs_apparent__P3_nested` = (1/K) Σ_f J(N_f, A) (tekrar başına);
  `jaccard_pairwise_folds__P3_nested` = fold maskelerinin ikili Jaccard ortalaması (K(K−1)/2 çift).

## 4. Testler

- **Permütasyon (E5, R3-71):** p = (1 + #{b : T_b ≥ T_obs}) / (B + 1), tek yönlü; T = nested fold-mean R²
  (ayrıca apparent). Grapevine: serbest ve plot içi kısıtlı; potato: eşit boyutlu Field blok permütasyonu.
  ryckewaert × LASSO: B = 50 (DEVIATIONS.md D2), en küçük p = 1/51.
- **Seed-Wilcoxon (eski 12 seed, R3-51/53/54):** eşli Wilcoxon signed-rank (scipy varsayılanı:
  n ≤ 50 ve bağ yoksa tam dağılım), apparent vs nested. Yalnız seçimi seed'e bağlı (stokastik)
  seçicilerde tanımlı: aile = 8 stokastik kombinasyon (GA × 4 + MI × 4); Benjamini–Hochberg bu 8 test
  üzerinden (fold-mean ve pooled ayrı aileler). Etki büyüklüğü olarak Cliff's δ yerine eşli fark özeti
  (ortalama, medyan, Q1, Q3, IQR). Pratik eşik (0.02) kullanılmaz.
- **Spearman (R3-75):** ρ(örtüşme, Δ). 12 satırlık eski analizde ryckewaert/flavonol'ün PLS ve LightGBM
  satırları aynı LASSO/RFE/MI maskesini paylaşır → 9 benzersiz seçim. p: örtüşme değerleri 9 seçim
  arasında permüte edilir (aynı maskeyi paylaşan satırlar birlikte), 9! = 362 880 atamanın hepsi sayılır;
  iki yönlü p = P(|ρ*| ≥ |ρ_obs|).

## 5. Hesaplama bütçesi (R3-39)

Seçim başına model fit sayısı: GA = değerlendirilen benzersiz aday × 5 iç fold (aday önbelleği RNG akışını
değiştirmez); RFE = eleme turu + 1 Ridge fit; LASSO = α grid (100) × 5 iç fold + 1; MI = 0 (öznitelik
başına k-NN MI tahmini). `compute_budget.csv`.
