# Plan Bölüm 1 tablosu — kod/veri doğrulaması

Durum: **DOĞRULANDI** = plandaki bulgu kodda/veride teyit edildi; **DÜZELTME** = plan metni güncellenmeli;
**NOT** = bilgi.

| konu | durum | kanıt |
|---|---|---|
| R3-19 flavonol-türevli 5 dönüşüm | DOĞRULANDI | src/m04_features/extraction.py:112; 5 dönüşüm yalnız yansımadan; NBI = NDRE/FRI (src/m03_indices/indices.py:168). builder X'e hedef eklemiyor (src/m05_dataset/builder.py:120). Veri: 520 sütunda max|r| Chl=0.923, Flav=0.534, NBI=0.868 (hiçbiri 1 değil); Dualex/Flav/Chl adlı sütun: 0; öznitelik adları kod sırasıyla aynı: True. |
| R3-33 LassoCV iç CV | DOĞRULANDI | src/m04_features/selectors.py:87: `cv=5` → `KFold(5, shuffle=False)`, grup yok. Potato: iç fold'lara bölünen Field sayısı apparent=24/24, nested dış-train başına [19, 19, 19, 19, 20] → iç sızıntı VAR. Grapevine leaf: 0 (her leaf tek örnek → etkisiz). EK NOT: düz KFold ardışık blok üretir; grapevine satırları dosya adına göre sıralı olduğundan iç fold'lar plot/tarih bloklarına denk gelir (plot grubu bölünmesi apparent=4/10). |
| R3-26 ölçekleme | DOĞRULANDI | LASSO: `StandardScaler().fit_transform(X)` seçicinin içinde, yalnız verilen X'e (src/m04_features/selectors.py:86). RFE: `Ridge(alpha=1.0)` ölçeklemesiz (src/m04_features/selectors.py:60); sütun std aralığı 1.63e-04 – 7.90e+01 → RFE sıralaması ölçeğe bağlı. PLS: `PLSRegression` varsayılan `scale=True` (model içinde, fold-train'e fit). |
| R3-31 GA iç CV | DOĞRULANDI | src/m04_features/ga_feature_selection.py:220: GroupKFold(5), seed'den bağımsız=True, grup bölüyor=False; aynı bölme tüm nesillerde sabit. |
| R3-30 RNG | DOĞRULANDI | src/m04_features/ga_feature_selection.py:305: ana süreçte `random.seed`/`np.random.seed`; işçiler `mp.Pool` (joblib değil) ile yalnız fitness hesaplar (src/m04_features/ga_feature_selection.py:322); PLS deterministik, LightGBM random_state=42 sabit. |
| R3-28 fitness kırpma | DOĞRULANDI | src/m04_features/ga_feature_selection.py:231: `max(mean_r2, 0)`; MIN_FEATURES=5 altı → 0.0. Sinyalsiz y ile 10 rastgele bireyin 10'i tam 0 fitness aldı. |
| Dış CV | DOĞRULANDI | src/core/cv.py:181: shuffle'sız GroupKFold. 12 seed'de dış fold'lar aynı (grapevine True, potato True). sklearn 1.8.0 `GroupKFold(shuffle=…)`: True. |
| R3-9 plot grupları | DOĞRULANDI | `groups_plot.npy` = variety|plotLocation|collectionDate (src/m05_dataset/builder.py:217); 10 plot grubu, plot başına [10, 10, 15, 15, 22, 23, 25, 25, 25, 34] leaf. Bölümleme description-2.tab ile aynı: True. NOT: kayıtlı dizgelerde U+FFFD var (False; eski ayrıştırıcı) — bölümleme etkilenmez. NOT: plot = çeşit × konum × tarih; çeşit ve tarih ile karışık (R3-10). |
| R3-11 potato grubu | DOĞRULANDI | `groups.append(int(row["Field"]))` (src/m01_io/dataset_registry.py:272); 24 Field, Field başına örnek: [3]; npz sırası CHESS_FOLDS.CSV ile aynı: True. Orijinal 5-fold'da fold başına [5, 5, 5, 5, 4] Field. |
| R3-22 potato ön işleme | DOĞRULANDI | src/m01_io/dataset_registry.py:183: SavGol(11, 2, deriv=0) + SNV, satır başına (örnek-yerel). Önbellek: satır ort. max|·|=6.6e-16, std−1 max|·|=2.2e-16. 'raw channels' ifadesi düzeltilmeli. |
| R3-71 permütasyon p | DOĞRULANDI | src/m06_selection_bias/synthetic_null.py:314: p = (1 + #{perm ≥ obs}) / (B + 1), tek yönlü, istatistik nested fold-mean R². NOT: y serbest karıştırılıyor (grup kısıtı yok) — potato'da Field bloğu korunmuyor (E5'te ele alınacak). |
| Bekleyen dallar | DÜZELTME | Plan 'revision-1 = main + dört dal' diyor. Yayın main'i (7bf625c) kök commit (geçmiş sıkıştırılmış); içerik karşılaştırması: fix/wilcoxon-degeneracy-guard ve fix/figure2-label-overlap kodu main'de ZATEN VAR (git apply --reverse --check temiz; README içeriği yeniden yazılmış hâliyle mevcut). Eksik olanlar (fix/config-truthfulness: config, run_selection_bias, README; chore/logging-and-figure-labels: run.log, figür etiketleri) revision-1'e patch olarak, parça başına ayrı commit ile uygulandı (merge yok). GitHub'da yalnız main var (git ls-remote). |
| Yayına girmemiş betikler | NOT | run_null_n200.py kopyalandı. Yayındaki outputs3/data/null.pkl: NullResult n_repeats=200, method=filter, k=20 → makaledeki null run_null_n200.py ile üretilmiş (config varsayılanı null_repeats=30). regen_outputs3.py, regen_outputs3_rmse.py, outputs2/ henüz eşlenmedi (R3-81, sonraki aşama). |
| Eski checkpoint'ler | NOT | Kopyalanmadı (plan gereği); outputs/17_selection_bias/ çalışma klasöründe yok. |

## Veri tutarlılık kontrolleri
| kontrol | sonuç |
|---|---|
| X_equals_dataset_full_csv | True |
| y_flav_equals_csv_Flav | True |
| groups_leaf_equals_filename | True |
| all_leaves_found_in_tab | True |
| groups_plot_partition_matches_tab | True |
| groups_plot_has_U+FFFD | False |
| y_flav_equals_tab_Flav | True |
| potato_npz_order_equals_chess_csv | True |
| hdr_wavelengths_match_feature_names | True |
| feature_names_equal_code_order | True |
| outer_split_identical_across_seeds_ryck | True |
| ryck_targets_all_finite | True |
| outer_split_identical_across_seeds_potato | True |

## Davranış kontrolleri (seçici/fitness çağrıları; model başarımı ölçülmedi)
| kontrol | sonuç |
|---|---|
| ga_fitness_below_min_features | 0.0 |
| ga_MIN_FEATURES | 5 |
| ga_fitness_null_y_min_over_10 | 0.0 |
| ga_fitness_null_y_n_zero_over_10 | 10 |
| ga_inner_cv_type | GroupKFold |
| ga_inner_cv_seed_independent | True |
| ga_inner_cv_splits_groups | False |
| groupkfold_has_shuffle | True |
| lasso_mask_seed_independent | True |
| lasso_n_selected | 20 |
| rfe_mask_seed_independent | True |
| mi_mask_seed_independent | True |
| mi_jaccard_seed0_seed1 | 1.0 |

## SG / SNV yeniden hesap
| kontrol | sonuç |
|---|---|
| d1_recompute_max_abs_diff_delta1 | 4.0245584642661925e-16 |
| d1_recompute_max_abs_diff_delta_nm | 0.1278088708568376 |
| corr_snv_d1snv | -0.006925409301703962 |
| std_ratio_snv_over_d1 | 28.865318518248838 |
| snv_rows_max_abs_mean | 1.0454600148553557e-15 |
| snv_rows_max_abs_std_minus1 | 5.551115123125783e-16 |
| potato_rows_max_abs_mean | 6.631732200427602e-16 |
| potato_rows_max_abs_std_minus1 | 2.220446049250313e-16 |
| delta_passed_to_savgol_filter | False |

## Başlık sayılarına etkisi
Bu denetimde başlık sayılarını (0.615 / 0.269 / +0.346) değiştirecek bir hata bulunmadı:
tespit edilen konular (LassoCV iç CV'nin potato'da Field bölmesi, REP denkleminin standart dışı yazımı,
SG `delta=1`) yalnızca LASSO-potato satırlarını, indeks tanımının metnini ve türev biriminin
raporlanmasını ilgilendiriyor; başlık konfigürasyonu (Ryckewaert / flavonol / PLS / GA) bunlardan
etkilenmiyor.
