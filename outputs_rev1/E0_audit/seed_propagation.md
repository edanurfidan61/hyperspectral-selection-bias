# Seed nereye gidiyor (R3-37)

`nested_evaluation` içinde `seed = cfg["seed"]` (`src/m06_selection_bias/nested_ga.py:302`);
`sel_kwargs` (seed dahil) seçiciye iletilir. Çoklu-seed koşusu: seed ∈ {0, …, 11}.

| Bileşen | Seed kullanılıyor mu | Kanıt | Doğrulama |
|---|---|---|---|
| GA evrimi (başlangıç popülasyonu, seçim, çaprazlama, mutasyon) | **Evet** — ana süreçte `random.seed(seed)`, `np.random.seed(seed)` | `src/m04_features/ga_feature_selection.py:305` | işçiler (mp.Pool) yalnız deterministik fitness hesaplar → tek RNG akışı |
| GA fitness (iç CV) | Hayır — `make_cv_splitter(groups=…)` → `GroupKFold` (shuffle yok) | `src/m04_features/ga_feature_selection.py:220` | ✓ seed 0 ve 7 ile iç fold'lar aynı; grup bölünmesi: False |
| GA modelleri | PLS deterministik; LightGBM `random_state=42` sabit (seed'e bağlı DEĞİL) | `src/m04_features/ga_feature_selection.py:169` | — |
| MI (`mutual_info_regression`) | **Evet** — `random_state=seed` (kNN tahmincisine eklenen gürültü) | `src/m04_features/selectors.py:119` | seed 0/1 maske Jaccard = 1.00 |
| LASSO (`LassoCV`) | **Etkisiz** — `random_state=seed` veriliyor ama `selection='cyclic'` (varsayılan) iken kullanılmaz; iç CV `cv=5` → `KFold(shuffle=False)` | `src/m04_features/selectors.py:87` | ✓ seed 0/1 maskesi aynı |
| RFE (Ridge) | Hayır — deterministik | `src/m04_features/selectors.py:60` | ✓ seed 0/1 maskesi aynı |
| Dış CV (nested ve apparent) | **Etkisiz** — `GroupKFold(n_splits=5)` shuffle'sız; `random_state` yok sayılır | `src/core/cv.py:181` | ✓ grapevine, ✓ potato: 12 seed'de dış fold'lar birebir aynı |
| P1 (rastgele KFold) | **Evet** — `KFold(shuffle=True, random_state=seed)` | `src/m06_selection_bias/cv_comparison.py:96` | — |
| Bootstrap CI | **Evet** — `np.random.default_rng(seed)` | `src/core/metrics_ci.py:171` | — |
| Permütasyon | **Evet** — `np.random.default_rng(seed)` → `rng.permutation(y)` | `src/m06_selection_bias/synthetic_null.py:66` | — |
| Sentetik null | tekrar i → seed = 1000 + i (`run_null_n200.py`, BASE_SEED = 1000) | `run_null_n200.py` | — |

## Sonuç
- Seed'ler arasında değişen tek şey **seçici içindeki rastgelelik**tir (GA evrimi, MI tahmini).
  LASSO ve RFE seed'den bağımsızdır → 12 seed'de apparent–nested farkı sabit (Wilcoxon için etkin n = 1;
  `stats_tests.is_degenerate` bunu işaretliyor).
- Dış bölme 12 seed'in hepsinde AYNIDIR; seed varyasyonu bölme değişkenliğini İÇERMEZ.
  Bölme değişkenliği için `GroupKFold(shuffle=True, random_state=r)` gerekir
  (sklearn 1.8.0'da mevcut: ✓).
- `nested_evaluation` docstring'indeki "seed ayrıca dış/iç CV'ye akar" ifadesi grup varken
  geçersizdir (GroupKFold seed kullanmaz) — metinde düzeltilmeli.
