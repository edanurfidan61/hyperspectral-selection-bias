# Selection Bias in Small-Sample Leaf-HSI Regression

> Küçük örneklemli yaprak-HSI regresyonunda seçim yanlılığı

**Central claim / Ana iddia:**

> In small-sample leaf hyperspectral (HSI) regression, when wavelength/feature
> selection is performed **outside** the cross-validation loop, model performance
> is **systematically inflated**. This inflation is a property of *selecting
> features on data the validation folds later reuse* — not of any one selector
> family and not of any one dataset or plant. Honest reporting **requires
> nested-CV**.
>
> Küçük örneklemli yaprak-HSI regresyonunda, dalga boyu/öznitelik seçimi çapraz
> doğrulama döngüsünün **dışında** yapıldığında model performansı **sistematik
> olarak şişer**. Bu şişme, doğrulama fold'larının sonradan yeniden kullandığı
> veri üzerinde öznitelik seçmenin bir sonucudur — tek bir seçici ailesine, tek
> bir veri setine ya da tek bir bitkiye özgü değildir. Dürüst raporlama için
> **nested-CV zorunludur**.

---

## What this measures / Ne ölçülüyor

For each `(dataset, target, model, selector)` we compute two numbers:

- **apparent R²** *(biased)* — the selector picks features on **all** data by
  optimizing 5-fold CV R²; the reported score is the CV R² on that same data.
  Selection has "seen" the validation folds → optimistic.
- **nested R²** *(honest)* — an outer GroupKFold; in each outer fold the selector
  runs on the **outer-train only**, the chosen mask is fit on outer-train and
  evaluated on the held-out outer-test. Selection never sees the outer validation
  → unbiased.

**bias = apparent − nested.** Both arms use the *same* fold-mean aggregation, so
the difference is not an artifact of mixing pooled and per-fold scores.

The measurement is **selector-agnostic**. Four selector families are run through
the identical apparent/nested machinery:

| selector | family | notes |
|---|---|---|
| `ga` | search (genetic algorithm) | the most extreme case — a large search space over feature subsets |
| `rfe` | wrapper (recursive feature elimination) | deterministic, fixed-`k` |
| `lasso` | embedded (LassoCV, then top-`k` by \|coef\|) | deterministic, fixed-`k` |
| `mutual_info` | filter (mutual information) | fixed-`k` |

The GA is the sharpest illustration because its search space is the largest, but
the bias appears across all four families — that is the point: the leak is in the
*out-of-loop selection*, not in the GA.

A **synthetic-null** experiment (pure noise, no real signal) shows the same
inflation with no genuine relationship to exploit, and a **permutation** test
(real feature structure preserved, target shuffled) confirms the observed scores
sit far out in the permuted-null distribution.

### Headline result (Ryckewaert, flavonol + PLS, GA)

| metric | value |
|---|---|
| apparent R² (biased, fold-mean) | **0.615** |
| nested R² (honest) | **0.269 ± 0.026** |
| **bias** | **+0.346** |

The apparent number here is the **fold-mean** apparent R² (0.615), aggregated the
same way as the nested arm. A *pooled* apparent R² also exists for this
combination (0.619); it is used only in the aggregation comparison of Section 4.6
and is **not** the headline number. The two are close but not interchangeable —
see *Which output directory backs which table*.

---

## Installation

Requires **Python ≥ 3.10**.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions that produced the published results.

> **NumPy note.** The pipeline is run and validated under **NumPy 2.x** (the code
> uses the NumPy 2.0+ API, e.g. `np.trapezoid`). If your toolchain requires
> `numpy<2`, also downgrade `scipy<1.13` and `scikit-learn<1.5` and re-test —
> those are the components that gate the NumPy ABI. See the comment block at the
> top of `requirements.txt`.

(Developers can still use `pip install -e .` for an editable install.)

---

## Quick start

For a fast end-to-end smoke test (tiny GA, builds the dataset if missing):

```bash
python run_selection_bias.py --quick
```

`--quick` sets pop=20, ngen=8, null-repeats=3, n-perm=5.

### Full reproduction — one command per result

The published tables and figures come from **four separate runs**, each with the
selector set and seed count that its result requires. They are deliberately kept
apart: no single `--selectors` value is right for every table (Table 4 uses all
four selectors, Table 5 excludes the GA, the permutation test uses only
`rfe` + `mutual_info`), and the multi-seed mode and the synthetic null cannot
share a command (see the note below).

```bash
# A — Table 4 (headline + all 16 combinations, p_FDR / Cliff's δ) + seed-stability (Figure 4)
#     12 seeds (0–11); all four selectors.
python run_selection_bias.py --pop 150 --ngen 150 \
    --seeds 0,1,2,3,4,5,6,7,8,9,10,11 \
    --selectors ga,rfe,lasso,mutual_info \
    --stages dataset viz nested stats report
```

> **Why the synthetic null is a *separate* command (A2).** Passing `--seeds`
> switches the run into multi-seed stability mode, which **automatically skips
> the `null` stage** — the two are never produced together. If you add `null` to
> command A it is silently dropped. Run the null on its own, **without**
> `--seeds`:

```bash
# A2 — synthetic null (Figure 3, null.pkl, 200 repetitions). No --seeds.
python run_selection_bias.py --pop 150 --ngen 150 --null-repeats 200 \
    --selectors ga,rfe,lasso,mutual_info \
    --stages null report

# B — single-seed live cvcomp (P1/P2/P3 leakage decomposition, one seed). GA excluded.
#     A sanity check of the decomposition; the *authoritative* 12-seed Table 5
#     comes from regen_pipeline_comparison_foldmean.py (see Reproducibility).
python run_selection_bias.py --pop 150 --ngen 150 \
    --selectors rfe,lasso,mutual_info \
    --stages cvcomp

# C — permutation test (rfe + mutual_info, n_perm=200 → permutation_summary.csv).
python run_selection_bias.py --n-perm 200 \
    --selectors rfe,mutual_info \
    --stages permutation
```

**Which command produces what:**

| cmd | selectors | seeds | produces |
|---|---|---|---|
| **A** | ga, rfe, lasso, mutual_info | 12 (0–11) | Table 4 (16 combinations, p_FDR / Cliff's δ) + Figure 4 (seed stability) |
| **A2** | ga, rfe, lasso, mutual_info | 1 (default 42) | Figure 3 (synthetic-null distribution) + `null.pkl`, 200 repetitions |
| **B** | rfe, lasso, mutual_info | 1 (42) | live single-seed cvcomp demo (P1/P2/P3 for one seed) — *not* the authoritative Table 5 |
| **C** | rfe, mutual_info | 1 (42) | permutation test, `permutation_summary.csv`, n_perm=200 |

> **Table 5 (authoritative, 12-seed fold-mean).** The published Table 5 — P1/P2/P3
> means ± std over the same twelve seeds as Table 4 — is **not** produced by a
> live `run_selection_bias.py` stage, and that is by design, not an omission.
> `cvcomp` always runs a single seed (the orchestrator passes only `seeds[0]`,
> default 42), so it cannot produce a mean ± std across seeds. The 12-seed
> fold-mean table is reconstructed from the stored checkpoints by
> `regen_pipeline_comparison_foldmean.py` →
> `outputs3/tables_foldmean/pipeline_comparison.csv` — the **same
> checkpoint-based reconstruction** used for Table 4's fold-mean apparent arm, so
> both tables share one aggregation over one set of predictions. Command B is a
> single-seed sanity check of the same decomposition, not the paper's Table 5.

Outputs land in `outputs/17_selection_bias/`: `selection_bias_summary.csv`,
`run_config.json` (the exact settings of the run — see *Reproducibility*),
`stats_summary.csv`, `cv_comparison_summary.csv`, `permutation_summary.csv`,
`fig_apparent_vs_nested.png`, `fig_fold_variance.png`, `fig_band_frequency_*.png`,
`fig_null_distribution.png`, `fig_permutation.png`, `fig_seed_stability.png`,
plus `results.pkl` / `null.pkl` / `permutation.pkl`.

### Selected flags

| flag | meaning |
|---|---|
| `--quick` | pop=20, ngen=8, null-repeats=3, n-perm=5 (smoke test) |
| `--pop / --ngen` | GA population / generations |
| `--selectors ga,rfe,lasso,mutual_info` | feature-selector families to run |
| `--k 20` | features kept by the fixed-`k` selectors (rfe/lasso/mutual_info) |
| `--n-outer 5` | outer fold count |
| `--seeds 0,1,…,11` | multi-seed stability run (checkpointed); switches off the `null` stage |
| `--null-repeats 200` | synthetic-null repetitions (feeds the `null` stage) |
| `--n-perm 200` | permutation-test repetitions (feeds the `permutation` stage) |
| `--no-null` | skip the synthetic-null stage |
| `--include-lambrusco` | add the lambrusco binary targets back into the analysis |
| `--stages dataset viz nested null cvcomp stats permutation report` | run a subset |

`--null-repeats` and `--n-perm` drive **two different experiments** — the
synthetic null and the permutation test respectively — and are recorded
separately in `run_config.json`.

### Stages (`--stages`)

The accepted stage names are `dataset viz nested null cvcomp stats permutation
report`. An unrecognized name is silently ignored, so spelling matters.

| stage | what it does |
|---|---|
| `dataset` | build/load the feature matrix from `data/` (skips if `outputs/01_dataset` exists) |
| `viz` | dataset-visualisation figures for every registered dataset |
| `nested` | the core analysis — apparent vs nested-CV per `(dataset, target, selector)`. With `--seeds` this runs the multi-seed stability variant |
| `null` | synthetic-null experiment (pure noise; `--null-repeats` sets the repetition count). Skipped automatically when `--seeds` is set |
| `cvcomp` | three-pipeline CV comparison (P1/P2/P3), separating group- from selection-leakage. Always **single-seed**: even with `--seeds 0,…,11` the orchestrator passes only `seeds[0]` (else 42), so it never yields a mean ± std. The authoritative 12-seed Table 5 is regenerated offline — see *Reproducibility* |
| `stats` | Wilcoxon + BH-FDR + Cliff's δ over the multi-seed results. **Requires a prior `--seeds` run**, since it reads `selection_bias_multiseed_raw.csv`; without it the stage errors out |
| `permutation` | permutation test (real `X` fixed, `y` shuffled; `--n-perm` sets the count) |
| `report` | writes `selection_bias_summary.csv` + the apparent/nested and null figures from the collected results |

---

## Datasets

Only the targets that the analysis actually runs are listed here:

| name | shape | analysed targets | groups | notes |
|---|---|---|---|---|
| **ryckewaert** | 204 × 520 | chlorophyll, flavonol | 204 leaves | grapevine ENVI cubes (main example) |
| **deep_patato** | 72 × 150 | chlorophyll (SPAD) | 24 fields | potato HSI cubes, clean regression |
| **lambrusco** | 158 × 24 | *(excluded — see below)* | 28 sets | 6-band multispectral (24 = 6 bands × {mean, std, p25, p75}); binary FD/ESCA targets |

> **Loaded but not analysed.** The Ryckewaert loader also exposes an `nbi` target,
> and the deep_patato loader exposes `rwc`, `fvfm`, `pi`, `ndvi`, and
> `soil_moisture`. These are read into the `HSIDataset` objects but are **not**
> part of the analysed combinations, so they feed **no** table or figure. The
> analysed set is fixed by the combination map in `multi_dataset.py`, not by what
> the loaders happen to expose.

<!-- -->

> **Lambrusco is excluded from the regression analysis.** Its FD/ESCA targets are
> **binary**, so an R²-based apparent/nested comparison is not the appropriate
> metric; it belongs in a separate classification sub-analysis (e.g. F1 /
> balanced accuracy) rather than the regression tables. The loader stays
> registered and can be added back with `--include-lambrusco` for exploratory
> runs, but no claim is made about it here.

All datasets are exposed through `src/m01_io/dataset_registry.py` as a common
`HSIDataset` (X, targets, groups, feature_names, wavelengths). Raw data lives
under `data/<name>/` and is **git-ignored** (large); loaders cache derived
feature matrices to `outputs/_datasets_cache/`.

Cross-dataset comparison: `python -m src.m06_selection_bias.multi_dataset`.

### Downloading the data

**The raw data is NOT included in this repository** (tens of GB). Download it
from the public archives and place it under `data/<name>/` yourself:

| dataset | source | DOI / link |
|---|---|---|
| **Ryckewaert** (grapevine) | Recherche Data Gouv | [10.57745/WW7TY7](https://doi.org/10.57745/WW7TY7) |
| **Deep Potato** (potato) | Mendeley Data | [10.17632/xn2wy75f8m.2](https://doi.org/10.17632/xn2wy75f8m.2) |
| **Lambrusco** (grapevine, multispectral) | Zenodo | [10.5281/zenodo.14936376](https://doi.org/10.5281/zenodo.14936376) |

The Lambrusco archive is only needed for exploratory `--include-lambrusco` runs;
it does not feed the main regression analysis (binary targets — see *Datasets*).

Expected layout after download:

```text
data/
  ryckewaert/
    raw/<leaf>/results/REFLECTANCE_*.dat (+ .hdr)
    metadata/description-2.tab
  deep_patato/
    ...                         # as distributed by the Mendeley archive
  lambrusco/                    # only for --include-lambrusco runs
    annotation.json
    processed/                  # <key>SET/ dirs of IMG_*.png groups
    raw/
```

The exact paths are configurable in `config/default.yaml` (`data:` section).
On first run, `--stages dataset` builds `outputs/01_dataset/` from this raw data.

---

## Code layout

```text
run_selection_bias.py            # end-to-end orchestrator (the entry point)
src/
  core/                          # config, paths, CV splitters, metrics_ci (per-combo bootstrap CI)
  m01_io/                        # ENVI loader + dataset_registry (HSIDataset, loaders)
  m02_preprocessing/             # SavGol / SNV / segmentation
  m03_indices/                   # vegetation indices (feature extraction)
  m04_features/
    extraction.py                # cube → feature vector
    ga_feature_selection.py      # GA; run_ga_core() is the pure, side-effect-free selector
    selectors.py                 # the four selector families behind one common signature
    ga_wavelength_viz.py         # GA wavelength-visualisation helper (auto-called after a GA run)
  m05_dataset/                   # 01_dataset builder + ground truth + dataset_viz
  m06_selection_bias/            # THE PAPER
    multiseed.py                 #   drives the 12-seed run (seeds 0–11) behind command A
    nested_ga.py                 #   apparent vs nested-CV (NestedResult), selector-agnostic
    synthetic_null.py            #   pure-noise null + permutation test
    cv_comparison.py             #   three-pipeline P1/P2/P3 leakage decomposition
    multi_dataset.py             #   loop over REGISTRY × selectors
    stats_tests.py               #   Wilcoxon + BH-FDR + Cliff's δ over multi-seed results
    report.py                    #   summary CSV + figures
```

The selection-bias pipeline reuses the feature-extraction chain (m01–m05) to
build datasets, but its only claim-bearing logic lives in `m06_selection_bias`.
The core evaluator (`nested_ga.nested_evaluation`) takes a `selector_fn`, so the
apparent/nested comparison is shared by all four selector families; the GA path
is just one caller. The regressors the selectors optimize (PLS, Ridge, RF,
LightGBM, …) are built self-contained in
`m04_features/ga_feature_selection.py` (`_build_regressor`); there is no separate
model package.

---

## Method notes

- **Groups matter.** Outer splits are group-aware (GroupKFold) so leaves/fields
  from the same unit never straddle train/test.
- **Fold variance.** Small samples make nested fold scores highly variable (e.g.
  one Ryckewaert outer fold goes negative) — itself evidence that out-of-loop
  selection masks instability.
- **Band instability.** Per-fold masks select different subsets; band-selection
  frequency (`fig_band_frequency_*`) shows the selection is fitting noise.

---

## Statistical significance

The significance chain runs over the **12-seed** results
(`selection_bias_multiseed_raw.csv`, produced by command A):

1. Per combination, the 12 paired `(apparent, nested)` R² values across seeds.
2. **Wilcoxon signed-rank** test (paired, non-parametric) on those pairs.
3. **Benjamini–Hochberg (FDR)** correction across the combinations where the test
   is defined.
4. **Cliff's δ** effect size (with Cohen's *d* reported alongside).

**Degeneracy.** Of the 16 combinations, **8 use a deterministic selector (all
LASSO + all RFE)**: their apparent − nested difference is **constant across
seeds**, so the effective sample size is 1, not 12, and the seed-based Wilcoxon
test is **undefined**. Those rows carry `degenerate = True` and `p_value` /
`p_value_fdr` / `cliffs_delta` / `cohens_d` of `nan`, while their **bias is still
reported**. BH-FDR therefore runs over the **8** combinations where the test is
defined. (This matches the Table 4 footnote in the manuscript exactly.)

**Bootstrap CIs are not part of this chain.** `core/metrics_ci` computes a
per-combination bootstrap confidence interval around the pooled predictions and
stores it on each result for reporting; it is *not* used as a significance
criterion. Significance is the Wilcoxon → BH-FDR → Cliff's δ chain above.

---

## Reproducibility

- **Fixed seeds.** All stochastic steps (GA, CV splits, synthetic null,
  permutation shuffles) are seeded; the default seed is `42`
  (`models.random_state` in `config/default.yaml`). Multi-seed stability runs use
  `--seeds 0,…,11`.
- **`run_config.json`.** Every run of `run_selection_bias.py` writes
  `outputs/17_selection_bias/run_config.json` recording the *exact* settings
  used: seed(s), `pop`, `ngen`, `k`, `n_outer`, the selector list, `null_repeats`
  **and** `n_perm` (kept as separate fields — they drive different experiments),
  the dataset list, timestamp, Python/platform, the full command line, and the
  **git commit hash** (when run inside a git checkout). This file is the canonical
  record to cite in the paper's appendix as *"produced with exactly these
  settings."*
- **Outputs are regenerable.** Figures, tables and pickles under
  `outputs/17_selection_bias/` are derived purely from the raw data + the pinned
  dependencies + `run_config.json`; they are git-ignored and rebuilt by re-running
  the four commands above.

To reproduce the published numbers: install the pinned `requirements.txt`,
download the datasets (above), and run commands A / A2 / B / C from *Full
reproduction*. Compare your `run_config.json` against the appendix to confirm the
configuration matches.

### Which output directory backs which table

The 12-seed results are kept under three aggregation schemes. The paper cites
each in a different place, and Section 4.6 compares them directly.

| Directory | Aggregation | Status |
| --- | --- | --- |
| `outputs3/tables_foldmean/` | apparent **and** nested both fold-mean (symmetric) | **Authoritative.** Backs Table 4 and all headline numbers (0.615 apparent). |
| `outputs3/tables_pooled/` | apparent **and** nested both pooled | Backs Section 4.6's fully-pooled aggregation comparison (`+0.341`, positive in 13/16). |
| `outputs3/tables/` | apparent pooled (0.619), nested fold-mean (mixed) | Pooled-apparent reference run, cited in Section 4.6 as the mixed-aggregation example. Not superseded. |

**Table 5** is backed by `outputs3/tables_foldmean/pipeline_comparison.csv`
(12-seed fold-mean P1/P2/P3 with mean ± std), regenerated by
`regen_pipeline_comparison_foldmean.py`. This is reconstructed from the stored
checkpoints — **not** run live — for exactly the same reason Table 4's fold-mean
apparent arm is: it re-slices predictions that already exist in the checkpoints
rather than re-running the GA, so Tables 4 and 5 share one aggregation over one
set of predictions. The `cvcomp` stage (Command B) is a single-seed live sanity
check of the same P1/P2/P3 decomposition, not the source of the paper's Table 5.

> **Note — duplicate filenames across schemes.** `selection_bias_multiseed.csv`
> and `stats_summary.csv` exist under *both* `outputs3/tables_foldmean/` and
> `outputs3/tables/`. They are **different aggregations with the same name**: the
> `tables_foldmean/` copies are **authoritative** (symmetric fold-mean — Table 4
> and the significance chain); the `tables/` copies are the **mixed-aggregation
> reference** for Section 4.6 only. When in doubt, `tables_foldmean/` is the one
> the headline and Table 4 cite.

- **Why fold-mean is authoritative.** Pooling the apparent arm while averaging
  the nested arm per fold makes the two arms non-comparable, and the asymmetry
  leaks into the apparent − nested difference. In the current pipeline both arms
  are computed as fold-means at run time — the apparent arm is a fold-mean over
  the same GroupKFold(5) partition as the nested arm, with the pooled value kept
  in a dedicated field for the Section 4.6 comparison rather than mixed into the
  headline.
- **What the pooled schemes are for.** `outputs3/tables_pooled/` holds the
  fully-pooled figures (both arms pooled: `+0.341` for the principal
  configuration, positive in 13/16). `outputs3/tables/` holds the mixed scheme,
  which Section 4.6 quotes as the worked example of what mixed aggregation would
  report — two combinations flip sign between the schemes, and that flip is the
  point of the section, so both reference runs stay reproducible.

---

## Citation / DOI

If you use this code or method, please cite the paper (citation to be added on
acceptance) and this software archive.

The tagged GitHub release is archived on **Zenodo**, which mints a DOI:

```text
Fidan, E. & Aktaş, F. (2026). Selection bias in small-sample leaf-HSI regression
[Software]. Zenodo. https://doi.org/10.5281/zenodo.21439965

This release: v1.0.0, https://doi.org/10.5281/zenodo.21439966
```

> **Concept DOI: `10.5281/zenodo.21439965`** — always resolves to the latest
> version. **Version DOI: `10.5281/zenodo.21439966`** — pins this specific
> release (v1.0.0). Cite the concept DOI for the project in general, or the
> version DOI to reference exactly what you used.
