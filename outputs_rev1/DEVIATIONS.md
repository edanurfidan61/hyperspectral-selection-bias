# DEVIATIONS.md — revision-1

Deviations from, and applications of decision rules in, the prespecified grid
(`config/revision.yaml`, committed before the runs; plan §0.3). Each entry: what, why,
basis, consequence, and how the outputs mark it. Dates are local (Europe/Istanbul).

---

## D1 — E4 S-null: prespecified reduced-repeats rule applied (2026-10-31)

- **Rule (prespecified):** `E4_simulation.s_null.reduced_repeats_rule` — if the projected
  wall time from the one-repeat timing exceeds 3.0 h, the S-null arms for
  `lasso` and `mutual_info` run with 100 repeats instead of 200.
- **Basis:** `outputs_rev1/_timing/E4_simulation_timing.json` (commit f6a9721):
  projected 3.94 h (LPT, 14 workers, 55.1 CPU-h) > 3.0 h → rule triggered.
- **Applied as:** `python -m src.m07_revision.e4_simulation --reduced-repeats lasso,mutual_info`;
  the decision is frozen in `outputs_rev1/E4_simulation/repeat_decision.json` at run start
  (the script refuses a different decision on resume).
- **Unchanged:** S-null `pearson_topk`, `rfe`, `random_k` = 200 repeats; R3-65 grid = 200;
  S-signal = 100 (no rule for S-signal).
- **Consequence:** wider paired-difference CIs for S-null LASSO/MI (≈ √2 × the 200-repeat SE).
  Repeats i = 0..99 are the first 100 data seeds (1000 + i) of the 200-repeat stream, so the
  100-repeat cells are directly comparable with the 200-repeat cells on those seeds.
- **Marked in outputs:** `n_repeats` column of `s_null_summary.csv`.
- **Approved by:** user, 2026-10-31.

## D2 — E5 grapevine LASSO permutation: B = 50 instead of B = 200 (2026-10-31)

- **Prespecified:** `E5_permutation.B = 200` for every (dataset, target, selector, scheme).
- **Deviation:** for `ryckewaert × lasso` (flavonol and chlorophyll; schemes `free` and
  `within_plot`) B = 50. All other E5 cells keep B = 200. No prespecified reduction rule
  existed for this cell, so this is a deviation decided by the user.
- **Why:** LassoCV (inner = group, max_iter = 20000, tol = 1e-3) converges far more slowly on
  permuted y than on observed y. Timing (`outputs_rev1/_timing/E5_permutation_timing.json`,
  commit e1e5d93): one permuted evaluation (6 LassoCV fits) = 625–1091 s vs 30–280 s observed;
  projected ≈ 202 CPU-h, ≈ 14.5 h wall at moderate load (≈ 20–27 h under full 14-worker load)
  for this cell alone, versus ≈ 0.7 h for the rest of E5.
- **Applied as:** `python -m src.m07_revision.e5_permutation --only ryckewaert/lasso --n-perm 50`
  (after the rest of E5 ran with `--exclude ryckewaert/lasso`).
- **Consequence:** the smallest attainable p is 1/(B+1) = 1/51 ≈ 0.0196 (vs 1/201 ≈ 0.0050);
  p values for these rows are coarser. Permutation seeds are `derive_seed(base, b, −1,
  "perm:<scheme>")`, so b = 0..49 are exactly the first 50 permutations of the B = 200
  stream; re-running without `--n-perm` would extend the same test to B = 200 from checkpoint.
- **Marked in outputs:** `permutation_summary.csv` columns `B` (= 50), `B_planned` (= 200) and
  `B_deviation` (= "DEVIATIONS.md D2").
- **Approved by:** user, 2026-10-31.
