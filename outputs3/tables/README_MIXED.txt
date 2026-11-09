================================================================================
  NOT AUTHORITATIVE — MIXED-AGGREGATION REFERENCE ONLY
================================================================================

EN
--------------------------------------------------------------------------------
This folder (outputs3/tables/) does NOT back the headline numbers of the paper.

The CSV files here use a MIXED aggregation: the apparent arm is POOLED
(single score over the pooled out-of-fold predictions) while the nested arm is
FOLD-MEAN (averaged over the five outer folds). Because the two arms are
aggregated differently, the "bias" column here combines a second discrepancy
on top of the one the study measures. This is deliberate: these files are kept
only as the worked example of mixed aggregation discussed in Section 4.6 of the
paper.

As a result, the numbers in this folder differ from the paper's headline, e.g.:
    apparent R2 = 0.619  (pooled)      instead of 0.615  (fold-mean)
    bias        = 0.349  (mixed)       instead of 0.346  (fold-mean)
    positive in 15/16 combinations     instead of 14/16  (fold-mean)
These are NOT errors and do NOT contradict the paper.

>>> The AUTHORITATIVE outputs, which back Table 4, all headline numbers, and
    the significance chain, are in:

        outputs3/tables_foldmean/     (both arms fold-mean — symmetric)

    The fully-pooled aggregation (both arms pooled; Section 4.6 comparison,
    +0.341, positive in 13/16) is in:

        outputs3/tables_pooled/

When in doubt, use outputs3/tables_foldmean/.

================================================================================