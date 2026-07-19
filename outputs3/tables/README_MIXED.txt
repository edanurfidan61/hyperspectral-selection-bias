================================================================================
  NOT AUTHORITATIVE — MIXED-AGGREGATION REFERENCE ONLY
  OTORİTER DEĞİLDİR — YALNIZCA KARIŞIK-AGGREGATION REFERANSI
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

TR
--------------------------------------------------------------------------------
Bu klasör (outputs3/tables/) makalenin manşet sayılarının kaynağı DEĞİLDİR.

Buradaki CSV dosyaları KARIŞIK bir aggregation kullanır: apparent kolu POOLED
(fold'lar birleştirilerek tek skor), nested kolu ise FOLD-ORTALAMASI (beş dış
fold'un ortalaması) ile hesaplanmıştır. İki kol farklı toplandığı için buradaki
"bias" kolonu, çalışmanın ölçtüğü farkın üstüne ikinci bir tutarsızlık ekler.
Bu bilinçli bir tercihtir: bu dosyalar yalnızca makalenin 4.6. bölümünde
tartışılan karışık-aggregation örneği olarak saklanmaktadır.

Bu nedenle buradaki sayılar makalenin manşetinden farklıdır, örn.:
    apparent R2 = 0.619  (pooled)      manşette 0.615  (fold-ortalaması)
    bias        = 0.349  (karışık)     manşette 0.346  (fold-ortalaması)
    15/16 kombinasyonda pozitif        manşette 14/16  (fold-ortalaması)
Bunlar HATA DEĞİLDİR ve makaleyle ÇELİŞMEZ.

>>> Tablo 4'ü, tüm manşet sayılarını ve anlamlılık zincirini besleyen OTORİTER
    çıktılar şurada:

        outputs3/tables_foldmean/     (her iki kol da fold-ortalaması — simetrik)

    Tümüyle pooled aggregation (her iki kol pooled; 4.6 karşılaştırması,
    +0.341, 13/16 pozitif) şurada:

        outputs3/tables_pooled/

Şüphe durumunda outputs3/tables_foldmean/ kullanın.
================================================================================