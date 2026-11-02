"""Revizyon 1 (CHEMOLAB-D-26-00975) modülleri.

Her deney ayrı bir modüldür ve ``python -m src.m07_revision.<modül> [--quick]``
ile bağımsız çalışır. Plan: ``docs/REVISION_PLAN.md``.

BLAS/OpenMP iş parçacıkları burada 1'e sabitlenir: paralellik YALNIZ joblib
düzeyinde (tekrar/görev başına bir süreç) olmalı; aksi halde 14 süreç × N BLAS
iş parçacığı çekirdekleri aşırı abone eder. Bu dosya ``src.m07_revision.*``
içindeki her modülden ÖNCE yüklenir (``python -m`` paket ``__init__``'ini önce
çalıştırır) ve loky işçileri ``os.environ``'u devralır. numpy bu noktadan önce
yüklenmişse ortam değişkeni geç kalır; işçi fonksiyonları ayrıca
``threadpoolctl.threadpool_limits(1)`` kullanır.
"""

import os

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for _v in THREAD_ENV_VARS:
    os.environ[_v] = "1"
