"""Revizyon modelleri — E3 ``pls_components`` ekseni (R3-50).

:class:`PLSInnerCV`: PLS bileşen sayısını YALNIZ verilen eğitim verisinde (nested'de
dış-train) iç ``GroupKFold(5)`` ile seçer, sonra tüm eğitim verisine yeniden fit
eder. Dış-test hiç görülmez. Izgara ``1..10`` (``min(c, n_seçili, n_iç_train − 1)``
ile kırpılır). Eşitlikte küçük bileşen sayısı seçilir. ``groups`` verilmezse iç
bölme ``KFold(5, shuffle=True, random_state=0)``.

Engine bu modeli ``model="pls_tuned"`` adıyla kurar ve ``fit(X, y, groups=g_tr)``
çağırır (``engine._fit_predict``).
"""

from __future__ import annotations

import numpy as np


class PLSInnerCV:
    """İç CV ile ayarlanan PLS — sklearn benzeri arayüz (fit/predict, coef_, n_components)."""

    def __init__(self, grid=tuple(range(1, 11)), n_inner: int = 5):
        self.grid = tuple(int(c) for c in grid)
        self.n_inner = int(n_inner)
        self.n_components = None
        self.inner_scores_ = {}
        self._pls = None

    def _inner_splits(self, X, y, groups):
        from sklearn.model_selection import GroupKFold, KFold
        if groups is not None and len(np.unique(groups)) >= self.n_inner:
            return list(GroupKFold(n_splits=self.n_inner).split(X, y, groups))
        return list(KFold(n_splits=self.n_inner, shuffle=True, random_state=0).split(X, y))

    def fit(self, X, y, groups=None):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.metrics import r2_score

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        splits = self._inner_splits(X, y, groups)
        min_tr = min(len(tr) for tr, _ in splits)
        cands = sorted({max(1, min(c, X.shape[1], min_tr - 1)) for c in self.grid})
        scores = {}
        for c in cands:
            fold = []
            for tr, te in splits:
                m = PLSRegression(n_components=c).fit(X[tr], y[tr])
                fold.append(r2_score(y[te], np.asarray(m.predict(X[te])).ravel()))
            scores[c] = float(np.mean(fold))
        best = max(cands, key=lambda c: (scores[c], -c))   # eşitlikte küçük c
        self.inner_scores_ = scores
        self.n_components = int(best)
        self._pls = PLSRegression(n_components=self.n_components).fit(X, y)
        return self

    def predict(self, X):
        return self._pls.predict(np.asarray(X, dtype=float))

    @property
    def coef_(self):
        return None if self._pls is None else self._pls.coef_
