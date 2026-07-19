"""Group-aware cross-validation helper'ları.

Pipeline genelinde tek noktadan CV splitter üretmek için kullanılır.
Aynı yaprağın (veya aynı bağ/tarihten gelen yaprakların) hem train hem
test'te bulunmasını engellemek için ``groups`` parametresi verilir;
verilmezse klasik StratifiedKFold/KFold'a düşülür.

| task           | groups var mı? | döndürülen splitter   |
|----------------|----------------|-----------------------|
| classification | evet           | StratifiedGroupKFold  |
| classification | hayır          | StratifiedKFold       |
| regression     | evet           | GroupKFold            |
| regression     | hayır          | KFold                 |

Hold-out (train/test) split'i için ``make_holdout_split`` aynı mantığı
ShuffleSplit varyantlarıyla uygular.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal, Optional

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

Task = Literal["regression", "classification"]


def bin_continuous(
    y: np.ndarray, n_bins: int = 5, strategy: str = "quantile",
) -> np.ndarray:
    """Sürekli y'yi tamsayı bin etiketlerine dönüştür (stratification için).

    Parameters
    ----------
    y : ndarray
        Sürekli değerli hedef.
    n_bins : int
        Bin sayısı (default 5).
    strategy : {"quantile", "uniform"}
        - ``"quantile"`` → eşit-örnekli bin'ler (her bin yaklaşık aynı sayıda
          örnek alır). Çarpık dağılımlar için tercih edilir.
        - ``"uniform"``  → eşit-genişlikli bin'ler ([min, max] aralığını
          ``n_bins`` parçaya böler).

    Returns
    -------
    bins : ndarray (int)
        ``[0, n_bins-1]`` aralığında bin etiketleri. NaN'lar için ``-1`` döner;
        çağıran kod bu örnekleri zaten geçerli/valid maskesiyle filtrelemiş
        olmalıdır.
    """
    y = np.asarray(y, dtype=np.float64)
    out = np.full(y.shape, -1, dtype=np.int64)
    finite = np.isfinite(y)
    if not finite.any():
        return out
    yv = y[finite]
    if strategy == "quantile":
        edges = np.quantile(yv, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)  # tekrarlı kenarları kaldır
        if edges.size < 2:
            out[finite] = 0
            return out
        idx = np.clip(np.digitize(yv, edges[1:-1], right=False), 0, edges.size - 2)
    elif strategy == "uniform":
        lo, hi = float(yv.min()), float(yv.max())
        if hi <= lo:
            out[finite] = 0
            return out
        edges = np.linspace(lo, hi, n_bins + 1)
        idx = np.clip(np.digitize(yv, edges[1:-1], right=False), 0, n_bins - 1)
    else:
        raise ValueError(f"Bilinmeyen strategy: {strategy!r}")
    out[finite] = idx.astype(np.int64)
    return out


class StratifiedRegressionKFold:
    """Sürekli y için stratified K-fold (binning ile).

    sklearn-uyumlu splitter. ``split(X, y, groups)`` çağrısında y'yi quantile
    bin'lerine ayırıp ``StratifiedKFold`` (groups=None) veya
    ``StratifiedGroupKFold`` (groups verilirse) ile devam eder.

    Düşük örnekli bin'ler n_splits'ten küçük olursa otomatik olarak n_bins
    aşağı düşürülür ya da fallback olarak normal KFold/GroupKFold'a düşer.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_bins: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        strategy: str = "quantile",
    ) -> None:
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.shuffle = shuffle
        self.random_state = random_state
        self.strategy = strategy

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y, groups=None):
        y = np.asarray(y).ravel()
        n_bins = self.n_bins
        # Bin sayısını n_splits'e uyduracak şekilde aşağı düşür
        for _ in range(n_bins):
            binned = bin_continuous(y, n_bins=n_bins, strategy=self.strategy)
            _, counts = np.unique(binned[binned >= 0], return_counts=True)
            if counts.size >= 2 and counts.min() >= self.n_splits:
                break
            n_bins = max(2, n_bins - 1)
            if n_bins < 2:
                break

        if n_bins < 2:
            # Fallback: stratification yapılamıyor → normal KFold
            inner = (
                GroupKFold(n_splits=self.n_splits) if groups is not None
                else KFold(n_splits=self.n_splits, shuffle=self.shuffle,
                           random_state=self.random_state)
            )
            if groups is not None:
                yield from inner.split(X, y, groups=groups)
            else:
                yield from inner.split(X, y)
            return

        if groups is not None:
            inner = StratifiedGroupKFold(
                n_splits=self.n_splits, shuffle=self.shuffle,
                random_state=self.random_state,
            )
            yield from inner.split(X, binned, groups=groups)
        else:
            inner = StratifiedKFold(
                n_splits=self.n_splits, shuffle=self.shuffle,
                random_state=self.random_state,
            )
            yield from inner.split(X, binned)


def make_cv_splitter(
    *,
    n_splits: int,
    task: Task,
    groups: Optional[np.ndarray] = None,
    random_state: int = 42,
    stratify_regression: bool = False,
    n_bins: int = 5,
):
    """Görev tipine ve groups parametresine göre uygun CV splitter üret.

    ``stratify_regression=True`` ise regresyon için ``StratifiedRegressionKFold``
    döner — y quantile-bin'lerine ayrılarak stratification yapılır. Çarpık veya
    küçük dataset'lerde fold'lar arası dağılım benzerliğini garanti eder.
    """
    if task == "regression" and stratify_regression:
        return StratifiedRegressionKFold(
            n_splits=n_splits, n_bins=n_bins, shuffle=True,
            random_state=random_state,
        )
    if groups is not None:
        if task == "classification":
            return StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state,
            )
        return GroupKFold(n_splits=n_splits)
    if task == "classification":
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state,
        )
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def split_indices(
    splitter,
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Splitter.split() çağrı çeşitliliğini soyutla — groups varsa geçir."""
    if groups is not None:
        return splitter.split(X, y, groups=groups)
    return splitter.split(X, y)


def make_holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    task: Task = "regression",
) -> tuple[np.ndarray, np.ndarray]:
    """Train/test indekslerini döndür (group/stratified-aware).

    - ``groups`` verilirse: ``GroupShuffleSplit`` (aynı grup hep aynı tarafta)
    - sınıflandırma + groups yok: ``StratifiedShuffleSplit``
    - regresyon + groups yok: ``ShuffleSplit``
    """
    if groups is not None:
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state,
        )
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
    elif task == "classification":
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state,
        )
        train_idx, test_idx = next(sss.split(X, y))
    else:
        ss = ShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state,
        )
        train_idx, test_idx = next(ss.split(X))
    return np.asarray(train_idx), np.asarray(test_idx)


def load_groups(
    dataset_dir: Path | str,
    *,
    key: str = "leaf",
) -> Optional[np.ndarray]:
    """01_dataset altından groups dizisini yükle.

    Parameters
    ----------
    dataset_dir : Path | str
        ``outputs/01_dataset`` veya benzeri klasör.
    key : {"leaf", "plot", "none"}
        - ``"leaf"`` → ``groups_leaf.npy`` (her yaprak farklı grup; varsayılan)
        - ``"plot"`` → ``groups_plot.npy`` (variety+plotLocation+collectionDate)
        - ``"none"`` → ``None`` döner (group split kapalı)
    """
    if key == "none":
        return None
    fname = f"groups_{key}.npy"
    path = Path(dataset_dir) / fname
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)
