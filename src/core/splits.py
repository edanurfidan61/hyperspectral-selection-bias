"""Dış bölme (outer split) ve seed üretimi — Revizyon 1 (plan §2.2).

Tüm deneyler dış bölmeleri BURADAN alır; böylece aynı ``(scheme, n_splits,
repeat)`` için tüm kollar ve tüm seçiciler BİREBİR aynı bölmeyi paylaşır
(eşli tasarım). Bölme yalnız ``n``, ``groups`` ve ``repeat``'e bağlıdır — X ve y'ye
bağlı DEĞİLDİR (GroupKFold/KFold/LeaveOneGroupOut zaten y kullanmaz).

Şemalar
-------
``"group"``  GroupKFold(n_splits, shuffle=True, random_state=repeat).
             ``shuffle=False`` → orijinal makale bölmesi (sklearn'in deterministik,
             grup-boyutu dengeleyen ataması; ``repeat`` yok sayılır). Geriye uyum:
             ``repeat=0, shuffle=False`` = ``core.cv.make_cv_splitter(groups=...)``.
``"plot"``   ``"group"`` ile aynı algoritma; çağıran ``groups`` olarak plot
             gruplarını verir (R3-9). Ayrı ad, çıktı tablolarında okunabilirlik için.
``"random"`` KFold(n_splits, shuffle=True, random_state=repeat) — grupları yok sayar (P1).
``"logo"``   LeaveOneGroupOut — ``n_splits`` ve ``repeat`` yok sayılır (potato, 24 Field).

Seed üretimi
------------
``derive_seed(base, repeat, fold, selector_id)`` →
``np.random.SeedSequence([base, repeat, fold, selector_id]).generate_state(1)[0]``.
Bağımsız, çakışmasız ve yeniden üretilebilir alt-akışlar verir. ``fold=-1``
"tüm veri" (apparent/P1/P2 seçimi) için ayrılmıştır.

Çalıştırma (duman testi)::

    python -m src.core.splits
"""

from __future__ import annotations

import hashlib
import zlib
from typing import Sequence

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

SCHEMES = ("group", "plot", "random", "logo")

Split = tuple[np.ndarray, np.ndarray]


def make_outer_splits(
    n: int,
    groups: np.ndarray | None,
    scheme: str = "group",
    n_splits: int = 5,
    repeat: int = 0,
    *,
    shuffle: bool = True,
) -> list[Split]:
    """Dış bölmeleri ``[(train_idx, test_idx), ...]`` listesi olarak üret.

    Parameters
    ----------
    n : int
        Örnek sayısı (NaN hedefler ÖNCEDEN atılmış olmalı).
    groups : ndarray | None
        Grup etiketleri; ``"random"`` dışında zorunlu.
    scheme : {"group", "plot", "random", "logo"}
    n_splits : int
        Fold sayısı (``"logo"``'da yok sayılır).
    repeat : int
        Tekrar indeksi = ``random_state`` (shuffle=True iken).
    shuffle : bool
        ``"group"/"plot"`` için. ``False`` → orijinal (shuffle'sız) GroupKFold.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"Bilinmeyen şema: {scheme!r}. Seçenekler: {SCHEMES}")
    idx = np.arange(int(n))
    if scheme == "random":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=int(repeat))
        return [(np.asarray(tr), np.asarray(te)) for tr, te in cv.split(idx)]

    if groups is None:
        raise ValueError(f"'{scheme}' şeması groups gerektirir")
    groups = np.asarray(groups)
    if len(groups) != n:
        raise ValueError(f"groups uzunluğu {len(groups)} ≠ n {n}")
    if scheme == "logo":
        cv = LeaveOneGroupOut()
    elif shuffle:
        cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=int(repeat))
    else:
        # Orijinal makale bölmesi: core.cv.make_cv_splitter ile birebir aynı çağrı.
        cv = GroupKFold(n_splits=n_splits)
    return [(np.asarray(tr), np.asarray(te)) for tr, te in cv.split(idx, groups=groups)]


def selector_id(name: str | int) -> int:
    """Seçici adını kararlı bir tamsayıya çevir (Python ``hash`` süreçler arası değişir)."""
    if isinstance(name, (int, np.integer)):
        return int(name)
    return zlib.crc32(str(name).encode("utf-8")) & 0x7FFFFFFF


def derive_seed(base: int, repeat: int, fold: int, sel: str | int) -> int:
    """``SeedSequence([base, repeat, fold, selector_id])`` → tek 32-bit seed.

    ``fold=-1`` tüm-veri seçimleri için kullanılır (SeedSequence negatif girdi
    kabul etmediği için içeride ``fold + 1`` kullanılır; eşleme birebirdir).
    """
    entropy = [int(base), int(repeat), int(fold) + 1, selector_id(sel)]
    if min(entropy) < 0:
        raise ValueError(f"negatif seed bileşeni: {entropy}")
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def splits_signature(splits: Sequence[Split]) -> str:
    """Bölmenin kısa özeti (SHA-1, 12 hane) — kollar/deneyler arası eşliği doğrulamak için."""
    h = hashlib.sha1()
    for k, (_tr, te) in enumerate(splits):
        h.update(f"{k}:".encode())
        h.update(np.sort(np.asarray(te, dtype=np.int64)).tobytes())
    return h.hexdigest()[:12]


if __name__ == "__main__":
    # Duman testi: geriye uyum + tekrar farklılığı + seed determinizmi
    from src.core.cv import make_cv_splitter

    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(24), 3)
    n = len(g)
    old = [te for _, te in make_cv_splitter(n_splits=5, task="regression", groups=g).split(np.zeros(n), np.zeros(n), g)]
    new = [te for _, te in make_outer_splits(n, g, "group", 5, repeat=0, shuffle=False)]
    assert all(np.array_equal(a, b) for a, b in zip(old, new)), "geriye uyum bozuk"
    s0 = splits_signature(make_outer_splits(n, g, "group", 5, repeat=0))
    s1 = splits_signature(make_outer_splits(n, g, "group", 5, repeat=1))
    assert s0 != s1, "tekrarlar aynı bölmeyi üretti"
    assert derive_seed(1, 2, 3, "lasso") == derive_seed(1, 2, 3, "lasso")
    assert derive_seed(1, 2, 3, "lasso") != derive_seed(1, 2, 4, "lasso")
    print("logo fold sayısı:", len(make_outer_splits(n, g, "logo")))
    print("splits.py duman testi OK  |  imza r0 =", s0, " r1 =", s1)
