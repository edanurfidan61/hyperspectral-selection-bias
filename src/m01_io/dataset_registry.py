"""Veri seti kayıt defteri — selection-bias çalışmasının çoklu-dataset omurgası.

Seçim-yanlılığının tek bir bitki/veri setine özgü olmadığını göstermek için,
tüm veri setlerini ortak bir arayüzde (``HSIDataset``) sunarız. Her dataset:
özellik matrisi ``X``, bir veya birden çok hedef (``targets``), CV için ``groups``,
``feature_names`` ve (varsa) ``wavelengths`` taşır.

Şimdilik sadece Ryckewaert (mevcut tez pipeline'ının 01_dataset çıktısı) kayıtlı.
Diğer veri setleri (örn. Nalepa klorofil) Adım 5'te eklenecek.

Çalıştırma (smoke test):
    python -m src.m01_io.dataset_registry
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.core import paths
from src.core.cv import load_groups
from src.core.logging_setup import get as get_logger

log = get_logger("m01_io.dataset_registry")


@dataclass
class HSIDataset:
    """Hiperspektral yaprak veri setinin ortak temsili.

    Attributes
    ----------
    name : str
        Kısa, benzersiz tanımlayıcı (örn. ``"ryckewaert"``).
    X : ndarray, shape (n_samples, n_features)
        Özellik matrisi.
    targets : dict[str, ndarray]
        Hedef adı → (n_samples,) hedef vektörü. NaN değerler regresyonda
        downstream filtrelenir.
    groups : ndarray | None, shape (n_samples,)
        GroupKFold için grup etiketleri (örn. yaprak kimliği). None ise
        gruplama kapalı.
    feature_names : list[str]
        ``X`` sütunlarının isimleri (len == n_features).
    wavelengths : ndarray | None
        Ham bant dalga boyları (nm). Öznitelik-türevli setlerde None olabilir.
    """

    name: str
    X: np.ndarray
    targets: dict[str, np.ndarray]
    groups: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    wavelengths: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = self.X.shape[0]
        for tname, tvec in self.targets.items():
            if len(tvec) != n:
                raise ValueError(
                    f"[{self.name}] hedef {tname!r} uzunluğu {len(tvec)} "
                    f"≠ n_samples {n}"
                )
        if self.groups is not None and len(self.groups) != n:
            raise ValueError(
                f"[{self.name}] groups uzunluğu {len(self.groups)} ≠ n_samples {n}"
            )
        if self.feature_names and len(self.feature_names) != self.X.shape[1]:
            raise ValueError(
                f"[{self.name}] feature_names uzunluğu {len(self.feature_names)} "
                f"≠ n_features {self.X.shape[1]}"
            )

    def target(self, name: str) -> np.ndarray:
        """Tek bir hedef vektörünü döndür (yoksa anlaşılır hata)."""
        if name not in self.targets:
            raise KeyError(
                f"[{self.name}] hedef {name!r} yok. Mevcut: {list(self.targets)}"
            )
        return self.targets[name]

    def __repr__(self) -> str:  # pragma: no cover - sadece insan-okuru
        return (
            f"HSIDataset(name={self.name!r}, X={self.X.shape}, "
            f"targets={list(self.targets)}, "
            f"groups={'yok' if self.groups is None else len(set(self.groups.tolist()))}, "
            f"wavelengths={'yok' if self.wavelengths is None else len(self.wavelengths)})"
        )


# ---------------------------------------------------------------------------
# Yükleyiciler
# ---------------------------------------------------------------------------
#: Ryckewaert 01_dataset hedef dosya adları → kanonik hedef adı.
_RYCKEWAERT_TARGETS = {
    "chlorophyll": "y_chl.npy",
    "flavonol": "y_flav.npy",
    "nbi": "y_nbi.npy",
}


def load_ryckewaert(
    dataset_dir: Path | str | None = None,
    *,
    group_key: str = "leaf",
) -> HSIDataset:
    """Mevcut tez pipeline'ının ``outputs/01_dataset`` çıktısını sarmala.

    Bu, GA modülündeki ``_load_dataset`` ile aynı dosyaları okur ama tüm
    hedefleri tek nesnede toplar (filtreleme yapmaz — NaN'ler korunur,
    downstream nested-CV her hedef için ayrı maskeler).

    Parameters
    ----------
    dataset_dir : Path | str | None
        01_dataset klasörü. None ise ``outputs/01_dataset``.
    group_key : {"leaf", "plot", "none"}
        CV grup anahtarı.

    Returns
    -------
    HSIDataset
    """
    ds_dir = Path(dataset_dir) if dataset_dir is not None \
        else paths.OUTPUTS_DIR / "01_dataset"

    x_path = ds_dir / "X.npy"
    fn_path = ds_dir / "feature_names.json"
    if not x_path.exists() or not fn_path.exists():
        raise FileNotFoundError(
            f"Ryckewaert dataset bulunamadı: {ds_dir}\n"
            f"Önce dataset aşamasını çalıştırın: python main.py --stages 01_dataset"
        )

    X = np.load(x_path)
    feature_names = json.loads(fn_path.read_text(encoding="utf-8"))
    groups = load_groups(ds_dir, key=group_key)

    targets: dict[str, np.ndarray] = {}
    for name, fname in _RYCKEWAERT_TARGETS.items():
        p = ds_dir / fname
        if p.exists():
            targets[name] = np.load(p)

    # Dalga boyları öznitelik-türevli sette zorunlu değil; varsa yükle.
    wl_path = ds_dir / "wavelengths.npy"
    wavelengths = np.load(wl_path) if wl_path.exists() else None

    return HSIDataset(
        name="ryckewaert",
        X=X,
        targets=targets,
        groups=groups,
        feature_names=feature_names,
        wavelengths=wavelengths,
    )


# ---------------------------------------------------------------------------
# Ortak yardımcılar (önbellek + ön işleme)
# ---------------------------------------------------------------------------
def _cache_path(name: str) -> Path:
    d = paths.OUTPUTS_DIR / "_datasets_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.npz"


def _preprocess_spectra(X: np.ndarray, savgol_window: int = 11) -> np.ndarray:
    """SavGol (gürültü) + SNV (saçılma) — bant ekseni boyunca.

    Bant sayısı SavGol penceresinden küçükse SavGol atlanır (örn. 6-bantlı
    multispektral). m02_preprocessing/spectral.py yeniden kullanılır.
    """
    from src.m02_preprocessing import spectral as sp

    Xp = np.asarray(X, dtype=float)
    n_bands = Xp.shape[1]
    if n_bands >= savgol_window:
        win = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
        Xp = sp.savitzky_golay(Xp, window_length=win, polyorder=2)
    Xp = sp.snv(Xp)
    return Xp


def _save_cache(name: str, ds: HSIDataset) -> None:
    payload = {
        "X": ds.X,
        "feature_names": np.array(ds.feature_names, dtype=object),
        "groups": np.array([]) if ds.groups is None else np.asarray(ds.groups, dtype=object),
        "target_names": np.array(list(ds.targets), dtype=object),
    }
    for t, v in ds.targets.items():
        payload[f"target__{t}"] = v
    np.savez(_cache_path(name), **payload)


def _load_cache(name: str) -> HSIDataset | None:
    p = _cache_path(name)
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    target_names = list(d["target_names"])
    targets = {t: d[f"target__{t}"] for t in target_names}
    groups = d["groups"]
    groups = None if groups.size == 0 else groups
    return HSIDataset(
        name=name, X=d["X"], targets=targets, groups=groups,
        feature_names=list(d["feature_names"]),
    )


# ---------------------------------------------------------------------------
# deep_patato — hiperspektral patates yaprağı küpleri (regresyon)
# ---------------------------------------------------------------------------
#: CHESS_FOLDS.CSV hedef sütunları → kanonik hedef adı. SPAD klorofil proxy'sidir.
_DEEP_PATATO_TARGETS = {
    "chlorophyll": "SPAD",
    "rwc": "RWC",
    "fvfm": "FvFm",
    "pi": "PI",
    "ndvi": "NDVI",
    "soil_moisture": "Soil_Moisture",
}


def load_deep_patato(
    data_dir: Path | str = "data/deep_patato",
    *,
    use_cache: bool = True,
    preprocess: bool = True,
) -> HSIDataset:
    """deep_patato: her örnek bir (150-bant) HSI küpü; maske-içi ortalama spektrum.

    Hedefler CHESS_FOLDS.CSV'den (SPAD=klorofil, RWC, FvFm, PI, NDVI, Soil_Moisture);
    gruplar tarla (``Field``) bazlı. Maske ile yaprak pikselleri seçilir, bant başına
    ortalama alınarak (150,) öznitelik vektörü üretilir.
    """
    import pandas as pd

    if use_cache and (cached := _load_cache("deep_patato")) is not None:
        return cached

    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "measurements" / "CHESS_FOLDS.CSV")
    img_dir = data_dir / "images"

    n_bands = None
    rows_X, groups, kept_idx = [], [], []
    for _, row in df.iterrows():
        idx = int(row["Sample_Index"])
        npz = img_dir / f"{idx}.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        cube = d["data"].astype(float)        # (B, H, W)
        mask = d["mask"].astype(bool)         # (B, H, W)
        B = cube.shape[0]
        # Bant başına maske-içi ortalama (maske boşsa o bant için tüm-piksel ort.)
        spec = np.empty(B, dtype=float)
        for b in range(B):
            mb = mask[b]
            spec[b] = cube[b][mb].mean() if mb.any() else cube[b].mean()
        if n_bands is None:
            n_bands = B
        elif B != n_bands:
            log.warning("deep_patato idx=%d bant sayısı %d ≠ %d, atlanıyor", idx, B, n_bands)
            continue
        rows_X.append(spec)
        groups.append(int(row["Field"]))
        kept_idx.append(idx)

    X = np.vstack(rows_X)
    if preprocess:
        X = _preprocess_spectra(X)

    df_kept = df.set_index("Sample_Index").loc[kept_idx]
    targets = {
        name: df_kept[col].to_numpy(dtype=float)
        for name, col in _DEEP_PATATO_TARGETS.items() if col in df_kept.columns
    }
    feature_names = [f"band_{i:03d}" for i in range(X.shape[1])]

    ds = HSIDataset(
        name="deep_patato", X=X, targets=targets,
        groups=np.asarray(groups), feature_names=feature_names,
    )
    _save_cache("deep_patato", ds)
    log.info("deep_patato yüklendi: %s", ds)
    return ds


# ---------------------------------------------------------------------------
# lambrusco — 6-bant multispektral; ikili FD/ESCA (sınıflandırma-tadında)
# ---------------------------------------------------------------------------
# NOT: lambrusco hedefleri İKİLİdir (FD = flavescence dorée, ESCA). Mevcut R²
# tabanlı nested_ga çerçevesi bunları 0/1 regresyon hedefi olarak ele alır;
# apparent-vs-nested uçurumu yine ölçülebilir ama yorumda "binary" olduğu
# unutulmamalı. Maskeleme yoktur (tam-görüntü istatistikleri) — arka plan gürültüsü
# bir miktar dahil olur; bias gösterimi için kabul edilebilir.
def load_lambrusco(
    data_dir: Path | str = "data/lambrusco",
    *,
    use_cache: bool = True,
    preprocess: bool = False,
) -> HSIDataset:
    """lambrusco: SET başına ikili etiket; her IMG grubu 6 grayscale bant.

    Öznitelik: bant başına {mean, std, p25, p75} (6×4=24). Hedef: FD, ESCA (0/1).
    Gruplar: SET kimliği (sızıntıyı önlemek için aynı SET'in görüntüleri birlikte).
    """
    import glob
    import json
    import re

    from PIL import Image

    if use_cache and (cached := _load_cache("lambrusco")) is not None:
        return cached

    data_dir = Path(data_dir)
    ann = json.loads((data_dir / "annotation.json").read_text(encoding="utf-8"))
    proc = data_dir / "processed"

    # SET kimliği → annotation (type/FD/ESCA). SET dizin adı "<key>SET".
    rows_X, fd, esca, groups = [], [], [], []
    for set_dir in sorted(proc.glob("*SET")):
        key = set_dir.name.replace("SET", "")
        meta = ann.get(key)
        if meta is None or meta.get("type") == "calibration_panel":
            continue
        # SET içindeki IMG gruplarını bul
        img_groups: dict[str, list[str]] = {}
        for p in glob.glob(str(set_dir / "IMG_*.png")):
            m = re.match(r"IMG_(\d+)_(\d+)\.png", Path(p).name)
            if m:
                img_groups.setdefault(m.group(1), []).append(p)
        for gid, files in img_groups.items():
            files = sorted(files, key=lambda f: int(re.search(r"_(\d+)\.png$", f).group(1)))
            if len(files) != 6:
                continue
            feats = []
            for f in files:
                arr = np.asarray(Image.open(f).convert("L"), dtype=float).ravel()
                feats.extend([arr.mean(), arr.std(),
                              np.percentile(arr, 25), np.percentile(arr, 75)])
            rows_X.append(feats)
            fd.append(1.0 if meta.get("FD") else 0.0)
            esca.append(1.0 if meta.get("ESCA") else 0.0)
            groups.append(key)

    X = np.asarray(rows_X, dtype=float)
    if preprocess:
        X = _preprocess_spectra(X)
    feature_names = [f"band{b}_{stat}" for b in range(6)
                     for stat in ("mean", "std", "p25", "p75")]
    ds = HSIDataset(
        name="lambrusco", X=X,
        targets={"fd": np.asarray(fd), "esca": np.asarray(esca)},
        groups=np.asarray(groups, dtype=object), feature_names=feature_names,
    )
    _save_cache("lambrusco", ds)
    log.info("lambrusco yüklendi: %s", ds)
    return ds


#: Dataset adı → sıfır-argümanlı yükleyici (REGISTRY[name]() çağrılır).
REGISTRY: dict[str, callable] = {
    "ryckewaert": load_ryckewaert,
    "deep_patato": load_deep_patato,
    "lambrusco": load_lambrusco,
}


def load(name: str) -> HSIDataset:
    """REGISTRY'den isimle dataset yükle."""
    if name not in REGISTRY:
        raise KeyError(f"Bilinmeyen dataset: {name!r}. Mevcut: {list(REGISTRY)}")
    return REGISTRY[name]()


if __name__ == "__main__":
    # Smoke test: 01_dataset çıktısı varsa yükle ve özetle; yoksa yapıyı doğrula.
    try:
        ds = load_ryckewaert()
        print("Yüklendi:", ds)
        for t, v in ds.targets.items():
            finite = int(np.isfinite(v).sum())
            print(f"  hedef {t:>11}: n={len(v)}  geçerli={finite}")
    except FileNotFoundError as exc:
        print("01_dataset çıktısı yok (beklenen — henüz üretilmedi):")
        print(" ", str(exc).splitlines()[0])
        # Yapısal doğrulama: sentetik HSIDataset kur
        rng = np.random.default_rng(0)
        ds = HSIDataset(
            name="dummy",
            X=rng.normal(size=(10, 4)),
            targets={"y": rng.normal(size=10)},
            groups=np.arange(10),
            feature_names=[f"f{i}" for i in range(4)],
        )
        print("Sentetik doğrulama OK:", ds)
    print("REGISTRY:", list(REGISTRY))
