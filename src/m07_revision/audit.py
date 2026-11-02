"""E0 — Denetim (audit): koşu gerektirmeyen, koddan ve veriden okunan bulgular.

Hakem 3 yorumları
R3-9, 10, 11, 13, 18–22, 24, 35, 37, 71 (+ Bölüm 1 tablosunun kodda doğrulanması).

Bu modül YENİ DENEY KOŞMAZ. Yaptığı tek hesaplama:
  * mevcut veri dosyalarından betimsel istatistik (dağılım, SVD, fold bileşimi),
  * mevcut kodun davranışını küçük, deterministik çağrılarla doğrulama
    (ör. dış GroupKFold'un seed'den bağımsız olduğu, LassoCV'nin seed'i
    kullanmadığı). Bunlar model başarımı ölçmez; başlık sayılarını etkilemez.

Çıktılar (``outputs_rev1/E0_audit/``)
------------------------------------
feature_audit.csv          520 özniteliğin her biri → blok, dönüşüm, kaynak bantlar
                           (hedef nm + gerçek nm), örnek-yerel mi, denklem.
flavonol_transforms.md     5 flavonol-türevli dönüşümün tam denklemi + gerçek nm.
continuum_removal.md       çapa noktaları, interpolasyon, eksik değer kuralı.
savgol_delta.md            savgol_filter ``delta`` değeri, bant aralığı (nm).
samples/groups.csv         örnek başına grup kimlikleri (grapevine leaf→plot,
                           potato örnek→Field). Örnek düzeyi → git-ignored
                           (``outputs_rev1/**/samples/``).
groups_summary.csv         grup başına örnek sayısı.
target_distributions.csv   hedef dağılımları variety/collectionDate/plot bazında
(+ .png)                   (R3-10).
fold_composition.csv       orijinal 5-fold bölmesinde fold başına grup/örnek (R3-13).
lasso_inner_cv.csv         LassoCV iç CV'sinin (cv=5 → düz KFold) grupları bölüp
                           bölmediği (R3-33).
svd.csv, svd_summary.csv   standardize sütunların tekil değerleri, %90/95/99
(+ .png)                   varyans için bileşen sayısı, etkin rank (R3-20).
seed_propagation.md        seed nereye gidiyor (R3-37).
section1_verification.md   plan Bölüm 1 tablosunun satır satır kod/veri kanıtı.
run_config.json            commit SHA, komut, sürümler, girdi SHA-256'ları, tarih.

Terminoloji (plan §0.8): "apparent", "nested", "apparent–nested difference".

Çalıştırma::

    .venv\\Scripts\\python.exe -m src.m07_revision.audit            # tam (figürlü)
    .venv\\Scripts\\python.exe -m src.m07_revision.audit --quick    # figürsüz, hızlı
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import paths

# ---------------------------------------------------------------------------
# Sabit yollar
# ---------------------------------------------------------------------------
ROOT = paths.ROOT
DS_DIR = ROOT / "outputs" / "01_dataset"
POTATO_NPZ = ROOT / "outputs" / "_datasets_cache" / "deep_patato.npz"
CHESS_CSV = ROOT / "data" / "deep_patato" / "measurements" / "CHESS_FOLDS.CSV"
DESC_TAB = ROOT / "data" / "ryckewaert" / "metadata" / "description-2.tab"
# Gerçek dalga boyları: 01_dataset'te wavelengths.npy YOK; öznitelik adları
# yuvarlanmış nm taşır. Gerçek nm için ilk (sıralı) yaprağın ENVI başlığı
# kaynak klasörden kopyalandı (builder dalga boylarını ilk yapraktan alır).
HDR_PATH = (ROOT / "data" / "ryckewaert" / "raw_hdr" / "2020-09-10_003" / "results"
            / "REFLECTANCE_2020-09-10_003.hdr")
OUT_DIR = ROOT / "outputs_rev1" / "E0_audit"

TARGETS_RYCK = {"chlorophyll": "y_chl.npy", "flavonol": "y_flav.npy", "nbi": "y_nbi.npy"}
SEEDS_PAPER = list(range(12))   # makaledeki çoklu-seed koşusu: 0..11


# ---------------------------------------------------------------------------
# Küçük yardımcılar
# ---------------------------------------------------------------------------
def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _where(rel_path: str, pattern: str) -> str:
    """Kaynak dosyada ``pattern`` (regex) geçen İLK satırı ``yol:satır`` olarak döndür.

    Satır numaraları sabit yazılmaz; kod değişirse kanıt referansı da kendiliğinden
    güncellenir. Bulunamazsa açıkça "BULUNAMADI" yazar (sessiz geçmez).
    """
    text = (ROOT / rel_path).read_text(encoding="utf-8").splitlines()
    rx = re.compile(pattern)
    for i, line in enumerate(text, start=1):
        if rx.search(line):
            return f"{rel_path}:{i}"
    return f"{rel_path}:BULUNAMADI({pattern})"


def _refs() -> dict[str, str]:
    """Rapor metinlerinde kullanılan kod kanıtı referansları (``yol:satır``).

    Önceden hesaplanır: Python 3.11'de f-string ifadesi içinde ters bölü
    (regex kaçışları) kullanılamaz.
    """
    spec = {
        "seed_cfg": ("src/m06_selection_bias/nested_ga.py", r'seed = cfg\["seed"\]'),
        "ga_seed": ("src/m04_features/ga_feature_selection.py", r"random\.seed\(seed\)"),
        "ga_cv": ("src/m04_features/ga_feature_selection.py", r"cv = make_cv_splitter"),
        "ga_lgbm": ("src/m04_features/ga_feature_selection.py", r"LGBMRegressor\(n_estimators=100, random_state=42"),
        "ga_pool": ("src/m04_features/ga_feature_selection.py", r"pool = mp\.Pool"),
        "ga_clip": ("src/m04_features/ga_feature_selection.py", r"return \(max\(mean_r2, 0\.0\),\)"),
        "mi": ("src/m04_features/selectors.py", r"mutual_info_regression\(X, y, random_state"),
        "lasso": ("src/m04_features/selectors.py", r"LassoCV\(cv=5"),
        "lasso_scaler": ("src/m04_features/selectors.py", r"Xs = StandardScaler\(\)\.fit_transform"),
        "rfe": ("src/m04_features/selectors.py", r"RFE\(estimator=Ridge"),
        "outer": ("src/core/cv.py", r"return GroupKFold\(n_splits=n_splits\)"),
        "p1": ("src/m06_selection_bias/cv_comparison.py", r"cv1 = KFold"),
        "boot": ("src/core/metrics_ci.py", r"random_state=np\.random\.default_rng\(seed\)"),
        "perm_rng": ("src/m06_selection_bias/synthetic_null.py", r"rng = np\.random\.default_rng\(seed\)"),
        "perm_p": ("src/m06_selection_bias/synthetic_null.py", r"perm_p_value = float\(\(1 \+ np\.sum"),
        "derived": ("src/m04_features/extraction.py", r"def _build_derived_maps"),
        "calc_nbi": ("src/m03_indices/indices.py", r"def calc_nbi\("),
        "builder_feat": ("src/m05_dataset/builder.py", r"feat = extract_features"),
        "builder_plot": ("src/m05_dataset/builder.py", r"groups_plot\[i\] = "),
        "potato_field": ("src/m01_io/dataset_registry.py", r"groups\.append\(int\(row\[.Field.\]\)\)"),
        "potato_sg": ("src/m01_io/dataset_registry.py", r"Xp = sp\.savitzky_golay"),
    }
    return {k: _where(p, rx) for k, (p, rx) in spec.items()}


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _read_wavelengths(hdr: Path) -> np.ndarray:
    """ENVI .hdr içindeki ``wavelength = {...}`` bloğunu oku (envi_loader ile aynı regex)."""
    content = hdr.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"wavelength\s*=\s*\{([^}]+)\}", content, re.IGNORECASE | re.DOTALL)
    if not m:
        raise RuntimeError(f"wavelength bloğu yok: {hdr}")
    return np.array([float(v) for v in m.group(1).replace("\n", "").split(",") if v.strip()])


def _md_table(df: pd.DataFrame) -> str:
    """tabulate bağımlılığı olmadan basit markdown tablo."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------
def load_inputs() -> dict:
    """Grapevine (Ryckewaert) + potato verisini ve meta-verisini yükle.

    Tutarlılık kontrolleri burada yapılır ve ``checks`` içine yazılır:
    X.npy == dataset_full.csv öznitelikleri; groups_plot.npy bölümlemesi ==
    description-2.tab'dan yeniden kurulan bölümleme; potato npz sırası ==
    CHESS_FOLDS.CSV sırası (Field + SPAD eşleşmesiyle).
    """
    from src.m05_dataset.ground_truth import load_ground_truth

    checks: dict = {}
    X = np.load(DS_DIR / "X.npy")
    fn = json.loads((DS_DIR / "feature_names.json").read_text(encoding="utf-8"))
    ys = {t: np.load(DS_DIR / f) for t, f in TARGETS_RYCK.items()}
    g_leaf = np.load(DS_DIR / "groups_leaf.npy", allow_pickle=True)
    g_plot = np.load(DS_DIR / "groups_plot.npy", allow_pickle=True)
    full = pd.read_csv(DS_DIR / "dataset_full.csv")

    # X.npy ↔ dataset_full.csv (aynı satır sırası, aynı değerler)
    checks["X_equals_dataset_full_csv"] = bool(
        np.allclose(full[fn].to_numpy(float), X, rtol=0, atol=1e-9))
    checks["y_flav_equals_csv_Flav"] = bool(np.allclose(full["Flav"].to_numpy(float), ys["flavonol"]))
    checks["groups_leaf_equals_filename"] = bool(
        (full["filename"].astype(str).to_numpy() == g_leaf.astype(str)).all())

    # description-2.tab → yaprak meta-verisi (builder ile aynı ayrıştırıcı)
    gt = load_ground_truth(DESC_TAB)
    gt = gt.drop_duplicates("filename_lower").set_index("filename_lower")
    keys = pd.Series(g_leaf.astype(str)).str.strip().str.lower()
    meta = gt.reindex(keys.to_numpy())
    checks["all_leaves_found_in_tab"] = bool(meta["variety"].notna().all())
    meta = meta.reset_index(drop=True)
    plot_rebuilt = (meta["variety"].astype(str).str.strip() + "|"
                    + meta["plotLocation"].astype(str).str.strip() + "|"
                    + meta["collectionDate"].astype(str).str.strip())
    # Karşılaştırma STRING değil BÖLÜMLEME üzerinden: kayıtlı groups_plot.npy eski
    # bir ayrıştırıcı sürümüyle (U+FFFD karakterli) üretildi; bölümleme aynıysa
    # analiz açısından eşdeğerdir.
    a = pd.factorize(g_plot.astype(str))[0]
    b = pd.factorize(plot_rebuilt.to_numpy())[0]
    checks["groups_plot_partition_matches_tab"] = bool(
        pd.crosstab(a, b).astype(bool).sum(axis=1).eq(1).all()
        and pd.crosstab(a, b).astype(bool).sum(axis=0).eq(1).all())
    checks["groups_plot_has_U+FFFD"] = bool(any("�" in s for s in g_plot.astype(str)))
    # Hedefler tab ile aynı mı (Dualex ölçümü)?
    checks["y_flav_equals_tab_Flav"] = bool(np.allclose(
        pd.to_numeric(meta["Flav"], errors="coerce").to_numpy(float), ys["flavonol"]))

    # Potato
    z = np.load(POTATO_NPZ, allow_pickle=True)
    Xp = np.asarray(z["X"], dtype=float)
    gp = np.asarray(z["groups"]).astype(int)
    yp = np.asarray(z["target__chlorophyll"], dtype=float)
    chess = pd.read_csv(CHESS_CSV)
    # Kayıtlı npz'de Sample_Index yok; CSV sırasıyla birebir mi?
    same_order = (len(chess) == len(gp)
                  and np.array_equal(chess["Field"].to_numpy(int), gp)
                  and np.allclose(chess["SPAD"].to_numpy(float), yp))
    checks["potato_npz_order_equals_chess_csv"] = bool(same_order)
    potato_sample_index = (chess["Sample_Index"].to_numpy(int) if same_order
                           else np.full(len(gp), -1))

    wl = _read_wavelengths(HDR_PATH)
    snv_nm = [int(n.split("R")[1]) for n in fn if n.startswith("snv_R")]
    checks["hdr_wavelengths_match_feature_names"] = bool(
        len(wl) == len(snv_nm) and all(int(round(w)) == s for w, s in zip(wl, snv_nm)))

    return {
        "X": X, "fn": fn, "ys": ys, "g_leaf": g_leaf, "g_plot": g_plot, "meta": meta,
        "Xp": Xp, "gp": gp, "yp": yp, "potato_fn": [str(s) for s in z["feature_names"]],
        "potato_sample_index": potato_sample_index, "chess": chess,
        "wl": wl, "checks": checks,
    }


# ---------------------------------------------------------------------------
# 1) feature_audit.csv + flavonol_transforms.md
# ---------------------------------------------------------------------------
def _index_bands_from_code() -> dict[str, list[int]]:
    """13 indeksin kullandığı hedef nm'leri KAYNAK KODDAN oku (elle yazma).

    ``calc_*`` fonksiyonlarının gövdesindeki ``_band(data, wavelengths, NNN)``
    çağrıları regex ile toplanır → kod değişirse denetim de değişir.
    """
    from src.m03_indices.indices import INDEX_FUNCS

    out = {}
    for name, fn in INDEX_FUNCS.items():
        src = inspect.getsource(fn)
        out[name] = [int(v) for v in re.findall(r"_band\(data,\s*wavelengths,\s*(\d+)\)", src)]
    return out


#: İndeks denklemleri — src/m03_indices/indices.py fonksiyon gövdelerinden.
#: (REP kodda Guyot–Baret'in standart (R670+R780)/2 biçiminden FARKLI yazılmış;
#: kodda olan yazılır, not düşülür.)
_INDEX_EQ = {
    "NDVI": "(R800-R670)/(R800+R670)",
    "GNDVI": "(R800-R550)/(R800+R550)",
    "ARI": "1/R550 - 1/R700",
    "RVSI": "(R714+R752)/2 - R733",
    "ZTM": "R750/R710",
    "CRI": "1/R510 - 1/R550",
    "PRI": "(R531-R570)/(R531+R570)",
    "mARI": "(1/R550 - 1/R700)*R800",
    "SIPI": "(R800-R445)/(R800-R680)",
    "WBI": "R900/R970",
    "REP": "700 + 40*((R700+R740)/2 - R710)/(R720 - R710)  [NOT: standart Guyot-Baret 1988 "
           "biçimi 700+40*((R670+R780)/2 - R700)/(R740-R700); kod farklı]",
    "BES": "(R450-R400)/50",
    "FLAVI": "R400/(R420*R700)",
}

#: 8 temel + 5 türetilmiş katman: (kaynak hedef nm, denklem)
_LAYER_DEF = {
    "R410": ([410], "R410 (en yakın bant yansıması)"),
    "R460": ([460], "R460"),
    "R670": ([670], "R670"),
    "R700": ([700], "R700"),
    "R800": ([800], "R800"),
    "FRI": ([410, 460], "FRI = 1/R410 - 1/R460  (1/x, x<=1e-10 ise 0)"),
    "NDRE": ([700, 800], "NDRE = (R800-R700)/(R800+R700)  (payda<=1e-10 ise 0)"),
    "NBI": ([410, 460, 700, 800], "NBI = NDRE/FRI  (|FRI|<=1e-10 ise 0) — SPEKTRAL, Dualex değil"),
    "logR460_R410": ([460, 410], "log(R460/R410 + 1e-10)  (oran<=1e-10 ise 0; |R410|<=1e-10 ise oran=0)"),
    "R670_R460": ([670, 460], "R670/R460  (|R460|<=1e-10 ise 0)"),
    "R800_R460": ([800, 460], "R800/R460  (|R460|<=1e-10 ise 0)"),
    "FRI_x_NDRE": ([410, 460, 700, 800], "FRI * NDRE = (1/R410 - 1/R460) * (R800-R700)/(R800+R700)"),
    "logNBI": ([410, 460, 700, 800], "log(max(NBI,0) + 1 + 1e-10)"),
}
FLAVONOL_DERIVED = ["logR460_R410", "R670_R460", "R800_R460", "FRI_x_NDRE", "logNBI"]

_STAT_EQ = {"mean": "ortalama", "std": "std (ddof=0)", "p25": "25. persentil", "p75": "75. persentil"}


def build_feature_audit(d: dict) -> pd.DataFrame:
    """520 özniteliği koddaki üretim sırasıyla (extraction.get_feature_names) eşle."""
    from src.core.spectral_utils import find_band
    from src.m04_features import extraction as ex

    wl = d["wl"]
    fn = d["fn"]
    X = d["X"]
    # Kodun kendi ad listesi ile kayıtlı ad listesi aynı mı (sıra dahil)?
    code_names = ex.get_feature_names(wl)
    d["checks"]["feature_names_equal_code_order"] = bool(code_names == fn)

    idx_bands = _index_bands_from_code()

    def nm_actual(targets: list[int]) -> str:
        return ";".join(f"{wl[find_band(wl, t)]:.2f}" for t in targets)

    rows = []
    for j, name in enumerate(fn):
        r = {"idx": j, "name": name}
        layer, stat = None, None
        if j < 52:  # A) 8 temel + 5 türetilmiş katman × 4 istatistik
            layer, stat = name.rsplit("_", 1)
            tnm, eq = _LAYER_DEF[layer]
            r.update(block="A_layer_stats",
                     transform=("flavonol-derived" if layer in FLAVONOL_DERIVED else
                                "raw_band" if layer.startswith("R") and layer[1:].isdigit()
                                else "spectral_index"),
                     preprocessing="ham yansıma (SNV yok), piksel başına harita → yaprak pikselleri üzerinde özet",
                     source_targets_nm=";".join(map(str, tnm)), source_actual_nm=nm_actual(tnm),
                     equation=f"{_STAT_EQ[stat]}[{eq}]")
        elif j < 104:  # B) 13 indeks × 4 istatistik
            layer, stat = name.rsplit("_", 1)
            tnm = idx_bands[layer]
            r.update(block="B_index_stats", transform="vegetation_index",
                     preprocessing="ham yansıma (SNV yok), piksel başına harita → yaprak pikselleri üzerinde özet",
                     source_targets_nm=";".join(map(str, tnm)), source_actual_nm=nm_actual(tnm),
                     equation=f"{_STAT_EQ[stat]}[{_INDEX_EQ[layer]}]")
        elif name.startswith("snv_R"):  # C) SNV yaprak-ortalama spektrumu
            b = j - 104
            r.update(block="C_snv_spectrum", transform="snv",
                     preprocessing="yaprak-ortalama spektrum → SNV (örnek içi: (x-mean)/std, 204 bant)",
                     source_targets_nm=str(int(round(wl[b]))), source_actual_nm=f"{wl[b]:.2f}",
                     equation="(m_b - mean_b(m)) / std_b(m),  m = yaprak-ortalama yansıma")
        elif name.startswith("d1snv_R"):  # D) SG 1. türev
            b = j - 104 - len(wl)
            r.update(block="D_sg_d1_snv", transform="savgol_deriv1",
                     preprocessing="SNV spektrumu → savgol_filter(window=11, polyorder=2, deriv=1, delta=1, mode='interp')",
                     source_targets_nm=str(int(round(wl[b]))),
                     source_actual_nm=";".join(f"{w:.2f}" for w in wl[max(0, b - 5):b + 6]),
                     equation="bant-indeksi başına türev (delta=1); nm başına değil")
        elif name.startswith("cont_BandI"):  # E) continuum removal
            r.update(block="E_continuum_bandI", transform="continuum_removal_linear",
                     preprocessing="yaprak-ortalama HAM spektrum (SNV öncesi), 410–510 nm",
                     source_targets_nm="410-510", source_actual_nm=f"{_band_i_nm(wl)[0]:.2f}-{_band_i_nm(wl)[1]:.2f}",
                     equation=("1 - min(R/baseline)" if name.endswith("depth")
                               else "trapz(max(1 - R/baseline, 0), nm)"))
        else:  # F) türev tepe noktaları
            region = name.split("_")[1]
            win = {"red": "680-760", "blue": "430-510", "all": "397-1004"}[region]
            r.update(block="F_d1_peaks", transform="argmax_abs_d1",
                     preprocessing="D bloğundaki d1 spektrumu",
                     source_targets_nm=win, source_actual_nm=win,
                     equation=("λ[argmax |d1| pencere içinde]" if name.endswith("_wl")
                               else "d1[argmax |d1| pencere içinde] (işaretli)"))
        r["layer"] = layer or ""
        r["stat"] = stat or ""
        # Tüm öznitelikler ÖRNEK-YEREL: extract_features tek yaprağın küpünden
        # hesaplanır, örnekler arası hiçbir adım (ölçekleme, PCA, ...) yoktur.
        r["sample_local"] = True
        col = X[:, j]
        r["col_std"] = float(np.std(col))
        r["n_unique"] = int(len(np.unique(col)))
        for t, y in d["ys"].items():
            r[f"r_{t}"] = float(np.corrcoef(col, y)[0, 1]) if np.std(col) > 0 else np.nan
        rows.append(r)

    df = pd.DataFrame(rows)
    order = ["idx", "name", "block", "layer", "stat", "transform", "preprocessing",
             "source_targets_nm", "source_actual_nm", "sample_local", "equation",
             "col_std", "n_unique", "r_chlorophyll", "r_flavonol", "r_nbi"]
    return df[order]


def _band_i_nm(wl: np.ndarray) -> tuple[float, float, int]:
    """Continuum-removal Band-I aralığına düşen gerçek bantlar: (ilk nm, son nm, sayı)."""
    from src.m04_features.extraction import _BAND_I_NM

    lo, hi = _BAND_I_NM
    idx = np.where((wl >= lo) & (wl <= hi))[0]
    return float(wl[idx[0]]), float(wl[idx[-1]]), int(idx.size)


def write_flavonol_md(d: dict, fa: pd.DataFrame, out: Path) -> None:
    from src.core.spectral_utils import find_band

    wl = d["wl"]
    rows = []
    for layer in FLAVONOL_DERIVED:
        tnm, eq = _LAYER_DEF[layer]
        sub = fa[fa["layer"] == layer]
        rows.append({
            "dönüşüm": layer,
            "denklem (piksel başına)": eq,
            "hedef nm": ", ".join(map(str, tnm)),
            "gerçek nm (bant indeksi)": ", ".join(
                f"{wl[find_band(wl, t)]:.2f} (#{find_band(wl, t)})" for t in tnm),
            "max |r| Flav (4 istatistik)": f"{sub['r_flavonol'].abs().max():.3f}",
            "max |r| NBI (4 istatistik)": f"{sub['r_nbi'].abs().max():.3f}",
        })
    tbl = pd.DataFrame(rows)
    ex_ref = _where("src/m04_features/extraction.py", r"def _build_derived_maps")
    nbi_ref = _where("src/m03_indices/indices.py", r"def calc_nbi\(")
    text = f"""# Flavonol-türevli 5 dönüşüm (R3-19)

Kaynak: `{ex_ref}`; NBI tanımı `{nbi_ref}`.
Bant seçimi `find_band` ile (hedef nm'ye en yakın bant; Specim IQ, 204 bant).
Gerçek nm değerleri ilk yaprağın ENVI başlığından (`{HDR_PATH.relative_to(ROOT).as_posix()}`).

{_md_table(tbl)}

## Hesaplama sırası
1. Ham yansıma küpünden (SNV/SG UYGULANMADAN) R410, R460, R670, R700, R800 piksel haritaları.
2. FRI, NDRE, NBI piksel haritaları (`calc_nbi_maps`), sonra yukarıdaki 5 dönüşüm.
3. Sonlu olmayan piksel değerleri 0'a çekilir (`np.where(np.isfinite(v), v, 0.0)`).
4. Her harita yaprak pikselleri üzerinde mean / std / p25 / p75 ile özetlenir → 5 × 4 = 20 öznitelik.

## Dualex ile ilişki
- Hiçbir dönüşüm Dualex ölçümünü (hedef `Flav`, `Chl`, `NBI`) girdi olarak kullanmaz;
  hepsi yalnız yansıma bantlarından hesaplanır. Öznitelik adındaki `NBI`, spektral
  `NDRE/FRI` oranıdır; Dualex NBI'si (= Chl/Flav) DEĞİLDİR.
- Hedeflerle en büyük mutlak korelasyon tabloda; |r| = 1 olan sütun yok
  (tüm 520 sütun için: bkz. `section1_verification.md`).
"""
    (out / "flavonol_transforms.md").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 2) continuum_removal.md + savgol_delta.md
# ---------------------------------------------------------------------------
def write_continuum_md(d: dict, out: Path) -> None:
    wl = d["wl"]
    lo_nm, hi_nm, nb = _band_i_nm(wl)
    fn = d["fn"]
    depth = d["X"][:, fn.index("cont_BandI_depth")]
    area = d["X"][:, fn.index("cont_BandI_area")]
    ref = _where("src/m04_features/extraction.py", r"def _continuum_band_i")
    alt = _where("src/m02_preprocessing/spectral.py", r"def continuum_removal")
    text = f"""# Continuum removal — Band-I (kodu okuyarak)

Kaynak: `{ref}` (öznitelik çıkarımında kullanılan); `{alt}` içindeki genel
`continuum_removal` (convex_hull/linear) öznitelik hattında **kullanılmıyor**.

| Öğe | Değer |
|---|---|
| Girdi spektrum | yaprak-ortalama **ham** yansıma (`_leaf_mean_spectrum`; SNV ve SG öncesi) |
| Aralık (kod) | `_BAND_I_NM = (410.0, 510.0)` nm, kapalı aralık `wl >= 410 & wl <= 510` |
| Aralıktaki gerçek bantlar | {nb} bant, {lo_nm:.2f} – {hi_nm:.2f} nm |
| Çapa noktaları | aralıktaki **ilk** ({lo_nm:.2f} nm) ve **son** ({hi_nm:.2f} nm) bant — sabit; yerel maksimum / dışbükey zarf aranmaz |
| İnterpolasyon | `np.linspace(sub[0], sub[-1], num=len(sub))` → bant **indeksi** üzerinde doğrusal (nm üzerinde değil; bant aralığı ~2.9–3.0 nm olduğundan fark küçük) |
| Continuum-removed | `cr = sub / baseline` |
| depth | `1 - min(cr)` |
| area | `trapezoid(max(1 - cr, 0), wl[idx])` — nm ekseninde integral |
| Eksik değer kuralı | aralıkta < 3 bant → (0, 0). Aralıkta sonlu olmayan değer **veya** uçlardan biri ≤ 0 ise: sonlu olmayan ya da ≤ 0 değerler `1e-9` ile değiştirilir. `baseline` alt sınırı `1e-9`. Son vektörde sonlu olmayan değerler 0'a çekilir. |
| Örnek-yerel mi | Evet (tek yaprağın spektrumu) |

## Veride gözlenen
- depth: min {depth.min():.4f}, medyan {np.median(depth):.4f}, max {depth.max():.4f};
  depth < 0 olan örnek sayısı: {int((depth < 0).sum())} / {len(depth)}
  (doğrusal taban dışbükey zarf olmadığı için `cr > 1` mümkündür → negatif depth).
- area: min {area.min():.4f}, medyan {np.median(area):.4f}, max {area.max():.4f};
  area = 0 olan örnek: {int((area == 0).sum())}.
"""
    (out / "continuum_removal.md").write_text(text, encoding="utf-8")


def write_savgol_md(d: dict, out: Path) -> dict:
    """SG türev ayarlarını yaz ve d1snv'yi snv sütunlarından YENİDEN HESAPLAYARAK doğrula."""
    from scipy.signal import savgol_filter

    from src.m02_preprocessing import spectral as sp

    wl = d["wl"]
    dl = np.diff(wl)
    fn, X = d["fn"], d["X"]
    s_idx = [i for i, n in enumerate(fn) if n.startswith("snv_R")]
    d_idx = [i for i, n in enumerate(fn) if n.startswith("d1snv_R")]
    S, D = X[:, s_idx], X[:, d_idx]
    # Yeniden hesap: kodun kullandığı çağrı (delta verilmez → 1.0)
    D_re = savgol_filter(S, window_length=11, polyorder=2, deriv=1, axis=1)
    max_diff = float(np.max(np.abs(D_re - D)))
    D_nm = savgol_filter(S, window_length=11, polyorder=2, deriv=1, delta=float(np.mean(dl)), axis=1)
    max_diff_nm = float(np.max(np.abs(D_nm - D)))
    # deriv=0 olsaydı d1snv ≈ snv olurdu: korelasyon kontrolü (config-truthfulness dalındaki iddia)
    corr = float(np.corrcoef(S.ravel(), D.ravel())[0, 1])
    std_ratio = float(np.std(S) / np.std(D))
    # SNV örnek-yerel mi: her satır ortalama 0, std 1
    snv_row_mean = float(np.max(np.abs(S.mean(axis=1))))
    snv_row_std = float(np.max(np.abs(S.std(axis=1) - 1)))

    sig = inspect.signature(sp.sg_first_derivative)
    sg_src = inspect.getsource(sp.savitzky_golay)
    delta_passed = "delta" in sg_src.split("savgol_filter(")[1].split(")")[0]
    # Potato: deriv=0 yumuşatma + SNV
    Xp = d["Xp"]
    p_row_mean = float(np.max(np.abs(Xp.mean(axis=1))))
    p_row_std = float(np.max(np.abs(Xp.std(axis=1) - 1)))

    res = {"d1_recompute_max_abs_diff_delta1": max_diff,
           "d1_recompute_max_abs_diff_delta_nm": max_diff_nm,
           "corr_snv_d1snv": corr, "std_ratio_snv_over_d1": std_ratio,
           "snv_rows_max_abs_mean": snv_row_mean, "snv_rows_max_abs_std_minus1": snv_row_std,
           "potato_rows_max_abs_mean": p_row_mean, "potato_rows_max_abs_std_minus1": p_row_std,
           "delta_passed_to_savgol_filter": bool(delta_passed)}

    text = f"""# Savitzky–Golay `delta` ve bant aralığı

## Grapevine (Ryckewaert) — `d1snv_R*` (204 öznitelik)
| Öğe | Değer | Kaynak |
|---|---|---|
| Fonksiyon | `sg_first_derivative(window_length={sig.parameters['window_length'].default}, polyorder={sig.parameters['polyorder'].default})` → `savitzky_golay(..., deriv=1)` | `{_where('src/m02_preprocessing/spectral.py', r'def sg_first_derivative')}` |
| `savgol_filter` çağrısı | `savgol_filter(spectra, window_length, polyorder, deriv, axis=1)` | `{_where('src/m02_preprocessing/spectral.py', r'result = savgol_filter')}` |
| `delta` | **verilmiyor → scipy varsayılanı `delta=1.0`** (türev bant-indeksi başına; nm başına DEĞİL) | aynı satır |
| `mode` | verilmiyor → varsayılan `'interp'` (kenar 5 bantta pencereye polinom uydurma) | aynı satır |
| `config/default.yaml` `preprocessing.savgol.deriv` | `0` yazıyor ama d1 için **okunmuyor**; `deriv=1` kodda sabit | `{_where('config/default.yaml', r'deriv:')}` |
| Girdi | yaprak-ortalama spektrumun SNV'si (`snv_R*`) | `{_where('src/m04_features/extraction.py', r'der_spec = sg_first_derivative')}` |

### Bant aralığı (Specim IQ, 204 bant, ilk yaprağın .hdr dosyası)
- Aralık: {wl[0]:.2f} – {wl[-1]:.2f} nm
- Δλ: min {dl.min():.3f}, medyan {np.median(dl):.3f}, ortalama {dl.mean():.3f}, max {dl.max():.3f} nm
- nm başına türeve dönüşüm: `dR/dλ ≈ d1snv / Δλ` (≈ d1snv / {dl.mean():.3f})

### Doğrulama (kayıtlı X.npy üzerinden, config'e güvenmeden)
- `snv_R*` sütunlarından `savgol_filter(11, 2, deriv=1)` (delta=1) ile yeniden hesap → kayıtlı `d1snv_R*` ile
  max |fark| = **{max_diff:.2e}**  (delta = ortalama Δλ ile: {max_diff_nm:.2e}) → **delta = 1 doğrulandı**.
- corr(snv, d1snv) = {corr:+.3f}, std(snv)/std(d1snv) = {std_ratio:.1f} → deriv = 1 (deriv = 0 olsaydı korelasyon ≈ +1 olurdu).
- SNV örnek-yerel: satır ortalaması max |·| = {snv_row_mean:.1e}, satır std − 1 max |·| = {snv_row_std:.1e}.

## Potato (deep_patato) — 150 kanal
- `_preprocess_spectra`: `savitzky_golay(window=11, polyorder=2)` → **deriv=0 (yumuşatma)**, ardından SNV
  (`{_where('src/m01_io/dataset_registry.py', r'def _preprocess_spectra')}`). `delta` deriv=0'da etkisizdir.
- Kayıtlı önbellek doğrulaması: satır ortalaması max |·| = {p_row_mean:.1e}, satır std − 1 max |·| = {p_row_std:.1e}
  → SNV uygulanmış, örnek-yerel. Kanallar "ham" değildir (R3-22: makaledeki "raw channels" ifadesi düzeltilmeli).
- Potato kanal dalga boyları önbellekte yok (öznitelik adları `band_000`…`band_149`).
"""
    (out / "savgol_delta.md").write_text(text, encoding="utf-8")
    return res


# ---------------------------------------------------------------------------
# 3) groups.csv, groups_summary.csv, target_distributions.csv(+png)
# ---------------------------------------------------------------------------
def build_groups(d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = d["meta"]
    g = pd.DataFrame({
        "dataset": "ryckewaert",
        "sample_idx": np.arange(len(d["g_leaf"])),
        "group_id": d["g_leaf"].astype(str),          # analizde kullanılan grup = yaprak
        "leaf_id": d["g_leaf"].astype(str),
        "image_id": m["imageID"].astype(str).to_numpy(),
        "plot_group": d["g_plot"].astype(str),        # variety|plotLocation|collectionDate
        "variety": m["variety"].astype(str).to_numpy(),
        "plot_location": m["plotLocation"].astype(str).to_numpy(),
        "collection_date": m["collectionDate"].astype(str).to_numpy(),
        "symptom": m["symptom"].astype(str).to_numpy(),
    })
    p = pd.DataFrame({
        "dataset": "deep_patato",
        "sample_idx": np.arange(len(d["gp"])),
        "group_id": d["gp"].astype(str),              # analizde kullanılan grup = Field
        "sample_index_csv": d["potato_sample_index"],
        "field": d["gp"],
        "flight_date": d["chess"]["Flight_Date"].astype(str).to_numpy()
        if d["checks"]["potato_npz_order_equals_chess_csv"] else "",
    })
    groups = pd.concat([g, p], ignore_index=True)

    summ = []
    for name, col, df in [("ryckewaert", "group_id", g), ("ryckewaert", "plot_group", g),
                          ("ryckewaert", "variety", g), ("ryckewaert", "collection_date", g),
                          ("deep_patato", "group_id", p)]:
        vc = df[col].value_counts().sort_index()
        level = {"group_id": "analysis_group(leaf)" if name == "ryckewaert" else "analysis_group(Field)",
                 }.get(col, col)
        if name == "ryckewaert" and col == "group_id":
            # 204 tekil yaprak — tek satıra özetle
            summ.append({"dataset": name, "level": level, "group": "(her yaprak)",
                         "n_samples": int(vc.iloc[0]) if vc.nunique() == 1 else -1,
                         "n_groups_at_level": int(len(vc))})
            continue
        for k, v in vc.items():
            summ.append({"dataset": name, "level": level, "group": str(k),
                         "n_samples": int(v), "n_groups_at_level": int(len(vc))})
    return groups, pd.DataFrame(summ)


def build_target_distributions(d: dict, groups: pd.DataFrame) -> pd.DataFrame:
    g = groups[groups.dataset == "ryckewaert"].reset_index(drop=True)
    rows = []
    for t, y in d["ys"].items():
        for by in ["ALL", "variety", "collection_date", "plot_group", "plot_location", "symptom"]:
            keys = pd.Series(["ALL"] * len(y)) if by == "ALL" else g[by]
            for k, idx in keys.groupby(keys).groups.items():
                v = y[np.asarray(list(idx))]
                rows.append({"dataset": "ryckewaert", "target": t, "by": by, "level": k,
                             "n": len(v), "mean": v.mean(), "std": v.std(ddof=1) if len(v) > 1 else np.nan,
                             "min": v.min(), "q25": np.quantile(v, .25), "median": np.median(v),
                             "q75": np.quantile(v, .75), "max": v.max()})
    yp, gp = d["yp"], d["gp"]
    for by in ["ALL", "field"]:
        keys = pd.Series(["ALL"] * len(yp)) if by == "ALL" else pd.Series(gp)
        for k, idx in keys.groupby(keys).groups.items():
            v = yp[np.asarray(list(idx))]
            rows.append({"dataset": "deep_patato", "target": "chlorophyll(SPAD)", "by": by, "level": k,
                         "n": len(v), "mean": v.mean(), "std": v.std(ddof=1) if len(v) > 1 else np.nan,
                         "min": v.min(), "q25": np.quantile(v, .25), "median": np.median(v),
                         "q75": np.quantile(v, .75), "max": v.max()})
    return pd.DataFrame(rows)


def plot_target_distributions(d: dict, groups: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = groups[groups.dataset == "ryckewaert"].reset_index(drop=True)
    levels = sorted(g["plot_group"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)
    for ax, (t, y) in zip(axes, d["ys"].items()):
        data = [y[(g["plot_group"] == lv).to_numpy()] for lv in levels]
        ax.boxplot(data, vert=False)
        ax.set_yticks(range(1, len(levels) + 1))
        ax.set_yticklabels([lv.replace("�", "?") for lv in levels], fontsize=7)
        ax.set_title(f"{t} by plot group (variety|location|date)", fontsize=9)
        ax.set_xlabel(t)
    plt.tight_layout()
    fig.savefig(out / "target_distributions.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4) fold_composition.csv + lasso_inner_cv.csv
# ---------------------------------------------------------------------------
def _original_outer_splits(X, y, groups, seed: int = 42):
    """Makaledeki dış bölme: nested_ga.nested_evaluation ile AYNI çağrı."""
    from src.core.cv import make_cv_splitter

    cv = make_cv_splitter(n_splits=5, task="regression", groups=groups, random_state=seed)
    return [(np.asarray(tr), np.asarray(te)) for tr, te in cv.split(X, y, groups)]


def build_fold_composition(d: dict, groups: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    g = groups[groups.dataset == "ryckewaert"].reset_index(drop=True)
    rows, checks = [], {}
    y = d["ys"]["flavonol"]
    sp = _original_outer_splits(d["X"], y, d["g_leaf"])
    # Seed'den bağımsızlık: makalenin 12 seed'i için dış bölme aynı mı?
    same = all(all(np.array_equal(a[1], b[1]) for a, b in
                   zip(sp, _original_outer_splits(d["X"], y, d["g_leaf"], seed=s)))
               for s in SEEDS_PAPER)
    checks["outer_split_identical_across_seeds_ryck"] = bool(same)
    # Tüm hedefler sonlu → üç hedefte de bölme aynı
    checks["ryck_targets_all_finite"] = bool(all(np.isfinite(v).all() for v in d["ys"].values()))
    for k, (tr, te) in enumerate(sp):
        gt = g.iloc[te]
        rows.append({
            "dataset": "ryckewaert", "group_level": "leaf", "fold": k, "n_train": len(tr),
            "n_test": len(te), "n_test_groups": gt["leaf_id"].nunique(),
            "n_test_plot_groups": gt["plot_group"].nunique(),
            "n_plot_groups_shared_with_train": len(set(gt["plot_group"]) & set(g.iloc[tr]["plot_group"])),
            "test_varieties": "; ".join(f"{a}:{b}" for a, b in gt["variety"].value_counts().sort_index().items()),
            "test_dates": "; ".join(f"{a}:{b}" for a, b in gt["collection_date"].value_counts().sort_index().items()),
            "test_flavonol_mean": float(d["ys"]["flavonol"][te].mean()),
            "test_chlorophyll_mean": float(d["ys"]["chlorophyll"][te].mean()),
            "test_sample_idx_range": f"{te.min()}-{te.max()}",
        })
    spp = _original_outer_splits(d["Xp"], d["yp"], d["gp"])
    samep = all(all(np.array_equal(a[1], b[1]) for a, b in
                    zip(spp, _original_outer_splits(d["Xp"], d["yp"], d["gp"], seed=s)))
                for s in SEEDS_PAPER)
    checks["outer_split_identical_across_seeds_potato"] = bool(samep)
    for k, (tr, te) in enumerate(spp):
        fields = sorted(set(d["gp"][te].tolist()))
        rows.append({
            "dataset": "deep_patato", "group_level": "Field", "fold": k, "n_train": len(tr),
            "n_test": len(te), "n_test_groups": len(fields),
            "n_test_plot_groups": np.nan, "n_plot_groups_shared_with_train": np.nan,
            "test_varieties": "", "test_dates": "",
            "test_flavonol_mean": np.nan, "test_chlorophyll_mean": float(d["yp"][te].mean()),
            "test_sample_idx_range": "fields=" + ",".join(map(str, fields)),
        })
    return pd.DataFrame(rows), checks


def build_lasso_inner_cv(d: dict) -> pd.DataFrame:
    """LassoCV(cv=5) iç bölmesinin grupları bölüp bölmediği (R3-33).

    ``LassoCV(cv=5)`` regresyonda ``check_cv`` → ``KFold(5, shuffle=False)``:
    satır SIRASINA göre ardışık bloklar, grup bilgisi YOK. Burada select_lasso'ya
    gerçekten verilen X'ler (apparent: tüm veri; nested: her dış-train) için iç
    fold'ları üretip kaç grubun birden fazla iç fold'a dağıldığını sayıyoruz.
    """
    from sklearn.model_selection import check_cv

    rows = []
    inner = check_cv(5, d["yp"], classifier=False)
    rows_meta = [("ryckewaert", "leaf", d["g_leaf"].astype(str)),
                 ("ryckewaert", "plot_group", d["g_plot"].astype(str)),
                 ("deep_patato", "Field", d["gp"].astype(str))]
    for ds, level, grp in rows_meta:
        if ds == "ryckewaert":
            X, y, outer_g = d["X"], d["ys"]["flavonol"], d["g_leaf"]
        else:
            X, y, outer_g = d["Xp"], d["yp"], d["gp"]
        scopes = [("apparent(full)", np.arange(len(y)))]
        scopes += [(f"nested_outer_train_fold{k}", tr)
                   for k, (tr, _te) in enumerate(_original_outer_splits(X, y, outer_g))]
        for scope, idx in scopes:
            gsub = grp[idx]
            fold_of = np.empty(len(idx), dtype=int)
            for f, (_tr, te) in enumerate(inner.split(X[idx], y[idx])):
                fold_of[te] = f
            df = pd.DataFrame({"g": gsub, "f": fold_of})
            nf = df.groupby("g")["f"].nunique()
            # bir grubun en az bir örneği iç-test'te, en az bir örneği iç-train'de olan
            # (iç fold, grup) çifti sayısı = iç sızıntı fırsatı
            leak_pairs = int(sum(((df.f == f) & (df.g == gg)).any() and ((df.f != f) & (df.g == gg)).any()
                                 for f in range(5) for gg in nf.index[nf > 1]))
            rows.append({"dataset": ds, "group_level": level, "scope": scope,
                         "inner_cv": repr(inner), "n_rows": len(idx),
                         "n_groups": int(nf.size), "n_groups_split_across_inner_folds": int((nf > 1).sum()),
                         "inner_fold_group_leak_pairs": leak_pairs})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5) svd.csv, svd_summary.csv (+png)
# ---------------------------------------------------------------------------
def build_svd(d: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, summ, Z_keep = [], [], {}
    for ds, X in [("ryckewaert", d["X"]), ("deep_patato", d["Xp"])]:
        sd = X.std(axis=0)
        keep = sd > 0
        Z = (X[:, keep] - X[:, keep].mean(axis=0)) / sd[keep]
        Z_keep[ds] = (Z, keep)
        s = np.linalg.svd(Z, compute_uv=False)
        lam = s ** 2
        evr = lam / lam.sum()
        cum = np.cumsum(evr)
        for i, (sv, e, c) in enumerate(zip(s, evr, cum), start=1):
            rows.append({"dataset": ds, "component": i, "singular_value": sv,
                         "explained_var_ratio": e, "cumulative": c})
        p = s / s.sum()
        p = p[p > 0]
        summ.append({
            "dataset": ds, "n_samples": X.shape[0], "n_features": X.shape[1],
            "n_constant_dropped": int((~keep).sum()),
            "constant_columns": ";".join(np.array(d["fn"] if ds == "ryckewaert" else d["potato_fn"])[~keep]),
            "numerical_rank": int(np.linalg.matrix_rank(Z)),
            "n_comp_90": int(np.searchsorted(cum, 0.90) + 1),
            "n_comp_95": int(np.searchsorted(cum, 0.95) + 1),
            "n_comp_99": int(np.searchsorted(cum, 0.99) + 1),
            # Katılım oranı (participation ratio) — özdeğerler λ = s² üzerinden
            "participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()),
            # Entropi-etkin rank (Roy & Vetterli 2007) — tekil değerler üzerinden
            "entropy_effective_rank": float(np.exp(-(p * np.log(p)).sum())),
            "frac_abs_corr_gt_0.9": float(np.mean(np.abs(np.corrcoef(Z, rowvar=False))[np.triu_indices(Z.shape[1], 1)] > 0.9)),
        })
    return pd.DataFrame(rows), pd.DataFrame(summ), Z_keep


def plot_svd(d: dict, svd: pd.DataFrame, Z_keep: dict, fa: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for ds, sub in svd.groupby("dataset"):
        ax[0].semilogy(sub["component"], sub["singular_value"], label=ds)
        ax[1].plot(sub["component"], sub["cumulative"], label=ds)
    for v in (0.90, 0.95, 0.99):
        ax[1].axhline(v, color="gray", lw=0.6, ls="--")
    ax[0].set_xlabel("component"); ax[0].set_ylabel("singular value (standardized X)")
    ax[1].set_xlabel("component"); ax[1].set_ylabel("cumulative explained variance")
    ax[1].set_xlim(0, 60)
    for a in ax:
        a.legend()
    plt.tight_layout(); fig.savefig(out / "svd.png", dpi=130); plt.close(fig)

    Z, keep = Z_keep["ryckewaert"]
    C = np.corrcoef(Z, rowvar=False)
    fig, axh = plt.subplots(figsize=(8.5, 7.5))
    im = axh.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    blocks = fa.loc[keep, "block"].to_numpy()
    edges = [i for i in range(1, len(blocks)) if blocks[i] != blocks[i - 1]]
    for e in edges:
        axh.axhline(e - 0.5, color="k", lw=0.5); axh.axvline(e - 0.5, color="k", lw=0.5)
    axh.set_title("Ryckewaert: correlation of standardized features (block boundaries)", fontsize=9)
    fig.colorbar(im, ax=axh, fraction=0.046)
    plt.tight_layout(); fig.savefig(out / "correlation_heatmap.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# 6) Davranış doğrulamaları (seed, fitness, seçiciler) — model başarımı ÖLÇMEZ
# ---------------------------------------------------------------------------
def behaviour_checks(d: dict, quick: bool) -> dict:
    """Bölüm 1 tablosundaki davranış iddialarını küçük çağrılarla doğrula.

    Ucuz olsun diye potato (72×150) kullanılır; sonuçlar seçici koduna özgüdür,
    veri setine değil. ``quick`` iken seçici çağrıları atlanır.
    """
    import warnings

    from src.m04_features import ga_feature_selection as gfs
    from src.m04_features import selectors as sel

    out: dict = {}
    Xp, yp, gp = d["Xp"], d["yp"], d["gp"]

    # GA fitness: MIN_FEATURES altı 0; negatif R² 0'a kırpılıyor mu?
    gfs._init_worker(Xp, yp, "pls", 0, gp)
    ind_small = [1] * (gfs.MIN_FEATURES - 1) + [0] * (Xp.shape[1] - gfs.MIN_FEATURES + 1)
    out["ga_fitness_below_min_features"] = gfs._evaluate_individual(ind_small)[0]
    out["ga_MIN_FEATURES"] = int(gfs.MIN_FEATURES)
    rng = np.random.default_rng(0)
    gfs._init_worker(Xp, rng.permutation(yp), "pls", 0, gp)   # sinyalsiz y → R² ≤ 0 beklenir
    vals = [gfs._evaluate_individual(list(rng.integers(0, 2, Xp.shape[1])))[0] for _ in range(10)]
    out["ga_fitness_null_y_min_over_10"] = float(min(vals))
    out["ga_fitness_null_y_n_zero_over_10"] = int(sum(v == 0.0 for v in vals))

    # GA iç CV: make_cv_splitter(groups=...) → GroupKFold, seed'den bağımsız
    from src.core.cv import make_cv_splitter
    a = [te for _, te in make_cv_splitter(n_splits=5, task="regression", groups=gp, random_state=0).split(Xp, yp, gp)]
    b = [te for _, te in make_cv_splitter(n_splits=5, task="regression", groups=gp, random_state=7).split(Xp, yp, gp)]
    out["ga_inner_cv_type"] = type(make_cv_splitter(n_splits=5, task="regression", groups=gp)).__name__
    out["ga_inner_cv_seed_independent"] = bool(all(np.array_equal(x, y) for x, y in zip(a, b)))
    out["ga_inner_cv_splits_groups"] = bool(any(
        len(set(gp[te]) & set(np.delete(gp, te))) > 0 for te in a))

    # sklearn GroupKFold(shuffle=True) var mı (tekrarlı bölme için)?
    from sklearn.model_selection import GroupKFold
    out["groupkfold_has_shuffle"] = "shuffle" in inspect.signature(GroupKFold).parameters

    if not quick:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m0 = sel.select_lasso(Xp, yp, 20, seed=0, inner="legacy")
            m1 = sel.select_lasso(Xp, yp, 20, seed=1, inner="legacy")
            out["lasso_mask_seed_independent"] = bool(np.array_equal(m0, m1))
            out["lasso_n_selected"] = int(m0.sum())
            r0 = sel.select_rfe(Xp, yp, 20, seed=0)
            r1 = sel.select_rfe(Xp, yp, 20, seed=1)
            out["rfe_mask_seed_independent"] = bool(np.array_equal(r0, r1))
            i0 = sel.select_mutual_info(Xp, yp, 20, seed=0)
            i1 = sel.select_mutual_info(Xp, yp, 20, seed=1)
            out["mi_mask_seed_independent"] = bool(np.array_equal(i0, i1))
            out["mi_jaccard_seed0_seed1"] = float((i0 & i1).sum() / (i0 | i1).sum())
    return out


# ---------------------------------------------------------------------------
# 7) seed_propagation.md + section1_verification.md
# ---------------------------------------------------------------------------
def write_seed_md(beh: dict, checks: dict, out: Path) -> None:
    R = _refs()
    ok = lambda b: "✓" if b else "✗"  # noqa: E731
    lasso_line = (f"{ok(beh['lasso_mask_seed_independent'])} seed 0/1 maskesi aynı"
                  if "lasso_mask_seed_independent" in beh else "(quick: atlandı)")
    mi_line = (f"seed 0/1 maske Jaccard = {beh['mi_jaccard_seed0_seed1']:.2f}"
               if "mi_jaccard_seed0_seed1" in beh else "(quick: atlandı)")
    rfe_line = (f"{ok(beh['rfe_mask_seed_independent'])} seed 0/1 maskesi aynı"
                if "rfe_mask_seed_independent" in beh else "(quick: atlandı)")
    text = f"""# Seed nereye gidiyor (R3-37)

`nested_evaluation` içinde `seed = cfg["seed"]` (`{R['seed_cfg']}`);
`sel_kwargs` (seed dahil) seçiciye iletilir. Çoklu-seed koşusu: seed ∈ {{0, …, 11}}.

| Bileşen | Seed kullanılıyor mu | Kanıt | Doğrulama |
|---|---|---|---|
| GA evrimi (başlangıç popülasyonu, seçim, çaprazlama, mutasyon) | **Evet** — ana süreçte `random.seed(seed)`, `np.random.seed(seed)` | `{R['ga_seed']}` | işçiler (mp.Pool) yalnız deterministik fitness hesaplar → tek RNG akışı |
| GA fitness (iç CV) | Hayır — `make_cv_splitter(groups=…)` → `{beh['ga_inner_cv_type']}` (shuffle yok) | `{R['ga_cv']}` | {ok(beh['ga_inner_cv_seed_independent'])} seed 0 ve 7 ile iç fold'lar aynı; grup bölünmesi: {beh['ga_inner_cv_splits_groups']} |
| GA modelleri | PLS deterministik; LightGBM `random_state=42` sabit (seed'e bağlı DEĞİL) | `{R['ga_lgbm']}` | — |
| MI (`mutual_info_regression`) | **Evet** — `random_state=seed` (kNN tahmincisine eklenen gürültü) | `{R['mi']}` | {mi_line} |
| LASSO (`LassoCV`) | **Etkisiz** — `random_state=seed` veriliyor ama `selection='cyclic'` (varsayılan) iken kullanılmaz; iç CV `cv=5` → `KFold(shuffle=False)` | `{R['lasso']}` | {lasso_line} |
| RFE (Ridge) | Hayır — deterministik | `{R['rfe']}` | {rfe_line} |
| Dış CV (nested ve apparent) | **Etkisiz** — `GroupKFold(n_splits=5)` shuffle'sız; `random_state` yok sayılır | `{R['outer']}` | {ok(checks['outer_split_identical_across_seeds_ryck'])} grapevine, {ok(checks['outer_split_identical_across_seeds_potato'])} potato: 12 seed'de dış fold'lar birebir aynı |
| P1 (rastgele KFold) | **Evet** — `KFold(shuffle=True, random_state=seed)` | `{R['p1']}` | — |
| Bootstrap CI | **Evet** — `np.random.default_rng(seed)` | `{R['boot']}` | — |
| Permütasyon | **Evet** — `np.random.default_rng(seed)` → `rng.permutation(y)` | `{R['perm_rng']}` | — |
| Sentetik null | tekrar i → seed = 1000 + i (`run_null_n200.py`, BASE_SEED = 1000) | `run_null_n200.py` | — |

## Sonuç
- Seed'ler arasında değişen tek şey **seçici içindeki rastgelelik**tir (GA evrimi, MI tahmini).
  LASSO ve RFE seed'den bağımsızdır → 12 seed'de apparent–nested farkı sabit (Wilcoxon için etkin n = 1;
  `stats_tests.is_degenerate` bunu işaretliyor).
- Dış bölme 12 seed'in hepsinde AYNIDIR; seed varyasyonu bölme değişkenliğini İÇERMEZ.
  Bölme değişkenliği için `GroupKFold(shuffle=True, random_state=r)` gerekir
  (sklearn 1.8.0'da mevcut: {ok(beh['groupkfold_has_shuffle'])}).
- `nested_evaluation` docstring'indeki "seed ayrıca dış/iç CV'ye akar" ifadesi grup varken
  geçersizdir (GroupKFold seed kullanmaz) — metinde düzeltilmeli.
"""
    (out / "seed_propagation.md").write_text(text, encoding="utf-8")


def write_section1_md(d: dict, fa: pd.DataFrame, sg: dict, beh: dict, lasso_cv: pd.DataFrame,
                      fold_df: pd.DataFrame, out: Path) -> list[dict]:
    """Plan Bölüm 1 tablosunu satır satır kanıtla. Her satır: durum + kanıt."""
    R = _refs()
    c = d["checks"]
    r_max = fa[["r_chlorophyll", "r_flavonol", "r_nbi"]].abs().max()
    # Dualex hedef adları birebir (Flav/Chl/NBI) ya da "dualex" geçen ad. FLAVI_* (spektral
    # indeks R400/(R420·R700)) ve NBI_* (spektral NDRE/FRI) yansımadan gelir → hariç.
    bad_names = [n for n in d["fn"] if re.search(r"dualex|^(flav|chl|nbi)$", n, re.I)]
    pot_leak = lasso_cv[(lasso_cv.dataset == "deep_patato")]
    ryck_leaf = lasso_cv[(lasso_cv.dataset == "ryckewaert") & (lasso_cv.group_level == "leaf")]
    ryck_plot = lasso_cv[(lasso_cv.dataset == "ryckewaert") & (lasso_cv.group_level == "plot_group")]
    n_fields = int(len(np.unique(d["gp"])))
    per_field = sorted(set(pd.Series(d["gp"]).value_counts().tolist()))

    V, F, N = "DOĞRULANDI", "DÜZELTME", "NOT"
    rows = [
        {"konu": "R3-19 flavonol-türevli 5 dönüşüm",
         "durum": V if (max(r_max) < 0.999 and not bad_names and c["feature_names_equal_code_order"]) else F,
         "kanıt": f"{R['derived']}; 5 dönüşüm yalnız yansımadan; "
                  f"NBI = NDRE/FRI ({R['calc_nbi']}). "
                  f"builder X'e hedef eklemiyor ({R['builder_feat']}). "
                  f"Veri: 520 sütunda max|r| Chl={r_max['r_chlorophyll']:.3f}, Flav={r_max['r_flavonol']:.3f}, "
                  f"NBI={r_max['r_nbi']:.3f} (hiçbiri 1 değil); Dualex/Flav/Chl adlı sütun: {len(bad_names)}; "
                  f"öznitelik adları kod sırasıyla aynı: {c['feature_names_equal_code_order']}."},
        {"konu": "R3-33 LassoCV iç CV",
         "durum": V,
         "kanıt": f"{R['lasso']}: `cv=5` → `KFold(5, shuffle=False)`, grup yok. "
                  f"Potato: iç fold'lara bölünen Field sayısı apparent={int(pot_leak.iloc[0].n_groups_split_across_inner_folds)}/{n_fields}, "
                  f"nested dış-train başına {pot_leak.iloc[1:].n_groups_split_across_inner_folds.tolist()} → iç sızıntı VAR. "
                  f"Grapevine leaf: {int(ryck_leaf.n_groups_split_across_inner_folds.max())} (her leaf tek örnek → etkisiz). "
                  f"EK NOT: düz KFold ardışık blok üretir; grapevine satırları dosya adına göre sıralı olduğundan iç fold'lar "
                  f"plot/tarih bloklarına denk gelir (plot grubu bölünmesi apparent={int(ryck_plot.iloc[0].n_groups_split_across_inner_folds)}/10)."},
        {"konu": "R3-26 ölçekleme",
         "durum": V,
         "kanıt": f"LASSO: `StandardScaler().fit_transform(X)` seçicinin içinde, yalnız verilen X'e "
                  f"({R['lasso_scaler']}). "
                  f"RFE: `Ridge(alpha=1.0)` ölçeklemesiz ({R['rfe']}); "
                  f"sütun std aralığı {fa.col_std[fa.col_std > 0].min():.2e} – {fa.col_std.max():.2e} → RFE sıralaması ölçeğe bağlı. "
                  f"PLS: `PLSRegression` varsayılan `scale=True` (model içinde, fold-train'e fit)."},
        {"konu": "R3-31 GA iç CV",
         "durum": V if (beh["ga_inner_cv_type"] == "GroupKFold" and beh["ga_inner_cv_seed_independent"]
                        and not beh["ga_inner_cv_splits_groups"]) else F,
         "kanıt": f"{R['ga_cv']}: "
                  f"{beh['ga_inner_cv_type']}(5), seed'den bağımsız={beh['ga_inner_cv_seed_independent']}, "
                  f"grup bölüyor={beh['ga_inner_cv_splits_groups']}; aynı bölme tüm nesillerde sabit."},
        {"konu": "R3-30 RNG",
         "durum": V,
         "kanıt": f"{R['ga_seed']}: ana süreçte "
                  f"`random.seed`/`np.random.seed`; işçiler `mp.Pool` (joblib değil) ile yalnız fitness hesaplar "
                  f"({R['ga_pool']}); PLS deterministik, "
                  f"LightGBM random_state=42 sabit."},
        {"konu": "R3-28 fitness kırpma",
         "durum": V if (beh["ga_fitness_below_min_features"] == 0.0 and beh["ga_fitness_null_y_min_over_10"] >= 0) else F,
         "kanıt": f"{R['ga_clip']}: `max(mean_r2, 0)`; "
                  f"MIN_FEATURES={beh['ga_MIN_FEATURES']} altı → {beh['ga_fitness_below_min_features']}. "
                  f"Sinyalsiz y ile 10 rastgele bireyin {beh['ga_fitness_null_y_n_zero_over_10']}'i tam 0 fitness aldı."},
        {"konu": "Dış CV",
         "durum": V if (c["outer_split_identical_across_seeds_ryck"] and c["outer_split_identical_across_seeds_potato"]
                        and beh["groupkfold_has_shuffle"]) else F,
         "kanıt": f"{R['outer']}: shuffle'sız GroupKFold. "
                  f"12 seed'de dış fold'lar aynı (grapevine {c['outer_split_identical_across_seeds_ryck']}, "
                  f"potato {c['outer_split_identical_across_seeds_potato']}). sklearn 1.8.0 `GroupKFold(shuffle=…)`: "
                  f"{beh['groupkfold_has_shuffle']}."},
        {"konu": "R3-9 plot grupları",
         "durum": V if c["groups_plot_partition_matches_tab"] else F,
         "kanıt": f"`groups_plot.npy` = variety|plotLocation|collectionDate ({R['builder_plot']}); "
                  f"{len(np.unique(d['g_plot']))} plot grubu, plot başına {sorted(pd.Series(d['g_plot']).value_counts().tolist())} leaf. "
                  f"Bölümleme description-2.tab ile aynı: {c['groups_plot_partition_matches_tab']}. "
                  f"NOT: kayıtlı dizgelerde U+FFFD var ({c['groups_plot_has_U+FFFD']}; eski ayrıştırıcı) — bölümleme etkilenmez. "
                  f"NOT: plot = çeşit × konum × tarih; çeşit ve tarih ile karışık (R3-10)."},
        {"konu": "R3-11 potato grubu",
         "durum": V if (c["potato_npz_order_equals_chess_csv"]) else F,
         "kanıt": f"`groups.append(int(row[\"Field\"]))` ({R['potato_field']}); "
                  f"{n_fields} Field, Field başına örnek: {per_field}; npz sırası CHESS_FOLDS.CSV ile aynı: "
                  f"{c['potato_npz_order_equals_chess_csv']}. Orijinal 5-fold'da fold başına "
                  f"{fold_df[fold_df.dataset == 'deep_patato'].n_test_groups.tolist()} Field."},
        {"konu": "R3-22 potato ön işleme",
         "durum": V,
         "kanıt": f"{R['potato_sg']}: SavGol(11, 2, deriv=0) + SNV, satır başına "
                  f"(örnek-yerel). Önbellek: satır ort. max|·|={sg['potato_rows_max_abs_mean']:.1e}, std−1 max|·|="
                  f"{sg['potato_rows_max_abs_std_minus1']:.1e}. 'raw channels' ifadesi düzeltilmeli."},
        {"konu": "R3-71 permütasyon p",
         "durum": V,
         "kanıt": f"{R['perm_p']}: "
                  f"p = (1 + #{{perm ≥ obs}}) / (B + 1), tek yönlü, istatistik nested fold-mean R². "
                  f"NOT: y serbest karıştırılıyor (grup kısıtı yok) — potato'da Field bloğu korunmuyor (E5'te ele alınacak)."},
        {"konu": "Bekleyen dallar",
         "durum": F,
         "kanıt": "Plan 'revision-1 = main + dört dal' diyor. Yayın main'i (7bf625c) kök commit (geçmiş sıkıştırılmış); "
                  "içerik karşılaştırması: fix/wilcoxon-degeneracy-guard ve fix/figure2-label-overlap kodu main'de ZATEN VAR "
                  "(git apply --reverse --check temiz; README içeriği yeniden yazılmış hâliyle mevcut). "
                  "Eksik olanlar (fix/config-truthfulness: config, run_selection_bias, README; "
                  "chore/logging-and-figure-labels: run.log, figür etiketleri) revision-1'e patch olarak, "
                  "parça başına ayrı commit ile uygulandı (merge yok). GitHub'da yalnız main var (git ls-remote)."},
        {"konu": "Yayına girmemiş betikler",
         "durum": N,
         "kanıt": "run_null_n200.py kopyalandı. Yayındaki outputs3/data/null.pkl: NullResult n_repeats=200, method=filter, "
                  "k=20 → makaledeki null run_null_n200.py ile üretilmiş (config varsayılanı null_repeats=30). "
                  "regen_outputs3.py, regen_outputs3_rmse.py, outputs2/ henüz eşlenmedi (R3-81, sonraki aşama)."},
        {"konu": "Eski checkpoint'ler",
         "durum": N,
         "kanıt": "Kopyalanmadı (plan gereği); outputs/17_selection_bias/ çalışma klasöründe yok."},
    ]
    df = pd.DataFrame(rows)
    sanity = pd.DataFrame([{"kontrol": k, "sonuç": v} for k, v in c.items()])
    text = f"""# Plan Bölüm 1 tablosu — kod/veri doğrulaması

Durum: **{V}** = plandaki bulgu kodda/veride teyit edildi; **{F}** = plan metni güncellenmeli;
**{N}** = bilgi.

{_md_table(df)}

## Veri tutarlılık kontrolleri
{_md_table(sanity)}

## Davranış kontrolleri (seçici/fitness çağrıları; model başarımı ölçülmedi)
{_md_table(pd.DataFrame([{"kontrol": k, "sonuç": v} for k, v in beh.items()]))}

## SG / SNV yeniden hesap
{_md_table(pd.DataFrame([{"kontrol": k, "sonuç": v} for k, v in sg.items()]))}

## Başlık sayılarına etkisi
Bu denetimde başlık sayılarını (0.615 / 0.269 / +0.346) değiştirecek bir hata bulunmadı:
tespit edilen konular (LassoCV iç CV'nin potato'da Field bölmesi, REP denkleminin standart dışı yazımı,
SG `delta=1`) yalnızca LASSO-potato satırlarını, indeks tanımının metnini ve türev biriminin
raporlanmasını ilgilendiriyor; başlık konfigürasyonu (Ryckewaert / flavonol / PLS / GA) bunlardan
etkilenmiyor.
"""
    (out / "section1_verification.md").write_text(text, encoding="utf-8")
    return rows


# ---------------------------------------------------------------------------
# run_config.json
# ---------------------------------------------------------------------------
def write_run_config(out: Path, t0: float, quick: bool) -> None:
    pkgs = {}
    for p in ["numpy", "scipy", "scikit-learn", "pandas", "deap", "lightgbm", "statsmodels",
              "joblib", "matplotlib", "pyarrow"]:
        try:
            pkgs[p] = importlib_metadata.version(p)
        except importlib_metadata.PackageNotFoundError:
            pkgs[p] = None
    inputs = [DS_DIR / f for f in sorted(p.name for p in DS_DIR.iterdir())] + [POTATO_NPZ, CHESS_CSV, DESC_TAB, HDR_PATH]
    payload = {
        "experiment": "E0_audit",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_files": (_git("status", "--porcelain") or "").splitlines(),
        "command": " ".join([Path(sys.executable).name, "-m", "src.m07_revision.audit", *sys.argv[1:]]),
        "quick": quick,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": pkgs,
        "seeds": {"outer_split_checked": SEEDS_PAPER, "behaviour_checks": [0, 1, 7],
                  "note": "E0 model başarımı ölçmez; seed'ler yalnız davranış doğrulamasında kullanıldı."},
        "inputs_sha256": {str(p.relative_to(ROOT).as_posix()): _sha256(p) for p in inputs},
        "elapsed_s": round(time.time() - t0, 2),
    }
    (out / "run_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def run(quick: bool = False, out_dir: Path = OUT_DIR) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load_inputs()

    fa = build_feature_audit(d)
    fa.to_csv(out_dir / "feature_audit.csv", index=False, encoding="utf-8")
    write_flavonol_md(d, fa, out_dir)
    write_continuum_md(d, out_dir)
    sg = write_savgol_md(d, out_dir)

    groups, gsum = build_groups(d)
    # Örnek düzeyi (leaf/sample id içeren) dosyalar samples/ altına → .gitignore'da
    # (commit politikası: yalnız kod + özet tablolar repoya girer).
    (out_dir / "samples").mkdir(exist_ok=True)
    groups.to_csv(out_dir / "samples" / "groups.csv", index=False, encoding="utf-8")
    gsum.to_csv(out_dir / "groups_summary.csv", index=False, encoding="utf-8")
    build_target_distributions(d, groups).to_csv(out_dir / "target_distributions.csv",
                                                 index=False, encoding="utf-8")

    fold_df, fchecks = build_fold_composition(d, groups)
    d["checks"].update(fchecks)
    fold_df.to_csv(out_dir / "fold_composition.csv", index=False, encoding="utf-8")
    lasso_cv = build_lasso_inner_cv(d)
    lasso_cv.to_csv(out_dir / "lasso_inner_cv.csv", index=False, encoding="utf-8")

    svd, svd_sum, Z_keep = build_svd(d)
    svd.to_csv(out_dir / "svd.csv", index=False, encoding="utf-8")
    svd_sum.to_csv(out_dir / "svd_summary.csv", index=False, encoding="utf-8")

    if not quick:
        plot_target_distributions(d, groups, out_dir)
        plot_svd(d, svd, Z_keep, fa, out_dir)

    beh = behaviour_checks(d, quick)
    write_seed_md(beh, d["checks"], out_dir)
    rows = write_section1_md(d, fa, sg, beh, lasso_cv, fold_df, out_dir)
    write_run_config(out_dir, t0, quick)
    return {"checks": d["checks"], "behaviour": beh, "sg": sg, "section1": rows,
            "svd_summary": svd_sum, "lasso_inner_cv": lasso_cv, "fold_composition": fold_df}


def _parse(argv=None):
    ap = argparse.ArgumentParser(description="E0 denetimi (koşu gerektirmez).")
    ap.add_argument("--quick", action="store_true",
                    help="figürleri ve seçici-determinizm çağrılarını atla (duman testi)")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="çıktı klasörü")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _parse()
    res = run(quick=args.quick, out_dir=args.out)
    # Duman testi: tüm veri tutarlılık kontrolleri geçmeli
    failed = [k for k, v in res["checks"].items() if v is False and k != "groups_plot_has_U+FFFD"]
    print("Veri kontrolleri:", "HEPSİ GEÇTİ" if not failed else f"BAŞARISIZ: {failed}")
    for r in res["section1"]:
        print(f"  [{r['durum']:>10}] {r['konu']}")
    print(res["svd_summary"][["dataset", "n_constant_dropped", "numerical_rank", "n_comp_90",
                              "n_comp_95", "n_comp_99", "participation_ratio",
                              "entropy_effective_rank"]].to_string(index=False))
    print("Çıktı:", args.out)
    sys.exit(1 if failed else 0)
