"""
Figure 3 — Synthetic null: distribution of apparent and nested R2 when no signal exists.

Amac: apparent ve nested tahminlerin AYNI estimator'un gurultulu iki versiyonu
OLMADIGINI gostermek. Sinyal yokken apparent pozitif, nested negatif kaliyor;
iki dagilim sifirin iki yanina duşuyor.

TERMINOLOJI KILIDI: Bu figur nested tasarimin VALIDASYONUDUR, leakage'in kaniti
degildir. Caption'da "proves leakage" tarzi bir ifade KULLANILMAMALIDIR.

KAPSAM: Deney tek selector ile yapilmistir: filtre-tabanli, hedefle MUTLAK PEARSON
KORELASYONU en yuksek k=20 oznitelik (bkz synthetic_null._topk_corr_idx). Mutual
information DEGILDIR. Bu, Abstract'ta da isaretlenmistir (D6); caption'da
belirtilmesi zorunludur, aksi halde tum selector aileleri icin gosterilmis gibi okunur.

Girdi : null deneyinin tekrar-bazli R2 degerleri.
        Oncelik: outputs3/data/null.pkl (NullResult; .apparent / .nested = 200 tekrar).
        Alternatif: --csv ile bir CSV verilirse (sutunlar apparent_r2, nested_r2)
        o okunur. Her iki durumda da GERCEK tekrar degerleri kullanilir; ozet
        mean/SD'den yeniden uretim YAPILMAZ (kod-gercek sadakati).

Cikti : outputs3/figures/paper_visuals/fig_null_distribution.(pdf|png)

Bu betik pipeline'dan BAGIMSIZ calisir; proje kokunu kendi konumundan cozer:
    python paper_visuals/fig_null_distribution.py

Tasarim notlari:
  - Baslik figurun icinde yoktur; Elsevier basligi caption'a ister.
  - Grayscale baskida ayrism icin dolgu tonu + desen birlikte kodlanir
    (eski surumdeki kirmizi/mavi grayscale'de birbirine yaklasiyordu).
  - Sifirda dikey referans cizgisi: figurun tum retorigi bu cizginin iki yaninda.
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Yol cozumu ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
# synthetic_null modulu unpickle icin import edilebilir olmali
sys.path.insert(0, str(ROOT))

# --- Ayarlar ---------------------------------------------------------------

NULL_PKL = ROOT / "outputs3" / "data" / "null.pkl"
OUTDIR = ROOT / "outputs3" / "figures" / "paper_visuals"
STEM = "fig_null_distribution"

# CILS tek sutun ~90 mm — bu figur tek panel, dar durabilir
WIDTH_IN = 130 / 25.4
HEIGHT_IN = 75 / 25.4

N_BINS = 24

STYLE = {
    "apparent": dict(facecolor="0.55", edgecolor="0.2", hatch="",
                     label="Apparent"),
    "nested":   dict(facecolor="white", edgecolor="0.2", hatch="///",
                     label="Nested"),
}


# --- Veri ------------------------------------------------------------------

def load_from_pkl(pkl_path: Path) -> pd.DataFrame:
    """NullResult pkl'inden tekrar-bazli apparent/nested R2 degerlerini okur."""
    with open(pkl_path, "rb") as fh:
        res = pickle.load(fh)
    app = np.asarray(res.apparent, dtype=float)
    nst = np.asarray(res.nested, dtype=float)
    return pd.DataFrame({"apparent_r2": app, "nested_r2": nst}).dropna()


def load_from_csv(csv_path: Path) -> pd.DataFrame:
    """CSV'den tekrar-bazli R2 degerlerini okur."""
    df = pd.read_csv(csv_path)
    for col in ("apparent_r2", "nested_r2"):
        if col not in df.columns:
            raise ValueError(f"'{col}' sutunu yok. Mevcut: {list(df.columns)}")
    return df.dropna(subset=["apparent_r2", "nested_r2"])


def load(csv: Path | None, pkl: Path) -> pd.DataFrame:
    if csv is not None:
        return load_from_csv(csv)
    if not pkl.exists():
        raise FileNotFoundError(
            f"{pkl} bulunamadi. Bu figur gercek tekrar degerlerinden cizilmelidir; "
            f"ozet mean/SD'den yeniden uretim yapilmaz. --csv ile bir dosya verin."
        )
    return load_from_pkl(pkl)


# --- Cizim -----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 3 — synthetic null distribution")
    parser.add_argument("--csv", type=Path, default=None,
                        help="tekrar-bazli R2 CSV'si (sutunlar: apparent_r2, nested_r2)")
    parser.add_argument("--pkl", type=Path, default=NULL_PKL,
                        help=f"NullResult pkl (varsayilan: {NULL_PKL})")
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    d = load(args.csv, args.pkl)
    app = d["apparent_r2"].to_numpy()
    nst = d["nested_r2"].to_numpy()

    # Ortak bin kenarlari: iki dagilim ayni olcekte okunmali
    lo = min(app.min(), nst.min())
    hi = max(app.max(), nst.max())
    pad = 0.05 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, N_BINS + 1)

    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))

    ax.hist(nst, bins=bins, lw=0.7, zorder=2, **STYLE["nested"])
    ax.hist(app, bins=bins, lw=0.7, zorder=2, **STYLE["apparent"])

    # Sifir cizgisi: figurun tum mesaji bu cizginin iki yaninda
    ax.axvline(0, color="black", lw=1.0, ls="--", zorder=3)

    # Ortalama isaretleri
    for arr, name in ((app, "apparent"), (nst, "nested")):
        ax.axvline(arr.mean(), color="0.2", lw=0.8, ls=":", zorder=3)

    ax.set_xlabel("$R^2$", fontsize=9)
    ax.set_ylabel(f"Number of repeats (n = {len(d)})", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="0.9")
    ax.set_axisbelow(True)

    # Legend'a ortalama + SD yazilir
    handles, _ = ax.get_legend_handles_labels()
    labels = [
        f"Nested (mean {nst.mean():+.3f}, SD {nst.std(ddof=1):.3f})",
        f"Apparent (mean {app.mean():+.3f}, SD {app.std(ddof=1):.3f})",
    ]
    ax.legend(handles, labels, fontsize=7.5, frameon=False, loc="upper left")

    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.16)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.outdir / f"{STEM}.{ext}", dpi=600, bbox_inches="tight")
    print(f"yazildi: {args.outdir / STEM}.pdf ve .png")
    print(f"  apparent: {app.mean():+.3f} +/- {app.std(ddof=1):.3f}  (n={len(app)})")
    print(f"  nested  : {nst.mean():+.3f} +/- {nst.std(ddof=1):.3f}  (n={len(nst)})")


if __name__ == "__main__":
    main()
