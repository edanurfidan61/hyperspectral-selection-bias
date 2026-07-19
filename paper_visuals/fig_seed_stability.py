"""
Figure 4 — Seed stability of the headline combination.

Amac: apparent ve nested tahminlerin 12 seed boyunca AYRIK kaldigini gostermek.
Ikinci mesaj: nested tahmin sadece daha dusuk degil, ayni zamanda daha degiskendir
(apparent SD ~0.011, nested SD ~0.026) — Discussion 4.x'teki iddianin gorsel karsiligi.

DIKKAT: Eski figurun basligi "apparent (unstable) vs nested (stable)" idi; bu makale
metniyle CELISIYORDU. Dogru yon: nested daha degisken. Bu script o hatayi tasimaz.

Girdi : outputs3/tables_foldmean/selection_bias_multiseed_raw.csv  (fold-mean)
        Beklenen sutunlar: seed, dataset, target, model, selector, apparent_r2, nested_r2
        NOT: apparent_r2 fold-mean'dir; pooled deger ayri sutunda (apparent_r2_pooled)
        durur ve KULLANILMAZ.
Cikti : outputs3/figures/paper_visuals/fig_seed_stability.(svg|pdf|png)

Bu betik pipeline'dan BAGIMSIZ calisir; proje kokunu kendi konumundan cozer:
    python paper_visuals/fig_seed_stability.py
Yol ezmek icin: --csv <yol> --outdir <yol>

Tasarim notlari:
  - Sadece headline kombinasyon gosterilir. 16 kombinasyonun tamami Figure 2'de
    zaten var; burada tekrarlamak figuru okunmaz yapardi (eski surumun sorunu).
  - X ekseni seed'dir ama seed sirali bir degisken DEGILDIR: trend/regresyon
    cizgisi bilincli olarak yoktur.
  - Grayscale baskida ayrism icin renk + marker + cizgi tipi birlikte kodlanir.
  - Baslik figurun icinde yoktur; Elsevier basligi caption'a ister.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# --- Yol cozumu ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# --- Ayarlar ---------------------------------------------------------------

CSV = ROOT / "outputs3" / "tables_foldmean" / "selection_bias_multiseed_raw.csv"
OUTDIR = ROOT / "outputs3" / "figures" / "paper_visuals"
STEM = "fig_seed_stability"

# CILS tek sutun ~90 mm; bu figur dar ve okunabilir olmali
WIDTH_IN = 140 / 25.4
HEIGHT_IN = 75 / 25.4

HEADLINE = dict(dataset="ryckewaert", target="flavonol", model="pls", selector="ga")

STYLE = {
    "apparent": dict(color="0.15", marker="o", ls="-", ms=4.5, lw=1.2,
                     mfc="0.15", label="Apparent"),
    "nested":   dict(color="0.15", marker="s", ls="--", ms=4.5, lw=1.2,
                     mfc="white", label="Nested"),
}


# --- Veri ------------------------------------------------------------------

def load(csv_path: Path) -> pd.DataFrame:
    """Headline kombinasyonun seed-bazli satirlarini dondurur."""
    df = pd.read_csv(csv_path)
    mask = True
    for col, val in HEADLINE.items():
        mask &= df[col] == val
    d = df[mask].sort_values("seed").reset_index(drop=True)
    if d.empty:
        raise ValueError(f"Headline kombinasyon bulunamadi: {HEADLINE}")
    if len(d) != 12:
        raise ValueError(f"12 seed bekleniyordu, {len(d)} bulundu")
    return d


# --- Cizim -----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 4 — seed stability")
    parser.add_argument("--csv", type=Path, default=CSV)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    d = load(args.csv)

    fig, (ax, ax_box) = plt.subplots(
        1, 2, figsize=(WIDTH_IN, HEIGHT_IN),
        gridspec_kw=dict(width_ratios=[3.4, 1], wspace=0.06),
        sharey=True,
    )

    # --- Sol panel: seed bazinda iki seri --------------------------------
    ax.plot(d["seed"], d["apparent_r2"], **STYLE["apparent"])
    ax.plot(d["seed"], d["nested_r2"], **STYLE["nested"])

    # Iki serinin arasindaki bosluk = bias; hafif gölge ile gorsellestirilir
    ax.fill_between(d["seed"], d["nested_r2"], d["apparent_r2"],
                    color="0.85", alpha=0.55, zorder=0, lw=0)

    ax.set_xticks(d["seed"])
    ax.set_xlabel("Seed", fontsize=9)
    ax.set_ylabel("$R^2$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="0.9")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="center right")

    # Ortadaki bosluga bias etiketi
    mid = len(d) // 2
    y_mid = (d["apparent_r2"].iloc[mid] + d["nested_r2"].iloc[mid]) / 2
    ax.annotate(f"bias = {(d['apparent_r2'] - d['nested_r2']).mean():+.3f}",
                xy=(d["seed"].iloc[mid], y_mid), fontsize=7.5,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="0.6", lw=0.5))

    # --- Sag panel: dagilim genisligi ------------------------------------
    # Nested'in daha genis oldugu buradan okunur (makale metniyle tutarli)
    bp = ax_box.boxplot(
        [d["apparent_r2"], d["nested_r2"]],
        widths=0.55, showfliers=False, patch_artist=True,
        medianprops=dict(color="black", lw=1.0),
        whiskerprops=dict(color="0.3", lw=0.8),
        capprops=dict(color="0.3", lw=0.8),
        boxprops=dict(edgecolor="0.3", lw=0.8),
    )
    for patch, fc in zip(bp["boxes"], ["0.55", "white"]):
        patch.set_facecolor(fc)
    bp["boxes"][1].set_hatch("///")

    ax_box.set_xticks([1, 2])
    ax_box.set_xticklabels(["App.", "Nest."], fontsize=8)
    ax_box.tick_params(labelsize=8)
    ax_box.spines[["top", "right", "left"]].set_visible(False)
    ax_box.tick_params(axis="y", left=False)
    ax_box.grid(axis="y", lw=0.4, color="0.9")
    ax_box.set_axisbelow(True)

    # SD degerleri: nested'in daha genis oldugunu sayiyla da soyler
    ax_box.text(1, d["apparent_r2"].max() + 0.012,
                f"SD\n{d['apparent_r2'].std(ddof=1):.3f}",
                ha="center", va="bottom", fontsize=6.5)
    ax_box.text(2, d["nested_r2"].max() + 0.012,
                f"SD\n{d['nested_r2'].std(ddof=1):.3f}",
                ha="center", va="bottom", fontsize=6.5)

    fig.subplots_adjust(left=0.10, right=0.99, top=0.94, bottom=0.16)
    args.outdir.mkdir(parents=True, exist_ok=True)
    # Vektorel (SVG + PDF) makale icin; PNG hizli onizleme icin.
    for ext in ("svg", "pdf", "png"):
        fig.savefig(args.outdir / f"{STEM}.{ext}", dpi=600, bbox_inches="tight")
    print(f"yazildi: {args.outdir / STEM}.svg, .pdf ve .png")
    print(f"  apparent: {d['apparent_r2'].mean():.3f} +/- {d['apparent_r2'].std(ddof=1):.3f}")
    print(f"  nested  : {d['nested_r2'].mean():.3f} +/- {d['nested_r2'].std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
