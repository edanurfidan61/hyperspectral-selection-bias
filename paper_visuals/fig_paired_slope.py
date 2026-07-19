"""
Figure 2 — Apparent vs nested R2 across the sixteen analysed combinations.

Iki panel:
  (a) Paired slope plot : her kombinasyon icin apparent -> nested gecisi.
  (b) Bias bar chart     : ayni kombinasyonlarin bias degerleri (apparent - nested),
                           buyukluge gore siralanmis. Tek negatif satir hemen ayrisir.

Girdi : outputs3/tables_foldmean/stats_summary.csv   (fold-mean agregasyon)
        Beklenen sutunlar: dataset, target, model, selector,
                           mean_apparent, mean_nested, mean_bias
        Dosyanin sonundaki '#' ile baslayan ozet yorum satirlari otomatik atilir.

Cikti : outputs3/figures/paper_visuals/fig_paired_slope.(svg|pdf)  — vektorel

Bu betik pipeline'dan BAGIMSIZ calisir. Proje kokunu kendi konumundan cozer,
dolayisiyla nereden calistirilirsa calistirilsin dogru girdi/cikti yollarini bulur:

    python paper_visuals/fig_paired_slope.py

Yollari ezmek icin (istege bagli):
    python paper_visuals/fig_paired_slope.py --csv <yol> --outdir <yol>

Vurgulanan kombinasyonlar (fold-mean):
  - Headline : ryckewaert / flavonol / pls / ga  (bias ~ +0.346, hard-coded tuple;
               etiket degeri CSV'den dinamik okunur)
  - Istisnalar : mean_bias < 0 olan TUM kombinasyonlar (KOSUL, hard-coded degil).
               Fold-mean veride iki tane vardir:
                 deep_patato / chlorophyll / pls / rfe          (~ -0.062)
                 deep_patato / chlorophyll / pls / mutual_info  (~ -0.005)
               Yeni bir kombinasyon negatife dönerse otomatik istisna stili alir.

Tasarim notlari:
  - Degerler 12 seed ortalamasidir; caption'da belirtilmelidir.
  - Anlamlilik gosterimi (yildiz / p-degeri) bilincli olarak YOKTUR. Anlamlilik
    zinciri Table 4'te raporlanir; figurde tekrarlanmasi Abstract'ta yumusatilan
    iddiayi geri getirirdi.
  - Grayscale baskida ayrism icin renk + cizgi tipi birlikte kodlanmistir.
  - Panel genisligi CILS cift sutun (~190 mm) hedeflenerek verilmistir; TUM yazi
    boyutlari >= 7 pt (MIN_PT). Eskiden (b) yticklabel'lari 6.5 pt idi — kisit
    ihlali; duzeltildi.

Yerlesim (b) panelinin etiketleri icin
--------------------------------------
(b)'nin kombinasyon etiketleri UZUN ("Grapevine / flavonol / LightGBM / LASSO",
39 karakter, 7 pt'de ~50 mm = figur genisliginin ~%26'si). Eskiden sol margin
bu genislige gore acilmadigi icin etiketler (a) panelinin cizim alanina tasiyor,
slope cizgilerinin uzerine biniyordu.

Cozum: etiketler (b)'nin KENDI yticklabel'lari olarak kalir (iki panelin y
eksenleri farkli cinsten — (a) surekli R2, (b) kategorik satir — ortak etiket
paylasamazlar). Gereken bosluk GOZLE DENEME ile degil, renderer'dan OLCULEREK
hesaplanir: _measure_label_width() en uzun etiketi gercek renderer ile olcer,
_solve_layout() bundan wspace'i cozer. Font/etiket degisirse yerlesim otomatik
uyarlanir.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# --- Yol cozumu ------------------------------------------------------------
# Proje koku: bu dosya paper_visuals/ icinde -> bir ust dizin.
ROOT = Path(__file__).resolve().parent.parent

# --- Ayarlar ---------------------------------------------------------------

CSV = ROOT / "outputs3" / "tables_foldmean" / "stats_summary.csv"
OUTDIR = ROOT / "outputs3" / "figures" / "paper_visuals"
STEM = "fig_paired_slope"

# CILS cift sutun ~190 mm genislik (inch cinsinden)
WIDTH_IN = 190 / 25.4
HEIGHT_IN = 105 / 25.4

# Dergi kisiti: figurdeki HICBIR yazi bunun altina inmez.
MIN_PT = 7.0
# (b) kombinasyon etiketleri — en uzunu ~39 karakter; MIN_PT'de tutulur.
TICK_PT = MIN_PT
# Panel basliklari / eksen etiketleri.
AXIS_PT = 9.0
# Deger etiketleri ("+0.346") de kisita tabi.
VALUE_PT = MIN_PT

# Yerlesim payi: olculen etiket genisligine eklenen nefes payi (inch).
# 0.06 yetmiyordu — en uzun etiket (a)'nin cizim alanina 0.9 mm giriyordu (goz
# ayirt etmiyor, olcum ediyor: bkz. _assert_layout). tick isaretleri + etiket-
# eksen bosluguna karsilik ~2.5 mm pay.
LABEL_PAD_IN = 0.10
# Figur kenar bosluklari (subplots_adjust ile ayni birim: figur orani).
LEFT_MARGIN = 0.075
RIGHT_MARGIN = 0.985
# width_ratios — (a) dar, (b) genis.
WIDTH_RATIOS = (1.0, 1.6)

# Headline yalniz vurgu tuple'i (etiket degeri CSV'den dinamik). Istisna ise
# SABIT tuple DEGIL, mean_bias < 0 KOSULU (bkz. _kind) — birden fazla negatif
# kombinasyon otomatik istisna stili alir.
HEADLINE = ("ryckewaert", "flavonol", "pls", "ga")

# Cizgi/bar stilleri: renk + cizgi tipi birlikte kodlanir (grayscale guvenli)
STYLE = {
    "normal":    dict(color="0.65", lw=1.0, ls="-",  zorder=1, alpha=0.9),
    "headline":  dict(color="black", lw=2.6, ls="-",  zorder=3),
    "exception": dict(color="0.15", lw=1.8, ls="--", zorder=2),
}
BAR_FACE = {"normal": "0.75", "headline": "0.15", "exception": "white"}
BAR_EDGE = {"normal": "0.45", "headline": "black", "exception": "0.15"}
# Istisna hatch'i siki ("////"): -0.005 bar'i cok kisa, seyrek hatch'te ("///")
# icine tek egik cizgi bile dusmuyor ve duz beyaz gorunuyordu.
BAR_HATCH = {"normal": "", "headline": "", "exception": "////"}

LABEL = {
    "ga": "GA", "rfe": "RFE", "lasso": "LASSO", "mutual_info": "MI",
    "pls": "PLS", "lightgbm": "LightGBM",
    "ryckewaert": "Grapevine", "deep_patato": "Potato",
    "flavonol": "flavonol", "chlorophyll": "chlorophyll",
}


# --- Veri ------------------------------------------------------------------

def load(csv_path: Path) -> pd.DataFrame:
    """stats_summary.csv'yi okur, ozet yorum satirlarini atar, 16 kombinasyonu dondurur."""
    df = pd.read_csv(csv_path)
    df = df[~df["dataset"].astype(str).str.startswith("#")].copy()
    df = df.dropna(subset=["mean_apparent", "mean_nested", "mean_bias"])
    if len(df) != 16:
        raise ValueError(f"16 kombinasyon bekleniyordu, {len(df)} bulundu")
    df["kind"] = df.apply(_kind, axis=1)
    df["label"] = df.apply(_pretty, axis=1)
    return df


def _kind(row) -> str:
    key = (row["dataset"], row["target"], row["model"], row["selector"])
    if key == HEADLINE:
        return "headline"
    # Istisna artik SABIT tuple degil, KOSUL: negatif bias olan her kombinasyon.
    # Fold-mean veride bu iki satirdir (rfe ~ -0.062, mutual_info ~ -0.005) ama
    # kod bunlari sabitlemez — mean_bias isaretine gore otomatik ayrisir.
    if row["mean_bias"] < 0:
        return "exception"
    return "normal"


def _pretty(row) -> str:
    return (f"{LABEL[row['dataset']]} / {LABEL[row['target']]} / "
            f"{LABEL[row['model']]} / {LABEL[row['selector']]}")


# --- Yerlesim olcumu -------------------------------------------------------

def _measure_label_width(fig, labels, fontsize: float) -> float:
    """En uzun etiketin GERCEK genisligi (inch) — renderer'dan olculur.

    Karakter sayisi x tahmini genislik gibi yaklasimlar yaniltir (orantisiz
    font, ' / ' ayraclari). Burada etiket gecici bir Text olarak cizilip
    get_window_extent ile olculur, sonra kaldirilir.
    """
    renderer = fig.canvas.get_renderer()
    widest = 0.0
    probe = fig.text(0, 0, "", fontsize=fontsize)
    for lab in labels:
        probe.set_text(str(lab))
        bb = probe.get_window_extent(renderer=renderer)
        widest = max(widest, bb.width / fig.dpi)
    probe.remove()
    return widest


def _solve_layout(fig, label_w_in: float) -> float:
    """Olculen etiket genisliginden gereken wspace'i coz.

    matplotlib'de wspace, subplot'lar arasi bosluk / ORTALAMA subplot genisligi
    oranidir. (b)'nin etiketleri kendi ekseninin SOLUNA tasar, yani (a) ile (b)
    arasindaki boslugun etiketi tamamen yutmasi gerekir; yoksa etiketler (a)'nin
    cizim alanina biner (eski hatanin tam sebebi).

        available   = figur genisligi - kenar bosluklari
        w_a + w_b   = available - gap             (w_b / w_a = ratio_b / ratio_a)
        gap         = label_w_in + LABEL_PAD_IN   (etiket + nefes payi)
        wspace      = gap / ((w_a + w_b) / 2)

    Donus: wspace (subplots_adjust/gridspec ile ayni birim).
    """
    fig_w = fig.get_size_inches()[0]
    available = fig_w * (RIGHT_MARGIN - LEFT_MARGIN)
    gap = label_w_in + LABEL_PAD_IN
    panels_w = available - gap
    if panels_w <= 0:
        raise ValueError(
            f"Etiketler figure sigmiyor: gap={gap:.2f} in, available={available:.2f} in. "
            "Figur genisligini artirin veya etiketleri kisaltin."
        )
    return gap / (panels_w / 2.0)


def _assert_layout(fig, ax_a, ax_b) -> None:
    """Cizimden SONRA yerlesimi dogrula — sessizce bozulmasin.

    Iki kisiti olcerek denetler (goz yanilir, olcum yanilmaz):
      1. (b)'nin hicbir kombinasyon etiketi (a)'nin cizim alanina girmez.
      2. Figurdeki hicbir yazi MIN_PT altina inmez.
    Ihlalde RuntimeError — bozuk figur sessizce yazilmaz.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    a_right = ax_a.get_window_extent(r).x1

    intruding = [t.get_text() for t in ax_b.get_yticklabels()
                 if t.get_window_extent(r).x0 < a_right]
    if intruding:
        raise RuntimeError(
            f"(b) etiketleri (a) panelinin cizim alanina giriyor: {intruding}. "
            f"LABEL_PAD_IN'i artirin (su an {LABEL_PAD_IN})."
        )

    small = []
    for ax in (ax_a, ax_b):
        items = (list(ax.get_xticklabels()) + list(ax.get_yticklabels())
                 + [ax.xaxis.label, ax.yaxis.label, ax.title] + list(ax.texts))
        small += [(t.get_text(), t.get_fontsize()) for t in items
                  if t.get_text() and t.get_fontsize() < MIN_PT]
    if small:
        raise RuntimeError(f"{MIN_PT} pt altinda yazi var: {small}")


# --- Panel (a): paired slope ----------------------------------------------

def panel_slope(ax, df: pd.DataFrame) -> None:
    x = [0, 1]
    for _, row in df.iterrows():
        k = row["kind"]
        ax.plot(x, [row["mean_apparent"], row["mean_nested"]], **STYLE[k])
        if k == "normal":
            continue
        # Sadece iki vurgulu satir isaretlenir; 16 etiket okunmaz olurdu
        ax.plot(x, [row["mean_apparent"], row["mean_nested"]],
                marker="o", ms=4, ls="none",
                color=STYLE[k]["color"], zorder=STYLE[k]["zorder"] + 1)

    ax.axhline(0, color="0.4", lw=0.7, ls=":", zorder=0)
    ax.set_xlim(-0.15, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(["Apparent", "Nested"], fontsize=AXIS_PT)
    ax.set_ylabel("$R^2$ (mean over 12 seeds)", fontsize=AXIS_PT)
    ax.tick_params(axis="y", labelsize=MIN_PT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="0.9")
    ax.set_axisbelow(True)
    ax.set_title("(a) Apparent vs nested estimate", fontsize=AXIS_PT, loc="left", pad=8)


# --- Panel (b): bias bar ---------------------------------------------------

def panel_bias(ax, df: pd.DataFrame) -> None:
    # Caption "ordered by magnitude" diyor. barh y'yi asagidan yukari cizer,
    # dolayisiyla ARTAN siralayip cizince gorsel sonuc YUKARIDAN ASAGI AZALAN
    # olur: en ustte headline (+0.346), en altta Potato/chl/RFE (-0.062).
    d = df.sort_values("mean_bias", ascending=True).reset_index(drop=True)
    ypos = range(len(d))

    for y, (_, row) in zip(ypos, d.iterrows()):
        k = row["kind"]
        ax.barh(y, row["mean_bias"],
                height=0.72,
                facecolor=BAR_FACE[k], edgecolor=BAR_EDGE[k],
                hatch=BAR_HATCH[k], linewidth=0.8, zorder=2)

    ax.axvline(0, color="0.3", lw=0.8, zorder=3)

    ax.set_yticks(list(ypos))
    ax.set_yticklabels(d["label"], fontsize=TICK_PT)
    for tick, k in zip(ax.get_yticklabels(), d["kind"]):
        if k == "headline":
            tick.set_fontweight("bold")

    ax.set_xlabel("Bias  (apparent $-$ nested $R^2$)", fontsize=AXIS_PT)
    ax.tick_params(axis="x", labelsize=MIN_PT)
    # Fold-mean veri araligi [-0.062, +0.346]; en negatif bari (-0.062) ve deger
    # etiketlerini kirpmadan sigdir.
    ax.set_xlim(-0.10, 0.40)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", lw=0.4, color="0.9")
    ax.set_axisbelow(True)
    ax.set_title("(b) Bias per combination", fontsize=AXIS_PT, loc="left", pad=8)

    # Vurgulu satirlara deger etiketi.
    # Negatif barlarda etiket bar'in SOLUNA degil, sifir cizgisinin SAGINA
    # konur: sola konunca kisa barlarda (-0.005/-0.062) x-limitini asip eksenin
    # disina, (b)'nin kendi yticklabel'inin uzerine tasiyordu ("RFE0.062").
    # Sifirin sagi bu satirlarda bos, cakisma yok.
    for y, (_, row) in zip(ypos, d.iterrows()):
        if row["kind"] == "normal":
            continue
        v = row["mean_bias"]
        if v >= 0:
            xpos, ha = v + 0.008, "left"
        else:
            xpos, ha = 0.008, "left"
        ax.text(xpos, y, f"{v:+.3f}",
                va="center", ha=ha, fontsize=VALUE_PT,
                fontweight="bold" if row["kind"] == "headline" else "normal",
                zorder=4)


# --- Ana akis --------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 2 — paired slope + bias bar")
    parser.add_argument("--csv", type=Path, default=CSV,
                        help=f"stats_summary.csv yolu (varsayilan: {CSV})")
    parser.add_argument("--outdir", type=Path, default=OUTDIR,
                        help=f"cikti dizini (varsayilan: {OUTDIR})")
    args = parser.parse_args()

    df = load(args.csv)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(WIDTH_IN, HEIGHT_IN),
        gridspec_kw=dict(width_ratios=list(WIDTH_RATIOS)),
    )
    panel_slope(ax_a, df)
    panel_bias(ax_b, df)

    # Yerlesim OLCULEREK cozulur (gozle deneme yok): en uzun (b) etiketinin
    # gercek genisligi -> gereken wspace.
    label_w = _measure_label_width(fig, df["label"], TICK_PT)
    wspace = _solve_layout(fig, label_w)
    fig.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN,
                        top=0.90, bottom=0.13, wspace=wspace)
    print(f"[yerlesim] en uzun etiket = {label_w * 25.4:.1f} mm @ {TICK_PT} pt "
          f"-> wspace = {wspace:.3f}")

    # Yazmadan ONCE dogrula: cakisma / kucuk font varsa burada patlar.
    _assert_layout(fig, ax_a, ax_b)
    print("[dogrulama] etiket cakismasi yok, tum yazilar >= "
          f"{MIN_PT} pt — OK")

    args.outdir.mkdir(parents=True, exist_ok=True)
    # SVG + PDF (makale icin vektorel). bbox_inches="tight" KULLANILMAZ:
    # hesaplanan margin'leri ve 190 mm hedef genisligi ezer.
    for ext in ("svg", "pdf"):
        fig.savefig(args.outdir / f"{STEM}.{ext}")
    print(f"yazildi: {args.outdir / STEM}.svg ve .pdf")


if __name__ == "__main__":
    main()
