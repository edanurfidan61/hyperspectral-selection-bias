"""İstatistiksel anlamlılık testleri (Adım 6c).

Amaç: "bias var" değil, "istatistiksel olarak **anlamlı** bias var" diyebilmek.
Çoklu-seed koşusunun ürettiği per-seed apparent/nested R² çiftleri üzerinde
çalışır; her (dataset, target, model, selector) kombinasyonu için seed'ler
boyunca eşleştirilmiş bir karşılaştırma yapar.

Girdi
-----
``selection_bias_multiseed_raw.csv`` (report.build_multiseed_csv çıktısı).
Kolonlar: seed, dataset, target, model, selector, apparent_r2, nested_r2, bias.
Her kombinasyon için seed'ler boyunca iki eşleştirilmiş dizi (apparent_r2,
nested_r2) toplanır.

Testler (her kombinasyon için)
------------------------------
* **Wilcoxon signed-rank** (eşleştirilmiş, non-parametrik, küçük n'e uygun):
  apparent vs nested. n<6 ise ``low_n_warning=True`` (Wilcoxon güvenilmez) ama
  yine de hesaplanır.
* **Cliff's delta** (non-parametrik etki büyüklüğü) — elle hesaplanır.
  Eşikler: |d|<0.147 ihmal, <0.33 küçük, <0.474 orta, ≥0.474 büyük.
* **Cohen's d** (eşleştirilmiş): fark ortalaması / fark std.

Çoklu karşılaştırma
-------------------
Benjamini-Hochberg (FDR) düzeltmesi tüm kombinasyonların ham p-değerlerine
uygulanır (statsmodels multipletests, fdr_bh). Hem ham (p_value) hem düzeltilmiş
(p_value_fdr) saklanır.

Çıktı
-----
``outputs/17_selection_bias/stats_summary.csv`` + sonuna toplu özet bloğu.

Çalıştırma orchestrator üzerinden:
    python run_selection_bias.py --stages stats
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from src.core import paths
from src.core.logging_setup import get as get_logger

log = get_logger("m06_selection_bias.stats_tests")

# Wilcoxon n<6'da güvenilmez kabul edilir (küçük örneklem uyarısı).
LOW_N_THRESHOLD = 6

# FDR düzeltilmiş p-değeri anlamlılık eşiği.
ALPHA = 0.05

# Cliff's delta yorum eşikleri (Romano ve ark.).
_CLIFF_BINS = [
    (0.147, "ihmal"),     # |d| < 0.147
    (0.33, "küçük"),      # < 0.33
    (0.474, "orta"),      # < 0.474
    (float("inf"), "büyük"),  # >= 0.474
]

GROUP_KEYS = ["dataset", "target", "model", "selector"]


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a>b) − P(a<b), tüm a×b çiftleri üzerinden.

    Non-parametrik, sıralama-tabanlı etki büyüklüğü. +1 ↔ a tamamen b'nin
    üstünde, −1 ↔ tamamen altında, 0 ↔ örtüşme. scipy'de yok; elle hesaplanır.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    # diff[i, j] = sign(a_i - b_j); ortalaması Cliff's delta.
    diff = np.sign(a[:, None] - b[None, :])
    return float(diff.mean())


def cliffs_interpretation(delta: float) -> str:
    """|delta| → ihmal/küçük/orta/büyük."""
    if not np.isfinite(delta):
        return "n/a"
    ad = abs(delta)
    for threshold, label in _CLIFF_BINS:
        if ad < threshold:
            return label
    return "büyük"


def is_degenerate(diff: np.ndarray) -> bool:
    """Fark dizisi dejenere mi — yani seed'ler boyunca SABİT mi?

    Seçici deterministik olduğunda (LASSO, RFE) multiseed harness aynı tek
    hesabı 12 kez yazar; fark dizisi sabit olur. Wilcoxon bunu 12 bağımsız
    eşleştirilmiş gözlem sanıp dizinin UZUNLUĞUNU ölçer (n=12 için exact
    minimum 2/2^12), verinin gücünü değil. Gerçek örneklem büyüklüğü n=1 ve
    seed-tabanlı test TANIMSIZDIR.

    Dikkat: kriter "diff == 0" (apparent == nested) DEĞİL, "diff sabit"tir.
    Sıfırdan farklı ama sabit bir fark da dejeneredir.
    """
    diff = np.asarray(diff, dtype=float)
    if diff.size == 0:
        return True
    return bool(np.allclose(diff, diff[0], atol=1e-12))


def cohens_d_paired(diff: np.ndarray) -> float:
    """Eşleştirilmiş Cohen's d: fark ortalaması / fark std (ddof=1).

    Dejenere (sabit fark) durumda nan — Wilcoxon/Cliff ile aynı kriter. Eskiden
    sd==0'da ±inf dönüyordu; bu bir etki büyüklüğü değil, sıfıra bölmenin
    artefaktıdır (tek deterministik hesabın 12 kopyası). Sabit farkta seed
    varyansı yoktur, dolayısıyla standartlaştırılmış etki TANIMSIZDIR.
    """
    diff = np.asarray(diff, dtype=float)
    if diff.size < 2 or is_degenerate(diff):
        return float("nan")
    sd = diff.std(ddof=1)
    if sd == 0:
        # is_degenerate sonrası buraya normalde düşülmez; savunma amaçlı.
        return float("nan")
    return float(diff.mean() / sd)


def _wilcoxon_safe(apparent: np.ndarray, nested: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank; dejenere (sabit fark) durumda (nan, nan) döner.

    nan ↔ "test tanımsız" (p=1.0 gibi "anlamsız" DEĞİL). Dejenere satırlar
    tabloda kalır — bias'ları görünür — ama p/FDR taşımazlar.
    """
    diff = apparent - nested
    if is_degenerate(diff):
        return float("nan"), float("nan")
    try:
        stat, p = wilcoxon(apparent, nested)
        return float(stat), float(p)
    except ValueError:
        # ör. örneklem çok küçük → test tanımsız.
        return float("nan"), float("nan")


def compute_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-seed ham tablodan kombinasyon-bazlı istatistik tablosu üret.

    Parameters
    ----------
    raw : DataFrame
        En az şu kolonlar: dataset, target, model, selector, apparent_r2,
        nested_r2. (seed bilgisi gruplamaya gerek değil; satırlar seed'lerdir.)

    Returns
    -------
    DataFrame
        Her kombinasyon için bir satır; kolonlar prompt'taki sıraya göre.
        FDR düzeltmesi tüm kombinasyonlar boyunca uygulanmıştır.
    """
    missing = [c for c in (*GROUP_KEYS, "apparent_r2", "nested_r2") if c not in raw.columns]
    if missing:
        raise ValueError(f"Ham tabloda eksik kolon(lar): {missing}")

    rows: list[dict] = []
    for keys, g in raw.groupby(GROUP_KEYS, sort=False):
        dataset, target, model, selector = keys
        app = g["apparent_r2"].to_numpy(dtype=float)
        nes = g["nested_r2"].to_numpy(dtype=float)
        diff = app - nes
        n = len(g)

        stat, p = _wilcoxon_safe(app, nes)
        # Cliff's delta de aynı dejenerasyon kriterine bağlı: sabit farkta iki
        # örtüşmeyen sabit ±1.0 verir — bu etki büyüklüğü değil, artefakttır.
        degenerate = is_degenerate(diff)
        delta = float("nan") if degenerate else cliffs_delta(app, nes)
        rows.append({
            "dataset": dataset, "target": target, "model": model,
            "selector": selector, "n_seeds": n,
            "n_unique_diff": int(np.unique(diff).size),
            "degenerate": degenerate,
            "mean_apparent": float(np.mean(app)),
            "mean_nested": float(np.mean(nes)),
            "mean_bias": float(np.mean(diff)),
            "wilcoxon_stat": stat,
            "p_value": p,
            "p_value_fdr": np.nan,  # aşağıda doldurulur
            "significant_fdr": False,
            "cliffs_delta": delta,
            "cliffs_interpretation": cliffs_interpretation(delta),
            "cohens_d": cohens_d_paired(diff),
            "low_n_warning": n < LOW_N_THRESHOLD,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Benjamini-Hochberg FDR — yalnız geçerli (non-nan) p-değerleri üzerinde.
    valid = df["p_value"].notna().to_numpy()
    p_fdr = np.full(len(df), np.nan)
    if valid.any():
        _, p_corr, _, _ = multipletests(
            df.loc[valid, "p_value"].to_numpy(), alpha=ALPHA, method="fdr_bh",
        )
        p_fdr[valid] = p_corr
    df["p_value_fdr"] = p_fdr
    df["significant_fdr"] = (df["p_value_fdr"] < ALPHA) & df["p_value_fdr"].notna()
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Toplu özet: kaç kombinasyon anlamlı + pozitif bias verdi."""
    n_total = len(df)
    pos = df["mean_bias"] > 0
    sig = df["significant_fdr"]
    sig_pos = (sig & pos).sum()
    low_n = df["low_n_warning"].sum()
    # Dejenere (sabit fark) satırlarda seed-tabanlı test TANIMSIZ; testli satır
    # sayısı = BH'ye giren test sayısı. Anlamlılık oranı bu payda üzerinden
    # okunmalı — 16 üzerinden değil.
    n_tested = int(df["p_value"].notna().sum())
    n_degenerate = int(n_total - n_tested)
    return {
        "n_combos": int(n_total),
        "n_positive_bias": int(pos.sum()),
        "n_significant_fdr": int(sig.sum()),
        "n_significant_positive": int(sig_pos),
        "n_low_n_warning": int(low_n),
        "n_tested": n_tested,
        "n_degenerate": n_degenerate,
        "text": (
            f"{n_tested}/{n_total} kombinasyonda seed-tabanlı test tanımlı "
            f"({n_degenerate} dejenere: seçici deterministik, fark seed'ler "
            f"boyunca sabit → p/δ tanımsız, bias yine de raporlanır). "
            f"BH-FDR {n_tested} test üzerinden. "
            f"Testli satırların {int(sig_pos)}'inde p_fdr<{ALPHA} ve pozitif "
            f"bias (anlamlı + yönü doğru). Pozitif bias: {int(pos.sum())}/{n_total}; "
            f"FDR-anlamlı: {int(sig.sum())}/{n_total}; "
            f"düşük-n uyarısı: {int(low_n)}/{n_total}."
        ),
    }


def _write_with_summary(df: pd.DataFrame, summary: dict, path: Path) -> None:
    """Kombinasyon tablosunu + sonuna toplu özet bloğunu CSV'ye yaz.

    Önce normal tablo, ardından boş satır, ardından "# ÖZET" başlığı ve özet
    metni/sayıları yorum satırları olarak eklenir (pandas tekrar okurken bu
    satırlar comment='#' ile atlanabilir).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    lines = [
        "",
        "# ÖZET (toplu)",
        f"# {summary['text']}",
        f"# n_combos,{summary['n_combos']}",
        f"# n_tested,{summary.get('n_tested', '')}",
        f"# n_degenerate,{summary.get('n_degenerate', '')}",
        f"# n_positive_bias,{summary['n_positive_bias']}",
        f"# n_significant_fdr,{summary['n_significant_fdr']}",
        f"# n_significant_positive,{summary['n_significant_positive']}",
        f"# n_low_n_warning,{summary['n_low_n_warning']}",
    ]
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def run_stats(
    raw_csv: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Çoklu-seed ham CSV'sini oku, istatistikleri hesapla, stats_summary.csv yaz.

    Parameters
    ----------
    raw_csv : Path | None
        Per-seed ham tablo. None → ``out_dir/selection_bias_multiseed_raw.csv``.
    out_dir : Path | None
        Çıktı kökü. None → outputs/17_selection_bias.

    Returns
    -------
    Path
        Yazılan ``stats_summary.csv``.
    """
    out_dir = out_dir or (paths.OUTPUTS_DIR / "17_selection_bias")
    raw_csv = raw_csv or (out_dir / "selection_bias_multiseed_raw.csv")
    if not Path(raw_csv).exists():
        raise FileNotFoundError(
            f"Per-seed ham tablo bulunamadı: {raw_csv}\n"
            "Önce çoklu-seed koşusunu çalıştırın: "
            "python run_selection_bias.py --seeds 0,1,2,3,4 ..."
        )
    raw = pd.read_csv(raw_csv)
    log.info("Ham per-seed tablo okundu: %s (%d satır)", raw_csv, len(raw))

    df = compute_stats(raw)
    if df.empty:
        log.warning("İstatistik tablosu boş — kombinasyon yok.")
    summary = summarize(df) if not df.empty else {
        "n_combos": 0, "n_positive_bias": 0, "n_significant_fdr": 0,
        "n_significant_positive": 0, "n_low_n_warning": 0,
        "n_tested": 0, "n_degenerate": 0,
        "text": "Kombinasyon yok.",
    }

    path = out_dir / "stats_summary.csv"
    _write_with_summary(df, summary, path)
    log.info("İstatistik özeti yazıldı: %s (%d kombinasyon)", path, len(df))
    log.info("TOPLU ÖZET: %s", summary["text"])
    return path


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO)
    run_stats()
