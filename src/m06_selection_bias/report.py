"""Seçim-yanlılığı raporu — özet tablo + figürler (Adım 6).

Girdi: bir veya birden çok ``NestedResult`` (nested_ga) ve opsiyonel
``NullResult`` (synthetic_null). Çıktı: ``selection_bias_summary.csv`` ve figürler.

Çalıştırma: orchestrator (run_selection_bias.py) tarafından çağrılır, ya da:
    python -m src.m06_selection_bias.report
(tek bir gerçek-veri sonucu üretip yazar — hızlı duman testi için ağır ayar değil).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from src.core import paths
from src.core.logging_setup import get as get_logger
from src.m06_selection_bias.nested_ga import NestedResult

log = get_logger("m06_selection_bias.report")


def _out_dir() -> Path:
    d = paths.OUTPUTS_DIR / "17_selection_bias"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_summary_csv(results: list[NestedResult], out_dir: Path | None = None) -> Path:
    """dataset×target×model → apparent/nested/bias/std tablosu yaz."""
    out_dir = out_dir or _out_dir()
    rows = []
    for r in results:
        a_ci = r.apparent_ci or {}
        n_ci = r.nested_ci or {}
        rows.append({
            "dataset": r.dataset, "target": r.target, "model": r.model,
            "selector": getattr(r, "selector", "ga"), "k": getattr(r, "k", None),
            "n_samples": r.n_samples, "n_features": r.n_features,
            "n_groups": r.n_groups,
            "apparent_r2": r.apparent_r2,
            "apparent_ci_lo": a_ci.get("lo"), "apparent_ci_hi": a_ci.get("hi"),
            "nested_r2": r.nested_r2, "nested_r2_std": r.nested_r2_std,
            "nested_pooled_r2": r.nested_pooled_r2,
            "nested_ci_lo": n_ci.get("lo"), "nested_ci_hi": n_ci.get("hi"),
            "bias": r.bias,
            # Çoklu metrik: RMSE/MAE (bias yönü R²'nin tersi — pozitif = apparent iyimser)
            "apparent_rmse": getattr(r, "apparent_rmse", None),
            "nested_rmse": getattr(r, "nested_rmse", None),
            "bias_rmse": getattr(r, "bias_rmse", None),
            "apparent_mae": getattr(r, "apparent_mae", None),
            "nested_mae": getattr(r, "nested_mae", None),
            "bias_mae": getattr(r, "bias_mae", None),
            "fold_scores": ";".join(f"{s:.4f}" for s in r.fold_scores),
        })
    df = pd.DataFrame(rows)
    path = out_dir / "selection_bias_summary.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Özet yazıldı: %s (%d satır)", path, len(df))
    return path


def fig_apparent_vs_nested(results: list[NestedResult], out_dir: Path | None = None) -> Path:
    """apparent vs nested R² — gruplu bar grafiği."""
    out_dir = out_dir or _out_dir()
    labels = [f"{r.dataset}\n{r.target}/{r.model}/{getattr(r, 'selector', 'ga')}"
              for r in results]
    app = [r.apparent_r2 for r in results]
    nes = [r.nested_r2 for r in results]

    def _yerr(rs, attr_ci, centers):
        """%95 bootstrap GA varsa asimetrik hata çubuğu, yoksa None."""
        lo, hi = [], []
        ok = True
        for r, c in zip(rs, centers):
            ci = getattr(r, attr_ci)
            if ci is None or not np.isfinite(ci.get("lo", np.nan)):
                ok = False
                break
            lo.append(max(0.0, c - ci["lo"])); hi.append(max(0.0, ci["hi"] - c))
        return np.array([lo, hi]) if ok else None

    app_err = _yerr(results, "apparent_ci", app)
    nes_err = _yerr(results, "nested_ci", nes)
    if nes_err is None:  # CI yoksa fold std'ye düş
        nes_err = [r.nested_r2_std for r in results]

    x = np.arange(len(results))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(results)), 5))
    ax.bar(x - w / 2, app, w, yerr=app_err, capsize=4,
           label="apparent (biased)", color="indianred")
    ax.bar(x + w / 2, nes, w, yerr=nes_err, capsize=4,
           label="nested (honest)", color="steelblue")
    for i, (a, n) in enumerate(zip(app, nes)):
        ax.annotate(f"bias\n{a - n:+.2f}", (i, max(a, n) + 0.02),
                    ha="center", fontsize=8, color="black")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("R²")
    ax.set_title("Seçim yanlılığı: apparent vs nested-CV R²", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    p = out_dir / "fig_apparent_vs_nested.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def fig_fold_variance(results: list[NestedResult], out_dir: Path | None = None) -> Path:
    """Dış-fold R² dağılımı — küçük örneklemde nested varyansının yüksekliği."""
    out_dir = out_dir or _out_dir()
    data = [r.fold_scores for r in results]
    labels = [f"{r.dataset}\n{r.target}/{r.model}/{getattr(r, 'selector', 'ga')}"
              for r in results]
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(results)), 5))
    ax.boxplot(data, labels=labels, showmeans=True)
    for i, fs in enumerate(data, start=1):
        ax.scatter(np.full(len(fs), i), fs, alpha=0.6, color="darkorange", zorder=3)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Dış-fold nested R²")
    ax.set_title("Fold varyansı (nested-CV)", fontweight="bold")
    plt.tight_layout()
    p = out_dir / "fig_fold_variance.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def fig_band_frequency(result: NestedResult, out_dir: Path | None = None) -> Path:
    """Bir sonucun dış-fold maskelerinde band seçilme frekansı."""
    out_dir = out_dir or _out_dir()
    if not result.masks:
        raise ValueError("masks boş — band frekansı çizilemiyor.")
    freq = np.mean(np.vstack([m.astype(float) for m in result.masks]), axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(freq)), freq, color="seagreen", width=1.0)
    ax.set_xlabel("Öznitelik indeksi")
    ax.set_ylabel("Seçilme frekansı (dış fold'lar)")
    sel = getattr(result, "selector", "ga")
    ax.set_title(
        f"Band seçilme kararlılığı — {result.dataset}/{result.target}/{result.model}/{sel}",
        fontweight="bold")
    plt.tight_layout()
    p = out_dir / (f"fig_band_frequency_{result.dataset}_{result.target}_"
                   f"{result.model}_{sel}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def fig_null_distribution(null_result, out_dir: Path | None = None) -> Path:
    """Sentetik null: apparent vs nested R² dağılımı (histogram)."""
    out_dir = out_dir or _out_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(min(null_result.apparent), min(null_result.nested)) - 0.05,
        max(max(null_result.apparent), max(null_result.nested)) + 0.05, 25)
    ax.hist(null_result.apparent, bins=bins, alpha=0.6, color="indianred",
            label=f"apparent (ort={null_result.apparent_mean:+.3f})")
    ax.hist(null_result.nested, bins=bins, alpha=0.6, color="steelblue",
            label=f"nested (ort={null_result.nested_mean:+.3f})")
    ax.axvline(0, color="gray", lw=1.0, ls="--")
    ax.set_xlabel("R²"); ax.set_ylabel("Tekrar sayısı")
    ax.set_title("Sentetik null: gerçek sinyal yokken R² dağılımı", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    p = out_dir / "fig_null_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def _fig_one_permutation(perm: dict, out_dir: Path) -> Path:
    """Tek kombinasyon için permütasyon histogramı (apparent + nested) + observed.

    İki panel: solda apparent_r2, sağda nested_r2. Permütasyon null dağılımı
    histogram; observed (gerçek y) değeri dikey çizgi. Beklenti: apparent dağılımı
    POZİTİF tarafa yığılır (sinyal yokken bile), nested ≈0; observed apparent ise
    dağılımın daha da sağında (gerçek sinyal + seçim yanlılığı).
    """
    pa = np.asarray(perm["perm_apparent_r2"], dtype=float)
    pn = np.asarray(perm["perm_nested_r2"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    panels = [
        (axes[0], pa, perm["observed_apparent_r2"], "apparent (CV-dışı seçim)",
         "indianred", perm["perm_p_value_apparent"]),
        (axes[1], pn, perm["observed_nested_r2"], "nested (dürüst)",
         "steelblue", perm["perm_p_value"]),
    ]
    for ax, dist, obs, title, color, pval in panels:
        finite = dist[np.isfinite(dist)]
        if finite.size:
            lo = min(finite.min(), obs) - 0.05
            hi = max(finite.max(), obs) + 0.05
            ax.hist(finite, bins=np.linspace(lo, hi, 30), color=color, alpha=0.6,
                    label=f"permütasyon (ort={np.mean(finite):+.3f})")
        ax.axvline(0, color="gray", lw=1.0, ls="--")
        ax.axvline(obs, color="black", lw=2.0,
                   label=f"observed={obs:+.3f}\np={pval:.4f}")
        ax.set_xlabel("R²"); ax.set_ylabel("Permütasyon sayısı")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Permütasyon testi — {perm['dataset']}/{perm['target']}/"
        f"{perm['model']}/{perm['selector']} (n_perm={perm['n_perm']})",
        fontweight="bold")
    plt.tight_layout()
    p = out_dir / (f"fig_permutation_{perm['dataset']}_{perm['target']}_"
                   f"{perm['model']}_{perm['selector']}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def fig_permutation(perm_results: list[dict], out_dir: Path | None = None) -> dict[str, Path]:
    """Permütasyon testi figürleri: her kombinasyon için bir panel-çifti.

    Ek olarak ana kombinasyon (ryckewaert/flavonol/ga varsa, yoksa ilki)
    ``fig_permutation.png`` adıyla da kaydedilir (makale ana figürü).

    Returns
    -------
    dict[str, Path]
        {"permutation": <ana fig>, "permutation_<combo>": <per-combo fig>, ...}
    """
    out_dir = out_dir or _out_dir()
    if not perm_results:
        raise ValueError("perm_results boş — permütasyon figürü çizilemiyor.")
    produced: dict[str, Path] = {}
    for perm in perm_results:
        key = f"permutation_{perm['dataset']}_{perm['target']}_{perm['model']}_{perm['selector']}"
        produced[key] = _fig_one_permutation(perm, out_dir)

    # Ana kombinasyon → fig_permutation.png
    main = next((d for d in perm_results
                 if d["dataset"] == "ryckewaert" and d["target"] == "flavonol"
                 and d["selector"] == "ga"), perm_results[0])
    main_path = out_dir / "fig_permutation.png"
    import shutil
    src_key = f"permutation_{main['dataset']}_{main['target']}_{main['model']}_{main['selector']}"
    shutil.copyfile(produced[src_key], main_path)
    produced["permutation"] = main_path
    log.info("Permütasyon figürleri yazıldı: %d (+ ana fig_permutation.png)",
             len(perm_results))
    return produced


def build_multiseed_csv(records: list[dict], out_dir: Path | None = None) -> tuple[Path, Path]:
    """Çoklu-seed kayıtlarından özet + ham CSV yaz.

    records : list[dict]
        multiseed.run_multiseed çıktısı; her eleman {seed, dataset, target,
        model, result: NestedResult}.

    Returns
    -------
    (summary_path, raw_path)
        selection_bias_multiseed.csv (seed'ler boyunca ort/std/min/max) ve
        selection_bias_multiseed_raw.csv (ham per-seed apparent/nested/bias).
    """
    out_dir = out_dir or _out_dir()
    raw_rows = []
    for rec in records:
        r = rec["result"]
        raw_rows.append({
            "seed": rec["seed"], "dataset": rec["dataset"],
            "target": rec["target"], "model": rec["model"],
            "selector": rec.get("selector", getattr(r, "selector", "ga")),
            "apparent_r2": r.apparent_r2, "nested_r2": r.nested_r2, "bias": r.bias,
            "apparent_rmse": getattr(r, "apparent_rmse", None),
            "nested_rmse": getattr(r, "nested_rmse", None),
            "bias_rmse": getattr(r, "bias_rmse", None),
            "apparent_mae": getattr(r, "apparent_mae", None),
            "nested_mae": getattr(r, "nested_mae", None),
            "bias_mae": getattr(r, "bias_mae", None),
        })
    raw = pd.DataFrame(raw_rows).sort_values(
        ["dataset", "target", "model", "selector", "seed"]).reset_index(drop=True)
    raw_path = out_dir / "selection_bias_multiseed_raw.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8")

    agg = raw.groupby(["dataset", "target", "model", "selector"],
                      as_index=False, sort=False).agg(
        n_seeds=("seed", "nunique"),
        apparent_mean=("apparent_r2", "mean"), apparent_std=("apparent_r2", "std"),
        apparent_min=("apparent_r2", "min"), apparent_max=("apparent_r2", "max"),
        nested_mean=("nested_r2", "mean"), nested_std=("nested_r2", "std"),
        nested_min=("nested_r2", "min"), nested_max=("nested_r2", "max"),
        bias_mean=("bias", "mean"), bias_std=("bias", "std"),
    )
    sum_path = out_dir / "selection_bias_multiseed.csv"
    agg.to_csv(sum_path, index=False, encoding="utf-8")
    log.info("Çoklu-seed özet yazıldı: %s (%d kombinasyon) + ham %s (%d satır)",
             sum_path, len(agg), raw_path, len(raw))
    return sum_path, raw_path


def fig_seed_stability(records: list[dict], out_dir: Path | None = None) -> Path:
    """Her kombinasyon için apparent (kırmızı) vs nested (mavi) R² dağılımı.

    Yan yana box + strip plot. Beklenti: apparent kutusu geniş ve yüksek
    (kararsız), nested kutusu dar ve düşük (kararlı).
    """
    from matplotlib.patches import Patch

    out_dir = out_dir or _out_dir()
    combos: list[tuple[str, str, str, str]] = []
    data: dict[tuple[str, str, str, str], dict[str, list[float]]] = {}
    for rec in records:
        sel = rec.get("selector", getattr(rec["result"], "selector", "ga"))
        key = (rec["dataset"], rec["target"], rec["model"], sel)
        if key not in data:
            data[key] = {"apparent": [], "nested": []}
            combos.append(key)
        data[key]["apparent"].append(rec["result"].apparent_r2)
        data[key]["nested"].append(rec["result"].nested_r2)

    labels = [f"{d}\n{t}/{m}/{s}" for d, t, m, s in combos]
    app_data = [data[k]["apparent"] for k in combos]
    nes_data = [data[k]["nested"] for k in combos]
    x = np.arange(len(combos))
    w = 0.3

    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(combos)), 5))
    bp1 = ax.boxplot(app_data, positions=x - w / 2, widths=w * 0.8,
                     patch_artist=True, manage_ticks=False)
    bp2 = ax.boxplot(nes_data, positions=x + w / 2, widths=w * 0.8,
                     patch_artist=True, manage_ticks=False)
    for b in bp1["boxes"]:
        b.set_facecolor("indianred"); b.set_alpha(0.6)
    for b in bp2["boxes"]:
        b.set_facecolor("steelblue"); b.set_alpha(0.6)
    for i in range(len(combos)):
        ax.scatter(np.full(len(app_data[i]), x[i] - w / 2), app_data[i],
                   color="darkred", s=18, zorder=3)
        ax.scatter(np.full(len(nes_data[i]), x[i] + w / 2), nes_data[i],
                   color="navy", s=18, zorder=3)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("R²")
    ax.set_title("Seed kararlılığı: apparent (kararsız) vs nested (kararlı)",
                 fontweight="bold")
    ax.legend(handles=[
        Patch(facecolor="indianred", alpha=0.6, label="apparent (biased)"),
        Patch(facecolor="steelblue", alpha=0.6, label="nested (honest)"),
    ])
    plt.tight_layout()
    p = out_dir / "fig_seed_stability.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def fig_cv_comparison(rows: list[dict], out_dir: Path | None = None) -> Path:
    """Üç-pipeline (P1/P2/P3) R² gruplu bar — azalan basamak + leakage etiketleri.

    Her kombinasyon için P1 (random+dış seçim), P2 (group+dış seçim), P3 (nested)
    üç bar yan yana. P1→P2→P3 azalan görünmeli. group_leakage (P1−P2) ve
    selection_leakage (P2−P3) bar üstünde etiketlenir.
    """
    out_dir = out_dir or _out_dir()
    if not rows:
        raise ValueError("rows boş — CV karşılaştırma figürü çizilemiyor.")
    labels = [f"{r['dataset']}\n{r['target']}/{r['model']}/{r['selector']}\n"
              f"(n_grp={r['n_groups']})" for r in rows]
    p1 = [r["p1_r2"] for r in rows]
    p2 = [r["p2_r2"] for r in rows]
    p3 = [r["p3_r2"] for r in rows]

    x = np.arange(len(rows))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 2.0 * len(rows)), 5.5))
    ax.bar(x - w, p1, w, label="P1 random + dış seçim", color="firebrick")
    ax.bar(x, p2, w, label="P2 group + dış seçim", color="darkorange")
    ax.bar(x + w, p3, w, label="P3 nested (dürüst)", color="steelblue")

    for i, r in enumerate(rows):
        top = max([v for v in (p1[i], p2[i], p3[i]) if np.isfinite(v)] or [0.0])
        gl, sl = r["group_leakage"], r["selection_leakage"]
        gl_s = f"{gl:+.2f}" if np.isfinite(gl) else "n/a"
        sl_s = f"{sl:+.2f}" if np.isfinite(sl) else "n/a"
        ax.annotate(f"grup {gl_s}\nseçim {sl_s}", (i, top + 0.02),
                    ha="center", fontsize=8, color="black")

    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("R²")
    ax.set_title("Üç-pipeline CV: grup vs seçim sızıntısının ayrıştırılması",
                 fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    p = out_dir / "fig_cv_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info("CV karşılaştırma figürü yazıldı: %s", p)
    return p


def generate_multiseed(records: list[dict], out_dir: Path | None = None) -> dict[str, Path]:
    """Çoklu-seed çıktılarını üret: özet CSV + ham CSV + kararlılık figürü."""
    out_dir = out_dir or _out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, Path] = {}
    if not records:
        log.warning("Çoklu-seed kaydı yok — çıktı üretilmedi.")
        return produced
    sum_path, raw_path = build_multiseed_csv(records, out_dir)
    produced["multiseed_csv"] = sum_path
    produced["multiseed_raw_csv"] = raw_path
    produced["seed_stability"] = fig_seed_stability(records, out_dir)
    log.info("Çoklu-seed raporu: %d çıktı → %s", len(produced), out_dir)
    return produced


def generate_all(
    results: list[NestedResult],
    null_result=None,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """Tüm rapor çıktılarını üret. Eksik/boş girdiler atlanır."""
    out_dir = out_dir or _out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, Path] = {}
    if results:
        produced["summary_csv"] = build_summary_csv(results, out_dir)
        produced["apparent_vs_nested"] = fig_apparent_vs_nested(results, out_dir)
        produced["fold_variance"] = fig_fold_variance(results, out_dir)
        for r in results:
            if r.masks:
                sel = getattr(r, "selector", "ga")
                key = f"band_freq_{r.dataset}_{r.target}_{r.model}_{sel}"
                produced[key] = fig_band_frequency(r, out_dir)
    if null_result is not None:
        produced["null_distribution"] = fig_null_distribution(null_result, out_dir)
    log.info("Rapor üretildi: %d çıktı → %s", len(produced), out_dir)
    return produced


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    from src.m01_io.dataset_registry import load_ryckewaert
    from src.m06_selection_bias.nested_ga import nested_ga_evaluation

    ds = load_ryckewaert()
    r = nested_ga_evaluation(
        ds.X, ds.target("flavonol"), ds.groups, model="pls", n_outer=5,
        ga_cfg={"pop": 20, "ngen": 8, "n_jobs": -1},
        dataset=ds.name, target="flavonol",
    )
    out = generate_all([r])
    print("Üretilen:", {k: str(v) for k, v in out.items()})
