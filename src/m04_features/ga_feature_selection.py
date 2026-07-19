"""Genetik Algoritma (GA) tabanlı özellik seçimi — flavonol regresyonu için.

Bu modül, mevcut hiperspektral pipeline'a entegre çalışır. 462 özellikten
oluşan binary maske üzerinde GA (DEAP) ile arama yapar; fitness olarak
seçilen alt küme üzerinde 5-fold CV R² skorunu kullanır.

Çalıştırma:
    # Pipeline aşaması olarak (main.py içinden)
    from src.m04_features import ga_feature_selection
    ga_feature_selection.run(cfg)

    # Bağımsız komut satırı (Windows PowerShell):
    python -m src.m04_features.ga_feature_selection --target flavonol --model ridge
    python -m src.m04_features.ga_feature_selection --model lightgbm --pop 200 --ngen 150

Çıktılar (varsayılan ``outputs/12_ga_feature_selection/<target>_<model>/``;
``--output-dir`` ile ek bir kopya hedefi belirtilebilir):
    - ga_best_mask.npy
    - ga_best_features.txt
    - ga_logbook.csv
    - ga_convergence.png
    - ga_comparison.csv
    - ga_summary.txt
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# LGBMRegressor pandas öznitelik adlarıyla fit edilip ndarray ile predict
# edilince çıkan zararsız uyarıyı sustur. Modül seviyesinde — böylece GA'nın
# loky/joblib işçi süreçlerinde de (modülü yeniden import ederler) geçerli olur.
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning,
)

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.core import paths
from src.core.cv import load_groups, make_cv_splitter
from src.core.logging_setup import get as get_logger

log = get_logger("m04_features.ga_feature_selection")

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
TARGETS: dict[str, str] = {
    "chlorophyll": "y_chl",
    "flavonol": "y_flav",
    "nbi": "y_nbi",
}

MODEL_CHOICES = ("ridge", "rf", "lightgbm", "pls", "elasticnet", "svr", "xgboost")
MIN_FEATURES = 5  # birey en az bu kadar özellik seçmeli (yoksa penalty)


# ---------------------------------------------------------------------------
# DEAP'ı tembel-import et (workerlarda yeniden import edileceği için)
# ---------------------------------------------------------------------------
def _check_deap() -> None:
    """DEAP yoksa anlaşılır hata mesajı ver ve çık."""
    try:
        import deap  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "HATA: DEAP yüklü değil. Lütfen şunu çalıştırın:\n"
            "    pip install deap\n"
        )
        raise


# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------
def _load_dataset(
    target: str, group_key: str = "leaf",
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    """Pipeline'ın 01_dataset çıktılarından X, y_<target>, özellik isimlerini yükle.

    Parameters
    ----------
    target : {"chlorophyll", "flavonol", "nbi"}
        Hedef değişken adı.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    feature_names : list[str]

    Raises
    ------
    FileNotFoundError
        Beklenen dosyalar 01_dataset altında yoksa.
    """
    if target not in TARGETS:
        raise ValueError(f"Bilinmeyen hedef: {target!r}. Seçenekler: {list(TARGETS)}")

    ds_dir = paths.OUTPUTS_DIR / "01_dataset"
    x_path = ds_dir / "X.npy"
    y_path = ds_dir / f"{TARGETS[target]}.npy"
    fn_path = ds_dir / "feature_names.json"

    for p in (x_path, y_path, fn_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Beklenen dosya yok: {p}\n"
                f"Önce dataset aşamasını çalıştırın: python main.py --stages 01_dataset"
            )

    X = np.load(x_path)
    y = np.load(y_path)
    feature_names = json.loads(fn_path.read_text(encoding="utf-8"))
    groups = load_groups(ds_dir, key=group_key)

    valid = np.isfinite(y)
    groups_v = groups[valid] if groups is not None else None
    return X[valid], y[valid], feature_names, groups_v


# ---------------------------------------------------------------------------
# Fitness fonksiyonu
# ---------------------------------------------------------------------------
# Worker süreçleri global X/y'ye erişsin diye modül seviyesinde tutulur
_GLOBAL_X: np.ndarray | None = None
_GLOBAL_Y: np.ndarray | None = None
_GLOBAL_GROUPS: np.ndarray | None = None
_GLOBAL_MODEL: str = "ridge"
_GLOBAL_SEED: int = 42


def _build_regressor(model_type: str, n_features_sel: int, n_samples: int):
    """Seçilen alt küme için sklearn-uyumlu regresör örneği üret.

    Worker içinde çağrıldığı için model ``n_jobs=1`` ile sınırlanır
    (multiprocessing × intra-model paralelizm çakışmasını önler).
    """
    if model_type == "ridge":
        return Ridge(alpha=1.0)
    if model_type == "rf":
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    if model_type == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=100, random_state=42, n_jobs=1,
                             verbose=-1, verbosity=-1, force_col_wise=True)
    if model_type == "pls":
        n_comp = max(1, min(10, n_features_sel, n_samples - 1))
        return PLSRegression(n_components=n_comp)
    if model_type == "elasticnet":
        # Hiperspektral bantlar yüksek korelasyonlu — ElasticNet (L1+L2) Ridge'den
        # daha seyrek bir çözüm üretir; GA seçimi üstüne ekstra düzenlileştirme.
        return SkPipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42)),
        ])
    if model_type == "svr":
        # Küçük n + non-linear ilişkiler için RBF-SVR. Ölçek hassas → scaler şart.
        return SkPipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=1.0, gamma="scale")),
        ])
    if model_type == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=1,
            verbosity=0,
            tree_method="hist",
        )
    raise ValueError(f"Bilinmeyen model: {model_type!r}")


def _evaluate_individual(individual: list[int]) -> tuple[float]:
    """Tek bir bireyin (binary maske) fitness değerini hesapla.

    Çok az özellik seçen bireylere penalty (0.0). 5-fold CV R² ortalaması
    döndürülür; negatif R² 0.0'a sabitlenir.
    """
    selected = np.asarray(individual, dtype=bool)
    if int(selected.sum()) < MIN_FEATURES:
        return (0.0,)

    assert _GLOBAL_X is not None and _GLOBAL_Y is not None
    X_sel = _GLOBAL_X[:, selected]
    y = _GLOBAL_Y
    groups = _GLOBAL_GROUPS

    try:
        model = _build_regressor(_GLOBAL_MODEL, X_sel.shape[1], X_sel.shape[0])
        cv = make_cv_splitter(
            n_splits=5, task="regression",
            groups=groups, random_state=_GLOBAL_SEED,
        )
        if groups is not None:
            scores = cross_val_score(
                model, X_sel, y, cv=cv, scoring="r2", n_jobs=1, groups=groups,
            )
        else:
            scores = cross_val_score(model, X_sel, y, cv=cv, scoring="r2", n_jobs=1)
        mean_r2 = float(np.mean(scores))
        return (max(mean_r2, 0.0),)
    except Exception:
        return (0.0,)


def _init_worker(
    X: np.ndarray,
    y: np.ndarray,
    model: str,
    seed: int,
    groups: np.ndarray | None = None,
) -> None:
    """multiprocessing.Pool initializer — global'leri her worker'a yerleştir."""
    global _GLOBAL_X, _GLOBAL_Y, _GLOBAL_GROUPS, _GLOBAL_MODEL, _GLOBAL_SEED
    _GLOBAL_X = X
    _GLOBAL_Y = y
    _GLOBAL_GROUPS = groups
    _GLOBAL_MODEL = model
    _GLOBAL_SEED = int(seed)


# ---------------------------------------------------------------------------
# DEAP toolbox kurulumu
# ---------------------------------------------------------------------------
def _build_toolbox(n_features: int, seed: int):
    """DEAP toolbox'ını kur: birey/popülasyon/operatör tanımları."""
    from deap import base, creator, tools

    # creator sınıfları process-safe olsun diye hasattr ile kontrollü tanım
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n_features,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", _evaluate_individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / n_features)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


# ---------------------------------------------------------------------------
# GA döngüsü
# ---------------------------------------------------------------------------
def _run_ga(
    X: np.ndarray,
    y: np.ndarray,
    model: str,
    pop_size: int,
    n_gen: int,
    seed: int,
    n_jobs: int,
    groups: np.ndarray | None = None,
    write_checkpoint: bool = True,
) -> tuple[np.ndarray, list[dict], object]:
    """GA döngüsünü çalıştır ve en iyi maske + logbook + HOF döndür.

    ``write_checkpoint`` False ise diske checkpoint yazılmaz (saf/yan-etkisiz
    mod; nested-CV içinde dış-fold başına çağrılırken kullanılır). True iken
    eski pipeline davranışı birebir korunur.
    """
    from deap import tools

    n_features = X.shape[1]
    random.seed(seed)
    np.random.seed(seed)

    toolbox = _build_toolbox(n_features, seed)
    hof = tools.HallOfFame(5)
    pop = toolbox.population(n=pop_size)

    cxpb, mutpb = 0.7, 0.3

    # multiprocessing pool (Windows uyumlu: __main__ guard zorunlu)
    pool = None
    if n_jobs is None or n_jobs == 0:
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = max(1, mp.cpu_count())

    if n_jobs > 1:
        pool = mp.Pool(
            processes=n_jobs,
            initializer=_init_worker,
            initargs=(X, y, model, seed, groups),
        )
        toolbox.register("map", pool.map)
        log.info("Pool: %d worker", n_jobs)
    else:
        _init_worker(X, y, model, seed, groups)

    # Checkpoint klasörü — Ctrl+C / erken durdurmada güncel HOF'u kurtarır
    # (eski reports/ga/_checkpoint kaldırıldı → numaralı aşama dizini)
    ckpt_dir = paths.OUTPUTS_DIR / "12_ga_feature_selection" / "_checkpoint"
    if write_checkpoint:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(gen_done: int) -> None:
        if not write_checkpoint or len(hof) == 0:
            return
        np.save(ckpt_dir / "best_mask.npy", np.asarray(hof[0], dtype=bool))
        (ckpt_dir / "info.txt").write_text(
            f"gen_done={gen_done}\nbest_r2={hof[0].fitness.values[0]:.6f}\n"
            f"n_selected={int(np.asarray(hof[0]).sum())}\n",
            encoding="utf-8",
        )

    log_rows: list[dict] = []
    interrupted = False
    try:
        # İlk popülasyonu değerlendir
        invalid = [ind for ind in pop if not ind.fitness.valid]
        fits = list(toolbox.map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit
        hof.update(pop)
        _log_generation(0, n_gen, pop, log_rows)
        _save_checkpoint(0)

        for gen in range(1, n_gen + 1):
            # Seçim — elitleri (HOF) ekle, böylece her zaman korunur
            offspring = toolbox.select(pop, k=len(pop) - len(hof))
            offspring = [list(ind) for ind in offspring]
            from deap import creator
            offspring = [creator.Individual(o) for o in offspring]

            # Çaprazlama
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            # Mutasyon
            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Elitleri ekle
            elites = [creator.Individual(list(h)) for h in hof]
            for e, h in zip(elites, hof):
                e.fitness.values = h.fitness.values
            offspring.extend(elites)

            # Yeniden değerlendirme
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(toolbox.map(toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)
            _log_generation(gen, n_gen, pop, log_rows)
            _save_checkpoint(gen)
    except KeyboardInterrupt:
        interrupted = True
        log.warning("KeyboardInterrupt — mevcut en iyi sonuç kaydediliyor "
                    "(tamamlanan nesil=%d)", log_rows[-1]["gen"] if log_rows else 0)
    finally:
        if pool is not None:
            pool.terminate() if interrupted else pool.close()
            pool.join()

    best = np.asarray(hof[0], dtype=bool)
    return best, log_rows, hof


def run_ga_core(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    model: str,
    pop: int = 150,
    ngen: int = 150,
    seed: int = 42,
    n_jobs: int = -1,
) -> np.ndarray:
    """Saf GA: verilen (X, y, groups) üzerinde en iyi band maskesini döndürür.

    DOSYA YAZMAZ, çizmez, checkpoint tutmaz — sadece algoritma. Nested-CV'de
    her dış-fold'un dış-train'i üzerinde bağımsızca çağrılmak için tasarlandı.
    ``run(cfg)`` / pipeline çıktı davranışı bu fonksiyondan etkilenmez.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    groups : ndarray | None
        GroupKFold için grup etiketleri (None → gruplama kapalı).
    model : str
        Fitness regresörü (bkz. ``MODEL_CHOICES``).
    pop, ngen, seed, n_jobs : int
        GA hiperparametreleri (``_run_ga`` ile aynı anlam).

    Returns
    -------
    best_mask : ndarray[bool], shape (n_features,)
        En iyi bireyin band maskesi.
    """
    best_mask, _log_rows, _hof = _run_ga(
        X, y, model=model, pop_size=pop, n_gen=ngen,
        seed=seed, n_jobs=n_jobs, groups=groups,
        write_checkpoint=False,
    )
    return best_mask


def _log_generation(gen: int, n_gen: int, pop: list, log_rows: list[dict]) -> None:
    """Bir nesil için istatistikleri hesapla, logla ve logbook'a ekle."""
    fitnesses = np.asarray([ind.fitness.values[0] for ind in pop], dtype=float)
    sizes = np.asarray([sum(ind) for ind in pop], dtype=float)
    row = {
        "gen": gen,
        "max_r2": float(fitnesses.max()),
        "avg_r2": float(fitnesses.mean()),
        "min_r2": float(fitnesses.min()),
        "std_r2": float(fitnesses.std()),
        "avg_n_features": float(sizes.mean()),
        "std_n_features": float(sizes.std()),
    }
    log_rows.append(row)
    if gen == 0 or gen == n_gen or gen % 10 == 0:
        log.info(
            "Nesil %3d/%d | Max R²: %.3f | Ort R²: %.3f | "
            "Seçilen özellik: %.0f±%.0f",
            gen, n_gen, row["max_r2"], row["avg_r2"],
            row["avg_n_features"], row["std_n_features"],
        )


# ---------------------------------------------------------------------------
# Karşılaştırma raporu
# ---------------------------------------------------------------------------
def _evaluate_models_cv(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    groups: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Verilen X üzerinde 4 modeli 5-fold CV ile değerlendir, R²/RMSE/RPD döndür."""
    cv = make_cv_splitter(
        n_splits=5, task="regression", groups=groups, random_state=seed,
    )
    results: dict[str, dict[str, float]] = {}
    for name in MODEL_CHOICES:
        try:
            model = _build_regressor(name, X.shape[1], X.shape[0])
            from sklearn.model_selection import cross_val_predict
            if groups is not None:
                y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1, groups=groups)
            else:
                y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
            r2 = float(r2_score(y, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
            rpd = float(np.std(y) / rmse) if rmse > 0 else 0.0
            results[name] = {"R2": r2, "RMSE": rmse, "RPD": rpd}
        except Exception as exc:
            log.warning("Model %s değerlendirilemedi: %s", name, exc)
            results[name] = {"R2": float("nan"), "RMSE": float("nan"), "RPD": float("nan")}
    return results


def _build_comparison(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    seed: int,
    groups: np.ndarray | None = None,
) -> pd.DataFrame:
    """Tüm özellikler vs GA seçimi karşılaştırmasını üret."""
    log.info("Karşılaştırma: tüm özellikler (n=%d) vs GA seçimi (n=%d)",
             X.shape[1], int(mask.sum()))
    full = _evaluate_models_cv(X, y, seed, groups=groups)
    selected = _evaluate_models_cv(X[:, mask], y, seed, groups=groups)

    rows = []
    for m in MODEL_CHOICES:
        f = full[m]
        s = selected[m]
        rows.append({
            "model": m,
            "n_features_full": X.shape[1],
            "R2_full": f["R2"], "RMSE_full": f["RMSE"], "RPD_full": f["RPD"],
            "n_features_ga": int(mask.sum()),
            "R2_ga": s["R2"], "RMSE_ga": s["RMSE"], "RPD_ga": s["RPD"],
            "delta_R2": s["R2"] - f["R2"],
            "delta_RMSE": s["RMSE"] - f["RMSE"],
            "delta_RPD": s["RPD"] - f["RPD"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Çıktı yazma
# ---------------------------------------------------------------------------
def _save_outputs(
    out_dir: Path,
    stage_dir: Path,
    mask: np.ndarray,
    feature_names: list[str],
    log_rows: list[dict],
    comparison: pd.DataFrame,
    target: str,
    model: str,
    pop_size: int,
    n_gen: int,
    elapsed: float,
    cfg_source: Path | None,
) -> None:
    """GA çıktılarını ilgili dizin(ler)e yaz.

    Varsayılan: out_dir == stage_dir (numaralı aşama dizini). CLI ile
    ``--output-dir`` verildiyse out_dir farklı olur ve oraya da kopya yazılır.
    Aynı yola iki kez yazmayı önlemek için set ile tekille.
    """
    dirs = [out_dir] if out_dir == stage_dir else [out_dir, stage_dir]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # 1) En iyi maske
    for d in dirs:
        np.save(d / "ga_best_mask.npy", mask)

    # 2) Seçilen özellik isimleri
    selected_names = [feature_names[i] for i in np.where(mask)[0]]
    feat_txt = "\n".join(selected_names) + "\n"
    for d in dirs:
        (d / "ga_best_features.txt").write_text(feat_txt, encoding="utf-8")

    # 3) Logbook
    log_df = pd.DataFrame(log_rows)
    for d in dirs:
        log_df.to_csv(d / "ga_logbook.csv", index=False, encoding="utf-8")

    # 4) Convergence grafiği
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(log_df["gen"], log_df["max_r2"], color="steelblue",
            linewidth=2, label="Max R² (HOF)")
    ax.plot(log_df["gen"], log_df["avg_r2"], color="darkorange",
            linewidth=1.2, alpha=0.8, label="Ort. R²")
    ax.fill_between(
        log_df["gen"],
        log_df["avg_r2"] - log_df["std_r2"],
        log_df["avg_r2"] + log_df["std_r2"],
        alpha=0.2, color="darkorange", label="±1 std (popülasyon)",
    )
    ax.set_xlabel("Nesil")
    ax.set_ylabel("R² (5-fold CV)")
    ax.set_title(f"GA convergence — target={target}, model={model}",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    for d in dirs:
        fig.savefig(d / "ga_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5) Karşılaştırma CSV
    for d in dirs:
        comparison.to_csv(d / "ga_comparison.csv", index=False, encoding="utf-8")

    # 6) Türkçe özet
    best = comparison.iloc[comparison["R2_ga"].idxmax()]
    summary = [
        "GA Tabanlı Özellik Seçimi — Özet",
        "=" * 50,
        f"Hedef değişken     : {target}",
        f"Fitness modeli     : {model}",
        f"Popülasyon         : {pop_size}",
        f"Nesil sayısı       : {n_gen}",
        f"Toplam süre        : {elapsed:.1f} sn ({elapsed/60:.1f} dk)",
        "",
        f"Toplam özellik     : {len(mask)}",
        f"GA tarafından seçilen: {int(mask.sum())} ({int(mask.sum())/len(mask)*100:.1f}%)",
        "",
        "Karşılaştırma (5-fold CV):",
        "-" * 50,
    ]
    for _, r in comparison.iterrows():
        summary.append(
            f"  {r['model']:>9}  full R²={r['R2_full']:+.3f}  "
            f"→ GA R²={r['R2_ga']:+.3f}  Δ={r['delta_R2']:+.3f}  "
            f"(RPD: {r['RPD_full']:.2f} → {r['RPD_ga']:.2f})"
        )
    summary += [
        "-" * 50,
        f"En iyi GA modeli   : {best['model']} (R²={best['R2_ga']:.3f}, RPD={best['RPD_ga']:.2f})",
        "",
        f"Çıktılar           : {out_dir}",
    ]
    summary_text = "\n".join(summary) + "\n"
    # UTF-8 BOM: Windows Notepad/Excel'in CP1254 sanmaması için
    for d in dirs:
        (d / "ga_summary.txt").write_text(summary_text, encoding="utf-8-sig")

    paths.write_source_marker(
        stage_dir,
        producer="src/m04_features/ga_feature_selection.py",
        config_source=cfg_source,
    )

    # GÖREV 9: GA seçili dalga boyu görselleştirmesi (her GA bitiminde otomatik)
    try:
        from src.m04_features.ga_wavelength_viz import plot_for as _ga_viz
        for d in dirs:
            _ga_viz(d)
    except Exception as exc:
        log.warning("GA wavelength viz atlandı: %s", exc)


# ---------------------------------------------------------------------------
# Pipeline & CLI giriş noktaları
# ---------------------------------------------------------------------------
#: Downstream aşamaların (13_flavonol_combos, 16_final_combos) ihtiyaç duyduğu
#: minimum (target, model) GA kombinasyonları. Liste cfg ile genişletilebilir
#: (``ga.combos: [[target, model], ...]``). Pipeline aşaması olarak çağrıldığında
#: bu listenin tamamı tek tek koşulur; mevcut maskeler atlanır (force ile yenilenir).
_DEFAULT_GA_COMBOS: tuple[tuple[str, str], ...] = (
    ("flavonol", "pls"),
    ("flavonol", "lightgbm"),
)


def _run_single(
    cfg, target: str, model: str, pop_size: int, n_gen: int, seed: int,
    n_jobs: int, group_key: str, output_dir: Path | None, force: bool,
) -> Path:
    """Tek bir (target, model) GA kombinasyonunu çalıştır (eski `run` gövdesi)."""
    _check_deap()
    stage_dir = paths.stage_dir("12_ga_feature_selection", f"{target}_{model}")
    out_dir = Path(output_dir) if output_dir is not None else stage_dir
    mask_path = stage_dir / "ga_best_mask.npy"
    # Cache: maske zaten varsa ve force değilse atla — pipeline'da idempotent
    if mask_path.exists() and not force:
        log.info("GA atlandı (cache): %s/%s → %s", target, model, mask_path)
        return out_dir / "ga_summary.txt"

    log.info("GA başlıyor: target=%s, model=%s, pop=%d, ngen=%d, seed=%d, group=%s",
             target, model, pop_size, n_gen, seed, group_key)
    X, y, feature_names, groups = _load_dataset(target, group_key=group_key)
    log.info("Dataset: X=%s, y=%s (%s) | groups=%s",
             X.shape, y.shape, target,
             "yok" if groups is None else f"{group_key} ({len(set(groups.tolist()))} unique)")

    t0 = time.time()
    best_mask, log_rows, _ = _run_ga(
        X, y, model=model, pop_size=pop_size, n_gen=n_gen,
        seed=seed, n_jobs=n_jobs, groups=groups,
    )
    elapsed = time.time() - t0
    log.info("GA bitti: %d özellik seçildi (%.1f sn)", int(best_mask.sum()), elapsed)

    comparison = _build_comparison(X, y, best_mask, seed, groups=groups)
    _save_outputs(
        out_dir, stage_dir, best_mask, feature_names, log_rows,
        comparison, target, model, pop_size, n_gen, elapsed,
        cfg_source=cfg.source if cfg is not None else None,
    )
    log.info("Çıktılar yazıldı: %s", out_dir)
    return out_dir / "ga_summary.txt"


def run(
    cfg=None,
    target: str = "flavonol",
    model: str = "ridge",
    pop_size: int = 150,
    n_gen: int = 150,
    seed: int = 42,
    n_jobs: int = -1,
    output_dir: Path | None = None,
    group_key: str = "leaf",
    force: bool = False,
) -> Path:
    """Pipeline aşaması: GA tabanlı özellik seçimini çalıştır.

    Parameters
    ----------
    cfg : Config | None
        ``main.py``'den iletilen konfigürasyon nesnesi. None ise varsayılanlar.
    target : str
        ``"chlorophyll" | "flavonol" | "nbi"``.
    model : str
        Fitness regresörü: ``"ridge" | "rf" | "lightgbm" | "pls"``.
    pop_size : int
        Popülasyon boyutu.
    n_gen : int
        Nesil sayısı.
    seed : int
        Random seed.
    n_jobs : int
        Paralel iş sayısı (-1 = tüm çekirdekler).
    output_dir : Path | None
        Override; varsayılan ``outputs/12_ga_feature_selection/<target>_<model>/``.

    Returns
    -------
    Path
        Yazılan ``ga_summary.txt`` yolu.
    """
    # cfg varsa varsayılanları override et
    if cfg is not None:
        pop_size = int(cfg.get("ga.pop", pop_size))
        n_gen = int(cfg.get("ga.ngen", n_gen))
        seed = int(cfg.get("models.random_state", seed))
        group_key = str(cfg.get("cv.group_key", group_key))

    # Pipeline modu: cfg'den combos listesi oku; yoksa default minimum set.
    # CLI tek-kombo modunda `output_dir` verildiğinde geri-uyumluluk için
    # SADECE (target, model) çalıştır.
    if output_dir is not None:
        return _run_single(cfg, target, model, pop_size, n_gen, seed,
                           n_jobs, group_key, output_dir, force)

    combos_cfg = (cfg.get("ga.combos", None) if cfg is not None else None) \
                 or list(_DEFAULT_GA_COMBOS)
    combos = [tuple(c) for c in combos_cfg]
    # cfg.ga.target/model verildiyse listeye ekle (geri-uyumlu CLI)
    if cfg is not None:
        t = cfg.get("ga.target", None)
        m = cfg.get("ga.model", None)
        if t and m and (str(t), str(m)) not in combos:
            combos.append((str(t), str(m)))

    log.info("GA pipeline: %d kombinasyon → %s", len(combos),
             ", ".join(f"{t}_{m}" for t, m in combos))

    last_path: Path | None = None
    for tgt, mdl in combos:
        try:
            last_path = _run_single(cfg, tgt, mdl, pop_size, n_gen, seed,
                                    n_jobs, group_key, None, force)
        except Exception as exc:
            log.exception("GA %s_%s HATA (devam): %s", tgt, mdl, exc)
    return last_path or paths.stage_dir("12_ga_feature_selection")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genetik Algoritma tabanlı özellik seçimi (DEAP)",
    )
    p.add_argument("--target", default="flavonol", choices=list(TARGETS),
                   help="Hedef değişken (default: flavonol)")
    p.add_argument("--model", default="ridge", choices=list(MODEL_CHOICES),
                   help="Fitness regresörü (default: ridge)")
    p.add_argument("--pop", type=int, default=150, help="Popülasyon boyutu (default: 150)")
    p.add_argument("--ngen", type=int, default=150, help="Nesil sayısı (default: 150)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Paralel iş sayısı (default: -1, tüm çekirdekler)")
    p.add_argument("--output-dir", default=None,
                   help="Çıktı dizini (default: outputs/12_ga_feature_selection/<target>_<model>/)")
    p.add_argument("--group-key", default="leaf", choices=["leaf", "plot", "none"],
                   help="CV group key: leaf=yaprak-bazlı, plot=variety+plot+date, none=kapalı")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI giriş noktası: ``python -m src.m04_features.ga_feature_selection ...``"""
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s | %(message)s",
                         datefmt="%H:%M:%S")
    args = _parse_args(argv)
    run(
        cfg=None,
        target=args.target,
        model=args.model,
        pop_size=args.pop,
        n_gen=args.ngen,
        seed=args.seed,
        n_jobs=args.n_jobs,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        group_key=args.group_key,
    )


if __name__ == "__main__":
    # Windows multiprocessing: __main__ guard zorunlu
    main()
