"""Sentetik null deneyi — seçim-yanlılığının en güçlü kanıtı.

Saf gürültü X ve X'ten TAMAMEN bağımsız y üretiriz. Gerçek bir sinyal YOKTUR,
dolayısıyla dürüst R² beklentisi 0'dır. Buna rağmen, öznitelik seçimi dış
doğrulama döngüsünün DIŞINDA yapıldığında apparent R² belirgin POZİTİF çıkar;
nested-CV ise ≈0 verir. Bu, yanlılığın bir "kötü veri seti" eseri değil,
metodolojik bir artefakt olduğunu gösterir.

İki null çeşidi:

filtre-tabanlı (VARSAYILAN, ``run_filter_null_experiment``)
    En yüksek |korelasyon|lu k bandı seçer. AZ öznitelik seçtiği için model aşırı
    uymaz; sahte korelasyon kalır → apparent net pozitif, nested ≈0. Standart ve
    hızlı (GA yok). Pipeline bunu kullanır.

GA-tabanlı (``run_null_experiment``, ikincil)
    Tam GA ile seçim. Düz fitness manzarasında GA çok fazla öznitelik seçtiğinden
    PLS gürültüye aşırı uyar → her iki R² de derin negatif çıkar (yanlılık yönü
    yine pozitif ama mutlak değerler yorum açısından yanıltıcı). Tarihsel/karşılaştırma
    amaçlı korunur.

Çalıştırma:
    python -m src.m06_selection_bias.synthetic_null
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict

from src.core.cv import make_cv_splitter
from src.core.logging_setup import get as get_logger
from src.m06_selection_bias.nested_ga import nested_ga_evaluation
from src.m04_features.ga_feature_selection import _build_regressor

log = get_logger("m06_selection_bias.synthetic_null")

#: Filtre-tabanlı null'da seçilecek öznitelik sayısı (az → aşırı-uyum yok).
NULL_FILTER_K: int = 20


#: Null deneyinde GA ayarı — tez ayarından hafif (30 tekrar × 6 GA koşusu makul
#: sürede bitsin diye). Yanlılık gösterimi için ağır arama gerekmez; gürültüye
#: aşırı-uyum zaten ortaya çıkar.
NULL_GA_CFG: dict = {"pop": 40, "ngen": 20, "seed": 42, "n_jobs": -1}


def make_null_data(
    n_samples: int = 204,
    n_features: int = 520,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Saf gürültü X + ilişkisiz y + grup etiketleri üret.

    X ~ N(0,1), y ~ N(0,1) bağımsız çekilir → gerçek R² beklentisi 0.
    Gruplar, gerçek Ryckewaert düzenini taklit eder (her örnek kendi grubu);
    None istenirse çağıran ``groups=None`` geçebilir.

    Returns
    -------
    (X, y, groups)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = rng.standard_normal(n_samples)
    groups = np.arange(n_samples)
    return X, y, groups


@dataclass
class NullResult:
    n_repeats: int
    apparent: list[float] = field(default_factory=list)
    nested: list[float] = field(default_factory=list)
    n_samples: int = 0
    n_features: int = 0
    method: str = "filter"   # "filter" | "ga"
    k: int | None = None     # filtre-tabanlı null'da seçilen öznitelik sayısı

    @property
    def apparent_mean(self) -> float:
        return float(np.mean(self.apparent))

    @property
    def nested_mean(self) -> float:
        return float(np.mean(self.nested))

    def summary(self) -> str:
        head = f"NULL [{self.method}" + (f", k={self.k}" if self.k else "") + "]"
        return (
            f"{head} (tekrar={self.n_repeats}, n={self.n_samples}, p={self.n_features})\n"
            f"  apparent R²: ort={self.apparent_mean:+.3f} "
            f"std={np.std(self.apparent):.3f} "
            f"[{min(self.apparent):+.3f}, {max(self.apparent):+.3f}]\n"
            f"  nested   R²: ort={self.nested_mean:+.3f} "
            f"std={np.std(self.nested):.3f} "
            f"[{min(self.nested):+.3f}, {max(self.nested):+.3f}]\n"
            f"  ortalama bias = {self.apparent_mean - self.nested_mean:+.3f}"
        )


# ---------------------------------------------------------------------------
# Filtre-tabanlı null (VARSAYILAN) — en korelasyonlu k bandı seç
# ---------------------------------------------------------------------------
def _topk_corr_idx(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """y ile |Pearson korelasyonu| en yüksek k öznitelik indeksini döndür."""
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    denom = np.sqrt((Xc ** 2).sum(axis=0)) * np.sqrt((yc ** 2).sum())
    denom[denom == 0] = 1.0
    corr = np.abs((Xc * yc[:, None]).sum(axis=0) / denom)
    return np.argsort(corr)[::-1][:k]


def filter_apparent_nested(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    k: int = NULL_FILTER_K,
    model: str = "pls",
    n_outer: int = 5,
    seed: int = 42,
) -> tuple[float, float]:
    """Filtre seçimiyle apparent (CV-dışı seçim) ve nested (CV-içi seçim) R².

    apparent : k bandı TÜM veride seç → aynı veride 5-fold CV R² (sızıntılı).
    nested   : her dış fold'da k bandı SADECE dış-train'de seç → dış-test'te R².
    """
    # apparent — seçim tüm veriyi (dolayısıyla doğrulama fold'larını) görür
    idx = _topk_corr_idx(X, y, k)
    cv = make_cv_splitter(n_splits=5, task="regression", groups=groups, random_state=seed)
    reg = _build_regressor(model, k, X.shape[0])
    if groups is not None:
        y_pred = cross_val_predict(reg, X[:, idx], y, cv=cv, n_jobs=1, groups=groups)
    else:
        y_pred = cross_val_predict(reg, X[:, idx], y, cv=cv, n_jobs=1)
    apparent = float(r2_score(y, y_pred))

    # nested — seçim her fold'da yalnızca dış-train'de
    outer = make_cv_splitter(n_splits=n_outer, task="regression",
                             groups=groups, random_state=seed)
    split_iter = outer.split(X, y, groups) if groups is not None else outer.split(X, y)
    pooled_true = np.full(X.shape[0], np.nan)
    pooled_pred = np.full(X.shape[0], np.nan)
    for tr, te in split_iter:
        idx_tr = _topk_corr_idx(X[tr], y[tr], k)
        reg = _build_regressor(model, k, len(tr))
        reg.fit(X[tr][:, idx_tr], y[tr])
        pooled_true[te] = y[te]
        pooled_pred[te] = np.asarray(reg.predict(X[te][:, idx_tr])).ravel()
    nested = float(r2_score(pooled_true, pooled_pred))
    return apparent, nested


def run_filter_null_experiment(
    n_repeats: int = 30,
    n_samples: int = 204,
    n_features: int = 520,
    k: int = NULL_FILTER_K,
    model: str = "pls",
    n_outer: int = 5,
    base_seed: int = 1000,
) -> NullResult:
    """Filtre-tabanlı sentetik null (VARSAYILAN). Beklenti: apparent>0, nested≈0."""
    res = NullResult(n_repeats=n_repeats, n_samples=n_samples,
                     n_features=n_features, method="filter", k=k)
    for r in range(n_repeats):
        X, y, groups = make_null_data(n_samples, n_features, seed=base_seed + r)
        app, nes = filter_apparent_nested(
            X, y, groups, k=k, model=model, n_outer=n_outer, seed=base_seed + r)
        res.apparent.append(app)
        res.nested.append(nes)
        log.info("Tekrar %2d/%d [filter k=%d]: apparent=%.3f nested=%.3f",
                 r + 1, n_repeats, k, app, nes)
    log.info("\n%s", res.summary())
    return res


def run_null_experiment(
    n_repeats: int = 30,
    n_samples: int = 204,
    n_features: int = 520,
    model: str = "pls",
    n_outer: int = 5,
    ga_cfg: dict | None = None,
    base_seed: int = 1000,
) -> NullResult:
    """GA-tabanlı sentetik null (İKİNCİL — bkz modül docstring).

    Her tekrar farklı tohumla yeni gürültü üretir; nested_ga_evaluation çağrılır.
    Düz fitness manzarası nedeniyle her iki R² de derin negatif çıkabilir; yanlılık
    yönü (apparent>nested) yine korunur. Temiz gösterim için ``run_filter_null_experiment``.
    """
    cfg = ga_cfg or NULL_GA_CFG
    res = NullResult(n_repeats=n_repeats, n_samples=n_samples,
                     n_features=n_features, method="ga")
    for r in range(n_repeats):
        X, y, groups = make_null_data(n_samples, n_features, seed=base_seed + r)
        out = nested_ga_evaluation(
            X, y, groups, model=model, n_outer=n_outer,
            ga_cfg={**cfg, "seed": base_seed + r},
            dataset="synthetic_null", target="noise",
        )
        res.apparent.append(out.apparent_r2)
        res.nested.append(out.nested_r2)
        log.info("Tekrar %2d/%d: apparent=%.3f nested=%.3f",
                 r + 1, n_repeats, out.apparent_r2, out.nested_r2)
    log.info("\n%s", res.summary())
    return res


# ---------------------------------------------------------------------------
# Permütasyon testi — "daha gerçekçi null"
# ---------------------------------------------------------------------------
# Sentetik null X'i de y'yi de sıfırdan rastgele üretir; permütasyon ise GERÇEK
# X yapısını (spektral kovaryans, bant-bant korelasyon) AYNEN korur, sadece y'yi
# karıştırarak X–y ilişkisini bozar. Soru: gerçek kovaryans yapısı altında bile,
# sinyal yokken apparent R² pozitife yığılıyor ve nested ≈0 kalıyor mu?
#: Permütasyon testi varsayılan tekrar sayısı (ağır olursa CLI ile düşürülür).
PERM_DEFAULT_N: int = 200


def run_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    model: str,
    selector_fn,
    k: int | None,
    n_perm: int = PERM_DEFAULT_N,
    seed: int = 42,
    *,
    n_outer: int = 5,
    ga_cfg: dict | None = None,
    dataset: str = "?",
    target: str = "?",
    selector_name: str = "?",
) -> dict:
    """Permütasyon testi: gerçek X sabit, y karıştırılır → X–y ilişkisi bozulur.

    Sentetik null'dan farkı: X'in gerçek spektral kovaryans yapısı korunur.
    ``n_perm`` kez y permüte edilir; her permütasyonda
    :func:`nested_ga.nested_evaluation` ile hem apparent (CV-dışı seçim) hem
    nested (dürüst) R² hesaplanır. Karıştırılmamış (gerçek) y ile bir kez de
    referans (observed) ölçülür.

    Permütasyon p-değeri: gerçek (observed) skorun, permütasyon null dağılımının
    neresinde olduğudur — ``(1 + #{perm >= observed}) / (n_perm + 1)`` (tek-yönlü).
    Asıl ilgi nested R²'dedir (dürüst metrik → "gerçek sinyal var mı?"); apparent
    için de ayrıca hesaplanır (bilgi amaçlı; apparent zaten null'da pozitife yığılır).

    Parameters
    ----------
    X, y : ndarray
        GERÇEK öznitelik matrisi ve hedef. NaN hedef satırları otomatik atılır.
    groups : ndarray | None
        Dış GroupKFold grup etiketleri. y karıştırılırken X ve groups sabit kalır
        (gruplar zaten bölmede kullanılır; basit y-shuffle yeterli).
    selector_fn : callable
        ``selectors`` ailesinden ortak imzalı seçici.
    k : int | None
        Sabit-k seçicilerde hedef öznitelik sayısı (GA yok sayar).
    n_perm : int
        Permütasyon sayısı (varsayılan 200; ağır olursa düşür).
    seed : int
        Permütasyon RNG + CV/bootstrap seed'i.

    Returns
    -------
    dict
        Etiketler, observed_* skorları, perm_* dağılım dizileri (uzunluk n_perm),
        ortalamalar ve perm_p_value (nested-tabanlı) + perm_p_value_apparent.
    """
    from src.m06_selection_bias.nested_ga import nested_evaluation

    # NaN hedefleri at (nested_evaluation da atar; burada permütasyon uzunluğu
    # ve referans tutarlı olsun diye önceden hizalıyoruz).
    fin = np.isfinite(y)
    X, y = X[fin], y[fin]
    g = groups[fin] if groups is not None else None

    # Referans: gerçek (karıştırılmamış) y ile bir kez apparent + nested
    obs = nested_evaluation(
        X, y, g, model, selector_fn, n_outer=n_outer, ga_cfg=ga_cfg, k=k,
        dataset=dataset, target=target, selector_name=selector_name,
    )
    observed_apparent_r2 = float(obs.apparent_r2)
    observed_nested_r2 = float(obs.nested_r2)
    log.info("OBSERVED [%s/%s/%s/%s]: apparent=%.3f nested=%.3f",
             dataset, target, model, selector_name,
             observed_apparent_r2, observed_nested_r2)

    rng = np.random.default_rng(seed)
    perm_apparent_r2 = np.empty(n_perm, dtype=float)
    perm_nested_r2 = np.empty(n_perm, dtype=float)
    perm_apparent_bias = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        yp = rng.permutation(y)   # X sabit; sadece y–X eşlemesi bozulur
        r = nested_evaluation(
            X, yp, g, model, selector_fn, n_outer=n_outer, ga_cfg=ga_cfg, k=k,
            dataset=dataset, target=target, selector_name=selector_name,
        )
        perm_apparent_r2[i] = r.apparent_r2
        perm_nested_r2[i] = r.nested_r2
        perm_apparent_bias[i] = r.apparent_r2 - r.nested_r2
        log.info("Permütasyon %3d/%d [%s/%s/%s/%s]: apparent=%+.3f nested=%+.3f",
                 i + 1, n_perm, dataset, target, model, selector_name,
                 r.apparent_r2, r.nested_r2)

    # Tek-yönlü permütasyon p-değeri (+1 düzeltmesi → asla 0 olmaz)
    perm_p_value = float((1 + np.sum(perm_nested_r2 >= observed_nested_r2)) / (n_perm + 1))
    perm_p_value_apparent = float(
        (1 + np.sum(perm_apparent_r2 >= observed_apparent_r2)) / (n_perm + 1))

    log.info("PERM ÖZET [%s/%s/%s/%s]: mean_perm_apparent=%+.3f mean_perm_nested=%+.3f "
             "p(nested)=%.4f p(apparent)=%.4f",
             dataset, target, model, selector_name,
             float(np.mean(perm_apparent_r2)), float(np.mean(perm_nested_r2)),
             perm_p_value, perm_p_value_apparent)

    return {
        "dataset": dataset, "target": target, "model": model,
        "selector": selector_name, "k": k, "n_perm": int(n_perm), "seed": int(seed),
        "observed_apparent_r2": observed_apparent_r2,
        "observed_nested_r2": observed_nested_r2,
        "perm_apparent_r2": perm_apparent_r2,
        "perm_nested_r2": perm_nested_r2,
        "perm_apparent_bias": perm_apparent_bias,
        "mean_perm_apparent_r2": float(np.mean(perm_apparent_r2)),
        "mean_perm_nested_r2": float(np.mean(perm_nested_r2)),
        "perm_p_value": perm_p_value,
        "perm_p_value_apparent": perm_p_value_apparent,
    }


def _write_permutation_csv(perm_results: list[dict], out_dir) -> "Path":
    """Permütasyon özetini permutation_summary.csv'ye yaz (her combo bir satır).

    Checkpoint amaçlı: her kombinasyon bittikçe çağrılır; çökerse o ana kadarki
    kombinasyonlar diskte kalır.
    """
    import pandas as pd

    rows = [{
        "dataset": d["dataset"], "target": d["target"], "model": d["model"],
        "selector": d["selector"], "k": d["k"], "n_perm": d["n_perm"], "seed": d["seed"],
        "mean_perm_apparent_r2": d["mean_perm_apparent_r2"],
        "mean_perm_nested_r2": d["mean_perm_nested_r2"],
        "observed_apparent_r2": d["observed_apparent_r2"],
        "observed_nested_r2": d["observed_nested_r2"],
        "perm_p_value": d["perm_p_value"],
        "perm_p_value_apparent": d["perm_p_value_apparent"],
    } for d in perm_results]
    df = pd.DataFrame(rows)
    path = out_dir / "permutation_summary.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def run_permutation_multi(
    n_perm: int = PERM_DEFAULT_N,
    seed: int = 42,
    n_outer: int = 5,
    ga_cfg: dict | None = None,
    include_lambrusco: bool = False,
    selectors: list[str] | None = None,
    k: int = NULL_FILTER_K,
    out_dir=None,
) -> list[dict]:
    """Tüm (dataset, target, model, selector) kombinasyonlarında permütasyon testi.

    multi_dataset.iter_combos ile aynı kombinasyon mantığını paylaşır. Her
    kombinasyon bittiğinde ara permutation_summary.csv + permutation.pkl yazılır
    (checkpoint: çökerse kayıp olmasın). Tamamlanan kombinasyonlar tekrar
    koşulmaz (permutation.pkl varsa devam edilir).

    Returns
    -------
    list[dict]
        run_permutation_test çıktıları (figür/CSV için).
    """
    import pickle

    from src.core import paths
    from src.m04_features.selectors import get_selector
    from src.m06_selection_bias.multi_dataset import iter_combos

    out_dir = out_dir or (paths.OUTPUTS_DIR / "17_selection_bias")
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "permutation.pkl"

    # Checkpoint: önceki koşudan tamamlanmış kombinasyonları yükle
    perm_results: list[dict] = []
    done: set[tuple] = set()
    if pkl_path.exists():
        try:
            perm_results = pickle.loads(pkl_path.read_bytes())
            done = {(d["dataset"], d["target"], d["model"], d["selector"])
                    for d in perm_results}
            log.info("Permütasyon checkpoint: %d kombinasyon zaten tamam, atlanacak",
                     len(done))
        except Exception as exc:
            log.warning("permutation.pkl okunamadı (sıfırdan): %s", exc)

    for name, ds, target, model, selector in iter_combos(
        include_lambrusco=include_lambrusco, selectors=selectors,
    ):
        key = (name, target, model, selector)
        if key in done:
            log.info("Atlanıyor (checkpoint): %s/%s/%s/%s", *key)
            continue
        log.info("=== PERMÜTASYON %s/%s/%s/%s (n_perm=%d) ===",
                 name, target, model, selector, n_perm)
        res = run_permutation_test(
            ds.X, ds.target(target), ds.groups, model=model,
            selector_fn=get_selector(selector), k=k, n_perm=n_perm, seed=seed,
            n_outer=n_outer, ga_cfg=ga_cfg,
            dataset=name, target=target, selector_name=selector,
        )
        perm_results.append(res)
        # Ara checkpoint: her kombinasyon bitince yaz
        pkl_path.write_bytes(pickle.dumps(perm_results))
        csv_path = _write_permutation_csv(perm_results, out_dir)
        log.info("Checkpoint yazıldı: %s + %s (%d combo)",
                 pkl_path.name, csv_path.name, len(perm_results))

    return perm_results


def run(cfg=None):
    """Pipeline aşaması 17b_synthetic_null: filtre-tabanlı null + dağılım figürü."""
    import pickle

    from src.core import paths
    from src.m06_selection_bias import report

    out_dir = paths.stage_dir("17b_synthetic_null")
    n_repeats = int(cfg.get("selection_bias.null_repeats", 30)) if cfg is not None else 30
    k = int(cfg.get("selection_bias.null_k", NULL_FILTER_K)) if cfg is not None else NULL_FILTER_K

    res = run_filter_null_experiment(n_repeats=n_repeats, k=k)
    (out_dir / "null.pkl").write_bytes(pickle.dumps(res))
    report.generate_all([], null_result=res, out_dir=out_dir)
    (out_dir / "null_summary.txt").write_text(res.summary(), encoding="utf-8")
    log.info("17b_synthetic_null tamamlandı → %s", out_dir)
    return out_dir


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s | %(message)s",
                         datefmt="%H:%M:%S")
    res = run_filter_null_experiment(n_repeats=30)
    print(res.summary())
