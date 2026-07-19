"""Çoklu veri seti karşılaştırması (Adım 5).

REGISTRY'deki her veri seti için, belirtilen (target, model) kombinasyonlarında
nested-vs-apparent karşılaştırması koşar ve tek bir özet liste döndürür. Bu liste
report.generate_all'a verilerek tablo + figürlere dönüşür.

Not: lambrusco ve deep_patato yükleyicileri henüz dataset_registry'ye eklenmedi
(Adım 5'in veri-hazırlık kısmı). REGISTRY'ye eklendikçe burası otomatik kapsar.

Çalıştırma:
    python -m src.m06_selection_bias.multi_dataset
"""

from __future__ import annotations

import numpy as np

from src.core.logging_setup import get as get_logger
from src.m01_io import dataset_registry as registry
from src.m04_features.selectors import get_selector
from src.m06_selection_bias.nested_ga import NestedResult, nested_evaluation

log = get_logger("m06_selection_bias.multi_dataset")

#: Varsayılan seçici ekseni — GA dışında 3 farklı aile. "Sorun GA değil, CV-dışı
#: seçimin kendisi" tezini kanıtlamak için hepsinde bias pozitif beklenir.
DEFAULT_SELECTORS: list[str] = ["ga", "rfe", "lasso", "mutual_info"]

#: Sabit-k seçicilerde (rfe/lasso/mutual_info) seçilen öznitelik sayısı.
DEFAULT_K: int = 20


#: Her veri seti için koşulacak (target, model) kombinasyonları. Bir veri setinde
#: hedef yoksa atlanır. Ryckewaert flavonol+pls makalenin ana örneği.
#: deep_patato chlorophyll/pls combo'su FALLBACK üzerinden (ilk hedef) gelir.
# Lambrusco ana regresyon analizinden çıkarıldı: FD ikili bir sınıflandırma
# hedefi, R² regresyon metriği uygun değil. Sınıflandırma alt-analizi için ayrı
# ele alınmalı (F1/balanced accuracy). --include-lambrusco ile geri eklenebilir.
DEFAULT_COMBOS: dict[str, list[tuple[str, str]]] = {
    "ryckewaert": [("flavonol", "pls"), ("flavonol", "lightgbm"),
                   ("chlorophyll", "pls")],
    "deep_patato": [("chlorophyll", "pls")],
}

#: Lambrusco varsayılan olarak ana nested analizden hariç tutulur (yukarıdaki not).
#: --include-lambrusco bayrağı True olunca eski fd/pls combo'su geri eklenir.
LAMBRUSCO_COMBOS: list[tuple[str, str]] = [("fd", "pls")]

#: Bir veri setinde combo tanımlı değilse uygulanacak yedek.
FALLBACK_COMBOS: list[tuple[str, str]] = [("__first_target__", "pls")]


def run_multi_dataset(
    datasets: list[str] | None = None,
    combos: dict[str, list[tuple[str, str]]] | None = None,
    n_outer: int = 5,
    ga_cfg: dict | None = None,
    include_lambrusco: bool = False,
    selectors: list[str] | None = None,
    k: int = DEFAULT_K,
) -> list[NestedResult]:
    """REGISTRY veri setlerinde nested-vs-apparent karşılaştırması.

    Parameters
    ----------
    datasets : list[str] | None
        Koşulacak veri seti adları. None → REGISTRY'deki tümü.
    combos : dict | None
        dataset → [(target, model), ...]. None → DEFAULT_COMBOS.
    n_outer : int
        Dış fold sayısı.
    ga_cfg : dict | None
        Seçici hiperparametreleri (nested_ga.DEFAULT_GA_CFG ile birleşir).
    include_lambrusco : bool
        True ise lambrusco (fd/pls) ana analize geri eklenir. Varsayılan False —
        lambrusco ikili sınıflandırma hedefi olduğu için ana regresyon analizinden
        hariç tutulur (bkz. DEFAULT_COMBOS üstündeki not).
    selectors : list[str] | None
        Koşulacak seçici aileleri. None → DEFAULT_SELECTORS (ga, rfe, lasso, mutual_info).
    k : int
        Sabit-k seçicilerde hedef öznitelik sayısı (GA yok sayar).

    Returns
    -------
    list[NestedResult]
    """
    results: list[NestedResult] = []
    for name, ds, target, model, selector in iter_combos(
        datasets=datasets, combos=combos, include_lambrusco=include_lambrusco,
        selectors=selectors,
    ):
        log.info("=== %s / %s / %s / %s ===", name, target, model, selector)
        res = nested_evaluation(
            ds.X, ds.target(target), ds.groups, model=model,
            selector_fn=get_selector(selector), n_outer=n_outer, ga_cfg=ga_cfg,
            k=k, dataset=name, target=target, selector_name=selector,
        )
        results.append(res)
        log.info(res.summary())

    return results


def iter_combos(
    datasets: list[str] | None = None,
    combos: dict[str, list[tuple[str, str]]] | None = None,
    include_lambrusco: bool = False,
    selectors: list[str] | None = None,
):
    """Koşulacak (name, HSIDataset, target, model, selector) beşlilerini üret.

    run_multi_dataset ve multiseed runner'ı aynı kombinasyon/atlama mantığını
    paylaşsın diye ortak nokta. Yüklenemeyen set, eksik hedef veya yetersiz
    örnek (<20 geçerli) atlanır; lambrusco varsayılan olarak hariç tutulur.
    Her (target, model) kombinasyonu her seçici için ayrı yield edilir.

    Yields
    ------
    (name, ds, target, model, selector)
    """
    names = datasets or list(registry.REGISTRY)
    combo_map = combos or DEFAULT_COMBOS
    sel_list = selectors or DEFAULT_SELECTORS
    if combos is None and include_lambrusco:
        combo_map = {**combo_map, "lambrusco": LAMBRUSCO_COMBOS}

    for name in names:
        # Lambrusco'yu varsayılan olarak ana analizden çıkar (explicit combo
        # verilmediyse ve bayrak kapalıysa). Loader/REGISTRY kaydı korunur.
        if name == "lambrusco" and combos is None and not include_lambrusco:
            log.info("lambrusco ana regresyon analizinden atlanıyor "
                     "(--include-lambrusco ile eklenebilir)")
            continue
        try:
            ds = registry.load(name)
        except Exception as exc:
            log.warning("Veri seti %s yüklenemedi (atlanıyor): %s", name, exc)
            continue

        ds_combos = combo_map.get(name)
        if not ds_combos:
            first = next(iter(ds.targets), None)
            if first is None:
                log.warning("%s: hedef yok, atlanıyor", name)
                continue
            ds_combos = [(first, "pls")]

        for target, model in ds_combos:
            if target not in ds.targets:
                log.warning("%s: hedef %r yok, atlanıyor", name, target)
                continue
            if int(np.isfinite(ds.target(target)).sum()) < 20:
                log.warning("%s/%s: yeterli geçerli örnek yok, atlanıyor", name, target)
                continue
            for selector in sel_list:
                yield name, ds, target, model, selector


def run(cfg=None):
    """Pipeline aşaması 17c_multi_dataset: REGISTRY'deki tüm setlerde karşılaştırma."""
    import pickle

    from src.core import paths
    from src.m06_selection_bias import report
    from src.m06_selection_bias.nested_ga import ga_cfg_from

    out_dir = paths.stage_dir("17c_multi_dataset")
    ga_cfg = ga_cfg_from(cfg)
    n_outer = int(cfg.get("cv.n_splits", 5)) if cfg is not None else 5
    include_lambrusco = bool(cfg.get("selection_bias.include_lambrusco", False)) \
        if cfg is not None else False
    selectors = (cfg.get("selection_bias.selectors", None) if cfg is not None else None) \
        or DEFAULT_SELECTORS
    k = int(cfg.get("selection_bias.k", DEFAULT_K)) if cfg is not None else DEFAULT_K

    results = run_multi_dataset(n_outer=n_outer, ga_cfg=ga_cfg,
                                include_lambrusco=include_lambrusco,
                                selectors=list(selectors), k=k)
    (out_dir / "results.pkl").write_bytes(pickle.dumps(results))
    report.generate_all(results, out_dir=out_dir)
    log.info("17c_multi_dataset tamamlandı (%d sonuç) → %s", len(results), out_dir)
    return out_dir


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s | %(message)s",
                         datefmt="%H:%M:%S")
    # Hızlı duman testi: ryckewaert flavonol+pls, 4 seçici, hafif GA
    res = run_multi_dataset(
        combos={"ryckewaert": [("flavonol", "pls")]},
        ga_cfg={"pop": 20, "ngen": 8, "n_jobs": -1},
        selectors=DEFAULT_SELECTORS, k=20,
    )
    for r in res:
        print(r.summary())
