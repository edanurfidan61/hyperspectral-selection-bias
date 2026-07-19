"""Çoklu-seed kararlılık analizi (Adım 6b).

Amaç: apparent ve nested R²'nin seed'den seed'e nasıl değiştiğini ölçmek.
Beklenti: apparent **kararsız** (geniş dağılım), nested **kararlı** (dar dağılım)
→ "seçim yanlılığı sadece şişirmez, aynı zamanda kararsızdır" figürü.

Tasarım notları
---------------
* Seed listesi BAŞTAN sabittir; hiçbir seed elenmez, hepsinin sonucu saklanır.
* **Checkpoint:** her (seed, dataset, hedef, model, seçici) bitince diske yazılır
  (``_checkpoints/<cfg_tag>/seed<N>_<dataset>_<target>_<model>_<selector>.pkl``).
  ``cfg_tag`` GA boyutları + k + dış-fold sayısını kodlar; farklı ayarlı koşular
  (duman pop=20 vs final pop=150) ayrı klasöre yazar, böylece çakışıp kirlenmez.
  Çökerse tamamlanan kombinasyonlar atlanıp eksikten devam edilir — tam koşu
  ~20-25 saat sürdüğü için kritik.
* Seed, ``ga_cfg["seed"]`` üzerinden hem GA'ya hem dış/iç CV splitter'a hem de
  bootstrap'a akar; yani seed değişince apparent ve nested birlikte değişir.

Çalıştırma orchestrator üzerinden:
    python run_selection_bias.py --seeds 0,1,2,3,4 --pop 150 --ngen 150
"""

from __future__ import annotations

import pickle
from pathlib import Path

from src.core import paths
from src.core.logging_setup import get as get_logger
from src.m04_features.selectors import get_selector
from src.m06_selection_bias.multi_dataset import DEFAULT_K, iter_combos
from src.m06_selection_bias.nested_ga import NestedResult, nested_evaluation

log = get_logger("m06_selection_bias.multiseed")


def _combo_tag(dataset: str, target: str, model: str, selector: str) -> str:
    return f"{dataset}_{target}_{model}_{selector}"


def cfg_tag(ga_cfg: dict | None, k: int, n_outer: int) -> str:
    """Checkpoint namespace imzası — GA boyutları + k + dış-fold sayısı.

    Checkpoint'ler bu imzaya göre alt klasöre yazılır. Böylece farklı ayarlı
    koşular (örn. duman testi pop=20 vs final pop=150) ASLA aynı dosyayı
    paylaşmaz — aksi halde final, duman testinin ucuz GA sonucunu yükleyip
    sessizce kirlenirdi. Aynı imzalı yarıda kesilmiş final ise sorunsuz devam eder.
    """
    g = ga_cfg or {}
    pop = g.get("pop", "NA")
    ngen = g.get("ngen", "NA")
    return f"pop{pop}_ngen{ngen}_k{k}_no{n_outer}"


def run_multiseed(
    seeds: list[int],
    datasets: list[str] | None = None,
    combos: dict[str, list[tuple[str, str]]] | None = None,
    n_outer: int = 5,
    ga_cfg: dict | None = None,
    include_lambrusco: bool = False,
    selectors: list[str] | None = None,
    k: int = DEFAULT_K,
    out_dir: Path | None = None,
) -> list[dict]:
    """Her seed için tüm nested analizini koş, kombinasyon-bazlı checkpoint'le.

    Parameters
    ----------
    seeds : list[int]
        Koşulacak seed'ler (baştan sabit; hiçbiri elenmez).
    datasets, combos, n_outer, ga_cfg, include_lambrusco
        ``iter_combos`` / ``nested_ga_evaluation`` ile aynı anlam.
    out_dir : Path | None
        Çıktı kökü. None → outputs/17_selection_bias. Checkpoint'ler alt
        klasör ``_checkpoints/`` altına yazılır.

    Returns
    -------
    list[dict]
        Her eleman: {"seed", "dataset", "target", "model", "result": NestedResult}.
        Tüm (seed × kombinasyon) çarpımı (yüklenemeyen/atlanan kombolar hariç).
    """
    out_dir = out_dir or (paths.OUTPUTS_DIR / "17_selection_bias")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Checkpoint'ler config imzasına göre namespace'lenir (duman vs final çakışmaz).
    ckpt_dir = out_dir / "_checkpoints" / cfg_tag(ga_cfg, k, n_outer)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Kombinasyonları bir kez topla (veri setlerini bir kez yükler; registry
    # cache'lediği için seed'ler arasında yeniden yüklenmez).
    combo_list = list(iter_combos(
        datasets=datasets, combos=combos, include_lambrusco=include_lambrusco,
        selectors=selectors,
    ))
    if not combo_list:
        log.warning("Koşulacak kombinasyon yok.")
        return []

    total = len(seeds) * len(combo_list)
    log.info("Çoklu-seed: %d seed × %d kombinasyon = %d koşu (checkpoint: %s)",
             len(seeds), len(combo_list), total, ckpt_dir)

    records: list[dict] = []
    done = 0
    for seed in seeds:
        for name, ds, target, model, selector in combo_list:
            done += 1
            ckpt = ckpt_dir / f"seed{seed}_{_combo_tag(name, target, model, selector)}.pkl"
            if ckpt.exists():
                res = pickle.loads(ckpt.read_bytes())
                log.info("[%d/%d] ATLA (checkpoint var): seed=%d %s/%s/%s/%s",
                         done, total, seed, name, target, model, selector)
            else:
                log.info("[%d/%d] KOŞ: seed=%d %s/%s/%s/%s",
                         done, total, seed, name, target, model, selector)
                seed_cfg = {**(ga_cfg or {}), "seed": seed}
                res = nested_evaluation(
                    ds.X, ds.target(target), ds.groups, model=model,
                    selector_fn=get_selector(selector), n_outer=n_outer,
                    ga_cfg=seed_cfg, k=k, dataset=name, target=target,
                    selector_name=selector,
                )
                ckpt.write_bytes(pickle.dumps(res))
                log.info("checkpoint yazıldı: %s", ckpt.name)
            records.append({
                "seed": seed, "dataset": name, "target": target,
                "model": model, "selector": selector, "result": res,
            })

    return records


def load_checkpoints(
    out_dir: Path | None = None,
    ga_cfg: dict | None = None,
    k: int = DEFAULT_K,
    n_outer: int = 5,
) -> list[dict]:
    """Diskteki seed checkpoint'lerini yükle (report ayrı koşulduğunda).

    ``ga_cfg/k/n_outer`` ile aynı config namespace'i hedeflenir — koşuyla
    eşleşen checkpoint'ler yüklensin (farklı pop/ngen ayrı klasörde). Dosya adı
    ``seed<N>_...`` formatından seed ayrıştırılır; dataset/target/model/selector
    gövdedeki NestedResult'tan okunur (dataset adında alt çizgi olabilir).
    """
    out_dir = out_dir or (paths.OUTPUTS_DIR / "17_selection_bias")
    ckpt_dir = out_dir / "_checkpoints" / cfg_tag(ga_cfg, k, n_outer)
    records: list[dict] = []
    if not ckpt_dir.exists():
        return records
    for ckpt in sorted(ckpt_dir.glob("seed*.pkl")):
        res: NestedResult = pickle.loads(ckpt.read_bytes())
        seed = int(ckpt.name[len("seed"):].split("_", 1)[0])
        records.append({
            "seed": seed, "dataset": res.dataset, "target": res.target,
            "model": res.model, "selector": getattr(res, "selector", "ga"),
            "result": res,
        })
    return records
