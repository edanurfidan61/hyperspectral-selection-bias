"""Seçim-yanlılığı makalesi — uçtan uca temiz çalıştırıcı.

Tek komutla: (1) gerekirse 01_dataset üret, (2) gerçek veride nested-vs-apparent
GA, (3) sentetik null deneyi, (4) özet tablo + figürler. Sonuçlar pickle olarak
da saklanır (yeniden çizim/analiz için).

Evde tam güçle:
    python run_selection_bias.py --pop 150 --ngen 150 --null-repeats 30

Hızlı duman testi:
    python run_selection_bias.py --quick

Seçimli aşama:
    python run_selection_bias.py --no-null            # null'ı atla
    python run_selection_bias.py --stages nested null report

Çıktılar: outputs/17_selection_bias/
    selection_bias_summary.csv, fig_*.png, results.pkl, null.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

from src.core import config as cfg_mod
from src.core import paths


def _log_setup() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # LGBMRegressor pandas öznitelik adlarıyla fit edilip ndarray ile predict
    # edilince çıkan zararsız sklearn uyarısını sustur (ekranı kirletiyor).
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names",
        category=UserWarning,
    )


def dump_run_config(
    args: argparse.Namespace,
    *,
    seeds: list[int] | None,
    selectors: list[str],
    stages: set[str],
    out_dir: Path,
) -> Path:
    """Tüm efektif çalışma ayarlarını ``run_config.json`` olarak yaz.

    Makale ekine "tam olarak şu ayarlarla üretildi" diye girer: seed(ler), GA
    boyutları, k, dış-fold sayısı, seçici listesi, null tekrarı, aşamalar, tarih
    ve (varsa) git commit SHA. Mevcut akışı bozmaz; sadece dökümdür.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": paths._git_sha(),
        "command": " ".join(sys.argv),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": getattr(args, "seed", None) if seeds is None else None,
        "seeds": seeds,
        "pop": args.pop,
        "ngen": args.ngen,
        "k": args.k,
        "n_outer": args.n_outer,
        "selectors": selectors,
        "n_perm": args.n_perm,
        "null_repeats": args.null_repeats,
        "null_k": args.null_k,
        "n_jobs": args.n_jobs,
        "stages": sorted(stages),
        "include_lambrusco": args.include_lambrusco,
        "quick": args.quick,
        "datasets": _dataset_names(args.include_lambrusco),
    }
    target = out_dir / "run_config.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("run_config yazıldı → %s", target)
    return target


def _dataset_names(include_lambrusco: bool) -> list[str]:
    """REGISTRY'deki dataset adlarını döndür (import patlarsa boş liste)."""
    try:
        from src.m01_io.dataset_registry import REGISTRY
        names = list(REGISTRY.keys())
    except Exception:
        return []
    if not include_lambrusco:
        names = [n for n in names if n != "lambrusco"]
    return names


def ensure_dataset(force: bool = False) -> None:
    """01_dataset yoksa (veya force) üret."""
    ds_dir = paths.OUTPUTS_DIR / "01_dataset"
    if not force and (ds_dir / "X.npy").exists():
        logging.info("01_dataset mevcut → atlanıyor (%s)", ds_dir)
        return
    logging.info("01_dataset üretiliyor...")
    from src.m05_dataset import builder as ds_builder
    cfg = cfg_mod.load("config/default.yaml")
    ds_builder.build(cfg, force=force)


def main(argv: list[str] | None = None) -> None:
    _log_setup()
    p = argparse.ArgumentParser(description="Seçim-yanlılığı uçtan uca çalıştırıcı")
    p.add_argument("--pop", type=int, default=150, help="GA popülasyon (default 150)")
    p.add_argument("--ngen", type=int, default=150, help="GA nesil (default 150)")
    p.add_argument("--n-jobs", type=int, default=-1, help="Paralel iş (default -1)")
    p.add_argument("--n-outer", type=int, default=5, help="Dış fold sayısı")
    p.add_argument("--null-repeats", type=int, default=30, help="Null tekrar sayısı")
    p.add_argument("--n-perm", type=int, default=200,
                   help="Permütasyon testi tekrar sayısı (default 200; ağır "
                        "olursa düşür). 'permutation' aşamasında kullanılır.")
    p.add_argument("--null-k", type=int, default=20,
                   help="Filtre-tabanlı null'da seçilen öznitelik sayısı (default 20)")
    p.add_argument("--selectors", type=str, default="ga,rfe,lasso,mutual_info",
                   help="Nested analizde koşulacak seçici aileleri (virgülle "
                        "ayrılmış: ga,rfe,lasso,mutual_info). Hepsinde bias pozitif "
                        "beklenir — 'sorun GA değil, CV-dışı seçim' tezi.")
    p.add_argument("--k", type=int, default=20,
                   help="Sabit-k seçicilerde (rfe/lasso/mutual_info) seçilen "
                        "öznitelik sayısı (GA yok sayar; default 20)")
    p.add_argument("--stages", nargs="*",
                   default=["dataset", "viz", "nested", "null", "report"],
                   help="Çalıştırılacak aşamalar: dataset viz nested null report "
                        "cvcomp (üç-pipeline grup vs seçim sızıntısı ayrıştırması) "
                        "stats (çoklu-seed üzerinde istatistiksel anlamlılık testleri) "
                        "permutation (gerçek X sabit, y karıştır → permütasyon testi)")
    p.add_argument("--no-null", action="store_true", help="Null aşamasını atla")
    p.add_argument("--force-dataset", action="store_true",
                   help="01_dataset'i yeniden üret")
    p.add_argument("--quick", action="store_true",
                   help="Hızlı duman testi (pop=20, ngen=8, null-repeats=3)")
    p.add_argument("--include-lambrusco", action="store_true",
                   help="Lambrusco'yu (fd/pls) ana nested analize geri ekle "
                        "(varsayılan hariç: ikili sınıflandırma hedefi)")
    p.add_argument("--seeds", type=str, default=None,
                   help="Çoklu-seed kararlılık koşusu: virgülle ayrılmış seed "
                        "listesi (örn. 0,1,2,3,4). Her seed için tüm nested "
                        "analiz koşar; kombinasyon-bazlı checkpoint'le devam edilebilir.")
    args = p.parse_args(argv)

    seeds = None
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
        if not seeds:
            p.error("--seeds boş; en az bir tam sayı verin (örn. --seeds 0,1,2)")

    selectors = [s.strip() for s in args.selectors.split(",") if s.strip()]
    if not selectors:
        p.error("--selectors boş; en az bir seçici verin (örn. ga,rfe)")

    if args.quick:
        args.pop, args.ngen, args.null_repeats = 20, 8, 3
        args.n_perm = 5

    stages = set(args.stages)
    if args.no_null:
        stages.discard("null")
    if seeds is not None and "null" in stages:
        # Çoklu-seed analizi yalnız nested kararlılığına odaklanır; null ayrı koşulur.
        logging.info("Çoklu-seed modu: 'null' aşaması atlanıyor.")
        stages.discard("null")

    out_dir = paths.OUTPUTS_DIR / "17_selection_bias"
    out_dir.mkdir(parents=True, exist_ok=True)
    ga_cfg = {"pop": args.pop, "ngen": args.ngen, "n_jobs": args.n_jobs}

    # Reproducibility dökümü: bu koşunun tam ayarları (makale eki için).
    dump_run_config(args, seeds=seeds, selectors=selectors, stages=stages, out_dir=out_dir)

    t0 = time.time()
    results = []
    null_result = None
    ms_records = None

    # 1) Dataset
    if "dataset" in stages:
        ensure_dataset(force=args.force_dataset)

    # 1b) Veri seti görselleştirme (REGISTRY'deki tüm setler)
    if "viz" in stages:
        from src.m05_dataset import dataset_viz
        logging.info("=== VERİ SETİ GÖRSELLEŞTİRME ===")
        dataset_viz.run()

    # 2) Nested-vs-apparent (gerçek veri, çoklu dataset)
    if "nested" in stages:
        if seeds is not None:
            from src.m06_selection_bias.multiseed import run_multiseed
            logging.info("=== ÇOKLU-SEED NESTED (seeds=%s) ===", seeds)
            ms_records = run_multiseed(
                seeds, n_outer=args.n_outer, ga_cfg=ga_cfg,
                include_lambrusco=args.include_lambrusco,
                selectors=selectors, k=args.k, out_dir=out_dir,
            )
            logging.info("çoklu-seed koşusu tamamlandı: %d kayıt", len(ms_records))
        else:
            from src.m06_selection_bias.multi_dataset import run_multi_dataset
            logging.info("=== NESTED vs APPARENT (gerçek veri) ===")
            results = run_multi_dataset(n_outer=args.n_outer, ga_cfg=ga_cfg,
                                        include_lambrusco=args.include_lambrusco,
                                        selectors=selectors, k=args.k)
            (out_dir / "results.pkl").write_bytes(pickle.dumps(results))
            logging.info("nested sonuçları kaydedildi: %d", len(results))

    # 3) Sentetik null (filtre-tabanlı — hızlı, GA yok)
    if "null" in stages:
        from src.m06_selection_bias.synthetic_null import run_filter_null_experiment
        logging.info("=== SENTETİK NULL (filter, %d tekrar, k=%d) ===",
                     args.null_repeats, args.null_k)
        null_result = run_filter_null_experiment(
            n_repeats=args.null_repeats, k=args.null_k, n_outer=args.n_outer,
        )
        (out_dir / "null.pkl").write_bytes(pickle.dumps(null_result))
        logging.info("\n%s", null_result.summary())

    # 3b) Üç-pipeline CV karşılaştırması (grup vs seçim sızıntısı ayrıştırması)
    if "cvcomp" in stages:
        from src.m06_selection_bias import cv_comparison, report
        logging.info("=== ÜÇ-PIPELINE CV KARŞILAŞTIRMASI (grup vs seçim) ===")
        cmp_rows = cv_comparison.run_cv_comparison_multi(
            n_splits=args.n_outer, seed=(seeds[0] if seeds else 42), ga_cfg=ga_cfg,
            include_lambrusco=args.include_lambrusco,
            selectors=selectors, k=args.k,
        )
        cv_comparison.write_summary_csv(cmp_rows, out_dir)
        report.fig_cv_comparison(cmp_rows, out_dir)
        logging.info("cv_comparison tamamlandı: %d satır", len(cmp_rows))

    # 3c) İstatistiksel anlamlılık testleri (çoklu-seed ham CSV üzerinde)
    if "stats" in stages:
        from src.m06_selection_bias import stats_tests
        logging.info("=== İSTATİSTİKSEL ANLAMLILIK TESTLERİ (çoklu-seed) ===")
        stats_path = stats_tests.run_stats(out_dir=out_dir)
        logging.info("istatistik özeti yazıldı → %s", stats_path)

    # 3d) Permütasyon testi (gerçek X sabit, y karıştır → "daha gerçekçi null")
    if "permutation" in stages:
        from src.m06_selection_bias import report
        from src.m06_selection_bias.synthetic_null import run_permutation_multi
        logging.info("=== PERMÜTASYON TESTİ (n_perm=%d, gerçek X yapısı korunur) ===",
                     args.n_perm)
        perm_results = run_permutation_multi(
            n_perm=args.n_perm, seed=(seeds[0] if seeds else 42),
            n_outer=args.n_outer, ga_cfg=ga_cfg,
            include_lambrusco=args.include_lambrusco,
            selectors=selectors, k=args.k, out_dir=out_dir,
        )
        if perm_results:
            produced = report.fig_permutation(perm_results, out_dir)
            logging.info("permütasyon tamamlandı: %d kombinasyon, figürler: %s",
                         len(perm_results), list(produced))

    # 4) Rapor
    if "report" in stages:
        from src.m06_selection_bias import report
        if seeds is not None:
            # Çoklu-seed raporu: bellekte yoksa checkpoint'lerden topla
            if ms_records is None:
                from src.m06_selection_bias.multiseed import load_checkpoints
                ms_records = load_checkpoints(out_dir=out_dir, ga_cfg=ga_cfg,
                                              k=args.k, n_outer=args.n_outer)
            produced = report.generate_multiseed(ms_records, out_dir=out_dir)
            logging.info("Çoklu-seed rapor çıktıları: %s", list(produced))
        else:
            # Bellekte yoksa diskten yükle (aşamalar ayrı koşulduysa)
            if not results and (out_dir / "results.pkl").exists():
                results = pickle.loads((out_dir / "results.pkl").read_bytes())
            if null_result is None and (out_dir / "null.pkl").exists():
                null_result = pickle.loads((out_dir / "null.pkl").read_bytes())
            produced = report.generate_all(results, null_result, out_dir=out_dir)
            logging.info("Rapor çıktıları: %s", list(produced))

    logging.info("=== BİTTİ (%.1f dk) → %s ===", (time.time() - t0) / 60, out_dir)


if __name__ == "__main__":
    main()
