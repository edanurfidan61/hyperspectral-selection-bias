"""Revizyon GA çekirdeği — arşivli, aday önbellekli, varyantlı (plan §3 E6).

Makaledeki GA (``src.m04_features.ga_feature_selection._run_ga``) ile BİREBİR
aynı evrim döngüsü:

* ``random.seed(seed)`` + ``np.random.seed(seed)``; DEAP toolbox
  ``_build_toolbox`` ile (attr_bool ``random.randint``, ``cxUniform(0.5)``,
  ``mutFlipBit(1/p)``, ``selTournament(3)``), ``HallOfFame(5)``, cxpb 0.7,
  mutpb 0.3, elitler (HOF) her nesle eklenir.
* Fitness: seçilen sütunlarda ``cross_val_score(PLS, cv=GroupKFold(5))`` ortalama
  R² (iç CV deterministik, evrim boyunca sabit — R3-31); ``MIN_FEATURES = 5`` altı
  ceza; ``max(R², 0)`` kırpma (R3-28).

Bu modülün ekledikleri (sonucu DEĞİŞTİRMEZ):

* **Aday önbelleği** (maske özeti → fitness). Değerlendirmede rastgelelik yoktur;
  RNG akışı yalnız seçim/çaprazlama/mutasyonda ilerler → önbellek açık/kapalı
  aynı maskeyi verir (``tests`` + ``--quick``'te ``cache_invariance.csv``).
  Orijinal döngü her nesilde HER yavruyu yeniden değerlendirir (yeni
  ``Individual`` nesneleri fitness taşımaz); önbellek bu tekrarları atlar.
* **Arşiv** (R3-78, R3-29): nesil başına en iyi/ortalama/std fitness,
  popülasyon kardinaliteleri, istenen / benzersiz değerlendirme sayıları; son
  popülasyon, hall-of-fame maskeleri ve iç fitness'ları.
* **Varyantlar**: ``representation="fixed_k"`` (G2; birey = k indeks, çaprazlama =
  birleşimden k örnekle, mutasyon = bir içeri/bir dışarı takas) ve
  ``clip=False`` (G3; kırpmasız fitness, MIN_FEATURES altı / hata → −∞).

Paralellik: bir GA koşusu TEK süreçte seri koşar (görev düzeyinde joblib/loky
paralelliği, ``e6_ga``). Orijinal ``mp.Pool.map`` sırayı koruduğu ve fitness
deterministik olduğu için seri değerlendirme aynı sonucu verir.

Duman testi::

    python -m src.m07_revision.ga
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from pathlib import Path

import numpy as np

#: Makaledeki evrim sabitleri (``_run_ga``)
CXPB, MUTPB, HOF_SIZE, TOURNSIZE = 0.7, 0.3, 5, 3


# ---------------------------------------------------------------------------
# Fitness (``_evaluate_individual`` ile aynı hesap; kırpma bayraklı + önbellekli)
# ---------------------------------------------------------------------------
class Fitness:
    """Maske → iç CV R². ``clip=True`` makaledeki fitness ile birebir aynıdır."""

    def __init__(self, X, y, groups, model: str = "pls", seed: int = 42, *, clip: bool = True,
                 use_cache: bool = True):
        from src.core.cv import make_cv_splitter
        from src.m04_features.ga_feature_selection import MIN_FEATURES

        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.groups = None if groups is None else np.asarray(groups)
        self.model = model
        self.clip = bool(clip)
        self.min_features = int(MIN_FEATURES)
        self.penalty = 0.0 if self.clip else float("-inf")
        # Orijinal: her değerlendirmede make_cv_splitter(n_splits=5, groups, random_state=seed)
        self.cv = make_cv_splitter(n_splits=5, task="regression", groups=self.groups,
                                   random_state=int(seed))
        self.use_cache = bool(use_cache)
        self.cache: dict[bytes, float] = {}
        self.n_requested = 0
        self.n_computed = 0

    def _compute(self, mask: np.ndarray) -> float:
        from sklearn.model_selection import cross_val_score

        from src.m04_features.ga_feature_selection import _build_regressor

        if int(mask.sum()) < self.min_features:
            return self.penalty
        Xs = self.X[:, mask]
        try:
            reg = _build_regressor(self.model, Xs.shape[1], Xs.shape[0])
            if self.groups is not None:
                sc = cross_val_score(reg, Xs, self.y, cv=self.cv, scoring="r2", n_jobs=1,
                                     groups=self.groups)
            else:
                sc = cross_val_score(reg, Xs, self.y, cv=self.cv, scoring="r2", n_jobs=1)
            m = float(np.mean(sc))
            return max(m, 0.0) if self.clip else m
        except Exception:
            return self.penalty

    def __call__(self, mask: np.ndarray) -> float:
        mask = np.asarray(mask, dtype=bool)
        self.n_requested += 1
        if not self.use_cache:
            self.n_computed += 1
            return self._compute(mask)
        key = np.packbits(mask).tobytes()
        v = self.cache.get(key)
        if v is None:
            self.n_computed += 1
            v = self._compute(mask)
            self.cache[key] = v
        return v


# ---------------------------------------------------------------------------
# Temsiller
# ---------------------------------------------------------------------------
def _binary_toolbox(p: int, seed: int):
    """Makaledeki toolbox (aynı fonksiyon) — RNG çağrı sırası birebir korunur."""
    from src.m04_features.ga_feature_selection import _build_toolbox
    return _build_toolbox(p, seed)


def _fixed_k_toolbox(p: int, k: int):
    """G2 kardinalite-sabit GA (R3-27, R3-74). Birey = sıralı k indeks listesi."""
    from deap import base, creator, tools

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    def init_ind():
        return creator.Individual(sorted(random.sample(range(p), k)))

    def mate(c1, c2):
        # Birleşimden k örnekle (iki çocuk bağımsız örneklenir)
        union = sorted(set(c1) | set(c2))
        a = sorted(random.sample(union, k))
        b = sorted(random.sample(union, k))
        c1[:] = a
        c2[:] = b
        return c1, c2

    def mutate(ind):
        # Bir içeri / bir dışarı takas
        inside = set(ind)
        out = random.choice(list(ind))
        cand = [j for j in range(p) if j not in inside]
        inn = random.choice(cand)
        ind[:] = sorted((inside - {out}) | {inn})
        return (ind,)

    tb = base.Toolbox()
    tb.register("individual", init_ind)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("mate", mate)
    tb.register("mutate", mutate)
    tb.register("select", tools.selTournament, tournsize=TOURNSIZE)
    return tb


def _to_mask(ind, p: int, representation: str) -> np.ndarray:
    if representation == "binary":
        return np.asarray(ind, dtype=bool)
    m = np.zeros(p, dtype=bool)
    m[np.asarray(list(ind), dtype=int)] = True
    return m


# ---------------------------------------------------------------------------
# Evrim döngüsü (``_run_ga`` ile aynı)
# ---------------------------------------------------------------------------
def run_ga(X, y, groups, *, model: str = "pls", pop: int = 150, ngen: int = 150, seed: int = 42,
           representation: str = "binary", k: int | None = None, clip: bool = True,
           candidate_cache: bool = True) -> tuple[np.ndarray, dict, dict]:
    """Tek GA koşusu. Dönüş: (en iyi maske, info, arşiv dict'i).

    ``info`` JSON-uyumludur (engine fold kaydına girer); arşiv numpy dizileri içerir
    (:func:`save_archive`).
    """
    from deap import creator, tools

    t0 = time.time()
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    if representation not in ("binary", "fixed_k"):
        raise ValueError(f"representation {representation!r}")
    if representation == "fixed_k" and not k:
        raise ValueError("fixed_k için k gerekli")

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    tb = _binary_toolbox(p, seed) if representation == "binary" else _fixed_k_toolbox(p, int(k))
    fit = Fitness(X, y, groups, model, seed, clip=clip, use_cache=candidate_cache)
    hof = tools.HallOfFame(HOF_SIZE)
    population = tb.population(n=pop)

    fit_hist = np.full((ngen + 1, pop), np.nan, dtype=np.float32)
    size_hist = np.zeros((ngen + 1, pop), dtype=np.int16)
    req_hist = np.zeros(ngen + 1, dtype=np.int64)
    comp_hist = np.zeros(ngen + 1, dtype=np.int64)

    def evaluate(inds):
        for ind in inds:
            ind.fitness.values = (fit(_to_mask(ind, p, representation)),)

    def log_gen(g):
        fit_hist[g] = [ind.fitness.values[0] for ind in population]
        size_hist[g] = [int(_to_mask(ind, p, representation).sum()) for ind in population]
        req_hist[g] = fit.n_requested
        comp_hist[g] = fit.n_computed

    evaluate([ind for ind in population if not ind.fitness.valid])
    hof.update(population)
    log_gen(0)
    for g in range(1, ngen + 1):
        offspring = tb.select(population, k=len(population) - len(hof))
        offspring = [creator.Individual(list(ind)) for ind in offspring]
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                tb.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values
        for mutant in offspring:
            if random.random() < MUTPB:
                tb.mutate(mutant)
                del mutant.fitness.values
        elites = [creator.Individual(list(h)) for h in hof]
        for e, h in zip(elites, hof):
            e.fitness.values = h.fitness.values
        offspring.extend(elites)
        evaluate([ind for ind in offspring if not ind.fitness.valid])
        population[:] = offspring
        hof.update(population)
        log_gen(g)

    best = _to_mask(hof[0], p, representation)
    hof_masks = np.stack([_to_mask(h, p, representation) for h in hof])
    hof_fit = np.array([h.fitness.values[0] for h in hof], dtype=float)
    final_pop = np.stack([_to_mask(ind, p, representation) for ind in population])
    secs = round(time.time() - t0, 2)
    info = {"selector": "ga", "representation": representation, "k_fixed": k, "clip": bool(clip),
            "pop": int(pop), "ngen": int(ngen), "seed": int(seed), "model": model,
            "candidate_cache": bool(candidate_cache),
            "inner_fitness": float(hof_fit[0]),
            "n_selected": int(best.sum()),
            "n_evaluations_requested": int(fit.n_requested),
            "n_unique_evaluated": int(len(fit.cache)) if candidate_cache else None,
            "n_evaluations_computed": int(fit.n_computed),
            "inner_folds": 5, "ga_seconds": secs}
    arch = {"best_mask": best, "hof_masks": hof_masks, "hof_fitness": hof_fit,
            "final_population": final_pop, "fitness_by_gen": fit_hist, "size_by_gen": size_hist,
            "requested_cum_by_gen": req_hist, "computed_cum_by_gen": comp_hist}
    return best, info, arch


def generation_stats(arch: dict) -> dict:
    """Arşivden nesil başına özet (logbook) — sonlu fitness'lar üzerinden."""
    f = arch["fitness_by_gen"].astype(float)
    f = np.where(np.isfinite(f), f, np.nan)
    s = arch["size_by_gen"].astype(float)
    req, comp = arch["requested_cum_by_gen"], arch["computed_cum_by_gen"]
    return {"gen": np.arange(f.shape[0]),
            "best_fitness": np.nanmax(f, axis=1), "mean_fitness": np.nanmean(f, axis=1),
            "std_fitness": np.nanstd(f, axis=1), "min_fitness": np.nanmin(f, axis=1),
            "n_infeasible": np.isnan(f).sum(axis=1),
            "card_mean": s.mean(axis=1), "card_std": s.std(axis=1),
            "card_min": s.min(axis=1), "card_q25": np.percentile(s, 25, axis=1),
            "card_median": np.median(s, axis=1), "card_q75": np.percentile(s, 75, axis=1),
            "card_max": s.max(axis=1),
            "n_requested_gen": np.diff(np.r_[0, req]), "n_new_unique_gen": np.diff(np.r_[0, comp]),
            "n_unique_cum": comp}


def save_archive(path: Path, arch: dict, info: dict) -> None:
    """Arşivi atomik yaz (maskeler packbits ile sıkıştırılır)."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = arch["best_mask"].shape[0]
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, n_features=np.int64(p),
            best_mask=np.packbits(arch["best_mask"]),
            hof_masks=np.packbits(arch["hof_masks"], axis=1), hof_fitness=arch["hof_fitness"],
            final_population=np.packbits(arch["final_population"], axis=1),
            fitness_by_gen=arch["fitness_by_gen"], size_by_gen=arch["size_by_gen"],
            requested_cum_by_gen=arch["requested_cum_by_gen"],
            computed_cum_by_gen=arch["computed_cum_by_gen"],
            info_json=np.array(json.dumps(info, default=str)))
    tmp.replace(path)


def load_archive(path: Path) -> dict:
    """:func:`save_archive` çıktısını aç (maskeler bool'a çözülür)."""
    import json
    z = np.load(path, allow_pickle=False)
    p = int(z["n_features"])
    unpack = lambda a: np.unpackbits(a, axis=-1, count=p).astype(bool)  # noqa: E731
    return {"best_mask": unpack(z["best_mask"]), "hof_masks": unpack(z["hof_masks"]),
            "hof_fitness": z["hof_fitness"], "final_population": unpack(z["final_population"]),
            "fitness_by_gen": z["fitness_by_gen"], "size_by_gen": z["size_by_gen"],
            "requested_cum_by_gen": z["requested_cum_by_gen"],
            "computed_cum_by_gen": z["computed_cum_by_gen"],
            "info": json.loads(str(z["info_json"]))}


def mask_sha1(mask: np.ndarray) -> str:
    return hashlib.sha1(np.packbits(np.asarray(mask, dtype=bool)).tobytes()).hexdigest()[:16]


if __name__ == "__main__":
    # Duman testi: küçük bütçe; (1) makale GA'sı ile aynı maske, (2) önbellek açık/kapalı aynı,
    # (3) fixed_k tam k seçer, (4) kırpmasız fitness çalışır.
    import tempfile

    from src.m04_features.ga_feature_selection import run_ga_core

    rng = np.random.default_rng(0)
    n, p = 60, 25
    g = np.repeat(np.arange(15), 4)
    X = rng.normal(size=(n, p))
    y = X[:, :3].sum(axis=1) + rng.normal(size=n)
    ref = run_ga_core(X, y, g, model="pls", pop=16, ngen=4, seed=3, n_jobs=1)
    m1, i1, a1 = run_ga(X, y, g, pop=16, ngen=4, seed=3, candidate_cache=True)
    m2, i2, _ = run_ga(X, y, g, pop=16, ngen=4, seed=3, candidate_cache=False)
    assert np.array_equal(ref, m1) and np.array_equal(m1, m2), "makale GA'sı ile fark"
    print("makale GA maskesi = revizyon GA maskesi (önbellek açık/kapalı):", int(m1.sum()), "öznitelik")
    print("değerlendirme: istenen", i1["n_evaluations_requested"], "| benzersiz", i1["n_unique_evaluated"],
          "| önbelleksiz", i2["n_evaluations_computed"])
    mk, ik, _ = run_ga(X, y, g, pop=16, ngen=4, seed=3, representation="fixed_k", k=6)
    assert mk.sum() == 6
    mu, iu, au = run_ga(X, y, g, pop=16, ngen=4, seed=3, clip=False)
    with tempfile.TemporaryDirectory() as d:
        save_archive(Path(d) / "a.npz", a1, i1)
        b = load_archive(Path(d) / "a.npz")
        assert np.array_equal(b["best_mask"], m1) and b["final_population"].shape == (16, p)
    st = generation_stats(a1)
    print("nesil özeti:", {k_: np.round(v[-1], 3) for k_, v in st.items() if k_ in ("best_fitness", "card_mean")})
    print("ga.py duman testi OK")
