"""Bootstrap güven aralıkları — regresyon metrikleri için (BCa).

Seçim-yanlılığı (selection bias) makalesinde, apparent vs nested R² farklarının
istatistiksel anlamlılığını raporlamak için R²'nin %95 bootstrap güven aralığını
kullanırız. Bu modül o işin tek sorumlusu; herhangi bir pipeline aşamasına bağlı
değildir (saf yardımcı).

YÖNTEM — BCa (bias-corrected and accelerated) bootstrap:
    Eski sürüm düz "percentile" bootstrap kullanıyordu: bootstrap dağılımının
    ham [alpha/2, 1-alpha/2] yüzdelikleri. Percentile yöntemi, tahmin edicide
    yanlılık (bias) veya çarpıklık (skew) varsa hatalı kapsama verir. BCa iki
    düzeltme ekler:
      * z0 (bias-correction): bootstrap dağılımının, nokta tahminin kaçta
        kaçının altında kaldığına bakarak kaymayı düzeltir.
      * a (acceleration): jackknife ile hesaplanan, standart hatanın gerçek
        değere göre değişim hızını (skew) düzeltir.
    Referans: DiCiccio & Efron (1996), "Bootstrap Confidence Intervals",
    Statistical Science 11(3):189-228. Efron (1987), JASA 82:171-185.

    Uygulama ``scipy.stats.bootstrap(method="BCa")`` üzerinden yapılır. R²
    eşleştirilmiş (paired) bir istatistik olduğundan (aynı örnek için hem
    y_true hem y_pred birlikte yeniden örneklenmeli), scipy'ye ham vektörleri
    değil GÖZLEM İNDEKSLERİNİ "veri" olarak veririz; istatistik fonksiyonu bu
    indeksler üzerinden orijinal çiftlere erişip R² hesaplar. Böylece
    (y_true, y_pred) eşleşmesi her yeniden örneklemde korunur.

SÖZLEŞME (değişmedi — aşağı akış report.py/nested_ga.py buna bağlı):
    Dönüş dict'i her zaman şu anahtarları içerir:
      {"r2", "lo", "hi", "se", "n_boot"}  (+ bilgi amaçlı "method").
    Nokta tahmini "r2" doğrudan r2_score'dan gelir; BCa yalnızca "lo"/"hi"yi
    (ve "se"yi) etkiler.

DAYANIKLILIK:
    BCa dejenere durumlarda başarısız olabilir (tüm bootstrap R²'leri aynı →
    z0 sonsuz; jackknife varyansı sıfır → a tanımsız). Bu gibi durumlarda
    sessizce düz percentile bootstrap'a düşeriz (fallback) ve "method"
    alanında bunu belirtiriz. Dejenere yeniden-örneklem filtresi (tüm y_true
    aynı → R² tanımsız) korunmuştur.

Çalıştırma (smoke test):
    python -m src.core.metrics_ci
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import bootstrap as _scipy_bootstrap


def _r2_from_index(idx: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Verilen gözlem indeksleri için eşleştirilmiş R².

    scipy.stats.bootstrap, ``idx`` dizisini yerine-koymalı yeniden örnekler ve
    bu fonksiyonu her yeniden örneklem için çağırır. ``idx`` orijinal (y_true,
    y_pred) çiftlerine işaret ettiği için eşleşme korunur. Dejenere örneklemde
    (tüm y_true aynı → R² tanımsız/±inf) NaN döneriz; scipy bunları hesaba
    katmaz (percentile/BCa NaN-güvenlidir değil, bu yüzden fallback'te
    ayrıca filtreliyoruz).
    """
    idx = np.asarray(idx, dtype=int)
    yt = y_true[idx]
    yp = y_pred[idx]
    if np.ptp(yt) == 0.0:
        return np.nan
    return float(r2_score(yt, yp))


def _percentile_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    """Düz percentile bootstrap — BCa başarısız olursa yedek yol.

    Dejenere yeniden-örneklemleri (tüm y_true aynı) atlar; geçerli tekrarların
    yüzdeliklerini döndürür. Dönüş: (lo, hi, se, n_valid).
    """
    n = y_true.shape[0]
    boot = np.empty(n_boot, dtype=float)
    n_valid = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if np.ptp(yt) == 0.0:
            continue
        boot[n_valid] = r2_score(yt, yp)
        n_valid += 1
    boot = boot[:n_valid]
    if n_valid == 0:
        return float("nan"), float("nan"), float("nan"), 0
    lo = float(np.percentile(boot, 100 * (alpha / 2)))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    se = float(np.std(boot, ddof=1)) if n_valid > 1 else float("nan")
    return lo, hi, se, n_valid


def bootstrap_r2_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """R² için BCa bootstrap (%95 varsayılan) güven aralığı.

    Örneklem çiftleri ``(y_true, y_pred)`` üzerinde ``n_boot`` kez yerine-koymalı
    yeniden örnekleme yapılır; her tekrarda R² hesaplanır ve BCa (bias-corrected
    and accelerated) yöntemiyle ``(1-alpha)`` güven aralığı üretilir. BCa
    hesaplanamazsa düz percentile'a düşülür.

    Parameters
    ----------
    y_true, y_pred : ndarray, shape (n_samples,)
        Gerçek ve tahmin edilen değerler (eşleştirilmiş — aynı indeks aynı örnek).
    n_boot : int
        Bootstrap tekrar sayısı (scipy: n_resamples).
    seed : int
        Yeniden üretilebilirlik için RNG tohumu. ``np.random.default_rng(seed)``
        hem scipy'ye hem de percentile fallback'e verilir (deterministik).
    alpha : float
        Anlamlılık düzeyi (0.05 → %95 GA). scipy confidence_level = 1 - alpha.

    Returns
    -------
    dict
        ``{"r2": nokta tahmini, "lo": alt sınır, "hi": üst sınır,
        "se": bootstrap std hatası, "n_boot": geçerli tekrar sayısı,
        "method": "BCa" | "percentile-fallback" | "degenerate"}``.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true {y_true.shape} ve y_pred {y_pred.shape} aynı boyutta olmalı."
        )
    n = y_true.shape[0]
    if n < 2:
        raise ValueError("En az 2 örnek gerekli.")

    # Nokta tahmini — BCa/percentile'dan bağımsız, doğrudan tüm veriden.
    point = float(r2_score(y_true, y_pred))

    # Nokta tahmini bile tanımsızsa (tüm y_true aynı) hiç bootstrap'a girme.
    if np.ptp(y_true) == 0.0:
        return {"r2": point, "lo": float("nan"), "hi": float("nan"),
                "se": float("nan"), "n_boot": 0, "method": "degenerate"}

    # scipy.stats.bootstrap deterministik olsun diye RNG'yi seed'den türet.
    # Aynı Generator'ı fallback percentile için de kullanacağız; ama scipy
    # onu tüketebileceğinden fallback'e ayrı, aynı tohumlu bir Generator veririz.
    idx_all = np.arange(n)

    # BCa denemesi ----------------------------------------------------------
    try:
        with warnings.catch_warnings():
            # BCa dejenere uçlarda DegenerateDataWarning verebilir; yakalayıp
            # fallback'e yönlendirmek için uyarıyı hataya çeviriyoruz.
            warnings.simplefilter("error")
            res = _scipy_bootstrap(
                (idx_all,),  # "veri" = indeksler (paired resampling için)
                statistic=lambda ix: _r2_from_index(ix, y_true, y_pred),
                n_resamples=n_boot,
                confidence_level=1.0 - alpha,
                method="BCa",
                vectorized=False,
                random_state=np.random.default_rng(seed),
            )
        lo = float(res.confidence_interval.low)
        hi = float(res.confidence_interval.high)
        se = float(res.standard_error)
        # BCa sonucu NaN/inf verdiyse (dejenere) fallback'e düş.
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError("BCa sonlu olmayan sınır üretti.")
        return {"r2": point, "lo": lo, "hi": hi, "se": se,
                "n_boot": int(n_boot), "method": "BCa"}
    except Exception as exc:  # noqa: BLE001 — her BCa hatasında güvenli yola düş
        warnings.warn(
            f"BCa başarısız ({type(exc).__name__}: {exc}); "
            "düz percentile bootstrap'a düşülüyor.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Percentile fallback ---------------------------------------------------
    lo, hi, se, n_valid = _percentile_ci(
        y_true, y_pred, n_boot, alpha, np.random.default_rng(seed)
    )
    if n_valid == 0:
        return {"r2": point, "lo": float("nan"), "hi": float("nan"),
                "se": float("nan"), "n_boot": 0, "method": "degenerate"}
    return {"r2": point, "lo": lo, "hi": hi, "se": se,
            "n_boot": int(n_valid), "method": "percentile-fallback"}


if __name__ == "__main__":
    # Sahte veriyle smoke test: güçlü ilişki → GA pozitif ve dar olmalı,
    # saf gürültü → GA sıfırı kapsamalı, dejenere → NaN + "degenerate".
    rng = np.random.default_rng(0)
    n = 200

    # 1) Güçlü doğrusal ilişki — BCa çalışmalı, alt sınır yüksek olmalı.
    x = rng.normal(size=n)
    y_true = 3.0 * x + rng.normal(scale=0.5, size=n)
    y_pred = 3.0 * x  # iyi tahmin
    ci = bootstrap_r2_ci(y_true, y_pred)
    print(f"[güçlü ilişki]  R²={ci['r2']:.3f}  "
          f"%95 GA=[{ci['lo']:.3f}, {ci['hi']:.3f}]  se={ci['se']:.3f}  "
          f"method={ci['method']}")
    assert ci["method"] == "BCa", "Sağlıklı veride BCa kullanılmalı"
    assert ci["lo"] > 0.5, "Güçlü ilişkide alt sınır yüksek olmalı"
    assert ci["lo"] <= ci["r2"] <= ci["hi"], "Nokta tahmin GA içinde olmalı"

    # 2) İlişkisiz (null) — tahmin gerçekle alakasız, GA ~0 civarı.
    y_pred_null = rng.normal(size=n)
    ci0 = bootstrap_r2_ci(y_true, y_pred_null)
    print(f"[null]          R²={ci0['r2']:.3f}  "
          f"%95 GA=[{ci0['lo']:.3f}, {ci0['hi']:.3f}]  se={ci0['se']:.3f}  "
          f"method={ci0['method']}")
    assert ci0["lo"] <= 0.0 <= ci0["hi"] or ci0["hi"] < 0.1, \
        "Null durumda GA ~0 civarı olmalı"

    # 3) Determinizm — aynı seed, aynı sonuç.
    ci_a = bootstrap_r2_ci(y_true, y_pred, seed=7)
    ci_b = bootstrap_r2_ci(y_true, y_pred, seed=7)
    assert ci_a == ci_b, "Aynı seed deterministik sonuç vermeli"
    print(f"[determinizm]   seed=7 tekrar -> ayni GA ({ci_a['method']}) OK")

    # 4) Dejenere — tüm y_true aynı → R² tanımsız, "degenerate".
    y_const = np.full(n, 5.0)
    ci_deg = bootstrap_r2_ci(y_const, y_pred)
    print(f"[dejenere]      method={ci_deg['method']}  "
          f"lo={ci_deg['lo']}  hi={ci_deg['hi']}  n_boot={ci_deg['n_boot']}")
    assert ci_deg["method"] == "degenerate", "Sabit y_true dejenere olmalı"
    assert ci_deg["n_boot"] == 0

    print("OK: bootstrap_r2_ci (BCa) smoke test geçti.")
