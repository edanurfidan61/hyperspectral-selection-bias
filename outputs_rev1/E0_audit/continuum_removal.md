# Continuum removal — Band-I (kodu okuyarak)

Kaynak: `src/m04_features/extraction.py:149` (öznitelik çıkarımında kullanılan); `src/m02_preprocessing/spectral.py:91` içindeki genel
`continuum_removal` (convex_hull/linear) öznitelik hattında **kullanılmıyor**.

| Öğe | Değer |
|---|---|
| Girdi spektrum | yaprak-ortalama **ham** yansıma (`_leaf_mean_spectrum`; SNV ve SG öncesi) |
| Aralık (kod) | `_BAND_I_NM = (410.0, 510.0)` nm, kapalı aralık `wl >= 410 & wl <= 510` |
| Aralıktaki gerçek bantlar | 34 bant, 411.74 – 507.56 nm |
| Çapa noktaları | aralıktaki **ilk** (411.74 nm) ve **son** (507.56 nm) bant — sabit; yerel maksimum / dışbükey zarf aranmaz |
| İnterpolasyon | `np.linspace(sub[0], sub[-1], num=len(sub))` → bant **indeksi** üzerinde doğrusal (nm üzerinde değil; bant aralığı ~2.9–3.0 nm olduğundan fark küçük) |
| Continuum-removed | `cr = sub / baseline` |
| depth | `1 - min(cr)` |
| area | `trapezoid(max(1 - cr, 0), wl[idx])` — nm ekseninde integral |
| Eksik değer kuralı | aralıkta < 3 bant → (0, 0). Aralıkta sonlu olmayan değer **veya** uçlardan biri ≤ 0 ise: sonlu olmayan ya da ≤ 0 değerler `1e-9` ile değiştirilir. `baseline` alt sınırı `1e-9`. Son vektörde sonlu olmayan değerler 0'a çekilir. |
| Örnek-yerel mi | Evet (tek yaprağın spektrumu) |

## Veride gözlenen
- depth: min 0.0736, medyan 0.1488, max 0.3886;
  depth < 0 olan örnek sayısı: 0 / 204
  (doğrusal taban dışbükey zarf olmadığı için `cr > 1` mümkündür → negatif depth).
- area: min 1.6336, medyan 10.0103, max 25.7502;
  area = 0 olan örnek: 0.
