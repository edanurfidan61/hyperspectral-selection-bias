"""Seçim-yanlılığı (selection bias) deneyleri — makale omurgası.

Bu paket, küçük örneklemli yaprak-HSI regresyonunda öznitelik/dalga-boyu
seçiminin dış doğrulama fold'unun DIŞINDA yapılmasının performansı sistematik
olarak şişirdiğini ölçer. Tek iddia: dürüst raporlama için nested-CV zorunludur.

Modüller:
    nested_ga       — apparent (biased) vs nested-CV GA değerlendirmesi
    synthetic_null  — saf gürültüde yanlılığın gösterimi (Adım 4)
    multi_dataset   — birden çok veri setinde tekrar (Adım 5)
    report          — özet tablo + figürler (Adım 6)
"""
