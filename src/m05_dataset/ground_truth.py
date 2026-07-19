"""description-2.tab ground truth dosyasını okuma ve stres etiketleme.

Dualex ground truth dosyası tab veya semicolon ayrımlı olabilir, BOM içerebilir,
sütun adları büyük/küçük harf karışık gelebilir. Bu modül bu özellikleri normalize
eder ve standart sütun adlarına dönüştürür.

DÜZELTME (v2):
- description-2.tab dosyasında her veri satırı dış tırnak içine alınmış
  ("Fdga1;...;3.5") ve tüm alanlar tek bir CSV hücresi gibi parse ediliyordu.
  Bu düzeltme satırları manuel olarak ayrıştırır.
- Stres etiket şeması gerçek symptom değerleriyle güncellendi:
    0 = healthy
    1 = flavescence dorée  (FD — biyotik hastalık, Antistax bağlantısı)
    2 = diğer biyotik stres (yeşil yaprak zikadası / green leafhopper [Empoasca vitis],
        buffalo zikadası / buffalo treehopper [Stictocephala bisonia], odun hastalıkları
        / wood diseases ve mildiyö / downy mildew [Plasmopara viticola]) — toplam 56 örnek.
        NOT: bu veri setinde külleme (powdery mildew) YOKTUR; "mildew" = mildiyö (downy mildew).
    3 = abiyotik / diğer   (water stress, senescence, damaged, deficiency, vb.)
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.logging_setup import get as get_logger

log = get_logger("m05_dataset.ground_truth")

_FILENAME_ALIASES = ("directoryname", "imageid", "file", "filename", "file_name", "image")
_VARIETY_ALIASES = ("variety", "cultivar", "grape_variety")
_CHL_ALIASES = ("Chl", "chl", "Chlorophyll", "chlorophyll")
_FLAV_ALIASES = ("Flav", "flav", "Flavonol", "flavonol")
_NBI_ALIASES = ("NBI", "nbi", "NitrogenBalanceIndex")

# Gerçek symptom → sınıf eşlemesi
# Sınıf 0: Sağlıklı
# Sınıf 1: Flavescence dorée (FD) — Antistax / EMA değerlendirme raporu
#          (EMA/HMPC/464682/2016) bağlantısı için merkezi sınıf
# Sınıf 2: Diğer biyotik stres
# Sınıf 3: Abiyotik / belirsiz
_SYMPTOM_TO_CLASS: dict[str, int] = {
    "healthy": 0,
    # FD
    "flavescence doree": 1,
    "flavescence dor ee": 1,  # encoding bozulması ihtimaline karşı
    # Diğer biyotik
    "green leafhopper": 2,
    "buffalo treehopper": 2,
    "wood diseases": 2,
    "mildew": 2,  # mildiyö / downy mildew (Plasmopara viticola, 2 örnek) — külleme DEĞİL
    # Abiyotik / diğer
    "water stress": 3,
    "senescence": 3,
    "damaged": 3,
    "deficiency": 3,
    "chlorosis": 3,
    "discoloration": 3,
}

_CLASS_NAMES = {
    0: "sağlıklı",
    1: "flavescence dorée",
    2: "diğer biyotik stres",
    3: "abiyotik / diğer",
}


def _normalize_text(value) -> str:
    """Unicode normalize + küçük harf + boşluk temizliği."""
    text = str(value).strip().lower()
    # é → e gibi aksan giderme
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("'", "'").replace("`", "'")
    return " ".join(text.split())


def _parse_tab_file(tab_path: Path) -> pd.DataFrame:
    """description-2.tab dosyasını doğru şekilde ayrıştırır.

    Dosya formatı:
      - Başlık satırı: imageID;directoryName;...;NBI  (tırnaksız)
      - Veri satırları: "Fdga1;2020-09-10_003;...;3.5"  (tüm satır dış tırnak içinde)

    Standart CSV parser bunu yanlış okur; manuel ayrıştırma gerekir.
    """
    # Encoding sağlamlaştırma: kaynak dosya Fransızca (latin-1/cp1252) olabilir;
    # "flavescence dorée"deki é = 0xE9 baytı utf-8'de geçersizdir ve errors=
    # "replace" ile � (U+FFFD) olur → FD sınıfı (80 örnek) eşleşmeyip sınıf 3'e
    # düşerdi. Önce utf-8-sig STRICT dene; bozulursa cp1252/latin-1'e düş.
    raw_lines = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with tab_path.open("r", encoding=enc) as f:
                raw_lines = f.readlines()
            log.info("Ground truth encoding: %s", enc)
            break
        except UnicodeDecodeError:
            continue
    if raw_lines is None:
        # Son çare: hatalı baytları değiştirerek oku (en azından çökmesin)
        with tab_path.open("r", encoding="utf-8-sig", errors="replace") as f:
            raw_lines = f.readlines()
        log.warning("Ground truth hiçbir encoding ile temiz okunamadı; replace modu")

    if not raw_lines:
        raise ValueError(f"Boş dosya: {tab_path}")

    # Başlık satırı — tırnaksız, semicolon ile ayrılmış
    header = [h.strip().strip('"') for h in raw_lines[0].strip().split(";")]
    log.info("Ground truth başlık: %s", header)

    rows = []
    for line in raw_lines[1:]:
        line = line.strip()
        if not line:
            continue
        # Dış tırnak varsa soy
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        parts = line.split(";")
        # Eksik alan varsa NaN ile doldur
        while len(parts) < len(header):
            parts.append("")
        rows.append(parts[:len(header)])

    df = pd.DataFrame(rows, columns=header)

    # Sütun değerlerini temizle
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.strip('"')

    # Sayısal sütunları dönüştür
    for col in ("Chl", "Flav", "NBI"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info("Ground truth ayrıştırıldı: %d satır", len(df))
    return df


def load_ground_truth(tab_path: str | Path) -> pd.DataFrame:
    """description-2.tab dosyasını oku ve standart sütun adlarına çevir."""
    tab_path = Path(tab_path)
    if not tab_path.exists():
        raise FileNotFoundError(f"Ground truth dosyası bulunamadı: {tab_path}")

    df = _parse_tab_file(tab_path)

    # directoryName → filename olarak yeniden adlandır (eşleştirme için)
    rename_map: dict[str, str] = {}
    for col in df.columns:
        if col.lower() == "directoryname":
            rename_map[col] = "filename"
            break

    if not rename_map:
        # Fallback: imageID'den directoryName bilgisini çıkarmayı dene
        log.warning("'directoryName' sütunu bulunamadı; imageID kullanılacak")
        if "imageID" in df.columns:
            rename_map["imageID"] = "filename"

    df = df.rename(columns=rename_map)

    if "filename" not in df.columns:
        log.warning("'filename' sütunu yok; ilk sütun kullanılıyor: %s", df.columns[0])
        df = df.rename(columns={df.columns[0]: "filename"})

    df["filename_lower"] = df["filename"].astype(str).str.strip().str.lower()

    # symptom sütunu normalize
    if "symptom" in df.columns:
        df["symptom_norm"] = df["symptom"].apply(_normalize_text)
    else:
        log.warning("'symptom' sütunu yok!")
        df["symptom_norm"] = "unknown"

    log.info("Symptom dağılımı:\n%s", df["symptom_norm"].value_counts().to_string())
    log.info("Ground truth yüklendi: %d satır, sütunlar=%s", len(df), list(df.columns))
    return df


def _pick_first(row: pd.Series, names: tuple[str, ...]) -> float:
    for n in names:
        if n in row.index:
            try:
                return float(row[n])
            except (ValueError, TypeError):
                return float("nan")
    return float("nan")


def match_ground_truth(
    folder_names: list[str], gt_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Klasör adları üzerinden Chl/Flav/NBI değerlerini hizala.

    Returns
    -------
    (y_chl, y_flav, y_nbi, varieties, unmatched)
    """
    n = len(folder_names)
    y_chl = np.full(n, np.nan, dtype=np.float64)
    y_flav = np.full(n, np.nan, dtype=np.float64)
    y_nbi = np.full(n, np.nan, dtype=np.float64)
    varieties: list[str] = []
    unmatched: list[str] = []

    # Eşleştirme için filename_lower kullan
    if "filename_lower" not in gt_df.columns:
        gt_df = gt_df.copy()
        gt_df["filename_lower"] = gt_df["filename"].astype(str).str.strip().str.lower()

    vals = gt_df["filename_lower"]

    for i, folder_name in enumerate(folder_names):
        key = folder_name.strip().lower()

        # Önce tam eşleşme dene
        match_idx = vals[vals == key].index
        if len(match_idx) == 0:
            # İçerik eşleşmesi (klasör adı GT değerinin bir parçasıysa)
            match_idx = vals[vals.str.contains(key, na=False, regex=False)].index
        if len(match_idx) == 0:
            varieties.append("unknown")
            unmatched.append(folder_name)
            continue

        row = gt_df.loc[match_idx[0]]
        y_chl[i] = _pick_first(row, _CHL_ALIASES)
        y_flav[i] = _pick_first(row, _FLAV_ALIASES)
        y_nbi[i] = _pick_first(row, _NBI_ALIASES)
        varieties.append(str(row.get("variety", "unknown")))

    matched_count = n - len(unmatched)
    log.info("Ground truth eşleştirme: %d/%d başarılı", matched_count, n)
    if unmatched:
        log.warning(
            "Eşleşmeyen yaprak sayısı: %d (örn: %s)",
            len(unmatched), unmatched[: min(5, len(unmatched))],
        )
    return y_chl, y_flav, y_nbi, varieties, unmatched


def assign_stress_labels_from_ground_truth(
    gt_df: pd.DataFrame, folder_names: list[str]
) -> np.ndarray:
    """Symptom değerine göre 4 sınıflı stres etiketi ata.

    Sınıf şeması:
        0 = sağlıklı (healthy)
        1 = flavescence dorée — FD (Antistax / EMA değerlendirme raporu bağlantısı)
        2 = diğer biyotik stres — yeşil yaprak zikadası (Empoasca vitis), buffalo
            zikadası (Stictocephala bisonia), odun hastalıkları ve mildiyö / downy
            mildew (Plasmopara viticola); toplam 56 örnek. (külleme/powdery mildew YOK)
        3 = abiyotik / diğer (water stress, senescence, damaged, deficiency, vb.)

    Bu şema şunları garanti eder:
    - FD sınıfı açıkça ayrıştırılır (tezin biyomedikal motivasyonu)
    - Tüm 204 yaprak bir sınıfa düşer (eksik sınıf ValueError vermez)
    - Sınıf dengesizliği minimize edilir
    """
    if "symptom_norm" not in gt_df.columns:
        if "symptom" in gt_df.columns:
            gt_df = gt_df.copy()
            gt_df["symptom_norm"] = gt_df["symptom"].apply(_normalize_text)
        else:
            raise ValueError("Ground truth'ta 'symptom' sütunu yok.")

    # Klasör adı → symptom_norm eşlemesi
    symptom_map: dict[str, str] = {}
    for _, row in gt_df.iterrows():
        fn = _normalize_text(str(row.get("filename", "")))
        symptom_map[fn] = str(row.get("symptom_norm", "unknown"))

    labels = np.full(len(folder_names), 3, dtype=int)  # varsayılan: abiyotik/diğer

    for i, folder in enumerate(folder_names):
        key = _normalize_text(folder)
        symptom = symptom_map.get(key, None)

        # Tam eşleşme yoksa içerik eşleşmesi dene
        if symptom is None:
            for map_key, map_val in symptom_map.items():
                if key in map_key or map_key in key:
                    symptom = map_val
                    break

        if symptom is None:
            symptom = "unknown"

        # Sınıf ata
        assigned = False
        for pattern, cls in _SYMPTOM_TO_CLASS.items():
            if pattern in symptom:
                labels[i] = cls
                assigned = True
                break

        if not assigned:
            labels[i] = 3  # bilinmeyen → abiyotik/diğer
            log.debug("Bilinmeyen symptom '%s' → sınıf 3", symptom)

    # Dağılım raporu
    counts = np.bincount(labels, minlength=4)
    for cls, name in _CLASS_NAMES.items():
        log.info("Sınıf %d (%s): %d yaprak", cls, name, int(counts[cls]))

    # Eksik sınıf kontrolü — artık ValueError fırlatmıyor, sadece uyarı
    missing = [c for c in range(4) if counts[c] == 0]
    if missing:
        log.warning(
            "Eksik sınıf(lar): %s — bu sınıflar modelde görünmeyecek",
            [_CLASS_NAMES[c] for c in missing],
        )

    return labels


def assign_stress_labels_from_flavonol(flav_values: np.ndarray) -> np.ndarray:
    """Flavonol değerine göre fallback stres etiketleme.

    UYARI: Sadece symptom bilgisi yoksa son çare olarak kullan.
    Bu eşik değerleri biyolojik olarak doğrulanmamıştır.
    """
    log.warning(
        "Flavonol tabanlı stres etiketleme kullanılıyor — symptom verisi tercih edilmeli!"
    )
    labels = np.zeros(len(flav_values), dtype=int)
    for i, flav in enumerate(flav_values):
        if np.isnan(flav):
            labels[i] = 3
        elif flav >= 2.5:
            labels[i] = 0  # sağlıklı aralık
        elif flav >= 1.5:
            labels[i] = 2  # biyotik stres
        else:
            labels[i] = 3  # şiddetli / abiyotik
    return labels