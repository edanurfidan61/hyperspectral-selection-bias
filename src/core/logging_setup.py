"""Pipeline boyunca kullanılan loglama yapılandırması.

Eski kod ``print()`` çağrılarıyla doluydu; burada tüm modüller
``logging.getLogger("hyperspectral.<modul>")`` ile seviyelendirilmiş loglar
yazar.

Konsol vs dosya politikası (GÖREV 5 sadeleştirmesi):
- **Konsol**: sadece INFO+ — kullanıcı ekranını DEBUG ile boğmaz.
- **Dosya**  (``outputs/pipeline.log``): yapılandırılmış seviyenin tamamı
  (DEBUG dahil) yazılır → ayrıntılı izleme dosyada kalır.

Logger isimleri sadeleşir: ``hyperspectral.m05_dataset.builder`` →
``m05.builder`` (konsolda da dosyada da). Ayrıca konsol handler ``tqdm.write``
üzerinden çalışır; aktif bir ilerleme çubuğunu bozmaz.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .paths import OUTPUTS_DIR

ROOT_LOGGER_NAME = "hyperspectral"
_INITIALIZED = False

# "hyperspectral." köke önekini ve "mNN_<alt>." modül-paket önekini sadeleştir.
_NAME_PREFIX_RE = re.compile(rf"^{ROOT_LOGGER_NAME}\.")
# Örn: "m05_dataset." → "m05."   |   "m07_ensemble." → "m07."
_MODULE_PREFIX_RE = re.compile(r"\bm(\d+[a-z]?)_[a-z_]+\.")


def _shorten(name: str) -> str:
    """Logger adını okunabilir kısa biçime indir."""
    name = _NAME_PREFIX_RE.sub("", name)
    name = _MODULE_PREFIX_RE.sub(lambda m: f"m{m.group(1)}.", name)
    return name


class _ShortNameFormatter(logging.Formatter):
    """Format ederken ``record.name``'i geçici olarak kısaltır.

    Aynı record birden çok handler'a gittiği için orijinal değeri geri yükleriz.
    """

    def format(self, record: logging.LogRecord) -> str:
        original = record.name
        record.name = _shorten(original)
        try:
            return super().format(record)
        finally:
            record.name = original


class _TqdmStreamHandler(logging.StreamHandler):
    """``tqdm.write`` ile uyumlu konsol handler — ilerleme çubuğunu bozmaz."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            try:
                # tqdm yüklü değilse veya aktif bar yoksa düz akışa düşeriz.
                from tqdm import tqdm  # type: ignore

                tqdm.write(msg, file=self.stream)
            except Exception:
                self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def init(cfg) -> logging.Logger:
    """Konfigürasyondan logging seviyelerini ve hedeflerini ayarla."""
    global _INITIALIZED
    if _INITIALIZED:
        return logging.getLogger(ROOT_LOGGER_NAME)

    level_name = (cfg.get("logging.level", "INFO") or "INFO").upper()
    file_level = getattr(logging, level_name, logging.INFO)
    to_console = bool(cfg.get("logging.console", True))
    file_path = cfg.get("logging.file")

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    # Root'u en geniş (en alçak) seviyede tut; handler'lar kendi eşiğini uygular.
    logger.setLevel(min(file_level, logging.INFO))
    logger.handlers.clear()

    fmt = _ShortNameFormatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if to_console:
        ch = _TqdmStreamHandler()
        ch.setFormatter(fmt)
        ch.setLevel(logging.INFO)        # konsola SADECE INFO+
        logger.addHandler(ch)

    if file_path:
        target = Path(file_path)
        if not target.is_absolute():
            target = OUTPUTS_DIR.parent / file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(target, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(file_level)          # dosyaya yapılandırılan seviye (DEBUG dahil)
        logger.addHandler(fh)

    logger.propagate = False
    _INITIALIZED = True
    return logger


def get(name: str) -> logging.Logger:
    """``hyperspectral.<name>`` altında alt-logger döndür."""
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
