"""Konfigürasyon yükleme yardımcıları.

Pipeline boyunca tüm sihirli sayılar `config/default.yaml` dosyasında durur.
Bu modül YAML'i yükler ve ``cfg.get("models.ridge.alphas")`` gibi noktalı
erişimle değer çekmeyi sağlar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """YAML konfigürasyonunu noktalı anahtarla erişilebilir hale getiren sarmalayıcı."""

    def __init__(self, data: dict[str, Any], source: Path | None = None) -> None:
        self._data = data
        self.source = source

    def get(self, key: str, default: Any = None) -> Any:
        """``cfg.get("models.ridge.alphas")`` formunda nokta-yollu erişim."""
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def set(self, key: str, value: Any) -> None:
        """Nokta-yollu anahtara değer ata (ör. run_all --quick enjeksiyonu).

        Ara sözlükler yoksa oluşturulur; mevcut YAML değerini override eder.
        """
        node = self._data
        parts = key.split(".")
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def as_dict(self) -> dict[str, Any]:
        return self._data


def load(path: str | Path = "config/default.yaml") -> Config:
    """YAML dosyasını yükle ve `Config` döndür."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(data, source=path)
