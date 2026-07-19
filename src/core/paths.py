"""Tüm proje yollarını tek noktadan yöneten modül.

Eski kod tabanında her dosya kendi başına ``_THIS_DIR = os.path.dirname(...)``
hesaplıyordu (24 yerde). Burada kök, config, data ve outputs yolları tek
noktada toplanır; ayrıca her aşama klasörü için ``stage_dir()`` ve
``write_source_marker()`` yardımcıları sağlanır.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

# Paket içinden kök hesabı: src/core/paths.py → ../../.. → proje kökü
ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = ROOT / "config"
DATA_DIR: Path = ROOT / "data"
OUTPUTS_DIR: Path = ROOT / "outputs"

# Pipeline aşama isimleri (sıra önemli — main.py bu listeyi kullanır)
PIPELINE_STAGES: tuple[str, ...] = (
    "01_dataset",
    "02_eda",
    "03_visualization",
    "04_feature_shap",
    "05_feature_rfe",
    "06_regression",
    # "07_classification" listeden çıkarıldı: SMOTE açıkken (default) çıktılar
    # 07_classification_resampled'a yazılıyor → baseline klasörü boş kalıyordu.
    # stage_dir() lazy oluşturduğundan gerçekten yazılırsa kendiliğinden doğar.
    "08_deep_learning",
    "09_ordinal_flavonol",
    "10_anomaly_flavonol",
    "11_ensemble",
    "12_ga_feature_selection",
    "13_flavonol_combos",
)


def stage_dir(stage: str, *sub: str) -> Path:
    """``outputs/<stage>/<sub...>`` klasörünü döndür ve oluştur."""
    p = OUTPUTS_DIR.joinpath(stage, *sub)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_outputs_tree() -> None:
    """Tüm pipeline aşama klasörlerini önceden oluştur."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for stage in PIPELINE_STAGES:
        (OUTPUTS_DIR / stage).mkdir(parents=True, exist_ok=True)


def _git_sha() -> str | None:
    """Mevcut git commit SHA'sını al; repo yoksa None."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def write_source_marker(stage_path: Path, *, producer: str, config_source: Path | None) -> None:
    """Bir aşama klasörüne ``source.txt`` bırak.

    Hangi modülün ürettiği, hangi config'in kullanıldığı, git SHA ve tarihi
    yazar — kullanıcı klasöre baktığında çıktının nereden geldiğini anlar.
    """
    stage_path.mkdir(parents=True, exist_ok=True)
    lines = [
        f"producer    : {producer}",
        f"config      : {config_source if config_source else '(belirtilmemiş)'}",
        f"git_sha     : {_git_sha() or '(repo yok)'}",
        f"timestamp   : {datetime.now().isoformat(timespec='seconds')}",
        f"platform    : {platform.platform()}",
    ]
    (stage_path / "source.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


class RunManifest:
    """Tüm pipeline çalışmasının özetini ``run_manifest.json`` olarak yazar."""

    def __init__(self, config_data: dict[str, Any], config_source: Path | None) -> None:
        self.config = config_data
        self.config_source = str(config_source) if config_source else None
        self.git_sha = _git_sha()
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.stages: list[dict[str, Any]] = []

    def record_stage(self, name: str, duration: float, status: str = "ok") -> None:
        self.stages.append(
            {
                "name": name,
                "duration_seconds": round(duration, 2),
                "status": status,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

    def write(self) -> Path:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        target = OUTPUTS_DIR / "run_manifest.json"
        payload = {
            "config_source": self.config_source,
            "git_sha": self.git_sha,
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "stages": self.stages,
            "config": self.config,
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return target


def start_manifest(cfg) -> RunManifest:
    """Bir Config nesnesinden RunManifest oluştur."""
    return RunManifest(cfg.as_dict(), cfg.source)
