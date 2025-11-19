"""Funciones utilitarias compartidas en el proyecto."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class MissingDependencyError(RuntimeError):
	"""Se lanza cuando una librería requerida no está instalada."""


def ensure_dir(path: Path | str) -> Path:
	directory = Path(path)
	directory.mkdir(parents=True, exist_ok=True)
	return directory


def load_yaml(path: Path | str) -> Dict[str, Any]:
	try:
		import yaml  # type: ignore
	except ModuleNotFoundError as exc:  # pragma: no cover - validación de dependencia
		raise MissingDependencyError(
			"PyYAML no está instalado. Ejecuta 'pip install pyyaml' o instala los requisitos."
		) from exc
	yaml_path = Path(path)
	if not yaml_path.exists():
		raise FileNotFoundError(f"No se encontró el archivo YAML: {yaml_path}")
	return yaml.safe_load(yaml_path.read_text(encoding="utf-8"))


def save_json(data: Dict[str, Any], path: Path | str, indent: int = 2) -> Path:
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
	return output_path


def timestamp_slug(prefix: str = "run") -> str:
	stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
	return f"{prefix}-{stamp}"


__all__ = [
	"MissingDependencyError",
	"ensure_dir",
	"load_yaml",
	"save_json",
	"timestamp_slug",
]
