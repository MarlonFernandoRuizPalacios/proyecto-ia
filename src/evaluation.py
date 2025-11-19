"""Evaluación de modelos entrenados."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from utils.helpers import MissingDependencyError


def _import_yolo():
	try:
		from ultralytics import YOLO  # type: ignore
	except ModuleNotFoundError as exc:  # pragma: no cover
		raise MissingDependencyError(
			"Ultralytics no está instalado. Ejecuta 'pip install ultralytics'"
		) from exc
	return YOLO


def evaluate_model(
	weights_path: Path | str = Path("models/fracture_detector.pt"),
	data_yaml: Path | str = Path("data/raw/data.yaml"),
	split: str = "val",
	imgsz: int = 640,
	conf: float = 0.25,
) -> Dict[str, float]:
	YOLO = _import_yolo()
	model = YOLO(str(weights_path))
	results = model.val(data=str(data_yaml), imgsz=imgsz, conf=conf, split=split)
	return results.results_dict


__all__ = ["evaluate_model"]
