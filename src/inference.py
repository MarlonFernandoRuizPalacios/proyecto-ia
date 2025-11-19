"""Inferencia para detección de fracturas."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from src.telemetry import InferenceLogger
from utils.helpers import MissingDependencyError, ensure_dir


def _import_yolo():
	try:
		from ultralytics import YOLO  # type: ignore
	except ModuleNotFoundError as exc:  # pragma: no cover - error de dependencia
		raise MissingDependencyError(
			"Ultralytics no está instalado. Ejecuta 'pip install ultralytics' o usa requirements.txt"
		) from exc
	return YOLO


@dataclass
class Detection:
	label: str
	confidence: float
	bbox_xyxy: Sequence[float]


@dataclass
class FractureDetectionResult:
	detections: List[Detection] = field(default_factory=list)
	summary: str = ""
	annotated_image: Optional[np.ndarray] = None
	source: Optional[str] = None

	def to_dict(self) -> dict:
		return {
			"summary": self.summary,
			"source": self.source,
			"detections": [
				{"label": det.label, "confidence": det.confidence, "bbox_xyxy": list(det.bbox_xyxy)}
				for det in self.detections
			],
		}


class FractureDetector:
	def __init__(
		self,
		weights_path: Path | str = Path("models/fracture_detector.pt"),
		fallback_weights: str = "yolov8n.pt",
		imgsz: int = 640,
		conf: float = 0.25,
		model=None,
		logger: Optional[InferenceLogger] = None,
	) -> None:
		self.weights_path = Path(weights_path)
		self.fallback_weights = fallback_weights
		self.imgsz = imgsz
		self.conf = conf
		self.model = model or self._load_model()
		self.logger = logger or InferenceLogger()

	def _load_model(self):
		YOLO = _import_yolo()
		weights = self.weights_path if self.weights_path.exists() else self.fallback_weights
		if weights == self.fallback_weights:
			ensure_dir(self.weights_path.parent)
		return YOLO(str(weights))

	def predict(self, image_source, conf: Optional[float] = None) -> FractureDetectionResult:
		confidence = conf or self.conf
		predictions = self.model(image_source, imgsz=self.imgsz, conf=confidence, verbose=False)
		result = predictions[0]
		detections = self._parse_detections(result)
		annotated = result.plot()  # numpy array BGR
		annotated_rgb = annotated[:, :, ::-1]
		summary = self._build_summary(detections)
		result_obj = FractureDetectionResult(
			detections=detections,
			summary=summary,
			annotated_image=annotated_rgb,
			source=getattr(result, "path", None),
		)
		if self.logger:
			self.logger.log_result(result_obj, source=result_obj.source)
		return result_obj

	def _parse_detections(self, result) -> List[Detection]:
		detections: List[Detection] = []
		names = result.names or {}
		if not hasattr(result, "boxes") or result.boxes is None:
			return detections

		for box in result.boxes:
			cls_value = box.cls.item() if hasattr(box.cls, "item") else float(box.cls)
			cls_id = int(cls_value)
			label = names.get(cls_id, str(cls_id))
			conf_value = box.conf.item() if hasattr(box.conf, "item") else float(box.conf)
			confidence = float(conf_value)
			bbox = box.xyxy.cpu().numpy().tolist()[0]
			detections.append(Detection(label=label, confidence=confidence, bbox_xyxy=bbox))
		return detections

	@staticmethod
	def _build_summary(detections: Sequence[Detection]) -> str:
		if not detections:
			return "No se detectaron fracturas evidentes en la imagen analizada."
		lines = ["Resumen de hallazgos:"]
		for det in detections:
			confidence_pct = round(det.confidence * 100, 1)
			lines.append(f"- {det.label} (confianza {confidence_pct}%)")
		return "\n".join(lines)


__all__ = [
	"Detection",
	"FractureDetectionResult",
	"FractureDetector",
]
