"""Registro básico de inferencias para trazabilidad."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from utils.helpers import ensure_dir


@dataclass
class InferenceRecord:
    timestamp: str
    source: Optional[str]
    summary: str
    detections: Iterable[Dict[str, Any]]


class InferenceLogger:
    def __init__(self, output_path: Path | str = Path("logs/inference.jsonl")) -> None:
        self.output_path = Path(output_path)
        ensure_dir(self.output_path.parent)

    def log(self, record: InferenceRecord) -> None:
        payload = {
            "timestamp": record.timestamp,
            "source": record.source,
            "summary": record.summary,
            "detections": list(record.detections),
        }
        with self.output_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_result(self, result, source: Optional[str] = None) -> None:
        timestamp = datetime.now(UTC).isoformat()
        detections = [
            {
                "label": det.label,
                "confidence": det.confidence,
                "bbox_xyxy": list(det.bbox_xyxy),
            }
            for det in result.detections
        ]
        self.log(
            InferenceRecord(
                timestamp=timestamp,
                source=source or result.source,
                summary=result.summary,
                detections=detections,
            )
        )


__all__ = ["InferenceLogger", "InferenceRecord"]
