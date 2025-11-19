import json
from pathlib import Path

from src.inference import Detection, FractureDetectionResult
from src.telemetry import InferenceLogger


def test_inference_logger_writes_jsonl(tmp_path: Path):
    log_path = tmp_path / "inference.jsonl"
    logger = InferenceLogger(output_path=log_path)

    result = FractureDetectionResult(
        summary="Encontré fractura",
        detections=[Detection(label="radio", confidence=0.9, bbox_xyxy=[0, 0, 10, 10])],
        source="imagen.png",
    )

    logger.log_result(result)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["summary"] == "Encontré fractura"
    assert payload["source"] == "imagen.png"
    assert payload["detections"][0]["label"] == "radio"
