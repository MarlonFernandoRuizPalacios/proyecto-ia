import numpy as np

from src.inference import FractureDetector


class _FakeTensor:
    def __init__(self, data):
        self._data = np.array([data], dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._data

    def tolist(self):  # pragma: no cover - compatibilidad
        return self._data.tolist()


class _FakeBox:
    def __init__(self, cls_id, conf, bbox):
        self.cls = np.array([cls_id])
        self.conf = np.array([conf])
        self.xyxy = _FakeTensor(bbox)


class _FakeResult:
    def __init__(self):
        self.names = {0: "tibia fracture"}
        self.boxes = [_FakeBox(0, 0.9, [0, 0, 10, 10])]
        self.path = "test"

    def plot(self):
        return np.zeros((640, 640, 3), dtype=np.uint8)


class _FakeModel:
    def __call__(self, *args, **kwargs):
        return [_FakeResult()]


class _NullLogger:
    def log_result(self, *args, **kwargs):
        return None


def test_fracture_detector_builds_summary():
    detector = FractureDetector(model=_FakeModel(), logger=_NullLogger())
    result = detector.predict(np.zeros((640, 640, 3), dtype=np.uint8))
    assert result.detections
    assert "Resumen" in result.summary
