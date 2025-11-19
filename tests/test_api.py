import io

from fastapi.testclient import TestClient
from PIL import Image

from src.api import app, get_chat_agent, get_detector
from src.inference import Detection, FractureDetectionResult


class FakeDetector:
    def predict(self, image):
        return FractureDetectionResult(
            summary="Fractura simulada",
            detections=[Detection(label="radio", confidence=0.9, bbox_xyxy=[0, 0, 10, 10])],
        )


class FakeAgent:
    def __init__(self):
        self.updated = False

    def update_context(self, result):
        self.updated = True

    def answer(self, message: str) -> str:
        return f"Respuesta a: {message}"


client = TestClient(app)


def _override_dependencies():
    fake_detector = FakeDetector()
    fake_agent = FakeAgent()
    app.dependency_overrides[get_detector] = lambda: fake_detector
    app.dependency_overrides[get_chat_agent] = lambda: fake_agent
    return fake_agent


def _clear_overrides():
    app.dependency_overrides.clear()


def test_detect_endpoint_returns_summary():
    agent = _override_dependencies()
    image = Image.new("RGB", (8, 8), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post("/detect", files={"file": ("test.png", buffer, "image/png")})

    assert response.status_code == 200
    data = response.json()
    assert data["summary"].startswith("Fractura")
    assert data["detections"][0]["label"] == "radio"
    _clear_overrides()
    assert agent.updated is True


def test_chat_endpoint_returns_response():
    _override_dependencies()
    response = client.post("/chat", json={"message": "Hola"})
    assert response.status_code == 200
    assert "Respuesta" in response.json()["response"]
    _clear_overrides()
