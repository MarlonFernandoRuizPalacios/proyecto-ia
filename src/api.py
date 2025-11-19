"""API REST para detección de fracturas y chatbot clínico."""

from __future__ import annotations

import io
from typing import Dict, List

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from pydantic import BaseModel  # type: ignore
from PIL import Image

from chatbot.agent import FractureChatAgent
from src.inference import FractureDetector

app = FastAPI(title="Fracture Detector API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector: FractureDetector | None = None
_chat_agent = FractureChatAgent()


def get_detector() -> FractureDetector:
    global _detector
    if _detector is None:
        _detector = FractureDetector()
    return _detector


def get_chat_agent() -> FractureChatAgent:
    return _chat_agent


class DetectResponse(BaseModel):
    summary: str
    detections: List[Dict]


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/health", tags=["monitoring"])
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse, tags=["inference"])
async def detect_fracture(
    file: UploadFile = File(...),
    detector: FractureDetector = Depends(get_detector),
    agent: FractureChatAgent = Depends(get_chat_agent),
) -> DetectResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:  # pragma: no cover - validación
        raise HTTPException(status_code=400, detail="No pude procesar la imagen.") from exc

    result = detector.predict(image)
    agent.update_context(result)
    detections = [
        {"label": det.label, "confidence": det.confidence, "bbox_xyxy": list(det.bbox_xyxy)}
        for det in result.detections
    ]
    return DetectResponse(summary=result.summary, detections=detections)


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat_endpoint(
    payload: ChatRequest,
    agent: FractureChatAgent = Depends(get_chat_agent),
) -> ChatResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")
    response = agent.answer(payload.message)
    return ChatResponse(response=response)


__all__ = ["app", "get_detector", "get_chat_agent"]
