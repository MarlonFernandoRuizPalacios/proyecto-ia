"""Interfaz visual del asistente de detección de fracturas."""

from __future__ import annotations

import tempfile
from typing import List, Tuple

from PIL import Image

from chatbot.agent import FractureChatAgent
from src.inference import FractureDetector
from utils.helpers import MissingDependencyError

DISCLAIMER_TEXT = (
	"> ⚠️ Esta herramienta es informativa y no sustituye una valoración presencial. "
	"Consulta a un profesional de la salud ante síntomas graves o progresivos."
)


_GRADIO = None


def _get_gradio():
	global _GRADIO
	if _GRADIO is None:
		try:
			import gradio as gr  # type: ignore
		except ModuleNotFoundError as exc:  # pragma: no cover
			raise MissingDependencyError(
				"Gradio no está instalado. Ejecuta 'pip install gradio' o instala los requisitos."
			) from exc
		_GRADIO = gr
	return _GRADIO


detector = FractureDetector()
chat_agent = FractureChatAgent()


def _format_rows(detections) -> List[List[str]]:
	rows: List[List[str]] = []
	for det in detections:
		bbox = [round(coord, 1) for coord in det.bbox_xyxy]
		rows.append([
			det.label,
			f"{det.confidence * 100:.1f}%",
			*bbox,
		])
	return rows


def _save_temp_image(image: Image.Image) -> str:
	tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
	image.save(tmp.name)
	return tmp.name


def _summary_with_disclaimer(summary: str) -> str:
	return f"{summary}\n\n{DISCLAIMER_TEXT}"


def analyze_image(image: Image.Image):
	gr = _get_gradio()
	if image is None:
		raise gr.Error("Sube una radiografía en formato PNG o JPG.")

	result = detector.predict(image)
	chat_agent.update_context(result)

	annotated = result.annotated_image
	if annotated is None:
		annotated_pil = image
	else:
		annotated_pil = Image.fromarray(annotated)

	download_path = _save_temp_image(annotated_pil)

	table_rows = _format_rows(result.detections)
	return annotated_pil, _summary_with_disclaimer(result.summary), table_rows, download_path


def chat_with_agent(message: str, history: List[Tuple[str, str]]):
	gr = _get_gradio()
	history = history or []
	if not message:
		return history, gr.update(value="")
	response = chat_agent.answer(message)
	history = history + [(message, response)]
	return history, gr.update(value="")


def reset_chat():
	gr = _get_gradio()
	chat_agent.reset()
	return [], gr.update(value="")


def start_gui(server_name: str = "127.0.0.1", server_port: int = 7860):
	gr = _get_gradio()
	with gr.Blocks(title="Detector de Fracturas") as demo:
		gr.Markdown("## 🦴 Detección de Fracturas en Radiografías")
		gr.Markdown(
			"Sube una imagen, revisa la detección automática y conversa con el asistente para resolver dudas."
		)

		with gr.Row():
			with gr.Column():
				image_input = gr.Image(label="Radiografía", type="pil")
				analyze_btn = gr.Button("Analizar imagen", variant="primary")
			with gr.Column():
				annotated_output = gr.Image(label="Resultado anotado", type="pil")
				download_output = gr.File(label="Descargar anotación")

		summary_output = gr.Markdown(label="Resumen de hallazgos")
		disclaimer_md = gr.Markdown(DISCLAIMER_TEXT)
		table_output = gr.Dataframe(
			headers=["Lesión", "Confianza", "x1", "y1", "x2", "y2"],
			datatype=["str", "str", "number", "number", "number", "number"],
			interactive=False,
			row_count=(0, "dynamic"),
			label="Detecciones",
		)

		analyze_btn.click(
			analyze_image,
			inputs=image_input,
			outputs=[annotated_output, summary_output, table_output, download_output],
		)

		gr.Markdown("---")
		gr.Markdown("### Chat con el asistente clínico")
		chatbot_component = gr.Chatbot(label="Conversación", type="tuples")
		chat_input = gr.Textbox(label="Pregunta", placeholder="¿Qué tipo de fractura se observa?", lines=2)
		with gr.Row():
			send_btn = gr.Button("Enviar", variant="primary")
			clear_btn = gr.Button("Limpiar conversación")

		send_btn.click(
			chat_with_agent,
			inputs=[chat_input, chatbot_component],
			outputs=[chatbot_component, chat_input],
		)

		clear_btn.click(
			reset_chat,
			inputs=None,
			outputs=[chatbot_component, chat_input],
		)

	demo.launch(server_name=server_name, server_port=server_port)


__all__ = ["start_gui"]
