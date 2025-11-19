"""Agente de chat para responder preguntas sobre fracturas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from chatbot.knowledge_base import FractureInfo, KnowledgeBase
from chatbot.prompts import (
	DISCLAIMER_PROMPT,
	FOLLOW_UP_PROMPT,
	FORMAT_TEMPLATE,
	SYSTEM_PROMPT_ES,
)
from src.inference import FractureDetectionResult


@dataclass
class ConversationTurn:
	user: str
	assistant: str


class FractureChatAgent:
	def __init__(self, knowledge_base: Optional[KnowledgeBase] = None) -> None:
		self.knowledge_base = knowledge_base or KnowledgeBase()
		self.history: List[ConversationTurn] = []
		self.last_result: Optional[FractureDetectionResult] = None

	def update_context(self, detection_result: FractureDetectionResult) -> None:
		self.last_result = detection_result

	def reset(self) -> None:
		"""Limpia el historial y el contexto del último resultado."""
		self.history.clear()
		self.last_result = None

	def answer(self, question: str) -> str:
		info = self.knowledge_base.find_match(question)
		summary = self._build_summary()
		anatomy_details = self._format_info(info) if info else self._fallback_info()
		care_steps = self._care_plan(info)
		disclaimer = self._disclaimer()

		response = FORMAT_TEMPLATE.format(
			summary=summary,
			anatomy_details=anatomy_details,
			care_steps=care_steps,
			disclaimer=disclaimer,
		)
		response = f"{SYSTEM_PROMPT_ES}\n\n{response}\n\n{FOLLOW_UP_PROMPT}"
		self.history.append(ConversationTurn(user=question, assistant=response))
		return response

	def _build_summary(self) -> str:
		if self.last_result:
			return self.last_result.summary
		return "Aún no he analizado una imagen reciente. Puedes subir una radiografía para obtener hallazgos específicos."

	def _format_info(self, info: FractureInfo) -> str:
		symptoms = ", ".join(info.symptoms)
		return (
			f"Hueso: {info.name}\n"
			f"Descripción: {info.description}\n"
			f"Síntomas frecuentes: {symptoms}.\n"
			f"Sugerencia inicial: {info.treatment}"
		)

	def _fallback_info(self) -> str:
		bones = ", ".join(self.knowledge_base.list_bones())
		return (
			"No identifiqué un hueso específico en tu pregunta. "
			f"Puedo orientarte sobre estas zonas: {bones}. Indícame cuál te preocupa."
		)

	def _care_plan(self, info: Optional[FractureInfo]) -> str:
		base = [
			"Reposo relativo y evitar movimientos bruscos.",
			"Aplicar hielo envuelto en tela 15 min cada hora durante las primeras 48 h.",
			"Inmovilizar la zona con férula o cabestrillo de ser posible.",
		]
		if info:
			base.append(f"Consultar con traumatología para confirmar lesiones en {info.name.lower()}.")
		else:
			base.append("Acudir a urgencias si hay dolor intenso, entumecimiento o falta de movilidad.")
		return "\n".join(f"- {step}" for step in base)

	def _disclaimer(self) -> str:
		return (
			f"{DISCLAIMER_PROMPT} "
			"Esta guía es informativa y no sustituye una valoración presencial. "
			"Busca atención médica inmediata ante síntomas graves o progresivos."
		)


__all__ = ["FractureChatAgent"]
