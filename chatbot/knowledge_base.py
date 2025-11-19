"""Base de conocimiento simplificada para el chatbot médico."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class FractureInfo:
    name: str
    description: str
    symptoms: List[str]
    treatment: str


class KnowledgeBase:
    def __init__(self) -> None:
        self._entries: Dict[str, FractureInfo] = {
            "tibia": FractureInfo(
                name="Tibia",
                description="Hueso principal de la pierna inferior; soporta la mayor carga de peso.",
                symptoms=["Dolor agudo", "Inflamación", "Dificultad para apoyar el pie"],
                treatment="Inmovilización con yeso y, si hay desplazamiento, cirugía con placas o clavos.",
            ),
            "radio": FractureInfo(
                name="Radio",
                description="Hueso lateral del antebrazo, clave para la movilidad de la muñeca.",
                symptoms=["Dolor en la muñeca", "Pérdida de fuerza", "Hinchazón"],
                treatment="Férula o yeso; cirugía en fracturas complejas.",
            ),
            "húmero": FractureInfo(
                name="Húmero",
                description="Hueso largo del brazo que conecta el hombro con el codo.",
                symptoms=["Dolor intenso", "Limitación funcional", "Deformidad"],
                treatment="Inmovilización y fisioterapia; fijación interna si hay desplazamiento.",
            ),
            "fémur": FractureInfo(
                name="Fémur",
                description="Hueso más largo del cuerpo, esencial para la marcha.",
                symptoms=["Dolor incapacitante", "Imposibilidad para caminar", "Acortamiento de la pierna"],
                treatment="Cirugía inmediata con clavos intramedulares o placas.",
            ),
        }

    def list_bones(self) -> List[str]:
        return list(self._entries.keys())

    def find_match(self, text: str) -> Optional[FractureInfo]:
        text_lower = text.lower()
        for key, info in self._entries.items():
            if key in text_lower:
                return info
        return None

    def get(self, bone: str) -> Optional[FractureInfo]:
        return self._entries.get(bone.lower())


__all__ = ["FractureInfo", "KnowledgeBase"]
