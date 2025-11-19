from chatbot.agent import FractureChatAgent
from src.inference import FractureDetectionResult


def test_agent_includes_disclaimer_and_summary():
    agent = FractureChatAgent()
    agent.update_context(FractureDetectionResult(summary="Resumen de prueba"))

    response = agent.answer("¿Qué sabes de la tibia?")

    assert "Resumen de prueba" in response
    assert "informativa" in response.lower()
    assert "tibia" in response.lower()

def test_agent_fallback_lists_known_bones():
    agent = FractureChatAgent()
    response = agent.answer("No menciono huesos")
    assert "húmero" in response.lower() or "tibia" in response.lower()


def test_agent_reset_clears_history_and_context():
    agent = FractureChatAgent()
    agent.update_context(FractureDetectionResult(summary="Resumen"))
    agent.answer("Pregunta 1")
    assert agent.history

    agent.reset()
    assert agent.history == []
    assert agent.last_result is None
