# Detección de Fracturas Óseas en Radiografías

Asistente clínico que combina detección automática de fracturas (YOLOv8) con un chatbot contextual en español. Permite entrenar modelos con el dataset de `data/raw`, ejecutar inferencia sobre radiografías y resolver dudas mediante una interfaz gráfica basada en Gradio.

## 📦 Requisitos
- Python 3.10+
- Dependencias indicadas en `requirements.txt`
- Dataset con estructura YOLO (ver `data/raw/data.yaml`)

Instalación rápida:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Ejecución de la interfaz

```bash
python main.py
```

Esto lanzará la interfaz en `http://127.0.0.1:7860` con:
1. **Carga de radiografías** y visualización de anotaciones.
2. **Resumen textual** de hallazgos con recordatorio clínico.
3. **Descarga de la imagen anotada** para compartir o documentar.
4. **Chatbot médico** que responde preguntas sobre la fractura detectada u otros huesos soportados.
5. **Botón para limpiar la conversación**, útil para comenzar nuevas consultas.

## 🌐 API REST (FastAPI)

Inicia el servicio HTTP con:

```bash
uvicorn src.api:app --reload
```

Endpoints principales:
- `GET /health` → verificación básica.
- `POST /detect` → recibe un archivo de imagen (`multipart/form-data`) y devuelve el resumen y las detecciones (se actualiza el contexto del chatbot).
- `POST /chat` → recibe `{ "message": "..." }` y responde usando el mismo contexto que la UI.

El detector y el chatbot son reutilizados en memoria, por lo que el servicio es liviano y mantiene historial entre peticiones.

## 📝 Trazabilidad de inferencias
- Cada ejecución exitosamente analizada se registra en `logs/inference.jsonl`.
- El registro incluye marca de tiempo UTC, imagen origen (si está disponible), resumen y todas las detecciones.
- Puedes apuntar a otro archivo pasando `logger=InferenceLogger(output_path="otros_logs.jsonl")` al crear `FractureDetector`.

## 🧠 Entrenamiento del detector

```python
from src.training import TrainingConfig, train_detector

config = TrainingConfig(
    data_yaml="data/raw/data.yaml",
    model_variant="yolov8s.pt",
    epochs=75,
    batch=16,
)
train_detector(config)
```

También puedes entrenar desde la línea de comandos:

> Ejecuta estos scripts desde la raíz del repositorio (tras activar tu entorno virtual).

```bash
python scripts/train_detector.py --data data/raw/data.yaml --model yolov8s.pt --epochs 75 --batch 16
```

El mejor modelo se guardará en `models/fracture_detector.pt`, que es el archivo cargado por el módulo de inferencia.

### Notebook estilo Colab (VS Code)

Si prefieres un flujo interactivo con gráficas embebidas, abre `notebooks/entrenamiento_colab_vs_code.ipynb` y ejecuta las celdas en orden:

1. **Dependencias** → instala `tensorboard` y `pandas` junto con el resto del proyecto.
2. **Configuración** → detecta automáticamente si hay GPU disponible (similar a Colab) y genera un nombre de experimento.
3. **Entrenamiento** → lanza `TrainingConfig` apuntando a `runs/notebooks` y exporta los pesos a `models/<experimento>.pt`.
4. **Gráficas** → genera plots de pérdidas y métricas usando Matplotlib.
5. **TensorBoard** → abre el panel dentro de VS Code con `%tensorboard --logdir runs/notebooks` para explorar la evolución completa.

Este notebook replica la experiencia típica de Colab pero utilizando tus archivos locales (dataset e histórico) sin subirlos a la nube.

## 🔍 Inferencia programática

```python
from src.inference import FractureDetector

model = FractureDetector()
result = model.predict("data/raw/test/images/ejemplo.png")
print(result.summary)
print(result.to_dict())
```

Para analizar varias imágenes de una carpeta y guardar los resultados anotados:

```bash
python scripts/batch_infer.py data/raw/test/images --weights models/fracture_detector.pt --output runs/inference
```

## 📊 Evaluación rápida

```python
from src.evaluation import evaluate_model

metrics = evaluate_model()
print(metrics)
```

## 🤖 Chatbot
- Implementado en `chatbot/agent.py` con una base de conocimiento ligera (`chatbot/knowledge_base.py`).
- Recibe automáticamente el último resultado de detección para responder preguntas como *"¿Qué fractura detectaste?"* o *"¿Cuáles son los síntomas de una fractura de radio?"*.

## 🧪 Pruebas

```bash
python -m pytest tests
```

## 📚 Documentación adicional
- `docs/architecture.md`: descripción detallada del flujo de datos, componentes y próximos pasos.

---
Este proyecto es una guía educativa y **no reemplaza** la valoración de un profesional de la salud.
