# Arquitectura del Sistema de Detección de Fracturas

## 1. Visión General
El proyecto implementa un asistente clínico que combina tres capacidades principales:

1. **Entrenamiento y actualización de modelos**: aprovechando un dataset en formato YOLO almacenado en `data/raw`.
2. **Servicio de inferencia**: expone un detector de fracturas basado en modelos YOLO preentrenados o fine-tuned.
3. **Interfaz asistida**: una UI con carga de imágenes y un chatbot contextual que responde preguntas sobre hallazgos y tratamientos.

```
Usuario ─┐
        │  (imágenes / preguntas)
        ▼
    Interfaz Gradio (carga de imágenes + chat)
        │
        ├─► Pipeline de Inferencia (preprocesado + YOLO)
        │       └─► Resultados estructurados + imagen anotada
        │
        └─► Agente Conversacional (conocimiento médico + contexto de detección)
```

## 2. Flujo de Datos
1. **Imágenes crudas** → módulo `src.preprocessing` aplica normalización, CLAHE y resize.
2. **Modelo YOLO** → archivo `src.training` orquesta fine-tuning usando `ultralytics.YOLO` y guarda pesos en `models/fracture_detector.pt`.
3. **Inferencia** → `src.inference.FractureDetector` carga los pesos, ejecuta predicciones en CPU/GPU y produce:
   - Lista de detecciones con etiquetas, confianza y cajas.
   - Imagen anotada en memoria (`numpy`/`PIL`).
   - Resumen textual con niveles de riesgo.
4. **Chatbot** → `chatbot.agent.FractureChatAgent` consume el resumen y responde preguntas con base en `chatbot.prompts` + knowledge base.

## 3. Componentes Clave
### 3.1 Preprocesamiento (`src/preprocessing.py`)
- Carga segura de imágenes (`PIL`/`cv2`).
- Filtros opcionales: denoise, CLAHE, normalización.
- Conversión a tensores compatibles con YOLO (640×640 por defecto).

### 3.2 Entrenamiento (`src/training.py`)
- Función `train_detector()` parametrizable (epochs, batch, img size).
- Reutiliza `ultralytics.YOLO` para transfer learning desde modelos base (`yolov8n.pt`, `yolov8s.pt`, etc.).
- Guarda directorio de experimentos en `models/cnn/` y expone utilidades para exportar a ONNX/TFLite.

### 3.3 Inferencia (`src/inference.py`)
- Clase `FractureDetector` con métodos:
  - `predict(image_source)` → devuelve `FractureDetectionResult` (detecciones + resumen).
  - `render(image_source, detections)` → genera imagen anotada.
  - `predict_from_path()` helper usado por backend/tests.
- Maneja múltiples tipos de entrada (ruta, bytes, `PIL.Image`, `numpy array`).
- Incluye lógica de fallback: si no hay pesos entrenados, usa un modelo general (`yolov8n.pt`).

### 3.4 Evaluación (`src/evaluation.py`)
- Wrapper sobre `model.val()` para obtener métricas (mAP, precision, recall).
- Produce reportes JSON/CSV reutilizables por el UI o dashboards.

### 3.5 Chatbot (`chatbot/agent.py`)
- Mecanismo rule-based + retrieval sobre `knowledge_base`.
- Memoria volátil con el último resultado de inferencia para contestar: “¿Qué se encontró en esta radiografía?”
- `chatbot/prompts.py` contiene plantillas bilingües (ES/EN) y mensajes del sistema.

### 3.6 Interfaz (`ui/gui.py`)
- Implementada con **Gradio Blocks** para mantener simplicidad multiplataforma.
- Secciones:
  1. **Detector**: carga imagen → muestra anotaciones, tabla de detecciones y resumen textual.
  2. **Chatbot**: historial conversacional sincronizado con el detector.
- Expone `start_gui()` para integrarse con `main.py` y permitir despliegue local o vía `uvicorn`.

## 4. Entrenamiento Personalizado
- Dataset definido en `data/raw/data.yaml` (estándar YOLO, 7 clases) → no requiere conversiones adicionales.
- Comando típico:
  ```python
  from src.training import train_detector
  train_detector(data_yaml="data/raw/data.yaml", model_variant="yolov8s.pt", epochs=75)
  ```
- Pesos resultantes -> `models/fracture_detector.pt` (ruta por defecto usada en inferencia/UI).

## 5. Integración con Chatbot
1. El detector guarda el último resultado (`FractureDetectionResult.to_context()`).
2. `FractureChatAgent.update_context()` recibe ese objeto.
3. Cada pregunta pasa por un analizador semántico básico que:
   - Busca huesos específicos (tibia, radio, húmero, etc.).
   - Identifica intención: hallazgos vs síntomas vs tratamiento.
   - Construye la respuesta combinando contexto + base de conocimiento.

## 6. Despliegue y Automatización
- **Local**: `python main.py` levanta la interfaz Gradio.
- **API**: `interface/web_app.py` mantiene compatibilidad Flask para integraciones ligeras.
- **CI/Test**: `pytest` en `tests/` valida componentes clave (preprocesamiento e inferencia mockeada).

## 7. Próximos Pasos
- Integrar seguimiento de experimentos (Weights & Biases) en `training.py`.
- Añadir modo batch inference para carpetas completas.
- Extender chatbot con un modelo generativo ligero (ej. `transformers` + `distilbert`).
