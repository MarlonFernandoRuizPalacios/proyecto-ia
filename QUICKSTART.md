# 🚀 Guía de Inicio Rápido

## ⚡ Para usuarios (solo inferencia)

**Tiempo estimado: 2 minutos**

```bash
# 1. Clonar
git clone https://github.com/MarlonFernandoRuizPalacios/proyecto-ia.git
cd proyecto-ia

# 2. Instalar
pip install -r requirements.txt

# 3. Descargar modelo (Git LFS)
git lfs pull

# 4. Verificar instalación (opcional)
python verify_installation.py

# 5. ¡Usar!
python main.py
```

Abre tu navegador en `http://127.0.0.1:7860`

---

## 🔧 Para desarrolladores (con entrenamiento)

**Requisitos:**
- Python 3.8+
- GPU NVIDIA (recomendado)
- Dataset en formato YOLO

```bash
# 1. Entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Instalar
pip install -r requirements.txt

# 3. Descargar modelo
git lfs pull

# 4. Preparar dataset
# Coloca tu dataset en data/raw/:
#   data/raw/
#     ├── data.yaml
#     ├── train/
#     │   ├── images/
#     │   └── labels/
#     └── valid/
#         ├── images/
#         └── labels/

# 5. Entrenar
python scripts/train_detector.py \
  --data data/raw/data.yaml \
  --model yolov8m.pt \
  --epochs 95 \
  --batch 4 \
  --imgsz 800

# 6. El modelo entrenado se guarda en:
# models/fracture_detector.pt
```

---

## 📝 Formato del Dataset (YOLO)

### data.yaml
```yaml
path: ../data/raw
train: train/images
val: valid/images

nc: 2  # número de clases
names: ['fractured', 'not fractured']
```

### Estructura de directorios
```
data/raw/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── labels/
│       ├── img1.txt
│       ├── img2.txt
│       └── ...
└── valid/
    ├── images/
    └── labels/
```

### Formato de etiquetas (.txt)
Cada archivo `.txt` debe tener el mismo nombre que su imagen correspondiente:

```
# img1.txt
0 0.5 0.5 0.3 0.4  # clase x_center y_center width height (normalizados 0-1)
```

---

## 🔍 API REST

```bash
# Iniciar servidor
uvicorn src.api:app --reload
```

### Endpoints

**POST /detect** - Detectar fracturas
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@radiografia.jpg"
```

**POST /chat** - Consultar al chatbot
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Qué tipo de fractura es?"}'
```

**GET /health** - Verificar estado
```bash
curl http://localhost:8000/health
```

---

## 🐛 Solución de Problemas

### El modelo no se descarga
```bash
# Instalar Git LFS
# Windows: https://git-lfs.github.com/
# Linux: sudo apt install git-lfs
# Mac: brew install git-lfs

# Inicializar y descargar
git lfs install
git lfs pull
```

### Error de CUDA
```bash
# Verificar PyTorch con CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Si no está disponible, instalar:
# https://pytorch.org/get-started/locally/
```

### Falta alguna dependencia
```bash
pip install -r requirements.txt --upgrade
```

### El modelo no carga
```bash
# Verificar que el archivo existe y tiene tamaño correcto
ls -lh models/fracture_detector.pt
# Debe ser ~148 MB

# Si es muy pequeño (<1MB), es un pointer de Git LFS
git lfs pull
```

---

## 📊 Rendimiento del Modelo Incluido

- **Precisión**: 75.4%
- **Recall**: 78.6%
- **mAP50**: 74.0%
- **mAP50-95**: 62.2%
- **Dataset**: 10,119 imágenes (fracturas de huesos)
- **Modelo**: YOLOv8m (25.9M parámetros)

---

## 📚 Más Información

- [README.md](README.md) - Documentación completa
- [docs/architecture.md](docs/architecture.md) - Arquitectura del sistema
- [docs/quick_training_guide.md](docs/quick_training_guide.md) - Guía de entrenamiento
- [GitHub](https://github.com/MarlonFernandoRuizPalacios/proyecto-ia)

---

## ⚠️ Importante

Esta herramienta es **educativa y no reemplaza** la valoración de un profesional de la salud. Ante síntomas graves o progresivos, consulta a un médico.
