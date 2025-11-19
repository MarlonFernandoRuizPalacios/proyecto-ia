# 🖼️ Imágenes de Ejemplo para Pruebas

Este proyecto **no incluye imágenes de radiografías** en el repositorio para mantener un tamaño reducido y respetar derechos de autor.

## 🔍 Opciones para Obtener Imágenes de Prueba

### Opción 1: Dataset Público (Recomendado)

Puedes descargar imágenes de datasets públicos de radiografías de fracturas:

**Roboflow Datasets:**
- [Bone Fracture Detection](https://universe.roboflow.com/bone-fracture/bone-fracture-detection)
- [FracAtlas Dataset](https://universe.roboflow.com/fracatlas)

**Kaggle Datasets:**
- [Bone Fracture Detection](https://www.kaggle.com/datasets/vuppalaadithyasairam/bone-fracture-detection-using-xrays)

### Opción 2: Usar Imágenes de Búsqueda

Puedes buscar en Google Images:
```
"fractured bone X-ray"
"wrist fracture radiograph"
"ankle fracture X-ray"
```

**⚠️ Importante:** Solo para uso educativo/personal. No redistribuir.

### Opción 3: Imágenes Sintéticas

Genera imágenes sintéticas de radiografías usando herramientas como:
- DALL-E
- Stable Diffusion
- Midjourney

---

## 📁 Dónde Colocar las Imágenes

Una vez descargadas, puedes probarlas directamente desde la interfaz web:

```bash
python main.py
# Abre http://127.0.0.1:7860
# Sube tu imagen en la interfaz
```

O usar la inferencia programática:

```python
from src.inference import FractureDetector

detector = FractureDetector()
result = detector.predict("ruta/a/tu/imagen.jpg")
print(result.summary)
```

---

## 🎯 Formato de Imágenes Soportado

- **Formatos:** JPG, PNG, BMP
- **Resolución:** Mínimo 640x640px (se redimensiona automáticamente)
- **Tipo:** Radiografías de huesos (cualquier región anatómica)
- **Orientación:** Cualquiera

---

## 📊 Ejemplos de Resultados

El modelo puede detectar:
- ✅ **Fracturas visibles** (clasifica como "fractured")
- ✅ **Sin fracturas** (clasifica como "not fractured")
- ✅ **Múltiples fracturas** en una misma imagen
- ✅ **Confidence score** para cada detección

---

## 🧪 Prueba Rápida sin Imágenes

Si quieres verificar que el sistema funciona sin imágenes, ejecuta:

```bash
python verify_installation.py
```

Esto cargará el modelo y verificará que todas las dependencias estén correctamente instaladas.

---

## ⚠️ Nota Legal

Este proyecto es **educativo**. Las predicciones del modelo **no sustituyen** el diagnóstico médico profesional. Siempre consulta con un médico certificado para evaluación clínica real.
