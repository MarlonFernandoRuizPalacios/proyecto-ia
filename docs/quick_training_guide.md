# ⚡ Configuración Optimizada para 12 Horas Máximo

## ✅ Configuración Actual (3-4 horas)

**Parámetros ajustados automáticamente:**
- 🎯 Modelo: **YOLOv8s** (11M params)
- 📐 Imagen: **896px** (balance precisión/velocidad)
- 🔢 Batch: **16** (óptimo para RTX 5060)
- 📊 Épocas: **200** (con early stopping patience=40)
- 💾 VRAM: **~5-6GB** (dentro del límite de 8GB)
- ⏱️ Tiempo/época: **~45-60 segundos**
- 🎯 **Tiempo total estimado: 3-4 horas**

## 🚀 Comandos Disponibles

### 1. Integrar Nuevo Dataset (PRIMERO)
```powershell
python scripts/integrate_bone_break_dataset.py
```

**Esto añadirá ~8,000-10,000 imágenes** del dataset Bone Break Classification.

### 2. Entrenar Modelo (DESPUÉS)
```powershell
python scripts/train_detector.py
```

Con el dataset aumentado, el entrenamiento será más efectivo.

---

## 📊 Tiempo Estimado por Configuración

| Config | Modelo | Batch | Img | Época | 200 Épocas | VRAM | mAP50 |
|--------|--------|-------|-----|-------|-----------|------|-------|
| **Actual** | YOLOv8s | 16 | 896 | 45s | **2.5h** | 5-6G ✅ | 65-75% |
| Rápida | YOLOv8n | 32 | 640 | 25s | **1.4h** | 3-4G ✅ | 55-65% |
| Precisa | YOLOv8m | 12 | 1024 | 90s | **5h** | 7-8G ⚠️ | 70-80% |

---

## 🎯 Precisión Esperada

### Dataset Actual (3,631 imágenes):
- mAP50: 60-70%
- Precision promedio: 70-85%

### Después de integrar Bone Break (~12,000 imágenes totales):
- mAP50: **75-85%** ✅
- Precision promedio: **80-92%** ✅
- **Algunas clases >90% Precision** 🎯

---

## ⚙️ Opciones Avanzadas

### Si necesitas MÁS VELOCIDAD:
```powershell
python scripts/train_detector.py --model yolov8n.pt --batch 32 --imgsz 640 --epochs 150
```
- Tiempo: ~1.5 horas
- mAP50: 60-70%

### Si necesitas MÁS PRECISIÓN (hasta 8 horas):
```powershell
python scripts/train_detector.py --model yolov8m.pt --batch 8 --imgsz 1024 --epochs 250
```
- Tiempo: ~6-8 horas
- mAP50: 75-85%
- ⚠️ Usa más VRAM (~7-8GB)

---

## 🔧 Solución de Problemas

### Si da MemoryError o CUDA Out of Memory:
```powershell
# Reducir batch y/o imagen
python scripts/train_detector.py --batch 8 --imgsz 768
```

### Si el entrenamiento es muy lento:
```powershell
# Modelo más pequeño
python scripts/train_detector.py --model yolov8n.pt --batch 32
```

### Si quieres entrenar en CPU (no recomendado):
```powershell
python scripts/train_detector.py --device cpu --batch 4 --workers 2
```

---

## 📈 Progreso Durante Entrenamiento

**Deberías ver algo así:**

```
Epoch 1/200:
  GPU_mem: 5.5G    ← Dentro del límite ✅
  box_loss: 2.5    ← Bajará gradualmente
  cls_loss: 8.3    ← Bajará a ~1.0
  mAP50: 0.15      ← Subirá a 0.70-0.85

Época ~10:
  mAP50: 0.40-0.50 ← Ya es útil

Época ~50:
  mAP50: 0.65-0.75 ← Excelente

Época ~100-150:
  mAP50: 0.75-0.85 ← Convergencia
```

---

## 🎓 Métricas Objetivo

| Dataset | mAP50 Esperado | Precision | Tiempo |
|---------|---------------|-----------|--------|
| **Actual (3.6k imgs)** | 65-75% | 75-85% | 2.5h |
| **Con Bone Break (12k imgs)** | 75-85% | 85-92% | 3-4h |
| **Con augmentation extremo** | 80-90% | 88-95% | 4-5h |

---

## ✅ PASOS FINALES

### 1. Integra el nuevo dataset:
```powershell
python scripts/integrate_bone_break_dataset.py
```

### 2. Entrena el modelo:
```powershell
python scripts/train_detector.py
```

### 3. Monitorea el progreso:
- Revisa la terminal cada 10-20 épocas
- Si mAP50 no sube después de 50 épocas, detén (Ctrl+C)
- El mejor modelo se guarda automáticamente en `models/fracture_detector.pt`

**¡Listo! El entrenamiento completará en 3-4 horas con buena precisión (75-85% mAP50).**
