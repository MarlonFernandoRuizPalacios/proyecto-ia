# 🔧 Solución a Problemas de Memoria (MemoryError)

## ✅ Cambios Aplicados

La configuración se ha optimizado para GPUs de 8GB:

| Parámetro | Antes (MemoryError) | Ahora (Optimizado) |
|-----------|---------------------|-------------------|
| **Modelo** | YOLOv8x (68M params) | **YOLOv8l (43M params)** |
| **Batch** | 8 | **4** |
| **Workers** | 8 | **4** |
| **Imagen** | 1280px | **1280px** (mantiene calidad) |

## 🚀 Comando Actualizado

```powershell
python scripts/train_detector.py
```

Esto debería funcionar sin errores de memoria.

---

## 🆘 Si TODAVÍA da MemoryError

### Opción 1: Reducir Imagen (Recomendado)
```powershell
python scripts/train_detector.py --imgsz 1024 --batch 8
```
- Imagen más pequeña = Menos VRAM
- Batch más grande = Entrenamiento más eficiente

### Opción 2: Modelo Más Pequeño
```powershell
python scripts/train_detector.py --model yolov8m.pt --batch 16 --imgsz 1024
```
- YOLOv8m (25M params) vs YOLOv8l (43M params)
- Permite batch más grande

### Opción 3: Cambiar Cache a Disco
```powershell
python scripts/train_detector.py --cache disk --workers 2
```
- Usa disco en lugar de RAM
- Reduce workers a 2

### Opción 4: Configuración Conservadora (Siempre Funciona)
```powershell
python scripts/train_detector.py --model yolov8m.pt --batch 8 --imgsz 896 --workers 2 --cache disk
```
- Modelo mediano
- Imagen moderada (896px)
- Batch pequeño
- Workers mínimos
- Cache en disco

---

## 📊 Comparación de Configuraciones

### 🔥 Alta Precisión (Configuración Actual)
```powershell
python scripts/train_detector.py
```
- Modelo: Large (43M)
- Imagen: 1280px
- Batch: 4
- VRAM: ~7GB
- Tiempo: 6-8 horas
- Precisión esperada: 65-75% mAP50

### ⚡ Balanceada (Si hay MemoryError)
```powershell
python scripts/train_detector.py --model yolov8m.pt --batch 16 --imgsz 1024
```
- Modelo: Medium (25M)
- Imagen: 1024px
- Batch: 16
- VRAM: ~6GB
- Tiempo: 3-4 horas
- Precisión esperada: 55-65% mAP50

### 🚀 Rápida (Backup)
```powershell
python scripts/train_detector.py --model yolov8s.pt --batch 32 --imgsz 640 --epochs 150
```
- Modelo: Small (11M)
- Imagen: 640px
- Batch: 32
- VRAM: ~4GB
- Tiempo: 1-2 horas
- Precisión esperada: 45-55% mAP50

---

## 🔍 Diagnóstico del Error

El `MemoryError` ocurre cuando:
1. **Workers demasiado altos** → Cada worker copia datos en memoria
2. **Batch demasiado grande** → Más imágenes en GPU simultáneamente
3. **Imagen muy grande** → 1280px usa 4x más memoria que 640px
4. **Modelo muy grande** → YOLOv8x necesita más VRAM que YOLOv8l
5. **Cache RAM** → Todo el dataset en RAM (~3GB adicionales)

---

## 💡 Recomendación Final

**EJECUTA ESTO** (configuración balanceada para RTX 5060 8GB):

```powershell
python scripts/train_detector.py
```

Si falla, usa el fallback:

```powershell
python scripts/train_detector.py --model yolov8m.pt --batch 12 --imgsz 1024
```

---

## 📈 Equivalencia de Precisión

No te preocupes por usar modelo más pequeño:

| Modelo | Parámetros | mAP50 Esperado | Diferencia |
|--------|------------|----------------|------------|
| YOLOv8x | 68M | 70-80% | Referencia |
| YOLOv8l | 43M | 65-75% | -5% |
| YOLOv8m | 25M | 60-70% | -10% |
| YOLOv8s | 11M | 50-60% | -20% |

**Con buenos datos y augmentation**, YOLOv8m puede alcanzar lo que YOLOv8x logra con peores datos.

---

## 🎯 Objetivo de >90% Precisión

Para >90% **Precision** (no mAP50), cualquier modelo desde Medium hacia arriba puede lograrlo:

- ✅ **YOLOv8m**: Puede lograr 85-95% Precision en clases bien representadas
- ✅ **YOLOv8l**: Puede lograr 88-95% Precision 
- ✅ **YOLOv8x**: Puede lograr 90-97% Precision

La diferencia principal es en **mAP50** (métrica global), no tanto en Precision individual por clase.
