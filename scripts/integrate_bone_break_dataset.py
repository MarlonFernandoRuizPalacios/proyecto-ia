"""Script para integrar el dataset 'Bone Break Classification' al dataset actual."""

from __future__ import annotations
import shutil
from pathlib import Path
from typing import Dict
import random

# Configuración
BONE_BREAK_ROOT = Path(r"C:\Users\Marlon\Desktop\datasets\Bone Break Classification\Bone Break Classification")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_DATA_ROOT = PROJECT_ROOT / "data" / "raw"

# Todas las fracturas se mapean a una sola clase "fracture" (class_id = 2 según tu data.yaml)
FRACTURE_CLASS_ID = 2  # "forearm fracture" en tu data.yaml
FRACTURE_CLASS_NAME = "forearm fracture"


def copy_images_and_create_labels(
    source_folder: Path,
    dest_images: Path,
    dest_labels: Path,
    class_id: int,
    prefix: str,
) -> int:
    """Copia imágenes y crea labels YOLO con bounding box completo.
    
    Args:
        source_folder: Carpeta Train o Test del dataset nuevo
        dest_images: Destino para imágenes
        dest_labels: Destino para labels
        class_id: ID de clase en YOLO
        prefix: Prefijo para evitar conflictos de nombres
    
    Returns:
        Número de imágenes copiadas
    """
    count = 0
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    for img_path in source_folder.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in valid_extensions:
            continue
        
        # Copiar imagen con prefijo único
        new_name = f"{prefix}_{img_path.stem}{img_path.suffix}"
        dest_img = dest_images / new_name
        shutil.copy2(img_path, dest_img)
        
        # Crear label YOLO (bbox completo: centro 0.5,0.5, tamaño 1.0,1.0)
        label_path = dest_labels / f"{dest_img.stem}.txt"
        with open(label_path, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
        
        count += 1
    
    return count


def integrate_dataset():
    """Integra el dataset Bone Break Classification al actual."""
    
    if not BONE_BREAK_ROOT.exists():
        print(f"❌ Error: No se encontró {BONE_BREAK_ROOT}")
        print("⚠️  Verifica que la ruta sea correcta")
        return
    
    print("="*70)
    print("🔄 INTEGRANDO BONE BREAK CLASSIFICATION DATASET")
    print("="*70)
    print(f"\n📂 Dataset origen: {BONE_BREAK_ROOT}")
    print(f"📂 Dataset destino: {CURRENT_DATA_ROOT}")
    print(f"🏷️  Todas las fracturas → clase '{FRACTURE_CLASS_NAME}' (ID: {FRACTURE_CLASS_ID})\n")
    
    # Directorios destino
    train_images = CURRENT_DATA_ROOT / "train" / "images"
    train_labels = CURRENT_DATA_ROOT / "train" / "labels"
    val_images = CURRENT_DATA_ROOT / "valid" / "images"
    val_labels = CURRENT_DATA_ROOT / "valid" / "labels"
    
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)
    
    total_train = 0
    total_val = 0
    
    # Obtener todas las carpetas de fracturas
    fracture_folders = [f for f in BONE_BREAK_ROOT.iterdir() if f.is_dir()]
    
    if not fracture_folders:
        print("❌ No se encontraron carpetas de fracturas")
        return
    
    print(f"📁 Carpetas encontradas: {len(fracture_folders)}\n")
    
    # Procesar cada tipo de fractura
    for folder in fracture_folders:
        folder_name = folder.name
        prefix = folder_name.replace(" ", "_").lower()
        
        print(f"📁 Procesando: {folder_name}")
        
        # Copiar Train
        train_folder = folder / "Train"
        if train_folder.exists() and train_folder.is_dir():
            count = copy_images_and_create_labels(
                train_folder, train_images, train_labels, FRACTURE_CLASS_ID, f"{prefix}_train"
            )
            total_train += count
            print(f"   ✅ Train: {count} imágenes")
        else:
            print(f"   ⚠️  No se encontró carpeta Train")
        
        # Copiar Test → Val
        test_folder = folder / "Test"
        if test_folder.exists() and test_folder.is_dir():
            count = copy_images_and_create_labels(
                test_folder, val_images, val_labels, FRACTURE_CLASS_ID, f"{prefix}_test"
            )
            total_val += count
            print(f"   ✅ Test→Val: {count} imágenes")
        else:
            print(f"   ⚠️  No se encontró carpeta Test")
        
        print()
    
    print("="*70)
    print("✅ INTEGRACIÓN COMPLETADA")
    print("="*70)
    print(f"📊 Total imágenes Train añadidas: {total_train}")
    print(f"📊 Total imágenes Val añadidas: {total_val}")
    print(f"📊 Total general: {total_train + total_val}")
    print("="*70)
    
    # Mostrar resumen del dataset completo
    print(f"\n📈 RESUMEN DATASET COMPLETO:")
    train_total = len(list(train_images.glob("*.[jp][pn][g]")))
    val_total = len(list(val_images.glob("*.[jp][pn][g]")))
    print(f"   Train: {train_total} imágenes totales")
    print(f"   Val: {val_total} imágenes totales")
    print(f"   Total: {train_total + val_total} imágenes")
    
    print(f"\n🚀 Siguiente paso:")
    print(f"   python scripts/train_detector.py")
    print()


if __name__ == "__main__":
    try:
        integrate_dataset()
    except Exception as e:
        print(f"\n❌ Error durante la integración: {e}")
        import traceback
        traceback.print_exc()
