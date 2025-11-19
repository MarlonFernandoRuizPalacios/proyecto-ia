"""Script simple para configurar dataset Bone Break Classification."""

from __future__ import annotations
import shutil
from pathlib import Path
from collections import Counter
import yaml

# RUTAS
BONE_BREAK_SOURCE = Path(r"C:\Users\Marlon\Desktop\datasets\Bone Break Classification\Bone Break Classification")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DEST = PROJECT_ROOT / "data" / "raw"

# Clases del dataset
FRACTURE_TYPES = [
    "Avulsion fracture",
    "Comminuted fracture",
    "Fracture Dislocation",
    "Greenstick fracture",
    "Hairline Fracture",
    "Impacted fracture",
    "Longitudinal fracture",
    "Oblique fracture",
    "Pathological fracture",
    "Spiral Fracture",
]

CLASS_NAMES = [
    "avulsion",
    "comminuted",
    "dislocation",
    "greenstick",
    "hairline",
    "impacted",
    "longitudinal",
    "oblique",
    "pathological",
    "spiral",
]


def setup_directories():
    """Limpia y crea directorios."""
    # Limpiar data/raw si existe
    if DATA_DEST.exists():
        print(f"🗑️  Eliminando dataset anterior: {DATA_DEST}")
        shutil.rmtree(DATA_DEST)
    
    # Crear estructura YOLO
    (DATA_DEST / "images" / "train").mkdir(parents=True, exist_ok=True)
    (DATA_DEST / "images" / "val").mkdir(parents=True, exist_ok=True)
    (DATA_DEST / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (DATA_DEST / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Directorios creados en: {DATA_DEST}")


def copy_images_and_labels(source_folder: Path, dest_split: str, class_id: int, class_name: str) -> int:
    """Copia imágenes y crea labels YOLO."""
    if not source_folder.exists():
        return 0
    
    dest_images = DATA_DEST / "images" / dest_split
    dest_labels = DATA_DEST / "labels" / dest_split
    
    count = 0
    for img_path in source_folder.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
        
        # Nombre único: clase_original.ext
        new_name = f"{class_name}_{img_path.name}"
        
        # Copiar imagen
        shutil.copy2(img_path, dest_images / new_name)
        
        # Crear label (bbox completo para clasificación)
        label_file = dest_labels / f"{Path(new_name).stem}.txt"
        with open(label_file, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
        
        count += 1
    
    return count


def create_dataset():
    """Configura todo el dataset."""
    
    # Verificar que existe el dataset origen
    if not BONE_BREAK_SOURCE.exists():
        print(f"❌ ERROR: No se encontró el dataset en:")
        print(f"   {BONE_BREAK_SOURCE}")
        print(f"\n⚠️  Verifica que la ruta sea correcta")
        return False
    
    print("\n" + "="*70)
    print("CONFIGURANDO DATASET BONE BREAK CLASSIFICATION")
    print("="*70)
    print(f"Origen: {BONE_BREAK_SOURCE}")
    print(f"Destino: {DATA_DEST}")
    print("="*70 + "\n")
    
    # Limpiar y crear directorios
    setup_directories()
    
    # Estadísticas
    stats = {"train": Counter(), "val": Counter()}
    
    # Procesar cada tipo de fractura
    for class_id, (folder_name, class_name) in enumerate(zip(FRACTURE_TYPES, CLASS_NAMES)):
        folder_path = BONE_BREAK_SOURCE / folder_name
        
        if not folder_path.exists():
            print(f"⚠️  No encontrado: {folder_name}")
            continue
        
        print(f"📁 Procesando: {folder_name}")
        
        # Copiar Train
        train_folder = folder_path / "Train"
        if train_folder.exists():
            count = copy_images_and_labels(train_folder, "train", class_id, class_name)
            stats["train"][class_id] = count
            print(f"   ✅ Train: {count} imágenes → data/raw/images/train/")
        
        # Copiar Test como validación
        test_folder = folder_path / "Test"
        if test_folder.exists():
            count = copy_images_and_labels(test_folder, "val", class_id, class_name)
            stats["val"][class_id] = count
            print(f"   ✅ Test→Val: {count} imágenes → data/raw/images/val/")
    
    # Crear data.yaml
    data_yaml = DATA_DEST / "data.yaml"
    yaml_content = {
        "path": str(DATA_DEST.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    
    with open(data_yaml, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    # Resumen final
    total_train = sum(stats["train"].values())
    total_val = sum(stats["val"].values())
    
    print("\n" + "="*70)
    print("✅ DATASET CONFIGURADO EXITOSAMENTE")
    print("="*70)
    print(f"\n📊 Total:")
    print(f"   Train: {total_train} imágenes")
    print(f"   Val: {total_val} imágenes")
    print(f"   Total: {total_train + total_val} imágenes")
    
    print(f"\n📋 Distribución por clase:")
    for class_id, name in enumerate(CLASS_NAMES):
        train_c = stats["train"].get(class_id, 0)
        val_c = stats["val"].get(class_id, 0)
        total = train_c + val_c
        print(f"   {name:15s}: {train_c:4d} train + {val_c:3d} val = {total:4d} total")
    
    print(f"\n📄 Configuración YOLO: {data_yaml}")
    print(f"\n🚀 Siguiente paso:")
    print(f"   python scripts/train_detector.py")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = create_dataset()
    if not success:
        print("\n❌ Configuración fallida. Verifica la ruta del dataset.")
    else:
        print("✅ Listo para entrenar!")
