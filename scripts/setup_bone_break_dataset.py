"""Configura el dataset Bone Break Classification desde cero."""

from __future__ import annotations
import shutil
from pathlib import Path
from collections import Counter
import yaml

# Configuración
BONE_BREAK_ROOT = Path(r"C:\Users\Marlon\Desktop\datasets\Bone Break Classification\Bone Break Classification")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Mapeo de categorías a clases YOLO
CLASS_MAPPING = {
    "Avulsion fracture": 0,
    "Comminuted fracture": 1,
    "Fracture Dislocation": 2,
    "Greenstick fracture": 3,
    "Hairline Fracture": 4,
    "Impacted fracture": 5,
    "Longitudinal fracture": 6,
    "Oblique fracture": 7,
    "Pathological fracture": 8,
    "Spiral Fracture": 9,
}

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


def clean_old_dataset():
    """Elimina el dataset anterior."""
    print("🗑️  Limpiando dataset anterior...")
    
    if DATA_DIR.exists():
        # Eliminar solo las carpetas de datos, mantener estructura
        for folder in ["train", "valid", "test"]:
            folder_path = DATA_DIR / folder
            if folder_path.exists():
                shutil.rmtree(folder_path)
                print(f"   ✅ Eliminado: {folder}")
    
    # Eliminar data.yaml antiguo
    old_yaml = DATA_DIR / "data.yaml"
    if old_yaml.exists():
        old_yaml.unlink()
        print(f"   ✅ Eliminado: data.yaml")
    
    print("✅ Limpieza completada\n")


def setup_directories():
    """Crea estructura de directorios YOLO."""
    for split in ["train", "valid"]:
        (DATA_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    print("✅ Directorios creados")


def process_images(source_folder: Path, dest_images: Path, dest_labels: Path, class_id: int) -> int:
    """Procesa imágenes de una carpeta y crea labels YOLO."""
    count = 0
    
    if not source_folder.exists():
        return 0
    
    for img_path in source_folder.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
        
        # Copiar imagen con nombre único
        dest_img = dest_images / f"{source_folder.parent.name.replace(' ', '_')}_{img_path.name}"
        shutil.copy2(img_path, dest_img)
        
        # Crear label YOLO (bbox completo para clasificación)
        label_path = dest_labels / f"{dest_img.stem}.txt"
        with open(label_path, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
        
        count += 1
    
    return count


def create_dataset():
    """Crea el dataset completo desde Bone Break Classification."""
    
    if not BONE_BREAK_ROOT.exists():
        print(f"❌ No se encontró: {BONE_BREAK_ROOT}")
        print("⚠️ Verifica la ruta del dataset")
        return False
    
    # Limpiar dataset anterior
    clean_old_dataset()
    
    # Crear estructura nueva
    setup_directories()
    
    stats = {"train": Counter(), "valid": Counter()}
    
    print(f"\n{'='*70}")
    print("PROCESANDO DATASET BONE BREAK CLASSIFICATION")
    print(f"{'='*70}\n")
    
    # Procesar cada tipo de fractura
    for folder_name, class_id in CLASS_MAPPING.items():
        folder_path = BONE_BREAK_ROOT / folder_name
        
        if not folder_path.exists():
            print(f"⚠️ No encontrado: {folder_name}")
            continue
        
        print(f"📁 {folder_name} (clase {class_id})")
        
        # Procesar Train
        train_folder = folder_path / "Train"
        if train_folder.exists():
            count = process_images(
                train_folder,
                DATA_DIR / "train" / "images",
                DATA_DIR / "train" / "labels",
                class_id
            )
            stats["train"][class_id] = count
            print(f"   ✅ Train: {count} imágenes")
        
        # Procesar Test como validación
        test_folder = folder_path / "Test"
        if test_folder.exists():
            count = process_images(
                test_folder,
                DATA_DIR / "valid" / "images",
                DATA_DIR / "valid" / "labels",
                class_id
            )
            stats["valid"][class_id] = count
            print(f"   ✅ Val: {count} imágenes")
    
    # Crear data.yaml
    data_yaml = DATA_DIR / "data.yaml"
    yaml_content = {
        "train": "../train/images",
        "val": "../valid/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    
    with open(data_yaml, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\n{'='*70}")
    print("✅ DATASET CREADO EXITOSAMENTE")
    print(f"{'='*70}")
    print(f"\n📊 Estadísticas:")
    print(f"   Train: {sum(stats['train'].values())} imágenes")
    print(f"   Val: {sum(stats['valid'].values())} imágenes")
    print(f"   Total: {sum(stats['train'].values()) + sum(stats['valid'].values())} imágenes")
    print(f"\n📋 Distribución por clase:")
    for class_id, name in enumerate(CLASS_NAMES):
        train_count = stats['train'].get(class_id, 0)
        val_count = stats['valid'].get(class_id, 0)
        total = train_count + val_count
        print(f"   {name:15s}: {train_count:4d} train + {val_count:3d} val = {total:4d} total")
    
    print(f"\n📄 Archivo de configuración: {data_yaml}")
    print(f"\n🚀 Siguiente paso:")
    print(f"   python scripts/train_detector.py")
    
    return True


if __name__ == "__main__":
    print("🔄 Iniciando configuración del dataset...\n")
    success = create_dataset()
    if not success:
        exit(1)
