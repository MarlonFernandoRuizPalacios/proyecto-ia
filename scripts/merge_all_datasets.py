"""
Script para fusionar TODOS los datasets de fracturas disponibles.
Combina: Bone fracture dataset + Bone-Fracture-detection datasets
"""

import os
import shutil
from pathlib import Path
import random
from typing import List, Dict
import yaml

# Configuración
BONE_FRACTURE_DATASET = Path("data/bone fracture dataset/Dataset")
BONE_DETECTION_V4 = Path("data/Bone-Fracture-detection/bone fracture detection.v4-v4.yolov8")
OUTPUT_DATASET = Path("data/merged_full")
TRAIN_SPLIT = 0.8

# Mapeo de clases: consolidar todas las clases de fracturas
# Dataset 1 (Bone fracture dataset): binario fracture/normal -> distribuir en clases generales
# Dataset 2 (Bone-Fracture-detection): 7 clases específicas
CLASS_MAPPING = {
    # Clases del dataset de detección (mantener)
    'elbow positive': 0,
    'fingers positive': 1,
    'forearm fracture': 2,
    'humerus fracture': 3,
    'humerus': 4,
    'shoulder fracture': 5,
    'wrist positive': 6,
    # Clases del dataset binario (mapear a genérico)
    'fracture': 7,  # Fractura genérica
    'normal': 8     # Sin fractura
}

CLASSES = list(CLASS_MAPPING.keys())
NUM_CLASSES = len(CLASSES)

def setup_output_structure():
    """Crea la estructura de directorios para el dataset fusionado."""
    for split in ['train', 'valid']:
        for subdir in ['images', 'labels']:
            output_dir = OUTPUT_DATASET / split / subdir
            output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Estructura de directorios creada en {OUTPUT_DATASET}")

def get_image_files(directory: Path) -> List[Path]:
    """Obtiene lista de archivos de imagen de un directorio."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    images = []
    if directory.exists():
        for ext in image_extensions:
            images.extend(directory.glob(f'*{ext}'))
            images.extend(directory.glob(f'*{ext.upper()}'))
    return images

def copy_yolo_dataset(source_path: Path, dataset_name: str, class_offset: int = 0) -> Dict[str, int]:
    """
    Copia un dataset YOLO completo al dataset fusionado.
    
    Args:
        source_path: Ruta al dataset fuente
        dataset_name: Nombre descriptivo del dataset
        class_offset: Offset para ajustar los IDs de clase en los labels
    """
    print(f"\n📦 Procesando dataset: {dataset_name}")
    print(f"   Fuente: {source_path}")
    
    stats = {'train': 0, 'valid': 0}
    
    for split_src, split_dst in [('train', 'train'), ('valid', 'valid'), ('test', 'valid')]:
        src_images = source_path / split_src / 'images'
        src_labels = source_path / split_src / 'labels'
        
        if not src_images.exists():
            continue
        
        images = get_image_files(src_images)
        
        for img_path in images:
            # Crear nombre único para evitar colisiones
            new_name = f"{dataset_name}_{img_path.stem}{img_path.suffix}"
            
            # Copiar imagen
            dst_image = OUTPUT_DATASET / split_dst / 'images' / new_name
            shutil.copy2(img_path, dst_image)
            
            # Copiar y ajustar label si existe
            label_path = src_labels / f"{img_path.stem}.txt"
            dst_label = OUTPUT_DATASET / split_dst / 'labels' / f"{new_name.rsplit('.', 1)[0]}.txt"
            
            if label_path.exists():
                # Leer y ajustar IDs de clase
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                with open(dst_label, 'w') as f:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Ajustar ID de clase
                            class_id = int(parts[0]) + class_offset
                            parts[0] = str(class_id)
                            f.write(' '.join(parts) + '\n')
            else:
                # Crear label vacío si no existe
                dst_label.touch()
            
            stats[split_dst] += 1
    
    print(f"   ✓ {stats['train']} imágenes train, {stats['valid']} imágenes valid")
    return stats

def process_binary_dataset(source_path: Path, dataset_name: str) -> Dict[str, int]:
    """
    Procesa el dataset binario (fracture/normal).
    """
    print(f"\n📦 Procesando dataset binario: {dataset_name}")
    print(f"   Fuente: {source_path}")
    
    stats = {'train': 0, 'valid': 0}
    
    # Procesar fracturas
    fracture_images = get_image_files(source_path / 'fracture')
    random.shuffle(fracture_images)
    num_train = int(len(fracture_images) * TRAIN_SPLIT)
    
    for split, images in [('train', fracture_images[:num_train]), 
                          ('valid', fracture_images[num_train:])]:
        for img_path in images:
            new_name = f"{dataset_name}_fracture_{img_path.stem}{img_path.suffix}"
            
            # Copiar imagen
            dst_image = OUTPUT_DATASET / split / 'images' / new_name
            shutil.copy2(img_path, dst_image)
            
            # Crear label para fractura genérica
            dst_label = OUTPUT_DATASET / split / 'labels' / f"{new_name.rsplit('.', 1)[0]}.txt"
            with open(dst_label, 'w') as f:
                # Clase 7 (fracture genérica) - cubre 80% de la imagen
                f.write(f"{CLASS_MAPPING['fracture']} 0.5 0.5 0.8 0.8\n")
            
            stats[split] += 1
    
    print(f"   ✓ {len(fracture_images)} imágenes de fracturas")
    
    # Procesar normales
    normal_images = get_image_files(source_path / 'normal')
    random.shuffle(normal_images)
    num_train = int(len(normal_images) * TRAIN_SPLIT)
    
    for split, images in [('train', normal_images[:num_train]), 
                          ('valid', normal_images[num_train:])]:
        for img_path in images:
            new_name = f"{dataset_name}_normal_{img_path.stem}{img_path.suffix}"
            
            # Copiar imagen
            dst_image = OUTPUT_DATASET / split / 'images' / new_name
            shutil.copy2(img_path, dst_image)
            
            # Crear label para imagen normal
            dst_label = OUTPUT_DATASET / split / 'labels' / f"{new_name.rsplit('.', 1)[0]}.txt"
            with open(dst_label, 'w') as f:
                # Clase 8 (normal) - cubre toda la imagen
                f.write(f"{CLASS_MAPPING['normal']} 0.5 0.5 1.0 1.0\n")
            
            stats[split] += 1
    
    print(f"   ✓ {len(normal_images)} imágenes normales")
    return stats

def create_data_yaml():
    """Crea el archivo data.yaml para el dataset fusionado."""
    data_config = {
        'path': str(OUTPUT_DATASET.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'nc': NUM_CLASSES,
        'names': CLASSES
    }
    
    yaml_path = OUTPUT_DATASET / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Configuración YOLO guardada en {yaml_path}")
    print(f"  Clases totales: {NUM_CLASSES}")
    print(f"  Nombres: {CLASSES}")

def print_summary():
    """Imprime un resumen del dataset fusionado."""
    print("\n" + "="*70)
    print("RESUMEN DEL DATASET COMPLETO FUSIONADO")
    print("="*70)
    
    total_train = 0
    total_valid = 0
    
    for split in ['train', 'valid']:
        images_dir = OUTPUT_DATASET / split / 'images'
        labels_dir = OUTPUT_DATASET / split / 'labels'
        
        num_images = len(list(images_dir.glob('*.*')))
        num_labels = len(list(labels_dir.glob('*.txt')))
        
        if split == 'train':
            total_train = num_images
        else:
            total_valid = num_images
        
        print(f"\n{split.upper()}:")
        print(f"  Imágenes: {num_images:,}")
        print(f"  Labels: {num_labels:,}")
    
    print("\n" + "="*70)
    print(f"TOTAL: {total_train + total_valid:,} imágenes")
    print(f"  Train: {total_train:,} ({total_train/(total_train+total_valid)*100:.1f}%)")
    print(f"  Valid: {total_valid:,} ({total_valid/(total_train+total_valid)*100:.1f}%)")
    print(f"\nDataset listo en: {OUTPUT_DATASET.absolute()}")
    print(f"Configuración: {(OUTPUT_DATASET / 'data.yaml').absolute()}")
    print("="*70)

def main():
    print("="*70)
    print("FUSIÓN COMPLETA DE TODOS LOS DATASETS DE FRACTURAS")
    print("="*70)
    print()
    
    # Validar que existen los datasets
    datasets_to_merge = [
        (BONE_FRACTURE_DATASET, "Bone fracture dataset", "binary"),
        (BONE_DETECTION_V4, "Bone-Fracture-detection-v4", "yolo"),
    ]
    
    for dataset_path, name, _ in datasets_to_merge:
        if not dataset_path.exists():
            print(f"⚠️  Advertencia: {name} no encontrado en {dataset_path}")
    
    # Limpiar directorio de salida si existe
    if OUTPUT_DATASET.exists():
        print(f"⚠️  Limpiando directorio existente: {OUTPUT_DATASET}")
        shutil.rmtree(OUTPUT_DATASET)
    
    # Crear estructura
    print("\n1. Creando estructura de directorios...")
    setup_output_structure()
    
    # Fusionar datasets
    print("\n2. Fusionando datasets...")
    
    total_stats = {'train': 0, 'valid': 0}
    
    # Dataset binario (fracture/normal)
    if BONE_FRACTURE_DATASET.exists():
        stats = process_binary_dataset(BONE_FRACTURE_DATASET, "bone_fracture")
        total_stats['train'] += stats['train']
        total_stats['valid'] += stats['valid']
    
    # Dataset de detección con 7 clases
    if BONE_DETECTION_V4.exists():
        stats = copy_yolo_dataset(BONE_DETECTION_V4, "bone_detection_v4", class_offset=0)
        total_stats['train'] += stats['train']
        total_stats['valid'] += stats['valid']
    
    # Crear configuración YOLO
    print("\n3. Creando configuración YOLO...")
    create_data_yaml()
    
    # Resumen
    print_summary()
    print(f"\n✅ ¡Fusión completada exitosamente!")
    print(f"\nPara entrenar ejecuta:")
    print(f"  python scripts/train_detector.py --data {OUTPUT_DATASET / 'data.yaml'} --epochs 200")

if __name__ == "__main__":
    random.seed(42)  # Para reproducibilidad
    main()
