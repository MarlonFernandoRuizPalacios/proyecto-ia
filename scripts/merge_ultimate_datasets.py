"""
Script para fusionar TODOS los datasets disponibles incluyendo Fracatlas y archive(6).
Dataset final: ~17,000 imágenes de fracturas.
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
FRACATLAS_DATASET = Path("data/Fracatlas")
ARCHIVE_DATASET = Path("data/archive (6)")
OUTPUT_DATASET = Path("data/merged_ultimate")
TRAIN_SPLIT = 0.85  # 85% train, 15% valid para dataset grande

# Mapeo consolidado de clases
CLASS_MAPPING = {
    # Dataset Bone-Fracture-detection (7 clases específicas)
    'elbow positive': 0,
    'fingers positive': 1,
    'forearm fracture': 2,
    'humerus fracture': 3,
    'humerus': 4,
    'shoulder fracture': 5,
    'wrist positive': 6,
    # Datasets binarios/genéricos
    'fracture': 7,      # Fractura genérica
    'fractured': 7,     # Fracatlas usa 'fractured' - mapear a misma clase
    'normal': 8,        # Sin fractura
    'not fractured': 8  # archive(6) usa 'not fractured' - mapear a misma clase
}

CLASSES = ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 
           'humerus', 'shoulder fracture', 'wrist positive', 'fractured', 'normal']
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
    """Copia un dataset YOLO completo con labels."""
    print(f"\n📦 Procesando dataset YOLO: {dataset_name}")
    print(f"   Fuente: {source_path}")
    
    stats = {'train': 0, 'valid': 0}
    
    for split_src, split_dst in [('train', 'train'), ('valid', 'valid'), ('validation', 'valid'), ('test', 'valid')]:
        src_images = source_path / split_src / 'images'
        src_labels = source_path / split_src / 'labels'
        
        if not src_images.exists():
            continue
        
        images = get_image_files(src_images)
        
        for img_path in images:
            new_name = f"{dataset_name}_{img_path.stem}{img_path.suffix}"
            
            # Copiar imagen
            dst_image = OUTPUT_DATASET / split_dst / 'images' / new_name
            shutil.copy2(img_path, dst_image)
            
            # Copiar y ajustar label
            label_path = src_labels / f"{img_path.stem}.txt"
            dst_label = OUTPUT_DATASET / split_dst / 'labels' / f"{new_name.rsplit('.', 1)[0]}.txt"
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                with open(dst_label, 'w') as f:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0]) + class_offset
                            parts[0] = str(class_id)
                            f.write(' '.join(parts) + '\n')
            else:
                dst_label.touch()
            
            stats[split_dst] += 1
    
    print(f"   ✓ {stats['train']} train, {stats['valid']} valid")
    return stats

def process_binary_dataset(source_path: Path, dataset_name: str, 
                          fracture_class: str = 'fracture', 
                          normal_class: str = 'normal') -> Dict[str, int]:
    """Procesa dataset binario con estructura fracture/normal."""
    print(f"\n📦 Procesando dataset binario: {dataset_name}")
    print(f"   Fuente: {source_path}")
    
    stats = {'train': 0, 'valid': 0}
    
    # Buscar carpetas de fracturas
    fracture_dirs = []
    for name in ['fracture', 'fractured', 'Fractured']:
        if (source_path / name).exists():
            fracture_dirs.append(source_path / name)
    
    # Procesar fracturas
    fracture_images = []
    for fdir in fracture_dirs:
        fracture_images.extend(get_image_files(fdir))
    
    if fracture_images:
        random.shuffle(fracture_images)
        num_train = int(len(fracture_images) * TRAIN_SPLIT)
        
        for split, images in [('train', fracture_images[:num_train]), 
                              ('valid', fracture_images[num_train:])]:
            for img_path in images:
                new_name = f"{dataset_name}_frac_{img_path.stem}{img_path.suffix}"
                
                dst_image = OUTPUT_DATASET / split / 'images' / new_name
                shutil.copy2(img_path, dst_image)
                
                # Label para fractura
                dst_label = OUTPUT_DATASET / split / 'labels' / f"{new_name.rsplit('.', 1)[0]}.txt"
                with open(dst_label, 'w') as f:
                    f.write(f"{CLASS_MAPPING[fracture_class]} 0.5 0.5 0.8 0.8\n")
                
                stats[split] += 1
        
        print(f"   ✓ {len(fracture_images)} fracturas")
    
    # Buscar carpetas normales
    normal_dirs = []
    for name in ['normal', 'not fractured', 'Normal', 'Not Fractured']:
        if (source_path / name).exists():
            normal_dirs.append(source_path / name)
    
    # Procesar normales
    normal_images = []
    for ndir in normal_dirs:
        normal_images.extend(get_image_files(ndir))
    
    if normal_images:
        random.shuffle(normal_images)
        num_train = int(len(normal_images) * TRAIN_SPLIT)
        
        for split, images in [('train', normal_images[:num_train]), 
                              ('valid', normal_images[num_train:])]:
            for img_path in images:
                new_name = f"{dataset_name}_norm_{img_path.stem}{img_path.suffix}"
                
                dst_image = OUTPUT_DATASET / split / 'images' / new_name
                shutil.copy2(img_path, dst_image)
                
                # Label vacío para normal
                dst_label = OUTPUT_DATASET / split / 'labels' / f"{new_name.rsplit('.', 1)[0]}.txt"
                with open(dst_label, 'w') as f:
                    f.write(f"{CLASS_MAPPING[normal_class]} 0.5 0.5 1.0 1.0\n")
                
                stats[split] += 1
        
        print(f"   ✓ {len(normal_images)} normales")
    
    return stats

def process_archive_dataset(source_path: Path, dataset_name: str) -> Dict[str, int]:
    """Procesa archive(6) que tiene train/val con subdirectorios fractured/not fractured."""
    print(f"\n📦 Procesando dataset archive: {dataset_name}")
    print(f"   Fuente: {source_path}")
    
    stats = {'train': 0, 'valid': 0}
    
    # Mapeo de splits
    split_mapping = {'train': 'train', 'val': 'valid'}
    
    for src_split, dst_split in split_mapping.items():
        split_path = source_path / src_split
        if not split_path.exists():
            continue
        
        # Procesar fractured
        frac_path = split_path / 'fractured'
        if frac_path.exists():
            frac_images = get_image_files(frac_path)
            for img_path in frac_images:
                new_name = f"{dataset_name}_frac_{img_path.stem}{img_path.suffix}"
                
                dst_image = OUTPUT_DATASET / dst_split / 'images' / new_name
                shutil.copy2(img_path, dst_image)
                
                dst_label = OUTPUT_DATASET / dst_split / 'labels' / f"{new_name.rsplit('.', 1)[0]}.txt"
                with open(dst_label, 'w') as f:
                    f.write(f"{CLASS_MAPPING['fractured']} 0.5 0.5 0.8 0.8\n")
                
                stats[dst_split] += 1
        
        # Procesar not fractured
        norm_path = split_path / 'not fractured'
        if norm_path.exists():
            norm_images = get_image_files(norm_path)
            for img_path in norm_images:
                new_name = f"{dataset_name}_norm_{img_path.stem}{img_path.suffix}"
                
                dst_image = OUTPUT_DATASET / dst_split / 'images' / new_name
                shutil.copy2(img_path, dst_image)
                
                dst_label = OUTPUT_DATASET / dst_split / 'labels' / f"{new_name.rsplit('.', 1)[0]}.txt"
                with open(dst_label, 'w') as f:
                    f.write(f"{CLASS_MAPPING['not fractured']} 0.5 0.5 1.0 1.0\n")
                
                stats[dst_split] += 1
    
    print(f"   ✓ {stats['train']} train, {stats['valid']} valid")
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

def print_summary():
    """Imprime un resumen del dataset fusionado."""
    print("\n" + "="*70)
    print("RESUMEN DEL DATASET ULTIMATE")
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
    print(f"\nClases: {NUM_CLASSES}")
    print(f"Dataset listo en: {OUTPUT_DATASET.absolute()}")
    print("="*70)

def main():
    print("="*70)
    print("FUSIÓN ULTIMATE - TODOS LOS DATASETS")
    print("="*70)
    print()
    
    if OUTPUT_DATASET.exists():
        print(f"⚠️  Limpiando directorio existente: {OUTPUT_DATASET}")
        shutil.rmtree(OUTPUT_DATASET)
    
    print("\n1. Creando estructura...")
    setup_output_structure()
    
    print("\n2. Fusionando datasets...")
    total_stats = {'train': 0, 'valid': 0}
    
    # Dataset 1: Bone fracture (binario simple)
    if BONE_FRACTURE_DATASET.exists():
        stats = process_binary_dataset(BONE_FRACTURE_DATASET, "bone_frac", 'fracture', 'normal')
        total_stats['train'] += stats['train']
        total_stats['valid'] += stats['valid']
    
    # Dataset 2: Bone-Fracture-detection (7 clases YOLO)
    if BONE_DETECTION_V4.exists():
        stats = copy_yolo_dataset(BONE_DETECTION_V4, "bone_det_v4", class_offset=0)
        total_stats['train'] += stats['train']
        total_stats['valid'] += stats['valid']
    
    # Dataset 3: Fracatlas (1 clase YOLO)
    if FRACATLAS_DATASET.exists():
        stats = copy_yolo_dataset(FRACATLAS_DATASET, "fracatlas", class_offset=7)
        total_stats['train'] += stats['train']
        total_stats['valid'] += stats['valid']
    
    # Dataset 4: archive(6) (binario con train/val splits)
    if ARCHIVE_DATASET.exists():
        stats = process_archive_dataset(ARCHIVE_DATASET, "archive6")
        total_stats['train'] += stats['train']
        total_stats['valid'] += stats['valid']
    
    print("\n3. Creando configuración YOLO...")
    create_data_yaml()
    
    print_summary()
    print(f"\n✅ ¡Fusión completada!")
    print(f"\nPara entrenar:")
    print(f"  python scripts/train_detector.py --data {OUTPUT_DATASET / 'data.yaml'} --epochs 300")

if __name__ == "__main__":
    random.seed(42)
    main()
