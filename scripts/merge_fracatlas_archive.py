"""
Script para fusionar SOLO Fracatlas y archive(6).
Dataset optimizado con las 2 fuentes más grandes.
"""

import os
import shutil
from pathlib import Path
import random
from typing import List, Dict
import yaml

# Configuración
FRACATLAS_DATASET = Path("data/Fracatlas")
ARCHIVE_DATASET = Path("data/archive (6)")
OUTPUT_DATASET = Path("data/merged_fracatlas_archive")
TRAIN_SPLIT = 0.85

# Mapeo de clases - binario simplificado
CLASS_MAPPING = {
    'fractured': 0,
    'not fractured': 1
}

CLASSES = ['fractured', 'not fractured']
NUM_CLASSES = len(CLASSES)

def setup_output_structure():
    """Crea la estructura de directorios."""
    for split in ['train', 'valid']:
        for subdir in ['images', 'labels']:
            output_dir = OUTPUT_DATASET / split / subdir
            output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Estructura creada en {OUTPUT_DATASET}")

def get_image_files(directory: Path) -> List[Path]:
    """Obtiene lista de archivos de imagen."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    images = []
    if directory.exists():
        for ext in image_extensions:
            images.extend(directory.glob(f'*{ext}'))
            images.extend(directory.glob(f'*{ext.upper()}'))
    return images

def process_fracatlas(dataset_name: str = "fracatlas") -> Dict[str, int]:
    """Procesa Fracatlas (solo fracturas)."""
    print(f"\n📦 Procesando Fracatlas")
    
    stats = {'train': 0, 'valid': 0}
    
    # Mapeo de splits
    split_mapping = {'train': 'train', 'validation': 'valid'}
    
    for src_split, dst_split in split_mapping.items():
        src_images = FRACATLAS_DATASET / src_split / 'images'
        src_labels = FRACATLAS_DATASET / src_split / 'labels'
        
        if not src_images.exists():
            continue
        
        images = get_image_files(src_images)
        
        for img_path in images:
            new_name = f"{dataset_name}_{img_path.stem}{img_path.suffix}"
            
            # Copiar imagen
            dst_image = OUTPUT_DATASET / dst_split / 'images' / new_name
            shutil.copy2(img_path, dst_image)
            
            # Copiar/ajustar label
            label_path = src_labels / f"{img_path.stem}.txt"
            dst_label = OUTPUT_DATASET / dst_split / 'labels' / f"{new_name.rsplit('.', 1)[0]}.txt"
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                with open(dst_label, 'w') as f:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Clase 0 = fractured
                            parts[0] = '0'
                            f.write(' '.join(parts) + '\n')
            else:
                # Crear label para fractura
                with open(dst_label, 'w') as f:
                    f.write(f"0 0.5 0.5 0.8 0.8\n")
            
            stats[dst_split] += 1
    
    print(f"   ✓ {stats['train']} train, {stats['valid']} valid")
    return stats

def process_archive(dataset_name: str = "archive6") -> Dict[str, int]:
    """Procesa archive(6) con train/val splits."""
    print(f"\n📦 Procesando archive(6)")
    
    stats = {'train': 0, 'valid': 0}
    
    split_mapping = {'train': 'train', 'val': 'valid'}
    
    for src_split, dst_split in split_mapping.items():
        split_path = ARCHIVE_DATASET / src_split
        if not split_path.exists():
            continue
        
        # Procesar fractured (clase 0)
        frac_path = split_path / 'fractured'
        if frac_path.exists():
            frac_images = get_image_files(frac_path)
            for img_path in frac_images:
                new_name = f"{dataset_name}_frac_{img_path.stem}{img_path.suffix}"
                
                dst_image = OUTPUT_DATASET / dst_split / 'images' / new_name
                shutil.copy2(img_path, dst_image)
                
                dst_label = OUTPUT_DATASET / dst_split / 'labels' / f"{new_name.rsplit('.', 1)[0]}.txt"
                with open(dst_label, 'w') as f:
                    f.write(f"0 0.5 0.5 0.8 0.8\n")
                
                stats[dst_split] += 1
        
        # Procesar not fractured (clase 1)
        norm_path = split_path / 'not fractured'
        if norm_path.exists():
            norm_images = get_image_files(norm_path)
            for img_path in norm_images:
                new_name = f"{dataset_name}_norm_{img_path.stem}{img_path.suffix}"
                
                dst_image = OUTPUT_DATASET / dst_split / 'images' / new_name
                shutil.copy2(img_path, dst_image)
                
                dst_label = OUTPUT_DATASET / dst_split / 'labels' / f"{new_name.rsplit('.', 1)[0]}.txt"
                with open(dst_label, 'w') as f:
                    f.write(f"1 0.5 0.5 1.0 1.0\n")
                
                stats[dst_split] += 1
    
    print(f"   ✓ {stats['train']} train, {stats['valid']} valid")
    return stats

def create_data_yaml():
    """Crea el archivo data.yaml."""
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
    
    print(f"\n✓ Configuración guardada en {yaml_path}")

def print_summary():
    """Imprime resumen del dataset."""
    print("\n" + "="*70)
    print("DATASET FRACATLAS + ARCHIVE(6)")
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
    print(f"\nClases: {NUM_CLASSES} (fractured, not fractured)")
    print(f"Dataset listo en: {OUTPUT_DATASET.absolute()}")
    print("="*70)

def main():
    print("="*70)
    print("FUSIÓN: FRACATLAS + ARCHIVE(6)")
    print("="*70)
    
    if OUTPUT_DATASET.exists():
        print(f"⚠️  Limpiando directorio existente")
        shutil.rmtree(OUTPUT_DATASET)
    
    print("\n1. Creando estructura...")
    setup_output_structure()
    
    print("\n2. Fusionando datasets...")
    total_stats = {'train': 0, 'valid': 0}
    
    # Fracatlas
    if FRACATLAS_DATASET.exists():
        stats = process_fracatlas()
        total_stats['train'] += stats['train']
        total_stats['valid'] += stats['valid']
    
    # archive(6)
    if ARCHIVE_DATASET.exists():
        stats = process_archive()
        total_stats['train'] += stats['train']
        total_stats['valid'] += stats['valid']
    
    print("\n3. Creando configuración...")
    create_data_yaml()
    
    print_summary()
    print(f"\n✅ Listo para entrenar!")

if __name__ == "__main__":
    random.seed(42)
    main()
