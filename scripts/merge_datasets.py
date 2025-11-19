"""
Script para fusionar el dataset actual de Bone Break Classification con el nuevo dataset binario.
Distribuye las imágenes de 'fracture' del nuevo dataset entre las 10 clases existentes.
"""

import os
import shutil
from pathlib import Path
import random
from typing import List, Tuple
import yaml

# Configuración
CURRENT_DATASET = Path("data/raw")
NEW_DATASET = Path("data/bone fracture dataset/Dataset")
OUTPUT_DATASET = Path("data/merged")
TRAIN_SPLIT = 0.8

# 10 clases de fracturas del dataset original
FRACTURE_CLASSES = [
    "avulsion",
    "comminuted", 
    "dislocation",
    "greenstick",
    "hairline",
    "impacted",
    "longitudinal",
    "oblique",
    "pathological",
    "spiral"
]

def setup_output_structure():
    """Crea la estructura de directorios para el dataset fusionado."""
    for split in ['train', 'valid']:
        for subdir in ['images', 'labels']:
            output_dir = OUTPUT_DATASET / split / subdir
            output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Estructura de directorios creada en {OUTPUT_DATASET}")

def copy_current_dataset():
    """Copia el dataset actual al nuevo directorio."""
    copied_images = 0
    copied_labels = 0
    
    for split in ['train', 'valid']:
        # Copiar imágenes
        src_images = CURRENT_DATASET / split / 'images'
        dst_images = OUTPUT_DATASET / split / 'images'
        
        if src_images.exists():
            for img_file in src_images.glob('*.*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    shutil.copy2(img_file, dst_images / img_file.name)
                    copied_images += 1
        
        # Copiar labels
        src_labels = CURRENT_DATASET / split / 'labels'
        dst_labels = OUTPUT_DATASET / split / 'labels'
        
        if src_labels.exists():
            for label_file in src_labels.glob('*.txt'):
                shutil.copy2(label_file, dst_labels / label_file.name)
                copied_labels += 1
    
    print(f"✓ Dataset actual copiado: {copied_images} imágenes, {copied_labels} labels")
    return copied_images

def get_image_files(directory: Path) -> List[Path]:
    """Obtiene lista de archivos de imagen de un directorio."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    images = []
    for ext in image_extensions:
        images.extend(directory.glob(f'*{ext}'))
        images.extend(directory.glob(f'*{ext.upper()}'))
    return images

def create_empty_label(image_path: Path, output_label_path: Path):
    """Crea un archivo de label vacío para imágenes 'normal' (sin fracturas)."""
    # Para imágenes normales sin objetos, creamos un archivo txt vacío
    output_label_path.touch()

def distribute_fracture_images():
    """
    Distribuye las imágenes de 'fracture' del nuevo dataset entre las 10 clases.
    Como no tenemos etiquetas específicas, las distribuiremos de forma balanceada.
    """
    fracture_images = get_image_files(NEW_DATASET / 'fracture')
    random.shuffle(fracture_images)
    
    # Calcular cuántas imágenes por clase
    images_per_class = len(fracture_images) // len(FRACTURE_CLASSES)
    extra_images = len(fracture_images) % len(FRACTURE_CLASSES)
    
    print(f"✓ Distribuyendo {len(fracture_images)} imágenes de fracturas en {len(FRACTURE_CLASSES)} clases")
    print(f"  ~{images_per_class} imágenes por clase")
    
    # Split train/valid
    num_train = int(len(fracture_images) * TRAIN_SPLIT)
    train_images = fracture_images[:num_train]
    valid_images = fracture_images[num_train:]
    
    def process_images(images: List[Path], split: str):
        """Procesa un conjunto de imágenes para un split específico."""
        images_per_class_split = len(images) // len(FRACTURE_CLASSES)
        
        for class_idx, class_name in enumerate(FRACTURE_CLASSES):
            start_idx = class_idx * images_per_class_split
            end_idx = start_idx + images_per_class_split
            
            # Añadir imágenes extra a las primeras clases
            if class_idx < extra_images and split == 'train':
                end_idx += 1
            
            class_images = images[start_idx:end_idx]
            
            for img_path in class_images:
                # Copiar imagen
                new_name = f"{class_name}_{img_path.stem}{img_path.suffix}"
                dst_image = OUTPUT_DATASET / split / 'images' / new_name
                shutil.copy2(img_path, dst_image)
                
                # Crear label YOLO
                # Como no tenemos bounding boxes, creamos un label que indica
                # que toda la imagen contiene una fractura de este tipo
                # Formato: class_id center_x center_y width height (normalized)
                dst_label = OUTPUT_DATASET / split / 'labels' / f"{new_name.rsplit('.', 1)[0]}.txt"
                with open(dst_label, 'w') as f:
                    # Asumimos que la fractura ocupa el centro de la imagen
                    # con un área del 80% (puedes ajustar esto)
                    f.write(f"{class_idx} 0.5 0.5 0.8 0.8\n")
        
        return len(images)
    
    train_added = process_images(train_images, 'train')
    valid_added = process_images(valid_images, 'valid')
    
    print(f"✓ Agregadas {train_added} imágenes de fractura a train")
    print(f"✓ Agregadas {valid_added} imágenes de fractura a valid")
    
    return train_added + valid_added

def add_normal_images():
    """
    Añade imágenes 'normal' como ejemplos negativos.
    Estas imágenes tendrán labels vacíos (sin objetos detectados).
    """
    normal_images = get_image_files(NEW_DATASET / 'normal')
    
    # Split train/valid
    num_train = int(len(normal_images) * TRAIN_SPLIT)
    random.shuffle(normal_images)
    
    train_images = normal_images[:num_train]
    valid_images = normal_images[num_train:]
    
    def process_normal_images(images: List[Path], split: str):
        """Procesa imágenes normales para un split específico."""
        for img_path in images:
            # Copiar imagen
            new_name = f"normal_{img_path.stem}{img_path.suffix}"
            dst_image = OUTPUT_DATASET / split / 'images' / new_name
            shutil.copy2(img_path, dst_image)
            
            # Crear label vacío (sin objetos)
            dst_label = OUTPUT_DATASET / split / 'labels' / f"{new_name.rsplit('.', 1)[0]}.txt"
            create_empty_label(img_path, dst_label)
        
        return len(images)
    
    train_added = process_normal_images(train_images, 'train')
    valid_added = process_normal_images(valid_images, 'valid')
    
    print(f"✓ Agregadas {train_added} imágenes normales a train")
    print(f"✓ Agregadas {valid_added} imágenes normales a valid")
    
    return train_added + valid_added

def create_data_yaml():
    """Crea el archivo data.yaml para el dataset fusionado."""
    data_config = {
        'path': str(OUTPUT_DATASET.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'nc': len(FRACTURE_CLASSES),
        'names': FRACTURE_CLASSES
    }
    
    yaml_path = OUTPUT_DATASET / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Configuración YOLO guardada en {yaml_path}")

def print_summary():
    """Imprime un resumen del dataset fusionado."""
    print("\n" + "="*60)
    print("RESUMEN DEL DATASET FUSIONADO")
    print("="*60)
    
    for split in ['train', 'valid']:
        images_dir = OUTPUT_DATASET / split / 'images'
        labels_dir = OUTPUT_DATASET / split / 'labels'
        
        num_images = len(list(images_dir.glob('*.*')))
        num_labels = len(list(labels_dir.glob('*.txt')))
        
        print(f"\n{split.upper()}:")
        print(f"  Imágenes: {num_images}")
        print(f"  Labels: {num_labels}")
    
    print("\n" + "="*60)
    print(f"Dataset listo en: {OUTPUT_DATASET.absolute()}")
    print(f"Configuración: {(OUTPUT_DATASET / 'data.yaml').absolute()}")
    print("="*60)

def main():
    print("="*60)
    print("FUSIÓN DE DATASETS PARA DETECCIÓN DE FRACTURAS")
    print("="*60)
    print()
    
    # Validar que existen los datasets de entrada
    if not CURRENT_DATASET.exists():
        print(f"❌ Error: No se encuentra el dataset actual en {CURRENT_DATASET}")
        return
    
    if not NEW_DATASET.exists():
        print(f"❌ Error: No se encuentra el nuevo dataset en {NEW_DATASET}")
        return
    
    print(f"Dataset actual: {CURRENT_DATASET.absolute()}")
    print(f"Nuevo dataset: {NEW_DATASET.absolute()}")
    print(f"Dataset fusionado: {OUTPUT_DATASET.absolute()}")
    print()
    
    # Limpiar directorio de salida si existe
    if OUTPUT_DATASET.exists():
        print(f"⚠️  Limpiando directorio existente: {OUTPUT_DATASET}")
        shutil.rmtree(OUTPUT_DATASET)
    
    # Proceso de fusión
    print("\n1. Creando estructura de directorios...")
    setup_output_structure()
    
    print("\n2. Copiando dataset actual...")
    current_count = copy_current_dataset()
    
    print("\n3. Distribuyendo imágenes de fracturas del nuevo dataset...")
    fracture_count = distribute_fracture_images()
    
    print("\n4. Agregando imágenes normales como ejemplos negativos...")
    normal_count = add_normal_images()
    
    print("\n5. Creando configuración YOLO...")
    create_data_yaml()
    
    # Resumen
    print_summary()
    print(f"\n✅ ¡Fusión completada exitosamente!")
    print(f"   Total de imágenes agregadas: {fracture_count + normal_count}")
    print(f"   Dataset final: ~{current_count + fracture_count + normal_count} imágenes")
    print(f"\nAhora puedes entrenar con: python scripts/train_detector.py --data {OUTPUT_DATASET / 'data.yaml'}")

if __name__ == "__main__":
    random.seed(42)  # Para reproducibilidad
    main()
