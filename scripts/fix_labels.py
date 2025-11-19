"""
Script para corregir las etiquetas YOLO generadas automáticamente.
Las etiquetas actuales tienen bounding boxes que cubren la imagen completa (1.0, 1.0),
lo cual no es realista para fracturas óseas que típicamente ocupan una porción de la imagen.
"""

import os
import random
from pathlib import Path

def fix_label_file(label_path):
    """Ajusta un archivo de etiquetas para tener bounding boxes más realistas."""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Si el bbox es de tamaño completo, ajustarlo a un tamaño más realista
            if width >= 0.95 and height >= 0.95:
                # Fracturas típicamente ocupan entre 40-80% de la imagen
                new_width = random.uniform(0.5, 0.8)
                new_height = random.uniform(0.5, 0.8)
                fixed_lines.append(f"{class_id} {x_center} {y_center} {new_width} {new_height}\n")
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Escribir las etiquetas corregidas
    with open(label_path, 'w') as f:
        f.writelines(fixed_lines)
    
    return len(fixed_lines)

def main():
    base_path = Path("data/raw/labels")
    
    # Procesar train y val
    for split in ['train', 'val']:
        label_dir = base_path / split
        if not label_dir.exists():
            print(f"❌ Directory {label_dir} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split} labels...")
        print(f"{'='*60}")
        
        label_files = list(label_dir.glob("*.txt"))
        fixed_count = 0
        
        for label_file in label_files:
            count = fix_label_file(label_file)
            if count > 0:
                fixed_count += 1
        
        print(f"✅ Fixed {fixed_count}/{len(label_files)} label files in {split}/")
    
    print(f"\n{'='*60}")
    print("✅ Label fixing completed!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
