"""
Script de entrenamiento optimizado para dataset ultimate de 17,011 imágenes.
Configuración óptima para RTX 5060 8GB.
"""

import subprocess
import sys
from pathlib import Path

def train_optimal():
    """Entrenamiento con configuración óptima."""
    
    cmd = [
        sys.executable,
        "scripts/train_detector.py",
        
        # Dataset
        "--data", "data/merged_ultimate/data.yaml",
        
        # Modelo: YOLOv8l para mejor precisión con dataset grande
        "--model", "yolov8l.pt",
        
        # Épocas: Más épocas para dataset grande
        "--epochs", "300",
        
        # Batch: Reducido para modelo grande + 1280px
        "--batch", "4",
        
        # Resolución: Mayor para mejor detección
        "--imgsz", "1280",
        
        # Device
        "--device", "0",
        
        # Paciencia: Mayor para dataset grande
        "--patience", "80",
        
        # Workers: Usar más CPU para preparar datos mientras GPU entrena
        "--workers", "8",
        
        # Cache: Cachear en RAM para acelerar (si tienes suficiente RAM)
        "--cache", "ram",
        
        # Optimizador
        "--optimizer", "AdamW",
        
        # Learning rates optimizados
        "--lr0", "0.001",
        "--lrf", "0.0001",
        
        # Augmentations optimizados para dataset grande
        "--augment",
        "--cos-lr",
        "--mosaic", "1.0",
        "--mixup", "0.15",
        "--copy-paste", "0.1",
        
        # Guardar cada 20 épocas
        "--save-period", "20"
    ]
    
    print("="*70)
    print("ENTRENAMIENTO ÓPTIMO - DATASET ULTIMATE")
    print("="*70)
    print(f"\nDataset: 17,011 imágenes (15,144 train + 1,867 valid)")
    print(f"Modelo: YOLOv8l (mejor que YOLOv8s para dataset grande)")
    print(f"Resolución: 1280px (mejor detección)")
    print(f"Épocas: 300 (más validaciones totales)")
    print(f"Workers: 8 (CPU prepara datos mientras GPU entrena)")
    print(f"Batch: 4 (ajustado para 1280px en 8GB VRAM)")
    print(f"\nTiempo estimado: 12-15 horas")
    print(f"mAP50 esperado: 75-85%")
    print("="*70)
    print()
    
    # Confirmar
    response = input("¿Iniciar entrenamiento? (s/n): ")
    if response.lower() != 's':
        print("Entrenamiento cancelado.")
        return
    
    print("\n🚀 Iniciando entrenamiento...\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    train_optimal()
