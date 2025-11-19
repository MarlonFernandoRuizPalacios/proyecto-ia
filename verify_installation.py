"""
Script de verificación de instalación.
Verifica que todas las dependencias y el modelo estén correctamente instalados.
"""

import sys
from pathlib import Path

def check_python_version():
    """Verificar versión de Python."""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠️  Se recomienda Python 3.8 o superior")
        return False
    return True

def check_dependencies():
    """Verificar dependencias principales."""
    dependencies = {
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics (YOLOv8)',
        'gradio': 'Gradio',
        'fastapi': 'FastAPI',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NO INSTALADO")
            all_ok = False
    
    return all_ok

def check_model():
    """Verificar que el modelo esté descargado."""
    model_path = Path("models/fracture_detector.pt")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Modelo encontrado ({size_mb:.1f} MB)")
        return True
    else:
        print("✗ Modelo NO encontrado")
        print("  Ejecuta: git lfs pull")
        return False

def check_cuda():
    """Verificar soporte CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA disponible - GPU: {gpu_name}")
            return True
        else:
            print("⚠️  CUDA no disponible (se usará CPU)")
            return True
    except:
        print("⚠️  No se pudo verificar CUDA")
        return True

def test_model_loading():
    """Probar que el modelo se carga correctamente."""
    try:
        from src.inference import FractureDetector
        detector = FractureDetector()
        classes = detector.model.names
        print(f"✓ Modelo cargado correctamente")
        print(f"  Clases: {list(classes.values())}")
        return True
    except Exception as e:
        print(f"✗ Error al cargar modelo: {e}")
        return False

def main():
    print("="*60)
    print("VERIFICACIÓN DE INSTALACIÓN - Detector de Fracturas")
    print("="*60)
    print()
    
    checks = [
        ("Python", check_python_version),
        ("Dependencias", check_dependencies),
        ("Modelo", check_model),
        ("CUDA/GPU", check_cuda),
        ("Carga del modelo", test_model_loading),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n[{name}]")
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append(False)
    
    print()
    print("="*60)
    if all(results):
        print("✓ ¡TODO CORRECTO! Ejecuta 'python main.py' para iniciar")
    else:
        print("✗ Hay problemas. Revisa los errores arriba")
        print("\nSoluciones comunes:")
        print("  1. Instalar dependencias: pip install -r requirements.txt")
        print("  2. Descargar modelo: git lfs pull")
        print("  3. Verificar Python 3.8+")
    print("="*60)

if __name__ == "__main__":
    main()
