"""CLI para entrenar el detector basado en YOLOv8."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.training import TrainingConfig, train_detector


def resolve_device_preference(device_arg: Optional[str]) -> str:
    """Devuelve un dispositivo compatible con Ultralytics.

    Si el usuario pasa "auto" (valor por defecto anterior), se intenta usar la primera GPU
    disponible; si no existe CUDA/MPS, se fuerza CPU para evitar errores.
    """

    requested = (device_arg or "auto").strip()
    if requested.lower() != "auto":
        return requested

    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "0"
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available():
                return "mps"
    except Exception:
        # Si torch no está instalado o falla la detección dejamos que Ultralytics maneje el valor.
        pass

    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena el detector de fracturas")
    parser.add_argument("--data", default="data/raw/data.yaml", help="Ruta al archivo data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="Modelo base de Ultralytics (n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=200, help="Número de épocas")
    parser.add_argument("--batch", type=int, default=8, help="Tamaño de batch")
    parser.add_argument("--imgsz", type=int, default=768, help="Resolución de entrenamiento")
    parser.add_argument("--device", default="auto", help="Dispositivo (auto, cpu, 0, 0,1, etc.)")
    parser.add_argument("--export", default="models/fracture_detector.pt", help="Archivo final de pesos")
    parser.add_argument(
        "--project",
        default="models/cnn",
        help="Directorio donde Ultralytics guardará los experimentos",
    )
    parser.add_argument("--name", default=None, help="Nombre del experimento (opcional)")
    parser.add_argument("--cache", default="ram", help="Cache de datos (ram, disk, o False)")
    parser.add_argument("--workers", type=int, default=4, help="Número de workers para carga de datos")
    parser.add_argument("--patience", type=int, default=40, help="Paciencia para early stopping")
    parser.add_argument("--lr0", type=float, default=0.0003, help="Learning rate inicial")
    parser.add_argument("--lrf", type=float, default=0.00001, help="Learning rate final")
    parser.add_argument("--cos-lr", action="store_true", default=True, help="Usar cosine learning rate scheduler")
    parser.add_argument("--augment", action="store_true", default=True, help="Usar augmentation fuerte")
    parser.add_argument("--optimizer", default="AdamW", help="Optimizador (SGD, Adam, AdamW)")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay para regularización")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    resolved_device = resolve_device_preference(args.device)

    if args.device and args.device.lower() == "auto":
        print(f"[INFO] --device=auto resuelto como '{resolved_device}'")

    config = TrainingConfig(
        data_yaml=Path(args.data),
        model_variant=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=resolved_device,
        project_dir=Path(args.project),
        experiment_name=args.name,
        cache=args.cache,
        workers=args.workers,
        patience=args.patience,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        augment=args.augment,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        export_path=Path(args.export),
    )
    output = train_detector(config)
    print(f"Modelo guardado en {output}")


if __name__ == "__main__":
    main()
