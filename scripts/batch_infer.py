"""CLI para analizar carpetas de imágenes con el detector entrenado."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.inference import FractureDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inferencia masiva de radiografías")
    parser.add_argument("input", help="Carpeta con imágenes o una sola imagen")
    parser.add_argument(
        "--weights",
        default="models/fracture_detector.pt",
        help="Ruta a los pesos entrenados",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Carpeta donde se guardarán imágenes anotadas y archivos .txt",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confianza mínima")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de entrada")
    return parser


def process_image(detector: FractureDetector, image_path: Path, output_dir: Path):
    result = detector.predict(str(image_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / f"{image_path.stem}_annotated.png"
    if result.annotated_image is not None:
        Image.fromarray(result.annotated_image).save(annotated_path)
    report_path = output_dir / f"{image_path.stem}_report.txt"
    report_path.write_text(result.summary, encoding="utf-8")
    print(f"Procesada {image_path.name} -> {annotated_path.name}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    detector = FractureDetector(weights_path=args.weights, conf=args.conf, imgsz=args.imgsz)
    input_path = Path(args.input)
    images = [input_path] if input_path.is_file() else sorted(input_path.glob("*.png")) + sorted(input_path.glob("*.jpg"))
    if not images:
        raise SystemExit("No se encontraron imágenes en la ruta indicada")

    output_dir = Path(args.output)
    for img in images:
        process_image(detector, img, output_dir)


if __name__ == "__main__":
    main()
