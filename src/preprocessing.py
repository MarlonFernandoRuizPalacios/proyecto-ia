"""Utilidades de preprocesamiento para radiografías."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union

import io

import cv2
import numpy as np
from PIL import Image


ImageSource = Union[str, Path, bytes, bytearray, np.ndarray, Image.Image]


class PreprocessingError(RuntimeError):
	"""Se lanza cuando una imagen no puede procesarse correctamente."""


@dataclass
class PreprocessConfig:
	target_size: Tuple[int, int] = (640, 640)
	clahe: bool = True
	clahe_clip_limit: float = 2.0
	clahe_grid_size: Tuple[int, int] = (8, 8)
	normalize: bool = True
	convert_gray: bool = False


def _to_numpy(image: Image.Image) -> np.ndarray:
	array = np.array(image)
	if array.ndim == 2:
		return np.stack([array] * 3, axis=-1)
	if array.shape[2] == 4:
		# eliminar canal alfa
		return array[:, :, :3]
	return array


def load_image(source: ImageSource) -> np.ndarray:
	"""Carga una imagen desde cualquiera de las fuentes permitidas."""

	if isinstance(source, np.ndarray):
		array = source.copy()
		if array.ndim == 2:
			array = np.stack([array] * 3, axis=-1)
		return array

	if isinstance(source, Image.Image):
		return _to_numpy(source.convert("RGB"))

	if isinstance(source, (str, Path)):
		path = Path(source)
		if not path.exists():
			raise PreprocessingError(f"La ruta {path} no existe")
		with Image.open(path) as img:
			return _to_numpy(img.convert("RGB"))

	if isinstance(source, (bytes, bytearray)):
		buffer = io.BytesIO(source)
		with Image.open(buffer) as img:
			return _to_numpy(img.convert("RGB"))

	# Manejar buffers en memoria usando PIL directamente
	try:
		with Image.open(source) as img:  # type: ignore[arg-type]
			return _to_numpy(img.convert("RGB"))
	except Exception as exc:  # pragma: no cover - camino de error
		raise PreprocessingError("No fue posible cargar la imagen") from exc


def apply_clahe(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
	if not config.clahe:
		return image

	lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
	l, a, b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=config.clahe_grid_size)
	l_eq = clahe.apply(l)
	merged = cv2.merge((l_eq, a, b))
	return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
	return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
	return (image / 255.0).astype(np.float32)


def preprocess_for_inference(source: ImageSource, config: Optional[PreprocessConfig] = None) -> np.ndarray:
	cfg = config or PreprocessConfig()
	image = load_image(source)

	if cfg.convert_gray:
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

	image = apply_clahe(image, cfg)
	image = resize_image(image, cfg.target_size)
	if cfg.normalize:
		image = normalize_image(image)
	return image


def batch_preprocess(sources: Iterable[ImageSource], config: Optional[PreprocessConfig] = None) -> np.ndarray:
	processed = [preprocess_for_inference(src, config) for src in sources]
	return np.stack(processed, axis=0)


def save_image(array: np.ndarray, path: Union[str, Path]) -> Path:
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	image = Image.fromarray((array * 255).astype(np.uint8)) if array.dtype != np.uint8 else Image.fromarray(array)
	image.save(output_path)
	return output_path


__all__ = [
	"PreprocessConfig",
	"PreprocessingError",
	"apply_clahe",
	"batch_preprocess",
	"load_image",
	"normalize_image",
	"preprocess_for_inference",
	"resize_image",
	"save_image",
]
