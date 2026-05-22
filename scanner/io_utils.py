import os
from pathlib import Path

import cv2
import numpy as np

try:
    import img2pdf
    HAS_IMG2PDF = True
except ImportError:
    HAS_IMG2PDF = False


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def save_image(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def collect_images(input_dir: str | Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    input_dir = Path(input_dir)
    paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in extensions
    )
    if not paths:
        raise FileNotFoundError(f"No images found in: {input_dir}")
    return paths


def save_as_pdf(image_paths: list[str | Path], output_path: str | Path) -> None:
    if not HAS_IMG2PDF:
        raise ImportError("img2pdf is required for PDF export. Install: pip install img2pdf")

    image_paths = [str(p) for p in image_paths]
    pdf_bytes = img2pdf.convert(image_paths)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pdf_bytes)
