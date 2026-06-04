"""
IO 工具模块 — 图像读写、批量收集、PDF 导出
=============================================

负责文件系统层面的操作：
- 单张图片的加载与保存
- 从目录中批量收集图片文件
- 将扫描结果导出为 PDF（依赖 img2pdf 库）
"""

from pathlib import Path

import cv2
import numpy as np

try:
    import img2pdf
    HAS_IMG2PDF = True
except ImportError:
    HAS_IMG2PDF = False


def load_image(path: str | Path) -> np.ndarray:
    """加载图像文件为 OpenCV BGR 数组。

    支持的格式由 OpenCV 决定（jpg/png/bmp/tiff 等）。

    Args:
        path: 图像文件路径

    Returns:
        BGR 格式的 np.ndarray

    Raises:
        FileNotFoundError: 文件不存在或 OpenCV 无法读取
    """
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def save_image(image: np.ndarray, path: str | Path) -> None:
    """将 BGR 数组保存为图像文件，自动创建父目录。

    Args:
        image: BGR 格式的 np.ndarray
        path: 输出文件路径（格式由扩展名决定）
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def collect_images(input_dir: str | Path) -> list[Path]:
    """收集目录中所有支持的图片文件，按文件名排序。

    支持的扩展名: .jpg .jpeg .png .bmp .tiff .tif

    Args:
        input_dir: 输入目录路径

    Returns:
        排序后的图片文件路径列表

    Raises:
        FileNotFoundError: 目录不存在或目录中没有图片文件
    """
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
    """将多张图片合并导出为一个 PDF 文件。

    使用 img2pdf 库将图片直接嵌入 PDF（不经过 PIL 再编码），
    保持原始图片质量。

    Args:
        image_paths: 图片文件路径列表
        output_path: 输出 PDF 文件路径

    Raises:
        ImportError: img2pdf 未安装时抛出
    """
    if not HAS_IMG2PDF:
        raise ImportError("img2pdf is required for PDF export. Install: pip install img2pdf")

    image_paths = [str(p) for p in image_paths]
    pdf_bytes = img2pdf.convert(image_paths)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pdf_bytes)
