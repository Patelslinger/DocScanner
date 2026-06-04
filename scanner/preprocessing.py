"""
图像预处理模块 — 灰度化、降噪、大图缩放
===========================================

Pipeline 的前处理步骤，在文档角点检测之前执行。
主要功能：
- 彩色图转灰度图
- 高斯模糊降噪
- 超大图像等比缩放到合理尺寸（控制后续算法耗时）
- BGR 先模糊再灰度化（专为 Sobel 固定阈值策略设计）
"""

import cv2
import numpy as np

from .config import default_config as _cfg


def grayscale(image: np.ndarray) -> np.ndarray:
    """将 BGR 彩色图像转换为灰度图。

    如果输入已经是灰度图则直接返回，不做重复转换。

    Args:
        image: BGR (H,W,3) 或灰度 (H,W) 图像

    Returns:
        灰度图像 (H,W)
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """高斯模糊降噪。

    Args:
        image: 输入图像
        kernel_size: 高斯核大小，偶数会自动加 1

    Returns:
        模糊后的图像
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def resize_if_large(image: np.ndarray, max_dim: int | None = None) -> np.ndarray:
    """如果图像长边超过 max_dim，等比缩小到 max_dim。

    使用 INTER_AREA 插值，适合缩小场景，保留更多结构信息。
    控制检测阶段的图像尺寸，保证处理速度可接受。

    Args:
        image: 输入图像
        max_dim: 最大边长，默认使用配置中的 preprocess.max_dim（1500px）

    Returns:
        缩放后（或原图）的图像
    """
    if max_dim is None:
        max_dim = _cfg.preprocess.max_dim
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def blur_bgr_then_gray(bgr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """先对 BGR 三通道做高斯模糊去噪，再转灰度。

    与「先灰度再高斯模糊」的区别：在 BGR 空间做模糊可以
    利用三通道信息压制彩色噪声，灰度化后边缘更干净。
    专为 _sobel_fixed_edge 策略设计。

    Args:
        bgr: BGR 彩色图像
        kernel_size: 高斯核大小

    Returns:
        先模糊再灰度化的结果
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(bgr, (kernel_size, kernel_size), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
