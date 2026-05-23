import cv2
import numpy as np


def grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def resize_if_large(image: np.ndarray, max_dim: int = 1500) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def blur_bgr_then_gray(bgr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """去噪 → 灰度化：先对 BGR 做高斯模糊抑制噪声，再转灰度。"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(bgr, (kernel_size, kernel_size), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
