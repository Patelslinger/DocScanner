import cv2
import numpy as np

from .config import default_config as _cfg
from .detection import order_corners


def perspective_transform(image: np.ndarray, corners: np.ndarray,
                          output_size: tuple[int, int] | None = None) -> np.ndarray:
    corners = order_corners(corners)
    if output_size is None:
        tl, tr, br, bl = corners
        width_top = np.linalg.norm(tr - tl)
        width_bot = np.linalg.norm(br - bl)
        max_width = max(int(width_top), int(width_bot))

        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_height = max(int(height_left), int(height_right))

        output_size = (max_width, max_height)

    dst = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image, M, output_size)


def pad_to_a4(image: np.ndarray) -> np.ndarray:
    """将矫正后的图像居中放置到 A4 比例 (210:297 或 297:210) 的白色画布上.

    根据图像的实际方向匹配对应的 A4 方向：横图用横版 A4，竖图用竖版 A4。
    """
    h, w = image.shape[:2]
    a4_w, a4_h = _cfg.transform.a4_width, _cfg.transform.a4_height

    # 横图用横版 A4，竖图用竖版 A4
    if w > h:
        target_ratio = a4_h / a4_w  # 横版 297/210 ≈ 1.414
    else:
        target_ratio = a4_w / a4_h  # 竖版 210/297 ≈ 0.707

    current_ratio = w / h
    if abs(current_ratio - target_ratio) < _cfg.transform.ratio_tolerance:
        return image

    if current_ratio > target_ratio:
        new_h = int(w / target_ratio)
        pad_top = (new_h - h) // 2
        pad_bot = new_h - h - pad_top
        pad_left, pad_right = 0, 0
    else:
        new_w = int(h * target_ratio)
        pad_left = (new_w - w) // 2
        pad_right = new_w - w - pad_left
        pad_top, pad_bot = 0, 0

    if len(image.shape) == 2:
        return cv2.copyMakeBorder(image, pad_top, pad_bot, pad_left, pad_right,
                                  cv2.BORDER_CONSTANT, value=255)
    else:
        return cv2.copyMakeBorder(image, pad_top, pad_bot, pad_left, pad_right,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
