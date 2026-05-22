import cv2
import numpy as np


def perspective_transform(image: np.ndarray, corners: np.ndarray,
                          output_size: tuple[int, int] | None = None) -> np.ndarray:
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
    """将矫正后的图像居中放置到 A4 比例 (210:297) 的白色画布上."""
    h, w = image.shape[:2]
    target_ratio = 210 / 297

    current_ratio = w / h
    if abs(current_ratio - target_ratio) < 0.02:
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
