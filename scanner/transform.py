"""
透视变换模块 — 文档透视矫正 + A4 画布填充
=============================================

核心功能：
1. perspective_transform：将检测到的纸张四边形透视变换为矩形正视图
2. pad_to_a4：将矫正后的图像填充白边到 A4 比例（方便打印/展示）

透视变换原理：
使用 4 组对应点求解 3×3 的单应矩阵（8 自由度），将源四边形映射到目标矩形。
与仿射变换（3 组点，保持平行线）不同，透视变换可以处理近大远小的梯形变形。
"""

import cv2
import numpy as np

from .config import default_config as _cfg
from .detection import order_corners


def perspective_transform(image: np.ndarray, corners: np.ndarray,
                          output_size: tuple[int, int] | None = None) -> np.ndarray:
    """将检测到的文档区域透视变换为正面俯视图。

    步骤：
    1. 调用 order_corners 将四个角点排序为 [TL, TR, BR, BL]
    2. 如果未指定输出尺寸，按四边形各边长的最大值计算
    3. 构建目标矩形的四个顶点
    4. 用 cv2.getPerspectiveTransform 计算 3×3 单应矩阵
    5. 用 cv2.warpPerspective 执行变换（双线性插值）

    Args:
        image: 原始 BGR 图像
        corners: 四个角点坐标，(4, 2) 的 float32 数组
        output_size: 输出图像尺寸 (width, height)，默认自动计算

    Returns:
        透视矫正后的正面矩形图像
    """
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
    """将矫正后的文档图像填充白边到 A4 比例。

    根据图像的实际方向自动选择横版（297:210）或竖版（210:297）A4，
    将原图居中放置，两边补白。输出图像看起来像扫描件，方便打印和展示。

    如果图像宽高比已经接近 A4（容差 < ratio_tolerance），则不操作直接返回。

    Args:
        image: 透视矫正后的文档图像（BGR 或灰度）

    Returns:
        居中放置在 A4 比例白画布上的图像
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
