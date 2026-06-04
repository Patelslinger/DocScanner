"""
图像增强模块 — CLAHE / 锐化 / 二值化 / 去噪
===============================================

对透视矫正后的文档图像做后处理增强，使其看起来像扫描仪输出。
各功能可独立开关、自由组合。

增强管线（按执行顺序）:
1. CLAHE — 局部对比度增强，改善光照不均
2. 锐化 — 高反差保留（Unsharp Mask），使文字边缘清晰
3. 二值化 — Otsu 或 Adaptive 阈值，输出黑白图
   （若 CLAHE+二值化同时开启，可选择先 CLAHE 再二值化以抑制阴影）
4. 形态学去噪 — 开运算去白点 + 闭运算填黑孔
"""

import cv2
import numpy as np

from .config import default_config as _cfg


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """限制对比度自适应直方图均衡化（CLAHE）。

    将图像分成 tile_grid_size 个小块，每个块独立做直方图均衡，
    再对每个块的对比度进行 clip_limit 裁剪限制。
    相比全局直方图均衡化，CLAHE 能增强局部细节而不过度放大噪声。

    对彩色图：在 LAB 空间的 L（亮度）通道上做 CLAHE，保持颜色不变。
    对灰度图：直接应用。

    Args:
        image: BGR 或灰度图像
        clip_limit: 对比度裁剪限幅，越大增强越强，但噪声也越明显
        tile_grid_size: 分块网格大小

    Returns:
        CLAHE 增强后的图像
    """
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l = clahe_obj.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe_obj.apply(image)


def sharpen(image: np.ndarray, strength: float | None = None) -> np.ndarray:
    """高反差保留锐化（Unsharp Mask）。

    原理：原图 + strength × (原图 - 高斯模糊版本)
    即提取模糊版本中的"低频"信息，从原图中减去以突出"高频"细节（边缘、纹理）。

    Args:
        image: 输入图像
        strength: 锐化强度，默认使用 enhance.sharpen_strength（1.5）

    Returns:
        锐化后的图像
    """
    if strength is None:
        strength = _cfg.enhance.sharpen_strength
    blurred = cv2.GaussianBlur(image, (0, 0), _cfg.enhance.sharpen_sigma)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)


def _denoise_binary(binary: np.ndarray, kernel_size: int | None = None) -> np.ndarray:
    """二值图形态学去噪。

    先做开运算（腐蚀 → 膨胀），去除白色孤立噪点；
    再做闭运算（膨胀 → 腐蚀），填充黑色小孔。

    Args:
        binary: 二值图（0 和 255）
        kernel_size: 形态学核大小

    Returns:
        去噪后的二值图
    """
    if kernel_size is None:
        kernel_size = _cfg.binarize.denoise_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    denoised = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return denoised


def binarize(image: np.ndarray, method: str = "otsu",
             block_size: int | None = None, c: int | None = None) -> np.ndarray:
    """图像二值化：将灰度图转为黑白（0 / 255）。

    支持两种方法：
    - otsu: 大津法，自动计算全局最优阈值，适合光照均匀的文档
    - adaptive: 自适应阈值，每个像素的阈值由其邻域决定，适合光照不均匀的文档

    Args:
        image: BGR 或灰度图像
        method: "otsu" 或 "adaptive"
        block_size: Adaptive 方法的块大小（奇数）
        c: Adaptive 方法的常数偏移量

    Returns:
        二值图像（0 和 255）
    """
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        if block_size is None:
            block_size = _cfg.binarize.adaptive_block_size
        if c is None:
            c = _cfg.binarize.adaptive_c
        if block_size % 2 == 0:
            block_size += 1
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, c)
    else:
        raise ValueError(f"Unknown binarize method: {method}")

    return binary


def enhance(image: np.ndarray, do_clahe: bool = True, do_sharpen: bool = True,
            do_binarize: bool = False, binarize_method: str = "otsu",
            binarize_block_size: int | None = None,
            binarize_c: int | None = None,
            denoise: bool = True,
            denoise_kernel: int | None = None) -> dict[str, np.ndarray]:
    """图像增强主函数：依次执行 CLAHE → 锐化 → 二值化 → 去噪。

    各步骤可独立开关，并根据开关状态调整执行顺序：
    - 如果同时开启 CLAHE 和 二值化（且配置允许），CLAHE 会在二值化之前执行
      以先均衡光照，减少阴影对二值化的干扰
    - 否则 CLAHE 和 锐化先做，生成增强后的彩色图；二值化基于增强结果

    Returns:
        字典，包含各阶段的结果图。键值可能包含：
        - "original":   输入原图副本
        - "clahe":      CLAHE 增强后（如有）
        - "sharpened":  锐化后（如有）
        - "binary":     二值化结果（如有）
        - "binary_denoised": 形态学去噪后的二值图（如有）
        - "final":      最终输出（二值图或增强彩色图）
    """
    results = {"original": image}

    img = image.copy()
    # 勾选 CLAHE 且勾选二值化 + clahe_before_binarize 时，先 CLAHE 再二值化
    clahe_before = do_clahe and do_binarize and _cfg.binarize.clahe_before_binarize

    if do_clahe and not clahe_before:
        img = clahe(img)
        results["clahe"] = img
    if do_sharpen:
        img = sharpen(img)
        results["sharpened"] = img
    if do_binarize:
        if clahe_before:
            # 对裁剪后的图做 CLAHE 再二值化，抑制阴影
            clahe_img = clahe(image.copy())
            binary = binarize(clahe_img, method=binarize_method,
                              block_size=binarize_block_size, c=binarize_c)
            results["clahe"] = clahe_img
        else:
            binary = binarize(img, method=binarize_method,
                              block_size=binarize_block_size, c=binarize_c)
        results["binary"] = binary

        # 二值化后去噪
        if denoise:
            binary = _denoise_binary(binary, kernel_size=denoise_kernel)
            results["binary_denoised"] = binary

    results["final"] = binary if do_binarize else img
    return results
