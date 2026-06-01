import cv2
import numpy as np

from .config import default_config as _cfg


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
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
    if strength is None:
        strength = _cfg.enhance.sharpen_strength
    blurred = cv2.GaussianBlur(image, (0, 0), _cfg.enhance.sharpen_sigma)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)


def _denoise_binary(binary: np.ndarray, kernel_size: int | None = None) -> np.ndarray:
    """二值图形态学去噪：开运算去白噪点 + 闭运算填小黑孔。"""
    if kernel_size is None:
        kernel_size = _cfg.binarize.denoise_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    denoised = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return denoised


def binarize(image: np.ndarray, method: str = "otsu",
             block_size: int | None = None, c: int | None = None) -> np.ndarray:
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
