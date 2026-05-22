import cv2
import numpy as np


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


def sharpen(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ], dtype=np.float32)

    # 5 为中性值，通过 strength 控制锐化程度
    kernel[1, 1] = 4 * strength + 1
    kernel = kernel * (1.0 / (1 + 4 * strength)) + (1 - 1.0 / (1 + 4 * strength)) * np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ], dtype=np.float32)

    return cv2.filter2D(image, -1, kernel)


def binarize(image: np.ndarray, method: str = "otsu", block_size: int = 11, c: int = 2) -> np.ndarray:
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, c)
    else:
        raise ValueError(f"Unknown binarize method: {method}")

    return binary


def enhance(image: np.ndarray, do_clahe: bool = True, do_sharpen: bool = True,
            do_binarize: bool = False, binarize_method: str = "otsu") -> dict[str, np.ndarray]:
    results = {"original": image}

    img = image.copy()
    if do_clahe:
        img = clahe(img)
        results["clahe"] = img
    if do_sharpen:
        img = sharpen(img)
        results["sharpened"] = img
    if do_binarize:
        binary = binarize(img, method=binarize_method)
        results["binary"] = binary

    results["final"] = binary if do_binarize else img
    return results
