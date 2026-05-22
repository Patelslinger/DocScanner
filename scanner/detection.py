import cv2
import numpy as np


def canny_edge(image: np.ndarray, low: int = 75, high: int = 200) -> np.ndarray:
    return cv2.Canny(image, low, high)


def find_contours(edges: np.ndarray) -> list:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def sort_contours_by_area(contours: list) -> list:
    return sorted(contours, key=cv2.contourArea, reverse=True)


def approximate_polygon(contour: np.ndarray, epsilon_factor: float = 0.02) -> np.ndarray:
    peri = cv2.arcLength(contour, True)
    epsilon = epsilon_factor * peri
    return cv2.approxPolyDP(contour, epsilon, True)


def find_document_contour(image: np.ndarray, edges: np.ndarray | None = None) -> np.ndarray | None:
    if edges is None:
        edges = canny_edge(image)

    contours = find_contours(edges)
    contours = sort_contours_by_area(contours)

    h, w = image.shape[:2]
    min_area = (h * w) * 0.05

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        approx = approximate_polygon(cnt, epsilon_factor=0.02)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2)

    # 放宽条件：尝试更大的 epsilon
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        approx = approximate_polygon(cnt, epsilon_factor=0.05)
        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None


def order_corners(corners: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)

    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]  # top-left
    rect[2] = corners[np.argmax(s)]  # bottom-right

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]  # top-right
    rect[3] = corners[np.argmax(diff)]  # bottom-left

    return rect
