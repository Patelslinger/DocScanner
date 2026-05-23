import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def auto_canny(image: np.ndarray, sigma: float = 0.33) -> tuple[int, int]:
    median = np.median(image)
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))
    return low, high


def canny_edge(image: np.ndarray, low: int | None = None, high: int | None = None) -> np.ndarray:
    if low is None or high is None:
        low, high = auto_canny(image)
    return cv2.Canny(image, low, high)


def _find_contours(edges: np.ndarray) -> list:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _sort_by_area(contours: list) -> list:
    return sorted(contours, key=cv2.contourArea, reverse=True)


# ---------------------------------------------------------------------------
# 四边形提取 & 评分
# ---------------------------------------------------------------------------

def _extract_quad(approx: np.ndarray) -> np.ndarray | None:
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)

    if len(approx) > 4:
        hull = cv2.convexHull(approx)
        peri = cv2.arcLength(hull, True)
        for eps in (0.02, 0.05, 0.08, 0.12):
            simplified = cv2.approxPolyDP(hull, eps * peri, True)
            if len(simplified) == 4:
                return simplified.reshape(4, 2).astype(np.float32)
            if len(simplified) < 4:
                continue
        rect = cv2.minAreaRect(approx)
        return cv2.boxPoints(rect).astype(np.float32)

    return None


def _score_quad(quad: np.ndarray, w: int, h: int) -> float:
    """给四边形打分，越高越好。"""
    quad = order_corners(quad)
    tl, tr, br, bl = quad

    # ---- 边长合法性 ----
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_lef = np.linalg.norm(bl - tl)
    h_rig = np.linalg.norm(br - tr)
    min_side = min(w_top, w_bot, h_lef, h_rig)
    if min_side < 15:
        return -1000

    # ---- 面积比例 ----
    area = cv2.contourArea(quad.reshape(4, 1, 2))
    img_area = w * h
    ratio = area / img_area

    if ratio < 0.06:
        return -1000
    if ratio > 0.98:
        return -500  # 整张图，说明没检测到纸张

    area_score = 25 if 0.20 <= ratio <= 0.93 else (10 if 0.10 <= ratio < 0.20 else 0)

    # ---- 角点贴边检测 ----
    # 允许 1-2 个角点贴边（纸张可能被画面裁剪），但 3+ 个贴边则为误检
    margin = 5
    border_hit = 0
    for pt in quad:
        x, y = pt
        if x <= margin or x >= w - 1 - margin or y <= margin or y >= h - 1 - margin:
            border_hit += 1

    if border_hit >= 3:
        return -1000
    border_penalty = border_hit * 12  # 适度扣分，不直接拒绝

    # ---- 对边平行度 ----
    def _side_ratio(a, b):
        return min(a, b) / max(a, b) if max(a, b) > 0 else 0
    parallel_score = (_side_ratio(w_top, w_bot) + _side_ratio(h_lef, h_rig)) * 15

    # ---- 内角合理性 ----
    angle_score = 0
    pts = quad.astype(np.float64)
    for i in range(4):
        p0, p1, p2 = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        v1, v2 = p0 - p1, p2 - p1
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1 or n2 < 1:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        deg = np.degrees(np.arccos(cos_a))
        if 55 <= deg <= 125:
            angle_score += 6
        elif 40 <= deg <= 140:
            angle_score += 2

    return area_score + parallel_score + angle_score - border_penalty


# ---------------------------------------------------------------------------
# 策略 0：亮度区域分割（最优先，最可靠）
# ---------------------------------------------------------------------------

def _find_by_bright_region(gray: np.ndarray, min_area: float) -> np.ndarray | None:
    """将纸张作为图像中最亮的连通区域来检测，使用中心参考阈值。"""
    h, w = gray.shape[:2]

    # 采样图像中心 1/4 区域作为纸张亮度参考
    ch, cw = h // 4, w // 4
    center_region = gray[ch:3 * ch, cw:3 * cw]
    center_mean = float(center_region.mean())
    center_std = float(center_region.std())

    # 方法 A：基于中心亮度（中心区域通常就是纸张）
    for n_std in (0.8, 1.0, 1.3, 1.6, 2.0, 2.5):
        thresh_val = center_mean - n_std * center_std
        if thresh_val < 10 or thresh_val > 248:
            continue

        quad = _bright_region_from_threshold(gray, int(thresh_val), min_area, h, w)
        if quad is not None:
            return quad

    # 方法 B：基于 Otsu 阈值（整体前景/背景分离）
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for offset in (0, 10, 20, -10, -20):
        tv = otsu_val + offset
        if 10 <= tv <= 248:
            quad = _bright_region_from_threshold(gray, tv, min_area, h, w)
            if quad is not None:
                return quad

    # 方法 C：百分位阈值（回退）
    for pct in (55, 60, 65, 70, 75):
        thresh_val = np.percentile(gray, pct)
        if thresh_val < 10 or thresh_val > 248:
            continue
        quad = _bright_region_from_threshold(gray, int(thresh_val), min_area, h, w)
        if quad is not None:
            return quad

    return None


def _bright_region_from_threshold(gray: np.ndarray, thresh_val: int,
                                   min_area: float, h: int, w: int) -> np.ndarray | None:
    """给定阈值，从二值图中提取最大的连通区域并逼近四边形。"""
    border = 25
    padded = cv2.copyMakeBorder(gray, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=0)

    _, binary = cv2.threshold(padded, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    contours = _find_contours(opened)
    if not contours:
        return None

    contours = _sort_by_area(contours)
    for cnt in contours[:3]:
        if cv2.contourArea(cnt) < min_area:
            continue

        # 方法 A：直接用轮廓逼近（比 convexHull 更贴近实际纸张边界）
        peri = cv2.arcLength(cnt, True)
        for eps in (0.01, 0.02, 0.04, 0.06, 0.10):
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            quad = _extract_quad(approx)
            if quad is None:
                continue
            if not cv2.isContourConvex(quad.reshape(4, 1, 2)):
                continue
            quad_adj = quad - border
            s = _score_quad(quad_adj, w, h)
            if s > 0:
                return quad_adj

        # 方法 B：用 minAreaRect（最紧致的外接矩形，不会碰到图像边框）
        rot_rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rot_rect).astype(np.float32)
        box_adj = box - border
        if cv2.isContourConvex(box_adj.reshape(4, 1, 2)):
            s = _score_quad(box_adj, w, h)
            if s > 0:
                return box_adj

    return None


# ---------------------------------------------------------------------------
# 策略 1：多参数 Canny 扫描
# ---------------------------------------------------------------------------

def _canny_sweep(gray: np.ndarray, min_area: float) -> np.ndarray | None:
    h, w = gray.shape[:2]
    best_quad, best_score = None, -9999

    # 给图像加黑边，防止纸张与图像边界粘连
    border = 25
    padded = cv2.copyMakeBorder(gray, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=0)

    sigmas = [0.12, 0.20, 0.28, 0.36, 0.44, 0.52]
    blurs = [3, 5]
    dilates = [2, 3, 4]
    epsilons = [0.01, 0.02, 0.04, 0.06, 0.10]

    for sigma in sigmas:
        low, high = auto_canny(padded, sigma)
        for bk in blurs:
            blurred = cv2.GaussianBlur(padded, (bk, bk), 0)
            edges = cv2.Canny(blurred, low, high)
            for dil_iter in dilates:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                dilated = cv2.dilate(edges, kernel, iterations=dil_iter)

                contours = _find_contours(dilated)
                contours = _sort_by_area(contours)

                for cnt in contours[:8]:
                    if cv2.contourArea(cnt) < min_area:
                        continue
                    peri = cv2.arcLength(cnt, True)
                    for eps in epsilons:
                        approx = cv2.approxPolyDP(cnt, eps * peri, True)
                        quad = _extract_quad(approx)
                        if quad is None:
                            continue
                        if not cv2.isContourConvex(quad.reshape(4, 1, 2)):
                            continue
                        # 去掉 padding 偏移
                        quad_adj = quad - border
                        s = _score_quad(quad_adj, w, h)
                        if s > best_score:
                            best_score = s
                            best_quad = quad_adj

    return best_quad if best_score > 0 else None


# ---------------------------------------------------------------------------
# 策略 2：自适应阈值
# ---------------------------------------------------------------------------

def _adaptive_threshold(gray: np.ndarray, min_area: float) -> np.ndarray | None:
    h, w = gray.shape[:2]
    border = 25
    padded = cv2.copyMakeBorder(gray, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=0)

    for bs in [41, 61, 81, 111, 151]:
        if bs % 2 == 0:
            bs += 1
        th = cv2.adaptiveThreshold(padded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, bs, 4)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=5)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

        contours = _find_contours(opened)
        contours = _sort_by_area(contours)

        for cnt in contours[:5]:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in (0.01, 0.02, 0.04, 0.06, 0.10):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                quad = _extract_quad(approx)
                if quad is None:
                    continue
                if not cv2.isContourConvex(quad.reshape(4, 1, 2)):
                    continue
                quad_adj = quad - border
                s = _score_quad(quad_adj, w, h)
                if s > 0:
                    return quad_adj

    return None


# ---------------------------------------------------------------------------
# 策略 3：Sobel 梯度
# ---------------------------------------------------------------------------

def _sobel_gradient(gray: np.ndarray, min_area: float) -> np.ndarray | None:
    h, w = gray.shape[:2]
    border = 25
    padded = cv2.copyMakeBorder(gray, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=0)

    gx = cv2.Sobel(padded, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(padded, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = np.uint8(np.clip(mag, 0, 255))

    for thresh in (25, 40, 60, 80):
        _, binary = cv2.threshold(mag, thresh, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=3)

        contours = _find_contours(dilated)
        contours = _sort_by_area(contours)

        for cnt in contours[:5]:
            if cv2.contourArea(cnt) < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in (0.01, 0.02, 0.04, 0.06, 0.10):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                quad = _extract_quad(approx)
                if quad is None:
                    continue
                if not cv2.isContourConvex(quad.reshape(4, 1, 2)):
                    continue
                quad_adj = quad - border
                s = _score_quad(quad_adj, w, h)
                if s > 0:
                    return quad_adj

    return None


# ---------------------------------------------------------------------------
# 策略 4：颜色分割
# ---------------------------------------------------------------------------

def _color_segmentation(bgr: np.ndarray, min_area: float) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    border = 25
    padded = cv2.copyMakeBorder(bgr, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    hsv = cv2.cvtColor(padded, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(padded, cv2.COLOR_BGR2LAB)
    s_ch, v_ch = hsv[:, :, 1], hsv[:, :, 2]
    l_ch = lab[:, :, 0]

    masks = [
        (s_ch < 30) & (v_ch > 130),
        (s_ch < 40) & (v_ch > 150),
        s_ch < 35,
        l_ch > 170,
        (s_ch < 25) & (v_ch > 100),
    ]

    for raw in masks:
        mask = raw.astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours = _find_contours(mask)
        if not contours:
            continue
        contours = _sort_by_area(contours)

        for cnt in contours[:3]:
            if cv2.contourArea(cnt) < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in (0.01, 0.02, 0.04, 0.06, 0.10):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                quad = _extract_quad(approx)
                if quad is None:
                    continue
                if not cv2.isContourConvex(quad.reshape(4, 1, 2)):
                    continue
                quad_adj = quad - border
                s = _score_quad(quad_adj, w, h)
                if s > 0:
                    return quad_adj

    return None


# ---------------------------------------------------------------------------
# 策略 5：Hough 直线检测
# ---------------------------------------------------------------------------

def _hough_lines(gray: np.ndarray) -> np.ndarray | None:
    h, w = gray.shape[:2]
    border = 25
    padded = cv2.copyMakeBorder(gray, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=0)
    ph, pw = padded.shape[:2]

    for low, high in [(30, 90), (50, 150), (75, 200), (25, 75)]:
        edges = cv2.Canny(padded, low, high)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=min(pw, ph) // 5,
                                maxLineGap=60)
        if lines is None or len(lines) < 4:
            continue

        all_pts = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            all_pts.append([x1, y1])
            all_pts.append([x2, y2])

        pts = np.array(all_pts, dtype=np.int32)
        hull = cv2.convexHull(pts)
        if hull is None or len(hull) < 4:
            continue

        peri = cv2.arcLength(hull, True)
        for eps in (0.01, 0.02, 0.04, 0.06, 0.10):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            quad = _extract_quad(approx)
            if quad is None:
                continue
            if not cv2.isContourConvex(quad.reshape(4, 1, 2)):
                continue
            quad_adj = quad - border
            s = _score_quad(quad_adj, w, h)
            if s > 0:
                return quad_adj

    return None


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def find_document_contour(image: np.ndarray,
                          edges: np.ndarray | None = None,
                          bgr: np.ndarray | None = None) -> np.ndarray | None:
    """多策略文档角点检测。

    按顺序尝试：亮度区域 → 多参数 Canny → 自适应阈值 → Sobel →
    颜色分割 → Hough 直线。返回 (4, 2) 的角点数组，或 None。
    """
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bgr_input = bgr if bgr is not None else (image if len(image.shape) == 3 else None)
    h, w = gray.shape[:2]
    min_area = (h * w) * 0.05

    strategies = [
        ("bright",    lambda: _find_by_bright_region(gray, min_area)),
        ("canny",     lambda: _canny_sweep(gray, min_area)),
        ("adaptive",  lambda: _adaptive_threshold(gray, min_area)),
        ("sobel",     lambda: _sobel_gradient(gray, min_area)),
        ("color",     lambda: _color_segmentation(bgr_input, min_area) if bgr_input is not None else None),
        ("hough",     lambda: _hough_lines(gray)),
    ]

    best_quad, best_score = None, -9999

    for _name, strategy in strategies:
        try:
            quad = strategy()
            if quad is not None:
                s = _score_quad(quad, w, h)
                if s > best_score:
                    best_score = s
                    best_quad = quad
        except Exception:
            continue

    return best_quad


# ---------------------------------------------------------------------------
# 角点排序
# ---------------------------------------------------------------------------

def order_corners(corners: np.ndarray) -> np.ndarray:
    """排序为 [top-left, top-right, bottom-right, bottom-left]"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    return rect
