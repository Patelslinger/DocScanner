import cv2
import numpy as np

from .config import default_config as _cfg


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
        for eps in _cfg.extract.eps_values:
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
    if min_side < _cfg.scoring.min_side:
        return -1000

    # ---- 面积比例 ----
    area = cv2.contourArea(quad.reshape(4, 1, 2))
    img_area = w * h
    ratio = area / img_area

    if ratio < _cfg.scoring.area_min_ratio:
        return -1000
    if ratio > _cfg.scoring.area_max_ratio:
        return -500  # 整张图，说明没检测到纸张

    area_score = (_cfg.scoring.area_full_score
                  if _cfg.scoring.area_full_min <= ratio <= _cfg.scoring.area_full_max
                  else (_cfg.scoring.area_partial_score
                        if _cfg.scoring.area_partial_min <= ratio < _cfg.scoring.area_full_min
                        else 0))

    # ---- 角点贴边检测 ----
    margin = _cfg.scoring.border_margin
    border_hit = 0
    for pt in quad:
        x, y = pt
        if x <= margin or x >= w - 1 - margin or y <= margin or y >= h - 1 - margin:
            border_hit += 1

    if border_hit >= _cfg.scoring.border_max_hits:
        return -1000
    border_penalty = border_hit * _cfg.scoring.border_penalty

    # ---- 对边平行度 ----
    def _side_ratio(a, b):
        return min(a, b) / max(a, b) if max(a, b) > 0 else 0
    parallel_score = (_side_ratio(w_top, w_bot) + _side_ratio(h_lef, h_rig)) * _cfg.scoring.parallel_weight

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
        if _cfg.scoring.angle_full_min <= deg <= _cfg.scoring.angle_full_max:
            angle_score += _cfg.scoring.angle_full_score
        elif _cfg.scoring.angle_partial_min <= deg <= _cfg.scoring.angle_partial_max:
            angle_score += _cfg.scoring.angle_partial_score

    # ---- 矩形度：轮廓面积 / boundingRect 面积 ----
    x, y, bw, bh = cv2.boundingRect(quad.reshape(4, 1, 2).astype(np.int32))
    bbox_area = bw * bh
    if bbox_area > 0:
        rect_ratio = area / bbox_area
        if rect_ratio < _cfg.scoring.rect_ratio_reject:
            return -500
        rect_score = (rect_ratio - _cfg.scoring.rect_ratio_base) * _cfg.scoring.rect_ratio_weight
    else:
        rect_score = 0

    return area_score + parallel_score + angle_score + rect_score - border_penalty


# ---------------------------------------------------------------------------
# 策略 0：射线搜索（最优先，不依赖完整边缘）
# ---------------------------------------------------------------------------

def _radial_search(gray: np.ndarray, bgr: np.ndarray | None,
                   min_area: float) -> np.ndarray | None:
    """从图像中心向外发射射线，沿每条射线找梯度最大处作为纸张边界。"""
    h, w = gray.shape[:2]
    cy, cx = h // 2, w // 2

    blur_k = max(_cfg.radial.blur_k_min, min(h, w) // _cfg.radial.blur_k_div)
    if blur_k % 2 == 0:
        blur_k += 1
    gray_blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    gx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    boundary_pts = []
    max_radius = max(w, h)

    for angle_deg in range(0, 360, _cfg.radial.angle_step):
        rad = np.radians(angle_deg)
        dx, dy = np.cos(rad), np.sin(rad)
        best_step, best_grad = 0, -1
        for step in range(_cfg.radial.step_start, max_radius, _cfg.radial.step_inc):
            x = int(cx + dx * step)
            y = int(cy + dy * step)
            if x < 2 or x >= w - 2 or y < 2 or y >= h - 2:
                if best_step == 0:
                    boundary_pts.append((x, y))
                break
            g = grad_mag[y, x]
            if g > best_grad:
                best_grad = g
                best_step = step
        if best_step > 0:
            bx = int(cx + dx * best_step)
            by = int(cy + dy * best_step)
            boundary_pts.append((bx, by))

    if len(boundary_pts) < _cfg.radial.min_boundary_pts:
        return None

    pts_array = np.array(boundary_pts, dtype=np.float32)
    distances = np.sqrt((pts_array[:, 0] - cx) ** 2 + (pts_array[:, 1] - cy) ** 2)
    median_dist = np.median(distances)
    if median_dist < _cfg.radial.min_median_dist:
        return None
    keep = (distances > median_dist * _cfg.radial.dist_lower_ratio) & (distances < median_dist * _cfg.radial.dist_upper_ratio)
    pts_filtered = pts_array[keep]

    if len(pts_filtered) < _cfg.radial.min_filtered_pts:
        return None

    centroid_x, centroid_y = pts_filtered[:, 0].mean(), pts_filtered[:, 1].mean()
    quad = _quadrant_fit_quad(pts_filtered, centroid_x, centroid_y, w, h)
    if quad is None:
        return None

    quad = _refine_corners_on_edges(quad, gray, w, h)
    s = _score_quad(quad, w, h)
    if s > 0:
        return quad
    return None


def _quadrant_fit_quad(points: np.ndarray, cx: float, cy: float,
                        w: int, h: int) -> np.ndarray | None:
    """将边界点按相对中心的方位分 4 组，每组拟合一条边，求交点。"""
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    a = _cfg.quadrant.angles
    angles = np.degrees(np.arctan2(dy, dx)) % 360

    groups = {
        'right': points[(angles < a[0]) | (angles >= a[3])],
        'bottom': points[(angles >= a[0]) & (angles < a[1])],
        'left': points[(angles >= a[1]) & (angles < a[2])],
        'top': points[(angles >= a[2]) & (angles < a[3])],
    }

    lines = {}
    for name, group_pts in groups.items():
        if len(group_pts) < _cfg.quadrant.min_group_pts:
            continue
        line = _ransac_line(group_pts)
        if line is not None:
            a, b, c = line
            if name == 'top' and b > 0:
                a, b, c = -a, -b, -c
            elif name == 'bottom' and b < 0:
                a, b, c = -a, -b, -c
            elif name == 'left' and a > 0:
                a, b, c = -a, -b, -c
            elif name == 'right' and a < 0:
                a, b, c = -a, -b, -c
            lines[name] = (a, b, c)

    if len(lines) < 3:
        return None

    lines = _iterative_line_fit(lines, points, _cfg.quadrant.fit_iterations)

    if len(lines) < 3:
        return None

    off = _cfg.quadrant.missing_edge_offset
    if len(lines) == 3:
        if 'top' not in lines and 'bottom' in lines:
            a, b, c = lines['bottom']
            lines['top'] = (a, b, c - off)
        elif 'bottom' not in lines and 'top' in lines:
            a, b, c = lines['top']
            lines['bottom'] = (a, b, c + off)
        elif 'left' not in lines and 'right' in lines:
            a, b, c = lines['right']
            lines['left'] = (a, b, c - off)
        elif 'right' not in lines and 'left' in lines:
            a, b, c = lines['left']
            lines['right'] = (a, b, c + off)

    if len(lines) < 4:
        return None

    order = ['top', 'right', 'bottom', 'left']
    corners = []
    for i in range(4):
        name_a, name_b = order[i], order[(i + 1) % 4]
        if name_a not in lines or name_b not in lines:
            return None
        a1, b1, c1 = lines[name_a]
        a2, b2, c2 = lines[name_b]
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-6:
            return None
        px = (b1 * c2 - b2 * c1) / det
        py = (a2 * c1 - a1 * c2) / det
        corners.append([px, py])

    quad = np.array(corners, dtype=np.float32)
    if not cv2.isContourConvex(quad.reshape(4, 1, 2)):
        return None
    return quad


def _ransac_line(points: np.ndarray) -> tuple | None:
    """对一组点做 RANSAC 直线拟合，返回 (a, b, c) 使得 ax+by+c=0。"""
    if len(points) < _cfg.ransac.min_points:
        return None
    best_line = None
    best_inliers = 0
    for _ in range(_cfg.ransac.iterations):
        idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        if np.linalg.norm(p2 - p1) < _cfg.ransac.min_pt_distance:
            continue
        a = p1[1] - p2[1]
        b = p2[0] - p1[0]
        c = p1[0] * p2[1] - p2[0] * p1[1]
        norm = np.sqrt(a * a + b * b)
        if norm < 1e-6:
            continue
        a, b, c = a / norm, b / norm, c / norm
        dists = np.abs(a * points[:, 0] + b * points[:, 1] + c)
        inliers = np.sum(dists < _cfg.ransac.inlier_threshold)
        if inliers > best_inliers:
            best_inliers = inliers
            best_line = (a, b, c)
    return best_line


def _iterative_line_fit(lines: dict, points: np.ndarray,
                         iterations: int = 3) -> dict:
    """迭代精修：将每个点分配到最近的线，重新拟合，重复 N 次。"""
    if len(lines) < 2:
        return lines
    for _ in range(iterations):
        assigned = {name: [] for name in lines}
        for pt in points:
            best_name, best_dist = None, float('inf')
            for name, (a, b, c) in lines.items():
                dist = abs(a * pt[0] + b * pt[1] + c)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
            if best_name is not None:
                assigned[best_name].append(pt)
        new_lines = {}
        for name, pts in assigned.items():
            if len(pts) < _cfg.ransac.min_points:
                new_lines[name] = lines[name]
            else:
                pts_arr = np.array(pts, dtype=np.float32)
                new_line = _ransac_line(pts_arr)
                if new_line is not None:
                    a, b, c = new_line
                    old_a, old_b, old_c = lines[name]
                    if a * old_a + b * old_b < 0:
                        a, b, c = -a, -b, -c
                    new_lines[name] = (a, b, c)
                else:
                    new_lines[name] = lines[name]
        lines = new_lines
    return lines


def _refine_corners_on_edges(quad: np.ndarray, gray: np.ndarray,
                              w: int, h: int) -> np.ndarray:
    """沿每条边采样多个点，分别搜索法线方向最强梯度，微调角点位置。"""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    refined = quad.copy()
    for i in range(4):
        p1 = refined[i]
        p2 = refined[(i + 1) % 4]
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < _cfg.refine.min_edge_len:
            continue
        edge_dir = edge_vec / edge_len
        normal = np.array([-edge_dir[1], edge_dir[0]])
        shifts = []
        for t in np.linspace(_cfg.refine.sample_start, _cfg.refine.sample_end, _cfg.refine.sample_count):
            sample_pt = p1 + edge_vec * t
            best_shift = 0
            best_grad = 0
            for s in range(_cfg.refine.search_start, _cfg.refine.search_end, _cfg.refine.search_step):
                sp = sample_pt + normal * s
                m = _cfg.refine.search_margin
                sx, sy = int(np.clip(sp[0], m, w - 1 - m)), int(np.clip(sp[1], m, h - 1 - m))
                g = grad_mag[sy - 1:sy + 2, sx - 1:sx + 2].mean()
                if g > best_grad:
                    best_grad = g
                    best_shift = s
            shifts.append(best_shift)
        if shifts:
            med_shift = np.median(shifts)
            if abs(med_shift) >= 1:
                refined[i] = refined[i] + normal * med_shift
                refined[(i + 1) % 4] = refined[(i + 1) % 4] + normal * med_shift
    m = _cfg.refine.search_margin
    refined[:, 0] = np.clip(refined[:, 0], m - 1, w - m)
    refined[:, 1] = np.clip(refined[:, 1], m - 1, h - m)
    return refined


def _ransac_fit_quad(points: np.ndarray, w: int, h: int) -> np.ndarray | None:
    """纯 RANSAC 同时拟合 4 条直线，计算交点得到四边形角点（备用）。"""
    best_quad, best_score = None, -9999
    for _ in range(_cfg.ransac_quad.iterations):
        if len(points) < 8:
            break
        n_pts = len(points)
        lines = []
        for _ in range(4):
            idx = np.random.choice(n_pts, 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]
            if np.linalg.norm(p2 - p1) < _cfg.ransac_quad.min_pt_distance:
                continue
            a = p1[1] - p2[1]
            b = p2[0] - p1[0]
            c = p1[0] * p2[1] - p2[0] * p1[1]
            norm = np.sqrt(a * a + b * b)
            if norm < 1e-6:
                continue
            a, b, c = a / norm, b / norm, c / norm
            cx_img, cy_img = w / 2, h / 2
            if a * cx_img + b * cy_img + c > 0:
                a, b, c = -a, -b, -c
            lines.append((a, b, c))
        if len(lines) < 4:
            continue
        corners = []
        for i in range(4):
            for j in range(i + 1, 4):
                a1, b1, c1 = lines[i]
                a2, b2, c2 = lines[j]
                det = a1 * b2 - a2 * b1
                if abs(det) < 1e-6:
                    continue
                px = (b1 * c2 - b2 * c1) / det
                py = (a2 * c1 - a1 * c2) / det
                m = _cfg.ransac_quad.outlier_margin
                if -m < px < w + m and -m < py < h + m:
                    corners.append((px, py))
        if len(corners) < 4:
            continue
        corners = np.array(corners, dtype=np.float32)
        hull = cv2.convexHull(corners.reshape(-1, 1, 2))
        if hull is None or len(hull) < 4:
            continue
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.05 * peri, True)
        if len(approx) != 4:
            continue
        quad = approx.reshape(4, 2).astype(np.float32)
        if not cv2.isContourConvex(quad.reshape(4, 1, 2)):
            continue
        s = _score_quad(quad, w, h)
        if s > best_score:
            best_score = s
            best_quad = quad
    return best_quad


# ---------------------------------------------------------------------------
# 策略 1：Sobel 固定阈值边缘检测
# ---------------------------------------------------------------------------

def _sobel_fixed_edge(bgr: np.ndarray, min_area: float) -> np.ndarray | None:
    """Sobel X+Y 固定阈值边缘检测：先 BGR 模糊去噪，再灰度化 + Sobel。"""
    h, w = bgr.shape[:2]
    border = _cfg.detection.border_size
    padded = cv2.copyMakeBorder(bgr, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    bk = _cfg.sobel_fixed.blur_kernel
    blurred = cv2.GaussianBlur(padded, (bk, bk), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    scale = _cfg.sobel_fixed.sobel_scale
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.uint8(np.absolute(sobel_x) * scale)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.uint8(np.absolute(sobel_y) * scale)
    edge = cv2.bitwise_or(sobel_x, sobel_y)
    for thresh_val in _cfg.sobel_fixed.thresholds:
        _, edge_binary = cv2.threshold(edge, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edge_binary, kernel, iterations=_cfg.sobel_fixed.dilate_iter)
        contours = _find_contours(dilated)
        if not contours:
            continue
        contours = _sort_by_area(contours)
        for cnt in contours[:_cfg.sobel_fixed.max_contours]:
            if cv2.contourArea(cnt) < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in _cfg.extract.contour_eps_values:
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
# 策略 2：亮度区域分割

def _find_by_bright_region(gray: np.ndarray, min_area: float) -> np.ndarray | None:
    """将纸张作为图像中最亮的连通区域来检测，使用中心参考阈值。"""
    h, w = gray.shape[:2]

    # 采样图像中心 1/4 区域作为纸张亮度参考
    ch, cw = h // 4, w // 4
    center_region = gray[ch:3 * ch, cw:3 * cw]
    center_mean = float(center_region.mean())
    center_std = float(center_region.std())

    # 方法 A：基于中心亮度（中心区域通常就是纸张）
    for n_std in _cfg.bright.n_std_values:
        thresh_val = center_mean - n_std * center_std
        if thresh_val < 10 or thresh_val > 248:
            continue

        quad = _bright_region_from_threshold(gray, int(thresh_val), min_area, h, w)
        if quad is not None:
            return quad

    # 方法 B：基于 Otsu 阈值（整体前景/背景分离）
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for offset in _cfg.bright.otsu_offsets:
        tv = otsu_val + offset
        if 10 <= tv <= 248:
            quad = _bright_region_from_threshold(gray, tv, min_area, h, w)
            if quad is not None:
                return quad

    # 方法 C：百分位阈值（回退）
    for pct in _cfg.bright.percentiles:
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
    border = _cfg.detection.border_size
    padded = cv2.copyMakeBorder(gray, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=0)

    _, binary = cv2.threshold(padded, thresh_val, 255, cv2.THRESH_BINARY)

    k = _cfg.bright.morph_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=_cfg.bright.morph_close_iter)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=_cfg.bright.morph_open_iter)

    contours = _find_contours(opened)
    if not contours:
        return None

    contours = _sort_by_area(contours)
    for cnt in contours[:_cfg.bright.max_contours]:
        if cv2.contourArea(cnt) < min_area:
            continue

        # 方法 A：直接用轮廓逼近（比 convexHull 更贴近实际纸张边界）
        peri = cv2.arcLength(cnt, True)
        for eps in _cfg.extract.contour_eps_values:
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

def _canny_sweep(gray: np.ndarray, min_area: float,
                  canny_low: int | None = None,
                  canny_high: int | None = None) -> np.ndarray | None:
    h, w = gray.shape[:2]
    best_quad, best_score = None, -9999

    border = _cfg.detection.border_size
    padded = cv2.copyMakeBorder(gray, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=0)

    if canny_low is not None and canny_high is not None:
        threshold_pairs = [(canny_low, canny_high)]
    else:
        threshold_pairs = []
        for sigma in _cfg.canny_sweep.sigma_values:
            threshold_pairs.append(auto_canny(padded, sigma))

    for low, high in threshold_pairs:
        for bk in _cfg.canny_sweep.blur_kernels:
            blurred = cv2.GaussianBlur(padded, (bk, bk), 0)
            edges = cv2.Canny(blurred, low, high)
            for dil_iter in _cfg.canny_sweep.dilate_iters:
                dk = _cfg.canny_sweep.dilate_kernel
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dk, dk))
                dilated = cv2.dilate(edges, kernel, iterations=dil_iter)

                contours = _find_contours(dilated)
                contours = _sort_by_area(contours)

                for cnt in contours[:_cfg.canny_sweep.max_contours]:
                    if cv2.contourArea(cnt) < min_area:
                        continue
                    peri = cv2.arcLength(cnt, True)
                    for eps in _cfg.extract.contour_eps_values:
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
    border = _cfg.detection.border_size
    padded = cv2.copyMakeBorder(gray, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=0)

    for bs in _cfg.adaptive.block_sizes:
        if bs % 2 == 0:
            bs += 1
        th = cv2.adaptiveThreshold(padded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, bs, _cfg.adaptive.c_value)

        k = _cfg.adaptive.morph_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=_cfg.adaptive.morph_close_iter)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=_cfg.adaptive.morph_open_iter)

        contours = _find_contours(opened)
        contours = _sort_by_area(contours)

        for cnt in contours[:_cfg.adaptive.max_contours]:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in _cfg.extract.contour_eps_values:
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
    border = _cfg.detection.border_size
    padded = cv2.copyMakeBorder(gray, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=0)

    gx = cv2.Sobel(padded, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(padded, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = np.uint8(np.clip(mag, 0, 255))

    for thresh in _cfg.sobel_grad.thresholds:
        _, binary = cv2.threshold(mag, thresh, 255, cv2.THRESH_BINARY)

        dk = _cfg.sobel_grad.dilate_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dk, dk))
        dilated = cv2.dilate(binary, kernel, iterations=_cfg.sobel_grad.dilate_iter)

        contours = _find_contours(dilated)
        contours = _sort_by_area(contours)

        for cnt in contours[:_cfg.sobel_grad.max_contours]:
            if cv2.contourArea(cnt) < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in _cfg.extract.contour_eps_values:
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
    border = _cfg.detection.border_size
    padded = cv2.copyMakeBorder(bgr, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    hsv = cv2.cvtColor(padded, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(padded, cv2.COLOR_BGR2LAB)
    s_ch, v_ch = hsv[:, :, 1], hsv[:, :, 2]
    l_ch = lab[:, :, 0]

    p = _cfg.color_seg.mask_params
    masks = []
    # 按配置生成各 mask
    mp = p[0]; masks.append((s_ch < mp["s_max"]) & (v_ch > mp["v_min"]))
    mp = p[1]; masks.append((s_ch < mp["s_max"]) & (v_ch > mp["v_min"]))
    mp = p[2]; masks.append(s_ch < mp["s_max"])
    mp = p[3]; masks.append(l_ch > mp["l_min"])
    mp = p[4]; masks.append((s_ch < mp["s_max"]) & (v_ch > mp["v_min"]))

    for raw in masks:
        mask = raw.astype(np.uint8) * 255

        k = _cfg.color_seg.morph_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=_cfg.color_seg.morph_close_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=_cfg.color_seg.morph_open_iter)

        contours = _find_contours(mask)
        if not contours:
            continue
        contours = _sort_by_area(contours)

        for cnt in contours[:_cfg.color_seg.max_contours]:
            if cv2.contourArea(cnt) < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in _cfg.extract.contour_eps_values:
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

def _hough_lines(gray: np.ndarray,
                  canny_low: int | None = None,
                  canny_high: int | None = None) -> np.ndarray | None:
    h, w = gray.shape[:2]
    border = _cfg.detection.border_size
    padded = cv2.copyMakeBorder(gray, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=0)
    ph, pw = padded.shape[:2]

    if canny_low is not None and canny_high is not None:
        threshold_pairs = [(canny_low, canny_high)]
    else:
        threshold_pairs = list(_cfg.hough.canny_thresholds)

    for low, high in threshold_pairs:
        edges = cv2.Canny(padded, low, high)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=_cfg.hough.hough_threshold,
                                minLineLength=min(pw, ph) // _cfg.hough.min_line_div,
                                maxLineGap=_cfg.hough.max_line_gap)
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
        for eps in _cfg.extract.contour_eps_values:
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
# 安全网回退：所有策略失败时，用最简单的 Canny + 最大四边形兜底
# ---------------------------------------------------------------------------

def _fallback_detect(gray: np.ndarray, min_area: float,
                     w: int, h: int) -> np.ndarray | None:
    """简单 Canny + 最大凸四边形检测，放宽评分要求作为最后兜底。"""
    border = _cfg.detection.border_size
    padded = cv2.copyMakeBorder(gray, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=0)
    low, high = auto_canny(padded)
    edges = cv2.Canny(padded, low, high)
    fk = _cfg.fallback.dilate_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (fk, fk))
    dilated = cv2.dilate(edges, kernel, iterations=_cfg.fallback.dilate_iter)
    contours = _find_contours(dilated)
    if not contours:
        return None

    for cnt in _sort_by_area(contours)[:_cfg.fallback.max_contours]:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        for eps in _cfg.extract.contour_eps_values:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            quad = _extract_quad(approx)
            if quad is None:
                continue
            if not cv2.isContourConvex(quad.reshape(4, 1, 2)):
                continue
            quad_adj = quad - border
            quad_adj[:, 0] = np.clip(quad_adj[:, 0], 0, w - 1)
            quad_adj[:, 1] = np.clip(quad_adj[:, 1], 0, h - 1)
            s = _score_quad(quad_adj, w, h)
            if s > _cfg.fallback.min_score:
                return quad_adj

    return None


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def find_document_contour(image: np.ndarray,
                          edges: np.ndarray | None = None,
                          bgr: np.ndarray | None = None,
                          canny_low: int | None = None,
                          canny_high: int | None = None) -> np.ndarray | None:
    """多策略文档角点检测。

    按顺序尝试：亮度区域 → 多参数 Canny → 自适应阈值 → Sobel →
    颜色分割 → Hough 直线。返回 (4, 2) 的角点数组，或 None。
    """
    np.random.seed(12)  # 固定随机种子，确保 RANSAC 结果可重复
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bgr_input = bgr if bgr is not None else (image if len(image.shape) == 3 else None)
    h, w = gray.shape[:2]
    min_area = (h * w) * _cfg.detection.min_area_ratio

    strategies = [
        ("radial",       lambda: _radial_search(gray, bgr_input, min_area)),
        ("sobel_fixed",  lambda: _sobel_fixed_edge(bgr_input, min_area) if bgr_input is not None else None),
        ("bright",       lambda: _find_by_bright_region(gray, min_area)),
        ("canny",        lambda: _canny_sweep(gray, min_area, canny_low, canny_high)),
        ("adaptive",     lambda: _adaptive_threshold(gray, min_area)),
        ("sobel",        lambda: _sobel_gradient(gray, min_area)),
        ("color",        lambda: _color_segmentation(bgr_input, min_area) if bgr_input is not None else None),
        ("hough",        lambda: _hough_lines(gray, canny_low, canny_high)),
    ]

    best_quad, best_score = None, -9999

    for _name, strategy in strategies:
        try:
            quad = strategy()
            if quad is not None:
                quad = _refine_corners_on_edges(quad, gray, w, h)
                s = _score_quad(quad, w, h)
                if s > best_score:
                    best_score = s
                    best_quad = quad
                    if s > _cfg.scoring.early_break_score:
                        break
        except Exception:
            continue

    if best_quad is None:
        best_quad = _fallback_detect(gray, min_area, w, h)

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


def draw_document_contour(image: np.ndarray, corners: np.ndarray,
                          title: str = "Document Detection") -> np.ndarray:
    """在图像上绘制检测到的文档轮廓和角点，用于调试和可视化。

    返回一份带标注的副本，不修改原图。
    """
    out = image.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    pts = corners.reshape(1, 4, 2).astype(np.int32)
    cv2.polylines(out, [pts], True, (0, 255, 0), 3)

    labels = ["TL", "TR", "BR", "BL"]
    for i, (x, y) in enumerate(corners.astype(np.int32)):
        cv2.circle(out, (x, y), 8, (255, 0, 0), cv2.FILLED)
        cv2.putText(out, labels[i], (x + 12, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return out


def draw_corners_on_canny(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """在 Canny 边缘图上叠加检测到的角点和四边形，用于调试中间过程。

    返回 BGR 彩色图像，黑色背景 + 白色边缘 + 绿色四边形 + 蓝色角点。
    """
    low, high = auto_canny(gray)
    edges = cv2.Canny(gray, low, high)
    out = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    pts = corners.reshape(1, 4, 2).astype(np.int32)
    cv2.polylines(out, [pts], True, (0, 255, 0), 2)

    labels = ["TL", "TR", "BR", "BL"]
    for i, (x, y) in enumerate(corners.astype(np.int32)):
        cv2.circle(out, (x, y), 6, (255, 0, 0), cv2.FILLED)
        cv2.putText(out, labels[i], (x + 10, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return out
