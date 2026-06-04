"""
文档角点检测模块 — 8 种级联策略 + 四边形评分系统
====================================================

这是项目最核心的模块，负责从图像中定位文档的四个角点。
因为单一的边缘检测方法无法适应所有场景（光照、背景、纸张颜色等），
所以设计了 8 种互补的检测策略按优先级级联执行，每种子策略返回候选四边形，
由评分系统选出最优结果。

评分维度（见 _score_quad）:
- 面积占比（纸张应占画面一定比例）
- 对边平行度（透视变形后对边仍大致平行）
- 内角合理性（四个角应在合理角度范围）
- 矩形度（轮廓面积/外接矩形面积）
- 贴边惩罚（角点贴到图像边缘说明检测可能出错）

策略执行顺序:
1. 射线搜索（_radial_search）— 最优先，不依赖完整闭合边缘
2. Sobel 固定阈值（_sobel_fixed_edge）
3. 亮度区域分割（_find_by_bright_region）
4. Canny 多参数扫描（_canny_sweep）
5. 自适应阈值（_adaptive_threshold）
6. Sobel 梯度（_sobel_gradient）
7. 颜色分割（_color_segmentation）
8. Hough 直线检测（_hough_lines）
9. 安全网回退（_fallback_detect）
"""

import cv2
import numpy as np

from .config import default_config as _cfg


# ---------------------------------------------------------------------------
# 基础工具 — Canny 阈值自动计算、轮廓提取、排序
# ---------------------------------------------------------------------------

def auto_canny(image: np.ndarray, sigma: float = 0.33) -> tuple[int, int]:
    """基于图像中位亮度自动计算 Canny 双阈值。

    算法：以图像中位数为基准，低阈值 = max(0, (1 - sigma) × 中位数)，
    高阈值 = min(255, (1 + sigma) × 中位数)。
    sigma 越小双阈值越窄，边缘越敏感。

    Args:
        image: 灰度图
        sigma: 阈值缩放因子，默认 0.33

    Returns:
        (低阈值, 高阈值)
    """
    median = np.median(image)
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))
    return low, high


def canny_edge(image: np.ndarray, low: int | None = None, high: int | None = None) -> np.ndarray:
    """执行 Canny 边缘检测，支持自动或手动阈值。

    Args:
        image: 灰度图
        low: Canny 低阈值，None 则自动计算
        high: Canny 高阈值，None 则自动计算

    Returns:
        边缘图（二值图，边缘为白）
    """
    if low is None or high is None:
        low, high = auto_canny(image)
    return cv2.Canny(image, low, high)


def _find_contours(edges: np.ndarray) -> list:
    """从边缘二值图中查找所有外轮廓。

    Args:
        edges: Canny 或其它边缘检测输出的二值图

    Returns:
        轮廓列表，每个轮廓是 N×1×2 的 ndarray
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _sort_by_area(contours: list) -> list:
    """按轮廓面积从大到小排序。

    Args:
        contours: 轮廓列表

    Returns:
        降序排列的轮廓列表
    """
    return sorted(contours, key=cv2.contourArea, reverse=True)


# ---------------------------------------------------------------------------
# 四边形提取 & 评分
# ---------------------------------------------------------------------------

def _extract_quad(approx: np.ndarray) -> np.ndarray | None:
    """从多边形近似结果中提取四边形（4 个顶点）。

    三种情况处理：
    1. 已经是四边形（len==4）→ 直接返回
    2. 多于 4 个点 → 取凸包，尝试多组 eps 参数逼近为四边形；
       如果都不行，用最小外接矩形（minAreaRect）兜底
    3. 少于 4 个点 → 无解，返回 None

    Args:
        approx: 多边形近似结果

    Returns:
        (4, 2) 的角点数组，或 None
    """
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
    """对候选四边形进行综合评分，分数越高说明越可能是纸张边界。

    评分维度（总分无上限，通常 0~100 之间）:
    1. 面积占比：面积 / 图像总面积，完美范围 20%~93%
    2. 对边平行度：上下边、左右边的长度比，越接近 1 越好
    3. 内角合理性：四个内角应在 45°~135° 之间
    4. 矩形度：轮廓面积 / 外接矩形面积，纸张通常接近矩形
    5. 贴边惩罚：角点贴到图像边缘说明可能检测错误

    拒绝条件（返回极低分）:
    - 最小边长 < 10px
    - 面积占比 < 6% 或 > 98%
    - 贴边角点 ≥ 3 个
    - 矩形度 < 0.5

    Args:
        quad: (4, 2) 的角点数组
        w: 图像宽度
        h: 图像高度

    Returns:
        综合评分（越高越好，负分表示该四边形不可用）
    """
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
    """[策略 0 — 射线搜索] 从图像中心发射射线，沿梯度最大处找边界。

    这是最优先的策略，因为它不依赖完整的闭合边缘。
    即使纸张边缘有断裂、阴影干扰，射线仍能找到梯度峰值。

    步骤：
    1. 高斯模糊降噪
    2. Sobel 计算梯度幅值图
    3. 从图像中心沿 360°（步长 1°）发射线
    4. 每条射线上找梯度响应最大的点作为边界候选
    5. 中位距离过滤离群点
    6. 象限分组 → RANSAC 直线拟合 → 求交点得到四边形
    7. 精修角点 → 评分

    Args:
        gray: 灰度图
        bgr: BGR 原图（仅用于传递给 refine）
        min_area: 最小面积阈值

    Returns:
        (4, 2) 的角点数组，或 None
    """
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
    """将边界点按相对中心的方位分 4 组，每组 RANSAC 拟合一条边，求交点得到四边形。

    分组逻辑（角度以图像中心为原点）:
    - right:  角度 [315°, 360°) ∪ [0°, 45°)
    - bottom: 角度 [45°, 135°)
    - left:   角度 [135°, 225°)
    - top:    角度 [225°, 315°)

    如果某组点数不足 5 个，跳过该边的拟合；
    如果至少 3 组有数据，用迭代重分配精修后求交点；
    如果只有 3 组，用最后一组的平行偏移补第 4 边。

    Args:
        points: 边界点集，形状 (N, 2)
        cx, cy: 图像中心坐标
        w, h: 图像尺寸

    Returns:
        (4, 2) 的角点数组，或 None
    """
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
    """对一组点做 RANSAC 直线拟合，返回 (a, b, c) 使得 ax + by + c = 0。

    随机采样 2 个点确定直线，计算所有点到直线的距离，
    统计 inlier（距离 < threshold）数量。
    迭代 N 次取 inlier 最多的直线。

    Args:
        points: 点集，形状 (N, 2)

    Returns:
        (a, b, c) 直线系数，归一化使得 √(a²+b²) = 1，或 None
    """
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
    """迭代精修边线：将每个点重新分配给最近的边，再重新拟合，重复 N 次。

    初始的象限分组可能把边界点分错组（比如右上角的点被分到 'top' 组），
    迭代过程逐步纠正，让每个点归属到离它最近的边。

    Args:
        lines: {name: (a, b, c)} 的字典，name 为 'top'/'right'/'bottom'/'left'
        points: 所有边界点
        iterations: 迭代轮数

    Returns:
        精修后的边线字典
    """
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
    """沿四边形边采样，在法线方向搜索最强梯度来精修角点位置。

    第一次逼近得到的角点可能不够精确，因为多边形的顶点不一定在真实的纸张边界上。
    这个方法沿每条边采样多个点，在每个点处沿法线方向搜索 Sobel 梯度最强的位置，
    用中位偏移量平移整条边，使角点更贴合真正的纸张边缘。

    Args:
        quad: (4, 2) 的角点数组
        gray: 灰度图
        w, h: 图像尺寸

    Returns:
        精修后的角点数组 (4, 2)
    """
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
    """[备用] 纯 RANSAC 同时拟合 4 条直线，求交点得四边形。

    与 _quadrant_fit_quad 不同，这个方法不依赖象限分组，
    直接从点集中随机采样 4 对点拟合 4 条直线，然后两两求交点。
    因为搜索空间大，迭代次数多，作为备用方案。

    Args:
        points: 边界点集
        w, h: 图像尺寸

    Returns:
        (4, 2) 角点数组，或 None
    """
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
    """[策略 1 — Sobel 固定阈值] 先模糊去噪 → Sobel 求梯度 → 固定阈值二值化 → 找四边形。

    与 Canny 相比，Sobel 固定阈值能保留更连续的边缘（因为不受 NMS 影响），
    适合纸张边界模糊或不连续的场景。

    步骤：
    1. BGR 图加黑边 padding
    2. 高斯模糊去噪
    3. Sobel X + Sobel Y 分别计算后合并
    4. 多组固定阈值尝试二值化
    5. 膨胀连接断开的边缘
    6. 找轮廓 → 多边形近似 → 四边形提取 → 评分

    Args:
        bgr: BGR 彩色图
        min_area: 最小面积阈值

    Returns:
        (4, 2) 角点数组，或 None
    """
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
    """[策略 2 — 亮度区域分割] 将纸张作为图像中最亮的区域来检测。

    基于一个简单观察：文档纸张通常是画面中最亮且面积最大的均匀区域。

    三种阈值策略依次尝试（任一种成功即返回）:
    方法 A — 中心参考：取图像中心 ¼ 区域的亮度均值和标准差，按 mean - n*std 做阈值
    方法 B — Otsu偏移：Otsu 全局阈值 ± 偏移量
    方法 C — 百分位：取全图第 55/60/65/70/75 百分位作为阈值

    每种阈值生成二值图 → 形态学闭运算（填孔）+ 开运算（去噪）→ 找轮廓 → 四边形提取

    Args:
        gray: 灰度图
        min_area: 最小面积阈值

    Returns:
        (4, 2) 的角点数组，或 None
    """
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
    """给定阈值，从二值图中提取亮度区域并逼近四边形。

    步骤：
    1. 灰度图加 padding
    2. 阈值二值化
    3. 形态学闭运算（填孔）+ 开运算（去噪）
    4. 找轮廓 → 按面积排序
    5. 方式 A：轮廓直接逼近四边形
    6. 方式 B：minAreaRect 最小外接矩形兜底

    Args:
        gray: 灰度图
        thresh_val: 二值化阈值
        min_area: 最小面积
        h, w: 图像尺寸

    Returns:
        (4, 2) 角点数组，或 None
    """
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
    """[策略 3 — Canny 多参数扫描] 多组参数遍历 Canny 检测，取最佳四边形。

    自动遍历 sigma/blur/dilate 参数组合，对每套参数做：
    高斯模糊 → Canny → 膨胀 → 找轮廓 → 四边形提取 → 评分

    保留所有组合中分数最高的四边形。

    Args:
        gray: 灰度图
        min_area: 最小面积阈值
        canny_low: 手动指定 Canny 低阈值（命令行参数传入）
        canny_high: 手动指定 Canny 高阈值

    Returns:
        (4, 2) 角点数组，或 None
    """
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
    """[策略 4 — 自适应阈值] 局部自适应二值化 → 四边形提取。

    对光照不均的场景特别有效（如部分区域有阴影）。
    尝试多个 block size（41/61/81/111/151），每块独立计算阈值。
    二值化后做形态学闭+开运算，然后提取四边形。

    Args:
        gray: 灰度图
        min_area: 最小面积阈值

    Returns:
        (4, 2) 角点数组，或 None
    """
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
    """[策略 5 — Sobel 梯度] 直接对 Sobel 梯度幅值阈值化 → 四边形提取。

    与 _sobel_fixed_edge 的区别：这里对灰度图做 Sobel，且直接用
    梯度幅值做阈值，没有先模糊 BGR。适合边缘锐利的场景。

    Args:
        gray: 灰度图
        min_area: 最小面积阈值

    Returns:
        (4, 2) 角点数组，或 None
    """
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
    """[策略 6 — 颜色分割] 在 HSV / LAB 空间通过颜色掩码提取纸张区域。

    通过 5 组不同松紧度的阈值在 HSV（色调/饱和度/明度）和 LAB（亮度）空间
    对图像做掩码，提取可能的纸张区域。特别适合：
    - 白色纸张在深色背景上
    - 彩色纸张在对比背景上
    - 普通亮度分割无法处理的场景

    Args:
        bgr: BGR 彩色图像
        min_area: 最小面积阈值

    Returns:
        (4, 2) 角点数组，或 None
    """
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
    """[策略 7 — Hough 直线检测] 用 Hough 变换找直线，从直线端点构造四边形。

    对于边缘断裂严重但直线段明显的场景（如强光下的文档），
    基于轮廓的方法可能找不到完整四边形。Hough 变换能检测出
    不连续的直线段，然后取所有直线端点的凸包来逼近四边形。

    Args:
        gray: 灰度图
        canny_low: Canny 低阈值
        canny_high: Canny 高阈值

    Returns:
        (4, 2) 角点数组，或 None
    """
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
    """[兜底 — 安全网] 所有策略都失败时的最终尝试。

    放宽评分门槛（min_score = -500），用最简单的 Canny + 轮廓检测
    取面积最大的近四边形作为结果。即使评分不高，也比完全没有结果好。

    同时将角点裁剪到图像边界内，防止越界。

    Args:
        gray: 灰度图
        min_area: 最小面积
        w, h: 图像尺寸

    Returns:
        (4, 2) 角点数组（质量可能不高），或 None
    """
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
    """文档角点检测主入口 — 8 种策略级联执行，返回最优四边形角点。

    执行流程：
    1. 依次执行 8 种检测策略（按优先级从高到低）
    2. 每个策略找到的四边形先经 _refine_corners_on_edges 精修
    3. 用 _score_quad 综合评分，保留当前最高分
    4. 如果最高分 > early_break_score（70分），提前退出，不再尝试后续策略
    5. 如果所有策略都失败，调用 _fallback_detect 兜底

    注意：`edges` 参数已废弃，仅保留是为了兼容性。

    Args:
        image: 灰度图（优先）或 BGR 图
        bgr: BGR 原图，用于依赖彩色信息的策略（颜色分割、Sobel 固定阈值）
        canny_low: 手动指定 Canny 低阈值（可选）
        canny_high: 手动指定 Canny 高阈值（可选）
        edges: 已废弃，保留仅为了接口兼容

    Returns:
        (4, 2) 的 float32 角点数组，未找到则返回 None
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
    """将四个角点按规范顺序排列：TL(左上) → TR(右上) → BR(右下) → BL(左下)。

    排序方法：
    - 左上角 = sum(x,y) 最小的点（坐标和最小）
    - 右下角 = sum(x,y) 最大的点（坐标和最大）
    - 右上角 = diff(y-x) 最小的点
    - 左下角 = diff(y-x) 最大的点

    这种排序是透视变换的前提条件，因为 getPerspectiveTransform 要求
    源点和目标点一一对应。

    Args:
        corners: (4, 2) 的角点数组，顺序任意

    Returns:
        (4, 2) 的排序后角点数组，顺序为 [TL, TR, BR, BL]
    """
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
    """在原图上绘制检测到的文档轮廓和角点标签（用于可视化/调试）。

    绿色四边形 + 蓝色实心圆标记角点 + TL/TR/BR/BL 文字标签。
    返回的是副本，不修改原图。

    Args:
        image: BGR 或灰度原图
        corners: (4, 2) 的角点数组
        title: 窗口标题（未使用，保留兼容）

    Returns:
        带标注的 BGR 图像副本
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
    """在 Canny 边缘图上叠加检测到的四边形和角点（调试用）。

    输出是 BGR 彩色图：
    - 黑色背景 + 白色边缘（Canny 结果）
    - 绿色四边形轮廓
    - 蓝色实心角点 + 黄色文字标签

    用于在 GUI 的「Canny 调试」Tab 中展示检测中间过程。

    Args:
        gray: 灰度图（用于计算 Canny 边缘）
        corners: (4, 2) 的角点数组

    Returns:
        BGR 彩色调试图
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
