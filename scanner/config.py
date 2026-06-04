"""
配置文件 —— 集中管理所有算法模块的可调参数
=============================================

采用 @dataclass + 嵌套聚合模式，所有参数集中定义在 ScannerConfig 这个顶层配置类中。
各模块通过 `from .config import default_config as _cfg` 引用，
例如 `_cfg.scoring.area_min_ratio`、`_cfg.detection.border_size`。

这样的设计便于：
1. 参数调优时只需改这一个文件，无需深入各算法代码
2. 可序列化保存多组参数配置（如"办公文档模式"、"名片模式"）
3. 所有参数有默认值和注释说明，降低使用门槛

分层结构:
ScannerConfig (顶层)
├── scoring        — 候选四边形评分
├── extract        — 轮廓近似为四边形
├── radial         — 射线搜索边界点
├── ransac         — RANSAC 直线拟合
├── quadrant       — 象限分组 + 线拟合
├── refine         — 角点精修
├── ransac_quad    — RANSAC 四边形拟合（备用）
├── sobel_fixed    — Sobel 固定阈值
├── bright         — 亮度区域分割
├── canny_sweep    — Canny 多参数扫描
├── adaptive       — 自适应阈值
├── sobel_grad     — Sobel 梯度
├── color_seg      — 颜色分割
├── hough          — Hough 直线检测
├── fallback       — 安全网回退
├── detection      — 检测主流程
├── pipeline       — 预处理管线
├── enhance        — 图像增强
├── transform      — 透视变换
├── preprocess     — 图像预处理
└── binarize       — 二值化
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 评分参数 — 对候选四边形进行打分，选出最可能为文档边界的四边形
# ---------------------------------------------------------------------------

@dataclass
class ScoreConfig:
    """候选四边形评分参数"""

    # ---- 面积过滤 ----
    min_side: int = 10                # 四边形最小边长 (px)，低于此值直接拒绝
    area_min_ratio: float = 0.06      # 最小面积占比：面积 / 图像总面积 < 此值 → 拒绝（滤除碎小轮廓）
    area_max_ratio: float = 0.98      # 最大面积占比：面积 / 图像总面积 > 此值 → 拒绝（滤除整张图轮廓）

    # ---- 面积得分 ----
    area_full_min: float = 0.20       # 面积满分下限：面积占比 >= 此值 → 面积项得满分
    area_full_max: float = 0.93       # 面积满分上限：面积占比 <= 此值 → 面积项得满分
    area_partial_min: float = 0.10    # 面积半分下限：面积占比 >= 此值 → 面积项得半分
    area_full_score: float = 25       # 面积满分分值
    area_partial_score: float = 10    # 面积半分分值

    # ---- 贴边惩罚 ----
    border_margin: int = 5            # 贴边判定距离 (px)：角点距图像边界 <= 此值视为贴边
    border_max_hits: int = 3          # 允许的最多贴边角点数，超过则每多一个扣分
    border_penalty: float = 12        # 每个超限贴边角的扣分值

    # ---- 平行度得分 ----
    parallel_weight: float = 15       # 平行度得分权重（对边夹角越小分越高）

    # ---- 内角得分 ----
    angle_full_min: float = 45        # 内角满分下限 (°)：内角 >= 此值且 <= angle_full_max 得满分
    angle_full_max: float = 135       # 内角满分上限 (°)
    angle_full_score: float = 6       # 每个满分角的得分
    angle_partial_min: float = 30     # 内角半分下限 (°)：内角在此区间得半分
    angle_partial_max: float = 150    # 内角半分上限 (°)
    angle_partial_score: float = 2    # 每个半分角的得分

    # ---- 矩形度得分 ----
    rect_ratio_reject: float = 0.50   # 矩形度阈值：四边形面积 / 外接矩形面积 < 此值 → 拒绝
    rect_ratio_base: float = 0.65     # 矩形度评分基线：矩形度 >= 此值开始按比例给分
    rect_ratio_weight: float = 40     # 矩形度评分权重

    # ---- 搜索策略 ----
    early_break_score: float = 70     # 多策略循环中的提前退出阈值：当前最高分 > 此值则不再尝试后续策略


# ---------------------------------------------------------------------------
# 四边形提取 — 轮廓近似为四边形的 eps 参数组合
# ---------------------------------------------------------------------------

@dataclass
class ExtractConfig:
    """从轮廓提取四边形的参数"""

    eps_values: tuple[float, ...] = (0.02, 0.05, 0.08, 0.12)
    """轮廓近似 eps 值列表，为轮廓周长的比例，值越小越接近原始轮廓"""

    contour_eps_values: tuple[float, ...] = (0.01, 0.02, 0.04, 0.06, 0.10)
    """轮廓合并前的近似 eps 值列表，用于简化轮廓后再合并"""


# ---------------------------------------------------------------------------
# 射线搜索 — 从图像中心向外发射射线，寻找文档边界点
# ---------------------------------------------------------------------------

@dataclass
class RadialConfig:
    """射线搜索边界点参数"""

    blur_k_min: int = 11              # 高斯模糊核的最小值 (px)
    blur_k_div: int = 50              # 模糊核计算除数：k = min(w,h) // 此值，然后 clamp 到 >= blur_k_min
    angle_step: int = 1               # 射线角度步长 (°)，值越小射线越密
    step_start: int = 10              # 射线起始步数（从中心向外搜索的起始距离）
    step_inc: int = 1                 # 射线步进增量（每次外扩的像素数）
    min_boundary_pts: int = 30        # 最小边界点数：有效边界点不足此值 → 放弃该策略
    min_median_dist: int = 20         # 最小中位距离 (px)：边界点中位距离过小 → 放弃（说明图像太小或无边界）
    dist_lower_ratio: float = 0.25    # 离群点过滤下限系数：距离 < 中位数 * 此值 → 剔除
    dist_upper_ratio: float = 2.5     # 离群点过滤上限系数：距离 > 中位数 * 此值 → 剔除
    min_filtered_pts: int = 20        # 过滤后最小点数：不足 → 放弃


# ---------------------------------------------------------------------------
# RANSAC 直线拟合 — 对边界点集用 RANSAC 拟合四条直线
# ---------------------------------------------------------------------------

@dataclass
class RansacConfig:
    """RANSAC 直线拟合参数"""

    min_points: int = 5               # 最小点数：输入点集不足此值 → 放弃拟合
    iterations: int = 50              # RANSAC 迭代次数
    min_pt_distance: float = 15       # 最小点间距 (px)：两点距离小于此值 → 跳过（避免重复采样）
    inlier_threshold: float = 4       # inlier 判定距离 (px)：点到直线的距离 <= 此值视为 inlier


# ---------------------------------------------------------------------------
# 象限分组 + 线拟合 — 将边界点按象限分组，每组拟合一条边
# ---------------------------------------------------------------------------

@dataclass
class QuadrantConfig:
    """象限分组与边线拟合参数"""

    angles: tuple[float, ...] = (45, 135, 225, 315)
    """象限角度分界 (°)：将 360° 按这些角度划分为四个象限"""

    min_group_pts: int = 5            # 每组最小点数：某象限点数不足 → 跳过该边拟合
    missing_edge_offset: float = 200  # 缺边偏移量 (px)：当某象限无点时，用此偏移量构造虚拟边
    fit_iterations: int = 3           # 迭代拟合次数：拟合后剔除离群点再重新拟合的轮数


# ---------------------------------------------------------------------------
# 角点精修 — 沿四边形边采样，在法线方向搜索精确边界来精修角点
# ---------------------------------------------------------------------------

@dataclass
class RefineConfig:
    """角点精修参数"""

    min_edge_len: float = 10          # 最小边长 (px)：边过短 → 跳过该边的精修
    sample_start: float = 0.15        # 采样起始位置（边长比例），避免角落处采样
    sample_end: float = 0.85          # 采样结束位置（边长比例）
    sample_count: int = 5             # 每条边的采样点数
    search_start: int = -20           # 法线搜索起点 (px)，负值表示向四边形内侧搜索
    search_end: int = 21              # 法线搜索终点 (px)，正值表示向四边形外侧搜索
    search_step: int = 2              # 法线搜索步长 (px)
    search_margin: int = 3            # 角点距图像边缘的最小距离 (px)，避免角点落在图外


# ---------------------------------------------------------------------------
# RANSAC 四边形拟合（备用）— 当主流程未找到合适四边形时的备选方案
# ---------------------------------------------------------------------------

@dataclass
class RansacQuadConfig:
    """RANSAC 直接拟合四边形参数（备用策略）"""

    iterations: int = 70              # RANSAC 迭代次数
    min_pt_distance: float = 10       # 最小点间距 (px)：两点距离小于此值 → 跳过
    outlier_margin: int = 50          # 交点可超出图像边缘的像素数，允许边线延伸到图外再求交点


# ---------------------------------------------------------------------------
# Sobel 固定阈值 — 使用固定阈值对 Sobel 边缘图二值化，提取轮廓
# ---------------------------------------------------------------------------

@dataclass
class SobelFixedConfig:
    """Sobel 固定阈值边缘检测参数"""

    sobel_scale: float = 1.8          # Sobel 响应放大系数：梯度幅值 * 此值后再阈值化
    thresholds: tuple[int, ...] = (30, 40, 50, 60, 70)
    """二值化阈值列表：多个阈值分别尝试，取最佳结果"""
    blur_kernel: int = 3              # 高斯模糊核大小 (px)
    dilate_iter: int = 6              # 膨胀迭代次数：连接断开的边缘
    max_contours: int = 5             # 最多保留的候选轮廓数


# ---------------------------------------------------------------------------
# 亮度区域分割 — 基于灰度直方图统计，提取亮度显著高于背景的区域
# ---------------------------------------------------------------------------

@dataclass
class BrightConfig:
    """亮度区域分割参数"""

    n_std_values: tuple[float, ...] = (0.8, 1.0, 1.3, 1.6, 2.0, 2.5)
    """标准差倍数列表：阈值 = mean + n_std * std，高于此值的像素视为亮区"""

    otsu_offsets: tuple[int, ...] = (0, 10, 20, -10, -20)
    """Otsu 阈值偏移量列表：在 Otsu 自动阈值基础上加减的灰度值"""

    percentiles: tuple[int, ...] = (55, 60, 65, 70, 75)
    """百分位数列表：取图像第 N 百分位数作为阈值"""

    morph_kernel: int = 5             # 形态学操作核大小 (px)
    morph_close_iter: int = 3         # 闭运算迭代次数：填充区域内部孔洞
    morph_open_iter: int = 7          # 开运算迭代次数：去除小噪点
    max_contours: int = 3             # 最多保留的候选轮廓数


# ---------------------------------------------------------------------------
# Canny 多参数扫描 — 用多组 sigma/blur/dilate 参数组合进行 Canny 边缘检测
# ---------------------------------------------------------------------------

@dataclass
class CannySweepConfig:
    """Canny 边缘检测多参数扫描配置"""

    sigma_values: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.45, 0.50)
    """高斯 sigma 值列表：控制 Canny 的平滑程度，值越大边缘越少"""

    blur_kernels: tuple[int, ...] = (3, 5)
    """高斯模糊核大小列表 (px)"""

    dilate_iters: tuple[int, ...] = (2, 3, 4)
    """膨胀迭代次数列表"""

    dilate_kernel: int = 3            # 膨胀操作核大小 (px)
    max_contours: int = 8             # 最多保留的候选轮廓数


# ---------------------------------------------------------------------------
# 自适应阈值 — 对图像局部区域使用自适应阈值二值化
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveConfig:
    """自适应阈值二值化参数"""

    block_sizes: tuple[int, ...] = (41, 61, 81, 111, 151)
    """自适应阈值块大小列表 (px)：每个块内独立计算阈值，必须为奇数"""

    c_value: int = 2                  # 从局部均值中减去的常数，值越大二值化越苛刻
    morph_kernel: int = 5             # 形态学操作核大小 (px)
    morph_close_iter: int = 3         # 闭运算迭代次数
    morph_open_iter: int = 4          # 开运算迭代次数
    max_contours: int = 5             # 最多保留的候选轮廓数


# ---------------------------------------------------------------------------
# Sobel 梯度 — 用 Sobel 算子计算梯度幅值后阈值化
# ---------------------------------------------------------------------------

@dataclass
class SobelGradConfig:
    """Sobel 梯度幅值检测参数"""

    thresholds: tuple[int, ...] = (25, 40, 60, 80)
    """二值化阈值列表：梯度幅值 > 阈值 → 边缘，多个阈值分别尝试"""

    dilate_kernel: int = 3            # 膨胀操作核大小 (px)
    dilate_iter: int = 3              # 膨胀迭代次数
    max_contours: int = 5             # 最多保留的候选轮廓数


# ---------------------------------------------------------------------------
# 颜色分割 — 在 HSV/LAB 空间对文档区域进行颜色掩码分割
# ---------------------------------------------------------------------------

@dataclass
class ColorSegConfig:
    """颜色分割参数（HSV / LAB 空间）"""

    # 5 组 HSV / LAB 掩码阈值参数
    # 每个字典可包含：s_max, v_min, l_min, s_min, v_max, l_max
    # 未指定的键表示不限制该通道
    # LAB 亮度通道 (l_min / l_max) 用于检测白色文档
    # HSV 通道 (s_min / s_max, v_min / v_max) 用于检测彩色/深色文档
    mask_params: tuple = (
        {"s_max": 30, "v_min": 130},    # 低饱和 + 高亮 → 白色纸张
        {"s_max": 40, "v_min": 150},    # 低饱和 + 更高亮 → 更严格的白色纸张
        {"s_max": 35},                   # 仅限制低饱和
        {"l_min": 170},                  # LAB 高亮度 → 白色区域
        {"s_max": 25, "v_min": 80},    # 更宽松的白纸阈值
    )
    morph_kernel: int = 5             # 形态学操作核大小 (px)
    morph_close_iter: int = 4         # 闭运算迭代次数
    morph_open_iter: int = 2          # 开运算迭代次数
    max_contours: int = 3             # 最多保留的候选轮廓数


# ---------------------------------------------------------------------------
# Hough 直线检测 — 用 Hough 变换检测直线，从直线交点构造四边形
# ---------------------------------------------------------------------------

@dataclass
class HoughConfig:
    """Hough 直线检测参数"""

    canny_thresholds: tuple[tuple[int, int], ...] = (
        (30, 90), (50, 150), (75, 200), (25, 75)
    )
    """Canny 双阈值列表：(low, high)，多组参数分别尝试"""

    hough_threshold: int = 80         # Hough 累加器阈值：交点投票数 >= 此值才视为直线
    min_line_div: int = 5             # 最小线段除数：minLineLength = min(w, h) // 此值
    max_line_gap: int = 60            # 最大线段间隙 (px)：间隙 <= 此值的共线线段会被合并


# ---------------------------------------------------------------------------
# 安全网回退 — 当所有策略都未找到有效四边形时的最终兜底方案
# ---------------------------------------------------------------------------

@dataclass
class FallbackConfig:
    """安全网回退参数（兜底策略）"""

    dilate_kernel: int = 3            # 膨胀操作核大小 (px)
    dilate_iter: int = 2              # 膨胀迭代次数
    max_contours: int = 10            # 最多保留的候选轮廓数
    min_score: float = -500           # 最低分数阈值：低于此值仍接受，确保兜底总能返回结果


# ---------------------------------------------------------------------------
# 检测主入口 — 控制检测流程的顶层参数
# ---------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    """检测主流程参数"""

    border_size: int = 25             # 图像 padding 黑边宽度 (px)，防止文档贴边时边缘检测失效
    min_area_ratio: float = 0.05      # 最小面积占比：min_area = w * h * 此值，低于此值的四边形直接丢弃


# ---------------------------------------------------------------------------
# Pipeline — 图像处理管线的预处理参数
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Pipeline 预处理参数"""

    clahe_clip: float = 2.0           # CLAHE 对比度裁剪限幅
    clahe_tile: int = 8               # CLAHE 网格分块大小 (px)


# ---------------------------------------------------------------------------
# 图像增强 — 对最终输出的文档图像进行锐化等后处理
# ---------------------------------------------------------------------------

@dataclass
class EnhanceConfig:
    """图像增强后处理参数"""

    sharpen_strength: float = 1.5     # 锐化强度：值越大锐化效果越强
    sharpen_sigma: int = 3            # 锐化高斯模糊 sigma：控制锐化半径


# ---------------------------------------------------------------------------
# 透视变换 — 将检测到的四边形矫正为矩形时的参数
# ---------------------------------------------------------------------------

@dataclass
class TransformConfig:
    """透视变换矫正参数"""

    a4_width: int = 210               # 目标输出宽度 (mm)，对应 A4 纸宽
    a4_height: int = 297              # 目标输出高度 (mm)，对应 A4 纸高
    ratio_tolerance: float = 0.02     # 宽高比容差：检测四边形的宽高比与 A4 比的偏差在此范围内视为匹配


# ---------------------------------------------------------------------------
# 二值化 — 自适应阈值参数与阴影抑制
# ---------------------------------------------------------------------------

@dataclass
class BinarizeConfig:
    """二值化参数（针对阴影、褶皱等不均匀光照）"""

    adaptive_block_size: int = 41       # Adaptive 块大小 (px)，越大越考虑全局，越小越敏感
    adaptive_c: int = 2                 # Adaptive C 值，从局部均值减去的常数，越大二值化越苛刻
    clahe_before_binarize: bool = True  # 二值化前先做 CLAHE 均衡亮度，对阴影褶皱有帮助
    denoise: bool = True                # 二值化后做形态学去噪（开运算去白点 + 闭运算填黑孔）
    denoise_kernel: int = 3             # 去噪核大小 (px)，越大去噪越强，但也会抹掉细笔画


# ---------------------------------------------------------------------------
# 预处理 — 输入图像的初始预处理参数
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """输入图像预处理参数"""

    max_dim: int = 1500               # 图像最大尺寸 (px)：长边超过此值则等比缩放，控制处理开销


# ---------------------------------------------------------------------------
# 总配置 — 聚合所有子配置的顶层配置类
# ---------------------------------------------------------------------------

@dataclass
class ScannerConfig:
    """文档扫描器总配置，聚合各模块的配置参数"""

    scoring: ScoreConfig = field(default_factory=ScoreConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    radial: RadialConfig = field(default_factory=RadialConfig)
    ransac: RansacConfig = field(default_factory=RansacConfig)
    quadrant: QuadrantConfig = field(default_factory=QuadrantConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    ransac_quad: RansacQuadConfig = field(default_factory=RansacQuadConfig)
    sobel_fixed: SobelFixedConfig = field(default_factory=SobelFixedConfig)
    bright: BrightConfig = field(default_factory=BrightConfig)
    canny_sweep: CannySweepConfig = field(default_factory=CannySweepConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    sobel_grad: SobelGradConfig = field(default_factory=SobelGradConfig)
    color_seg: ColorSegConfig = field(default_factory=ColorSegConfig)
    hough: HoughConfig = field(default_factory=HoughConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    enhance: EnhanceConfig = field(default_factory=EnhanceConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    binarize: BinarizeConfig = field(default_factory=BinarizeConfig)


# 全局默认配置实例
default_config = ScannerConfig()
