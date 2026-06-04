"""
scanner 包 — 智能文档扫描核心模块
====================================

提供完整的文档扫描管线：从图像加载 → 文档角点检测 → 透视变换矫正 →
图像增强 → 输出保存。纯经典计算机视觉算法，无机器学习依赖。

导出类:
    DocScanner: 扫描管线的入口类，提供 scan / scan_batch / scan_image 方法

导出函数:
    draw_document_contour: 在图像上绘制检测到的文档轮廓和角点标签
    draw_corners_on_canny: 在 Canny 边缘图上叠加文档四边形和角点
    order_corners: 将四个角点排序为 [TL, TR, BR, BL] 规范顺序
"""

from .pipeline import DocScanner
from .detection import draw_document_contour, draw_corners_on_canny, order_corners

__all__ = ["DocScanner", "draw_document_contour", "draw_corners_on_canny", "order_corners"]
