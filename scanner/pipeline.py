"""
文档扫描管线 — DocScanner 核心类
===================================

串联整个处理流程：加载图像 → 缩放 → 灰度化 → CLAHE（可选）→
角点检测 → 透视变换 → A4 画布填充 → 图像增强 → 保存输出。

DocScanner 类提供三个入口方法:
- scan(image_path):     扫描单张图片文件
- scan_image(image):    扫描内存中的图像（供 GUI 调用）
- scan_batch(dir):      批量扫描整个文件夹
- scan_batch_to_pdf():  批量扫描并导出为 PDF
"""

from pathlib import Path

import cv2
import numpy as np

from . import preprocessing, detection, transform, enhance, io_utils
from .config import default_config as _cfg


class DocScanner:
    """文档扫描器：检测 → 透视矫正 → 图像增强。

    该类串联整个扫描管线，协调各子模块工作。
    支持单张图片和批量目录两种模式。

    Args:
        canny_low: Canny 低阈值（手动覆盖）
        canny_high: Canny 高阈值（手动覆盖）
        clahe: 是否启用 CLAHE 对比度增强
        sharpen: 是否启用锐化
        binarize: 是否启用二值化
        binarize_method: 二值化方法，"otsu" 或 "adaptive"
        binarize_block_size: 自适应二值化的块大小
        binarize_c: 自适应二值化的 C 值
        output_dir: 输出目录（命令行模式使用）
        denoise: 是否启用二值化后去噪
        denoise_kernel: 去噪核大小
    """

    def __init__(self, canny_low: int | None = None, canny_high: int | None = None,
                 clahe: bool = True, sharpen: bool = True,
                 binarize: bool = False, binarize_method: str = "otsu",
                 binarize_block_size: int | None = None,
                 binarize_c: int | None = None,
                 output_dir: str | Path | None = None,
                 denoise: bool = True, denoise_kernel: int | None = None):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.do_clahe = clahe
        self.do_sharpen = sharpen
        self.do_binarize = binarize
        self.binarize_method = binarize_method
        self.binarize_block_size = binarize_block_size
        self.binarize_c = binarize_c
        self.denoise = denoise
        self.denoise_kernel = denoise_kernel
        self.output_dir = Path(output_dir) if output_dir else None

    def scan_image(self, image: np.ndarray) -> dict:
        """扫描单张图像（内存中），返回各阶段结果。

        完整处理管线：
        1. 灰度化
        2. 可选 CLAHE 产生第二个灰度版本
        3. 两个灰度版本分别过角点检测，选最优
        4. 角点排序 → 透视变换
        5. A4 画布填充
        6. 图像增强（CLAHE / 锐化 / 二值化）

        Args:
            image: BGR 图像

        Returns:
            字典，包含:
            - success: bool
            - error: 错误信息（失败时）
            - original: 原图
            - gray: 灰度图
            - corners: (4, 2) 角点
            - canny_debug: Canny 调试图
            - annotated: 角点标注图
            - warped: 透视矫正 + A4 画布（显示用）
            - warped_raw: 原始尺寸的矫正图（PDF 导出用）
            - final: 最终增强结果
            - final_raw: 原始尺寸的增强结果（PDF 导出用）
            - binary: 二值图（如有）
            - 以及其他 enhance() 返回的中间结果
        """
        original = image.copy()
        gray_raw = preprocessing.grayscale(image)

        gray_versions = [gray_raw]
        if self.do_clahe:
            gray_clahe = cv2.createCLAHE(
                clipLimit=_cfg.pipeline.clahe_clip,
                tileGridSize=(_cfg.pipeline.clahe_tile, _cfg.pipeline.clahe_tile),
            ).apply(gray_raw)
            gray_versions.append(gray_clahe)

        best_corners = None
        best_score = -9999
        h, w = gray_raw.shape[:2]

        for gray in gray_versions:
            corners = detection.find_document_contour(
                gray, bgr=original,
                canny_low=self.canny_low, canny_high=self.canny_high,
            )
            if corners is None:
                continue
            s = detection._score_quad(corners, w, h)
            # 只要 raw 检测成功，就不启用 CLAHE 的结果。
            # CLAHE 虽然分数可能更高，但容易把内部纹理误认为边缘。
            if best_corners is not None and gray is not gray_raw:
                continue
            if s > best_score:
                best_score = s
                best_corners = corners

        if best_corners is None:
            return {"success": False, "error": "Could not detect document corners", "original": original}

        ordered = detection.order_corners(best_corners)
        canny_debug = detection.draw_corners_on_canny(gray_raw, ordered)
        annotated = detection.draw_document_contour(original, ordered)
        warped_raw = transform.perspective_transform(original, ordered)  # 原始尺寸
        warped = transform.pad_to_a4(warped_raw)                        # A4 画布版（显示用）

        enhanced = enhance.enhance(
            warped,
            do_clahe=self.do_clahe,
            do_sharpen=self.do_sharpen,
            do_binarize=self.do_binarize,
            binarize_method=self.binarize_method,
            binarize_block_size=self.binarize_block_size,
            binarize_c=self.binarize_c,
            denoise=self.denoise,
            denoise_kernel=self.denoise_kernel,
        )

        # 原始尺寸版本（PDF 导出用）
        enhanced_raw = enhance.enhance(
            warped_raw,
            do_clahe=self.do_clahe,
            do_sharpen=self.do_sharpen,
            do_binarize=self.do_binarize,
            binarize_method=self.binarize_method,
            binarize_block_size=self.binarize_block_size,
            binarize_c=self.binarize_c,
            denoise=self.denoise,
            denoise_kernel=self.denoise_kernel,
        )

        return {
            "success": True,
            "original": original,
            "gray": gray_raw,
            "corners": ordered,
            "canny_debug": canny_debug,
            "annotated": annotated,
            "warped": warped,
            "warped_raw": warped_raw,
            "final_raw": enhanced_raw.get("final", warped_raw),
            **enhanced,
        }

    def scan(self, image_path: str | Path) -> dict:
        """扫描单张图片文件。

        加载 → 大图缩放 → scan_image → 自动保存到 output_dir。

        Args:
            image_path: 图片文件路径

        Returns:
            与 scan_image 相同的结果字典
        """
        image = io_utils.load_image(image_path)
        image = preprocessing.resize_if_large(image)

        result = self.scan_image(image)
        result["path"] = Path(image_path)

        if self.output_dir and result["success"]:
            self._save_result(result)

        return result

    def scan_batch(self, input_dir: str | Path) -> list[dict]:
        """批量扫描目录中的所有图片。

        Args:
            input_dir: 输入目录路径

        Returns:
            结果字典列表，每个元素对应一张图片的扫描结果
        """
        paths = io_utils.collect_images(input_dir)
        results = []
        for p in paths:
            result = self.scan(p)
            results.append(result)
        return results

    def scan_batch_to_pdf(self, input_dir: str | Path, pdf_path: str | Path) -> list[dict]:
        """批量扫描目录中的图片并导出为一份 PDF。

        Args:
            input_dir: 输入目录路径
            pdf_path: 输出 PDF 文件路径

        Returns:
            结果字典列表
        """
        results = self.scan_batch(input_dir)
        output_paths = [r.get("saved_path") for r in results if r["success"] and "saved_path" in r]
        if output_paths:
            io_utils.save_as_pdf(output_paths, pdf_path)
        return results

    def _save_result(self, result: dict) -> None:
        """将单张扫描结果保存到 output_dir。

        保存文件命名规则: {原文件名}_{阶段}.jpg
        - warped: 透视矫正图
        - canny_debug: Canny 调试图
        - annotated: 角点标注图
        - binary: 二值图
        - scanned: 最终扫描结果

        Args:
            result: scan_image 的返回字典
        """
        out_dir = self.output_dir
        stem = result["path"].stem

        io_utils.save_image(result["warped"], out_dir / f"{stem}_warped.jpg")
        io_utils.save_image(result["canny_debug"], out_dir / f"{stem}_canny_debug.jpg")
        io_utils.save_image(result["annotated"], out_dir / f"{stem}_annotated.jpg")

        if "binary" in result and result["binary"] is not result["final"]:
            io_utils.save_image(result["binary"], out_dir / f"{stem}_binary.jpg")

        io_utils.save_image(result["final"], out_dir / f"{stem}_scanned.jpg")
        result["saved_path"] = str(out_dir / f"{stem}_scanned.jpg")
