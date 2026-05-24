from pathlib import Path

import cv2
import numpy as np

from . import preprocessing, detection, transform, enhance, io_utils


class DocScanner:
    def __init__(self, canny_low: int | None = None, canny_high: int | None = None,
                 clahe: bool = True, sharpen: bool = True,
                 binarize: bool = False, binarize_method: str = "otsu",
                 output_dir: str | Path | None = None):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.do_clahe = clahe
        self.do_sharpen = sharpen
        self.do_binarize = binarize
        self.binarize_method = binarize_method
        self.output_dir = Path(output_dir) if output_dir else None

    def scan_image(self, image: np.ndarray) -> dict:
        original = image.copy()
        gray_raw = preprocessing.grayscale(image)

        gray_versions = [gray_raw]
        if self.do_clahe:
            gray_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_raw)
            gray_versions.append(gray_clahe)

        best_corners = None
        best_score = -9999
        h, w = gray_raw.shape[:2]

        for gray in gray_versions:
            corners = detection.find_document_contour(
                gray, bgr=original,
                canny_low=self.canny_low, canny_high=self.canny_high,
            )
            if corners is not None:
                s = detection._score_quad(corners, w, h)
                if s > best_score:
                    best_score = s
                    best_corners = corners

        if best_corners is None:
            return {"success": False, "error": "Could not detect document corners", "original": original}

        ordered = detection.order_corners(best_corners)
        canny_debug = detection.draw_corners_on_canny(gray_raw, ordered)
        warped = transform.perspective_transform(original, ordered)
        warped = transform.pad_to_a4(warped)

        enhanced = enhance.enhance(
            warped,
            do_clahe=self.do_clahe,
            do_sharpen=self.do_sharpen,
            do_binarize=self.do_binarize,
            binarize_method=self.binarize_method,
        )

        return {
            "success": True,
            "original": original,
            "gray": gray_raw,
            "corners": ordered,
            "canny_debug": canny_debug,
            "warped": warped,
            **enhanced,
        }

    def scan(self, image_path: str | Path) -> dict:
        image = io_utils.load_image(image_path)
        image = preprocessing.resize_if_large(image)

        result = self.scan_image(image)
        result["path"] = Path(image_path)

        if self.output_dir and result["success"]:
            self._save_result(result)

        return result

    def scan_batch(self, input_dir: str | Path) -> list[dict]:
        paths = io_utils.collect_images(input_dir)
        results = []
        for p in paths:
            result = self.scan(p)
            results.append(result)
        return results

    def scan_batch_to_pdf(self, input_dir: str | Path, pdf_path: str | Path) -> list[dict]:
        results = self.scan_batch(input_dir)
        output_paths = [r.get("saved_path") for r in results if r["success"] and "saved_path" in r]
        if output_paths:
            io_utils.save_as_pdf(output_paths, pdf_path)
        return results

    def _save_result(self, result: dict) -> None:
        out_dir = self.output_dir
        stem = result["path"].stem

        io_utils.save_image(result["warped"], out_dir / f"{stem}_warped.jpg")
        io_utils.save_image(result["canny_debug"], out_dir / f"{stem}_canny_debug.jpg")

        if "binary" in result and result["binary"] is not result["final"]:
            io_utils.save_image(result["binary"], out_dir / f"{stem}_binary.jpg")

        io_utils.save_image(result["final"], out_dir / f"{stem}_scanned.jpg")
        result["saved_path"] = str(out_dir / f"{stem}_scanned.jpg")
