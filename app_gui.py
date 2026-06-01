"""
窗口化文档扫描器 — Tkinter 桌面应用
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk

from scanner import DocScanner
from scanner import io_utils


class DocScannerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("智能文档扫描器")
        self.root.geometry("1280x800")
        self.root.minsize(960, 600)

        # ---------- 状态 ----------
        self.mode: str = "single"               # "single" | "batch"
        self.image_path: str | None = None       # 单张模式下的图片路径
        self.image_paths: list[Path] = []        # 批量模式下的图片路径列表
        self.current_index: int = 0              # 批量模式下当前索引
        self.batch_results: list[dict] = []      # 批量模式下的扫描结果
        self.original_bgr: np.ndarray | None = None  # 原始 BGR（未缩放）
        self.display_rgb: np.ndarray | None = None    # 用于显示的 RGB（已缩放）
        self.display_h, self.display_w = 0, 0

        self.result_cache = {}  # 缓存各阶段结果图
        self.scanner = DocScanner()

        # ---------- 参数 ----------
        self.canny_low = tk.IntVar(value=0)
        self.canny_high = tk.IntVar(value=0)
        self.enable_clahe = tk.BooleanVar(value=True)
        self.enable_sharpen = tk.BooleanVar(value=True)
        self.enable_binarize = tk.BooleanVar(value=False)
        self.binarize_method = tk.StringVar(value="otsu")
        self.binarize_block_size = tk.IntVar(value=41)
        self.binarize_c = tk.IntVar(value=2)
        self.enable_denoise = tk.BooleanVar(value=True)
        self.denoise_kernel = tk.IntVar(value=3)
        self.show_debug = tk.BooleanVar(value=False)

        # ---------- 构建 UI ----------
        self._build_ui()

    # ==================================================================
    # UI 构建
    # ==================================================================

    def _build_ui(self):
        # 主布局：左侧控制面板 + 右侧显示区
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ---- 左侧控制面板 ----
        left_frame = ttk.Frame(main_paned, width=260)
        main_paned.add(left_frame, weight=0)

        self._build_control_panel(left_frame)

        # ---- 右侧显示区 ----
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        self._build_display_panel(right_frame)

    def _build_control_panel(self, parent):
        pad = {"padx": 10, "pady": 4, "anchor": tk.W}
        pad_no_anchor = {"padx": 10, "pady": 4, "fill": tk.X}
        col_pad = {"padx": 3, "pady": 2}

        # ---- 可滚动画布 ----
        canvas = tk.Canvas(parent, highlightthickness=0, width=270)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        # 鼠标滚轮支持
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 标题
        ttk.Label(scroll_frame, text="📄 文档扫描器", font=("", 14, "bold")).pack(
            pady=(8, 4))

        # ---- 两列布局 ----
        top_row = ttk.Frame(scroll_frame)
        top_row.pack(fill=tk.X, padx=4)

        # 左列：文件 + 显示选项 + 批量导航
        left_col = ttk.Frame(top_row)
        left_col.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ---- 文件操作 ----
        file_frame = ttk.LabelFrame(left_col, text="文件", padding=6)
        file_frame.pack(fill=tk.X, pady=2, padx=1)
        ttk.Button(file_frame, text="📂 打开图片", command=self._open_image).pack(fill=tk.X, **col_pad)
        ttk.Button(file_frame, text="📁 打开文件夹", command=self._open_folder).pack(fill=tk.X, **col_pad)
        ttk.Separator(file_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3, padx=6)
        ttk.Button(file_frame, text="💾 保存图像", command=self._save_result).pack(fill=tk.X, **col_pad)
        ttk.Button(file_frame, text="📄 导出 PDF", command=self._save_pdf).pack(fill=tk.X, **col_pad)

        # ---- 显示选项 ----
        display_frame = ttk.LabelFrame(left_col, text="显示选项", padding=6)
        display_frame.pack(fill=tk.X, pady=2, padx=1)
        ttk.Checkbutton(display_frame, text="Canny 调试图",
                        variable=self.show_debug).pack(**col_pad)

        # ---- 批量导航 ----
        self.nav_frame = ttk.LabelFrame(left_col, text="批量导航", padding=6)
        nav_row = ttk.Frame(self.nav_frame)
        nav_row.pack(fill=tk.X)
        ttk.Button(nav_row, text="◀", command=self._navigate_prev, width=3).pack(side=tk.LEFT, padx=1)
        self.nav_label = ttk.Label(nav_row, text="—", font=("", 9, "bold"), anchor=tk.CENTER, width=8)
        self.nav_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(nav_row, text="▶", command=self._navigate_next, width=3).pack(side=tk.RIGHT, padx=1)
        self.nav_frame.pack_forget()

        # 右列：检测参数 + 增强参数
        right_col = ttk.Frame(top_row)
        right_col.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # ---- 检测参数 ----
        detect_frame = ttk.LabelFrame(right_col, text="检测参数", padding=6)
        detect_frame.pack(fill=tk.X, pady=2, padx=1)

        ttk.Label(detect_frame, text="Canny 低阈值:", font=("", 9)).pack(**col_pad)
        ttk.Scale(detect_frame, from_=0, to=300, variable=self.canny_low,
                  orient=tk.HORIZONTAL).pack(fill=tk.X, **col_pad)
        ttk.Label(detect_frame, textvariable=self.canny_low, font=("", 8)).pack(anchor=tk.W, padx=10)

        ttk.Label(detect_frame, text="Canny 高阈值:", font=("", 9)).pack(**col_pad)
        ttk.Scale(detect_frame, from_=0, to=500, variable=self.canny_high,
                  orient=tk.HORIZONTAL).pack(fill=tk.X, **col_pad)
        ttk.Label(detect_frame, textvariable=self.canny_high, font=("", 8)).pack(anchor=tk.W, padx=10)

        # ---- 增强参数 ----
        enh_frame = ttk.LabelFrame(right_col, text="增强参数", padding=6)
        enh_frame.pack(fill=tk.X, pady=2, padx=1)

        ttk.Checkbutton(enh_frame, text="CLAHE", variable=self.enable_clahe).pack(anchor=tk.W, **col_pad)
        ttk.Checkbutton(enh_frame, text="锐化", variable=self.enable_sharpen).pack(anchor=tk.W, **col_pad)

        ttk.Checkbutton(enh_frame, text="二值化", variable=self.enable_binarize).pack(anchor=tk.W, **col_pad)
        method_row = ttk.Frame(enh_frame)
        method_row.pack(**col_pad)
        ttk.Radiobutton(method_row, text="Otsu", variable=self.binarize_method,
                        value="otsu").pack(side=tk.LEFT, padx=1)
        ttk.Radiobutton(method_row, text="Adaptive", variable=self.binarize_method,
                        value="adaptive").pack(side=tk.LEFT, padx=1)

        ttk.Label(enh_frame, text="块大小:", font=("", 8)).pack(anchor=tk.W, padx=6)
        ttk.Scale(enh_frame, from_=11, to=199, variable=self.binarize_block_size,
                  orient=tk.HORIZONTAL,
                  command=lambda v: self.binarize_block_size.set(int(round(float(v))))).pack(fill=tk.X, padx=6)
        ttk.Label(enh_frame, textvariable=self.binarize_block_size, font=("", 8)).pack(anchor=tk.W, padx=10)

        ttk.Label(enh_frame, text="C 值:", font=("", 8)).pack(anchor=tk.W, padx=6)
        ttk.Scale(enh_frame, from_=0, to=20, variable=self.binarize_c,
                  orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6)
        ttk.Label(enh_frame, textvariable=self.binarize_c, font=("", 8)).pack(anchor=tk.W, padx=10)

        ttk.Checkbutton(enh_frame, text="去噪", variable=self.enable_denoise).pack(anchor=tk.W, **col_pad)
        ttk.Label(enh_frame, text="核大小:", font=("", 8)).pack(anchor=tk.W, padx=6)
        ttk.Scale(enh_frame, from_=3, to=15, variable=self.denoise_kernel,
                  orient=tk.HORIZONTAL,
                  command=lambda v: self.denoise_kernel.set(int(round(float(v))))).pack(fill=tk.X, padx=6)
        ttk.Label(enh_frame, textvariable=self.denoise_kernel, font=("", 8)).pack(anchor=tk.W, padx=10)

        # ---- 运行按钮 + 状态栏 ----
        bottom_frame = ttk.Frame(scroll_frame)
        bottom_frame.pack(fill=tk.X, pady=6, padx=8)
        ttk.Separator(bottom_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Button(bottom_frame, text="🚀 开始扫描", command=self._run_scan,
                   style="Accent.TButton").pack(fill=tk.X, pady=4)

        self.status_var = tk.StringVar(value="就绪 — 打开图片开始扫描")
        ttk.Label(bottom_frame, textvariable=self.status_var,
                  font=("", 8), foreground="gray", wraplength=240).pack(
            fill=tk.X, pady=2)

    def _build_display_panel(self, parent):
        # 标签栏切换不同视图
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 各 tab
        self.tabs = {}
        tab_info = [
            ("original",    "原图"),
            ("annotated",   "角点标注"),
            ("canny_debug", "Canny 调试"),
            ("warped",      "透视矫正"),
            ("final",       "最终输出"),
            ("binary",      "二值图"),
        ]

        for key, label in tab_info:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=label)

            # 图片显示用 Label
            img_label = ttk.Label(frame)
            img_label.pack(fill=tk.BOTH, expand=True)

            self.tabs[key] = {"frame": frame, "label": img_label}

    # ==================================================================
    # 功能
    # ==================================================================

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("所有文件", "*.*")],
        )
        if not path:
            return

        self.mode = "single"
        self.image_path = path
        self.batch_results = []
        self.image_paths = []
        self.nav_frame.pack_forget()

        self.original_bgr = cv2.imread(path)
        if self.original_bgr is None:
            messagebox.showerror("错误", f"无法读取图片: {path}")
            return

        # 重置结果
        self.result_cache = {}
        self.status_var.set(f"已加载: {Path(path).name}")

        # 显示原图
        self._show_image_on_tab("original", self.original_bgr)
        self.notebook.select(self.tabs["original"]["frame"])
        self._clear_all_tabs_except("original")

    def _run_scan(self):
        if self.mode == "batch":
            self._run_batch_scan()
            return

        if self.original_bgr is None:
            messagebox.showwarning("提示", "请先打开一张图片")
            return

        self.status_var.set("⏳ 正在检测文档角点...")
        self.root.update_idletasks()

        # 构建 scanner
        scanner = DocScanner(
            canny_low=self.canny_low.get() if self.canny_low.get() > 0 else None,
            canny_high=self.canny_high.get() if self.canny_high.get() > 0 else None,
            clahe=self.enable_clahe.get(),
            sharpen=self.enable_sharpen.get(),
            binarize=self.enable_binarize.get(),
            binarize_method=self.binarize_method.get(),
            binarize_block_size=self.binarize_block_size.get() if self.binarize_method.get() == "adaptive" else None,
            binarize_c=self.binarize_c.get() if self.binarize_method.get() == "adaptive" else None,
            denoise=self.enable_denoise.get(),
            denoise_kernel=self.denoise_kernel.get() if self.enable_denoise.get() else None,
        )

        try:
            result = scanner.scan_image(self.original_bgr)
        except Exception as e:
            messagebox.showerror("错误", f"扫描出错:\n{e}")
            self.status_var.set("❌ 扫描出错")
            return

        if not result["success"]:
            messagebox.showerror("检测失败", result.get("error", "未知错误"))
            self.status_var.set(f"❌ {result.get('error', '检测失败')}")
            return

        self.result_cache = result
        self._display_result(result)

        # 状态信息
        corners = result.get("corners")
        if corners is not None:
            labels = ["TL", "TR", "BR", "BL"]
            corner_str = " | ".join(f"{l}: ({x:.0f},{y:.0f})" for l, (x, y) in zip(labels, corners))
            self.status_var.set(f"✅ 检测成功 — {corner_str}")
        else:
            self.status_var.set("✅ 检测成功")

    def _run_batch_scan(self):
        if not self.image_paths:
            messagebox.showwarning("提示", "请先打开一个文件夹")
            return

        scanner = DocScanner(
            canny_low=self.canny_low.get() if self.canny_low.get() > 0 else None,
            canny_high=self.canny_high.get() if self.canny_high.get() > 0 else None,
            clahe=self.enable_clahe.get(),
            sharpen=self.enable_sharpen.get(),
            binarize=self.enable_binarize.get(),
            binarize_method=self.binarize_method.get(),
            binarize_block_size=self.binarize_block_size.get() if self.binarize_method.get() == "adaptive" else None,
            binarize_c=self.binarize_c.get() if self.binarize_method.get() == "adaptive" else None,
        )

        self.batch_results = []
        total = len(self.image_paths)
        success_count = 0

        for i, path in enumerate(self.image_paths):
            self.status_var.set(f"⏳ 扫描中 ({i + 1}/{total}): {path.name}")
            self.root.update_idletasks()

            bgr = cv2.imread(str(path))
            if bgr is None:
                self.batch_results.append({"success": False, "error": f"无法读取: {path.name}", "path": path})
                continue

            try:
                result = scanner.scan_image(bgr)
                result["path"] = path
            except Exception as e:
                result = {"success": False, "error": str(e), "path": path, "original": bgr}

            self.batch_results.append(result)
            if result["success"]:
                success_count += 1

        # 显示第一张成功的结果
        self.current_index = 0
        if success_count > 0:
            # 跳转到第一个成功的结果
            for idx, r in enumerate(self.batch_results):
                if r["success"]:
                    self.current_index = idx
                    break
            self._display_batch_result()
        else:
            messagebox.showerror("扫描失败", f"{total} 张图片全部扫描失败")
            self.status_var.set("❌ 批量扫描全部失败")
            return

        self.status_var.set(f"✅ 批量扫描完成: 成功 {success_count}/{total}")
        self.nav_label.config(text=f"{self.current_index + 1}/{total}")
        self.nav_frame.pack(fill=tk.X, pady=2, padx=1)

    def _save_result(self):
        if self.mode == "batch" and self.batch_results:
            self._save_batch_results()
            return

        if not self.result_cache:
            messagebox.showwarning("提示", "没有可保存的结果，请先运行扫描")
            return

        dir_path = filedialog.askdirectory(title="选择保存目录", initialdir="./output")
        if not dir_path:
            return

        out_dir = Path(dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self.image_path).stem if self.image_path else "scan"

        saved = []
        img_map = {
            "warped": f"{stem}_warped.jpg",
            "annotated": f"{stem}_annotated.jpg",
            "canny_debug": f"{stem}_canny_debug.jpg",
            "final": f"{stem}_scanned.jpg",
            "binary": f"{stem}_binary.jpg",
        }
        for key, filename in img_map.items():
            img = self.result_cache.get(key)
            if img is not None:
                path = out_dir / filename
                cv2.imwrite(str(path), img)
                saved.append(filename)

        if saved:
            messagebox.showinfo("保存成功", f"已保存 {len(saved)} 个文件到:\n{out_dir}")
            self.status_var.set(f"💾 已保存 {len(saved)} 个文件")
        else:
            messagebox.showwarning("提示", "没有可保存的图像数据")

    def _save_batch_results(self):
        dir_path = filedialog.askdirectory(title="选择保存目录", initialdir="./output")
        if not dir_path:
            return

        out_dir = Path(dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        total_saved = 0
        for result in self.batch_results:
            if not result["success"]:
                continue
            stem = result.get("path", Path("unknown")).stem
            img_map = {
                "final": f"{stem}_scanned.jpg",
                "warped": f"{stem}_warped.jpg",
                "annotated": f"{stem}_annotated.jpg",
            }
            for key, filename in img_map.items():
                img = result.get(key)
                if img is not None:
                    cv2.imwrite(str(out_dir / filename), img)
                    total_saved += 1

        messagebox.showinfo("保存成功", f"批量保存完成！共保存 {total_saved} 张图像到:\n{out_dir}")
        self.status_var.set(f"💾 批量保存 {total_saved} 张图像")

    # ==================================================================
    # 图像显示工具
    # ==================================================================

    def _show_image_on_tab(self, tab_key: str, bgr_img: np.ndarray):
        if bgr_img is None or bgr_img.size == 0:
            return
        label = self.tabs[tab_key]["label"]

        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # 缩放以适应标签区域
        label.update_idletasks()
        max_w = max(label.winfo_width(), 400)
        max_h = max(label.winfo_height(), 300)

        img_w, img_h = pil_img.size
        scale = min(max_w / img_w, max_h / img_h, 1.0)
        if scale < 1:
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil_img)
        label.config(image=tk_img)
        label.image = tk_img  # 防止被 GC

    def _clear_all_tabs_except(self, keep_key: str):
        for key, tab in self.tabs.items():
            if key != keep_key:
                tab["label"].config(image="")
                tab["label"].image = None

    # ==================================================================
    # 新增功能：文件夹输入 & PDF 导出
    # ==================================================================

    def _open_folder(self):
        dir_path = filedialog.askdirectory(title="选择图片文件夹")
        if not dir_path:
            return

        try:
            paths = io_utils.collect_images(dir_path)
        except FileNotFoundError as e:
            messagebox.showerror("错误", str(e))
            return

        self.mode = "batch"
        self.image_paths = paths
        self.current_index = 0
        self.batch_results = []
        self.result_cache = {}

        # 加载第一张预览
        self._load_batch_image(0)
        self.status_var.set(f"📁 已加载文件夹: {Path(dir_path).name} ({len(paths)} 张图片)")

    def _load_batch_image(self, index: int):
        """加载批量模式下的某张图片到预览（不扫描）"""
        if not self.image_paths or index < 0 or index >= len(self.image_paths):
            return
        self.current_index = index
        path = self.image_paths[index]
        self.original_bgr = cv2.imread(str(path))
        if self.original_bgr is None:
            messagebox.showerror("错误", f"无法读取图片: {path.name}")
            return

        self.result_cache = {}
        self._show_image_on_tab("original", self.original_bgr)
        self.notebook.select(self.tabs["original"]["frame"])
        self._clear_all_tabs_except("original")
        self.status_var.set(f"📁 [{index + 1}/{len(self.image_paths)}] {path.name}")

    def _navigate_prev(self):
        if not self.batch_results:
            return
        idx = (self.current_index - 1) % len(self.batch_results)
        self.current_index = idx
        self._display_batch_result()

    def _navigate_next(self):
        if not self.batch_results:
            return
        idx = (self.current_index + 1) % len(self.batch_results)
        self.current_index = idx
        self._display_batch_result()

    def _display_batch_result(self):
        """显示批量模式下当前索引的扫描结果"""
        idx = self.current_index
        if idx >= len(self.batch_results):
            return
        result = self.batch_results[idx]
        path = result.get("path", self.image_paths[idx] if idx < len(self.image_paths) else None)

        # 加载原图（无论扫描成功与否都显示原图）
        if "original" in result and result["original"] is not None:
            self.original_bgr = result["original"]
        elif path is not None:
            self.original_bgr = cv2.imread(str(path))

        self._clear_all_tabs_except("none")

        if self.original_bgr is not None:
            self._show_image_on_tab("original", self.original_bgr)

        total = len(self.batch_results)
        if result["success"]:
            self.result_cache = result
            self._display_result(result)
        elif self.original_bgr is not None:
            self.notebook.select(self.tabs["original"]["frame"])

        name = path.name if path else f"#{idx}"
        self.nav_label.config(text=f"{idx + 1}/{total}")
        self.status_var.set(f"{'✅' if result['success'] else '❌'} [{idx + 1}/{total}] {name}")

    def _display_result(self, result: dict):
        """展示单个扫描结果的各阶段图"""
        if "annotated" in result:
            self._show_image_on_tab("annotated", result["annotated"])
        if "canny_debug" in result:
            self._show_image_on_tab("canny_debug", result["canny_debug"])
        if "warped" in result:
            self._show_image_on_tab("warped", result["warped"])
        if "final" in result:
            self._show_image_on_tab("final", result["final"])
        if "binary" in result and result["binary"] is not None:
            self._show_image_on_tab("binary", result["binary"])

        # 切换到适当的视图
        if self.show_debug.get() and "canny_debug" in result:
            self.notebook.select(self.tabs["canny_debug"]["frame"])
        elif result.get("success"):
            self.notebook.select(self.tabs["annotated"]["frame"])

    def _save_pdf(self):
        """导出扫描结果为 PDF"""
        if self.mode == "batch" and self.batch_results:
            self._save_batch_pdf()
            return

        if not self.result_cache:
            messagebox.showwarning("提示", "没有可保存的结果，请先运行扫描")
            return

        pdf_path = filedialog.asksaveasfilename(
            title="导出 PDF",
            defaultextension=".pdf",
            filetypes=[("PDF 文件", "*.pdf")],
            initialdir="./output",
        )
        if not pdf_path:
            return

        # 对于单张模式：先保存临时图片，再转 PDF
        tmp_dir = Path(pdf_path).parent / ".tmp_pdf"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self.image_path).stem if self.image_path else "scan"

        tmp_path = tmp_dir / f"{stem}_scanned.jpg"
        pdf_img = self.result_cache.get("final_raw")
        if pdf_img is None:
            pdf_img = self.result_cache.get("final")
        if pdf_img is None:
            pdf_img = self.result_cache.get("warped")
        if pdf_img is not None:
            cv2.imwrite(str(tmp_path), pdf_img)
        else:
            messagebox.showerror("错误", "没有可导出的图像")
            return

        try:
            io_utils.save_as_pdf([tmp_path], pdf_path)
            # 清理临时文件
            tmp_path.unlink()
            tmp_dir.rmdir()
            messagebox.showinfo("导出成功", f"PDF 已保存至:\n{pdf_path}")
            self.status_var.set(f"📄 PDF 已导出: {Path(pdf_path).name}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    def _save_batch_pdf(self):
        """导出批量扫描结果为每个文档各自独立的 PDF"""
        dir_path = filedialog.askdirectory(title="选择 PDF 保存目录", initialdir="./output")
        if not dir_path:
            return

        out_dir = Path(dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        tmp_dir = out_dir / ".tmp_pdf_batch"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i, result in enumerate(self.batch_results):
            if not result["success"]:
                continue
            pdf_img = result.get("final_raw")
            if pdf_img is None:
                pdf_img = result.get("final")
            if pdf_img is None:
                pdf_img = result.get("warped")
            if pdf_img is None:
                continue
            stem = result.get("path", Path(f"page_{i:04d}")).stem
            tmp_path = tmp_dir / f"{stem}.jpg"
            cv2.imwrite(str(tmp_path), pdf_img)

            pdf_path = out_dir / f"{stem}.pdf"
            try:
                io_utils.save_as_pdf([tmp_path], pdf_path)
                tmp_path.unlink()
                saved += 1
            except Exception as e:
                self.status_var.set(f"❌ {stem}.pdf 导出失败")

        if tmp_dir.exists():
            try:
                tmp_dir.rmdir()
            except OSError:
                pass

        if saved:
            messagebox.showinfo("导出成功", f"共导出 {saved} 个 PDF 文件到:\n{out_dir}")
            self.status_var.set(f"📄 已导出 {saved} 个 PDF")
        else:
            messagebox.showwarning("提示", "没有扫描成功的图片可导出")

    # ==================================================================
    # 启动
    # ==================================================================

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = DocScannerApp()
    app.run()
