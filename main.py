"""
命令行入口 — 智能文档扫描器 CLI
==================================

提供两个子命令:
    scan <图片路径>    — 扫描单张图片
    batch <目录路径>   — 批量扫描整个文件夹

用法示例:
    python main.py scan input/1.jpg -o ./output
    python main.py batch input/ --binarize --pdf result.pdf
    python main.py scan input/1.jpg --canny-low 50 --canny-high 150 --no-clahe

增强选项:
    --no-clahe      禁用 CLAHE 对比度增强
    --no-sharpen    禁用锐化
    --binarize      启用二值化（默认关闭）
    --binarize-method otsu|adaptive  二值化方法
"""

import argparse
from pathlib import Path

from scanner import DocScanner


def main():
    parser = argparse.ArgumentParser(
        description="Smart Document Scanner — 智能文档扫描与透视矫正"
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # ---- scan 子命令：扫描单张图片 ----
    scan_parser = sub.add_parser("scan", help="扫描单张图片")
    scan_parser.add_argument("input", type=str, help="输入图片路径")
    scan_parser.add_argument("-o", "--output-dir", type=str, default="./output",
                             help="输出目录 (default: ./output)")
    scan_parser.add_argument("--canny-low", type=int, default=None,
                             help="Canny 低阈值 (default: 自动计算)")
    scan_parser.add_argument("--canny-high", type=int, default=None,
                             help="Canny 高阈值 (default: 自动计算)")
    scan_parser.add_argument("--no-clahe", action="store_true", help="禁用 CLAHE")
    scan_parser.add_argument("--no-sharpen", action="store_true", help="禁用锐化")
    scan_parser.add_argument("--binarize", action="store_true", help="启用二值化")
    scan_parser.add_argument("--binarize-method", choices=["otsu", "adaptive"],
                             default="otsu", help="二值化方法 (default: otsu)")

    # ---- batch 子命令：批量扫描 ----
    batch_parser = sub.add_parser("batch", help="批量扫描目录中的所有图片")
    batch_parser.add_argument("input_dir", type=str, help="输入目录")
    batch_parser.add_argument("-o", "--output-dir", type=str, default="./output",
                              help="输出目录 (default: ./output)")
    batch_parser.add_argument("--canny-low", type=int, default=None)
    batch_parser.add_argument("--canny-high", type=int, default=None)
    batch_parser.add_argument("--no-clahe", action="store_true")
    batch_parser.add_argument("--no-sharpen", action="store_true")
    batch_parser.add_argument("--binarize", action="store_true")
    batch_parser.add_argument("--binarize-method", choices=["otsu", "adaptive"], default="otsu")
    batch_parser.add_argument("--pdf", type=str, default=None,
                              help="同时导出为 PDF，指定 PDF 文件路径")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    scanner = DocScanner(
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        clahe=not args.no_clahe,
        sharpen=not args.no_sharpen,
        binarize=args.binarize,
        binarize_method=args.binarize_method,
        output_dir=Path(args.output_dir) if hasattr(args, "output_dir") else None,
    )

    if args.command == "scan":
        result = scanner.scan(args.input)
        if result["success"]:
            print(f"[OK]  已保存至 {args.output_dir}")
        else:
            print(f"[FAIL] {result['error']}")

    elif args.command == "batch":
        if args.pdf:
            results = scanner.scan_batch_to_pdf(args.input_dir, args.pdf)
            print(f"[OK]  PDF 已保存至 {args.pdf}")
        else:
            results = scanner.scan_batch(args.input_dir)

        ok = sum(1 for r in results if r["success"])
        fail = len(results) - ok
        print(f"[OK]  成功: {ok}, 失败: {fail}")


if __name__ == "__main__":
    main()
