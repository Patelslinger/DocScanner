# Smart Document Scanner

智能文档扫描与透视矫正 — 计算机视觉课设项目。

## 功能
- 给定拍摄的文档/纸张照片，自动定位四个角点
- 透视变换矫正为正视图
- 图像增强（CLAHE、锐化、二值化）输出扫描效果
- 批量处理、自动保存为 PDF

## 环境配置


### pip 安装
```bash
pip install opencv-python==4.8.1.78 numpy==1.26.4 Pillow==10.1.0 img2pdf==0.5.1
```

### conda 安装（推荐）
```bash
conda create -n docscanner python=3.10 -y
conda activate docscanner
conda install -c conda-forge opencv=4.8.1 numpy=1.26.4 pillow=10.1.0 matplotlib=3.8.2 -y
pip install img2pdf==0.5.1
```

## 版本建议
| 组件 | 推荐版本 |
|------|----------|
| Python | 3.10.x / 3.11.x |
| opencv | 4.8.1.78 |
| numpy | 1.24.x / 1.26.x |
| Pillow | 10.1.0 |
| img2pdf | 0.5.1 |

## 约束
- 禁止使用机器学习模型（SVM、CNN、预训练检测器等）
- 禁止调用封装的 OCR 引擎（如 Tesseract）
- 仅使用经典计算机视觉算法

## Pipeline
```
输入 → 灰度 → 高斯模糊 → Canny边缘检测 → 轮廓筛选 → 多边形近似 → 角点排序 → 透视变换 → 图像增强 → 输出
```
