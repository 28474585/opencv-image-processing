# Python 图像处理综合实验

> 项目时间：2026.05 – 2026.06 · 图像处理方向综合练习项目

基于 OpenCV 的三个图像处理实验，覆盖**视频抠像合成、滤波降噪对比、LBP 纹理特征提取与分类**三块内容。全部程序用 Python 编写，样例素材均为程序自动生成，克隆即可运行、无需联网下载。

## 项目结构

```
opencv-image-processing/
├── common/                    # 公共模块
│   ├── io.py                  # 中文路径安全的图片读写（Windows）
│   └── metrics.py             # PSNR 评估指标
├── chroma_key/                # 实验一：绿幕抠图（Chroma Key）
│   ├── generate_samples.py    # 生成绿幕/背景样例
│   └── chroma_key.py          # 抠图 + 合成视频
├── denoising/                 # 实验二：滤波降噪对比
│   └── denoise_compare.py     # 均值/中值/高斯滤波 + PSNR 量化
├── lbp/                       # 实验三：LBP 纹理识别
│   ├── generate_textures.py   # 生成岩石 / 草地 / 布料样例
│   └── lbp_texture.py         # LBP 特征提取 + 直方图分类
├── data/                      # 自动生成的样例素材
└── output/                    # 实验结果输出
```

## 环境要求

- Python 3.10+
- `pip install -r requirements.txt`（opencv-python / numpy / matplotlib）

## 实验一：绿幕抠图（Chroma Key）

**简介**：使用 OpenCV 库实现绿幕抠图系统，通过色度键（Chroma Key）技术将前景人物从绿色背景中分离并合成至自定义背景，最终输出合成视频。

**实现要点**：
- 基于 HSV 色度键提取绿色掩膜，通过形态学开/闭运算去除噪点与空洞；
- 高斯羽化得到柔和 alpha 通道，消除合成边缘的硬轮廓；
- 前景按 alpha 逐帧合成到自定义背景，输出 mp4 视频。

**运行**：

```bash
# 1. 生成样例（绿幕人物 + 自定义背景）
python chroma_key/generate_samples.py

# 2. 图片演示模式：前景上下弹跳 + 背景横向平移，合成一段 4 秒视频
python chroma_key/chroma_key.py --image data/greenscreen.png \
    --background data/background.jpg --output output/chroma_key_demo.mp4

# 3. 换成自己的绿幕视频：逐帧抠图合成（视频路径支持中文）
python chroma_key/chroma_key.py --video 绿幕视频.mp4 \
    --background data/background.jpg --output output/out.mp4
```

**结果**：`output/chroma_key_demo.mp4`、`output/chroma_key_demo_keyframe.png`（合成关键帧）。合成帧中人物内部绿色残留约 1%（主要为边缘羽化部分），背景替换完整。

## 实验二：滤波降噪对比

**简介**：对比分析均值滤波、中值滤波、高斯滤波在高斯噪声与椒盐噪声环境下的降噪性能，通过 PSNR（峰值信噪比）指标量化评估各滤波器在不同噪声强度下的表现。

**实验设计**：
- 高斯噪声：σ ∈ {10, 25, 50}
- 椒盐噪声：密度 ∈ {0.01, 0.05, 0.10}
- 三种滤波器统一 5×5 窗口，以 PSNR(dB) 量化（越大越好）

**运行**：

```bash
python denoising/denoise_compare.py
# 也可用自己的测试图: python denoising/denoise_compare.py --image 某图片
```

**结果**：终端打印对比表；`output/denoise_results.csv` 为结构化数据，`output/denoise_chart_*.png` 为分组柱状图，`output/denoise_visual_*.png` 为滤波效果对比图。

典型结论（本实验数据）：
- 椒盐噪声下**中值滤波**优势显著，各密度下 PSNR 均值约 42.8 dB，明显高于均值/高斯滤波（约 29 dB）；
- 高斯噪声下均值与高斯滤波在 5×5 窗口内表现接近，三者差异较小。

## 实验三：LBP 纹理特征提取与分类

**简介**：用 LBP（局部二值模式）算子提取纹理特征并统计直方图，对岩石、草地、布料三类纹理图像进行分类识别。

**实现要点**：
- **LBP 编码**：对每个像素比较 8 个邻域与中心像素的灰度，邻域不小于中心记 1，得到 8 位二进制编码；
- **均匀模式归并**：把 256 种编码中 0/1 跳变不超过 2 次的模式（均匀模式）单独保留，其余归为一类，直方图从 256 维降到 59 维，抗噪声更好；
- **分块直方图**：图像分成 4×4 个格子分别统计后拼接，保留一定空间信息，特征维度 4×4×59 = 944；
- **分类**：每类用训练样本的平均直方图作模板，按卡方距离做最近邻判别。

**运行**：

```bash
# 1. 生成三类合成纹理样例（每类训练 3 张 + 测试 3 张）
python lbp/generate_textures.py

# 2. 提取特征并分类，默认 4x4 分块
python lbp/lbp_texture.py
python lbp/lbp_texture.py --grid 8    # 换成 8x8 分块
```

**结果**：终端打印逐张预测、准确率与混淆矩阵；`output/lbp_results.csv` 为结构化结果，`output/lbp_samples.png` 为样例纹理，`output/lbp_codes.png` 为原图与 LBP 编码图，`output/lbp_histograms.png` 为三类纹理的平均直方图。

在合成的 9 张测试图上判别全部正确（9/9）。这是一个受控的小规模演示，用于验证特征提取流程，不代表真实场景下的分类准确率。

## 常见问题

- **视频写入失败**：确保输出路径所在目录存在、无中文之外的特殊字符问题；`chroma_key.py` 已内置中文路径兼容（先写临时文件再改名）。
- **抠图效果不佳（真实绿幕素材）**：绿幕光照不均时，可调整 `chroma_key.green_mask` 的 HSV 下界/上界参数。
- **LBP 分类不稳**：光照变化会影响 LBP 编码，实际使用时通常先做光照归一化；分块越大对空间差异越敏感，但特征维度也随之增加。
- **图中中文显示为方块**：`lbp_texture.py` 会自动查找系统中的中文字体（微软雅黑 / 黑体等）；若系统未安装，标签会退回默认字体。
