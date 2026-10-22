"""LBP（局部二值模式）纹理特征提取与分类实验。

流程：
1. 灰度图上计算 8 邻域 LBP 编码：邻域像素 >= 中心像素记 1，否则记 0；
2. 把 256 种编码按"均匀模式"归并成 59 类（58 种均匀模式 + 1 类非均匀模式），
   降低特征维度并让直方图对噪声更稳；
3. 图像分成 4x4 个格子，分别统计直方图后拼接（保留一定的空间信息）；
4. 每类用训练样本的平均直方图作为模板，卡方距离最近邻分类。

结果输出：
- 终端打印每张测试图的预测与总体准确率、混淆矩阵
- output/lbp_samples.png     三类训练样例
- output/lbp_codes.png       原图与 LBP 编码图
- output/lbp_histograms.png  三类纹理的平均 LBP 直方图
- output/lbp_results.csv     逐张测试结果

用法：
    python lbp_texture.py
    python lbp_texture.py --grid 8    # 改成 8x8 分块
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.io import imread, imwrite  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "textures"
OUT = ROOT / "output"
CLASSES = ("rock", "grass", "fabric")
CLASS_NAMES = {"rock": "岩石", "grass": "草地", "fabric": "布料"}

# 半径 1、8 邻域的采样点偏移 (dy, dx)
OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def _uniform_map():
    """返回长度 256 的映射表：均匀模式按首次出现顺序编号，非均匀模式统一为最后一类。"""
    codes, non_uniform, uniform_codes = {}, [], []
    for code in range(256):
        bits = [(code >> i) & 1 for i in range(8)]
        transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        if transitions <= 2:
            uniform_codes.append(code)
        else:
            non_uniform.append(code)
    for idx, code in enumerate(uniform_codes):
        codes[code] = idx
    last = len(uniform_codes)
    for code in non_uniform:
        codes[code] = last
    return np.array([codes[c] for c in range(256)], dtype=np.int32), last + 1


UNIFORM_LUT, BINS = _uniform_map()


def lbp_code(gray):
    """逐像素计算 8 邻域 LBP 编码，返回 uint8 编码图（严格 0-255）。"""
    img = gray.astype(np.int16)
    h, w = gray.shape
    padded = np.pad(img, 1, mode="edge")
    code = np.zeros((h, w), np.uint8)
    for i, (dy, dx) in enumerate(OFFSETS):
        neighbor = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        code |= ((neighbor >= img).astype(np.uint8) << i)
    return code


def lbp_features(gray, grid=4):
    """LBP 均匀模式直方图：分 grid×grid 块统计后拼接，每块内做 L1 归一化。"""
    code = lbp_code(gray)
    h, w = code.shape
    feats = []
    for gy in range(grid):
        for gx in range(grid):
            cell = code[gy * h // grid:(gy + 1) * h // grid, gx * w // grid:(gx + 1) * w // grid]
            hist = np.bincount(UNIFORM_LUT[cell.ravel()], minlength=BINS).astype(np.float64)
            total = hist.sum()
            feats.append(hist / total if total else hist)
    return np.concatenate(feats), code


def chi_square(p, q):
    """卡方距离：直方图之间的常用度量，对小值差异更敏感。"""
    denom = p + q
    denom[denom == 0] = 1.0
    return float(np.sum((p - q) ** 2 / denom))


def load_split(split):
    samples = {}
    for name in CLASSES:
        folder = DATA / split / name
        images = sorted(folder.glob("*.png"))
        samples[name] = [(imread(p, cv2.IMREAD_GRAYSCALE), p.name) for p in images]
    return samples


def _setup_font():
    """让图里的中文标签正常显示；找不到中文字体时退回英文标签。"""
    import matplotlib
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if name in installed:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


def plot_samples(samples):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, name in zip(axes, CLASSES):
        ax.imshow(samples[name][0][0], cmap="gray", vmin=0, vmax=255)
        ax.set_title(CLASS_NAMES[name])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "lbp_samples.png", dpi=150)
    plt.close(fig)


def plot_codes(samples):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for col, name in enumerate(CLASSES):
        gray = samples[name][0][0]
        _, code = lbp_features(gray)
        axes[0, col].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(f"{CLASS_NAMES[name]}：原图")
        axes[1, col].imshow(UNIFORM_LUT[code], cmap="viridis")
        axes[1, col].set_title("LBP 均匀模式编码")
        for ax in (axes[0, col], axes[1, col]):
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "lbp_codes.png", dpi=150)
    plt.close(fig)


def plot_histograms(templates):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in CLASSES:
        tpl = templates[name]
        # 四块格子取平均，展示单块的直方图形状
        n_cells = len(tpl) // BINS
        mean_hist = tpl.reshape(n_cells, BINS).mean(axis=0)
        ax.plot(mean_hist, label=CLASS_NAMES[name], linewidth=1.2)
    ax.set_xlabel("均匀 LBP 模式编号")
    ax.set_ylabel("归一化频率")
    ax.set_title("三类纹理的平均 LBP 直方图")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "lbp_histograms.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="LBP 纹理特征提取与分类")
    ap.add_argument("--grid", type=int, default=4, help="分块数，默认 4（4x4）")
    args = ap.parse_args()

    if not (DATA / "train").exists():
        raise SystemExit("缺少纹理样例，请先运行 python generate_textures.py")

    OUT.mkdir(exist_ok=True)
    train, test = load_split("train"), load_split("test")

    # 训练：每一类的所有样本特征取平均作为模板
    templates = {}
    for name in CLASSES:
        feats = [lbp_features(img, args.grid)[0] for img, _ in train[name]]
        templates[name] = np.mean(feats, axis=0)

    # 测试：卡方距离最近邻
    rows, correct = [], 0
    confusion = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    for truth in CLASSES:
        for gray, fname in test[truth]:
            feat, _ = lbp_features(gray, args.grid)
            dists = {name: chi_square(feat, templates[name]) for name in CLASSES}
            pred = min(dists, key=dists.get)
            ok = pred == truth
            correct += ok
            confusion[truth][pred] += 1
            rows.append([fname, CLASS_NAMES[truth], CLASS_NAMES[pred], "OK" if ok else "ERR"])

    total = len(rows)
    print(f"分块：{args.grid}x{args.grid}，特征维度：{len(templates[CLASSES[0]])}，LBP 直方图类别数：{BINS}")
    print(f"{'样本':<14}{'真实类别':<8}{'预测类别':<8}结果")
    for row in rows:
        print(f"{row[0]:<14}{row[1]:<8}{row[2]:<8}{row[3]}")
    print(f"\n准确率：{correct}/{total} = {correct / total:.2%}")
    print("混淆矩阵（行=真实，列=预测）：")
    print("        " + "".join(f"{CLASS_NAMES[c]:<8}" for c in CLASSES))
    for truth in CLASSES:
        print(f"{CLASS_NAMES[truth]:<8}" + "".join(f"{confusion[truth][p]:<8}" for p in CLASSES))

    with open(OUT / "lbp_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "true_class", "predicted_class", "result"])
        writer.writerows(rows)

    _setup_font()
    plot_samples(train)
    plot_codes(train)
    plot_histograms(templates)
    print(f"\n结果图与 CSV 已写入：{OUT}")


if __name__ == "__main__":
    main()
