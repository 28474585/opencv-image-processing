"""均值滤波 / 中值滤波 / 高斯滤波降噪性能对比实验。

对照两组噪声环境、多档强度，用 PSNR（峰值信噪比）量化评估三种滤波器：
- 高斯噪声   sigma ∈ {10, 25, 50}
- 椒盐噪声   density ∈ {0.01, 0.05, 0.10}

结果输出：
- 终端打印对比表
- output/denoise_results.csv   结构化结果
- output/denoise_chart_*.png   分组柱状图
- output/denoise_visual_*.png  滤波效果可视化（含滤波前噪声图）

用法：
    python denoise_compare.py [--image 可选测试图; 默认程序合成]
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.io import imread, imwrite  # noqa: E402
from common.metrics import psnr  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "output"
KERNEL = 5

NOISE_LEVELS = {
    "gaussian":   {"sigma": [10, 25, 50]},
    "salt_pepper": {"density": [0.01, 0.05, 0.10]},
}

FILTERS = {
    "mean":    lambda img: cv2.blur(img, (KERNEL, KERNEL)),
    "median":  lambda img: cv2.medianBlur(img, KERNEL),
    "gaussian": lambda img: cv2.GaussianBlur(img, (KERNEL, KERNEL), 1.5),
}


def make_test_image(size=(512, 512)):
    """程序合成一张含平滑区 + 边缘 + 细纹理的测试图（无需联网下载素材）。"""
    h, w = size
    img = np.zeros((h, w, 3), np.uint8)

    # 天空渐变
    for y in range(int(h * 0.55)):
        t = y / (h * 0.55)
        img[y, :] = (int(200 - 80 * t), int(120 - 60 * t), int(60 + 60 * t))
    # 地面渐变
    for y in range(int(h * 0.55), h):
        t = (y - h * 0.55) / (h * 0.45)
        img[y, :] = (int(40 + 30 * t), int(90 + 70 * t), int(60 + 30 * t))
    # 太阳（边缘）
    cv2.circle(img, (420, 90), 40, (60, 180, 255), -1)
    # 三角形山峰（多方向边缘）
    cv2.fillPoly(img, [np.array([[0, h], [140, int(h * 0.30)], [280, h]], np.int32)], (90, 110, 80))
    cv2.fillPoly(img, [np.array([[220, h], [360, int(h * 0.38)], [500, h]], np.int32)], (110, 130, 90))
    # 地面细纹理（正弦条纹，考验高频细节保留）
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    stripes = ((np.sin(xx * 0.35) * 30 + np.sin(yy * 0.22) * 20)) * (yy > h * 0.55)
    img = np.clip(img.astype(np.float64) + stripes[..., None] * 0.35, 0, 255).astype(np.uint8)
    return img


def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img, density):
    noisy = img.copy()
    h, w = img.shape[:2]
    n = int(density * h * w)
    # 白噪声（盐）
    ys, xs = np.random.randint(0, h, n), np.random.randint(0, w, n)
    noisy[ys, xs] = 255
    # 黑噪声（椒）
    ys, xs = np.random.randint(0, h, n), np.random.randint(0, w, n)
    noisy[ys, xs] = 0
    return noisy


def run_experiment(clean):
    rows = []
    results = {}
    for noise_type, levels in NOISE_LEVELS.items():
        for lv in levels[next(iter(levels))]:
            key = lv
            if noise_type == "gaussian":
                noisy = add_gaussian_noise(clean, lv)
                lv_label = f"σ={lv}"
            else:
                noisy = add_salt_pepper_noise(clean, lv)
                lv_label = f"density={lv}"

            noisy_psnr = psnr(clean, noisy)
            per_filter = {}
            for name, fn in FILTERS.items():
                denoised = fn(noisy)
                p = psnr(clean, denoised)
                per_filter[name] = p
                rows.append([noise_type, lv_label, name, round(p, 2)])

            # 中等强度保存一张可视化对比图
            if (noise_type == "gaussian" and lv == 25) or (noise_type == "salt_pepper" and lv == 0.05):
                save_visual(clean, noisy, noise_type, lv_label)
                results[noise_type] = {"noisy_psnr": round(noisy_psnr, 2), "filters": per_filter}
            print(f"[{noise_type} | {lv_label}] 噪声本身 PSNR={noisy_psnr:.2f} dB -> "
                  + "  ".join(f"{k}={v:.2f}" for k, v in per_filter.items()))

    write_csv(rows)
    draw_charts()
    print_summary(rows)


def save_visual(clean, noisy, noise_type, lv_label):
    row = [clean, noisy]
    for name, fn in FILTERS.items():
        row.append(fn(noisy))
    grid = np.hstack(row)
    h, w = grid.shape[:2]
    grid = cv2.resize(grid, (w // 2, h // 2))
    fname = OUT / f"denoise_visual_{noise_type}.png"
    imwrite(fname, grid)
    print("  可视化对比图:", fname)


def write_csv(rows):
    fname = OUT / "denoise_results.csv"
    with open(fname, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["noise_type", "noise_level", "filter", "psnr_db"])
        wr.writerows(rows)
    print("结果表:", fname)


def draw_charts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_data = {}
    with open(OUT / "denoise_results.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            csv_data.setdefault(row["noise_type"], {}).setdefault(row["noise_level"], {})[row["filter"]] = float(row["psnr_db"])

    colors = {"mean": "#2f6db5", "median": "#e8a33d", "gaussian": "#3fa13f"}
    for noise_type in ["gaussian", "salt_pepper"]:
        levels = list(csv_data[noise_type])
        width = 0.25
        x = np.arange(len(levels))
        plt.figure(figsize=(8, 5))
        for i, fn in enumerate(FILTERS):
            vals = [csv_data[noise_type][lv][fn] for lv in levels]
            plt.bar(x + (i - 1) * width, vals, width, label=fn, color=colors[fn])
        plt.xticks(x, levels)
        plt.xlabel("noise level")
        plt.ylabel("PSNR (dB)")
        plt.title(f"Denoising PSNR comparison - {noise_type} noise")
        plt.legend()
        plt.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.savefig(OUT / f"denoise_chart_{noise_type}.png", dpi=120)
        plt.close()


def print_summary(rows):
    by = {}
    for noise_type, lv, fn, p in rows:
        by.setdefault(noise_type, {}).setdefault(fn, []).append(p)
    print("\n===== 实验结论（按各档噪声强度 PSNR 均值，数据驱动）=====")
    for noise_type, fmap in by.items():
        means = {fn: np.mean(v) for fn, v in fmap.items()}
        best = max(means, key=means.get)
        desc = "  ".join(f"{fn}: {m:.2f} dB" for fn, m in means.items())
        print(f"[{noise_type}]  {desc}")
        print(f"  最优: {best}")
    print("说明: PSNR 越高越好。中值滤波对椒盐（脉冲）噪声优势显著;"
          "高斯/均值滤波在 5x5 窗口下对高斯噪声表现接近。")


def main():
    ap = argparse.ArgumentParser(description="三种滤波器降噪性能对比实验")
    ap.add_argument("--image", help="可选: 使用自己的测试图; 默认程序合成")
    args = ap.parse_args()

    clean = imread(args.image) if args.image else make_test_image()
    if clean is None:
        raise SystemExit(f"无法读取测试图: {args.image}")
    OUT.mkdir(parents=True, exist_ok=True)
    run_experiment(clean)


if __name__ == "__main__":
    main()
