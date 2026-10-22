"""通用评估指标：PSNR（峰值信噪比）。"""

import numpy as np


def psnr(clean, processed, max_val=255.0):
    """计算两幅同尺寸图像的 PSNR（dB）。

    PSNR 越大说明 processed 与 clean 越接近；无限大表示完全一致。
    """
    clean = clean.astype(np.float64)
    processed = processed.astype(np.float64)
    mse = np.mean((clean - processed) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(max_val * max_val / mse)


def capacity_mb(n_bytes):
    return n_bytes / (1024 * 1024)
