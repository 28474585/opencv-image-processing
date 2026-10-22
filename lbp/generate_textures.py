"""生成三类纹理样例（岩石 / 草地 / 布料），供 LBP 分类实验使用。

每类生成 TRAIN_PER_CLASS + TEST_PER_CLASS 张 256x256 灰度图，所有样例由程序
合成，不依赖外部素材。固定随机种子，重复运行得到完全相同的结果。

用法：
    python generate_textures.py
"""

from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "textures"
SIZE = 256
TRAIN_PER_CLASS = 3
TEST_PER_CLASS = 3
CLASSES = ("rock", "grass", "fabric")


def _value_noise(rng, size, cells, low=0, high=255):
    """在粗网格上取随机值再放大，得到一块连续变化的基础纹理。"""
    grid = rng.integers(low, high + 1, size=(cells, cells)).astype(np.float32)
    return cv2.resize(grid, (size, size), interpolation=cv2.INTER_CUBIC)


def make_rock(rng, size=SIZE):
    """岩石：大块明暗起伏 + 裂纹 + 颗粒噪声。"""
    img = _value_noise(rng, size, cells=6, low=70, high=190)
    img = cv2.GaussianBlur(img, (5, 5), 1.2)

    # 裂纹：随机折线，比底色暗
    crack = np.zeros((size, size), np.float32)
    for _ in range(4):
        pts = np.stack([
            np.arange(0, size, 8),
            rng.integers(0, size, size // 8) + np.linspace(0, rng.integers(-60, 60), size // 8),
        ], axis=1).astype(np.int32)
        cv2.polylines(crack, [pts], False, 1.0, thickness=int(rng.integers(1, 3)))
    crack = cv2.GaussianBlur(crack, (3, 3), 0.6)
    img = img * (1 - 0.65 * crack)

    # 颗粒：细碎亮暗点
    grain = rng.normal(0, 18, (size, size)).astype(np.float32)
    img = img + grain
    return np.clip(img, 0, 255).astype(np.uint8)


def make_grass(rng, size=SIZE):
    """草地：大量随机方向、随机长短的细线段，模拟草叶。"""
    img = np.full((size, size), 150, np.float32)
    blades = np.zeros((size, size), np.float32)
    for _ in range(900):
        x, y = rng.integers(0, size, 2)
        angle = rng.normal(-np.pi / 2, 0.5)
        length = rng.integers(6, 22)
        x2 = int(x + np.cos(angle) * length)
        y2 = int(y + np.sin(angle) * length)
        value = float(rng.uniform(0.4, 1.0))
        cv2.line(blades, (x, y), (x2, y2), value, thickness=int(rng.integers(1, 3)))
    img = img - 90 * blades
    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    img = img + rng.normal(0, 10, (size, size)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_fabric(rng, size=SIZE):
    """布料：横竖交织的周期性纱线纹理。"""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    period = float(rng.integers(10, 18))
    phase = rng.uniform(0, period, 2)
    weave_h = 0.5 + 0.5 * np.cos(2 * np.pi * (yy + phase[0]) / period)
    weave_v = 0.5 + 0.5 * np.cos(2 * np.pi * (xx + phase[1]) / period)
    img = 120 + 70 * np.maximum(weave_h, weave_v)

    # 纱线粗细不匀
    img = img + rng.normal(0, 12, (size, size)).astype(np.float32)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    return np.clip(img, 0, 255).astype(np.uint8)


GENERATORS = {"rock": make_rock, "grass": make_grass, "fabric": make_fabric}


def main():
    for split, count in (("train", TRAIN_PER_CLASS), ("test", TEST_PER_CLASS)):
        for name in CLASSES:
            out_dir = DATA / split / name
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                # 每张图独立种子，保证 train/test 的纹理互不相同
                seed = (CLASSES.index(name) + 1) * 1000 + (split == "test") * 500 + i
                rng = np.random.default_rng(seed)
                img = GENERATORS[name](rng)
                cv2.imwrite(str(out_dir / f"{name}_{i + 1}.png"), img)
    print(f"已生成纹理样例：{DATA}")


if __name__ == "__main__":
    main()
