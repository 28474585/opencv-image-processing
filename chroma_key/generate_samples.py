"""生成绿幕前景与合成背景样例，使演示无需外部素材即可离线运行。

生成两张图：
- data/greenscreen.png : 绿色背景上的"人物"（纯程序绘制）
- data/background.jpg  : 自定义合成背景（天空渐变 + 太阳 + 山 + 地面）
"""

from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
W, H = 640, 480


def _green_background():
    """纯绿背景加轻微亮度渐变，考验 HSV 区间选取的鲁棒性。"""
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (0, 255, 0)
    grad = np.linspace(0, 35, H, dtype=np.float64)[:, None, None]
    img = img.astype(np.float64) - grad
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_person(img):
    """在图上绘制一个简单的"人物"：头、躯干、手臂、腿、鞋。"""
    skin = (170, 200, 210)      # BGR 肤色
    shirt = (60, 60, 200)       # 红色上衣
    pants = (140, 110, 50)      # 深蓝裤子
    shoe = (60, 60, 60)

    cx, head_cy = 320, 230

    # 头
    cv2.circle(img, (cx, head_cy), 42, skin, -1)
    # 眼睛
    cv2.circle(img, (cx - 16, head_cy - 6), 5, (20, 20, 20), -1)
    cv2.circle(img, (cx + 16, head_cy - 6), 5, (20, 20, 20), -1)
    # 嘴
    cv2.ellipse(img, (cx, head_cy + 20), (16, 9), 0, 0, 180, (60, 60, 60), 3)

    # 躯干
    cv2.rectangle(img, (cx - 45, head_cy + 42), (cx + 45, head_cy + 150), shirt, -1)
    # 手臂（两侧）
    cv2.line(img, (cx - 40, head_cy + 55), (cx - 85, head_cy + 120), shirt, 22)
    cv2.line(img, (cx + 40, head_cy + 55), (cx + 85, head_cy + 120), shirt, 22)
    cv2.circle(img, (cx - 85, head_cy + 120), 12, skin, -1)
    cv2.circle(img, (cx + 85, head_cy + 120), 12, skin, -1)

    # 腿
    cv2.rectangle(img, (cx - 45, head_cy + 150), (cx - 8, head_cy + 225), pants, -1)
    cv2.rectangle(img, (cx + 8, head_cy + 150), (cx + 45, head_cy + 225), pants, -1)
    # 鞋
    cv2.rectangle(img, (cx - 48, head_cy + 220), (cx - 6, head_cy + 232), shoe, -1)
    cv2.rectangle(img, (cx + 6, head_cy + 220), (cx + 48, head_cy + 232), shoe, -1)

    # 给人物边缘叠一点点绿色反光（真实绿幕常有的"绿溢"），考验羽化处理
    spill = np.zeros_like(img, np.uint8)
    cv2.circle(spill, (cx, head_cy), 45, (0, 40, 0), 3)
    cv2.rectangle(spill, (cx - 48, head_cy + 40), (cx + 48, head_cy + 155), (0, 40, 0), 3)
    img[:] = np.clip(img.astype(np.int16) + spill.astype(np.int16), 0, 255).astype(np.uint8)


def _make_background():
    """自定义背景：天空渐变 + 太阳 + 远山 + 地面。"""
    img = np.zeros((H, W, 3), np.uint8)
    # 天空渐变（上深下浅的蓝色）
    for y in range(int(H * 0.62)):
        t = y / (H * 0.62)
        b = int(180 + 60 * (1 - t))
        g = int(120 + 100 * (1 - t))
        r = int(60 + 140 * (1 - t))
        img[y, :] = (b, g, r)
    # 太阳
    cv2.circle(img, (500, 80), 36, (60, 200, 255), -1)
    cv2.circle(img, (500, 80), 50, (60, 200, 255), 8)
    # 远山
    pts = np.array([[0, 320], [140, 180], [280, 320], [420, 210], [560, 320], [640, 260], [640, 320]], np.int32)
    cv2.fillPoly(img, [pts], (110, 130, 90))
    # 地面
    for y in range(int(H * 0.62), H):
        t = (y - H * 0.62) / (H * 0.38)
        img[y, :] = (int(60 + 30 * t), int(140 + 50 * t), int(70 + 40 * t))
    # 地面上的细纹理（测试抠图合成时背景的细节保留）
    for x in range(0, W, 24):
        cv2.line(img, (x, int(H * 0.62)), (x + 40, H), (0, 40, 30), 2)
    return img


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    gs = _green_background()
    _draw_person(gs)
    cv2.imwrite(str(DATA / "greenscreen.png"), gs)

    bg = _make_background()
    cv2.imwrite(str(DATA / "background.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print("已生成:")
    print(" ", DATA / "greenscreen.png")
    print(" ", DATA / "background.jpg")


if __name__ == "__main__":
    main()
