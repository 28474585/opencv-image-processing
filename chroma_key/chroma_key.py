"""绿幕抠图（Chroma Key）合成系统。

流程：
1. 从绿幕前景（图片或视频）中，基于 HSV 色度键生成前景掩膜；
2. 形态学去噪 + 高斯羽化，得到柔和的 alpha 通道；
3. 将前景按 alpha 合成到自定义背景上；
4. 输出合成视频（默认输出 mp4）。

支持两种输入：
- --image  单张绿幕图片（演示模式：前景在背景中上下弹跳、背景横向平移，生成一段动画视频）
- --video  绿幕视频（逐帧抠图，实时合成）

用法示例：
    python chroma_key.py --image  ../data/greenscreen.png \
        --background ../data/background.jpg --output ../output/chroma_key_demo.mp4
"""

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.io import imread, imwrite  # noqa: E402

W, H = 640, 480


def green_mask(bgr, h_low=35, h_high=85, s_min=40, v_min=40):
    """基于 HSV 色度键提取绿色前景掩膜，并做形态学 + 羽化处理。

    返回 float32 的掩膜，取值 [0, 255]，边缘柔和。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_low, s_min, v_min])
    upper = np.array([h_high, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # 开运算去掉背景上的细碎绿噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # 闭运算填补人物内部/边缘的小空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # 高斯模糊做边缘羽化，避免合成后出现生硬的矩形轮廓
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.5)
    return mask.astype(np.float32)


def composite_roi(fg, alpha, bg, x, y):
    """把前景图 fg（带 alpha）贴到背景 bg 的 (x, y) 位置。"""
    fh, fw = fg.shape[:2]
    bh, bw = bg.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(bw, x + fw), min(bh, y + fh)
    if x1 <= x0 or y1 <= y0:
        return bg
    sx0, sy0 = x0 - x, y0 - y
    sx1, sy1 = x1 - x, y1 - y

    a = (alpha[sy0:sy1, sx0:sx1, None] / 255.0).astype(np.float32)
    f = fg[sy0:sy1, sx0:sx1].astype(np.float32)
    b = bg[y0:y1, x0:x1].astype(np.float32)
    out = f * a + b * (1 - a)
    bg[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    return bg


def extract_foreground(img):
    """从单张绿幕图提取前景与 alpha（单通道、边缘羽化）。

    mask 在绿色像素处为 255，而 alpha 应在人物处为 255、绿色处为 0，
    因此这里取反后再作为前景权重。
    """
    mask = green_mask(img)
    alpha = (255 - mask).astype(np.uint8)
    fg = (img.astype(np.float32) * (alpha[..., None] / 255.0)).astype(np.uint8)
    return fg, alpha


def frame_background(bg, t, duration, fps):
    """第 t 帧的背景：对原背景做横向平移取景（模拟镜头移动）。"""
    fh, fw = H, W
    bh, bw = bg.shape[:2]
    max_off = max(0, bw - fw)
    offset = int(max_off * (t / duration))
    crop = bg[0:fh, offset:offset + fw]
    if crop.shape[1] < fw:
        crop = np.pad(crop, ((0, 0), (0, fw - crop.shape[1]), (0, 0)))
    return crop.copy()


def make_demo_video(fg, alpha, bg, output, duration=4, fps=25):
    """演示模式：前景上下弹跳 + 背景横向平移，合成一段视频。"""
    fh, fw = fg.shape[:2]
    total = int(duration * fps)

    writer = _unicode_video_writer(output, fps, (W, H))
    if writer is None:
        raise RuntimeError(f"无法创建视频文件: {output}")

    base_y = 30
    amp = 40
    for i in range(total):
        t = i / total
        # 背景帧：横向平移
        bgf = frame_background(bg, t, total, fps)
        # 前景位置：左右小幅摇摆 + 上下弹跳
        sway = int(np.sin(t * np.pi * 2 * 2) * 25)
        bounce = int(abs(np.sin(t * np.pi * 2 * 3)) * amp)
        x = (W - fw) // 2 + sway
        y = base_y + bounce
        frame = composite_roi(fg, alpha, bgf, x, y)
        writer.write(frame)

    writer.release()
    _finalize_unicode_video(output)


def make_video_from_greenscreen(greenscreen, bg, output, fps=25):
    """把绿幕视频逐帧抠图后合成到背景上，输出合成视频。"""
    cap = cv2.VideoCapture(str(greenscreen))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开绿幕视频: {greenscreen}")
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    bg = cv2.resize(bg, (fw, fh))

    writer = _unicode_video_writer(output, src_fps, (fw, fh))
    if writer is None:
        raise RuntimeError(f"无法创建视频文件: {output}")

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fg, alpha = extract_foreground(frame)
        composite = composite_roi(fg, alpha, bg.copy(), 0, 0)
        writer.write(composite)
        n += 1
    cap.release()
    writer.release()
    _finalize_unicode_video(output)
    print(f"已处理 {n} 帧，输出: {output}")


def _unicode_video_writer(path, fps, size):
    """VideoWriter 不支持中文路径，先写临时 ASCII 文件，最后改名。"""
    path = Path(path)
    if str(path).isascii():
        return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    tmp = Path(tempfile.mkdtemp()) / path.name
    _unicode_video_writer._tmp = (tmp, path)  # 记录临时位置，release 后改名
    return cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


def _finalize_unicode_video(path):
    tmp, final = getattr(_unicode_video_writer, "_tmp", (None, None))
    if tmp is not None and tmp.exists():
        tmp.replace(final)


def main():
    ap = argparse.ArgumentParser(description="绿幕抠图合成系统")
    ap.add_argument("--image", help="绿幕前景图片路径（演示模式）")
    ap.add_argument("--video", help="绿幕视频路径（逐帧抠图模式）")
    ap.add_argument("--background", required=True, help="自定义背景图片路径")
    ap.add_argument("--output", default="../output/chroma_key_demo.mp4", help="输出视频路径")
    ap.add_argument("--duration", type=float, default=4.0, help="演示模式视频时长（秒）")
    ap.add_argument("--fps", type=int, default=25, help="演示模式帧率")
    args = ap.parse_args()

    bg = imread(args.background)
    if bg is None:
        raise SystemExit(f"无法读取背景图片: {args.background}")
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.video:
        make_video_from_greenscreen(args.video, bg, out, args.fps)
    elif args.image:
        img = imread(args.image)
        if img is None:
            raise SystemExit(f"无法读取绿幕图片: {args.image}")
        fg, alpha = extract_foreground(img)
        make_demo_video(fg, alpha, bg, out, args.duration, args.fps)

        # 额外保存一张合成关键帧，方便仓库里直接查看效果
        key = (out.parent / (out.stem + "_keyframe.png")).resolve()
        bg0 = frame_background(bg, 0, int(args.duration * args.fps), args.fps)
        frame = composite_roi(fg, alpha, bg0, (W - fg.shape[1]) // 2, 40)
        imwrite(key, frame)
        print("合成关键帧已保存:", key)
    else:
        raise SystemExit("请提供 --image 或 --video 之一")

    print("合成视频已保存:", out)


if __name__ == "__main__":
    main()
