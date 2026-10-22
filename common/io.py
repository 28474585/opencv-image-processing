"""Unicode 安全的图像读写封装。

Windows 下 cv2.imread / cv2.imwrite 对包含中文等非 ASCII 字符的路径
会失败（返回 None / 不写入）。这里借助 np.fromfile / imencode 绕过，
保证在中文路径下也能正常工作。
"""

from pathlib import Path

import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)


def imwrite(path, img, params=None):
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix, img, params if params else [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def read_text(path, encoding="utf-8"):
    return Path(path).read_text(encoding=encoding)


def write_text(path, text, encoding="utf-8"):
    Path(path).write_text(text, encoding=encoding)
