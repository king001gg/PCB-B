"""
生成喷砂后 PCB 铜面合成样本图像。

模拟喷砂工艺形成的微观粗糙表面纹理，包含：
    - 随机锚纹（多方向微型凹痕 + 凸起）
    - 铜面金色基底（带 Perlin 噪声模拟真实轧制纹理）
    - 可注入的缺陷（氧化斑、磨料嵌入、未粗化区域、划痕）

输出格式：
    - 合成图像（JPEG）
    - 标注文件（JSON，含缺陷 bbox 和类别）

用途：
    - 分类器训练数据
    - 算法开发与调试
    - 系统演示

用法：
    python scripts/generate_sandblasted_samples.py --count 20 --output data/training/

依赖:
    - opencv-python, numpy
    - noise (pip install noise) — 可选，回退到简单实现
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Callable

import cv2
import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# 配置
# ============================================================================

SEED = 42
IMAGE_SIZE = (1024, 768)  # W, H

# 颜色（BGR）
COLORS = {
    "copper_light": (85, 185, 230),    # 亮铜色
    "copper_dark": (45, 140, 175),      # 暗铜色
    "copper_base": (65, 160, 200),      # 基础金色
    "oxidation_brown": (40, 90, 140),   # 氧化棕色
    "oxidation_dark": (20, 50, 100),    # 氧化深色
    "abrasive_white": (240, 240, 235),  # 磨料颗粒（亮白）
    "unroughened_glossy": (95, 200, 240),  # 未粗化光泽
    "scratch_dark": (30, 60, 110),      # 划痕
}

# 缺陷类型
DEFECT_TYPES = ["oxidation", "embedding", "unroughened", "scratch"]


# ============================================================================
# Perlin 噪声生成器
# ============================================================================

class PerlinNoise:
    """简化 Perlin 噪声实现（无需 noise 库）。

    用于生成逼真的铜面微纹理。
    """

    def __init__(self, seed: int = 42):
        self.permutation = list(range(256))
        random.seed(seed)
        random.shuffle(self.permutation)
        self.permutation = self.permutation * 2

    def _fade(self, t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    def _grad(self, hash_val: int, x: float, y: float) -> float:
        h = hash_val & 3
        u = x if h < 2 else y
        v = y if h < 2 else x
        return (u if h & 1 == 0 else -u) + (v if h & 2 == 0 else -v)

    def noise2d(self, x: float, y: float) -> float:
        """2D Perlin 噪声值 [−1, 1]。"""
        X = int(np.floor(x)) & 255
        Y = int(np.floor(y)) & 255
        x -= np.floor(x)
        y -= np.floor(y)

        u = self._fade(x)
        v = self._fade(y)

        a = self.permutation[X] + Y
        b = self.permutation[X + 1] + Y

        return self._lerp(
            self._lerp(
                self._grad(self.permutation[a], x, y),
                self._grad(self.permutation[b], x - 1, y),
                u,
            ),
            self._lerp(
                self._grad(self.permutation[a + 1], x, y - 1),
                self._grad(self.permutation[b + 1], x - 1, y - 1),
                u,
            ),
            v,
        )

    def fractal_noise(self, x: float, y: float, octaves: int = 4,
                      lacunarity: float = 2.0, gain: float = 0.5) -> float:
        """分形噪声（多倍频叠加）。"""
        value = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(octaves):
            value += amplitude * self.noise2d(x * frequency, y * frequency)
            max_value += amplitude
            amplitude *= gain
            frequency *= lacunarity

        return value / max_value


# ============================================================================
# 喷砂纹理生成
# ============================================================================

def generate_sandblasted_surface(
    width: int = 1024, height: int = 768, seed: int = None,
) -> np.ndarray:
    """生成带喷砂锚纹的铜面纹理。

    通过多层噪声叠加 + 随机划痕模拟喷砂效果：
        1. 铜面基础色（渐变 + 低频 Perlin 噪声）
        2. 锚纹（高频 Perlin + 随机方向划痕）
        3. 高斯噪声（模拟相机传感器噪声）

    Args:
        width: 图像宽度。
        height: 图像高度。
        seed: 随机种子。

    Returns:
        BGR 图像 (height, width, 3), uint8。
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    pn = PerlinNoise(seed or 42)
    image = np.zeros((height, width, 3), dtype=np.float64)

    # --- 1. 铜面基础色 ---
    for y in range(height):
        for x in range(width):
            # 低频 Perlin → 铜面色调变化
            base = 0.5 + 0.5 * pn.fractal_noise(
                x * 0.005, y * 0.005, octaves=3,
            )
            # 铜色渐变（中心略亮）
            dx = (x / width - 0.5) ** 2
            dy = (y / height - 0.5) ** 2
            vignette = 1.0 - 0.15 * (dx + dy)
            factor = base * vignette

            # 在亮铜和暗铜之间插值
            color = tuple(
                int(COLORS["copper_dark"][i] +
                    factor * (COLORS["copper_light"][i] -
                              COLORS["copper_dark"][i]))
                for i in range(3)
            )
            image[y, x] = color

    # --- 2. 锚纹（高频纹理） ---
    # 使用高频 Perlin 噪声调制像素亮度
    for y in range(height):
        for x in range(width):
            anchor = pn.fractal_noise(
                x * 0.08, y * 0.08, octaves=5, gain=0.6,
            )
            image[y, x] += anchor * 15  # ±15 的亮度调制

    # 随机方向微划痕（模拟喷砂撞击痕迹）
    n_scratches = int(width * height * 0.0003)  # ~230 per 1024x768
    for _ in range(n_scratches):
        cx = random.randint(0, width - 1)
        cy = random.randint(0, height - 1)
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(2, 8)
        thickness = random.randint(1, 2)
        ex = cx + int(length * np.cos(angle))
        ey = cy + int(length * np.sin(angle))
        # 随机明暗
        brightness = random.randint(-10, 5)
        cv2.line(
            image.astype(np.int16),
            (cx, cy), (ex, ey),
            (brightness, brightness, brightness),
            thickness,
        )

    # --- 3. 传感器噪声 ---
    noise = np.random.normal(0, 3, (height, width, 3))
    image += noise

    # --- 裁剪到 [0, 255] ---
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


# ============================================================================
# 缺陷注入器
# ============================================================================

def inject_oxidation(
    image: np.ndarray, count: int = None,
) -> List[dict]:
    """注入氧化斑缺陷。

    氧化斑呈现为深棕色不规则区域，面积 50-500 px²。
    """
    if count is None:
        count = random.randint(0, 3)

    h, w = image.shape[:2]
    annotations = []

    for _ in range(count):
        # 随机位置
        cx = random.randint(100, w - 100)
        cy = random.randint(100, h - 100)
        radius = random.randint(8, 30)

        # 不规则形状（随机多边形）
        n_pts = random.randint(8, 16)
        pts = []
        for i in range(n_pts):
            angle = 2 * np.pi * i / n_pts
            r = radius * random.uniform(0.5, 1.3)
            px = int(cx + r * np.cos(angle))
            py = int(cy + r * np.sin(angle))
            pts.append((px, py))

        pts_arr = np.array(pts, dtype=np.int32)

        # 氧化颜色（深棕渐变）
        ox_color = random.choice([
            COLORS["oxidation_brown"],
            COLORS["oxidation_dark"],
        ])

        # 填充
        cv2.fillPoly(image, [pts_arr], ox_color)
        # 边缘羽化
        cv2.polylines(image, [pts_arr], True,
                       COLORS["copper_dark"], 1)

        x1 = max(0, cx - radius - 4)
        y1 = max(0, cy - radius - 4)
        x2 = min(w, cx + radius + 4)
        y2 = min(h, cy + radius + 4)

        annotations.append({
            "bbox": [x1, y1, x2, y2],
            "class_name": "oxidation",
        })

    return annotations


def inject_abrasive_embedding(
    image: np.ndarray, count: int = None,
) -> List[dict]:
    """注入磨料嵌入缺陷。

    磨料颗粒呈现为微小亮点（~3-8 px），模拟嵌入铜面的
    氧化铝/碳化硅颗粒的光反射。
    """
    if count is None:
        count = random.randint(0, 15)

    h, w = image.shape[:2]
    annotations = []

    for _ in range(count):
        cx = random.randint(50, w - 50)
        cy = random.randint(50, h - 50)
        radius = random.randint(1, 4)

        # 微小圆形亮点
        brightness = random.randint(220, 255)
        color = (brightness, brightness, brightness - 10)
        cv2.circle(image, (cx, cy), radius, color, -1)
        cv2.circle(image, (cx, cy), radius,
                    COLORS["copper_light"], 1)

        x1 = cx - radius - 2
        y1 = cy - radius - 2
        x2 = cx + radius + 2
        y2 = cy + radius + 2

        annotations.append({
            "bbox": [x1, y1, x2, y2],
            "class_name": "embedding",
        })

    return annotations


def inject_unroughened_area(
    image: np.ndarray, count: int = None,
) -> List[dict]:
    """注入未粗化区域缺陷。

    未粗化区域呈现为光滑的铜面（缺少锚纹），
    比周围喷砂纹理更亮、更均匀。
    """
    if count is None:
        count = random.randint(0, 2)

    h, w = image.shape[:2]
    annotations = []

    for _ in range(count):
        cx = random.randint(150, w - 150)
        cy = random.randint(150, h - 150)

        # 椭圆光滑区域
        rx = random.randint(20, 60)
        ry = random.randint(15, 40)
        angle = random.randint(0, 180)

        # 创建光滑 patch
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)

        # 用亮铜色填充
        smooth_color = COLORS["unroughened_glossy"]
        image[mask > 0] = smooth_color
        # 边缘过渡
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_dilated = cv2.dilate(mask, kernel) - mask
        image[mask_dilated > 0] = COLORS["copper_base"]

        x1 = max(0, cx - rx - 5)
        y1 = max(0, cy - ry - 5)
        x2 = min(w, cx + rx + 5)
        y2 = min(h, cy + ry + 5)

        annotations.append({
            "bbox": [x1, y1, x2, y2],
            "class_name": "unroughened",
        })

    return annotations


def inject_scratch(
    image: np.ndarray, count: int = None,
) -> List[dict]:
    """注入划痕缺陷。

    划痕呈现为线性暗痕，模拟机械损伤。
    """
    if count is None:
        count = random.randint(0, 2)

    h, w = image.shape[:2]
    annotations = []

    for _ in range(count):
        x1 = random.randint(50, w - 100)
        y1 = random.randint(50, h - 100)
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(30, 150)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        thickness = random.randint(1, 3)

        cv2.line(image, (x1, y1), (x2, y2),
                  COLORS["scratch_dark"], thickness)

        # 轻微的高光边缘
        cv2.line(image, (x1 + 1, y1 + 1), (x2 + 1, y2 + 1),
                  COLORS["copper_light"], 1)

        annotations.append({
            "bbox": [
                min(x1, x2) - 5, min(y1, y2) - 5,
                max(x1, x2) + 5, max(y1, y2) + 5,
            ],
            "class_name": "scratch",
        })

    return annotations


# ============================================================================
# 样本生成器
# ============================================================================

def generate_sample(
    board_id: str = "sample_001",
    seed: int = None,
) -> Tuple[np.ndarray, List[dict]]:
    """生成一个完整的喷砂 PCB 样本（含随机缺陷）。

    Args:
        board_id: 样本编号。
        seed: 随机种子（实现可复现性）。

    Returns:
        (image_BGR, annotations_list)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # 1. 生成喷砂表面
    image = generate_sandblasted_surface(
        IMAGE_SIZE[0], IMAGE_SIZE[1], seed,
    )

    # 2. 注入缺陷（数量和类型随机）
    all_annotations = []

    for defect_type in DEFECT_TYPES:
        injectors = {
            "oxidation": inject_oxidation,
            "embedding": inject_abrasive_embedding,
            "unroughened": inject_unroughened_area,
            "scratch": inject_scratch,
        }
        injector = injectors[defect_type]
        annotations = injector(image)
        all_annotations.extend(annotations)

    return image, all_annotations


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="生成喷砂 PCB 铜面合成样本"
    )
    parser.add_argument(
        "--count", type=int, default=10,
        help="生成样本数量（默认: 10）"
    )
    parser.add_argument(
        "--output", default="data/training",
        help="输出目录（默认: data/training）"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="随机种子（默认: 42）"
    )
    parser.add_argument(
        "--size", default="1024x768",
        help="图像尺寸 WxH（默认: 1024x768）"
    )
    parser.add_argument(
        "--no-defects", action="store_true",
        help="仅生成 OK 样本（无缺陷）"
    )
    args = parser.parse_args()

    # 解析尺寸
    w_str, h_str = args.size.split("x")
    global IMAGE_SIZE
    IMAGE_SIZE = (int(w_str), int(h_str))

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # OK 样本目录
    ok_dir = output_dir / "ok"
    ok_dir.mkdir(exist_ok=True)

    # 缺陷样本目录
    defect_dirs = {}
    for dt in DEFECT_TYPES:
        d = output_dir / dt
        d.mkdir(exist_ok=True)
        defect_dirs[dt] = d

    print(f"生成 {args.count} 个合成喷砂 PCB 样本")
    print(f"尺寸: {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}")
    print(f"输出: {output_dir}")
    print()

    summary = {"ok": 0}
    for dt in DEFECT_TYPES:
        summary[dt] = 0

    for i in range(args.count):
        seed = args.seed + i
        np.random.seed(seed)
        random.seed(seed)

        image, annotations = generate_sample(
            board_id=f"sample_{i+1:04d}", seed=seed,
        )

        if args.no_defects:
            annotations = []

        # 确定保存目录
        if not annotations:
            save_dir = ok_dir
            summary["ok"] += 1
        else:
            # 放入主要缺陷类型对应的目录
            main_type = annotations[0]["class_name"]
            save_dir = defect_dirs[main_type]
            summary[main_type] += 1

        # 保存图像
        filename = f"sample_{i+1:04d}"
        img_path = save_dir / f"{filename}.jpg"
        cv2.imwrite(str(img_path), image,
                     [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 保存标注
        ann_path = save_dir / f"{filename}.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump({
                "image": f"{filename}.jpg",
                "image_size": {
                    "width": IMAGE_SIZE[0],
                    "height": IMAGE_SIZE[1],
                },
                "detections": annotations,
            }, f, indent=2, ensure_ascii=False)

        if (i + 1) % 5 == 0 or i == args.count - 1:
            print(f"  进度: {i+1}/{args.count}")

    # 打印摘要
    print(f"\n{'='*50}")
    print(f"{'类别':<20} {'数量':>8}")
    print(f"{'='*50}")
    for class_name, count in sorted(summary.items()):
        print(f"  {class_name:<18} {count:>8}")
    print(f"{'='*50}")
    print(f"\n样本已生成至: {output_dir}")
    print("目录结构:")
    for d in sorted(output_dir.iterdir()):
        if d.is_dir():
            n = len(list(d.glob("*.jpg")))
            print(f"  {d.name}/  ({n} 个样本)")


if __name__ == "__main__":
    main()
