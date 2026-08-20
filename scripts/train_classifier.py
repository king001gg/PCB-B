"""缺陷分类器训练脚本。

支持两种模式：
    (1) SVM 训练 — 使用手工纹理特征向量（GLCM+LBP+Gabor）
    (2) MobileNet 训练 — 使用原始图像块

用法：
    python scripts/train_classifier.py --mode svm --data data/training/
    python scripts/train_classifier.py --mode mobilenet --data data/training/ --epochs 50
"""

import os
import sys
import argparse
import numpy as np
import pickle
import glob
import json
from pathlib import Path

# 确保项目根在搜索路径上
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import yaml

from core.preprocessing import Preprocessor
from core.texture import TextureAnalyzer
from core.classifier import SVMClassifier, MobileNetClassifier, create_classifier


def load_config():
    """加载配置文件。"""
    config_path = PROJECT_ROOT / "config" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(data_dir: str):
    """从标注数据集目录加载图像和标签。

    数据集目录结构：
        data/training/
            ok/               ← OK 样本
            oxidation/        ← 氧化斑样本
            embedding/        ← 磨料嵌入样本
            unroughened/      ← 未粗化样本
            scratch/          ← 划痕样本
            或
        data/training/
            annotations.json  ← COCO 格式标注文件
            *.jpg / *.png     ← 图像文件

    Returns:
        X: np.ndarray 图像列表（RGB, H, W, 3）
        y: List[str] 标签列表
    """
    data_path = Path(data_dir)
    X, y = [], []

    # 尝试按文件夹加载
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    if class_dirs:
        print(f"发现 {len(class_dirs)} 个类别目录")
        for class_dir in class_dirs:
            class_name = class_dir.name
            image_files = list(class_dir.glob("*.jpg")) + \
                          list(class_dir.glob("*.jpeg")) + \
                          list(class_dir.glob("*.png")) + \
                          list(class_dir.glob("*.bmp"))

            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                y.append(class_name)

            print(f"  {class_name}: {len(image_files)} 个样本")
    else:
        # 尝试从 annotations.json 加载
        ann_file = data_path / "annotations.json"
        if ann_file.exists():
            with open(ann_file, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            for ann in annotations:
                img_path = data_path / ann["image"]
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                y.append(ann["class_name"])
            print(f"从 {ann_file} 加载 {len(X)} 个样本")
        else:
            raise FileNotFoundError(
                f"数据集目录 {data_dir} 中未找到类别子目录或 annotations.json"
            )

    if len(X) == 0:
        raise ValueError(f"未找到任何图像文件: {data_dir}")

    return np.array(X, dtype=object), y


def extract_features(images: np.ndarray, config: dict) -> np.ndarray:
    """从图像列表提取纹理特征向量。

    流程：预处理 → 纹理分析 → 拼接特征向量
    """
    print("提取纹理特征...")
    preprocessor = Preprocessor(config)
    analyzer = TextureAnalyzer(config)

    feature_vectors = []
    total = len(images)
    for i, img in enumerate(images):
        # 预处理
        gray = preprocessor.process(img)
        # 纹理分析
        tv = analyzer.analyze(gray)
        # 拼接特征向量
        fv = tv.flatten()
        feature_vectors.append(fv)

        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"  进度: {i+1}/{total} — 特征维度: {len(fv)}")

    X = np.array(feature_vectors)
    print(f"特征矩阵: {X.shape}")
    return X


def train_svm(config: dict, data_dir: str, output_dir: str):
    """训练 SVM 分类器。"""
    images, labels = load_dataset(data_dir)
    X = extract_features(images, config)

    clf = SVMClassifier(config)
    clf.grid_search = True  # 启用网格搜索
    clf.train(X, labels)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "svm_classifier.pkl")
    scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
    clf.save(model_path)

    # 保存标准化器参数也独立一份（用于特征提取管道）
    with open(scaler_path, "wb") as f:
        pickle.dump({
            "scaler": clf._scaler,
            "label_encoder": clf._label_encoder,
        }, f)

    print(f"\nSVM 分类器已保存至: {model_path}")
    print(f"标准化器已保存至: {scaler_path}")


def train_mobilenet(config: dict, data_dir: str, output_dir: str,
                     epochs: int, batch_size: int, lr: float):
    """训练 MobileNet 分类器。"""
    images, labels = load_dataset(data_dir)

    # MobileNet 使用原始图像块，仅需 resize
    clf = MobileNetClassifier(config)
    clf.train(images, labels, epochs=epochs, batch_size=batch_size, lr=lr)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "mobilenet_classifier.pt")
    clf.save(model_path)
    print(f"\nMobileNet 分类器已保存至: {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="训练 PCB 喷砂表面缺陷分类器"
    )
    parser.add_argument(
        "--mode", choices=["svm", "mobilenet"], default="svm",
        help="分类器类型（默认: svm）"
    )
    parser.add_argument(
        "--data", default="data/training",
        help="训练数据目录（默认: data/training）"
    )
    parser.add_argument(
        "--output", default="models",
        help="模型输出目录（默认: models）"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="训练轮数（仅 MobileNet，默认: 30）"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="批大小（仅 MobileNet，默认: 16）"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="学习率（仅 MobileNet，默认: 1e-4）"
    )
    parser.add_argument(
        "--config", default=None,
        help="配置文件路径（默认: config/default.yaml）"
    )

    args = parser.parse_args()

    # 加载配置
    config_path = args.config or str(PROJECT_ROOT / "config" / "default.yaml")
    if not os.path.exists(config_path):
        print(f"[WARN] 配置文件不存在: {config_path}")
        print("正在创建一个默认配置...")
        config = {
            "system": {"resolution_mm_per_pixel": 0.01},
            "inspection": {
                "preprocessing": {
                    "retinex": {"enabled": True, "sigma": [15, 80, 250], "gain": 128, "offset": 128},
                    "clahe": {"enabled": True, "clip_limit": 2.0, "tile_size": [8, 8]},
                },
                "roi": {"enabled": False},
                "texture": {
                    "glcm": {"distances": [1, 3, 5], "angles": [0, 45, 90, 135], "levels": 256, "symmetric": True, "normalize": True},
                    "lbp": {"radius_list": [1, 2, 3], "n_points_list": [8, 16, 24], "method": "uniform"},
                    "gabor": {"orientations": 8, "scales": [3, 5, 7], "frequencies": [0.1, 0.3, 0.5]},
                    "cv_window": 32, "local_entropy_window": 9,
                },
                "classifier": {"type": "svm", "confidence_threshold": 0.5},
            },
        }
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    print(f"模式: {args.mode}")
    print(f"数据: {args.data}")

    if args.mode == "svm":
        train_svm(config, args.data, args.output)
    else:
        train_mobilenet(
            config, args.data, args.output,
            args.epochs, args.batch_size, args.lr,
        )


if __name__ == "__main__":
    main()
