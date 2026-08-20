"""缺陷分类器模块。

提供两种分类器实现：

    SVMClassifier
        基于 scikit-learn SVM，使用手工纹理特征向量。
        小样本场景下表现稳定，无需 GPU。

    MobileNetClassifier
        基于 PyTorch MobileNetV3-Small，使用原始图像块。
        数据充足时可实现更高精度，支持像素级定位。

两者共享 DefectClassifier 抽象基类，可在配置文件中切换。
"""

import os
import pickle
import numpy as np
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# 缺陷类别定义
# ============================================================================

DEFECT_CLASSES = [
    "ok",                # 0: 无缺陷
    "oxidation",         # 1: 氧化斑
    "embedding",         # 2: 磨料嵌入
    "unroughened",       # 3: 未粗化
    "scratch",           # 4: 划痕
    "mixed",             # 5: 混合缺陷
]

# 类别权重（用于加权 F1）
CLASS_WEIGHTS = {
    "ok": 1.0,
    "oxidation": 1.2,
    "embedding": 1.0,
    "unroughened": 1.5,
    "scratch": 1.3,
    "mixed": 1.0,
}


# ============================================================================
# 抽象基类
# ============================================================================

class DefectClassifier(ABC):
    """缺陷分类器抽象基类。

    所有分类器必须实现 classify() 和 train() 方法。
    """

    def __init__(self, config: dict):
        self.config = config
        classifier_cfg = config.get("inspection", {}).get("classifier", {})
        self.conf_thresh = classifier_cfg.get("confidence_threshold", 0.5)

    @abstractmethod
    def classify(self, features: np.ndarray) -> Tuple[str, float]:
        """对特征向量进行分类。

        Args:
            features: 特征向量 (n_features,) 或 (n_samples, n_features)。

        Returns:
            (class_name, confidence) 元组。
            如果 n_samples > 1，返回多数投票结果。
        """
        ...

    @abstractmethod
    def train(self, X: np.ndarray, y: List[str]) -> None:
        """使用标注数据训练分类器。

        Args:
            X: 特征/图像数组，形状 (n_samples, n_features) 或 (n_samples, H, W)。
            y: 标签列表，长度 n_samples。
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型到文件。"""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """从文件加载模型。"""
        ...


# ============================================================================
# SVM 分类器
# ============================================================================

class SVMClassifier(DefectClassifier):
    """基于 SVM 的缺陷分类器。

    使用 RBF 核的 SVC，配合 StandardScaler 进行特征标准化。
    适用于中小规模数据集（100-1000 样本），对纹理特征向量效果良好。

    训练参数：
        - kernel: RBF（默认，非线性分类边界）
        - C: 正则化参数，通过网格搜索确定
        - gamma: RBF 核宽度，通过网格搜索确定
        - probability: True（输出校准概率作为置信度）
    """

    def __init__(self, config: dict):
        super().__init__(config)
        classifier_cfg = config.get("inspection", {}).get("classifier", {})
        self._svm = None
        self._scaler = StandardScaler()
        self._label_encoder = LabelEncoder()
        self._is_trained = False

        # SVM 超参数
        self.kernel = "rbf"
        self.C = 10.0
        self.gamma = "scale"
        self.probability = True

        # 网格搜索参数
        self.grid_search = False

    def classify(self, features: np.ndarray) -> Tuple[str, float]:
        """SVM 分类。

        Args:
            features: 特征向量，形状 (n_features,) 或 (n_samples, n_features)。

        Returns:
            (predicted_class, confidence)。
            如果输入是多样本（n_samples > 1），先对各样本分类再投票。
        """
        if not self._is_trained:
            raise RuntimeError("SVM 分类器未训练或未加载模型")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # 标准化
        X = self._scaler.transform(features)

        # 预测
        if X.shape[0] == 1:
            # 单个样本
            pred_id = int(self._svm.predict(X)[0])
            proba = self._svm.predict_proba(X)[0]
            confidence = float(proba[pred_id])
            class_name = self._label_encoder.inverse_transform([pred_id])[0]
        else:
            # 多样本 → 投票
            predictions = self._svm.predict(X)
            probas = self._svm.predict_proba(X)

            # 按类别加权置信度
            class_votes = {}
            for i, pred_id in enumerate(predictions):
                cname = self._label_encoder.inverse_transform([pred_id])[0]
                weight = CLASS_WEIGHTS.get(cname, 1.0) * probas[i, pred_id]
                class_votes[cname] = class_votes.get(cname, 0) + weight

            class_name = max(class_votes, key=class_votes.get)
            confidence = float(
                class_votes[class_name] / sum(class_votes.values())
            )

        return class_name, round(confidence, 4)

    def train(self, X: np.ndarray, y: List[str]) -> None:
        """训练 SVM 分类器。

        Args:
            X: 特征向量矩阵 (n_samples, n_features)。
            y: 标签列表。
        """
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn 未安装。请运行: pip install scikit-learn"
            )

        # 编码标签
        y_encoded = self._label_encoder.fit_transform(y)

        # 标准化特征
        X_scaled = self._scaler.fit_transform(X)

        # SVM
        self._svm = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=self.probability,
            class_weight="balanced",
            random_state=42,
        )

        if self.grid_search and len(y) >= 50:
            # 网格搜索最优超参数
            param_grid = {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.01, 0.1],
            }
            grid = GridSearchCV(
                self._svm, param_grid, cv=min(5, len(y) // 3),
                scoring="f1_weighted", n_jobs=-1, verbose=1,
            )
            grid.fit(X_scaled, y_encoded)
            self._svm = grid.best_estimator_
            print(f"[SVM] 最佳参数: {grid.best_params_}")
            print(f"[SVM] 最佳分数: {grid.best_score_:.4f}")
        else:
            self._svm.fit(X_scaled, y_encoded)

        self._is_trained = True
        print(f"[SVM] 训练完成 — {len(y_encoded)} 个样本, "
              f"{len(self._label_encoder.classes_)} 个类别")

    def save(self, path: str) -> None:
        """保存 SVM 模型、标准化器和标签编码器。"""
        if not self._is_trained:
            raise RuntimeError("训练后才能保存模型")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "svm": self._svm,
                "scaler": self._scaler,
                "label_encoder": self._label_encoder,
            }, f)
        print(f"[SVM] 模型已保存至: {path}")

    def load(self, path: str) -> None:
        """加载 SVM 模型。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"SVM 模型文件不存在: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._svm = data["svm"]
        self._scaler = data["scaler"]
        self._label_encoder = data["label_encoder"]
        self._is_trained = True
        print(f"[SVM] 模型已加载: {path} "
              f"({len(self._label_encoder.classes_)} 个类别)")


# ============================================================================
# MobileNet 分类器
# ============================================================================

class MobileNetClassifier(DefectClassifier):
    """基于 MobileNetV3-Small 的轻量 CNN 分类器。

    使用迁移学习策略：预训练 MobileNetV3 → 替换分类头 → 微调。
    适用于数据量充足（500+ 个标注图像块）的场景。

    训练配置：
        - 基础模型: MobileNetV3-Small（ImageNet 预训练权重）
        - 输入尺寸: 224 × 224
        - 优化器: Adam (lr=1e-4)
        - 损失函数: CrossEntropyLoss
    """

    def __init__(self, config: dict):
        super().__init__(config)
        classifier_cfg = config.get("inspection", {}).get("classifier", {})

        self.num_classes = len(DEFECT_CLASSES)
        self.input_size = (224, 224)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: Optional[nn.Module] = None
        self._is_trained = False

        # 图像预处理
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._build_model()

    def _build_model(self) -> None:
        """构建 MobileNetV3-Small 模型。"""
        if not HAS_TORCH:
            raise ImportError("PyTorch 未安装")
        self._model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        # 替换分类头
        in_features = self._model.classifier[-1].in_features
        self._model.classifier[-1] = nn.Linear(
            in_features, self.num_classes
        )
        self._model = self._model.to(self._device)

    # ------------------------------------------------------------------
    # 分类
    # ------------------------------------------------------------------

    def classify(self, image: np.ndarray) -> Tuple[str, float]:
        """对图像块进行分类。

        Args:
            image: 图像块 (H, W, 3) RGB uint8 或 (B, H, W, 3)。

        Returns:
            (class_name, confidence)。
        """
        if not self._is_trained:
            raise RuntimeError("MobileNet 分类器未训练或未加载模型")

        self._model.eval()

        # 预处理
        if image.ndim == 4:
            # 批量
            tensors = torch.stack([
                self._transform(img) for img in image
            ]).to(self._device)
        else:
            tensors = self._transform(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            outputs = self._model(tensors)
            probs = torch.softmax(outputs, dim=1)

        if image.ndim == 4:
            # 多样本投票
            mean_probs = probs.mean(dim=0)
            pred_id = int(torch.argmax(mean_probs))
            confidence = float(mean_probs[pred_id])
        else:
            pred_id = int(torch.argmax(probs[0]))
            confidence = float(probs[0, pred_id])

        return DEFECT_CLASSES[pred_id], round(confidence, 4)

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: List[str],
        epochs: int = 30,
        batch_size: int = 16,
        lr: float = 1e-4,
        val_split: float = 0.2,
    ) -> None:
        """微调 MobileNet 分类器。

        Args:
            X: 图像数组 (n_samples, H, W, 3)。
            y: 标签列表。
            epochs: 训练轮数。
            batch_size: 批大小。
            lr: 学习率。
            val_split: 验证集比例。
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch 未安装")

        # 标签编码
        label_to_idx = {name: i for i, name in enumerate(DEFECT_CLASSES)}
        y_idx = np.array([label_to_idx.get(lbl, 0) for lbl in y])

        # 划分数据集
        indices = np.arange(len(X))
        train_idx, val_idx = train_test_split(
            indices, test_size=val_split, stratify=y_idx,
            random_state=42,
        )

        # 创建 DataLoader
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim

        X_train = torch.stack([
            self._transform(X[i]) for i in train_idx
        ])
        y_train = torch.tensor(y_idx[train_idx], dtype=torch.long)
        X_val = torch.stack([
            self._transform(X[i]) for i in val_idx
        ])
        y_val = torch.tensor(y_idx[val_idx], dtype=torch.long)

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)

        # 优化器和损失函数
        optimizer = optim.Adam(self._model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.CrossEntropyLoss()

        # 训练循环
        best_acc = 0.0
        for epoch in range(epochs):
            # 训练
            self._model.train()
            train_loss = 0.0
            for bx, by in train_dl:
                bx, by = bx.to(self._device), by.to(self._device)
                optimizer.zero_grad()
                loss = criterion(self._model(bx), by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            # 验证
            self._model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for bx, by in val_dl:
                    bx, by = bx.to(self._device), by.to(self._device)
                    outputs = self._model(bx)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == by).sum().item()
                    total += by.size(0)
            acc = correct / total

            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"[MobileNet] Epoch {epoch+1}/{epochs} — "
                      f"Loss: {train_loss/len(train_dl):.4f}, "
                      f"Val Acc: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc

        self._is_trained = True
        print(f"[MobileNet] 训练完成 — 最佳验证精度: {best_acc:.4f}")

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """保存 MobileNet 权重。"""
        if not self._is_trained:
            raise RuntimeError("训练后才能保存模型")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "num_classes": self.num_classes,
        }, path)
        print(f"[MobileNet] 模型已保存至: {path}")

    def load(self, path: str) -> None:
        """加载 MobileNet 权重。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"MobileNet 模型文件不存在: {path}")
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._is_trained = True
        print(f"[MobileNet] 模型已加载: {path}")


# ============================================================================
# 工厂函数
# ============================================================================

def create_classifier(config: dict) -> DefectClassifier:
    """根据配置创建分类器实例。

    配置文件中的 classifier.type 决定:
        - "svm" → SVMClassifier
        - "mobilenet" → MobileNetClassifier
    """
    classifier_cfg = config.get("inspection", {}).get("classifier", {})
    clf_type = classifier_cfg.get("type", "svm")

    if clf_type == "svm":
        return SVMClassifier(config)
    elif clf_type == "mobilenet":
        return MobileNetClassifier(config)
    else:
        raise ValueError(f"不支持的分���器类型: {clf_type}。可选: svm, mobilenet")
