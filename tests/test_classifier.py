"""分类器模块单元测试。"""

import numpy as np
import pytest
import yaml
from pathlib import Path

from core.classifier import SVMClassifier


@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def dummy_data():
    """生成小型模拟数据集。"""
    np.random.seed(42)
    n_samples = 60  # divisible by 4 for clean splits
    n_features = 64

    X = np.random.randn(n_samples, n_features)
    half = n_samples // 2
    quarter = n_samples // 4
    y = ["ok"] * half + ["oxidation"] * quarter + \
        ["unroughened"] * quarter
    return X, y


class TestSVMClassifier:
    """SVM 分类器测试。"""

    def test_init(self, config):
        """初始化为未训练状态。"""
        clf = SVMClassifier(config)
        assert clf._is_trained is False

    def test_train(self, config, dummy_data):
        """训练后状态应变为已训练。"""
        X, y = dummy_data
        clf = SVMClassifier(config)
        clf.train(X, y)
        assert clf._is_trained is True

    def test_classify_single(self, config, dummy_data):
        """单样本分类应返回 (class_name, confidence)。"""
        X, y = dummy_data
        clf = SVMClassifier(config)
        clf.train(X, y)

        class_name, confidence = clf.classify(X[0])
        assert isinstance(class_name, str)
        assert 0 <= confidence <= 1

    def test_classify_batch(self, config, dummy_data):
        """多样本分类应投票。"""
        X, y = dummy_data
        clf = SVMClassifier(config)
        clf.train(X, y)

        # 投喂多样本
        class_name, confidence = clf.classify(X[:5])
        assert isinstance(class_name, str)
        assert 0 <= confidence <= 1

    def test_save_load(self, config, dummy_data, tmp_path):
        """保存和加载应保持分类器一致性。"""
        X, y = dummy_data
        clf = SVMClassifier(config)
        clf.train(X, y)

        # 保存
        save_path = tmp_path / "test_svm.pkl"
        clf.save(str(save_path))

        # 加载
        clf2 = SVMClassifier(config)
        clf2.load(str(save_path))
        assert clf2._is_trained is True

        # 分类结果应一致
        c1, _ = clf.classify(X[0])
        c2, _ = clf2.classify(X[0])
        assert c1 == c2

    def test_classify_untrained_raises(self, config, dummy_data):
        """未训练时分类应报错。"""
        X, _ = dummy_data
        clf = SVMClassifier(config)
        with pytest.raises(RuntimeError):
            clf.classify(X[0])

    def test_gaussian_data(self, config):
        """使用可分离的高斯分布数据测试分类准确率。"""
        np.random.seed(42)
        n = 60
        # 类别 0：均值 (-2, 0)，类别 1：均值 (2, 0)
        X = np.vstack([
            np.random.randn(n, 2) + np.array([-2, 0]),
            np.random.randn(n, 2) + np.array([2, 0]),
        ])
        y = ["ok"] * n + ["oxidation"] * n

        clf = SVMClassifier(config)
        clf.train(X, y)

        # 简单测试样本
        test1 = np.array([-2.5, 0])
        test2 = np.array([2.5, 0])
        c1, conf1 = clf.classify(test1)
        c2, conf2 = clf.classify(test2)
        assert conf1 > 0.5, f"类别 0 置信度过低: {conf1}"
        assert conf2 > 0.5, f"类别 1 置信度过低: {conf2}"
