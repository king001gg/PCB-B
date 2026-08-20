"""纹理特征提取模块单元测试。"""

import numpy as np
import pytest
import yaml
from pathlib import Path

from core.texture import (
    GLCMExtractor, LBPExtractor, GaborFilterBank,
    TextureAnalyzer, GLCMFeatures, TextureFeatureVector,
)


@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def gray_image():
    """200×200 灰度测试图像，带明暗条纹。"""
    img = np.zeros((200, 200), dtype=np.uint8)
    for i in range(200):
        stripe = 128 + 64 * np.sin(i * 0.1)
        img[i, :] = int(stripe)
    img += np.random.randint(0, 10, img.shape, dtype=np.uint8)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class TestGLCMExtractor:
    """GLCM 特征提取器测试。"""

    def test_compute_returns_features(self, gray_image):
        """计算应返回 GLCMFeatures 列表。"""
        glcm = GLCMExtractor(distances=[1], angles=[0])
        features = glcm.compute(gray_image)
        assert len(features) == 1
        f = features[0]
        assert isinstance(f, GLCMFeatures)
        assert f.contrast >= 0
        assert 0 <= f.energy <= 1
        assert f.entropy >= 0
        assert 0 <= f.homogeneity <= 1

    def test_multiple_distances_angles(self, gray_image):
        """多距离多角度应返回对应数量的特征。"""
        glcm = GLCMExtractor(
            distances=[1, 3], angles=[0, 45],
        )
        features = glcm.compute(gray_image)
        assert len(features) == 4  # 2 × 2

    def test_mean_features(self, gray_image):
        """均值特征应聚合所有方向。"""
        glcm = GLCMExtractor()
        mean_f = glcm.mean_features(gray_image)
        assert isinstance(mean_f, GLCMFeatures)
        assert mean_f.contrast >= 0

    def test_feature_vector_shape(self, gray_image):
        """特征向量维度应正确。"""
        glcm = GLCMExtractor(distances=[1, 3], angles=[0, 45, 90, 135])
        features = glcm.compute(gray_image)
        vec = features[0].as_vector()
        assert vec.shape == (6,)  # 6 个 GLCM 特征

    def test_none_input_raises(self):
        """None 输入应抛出异常。"""
        glcm = GLCMExtractor()
        with pytest.raises(ValueError):
            glcm.compute(None)


class TestLBPExtractor:
    """LBP 特征提取器测试。"""

    def test_compute_uniform(self, gray_image):
        """uniform LBP 计算。"""
        lbp = LBPExtractor(radius_list=[1], n_points_list=[8],
                            method="uniform")
        result = lbp.compute(gray_image)
        assert result is not None
        assert result.shape == gray_image.shape
        # uniform LBP 最多有 n_points + 2 个模式
        assert result.max() <= 10  # 8 + 2

    def test_histogram(self, gray_image):
        """LBP 直方图应归一化。"""
        lbp = LBPExtractor(radius_list=[1], n_points_list=[8])
        lbp_img = lbp.compute(gray_image)
        hist = lbp.histogram(lbp_img)
        assert abs(hist.sum() - 1.0) < 0.01
        assert hist.min() >= 0

    def test_multi_radius_histogram(self, gray_image):
        """多半径直方图应拼接所有半径。"""
        lbp = LBPExtractor(
            radius_list=[1, 2], n_points_list=[8, 16],
        )
        multi_hist = lbp.multi_radius_histogram(gray_image)
        assert multi_hist is not None
        assert len(multi_hist) > 0

    def test_unequal_lists_raises(self):
        """不一致的半径/点数列表应抛出异常。"""
        with pytest.raises(ValueError):
            LBPExtractor(radius_list=[1, 2], n_points_list=[8])


class TestGaborFilterBank:
    """Gabor 滤波器组测试。"""

    def test_init_builds_kernels(self):
        """初始化应预生成滤波核。"""
        gb = GaborFilterBank(orientations=4, scales=[3], frequencies=[0.3])
        # 1 scale × 1 freq × 4 orientations = 4 kernels
        assert gb.n_kernels == 4

    def test_filter_returns_responses(self, gray_image):
        """滤波应返回对应数量的响应图。"""
        gb = GaborFilterBank(orientations=4, scales=[3], frequencies=[0.3])
        responses = gb.filter(gray_image)
        assert len(responses) == gb.n_kernels
        assert responses[0].shape == gray_image.shape

    def test_energy_map(self, gray_image):
        """能量图尺寸应匹配输入。"""
        gb = GaborFilterBank(orientations=4, scales=[3], frequencies=[0.3])
        energy = gb.energy_map(gray_image)
        assert energy.shape == gray_image.shape
        assert energy.min() >= 0

    def test_direction_consistency_range(self, gray_image):
        """DCI 应在 [0, 1] 范围内。"""
        gb = GaborFilterBank(orientations=8)
        dci = gb.direction_consistency(gray_image)
        assert 0.0 <= dci <= 1.0

    def test_feature_vector(self, gray_image):
        """特征向量维度 = 2 × n_kernels。"""
        gb = GaborFilterBank(orientations=2, scales=[3], frequencies=[0.3])
        fv = gb.feature_vector(gray_image)
        assert fv.shape == (2 * gb.n_kernels,)


class TestTextureAnalyzer:
    """TextureAnalyzer 集成测试。"""

    def test_analyze_returns_vector(self, config, gray_image):
        """analyze() 应返回 TextureFeatureVector。"""
        ta = TextureAnalyzer(config)
        result = ta.analyze(gray_image)
        assert isinstance(result, TextureFeatureVector)
        assert len(result.glcm) > 0
        assert result.lbp_histogram is not None
        assert result.gabor_features is not None

    def test_cv_heatmap(self, config, gray_image):
        """CV 热力图应在 [0,1] 范围内。"""
        ta = TextureAnalyzer(config)
        heatmap = ta.compute_cv_heatmap(gray_image)
        assert heatmap.shape == gray_image.shape
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1

    def test_local_entropy(self, config, gray_image):
        """局部熵图应非负。"""
        ta = TextureAnalyzer(config)
        entropy = ta.compute_local_entropy(gray_image)
        assert entropy.shape == gray_image.shape
        assert entropy.min() >= 0

    def test_direction_consistency(self, config, gray_image):
        """方向一致性应在 [0,1] 范围。"""
        ta = TextureAnalyzer(config)
        dci = ta.direction_consistency(gray_image)
        assert 0.0 <= dci <= 1.0

    def test_flatten(self, config, gray_image):
        """flatten() 应返回一维向量。"""
        ta = TextureAnalyzer(config)
        tv = ta.analyze(gray_image)
        flat = tv.flatten()
        assert flat.ndim == 1
        assert len(flat) > 0
