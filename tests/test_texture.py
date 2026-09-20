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


class TestGaborConvolutionContract:
    """锁定 Gabor 卷积的数值契约。

    ``GaborFilterBank._convolve()`` 用 ``cv2.filter2D`` 复现
    ``scipy.signal.convolve2d(mode="same", boundary="symm")``（原实现，慢
    12～21 倍，在相机分辨率下占整条流水线耗时的 86%）。这里有两处改错了
    不会报错、只会静默算错的地方，故用 scipy 原实现把语义钉住：

      * ``filter2D`` 算的是相关、``convolve2d`` 算的是卷积 —— 核必须翻转；
      * ``BORDER_REFLECT`` 对应 scipy 的 ``"symm"``（边缘样本重复）；
        少一像素的 ``BORDER_REFLECT_101`` 是 scipy 的 ``"reflect"``，不是它。

    边界带单独断言：边界模式写错时内部仍然一致，只有边缘会偏。
    scipy 仅作测试参照，运行时已不再调用。
    """

    @pytest.fixture
    def reference(self):
        """scipy 原实现，作为数值参照。"""
        scipy_signal = pytest.importorskip("scipy.signal")

        def convolve(img, kernel):
            return scipy_signal.convolve2d(
                img, kernel, mode="same", boundary="symm"
            )
        return convolve

    @pytest.fixture
    def bank(self):
        return GaborFilterBank(orientations=4, scales=[3, 5], frequencies=[0.3])

    @pytest.fixture
    def image(self):
        """确定性灰度图（固定种子），避免用随机图像做数值断言。"""
        rng = np.random.default_rng(20260920)
        y, x = np.mgrid[0:120, 0:120]
        img = 128 + 60 * np.sin(x * 0.15) + 20 * np.cos(y * 0.1)
        img = np.clip(img + rng.integers(0, 10, img.shape), 0, 255)
        return img.astype(np.uint8)

    @pytest.mark.parametrize("scale,theta,freq", [
        (3, 0.0, 0.1), (5, np.pi / 4, 0.3), (7, 3 * np.pi / 4, 0.5),
    ])
    def test_convolve_matches_scipy(self, bank, image, reference,
                                    scale, theta, freq):
        """cv2 路径应与 scipy 原实现逐点一致，边界带同样一致。"""
        kernel = bank._gabor_kernel(scale, theta, freq)
        img = bank._as_float(image)
        got = bank._convolve(img, kernel)
        want = reference(img, kernel)

        assert got.shape == want.shape
        band = max(kernel.shape)
        assert np.abs(got - want).max() < 1e-12
        assert np.abs(got[:band] - want[:band]).max() < 1e-12
        assert np.abs(got[-band:] - want[-band:]).max() < 1e-12
        assert np.abs(got[:, :band] - want[:, :band]).max() < 1e-12
        assert np.abs(got[:, -band:] - want[:, -band:]).max() < 1e-12

    def test_scan_feature_vector_matches_scipy(self, bank, image, reference):
        """scan() 的特征向量应与逐核 scipy 卷积后取 mean/std 一致。"""
        img = bank._as_float(image)
        want = []
        for kernel in bank._kernels:
            r = reference(img, kernel)
            want.append(np.mean(r))
            want.append(np.std(r))

        got, _ = bank.scan(image)
        assert np.abs(got - np.array(want)).max() < 1e-12

    def test_scan_energies_grouping(self, bank, image):
        """能量按 (freq, scale) 分组，每组成员数 = 方向数。"""
        _, energies = bank.scan(image)

        assert bank.n_groups * bank.orientations == bank.n_kernels
        assert len(energies) == bank.n_groups
        assert all(len(g) == bank.orientations for g in energies)

    def test_scan_energies_match_scipy(self, bank, image, reference):
        """各组能量应与 scipy 原实现一致，且落位到正确的组。"""
        img = bank._as_float(image)
        want = [[] for _ in range(bank.n_groups)]
        for kernel, group in zip(bank._kernels, bank._kernel_group):
            want[group].append(np.mean(reference(img, kernel) ** 2))

        _, energies = bank.scan(image)
        assert len(energies) == len(want)
        for got_group, want_group in zip(energies, want):
            assert len(got_group) == len(want_group)
            assert np.abs(np.array(got_group) - np.array(want_group)).max() < 1e-12

    def test_direction_consistency_reuses_energies(self, bank, image):
        """传入 scan() 的能量应与自行扫描逐位相同（复用不得改变结果）。"""
        _, energies = bank.scan(image)
        assert (bank.direction_consistency(image, energies)
                == bank.direction_consistency(image))

    def test_energy_map_matches_filter(self, bank, image):
        """流式累加的能量图应与收集全部响应图后取均值逐位一致。"""
        responses = bank.filter(image)
        want = np.zeros_like(responses[0])
        for r in responses:
            want += r ** 2
        want /= len(responses)

        assert np.array_equal(bank.energy_map(image), want)

    def test_scan_returns_scalars_not_response_maps(self, bank, image):
        """scan() 只返回标量，不得返回响应图。

        相机分辨率（2448×2048）下 72 张 float64 响应图共约 2.9 GB，
        而 mean/std/mean(r²) 都只需要标量。这条断言防止有人为了「复用」
        把响应图缓存下来。
        """
        features, energies = bank.scan(image)

        assert features.shape == (2 * bank.n_kernels,)
        assert features.dtype == np.float64
        assert all(np.isscalar(v) for g in energies for v in g)
