"""预处理模块单元测试。"""

import numpy as np
import pytest
import cv2
import yaml
from pathlib import Path

from core.preprocessing import Preprocessor


@pytest.fixture
def config():
    """加载默认配置。"""
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_image():
    """生成测试用合成图像。"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:] = (65, 160, 200)  # 铜色
    # 添加一些纹理
    noise = np.random.randint(0, 20, (200, 200, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


@pytest.fixture
def gray_image():
    """灰度测试图像。"""
    img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    return img


class TestPreprocessor:
    """Preprocessor 类测试套件。"""

    def test_init(self, config):
        """测试初始化：应从配置文件加载参数。"""
        pp = Preprocessor(config)
        assert pp.retinex_enabled is True
        assert pp.clahe_enabled is True
        assert isinstance(pp.sigmas, list)

    def test_process_rgb(self, config, sample_image):
        """测试 RGB 图像处理：应返回有效的灰度输出。"""
        pp = Preprocessor(config)
        result = pp.process(sample_image)
        assert result is not None
        assert result.ndim == 2
        assert result.dtype == np.uint8
        assert result.shape[:2] == sample_image.shape[:2]

    def test_process_gray(self, config, gray_image):
        """测试灰度图像输入。"""
        pp = Preprocessor(config)
        result = pp.process(gray_image)
        assert result is not None
        assert result.ndim == 2
        assert result.shape == gray_image.shape

    def test_process_empty_raises(self, config):
        """测试空图像抛出异常。"""
        pp = Preprocessor(config)
        with pytest.raises(ValueError):
            pp.process(np.array([]))

    def test_process_none_raises(self, config):
        """测试 None 输入抛出异常。"""
        pp = Preprocessor(config)
        with pytest.raises(ValueError):
            pp.process(None)

    def test_clahe_enhance(self, config, gray_image):
        """测试 CLAHE 增强：输出应为 uint8 且尺寸不变。"""
        pp = Preprocessor(config)
        result = pp.enhance_texture(gray_image)
        assert result.dtype == np.uint8
        assert result.shape == gray_image.shape
        # CLAHE 不应使所有值相等
        assert result.std() > 0

    def test_retinex_correct(self, config, sample_image):
        """测试 Retinex 校正。"""
        pp = Preprocessor(config)
        result = pp.retinex_correct(sample_image)
        assert result is not None
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_roi_mask(self, sample_image):
        """测试静态 ROI 掩膜方法。"""
        mask = Preprocessor.get_roi_mask(
            sample_image,
            np.array([0, 30, 60]),
            np.array([25, 255, 255]),
        )
        assert mask is not None
        assert mask.shape == sample_image.shape[:2]
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

    def test_extract_roi(self, config, sample_image):
        """测试 ROI 提取。"""
        pp = Preprocessor(config)
        pp.roi_enabled = True
        gray = pp.extract_roi(sample_image)
        assert gray is not None
        assert gray.ndim == 2
        # 铜色图像应该产生非零掩膜
        assert gray.max() > 0

    def test_disabled_steps(self, config, sample_image):
        """测试禁用预处理步骤。"""
        pp = Preprocessor(config)
        pp.retinex_enabled = False
        pp.clahe_enabled = False
        pp.roi_enabled = False
        result = pp.process(sample_image)
        assert result is not None
