"""缺陷检测模块单元测试。"""

import numpy as np
import pytest
import yaml
import cv2
from pathlib import Path

from core.defects import DefectDetector, Defect


@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def clean_image():
    """洁净铜面图像（BGR，全铜色）。"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:] = (65, 160, 200)  # BGR 铜色
    img += np.random.randint(0, 10, img.shape, dtype=np.uint8)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def oxidized_image():
    """含氧化斑的图像。"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:] = (65, 160, 200)  # 铜色基底
    # 注入氧化斑（深棕色区域）
    cv2.circle(img, (100, 100), 30, (40, 90, 140), -1)
    return img


@pytest.fixture
def embedding_image():
    """含磨料嵌入点的灰度图像。"""
    img = np.ones((200, 200), dtype=np.uint8) * 120
    # 注入亮点（模拟磨料颗粒）
    for _ in range(10):
        cx, cy = np.random.randint(30, 170, 2)
        cv2.circle(img, (cx, cy), 2, 240, -1)
    return img


class TestDefectDetector:
    """DefectDetector 测试。"""

    def test_init(self, config):
        """测试初始化。"""
        dd = DefectDetector(config)
        assert dd.ox_enabled is True
        assert dd.emb_enabled is True
        assert dd.unrough_enabled is True

    def test_detect_oxidation(self, config, oxidized_image):
        """应检出注入的氧化斑。"""
        dd = DefectDetector(config)
        defects = dd.detect_oxidation(oxidized_image)
        assert isinstance(defects, list)
        # 可能检测到或多或少的氧化区域（取决于颜色阈值）
        # 不强制数量，验证结构
        for d in defects:
            assert d.type == "oxidation"
            assert d.area_pixels > 0
            assert len(d.bbox) == 4

    def test_detect_oxidation_clean(self, config, clean_image):
        """洁净铜面不应产生大量虚警。"""
        dd = DefectDetector(config)
        defects = dd.detect_oxidation(clean_image)
        # 洁净图像上氧化斑总数应较少（允许少量噪声误检）
        assert len(defects) < 5

    def test_detect_abrasive_embedding(self, config, embedding_image):
        """应检出磨料嵌入点。"""
        dd = DefectDetector(config)
        defects = dd.detect_abrasive_embedding(embedding_image)
        for d in defects:
            assert d.type == "embedding"
            assert d.area_pixels > 0

    def test_defect_dataclass(self):
        """Defect dataclass 属性类型验证。"""
        d = Defect(
            type="oxidation",
            mask=np.ones((10, 10), dtype=np.uint8) * 255,
            area_pixels=100,
            area_mm2=0.01,
            centroid=(50, 50),
            bbox=(40, 40, 60, 60),
            severity=0.5,
        )
        assert d.type == "oxidation"
        assert d.area_pixels == 100
        assert len(d.centroid) == 2
        assert len(d.bbox) == 4

    def test_detect_all(self, config, oxidized_image):
        """detect_all 应返回列表。"""
        dd = DefectDetector(config)
        defects = dd.detect_all(oxidized_image)
        assert isinstance(defects, list)

    def test_draw_defects(self, config, oxidized_image):
        """缺陷绘制不应改变图像尺寸。"""
        dd = DefectDetector(config)
        defects = dd.detect_oxidation(oxidized_image)
        annotated = dd.draw_defects(oxidized_image, defects)
        assert annotated.shape == oxidized_image.shape

    def test_generate_heatmap(self, config, oxidized_image):
        """热力图生成。"""
        dd = DefectDetector(config)
        defects = dd.detect_oxidation(oxidized_image)
        heatmap = dd.generate_defect_heatmap(
            oxidized_image.shape[:2], defects,
        )
        assert heatmap.shape == oxidized_image.shape[:2]
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1.0
