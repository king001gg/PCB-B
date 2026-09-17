"""喷砂工艺参数监测模块单元测试。

重点覆盖 core/process_monitor.py 中三条数值口径硬规则（R1/R2/R3）
与工艺判定规则的两处修正，这些是最容易被后续改动静默破坏的地方。
"""

import cv2
import numpy as np
import pytest

import core.process_monitor as pm
from core.process_monitor import (
    ProcessGLCMExtractor,
    ProcessGLCMFeatures,
    ProcessMonitor,
    ProcessVerdict,
)


# ============================================================================
# 测试用图像
# ============================================================================

@pytest.fixture
def flat_image():
    """常数灰度图 —— 无纹理，contrast 应为 0。"""
    return np.full((200, 200), 128, dtype=np.uint8)


@pytest.fixture
def checkerboard():
    """棋盘图 —— 高频纹理，contrast 应显著偏高。"""
    pattern = (np.indices((200, 200)).sum(axis=0) % 2) * 255
    return pattern.astype(np.uint8)


@pytest.fixture
def striped_image():
    """带条纹的灰度图 —— 介于两者之间的自然纹理。"""
    img = np.zeros((200, 200), dtype=np.uint8)
    for i in range(200):
        img[i, :] = int(128 + 64 * np.sin(i * 0.1))
    return img


# ============================================================================
# GLCM 提取器
# ============================================================================

class TestProcessGLCMExtractor:
    """8 级口径 GLCM 特征提取器测试。"""

    def test_constant_image_has_zero_contrast(self, flat_image):
        """常数图无灰度差，contrast 与 dissimilarity 应为 0。"""
        f = ProcessGLCMExtractor().compute(flat_image)
        assert isinstance(f, ProcessGLCMFeatures)
        assert f.contrast == pytest.approx(0.0, abs=1e-9)
        assert f.dissimilarity == pytest.approx(0.0, abs=1e-9)

    def test_checkerboard_has_high_contrast(self, flat_image, checkerboard):
        """棋盘图的 contrast 应显著高于常数图。"""
        extractor = ProcessGLCMExtractor()
        assert extractor.compute(checkerboard).contrast > \
            extractor.compute(flat_image).contrast + 1.0

    def test_all_features_are_finite(self, striped_image):
        """六个特征均应为有限值 —— 退化输入下 skimage 可能返回 nan。"""
        f = ProcessGLCMExtractor().compute(striped_image)
        for name, value in f.as_dict().items():
            assert np.isfinite(value), f"{name} 不是有限值: {value}"

    def test_energy_is_sqrt_of_asm_per_angle(self, striped_image):
        """energy 必须逐角度等于 sqrt(asm)。

        这是本模块最易出错的一点：skimage 的 graycoprops('energy') 返回
        sqrt(ASM)，而 core/texture.py 的同名字段存的却是 ASM。若有人把手写
        公式替换进来，数值口径会静默改变、继电器阈值随之失效。

        注意不能在取平均后比较：mean(sqrt(x)) != sqrt(mean(x))（Jensen 不等式），
        平均会破坏该恒等式。因此必须在单个角度的 GLCM 上验证。
        """
        from skimage.feature import graycomatrix

        extractor = ProcessGLCMExtractor()
        quantized = extractor._quantize(striped_image)
        glcm = graycomatrix(
            quantized,
            distances=[extractor.distance],
            angles=extractor._angles_rad,
            levels=extractor.levels,
            symmetric=True,
            normed=True,
        )
        props = extractor._props_from_glcm(glcm)
        assert props["energy"] == pytest.approx(np.sqrt(props["asm"]), rel=1e-9)

    def test_asm_equals_sum_of_squares(self, striped_image):
        """asm 应满足 sum(p²) 的定义，取值在 (0, 1]。"""
        f = ProcessGLCMExtractor().compute(striped_image)
        assert 0.0 < f.asm <= 1.0

    def test_gray_and_3channel_inputs_agree(self, striped_image):
        """R2：灰度图与其三通道副本应得到相同结果。

        验证 _to_gray 只做色彩空间转换、不夹带任何预处理。
        """
        bgr = cv2.cvtColor(striped_image, cv2.COLOR_GRAY2BGR)
        extractor = ProcessGLCMExtractor()
        from_gray = extractor.compute(striped_image)
        from_bgr = extractor.compute(bgr)
        assert from_bgr.as_dict() == pytest.approx(from_gray.as_dict(), rel=1e-9)

    def test_color_order_is_respected(self):
        """默认按 RGB 解释；在彩色图上 rgb 与 bgr 会得到不同灰度。

        本项目内部约定为 RGB（main.py 与 MainWindow._set_image 都先做 BGR2RGB，
        DefectDetector 也按 RGB 处理）。若有人把 BGR 图按默认参数传进来，灰度
        转换的 R/B 权重会对调，离线与在线两条路径将算出不同的值（实测最大差 27%），
        破坏 R4。注意用灰度图的三通道副本测不出这一点 —— 那里 R/B 本就相同。
        """
        rng = np.random.default_rng(3)
        colored = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)

        extractor = ProcessGLCMExtractor()

        # 默认即 RGB
        assert extractor.compute(colored).as_dict() == pytest.approx(
            extractor.compute(colored, color_order="rgb").as_dict(), rel=1e-12
        )

        # 两种顺序确实产生不同灰度与不同特征
        assert not np.array_equal(
            extractor._to_gray(colored, "rgb"), extractor._to_gray(colored, "bgr")
        )
        assert extractor.compute(colored).contrast != pytest.approx(
            extractor.compute(colored, color_order="bgr").contrast, rel=1e-6
        )

    def test_invalid_color_order_raises(self):
        """非法 color_order 应明确报错，而非静默按某个顺序处理。"""
        with pytest.raises(ValueError, match="color_order"):
            ProcessGLCMExtractor().compute(
                np.zeros((20, 20, 3), dtype=np.uint8), color_order="hsv"
            )

    def test_resize_is_idempotent(self):
        """R3：内部降采样与调用方预降采样应等价。

        离线路径处理整图、在线路径处理小图，若两者口径不一致，同一块板
        在两条路径下会得出不同的值，共用阈值将自相矛盾。这条是防回归的
        关键测试。
        """
        rng = np.random.default_rng(42)
        large = rng.integers(0, 256, (1000, 800), dtype=np.uint8)

        scale = 640 / 1000.0
        presized = cv2.resize(
            large, (int(round(800 * scale)), 640), interpolation=cv2.INTER_AREA
        )

        extractor = ProcessGLCMExtractor(resize_long_side=640)
        from_large = extractor.compute(large)
        from_presized = extractor.compute(presized)

        assert from_large.contrast == pytest.approx(from_presized.contrast, rel=1e-6)
        assert from_large.asm == pytest.approx(from_presized.asm, rel=1e-6)

    def test_small_image_is_not_upscaled(self):
        """小于 resize_long_side 的图像不应被放大。"""
        small = np.full((100, 100), 128, dtype=np.uint8)
        assert ProcessGLCMExtractor(resize_long_side=640)._resize(small).shape == (100, 100)

    def test_quantization_levels(self, flat_image):
        """R1：量化结果必须落在 [0, levels-1] 范围内。"""
        extractor = ProcessGLCMExtractor(levels=8)
        quantized = extractor._quantize(flat_image)
        assert quantized.min() >= 0
        assert quantized.max() <= 7

        ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
        assert extractor._quantize(ramp).max() <= 7

    def test_rejects_none_and_bad_dims(self):
        """非法输入应抛出明确异常。"""
        extractor = ProcessGLCMExtractor()
        with pytest.raises(ValueError):
            extractor.compute(None)
        with pytest.raises(ValueError):
            extractor.compute(np.zeros((4, 4, 4, 4), dtype=np.uint8))

    def test_missing_skimage_raises(self, monkeypatch):
        """skimage 缺失时应明确报错，而非静默降级到数值不一致的回退实现。"""
        monkeypatch.setattr(pm, "HAS_SKIMAGE", False)
        with pytest.raises(RuntimeError, match="scikit-image"):
            ProcessGLCMExtractor()


# ============================================================================
# 工艺判定
# ============================================================================

class TestProcessMonitor:
    """工艺判定规则测试。"""

    @staticmethod
    def _features(contrast: float) -> ProcessGLCMFeatures:
        return ProcessGLCMFeatures(contrast=contrast)

    @pytest.mark.parametrize("contrast, expected_level", [
        (0.00, "异常"),
        (0.29, "异常"),
        (0.31, "偏小"),
        (0.49, "偏小"),
        (0.51, "正常"),
        (1.00, "正常"),
        (1.99, "正常"),
        (2.01, "偏大"),
        (5.00, "偏大"),
    ])
    def test_boundary_levels(self, contrast, expected_level):
        """逐一覆盖阈值边界 0.3 / 0.5 / 2.0。"""
        monitor = ProcessMonitor()
        assert monitor.evaluate(self._features(contrast)).level == expected_level

    def test_alarm_only_in_lowest_band(self):
        """仅 contrast < 0.3 触发闪烁报警。"""
        monitor = ProcessMonitor()
        assert monitor.evaluate(self._features(0.29)).alarm is True
        assert monitor.evaluate(self._features(0.31)).alarm is False
        assert monitor.evaluate(self._features(1.00)).alarm is False
        assert monitor.evaluate(self._features(2.01)).alarm is False

    def test_alarm_branch_reports_abnormal_not_normal(self):
        """修正 1：报警分支的文本必须是「异常」。

        继电器原实现在该分支写入「正常」却同时点亮红灯，自相矛盾。
        """
        verdict = ProcessMonitor().evaluate(self._features(0.1))
        assert verdict.level == "异常"
        assert verdict.level != "正常"
        assert verdict.alarm is True

    def test_suggested_speed_is_clamped_non_negative(self):
        """修正 2：建议速度不得为负。

        原公式 speed_base - contrast 在 contrast ≥ speed_base 时产出负值，
        无物理意义。
        """
        monitor = ProcessMonitor()
        verdict = monitor.evaluate(self._features(10.0))
        assert "-" not in verdict.suggestion
        assert "0.10" in verdict.suggestion

    def test_suggested_speed_decreases_with_contrast(self):
        """纹理越强（contrast 越高）建议速度越低。

        取 0.4（偏小区间，建议提速）与 2.5（偏大区间，建议降速）对比。
        不能取正常区间 [0.5, 2.0] 内的值 —— 那里返回的是正常文案，不含速度。
        """
        monitor = ProcessMonitor()

        def speed_of(c):
            text = monitor.evaluate(self._features(c)).suggestion
            return float(text.replace("提高速度到 ", "").replace("降低速度到 ", ""))

        assert speed_of(0.4) > speed_of(2.5)

    def test_normal_band_uses_normal_advice(self):
        """正常区间应给出正常文案，而非速度建议。"""
        verdict = ProcessMonitor().evaluate(self._features(1.0))
        assert verdict.suggestion == "纹理清晰，检测环境理想"

    def test_verdict_colors(self):
        """三种状态使用不同颜色。"""
        monitor = ProcessMonitor()
        assert monitor.evaluate(self._features(1.0)).color == pm.COLOR_OK
        assert monitor.evaluate(self._features(0.4)).color == pm.COLOR_WARN
        assert monitor.evaluate(self._features(0.1)).color == pm.COLOR_ALARM

    def test_returns_verdict_instance(self):
        """返回值类型正确。"""
        assert isinstance(
            ProcessMonitor().evaluate(self._features(1.0)), ProcessVerdict
        )


# ============================================================================
# 配置
# ============================================================================

class TestConfigHandling:
    """配置读取与回退测试。"""

    def test_missing_section_falls_back_to_defaults(self):
        """配置缺失 process_monitor 段时应正常构造并使用默认阈值。

        保证旧版配置文件（config/default.yaml 在本模块引入之前的版本）
        仍可加载，不会导致系统启动失败。
        """
        monitor = ProcessMonitor({"inspection": {}, "system": {}})
        assert monitor.contrast_low_alarm == pm.DEFAULT_THRESHOLDS["contrast_low_alarm"]
        assert monitor.contrast_high == pm.DEFAULT_THRESHOLDS["contrast_high"]

    def test_empty_config_works(self):
        """完全不传配置也应可用。"""
        monitor = ProcessMonitor()
        assert monitor.evaluate(ProcessGLCMFeatures(contrast=1.0)).level == "正常"

    def test_none_config_works(self):
        """显式传入 None 也应可用。"""
        assert ProcessMonitor(None).enabled is True

    def test_custom_thresholds_are_respected(self):
        """自定义阈值应覆盖默认值。"""
        monitor = ProcessMonitor({
            "inspection": {
                "process_monitor": {
                    "thresholds": {"contrast_low_alarm": 1.0, "contrast_high": 9.0}
                }
            }
        })
        assert monitor.evaluate(ProcessGLCMFeatures(contrast=0.5)).alarm is True
        assert monitor.evaluate(ProcessGLCMFeatures(contrast=5.0)).level == "正常"

    def test_custom_glcm_params_are_respected(self):
        """GLCM 参数应可从配置覆盖。"""
        monitor = ProcessMonitor({
            "inspection": {"process_monitor": {"glcm": {"levels": 16, "distance": 2}}}
        })
        assert monitor.extractor.levels == 16
        assert monitor.extractor.distance == 2

    def test_custom_advice_template_is_rendered(self):
        """自定义建议模板应被渲染并填入速度值。"""
        monitor = ProcessMonitor({
            "inspection": {"process_monitor": {"advice": {"low": "请降速至 {speed:.1f} rpm"}}}
        })
        assert "rpm" in monitor.evaluate(ProcessGLCMFeatures(contrast=0.4)).suggestion

    def test_broken_advice_template_does_not_crash(self):
        """模板语法错误时应退回原文，而非抛异常。"""
        monitor = ProcessMonitor({
            "inspection": {"process_monitor": {"advice": {"low": "速度 {speed:"}}}
        })
        verdict = monitor.evaluate(ProcessGLCMFeatures(contrast=0.4))
        assert verdict.suggestion

    def test_disabled_flag(self):
        """enabled: false 应被读取。"""
        monitor = ProcessMonitor({
            "inspection": {"process_monitor": {"enabled": False}}
        })
        assert monitor.enabled is False
