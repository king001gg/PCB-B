"""core/color.py 的配置解析与校验。

分两类：

  (1) **缺项回退**。分析器要能在一个什么都不全的配置上构造出来 —— 老配置
      文件（没有 inspection.color 节）、单测里手搓的小字典、以及将来裁剪过
      的配置，都不该让它崩。

  (2) **校验**。``color_order`` 非法必须在**构造期**报错，不能拖到每帧。
      理由见 core/color.py 的 C2：这个量无法从数组形状推断，喂错不报错、
      只会静默把色相转掉 172°，所以宁可早失败。

这里用**合成配置**而不是 config/default.yaml —— 本文件测的是"没有配置时
怎么办"，依赖真配置里恰好有哪些键就测不到回退分支了。
"""

import numpy as np
import pytest

from core.color import ColorAnalyzer, HUE_UNIT_TO_DEG


def _minimal() -> dict:
    """只有 system 节的最小配置。"""
    return {"system": {"resolution_mm_per_pixel": 0.02}}


class TestFallbackDefaults:
    """配置缺项时的回退值。"""

    def test_empty_config_is_constructible(self):
        a = ColorAnalyzer({})

        assert a.enabled is True
        assert a.color_order == "rgb"
        assert a.hue_s_min == 30
        assert a.use_roi is False
        assert a.area_min_px == 100
        assert a.morph_kernel == 5
        assert a.max_regions == 50

    def test_missing_color_section(self):
        """老配置文件没有 inspection.color 节 —— 这是升级时的常态。"""
        a = ColorAnalyzer({"inspection": {"quality": {"ok_score_threshold": 60}}})
        assert a.color_order == "rgb"
        assert a.abs_enabled is True

    def test_missing_absolute_and_adaptive_subsections(self):
        a = ColorAnalyzer({"inspection": {"color": {"enabled": True}}})

        assert a.abs_hue_baseline == 34.0
        assert a.abs_hue_tol == 25.0
        assert a.abs_sat_baseline == 178.0
        assert a.abs_sat_tol == 50.0
        assert a.mad_k == 3.0
        assert a.min_hue_tol == 8.0
        assert a.min_sat_tol == 20.0
        assert a.min_valid_pixels == 1000

    def test_none_subsections_do_not_crash(self):
        """YAML 里写 ``absolute:`` 后面留空会得到 None，不是 {}。"""
        a = ColorAnalyzer({"inspection": {"color": {
            "absolute": None, "adaptive": None,
        }}})

        assert a.abs_hue_baseline == 34.0
        assert a.mad_k == 3.0

    def test_missing_system_section_uses_default_resolution(self):
        a = ColorAnalyzer({"inspection": {"color": {}}})
        assert a.mm_per_pixel == 0.01

    def test_resolution_is_read_from_system(self):
        a = ColorAnalyzer({"system": {"resolution_mm_per_pixel": 0.005}})
        assert a.mm_per_pixel == 0.005

    def test_analyzer_works_on_a_minimal_config(self):
        """回退值必须真的能跑通，而不只是属性存在。"""
        a = ColorAnalyzer(_minimal())
        img = np.zeros((60, 60, 3), dtype=np.uint8)
        img[:, :] = (60, 140, 200)

        feats = a.analyze(img, color_order="bgr")

        assert feats.available is True
        assert feats.hue_mean_deg == pytest.approx(34.0, abs=1.0)


class TestConfigValidation:
    """非法配置必须在构造期失败。"""

    @pytest.mark.parametrize("bad", ["gbr", "bgr ", "", "bgra", "hsv", "BRG"])
    def test_invalid_color_order_raises(self, bad):
        with pytest.raises(ValueError, match="color_order"):
            ColorAnalyzer({"inspection": {"color": {"color_order": bad}}})

    def test_error_message_lists_valid_choices(self):
        with pytest.raises(ValueError) as exc:
            ColorAnalyzer({"inspection": {"color": {"color_order": "xyz"}}})

        msg = str(exc.value)
        assert "bgr" in msg and "rgb" in msg
        assert "显式声明" in msg

    @pytest.mark.parametrize("good", ["bgr", "rgb", "BGR", "RGB", "Rgb", "bGr"])
    def test_valid_orders_are_accepted_and_lowercased(self, good):
        assert ColorAnalyzer(
            {"inspection": {"color": {"color_order": good}}}
        ).color_order == good.lower()

    def test_analyze_rejects_bad_order_at_call_time_too(self):
        """同一个实例要能服务多条路径，所以调用期也要挡一次。"""
        a = ColorAnalyzer({})
        with pytest.raises(ValueError, match="color_order"):
            a.analyze(np.zeros((4, 4, 3), dtype=np.uint8), color_order="nope")

    def test_none_uses_configured_order(self):
        """``color_order=None`` 表示"用配置里的"，不是"没指定所以随便"。"""
        img = np.zeros((60, 60, 3), dtype=np.uint8)
        img[:, :] = (60, 140, 200)

        as_rgb = ColorAnalyzer(
            {"inspection": {"color": {"color_order": "rgb"}}}
        ).analyze(img, color_order=None)
        as_bgr = ColorAnalyzer(
            {"inspection": {"color": {"color_order": "bgr"}}}
        ).analyze(img, color_order=None)

        assert as_rgb.color_order == "rgb"
        assert as_bgr.color_order == "bgr"
        assert as_rgb.hue_mean_deg != pytest.approx(as_bgr.hue_mean_deg)


class TestRealConfigRows:
    """真配置里的键名与结构必须与分析器读的一致。

    这些断言故意写死字面量：改名或挪位置时这里会红，提醒同步
    core/color.py 的读取路径。
    """

    def test_default_yaml_has_color_section(self, default_config):
        color = default_config["inspection"]["color"]

        assert set(color) >= {
            "enabled", "color_order", "hue_s_min", "use_roi", "area_min_px",
            "morph_kernel", "max_regions", "absolute", "adaptive",
        }
        assert set(color["absolute"]) == {
            "enabled", "hue_baseline_deg", "hue_tolerance_deg",
            "sat_baseline", "sat_tolerance",
        }
        assert set(color["adaptive"]) == {
            "enabled", "mad_k", "min_hue_tol_deg", "min_sat_tol",
            "min_valid_pixels",
        }

    def test_default_yaml_color_order_matches_the_gui_path(self, default_config):
        """默认值必须对齐 GUI / CLI —— 那两条路径在入口做了 BGR2RGB。

        写成 "bgr" 会让界面上显示的色度整体偏 172°，而且不报错。
        """
        assert default_config["inspection"]["color"]["color_order"] == "rgb"

    def test_default_yaml_pixel_format_is_color(self, default_config):
        """色度指标需要彩色输入，相机默认格式必须是彩色的。"""
        fmt = default_config["camera"]["pixel_format"]

        assert fmt == "BGR8", (
            "必须是 BGR8 而不是 RGB8：CameraBase 的契约是 BGR，"
            "上层 _to_rgb 会再转一次，填 RGB8 会让红蓝被换两次"
        )

    def test_hue_baseline_matches_the_measured_copper(self, default_config):
        """基准色相与实测的金色铜面对得上（H=17 单位 × 2 = 34°）。"""
        baseline = default_config["inspection"]["color"]["absolute"][
            "hue_baseline_deg"
        ]

        assert baseline == pytest.approx(17 * HUE_UNIT_TO_DEG)
