"""通道顺序（BGR / RGB）的**表征测试**。

"表征测试"= 只断言代码**当前**的行为，不改任何生产代码。目的是把现状钉死：
日后有人（包括未来的我）想"顺手统一一下通道顺序"时，这些用例会立刻告诉他在
哪些地方会连带改变结果。

三个用途：

  1. 给 `core/color.py` 的 C3 留下实测依据。C3 说"两种顺序下 S/V 逐位相同、
     只有 H 不同"，这是本模块 `color_order` 必须显式声明的全部理由。
  2. 把**当前存在的通道口径矛盾**白纸黑字写下来。矛盾本身是缺陷，不是设计。
  3. 钉住 `_localize` 与 `DefectDetector._extract_connected_components` 的等价性，
     避免两套连通域实现日后各自漂移。

⚠ 本文件中标注了「钉住缺陷」的用例，断言的是**缺陷现状**而非期望行为。
   缺陷修掉时这些用例会变红——那时应当改用例，**不要**把生产代码改回去。
   缺陷登记见 ``docs/2026-09-17-企业级测试报告.md``。

为什么需要这个文件：`tests/test_defects.py` 全程按 BGR 喂图，所以它对
`detect_oxidation` 的通道假设是"自洽"的，全绿；而生产上 GUI 与 CLI 走的是
**RGB** 入口。**测试通过的路径不是生产走的路径**——这是本文件要补的洞。
"""

import copy

import cv2
import numpy as np
import pytest

from core.acquisition import ImageAcquisition, ImageFrame
from core.color import ColorAnalyzer, HUE_UNIT_TO_DEG
from core.defects import DefectDetector
from core.pipeline import InspectionPipeline
from core.preprocessing import Preprocessor
from core.process_monitor import ProcessGLCMExtractor


# 纯金色铜面，BGR 顺序：B=60, G=140, R=200。
# 实测 H=17（OpenCV 单位）= 34°，S=178，V=200。
COPPER_BGR = (60, 140, 200)
COPPER_HUE_UNITS = 17
COPPER_HUE_DEG = COPPER_HUE_UNITS * HUE_UNIT_TO_DEG      # 34.0
# 按 RGB 解释同一份字节得到的色相：H=103 → 206°。环形距离 86 单位 = 172°。
WRONG_ORDER_HUE_UNITS = 103


def _solid(bgr: tuple, h: int = 80, w: int = 80) -> np.ndarray:
    """一张纯色图。用纯色是刻意的——色相没有分布，断言可以写死。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


def _varied_color_image(seed: int = 7, h: int = 96, w: int = 96) -> np.ndarray:
    """色彩有空间变化的图，用来验证逐位级别的恒等式。

    纯色图上 S/V 相同可能只是巧合（所有像素同值），必须在有分布的图上验。
    """
    rng = np.random.RandomState(seed)
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, :] = (200, 60, 30)          # 偏蓝（BGR 里 ch0 大）
    b = np.zeros((h, w, 3), dtype=np.uint8)
    b[:, :] = (30, 60, 200)          # 偏红（BGR 里 ch2 大）
    img = np.where(rng.rand(h, w, 1) > 0.5, a, b).astype(np.uint8)
    return cv2.GaussianBlur(img, (9, 9), 0)


class _OneFrameAcquisition(ImageAcquisition):
    """只吐一帧就结束的采集器，用来驱动 `process_acquisition`。"""

    def __init__(self, image: np.ndarray):
        self._image = image

    def acquire(self):
        if self._image is None:
            return None
        img, self._image = self._image, None
        return ImageFrame(image=img, timestamp="2026-01-01T00:00:00",
                          source_id="stub", frame_index=0)

    def reset(self):
        pass

    def close(self):
        pass


# ============================================================================
# C3 的实测依据：S/V 与顺序无关，H 强相关
# ============================================================================

class TestChannelOrderIdentity:
    """`core/color.py` 的 C3 为什么成立。"""

    def test_identity_holds(self):
        """恒等式：先 BGR2RGB 再 RGB2HSV == 直接 BGR2HSV。

        这条恒等式是"两种顺序只是同一件事的两种写法"的证明，也是
        `_to_hsv` 用一个 code 就够了的依据。
        """
        img = _varied_color_image()

        direct = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        roundtrip = cv2.cvtColor(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2HSV
        )

        assert np.array_equal(direct, roundtrip)

    def test_saturation_and_value_are_bit_identical_across_orders(self):
        """同一份字节按两种顺序解释，S 与 V **逐位**相同。

        这是 C3 的前半句。后果：顺序喂错时饱和度一路的量**完全看不出异常**，
        只有色相会错——错误因此极难被发现，必须靠显式声明挡住。
        """
        img = _varied_color_image()

        as_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        as_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        assert np.array_equal(as_bgr[:, :, 1], as_rgb[:, :, 1]), "S 应逐位相同"
        assert np.array_equal(as_bgr[:, :, 2], as_rgb[:, :, 2]), "V 应逐位相同"

    def test_only_hue_differs(self):
        """C3 的后半句：H 不同，而且差异很大。"""
        img = _varied_color_image()

        as_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        as_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        assert not np.array_equal(as_bgr[:, :, 0], as_rgb[:, :, 0])

    def test_copper_hue_shifts_by_172_degrees(self):
        """金色铜面的实测：H=17(34°) → H=103(206°)，环形距离 172°。

        172° 这个数字是 C3 里那个"172°"的出处，故意写死在这里。

        注意 uint8 相减会溢出（17-103 在 uint8 下是 170），必须先转 int。
        """
        img = _solid(COPPER_BGR)

        h_bgr = int(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[0, 0, 0])
        h_rgb = int(cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[0, 0, 0])

        assert (h_bgr, h_rgb) == (COPPER_HUE_UNITS, WRONG_ORDER_HUE_UNITS)

        delta_deg = abs(h_bgr - h_rgb) * HUE_UNIT_TO_DEG
        assert delta_deg == pytest.approx(172.0)

    def test_wrong_order_is_silent(self):
        """喂错顺序**不会报错**——所以 `ColorAnalyzer` 必须在构造期校验。

        这条用例是 `ColorAnalyzer._validate_order` 存在的理由：OpenCV 不给你
        任何提示，错就是静默错到底。
        """
        img = _solid(COPPER_BGR)

        # 不抛异常，只是答案不同
        cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        with pytest.raises(ValueError, match="color_order"):
            ColorAnalyzer({}).analyze(img, color_order="gbr")

    def test_color_analyzer_reports_the_shifted_hue_when_misordered(self):
        """同一个分析器，两个顺序，色相偏移差 172°——而饱和度几乎不动。"""
        img = _solid(COPPER_BGR)
        analyzer = ColorAnalyzer({})

        right = analyzer.analyze(img, color_order="bgr")
        wrong = analyzer.analyze(img, color_order="rgb")

        assert right.hue_mean_deg == pytest.approx(COPPER_HUE_DEG, abs=0.5)
        assert wrong.hue_mean_deg == pytest.approx(206.0, abs=0.5)
        # 饱和度一路毫无察觉
        assert right.sat_mean == pytest.approx(wrong.sat_mean)

    def test_color_analyzer_never_guesses_from_shape(self):
        """三通道图无法从 shape 判出顺序，所以默认值来自配置而非推断。

        `ColorAnalyzer` 读 `inspection.color.color_order`，与"猜"无关。
        """
        img = _solid(COPPER_BGR)

        as_configured_rgb = ColorAnalyzer(
            {"inspection": {"color": {"color_order": "rgb"}}}
        ).analyze(img)
        as_configured_bgr = ColorAnalyzer(
            {"inspection": {"color": {"color_order": "bgr"}}}
        ).analyze(img)

        assert as_configured_rgb.color_order == "rgb"
        assert as_configured_bgr.color_order == "bgr"
        assert as_configured_rgb.hue_mean_deg != as_configured_bgr.hue_mean_deg


# ============================================================================
# 旧路径的通道假设（钉住缺陷）
# ============================================================================

class TestLegacyPathChannelAssumptions:
    """旧路径各自假设了哪个顺序，以及它们互相矛盾的地方。

    ⚠ 本类钉的是**当前的缺陷现状**。矛盾清单：

        Preprocessor.process       按 BGR（L69 BGR2GRAY）
          └ 但 Retinex 分支按 RGB（L82/L87）  ← 同一函数内部自相矛盾
        Preprocessor.get_roi_mask  按 BGR
        DefectDetector.detect_oxidation  按 BGR（无条件 BGR2RGB）
        ProcessMonitor.analyze     **按 RGB**（默认 color_order="rgb"）

    而生产入口喂的是 RGB（GUI L570、CLI L155 都先做了 BGR2RGB）——于是
    `detect_oxidation` 在真实路径上拿到的是 RGB，色相被解释到 172° 之外。
    """

    # ---- Preprocessor -------------------------------------------------

    def test_preprocessor_gray_conversion_assumes_bgr(self, default_config):
        """⚠ 钉住缺陷：`process()` 的灰度化按 BGR 解释。

        关掉 Retinex 与 CLAHE、关掉 ROI 后，输出应当逐位等于 `BGR2GRAY`。
        """
        cfg = copy.deepcopy(default_config)
        cfg["inspection"]["preprocessing"]["retinex"]["enabled"] = False
        cfg["inspection"]["preprocessing"]["clahe"]["enabled"] = False
        cfg["inspection"]["roi"]["enabled"] = False
        pp = Preprocessor(cfg)

        img = _varied_color_image()
        out = pp.process(img)

        assert np.array_equal(out, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        assert not np.array_equal(out, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

    def test_retinex_branch_gray_conversion_assumes_rgb(self, default_config):
        """⚠ 钉住缺陷：同一个 `process()`，Retinex 分支却按 RGB 解释。

        与上一条同一个输入、同一个函数，只因为一个配置开关不同，就换了一套
        通道假设。两条灰度结果实测平均相差约 17 个灰度级——不是可以忽略的量。
        """
        cfg = copy.deepcopy(default_config)
        cfg["inspection"]["preprocessing"]["clahe"]["enabled"] = False
        cfg["inspection"]["roi"]["enabled"] = False
        pp = Preprocessor(cfg)
        assert pp.retinex_enabled is True, "本用例依赖 Retinex 默认开启"

        img = _varied_color_image()
        out = pp.process(img)
        corrected = pp.retinex_correct(img)

        assert np.array_equal(out, cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY))
        assert not np.array_equal(out, cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY))

    def test_the_two_branches_disagree_on_the_same_input(self, default_config):
        """两条分支的结果实测差多少。差值本身就是要钉住的量。"""
        img = _varied_color_image()

        off = copy.deepcopy(default_config)
        off["inspection"]["preprocessing"]["retinex"]["enabled"] = False
        off["inspection"]["preprocessing"]["clahe"]["enabled"] = False
        off["inspection"]["roi"]["enabled"] = False

        on = copy.deepcopy(off)
        on["inspection"]["preprocessing"]["retinex"]["enabled"] = True

        a = Preprocessor(off).process(img).astype(np.int16)
        b = Preprocessor(on).process(img).astype(np.int16)

        assert np.abs(a - b).mean() > 10.0, "两分支差异应当显著，不是舍入噪声"

    def test_retinex_preserves_the_input_channel_order(self, default_config):
        """`retinex_correct` 不重排通道——所以调用方喂什么顺序、出来就是什么顺序。

        这正是"L82 的 RGB2GRAY 是错的"的判据：既然输入按 BGR 解释（L69），
        而 retinex 保序，那么这里按 RGB 转灰度就是错的。
        """
        pp = Preprocessor(copy.deepcopy(default_config))
        img = np.zeros((80, 80, 3), dtype=np.uint8)
        img[:, :40] = (60, 60, 200)      # 左半偏红：BGR 下 ch2 最大
        img[:, 40:] = (200, 60, 60)      # 右半偏蓝：BGR 下 ch0 最大

        out = pp.retinex_correct(img)

        left, right = out[:, :40], out[:, 40:]
        assert left[:, :, 2].mean() > left[:, :, 0].mean(), "左半应仍是 R 占优"
        assert right[:, :, 0].mean() > right[:, :, 2].mean(), "右半应仍是 B 占优"

    # ---- DefectDetector -----------------------------------------------

    def test_defect_detector_oxidation_assumes_bgr(self, default_config):
        """⚠ 钉住缺陷：`detect_oxidation` 无条件做 BGR2RGB，即假定输入是 BGR。

        它的 docstring 写"BGR 或 RGB 图像"，代码却只有一条 BGR 分支。
        """
        det = DefectDetector(default_config)
        copper = _solid(COPPER_BGR)

        as_bgr = det.detect_oxidation(copper)
        as_rgb = det.detect_oxidation(cv2.cvtColor(copper, cv2.COLOR_BGR2RGB))

        assert len(as_bgr) == 1
        assert len(as_rgb) == 0

    # ---- ProcessMonitor -----------------------------------------------

    def test_process_monitor_defaults_to_rgb(self):
        """⚠ 钉住缺陷：`ProcessMonitor` 的方向与上面所有旧路径**相反**。

        它默认按 RGB 解释，且 R4 明确要求调用方声明——在旧路径里这是唯一
        显式声明的一处，代价是与 `Preprocessor` / `DefectDetector` 不一致。
        """
        import inspect

        default = inspect.signature(
            ProcessGLCMExtractor.compute
        ).parameters["color_order"].default

        assert default == "rgb"

    def test_process_monitor_rejects_unknown_order(self):
        """它至少会挡住非法值——这一点与新模块的做法一致。"""
        with pytest.raises(ValueError, match="color_order"):
            ProcessGLCMExtractor._to_gray(_solid(COPPER_BGR), color_order="gbr")

    def test_process_monitor_gray_differs_between_orders(self):
        """同一份字节，两种顺序，灰度结果不同——说明这个参数真的有影响。"""
        img = _varied_color_image()

        as_rgb = ProcessGLCMExtractor._to_gray(img, color_order="rgb")
        as_bgr = ProcessGLCMExtractor._to_gray(img, color_order="bgr")

        assert not np.array_equal(as_rgb, as_bgr)

    def test_two_legacy_modules_disagree_on_the_same_bytes(self):
        """⚠ 钉住缺陷：同一份字节，两个旧模块按相反的顺序解释。

        `ProcessGLCMExtractor._to_gray` 默认 RGB，
        `DefectDetector.detect_oxidation` 无条件按 BGR。
        两边都没有错——错的是它们对"流水线里传的是什么"没有共识。

        `compute()` 的 docstring 还写着"DefectDetector 也按 RGB 处理"，
        而代码是 BGR。文档与实现对不上，所以这里断言的是**代码**。
        """
        img = _varied_color_image()

        monitor_gray = ProcessGLCMExtractor._to_gray(img, color_order="rgb")
        detector_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        assert np.array_equal(monitor_gray, detector_gray) is False

    def test_process_monitor_ignores_grayscale_input(self):
        """灰度输入直接原样返回——这是它比旧路径稳健的地方。"""
        gray = np.full((32, 32), 120, dtype=np.uint8)

        assert np.array_equal(
            ProcessGLCMExtractor._to_gray(gray, color_order="rgb"), gray
        )

    # ---- 入口路径 ------------------------------------------------------

    def test_gui_and_cli_feed_rgb_into_a_bgr_detector(self, default_config):
        """⚠ 钉住缺陷：生产入口喂 RGB，而氧化检测按 BGR 解释。

        GUI 在 `ui/main_window.py:570` 做了 BGR2RGB，
        CLI 在 `main.py:155` 做了 BGR2RGB，
        两边随后都把结果交给 `detect_all`，最终进 `detect_oxidation`。

        本用例复现这条链路：同一块金色铜板，经生产入口走一遭 → 检出 0 处。
        """
        det = DefectDetector(default_config)
        copper_bgr = _solid(COPPER_BGR)

        # 生产入口的写法
        fed_by_gui = cv2.cvtColor(copper_bgr, cv2.COLOR_BGR2RGB)
        assert len(det.detect_oxidation(fed_by_gui)) == 0

        # 采集器路径的写法（不做 BGR2RGB）
        assert len(det.detect_oxidation(copper_bgr)) == 1

    def test_acquisition_path_and_gui_path_disagree_end_to_end(self, default_config):
        """⚠ 钉住缺陷：同一块板、同一个流水线，两条路径给出不同的缺陷数。

        `process_acquisition` 喂的是 BGR（`pipeline.py` 显式传 color_order="bgr"），
        GUI/CLI 喂的是 RGB。铜面检测因此一条路径对、一条路径错。
        """
        pipeline = InspectionPipeline(default_config)
        copper_bgr = _solid(COPPER_BGR)

        from_acquisition = pipeline.process_acquisition(
            _OneFrameAcquisition(copper_bgr), max_frames=1
        )[0]

        # GUI/CLI 路径：入口做了 BGR2RGB
        from_gui = pipeline.run(
            cv2.cvtColor(copper_bgr, cv2.COLOR_BGR2RGB), "gui"
        )

        # 只看氧化类：纯色图没有纹理，"未粗化"检测器在两条路径上都会报，
        # 拿总缺陷数比会被它掩盖掉真正的差异。
        def _oxidation(result):
            return [d for d in result.defects if d.type == DefectDetector.OXIDATION]

        assert len(_oxidation(from_acquisition)) == 1
        assert len(_oxidation(from_gui)) == 0

    # ---- 阈值语义矛盾 --------------------------------------------------

    def test_oxidation_hue_window_contains_normal_copper(self, default_config):
        """⚠ 钉住缺陷：`h_range` 既是"氧化色相范围"又包含正常铜面色相。

        `core/defects.py` 的 docstring 说"正常喷砂铜面呈金黄色 H≈15°-30°"，
        `config/default.yaml` 的注释说这是"氧化色相范围（背离金色铜面）"——
        同一区间两种对立含义。实测正常铜面 H=17 正落在 [10,30] 内，
        所以"检出整幅图"是必然的，不是巧合。
        """
        ox = default_config["inspection"]["defects"]["oxidation"]
        low, high = ox["h_range"]

        assert low <= COPPER_HUE_UNITS <= high

    def test_roi_hue_window_also_contains_normal_copper(self, default_config):
        """同理：铜面 ROI 的色相窗口也包含正常铜面的色相。

        后果：ROI 判据分辨不出"正常铜面"与"氧化铜面"，而这正是色度指标要测的
        量。这也是 `core/color.py` 的 `use_roi` 默认关掉的理由之一。
        """
        upper = default_config["inspection"]["roi"]["copper_hsv_upper"]
        lower = default_config["inspection"]["roi"]["copper_hsv_lower"]

        assert lower[0] <= COPPER_HUE_UNITS <= upper[0]

    def test_color_absolute_baseline_does_not_reuse_oxidation_h_range(
        self, default_config
    ):
        """新模块的绝对阈值**另立门户**，不沿用 oxidation 的阈值。

        决策 8：`oxidation.h_range/s_min/v_min` 的语义本身就是矛盾的
        （见上面两条），沿用会把矛盾一起继承过来。
        """
        color = default_config["inspection"]["color"]["absolute"]
        ox = default_config["inspection"]["defects"]["oxidation"]

        assert "h_range" not in color
        assert "s_min" not in color
        assert color["hue_baseline_deg"] != ox["h_range"][0]
        assert color["sat_baseline"] != ox["s_min"]


# ============================================================================
# ROI 掩膜的通道假设（钉住缺陷）
# ============================================================================

class TestRoiMaskAssumesBgr:
    """⚠ 钉住缺陷：`get_roi_mask` 把 `COLOR_BGR2HSV` 写死。"""

    def test_roi_mask_covers_everything_for_bgr_copper(self, default_config):
        roi = default_config["inspection"]["roi"]
        lower = np.array(roi["copper_hsv_lower"])
        upper = np.array(roi["copper_hsv_upper"])

        mask = Preprocessor.get_roi_mask(_solid(COPPER_BGR), lower, upper)

        assert mask.mean() / 255 * 100 == pytest.approx(100.0)

    def test_roi_mask_covers_nothing_for_rgb_input(self, default_config):
        """⚠ 钉住缺陷：喂 RGB 时 ROI 覆盖率为 0%。

        不会报错，只是整个 ROI 变成空集。上游若拿它做掩膜，结果会静默变成
        "全图无有效区域"。
        """
        roi = default_config["inspection"]["roi"]
        lower = np.array(roi["copper_hsv_lower"])
        upper = np.array(roi["copper_hsv_upper"])

        rgb = cv2.cvtColor(_solid(COPPER_BGR), cv2.COLOR_BGR2RGB)
        mask = Preprocessor.get_roi_mask(rgb, lower, upper)

        assert mask.mean() == 0.0

    def test_color_analyzer_does_not_reuse_get_roi_mask(self, default_config):
        """新模块**不复用** `get_roi_mask`——它把通道顺序写死了，与 C2 冲突。

        这条用例是"有意为之"的书面记录，防止日后有人"顺手复用一下"。
        """
        analyzer = ColorAnalyzer(default_config)
        assert analyzer.use_roi is False, "ROI 默认关，理由见 core/color.py"

        # 即使开着，`_selection_mask` 也走自己的路径，不再有"顺序写死"的问题
        img = _solid(COPPER_BGR)
        as_bgr = analyzer._selection_mask(img, "bgr").mean()
        as_rgb = analyzer._selection_mask(img, "rgb").mean()
        assert as_bgr > 0.0 and as_rgb > 0.0


# ============================================================================
# 连通域实现的契约：两套实现必须保持一致
# ============================================================================

class TestLocalizationContract:
    """`ColorAnalyzer._localize` 与 `DefectDetector._extract_connected_components`。

    两边**刻意**不复用（见 `_localize` 的 docstring：Defect 的 `type`/`severity`
    对色彩区域没有意义）。代价是可能各自漂移，所以用这组用例钉住：
    形态学处理与几何量的算法必须等价。
    """

    @staticmethod
    def _mask(h: int = 200, w: int = 200) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[20:60, 30:90] = 255
        mask[120:150, 100:170] = 255
        return mask

    def _localize(self, analyzer, mask, h=200, w=200):
        hue = np.full((h, w), float(COPPER_HUE_UNITS))
        sat = np.full((h, w), 178.0)
        valid = np.ones((h, w), dtype=bool)
        return analyzer._localize(
            mask, "absolute", hue, sat, valid, h * w,
            baseline_deg=COPPER_HUE_DEG, hue_tol=25.0,
            sat_baseline=178.0, sat_tol=50.0,
        )

    def test_geometry_matches_the_defect_detector(self, default_config):
        """同样的掩膜、同样的形态学预处理 → 包围盒、面积、质心完全一致。

        掩膜用规整矩形，开闭运算不改变它，所以两边比的是**同一张图**上的
        几何量，比的确实是算法而不是形态学差异。
        """
        analyzer = ColorAnalyzer(default_config)
        detector = DefectDetector(default_config)
        mask = self._mask()

        regions = self._localize(analyzer, mask)

        # 用分析器自己的核做同样的形态学处理，再交给缺陷检测器
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (analyzer.morph_kernel, analyzer.morph_kernel)
        )
        morphed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, k)
        defects = detector._extract_connected_components(
            morphed, DefectDetector.OXIDATION, analyzer.area_min_px
        )

        assert [r.bbox for r in regions] == [d.bbox for d in defects]
        assert [r.area_pixels for r in regions] == [d.area_pixels for d in defects]
        assert [tuple(map(int, r.centroid)) for r in regions] == \
               [d.centroid for d in defects]

    def test_area_min_threshold_is_inclusive_on_both_sides(self, default_config):
        """``area < area_min_px`` 才丢弃——正好等于阈值的连通域要保留。

        两边的比较运算符必须一致。这里在**缺陷检测器**上验谓词本身
        （它不做形态学，面积可控）；`_localize` 侧的形态学会改变面积，
        所以它的阈值读取另有一条用例。
        """
        detector = DefectDetector(default_config)
        area_min = 100

        for area, expected in ((area_min - 1, 0), (area_min, 1), (area_min + 1, 1)):
            # 单行连通域，面积精确等于像素数；画布要够宽，否则切片会被截断
            mask = np.zeros((10, area + 10), dtype=np.uint8)
            mask[5, 5:5 + area] = 255
            got = len(detector._extract_connected_components(
                mask, DefectDetector.OXIDATION, area_min
            ))
            assert got == expected, f"面积 {area}、阈值 {area_min} 时应有 {expected} 处"

    def test_localize_reads_its_area_min_from_config(self, default_config):
        """`_localize` 的面积阈值取自 `inspection.color.area_min_px`。

        刻意不拿"正好压线"的面积去比对——形态学开闭会改变连通域面积，
        那样测到的是形态学而不是阈值。这里把阈值抬到明显高于实际面积，
        验证过滤确实生效、且确实读的是这个键。
        """
        mask = self._mask()          # 两块：开闭后约 2392 / 2092 像素

        high = copy.deepcopy(default_config)
        high["inspection"]["color"]["area_min_px"] = 5000
        assert self._localize(ColorAnalyzer(high), mask) == []

        low = copy.deepcopy(default_config)
        low["inspection"]["color"]["area_min_px"] = 1000
        assert len(self._localize(ColorAnalyzer(low), mask)) == 2

    def test_morphology_despeckles_in_localize_but_not_in_extract(self, default_config):
        """两边**有意**的差异：`_localize` 先做形态学去噪，`_extract` 不做。

        这个差异是设计的一部分（色彩区域需要去噪，缺陷掩膜在调用方已经清理过），
        钉在这里免得日后被人当成 bug "修"掉。
        """
        analyzer = ColorAnalyzer(default_config)
        detector = DefectDetector(default_config)

        tiny = np.zeros((200, 200), dtype=np.uint8)
        tiny[5:8, 5:8] = 255          # 3×3，小于形态学核（默认 5×5）

        assert self._localize(analyzer, tiny) == [], "分析器应把噪点开运算掉"
        assert len(detector._extract_connected_components(
            tiny, DefectDetector.OXIDATION, 1
        )) == 1, "缺陷检测器不做形态学，原样保留"

    def test_analyzer_morph_kernel_is_the_shared_convention(self, default_config):
        """形态学核的尺寸来自配置，两边读的是同一个键。"""
        analyzer = ColorAnalyzer(default_config)

        assert analyzer.morph_kernel == default_config["inspection"]["color"][
            "morph_kernel"
        ]
        assert analyzer.area_min_px == default_config["inspection"]["color"][
            "area_min_px"
        ]
