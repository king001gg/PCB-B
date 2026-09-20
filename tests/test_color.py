"""core/color.py 单元测试 —— 色度 / 饱和度指标。

这个模块有两类东西必须钉死：

  (1) **圆统计**。色相是圆量，线性均值是错的，而且错得不明显：半圈 H=179
      (358°) 加半圈 H=1 (2°)，线性均值给 180°（恰好是最远的点），圆均值给
      0°。这类错误不会崩，只会安静地算出一个看起来合理的数字。
      同理，圆标准差 sqrt(-2·lnR) 在 R→0 时发散，必须被夹住。

  (2) **口径**。通道顺序、灰度输入、越界判据的独立性 —— 见模块 docstring
      的 C1–C7。喂错通道顺序不报错、只静默把色相转掉 172°，所以必须有
      用例直接断言这件事。

阈值全部走 config/default.yaml 的真配置（夹具 default_config），不在测试里
另写一份，配置改了这里会第一时间炸。
"""

import numpy as np
import pytest

from core.color import (
    CIRC_STD_MAX_DEG,
    ColorAnalyzer,
    ColorFeatures,
    ColorRegion,
    HUE_UNIT_TO_DEG,
    MAD_TO_SIGMA,
)


# ============================================================================
# 夹具与工具
# ============================================================================

# 实测的金色铜面：BGR=(60,140,200) → H=17(OpenCV 单位)=34°, S=178, V=200
COPPER_BGR = (60, 140, 200)
COPPER_HUE_DEG = 34.0
COPPER_SAT = 178.0


def _board(bgr, shape=(120, 120)) -> np.ndarray:
    """整幅单一颜色的假板。"""
    img = np.zeros((*shape, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


@pytest.fixture
def analyzer(default_config) -> ColorAnalyzer:
    """用真配置构造分析器。

    test_color_config.py 另有一组用「合成配置」的用例，用来验证缺项回退 ——
    那组不能依赖真配置里恰好有哪些键，所以两种夹具并存。
    """
    return ColorAnalyzer(default_config)


@pytest.fixture
def make_analyzer():
    """在真配置基础上覆盖 inspection.color 下的若干键。"""

    def _make(default_config, **overrides) -> ColorAnalyzer:
        import copy
        cfg = copy.deepcopy(default_config)
        cfg.setdefault("inspection", {}).setdefault("color", {}).update(overrides)
        return ColorAnalyzer(cfg)

    return _make


# ============================================================================
# 圆统计
# ============================================================================

class TestCircularStats:
    """Mardia 圆统计量。这里每一个断言都对应一种「线性做法会算错」的情形。"""

    def test_wraparound_mean_is_not_the_linear_mean(self, analyzer):
        """跨 0° 的两簇：圆均值 ≈ 0°，而线性均值给 180°（最远的点）。

        H 单位 179 → 358°，H 单位 1 → 2°。两者实际只差 4°，但线性均值会把
        它们扔到圆的正对面。
        """
        units = np.array([179.0] * 50 + [1.0] * 50)
        deg = units * HUE_UNIT_TO_DEG

        assert deg.mean() == pytest.approx(180.0, abs=1.0)   # 线性：错的

        mean_deg, _, r = analyzer._circular_stats(deg)
        assert min(mean_deg, 360.0 - mean_deg) == pytest.approx(0.0, abs=1.0)
        assert r == pytest.approx(1.0, abs=1e-3)             # 两簇几乎重合

    def test_identical_angles_give_zero_std_and_full_concentration(self, analyzer):
        """完全一致的色相：标准差为 0（到 1e-3 度量级即可）。

        容差不能卡到 1e-6：``sqrt(-2·lnR)`` 在 R 贴近 1 时对舍入极敏感，
        R=1-5e-13 会给出约 1.2e-6 度。那是数值噪声，不是真实的离散。
        """
        mean_deg, std_deg, r = analyzer._circular_stats(np.array([34.0] * 100))

        assert mean_deg == pytest.approx(34.0)
        assert std_deg == pytest.approx(0.0, abs=1e-3)
        assert r == pytest.approx(1.0)

    def test_antipodal_clusters_hit_the_clamp_exactly(self, analyzer):
        """正对面两簇：合向量恰好为 0 → 标准差取到上限 180°，且不 NaN。

        这是确定性构造成 R=0 的用例（cos 半圈 +1 半圈 -1 相消），不靠随机
        采样的运气。不夹的话 sqrt(-2·lnR) 会给出 inf/NaN。
        """
        angles = np.array([0.0] * 500 + [180.0] * 500)

        _, std_deg, r = analyzer._circular_stats(angles)

        assert r == pytest.approx(0.0, abs=1e-12)
        assert std_deg == CIRC_STD_MAX_DEG
        assert np.isfinite(std_deg)

    def test_uniform_noise_never_exceeds_the_clamp(self, analyzer):
        """均匀分布（真实方向完全分散）：标准差必须被夹住，不得发散。

        不夹的话这里会得到 400° 以上的数字（实测纯噪声给过 425.9°）。
        """
        rng = np.random.default_rng(20260920)
        noise = rng.uniform(0.0, 360.0, 20000)

        _, std_deg, r = analyzer._circular_stats(noise)

        assert r < 0.05
        assert std_deg <= CIRC_STD_MAX_DEG
        assert std_deg > 150.0, "均匀分布本应几乎完全分散"

    def test_std_is_never_negative(self, analyzer):
        """R 恰好为 1 时 -2·ln(1) 会给出 -0.0，圆标准差按定义非负。"""
        _, std_deg, _ = analyzer._circular_stats(np.array([10.0] * 100))
        assert std_deg >= 0.0
        assert not np.signbit(std_deg), "不能是 -0.0"

    def test_concentration_cannot_exceed_one(self, analyzer):
        """hypot 的浮点误差会让 R 略大于 1，进而 log(R)>0、根号下负数 → NaN。

        NaN 会顺着报告一路污染到数据库，所以 R 必须夹在 [0,1]。
        """
        for angles in ([10.0] * 100, [0.0] * 50, [359.9] * 77):
            _, std_deg, r = analyzer._circular_stats(np.array(angles))
            assert 0.0 <= r <= 1.0
            assert np.isfinite(std_deg)

    def test_single_pixel(self, analyzer):
        mean_deg, std_deg, r = analyzer._circular_stats(np.array([123.0]))

        assert mean_deg == pytest.approx(123.0)
        assert std_deg == pytest.approx(0.0, abs=1e-6)
        assert r == pytest.approx(1.0)

    def test_empty_input_is_total_dispersion(self, analyzer):
        mean_deg, std_deg, r = analyzer._circular_stats(np.array([]))

        assert mean_deg == 0.0
        assert std_deg == CIRC_STD_MAX_DEG
        assert r == 0.0

    def test_std_grows_as_spread_grows(self, analyzer):
        """单调性：铺得越开，圆标准差越大。"""
        tight = analyzer._circular_stats(np.array([34.0, 35.0, 33.0, 34.5]))[1]
        loose = analyzer._circular_stats(np.array([0.0, 90.0, 180.0, 270.0]))[1]

        assert tight < loose


class TestCircularMedian:
    """圆中位数用 180 桶直方图穷举，量化下限 1 单位 = 2°。"""

    def test_wraparound_median(self, analyzer):
        units = np.array([179.0] * 50 + [1.0] * 50)
        median_units = analyzer._circular_median_units(units)

        # 结果只能是 0（=0°）或 179（=358°），两者都贴着真实的中心
        assert median_units in (0.0, 179.0)

    def test_quantization_floor_is_two_degrees(self, analyzer):
        """中位数只能落在整数单位上 —— 这是自适应容差不能低于约 4° 的原因。"""
        median_units = analyzer._circular_median_units(np.array([10.0, 11.0, 12.0]))
        assert median_units == int(median_units)

    def test_empty_returns_zero(self, analyzer):
        assert analyzer._circular_median_units(np.array([])) == 0.0

    def test_out_of_range_units_are_clipped(self, analyzer):
        """越界输入不得让 bincount 下标越界崩溃。"""
        assert analyzer._circular_median_units(np.array([255.0, -3.0])) >= 0.0


class TestCircularDistance:
    """环形距离落在 [0, 180]。"""

    @pytest.mark.parametrize("a,b,expected", [
        (10.0, 20.0, 10.0),
        (350.0, 10.0, 20.0),     # 跨 0°
        (0.0, 180.0, 180.0),     # 正对面
        (0.0, 181.0, 179.0),
        (34.0, 34.0, 0.0),
    ])
    def test_distance(self, analyzer, a, b, expected):
        assert analyzer._circular_distance_deg(a, b) == pytest.approx(expected)

    def test_symmetric(self, analyzer):
        assert (analyzer._circular_distance_deg(10.0, 200.0)
                == analyzer._circular_distance_deg(200.0, 10.0))


class TestRobustSigma:
    """MAD → σ。系数 1.4826 与 SPC 3σ 同源。"""

    def test_scales_mad_by_consistency_factor(self, analyzer):
        x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])   # 中位数 3，偏差中位数 1
        assert analyzer._robust_sigma(x) == pytest.approx(MAD_TO_SIGMA * 1.0)

    def test_outlier_does_not_inflate_sigma(self, analyzer):
        """这正是用 MAD 而不是标准差的原因：孤立极值不该把阈值撑开。"""
        clean = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        with_outlier = np.array([10.0, 11.0, 12.0, 13.0, 10000.0])

        assert (analyzer._robust_sigma(with_outlier)
                == pytest.approx(analyzer._robust_sigma(clean)))

    def test_empty_is_zero(self, analyzer):
        assert analyzer._robust_sigma(np.array([])) == 0.0


# ============================================================================
# 输入口径（C1–C4）
# ============================================================================

class TestInputContract:
    """灰度、非法顺序、空输入 —— 都不准抛异常，也不准静默填 0。"""

    def test_gray_input_is_unavailable_not_zero(self, analyzer):
        """C4：灰度输入标为未测，绝不填 0（0 会被读成满分）。"""
        feats = analyzer.analyze(np.full((60, 60), 128, dtype=np.uint8))

        assert feats.available is False
        assert feats.note
        assert feats.hue_mean_deg is None
        assert feats.sat_mean is None
        assert feats.oor_abs_pct == 0.0
        assert feats.regions == []

    def test_gray_broadcast_to_three_channels_is_still_measurable(self, analyzer):
        """三通道但完全中性的输入：测得到，但提示"整板近中性"。

        这不是灰度输入，是真彩色相机拍到的灰色板 —— 两者要区分开。
        """
        feats = analyzer.analyze(_board((128, 128, 128)))

        assert feats.available is True
        assert feats.hue_mean_deg is None          # S=0，无有效色相像素
        assert feats.sat_mean == pytest.approx(0.0)
        assert "近中性" in feats.note

    def test_none_input_is_unavailable(self, analyzer):
        feats = analyzer.analyze(None)
        assert feats.available is False
        assert feats.note

    def test_two_channel_input_is_unavailable(self, analyzer):
        feats = analyzer.analyze(np.zeros((10, 10, 2), dtype=np.uint8))
        assert feats.available is False

    def test_four_channel_input_drops_alpha(self, analyzer):
        """BGRA/RGBA 取前三通道，不该崩。"""
        img = np.zeros((40, 40, 4), dtype=np.uint8)
        img[:, :, :3] = COPPER_BGR
        img[:, :, 3] = 255

        feats = analyzer.analyze(img, color_order="bgr")

        assert feats.available is True
        assert feats.hue_mean_deg == pytest.approx(COPPER_HUE_DEG, abs=1.0)

    def test_bad_color_order_raises_at_construction(self, default_config):
        """C2：顺序必须显式声明，非法值在构造期就报 —— 早失败好过每帧算错。"""
        import copy
        cfg = copy.deepcopy(default_config)
        cfg["inspection"]["color"]["color_order"] = "gbr"

        with pytest.raises(ValueError, match="color_order"):
            ColorAnalyzer(cfg)

    def test_bad_color_order_raises_at_call(self, analyzer):
        with pytest.raises(ValueError, match="color_order"):
            analyzer.analyze(_board(COPPER_BGR), color_order="bgrr")

    def test_color_order_is_case_insensitive(self, default_config):
        import copy
        cfg = copy.deepcopy(default_config)
        cfg["inspection"]["color"]["color_order"] = "BGR"

        assert ColorAnalyzer(cfg).color_order == "bgr"

    def test_disabled_analyzer_returns_unavailable(self, make_analyzer,
                                                   default_config):
        a = make_analyzer(default_config, enabled=False)
        feats = a.analyze(_board(COPPER_BGR), color_order="bgr")

        assert feats.available is False
        assert "未启用" in feats.note

    def test_call_order_overrides_configured_order(self, analyzer):
        """analyze() 的入参优先于配置 —— 同一实例要能服务多条路径。"""
        as_bgr = analyzer.analyze(_board(COPPER_BGR), color_order="bgr")
        as_rgb = analyzer.analyze(_board(COPPER_BGR), color_order="rgb")

        assert as_bgr.hue_mean_deg != pytest.approx(as_rgb.hue_mean_deg)


# ============================================================================
# 通道顺序（C3）
# ============================================================================

class TestChannelOrder:
    """喂错顺序不报错、只静默转掉色相 —— 所以必须有直接断言。"""

    def test_saturation_and_value_are_order_invariant(self, analyzer):
        """S、V 与通道顺序无关；只有 H 受影响。"""
        img = _board(COPPER_BGR)

        as_bgr = analyzer.analyze(img, color_order="bgr")
        as_rgb = analyzer.analyze(img, color_order="rgb")

        assert as_bgr.sat_mean == pytest.approx(as_rgb.sat_mean)
        assert as_bgr.val_mean == pytest.approx(as_rgb.val_mean)

    def test_hue_shifts_by_172_degrees_when_order_is_wrong(self, analyzer):
        """金色铜面 H=34°；按 RGB 解释同一份 BGR 字节会得到 206°，差 172°。

        这个数字是本模块存在的理由之一：它不报错，只是把"正常铜面"读成
        完全另一个颜色。
        """
        img = _board(COPPER_BGR)

        right = analyzer.analyze(img, color_order="bgr").hue_mean_deg
        wrong = analyzer.analyze(img, color_order="rgb").hue_mean_deg

        assert right == pytest.approx(COPPER_HUE_DEG, abs=1.0)
        assert wrong == pytest.approx(206.0, abs=1.0)
        assert analyzer._circular_distance_deg(right, wrong) == pytest.approx(
            172.0, abs=2.0
        )

    def test_wrong_order_is_not_detectable_from_shape(self, analyzer):
        """反证：两种顺序下数组形状完全相同 —— 所以无法靠形状推断。

        这条是 C2「必须显式声明」的直接依据。
        """
        img = _board(COPPER_BGR)
        a = analyzer.analyze(img, color_order="bgr")
        b = analyzer.analyze(img, color_order="rgb")

        assert a.pixel_count == b.pixel_count
        assert a.valid_hue_pixels == b.valid_hue_pixels
        assert a.color_order != b.color_order


# ============================================================================
# 绝对判据
# ============================================================================

class TestAbsoluteCriteria:
    """绝对阈值抓整板均匀变色 —— 面积法的结构性盲区。"""

    def test_baseline_copper_is_clean(self, analyzer):
        feats = analyzer.analyze(_board(COPPER_BGR), color_order="bgr")

        assert feats.available is True
        assert feats.hue_deviation_deg == pytest.approx(0.0, abs=1.0)
        assert feats.sat_mean == pytest.approx(COPPER_SAT, abs=1.0)
        assert feats.oor_abs_pct == 0.0
        assert feats.regions == []

    def test_uniform_discoloration_flags_the_whole_board(self, make_analyzer,
                                                         default_config):
        """整板均匀偏色：绝对判据应覆盖几乎全部像素。

        这正是面积法抓不到的情形 —— 没有任何连通域，但整板都变了色。
        """
        a = make_analyzer(default_config, adaptive={"enabled": False})
        feats = a.analyze(_board((200, 60, 60)), color_order="bgr")   # B=200 → 蓝

        assert feats.oor_abs_pct > 95.0
        assert feats.hue_deviation_deg > 25.0

    def test_saturation_drop_flags_even_without_hue_shift(self, make_analyzer,
                                                          default_config):
        """饱和度塌陷（近中性灰）时色相已不可测，但饱和判据仍要报出来。

        这是刻意保留的一路：均匀薄氧化膜把 S 压下去的同时 H 会变得不可靠，
        若两路都依赖色相就会一起失效。
        """
        a = make_analyzer(default_config, adaptive={"enabled": False})
        feats = a.analyze(_board((150, 150, 150)), color_order="bgr")

        assert feats.sat_mean == pytest.approx(0.0, abs=1.0)
        assert feats.hue_mean_deg is None            # 色相不可测
        assert feats.oor_abs_pct > 95.0              # 但饱和判据照样命中

    def test_tolerance_boundary_is_respected(self, make_analyzer, default_config):
        """恰好压在容差上的颜色不算越界，略微超过才算。

        这里用饱和度做探针（色相要构造精确角度更绕），边界两侧各取一点。
        """
        base = COPPER_SAT
        tol = default_config["inspection"]["color"]["absolute"]["sat_tolerance"]

        # 造一个 S 恰好比基准低 tol 的颜色：R 固定 200，调 B 抬升最暗通道
        inside = make_analyzer(
            default_config,
            adaptive={"enabled": False},
            absolute={"sat_tolerance": tol},
        )
        # S = (max-min)/max*255 = tol_off → min = 200*(1 - tol_off/255)
        def _bgr_with_sat(sat):
            mn = int(round(200 * (1.0 - sat / 255.0)))
            return (mn, 140, 200)

        just_inside = inside.analyze(_board(_bgr_with_sat(base - tol + 3)),
                                     color_order="bgr")
        just_outside = inside.analyze(_board(_bgr_with_sat(base - tol - 6)),
                                      color_order="bgr")

        assert just_inside.oor_abs_pct == 0.0
        assert just_outside.oor_abs_pct > 90.0

    def test_absolute_can_be_disabled(self, make_analyzer, default_config):
        a = make_analyzer(default_config,
                          absolute={"enabled": False},
                          adaptive={"enabled": False})
        feats = a.analyze(_board((200, 60, 60)), color_order="bgr")

        assert feats.oor_abs_pct == 0.0
        assert feats.oor_adaptive_pct == 0.0
        assert feats.hue_deviation_deg is None


# ============================================================================
# 自适应判据
# ============================================================================

class TestAdaptiveCriteria:
    """自适应阈值抓局部异常 —— 油污 / 水渍 / 指纹这一路。"""

    def _patch_board(self, patch_bgr, size=100, patch=(20, 40, 20, 40)):
        """铜面底 + 一块异色补丁。``patch`` 是 ``(y1, y2, x1, x2)``。

        默认 20×20 = 400 像素，在 100×100 上恰是 4%。
        """
        img = _board(COPPER_BGR, shape=(size, size))
        y1, y2, x1, x2 = patch
        img[y1:y2, x1:x2] = patch_bgr
        return img

    def test_local_anomaly_is_caught(self, analyzer):
        """补丁色相偏离 16°：超过自适应容差 8°，但仍在绝对容差 25° 内。

        所以只有自适应判据该命中 —— 这正好证明两套判据是独立的（C6）。
        """
        # B=60,G=177,R=200 → H=50°, S=178（与铜面同饱和，隔离色相这一路）
        feats = analyzer.analyze(self._patch_board((60, 177, 200)),
                                 color_order="bgr")

        assert feats.oor_adaptive_pct == pytest.approx(4.0, abs=1.0)
        assert feats.oor_abs_pct == 0.0, "补丁偏离在绝对容差内，不该命中绝对判据"

    def test_uniform_board_triggers_no_adaptive_outlier(self, analyzer):
        """整板一个颜色：自适应判据找不到异常（MAD=0，只剩容差下限）。"""
        feats = analyzer.analyze(_board(COPPER_BGR), color_order="bgr")

        assert feats.oor_adaptive_pct == 0.0
        assert feats.adaptive_hue_tol_deg == pytest.approx(
            analyzer.min_hue_tol, abs=1e-9
        )

    def test_wildly_off_board_triggers_absolute_only(self, analyzer):
        """整板大幅偏色：绝对判据全命中，自适应仍然沉默。

        这正是"两套都算"的意义 —— 只看自适应会漏掉整板均匀变色，只看绝对
        会把正常的批次色差当成缺陷。
        """
        feats = analyzer.analyze(_board((200, 60, 60)), color_order="bgr")

        assert feats.oor_abs_pct > 95.0
        assert feats.oor_adaptive_pct == 0.0

    def test_baseline_follows_this_frame_not_a_constant(self, analyzer):
        """自适应基线来自本帧分布，所以换一批偏色的板它跟着走。"""
        copper = analyzer.analyze(_board(COPPER_BGR), color_order="bgr")
        bluish = analyzer.analyze(_board((200, 60, 60)), color_order="bgr")

        assert copper.adaptive_hue_baseline_deg == pytest.approx(34.0, abs=1.0)
        assert bluish.adaptive_hue_baseline_deg == pytest.approx(240.0, abs=2.0)

    def test_too_few_valid_pixels_skips_adaptive_only(self, make_analyzer,
                                                      default_config):
        """有效像素不足 → 自适应留空，但绝对判据与 S/V 统计照常出。"""
        a = make_analyzer(default_config, adaptive={"min_valid_pixels": 10 ** 9})
        feats = a.analyze(_board((200, 60, 60)), color_order="bgr")

        assert feats.adaptive_hue_baseline_deg is None
        assert feats.adaptive_sat_baseline is None
        assert feats.oor_adaptive_pct == 0.0
        assert feats.oor_abs_pct > 95.0, "像素不足不该把绝对判据一起废掉"
        assert "自适应基线不可信" in feats.note

    def test_adaptive_can_be_disabled(self, make_analyzer, default_config):
        a = make_analyzer(default_config, adaptive={"enabled": False})
        feats = a.analyze(self._patch_board((60, 177, 200)), color_order="bgr")

        assert feats.adaptive_hue_baseline_deg is None
        assert feats.oor_adaptive_pct == 0.0

    def test_mad_k_widens_tolerance(self, make_analyzer, default_config):
        """mad_k 越大阈值越宽 —— 单调性，防止公式写反。"""
        img = self._patch_board((60, 177, 200))
        narrow = make_analyzer(default_config, adaptive={"mad_k": 0.1})
        wide = make_analyzer(default_config, adaptive={"mad_k": 50.0})

        assert (narrow.analyze(img, color_order="bgr").oor_adaptive_pct
                >= wide.analyze(img, color_order="bgr").oor_adaptive_pct)


# ============================================================================
# 越界区域定位
# ============================================================================

class TestLocalization:
    """几处、多大、在哪。"""

    def test_region_geometry_matches_the_patch(self, analyzer):
        img = _board(COPPER_BGR, shape=(100, 100))
        img[30:70, 10:50] = (200, 60, 60)          # 40×40 的蓝色块

        feats = analyzer.analyze(img, color_order="bgr")
        absolute = [r for r in feats.regions if r.basis == "absolute"]

        assert len(absolute) == 1
        r = absolute[0]
        assert r.area_pixels == pytest.approx(40 * 40, rel=0.05)
        assert r.area_pct == pytest.approx(16.0, abs=0.5)
        assert r.bbox == pytest.approx((10, 30, 50, 70), abs=1)
        assert r.centroid[0] == pytest.approx(30.0, abs=1.0)
        assert r.centroid[1] == pytest.approx(50.0, abs=1.0)

    def test_regions_from_both_bases_are_not_merged(self, analyzer):
        """同一块区域被两套判据都判出时，各出一条、basis 不同（C6）。"""
        img = _board(COPPER_BGR, shape=(100, 100))
        img[30:70, 10:50] = (200, 60, 60)

        feats = analyzer.analyze(img, color_order="bgr")
        bases = sorted(r.basis for r in feats.regions)

        assert bases == ["absolute", "adaptive"]
        assert feats.oor_count == 2
        areas = {r.area_pixels for r in feats.regions}
        assert len(areas) == 1, "同一块区域，两套判据给出的面积应一致"

    def test_small_speck_is_filtered_by_area_min(self, make_analyzer,
                                                 default_config):
        """小于 area_min_px 的连通域不单独成区域。"""
        a = make_analyzer(default_config, area_min_px=500)
        img = _board(COPPER_BGR, shape=(100, 100))
        img[10:14, 10:14] = (200, 60, 60)          # 16 像素，远小于 500

        feats = a.analyze(img, color_order="bgr")

        assert feats.regions == []
        # 但像素计数不受影响 —— 面积占比仍如实统计
        assert feats.oor_abs_pixels > 0
        assert feats.oor_abs_pct > 0.0

    def test_max_regions_truncates_and_notes_it(self, make_analyzer,
                                                default_config):
        """区域过多时按面积降序截断，并在 note 里说明 —— 不能悄悄丢。

        这里关掉自适应判据，让每块区域只出一条（否则同一块会被两套判据各记
        一条，面积相同，截断结果就不只看面积了）。
        """
        a = make_analyzer(default_config, max_regions=2, area_min_px=4,
                          adaptive={"enabled": False})
        img = _board(COPPER_BGR, shape=(100, 100))
        # 四块面积递减的越界块：20²=400 > 16²=256 > 12²=144 > 8²=64
        for y, x, s in [(5, 5, 20), (40, 5, 16), (5, 40, 12), (40, 40, 8)]:
            img[y:y + s, x:x + s] = (200, 60, 60)

        feats = a.analyze(img, color_order="bgr")

        assert len(feats.regions) == 2
        assert feats.regions[0].area_pixels > feats.regions[1].area_pixels
        assert feats.regions[0].area_pixels == pytest.approx(400, rel=0.05)
        assert feats.regions[1].area_pixels == pytest.approx(256, rel=0.05)
        assert "仅保留" in feats.note

    def test_area_mm2_uses_resolution(self, make_analyzer, default_config):
        """mm² 换算必须用 system.resolution_mm_per_pixel，不能用默认 0.01。"""
        a = make_analyzer(default_config)
        img = _board(COPPER_BGR, shape=(100, 100))
        img[30:70, 10:50] = (200, 60, 60)

        feats = a.analyze(img, color_order="bgr")
        r = feats.regions[0]
        mm = default_config["system"]["resolution_mm_per_pixel"]

        assert r.area_mm2 == pytest.approx(r.area_pixels * mm * mm, rel=1e-6)

    def test_morphology_removes_salt_noise(self, analyzer):
        """椒盐噪声经开运算后不该被当成几十处越界区域。"""
        rng = np.random.default_rng(7)
        img = _board(COPPER_BGR, shape=(100, 100))
        mask = rng.random((100, 100)) < 0.02
        img[mask] = (200, 60, 60)                  # 2% 的孤立蓝点

        feats = analyzer.analyze(img, color_order="bgr")

        assert len(feats.regions) < 10, "孤立噪点应被形态学开运算清掉"

    def test_region_reports_its_own_hue_and_sat(self, analyzer):
        img = _board(COPPER_BGR, shape=(100, 100))
        img[30:70, 10:50] = (200, 60, 60)

        feats = analyzer.analyze(img, color_order="bgr")
        r = [x for x in feats.regions if x.basis == "absolute"][0]

        assert r.mean_hue_deg == pytest.approx(240.0, abs=2.0)
        assert r.mean_sat == pytest.approx(178.0, abs=2.0)
        assert r.hue_deviation_deg > 25.0
        assert r.severity > 1.0, "越界倍数应大于 1"
        assert r.reason

    def test_region_hue_ignores_near_neutral_pixels(self, analyzer):
        """区域内的近中性像素不参与色相均值（C5），否则均值是噪声。"""
        img = _board(COPPER_BGR, shape=(100, 100))
        img[30:70, 10:50] = (200, 60, 60)
        img[35:45, 15:25] = (128, 128, 128)        # 区域内的灰色小块

        feats = analyzer.analyze(img, color_order="bgr")
        r = [x for x in feats.regions if x.basis == "absolute"][0]

        assert r.mean_hue_deg == pytest.approx(240.0, abs=2.0)


# ============================================================================
# 序列化
# ============================================================================

class TestColorFeaturesSerialization:

    def test_to_dict_includes_regions(self, analyzer):
        img = _board(COPPER_BGR, shape=(100, 100))
        img[30:70, 10:50] = (200, 60, 60)

        data = analyzer.analyze(img, color_order="bgr").to_dict()

        assert "regions" in data
        assert isinstance(data["regions"], list)
        assert data["regions"][0]["basis"] in ("absolute", "adaptive")
        assert "reason" in data["regions"][0]

    def test_to_dict_preserves_none(self, analyzer):
        """未测字段必须是 None，不能变成 0。"""
        data = analyzer.analyze(np.full((40, 40), 1, dtype=np.uint8)).to_dict()

        assert data["available"] is False
        assert data["hue_mean_deg"] is None
        assert data["sat_mean"] is None

    def test_to_dict_is_json_serializable(self, analyzer):
        """None / numpy 标量都不得漏出去 —— 报表层要直接 json.dumps。"""
        import json
        img = _board(COPPER_BGR, shape=(60, 60))
        img[20:40, 20:40] = (200, 60, 60)

        data = analyzer.analyze(img, color_order="bgr").to_dict()
        json.dumps(data)          # 不抛即通过

    def test_unavailable_classmethod_shape(self):
        feats = ColorFeatures.unavailable("测试原因", "rgb")

        assert feats.available is False
        assert feats.note == "测试原因"
        assert feats.color_order == "rgb"
        assert feats.regions == []
        assert feats.oor_count == 0


class TestColorRegionSerialization:

    def test_to_dict_rounds_and_lists(self):
        r = ColorRegion(basis="absolute", reason="r", area_pixels=100,
                        area_mm2=0.0123456, area_pct=1.23456,
                        centroid=(1.234, 5.678), bbox=(1, 2, 3, 4),
                        mean_hue_deg=34.567, mean_sat=177.77,
                        hue_deviation_deg=12.345, severity=1.2345)

        d = r.to_dict()

        assert d["area_mm2"] == pytest.approx(0.012346, abs=1e-9)
        assert d["area_pct"] == pytest.approx(1.2346, abs=1e-9)
        assert d["centroid"] == [1.23, 5.68]
        assert d["bbox"] == [1, 2, 3, 4]      # list，便于 JSON

    def test_to_dict_preserves_none_hue(self):
        r = ColorRegion(mean_hue_deg=None, hue_deviation_deg=None)
        d = r.to_dict()

        assert d["mean_hue_deg"] is None
        assert d["hue_deviation_deg"] is None


# ============================================================================
# ROI（默认关，但开关要真的有效）
# ============================================================================

class TestRoiSelection:

    def test_disabled_by_default(self, analyzer):
        assert analyzer.use_roi is False

    def test_enabled_roi_restricts_pixel_count(self, make_analyzer,
                                               default_config):
        """开 ROI 后参与统计的像素必须变少。

        这条不评判 ROI 判据好坏，只保证开关不是摆设 —— 默认关是有理由的
        （见 ColorAnalyzer._selection_mask 的说明）。
        """
        a = make_analyzer(default_config, use_roi=True)
        img = _board(COPPER_BGR, shape=(100, 100))
        img[:, :50] = (128, 128, 128)        # 一半是中性灰，应被 ROI 排除

        off = make_analyzer(default_config, use_roi=False).analyze(
            img, color_order="bgr"
        )
        on = a.analyze(img, color_order="bgr")

        assert on.pixel_count < off.pixel_count

    def test_roi_leaving_nothing_returns_unavailable(self, make_analyzer,
                                                     default_config):
        """ROI 把整幅都排掉时给出明确的未测结论，而不是除零崩掉。"""
        a = make_analyzer(default_config, use_roi=True)
        feats = a.analyze(_board((128, 128, 128), shape=(60, 60)),
                          color_order="bgr")

        assert feats.available is False
        assert "ROI" in feats.note
