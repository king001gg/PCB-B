"""core/quality.py 单元测试 —— 喷砂质检的 OK/NG 判定与评分。

为什么单独给这个模块写这么细的测试：
    这是全项目业务风险最高的模块。判定错了，要么把不良品放行（客户端
    失效），要么把良品报废（直接损失）——两种错法的代价都很高，而且
    都不像崩溃那样容易发现，只会在产线上安静地跑偏。

因此本文件重点压三类东西：
    (1) 五个 _score_* 的**阈值边界**：恰好等于阈值、阈值两侧各差一点、
        极端值（0 / 负数 / 极大值）。边界是这类分段函数最容易写错的地方，
        也是唯一能用单元测试钉死的部分。
    (2) OK/NG 结论**必须在阈值两侧相反**，且硬阈值（hard_limit）要能
        压过加权总分 —— 这条规则一旦失效就是放行不良品。
    (3) 评分**单调性**：指标变差时得分不允许变好。这条是防回归的底线，
        任何一次「顺手优化公式」都可能把它破坏掉。

测试全部确定性：热力图用带种子的随机数或常数填充构造，不读
data/samples，不联网，不依赖真实相机或 PLC。涉及项目自身
TextureAnalyzer 的用例属于跨模块口径检查（见 TestCvHeatmapContract）。

倾向说明：
    各断言写的是**当前实现的实际行为**，不是「我认为应该怎样」。
    凡是实际行为明显有业务风险的地方，用例标了 xfail 并在 reason 里
    写明缺陷，测试套件保持绿色；缺陷清单另行汇报，不在测试里顺手改代码。
"""

import json
from datetime import datetime

import numpy as np
import pytest

from core.defects import Defect
from core.quality import QualityAssessor, QualityReport
from core.texture import TextureAnalyzer


# ============================================================================
# 公共夹具与工具
# ============================================================================

HEAT_SHAPE = (100, 100)          # 热力图尺寸
HEAT_PIXELS = HEAT_SHAPE[0] * HEAT_SHAPE[1]  # = 10000，缺陷面积百分比的换算基数

# config/default.yaml 里 inspection.quality 的真实取值，供边界用例引用。
CV_MAX = 0.35
OXIDATION_MAX_PCT = 5.0
EMBEDDING_MAX_COUNT = 10
UNROUGHENED_MAX_PCT = 3.0
OK_SCORE_THRESHOLD = 60.0
OXIDATION_HARD_LIMIT = 15.0     # 不在 default.yaml 中，取代码默认值
UNROUGHENED_HARD_LIMIT = 10.0   # 同上


@pytest.fixture
def assessor(default_config):
    """用 config/default.yaml 的真实配置构造评估器。

    刻意读真配置而不是在测试里另写一份阈值 —— 阈值改了这里会第一时间炸。
    """
    return QualityAssessor(default_config)


@pytest.fixture
def quality_config_factory(config_factory):
    """在真实配置基础上覆盖 inspection.quality 下的若干键。"""

    def _make(**quality_overrides) -> dict:
        cfg = config_factory()
        cfg["inspection"]["quality"].update(quality_overrides)
        return cfg

    return _make


@pytest.fixture
def heatmap_factory():
    """CV 均匀性热力图工厂。

    ``mean`` 给定时返回常数热力图，此时 mean(heatmap) == mean，评分可直接手算；
    ``seed`` 给定时返回 [0, 2*mean) 的均匀随机热力图（带种子，可复现）。
    """

    def _make(mean: float = 0.1, shape=HEAT_SHAPE, seed: int = None):
        if seed is None:
            return np.full(shape, float(mean), dtype=np.float64)
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 2.0 * mean, shape).astype(np.float64)

    return _make


@pytest.fixture
def defect_factory():
    """Defect 工厂 —— 只关心类型和面积，掩膜给个形状正确的空图。"""

    def _make(defect_type: str, area_pixels: int = 0, shape=HEAT_SHAPE) -> Defect:
        return Defect(
            type=defect_type,
            mask=np.zeros(shape, dtype=np.uint8),
            area_pixels=area_pixels,
        )

    return _make


def board(area_px: int = 0, *, unrough_px: int = 0, embedding: int = 0,
          scratch: int = 0, defect_factory=None):
    """按像素面积拼一块板的缺陷列表。

    氧化斑 / 未粗化按面积累加（二者的百分比都由面积算出），
    磨料嵌入按「个数」计（评分只看计数）。
    """
    defects = []
    if area_px:
        defects.append(defect_factory("oxidation", area_px))
    if unrough_px:
        defects.append(defect_factory("unroughened", unrough_px))
    for _ in range(embedding):
        defects.append(defect_factory("embedding", 5))
    for _ in range(scratch):
        defects.append(defect_factory("scratch", 400))
    return defects


# ============================================================================
# (1) 粗糙度均匀性评分
# ============================================================================

class TestScoreRoughness:
    """_score_roughness —— CV 越低越好，阈值 cv_max = 0.35。"""

    def test_zero_cv_scores_full_marks(self, assessor):
        """CV = 0（完全均匀）应得 100 分。"""
        assert assessor._score_roughness(0.0) == 100.0

    def test_negative_cv_scores_full_marks(self, assessor):
        """负 CV 是无意义的输入，按「均匀」处理得满分（不报错、不扣分）。"""
        assert assessor._score_roughness(-1.0) == 100.0
        assert assessor._score_roughness(-1e9) == 100.0

    def test_half_threshold_is_linear(self, assessor):
        """阈值一半处的线性插值：0.175 → 80 分。"""
        assert assessor._score_roughness(CV_MAX / 2) == pytest.approx(80.0)

    def test_quarter_threshold_is_linear(self, assessor):
        """0.0875 → 90 分。"""
        assert assessor._score_roughness(CV_MAX / 4) == pytest.approx(90.0)

    def test_just_below_threshold_is_about_sixty(self, assessor):
        """阈值左极限应逼近 60 分（docstring 称 cv_max → 60 分）。"""
        assert assessor._score_roughness(0.3499) == pytest.approx(60.0, abs=0.05)

    def test_far_above_threshold_is_zero(self, assessor):
        """远超阈值应落到 0 分且不为负。"""
        assert assessor._score_roughness(1.0) == 0.0
        assert assessor._score_roughness(1e6) == 0.0

    @pytest.mark.parametrize("cv", [
        0.0, 0.01, 0.05, 0.1, 0.175, 0.34, 0.3499, 0.35, 0.3501, 0.5, 1.0, 100.0,
    ])
    def test_score_stays_within_range(self, assessor, cv):
        """任何输入下得分都必须落在 [0, 100]。"""
        assert 0.0 <= assessor._score_roughness(cv) <= 100.0

    def test_score_never_improves_as_cv_grows(self, assessor):
        """单调性：CV 变大（更不均匀）得分不得变好。"""
        cvs = [0.0, 0.05, 0.1, 0.175, 0.3, 0.34, 0.3499, 0.35, 0.4, 0.7, 1.2, 5.0]
        scores = [assessor._score_roughness(c) for c in cvs]
        assert scores == sorted(scores, reverse=True), f"评分非单调递减: {scores}"

    def test_growing_cv_never_keeps_score_equal(self, assessor):
        """指标确实变差时得分必须严格下降（单调性不能退化成常值）。"""
        assert assessor._score_roughness(0.1) > assessor._score_roughness(0.2)

    def test_config_threshold_shifts_the_curve(self, quality_config_factory):
        """roughness_cv_max 应可覆盖：放宽到 0.70 后同一个 CV 得分更高。"""
        strict = QualityAssessor(quality_config_factory(roughness_cv_max=0.35))
        loose = QualityAssessor(quality_config_factory(roughness_cv_max=0.70))
        assert loose.cv_max == 0.70
        assert loose._score_roughness(0.35) == pytest.approx(80.0)
        assert loose._score_roughness(0.35) > strict._score_roughness(0.35)

    @pytest.mark.xfail(
        reason="缺陷：_score_roughness 在 cv_max 处断层 —— 0.3499 得 60.01 分，"
               "恰好 0.35 却掉到 0 分（quality.py:290-292 的第二个分支用了 "
               "100*(1-cv/cv_max) 而非延续 100-(cv/cv_max)*40）。docstring "
               "（quality.py:285）明确写着「CV = cv_max → 60 分」。粗糙度权重 "
               "0.30，18 分的总分落差足以在阈值附近翻转 OK/NG。"
    )
    def test_score_is_continuous_at_threshold(self, assessor):
        """阈值两侧的得分应连续，恰好等于 cv_max 时应落在 60 分附近。"""
        assert assessor._score_roughness(CV_MAX) == pytest.approx(60.0, abs=1.0)


# ============================================================================
# (2) 方向一致性评分
# ============================================================================

class TestScoreDirection:
    """_score_direction —— DCI 越低（越各向同性）越好。"""

    def test_zero_dci_scores_full_marks(self, assessor):
        """DCI = 0（完全各向同性，理想喷砂）应得 100 分。"""
        assert assessor._score_direction(0.0) == 100.0

    def test_dci_one_scores_zero(self, assessor):
        """DCI = 1（最强方向性，最差）应得 0 分。"""
        assert assessor._score_direction(1.0) == 0.0

    def test_above_one_is_clamped_to_zero(self, assessor):
        """DCI 超出 [0,1] 上界时夹到 0，不出现负分。"""
        assert assessor._score_direction(1.0) == 0.0
        assert assessor._score_direction(5.0) == 0.0

    def test_half_dci_scores_fifty(self, assessor):
        """实际行为：DCI = 0.5 → 50 分。

        注意 docstring（quality.py:298）写的是「DCI=0.5 → 60 分」，与实现
        不符；这里记录实现值，供后续统一口径时定位。
        """
        assert assessor._score_direction(0.5) == 50.0

    def test_score_never_improves_as_dci_grows(self, assessor):
        """单调性：方向性越强（DCI 越大）得分不得变好。"""
        dcis = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.99, 1.0]
        scores = [assessor._score_direction(d) for d in dcis]
        assert scores == sorted(scores, reverse=True), f"评分非单调递减: {scores}"

    def test_score_stays_within_range_for_valid_dci(self, assessor):
        """契约内（DCI ∈ [0,1]）得分必须落在 [0, 100]。"""
        for dci in np.linspace(0.0, 1.0, 21):
            assert 0.0 <= assessor._score_direction(float(dci)) <= 100.0

    def test_negative_dci_exceeds_range(self, assessor, heatmap_factory):
        """契约外输入：DCI 为负时返回 200 分，可把总分推过 100。

        下界只在 max(0.0, ...) 处夹了一次，上界没夹。真实生产者的 DCI 经
        core/texture.py:611 夹到 [0,1]，所以这条只能由外部误传触发 ——
        记录行为，避免以后有人以为总分恒定 ≤ 100。
        """
        assert assessor._score_direction(-1.0) == 200.0

        report = assessor.assess([], heatmap_factory(mean=0.01), -1.0)
        assert report.overall_score > 100.0

    def test_min_threshold_config_does_not_change_scoring(self, quality_config_factory):
        """记录：direction_consistency_min 改动后评分完全不变（详见下方 xfail）。"""
        low = QualityAssessor(quality_config_factory(direction_consistency_min=0.01))
        high = QualityAssessor(quality_config_factory(direction_consistency_min=0.99))
        assert low._score_direction(0.9) == high._score_direction(0.9)

    @pytest.mark.xfail(
        reason="缺陷：配置项 direction_consistency_min 被读取却从不参与判定。"
               "quality.py:111-113 把它存进实例、quality.py:269 又导出到 "
               "detail['thresholds']['direction_min']，看起来是个生效的判据，"
               "但全模块没有任何地方引用它（_score_direction 只看 DCI 本身）。"
               "default.yaml:121 注释写明它是「锚纹方向一致性下限」，"
               "现场按此调参不会有任何效果。"
    )
    def test_min_threshold_affects_verdict(self, quality_config_factory, heatmap_factory):
        """把方向一致性下限从 0.01 改到 0.99，判定结果应当随之改变。

        刻意取 DCI=0.3（方向分 70，本身不触发「评分偏低」预警），把配置项
        的影响从通用的低分预警里隔离出来。
        """
        heatmap = heatmap_factory(mean=0.1)
        loose = QualityAssessor(quality_config_factory(direction_consistency_min=0.01))
        strict = QualityAssessor(quality_config_factory(direction_consistency_min=0.99))

        loose_report = loose.assess([], heatmap, 0.3)
        strict_report = strict.assess([], heatmap, 0.3)

        assert (strict_report.ok_ng, strict_report.warnings) != \
            (loose_report.ok_ng, loose_report.warnings)


# ============================================================================
# (3) 氧化斑评分
# ============================================================================

class TestScoreOxidation:
    """_score_oxidation —— 面积比越低越好，软阈值 oxidation_max_pct = 5.0。"""

    def test_zero_pct_scores_full_marks(self, assessor):
        """无氧化斑得 100 分。"""
        assert assessor._score_oxidation(0.0) == 100.0

    def test_negative_pct_scores_full_marks(self, assessor):
        """负面积比按 0 处理，不报错也不扣分。"""
        assert assessor._score_oxidation(-5.0) == 100.0

    def test_half_threshold_is_linear(self, assessor):
        """阈值一半处：2.5% → 80 分。"""
        assert assessor._score_oxidation(2.5) == pytest.approx(80.0)

    def test_just_below_threshold_is_about_sixty(self, assessor):
        """阈值左极限应逼近 60 分（docstring 称达到 max_pct → 60 分）。

        实测 4.99% → 60.08 分：线性段在阈值处落脚于 60，与 docstring 一致，
        与阈值右侧的 50 分之间才是断层（见下方 xfail）。
        """
        assert assessor._score_oxidation(4.99) == pytest.approx(60.0, abs=0.15)

    def test_ten_pct_scores_zero(self, assessor):
        """10% 时线性段已把分扣光。"""
        assert assessor._score_oxidation(10.0) == 0.0

    def test_huge_pct_is_clamped_to_zero(self, assessor):
        """极端面积比不得产生负分。"""
        assert assessor._score_oxidation(1e6) == 0.0

    @pytest.mark.parametrize("pct", [0.0, 0.05, 0.1, 0.11, 1.0, 2.5, 4.99, 5.0,
                                     5.01, 10.0, 50.0])
    def test_score_stays_within_range(self, assessor, pct):
        """任何输入下得分必须落在 [0, 100]。"""
        assert 0.0 <= assessor._score_oxidation(pct) <= 100.0

    def test_score_never_improves_as_pct_grows(self, assessor):
        """单调性：氧化面积越大得分不得变好。"""
        pcts = [0.0, 0.1, 0.5, 2.5, 4.99, 5.0, 6.0, 8.0, 10.0, 20.0]
        scores = [assessor._score_oxidation(p) for p in pcts]
        assert scores == sorted(scores, reverse=True), f"评分非单调递减: {scores}"

    def test_config_threshold_shifts_the_curve(self, quality_config_factory):
        """oxidation_max_pct 应可覆盖：放宽到 10 后同一个面积比得分更高。"""
        strict = QualityAssessor(quality_config_factory(oxidation_max_pct=5.0))
        loose = QualityAssessor(quality_config_factory(oxidation_max_pct=10.0))
        assert loose._score_oxidation(5.0) == pytest.approx(80.0)
        assert loose._score_oxidation(5.0) > strict._score_oxidation(5.0)

    @pytest.mark.xfail(
        reason="缺陷：_score_oxidation 在 oxidation_max_pct 处断层 —— 4.99% 得 "
               "60.08 分，恰好 5.0% 掉到 50 分（quality.py:309-311 的第二分支 "
               "用 100 - pct*10，斜率 10 与线性段的 40/5=8 不一致；同一处的 "
               "min(...,60) 上夹在 pct>=5 时永远不生效）。docstring"
               "（quality.py:306）称「达到 max_pct → 60 分」。"
    )
    def test_score_is_continuous_at_threshold(self, assessor):
        """阈值两侧的得分应连续，恰好等于 max_pct 时应落在 60 分附近。"""
        assert assessor._score_oxidation(OXIDATION_MAX_PCT) == pytest.approx(60.0, abs=1.0)


# ============================================================================
# (4) 磨料嵌入评分
# ============================================================================

class TestScoreEmbedding:
    """_score_embedding —— 计数越少越好，软阈值 embedding_max_count = 10。"""

    def test_zero_and_one_score_full_marks(self, assessor):
        """0 个或 1 个嵌入颗粒得满分（1 个在容差内）。"""
        assert assessor._score_embedding(0) == 100.0
        assert assessor._score_embedding(1) == 100.0

    def test_negative_count_scores_full_marks(self, assessor):
        """负计数按 0 处理。"""
        assert assessor._score_embedding(-3) == 100.0

    def test_half_threshold_is_linear(self, assessor):
        """阈值一半处：5 个 → 80 分。"""
        assert assessor._score_embedding(5) == pytest.approx(80.0)

    def test_just_below_threshold_is_about_sixty(self, assessor):
        """9 个 → 64 分，与阈值处衔接。"""
        assert assessor._score_embedding(9) == pytest.approx(64.0)

    def test_at_threshold_scores_sixty(self, assessor):
        """恰好达到 max_count → 60 分（该维度在阈值处是连续的）。"""
        assert assessor._score_embedding(EMBEDDING_MAX_COUNT) == 60.0

    def test_plateau_above_threshold_holds_sixty(self, assessor):
        """10~20 个仍为 60 分 —— 第二分支的 60 分上夹在此区间生效。"""
        assert assessor._score_embedding(20) == 60.0

    def test_far_above_threshold_is_zero(self, assessor):
        """50 个已把分扣光，且不为负。"""
        assert assessor._score_embedding(50) == 0.0
        assert assessor._score_embedding(10 ** 6) == 0.0

    @pytest.mark.parametrize("count", [0, 1, 2, 5, 9, 10, 11, 20, 21, 50, 1000])
    def test_score_stays_within_range(self, assessor, count):
        """任何输入下得分必须落在 [0, 100]。"""
        assert 0.0 <= assessor._score_embedding(count) <= 100.0

    def test_score_never_improves_as_count_grows(self, assessor):
        """单调性：嵌入颗粒越多得分不得变好。"""
        counts = [0, 1, 2, 5, 9, 10, 15, 20, 21, 30, 50, 100]
        scores = [assessor._score_embedding(c) for c in counts]
        assert scores == sorted(scores, reverse=True), f"评分非单调递减: {scores}"

    def test_config_threshold_shifts_the_curve(self, quality_config_factory):
        """embedding_max_count 应可覆盖：放宽到 20 后同样的计数得分更高。"""
        strict = QualityAssessor(quality_config_factory(embedding_max_count=10))
        loose = QualityAssessor(quality_config_factory(embedding_max_count=20))
        assert loose._score_embedding(10) == pytest.approx(80.0)
        assert loose._score_embedding(10) > strict._score_embedding(10)


# ============================================================================
# (5) 未粗化面积评分
# ============================================================================

class TestScoreUnroughened:
    """_score_unroughened —— 面积比越低越好，软阈值 unroughened_max_pct = 3.0。"""

    def test_zero_pct_scores_full_marks(self, assessor):
        """无未粗化区域得 100 分。"""
        assert assessor._score_unroughened(0.0) == 100.0

    def test_negative_pct_scores_full_marks(self, assessor):
        """负面积比按 0 处理。"""
        assert assessor._score_unroughened(-2.0) == 100.0

    def test_half_threshold_is_linear(self, assessor):
        """阈值一半处：1.5% → 80 分。"""
        assert assessor._score_unroughened(1.5) == pytest.approx(80.0)

    def test_just_below_threshold_is_about_sixty(self, assessor):
        """阈值左极限应逼近 60 分（实测 2.99% → 60.13 分）。"""
        assert assessor._score_unroughened(2.99) == pytest.approx(60.0, abs=0.15)

    def test_at_threshold_scores_sixty(self, assessor):
        """恰好达到 max_pct → 60 分（该维度在阈值处连续）。"""
        assert assessor._score_unroughened(UNROUGHENED_MAX_PCT) == 60.0

    def test_ten_pct_scores_zero(self, assessor):
        """10% 时已扣光（正好是硬阈值的位置）。"""
        assert assessor._score_unroughened(10.0) == 0.0
        assert assessor._score_unroughened(1e6) == 0.0

    @pytest.mark.parametrize("pct", [0.0, 0.1, 0.5, 1.5, 2.99, 3.0, 3.01, 6.0, 10.0])
    def test_score_stays_within_range(self, assessor, pct):
        """任何输入下得分必须落在 [0, 100]。"""
        assert 0.0 <= assessor._score_unroughened(pct) <= 100.0

    def test_score_never_improves_as_pct_grows(self, assessor):
        """单调性：未粗化面积越大得分不得变好。"""
        pcts = [0.0, 0.1, 1.0, 2.99, 3.0, 4.0, 6.0, 10.0, 30.0]
        scores = [assessor._score_unroughened(p) for p in pcts]
        assert scores == sorted(scores, reverse=True), f"评分非单调递减: {scores}"

    def test_config_threshold_shifts_the_curve(self, quality_config_factory):
        """unroughened_max_pct 应可覆盖：放宽到 6 后同一面积比得分更高。"""
        strict = QualityAssessor(quality_config_factory(unroughened_max_pct=3.0))
        loose = QualityAssessor(quality_config_factory(unroughened_max_pct=6.0))
        assert loose._score_unroughened(3.0) == pytest.approx(80.0)
        assert loose._score_unroughened(3.0) > strict._score_unroughened(3.0)


# ============================================================================
# assess() 返回的 QualityReport 各字段
# ============================================================================

class TestAssessReportFields:
    """assess() 的入参 → 报告字段的映射。"""

    def test_defaults_when_nothing_given(self, assessor):
        """空缺陷 + 理想 DCI 时应返回一份满分报告，而不是报错。

        这里显式传 dci=0.0；连 dci 一起省掉会掉到 85 分（方向维度按最差
        计），那是另一个缺陷，见 TestCvHeatmapContract 的两个用例。
        """
        report = assessor.assess([], None, 0.0)
        assert isinstance(report, QualityReport)
        assert report.overall_score == 100.0
        assert report.ok_ng is True
        assert report.roughness_uniformity == 100.0
        assert report.roughness_std == 0.0
        assert report.oxidation_percentage == 0.0
        assert report.unroughened_percentage == 0.0
        assert report.embedding_count == 0
        assert report.warnings == []

    def test_roughness_std_is_heatmap_mean(self, assessor, heatmap_factory):
        """roughness_std 取热力图均值（热力图存的就是 CV 值）。"""
        report = assessor.assess([], heatmap_factory(mean=0.1), 0.0)
        assert report.roughness_std == pytest.approx(0.1)
        assert report.roughness_uniformity == assessor._score_roughness(0.1)

    def test_roughness_std_follows_seeded_heatmap(self, assessor, heatmap_factory):
        """带种子的随机热力图：均值必须与 numpy 现算的一致，且可复现。"""
        hm = heatmap_factory(mean=0.1, seed=7)
        assert hm.shape == HEAT_SHAPE

        first = assessor.assess([], hm, 0.0)
        second = assessor.assess([], heatmap_factory(mean=0.1, seed=7), 0.0)
        assert first.roughness_std == pytest.approx(float(np.mean(hm)))
        assert first.overall_score == second.overall_score

    def test_defect_percentages_are_area_ratios(self, assessor, heatmap_factory, defect_factory):
        """氧化斑 / 未粗化面积比 = 缺陷面积 / 热力图像素数 * 100。"""
        defects = board(450, unrough_px=250, defect_factory=defect_factory)
        report = assessor.assess(defects, heatmap_factory(mean=0.1), 0.0)
        assert report.oxidation_percentage == pytest.approx(4.5)
        assert report.unroughened_percentage == pytest.approx(2.5)

    def test_same_type_defects_are_summed(self, assessor, heatmap_factory, defect_factory):
        """同类型的多个缺陷应累加面积，而不是只取第一个。"""
        defects = [defect_factory("oxidation", 300), defect_factory("oxidation", 150)]
        report = assessor.assess(defects, heatmap_factory(mean=0.1), 0.0)
        assert report.oxidation_percentage == pytest.approx(4.5)

    def test_embedding_is_counted_not_summed(self, assessor, heatmap_factory, defect_factory):
        """磨料嵌入按「个数」计，与各缺陷面积无关。"""
        defects = board(0, embedding=6, defect_factory=defect_factory)
        report = assessor.assess(defects, heatmap_factory(mean=0.1), 0.0)
        assert report.embedding_count == 6
        assert report.oxidation_percentage == 0.0

    def test_scratch_defects_do_not_affect_score(self, assessor, heatmap_factory, defect_factory):
        """划痕不参与五维评分 —— 记录该行为（评分体系里没有划痕维度）。"""
        clean = assessor.assess([], heatmap_factory(mean=0.1), 0.0)
        scratched = assessor.assess(
            board(0, scratch=10, defect_factory=defect_factory),
            heatmap_factory(mean=0.1), 0.0,
        )
        assert scratched.overall_score == clean.overall_score
        assert scratched.ok_ng == clean.ok_ng

    def test_direction_consistency_is_passed_through(self, assessor, heatmap_factory):
        """传入的 DCI 原样进入报告。"""
        report = assessor.assess([], heatmap_factory(mean=0.1), 0.42)
        assert report.direction_consistency == pytest.approx(0.42)

    def test_board_id_is_echoed(self, assessor, heatmap_factory):
        """板号原样回填到报告。"""
        report = assessor.assess([], heatmap_factory(mean=0.1), 0.0, board_id="PCB-0007")
        assert report.board_id == "PCB-0007"

    def test_timestamp_is_iso8601(self, assessor, heatmap_factory):
        """时间戳必须是可解析的 ISO 8601 字符串（要入库）。"""
        report = assessor.assess([], heatmap_factory(mean=0.1), 0.0)
        assert datetime.fromisoformat(report.timestamp)

    def test_detail_scores_cover_every_weighted_dimension(self, assessor, heatmap_factory):
        """detail['scores'] 的键必须与权重键一一对应，不能漏项。"""
        report = assessor.assess([], heatmap_factory(mean=0.1), 0.0)
        assert set(report.detail["scores"]) == set(assessor.weights)

    def test_detail_thresholds_echo_config(self, assessor, default_config):
        """detail['thresholds'] 应回显真实配置里的阈值。"""
        report = assessor.assess([], np.full(HEAT_SHAPE, 0.1), 0.0)
        thresholds = report.detail["thresholds"]
        quality_cfg = default_config["inspection"]["quality"]
        assert thresholds["roughness_cv_max"] == quality_cfg["roughness_cv_max"]
        assert thresholds["oxidation_max_pct"] == quality_cfg["oxidation_max_pct"]
        assert thresholds["embedding_max_cnt"] == quality_cfg["embedding_max_count"]
        assert thresholds["unroughened_max_pct"] == quality_cfg["unroughened_max_pct"]
        assert thresholds["direction_min"] == quality_cfg["direction_consistency_min"]

    def test_low_dimension_score_raises_warning(self, assessor, heatmap_factory, defect_factory):
        """任一维度低于 60 分都应留下预警（现场据此定位工艺问题）。"""
        defects = board(0, embedding=30, defect_factory=defect_factory)
        report = assessor.assess(defects, heatmap_factory(mean=0.1), 0.0)
        assert any("embedding" in w for w in report.warnings)

    def test_healthy_board_has_no_warning(self, assessor, heatmap_factory):
        """各维度都在阈值内时不应有预警。"""
        report = assessor.assess([], heatmap_factory(mean=0.05), 0.0)
        assert report.warnings == []


# ============================================================================
# OK/NG 判定
# ============================================================================

class TestOkNgDecision:
    """OK/NG 判定规则：硬阈值优先，其余看加权总分与 ok_score_threshold。"""

    # 与阈值比邻的一组输入：热力图均值 0.3、DCI 0.3、未粗化 250px、5 个嵌入，
    # 只让氧化斑面积在阈值两侧微调。690px(6.90%) → 60.0 分，695px(6.95%) → 59.8 分。
    @staticmethod
    def _straddle(oxidation_px, heatmap_factory, defect_factory):
        return board(oxidation_px, unrough_px=250, embedding=5,
                     defect_factory=defect_factory)

    def test_board_just_above_threshold_is_ok(self, assessor, heatmap_factory, defect_factory):
        """总分 60.0（恰好等于阈值）应判 OK —— 比较是「>=」而非「>」。"""
        report = assessor.assess(
            self._straddle(690, heatmap_factory, defect_factory),
            heatmap_factory(mean=0.3), 0.3,
        )
        assert report.overall_score == pytest.approx(60.0, abs=0.05)
        assert report.overall_score >= OK_SCORE_THRESHOLD
        assert report.ok_ng is True

    def test_board_just_below_threshold_is_ng(self, assessor, heatmap_factory, defect_factory):
        """总分 59.8（差 0.2 分）应判 NG —— 阈值两侧结论必须相反。"""
        report = assessor.assess(
            self._straddle(695, heatmap_factory, defect_factory),
            heatmap_factory(mean=0.3), 0.3,
        )
        assert report.overall_score == pytest.approx(59.8, abs=0.05)
        assert report.overall_score < OK_SCORE_THRESHOLD
        assert report.ok_ng is False

    def test_boundary_pair_differs_by_less_than_one_point(self, assessor, heatmap_factory,
                                                          defect_factory):
        """上面两块板的总分应几乎相同，证明判定的翻转确实是阈值造成的。"""
        ok = assessor.assess(self._straddle(690, heatmap_factory, defect_factory),
                             heatmap_factory(mean=0.3), 0.3)
        ng = assessor.assess(self._straddle(695, heatmap_factory, defect_factory),
                             heatmap_factory(mean=0.3), 0.3)
        assert ok.overall_score - ng.overall_score < 1.0
        assert ok.ok_ng != ng.ok_ng

    def test_threshold_comparison_is_inclusive(self, assessor, quality_config_factory,
                                               heatmap_factory, defect_factory):
        """把阈值设成正好等于某块板的总分 → OK；再抬高 0.1 分 → NG。"""
        defects = board(0, unrough_px=250, embedding=5, defect_factory=defect_factory)
        heatmap = heatmap_factory(mean=0.3)

        score = assessor.assess(defects, heatmap, 0.3).overall_score

        at = QualityAssessor(quality_config_factory(ok_score_threshold=score))
        assert at.assess(defects, heatmap, 0.3).ok_ng is True

        above = QualityAssessor(quality_config_factory(ok_score_threshold=score + 0.1))
        assert above.assess(defects, heatmap, 0.3).ok_ng is False

    def test_oxidation_hard_limit_is_strict(self, assessor, heatmap_factory, defect_factory):
        """氧化斑恰好 15% 不算超标（判据是「>」），15.01% 才直接 NG。"""
        heatmap = heatmap_factory(mean=0.1)

        at_limit = assessor.assess(
            board(1500, defect_factory=defect_factory), heatmap, 0.0
        )
        assert at_limit.oxidation_percentage == pytest.approx(OXIDATION_HARD_LIMIT)
        assert at_limit.ok_ng is True

        over = assessor.assess(
            board(1501, defect_factory=defect_factory), heatmap, 0.0
        )
        assert over.oxidation_percentage > OXIDATION_HARD_LIMIT
        assert over.ok_ng is False

    def test_unroughened_hard_limit_is_strict(self, assessor, heatmap_factory, defect_factory):
        """未粗化恰好 10% 不算超标，10.01% 直接 NG。"""
        heatmap = heatmap_factory(mean=0.1)

        at_limit = assessor.assess(
            board(0, unrough_px=1000, defect_factory=defect_factory), heatmap, 0.0
        )
        assert at_limit.unroughened_percentage == pytest.approx(UNROUGHENED_HARD_LIMIT)
        assert at_limit.ok_ng is True

        over = assessor.assess(
            board(0, unrough_px=1001, defect_factory=defect_factory), heatmap, 0.0
        )
        assert over.unroughened_percentage > UNROUGHENED_HARD_LIMIT
        assert over.ok_ng is False

    def test_hard_limit_overrides_high_score(self, assessor, heatmap_factory, defect_factory):
        """其余维度全满分、总分 75 分，氧化斑 20% 仍必须判 NG。

        这条是「放行不良品」的最后一道闸：硬阈值不能被高分盖过去。
        """
        report = assessor.assess(
            board(2000, defect_factory=defect_factory),
            heatmap_factory(mean=0.01), 0.0,
        )
        assert report.overall_score > OK_SCORE_THRESHOLD
        assert report.ok_ng is False

    def test_hard_limit_warning_mentions_value_and_limit(self, assessor, heatmap_factory,
                                                         defect_factory):
        """硬阈值预警要带上实测值与限制值，现场才能判断严重程度。"""
        report = assessor.assess(
            board(2000, defect_factory=defect_factory),
            heatmap_factory(mean=0.1), 0.0,
        )
        assert any("20.0%" in w and "15.0%" in w for w in report.warnings)

    def test_custom_hard_limit_is_respected(self, quality_config_factory, heatmap_factory,
                                            defect_factory):
        """hard_limit 未写在 default.yaml 里，但确实可配置：收紧到 5% 后 6% 直接 NG。"""
        cfg = quality_config_factory(oxidation_hard_limit=5.0)
        assessor = QualityAssessor(cfg)
        report = assessor.assess(
            board(600, defect_factory=defect_factory),
            heatmap_factory(mean=0.1), 0.0,
        )
        assert report.oxidation_percentage == pytest.approx(6.0)
        assert report.ok_ng is False

    def test_ng_when_score_below_threshold(self, assessor, heatmap_factory):
        """无硬阈值超标时，总分低于阈值即 NG（例如 CV 极大 + 方向性极强）。"""
        report = assessor.assess([], heatmap_factory(mean=1.0), 1.0)
        assert report.overall_score < OK_SCORE_THRESHOLD
        assert report.ok_ng is False


# ============================================================================
# 权重与总分
# ============================================================================

class TestWeightsAndOverallScore:
    """加权求和的算术性质。"""

    def test_weights_sum_to_one(self, assessor):
        """权重和必须为 1，否则总分不再落在 0–100。"""
        assert sum(assessor.weights.values()) == pytest.approx(1.0)

    def test_weights_are_hardcoded(self, quality_config_factory):
        """记录：权重在 quality.py:128-134 硬编码，配置里给同名键也不会被读取。"""
        cfg = quality_config_factory(weights={"roughness": 0.9, "direction": 0.1})
        assessor = QualityAssessor(cfg)
        assert assessor.weights == {
            "roughness": 0.30, "direction": 0.15, "oxidation": 0.25,
            "embedding": 0.15, "unroughened": 0.15,
        }

    def test_all_full_marks_gives_one_hundred(self, assessor, heatmap_factory):
        """五个维度全满分 → 总分 100。"""
        report = assessor.assess([], heatmap_factory(mean=0.005), 0.0)
        assert report.detail["scores"] == pytest.approx({k: 100.0 for k in assessor.weights})
        assert report.overall_score == 100.0
        assert report.ok_ng is True

    def test_all_zero_marks_gives_zero(self, assessor, heatmap_factory, defect_factory):
        """五个维度全 0 分 → 总分 0 且判 NG。

        刻意把未粗化压在 10.0% —— 恰好等于硬阈值，不触发硬阈值分支，
        这样测的就是纯加权总分那条路径。
        """
        defects = board(1000, unrough_px=1000, embedding=50, defect_factory=defect_factory)
        report = assessor.assess(defects, heatmap_factory(mean=5.0), 1.0)
        assert report.detail["scores"] == pytest.approx({k: 0.0 for k in assessor.weights})
        assert report.overall_score == 0.0
        assert report.ok_ng is False

    def test_overall_equals_weighted_sum_of_dimension_scores(self, assessor, heatmap_factory,
                                                            defect_factory):
        """总分必须逐项等于 分项分 × 权重 之和（四舍五入到 1 位）。"""
        defects = board(320, unrough_px=180, embedding=4, defect_factory=defect_factory)
        report = assessor.assess(defects, heatmap_factory(mean=0.18), 0.25)

        scores, weights = report.detail["scores"], report.detail["weights"]
        expected = sum(scores[k] * weights[k] for k in weights)
        assert report.overall_score == pytest.approx(round(expected, 1))

    def test_overall_score_is_rounded_to_one_decimal(self, assessor, heatmap_factory):
        """总分保留 1 位小数（报表口径）。"""
        report = assessor.assess([], heatmap_factory(mean=0.13), 0.37)
        assert report.overall_score == round(report.overall_score, 1)


# ============================================================================
# batch_assess
# ============================================================================

class TestBatchAssess:
    """批量评估。"""

    def test_empty_list_returns_empty(self, assessor):
        """空列表 → 空结果，不报错。"""
        assert assessor.batch_assess([]) == []

    def test_single_item(self, assessor, heatmap_factory, defect_factory):
        """单元素批次的报告应与单次 assess 完全一致。"""
        defects = board(300, defect_factory=defect_factory)
        heatmap = heatmap_factory(mean=0.2)

        single = assessor.assess(defects, heatmap, 0.2, "B1")
        batch = assessor.batch_assess([(defects, heatmap, 0.2, "B1")])

        assert len(batch) == 1
        assert batch[0].overall_score == single.overall_score
        assert batch[0].ok_ng == single.ok_ng
        assert batch[0].board_id == "B1"

    def test_each_report_keeps_its_own_board_id(self, assessor, heatmap_factory):
        """板号不能串台 —— 报告要按板入库，串了就追不回是哪块板。"""
        heatmap = heatmap_factory(mean=0.1)
        reports = assessor.batch_assess([
            ([], heatmap, 0.0, "A-1"),
            ([], heatmap, 0.0, "A-2"),
            ([], heatmap, 0.0, "A-3"),
        ])
        assert [r.board_id for r in reports] == ["A-1", "A-2", "A-3"]

    def test_returns_independent_report_objects(self, assessor, heatmap_factory):
        """每个元素必须是独立对象，改一份不能影响另一份。"""
        heatmap = heatmap_factory(mean=0.1)
        reports = assessor.batch_assess([([], heatmap, 0.0, "A"), ([], heatmap, 0.0, "B")])
        assert reports[0] is not reports[1]
        reports[0].warnings.append("人为污染")
        assert reports[1].warnings == []

    def test_mixed_ok_and_ng(self, assessor, heatmap_factory, defect_factory):
        """混合批次里 OK/NG 不能被平均掉，必须逐板独立判定。"""
        heatmap = heatmap_factory(mean=0.1)
        good = board(0, embedding=1, defect_factory=defect_factory)
        bad = board(2000, defect_factory=defect_factory)   # 20% 氧化，硬阈值

        reports = assessor.batch_assess([
            (good, heatmap, 0.0, "OK-1"),
            (bad, heatmap, 0.0, "NG-1"),
            (good, heatmap, 0.0, "OK-2"),
        ])
        assert [r.ok_ng for r in reports] == [True, False, True]

    def test_order_is_preserved(self, assessor, heatmap_factory):
        """批次顺序即产出顺序（与产线节拍对应）。"""
        heatmap = heatmap_factory(mean=0.1)
        reports = assessor.batch_assess(
            [([], heatmap, 0.0, str(i)) for i in range(5)]
        )
        assert [r.board_id for r in reports] == ["0", "1", "2", "3", "4"]

    def test_wrong_tuple_arity_raises(self, assessor, heatmap_factory):
        """元组字段数不对时应立刻报错，而不是静默吞掉这块板。"""
        with pytest.raises(ValueError):
            assessor.batch_assess([([], heatmap_factory(mean=0.1), 0.0)])


# ============================================================================
# yield_stats
# ============================================================================

class TestYieldStats:
    """批次汇总统计。"""

    def test_empty_list_returns_empty_dict(self, assessor):
        """空批次 → 空字典（调用方据此跳过汇总，不能除零）。"""
        assert assessor.yield_stats([]) == {}

    def test_single_ok_report(self, assessor, heatmap_factory):
        """单块 OK 板：合格率 100%，NG 计数 0。"""
        reports = assessor.batch_assess([([], heatmap_factory(mean=0.05), 0.0, "B1")])
        stats = assessor.yield_stats(reports)
        assert stats["total"] == 1
        assert stats["ok_count"] == 1
        assert stats["ng_count"] == 0
        assert stats["yield_rate"] == 100.0

    def test_single_ng_report(self, assessor, heatmap_factory, defect_factory):
        """单块 NG 板：合格率 0%。"""
        reports = assessor.batch_assess([
            (board(2000, defect_factory=defect_factory), heatmap_factory(mean=0.1), 0.0, "B1")
        ])
        stats = assessor.yield_stats(reports)
        assert stats["total"] == 1
        assert stats["ok_count"] == 0
        assert stats["yield_rate"] == 0.0

    def test_all_ok(self, assessor, heatmap_factory):
        """全 OK：合格率 100%。"""
        heatmap = heatmap_factory(mean=0.05)
        reports = assessor.batch_assess([([], heatmap, 0.0, f"B{i}") for i in range(4)])
        stats = assessor.yield_stats(reports)
        assert (stats["total"], stats["ok_count"], stats["ng_count"]) == (4, 4, 0)
        assert stats["yield_rate"] == 100.0

    def test_all_ng(self, assessor, heatmap_factory, defect_factory):
        """全 NG：合格率 0%，且不应出现除零或 NaN。"""
        heatmap = heatmap_factory(mean=0.1)
        reports = assessor.batch_assess([
            (board(2000, defect_factory=defect_factory), heatmap, 0.0, f"B{i}")
            for i in range(3)
        ])
        stats = assessor.yield_stats(reports)
        assert (stats["total"], stats["ok_count"], stats["ng_count"]) == (3, 0, 3)
        assert stats["yield_rate"] == 0.0

    def test_mixed_yield_rate_is_rounded(self, assessor, heatmap_factory, defect_factory):
        """2/3 合格 → 66.67%，四舍五入到 2 位。"""
        heatmap = heatmap_factory(mean=0.1)
        good = board(0, embedding=1, defect_factory=defect_factory)
        bad = board(2000, defect_factory=defect_factory)

        reports = assessor.batch_assess([
            (good, heatmap, 0.0, "A"), (good, heatmap, 0.0, "B"), (bad, heatmap, 0.0, "C"),
        ])
        stats = assessor.yield_stats(reports)
        assert stats["yield_rate"] == pytest.approx(66.67)
        assert stats["ok_count"] + stats["ng_count"] == stats["total"]

    def test_avg_and_min_score(self, assessor, heatmap_factory, defect_factory):
        """平均分与最低分应等于各报告总分的对应统计量。"""
        heatmap = heatmap_factory(mean=0.1)
        reports = assessor.batch_assess([
            ([], heatmap, 0.0, "A"),
            (board(1500, defect_factory=defect_factory), heatmap, 0.0, "B"),
        ])
        scores = [r.overall_score for r in reports]
        stats = assessor.yield_stats(reports)
        assert stats["avg_score"] == pytest.approx(round(float(np.mean(scores)), 2))
        assert stats["min_score"] == pytest.approx(round(min(scores), 2))

    def test_average_dimension_metrics(self, assessor, heatmap_factory, defect_factory):
        """各维度的批次均值口径：面积比 2 位、嵌入计数 1 位。"""
        heatmap = heatmap_factory(mean=0.1)
        reports = assessor.batch_assess([
            (board(300, embedding=2, defect_factory=defect_factory), heatmap, 0.0, "A"),
            (board(500, embedding=4, defect_factory=defect_factory), heatmap, 0.0, "B"),
        ])
        stats = assessor.yield_stats(reports)
        assert stats["avg_oxidation_pct"] == pytest.approx(4.0)
        assert stats["avg_embedding_count"] == pytest.approx(3.0)
        assert stats["avg_unroughened_pct"] == pytest.approx(0.0)

    def test_top_warnings_most_common_first(self, assessor):
        """高频预警排前面，且最多 5 条。"""
        reports = [
            QualityReport(ok_ng=True, warnings=["B", "A"]),
            QualityReport(ok_ng=False, warnings=["B"]),
            QualityReport(ok_ng=True, warnings=["B", "C"]),
        ]
        stats = assessor.yield_stats(reports)
        assert stats["top_warnings"][0] == "B"
        assert len(stats["top_warnings"]) <= 5

    def test_top_warnings_deduplicated(self, assessor):
        """同一条预警在多块板上重复时只出现一次。"""
        reports = [QualityReport(warnings=["同一句"]), QualityReport(warnings=["同一句"])]
        assert assessor.yield_stats(reports)["top_warnings"] == ["同一句"]

    def test_expected_keys_present(self, assessor, heatmap_factory):
        """汇总字典的键必须齐全（报表/数据库按名取用）。"""
        reports = assessor.batch_assess([([], heatmap_factory(mean=0.1), 0.0, "A")])
        stats = assessor.yield_stats(reports)
        for key in ("total", "ok_count", "ng_count", "yield_rate", "avg_score",
                    "min_score", "avg_oxidation_pct", "avg_embedding_count",
                    "avg_unroughened_pct", "top_warnings"):
            assert key in stats


# ============================================================================
# 序列化与摘要
# ============================================================================

class TestSerialization:
    """to_dict / summary —— 入库与报表出口。"""

    def test_to_dict_has_all_documented_fields(self, assessor, heatmap_factory):
        """字段集合必须稳定，少一个字段下游就取不到值。"""
        report = assessor.assess([], heatmap_factory(mean=0.1), 0.2, board_id="B7")
        assert set(report.to_dict()) == {
            "overall_score", "ok_ng", "roughness_uniformity", "roughness_std",
            "direction_consistency", "oxidation_percentage", "embedding_count",
            "unroughened_percentage", "warnings", "timestamp", "board_id",
        }

    def test_to_dict_values_match_report(self, assessor, heatmap_factory, defect_factory):
        """字典值必须与对象字段一致，且板号/时间戳不得丢失。"""
        report = assessor.assess(
            board(450, unrough_px=250, embedding=3, defect_factory=defect_factory),
            heatmap_factory(mean=0.1), 0.25, board_id="B7",
        )
        data = report.to_dict()
        assert data["overall_score"] == report.overall_score
        assert data["ok_ng"] == report.ok_ng
        assert data["oxidation_percentage"] == pytest.approx(4.5)
        assert data["unroughened_percentage"] == pytest.approx(2.5)
        assert data["embedding_count"] == 3
        assert data["board_id"] == "B7"
        assert data["timestamp"] == report.timestamp

    def test_to_dict_rounds_floats(self, assessor, heatmap_factory, defect_factory):
        """方向 4 位、百分比 2 位 —— 入库口径固定，避免浮点尾巴。"""
        report = assessor.assess(
            board(453, defect_factory=defect_factory),
            heatmap_factory(mean=0.1), 1.0 / 3.0,
        )
        data = report.to_dict()
        assert data["direction_consistency"] == round(1.0 / 3.0, 4)
        assert data["oxidation_percentage"] == round(report.oxidation_percentage, 2)
        assert data["unroughened_percentage"] == round(report.unroughened_percentage, 2)

    def test_warnings_list_is_passed_through(self, assessor, heatmap_factory, defect_factory):
        """预警文本必须原样进入字典。"""
        report = assessor.assess(
            board(2000, defect_factory=defect_factory), heatmap_factory(mean=0.1), 0.0
        )
        assert report.warnings
        assert report.to_dict()["warnings"] == report.warnings

    def test_json_dumps_succeeds_for_plain_python_inputs(self, assessor, heatmap_factory,
                                                         defect_factory):
        """入参全是 Python 标量时，to_dict 必须能直接 json.dumps。"""
        report = assessor.assess(
            board(450, embedding=3, defect_factory=defect_factory),
            heatmap_factory(mean=0.1), 0.3, board_id="B7",
        )
        assert isinstance(report.ok_ng, bool), f"ok_ng 类型异常: {type(report.ok_ng)}"
        encoded = json.dumps(report.to_dict(), ensure_ascii=False)
        assert json.loads(encoded)["board_id"] == "B7"

    @pytest.mark.xfail(
        reason="缺陷：DCI 由生产路径传入时，to_dict() 里 ok_ng 是 numpy.bool_ 而非 "
               "Python bool，json.dumps 直接抛 TypeError。触发链："
               "core/texture.py:611 的 direction_consistency 返回 np.float64 → "
               "core/quality.py:247 总分变成 np.float64 → core/quality.py:255 的 "
               "比较产 np.bool_ → core/quality.py:62 原样放进 to_dict。"
               "core/reporter.py:97 的 to_json() 因此对每块板都崩，而文档"
               "（core/quality.py:59）说 to_dict 就是给 JSON 输出用的。"
    )
    def test_json_dumps_succeeds_with_numpy_dci(self, assessor, heatmap_factory):
        """真实管线传入的 DCI 是 numpy 标量，此时也必须能 json.dumps。"""
        report = assessor.assess([], heatmap_factory(mean=0.1), np.float64(0.3), "B7")
        json.dumps(report.to_dict(), ensure_ascii=False)

    def test_json_dumps_succeeds_for_default_report(self):
        """默认构造的报告也必须可序列化（空报告也要能入库）。"""
        encoded = json.dumps(QualityReport().to_dict(), ensure_ascii=False)
        assert json.loads(encoded)["ok_ng"] is True

    def test_summary_contains_status_and_score(self, assessor, heatmap_factory,
                                               defect_factory):
        """摘要单行文本要带状态、总分与关键指标。"""
        report = assessor.assess(
            board(450, embedding=3, defect_factory=defect_factory),
            heatmap_factory(mean=0.1), 0.25, board_id="B7",
        )
        text = report.summary()
        assert "OK" in text
        assert f"{report.overall_score:.1f}" in text
        assert "4.5%" in text                     # 氧化面积比
        assert "\n" not in text

    def test_summary_marks_ng(self, assessor, heatmap_factory, defect_factory):
        """NG 板的摘要必须能一眼看出是 NG，不能与 OK 混淆。"""
        report = assessor.assess(
            board(2000, defect_factory=defect_factory), heatmap_factory(mean=0.1), 0.0
        )
        assert report.ok_ng is False
        assert "NG" in report.summary()
        assert "OK" not in report.summary()


# ============================================================================
# 畸形输入与契约边界
# ============================================================================

class TestMalformedInput:
    """畸形输入下 assess() 的实际行为 —— 记录，不改。

    关注点不是「抛不抛异常」，而是「会不会安静地放行一块坏板」。
    """

    def test_none_heatmap_is_treated_as_no_data(self, assessor):
        """热力图缺省 → 按「无数据」处理，粗糙度给满分而不是 0。

        与「DCI 缺省 → 按最差处理」方向相反，两者不一致（见
        TestCvHeatmapContract 之后的缺失输入用例）。
        """
        report = assessor.assess([], None, 0.0)
        assert report.roughness_uniformity == 100.0
        assert report.roughness_std == 0.0
        assert report.overall_score == 100.0

    def test_empty_heatmap_is_treated_as_no_data(self, assessor):
        """空热力图（size == 0）同样按无数据处理，不除零。"""
        report = assessor.assess([], np.zeros((0, 0)), 0.0)
        assert report.roughness_uniformity == 100.0
        assert report.overall_score == 100.0

    def test_defect_percentages_are_zero_without_heatmap(self, assessor, defect_factory):
        """风险行为：没有热力图时，缺陷面积比恒为 0，硬阈值永不触发。

        百分比的分母取的是热力图像素数（quality.py:164-167、197-203），
        热力图缺失时整段换算被跳过。整块板铺满氧化斑也照样 ox%=0、判 OK。
        Defect 自带 mask，本可用 mask.size 兜底，但实现没有这么做。
        """
        defects = [defect_factory("oxidation", 9000), defect_factory("unroughened", 9000)]
        report = assessor.assess(defects, None, 0.0)
        assert report.oxidation_percentage == 0.0
        assert report.unroughened_percentage == 0.0
        assert report.warnings == []
        assert report.ok_ng is True

    def test_three_d_heatmap_is_accepted_silently(self, assessor):
        """三维数组不会被拒绝：size 照样算总数，均值照算，粗糙度分照给。"""
        report = assessor.assess([], np.full((4, 4, 3), 0.1), 0.0)
        assert report.roughness_std == pytest.approx(0.1)
        assert report.detail["scores"]["roughness"] == pytest.approx(
            100.0 - (0.1 / CV_MAX) * 40.0, abs=0.05
        )
        assert report.overall_score == pytest.approx(96.6, abs=0.05)

    def test_heatmap_as_python_list_raises_attribute_error(self, assessor):
        """传 list 而非 ndarray 会抛 AttributeError（没有 .size）。

        异常类型不算友好（既不是 ValueError 也不是 TypeError），
        但至少是立刻失败，不会静默按空数据处理。
        """
        with pytest.raises(AttributeError):
            assessor.assess([], [[0.1, 0.2], [0.3, 0.4]], 0.0)

    def test_none_defects_raises_type_error(self, assessor, heatmap_factory):
        """defects=None 会抛 TypeError（不可迭代）。

        签名虽写作 List[Defect]，缺省并不接受 None。立刻失败比静默当空列表好，
        但调用方若漏传会拿到一个不易读的错误类型。
        """
        with pytest.raises(TypeError):
            assessor.assess(None, heatmap_factory(mean=0.1), 0.0)

    def test_non_embedding_unknown_type_is_ignored(self, assessor, heatmap_factory,
                                                   defect_factory):
        """未知缺陷类型被静默忽略，不影响任何指标。"""
        report = assessor.assess(
            board(0, scratch=5, defect_factory=defect_factory),
            heatmap_factory(mean=0.1), 0.0,
        )
        assert report.embedding_count == 0
        assert report.oxidation_percentage == 0.0
        assert report.unroughened_percentage == 0.0

    def test_negative_defect_area_is_harmless(self, assessor, heatmap_factory, defect_factory):
        """负面积（不应出现的脏数据）不会扣分也不会崩，按 0 处理。"""
        report = assessor.assess(
            board(-500, defect_factory=defect_factory),
            heatmap_factory(mean=0.1), 0.0,
        )
        assert report.oxidation_percentage == pytest.approx(-5.0)
        assert report.ok_ng is True
        assert 0.0 <= report.detail["scores"]["oxidation"] <= 100.0

    def test_nan_heatmap_fails_safe_to_ng(self, assessor):
        """全 NaN 热力图 → 总分 NaN，判 NG。

        计算不会被拦住（没有 finite 校验），但最终比较 NaN >= 60 为假，
        所以结论落在 NG 一侧 —— 对「宁可误报不可放行」的场合是可接受的兜底。
        """
        report = assessor.assess([], np.full(HEAT_SHAPE, np.nan), 0.0)
        assert np.isnan(report.overall_score)
        assert report.ok_ng is False

    def test_float32_heatmap_is_accepted(self, assessor):
        """float32 热力图按 float 处理，结果与 float64 一致。"""
        report = assessor.assess(
            [], np.full(HEAT_SHAPE, 0.1, dtype=np.float32), 0.0
        )
        assert report.roughness_std == pytest.approx(0.1, abs=1e-6)

    def test_huge_defect_area_triggers_hard_limit(self, assessor, heatmap_factory,
                                                  defect_factory):
        """极大面积（远超图像）不应溢出或回绕，仍判 NG。"""
        defects = [defect_factory("oxidation", 10 ** 9)]
        report = assessor.assess(defects, heatmap_factory(mean=0.1), 0.0)
        assert report.oxidation_percentage > OXIDATION_HARD_LIMIT
        assert report.ok_ng is False
        assert report.detail["scores"]["oxidation"] == 0.0


# ============================================================================
# 与上游 TextureAnalyzer 的口径契约（跨模块）
# ============================================================================

class TestCvHeatmapContract:
    """CV 热力图口径检查：quality.py 与 texture.py 是同一份数据的上下游。

    这两个模块对「CV 热力图」的取值约定必须一致，否则粗糙度这一路
    （权重 0.30，是五个维度里最重的）会整体失真。
    """

    @pytest.fixture
    def analyzer(self, default_config):
        """真实配置下的纹理分析器。"""
        return TextureAnalyzer(default_config)

    @pytest.mark.slow
    def test_heatmap_range_matches_absolute_threshold(self, analyzer, assessor,
                                                      sandblasted_image, flat_image):
        """记录：上游热力图恒被拉伸到 [0,1]，而下游阈值 0.35 是绝对值。

        上游 core/texture.py:756-758 用**本图自身的 95 分位**做归一化，
        于是任何有纹理的图均值都逼近 1；而 quality.py:292 把它当绝对
        CV 与 0.35 比。实测正常喷砂面 0.964 → 0 分，完全平坦面 0.0 → 100 分。
        """
        good_map = analyzer.compute_cv_heatmap(sandblasted_image)
        flat_map = analyzer.compute_cv_heatmap(flat_image)

        assert good_map.mean() > CV_MAX
        assert assessor._score_roughness(float(good_map.mean())) == 0.0
        assert float(flat_map.mean()) == pytest.approx(0.0, abs=1e-6)
        assert assessor._score_roughness(float(flat_map.mean())) == 100.0

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason="缺陷：粗糙度评分在真实管线上失真且方向颠倒 —— 上游 "
               "core/texture.py:756-758 把 CV 图按自身 95 分位归一化到 [0,1]，"
               "core/quality.py:292 却拿均值与绝对阈值 0.35 比较。结果：正常"
               "喷砂面（热点图均值 0.964）粗糙度得 0 分并挂假预警，总分被扣掉"
               "约 30 分；而完全没粗化的平坦面（CV=0）反倒得 100 分。"
               "「CV=0 → 100 分」的单调性在模块内部自洽，坏在两个模块的"
               "取值约定没对齐。"
    )
    def test_normal_surface_scores_better_than_flat_surface(self, analyzer, assessor,
                                                           sandblasted_image, flat_image):
        """正常喷砂面的粗糙度分不应低于完全平坦面（粗化过的表面更均匀才对）。"""
        good = analyzer.compute_cv_heatmap(sandblasted_image)
        flat = analyzer.compute_cv_heatmap(flat_image)
        assert (assessor._score_roughness(float(good.mean()))
                > assessor._score_roughness(float(flat.mean())))

    def test_direction_consistency_producer_returns_numpy_scalar(self, analyzer):
        """记录：上游 DCI 返回的是 numpy 标量，不是 Python float。

        这正是 numpy 标量一路渗进 QualityReport 的源头（见
        TestSerialization 里 json 序列化失败的 xfail）。用一个带纹理的
        小图触发非零 DCI —— 常数图的 DCI 恰好被夹到字面量 0.0，返回的是
        Python float，测不出这个口径。
        """
        textured = np.random.default_rng(11).integers(0, 256, (96, 96), dtype=np.uint8)
        dci = analyzer.direction_consistency(textured)
        assert isinstance(dci, np.floating)

    @pytest.mark.xfail(
        reason="缺陷：assess() 对缺失的 DCI 与缺失的热力图给了方向相反的默认值。"
               "quality.py:180-181 只在 DCI 非 None 时覆盖，于是沿用 "
               "QualityReport.direction_consistency 的默认值 1.0（= 方向一致性"
               "最差，quality.py:45），方向维度直接吃 0 分并挂出「direction "
               "评分偏低」的假预警；而热力图缺失时 quality.py:175-177 给的是"
               "满分。同一块板、同一批缺陷，漏传一个 DCI 就从 OK 变 NG。"
    )
    def test_missing_direction_consistency_is_not_worst_case(self, assessor,
                                                             heatmap_factory, defect_factory):
        """未提供 DCI 时不应等价于「方向一致性最差」。"""
        defects = board(450, unrough_px=250, embedding=9, defect_factory=defect_factory)
        heatmap = heatmap_factory(mean=0.3)

        with_dci = assessor.assess(defects, heatmap, 0.0)      # 理想各向同性
        without = assessor.assess(defects, heatmap, None)      # 调用方漏传

        assert without.overall_score == with_dci.overall_score

    def test_missing_direction_consistency_flips_verdict(self, assessor, heatmap_factory,
                                                         defect_factory):
        """缺失 DCI 造成判定翻转的现场复现（同板同缺陷，仅 DCI 一个参数之差）。

        实测：传 DCI=0.0 → 70.3 分 OK；漏传 DCI → 55.3 分 NG。
        本用例只记录事实，缺陷本身由上一个 xfail 用例主张。
        """
        defects = board(450, unrough_px=250, embedding=9, defect_factory=defect_factory)
        heatmap = heatmap_factory(mean=0.3)

        with_dci = assessor.assess(defects, heatmap, 0.0)
        without = assessor.assess(defects, heatmap, None)

        assert (without.overall_score, without.ok_ng) == (55.3, False)
        assert (with_dci.overall_score, with_dci.ok_ng) == (70.3, True)


# ============================================================================
# 内部共享状态
# ============================================================================

class TestInternalStateSharing:
    """评估器内部对象的共享与污染。"""

    @pytest.mark.xfail(
        reason="缺陷：report.detail['weights'] 直接引用评估器的 self.weights"
               "（core/quality.py:266 未做拷贝），调用方拿到报告后改一下 "
               "detail['weights']，就会永久改掉这个评估器后续所有板的权重分配，"
               "而 UI（ui/main_window.py:478）正是逐项遍历 to_dict() 的调用方。"
    )
    def test_mutating_report_detail_does_not_change_assessor(self, assessor, heatmap_factory):
        """改动已返回报告的 detail 不得影响评估器后续的评分。"""
        report = assessor.assess([], heatmap_factory(mean=0.1), 0.0)
        report.detail["weights"]["roughness"] = 0.99
        assert assessor.weights["roughness"] == 0.30

    def test_assessor_is_reusable_across_boards(self, assessor, heatmap_factory,
                                                defect_factory):
        """同一评估器连续评估多块板，结果不应互相影响。"""
        heatmap = heatmap_factory(mean=0.2)
        first = assessor.assess([], heatmap, 0.1)
        assessor.assess(board(2000, embedding=20, defect_factory=defect_factory),
                        heatmap, 0.9)
        third = assessor.assess([], heatmap, 0.1)
        assert third.overall_score == first.overall_score
        assert third.warnings == first.warnings

    def test_reports_do_not_share_warning_lists(self, assessor, heatmap_factory):
        """每份报告的 warnings 必须是独立列表（dataclass 默认工厂）。"""
        heatmap = heatmap_factory(mean=0.1)
        first = assessor.assess([], heatmap, 0.0)
        second = assessor.assess([], heatmap, 0.0)
        assert first.warnings is not second.warnings


# ============================================================================
# 配置读取
# ============================================================================

class TestConfigHandling:
    """配置读取与回退。"""

    def test_missing_quality_section_falls_back_to_defaults(self):
        """配置里没有 inspection.quality 时应正常构造并使用代码默认阈值。

        保证旧版配置文件仍可加载，不会导致系统启动失败。
        """
        assessor = QualityAssessor({"inspection": {}, "system": {}})
        assert assessor.cv_max == CV_MAX
        assert assessor.oxidation_max_pct == OXIDATION_MAX_PCT
        assert assessor.embedding_max_count == EMBEDDING_MAX_COUNT
        assert assessor.unroughened_max_pct == UNROUGHENED_MAX_PCT
        assert assessor.ok_score_threshold == OK_SCORE_THRESHOLD
        assert assessor.hard_limits["oxidation_pct"] == OXIDATION_HARD_LIMIT
        assert assessor.hard_limits["unroughened_pct"] == UNROUGHENED_HARD_LIMIT

    def test_empty_config_works(self):
        """完全不传配置也应可用。"""
        assert QualityAssessor({}).assess([]).ok_ng is True

    def test_real_config_matches_module_constants(self, assessor, default_config):
        """断言真实配置与本文引用的常量一致，配置改动会在测试里显形。"""
        assert assessor.cv_max == default_config["inspection"]["quality"]["roughness_cv_max"]
        assert assessor.ok_score_threshold == \
            default_config["inspection"]["quality"]["ok_score_threshold"]

    def test_resolution_is_read_from_system_section(self, config_factory):
        """分辨率取自 system 节（缺陷面积换算用，缺省 0.01 mm/px）。"""
        cfg = config_factory()
        cfg["system"]["resolution_mm_per_pixel"] = 0.02
        assert QualityAssessor(cfg).resolution_mm_per_pixel == 0.02
        assert QualityAssessor({"system": {}}).resolution_mm_per_pixel == 0.01

    def test_custom_ok_threshold_is_respected(self, quality_config_factory, heatmap_factory):
        """ok_score_threshold 可覆盖：抬到 99 后满分以外的板一律 NG。"""
        assessor = QualityAssessor(quality_config_factory(ok_score_threshold=99))
        report = assessor.assess([], heatmap_factory(mean=0.1), 0.0)
        assert report.overall_score < 99
        assert report.ok_ng is False
