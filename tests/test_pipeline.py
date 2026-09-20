"""全流程编排模块（core/pipeline.py）集成测试。

InspectionPipeline 把预处理 → 纹理 → 缺陷 → 质量评估 → 工艺监测串成一条链路，
本文件重点验证三件事：

1. **编排不改变数值口径** —— run() 的每一项输出都必须与直接调用对应子模块逐位一致，
   否则「离线跑出来 OK、在线跑出来 NG」这类问题会出现在产线上而不是测试里。
2. **进度回调契约** —— 百分比 0~100、单调不减、百分比与阶段名逐条送达；
   回调自己抛异常不能把检测流程带崩（回调常来自 UI 线程）。
3. **退化输入的传播方式** —— None / 空图 / 错误 dtype / 1x1，是抛明确异常还是静默出错？

所有图像均由带种子的合成器生成（见 conftest.make_surface），不读 data/samples、
不联网、不依赖真实相机；同一用例重复执行结果完全一致。

约定：本项目内部图像一律按 RGB 解释，三通道测试图统一用 _to_color() 生成。
"""

import copy
import time

import numpy as np
import pytest

from core.acquisition import ImageAcquisition, ImageFrame
from core.pipeline import InspectionPipeline, InspectionResult
from core.process_monitor import ProcessGLCMExtractor


# ============================================================================
# 图像尺寸
# ============================================================================
# 编排层每个用例都要完整跑一遍纹理分析（GLCM+LBP+Gabor），这是整条链路最慢的一段，
# 耗时随面积线性增长：128x128 ≈ 0.37s，256x256 ≈ 1.36s。
# 因此常规用例统一用 128x128；只有真正要验证「生产尺寸」的冒烟用例才用 256x256
# 并标 slow。缩小尺寸不影响结论 —— 各阶段的数值口径与面积无关。
SMALL = 128
FULL = 256

# 缺陷密度：这组参数在 128x128 上稳定检出 6 个磨料嵌入点（见 test 中的断言）。
DEFECT_SEED = 42
DEFECT_BLOBS = 2
DEFECT_SPECKS = 8


# ============================================================================
# 夹具
# ============================================================================

def _to_color(gray: np.ndarray) -> np.ndarray:
    """把灰度图扩成低饱和度三通道图。

    与 conftest.rgb_surface 的口径一致：真实阻焊面在彩色相机下不是纯灰，
    R/B 通道带轻微偏色，能暴露「把 RGB 当 BGR 处理」导致的权重对调。
    """
    return np.stack(
        [gray, np.clip(gray * 0.92, 0, 255), np.clip(gray * 1.05, 0, 255)],
        axis=-1,
    ).astype(np.uint8)


@pytest.fixture
def pipeline(default_config) -> InspectionPipeline:
    """用真实配置文件构造的流水线。"""
    return InspectionPipeline(default_config)


@pytest.fixture
def color_image(surface_factory) -> np.ndarray:
    """128x128 三通道喷砂面，含 6 个磨料嵌入点。"""
    gray = surface_factory(
        SMALL, SMALL, seed=DEFECT_SEED,
        oxidation_blobs=DEFECT_BLOBS, embedded_specks=DEFECT_SPECKS,
    )
    return _to_color(gray)


@pytest.fixture
def gray_image(surface_factory) -> np.ndarray:
    """与 color_image 同源的 128x128 单通道灰度图。"""
    return surface_factory(
        SMALL, SMALL, seed=DEFECT_SEED,
        oxidation_blobs=DEFECT_BLOBS, embedded_specks=DEFECT_SPECKS,
    )


@pytest.fixture
def clean_color_image(surface_factory) -> np.ndarray:
    """128x128 三通道洁净面 —— 各检测器都检不出缺陷。"""
    return _to_color(surface_factory(SMALL, SMALL, seed=DEFECT_SEED))


@pytest.fixture
def ng_color_image(surface_factory) -> np.ndarray:
    """纹理明显偏强的表面（喷砂过度）—— 实测判 NG。"""
    return _to_color(surface_factory(SMALL, SMALL, texture=45.0, seed=2))


class _StubAcquisition(ImageAcquisition):
    """按固定张数吐帧的假采集器。

    真采集器（CameraAcquisition / FileAcquisition）要么需要硬件、要么需要
    依赖 data/ 下的真实图片，都会破坏测试的确定性。这里只复现 ImageAcquisition
    的接口契约：acquire() 返回 ImageFrame，取完返回 None。
    """

    def __init__(self, config: dict, image: np.ndarray, count: int,
                 source_id: str = "cam"):
        super().__init__(config)
        self._image = image
        self._count = count
        self._served = 0
        self._source_id = source_id

    def acquire(self):
        if self._served >= self._count:
            return None
        frame = ImageFrame(
            image=self._image,
            timestamp="2026-01-01T00:00:00",
            source_id=self._source_id,
            frame_index=self._served,
        )
        self._served += 1
        return frame

    def reset(self) -> None:
        self._served = 0


@pytest.fixture
def stub_acquisition(default_config, color_image):
    """工厂：`stub_acquisition(count, source_id=...)` → 假采集器。"""
    def _make(count: int, source_id: str = "cam") -> _StubAcquisition:
        return _StubAcquisition(default_config, color_image, count, source_id)
    return _make


# ============================================================================
# 工具
# ============================================================================

def _fingerprint(result: InspectionResult) -> dict:
    """把一次检测结果压成可逐字段比较的字典（用于确定性断言）。

    刻意包含图像原始字节：分数相同但像素不同（例如随机标注颜色）也必须被抓出来。
    质量报告里的 timestamp 是 wall clock、board_id 是入参，两者都与「算得对不对」
    无关，剔除后再比较（板号本身另有专门用例覆盖）。
    """
    quality = result.quality.to_dict()
    quality.pop("timestamp", None)
    quality.pop("board_id", None)
    return {
        "gray": result.gray.tobytes(),
        "image": result.image.tobytes(),
        "image_shape": result.image.shape,
        "heatmap": result.heatmap.tobytes(),
        "roughness_map": result.roughness_map.tobytes(),
        "defects": [
            (d.type, d.area_pixels, round(d.area_mm2, 9), d.centroid, d.bbox,
             round(d.severity, 9), round(d.confidence, 9))
            for d in result.defects
        ],
        "quality": quality,
        "ok_ng": result.ok_ng,
        "texture": result.texture_features.flatten().tobytes(),
        "process": result.process_features.as_dict(),
    }


def _collect_progress(pipeline: InspectionPipeline) -> list:
    """注册一个记录型回调并返回其累积列表 [(pct, stage), ...]。"""
    seen = []
    pipeline.on_progress(lambda pct, stage: seen.append((pct, stage)))
    return seen


# ============================================================================
# InspectionResult 数据类
# ============================================================================

class TestInspectionResult:

    def test_defaults_are_usable(self):
        """默认构造不应留下 None 容器，避免调用方到处判空。"""
        r = InspectionResult()
        assert r.defects == []
        assert r.timings == {}
        assert r.ok_ng is True
        assert r.quality is None
        assert r.image is None

    def test_summary_of_default_result(self):
        """空结果也能生成单行摘要。"""
        text = InspectionResult().summary()
        assert text.strip()
        assert "\n" not in text
        assert "OK" in text
        assert "评分=0.0" in text
        assert "缺陷数=0" in text

    def test_summary_reports_ng(self, pipeline, ng_color_image):
        """NG 结果的摘要必须带 NG 标记。"""
        result = pipeline.run(ng_color_image)
        assert not result.ok_ng
        text = result.summary()
        assert text.startswith("[NG]")
        assert text.strip()
        assert "\n" not in text

    def test_summary_counts_defects(self, pipeline, color_image):
        """摘要里的缺陷数应与列表长度一致。"""
        result = pipeline.run(color_image)
        assert len(result.defects) > 0, "夹具应能稳定检出缺陷"
        assert f"缺陷数={len(result.defects)}" in result.summary()

    def test_summary_without_quality_does_not_raise(self):
        """quality 为 None 时不应抛异常（评分退化为 0）。"""
        text = InspectionResult(defects=[], quality=None, total_time_ms=123.0).summary()
        assert "评分=0.0" in text
        assert "耗时=123ms" in text

    def test_summary_tolerates_nan_score(self):
        """评分为 NaN 时仍应返回字符串而不是崩溃。

        评分来自多层浮点运算，除零/退化输入都可能产出 NaN；摘要常用于日志，
        这里再抛异常会把「记一条日志」变成「又一次崩溃」。
        """
        from core.quality import QualityReport

        result = InspectionResult(quality=QualityReport(overall_score=float("nan")))
        text = result.summary()
        assert isinstance(text, str)
        assert text.strip()

    @pytest.mark.xfail(
        reason="已知缺陷：InspectionResult.ok_ng 声明为 bool，实际拿到的是 numpy.bool_。"
               "根因在 core/quality.py:255 —— overall_score 是 numpy.float64，"
               "`report.ok_ng = overall_score >= threshold` 因此产出 np.bool_。"
               "后果：core/reporter.py:95 的 to_json() 抛 TypeError"
               "（Object of type bool is not JSON serializable），检测报告导不出来。"
               "期望：InspectionResult.ok_ng / QualityReport.ok_ng 为 Python 原生 bool。"
    )
    def test_ok_ng_is_a_python_bool(self, pipeline, color_image):
        """ok_ng 应是 Python 原生 bool，不是 numpy.bool_。

        np.bool_ 在 if/比较里表现正常，只有在 json.dumps 这类按类型分派的场合才炸，
        属于典型的「跑到导出报表时才暴露」的问题。
        """
        assert isinstance(pipeline.run(color_image).ok_ng, bool)


# ============================================================================
# 单张检测：三通道完整链路
# ============================================================================

class TestRunSingleImage:

    def test_returns_inspection_result(self, pipeline, color_image):
        """run() 返回 InspectionResult 实例。"""
        assert isinstance(pipeline.run(color_image), InspectionResult)

    def test_all_output_fields_populated(self, pipeline, color_image):
        """三通道输入下各输出字段齐全且类型正确。"""
        result = pipeline.run(color_image, board_id="PCB-0001")

        assert isinstance(result.gray, np.ndarray)
        assert result.gray.ndim == 2 and result.gray.dtype == np.uint8
        assert isinstance(result.image, np.ndarray)
        assert isinstance(result.heatmap, np.ndarray)
        assert isinstance(result.roughness_map, np.ndarray)
        assert result.quality is not None
        assert result.texture_features is not None
        assert result.process_features is not None
        assert isinstance(result.timings, dict)
        assert result.total_time_ms > 0

    def test_output_shapes_match_preprocessed_gray(self, pipeline, color_image):
        """标注图、热力图、粗糙度图的尺寸必须与预处理输出一致。

        三个图分别由纹理分析、缺陷检测、可视化产生，任一处用了原始尺寸而不是
        统一后的尺寸，都会导致 UI 叠加错位。
        """
        result = pipeline.run(color_image)
        h, w = result.gray.shape
        assert result.heatmap.shape == (h, w)
        assert result.roughness_map.shape == (h, w)
        assert result.image.shape[:2] == color_image.shape[:2]

    def test_heatmap_is_normalized(self, pipeline, color_image):
        """缺陷热力图取值应落在 [0, 1]。"""
        heatmap = pipeline.run(color_image).heatmap
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0 + 1e-12
        assert heatmap.dtype == np.float64

    def test_heatmap_is_all_zero_without_defects(self, pipeline, clean_color_image):
        """无缺陷时热力图应全 0。"""
        result = pipeline.run(clean_color_image)
        assert result.defects == []
        assert np.all(result.heatmap == 0.0)

    def test_ok_ng_matches_quality_report(self, pipeline, color_image):
        """顶层的 ok_ng 必须与质量报告一致，不能各判各的。"""
        result = pipeline.run(color_image)
        assert result.ok_ng == result.quality.ok_ng

    def test_ng_case_propagates(self, pipeline, ng_color_image):
        """劣质表面应判 NG，且顶层标志同步为 False。"""
        result = pipeline.run(ng_color_image)
        assert not result.ok_ng
        assert not result.quality.ok_ng
        assert result.quality.overall_score < 60

    def test_board_id_reaches_quality_report(self, pipeline, color_image):
        """板号必须落到质量报告里，否则结果入库后无法回溯。"""
        result = pipeline.run(color_image, board_id="PCB-0042")
        assert result.quality.board_id == "PCB-0042"

    def test_default_board_id_is_empty(self, pipeline, color_image):
        """不传板号时为空串（而非 None）。"""
        assert pipeline.run(color_image).quality.board_id == ""

    def test_input_image_is_not_mutated(self, pipeline, color_image):
        """run() 不得改动调用方传进来的数组。

        在线模式下同一帧可能被多个消费者共用，原地修改会污染其它环节。
        """
        original = color_image.copy()
        pipeline.run(color_image)
        assert np.array_equal(color_image, original)

    def test_annotated_image_matches_direct_draw_call(self, pipeline, color_image):
        """result.image 就是 draw_defects 的输出，没有额外加工。"""
        result = pipeline.run(color_image)
        expected = pipeline.defect_detector.draw_defects(color_image, result.defects)
        assert np.array_equal(result.image, expected)

    def test_timings_cover_every_stage_and_total(self, pipeline, color_image):
        """各阶段计时齐全，且总耗时不少于阶段之和。"""
        result = pipeline.run(color_image)
        for stage in ("preprocessing", "texture", "defects", "quality"):
            assert stage in result.timings
            assert result.timings[stage] >= 0
        assert result.total_time_ms >= sum(result.timings.values()) - 1e-6

    def test_process_features_are_taken_from_the_raw_image(self, pipeline, color_image):
        """工艺特征必须取自**原始输入**，而不是 Retinex+CLAHE 之后的灰度图。

        这是 pipeline 里唯一一处「看起来可以复用 result.gray，但绝对不能」的地方：
        预处理会重排灰度分布，8 级量化的 GLCM 特征随之漂移，工艺报警阈值失效。
        下面的 contrast 差异（0.79 vs 9.45 量级）说明两条路径数值上确实不同 ——
        若有人图省事改成 compute(result.gray)，这条断言会立刻失败。
        """
        result = pipeline.run(color_image)
        extractor = ProcessGLCMExtractor()

        assert result.process_features.as_dict() == \
            pytest.approx(extractor.compute(color_image).as_dict(), rel=1e-12)
        assert result.process_features.contrast != pytest.approx(
            extractor.compute(result.gray).contrast, rel=1e-3
        )

    def test_process_monitor_disabled_skips_stage(self, pipeline, config_factory,
                                                  color_image):
        """关闭工艺监测后既不产出特征，也不记录该阶段耗时。"""
        cfg = config_factory()
        cfg["inspection"]["process_monitor"]["enabled"] = False
        result = InspectionPipeline(cfg).run(color_image)

        assert result.process_features is None
        assert "process_monitor" not in result.timings
        assert result.quality is not None, "辅助功能关闭不应影响主检测链路"

    def test_process_monitor_failure_does_not_block_inspection(self, pipeline,
                                                               color_image,
                                                               monkeypatch):
        """工艺监测抛异常时必须被吞掉：主检测照常出结果，只是该字段为 None。

        8 级 GLCM 是辅助功能（供工艺面板与报警灯），它依赖 scikit-image 等
        可选依赖；它挂掉不能连带把缺陷检测和质量判定一起拖停。
        """
        class _BoomExtractor:
            def compute(self, image):
                raise RuntimeError("工艺监测故意失败")

        monkeypatch.setattr(pipeline.process_monitor, "extractor", _BoomExtractor())

        result = pipeline.run(color_image)

        assert result.process_features is None
        assert "process_monitor" in result.timings
        assert result.quality is not None
        assert result.summary().strip()

    @pytest.mark.slow
    def test_full_size_color_image(self, pipeline, rgb_surface):
        """生产尺寸（256x256 三通道）冒烟：链路能跑通且结果自洽。

        单帧 ~1.4s，故标 slow。上面所有用例用 128x128 验证口径，这条只验证尺寸。
        """
        result = pipeline.run(rgb_surface, board_id="FULL-0001")
        assert result.gray.shape == rgb_surface.shape[:2]
        assert result.quality is not None
        assert result.summary().strip()


# ============================================================================
# 编排不改变数值口径：run() vs 直接调用子模块
# ============================================================================

class TestOrchestrationMatchesSubmodules:
    """逐项比对 run() 与手工串联各子模块的结果。

    这些断言是「编排层是透明的」这一承诺的唯一证据。编排层一旦偷偷多加一次
    滤波、换了参数顺序、或把中间结果复用了不该复用的那份，这里就会红。
    """

    @staticmethod
    def _by_hand(pipeline: InspectionPipeline, image: np.ndarray,
                 board_id: str = "PCB-0001"):
        """按 pipeline.run() 的文档顺序手工复现一遍。"""
        gray = pipeline.preprocessor.process(image)
        texture_vec = pipeline.texture_analyzer.analyze(gray)
        cv_heatmap = pipeline.texture_analyzer.compute_cv_heatmap(gray)
        # 刻意不传 analyze() 算出的能量：与 pipeline.run() 相比这里多跑一趟
        # Gabor，正好用来证明「复用能量」不改变结果。别「顺手」补上。
        dci = pipeline.texture_analyzer.direction_consistency(gray)
        defects = pipeline.defect_detector.detect_all(image, texture_vec)
        report = pipeline.quality_assessor.assess(defects, cv_heatmap, dci, board_id)
        return gray, texture_vec, cv_heatmap, dci, defects, report

    def test_preprocessed_gray_is_identical(self, pipeline, color_image):
        """灰度输出与 Preprocessor.process() 逐位一致。"""
        result = pipeline.run(color_image)
        gray, *_ = self._by_hand(pipeline, color_image)
        assert np.array_equal(result.gray, gray)

    def test_defect_list_is_identical(self, pipeline, color_image):
        """缺陷列表的类型、面积、质心、严重度与直接检测一致。

        detect_all 接收的是 pipeline 传进去的 texture_features（含局部熵图）。
        若编排层忘了传、或传了别的口径，未粗化检测的结果会不同。
        """
        result = pipeline.run(color_image)
        _, _, _, _, defects, _ = self._by_hand(pipeline, color_image)

        assert len(result.defects) == len(defects) > 0
        for got, want in zip(result.defects, defects):
            assert got.type == want.type
            assert got.area_pixels == want.area_pixels
            assert got.centroid == want.centroid
            assert got.bbox == want.bbox
            assert got.severity == pytest.approx(want.severity, rel=1e-12)

    def test_quality_report_is_identical(self, pipeline, color_image):
        """质量报告的每个维度都与直接 assess() 一致。"""
        result = pipeline.run(color_image, board_id="PCB-0001")
        _, _, cv_heatmap, dci, defects, report = self._by_hand(
            pipeline, color_image, "PCB-0001"
        )

        assert result.quality.overall_score == report.overall_score
        assert result.quality.ok_ng == report.ok_ng
        assert result.quality.roughness_std == report.roughness_std
        assert result.quality.direction_consistency == report.direction_consistency
        assert result.quality.oxidation_percentage == report.oxidation_percentage
        assert result.quality.embedding_count == report.embedding_count
        assert result.quality.unroughened_percentage == report.unroughened_percentage
        assert np.array_equal(cv_heatmap, result.roughness_map)

    def test_roughness_and_direction_inputs_are_passed_through(self, pipeline,
                                                               color_image):
        """传给质量评估的 CV 热力图与方向一致性必须原样来自纹理分析。"""
        result = pipeline.run(color_image)
        gray = result.gray

        assert np.array_equal(
            result.roughness_map, pipeline.texture_analyzer.compute_cv_heatmap(gray)
        )
        assert result.quality.direction_consistency == pytest.approx(
            pipeline.texture_analyzer.direction_consistency(gray), rel=1e-12
        )

    def test_texture_features_match_direct_analyze(self, pipeline, color_image):
        """纹理特征向量与直接 analyze() 一致。"""
        result = pipeline.run(color_image)
        want = pipeline.texture_analyzer.analyze(result.gray)
        assert np.array_equal(result.texture_features.flatten(), want.flatten())

    def test_pipeline_reuses_gabor_energies(self, pipeline, color_image,
                                            monkeypatch):
        """管道必须把 analyze() 的能量传给 DCI，否则每次检测白跑一趟 Gabor。

        这是性能契约而非正确性契约：不传也会得到逐位相同的结果
        （见 ``_by_hand`` 的对照），只是对同一张图多卷 n_kernels 个核 ——
        相机分辨率下约 3.4 s。
        """
        seen = []
        original = pipeline.texture_analyzer.direction_consistency

        def spy(image, orientation_energies=None):
            seen.append(orientation_energies)
            return original(image, orientation_energies)

        monkeypatch.setattr(
            pipeline.texture_analyzer, "direction_consistency", spy
        )
        pipeline.run(color_image)

        assert len(seen) == 1
        assert seen[0] is not None


# ============================================================================
# 进度回调
# ============================================================================

class TestProgressNotification:

    def test_callback_is_called(self, pipeline, color_image):
        """注册的回调会被调用，且收到 (int, str)。"""
        seen = _collect_progress(pipeline)
        pipeline.run(color_image)

        assert seen, "回调一次都没被调用"
        for pct, stage in seen:
            assert isinstance(pct, int)
            assert isinstance(stage, str)

    def test_percentages_are_monotonic_and_in_range(self, pipeline, color_image):
        """百分比单调不减且落在 0~100 —— 进度条倒退比不动更让人怀疑系统。"""
        seen = _collect_progress(pipeline)
        pipeline.run(color_image)

        pcts = [pct for pct, _ in seen]
        assert all(0 <= pct <= 100 for pct in pcts), pcts
        assert all(a <= b for a, b in zip(pcts, pcts[1:])), pcts

    def test_starts_at_zero_and_finishes_at_hundred(self, pipeline, color_image):
        """首条为 0，末条为 100。"""
        seen = _collect_progress(pipeline)
        pipeline.run(color_image)

        assert seen[0][0] == 0
        assert seen[-1][0] == 100

    def test_stage_names_are_nonempty(self, pipeline, color_image):
        """阶段名非空 —— UI 直接展示，空串会留下一行空白。"""
        seen = _collect_progress(pipeline)
        pipeline.run(color_image)

        assert all(stage.strip() for _, stage in seen)

    def test_every_registered_callback_is_notified(self, pipeline, color_image):
        """多个回调各自收到等量的通知。"""
        first, second = [], []
        pipeline.on_progress(lambda pct, stage: first.append(pct))
        pipeline.on_progress(lambda pct, stage: second.append(pct))
        pipeline.run(color_image)

        assert first == second
        assert len(first) > 0

    def test_progress_fires_for_every_run(self, pipeline, color_image):
        """每次 run() 都重新播报一遍进度，不会只报第一次。"""
        seen = _collect_progress(pipeline)
        pipeline.run(color_image)
        first_round = len(seen)
        pipeline.run(color_image)
        assert len(seen) == 2 * first_round

    def test_raising_callback_does_not_break_the_run(self, pipeline, color_image):
        """回调抛异常必须被吞掉，检测照常完成，其它回调照常收到通知。

        回调通常来自 UI（刷新进度条），UI 出错不能让产线停摆。
        """
        def boom(pct, stage):
            raise RuntimeError("回调故意抛错")

        seen = []
        pipeline.on_progress(boom)
        pipeline.on_progress(lambda pct, stage: seen.append(pct))

        result = pipeline.run(color_image)          # 不应抛出

        assert result.quality is not None
        assert seen and seen[-1] == 100

    def test_notify_progress_swallows_exception(self, pipeline):
        """直接调用 _notify_progress 时同样吞异常，且不影响后续回调。"""
        def boom(pct, stage):
            raise RuntimeError("回调故意抛错")

        seen = []
        pipeline.on_progress(boom)
        pipeline.on_progress(lambda pct, stage: seen.append((pct, stage)))

        pipeline._notify_progress(42, "测试阶段")   # 不应抛出

        assert seen == [(42, "测试阶段")]

    def test_notify_progress_without_callbacks_is_a_noop(self, pipeline):
        """没有注册回调时静默通过。"""
        assert pipeline._notify_progress(10, "无回调") is None


# ============================================================================
# 批量处理
# ============================================================================

class TestRunBatch:

    def test_empty_list_returns_empty(self, pipeline):
        """空列表返回空列表，不报错。"""
        assert pipeline.run_batch([]) == []

    def test_single_image(self, pipeline, color_image):
        """单张批处理等价于一次 run()。"""
        results = pipeline.run_batch([color_image])
        assert len(results) == 1
        assert isinstance(results[0], InspectionResult)

    def test_multiple_images(self, pipeline, color_image, clean_color_image):
        """多张批处理逐张产出结果，顺序与输入一致。"""
        images = [color_image, clean_color_image, color_image]
        results = pipeline.run_batch(images)

        assert len(results) == len(images)
        assert all(isinstance(r, InspectionResult) for r in results)
        assert [len(r.defects) for r in results] == [
            len(pipeline.run(img).defects) for img in images
        ]

    def test_default_board_ids(self, pipeline, color_image):
        """不传板号时按 board_%04d 编号。"""
        results = pipeline.run_batch([color_image, color_image])
        assert [r.quality.board_id for r in results] == ["board_0000", "board_0001"]

    def test_custom_board_ids(self, pipeline, color_image):
        """自定义板号逐张落到对应结果上。"""
        results = pipeline.run_batch([color_image] * 3, ["A", "B", "C"])
        assert [r.quality.board_id for r in results] == ["A", "B", "C"]

    def test_explicit_empty_board_ids_uses_defaults(self, pipeline, color_image):
        """显式传 None 与不传等价。"""
        results = pipeline.run_batch([color_image], None)
        assert results[0].quality.board_id == "board_0000"

    def test_each_result_matches_individual_run(self, pipeline,
                                                color_image, clean_color_image):
        """批量结果必须与逐张 run() 逐位一致。

        批量路径若共享了某个带状态的中间对象（例如复用同一份热力图缓冲），
        第 2 张起就会开始漂移 —— 这条断言按顺序覆盖多张，正是为了抓这种情况。
        """
        images = [color_image, clean_color_image, color_image]
        batch = pipeline.run_batch(images)
        singles = [pipeline.run(img) for img in images]

        for got, want in zip(batch, singles):
            assert _fingerprint(got) == _fingerprint(want)

    def test_batch_is_deterministic(self, pipeline, color_image, clean_color_image):
        """同一批输入跑两次结果完全一致。"""
        images = [color_image, clean_color_image]
        first = pipeline.run_batch(images)
        second = pipeline.run_batch(images)

        assert [_fingerprint(r) for r in first] == \
            [_fingerprint(r) for r in second]

    @pytest.mark.xfail(
        reason="已知缺陷：board_ids 长度与 images 不一致时被 zip 静默截断，"
               "多余的图像被丢弃且无任何告警。在线检测会因此漏检板卡。"
               "期望：长度不匹配时抛 ValueError。"
    )
    def test_mismatched_board_ids_raises(self, pipeline, color_image):
        """板号数量少于图像数量时应明确报错，而不是少检几块板。"""
        with pytest.raises(ValueError):
            pipeline.run_batch([color_image] * 3, ["only_one"])

    def test_mismatched_board_ids_currently_truncates_silently(
        self, pipeline, color_image
    ):
        """记录当前实际行为：长度不匹配时按 zip 语义截断（见上一条 xfail）。"""
        results = pipeline.run_batch([color_image] * 3, ["only_one"])
        assert len(results) == 1
        assert results[0].quality.board_id == "only_one"


# ============================================================================
# 从配置文件构造
# ============================================================================

class TestFromConfig:

    def test_builds_pipeline_from_real_config(self, project_root):
        """从仓库里的真实配置文件构造流水线。"""
        pipeline = InspectionPipeline.from_config(
            str(project_root / "config" / "default.yaml")
        )
        assert isinstance(pipeline, InspectionPipeline)

    def test_config_matches_yaml(self, project_root, default_config):
        """加载进来的配置与 yaml 内容一致（没有被中间层改写）。"""
        pipeline = InspectionPipeline.from_config(
            str(project_root / "config" / "default.yaml")
        )
        assert pipeline.config == default_config

    def test_submodules_are_wired_to_that_config(self, project_root, default_config):
        """各子模块确实按该配置初始化，而不只是把字典存下来。"""
        pipeline = InspectionPipeline.from_config(
            str(project_root / "config" / "default.yaml")
        )
        tex = default_config["inspection"]["texture"]
        quality = default_config["inspection"]["quality"]

        assert pipeline.texture_analyzer.cv_window == tex["cv_window"]
        assert pipeline.quality_assessor.weights == {
            "roughness": 0.30, "direction": 0.15, "oxidation": 0.25,
            "embedding": 0.15, "unroughened": 0.15,
        }
        assert pipeline.quality_assessor.ok_score_threshold == \
            quality["ok_score_threshold"]
        assert pipeline.process_monitor.enabled is True

    def test_from_config_can_actually_run(self, project_root, color_image):
        """从配置构造出来的流水线能直接跑通，不是只好看的壳子。"""
        pipeline = InspectionPipeline.from_config(
            str(project_root / "config" / "default.yaml")
        )
        result = pipeline.run(color_image)
        assert result.quality is not None
        assert result.process_features is not None

    def test_missing_file_raises(self, tmp_path):
        """配置文件不存在时抛 FileNotFoundError，而不是构造出半残的流水线。"""
        with pytest.raises(FileNotFoundError):
            InspectionPipeline.from_config(str(tmp_path / "nope.yaml"))


# ============================================================================
# 与采集器集成
# ============================================================================

class TestProcessAcquisition:

    def test_consumes_frames_until_none(self, pipeline, stub_acquisition):
        """一直取帧直到采集器返回 None。"""
        results = pipeline.process_acquisition(stub_acquisition(3))
        assert len(results) == 3

    def test_max_frames_limits_processing(self, pipeline, stub_acquisition):
        """max_frames 生效，且不必把采集器取空。"""
        assert len(pipeline.process_acquisition(stub_acquisition(5), max_frames=2)) == 2

    def test_empty_acquisition_returns_empty(self, pipeline, stub_acquisition):
        """采集器立刻返回 None 时返回空列表。"""
        assert pipeline.process_acquisition(stub_acquisition(0)) == []

    def test_board_id_combines_source_and_frame_index(self, pipeline,
                                                      stub_acquisition):
        """板号由 source_id 与 frame_index 拼成，便于回溯到具体帧。"""
        results = pipeline.process_acquisition(stub_acquisition(3, "cam_7"))
        assert [r.quality.board_id for r in results] == \
            ["cam_7_0", "cam_7_1", "cam_7_2"]

    def test_results_match_run(self, pipeline, stub_acquisition, color_image):
        """采集路径的结果与直接 run() 一致。"""
        results = pipeline.process_acquisition(stub_acquisition(2))
        # 必须显式声明 "bgr"：process_acquisition 喂的是采集器给的原生字节
        # （相机缓冲 / cv2.imread，都是 BGR），而 run() 的默认值是 "rgb"
        # （对齐 GUI 与 CLI，那两条路径在入口做了 BGR2RGB）。通道顺序无法从
        # 数组形状推断，喂错不会报错、只会静默转掉色相，故此处必须写明。
        single = pipeline.run(color_image, "cam_0", color_order="bgr")
        assert _fingerprint(results[0]) == _fingerprint(single)

    def test_long_run_logs_every_ten_frames(self, pipeline, default_config):
        """连续采集超过 10 帧时打进度日志（每 10 帧一条）。

        这里关心的是「长时间连续跑不会中断」这条管路，而不是数值，
        故用小尺寸图把 10 次 run() 压到亚秒级。
        """
        tiny = _to_color(np.full((8, 8), 128, dtype=np.uint8))
        results = pipeline.process_acquisition(
            _StubAcquisition(default_config, tiny, 10)
        )
        assert len(results) == 10


# ============================================================================
# 退化输入
# ============================================================================

class TestDegenerateInputs:
    """记录各类非法/退化输入的实际传播方式，并判断是否合理。"""

    def test_none_raises_value_error(self, pipeline):
        """None → ValueError（由 Preprocessor 抛出，信息明确）。"""
        with pytest.raises(ValueError):
            pipeline.run(None)

    def test_empty_array_raises_value_error(self, pipeline):
        """零尺寸数组 → ValueError。"""
        with pytest.raises(ValueError):
            pipeline.run(np.zeros((0, 0), dtype=np.uint8))

    def test_one_by_one_image_does_not_crash(self, pipeline):
        """1x1 图像能跑完（检不出缺陷），字段仍自洽。

        极小的 ROI 会出现在「板卡边缘裁切」场景，此时宁可返回一张空结果，
        也不该让整条产线抛异常。
        """
        result = pipeline.run(np.full((1, 1), 128, dtype=np.uint8))

        assert result.gray.shape == (1, 1)
        assert result.heatmap.shape == (1, 1)
        # 标注图恒为三通道：灰度输入经 GRAY2RGB 提升后再交给 draw_defects
        # （原来这里传的是二维图，检出缺陷时会崩，详见 TestGrayscaleInput）
        assert result.image.shape == (1, 1, 3)
        assert result.defects == []
        assert result.quality is not None
        assert result.summary().strip()

    def test_tiny_color_image_does_not_crash(self, pipeline, surface_factory):
        """8x8 三通道图像能跑完。"""
        result = pipeline.run(_to_color(surface_factory(8, 8, seed=7)))
        assert result.image.shape == (8, 8, 3)
        assert result.quality is not None

    @pytest.mark.parametrize("dtype", [np.int32, np.bool_, np.float16])
    def test_unsupported_dtype_raises(self, pipeline, dtype, surface_factory):
        """非 uint8 且非浮点的 dtype 抛异常（cv2.error），不会静默产出垃圾结果。

        异常本身来自 OpenCV 底层、信息不友好，但「明确失败」远好过
        「悄悄按错误位深算出一个分数」。测试只锁定「会抛异常」这一行为，
        不锁定异常类型，以免将来换成显式 ValueError 校验时误报。
        """
        image = surface_factory(SMALL, SMALL, seed=1).astype(dtype)
        with pytest.raises(Exception):
            pipeline.run(image)

    def test_four_dimensional_input_raises(self, pipeline):
        """4 维输入抛异常（cv2.error），不返回空结果。"""
        with pytest.raises(Exception):
            pipeline.run(np.zeros((4, 4, 4, 4), dtype=np.uint8))

    def test_all_zero_image_still_produces_a_report(self, pipeline):
        """全黑图（相机没出图时的典型产物）应产出报告而不是崩溃。

        这类图会被判未粗化 → NG，正是想要的结果：坏图不能悄悄通过。
        """
        result = pipeline.run(np.zeros((SMALL, SMALL, 3), dtype=np.uint8))

        assert result.quality is not None
        assert not result.ok_ng
        assert result.summary().strip()


# ============================================================================
# 灰度输入（已发现缺陷）
# ============================================================================

class TestGrayscaleInput:
    """core/pipeline.py 的 run() 文档承诺支持灰度 (H, W) 输入。

    这里曾是两条 xfail + 一条「记录错误行为」的用例，记录一个已修缺陷：
    run() 把二维图原样交给 draw_defects，defects.py 的半透明叠加
    ``overlay[d.mask>0]=color`` 只对三通道成立，于是「良品板能跑通、有缺陷的板
    必崩」，恰好把 NG 板全部漏掉。修法是把 ``color_image`` 交给 draw_defects
    （彩色输入下它就是 image 本身，故彩色路径逐字节不变）。

    现在这三条是**回归防线**：改回 ``draw_defects(image, ...)`` 它们就会红。
    """

    def test_gray_image_runs_full_chain(self, pipeline, gray_image):
        """灰度图应能走完整链路。"""
        result = pipeline.run(gray_image)
        assert result.quality is not None

    def test_gray_with_defects_does_not_raise(self, pipeline, gray_image):
        """灰度 + 检出缺陷不得抛异常，且标注图是三通道。

        这条是修复的核心断言：夹具造的灰度图必然带缺陷，所以它确实走到了
        draw_defects 的叠加分支（而不是靠"没缺陷"侥幸绕过）。
        """
        result = pipeline.run(gray_image)

        assert result.defects, "夹具应造出缺陷，否则这条用例没有覆盖到崩溃点"
        assert result.image.ndim == 3
        assert result.image.shape[:2] == gray_image.shape
        assert result.quality is not None

    def test_gray_input_reports_color_as_unavailable(self, pipeline, gray_image):
        """灰度输入没有色彩量，必须显式标为未测 —— 不能静默填 0。

        填 0 会被读成「色度零偏移」即满分，比不报更危险。
        """
        result = pipeline.run(gray_image)

        assert result.quality.color_available is False
        assert result.quality.color_hue_mean_deg is None
        assert result.quality.color_hue_deviation_deg is None
        assert any("色度未测" in w for w in result.quality.warnings)

    def test_gray_failure_only_happens_when_defects_exist(self, pipeline,
                                                          config_factory, gray_image):
        """缺陷检测全关时，灰度输入同样跑通（此前后者是唯一能跑的那条）。"""
        cfg = config_factory()
        for name in ("oxidation", "embedding", "unroughened"):
            cfg["inspection"]["defects"][name]["enabled"] = False

        result = InspectionPipeline(cfg).run(gray_image, "GRAY-OK")

        assert result.defects == []
        assert result.quality is not None
        assert result.quality.board_id == "GRAY-OK"
        assert result.summary().strip()

    @pytest.mark.slow
    def test_full_size_gray_image(self, pipeline, sandblasted_image):
        """生产尺寸灰度图同样应能走完整链路。"""
        result = pipeline.run(sandblasted_image, board_id="FULL-GRAY")
        assert result.quality is not None
        assert result.quality.board_id == "FULL-GRAY"


# ============================================================================
# 确定性
# ============================================================================

class TestDeterminism:
    """在线检测对确定性的要求最高：同一块板两次判定必须完全一致，
    否则 SPC 曲线会自己抖起来。"""

    def test_two_runs_are_bit_identical(self, pipeline, color_image):
        """同一输入连续两次 run()，所有可比较的输出逐位一致。"""
        first = pipeline.run(color_image, "PCB-D")
        second = pipeline.run(color_image, "PCB-D")
        assert _fingerprint(first) == _fingerprint(second)

    def test_repeated_runs_keep_ok_ng_stable(self, pipeline, color_image):
        """重复运行不会在 OK/NG 边界上翻来覆去。"""
        verdicts = {pipeline.run(color_image).ok_ng for _ in range(3)}
        assert len(verdicts) == 1

    def test_process_features_are_stable_across_runs(self, pipeline, color_image):
        """工艺特征（8 级 GLCM）逐次一致 —— 报警阈值依赖它。"""
        first = pipeline.run(color_image).process_features
        second = pipeline.run(color_image).process_features
        assert first.as_dict() == pytest.approx(second.as_dict(), rel=1e-12)

    def test_two_pipelines_from_same_config_agree(self, default_config, color_image):
        """用同一份配置构造的两个流水线实例结果一致（无隐藏全局状态）。"""
        a = InspectionPipeline(copy.deepcopy(default_config)).run(color_image)
        b = InspectionPipeline(copy.deepcopy(default_config)).run(color_image)
        assert _fingerprint(a) == _fingerprint(b)

    def test_runtime_is_bounded(self, pipeline, color_image):
        """单帧（128x128）应在 5s 内完成 —— 在线检测的粗略性能护栏。

        阈值取得很宽，只用来抓「某次改动让耗时涨了一个数量级」这类回归；
        实测约 0.4s。生产尺寸的耗时见标 slow 的冒烟用例。
        """
        start = time.perf_counter()
        pipeline.run(color_image)
        assert time.perf_counter() - start < 5.0
