"""core/reporter.py 单元测试。

覆盖 InspectionReport 的四种输出（字典 / JSON / 文本 / SPC 统计）以及在
磁盘上真正落盘的形态。所有写盘动作一律落在 tmp_report_dir，不碰 results/。

输入数据全部用构造出来的 InspectionResult / QualityReport，不跑流水线，
因此用例确定性、无副作用、毫秒级。
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from core.defects import Defect
from core.pipeline import InspectionResult
from core.quality import QualityReport
from core.reporter import InspectionReport


# ============================================================================
# 夹具与小工具
# ============================================================================

def _defect(
    dtype: str = "oxidation",
    area_mm2: float = 1.23456789,
    severity: float = 0.5,
    confidence: float = 0.91234,
    centroid=(3, 4),
    bbox=(1, 2, 5, 6),
):
    """造一个字段可控的缺陷对象。"""
    return Defect(
        type=dtype,
        mask=np.zeros((16, 16), dtype=np.uint8),
        area_pixels=100,
        area_mm2=area_mm2,
        centroid=centroid,
        bbox=bbox,
        severity=severity,
        confidence=confidence,
    )


@pytest.fixture
def report_factory():
    """按需组装 InspectionReport。

    用法::

        rep = report_factory(defects=[_defect()])
        rep = report_factory(quality=None)
        rep = report_factory(quality_overrides={"overall_score": 88.5})
    """

    def _build(
        defects=None,
        quality="default",
        quality_overrides: dict = None,
        roughness_map=None,
        timings: dict = None,
        total_time_ms: float = 12.3456,
        ok_ng: bool = True,
    ) -> InspectionReport:
        if quality == "default":
            quality = QualityReport(
                overall_score=88.5, board_id="PCB-0001", ok_ng=ok_ng
            )
        if quality is not None and quality_overrides:
            for key, value in quality_overrides.items():
                setattr(quality, key, value)

        result = InspectionResult(
            defects=list(defects or []),
            quality=quality,
            ok_ng=ok_ng,
            roughness_map=roughness_map,
            timings=dict(timings or {}),
            total_time_ms=total_time_ms,
        )
        return InspectionReport(result)

    return _build


def _strict_loads(text: str):
    """严格按 RFC 8259 解析：拒绝 NaN / Infinity 这类非标准字面量。"""

    def _reject(token: str):
        raise ValueError(f"JSON 规范不允许的字面量: {token}")

    return json.loads(text, parse_constant=_reject)


def _write_json(report: InspectionReport, path: Path) -> dict:
    """把报告写成 JSON 文件并读回，返回解析后的字典。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# to_dict
# ============================================================================

class TestToDict:
    """结构化字典输出。"""

    def test_empty_report_has_zero_defects(self, report_factory):
        """没有缺陷时：计数为 0、分类统计为空、明细为空列表。"""
        data = report_factory().to_dict()
        assert data["defects"]["total_count"] == 0
        assert data["defects"]["by_type"] == {}
        assert data["defects"]["items"] == []

    def test_report_section_fields(self, report_factory):
        """report 节携带板号、判定、总分与总耗时。"""
        data = report_factory().to_dict()
        assert data["report"]["board_id"] == "PCB-0001"
        assert data["report"]["ok_ng"] is True
        assert data["report"]["overall_score"] == 88.5
        assert data["report"]["total_time_ms"] == 12.35  # round(..., 2)

    def test_timestamp_is_iso(self, report_factory):
        """时间戳为合法 ISO 格式。"""
        from datetime import datetime

        datetime.fromisoformat(report_factory().to_dict()["report"]["timestamp"])

    def test_single_defect_item_is_fully_serialized(self, report_factory):
        """单条缺陷的全部字段都被写出，并按约定精度取整。"""
        rep = report_factory(defects=[_defect()])
        item = rep.to_dict()["defects"]["items"][0]
        assert item["type"] == "oxidation"
        assert item["area_mm2"] == 1.2346          # round(..., 4)
        assert item["severity"] == 0.5
        assert item["confidence"] == 0.9123
        assert item["centroid"] == [3, 4]          # 元组转列表
        assert item["bbox"] == [1, 2, 5, 6]

    def test_multiple_defects_by_type_counting(self, report_factory):
        """by_type 按缺陷类型分别计数。"""
        rep = report_factory(
            defects=[
                _defect("oxidation"),
                _defect("oxidation"),
                _defect("embedding"),
                _defect("unroughened"),
            ]
        )
        data = rep.to_dict()
        assert data["defects"]["total_count"] == 4
        assert data["defects"]["by_type"] == {
            "oxidation": 2, "embedding": 1, "unroughened": 1,
        }
        assert len(data["defects"]["items"]) == 4

    def test_quality_section_matches_quality_report(self, report_factory):
        """quality 节就是 QualityReport.to_dict() 的内容。"""
        rep = report_factory()
        assert rep.to_dict()["quality"] == rep.quality.to_dict()

    def test_performance_defaults_to_zero_for_missing_timings(self, report_factory):
        """timings 缺项时按 0 处理，不抛 KeyError。"""
        perf = report_factory(timings={}).to_dict()["performance"]
        assert perf == {
            "preprocessing_ms": 0, "texture_ms": 0, "defects_ms": 0,
            "quality_ms": 0, "total_ms": 12.3,
        }

    def test_performance_rounds_to_one_decimal(self, report_factory):
        """各阶段耗时保留一位小数。"""
        perf = report_factory(
            timings={"preprocessing": 1.26, "texture": 2.0,
                     "defects": 3.94, "quality": 0.05},
            total_time_ms=7.25,
        ).to_dict()["performance"]
        assert perf["preprocessing_ms"] == 1.3
        assert perf["defects_ms"] == 3.9
        assert perf["total_ms"] == 7.2

    def test_roughness_section_absent_without_map(self, report_factory):
        """没有粗糙度图时不输出 roughness 节。"""
        assert "roughness" not in report_factory().to_dict()

    def test_roughness_section_stats(self, report_factory):
        """有粗糙度图时输出均值/标准差/最大值。"""
        rmap = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        rough = report_factory(roughness_map=rmap).to_dict()["roughness"]
        assert rough["mean_cv"] == pytest.approx(0.25, abs=1e-4)
        assert rough["std_cv"] == pytest.approx(float(np.std(rmap)), abs=1e-4)
        assert rough["max_cv"] == pytest.approx(0.4, abs=1e-4)

    def test_missing_quality_is_handled(self, report_factory):
        """quality 为 None 时字典输出仍能生成（空板号 / 0 分 / 空 quality 节）。"""
        data = report_factory(quality=None).to_dict()
        assert data["report"]["board_id"] == ""
        assert data["report"]["overall_score"] == 0
        assert data["quality"] == {}

    def test_ok_ng_reflects_result_not_quality(self, report_factory):
        """report.ok_ng 取的是 result.ok_ng（判定结果），不是质量报告里的字段。"""
        rep = report_factory(ok_ng=False)
        rep.result.ok_ng = False
        assert rep.to_dict()["report"]["ok_ng"] is False

    def test_does_not_mutate_inputs(self, report_factory):
        """生成报告不应改动原始缺陷与质量对象。"""
        defect = _defect()
        rep = report_factory(defects=[defect])
        before = rep.quality.to_dict()
        rep.to_dict()
        assert rep.quality.to_dict() == before
        assert rep.result.defects == [defect]
        assert defect.centroid == (3, 4)  # 仍是元组，未被就地改写成列表


# ============================================================================
# to_json
# ============================================================================

class TestToJson:
    """JSON 字符串输出。"""

    def test_json_parses_and_equals_dict(self, report_factory):
        """JSON 文本可被 json.loads 解析，且与 to_dict() 等价。"""
        rep = report_factory(defects=[_defect()])
        assert json.loads(rep.to_json()) == rep.to_dict()

    def test_chinese_is_not_escaped(self, report_factory):
        """ensure_ascii=False：中文板号在文本里保持原字符，便于人眼核对。"""
        rep = report_factory(quality_overrides={"board_id": "板卡-甲-01"})
        text = rep.to_json()
        assert "板卡-甲-01" in text
        assert json.loads(text)["report"]["board_id"] == "板卡-甲-01"

    def test_indent_is_respected(self, report_factory):
        """indent 参数生效：None 是单行紧凑输出，数字则按缩进宽度排版。"""
        rep = report_factory()
        compact = rep.to_json(indent=None)
        wide = rep.to_json(indent=4)
        narrow = rep.to_json(indent=0)

        assert "\n" not in compact
        assert wide.count("\n") == narrow.count("\n")  # 换行位置只由结构决定
        assert wide.count("\n") > 0
        assert len(wide) > len(narrow)  # 缩进宽度体现在行长上
        # 三种排版解析结果必须完全相同
        assert json.loads(compact) == json.loads(wide) == json.loads(narrow) == rep.to_dict()

    def test_numbers_are_json_numbers_not_strings(self, report_factory):
        """数值字段是 JSON number，不是被引号包起来的字符串。"""
        data = json.loads(report_factory().to_json())
        assert isinstance(data["report"]["overall_score"], float)
        assert isinstance(data["report"]["total_time_ms"], float)
        assert isinstance(data["defects"]["total_count"], int)

    def test_float32_roughness_map_is_serializable(self, report_factory):
        """float32 粗糙度图经 float() 转换后可序列化（不残留 numpy 标量）。"""
        rmap = np.linspace(0.01, 0.4, 16, dtype=np.float32).reshape(4, 4)
        text = report_factory(roughness_map=rmap).to_json()
        assert json.loads(text)["roughness"]["max_cv"] == pytest.approx(0.4, abs=1e-3)

    def test_long_board_id_round_trips(self, report_factory):
        """超长字段（5000 字符）可正常序列化并原样还原。"""
        long_id = "板-" + "X" * 5000
        rep = report_factory(quality_overrides={"board_id": long_id})
        assert json.loads(rep.to_json())["report"]["board_id"] == long_id

    def test_special_characters_round_trip(self, report_factory):
        """引号、反斜杠、换行、制表符、emoji 等特殊字符原样往返。"""
        tricky = 'A"B\\C\nD\tE\rF\x00G ✓✗⚠ 板卡'
        rep = report_factory(quality_overrides={"board_id": tricky})
        assert json.loads(rep.to_json())["report"]["board_id"] == tricky

    def test_nan_score_is_rejected_by_strict_json(self, report_factory):
        """见同文件的 xfail 用例说明：本用例固定当前（不合规）行为。

        json.dumps 默认 allow_nan=True，会写出裸 NaN 字面量；Python 的
        json.loads 能读回来，但严格解析器（JS / Excel / 下游 SPC 工具）会拒收。
        """
        rep = report_factory(quality_overrides={"overall_score": float("nan")})
        text = rep.to_json()
        assert "NaN" in text
        assert json.loads(text)["report"]["overall_score"] != \
            json.loads(text)["report"]["overall_score"]  # NaN != NaN

    @pytest.mark.xfail(
        reason="to_json 未把 NaN/Infinity 归一化，quality.overall_score 为 NaN 时"
               "产出不符合 RFC 8259 的裸 NaN 字面量，严格 JSON 解析器会拒收",
        strict=False,
    )
    def test_nan_score_produces_strictly_valid_json(self, report_factory):
        """评分异常时也应产出合法 JSON，而不是裸 NaN。"""
        rep = report_factory(quality_overrides={"overall_score": float("nan")})
        _strict_loads(rep.to_json())


# ============================================================================
# to_text
# ============================================================================

class TestToText:
    """文本报告输出。"""

    def test_contains_header_and_identity(self, report_factory):
        """文本含标题、板号、时间与判定。"""
        text = report_factory().to_text()
        assert "PCB 喷砂表面质量检测报告" in text
        assert "PCB-0001" in text
        assert "88.5" in text
        assert "12093" not in text  # 耗时按毫秒整数展示
        assert "12 ms" in text

    def test_ok_and_ng_wording(self, report_factory):
        """OK / NG 两种判定文案不同。"""
        assert "OK" in report_factory(ok_ng=True).to_text()
        assert "NG" in report_factory(ok_ng=False).to_text()

    def test_empty_defect_list_is_explicit(self, report_factory):
        """无缺陷时明确写「(无缺陷)」。"""
        text = report_factory().to_text()
        assert "(无缺陷)" in text
        assert "--- 缺陷列表 (0 处) ---" in text

    def test_defects_are_listed_with_index_and_geometry(self, report_factory):
        """每条缺陷单独一行，带序号、类型、面积与包围盒坐标。"""
        rep = report_factory(
            defects=[_defect("oxidation", bbox=(1, 2, 5, 6)),
                     _defect("embedding", bbox=(7, 8, 9, 10))]
        )
        text = rep.to_text()
        assert "--- 缺陷列表 (2 处) ---" in text
        assert "1. [oxidation]" in text
        assert "2. [embedding]" in text
        assert "(1,2)→(5,6)" in text

    def test_warnings_section_appears_only_when_needed(self, report_factory):
        """有预警才输出预警段。"""
        assert "--- 预警 ---" not in report_factory().to_text()
        rep = report_factory(
            quality_overrides={"warnings": ["氧化斑面积超标: 20.0%"]}
        )
        text = rep.to_text()
        assert "--- 预警 ---" in text
        assert "氧化斑面积超标: 20.0%" in text

    def test_quality_metrics_are_rendered(self, report_factory):
        """五项质量指标都出现在文本里。"""
        rep = report_factory(
            quality_overrides={
                "roughness_uniformity": 77.7,
                "roughness_std": 0.1234,
                "direction_consistency": 0.4321,
                "oxidation_percentage": 2.5,
                "embedding_count": 3,
                "unroughened_percentage": 1.25,
            }
        )
        text = rep.to_text()
        assert "77.7/100" in text
        assert "0.1234" in text
        assert "2.50%" in text
        assert "3 个" in text
        assert "1.25%" in text

    def test_text_is_multiline_and_terminated(self, report_factory):
        """文本为多行，并以分隔线收尾。"""
        text = report_factory().to_text()
        assert text.count("\n") >= 10
        assert text.rstrip().endswith("=" * 60)

    @pytest.mark.xfail(
        reason="to_text 直接访问 q.overall_score / q.roughness_uniformity 等字段，"
               "quality 为 None 时抛 AttributeError；同一份结果走 to_dict/to_json 却是 "
               "None 安全的，两条出口行为不一致",
        strict=False,
    )
    def test_text_output_survives_missing_quality(self, report_factory):
        """没有质量报告时也应能出文本（与 to_dict 的 None 容忍度对齐）。"""
        assert report_factory(quality=None).to_text()


# ============================================================================
# 落盘
# ============================================================================

class TestFileOutput:
    """报告写出到磁盘（一律写进 tmp_report_dir）。"""

    def test_json_file_round_trip(self, report_factory, tmp_report_dir: Path):
        """写出的 JSON 文件能被重新解析，内容与内存中的报告一致。"""
        rep = report_factory(defects=[_defect(), _defect("embedding")])
        target = tmp_report_dir / "report.json"
        data = _write_json(rep, target)
        assert target.is_file()
        assert data == rep.to_dict()
        assert data["defects"]["total_count"] == 2

    def test_repeated_writes_overwrite_cleanly(self, report_factory, tmp_report_dir: Path):
        """同一路径重复写出应整体覆盖，不残留上一份的内容。"""
        target = tmp_report_dir / "report.json"
        _write_json(report_factory(defects=[_defect()]), target)
        data = _write_json(report_factory(), target)
        assert data["defects"]["total_count"] == 0

    def test_text_file_is_utf8_readable(self, report_factory, tmp_report_dir: Path):
        """文本报告按 UTF-8 写出，中文可原样读回。"""
        rep = report_factory()
        target = tmp_report_dir / "report.txt"
        target.write_text(rep.to_text(), encoding="utf-8")
        assert target.read_text(encoding="utf-8") == rep.to_text()
        assert "PCB-0001" in target.read_text(encoding="utf-8")

    def test_chinese_directory_path(self, report_factory, tmp_report_dir: Path):
        """中文目录 + 中文文件名可正常写出与读回。"""
        target = tmp_report_dir / "喷砂检测报告" / "第一块板.json"
        data = _write_json(report_factory(), target)
        assert target.is_file()
        assert data["report"]["board_id"] == "PCB-0001"

    def test_special_characters_survive_disk_round_trip(
        self, report_factory, tmp_report_dir: Path
    ):
        """特殊字符经文件往返后完全一致（不是只在内存里对）。"""
        tricky = 'A"B\\C\nD ✓ 板卡\U0001F600'
        rep = report_factory(quality_overrides={"board_id": tricky})
        data = _write_json(rep, tmp_report_dir / "tricky.json")
        assert data["report"]["board_id"] == tricky

    def test_very_large_report_is_written(self, report_factory, tmp_report_dir: Path):
        """上千条缺陷的报告能正常写出并读回，条目数不丢。"""
        defects = [_defect("oxidation") for _ in range(1000)]
        defects += [_defect("embedding") for _ in range(200)]
        data = _write_json(report_factory(defects=defects),
                           tmp_report_dir / "big.json")
        assert data["defects"]["total_count"] == 1200
        assert data["defects"]["by_type"] == {"oxidation": 1000, "embedding": 200}

    def test_report_is_not_written_into_results_dir(self, report_factory, tmp_report_dir: Path):
        """测试只落在临时目录：report_dir 必须是 tmp_path 下的路径。"""
        target = tmp_report_dir / "report.json"
        _write_json(report_factory(), target)
        assert "results" not in str(target.parent).replace("\\", "/").split("/")

    def test_flat_metrics_can_be_exported_as_csv(
        self, report_factory, tmp_report_dir: Path
    ):
        """报告可摊平成「指标,值」两列的 CSV（core/reporter 自身不含 CSV 出口，
        这里验证的是 to_dict 的扁平化数据足以喂给下游导出，列数严格一致）。"""
        data = report_factory().to_dict()
        rows = list(data["report"].items()) + list(data["quality"].items())
        target = tmp_report_dir / "metrics.csv"
        with open(target, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["指标", "值"])
            for key, value in rows:
                writer.writerow([key, value])

        with open(target, newline="", encoding="utf-8-sig") as f:
            parsed = list(csv.reader(f))
        assert len(parsed[0]) == 2
        assert all(len(row) == 2 for row in parsed)
        assert parsed[0] == ["指标", "值"]
        assert dict(parsed[1:])["board_id"] == "PCB-0001"
        assert len(parsed) == len(rows) + 1


# ============================================================================
# compute_spc
# ============================================================================

class TestComputeSPC:
    """SPC 统计。"""

    def test_insufficient_samples(self, report_factory):
        """少于 2 个数据点时返回 error，不抛异常。"""
        rep = report_factory()
        assert rep.compute_spc([]) == {"error": "样本不足（需要 ≥ 2 个数据点）"}
        assert "error" in rep.compute_spc([88.5])
        assert "error" in rep.compute_spc([88.5, 90.0][:1])

    def test_exactly_two_samples(self, report_factory):
        """2 个数据点即可出结果，均值/标准差按样本口径（ddof=1）。"""
        spc = report_factory().compute_spc([90.0, 92.0])
        assert spc["n"] == 2
        assert spc["mean"] == 91.0
        assert spc["std"] == 1.41
        assert spc["ucl"] == 95.24
        assert spc["lcl"] == 86.76
        assert spc["uwl"] == 93.83
        assert spc["lwl"] == 88.17
        assert spc["cp"] == 4.71
        assert spc["cpk"] == 2.12

    def test_control_limits_are_clamped_to_score_range(self, report_factory):
        """控制限被夹在 [0, 100]：均值接近满分时 ucl 不会超过 100。"""
        high = report_factory().compute_spc([99.0, 100.0])
        assert high["ucl"] == 100
        low = report_factory().compute_spc([0.0, 1.0])
        assert low["lcl"] == 0

    def test_zero_variance_gives_extreme_capability(self, report_factory):
        """标准差为 0 时 Cp/Cpk 记为 999（避免除零）。"""
        spc = report_factory().compute_spc([80.0, 80.0, 80.0])
        assert spc["std"] == 0.0
        assert spc["cp"] == 999.0
        assert spc["cpk"] == 999.0
        assert spc["ucl"] == 80.0
        assert spc["lcl"] == 80.0

    def test_out_of_control_flag(self, report_factory):
        """当前值落在 3σ 之外时 out_of_control 为 True。"""
        rep = report_factory()  # current = 88.5
        out = rep.compute_spc([96.0, 98.0])
        assert out["out_of_control"] is True
        assert out["warning"] is False

    def test_warning_flag_between_two_and_three_sigma(self, report_factory):
        """当前值落在 2σ~3σ 之间时 warning 为 True 而 out_of_control 为 False。"""
        warn = report_factory().compute_spc([84.0, 86.0])
        assert warn["warning"] is True
        assert warn["out_of_control"] is False

    def test_in_control_reports_neither_flag(self, report_factory):
        """当前值在 2σ 以内时两个标志都为 False。"""
        ok = report_factory().compute_spc([90.0, 92.0])
        assert ok["out_of_control"] is False
        assert ok["warning"] is False

    def test_current_comes_from_quality_score(self, report_factory):
        """current 取当前报告的质量评分，不受历史样本影响。"""
        rep = report_factory(quality_overrides={"overall_score": 72.0})
        assert rep.compute_spc([90.0, 92.0])["current"] == 72.0

    def test_current_is_zero_without_quality(self, report_factory):
        """没有质量报告时 current 记 0，而不是抛异常。"""
        assert report_factory(quality=None).compute_spc([90.0, 92.0])["current"] == 0

    def test_n_matches_input_length(self, report_factory):
        """n 为输入样本数（含当前值）。"""
        scores = [70.0, 75.5, 80.0, 88.5, 91.0]
        spc = report_factory().compute_spc(scores)
        assert spc["n"] == len(scores)
        assert spc["mean"] == pytest.approx(float(np.mean(scores)), abs=0.01)

    def test_all_returned_values_are_finite(self, report_factory):
        """所有数值字段都是有限值，不含 nan/inf。"""
        spc = report_factory().compute_spc([0.0, 100.0])
        for key, value in spc.items():
            if isinstance(value, float):
                assert np.isfinite(value), f"{key} 不是有限值: {value}"

    def test_mixed_ok_ng_history(self, report_factory):
        """含 OK/NG 混合历史时结果仍稳定。"""
        spc = report_factory().compute_spc([61.0, 62.0, 95.0, 96.0])
        assert spc["lcl"] >= 0
        assert spc["ucl"] <= 100
        assert spc["cpk"] < 1.0  # 历史波动远大于规格带宽
