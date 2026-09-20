"""导出层的**列契约**。

`data/export.py` 在改动前是 0% 覆盖。本次加色度字段时往里塞了 7 列，
而它最容易出的错不是崩溃，是**静默错位**：

    下游若有按列号解析 CSV 的脚本，把新列插在中间会让"氧化面积"读到
    "饱和度"的值 —— 不报错，只是数据全错。

所以本文件的核心不是"能导出"，而是：

  1. 前 11 列的**名字与顺序**逐字不变（老列契约）；
  2. 新列一律**追加在末尾**；
  3. 表头与数据**逐列对齐**（Excel 的表头在 `headers` 列表、数据在
     `ws.cell(column=N)` 里各写一遍，两处顺序写反就会错位）。

第 3 条是这里最值钱的一条 —— 两处顺序都是手写的，没有任何机制保证它们一致。

Excel 与 PDF 依赖 openpyxl / fpdf2，本环境两者都没装。Excel 走一个**记录型
假库**来测对齐（真库在的话也跑同一套断言，见 `TestRealOpenpyxl`）。
"""

import collections
import csv
import json
import sys
import types

import pytest

from core.defects import Defect
from core.quality import QualityReport
from data.export import export_csv, export_excel, export_json, export_pdf


#: 色度之前就有的 11 列，**顺序即契约**。
LEGACY_CSV_COLUMNS = (
    "board_id", "timestamp", "overall_score", "ok_ng",
    "roughness_uniformity", "roughness_std",
    "direction_consistency", "oxidation_percentage",
    "embedding_count", "unroughened_percentage",
    "warnings",
)

#: 追加在末尾的 7 列。
COLOR_CSV_COLUMNS = (
    "color_available", "color_hue_mean_deg",
    "color_hue_deviation_deg", "color_sat_mean",
    "color_oor_abs_pct", "color_oor_adaptive_pct",
    "color_oor_count",
)


def _report(**overrides) -> QualityReport:
    """一份默认值明确的报告，便于逐列断言。"""
    base = dict(
        board_id="B-001",
        timestamp="2026-09-20T10:00:00",
        overall_score=82.5,
        ok_ng=True,
        roughness_uniformity=90.0,
        roughness_std=0.1234,
        direction_consistency=0.8123,
        oxidation_percentage=1.25,
        embedding_count=3,
        unroughened_percentage=0.5,
        warnings=["粗糙度偏低"],
        color_available=True,
        color_hue_mean_deg=36.5,
        color_hue_deviation_deg=2.5,
        color_sat_mean=170.25,
        color_oor_abs_pct=1.75,
        color_oor_adaptive_pct=0.5,
        color_oor_count=2,
    )
    base.update(overrides)
    return QualityReport(**base)


@pytest.fixture
def csv_rows(tmp_path):
    """导出 CSV 再读回来，返回 ``(表头, 第一行)``。"""
    def _run(reports):
        path = tmp_path / "report.csv"
        export_csv(reports, str(path))
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        return rows[0], rows[1]
    return _run


# ============================================================================
# CSV 列契约
# ============================================================================

class TestCsvColumnContract:

    def test_header_is_legacy_columns_then_color_columns(self, csv_rows):
        header, _ = csv_rows([_report()])

        assert tuple(header) == LEGACY_CSV_COLUMNS + COLOR_CSV_COLUMNS

    def test_legacy_prefix_is_untouched(self, csv_rows):
        """前 11 列逐字、逐序不变 —— 这是本文件存在的理由。"""
        header, _ = csv_rows([_report()])

        assert tuple(header[:len(LEGACY_CSV_COLUMNS)]) == LEGACY_CSV_COLUMNS

    def test_color_columns_are_appended_not_inserted(self, csv_rows):
        header, _ = csv_rows([_report()])

        assert tuple(header[len(LEGACY_CSV_COLUMNS):]) == COLOR_CSV_COLUMNS

    def test_values_line_up_with_the_header(self, csv_rows):
        """逐列对表：每个色度值都落在自己那一列下面。

        这一条能抓到"改了行没改表头"或反之 —— 那是本文件最容易犯的错。
        """
        header, row = csv_rows([_report()])

        assert len(row) == len(header), "行宽必须与表头一致"
        by_name = dict(zip(header, row))

        assert by_name["color_available"] == "1"
        assert by_name["color_hue_mean_deg"] == "36.5"
        assert by_name["color_hue_deviation_deg"] == "2.5"
        assert by_name["color_sat_mean"] == "170.25"
        assert by_name["color_oor_abs_pct"] == "1.75"
        assert by_name["color_oor_adaptive_pct"] == "0.5"
        assert by_name["color_oor_count"] == "2"

    def test_legacy_values_line_up_too(self, csv_rows):
        """老列的值也不能因为加列而错位。"""
        header, row = csv_rows([_report()])
        by_name = dict(zip(header, row))

        assert by_name["board_id"] == "B-001"
        assert by_name["ok_ng"] == "1"
        assert by_name["oxidation_percentage"] == "1.25"
        assert by_name["embedding_count"] == "3"
        assert by_name["warnings"] == "粗糙度偏低"

    def test_unmeasured_hue_leaves_the_cell_empty(self, csv_rows):
        """未测 → **空单元格**，不是 0。

        写成 0 会被下游读成"色度零偏移"，即满分 —— 与"没测"正好相反。
        """
        header, row = csv_rows([_report(
            color_available=False,
            color_hue_mean_deg=None,
            color_hue_deviation_deg=None,
            color_sat_mean=None,
        )])
        by_name = dict(zip(header, row))

        assert by_name["color_available"] == "0"
        assert by_name["color_hue_mean_deg"] == ""
        assert by_name["color_hue_deviation_deg"] == ""
        assert by_name["color_sat_mean"] == ""

    def test_zero_deviation_is_written_as_zero_not_blank(self, csv_rows):
        """真正的 0.0（实测无偏移）必须写 0，不能空白。"""
        header, row = csv_rows([_report(
            color_hue_deviation_deg=0.0, color_oor_abs_pct=0.0,
            color_oor_count=0,
        )])
        by_name = dict(zip(header, row))

        assert by_name["color_hue_deviation_deg"] == "0.0"
        assert by_name["color_oor_abs_pct"] == "0.0"
        assert by_name["color_oor_count"] == "0"

    def test_header_can_be_suppressed(self, tmp_path):
        path = tmp_path / "no_header.csv"
        export_csv([_report()], str(path), include_header=False)

        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))

        assert len(rows) == 1

    def test_empty_report_list_produces_header_only(self, tmp_path):
        path = tmp_path / "empty.csv"
        export_csv([], str(path))

        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))

        assert len(rows) == 1
        assert tuple(rows[0]) == LEGACY_CSV_COLUMNS + COLOR_CSV_COLUMNS


# ============================================================================
# Excel 列对齐（用记录型假 openpyxl）
# ============================================================================

class _FakeCell:
    def __init__(self, row: int, column: int):
        self.row = row
        self.column = column
        self.value = None
        self.font = None
        self.fill = None
        self.column_letter = _col_letter(column)


class _FakeDim:
    width = 0


class _FakeWorksheet:
    """只记录 ``cell(row, column, value)`` 的调用，不做任何 Excel 的事。"""

    def __init__(self, title: str):
        self.title = title
        self.cells = {}
        # 真 openpyxl 的 column_dimensions 是 DimensionHolder，缺键时自建条目。
        # 用 defaultdict 模拟这一行为 —— `export_excel` 的自动列宽依赖它。
        self.column_dimensions = collections.defaultdict(_FakeDim)

    def cell(self, row, column, value=None):
        key = (row, column)
        cell = self.cells.get(key)
        if cell is None:
            cell = _FakeCell(row, column)
            self.cells[key] = cell
        if value is not None:      # 显式传 None = 留空，保持 None
            cell.value = value
        return cell

    def value_at(self, row: int, column: int):
        cell = self.cells.get((row, column))
        return None if cell is None else cell.value

    def row_values(self, row: int, width: int) -> list:
        return [self.value_at(row, c) for c in range(1, width + 1)]

    def header_values(self) -> list:
        """第 1 行里所有有值的单元格（表头是连续写的，取最大列号即可）。"""
        if not self.cells:
            return []
        width = max(col for _, col in self.cells)
        return self.row_values(1, width)

    @property
    def columns(self):
        """`export_excel` 的自动列宽要遍历它。"""
        if not self.cells:
            return []
        width = max(col for _, col in self.cells)
        out = []
        for col in range(1, width + 1):
            out.append(tuple(
                cell for (r, c), cell in sorted(self.cells.items()) if c == col
            ))
        return out


def _col_letter(index: int) -> str:
    """1 → A，26 → Z，27 → AA。"""
    letters = ""
    while index > 0:
        index, rem = divmod(index - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


class _RecordingWorkbook:
    """假的 ``openpyxl.Workbook``，把建出来的工作表留在 ``instances`` 里。"""

    instances = []

    def __init__(self):
        self._sheets = [_FakeWorksheet("Sheet")]
        self.saved_to = None
        _RecordingWorkbook.instances.append(self)

    @property
    def active(self):
        return self._sheets[0]

    @property
    def worksheets(self):
        return self._sheets

    def create_sheet(self, title):
        ws = _FakeWorksheet(title)
        self._sheets.append(ws)
        return ws

    def save(self, path):
        self.saved_to = path


@pytest.fixture
def fake_openpyxl(monkeypatch):
    """注入一个记录型假 openpyxl。

    刻意无条件覆盖 —— 真库装没装都跑同一套断言，结果可复现。
    真库在的环境另有一条 `TestRealOpenpyxl` 用真库验表头。
    """
    _RecordingWorkbook.instances = []

    module = types.ModuleType("openpyxl")
    module.Workbook = _RecordingWorkbook

    styles = types.ModuleType("openpyxl.styles")

    class _Style:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    styles.Font = _Style
    styles.PatternFill = _Style
    styles.Alignment = _Style
    module.styles = styles

    monkeypatch.setitem(sys.modules, "openpyxl", module)
    monkeypatch.setitem(sys.modules, "openpyxl.styles", styles)
    return _RecordingWorkbook


class TestExcelColumnAlignment:
    """表头与数据是两处手写的列表，顺序写反会静默错位。"""

    @staticmethod
    def _sheet(tmp_path, reports):
        export_excel(reports, output_path=str(tmp_path / "report.xlsx"))
        return _RecordingWorkbook.instances[-1].active

    def test_header_starts_with_the_legacy_eleven(self, fake_openpyxl, tmp_path):
        ws = self._sheet(tmp_path, [_report()])
        header = ws.header_values()

        assert header[:11] == [
            "板号", "时间", "总分", "判定", "粗糙度均匀性",
            "粗糙度标准差", "方向一致性", "氧化面积%",
            "磨料嵌入数", "未粗化面积%", "预警",
        ]

    def test_color_columns_are_appended_after_the_legacy_eleven(
        self, fake_openpyxl, tmp_path
    ):
        ws = self._sheet(tmp_path, [_report()])
        header = ws.header_values()

        assert len(header) == 18
        assert header[11:] == [
            "色度偏移(°)", "色相均值(°)", "饱和度",
            "越界面积%(绝对)", "越界面积%(自适应)", "越界区域数",
            "色度已测",
        ]

    def test_each_color_value_sits_under_its_own_header(
        self, fake_openpyxl, tmp_path
    ):
        """逐列对表 —— 表头顺序与 ``ws.cell(column=N)`` 的顺序必须一致。"""
        ws = self._sheet(tmp_path, [_report()])
        header = ws.header_values()
        row = ws.row_values(2, len(header))

        assert len(row) == len(header)
        by_name = dict(zip(header, row))

        assert by_name["色度偏移(°)"] == 2.5
        assert by_name["色相均值(°)"] == 36.5
        assert by_name["饱和度"] == 170.25
        assert by_name["越界面积%(绝对)"] == 1.75
        assert by_name["越界面积%(自适应)"] == 0.5
        assert by_name["越界区域数"] == 2
        assert by_name["色度已测"] == "是"

    def test_legacy_values_are_still_aligned(self, fake_openpyxl, tmp_path):
        ws = self._sheet(tmp_path, [_report()])
        header = ws.header_values()
        row = ws.row_values(2, len(header))
        by_name = dict(zip(header, row))

        assert by_name["板号"] == "B-001"
        assert by_name["总分"] == 82.5
        assert by_name["判定"] == "OK"
        assert by_name["氧化面积%"] == 1.25
        assert by_name["磨料嵌入数"] == 3

    def test_unmeasured_hue_leaves_cells_empty_not_zero(
        self, fake_openpyxl, tmp_path
    ):
        """未测的三列留空，而不是填 0（0 会被读成"色度零偏移"）。"""
        ws = self._sheet(tmp_path, [_report(
            color_available=False,
            color_hue_mean_deg=None,
            color_hue_deviation_deg=None,
            color_sat_mean=None,
        )])
        header = ws.header_values()
        row = ws.row_values(2, len(header))
        by_name = dict(zip(header, row))

        assert by_name["色度偏移(°)"] is None
        assert by_name["色相均值(°)"] is None
        assert by_name["饱和度"] is None
        assert by_name["色度已测"] == "否"

    def test_zero_deviation_is_written_as_zero(self, fake_openpyxl, tmp_path):
        ws = self._sheet(tmp_path, [_report(color_hue_deviation_deg=0.0)])
        header = ws.header_values()
        by_name = dict(zip(header, ws.row_values(2, len(header))))

        assert by_name["色度偏移(°)"] == 0.0
        assert by_name["色度偏移(°)"] is not None

    def test_defect_sheet_is_separate_and_unaligned_with_the_color_columns(
        self, fake_openpyxl, tmp_path
    ):
        """缺陷工作表有自己的 9 列，不该被色度列影响。"""
        defect = Defect(type="oxidation", mask=None, area_pixels=100,
                        area_mm2=0.01, centroid=(5, 6), bbox=(1, 2, 3, 4),
                        severity=0.5, confidence=0.8)
        export_excel([_report()], defects_list=[[defect]],
                     output_path=str(tmp_path / "report.xlsx"))
        wb = _RecordingWorkbook.instances[-1]
        detail = wb.worksheets[1]

        assert detail.title == "缺陷详情"
        assert detail.header_values() == [
            "板号", "缺陷类型", "面积(mm²)", "严重度",
            "置信度", "X1", "Y1", "X2", "Y2",
        ]
        assert detail.row_values(2, 9) == ["B-001", "oxidation", 0.01, 0.5,
                                          0.8, 1, 2, 3, 4]

    def test_falls_back_to_csv_when_openpyxl_is_missing(
        self, tmp_path, monkeypatch
    ):
        """没有 openpyxl 时回退 CSV —— 回退路径也得带上色度列。"""
        import builtins

        real_import = builtins.__import__

        def _no_openpyxl(name, *args, **kwargs):
            if name.startswith("openpyxl"):
                raise ImportError("模拟未安装 openpyxl")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_openpyxl)
        fallback = tmp_path / "fallback.csv"

        export_excel([_report()], output_path=str(fallback))

        with open(fallback, encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f))
        assert tuple(header) == LEGACY_CSV_COLUMNS + COLOR_CSV_COLUMNS


class TestRealOpenpyxl:
    """真 openpyxl 在的环境里，用真库再验一次表头。

    假库只保证我们自己那套顺序自洽；真库才能证明 ``openpyxl`` 收到的
    列号与表头一致（假库的 ``cell()`` 语义是自己写的）。
    """

    def test_real_workbook_header_matches(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "real.xlsx"

        export_excel([_report()], output_path=str(path))

        wb = openpyxl.load_workbook(str(path))
        ws = wb.active
        header = [c.value for c in ws[1]]
        row = [c.value for c in ws[2]]

        assert header == [
            "板号", "时间", "总分", "判定", "粗糙度均匀性",
            "粗糙度标准差", "方向一致性", "氧化面积%",
            "磨料嵌入数", "未粗化面积%", "预警",
            "色度偏移(°)", "色相均值(°)", "饱和度",
            "越界面积%(绝对)", "越界面积%(自适应)", "越界区域数",
            "色度已测",
        ]
        by_name = dict(zip(header, row))
        assert by_name["色度偏移(°)"] == 2.5
        assert by_name["色相均值(°)"] == 36.5
        assert by_name["氧化面积%"] == 1.25


# ============================================================================
# JSON / PDF
# ============================================================================

class TestJsonExport:

    def test_color_fields_are_present(self, tmp_path):
        path = tmp_path / "report.json"
        export_json([_report()], output_path=str(path))

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["color_available"] is True
        assert data[0]["color_hue_deviation_deg"] == 2.5
        assert data[0]["color_oor_count"] == 2

    def test_unmeasured_hue_is_null_not_zero(self, tmp_path):
        path = tmp_path / "report.json"
        export_json([_report(
            color_available=False,
            color_hue_deviation_deg=None,
            color_hue_mean_deg=None,
            color_sat_mean=None,
        )], output_path=str(path))

        with open(path, encoding="utf-8") as f:
            entry = json.load(f)[0]

        assert entry["color_available"] is False
        assert entry["color_hue_deviation_deg"] is None
        assert entry["color_hue_mean_deg"] is None

    def test_defects_are_attached_per_report(self, tmp_path):
        defect = Defect(type="embedding", mask=None, area_pixels=12,
                        area_mm2=0.0012, centroid=(3, 4), bbox=(1, 1, 5, 5),
                        severity=0.1, confidence=0.8)
        path = tmp_path / "report.json"
        export_json([_report()], defects_list=[[defect]], output_path=str(path))

        with open(path, encoding="utf-8") as f:
            entry = json.load(f)[0]

        assert len(entry["defects"]) == 1
        assert entry["defects"][0]["type"] == "embedding"
        assert entry["defects"][0]["bbox"] == [1, 1, 5, 5]


class TestPdfExport:
    """fpdf2 在本环境未安装，只能测"没装时不崩"这一条。

    ⚠ 色度那几行在 PDF 里的排版**未经测试** —— 这是已知缺口，
       装上 fpdf2 后应补一条断言 PDF 里出现"色度偏移"。
    """

    def test_missing_fpdf_returns_quietly(self, tmp_path):
        path = tmp_path / "report.pdf"

        export_pdf(_report(), output_path=str(path))

        # 没有 fpdf2 时只是打印一行警告，不抛异常、不留下半个文件
        assert not path.exists() or path.stat().st_size == 0
