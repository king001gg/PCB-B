"""SQLite 层的 schema 迁移与读回。

本文件里的绝大部分用例围绕**一个具体的崩溃**展开：

    ``CREATE TABLE IF NOT EXISTS`` 对**已存在**的库是空操作。老部署里的
    ``data/inspection.db`` 没有色度那 7 列，而 ``insert_inspection()`` 是照
    ``record.to_dict()`` 动态拼 INSERT 的 —— 升级后第一次检测必然抛

        sqlite3.OperationalError: table inspections has no column named color_available

光改建表语句只能覆盖全新部署。``_SCHEMA_MIGRATIONS`` + ``_migrate_schema``
就是补老库的那条路，所以它必须被真正测到。

``data/`` 下当前没有 ``.db`` 文件（版本库里不该有），所以旧库只能在临时目录里
按**旧版本的建表语句**合成。合成用的列清单是写死的字面量 —— 不能从
``_SCHEMA_MIGRATIONS`` 反推，那样就变成自证了。
"""

import sqlite3
import types

import pytest

from data.database import _SCHEMA_MIGRATIONS, InspectionDatabase
from data.models import InspectionRecord

# 上一版的建表语句（色度之前）。刻意写死，模拟真实升级场景。
_LEGACY_COLUMNS = (
    "board_id", "overall_score", "ok_ng",
    "roughness_uniformity", "roughness_std", "direction_consistency",
    "oxidation_percentage", "embedding_count", "unroughened_percentage",
    "defect_count", "warnings",
    "image_path", "result_image_path", "heatmap_path", "inspection_time",
)

_LEGACY_SCHEMA = """
CREATE TABLE inspections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    board_id        TEXT DEFAULT '',
    overall_score   REAL DEFAULT 0.0,
    ok_ng           INTEGER DEFAULT 1,
    roughness_uniformity REAL DEFAULT 0.0,
    roughness_std   REAL DEFAULT 0.0,
    direction_consistency REAL DEFAULT 0.0,
    oxidation_percentage REAL DEFAULT 0.0,
    embedding_count INTEGER DEFAULT 0,
    unroughened_percentage REAL DEFAULT 0.0,
    defect_count    INTEGER DEFAULT 0,
    warnings        TEXT DEFAULT '',
    image_path      TEXT DEFAULT '',
    result_image_path TEXT DEFAULT '',
    heatmap_path    TEXT DEFAULT '',
    inspection_time TEXT DEFAULT ''
);
CREATE TABLE defects (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id   INTEGER NOT NULL,
    defect_type     TEXT NOT NULL,
    area_pixels     INTEGER DEFAULT 0,
    area_mm2        REAL DEFAULT 0.0,
    centroid_x      REAL DEFAULT 0.0,
    centroid_y      REAL DEFAULT 0.0,
    bbox_x1         INTEGER DEFAULT 0,
    bbox_y1         INTEGER DEFAULT 0,
    bbox_x2         INTEGER DEFAULT 0,
    bbox_y2         INTEGER DEFAULT 0,
    severity        REAL DEFAULT 0.0,
    confidence      REAL DEFAULT 0.0
);
"""


def _columns_of(path: str) -> set:
    conn = sqlite3.connect(path)
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(inspections)")}
    finally:
        conn.close()


@pytest.fixture
def legacy_db(tmp_path) -> str:
    """一个"升级前"的库，里面已经有一条历史检测记录。"""
    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    conn.executescript(_LEGACY_SCHEMA)
    conn.execute(
        "INSERT INTO inspections "
        "(board_id, overall_score, ok_ng, warnings, inspection_time) "
        "VALUES (?, ?, ?, ?, ?)",
        ("OLD-001", 77.5, 1, "历史记录", "2026-01-01T00:00:00"),
    )
    conn.commit()
    conn.close()
    return path


# ============================================================================
# 迁移
# ============================================================================

class TestSchemaMigration:
    """`_migrate_schema` 必须把老库补齐，且不伤已有数据。"""

    def test_legacy_columns_and_migrations_are_disjoint(self):
        """迁移清单只准追加新列，不准重复声明老列。

        重复会让 ``ALTER TABLE ADD COLUMN`` 在已补齐的库上二次执行时抛错。
        """
        migrated = {column for column, _ in _SCHEMA_MIGRATIONS}

        assert migrated & set(_LEGACY_COLUMNS) == set()

    def test_legacy_database_gets_the_color_columns(self, legacy_db):
        """⚠ 本文件的核心用例：老库被补上 7 列。"""
        before = _columns_of(legacy_db)
        assert set(_LEGACY_COLUMNS) <= before
        assert "color_available" not in before, "前提：合成的库确实没有新列"

        db = InspectionDatabase(legacy_db)
        try:
            after = _columns_of(legacy_db)
        finally:
            db.close()

        for column, _ in _SCHEMA_MIGRATIONS:
            assert column in after, f"迁移后仍缺列 {column}"

    def test_first_migration_reports_the_added_count(self, legacy_db):
        """返回值是"本次补了几列"，可用来打日志。

        绕过 ``__init__``（它已经跑过一次迁移），单独调一次拿到首个返回值。
        """
        db = InspectionDatabase.__new__(InspectionDatabase)
        db.db_path = legacy_db
        db._conn = None
        try:
            assert db._migrate_schema() == len(_SCHEMA_MIGRATIONS)
        finally:
            db.close()

    def test_migration_is_idempotent(self, legacy_db):
        """重复调用无副作用、不抛错、不重复加列。"""
        db = InspectionDatabase(legacy_db)
        try:
            assert db._migrate_schema() == 0
            assert db._migrate_schema() == 0
            assert db._migrate_schema() == 0
        finally:
            db.close()

        assert set(_LEGACY_COLUMNS) | {c for c, _ in _SCHEMA_MIGRATIONS} <= \
               _columns_of(legacy_db)

    def test_fresh_database_needs_no_migration(self, tmp_path):
        """全新库由 CREATE TABLE 一次建全，迁移应当是空转。"""
        db = InspectionDatabase(str(tmp_path / "fresh.db"))
        try:
            assert db._migrate_schema() == 0
        finally:
            db.close()

    def test_legacy_rows_survive_the_migration(self, legacy_db):
        """ALTER TABLE ADD COLUMN 是就地加列，老数据必须原样还在。"""
        db = InspectionDatabase(legacy_db)
        try:
            record = db.get_inspection(1)
        finally:
            db.close()

        assert record is not None
        assert record.board_id == "OLD-001"
        assert record.overall_score == pytest.approx(77.5)
        assert record.ok_ng is True
        assert record.warnings == "历史记录"
        assert record.inspection_time == "2026-01-01T00:00:00"

    def test_legacy_row_color_fields_fall_back_to_defaults(self, legacy_db):
        """老行的新列是 NULL —— 读回来要落到文档规定的默认值，不能崩。

        色相类保持 ``None``（未测），而不是 0 —— 0 会被读成「色度零偏移」。
        """
        db = InspectionDatabase(legacy_db)
        try:
            record = db.get_inspection(1)
        finally:
            db.close()

        assert record.color_available is False
        assert record.color_hue_mean_deg is None
        assert record.color_hue_deviation_deg is None
        assert record.color_sat_mean is None
        assert record.color_oor_abs_pct == 0.0
        assert record.color_oor_adaptive_pct == 0.0
        assert record.color_oor_count == 0

    def test_migration_is_what_makes_the_dynamic_insert_work(self, legacy_db):
        """迁移存在的**全部理由**，用两条相反的断言写清楚。

        先证明老 schema 真的会拒绝新 INSERT（不是假想的风险），
        再证明迁移之后同一条 INSERT 就通过了。
        """
        record = InspectionRecord(board_id="NEW-001", overall_score=88.0)
        data = record.to_dict()
        sql = (
            f"INSERT INTO inspections ({', '.join(data)}) "
            f"VALUES ({', '.join('?' * len(data))})"
        )

        # (1) 未迁移的老库：拒绝
        conn = sqlite3.connect(legacy_db)
        try:
            with pytest.raises(sqlite3.OperationalError, match="color_available"):
                conn.execute(sql, list(data.values()))
        finally:
            conn.close()

        # (2) 迁移之后：接受
        db = InspectionDatabase(legacy_db)
        try:
            new_id = db.insert_inspection(record)
            assert new_id > 0
            assert db.get_inspection(new_id).board_id == "NEW-001"
        finally:
            db.close()

    def test_schema_and_migrations_cover_every_record_field(self, tmp_path):
        """通用不变式：``to_dict()`` 里的每个键都必须有对应的列。

        这条比逐列断言更有价值 —— 它挡的是**将来**新增字段时忘了改建表语句
        的情形，而那正是本次加色度字段踩到的坑。
        """
        db = InspectionDatabase(str(tmp_path / "fresh.db"))
        try:
            columns = {row[1] for row in db._conn.execute(
                "PRAGMA table_info(inspections)"
            )}
        finally:
            db.close()

        missing = set(InspectionRecord().to_dict()) - columns
        assert missing == set(), f"这些字段没有对应的列，插入时会炸：{sorted(missing)}"


# ============================================================================
# 色度字段的读回
# ============================================================================

class TestColorRoundTrip:
    """写入 → 读出必须一致，尤其是 ``None`` 与 ``0.0`` 的区别。"""

    def _round_trip(self, tmp_path, record: InspectionRecord) -> InspectionRecord:
        db = InspectionDatabase(str(tmp_path / "roundtrip.db"))
        try:
            new_id = db.insert_inspection(record)
            return db.get_inspection(new_id)
        finally:
            db.close()

    def test_measured_color_values_survive(self, tmp_path):
        record = InspectionRecord(
            board_id="COLOR-001",
            color_available=True,
            color_hue_mean_deg=41.25,
            color_hue_deviation_deg=7.25,
            color_sat_mean=163.5,
            color_oor_abs_pct=3.125,
            color_oor_adaptive_pct=1.5,
            color_oor_count=4,
        )

        got = self._round_trip(tmp_path, record)

        assert got.color_available is True
        assert got.color_hue_mean_deg == pytest.approx(41.25)
        assert got.color_hue_deviation_deg == pytest.approx(7.25)
        assert got.color_sat_mean == pytest.approx(163.5)
        assert got.color_oor_abs_pct == pytest.approx(3.125)
        assert got.color_oor_adaptive_pct == pytest.approx(1.5)
        assert got.color_oor_count == 4

    def test_unmeasured_hue_stays_none_not_zero(self, tmp_path):
        """灰度输入（未测）落库是 NULL，读回是 ``None``。

        若读回变成 0，上层会显示"色度偏移 0°"，即满分 —— 与"没测"完全相反。
        """
        record = InspectionRecord(board_id="GRAY-001", color_available=False)

        got = self._round_trip(tmp_path, record)

        assert got.color_available is False
        assert got.color_hue_mean_deg is None
        assert got.color_hue_deviation_deg is None
        assert got.color_sat_mean is None

    def test_zero_deviation_is_preserved_as_zero(self, tmp_path):
        """真正的 0.0（实测无偏移）不能被当成"没值"而丢掉。"""
        record = InspectionRecord(
            board_id="PERFECT-001",
            color_available=True,
            color_hue_mean_deg=34.0,
            color_hue_deviation_deg=0.0,
            color_sat_mean=178.0,
            color_oor_abs_pct=0.0,
            color_oor_adaptive_pct=0.0,
            color_oor_count=0,
        )

        got = self._round_trip(tmp_path, record)

        assert got.color_hue_deviation_deg == 0.0
        assert got.color_hue_deviation_deg is not None
        assert got.color_available is True

    def test_oor_count_zero_is_preserved(self, tmp_path):
        """``color_oor_count`` 同理：0 处越界 ≠ 没测。"""
        record = InspectionRecord(
            board_id="CLEAN-001", color_available=True, color_oor_count=0
        )

        got = self._round_trip(tmp_path, record)

        assert got.color_oor_count == 0
        assert got.color_available is True


# ============================================================================
# 从 QualityReport 构造
# ============================================================================

class TestFromQualityReport:
    """``from_quality_report`` 对老 report 对象要向后兼容。"""

    @staticmethod
    def _legacy_report():
        """一个**没有**色度属性的 report —— 模拟反序列化出来的旧对象。"""
        return types.SimpleNamespace(
            board_id="LEGACY-001",
            overall_score=71.0,
            ok_ng=True,
            roughness_uniformity=80.0,
            roughness_std=12.0,
            direction_consistency=0.75,
            oxidation_percentage=1.5,
            embedding_count=3,
            unroughened_percentage=0.5,
            warnings=["粗糙度偏低"],
            timestamp="2026-01-02T03:04:05",
        )

    def test_missing_color_attributes_fall_back(self):
        """老 report 没有 color_* 字段，``getattr`` 回退必须顶住。"""
        record = InspectionRecord.from_quality_report(self._legacy_report())

        assert record.color_available is False
        assert record.color_hue_mean_deg is None
        assert record.color_hue_deviation_deg is None
        assert record.color_sat_mean is None
        assert record.color_oor_abs_pct == 0.0
        assert record.color_oor_adaptive_pct == 0.0
        assert record.color_oor_count == 0

    def test_legacy_report_still_maps_the_original_fields(self):
        """回退逻辑不能影响原有字段的搬运。"""
        record = InspectionRecord.from_quality_report(
            self._legacy_report(), image_path="/tmp/a.png", defect_count=7
        )

        assert record.board_id == "LEGACY-001"
        assert record.overall_score == pytest.approx(71.0)
        assert record.oxidation_percentage == pytest.approx(1.5)
        assert record.embedding_count == 3
        assert record.warnings == "粗糙度偏低"
        assert record.image_path == "/tmp/a.png"
        assert record.defect_count == 7
        assert record.inspection_time == "2026-01-02T03:04:05"

    def test_color_attributes_are_carried_when_present(self):
        report = self._legacy_report()
        report.color_available = True
        report.color_hue_mean_deg = 38.5
        report.color_hue_deviation_deg = 4.5
        report.color_sat_mean = 170.0
        report.color_oor_abs_pct = 2.0
        report.color_oor_adaptive_pct = 0.5
        report.color_oor_count = 2

        record = InspectionRecord.from_quality_report(report)

        assert record.color_available is True
        assert record.color_hue_mean_deg == pytest.approx(38.5)
        assert record.color_hue_deviation_deg == pytest.approx(4.5)
        assert record.color_sat_mean == pytest.approx(170.0)
        assert record.color_oor_abs_pct == pytest.approx(2.0)
        assert record.color_oor_adaptive_pct == pytest.approx(0.5)
        assert record.color_oor_count == 2
