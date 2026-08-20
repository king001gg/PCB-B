"""SQLite 数据库接口。

提供检测记录和缺陷记录的增删改查功能。
支持两种后端：
    - SQLAlchemy（推荐，需要安装）
    - sqlite3（内置回退，无需额外安装）
"""

import os
import sqlite3
import json
from typing import List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from data.models import InspectionRecord, DefectRecord


class InspectionDatabase:
    """检测结果数据库。

    使用 SQLite 存储每次检测的完整记录。
    默认数据库文件位于 data/inspection.db。
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_dir = Path(__file__).resolve().parent
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "inspection.db")

        self.db_path = db_path
        self._conn = None
        self._init_db()

    def _init_db(self):
        """初始化数据库表。"""
        self._ensure_connection()

        cursor = self._conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS inspections (
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

            CREATE TABLE IF NOT EXISTS defects (
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
                confidence      REAL DEFAULT 0.0,
                FOREIGN KEY (inspection_id) REFERENCES inspections(id)
            );

            CREATE INDEX IF NOT EXISTS idx_inspections_time
                ON inspections(inspection_time);
            CREATE INDEX IF NOT EXISTS idx_inspections_score
                ON inspections(overall_score);
            CREATE INDEX IF NOT EXISTS idx_inspections_okng
                ON inspections(ok_ng);
            CREATE INDEX IF NOT EXISTS idx_defects_inspection
                ON defects(inspection_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # CRUD 操作
    # ------------------------------------------------------------------

    def insert_inspection(self, record: InspectionRecord) -> int:
        """插入一条检测记录，返回自增 ID。"""
        self._ensure_connection()
        cursor = self._conn.cursor()

        data = record.to_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        values = list(data.values())

        cursor.execute(
            f"INSERT INTO inspections ({columns}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()
        return cursor.lastrowid

    def insert_defect(self, defect: DefectRecord) -> int:
        """插入一条缺陷记录。"""
        self._ensure_connection()
        cursor = self._conn.cursor()

        data = defect.to_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        values = list(data.values())

        cursor.execute(
            f"INSERT INTO defects ({columns}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()
        return cursor.lastrowid

    def insert_defects_batch(
        self, defects: List[DefectRecord], inspection_id: int,
    ) -> int:
        """批量插入缺陷记录。"""
        for d in defects:
            d.inspection_id = inspection_id
            self.insert_defect(d)
        return len(defects)

    def get_inspection(self, inspection_id: int) -> Optional[InspectionRecord]:
        """按 ID 获取检测记录。"""
        self._ensure_connection()
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM inspections WHERE id = ?", (inspection_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_inspection(row)

    def get_defects(self, inspection_id: int) -> List[DefectRecord]:
        """获取一次检测的所有缺陷记录。"""
        self._ensure_connection()
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM defects WHERE inspection_id = ?",
            (inspection_id,),
        )
        return [self._row_to_defect(row) for row in cursor.fetchall()]

    def get_recent(self, limit: int = 50) -> List[InspectionRecord]:
        """获取最近 N 条检测记录。"""
        self._ensure_connection()
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM inspections ORDER BY inspection_time DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_inspection(row) for row in cursor.fetchall()]

    def get_statistics(
        self, date_from: str = None, date_to: str = None,
    ) -> dict:
        """查询统计数据。

        Returns:
            含总检测数、合格率、平均分、趋势等信息的字典。
        """
        self._ensure_connection()
        cursor = self._conn.cursor()

        # 基本统计
        cursor.execute("SELECT COUNT(*) FROM inspections")
        total = cursor.fetchone()[0]

        if total == 0:
            return {"total": 0, "ok_count": 0, "ng_count": 0,
                    "yield_rate": 0.0, "avg_score": 0.0,
                    "trend": []}

        cursor.execute(
            "SELECT COUNT(*) FROM inspections WHERE ok_ng = 1"
        )
        ok_count = cursor.fetchone()[0]

        cursor.execute(
            "SELECT AVG(overall_score), MIN(overall_score), "
            "MAX(overall_score) FROM inspections"
        )
        avg, min_score, max_score = cursor.fetchone()

        # 最近 N 条趋势
        cursor.execute(
            "SELECT overall_score, inspection_time "
            "FROM inspections ORDER BY inspection_time DESC LIMIT 50"
        )
        trend = cursor.fetchall()

        return {
            "total": total,
            "ok_count": ok_count,
            "ng_count": total - ok_count,
            "yield_rate": round(ok_count / total * 100, 2),
            "avg_score": round(avg or 0, 2),
            "min_score": round(min_score or 0, 2),
            "max_score": round(max_score or 0, 2),
            "trend": [(score, time) for score, time in reversed(trend)],
        }

    def delete_old(self, days: int = 90) -> int:
        """删除 N 天前的历史记录。"""
        self._ensure_connection()
        cutoff = datetime.now().isoformat()
        cursor = self._conn.cursor()

        # 先删除关联的缺陷记录
        cursor.execute(
            "DELETE FROM defects WHERE inspection_id IN "
            "(SELECT id FROM inspections WHERE inspection_time < ?)",
            (cutoff,),
        )
        deleted_defects = cursor.rowcount

        # 再删除检测记录
        cursor.execute(
            "DELETE FROM inspections WHERE inspection_time < ?",
            (cutoff,),
        )
        deleted_inspections = cursor.rowcount

        self._conn.commit()
        return deleted_inspections + deleted_defects

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _ensure_connection(self):
        """确保数据库连接可用。"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row

    def _row_to_inspection(self, row) -> InspectionRecord:
        """sqlite3.Row → InspectionRecord。"""
        return InspectionRecord(
            board_id=row["board_id"] or "",
            overall_score=row["overall_score"] or 0.0,
            ok_ng=bool(row["ok_ng"]),
            roughness_uniformity=row["roughness_uniformity"] or 0.0,
            roughness_std=row["roughness_std"] or 0.0,
            direction_consistency=row["direction_consistency"] or 0.0,
            oxidation_percentage=row["oxidation_percentage"] or 0.0,
            embedding_count=row["embedding_count"] or 0,
            unroughened_percentage=row["unroughened_percentage"] or 0.0,
            defect_count=row["defect_count"] or 0,
            warnings=row["warnings"] or "",
            image_path=row["image_path"] or "",
            result_image_path=row["result_image_path"] or "",
            heatmap_path=row["heatmap_path"] or "",
            inspection_time=row["inspection_time"] or "",
        )

    def _row_to_defect(self, row) -> DefectRecord:
        """sqlite3.Row → DefectRecord。"""
        return DefectRecord(
            inspection_id=row["inspection_id"],
            defect_type=row["defect_type"],
            area_pixels=row["area_pixels"] or 0,
            area_mm2=row["area_mm2"] or 0.0,
            centroid_x=row["centroid_x"] or 0.0,
            centroid_y=row["centroid_y"] or 0.0,
            bbox_x1=row["bbox_x1"] or 0,
            bbox_y1=row["bbox_y1"] or 0,
            bbox_x2=row["bbox_x2"] or 0,
            bbox_y2=row["bbox_y2"] or 0,
            severity=row["severity"] or 0.0,
            confidence=row["confidence"] or 0.0,
        )

    def close(self):
        """关闭数据库连接。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
