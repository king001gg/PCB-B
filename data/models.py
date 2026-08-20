"""数据模型定义。

使用 SQLAlchemy ORM 定义检测记录和缺陷记录的数据库表结构。
支持 SQLite（单机部署）和 PostgreSQL（企业部署）。
"""

from datetime import datetime
from typing import Optional


class InspectionRecord:
    """检测记录数据模型（纯 Python，兼容 SQLAlchemy 和纯 SQL）。"""

    def __init__(
        self,
        board_id: str = "",
        overall_score: float = 0.0,
        ok_ng: bool = True,
        roughness_uniformity: float = 0.0,
        roughness_std: float = 0.0,
        direction_consistency: float = 0.0,
        oxidation_percentage: float = 0.0,
        embedding_count: int = 0,
        unroughened_percentage: float = 0.0,
        defect_count: int = 0,
        warnings: str = "",
        image_path: str = "",
        result_image_path: str = "",
        heatmap_path: str = "",
        inspection_time: str = None,
    ):
        self.board_id = board_id
        self.overall_score = overall_score
        self.ok_ng = ok_ng
        self.roughness_uniformity = roughness_uniformity
        self.roughness_std = roughness_std
        self.direction_consistency = direction_consistency
        self.oxidation_percentage = oxidation_percentage
        self.embedding_count = embedding_count
        self.unroughened_percentage = unroughened_percentage
        self.defect_count = defect_count
        self.warnings = warnings
        self.image_path = image_path
        self.result_image_path = result_image_path
        self.heatmap_path = heatmap_path
        self.inspection_time = inspection_time or datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "board_id": self.board_id,
            "overall_score": self.overall_score,
            "ok_ng": self.ok_ng,
            "roughness_uniformity": self.roughness_uniformity,
            "roughness_std": self.roughness_std,
            "direction_consistency": self.direction_consistency,
            "oxidation_percentage": self.oxidation_percentage,
            "embedding_count": self.embedding_count,
            "unroughened_percentage": self.unroughened_percentage,
            "defect_count": self.defect_count,
            "warnings": self.warnings,
            "image_path": self.image_path,
            "result_image_path": self.result_image_path,
            "heatmap_path": self.heatmap_path,
            "inspection_time": self.inspection_time,
        }

    @classmethod
    def from_quality_report(cls, report, image_path="",
                            result_image_path="", heatmap_path="",
                            defect_count=0) -> "InspectionRecord":
        """从 QualityReport 创建记录。"""
        return cls(
            board_id=report.board_id,
            overall_score=report.overall_score,
            ok_ng=report.ok_ng,
            roughness_uniformity=report.roughness_uniformity,
            roughness_std=report.roughness_std,
            direction_consistency=report.direction_consistency,
            oxidation_percentage=report.oxidation_percentage,
            embedding_count=report.embedding_count,
            unroughened_percentage=report.unroughened_percentage,
            defect_count=defect_count,
            warnings="; ".join(report.warnings) if report.warnings else "",
            image_path=image_path,
            result_image_path=result_image_path,
            heatmap_path=heatmap_path,
            inspection_time=report.timestamp,
        )


class DefectRecord:
    """单项缺陷记录。"""

    def __init__(
        self,
        inspection_id: int = 0,
        defect_type: str = "",
        area_pixels: int = 0,
        area_mm2: float = 0.0,
        centroid_x: float = 0.0,
        centroid_y: float = 0.0,
        bbox_x1: int = 0,
        bbox_y1: int = 0,
        bbox_x2: int = 0,
        bbox_y2: int = 0,
        severity: float = 0.0,
        confidence: float = 0.0,
    ):
        self.inspection_id = inspection_id
        self.defect_type = defect_type
        self.area_pixels = area_pixels
        self.area_mm2 = area_mm2
        self.centroid_x = centroid_x
        self.centroid_y = centroid_y
        self.bbox_x1 = bbox_x1
        self.bbox_y1 = bbox_y1
        self.bbox_x2 = bbox_x2
        self.bbox_y2 = bbox_y2
        self.severity = severity
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {
            "inspection_id": self.inspection_id,
            "defect_type": self.defect_type,
            "area_pixels": self.area_pixels,
            "area_mm2": self.area_mm2,
            "centroid_x": self.centroid_x,
            "centroid_y": self.centroid_y,
            "bbox_x1": self.bbox_x1,
            "bbox_y1": self.bbox_y1,
            "bbox_x2": self.bbox_x2,
            "bbox_y2": self.bbox_y2,
            "severity": self.severity,
            "confidence": self.confidence,
        }

    @classmethod
    def from_defect(cls, defect, inspection_id: int = 0) -> "DefectRecord":
        """从 Defect dataclass 创建记录。"""
        cx, cy = defect.centroid
        x1, y1, x2, y2 = defect.bbox
        return cls(
            inspection_id=inspection_id,
            defect_type=defect.type,
            area_pixels=defect.area_pixels,
            area_mm2=defect.area_mm2,
            centroid_x=float(cx),
            centroid_y=float(cy),
            bbox_x1=x1, bbox_y1=y1,
            bbox_x2=x2, bbox_y2=y2,
            severity=defect.severity,
            confidence=defect.confidence,
        )
