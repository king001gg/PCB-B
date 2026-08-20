"""检测报告生成模块。

整合单次检测的全部信息，生成结构化报告。
支持文本、JSON、字典等多种输出格式。
"""

import json
import numpy as np
from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from core.quality import QualityReport


class InspectionReport:
    """检测报告生成器。

    汇总检测流水线的输出，提供：
        - 结构化字典 (to_dict)
        - JSON 字符串 (to_json)
        - 文本摘要 (to_text)
        - SPC 统计 (compute_spc)
    """

    def __init__(self, result):
        """从 InspectionResult 创建报告。

        Args:
            result: InspectionResult 对象。
        """
        self.result = result
        self.quality = result.quality
        self.timestamp = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # 输出格式
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """转换为结构化字典。"""
        r = self.result
        q = self.quality

        data = {
            "report": {
                "timestamp": self.timestamp,
                "board_id": q.board_id if q else "",
                "ok_ng": r.ok_ng,
                "overall_score": q.overall_score if q else 0,
                "total_time_ms": round(r.total_time_ms, 2),
            },
            "quality": q.to_dict() if q else {},
            "defects": {
                "total_count": len(r.defects),
                "by_type": self._count_by_type(),
                "items": [
                    {
                        "type": d.type,
                        "area_mm2": round(d.area_mm2, 4),
                        "severity": round(d.severity, 4),
                        "confidence": round(d.confidence, 4),
                        "centroid": list(d.centroid),
                        "bbox": list(d.bbox),
                    }
                    for d in r.defects
                ],
            },
            "performance": {
                "preprocessing_ms": round(
                    r.timings.get("preprocessing", 0), 1
                ),
                "texture_ms": round(
                    r.timings.get("texture", 0), 1
                ),
                "defects_ms": round(
                    r.timings.get("defects", 0), 1
                ),
                "quality_ms": round(
                    r.timings.get("quality", 0), 1
                ),
                "total_ms": round(r.total_time_ms, 1),
            },
        }

        if r.roughness_map is not None:
            data["roughness"] = {
                "mean_cv": round(float(np.mean(r.roughness_map)), 4),
                "std_cv": round(float(np.std(r.roughness_map)), 4),
                "max_cv": round(float(np.max(r.roughness_map)), 4),
            }

        return data

    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串。"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_text(self) -> str:
        """生成可读的文本报告。"""
        r = self.result
        q = self.quality

        lines = [
            "=" * 60,
            "  PCB 喷砂表面质量检测报告",
            "=" * 60,
            f"  时间: {self.timestamp}",
            f"  板号: {q.board_id if q else 'N/A'}",
            f"  判定: {'OK ✓ 合格' if r.ok_ng else 'NG ✗ 不合格'}",
            f"  评分: {q.overall_score:.1f} / 100",
            f"  耗时: {r.total_time_ms:.0f} ms",
            "",
            "--- 质量指标 ---",
            f"  粗糙度均匀性: {q.roughness_uniformity:.1f}/100",
            f"  粗糙度 CV:     {q.roughness_std:.4f}",
            f"  方向一致性:   {q.direction_consistency:.4f} (0=最好)",
            f"  氧化斑面积:   {q.oxidation_percentage:.2f}%",
            f"  磨料嵌入:     {q.embedding_count} 个",
            f"  未粗化面积:   {q.unroughened_percentage:.2f}%",
            "",
            f"--- 缺陷列表 ({len(r.defects)} 处) ---",
        ]

        if r.defects:
            for i, d in enumerate(r.defects, 1):
                lines.append(
                    f"  {i}. [{d.type}] 面积={d.area_mm2:.4f}mm² "
                    f"严重度={d.severity:.2f} "
                    f"({d.bbox[0]},{d.bbox[1]})→({d.bbox[2]},{d.bbox[3]})"
                )
        else:
            lines.append("  (无缺陷)")

        if q.warnings:
            lines.append("")
            lines.append("--- 预警 ---")
            for w in q.warnings:
                lines.append(f"  ⚠ {w}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # SPC 统计
    # ------------------------------------------------------------------

    def compute_spc(self, historical_scores: List[float]) -> dict:
        """计算统计过程控制（SPC）指标。

        Args:
            historical_scores: 历史质量评分列表（包含当前评分）。

        Returns:
            含均值、标准差、控制限、Cp/Cpk 的字典。
        """
        import numpy as np

        if len(historical_scores) < 2:
            return {"error": "样本不足（需要 ≥ 2 个数据点）"}

        data = np.array(historical_scores)
        mean = float(np.mean(data))
        std = float(np.std(data, ddof=1))

        # 控制限（±3σ）
        ucl = min(100, mean + 3 * std)
        lcl = max(0, mean - 3 * std)

        # 警告限（±2σ）
        uwl = min(100, mean + 2 * std)
        lwl = max(0, mean - 2 * std)

        # 过程能力指数（Cp、Cpk）
        usl = 100  # 规格上限
        lsl = 60   # 规格下限
        if std > 1e-6:
            cp = (usl - lsl) / (6 * std)
            cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
        else:
            cp = 999.0
            cpk = 999.0

        current = self.quality.overall_score if self.quality else 0

        return {
            "n": len(historical_scores),
            "mean": round(mean, 2),
            "std": round(std, 2),
            "ucl": round(ucl, 2),
            "lcl": round(lcl, 2),
            "uwl": round(uwl, 2),
            "lwl": round(lwl, 2),
            "cp": round(cp, 2),
            "cpk": round(cpk, 2),
            "current": round(current, 2),
            "out_of_control": current > ucl or current < lcl,
            "warning": (current > uwl or current < lwl) and not (
                current > ucl or current < lcl
            ),
        }

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _count_by_type(self) -> dict:
        """按缺陷类型统计数量。"""
        counts = {}
        for d in self.result.defects:
            counts[d.type] = counts.get(d.type, 0) + 1
        return counts
