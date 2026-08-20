"""质量评估模块。

综合多维度检测结果，输出质量评分和 OK/NG 判定。

质量评分体系：
    - 总分 0–100 分
    - 各项权重可配置
    - 低于阈值的触发 NG 信号和预警

评估维度：
    (1) 粗糙度均匀性（roughness_uniformity）— 基于 CV 热力图
    (2) 锚纹方向一致性（direction_consistency）— 基于 Gabor DCI
    (3) 氧化斑面积比（oxidation_percentage）
    (4) 磨料嵌入计数（embedding_count）
    (5) 未粗化面积比（unroughened_percentage）
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

from core.defects import Defect


# ============================================================================
# 质量报告数据结构
# ============================================================================

@dataclass
class QualityReport:
    """喷砂表面质量评估报告。

    包含所有五维度的评分和最终 OK/NG 判定。
    提供序列化方法用于存入数据库和导出报表。
    """

    # 总评
    overall_score: float = 100.0          # 综合评分 0–100
    ok_ng: bool = True                    # OK/NG 判定

    # 分项指标
    roughness_uniformity: float = 100.0   # 粗糙度均匀性评分
    roughness_std: float = 0.0            # Ra/Rz 空间分布标准差 (CV)
    direction_consistency: float = 1.0    # 锚纹方向一致性 (DCI, 0=最一致)
    oxidation_percentage: float = 0.0     # 氧化斑面积百分比
    embedding_count: int = 0              # 磨料嵌入颗粒数
    unroughened_percentage: float = 0.0   # 未粗化区域面积百分比

    # 元数据
    warnings: List[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    board_id: str = ""

    def to_dict(self) -> dict:
        """序列化为字典（用于 JSON 输出或数据库存储）。"""
        return {
            "overall_score": self.overall_score,
            "ok_ng": self.ok_ng,
            "roughness_uniformity": self.roughness_uniformity,
            "roughness_std": self.roughness_std,
            "direction_consistency": round(self.direction_consistency, 4),
            "oxidation_percentage": round(self.oxidation_percentage, 2),
            "embedding_count": self.embedding_count,
            "unroughened_percentage": round(self.unroughened_percentage, 2),
            "warnings": self.warnings,
            "timestamp": self.timestamp,
            "board_id": self.board_id,
        }

    def summary(self) -> str:
        """生成单行文本摘要。"""
        status = "OK ✓" if self.ok_ng else "NG ✗"
        return (
            f"[{self.timestamp}] {status} 评分={self.overall_score:.1f} "
            f"氧化={self.oxidation_percentage:.1f}% "
            f"磨料={self.embedding_count} "
            f"未粗化={self.unroughened_percentage:.1f}% "
            f"CV={self.roughness_std:.4f} "
            f"方向={self.direction_consistency:.4f}"
        )


# ============================================================================
# 质量评估器
# ============================================================================

class QualityAssessor:
    """喷砂表面质量评估器。

    汇总缺陷检测结果和纹理分析结果，生成多维度质量评分，
    做出 OK/NG 判定，并输出具体预警信息。

    评分逻辑：
        - 每个维度先做"单项评分"（0-100），然后按权重加权求和。
        - 若任一维度超过严格阈值（hard_limit），直接判 NG。
        - 否则根据加权总分与 ok_score_threshold 比较。

    Attributes:
        config: 质量评估配置字典。
    """

    def __init__(self, config: dict):
        quality_cfg = config.get("inspection", {}).get("quality", {})

        # 软阈值（用于扣分）
        self.cv_max = quality_cfg.get("roughness_cv_max", 0.35)
        self.direction_consistency_min = quality_cfg.get(
            "direction_consistency_min", 0.6
        )
        self.oxidation_max_pct = quality_cfg.get("oxidation_max_pct", 5.0)
        self.embedding_max_count = quality_cfg.get("embedding_max_count", 10)
        self.unroughened_max_pct = quality_cfg.get("unroughened_max_pct", 3.0)

        # OK/NG 判定
        self.ok_score_threshold = quality_cfg.get("ok_score_threshold", 60)

        # 硬阈值（任一超标直接 NG）
        self.hard_limits = {
            "oxidation_pct": quality_cfg.get("oxidation_hard_limit", 15.0),
            "unroughened_pct": quality_cfg.get("unroughened_hard_limit", 10.0),
        }

        # 权重分配（总和=1.0）
        self.weights = {
            "roughness": 0.30,
            "direction": 0.15,
            "oxidation": 0.25,
            "embedding": 0.15,
            "unroughened": 0.15,
        }

        # 图像分辨率
        self.resolution_mm_per_pixel = config.get(
            "system", {}
        ).get("resolution_mm_per_pixel", 0.01)

    # ------------------------------------------------------------------
    # 主评估入口
    # ------------------------------------------------------------------

    def assess(
        self,
        defects: List[Defect],
        cv_heatmap: Optional[np.ndarray] = None,
        direction_consistency: Optional[float] = None,
        board_id: str = "",
    ) -> QualityReport:
        """运行完整的质量评估。

        Args:
            defects: DefectDetector.detect_all() 返回的缺陷列表。
            cv_heatmap: CV 均匀性热力图 (H, W)。
            direction_consistency: Gabor DCI 值 [0, 1]。
            board_id: PCB 板编号。

        Returns:
            QualityReport 对象。
        """
        report = QualityReport(board_id=board_id)
        total_pixels = 0

        if cv_heatmap is not None:
            total_pixels = cv_heatmap.size

        # --- (1) 粗糙度均匀性评分 ---
        if cv_heatmap is not None and total_pixels > 0:
            report.roughness_std = float(np.mean(cv_heatmap))
            report.roughness_uniformity = self._score_roughness(
                report.roughness_std
            )
        else:
            report.roughness_uniformity = 100.0
            report.roughness_std = 0.0

        # --- (2) 方向一致性评分 ---
        if direction_consistency is not None:
            report.direction_consistency = direction_consistency
        # 评分在下面的加权阶段处理

        # --- (3)(4)(5) 缺陷统计 ---
        ox_area_px = 0
        emb_count = 0
        unrough_area_px = 0

        for d in defects:
            if d.type == "oxidation":
                ox_area_px += d.area_pixels
            elif d.type == "embedding":
                emb_count += 1
            elif d.type == "unroughened":
                unrough_area_px += d.area_pixels

        if total_pixels > 0:
            report.oxidation_percentage = (
                ox_area_px / total_pixels * 100
            )
            report.unroughened_percentage = (
                unrough_area_px / total_pixels * 100
            )
        report.embedding_count = emb_count

        # --- 硬阈值检查 ---
        if report.oxidation_percentage > self.hard_limits["oxidation_pct"]:
            report.ok_ng = False
            report.warnings.append(
                f"氧化斑面积超标: {report.oxidation_percentage:.1f}% "
                f"(限制: {self.hard_limits['oxidation_pct']:.1f}%)"
            )
        if report.unroughened_percentage > self.hard_limits["unroughened_pct"]:
            report.ok_ng = False
            report.warnings.append(
                f"未粗化面积超标: {report.unroughened_percentage:.1f}% "
                f"(限制: {self.hard_limits['unroughened_pct']:.1f}%)"
            )

        # --- 加权评分 ---
        scores = {}

        # 粗糙度评分
        scores["roughness"] = report.roughness_uniformity

        # 方向一致性评分
        scores["direction"] = self._score_direction(
            report.direction_consistency
        )

        # 氧化评分
        scores["oxidation"] = self._score_oxidation(
            report.oxidation_percentage
        )

        # 磨料评分
        scores["embedding"] = self._score_embedding(
            report.embedding_count
        )

        # 未粗化评分
        scores["unroughened"] = self._score_unroughened(
            report.unroughened_percentage
        )

        # 加权总分
        report.overall_score = sum(
            scores[k] * self.weights[k] for k in self.weights
        )
        report.overall_score = round(report.overall_score, 1)

        # --- OK/NG 判定 ---
        if not report.warnings:
            # 没有硬性超标的，根据总分判定
            report.ok_ng = report.overall_score >= self.ok_score_threshold

        # --- 预警信息 ---
        for key, score in scores.items():
            if score < 60:
                report.warnings.append(
                    f"{key} 评分偏低: {score:.1f}/100"
                )

        report.detail = {
            "scores": scores,
            "weights": self.weights,
            "thresholds": {
                "roughness_cv_max": self.cv_max,
                "direction_min": self.direction_consistency_min,
                "oxidation_max_pct": self.oxidation_max_pct,
                "embedding_max_cnt": self.embedding_max_count,
                "unroughened_max_pct": self.unroughened_max_pct,
            },
        }

        return report

    # ------------------------------------------------------------------
    # 单项评分函数
    # ------------------------------------------------------------------

    def _score_roughness(self, mean_cv: float) -> float:
        """粗糙度均匀性评分。

        CV 越低越好（均匀），CV = 0 → 100 分，CV = cv_max → 60 分。
        线性插值。
        """
        if mean_cv <= 0.01:
            return 100.0
        if mean_cv >= self.cv_max:
            return max(0.0, 100.0 * (1.0 - mean_cv / self.cv_max))
        return 100.0 - (mean_cv / self.cv_max) * 40.0

    def _score_direction(self, dci: float) -> float:
        """方向一致性评分。

        DCI（方向一致性指数）越低越各向同性越好。
        DCI=0 → 100 分（完全均匀），DCI=0.5 → 60 分，DCI=1.0 → 0 分。
        """
        return max(0.0, 100.0 - dci * 100.0)

    def _score_oxidation(self, pct: float) -> float:
        """氧化斑评分。

        面积比 0% → 100 分，达到 max_pct → 60 分，超过则线性递减。
        """
        if pct <= 0.1:
            return 100.0
        if pct >= self.oxidation_max_pct:
            base = max(0.0, 100.0 - pct * 10.0)
            return min(max(base, 0.0), 60.0)
        return 100.0 - (pct / self.oxidation_max_pct) * 40.0

    def _score_embedding(self, count: int) -> float:
        """磨料嵌入评分。

        数量 0 → 100 分，达到 max_count → 60 分，超过则线性递减。
        """
        if count <= 1:
            return 100.0
        if count >= self.embedding_max_count:
            base = max(0.0, 100.0 - count * 2.0)
            return min(max(base, 0.0), 60.0)
        return 100.0 - (count / self.embedding_max_count) * 40.0

    def _score_unroughened(self, pct: float) -> float:
        """未粗化面积评分。

        面积比 0% → 100 分，达到 max_pct → 60 分，超过则线性递减。
        """
        if pct <= 0.1:
            return 100.0
        if pct >= self.unroughened_max_pct:
            base = max(0.0, 100.0 - pct * 10.0)
            return min(max(base, 0.0), 60.0)
        return 100.0 - (pct / self.unroughened_max_pct) * 40.0

    # ------------------------------------------------------------------
    # 批量评估
    # ------------------------------------------------------------------

    def batch_assess(
        self, results: List[Tuple[List[Defect], np.ndarray, float, str]],
    ) -> List[QualityReport]:
        """批量质量评估。

        Args:
            results: [(defects, cv_heatmap, direction_consistency, board_id), ...]

        Returns:
            报告列表。
        """
        reports = []
        for defects, cv_map, dci, bid in results:
            report = self.assess(defects, cv_map, dci, bid)
            reports.append(report)
        return reports

    def yield_stats(self, reports: List[QualityReport]) -> dict:
        """统计多个报告的汇总指标。

        Returns:
            含合格率、平均分、各维度均值的字典。
        """
        if not reports:
            return {}

        n = len(reports)
        n_ok = sum(1 for r in reports if r.ok_ng)

        return {
            "total": n,
            "ok_count": n_ok,
            "ng_count": n - n_ok,
            "yield_rate": round(n_ok / n * 100, 2),
            "avg_score": round(np.mean([r.overall_score for r in reports]), 2),
            "min_score": round(min(r.overall_score for r in reports), 2),
            "avg_oxidation_pct": round(
                np.mean([r.oxidation_percentage for r in reports]), 2
            ),
            "avg_embedding_count": round(
                np.mean([r.embedding_count for r in reports]), 1
            ),
            "avg_unroughened_pct": round(
                np.mean([r.unroughened_percentage for r in reports]), 2
            ),
            "top_warnings": self._top_warnings(reports, 5),
        }

    @staticmethod
    def _top_warnings(reports: List[QualityReport], k: int) -> List[str]:
        """返回最常见的前 k 条预警信息。"""
        from collections import Counter
        counter = Counter()
        for r in reports:
            for w in r.warnings:
                counter[w] += 1
        return [w for w, _ in counter.most_common(k)]
