"""主检测流水线编排模块。

InspectionPipeline 将各处理阶段串联为统一的检测流程，
支持进度回调和中间结果访问。

InspectionResult 是流水线的最终输出，包含所有检测信息。
"""

import time
import cv2
import numpy as np
from typing import List, Optional, Callable
from dataclasses import dataclass, field

from core.preprocessing import Preprocessor
from core.texture import TextureAnalyzer, TextureFeatureVector
from core.defects import DefectDetector, Defect
from core.quality import QualityAssessor, QualityReport
from core.acquisition import ImageAcquisition, create_acquisition
from core.process_monitor import ProcessMonitor, ProcessGLCMFeatures


# ============================================================================
# 检测结果
# ============================================================================

@dataclass
class InspectionResult:
    """单次检测的完整结果。

    包含从预处理到质量评估的所有中间和最终输出。
    """
    # 输出图像
    image: Optional[np.ndarray] = None         # 带缺陷标注的图像（RGB）
    gray: Optional[np.ndarray] = None           # 预处理后的灰度图像
    heatmap: Optional[np.ndarray] = None        # 缺陷分布热力图
    roughness_map: Optional[np.ndarray] = None  # CV 粗糙度分布图

    # 检测结果
    defects: List[Defect] = field(default_factory=list)
    quality: Optional[QualityReport] = None
    ok_ng: bool = True

    # 特征向量（可选，用于分类器）
    texture_features: Optional[TextureFeatureVector] = None

    # 工艺参数监测特征（8 级 GLCM，供工艺面板使用）
    # 注意与 texture_features 是两套独立口径，不可混用，详见 core/process_monitor.py
    process_features: Optional[ProcessGLCMFeatures] = None

    # 性能计时
    timings: dict = field(default_factory=dict)
    total_time_ms: float = 0.0

    def summary(self) -> str:
        """生成单行摘要。"""
        status = "OK" if self.ok_ng else "NG"
        n_defects = len(self.defects)
        score = self.quality.overall_score if self.quality else 0
        return (
            f"[{status}] 评分={score:.1f} 缺陷数={n_defects} "
            f"耗时={self.total_time_ms:.0f}ms"
        )


# ============================================================================
# 主流水线
# ============================================================================

class InspectionPipeline:
    """喷砂表面质量检测主流水线。

    编排以下阶段：
        1. 图像采集（外置，由调用方传入图像）
        2. 预处理（ROI + Retinex + CLAHE）
        3. 纹理分析（GLCM + LBP + Gabor + CV 热力图）
        4. 缺陷检测（氧化斑 + 磨料嵌入 + 未粗化）
        5. 质量评估（多维度评分 + OK/NG 判定）

    Usage:
        pipeline = InspectionPipeline(config)
        result = pipeline.run(image)
        print(result.summary())
    """

    def __init__(self, config: dict):
        self.config = config

        # 初始化各处理阶段
        self.preprocessor = Preprocessor(config)
        self.texture_analyzer = TextureAnalyzer(config)
        self.defect_detector = DefectDetector(config)
        self.quality_assessor = QualityAssessor(config)
        self.process_monitor = ProcessMonitor(config)

        # 进度回调
        self._progress_callbacks: List[Callable[[int, str], None]] = []

    def on_progress(self, callback: Callable[[int, str], None]):
        """注册进度回调函数。

        Args:
            callback: (percentage: int, stage_name: str) → None
        """
        self._progress_callbacks.append(callback)

    def _notify_progress(self, pct: int, stage: str):
        for cb in self._progress_callbacks:
            try:
                cb(pct, stage)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 运行
    # ------------------------------------------------------------------

    def run(
        self,
        image: np.ndarray,
        board_id: str = "",
    ) -> InspectionResult:
        """运行完整的检测流水线。

        Args:
            image: 输入图像，BGR 或 RGB (H, W, 3) 或灰度 (H, W)。
            board_id: PCB 板编号。

        Returns:
            InspectionResult 包含所有检测信息。
        """
        result = InspectionResult()
        t_start = time.perf_counter()

        # --- Phase 1: 预处理 (0-20%) ---
        self._notify_progress(0, "预处理开始")
        t0 = time.perf_counter()

        gray = self.preprocessor.process(image)
        result.gray = gray

        t1 = time.perf_counter()
        result.timings["preprocessing"] = (t1 - t0) * 1000
        self._notify_progress(20, "预处理完成")

        # --- Phase 2: 纹理分析 (20-50%) ---
        self._notify_progress(21, "纹理分析开始")
        t0 = time.perf_counter()

        texture_vec = self.texture_analyzer.analyze(gray)
        result.texture_features = texture_vec

        # CV 热力图
        cv_heatmap = self.texture_analyzer.compute_cv_heatmap(gray)
        result.roughness_map = cv_heatmap

        # 方向一致性
        direction_consistency = self.texture_analyzer.direction_consistency(gray)

        t1 = time.perf_counter()
        result.timings["texture"] = (t1 - t0) * 1000
        self._notify_progress(50, "纹理分析完成")

        # --- 工艺参数监测（8 级口径，与上面的 256 级 GLCM 相互独立） ---
        # 必须从原始 image 取灰度，不能用 result.gray —— 后者经过 Retinex + CLAHE，
        # 会改变灰度分布使特征漂移，违反 core/process_monitor.py 的 R2 规则。
        if self.process_monitor.enabled:
            t0 = time.perf_counter()
            try:
                result.process_features = self.process_monitor.extractor.compute(image)
            except Exception as e:
                # 工艺监测是辅助功能，失败不应阻断主检测流程
                result.process_features = None
                print(f"警告: 工艺参数监测失败，已跳过: {e}")
            result.timings["process_monitor"] = (time.perf_counter() - t0) * 1000

        # --- Phase 3: 缺陷检测 (50-75%) ---
        self._notify_progress(51, "缺陷检测开始")
        t0 = time.perf_counter()

        # 确保有彩色图像用于氧化斑检测
        if image.ndim == 3:
            color_image = image
        else:
            color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if hasattr(cv2, 'cvtColor') else gray

        defects = self.defect_detector.detect_all(color_image, texture_vec)
        result.defects = defects

        t1 = time.perf_counter()
        result.timings["defects"] = (t1 - t0) * 1000
        self._notify_progress(75, "缺陷检测完成")

        # --- Phase 4: 质量评估 (75-90%) ---
        self._notify_progress(76, "质量评估中")
        t0 = time.perf_counter()

        report = self.quality_assessor.assess(
            defects, cv_heatmap, direction_consistency, board_id,
        )
        result.quality = report
        result.ok_ng = report.ok_ng

        t1 = time.perf_counter()
        result.timings["quality"] = (t1 - t0) * 1000
        self._notify_progress(90, "质量评估完成")

        # --- 后处理：生成标注和热力图 (90-100%) ---
        self._notify_progress(91, "生成输出图像")

        # 缺陷标注叠加
        annotated = self.defect_detector.draw_defects(image, defects)
        result.image = annotated

        # 缺陷热力图
        heatmap = self.defect_detector.generate_defect_heatmap(
            gray.shape, defects,
        )
        result.heatmap = heatmap

        # 总耗时
        t_end = time.perf_counter()
        result.total_time_ms = (t_end - t_start) * 1000
        self._notify_progress(100, "检测完成")

        return result

    # ------------------------------------------------------------------
    # 批量处理
    # ------------------------------------------------------------------

    def run_batch(
        self,
        images: List[np.ndarray],
        board_ids: List[str] = None,
    ) -> List[InspectionResult]:
        """批量处理多张图像。

        Args:
            images: 图像列表。
            board_ids: 可选的板号列表。

        Returns:
            检测结果列表。
        """
        if board_ids is None:
            board_ids = [f"board_{i:04d}" for i in range(len(images))]

        results = []
        for i, (img, bid) in enumerate(zip(images, board_ids)):
            result = self.run(img, bid)
            results.append(result)
            print(f"  [{i+1}/{len(images)}] {result.summary()}")

        return results

    # ------------------------------------------------------------------
    # 与采集器集成
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str) -> "InspectionPipeline":
        """从 YAML 配置文件创建流水线。"""
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def process_acquisition(
        self,
        acquisition: ImageAcquisition,
        max_frames: int = -1,
    ) -> List[InspectionResult]:
        """持续从采集器获取图像并处理。

        Args:
            acquisition: ImageAcquisition 实例。
            max_frames: 最大处理帧数（-1 = 无限）。
        """
        results = []
        frame_count = 0

        while max_frames < 0 or frame_count < max_frames:
            frame = acquisition.acquire()
            if frame is None:
                break

            board_id = f"{frame.source_id}_{frame.frame_index}"
            result = self.run(frame.image, board_id)
            results.append(result)

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"  已处理 {frame_count} 帧")

        return results
