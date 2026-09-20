"""后台工作线程模块。

将耗时的图像处理任务从 GUI 主线程分离，确保界面响应。
"""

import numpy as np
from PySide6.QtCore import QThread, Signal

from core.color import ColorAnalyzer
from core.preprocessing import Preprocessor
from core.texture import TextureAnalyzer
from core.defects import DefectDetector, Defect
from core.quality import QualityAssessor, QualityReport
from core.pipeline import InspectionResult
from core.process_monitor import ProcessMonitor


class DetectionWorker(QThread):
    """后台检测线程。

    运行完整的检测流水线（预处理 → 纹理分析 → 缺陷检测 → 质量评估），
    完成后通过信号返回结果。

    Signals:
        finished: (InspectionResult) 检测完成
        progress: (int) 进度百分比 0-100
        error: (str) 错误消息
    """

    finished = Signal(object)   # InspectionResult
    progress = Signal(int)
    error = Signal(str)

    def __init__(
        self,
        image: np.ndarray,
        preprocessor: Preprocessor,
        texture_analyzer: TextureAnalyzer,
        defect_detector: DefectDetector,
        quality_assessor: QualityAssessor,
        board_id: str = "",
        process_monitor: ProcessMonitor = None,
        color_analyzer: ColorAnalyzer = None,
        color_order: str = "rgb",
    ):
        super().__init__()
        self.image = image
        self.preprocessor = preprocessor
        self.texture_analyzer = texture_analyzer
        self.defect_detector = defect_detector
        self.quality_assessor = quality_assessor
        self.board_id = board_id
        self.process_monitor = process_monitor
        self.color_analyzer = color_analyzer
        # GUI 在 _set_image 里做了 BGR2RGB，所以喂进来的是 RGB。这个量无法从
        # 数组形状推断，喂错不报错、只会静默把色相转掉 172°，故显式声明。
        self.color_order = color_order

    def run(self):
        try:
            # (1) 预处理
            self.progress.emit(10)
            gray = self.preprocessor.process(self.image)

            # (2) 纹理分析
            self.progress.emit(30)
            texture_vec = self.texture_analyzer.analyze(gray)

            # (2.5) 工艺参数监测（8 级口径，与上面的 256 级 GLCM 相互独立）
            # 必须从原始 self.image 取灰度，不能用 gray —— 后者经过 Retinex + CLAHE，
            # 会改变灰度分布使特征漂移，违反 core/process_monitor.py 的 R2 规则。
            process_features = None
            if self.process_monitor is not None and self.process_monitor.enabled:
                try:
                    process_features = self.process_monitor.extractor.compute(self.image)
                except Exception:
                    # 工艺监测是辅助功能，失败不应阻断主检测流程
                    process_features = None

            # (3) CV 热力图
            self.progress.emit(50)
            cv_heatmap = self.texture_analyzer.compute_cv_heatmap(gray)

            # (4) 方向一致性 —— 复用 analyze() 那一趟 Gabor 的能量，
            # 避免对同一张图再卷一遍全部滤波器
            direction_consistency = self.texture_analyzer.direction_consistency(
                gray, texture_vec.gabor_orientation_energies
            )

            # (5) 缺陷检测
            self.progress.emit(70)
            defects = self.defect_detector.detect_all(self.image, texture_vec)

            # (6) 色度 / 饱和度（只监测，不进总分）。失败不应阻断主检测流程。
            color_features = None
            if self.color_analyzer is not None:
                try:
                    color_features = self.color_analyzer.analyze(
                        self.image, self.color_order
                    )
                except Exception:
                    color_features = None

            # (7) 质量评估
            self.progress.emit(85)
            report = self.quality_assessor.assess(
                defects, cv_heatmap, direction_consistency, self.board_id,
                color_features=color_features,
            )

            # (8) 标注图像
            annotated = self.defect_detector.draw_defects(self.image, defects)
            heatmap = self.defect_detector.generate_defect_heatmap(
                gray.shape, defects,
            )

            # 构建结果
            result = InspectionResult(
                image=annotated,
                gray=gray,
                quality=report,
                defects=defects,
                roughness_map=cv_heatmap,
                heatmap=heatmap,
                ok_ng=report.ok_ng,
                texture_features=texture_vec,
                process_features=process_features,
            )

            self.progress.emit(100)
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class CameraWorker(QThread):
    """后台相机采集线程。

    持续从相机抓取帧，通过信号传递给 GUI 显示。
    支持手动和外部触发模式。

    Signals:
        frame_ready: (np.ndarray) 新帧到达
        camera_error: (str) 相机错误
    """

    frame_ready = Signal(np.ndarray)
    camera_error = Signal(str)

    def __init__(self, camera, interval_ms: int = 100):
        super().__init__()
        self.camera = camera
        self.interval_ms = interval_ms
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            try:
                frame = self.camera.acquire()
                if frame is not None:
                    self.frame_ready.emit(frame)
            except Exception as e:
                self.camera_error.emit(str(e))
            self.msleep(self.interval_ms)

    def stop(self):
        self._running = False
