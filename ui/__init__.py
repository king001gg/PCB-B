"""PCB 阻焊前检测系统 — 图形用户界面。

Public API:
    MainWindow  — 主窗口
    ImageViewer — 带缩放/叠加的图像查看组件
    HeatmapWidget — 热力图可视化组件
    StatsPanel — 统计图表面板
    ConfigDialog — 参数配置对话框
    CameraDialog — 相机设置对话框
    DetectionWorker / CameraWorker — 后台线程
"""

from ui.main_window import MainWindow
from ui.image_viewer import ImageViewer
from ui.heatmap_widget import HeatmapWidget
from ui.stats_panel import StatsPanel
from ui.config_dialog import ConfigDialog
from ui.camera_dialog import CameraDialog
from ui.workers import DetectionWorker, CameraWorker

__all__ = [
    "MainWindow",
    "ImageViewer",
    "HeatmapWidget",
    "StatsPanel",
    "ConfigDialog",
    "CameraDialog",
    "DetectionWorker",
    "CameraWorker",
]
