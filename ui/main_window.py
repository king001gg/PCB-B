"""PCB 阻焊前喷砂质量在线检测系统 — PySide6 主窗口。

重构自旧版 main_window.py，新增：
    - 多标签页布局（检测 / 参数 / 统计 / 历史）
    - 热力图视图
    - 实时质量评分显示
    - 相机预览模式
    - 配置文件热更新

保留：
    - QThread 后台检测模式
    - 拖放图像功能
    - QSplitter 主布局
"""

import os
import sys
import time
import cv2
import yaml
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QApplication, QPushButton,
    QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QFileDialog, QTextEdit, QProgressBar,
    QSplitter, QStatusBar, QFrame,
    QTabWidget, QGroupBox, QGridLayout, QMenuBar, QMenu,
    QMessageBox, QDockWidget,
)
from PySide6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QAction
from PySide6.QtCore import Qt

# 核心模块
from core.preprocessing import Preprocessor
from core.texture import TextureAnalyzer
from core.defects import DefectDetector
from core.quality import QualityAssessor, QualityReport
from core.pipeline import InspectionResult
from core.process_monitor import ProcessMonitor

# UI 组件
from ui.image_viewer import ImageViewer
from ui.heatmap_widget import HeatmapWidget
from ui.stats_panel import StatsPanel
from ui.process_panel import ProcessPanel
from ui.config_dialog import ConfigDialog
from ui.camera_dialog import CameraDialog
from ui.camera_worker import CameraGrabWorker
from ui.workers import DetectionWorker

from hardware.camera import create_camera

# 工具
from utils.validators import validate_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


# ============================================================================
# 主窗口
# ============================================================================

class MainWindow(QMainWindow):

    SUPPORTED_FORMATS = "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"

    def __init__(self, config_path: str = None):
        super().__init__()
        self.setWindowTitle("PCB 阻焊前喷砂质量在线检测系统")
        self.resize(1400, 900)

        # --- 加载配置 ---
        self.config_path = config_path or str(DEFAULT_CONFIG)
        self.config = self._load_config(self.config_path)

        # --- 初始化核心模块 ---
        self._init_core_modules()

        # --- 状态 ---
        self.current_image_path: str = None
        self.current_raw_image: np.ndarray = None
        self.current_result: InspectionResult = None
        self.mode = self.config.get("system", {}).get("mode", "offline")
        self._live_mode = False
        # 在线抓拍的板号（抓拍图没有文件路径，用它填 board_id）
        self._snapshot_id: str = None

        # --- 构建 UI ---
        self._init_menu()
        self._init_ui()
        self._init_statusbar()

        # 支持拖放
        self.setAcceptDrops(True)

    # ------------------------------------------------------------------
    # 核心模块初始化
    # ------------------------------------------------------------------

    def _load_config(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        validate_config(config)
        return config

    def _init_core_modules(self):
        """初始化预处理、纹理分析、缺陷检测、质量评估模块。"""
        self.preprocessor = Preprocessor(self.config)
        self.texture_analyzer = TextureAnalyzer(self.config)
        self.defect_detector = DefectDetector(self.config)
        self.quality_assessor = QualityAssessor(self.config)
        self.process_monitor = ProcessMonitor(self.config)

    def reload_config(self):
        """重新加载配置并更新所有模块。"""
        self.config = self._load_config(self.config_path)
        self._init_core_modules()

    # ------------------------------------------------------------------
    # 菜单栏
    # ------------------------------------------------------------------

    def _init_menu(self):
        menubar = self.menuBar()

        # 文件
        file_menu = menubar.addMenu("文件(&F)")
        act_open = QAction("打开图像...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.load_image)
        file_menu.addAction(act_open)

        act_folder = QAction("打开文件夹...", self)
        act_folder.triggered.connect(self.load_folder)
        file_menu.addAction(act_folder)

        file_menu.addSeparator()

        act_export = QAction("导出报告...", self)
        act_export.triggered.connect(self.export_report)
        file_menu.addAction(act_export)

        file_menu.addSeparator()

        act_exit = QAction("退出", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # 视图
        view_menu = menubar.addMenu("视图(&V)")
        act_toggle_heatmap = QAction("切换热力图", self)
        act_toggle_heatmap.setShortcut("H")
        act_toggle_heatmap.triggered.connect(
            lambda: self.image_viewer.toggle_heatmap()
        )
        view_menu.addAction(act_toggle_heatmap)

        # 模式
        mode_menu = menubar.addMenu("模式(&M)")
        act_offline = QAction("离线模式", self)
        act_offline.triggered.connect(lambda: self._set_mode("offline"))
        mode_menu.addAction(act_offline)
        act_online = QAction("在线模式", self)
        act_online.triggered.connect(lambda: self._set_mode("online"))
        mode_menu.addAction(act_online)

        # 设置
        settings_menu = menubar.addMenu("设置(&S)")
        act_config = QAction("检测参数...", self)
        act_config.triggered.connect(self.open_config_dialog)
        settings_menu.addAction(act_config)

        act_camera = QAction("相机设置...", self)
        act_camera.triggered.connect(self.open_camera_dialog)
        settings_menu.addAction(act_camera)

        # 帮助
        help_menu = menubar.addMenu("帮助(&H)")
        act_about = QAction("关于...", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # ====== 主水平分割器 ======
        main_splitter = QSplitter(Qt.Horizontal)

        # ----- 左侧面板：图像显示 -----
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # 图像查看器（带缩放/热力图叠加）
        self.image_viewer = ImageViewer()
        left_layout.addWidget(self.image_viewer, 1)

        # 控制按钮行
        btn_row = QHBoxLayout()

        self.load_btn = QPushButton("📂 加载图像")
        # 按模式分派：离线选文件，在线从预览画面抓拍。
        # 不能直接连 load_image —— 在线模式下按钮文案是「拍照」却弹文件选择框。
        self.load_btn.clicked.connect(self._on_load_btn)
        btn_row.addWidget(self.load_btn)

        self.detect_btn = QPushButton("🔍 开始检测")
        self.detect_btn.clicked.connect(self.start_detection)
        self.detect_btn.setEnabled(False)
        self.detect_btn.setStyleSheet(
            "background: #2ecc71; color: white; font-weight: bold;"
        )
        btn_row.addWidget(self.detect_btn)

        self.live_btn = QPushButton("📷 实时预览")
        self.live_btn.setCheckable(True)
        self.live_btn.clicked.connect(self.toggle_live_mode)
        btn_row.addWidget(self.live_btn)

        left_layout.addLayout(btn_row)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(16)
        left_layout.addWidget(self.progress_bar)

        main_splitter.addWidget(left_panel)

        # ----- 右侧面板：结果 + 热力图 + 统计 -----
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        right_tabs = QTabWidget()

        # 标签页 1: 检测结果
        result_tab = QWidget()
        result_layout = QVBoxLayout(result_tab)

        # 质量评分卡片
        score_group = QGroupBox("质量评分")
        score_layout = QGridLayout(score_group)

        self.score_label = QLabel("--")
        self.score_label.setStyleSheet(
            "font-size: 36px; font-weight: bold; color: #2ecc71;"
        )
        self.score_label.setAlignment(Qt.AlignCenter)
        score_layout.addWidget(self.score_label, 0, 0, 1, 2)

        self.ok_ng_label = QLabel("等待检测...")
        self.ok_ng_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.ok_ng_label.setAlignment(Qt.AlignCenter)
        score_layout.addWidget(self.ok_ng_label, 1, 0, 1, 2)

        # 分项指标
        metrics = [
            ("粗糙度均匀性:", "roughness_value"),
            ("方向一致性:", "direction_value"),
            ("氧化斑面积:", "oxidation_value"),
            ("磨料嵌入:", "embedding_value"),
            ("未粗化面积:", "unroughened_value"),
        ]
        self.metric_labels = {}
        for row, (label, key) in enumerate(metrics):
            lbl = QLabel(label)
            val = QLabel("--")
            val.setStyleSheet("font-weight: bold;")
            score_layout.addWidget(lbl, row + 2, 0)
            score_layout.addWidget(val, row + 2, 1)
            self.metric_labels[key] = val

        score_group.setLayout(score_layout)
        result_layout.addWidget(score_group)

        # 缺陷列表
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("检测结果将显示在这里...")
        self.result_text.setMaximumHeight(200)
        result_layout.addWidget(self.result_text)

        right_tabs.addTab(result_tab, "检测结果")

        # 标签页 2: 热力图
        self.heatmap_widget = HeatmapWidget()
        right_tabs.addTab(self.heatmap_widget, "热力图")

        # 标签页 3: 统计
        self.stats_panel = StatsPanel()
        right_tabs.addTab(self.stats_panel, "统计")

        # 工艺参数监测面板 —— 置于标签页上方，切换标签页时报警灯仍可见
        self.process_panel = ProcessPanel()
        right_layout.addWidget(self.process_panel)

        right_layout.addWidget(right_tabs)
        main_splitter.addWidget(right_panel)

        # 分割比例 2:1
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 1)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.addWidget(main_splitter)

        # 启动时的按钮文案要跟配置里的模式一致（配置为在线时不能显示「加载图像」）
        self._sync_mode_controls()

    def _init_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 — 请加载 PCB 喷砂表面图像")

    # ------------------------------------------------------------------
    # 拖放支持
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if self._is_image(path):
                self._set_image(path)

    # ------------------------------------------------------------------
    # 槽函数
    # ------------------------------------------------------------------

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 PCB 喷砂表面图像", "",
            self.SUPPORTED_FORMATS,
        )
        if file_path:
            self._set_image(file_path)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "选择图像文件夹",
        )
        if folder:
            # 加载文件夹中第一张支持的图像
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                files = list(Path(folder).glob(f"*{ext}"))
                if files:
                    self._set_image(str(files[0]))
                    break

    def _on_load_btn(self):
        """「加载图像 / 拍照」按钮：按当前模式分派。"""
        if self.mode == "online":
            self._capture_frame()
        else:
            self.load_image()

    def _capture_frame(self) -> bool:
        """抓拍当前预览帧作为待检测图像，并停掉预览。

        抓拍即停预览，是刻意的：预览路径与检测路径都会写工艺面板，
        同时运行会让数值来回跳（见 ``toggle_live_mode``）。停掉之后
        整条流程与离线路径完全一致，也就不需要额外的互斥逻辑。

        Returns:
            是否成功抓到一帧。失败时会给出可操作的状态栏提示。
        """
        if not self._live_mode:
            self.status_bar.showMessage("请先点「实时预览」，再从预览画面抓拍")
            return False

        frame = self.current_raw_image
        if frame is None:
            self.status_bar.showMessage("尚未取到画面，请稍候再拍")
            return False

        # 先复制再停预览：停预览会释放相机。复制是因为在途帧可能还没派发完，
        # 而检测跑在后台线程里、跑完才读这张图，不能让它边跑边被覆盖。
        frozen = frame.copy()
        self._stop_live()

        self.current_image_path = None
        self._snapshot_id = f"抓拍-{time.strftime('%H%M%S')}"
        self.current_raw_image = frozen
        self.image_viewer.set_image(frozen)
        self.detect_btn.setEnabled(True)
        self._reset_result_views()
        self.status_bar.showMessage(
            f"已抓拍（{self._snapshot_id}）— 按「开始检测」运行完整检测"
        )
        return True

    def _current_board_id(self) -> str:
        """当前待检测图像的板号：文件用文件名，抓拍图用抓拍时间。"""
        if self.current_image_path:
            return os.path.basename(self.current_image_path)
        return self._snapshot_id or ""

    def start_detection(self):
        """启动后台检测线程。

        在线模式下先抓拍当前预览帧并停掉预览，之后走与离线完全相同的
        路径 —— 检测期间不会再有取流路径往工艺面板写值。
        """
        if self._live_mode and not self._capture_frame():
            return
        if self.current_raw_image is None:
            self.status_bar.showMessage("请先加载图像")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.detect_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        # 检测期间禁止再进预览：两条路径都会写工艺面板，同时跑数值会来回跳
        self.live_btn.setEnabled(False)
        self.status_bar.showMessage("正在检测中...")

        # 后台线程
        board_id = self._current_board_id()
        self.thread = DetectionWorker(
            image=self.current_raw_image,
            preprocessor=self.preprocessor,
            texture_analyzer=self.texture_analyzer,
            defect_detector=self.defect_detector,
            quality_assessor=self.quality_assessor,
            board_id=board_id,
            process_monitor=self.process_monitor,
        )
        self.thread.finished.connect(self._on_detection_finished)
        self.thread.progress.connect(self._on_progress)
        self.thread.error.connect(self._on_detection_error)
        self.thread.start()

    def _on_progress(self, value: int):
        self.progress_bar.setValue(value)

    def _on_detection_finished(self, result: InspectionResult):
        """检测完成，更新 UI。"""
        self.progress_bar.setVisible(False)
        self.detect_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.live_btn.setEnabled(True)

        self.current_result = result

        # 更新图像显示（含缺陷标注）
        self.image_viewer.set_image(result.image)

        # 更新工艺参数监测面板
        self._refresh_process_panel(result.process_features)

        # 更新结果面板
        self._update_result_panel(result)

        # 更新热力图
        self.heatmap_widget.set_data(
            defect_heatmap=result.heatmap,
            roughness_map=result.roughness_map,
        )

        # 更新统计
        self.stats_panel.add_report(result.quality)

        # 状态栏
        status = "OK ✓" if result.ok_ng else "NG ✗"
        self.status_bar.showMessage(
            f"检测完成 — {status} 评分: {result.quality.overall_score:.1f}/100  "
            f"缺陷数: {len(result.defects)}"
        )

    def _on_detection_error(self, error_msg: str):
        self.progress_bar.setVisible(False)
        self.detect_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.live_btn.setEnabled(True)
        self.result_text.setText(f"❌ 检测错误:\n{error_msg}")
        self.status_bar.showMessage("检测失败")
        QMessageBox.critical(self, "检测错误", error_msg)

    def toggle_live_mode(self):
        """切换实时预览模式。

        预览与检测不会同时运行 —— 两条路径都会写工艺面板，同时跑数值会
        来回跳。互斥靠「检测先抓拍并停掉预览」实现（见 _capture_frame），
        而不是靠禁用按钮：在线模式下必须先预览、再抓拍，才能拿到待检测的图。
        按钮的启停由 _start_live / _stop_live 统一负责 —— 菜单里的
        「离线模式」也会调 _stop_live，集中在一处才不会漏。
        """
        self._live_mode = self.live_btn.isChecked()
        if self._live_mode:
            self.live_btn.setText("⏹ 停止预览")
            self._start_live()
        else:
            self._stop_live()

    def open_config_dialog(self):
        """打开参数配置对话框。"""
        dialog = ConfigDialog(self.config, self.config_path, self)
        if dialog.exec():
            self.reload_config()
            self.status_bar.showMessage("配置已更新")

    def open_camera_dialog(self):
        """打开相机设置对话框。"""
        dialog = CameraDialog(self.config.get("camera", {}), self)
        if dialog.exec():
            # 必须取 dialog.camera_config：对话框内部对传入的 dict 做了 copy，
            # 改的是它自己那份。早先这里写回的是传进去的原对象，等于没改，
            # 相机参数点「确定」后全部丢失。
            self.config["camera"] = dialog.camera_config
            self.status_bar.showMessage("相机设置已更新")

    def export_report(self):
        """导出检测报告。"""
        if self.current_result is None:
            QMessageBox.information(self, "提示", "请先运行检测")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出检测报告", "inspection_report.csv",
            "CSV 文件 (*.csv);;Excel 文件 (*.xlsx);;PDF 文件 (*.pdf)",
        )
        if path:
            report = self.current_result.quality
            try:
                import csv
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(["指标", "值"])
                    for k, v in report.to_dict().items():
                        writer.writerow([k, v])
                self.status_bar.showMessage(f"报告已导出: {path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", str(e))

    # ------------------------------------------------------------------
    # UI 更新
    # ------------------------------------------------------------------

    def _set_image(self, path: str):
        """加载图像文件并显示。"""
        self.current_image_path = path
        image = cv2.imread(path)
        if image is None:
            QMessageBox.critical(self, "错误", f"无法读取图像:\n{path}")
            return

        self.current_raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_viewer.set_image(self.current_raw_image)

        self.detect_btn.setEnabled(True)
        self._snapshot_id = None
        self._reset_result_views()

        self.status_bar.showMessage(f"已加载: {os.path.basename(path)}")

    def _reset_result_views(self) -> None:
        """清空上一张图的检测结果与工艺面板。"""
        self.result_text.clear()
        self.score_label.setText("--")
        self.ok_ng_label.setText("等待检测...")
        for lbl in self.metric_labels.values():
            lbl.setText("--")
        # 清空工艺面板，避免上一张图的报警状态残留
        self.process_panel.reset()

    def _refresh_process_panel(self, features) -> None:
        """刷新工艺参数监测面板。

        Args:
            features: ProcessGLCMFeatures，为 None（未启用或计算失败）时跳过。
        """
        if features is None:
            return
        try:
            self.process_panel.update_features(features)
            self.process_panel.update_verdict(
                self.process_monitor.evaluate(features)
            )
        except Exception as e:
            # 面板刷新属辅助功能，失败不应中断检测流程
            self.status_bar.showMessage(f"工艺面板刷新失败: {e}")

    def _update_result_panel(self, result: InspectionResult):
        """更新检测结果面板。"""
        q = result.quality

        # 评分
        self.score_label.setText(f"{q.overall_score:.1f}")
        if q.ok_ng:
            self.score_label.setStyleSheet(
                "font-size: 36px; font-weight: bold; color: #2ecc71;"
            )
            self.ok_ng_label.setText("OK ✓ 合格")
            self.ok_ng_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #2ecc71;"
            )
        else:
            self.score_label.setStyleSheet(
                "font-size: 36px; font-weight: bold; color: #e74c3c;"
            )
            self.ok_ng_label.setText("NG ✗ 不合格")
            self.ok_ng_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #e74c3c;"
            )

        # 分项指标
        self.metric_labels["roughness_value"].setText(
            f"{q.roughness_uniformity:.1f}/100 (CV={q.roughness_std:.4f})"
        )
        self.metric_labels["direction_value"].setText(
            f"{q.direction_consistency:.4f} (0=最好)"
        )
        self.metric_labels["oxidation_value"].setText(
            f"{q.oxidation_percentage:.2f}%"
        )
        self.metric_labels["embedding_value"].setText(
            f"{q.embedding_count} 个"
        )
        self.metric_labels["unroughened_value"].setText(
            f"{q.unroughened_percentage:.2f}%"
        )

        # 缺陷列表
        if result.defects:
            lines = [
                f"检测到 {len(result.defects)} 处缺陷\n",
                f"{'类型':<16} {'面积(mm²)':>10} {'严重度':>8} {'位置'}",
                "-" * 60,
            ]
            for d in result.defects:
                lines.append(
                    f"{d.type:<16} {d.area_mm2:>10.4f} "
                    f"{d.severity:>8.2f} ({d.bbox[0]},{d.bbox[1]})"
                )
            self.result_text.setText("\n".join(lines))
        else:
            self.result_text.setText("✅ 未检测到缺陷。表面质量良好。")

        # 预警
        if q.warnings:
            self.result_text.append(
                "\n⚠ 预警信息:\n" + "\n".join(f"  • {w}" for w in q.warnings)
            )

    # ------------------------------------------------------------------
    # 实时模式
    # ------------------------------------------------------------------

    def _start_live(self):
        """启动相机实时预览。

        取流放在 ``CameraGrabWorker`` 线程里：GenICam 的 fetch() 是带超时的
        阻塞调用，留在 GUI 线程会直接冻住界面；OpenCV 的 read() 在相机拔线
        或带宽不足时同样会长时间阻塞。相机由工作线程自己 open()/release()，
        GUI 线程不碰设备句柄。
        """
        # 先收掉可能还在跑的上一轮，否则两个线程会抢同一个相机。
        # 只能用 _teardown_worker()，不能用 _stop_live() —— 后者会复位
        # _live_mode 和按钮文案，而 toggle_live_mode 刚把它们设成「预览中」。
        self._teardown_worker()

        try:
            camera = create_camera(self.config)
        except Exception as e:
            self._live_failed(f"相机初始化失败：{e}")
            return

        self.camera_worker = CameraGrabWorker(camera)
        self.camera_worker.frame_ready.connect(self._on_frame)
        self.camera_worker.opened.connect(self._on_live_opened)
        self.camera_worker.failed.connect(self._live_failed)
        self.camera_worker.stats.connect(self._on_live_stats)
        self.camera_worker.start()

        # 相机是异步打开的，这里先按「已进入预览」处理；真失败会走
        # _live_failed 把状态回滚。检测按钮在 _on_live_opened 里启用。
        self.status_bar.showMessage("正在打开相机…")

    def _teardown_worker(self) -> None:
        """停掉采集线程并释放相机，不动任何界面状态。"""
        worker = getattr(self, "camera_worker", None)
        if worker is None:
            return
        worker.stop()
        # acquire() 可能正阻塞在超时等待中，最坏要等一个超时周期
        if not worker.wait(3000):
            # 线程没退就强杀：继续留着会与下一次 open() 抢设备
            print("[Camera] 采集线程 3 秒内未退出，强制终止")
            worker.terminate()
            worker.wait(1000)
        worker.deleteLater()
        self.camera_worker = None

    def _stop_live(self):
        """停止实时预览。"""
        self._teardown_worker()

        # 菜单里的「离线模式」也会走到这里，故一并复位预览状态与按钮文案，
        # 避免出现按钮显示「停止预览」但实际已停止的不一致
        self._live_mode = False
        self.live_btn.setChecked(False)   # clicked 信号，setChecked 不会递归触发
        self.live_btn.setText("📷 实时预览")
        self.detect_btn.setEnabled(self.current_raw_image is not None)
        self.status_bar.showMessage("实时预览已停止")

    def _on_live_opened(self):
        """相机打开成功。"""
        # 预览期间允许检测：start_detection 会先抓拍当前帧并停掉预览，
        # 再走离线路径，两条路径不会同时写工艺面板
        self.detect_btn.setEnabled(True)
        self.status_bar.showMessage('实时预览模式 — 按"停止预览"退出')

    def _live_failed(self, message: str):
        """相机打开失败或连续取帧失败：回滚到未预览状态。

        相机没开起来就等于没进预览模式，必须保持检测按钮可用，
        否则用户既不能预览也不能检测。
        """
        self._stop_live()
        QMessageBox.critical(self, "相机错误", message)

    def _on_live_stats(self, fps: float, failures: int):
        """刷新状态栏的帧率显示。"""
        if not self._live_mode:
            return
        text = f"实时预览 — {fps:.1f} FPS"
        if failures:
            text += f"（取帧失败 {failures} 次）"
        self.status_bar.showMessage(text)

    def _on_frame(self, frame_rgb: np.ndarray):
        """处理工作线程投来的一帧（已在工作线程内转成 RGB）。

        GLCM 计算完必须解除工作线程的忙标志，否则预览只出一帧就卡住 ——
        用 finally 保证异常路径也会解除。
        """
        try:
            # 预览已停（例如刚抓拍完）时，在途的帧一律丢弃，
            # 否则会把抓拍冻结下来的那张图覆盖掉
            if not self._live_mode:
                return

            self.current_raw_image = frame_rgb
            self.image_viewer.set_image(frame_rgb)

            # 工艺参数监测走轻量路径：只算 8 级 GLCM（相机分辨率下实测 6.6 ms）。
            # 不跑 Gabor / SVM / 缺陷检测 —— 整条检测在相机分辨率下约 21 s，
            # 无法逐帧跑；需要完整结果时走「抓拍 + 开始检测」。
            # 这里拿到的是 RGB（工作线程边界上转好的），与离线路径一致。
            # 两条路径颜色顺序不一致会让灰度转换权重对调，同一块板算出不同
            # 的特征值，破坏 R3 的一致性要求。
            if self.process_monitor.enabled:
                try:
                    features = self.process_monitor.extractor.compute(frame_rgb)
                    self._refresh_process_panel(features)
                except Exception:
                    # 单帧计算失败不应中断预览
                    pass
        finally:
            worker = getattr(self, "camera_worker", None)
            if worker is not None:
                worker.notify_gui_busy(False)

    # ------------------------------------------------------------------
    # 模式切换
    # ------------------------------------------------------------------

    def _set_mode(self, mode: str):
        self.mode = mode
        self.config["system"]["mode"] = mode
        self._sync_mode_controls()
        if mode == "online":
            self.status_bar.showMessage("已切换到在线模式 — 请检查相机连接")
        else:
            self._stop_live()
            self.status_bar.showMessage("已切换到离线模式")

    def _sync_mode_controls(self) -> None:
        """按当前模式刷新按钮文案（不动预览状态、不写状态栏）。

        文案与分派逻辑必须一起看 ``_on_load_btn``：只改文案不改槽函数，
        就会变成「按『拍照』弹出文件选择框」—— 这正是修复前的状态。
        """
        self.load_btn.setText("📷 拍照" if self.mode == "online" else "📂 加载图像")

    # ------------------------------------------------------------------
    # 其他
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """关窗前先停掉采集线程。

        不这么做的话 Qt 会析构一个仍在运行的 QThread，运气好是
        「QThread: Destroyed while thread is still running」加内存泄漏，
        运气不好直接崩在退出路径上。
        """
        self._teardown_worker()
        super().closeEvent(event)

    def _show_about(self):
        QMessageBox.about(
            self, "关于",
            "PCB 阻焊前喷砂质量在线检测系统\n\n"
            "版本 2.0.0\n\n"
            "基于机器视觉的喷砂表面质量评估系统\n"
            "纹理分析（GLCM+LBP+Gabor）+ SVM/CNN 缺陷分类\n\n"
            "核心技术：\n"
            "  - ROI 提取 + Retinex 光照校正\n"
            "  - 多尺度纹理特征提取\n"
            "  - 氧化斑/磨料嵌入/未粗化 检测\n"
            "  - 多维度质量评分 + OK/NG 判定",
        )

    @staticmethod
    def _is_image(path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 全局样式
    app.setStyleSheet("""
        QMainWindow { background: #f0f0f0; }
        QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 4px; margin-top: 8px; padding-top: 8px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        QPushButton { padding: 6px 14px; border-radius: 3px; border: 1px solid #bbb; background: #fff; }
        QPushButton:hover { background: #e0e0e0; }
        QPushButton:pressed { background: #d0d0d0; }
        QTextEdit { border: 1px solid #ccc; border-radius: 3px; }
        QProgressBar { border: 1px solid #ccc; border-radius: 3px; text-align: center; }
        QProgressBar::chunk { background: #3498db; border-radius: 2px; }
    """)

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    window = MainWindow(config_path)
    window.show()
    sys.exit(app.exec())
