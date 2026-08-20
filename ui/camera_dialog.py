"""相机设置对话框。

配置工业相机的曝光时间、增益、分辨率、触发模式等参数。
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QSpinBox, QDoubleSpinBox, QComboBox, QPushButton,
    QGroupBox, QLabel, QMessageBox, QCheckBox,
)
from PySide6.QtCore import Qt


class CameraDialog(QDialog):
    """相机参数设置对话框。

    支持 OpenCV 直接控制（USB/GigE）和 GenICam 标准接口。
    """

    def __init__(self, camera_config: dict, parent=None):
        super().__init__(parent)
        self.camera_config = camera_config.copy()
        self.setWindowTitle("相机设置")
        self.resize(400, 400)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 基本设置
        basic_group = QGroupBox("基本设置")
        basic_form = QFormLayout(basic_group)

        self.driver_combo = QComboBox()
        self.driver_combo.addItems(["opencv", "harvesters (GenICam)"])
        idx = 0 if self.camera_config.get("driver") != "harvesters" else 1
        self.driver_combo.setCurrentIndex(idx)
        basic_form.addRow("驱动:", self.driver_combo)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(640, 8192)
        self.width_spin.setValue(self.camera_config.get("width", 2448))
        basic_form.addRow("宽度(px):", self.width_spin)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(480, 8192)
        self.height_spin.setValue(self.camera_config.get("height", 2048))
        basic_form.addRow("高度(px):", self.height_spin)

        layout.addWidget(basic_group)

        # 曝光和增益
        exposure_group = QGroupBox("曝光与增益")
        exposure_form = QFormLayout(exposure_group)

        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(10, 100000)
        self.exposure_spin.setSingleStep(100)
        self.exposure_spin.setValue(self.camera_config.get("exposure_us", 5000))
        self.exposure_spin.setSuffix(" μs")
        exposure_form.addRow("曝光时间:", self.exposure_spin)

        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(1.0, 32.0)
        self.gain_spin.setSingleStep(0.5)
        self.gain_spin.setValue(self.camera_config.get("gain", 1.0))
        exposure_form.addRow("增益:", self.gain_spin)

        layout.addWidget(exposure_group)

        # 触发模式
        trigger_group = QGroupBox("触发模式")
        trigger_form = QFormLayout(trigger_group)

        self.trigger_combo = QComboBox()
        self.trigger_combo.addItems(["手动触发", "外部信号触发", "连续采集"])
        mode_map = {"manual": 0, "external": 1, "continuous": 2}
        idx = mode_map.get(self.camera_config.get("trigger_mode", "manual"), 0)
        self.trigger_combo.setCurrentIndex(idx)
        trigger_form.addRow("触发模式:", self.trigger_combo)

        self.pixel_combo = QComboBox()
        self.pixel_combo.addItems(["Mono8", "Mono10", "Mono12", "RGB8"])
        idx = self.pixel_combo.findText(
            self.camera_config.get("pixel_format", "Mono8")
        )
        self.pixel_combo.setCurrentIndex(max(idx, 0))
        trigger_form.addRow("像素格式:", self.pixel_combo)

        layout.addWidget(trigger_group)

        # 按钮
        btn_layout = QHBoxLayout()
        self.test_btn = QPushButton("测试连接")
        self.test_btn.clicked.connect(self._test_connection)
        btn_layout.addWidget(self.test_btn)

        btn_layout.addStretch()

        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self._save_and_close)
        btn_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------------

    def _test_connection(self):
        """测试相机连接（使用 OpenCV）。"""
        try:
            import cv2
            cap = cv2.VideoCapture(0)  # 使用默认相机
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    QMessageBox.information(
                        self, "连接成功",
                        f"相机已连接\n分辨率: {w}×{h}\n"
                        f"下一步可设置具体参数。",
                    )
                else:
                    QMessageBox.warning(
                        self, "连接异常",
                        "相机已打开但无法读取帧。\n请检查镜头盖是否移除。",
                    )
                cap.release()
            else:
                QMessageBox.warning(
                    self, "连接失败",
                    "未检测到相机。\n请检查:\n"
                    "  - USB 连接\n"
                    "  - 驱动是否安装\n"
                    "  - 是否有其他程序占用相机",
                )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"相机初始化失败:\n{e}")

    def _save_and_close(self):
        """保存配置并关闭。"""
        trigger_map = {0: "manual", 1: "external", 2: "continuous"}
        self.camera_config.update({
            "driver": "opencv" if self.driver_combo.currentIndex() == 0 else "harvesters",
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "exposure_us": int(self.exposure_spin.value()),
            "gain": self.gain_spin.value(),
            "trigger_mode": trigger_map[self.trigger_combo.currentIndex()],
            "pixel_format": self.pixel_combo.currentText(),
        })
        self.accept()
