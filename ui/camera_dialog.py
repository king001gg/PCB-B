"""相机设置对话框。

配置工业相机的曝光时间、增益、分辨率、触发模式等参数，
并提供设备扫描与「测试连接」。

驱动下拉里的取值是**真实驱动名**（opencv / mvs / harvesters），存在
itemData 里用 currentData() 读回。早先用 ``currentIndex() == 0`` 做三元
判断，一旦选项增减就会错位 —— 加第三个驱动时必然踩。
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QSpinBox, QDoubleSpinBox, QComboBox, QPushButton,
    QGroupBox, QMessageBox, QLineEdit,
)
from PySide6.QtCore import Qt

from hardware.camera import (
    GENICAM_DRIVER_ALIASES,
    MVS_SDK_DRIVER_ALIASES,
    create_camera,
)


#: (显示名, 驱动名, 说明)。驱动名会写进 config 的 camera.driver。
#: mvs 排第一 —— 它是默认驱动，也是下拉在数据匹配不上时的兜底项。
_DRIVERS = (
    ("mvs — 海康官方 SDK（默认）", "mvs",
     "海康 MV 系列走 MvCameraControl.dll。支持曝光/增益/触发等全部参数。"),
    ("opencv — USB/网口摄像头", "opencv",
     "走 OpenCV VideoCapture。免驱 UVC 相机或笔记本摄像头用这个。"),
    ("harvesters — 通用 GenICam", "harvesters",
     "通用 GenTL 路线，适配 Basler / 大恒等。海康相机不要选：\n"
     "海康的 producer 不符合规范的 UTF-8 要求，会取到全黑帧。"),
)

#: 触发模式显示名 → 配置值。配置值即 camera.trigger.mode。
_TRIGGER_MODES = (
    ("连续采集", "continuous"),
    ("软触发", "software"),
    ("外部信号触发", "external"),
)

#: 像素格式下拉项。MvsCamera 会把 Mono8/RGB8 这类友好名解析成 SDK 枚举名。
#:
#: Bayer 四种必须列出来：很多海康彩色机**不提供** RGB8/BGR8 打包输出，原生只出
#: Bayer。缺了它们，用户会以为"相机不支持彩色"，转而去找别的驱动。
#: 下拉框本身是可编辑的，表里没有的枚举名也能直接填。
_PIXEL_FORMATS = (
    "Mono8", "Mono10", "Mono12",
    "RGB8", "BGR8",
    "BayerRG8", "BayerGB8", "BayerGR8", "BayerBG8",
    "YUV422",
)


class CameraDialog(QDialog):
    """相机参数设置对话框。"""

    def __init__(self, camera_config: dict, parent=None):
        super().__init__(parent)
        self.camera_config = camera_config.copy()
        self.setWindowTitle("相机设置")
        self.resize(460, 520)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ---- 基本设置 ----
        basic_group = QGroupBox("基本设置")
        basic_form = QFormLayout(basic_group)

        self.driver_combo = QComboBox()
        for label, key, tip in _DRIVERS:
            self.driver_combo.addItem(label, key)
            self.driver_combo.setItemData(
                self.driver_combo.count() - 1, tip, Qt.ToolTipRole)
        self._select_by_data(self.driver_combo,
                             self.camera_config.get("driver", "mvs"))
        basic_form.addRow("驱动:", self.driver_combo)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(640, 8192)
        self.width_spin.setValue(self.camera_config.get("width", 2448))
        basic_form.addRow("宽度(px):", self.width_spin)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(480, 8192)
        self.height_spin.setValue(self.camera_config.get("height", 2048))
        basic_form.addRow("高度(px):", self.height_spin)

        self.pixel_combo = QComboBox()
        self.pixel_combo.addItems(_PIXEL_FORMATS)
        # 允许直接填 SDK 枚举名（如 BayerRG8）—— 跟下面触发源下拉框同样的处理。
        #
        # **这句不能省**：下拉框不可编辑时，`findText` 找不到就返回 -1，
        # `setCurrentIndex(max(-1, 0))` 于是落到第 0 项，配置里写的 BayerRG8 /
        # RGB8_Packed 会被显示成、并在确定时**写回** Mono8。症状就是「换了彩色
        # 相机却还是黑白」，而且不报任何错。
        self.pixel_combo.setEditable(True)
        configured = self.camera_config.get("pixel_format", "Mono8")
        idx = self.pixel_combo.findText(configured)
        if idx >= 0:
            self.pixel_combo.setCurrentIndex(idx)
        else:
            self.pixel_combo.setEditText(configured)
        basic_form.addRow("像素格式:", self.pixel_combo)

        layout.addWidget(basic_group)

        # ---- 设备选择 ----
        device_group = QGroupBox("设备选择")
        device_form = QFormLayout(device_group)

        self.index_spin = QSpinBox()
        self.index_spin.setRange(0, 15)
        device_cfg = self.camera_config.get("device", {}) or {}
        self.index_spin.setValue(int(device_cfg.get("index", 0) or 0))
        self.index_spin.setToolTip(
            "opencv 驱动下表示摄像头编号（0 = 第一个）。\n"
            "mvs / harvesters 下用它按枚举顺序选设备。")
        device_form.addRow("设备编号:", self.index_spin)

        self.serial_edit = QLineEdit(str(device_cfg.get("serial_number", "") or ""))
        self.serial_edit.setPlaceholderText("留空 = 取枚举到的第一台")
        self.serial_edit.setToolTip(
            "多台相机时按序列号绑定。\n"
            "网口插拔顺序会变，靠「第一台」会拍错工位。\n"
            "opencv 驱动忽略此项。")
        device_form.addRow("序列号:", self.serial_edit)

        self.scan_btn = QPushButton("扫描设备")
        self.scan_btn.clicked.connect(self._scan_devices)
        device_form.addRow("", self.scan_btn)

        layout.addWidget(device_group)

        # ---- 曝光与增益 ----
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

        # ---- 触发 ----
        trigger_group = QGroupBox("触发")
        trigger_form = QFormLayout(trigger_group)

        self.trigger_combo = QComboBox()
        for label, key in _TRIGGER_MODES:
            self.trigger_combo.addItem(label, key)
        trigger_cfg = self.camera_config.get("trigger", {}) or {}
        # 兼容旧配置：触发模式早先散在 camera.trigger_mode
        self._select_by_data(
            self.trigger_combo,
            trigger_cfg.get("mode", self.camera_config.get("trigger_mode", "continuous")))
        trigger_form.addRow("触发模式:", self.trigger_combo)

        self.source_combo = QComboBox()
        self.source_combo.setEditable(True)   # 允许直接填 Line4 等非常见口
        self.source_combo.addItems(["Line0", "Line1", "Line2", "Line3"])
        self.source_combo.setCurrentText(str(
            trigger_cfg.get("source", self.camera_config.get("trigger_source", "Line0"))))
        self.source_combo.setToolTip("仅「外部信号触发」时有效。")
        trigger_form.addRow("触发源:", self.source_combo)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(100, 60000)
        self.timeout_spin.setSingleStep(100)
        self.timeout_spin.setSuffix(" ms")
        self.timeout_spin.setValue(
            int(self.camera_config.get("acquire_timeout_ms", 2000)))
        trigger_form.addRow("取帧超时:", self.timeout_spin)

        layout.addWidget(trigger_group)

        # ---- 按钮 ----
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
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _select_by_data(combo: QComboBox, value) -> None:
        """按 itemData 选中。找不到就退回第 0 项（不静默选错驱动）。"""
        idx = combo.findData(value)
        combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _form_config(self) -> dict:
        """用**表单当前值**组一份 config，而不是已保存的配置。

        测试连接要验的是用户刚填进去的东西，拿旧 config 测等于没测。
        """
        return {
            "camera": {
                "driver": self.driver_combo.currentData(),
                "width": self.width_spin.value(),
                "height": self.height_spin.value(),
                "exposure_us": int(self.exposure_spin.value()),
                "gain": self.gain_spin.value(),
                "pixel_format": self.pixel_combo.currentText(),
                "device": {
                    "index": self.index_spin.value(),
                    "serial_number": self.serial_edit.text().strip(),
                },
                "trigger": {
                    "mode": self.trigger_combo.currentData(),
                    "source": self.source_combo.currentText().strip() or "Line0",
                },
                "acquire_timeout_ms": self.timeout_spin.value(),
            }
        }

    # ------------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------------

    def _scan_devices(self):
        """枚举设备并把结果展示给用户，选中一项则回填表单。"""
        driver = self.driver_combo.currentData()

        try:
            if driver in MVS_SDK_DRIVER_ALIASES:
                found = self._scan_mvs()
            elif driver in GENICAM_DRIVER_ALIASES:
                found = self._scan_genicam()
            else:
                found = self._scan_opencv()
        except Exception as e:
            QMessageBox.critical(self, "扫描失败", f"枚举设备时出错:\n{e}")
            return

        if not found:
            QMessageBox.warning(
                self, "未发现设备",
                f"{driver} 驱动没有枚举到任何相机。\n\n"
                "请检查:\n"
                "  - 相机是否上电、网线/USB 是否插好\n"
                "  - 是否被 MVS 客户端或其它程序占用\n"
                "  - 可运行 tools/check_camera.py 查看详细原因")
            return

        # 有序列号的（mvs/genicam）让用户挑一台
        picks = [d for d in found if d.get("serial_number")]
        if picks:
            from PySide6.QtWidgets import QInputDialog
            labels = [
                f"{d.get('model', '?')}   SN:{d['serial_number']}"
                + (f"   IP:{d['ip']}" if d.get("ip") else "")
                for d in picks
            ]
            choice, ok = QInputDialog.getItem(
                self, "选择设备", f"枚举到 {len(picks)} 台相机:",
                labels, 0, False)
            if not ok:
                return
            picked = picks[labels.index(choice)]
            self.serial_edit.setText(picked["serial_number"])
            self.index_spin.setValue(int(picked.get("index", 0)))
            QMessageBox.information(
                self, "已选择",
                f"已绑定到 {picked.get('model', '?')}\n"
                f"序列号 {picked['serial_number']}")
        else:
            QMessageBox.information(
                self, "发现设备",
                "\n".join(f"[{d['index']}] {d.get('model', '摄像头')}"
                          for d in found))

    def _scan_mvs(self) -> list:
        from hardware.mvs_camera import list_mvs_devices
        return list_mvs_devices()

    def _scan_genicam(self) -> list:
        from hardware.camera import list_genicam_devices
        return list_genicam_devices()

    @staticmethod
    def _scan_opencv() -> list:
        """试开 0..4 号摄像头，能读出帧的算存在。"""
        import cv2
        found = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            try:
                if cap.isOpened():
                    found.append({"index": i, "model": f"摄像头 {i}"})
            finally:
                cap.release()
        return found

    def _test_connection(self):
        """按表单当前值真开一次相机并抓几帧。"""
        cfg = self._form_config()
        driver = cfg["camera"]["driver"]

        self.test_btn.setEnabled(False)
        self.test_btn.setText("测试中…")
        try:
            camera = create_camera(cfg)
            try:
                if not camera.open():
                    QMessageBox.warning(
                        self, "连接失败",
                        f"{driver} 驱动无法打开相机。\n\n"
                        "请检查:\n"
                        "  - 相机是否上电、线缆是否插好\n"
                        "  - 是否被其它程序占用\n"
                        "  - 可运行 tools/check_camera.py 查看详细原因")
                    return

                frames = []
                for _ in range(3):
                    img = camera.acquire()
                    if img is not None:
                        frames.append(img)
            finally:
                # 无论成败都要放掉设备，否则预览启动时会抢不到相机
                camera.release()

            if not frames:
                QMessageBox.warning(
                    self, "取帧失败",
                    "相机已打开，但 3 次取帧全部超时。\n\n"
                    "请检查:\n"
                    "  - 曝光时间是否过短\n"
                    "  - 网卡巨帧（Jumbo Frame）是否开启\n"
                    "  - 网络带宽是否被其它程序挤占")
                return

            info = camera.get_info()
            frame = frames[0]
            mean = float(frame.mean())
            h, w = frame.shape[:2]

            text = (f"相机已连接\n\n"
                    f"分辨率: {w}×{h}\n"
                    f"通道数: {frame.shape[2] if frame.ndim == 3 else 1}\n"
                    f"灰度均值: {mean:.1f}\n")
            if info:
                text += "\n" + "\n".join(f"{k}: {v}" for k, v in info.items())

            if mean <= 0.0:
                # 全黑是这套硬件上最常见、也最容易被误判成「相机坏了」的故障
                QMessageBox.warning(
                    self, "有信号但全黑",
                    text + "\n\n图像全黑，但相机通信正常。请检查:\n"
                           "  - 镜头盖是否已取下\n"
                           "  - 曝光时间是否过短\n"
                           "  - 光源是否开启")
            else:
                QMessageBox.information(self, "连接成功", text)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"相机初始化失败:\n{e}")
        finally:
            self.test_btn.setEnabled(True)
            self.test_btn.setText("测试连接")

    def _save_and_close(self):
        """保存配置并关闭。"""
        cam = self._form_config()["camera"]
        self.camera_config.update(cam)
        self.accept()
