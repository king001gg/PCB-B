"""工艺参数监测面板 —— 主窗口右侧常驻区块。

移植自电磁继电器动触簧表面缺陷识别系统的「实时分析数据」区域，
展示 8 级口径的 6 个 GLCM 特征、工艺判定、报警指示灯与维护建议。

与继电器版本的差异：
    - 沿用 PCB 现有浅色主题，不使用继电器的深色主题
    - 指示灯样式自包含地完整指定，不依赖全局样式表继承
    - 新增 reset()，加载新图像时清除上一次的报警残留
    - 报警文本在最低区间显示「异常」而非「正常」（见 core/process_monitor.py）
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit,
)

from core.process_monitor import ProcessGLCMFeatures, ProcessVerdict

#: 报警灯闪烁周期（毫秒）
FLASH_INTERVAL_MS = 500

#: 指示灯的熄灭底色
LED_IDLE_COLOR = "#d0d0d0"

#: 六个特征的显示顺序与中文名
FEATURE_ROWS = (
    ("contrast", "对比度"),
    ("correlation", "相关性"),
    ("energy", "能量值"),
    ("dissimilarity", "差异性"),
    ("homogeneity", "同质性"),
    ("asm", "ASM值"),
)

VALUE_PLACEHOLDER = "等待分析..."
SUGGESTION_PLACEHOLDER = "等待系统分析结果..."


class ProcessPanel(QGroupBox):
    """「实时分析数据」常驻区块。

    置于右侧标签页上方，切换标签页时报警灯始终可见。

    Usage:
        panel.update_features(features)
        panel.update_verdict(monitor.evaluate(features))
    """

    def __init__(self, parent=None):
        super().__init__("实时分析数据", parent)

        self._value_edits = {}
        self._verdict = None
        self._flash_on = False

        self._build_ui()

        self._flash_timer = QTimer(self)
        self._flash_timer.timeout.connect(self._toggle_flash)

        self.reset()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # --- 六个 GLCM 特征 ---
        grid = QGridLayout()
        grid.setSpacing(6)

        for row, (key, label_text) in enumerate(FEATURE_ROWS):
            label = QLabel(f"{label_text}:")
            label.setFixedWidth(56)

            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText(VALUE_PLACEHOLDER)

            grid.addWidget(label, row, 0)
            grid.addWidget(edit, row, 1)
            self._value_edits[key] = edit

        layout.addLayout(grid)

        # --- 工艺判定 ---
        level_row = QHBoxLayout()
        level_label = QLabel("工艺判定:")
        level_label.setFixedWidth(56)
        self._level_edit = QLineEdit()
        self._level_edit.setReadOnly(True)
        self._level_edit.setAlignment(Qt.AlignCenter)
        level_row.addWidget(level_label)
        level_row.addWidget(self._level_edit)
        layout.addLayout(level_row)

        # --- 报警指示灯 ---
        alarm_row = QHBoxLayout()
        alarm_label = QLabel("报警状态:")
        alarm_label.setFixedWidth(56)
        self._led = QLabel()
        self._led.setFixedSize(18, 18)
        alarm_row.addWidget(alarm_label)
        alarm_row.addWidget(self._led)
        alarm_row.addStretch(1)
        layout.addLayout(alarm_row)

        # --- 维护建议 ---
        self._suggestion_edit = QLineEdit()
        self._suggestion_edit.setReadOnly(True)
        suggestion_label = QLabel("智能维护建议:")
        layout.addWidget(suggestion_label)
        layout.addWidget(self._suggestion_edit)

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def update_features(self, features: ProcessGLCMFeatures) -> None:
        """填入 6 个 GLCM 特征值。"""
        if features is None:
            return
        for key, _ in FEATURE_ROWS:
            self._value_edits[key].setText(f"{getattr(features, key):.4f}")

    def update_verdict(self, verdict: ProcessVerdict) -> None:
        """填入工艺判定，并按需启动报警闪烁。"""
        if verdict is None:
            return

        self._verdict = verdict

        self._level_edit.setText(verdict.level)
        self._level_edit.setStyleSheet(f"color: {verdict.color}; font-weight: bold;")
        self._suggestion_edit.setText(verdict.suggestion)

        if verdict.alarm:
            self._flash_on = False
            self._flash_timer.start(FLASH_INTERVAL_MS)
            self._toggle_flash()   # 立即点亮，避免首个半周期显示为熄灭态
        else:
            self._stop_flash()
            self._set_led(verdict.color)

    def reset(self) -> None:
        """清空面板并停止报警闪烁。

        加载新图像或切换模式时调用，避免上一次的报警状态残留。
        """
        self._stop_flash()
        self._verdict = None

        for edit in self._value_edits.values():
            edit.clear()
            edit.setStyleSheet("")

        self._level_edit.setText("---")
        self._level_edit.setStyleSheet("")
        self._suggestion_edit.setText(SUGGESTION_PLACEHOLDER)
        self._set_led(LED_IDLE_COLOR)

    @property
    def is_flashing(self) -> bool:
        """报警灯是否处于闪烁状态。"""
        return self._flash_timer.isActive()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _toggle_flash(self) -> None:
        """报警灯亮灭交替。"""
        self._flash_on = not self._flash_on
        if self._flash_on:
            color = self._verdict.color if self._verdict else "#e74c3c"
            self._set_led(color)
        else:
            self._set_led("#ffffff")

    def _stop_flash(self) -> None:
        self._flash_timer.stop()
        self._flash_on = False

    def _set_led(self, color: str) -> None:
        """设置指示灯颜色。

        样式完整指定（背景 + 边框 + 圆角），不依赖全局样式表继承。
        继电器原实现只覆盖 background-color，其余属性从全局样式表继承，
        导致内联样式与全局样式互相干扰且无法恢复；此处刻意避免。
        """
        self._led.setStyleSheet(
            f"background-color: {color};"
            f"border: 1px solid #888;"
            f"border-radius: 9px;"
        )
