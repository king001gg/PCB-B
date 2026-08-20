"""热力图可视化组件。

将检测结果（缺陷分布、粗糙度分布、CV 均匀性）渲染为
彩色热力图并嵌入 QWidget 中显示。
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

import cv2


class HeatmapWidget(QWidget):
    """热力图可视化组件。

    支持三种热力图模式：
        - 缺陷分布（defects）：缺陷掩膜累加
        - 粗糙度分布（roughness）：CV 值映射
        - 组合视图（combined）：缺陷 + 粗糙度叠加

    使用 matplotlib 色谱（JET）进行伪彩色渲染。
    """

    COLORMAPS = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "inferno": cv2.COLORMAP_INFERNO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._defect_heatmap: np.ndarray = None
        self._roughness_map: np.ndarray = None
        self._colormap = "jet"
        self._mode = "combined"  # defects | roughness | combined

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # 模式选择
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "缺陷分布热力图",
            "粗糙度均匀性热力图",
            "组合视图",
        ])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_combo)

        # 热力图显示
        self.heatmap_label = QLabel()
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setMinimumSize(200, 200)
        self.heatmap_label.setStyleSheet(
            "background: #2b2b2b; border: 1px solid #555;"
        )
        self.heatmap_label.setText("尚未生成热力图")
        layout.addWidget(self.heatmap_label)

        # 图例
        self.legend_label = QLabel()
        self.legend_label.setMaximumHeight(30)
        self.legend_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.legend_label)

    def set_data(
        self,
        defect_heatmap: np.ndarray = None,
        roughness_map: np.ndarray = None,
    ):
        """设置热力图数据。

        Args:
            defect_heatmap: 缺陷分布热力图 (H, W), [0, 1]。
            roughness_map: 粗糙度 CV 分布图 (H, W), [0, 1]。
        """
        self._defect_heatmap = defect_heatmap
        self._roughness_map = roughness_map
        self._render()

    def _on_mode_changed(self, index: int):
        modes = ["defects", "roughness", "combined"]
        self._mode = modes[index]
        self._render()

    def _render(self):
        """渲染当前选择的热力图。"""
        if self._mode == "defects":
            data = self._defect_heatmap
            title = "缺陷分布"
        elif self._mode == "roughness":
            data = self._roughness_map
            title = "粗糙度均匀性"
        else:
            # combined: 缺陷叠加在粗糙度上
            data = self._blend_maps()
            title = "组合视图"

        if data is None:
            self.heatmap_label.setText("无数据")
            return

        # 伪彩色渲染
        colored = self._apply_colormap(data)

        # numpy → QPixmap
        h, w = colored.shape[:2]
        qimg = QImage(colored.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.heatmap_label.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.heatmap_label.setPixmap(pixmap)
        self.legend_label.setText(f"{title} | 色谱: {self._colormap}")

    def _blend_maps(self) -> np.ndarray:
        """叠加缺陷热力图和粗糙度热力图。"""
        maps = []
        if self._defect_heatmap is not None:
            maps.append(self._defect_heatmap)
        if self._roughness_map is not None:
            maps.append(self._roughness_map)

        if not maps:
            return None

        # 调整尺寸一致
        ref_shape = maps[0].shape
        for i, m in enumerate(maps):
            if m.shape != ref_shape:
                maps[i] = cv2.resize(m, (ref_shape[1], ref_shape[0]))

        blended = np.mean(maps, axis=0)
        # 重新归一化
        b_max = blended.max()
        if b_max > 1e-6:
            blended /= b_max
        return blended

    def _apply_colormap(self, data: np.ndarray) -> np.ndarray:
        """应用 colormap 进行伪彩色渲染。"""
        uint8_data = (np.clip(data, 0, 1) * 255).astype(np.uint8)
        cmap = self.COLORMAPS.get(self._colormap, cv2.COLORMAP_JET)
        colored = cv2.applyColorMap(uint8_data, cmap)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
