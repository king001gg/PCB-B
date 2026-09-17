"""可缩放图像显示组件。

支持：
    - 鼠标滚轮缩放
    - 鼠标拖拽平移
    - 缺陷叠加层（边界框 + 掩膜半透明叠加）
    - 热力图叠加
    - 适应窗口 / 实际像素切换
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout, QPushButton,
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QWheelEvent, QMouseEvent,
)
from PySide6.QtCore import Qt, QPoint, QRectF

from core.defects import Defect


#: 窄按钮（缩放 +/-）专用样式。
#: 主窗口的全局样式表给 QPushButton 设了 ``padding: 6px 14px``，左右合计
#: 28px。这两个按钮固定宽 30px，扣掉 padding 和 1px 边框后可用内容宽为 0，
#: 文字被整个裁掉 —— 界面上就是两个空白按钮。
#: 这里只把**水平** padding 归零，垂直方向保持 6px：改成 ``padding: 0``
#: 也会修好文字，但按钮高度会从 26px 掉到 14px，和旁边的「适应」「1:1」
#: 参差不齐。边框/底色/圆角仍由全局样式表提供。
_NARROW_BTN_STYLE = "QPushButton { padding: 6px 0; font-weight: bold; }"


class ImageViewer(QWidget):
    """带缩放和叠加的图像显示组件。

    集成于主界面的图像显示区域，可显示原始图像、
    检测标注叠加、热力图覆盖层。

    Attributes:
        image: 当前原始图像 (np.ndarray)。
        overlay: 叠加图像（含缺陷标注）。
        heatmap: 热力图数据 (np.ndarray)。
    """

    MIN_ZOOM = 0.1
    MAX_ZOOM = 10.0
    ZOOM_STEP = 0.1

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: np.ndarray = None
        self._overlay: np.ndarray = None
        self._heatmap: np.ndarray = None
        self._show_heatmap = False
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self._panning = False
        self._last_mouse_pos = QPoint()

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 工具栏
        toolbar = QHBoxLayout()
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedWidth(30)
        self.zoom_in_btn.setStyleSheet(_NARROW_BTN_STYLE)
        self.zoom_in_btn.clicked.connect(lambda: self.zoom(1.25))
        toolbar.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setFixedWidth(30)
        self.zoom_out_btn.setStyleSheet(_NARROW_BTN_STYLE)
        self.zoom_out_btn.clicked.connect(lambda: self.zoom(0.8))
        toolbar.addWidget(self.zoom_out_btn)

        self.fit_btn = QPushButton("适应")
        self.fit_btn.setFixedWidth(50)
        self.fit_btn.clicked.connect(self.fit_to_window)
        toolbar.addWidget(self.fit_btn)

        self.reset_btn = QPushButton("1:1")
        self.reset_btn.setFixedWidth(40)
        self.reset_btn.clicked.connect(self.reset_zoom)
        toolbar.addWidget(self.reset_btn)

        self.heatmap_btn = QPushButton("热力图")
        self.heatmap_btn.setCheckable(True)
        self.heatmap_btn.clicked.connect(self.toggle_heatmap)
        toolbar.addWidget(self.heatmap_btn)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # 图像显示
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet(
            "background: #2b2b2b; border: 1px solid #555;"
        )
        self.image_label.setText("等待加载图像...")
        self.image_label.setStyleSheet(
            "background: #2b2b2b; border: 1px solid #555; color: #888;"
        )
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def set_image(self, image: np.ndarray):
        """设置原始图像。"""
        self._image = image
        self._overlay = None
        self._heatmap = None
        self._refresh()

    def set_overlay(self, overlay: np.ndarray):
        """设置缺陷标注叠加图像。"""
        self._overlay = overlay
        self._refresh()

    def set_heatmap(self, heatmap: np.ndarray):
        """设置热力图。"""
        self._heatmap = heatmap
        if self._show_heatmap:
            self._refresh()

    def toggle_heatmap(self):
        """切换热力图显示。"""
        self._show_heatmap = self.heatmap_btn.isChecked()
        self._refresh()

    def zoom(self, factor: float):
        """缩放图像。"""
        new_zoom = self._zoom * factor
        if self.MIN_ZOOM <= new_zoom <= self.MAX_ZOOM:
            self._zoom = new_zoom
            self._refresh()

    def fit_to_window(self):
        """适应窗口显示。"""
        if self._image is not None:
            label_size = self.image_label.size()
            img_h, img_w = self._image.shape[:2]
            if img_w > 0 and img_h > 0:
                self._zoom = min(
                    label_size.width() / img_w,
                    label_size.height() / img_h,
                )
                self._offset = QPoint(0, 0)
                self._refresh()

    def reset_zoom(self):
        """重置为 1:1。"""
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self._refresh()

    # ------------------------------------------------------------------
    # 鼠标事件
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent):
        """滚轮缩放。"""
        delta = event.angleDelta().y()
        factor = 1.0 + (self.ZOOM_STEP if delta > 0 else -self.ZOOM_STEP)
        self.zoom(factor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._panning = True
            self._last_mouse_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning:
            delta = event.position().toPoint() - self._last_mouse_pos
            self._offset += delta
            self._last_mouse_pos = event.position().toPoint()
            self._refresh()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)

    # ------------------------------------------------------------------
    # 内部渲染
    # ------------------------------------------------------------------

    def _refresh(self):
        """重新渲染并显示图像。"""
        if self._image is None:
            return

        # 决定显示内容
        if self._show_heatmap and self._heatmap is not None:
            display = self._blend_heatmap()
        elif self._overlay is not None:
            display = self._overlay
        else:
            display = self._image

        # numpy → QPixmap
        pixmap = self._ndarray_to_qpixmap(display)
        if pixmap is None:
            return

        # 缩放
        scaled = pixmap.scaled(
            int(pixmap.width() * self._zoom),
            int(pixmap.height() * self._zoom),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        # 偏移
        if self._offset.x() != 0 or self._offset.y() != 0:
            final = QPixmap(scaled.size())
            final.fill(Qt.transparent)
            painter = QPainter(final)
            painter.drawPixmap(self._offset, scaled)
            painter.end()
            scaled = final

        self.image_label.setPixmap(scaled)

    def _blend_heatmap(self) -> np.ndarray:
        """将热力图半透明叠加在原始图像上。"""
        import cv2
        if self._heatmap is None:
            return self._image

        # 调整热力图尺寸
        if self._heatmap.shape[:2] != self._image.shape[:2]:
            heatmap = cv2.resize(
                self._heatmap,
                (self._image.shape[1], self._image.shape[0]),
            )
        else:
            heatmap = self._heatmap

        # 伪彩色热力图
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Alpha 混合
        alpha = 0.4
        if self._image.dtype == np.float64 or self._image.dtype == np.float32:
            image_uint8 = (self._image * 255).astype(np.uint8)
        else:
            image_uint8 = self._image

        if image_uint8.shape[2] == 3:
            blended = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_color, alpha, 0)
        else:
            # 灰度 → RGB
            image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
            blended = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_color, alpha, 0)

        return blended

    def _ndarray_to_qpixmap(self, image: np.ndarray) -> QPixmap:
        """numpy 数组转 QPixmap。"""
        if image is None:
            return None

        if image.dtype == np.float64 or image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)

        h, w = image.shape[:2]
        if image.ndim == 2:
            # 灰度
            qimg = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        elif image.shape[2] == 3:
            qimg = QImage(image.data, w, h, w * 3, QImage.Format_RGB888)
        elif image.shape[2] == 4:
            qimg = QImage(image.data, w, h, w * 4, QImage.Format_RGBA8888)
        else:
            return None

        return QPixmap.fromImage(qimg)
