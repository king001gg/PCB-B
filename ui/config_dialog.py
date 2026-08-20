"""参数配置对话框。

自动从 YAML 配置生成控件，支持实时调整检测参数。
"""

import yaml
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QPushButton, QTabWidget, QWidget, QGroupBox,
    QScrollArea, QLabel, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt


class ConfigDialog(QDialog):
    """检测参数配置对话框。

    自动解析配置文件结构并生成对应的表单控件。
    支持加载/保存配置到 YAML 文件。
    """

    def __init__(self, config: dict, config_path: str, parent=None):
        super().__init__(parent)
        self.config = config
        self.config_path = config_path
        self._widgets = {}  # key_path → (control, type)
        self.setWindowTitle("检测参数配置")
        self.resize(600, 500)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 标签页
        self.tabs = QTabWidget()

        # --- 预处理 ---
        preproc_tab = QScrollArea()
        preproc_tab.setWidgetResizable(True)
        preproc_widget = QWidget()
        preproc_layout = QVBoxLayout(preproc_widget)

        # ROI
        roi_group = QGroupBox("ROI 提取")
        roi_form = QFormLayout(roi_group)
        self._add_bool_widget(
            roi_form, "inspection.roi.enabled", "启用"
        )
        roi_group.setLayout(roi_form)
        preproc_layout.addWidget(roi_group)

        # Retinex
        retinex_group = QGroupBox("Retinex 光照校正")
        retinex_form = QFormLayout(retinex_group)
        self._add_bool_widget(
            retinex_form, "inspection.preprocessing.retinex.enabled", "启用"
        )
        self._add_spin_widget(
            retinex_form, "inspection.preprocessing.retinex.gain",
            "增益", 1, 255,
        )
        self._add_spin_widget(
            retinex_form, "inspection.preprocessing.retinex.offset",
            "偏置", 1, 255,
        )
        retinex_group.setLayout(retinex_form)
        preproc_layout.addWidget(retinex_group)

        # CLAHE
        clahe_group = QGroupBox("CLAHE 增强")
        clahe_form = QFormLayout(clahe_group)
        self._add_bool_widget(
            clahe_form, "inspection.preprocessing.clahe.enabled", "启用"
        )
        self._add_double_widget(
            clahe_form, "inspection.preprocessing.clahe.clip_limit",
            "裁剪限制", 0.5, 10.0, 0.5,
        )
        clahe_group.setLayout(clahe_form)
        preproc_layout.addWidget(clahe_group)

        preproc_layout.addStretch()
        preproc_tab.setWidget(preproc_widget)
        self.tabs.addTab(preproc_tab, "预处理")

        # --- 缺陷检测 ---
        defect_tab = QScrollArea()
        defect_tab.setWidgetResizable(True)
        defect_widget = QWidget()
        defect_layout = QVBoxLayout(defect_widget)

        # 氧化
        ox_group = QGroupBox("氧化斑检测")
        ox_form = QFormLayout(ox_group)
        self._add_bool_widget(
            ox_form, "inspection.defects.oxidation.enabled", "启用"
        )
        self._add_spin_widget(
            ox_form, "inspection.defects.oxidation.area_min_px",
            "最小面积(px)", 1, 10000,
        )
        ox_group.setLayout(ox_form)
        defect_layout.addWidget(ox_group)

        # 磨料
        emb_group = QGroupBox("磨料嵌入检测")
        emb_form = QFormLayout(emb_group)
        self._add_bool_widget(
            emb_form, "inspection.defects.embedding.enabled", "启用"
        )
        self._add_spin_widget(
            emb_form, "inspection.defects.embedding.threshold_binary",
            "二值化阈值", 1, 255,
        )
        self._add_spin_widget(
            emb_form, "inspection.defects.embedding.area_min_px",
            "最小面积(px)", 1, 100,
        )
        emb_group.setLayout(emb_form)
        defect_layout.addWidget(emb_group)

        # 未粗化
        unrough_group = QGroupBox("未粗化检测")
        unrough_form = QFormLayout(unrough_group)
        self._add_bool_widget(
            unrough_form, "inspection.defects.unroughened.enabled", "启用"
        )
        self._add_double_widget(
            unrough_form, "inspection.defects.unroughened.energy_threshold",
            "能量阈值", 0.01, 1.0, 0.01,
        )
        self._add_spin_widget(
            unrough_form, "inspection.defects.unroughened.area_min_px",
            "最小面积(px)", 10, 10000,
        )
        unrough_group.setLayout(unrough_form)
        defect_layout.addWidget(unrough_group)

        defect_layout.addStretch()
        defect_tab.setWidget(defect_widget)
        self.tabs.addTab(defect_tab, "缺陷检测")

        # --- 质量评估 ---
        quality_tab = QScrollArea()
        quality_tab.setWidgetResizable(True)
        quality_widget = QWidget()
        quality_layout = QVBoxLayout(quality_widget)

        quality_group = QGroupBox("质量阈值")
        quality_form = QFormLayout(quality_group)
        self._add_double_widget(
            quality_form, "inspection.quality.roughness_cv_max",
            "粗糙度 CV 上限", 0.05, 2.0, 0.05,
        )
        self._add_double_widget(
            quality_form, "inspection.quality.oxidation_max_pct",
            "氧化斑上限(%)", 1.0, 50.0, 1.0,
        )
        self._add_spin_widget(
            quality_form, "inspection.quality.embedding_max_count",
            "磨料嵌入上限(个)", 1, 100,
        )
        self._add_double_widget(
            quality_form, "inspection.quality.unroughened_max_pct",
            "未粗化上限(%)", 0.5, 20.0, 0.5,
        )
        self._add_double_widget(
            quality_form, "inspection.quality.ok_score_threshold",
            "OK 阈值", 10.0, 100.0, 5.0,
        )
        quality_group.setLayout(quality_form)
        quality_layout.addWidget(quality_group)
        quality_layout.addStretch()
        quality_tab.setWidget(quality_widget)
        self.tabs.addTab(quality_tab, "质量评估")

        layout.addWidget(self.tabs)

        # 按钮行
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_config)
        btn_layout.addWidget(self.save_btn)

        self.export_btn = QPushButton("导出...")
        self.export_btn.clicked.connect(self.export_config)
        btn_layout.addWidget(self.export_btn)

        self.load_btn = QPushButton("导入...")
        self.load_btn.clicked.connect(self.import_config)
        btn_layout.addWidget(self.load_btn)

        btn_layout.addStretch()

        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.ok_btn)

        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # 控件工厂
    # ------------------------------------------------------------------

    def _add_bool_widget(self, form: QFormLayout, key: str, label: str):
        """添加布尔控件。"""
        cb = QCheckBox()
        val = self._get_nested(self.config, key)
        cb.setChecked(bool(val))
        form.addRow(label, cb)

    def _add_spin_widget(
        self, form: QFormLayout, key: str, label: str,
        min_val: int, max_val: int,
    ):
        """添加整数 spinbox。"""
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        val = self._get_nested(self.config, key)
        if val is not None:
            spin.setValue(int(val))
        form.addRow(label, spin)

    def _add_double_widget(
        self, form: QFormLayout, key: str, label: str,
        min_val: float, max_val: float, step: float,
    ):
        """添加浮点 spinbox。"""
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        val = self._get_nested(self.config, key)
        if val is not None:
            spin.setValue(float(val))
        form.addRow(label, spin)

    @staticmethod
    def _get_nested(d: dict, dotted_key: str, default=None):
        """获取嵌套字典值，用 '.' 分隔键路径。"""
        keys = dotted_key.split(".")
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k)
            else:
                return default
        return d if d is not None else default

    # ------------------------------------------------------------------
    # 保存 / 加载
    # ------------------------------------------------------------------

    def save_config(self):
        """保存配置到文件。"""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            QMessageBox.information(self, "保存成功", f"配置已保存至:\n{self.config_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def export_config(self):
        """导出配置到指定文件。"""
        path, _ = QFileDialog.getSaveFileName(
            self, "导出配置", "config_export.yaml",
            "YAML 文件 (*.yaml *.yml)",
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    yaml.dump(self.config, f, default_flow_style=False,
                              allow_unicode=True)
                QMessageBox.information(self, "导出成功", f"配置已导出至:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", str(e))

    def import_config(self):
        """从文件导入配置。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "导入配置", "",
            "YAML 文件 (*.yaml *.yml)",
        )
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    imported = yaml.safe_load(f)
                self.config.update(imported)
                QMessageBox.information(self, "导入成功", f"配置已导入:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "导入失败", str(e))
