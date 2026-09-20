"""统计图表面板。

使用 matplotlib 嵌入 PySide6，显示：
    - 合格率趋势折线图
    - 缺陷分布饼图
    - 各维度评分雷达图
    - 历史质量 SPC 控制图
"""

from typing import List
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from PySide6.QtCore import Qt

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from ui.mpl_font import configure_cjk_font

    # 必须在建第一个 Figure 之前设：rcParams 是进程级全局的，设一次就够，
    # 之后所有图都继承。不设的话本文件里的中文标签会全部渲染成空心方块
    # —— matplotlib 默认的 DejaVu Sans 没有任何汉字字形。
    configure_cjk_font()
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from core.quality import QualityReport


class StatsPanel(QWidget):
    """统计图表面板。

    使用 QTabWidget 组织多个图表标签页。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._reports: List[QualityReport] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        if not HAS_MPL:
            from PySide6.QtWidgets import QLabel
            label = QLabel("matplotlib 未安装。\npip install matplotlib")
            label.setAlignment(Qt.AlignCenter)
            self.tabs.addTab(label, "统计")
            return

        # 标签页
        self.yield_tab = QWidget()
        self.pie_tab = QWidget()
        self.radar_tab = QWidget()

        self.tabs.addTab(self.yield_tab, "合格率趋势")
        self.tabs.addTab(self.pie_tab, "缺陷分布")
        self.tabs.addTab(self.radar_tab, "质量雷达图")

        # 初始化空白图表
        self._yield_fig = Figure(figsize=(4, 3), dpi=100)
        self._yield_canvas = FigureCanvas(self._yield_fig)
        y_layout = QVBoxLayout(self.yield_tab)
        y_layout.addWidget(self._yield_canvas)

        self._pie_fig = Figure(figsize=(4, 3), dpi=100)
        self._pie_canvas = FigureCanvas(self._pie_fig)
        p_layout = QVBoxLayout(self.pie_tab)
        p_layout.addWidget(self._pie_canvas)

        self._radar_fig = Figure(figsize=(4, 3), dpi=100)
        self._radar_canvas = FigureCanvas(self._radar_fig)
        r_layout = QVBoxLayout(self.radar_tab)
        r_layout.addWidget(self._radar_canvas)

    def add_report(self, report: QualityReport):
        """添加单次检测报告并更新图表。"""
        self._reports.append(report)
        # 保留最近 100 条记录
        if len(self._reports) > 100:
            self._reports = self._reports[-100:]
        self._update_charts()

    # ------------------------------------------------------------------
    # 图表更新
    # ------------------------------------------------------------------

    def _update_charts(self):
        if not HAS_MPL or not self._reports:
            return

        self._draw_yield_trend()
        self._draw_defect_pie()
        self._draw_quality_radar()

    def _draw_yield_trend(self):
        """绘制合格率趋势折线图。"""
        self._yield_fig.clear()
        ax = self._yield_fig.add_subplot(111)

        n = len(self._reports)
        scores = [r.overall_score for r in self._reports]
        x = list(range(1, n + 1))

        # 折线图
        ax.plot(x, scores, "b-o", markersize=3, linewidth=1, label="评分")
        # 阈值线
        ax.axhline(y=60, color="r", linestyle="--", alpha=0.5, label="OK 阈值")

        ax.set_xlabel("检测序号")
        ax.set_ylabel("综合评分")
        ax.set_title("质量评分趋势")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        self._yield_fig.tight_layout()
        self._yield_canvas.draw()

    def _draw_defect_pie(self):
        """绘制缺陷分布饼图（基于最近一次检测）。"""
        self._pie_fig.clear()
        ax = self._pie_fig.add_subplot(111)

        latest = self._reports[-1]

        labels = ["氧化斑", "磨料嵌入", "未粗化"]
        values = [
            latest.oxidation_percentage,
            latest.embedding_count,
            latest.unroughened_percentage,
        ]
        colors = ["#ff6b6b", "#4ecdc4", "#45b7d1"]

        # 过滤零值
        filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
        if not filtered:
            ax.text(0.5, 0.5, "无缺陷", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.set_title("缺陷分布")
            self._pie_fig.tight_layout()
            self._pie_canvas.draw()
            return

        labels, values, colors = zip(*filtered)

        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("缺陷分布（最近一次）")

        self._pie_fig.tight_layout()
        self._pie_canvas.draw()

    def _draw_quality_radar(self):
        """绘制五维质量雷达图。

        刻意保持五维：色度 / 饱和度是**只监测、不进总分**的指标，把它画进
        这张图会让人以为它参与了评分（雷达图的各项默认是可比的加权项）。
        色度的展示出口是主窗口的结果面板、报告文本与导出表格。
        """
        self._radar_fig.clear()
        ax = self._radar_fig.add_subplot(111, polar=True)

        latest = self._reports[-1]

        # 五个维度
        categories = ["粗糙度均匀", "方向一致", "氧化斑", "磨料嵌入", "未粗化"]
        values = [
            latest.roughness_uniformity / 100.0,
            1.0 - latest.direction_consistency,  # 越低越好 → 反转
            max(0, 1.0 - latest.oxidation_percentage / 20.0),
            max(0, 1.0 - latest.embedding_count / 30.0),
            max(0, 1.0 - latest.unroughened_percentage / 10.0),
        ]

        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        ax.plot(angles, values, "b-", linewidth=2)
        ax.fill(angles, values, "b", alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["20", "40", "60", "80", "100"])
        ax.set_title("质量雷达图", y=1.08)

        self._radar_fig.tight_layout()
        self._radar_canvas.draw()
