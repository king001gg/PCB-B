"""matplotlib 中文字体配置测试（``ui/mpl_font.py``）。

**缺陷背景**：matplotlib 默认字体 DejaVu Sans 没有任何汉字字形，中文标签会整片
渲染成空心方块（豆腐块），而数值正常 —— 用户截图里看到的就是这个。

**这个文件里带牙的是哪几条**：只断言「rcParams 里的字体名设对了」是骗得过的，
改成一个乱写的名字照样绿。所以主测试都落在**字形覆盖**和**真画一遍数警告**上：

    - ``test_default_font_reproduces_the_bug`` —— 先证明「不配置真的会缺字」。
      没有这条，下面那些「没有警告」的断言可能只是永远为真的空测试。
    - ``test_no_missing_glyph_warning_on_real_labels`` —— 真建 Figure、真 draw，
      数 matplotlib 报了几条 ``missing from font``。
    - ``test_every_plotted_chinese_string_is_covered`` —— 把 ``stats_panel.py``
      里所有会被画出来的中文字面量抽出来逐个验字形，**日后新加中文标签自动纳入
      覆盖**，不用记得回来改测试。
    - ``test_a_font_without_cjk_glyphs_is_rejected_even_though_it_exists`` ——
      钉住「必须验字形、不能只验名字」这个决定。
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ast
import warnings
from pathlib import Path

import pytest

# 导入 ui.* 会连带拉起整个 GUI 包（ui/__init__.py 是急切导入，会拉 main_window、
# camera_dialog 等），所以先确保 Qt 可用、且走离屏平台。
pytest.importorskip("PySide6.QtWidgets")

matplotlib = pytest.importorskip("matplotlib")

from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ft2font import FT2Font

from ui import mpl_font
from ui.mpl_font import configure_cjk_font

# 注意：导入 ui.stats_panel 时它会调 matplotlib.use("QtAgg")，把后端从 Agg 切走。
# 本文件全部用显式的 FigureCanvasAgg 建画布、不碰 pyplot，所以后端是谁无所谓。
STATS_PANEL = Path(__file__).resolve().parents[1] / "ui" / "stats_panel.py"

# stats_panel 里一定会画出来的中文，用来确认抽取逻辑没坏。
KNOWN_LABELS = (
    "质量评分趋势", "检测序号", "综合评分", "评分", "OK 阈值",
    "缺陷分布", "缺陷分布（最近一次）", "质量雷达图", "无缺陷",
    "氧化斑", "磨料嵌入", "未粗化", "粗糙度均匀", "方向一致",
)


# --------------------------------------------------------------------------
# 隔离：rcParams 是进程级全局状态，每个用例跑完必须还原，
# 否则「取消配置」的反向用例会把后面的用例一起带坏。
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _restore_matplotlib_state():
    saved_family = mpl_font._configured_family
    saved_list = list(rcParams["font.sans-serif"])
    saved_minus = rcParams["axes.unicode_minus"]
    yield
    mpl_font._configured_family = saved_family
    rcParams["font.sans-serif"] = saved_list
    rcParams["axes.unicode_minus"] = saved_minus


# --------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------
def _installed(family: str) -> bool:
    try:
        font_manager.findfont(
            font_manager.FontProperties(family=family), fallback_to_default=False
        )
        return True
    except Exception:
        return False


def _charmap(family: str):
    path = font_manager.findfont(
        font_manager.FontProperties(family=family), fallback_to_default=False
    )
    return FT2Font(path).get_charmap()


def _missing_glyphs(family: str, chars):
    cm = _charmap(family)
    return [c for c in chars if ord(c) not in cm]


def _draw(ax):
    """把 stats_panel 里真实出现过的标签画一遍。"""
    ax.set_title("质量评分趋势")
    ax.set_xlabel("检测序号")
    ax.set_ylabel("综合评分")
    ax.plot([1, 2], [1, 2], label="评分")
    ax.axhline(y=2, color="r", linestyle="--", label="OK 阈值")
    ax.legend()


def _draw_negative_ticks(ax):
    """负刻度 —— 专门用来暴露 U+2212 缺字形。"""
    ax.set_ylim(-5, 5)


def _missing_glyph_warnings(build) -> list:
    """建一张图、画一遍，返回 matplotlib 报的缺字警告文本。"""
    fig = Figure(figsize=(3, 2), dpi=60)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    build(ax)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    return [str(w.message) for w in caught if "missing from font" in str(w.message)]


def _plotting_strings():
    """抽出 ``ui/stats_panel.py`` 里所有可能被画出来的中文字符串字面量。

    规则：全部字符串字面量，**排除文档字符串**（那些永远不会被渲染，拿它们
    卡字形只会误伤）。这样既覆盖 ``labels = [...]`` 这类先赋值后传参的写法，
    也覆盖模块级的标签常量。
    """
    tree = ast.parse(STATS_PANEL.read_text(encoding="utf-8"))

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))

    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            out.append(node.value)
    return out


def _plotted_cjk_chars():
    return sorted({c for s in _plotting_strings() for c in s
                   if "一" <= c <= "鿿"})


# --------------------------------------------------------------------------
# 1. 先证明缺陷是真的 —— 否则下面的断言可能只是空转
# --------------------------------------------------------------------------
class TestTheBugIsReal:
    def test_default_font_reproduces_the_bug(self):
        """不配置时必须缺字。这条挂了说明前提变了，下面几条也就没有意义。"""
        rcParams["font.sans-serif"] = ["DejaVu Sans"]
        mpl_font._configured_family = None
        reported = _missing_glyph_warnings(_draw)
        assert reported, "默认字体下竟然没报缺字？这个缺陷的前提不成立了"
        # 报的应当是汉字码位（「质」= U+8D28），不是别的什么缺字。
        assert any("8D28" in m for m in reported), reported

    def test_configuring_removes_every_missing_glyph_warning(self):
        assert configure_cjk_font(force=True) is not None
        assert _missing_glyph_warnings(_draw) == []

    def test_configuring_also_fixes_negative_ticks(self):
        assert configure_cjk_font(force=True) is not None
        assert _missing_glyph_warnings(_draw_negative_ticks) == []


# --------------------------------------------------------------------------
# 2. 选中字体的字形覆盖
# --------------------------------------------------------------------------
class TestChosenFont:
    def test_chosen_font_actually_has_cjk_glyphs(self):
        """验的是**字形**不是名字 —— 名字随便写一个也「设对了」。"""
        family = configure_cjk_font(force=True)
        assert family is not None
        assert _missing_glyphs(family, mpl_font._PROBE_CHARS) == []

    @pytest.mark.skipif(not _installed("Microsoft YaHei"),
                        reason="本机未装微软雅黑")
    def test_this_machine_picks_microsoft_yahei(self):
        assert configure_cjk_font(force=True) == "Microsoft YaHei"

    def test_microsoft_yahei_is_the_first_choice(self):
        """首选是明确选定的（屏幕可读性 + 自带拉丁字形），别被顺手改掉。"""
        assert mpl_font._CJK_CANDIDATES[0] == "Microsoft YaHei"

    def test_candidate_list_covers_all_three_platforms(self):
        """只留 Windows 字体的话，换到 Linux/macOS 上会原样复现豆腐块。"""
        names = set(mpl_font._CJK_CANDIDATES)
        assert any(n in names for n in ("Microsoft YaHei", "SimHei", "SimSun"))
        assert any(n in names for n in
                   ("Noto Sans SC", "Source Han Sans SC", "WenQuanYi Micro Hei"))
        assert any(n in names for n in ("PingFang SC", "Hiragino Sans GB"))

    def test_a_font_without_cjk_glyphs_is_rejected_even_though_it_exists(self):
        """DejaVu Sans 装了、但一个汉字都没有。

        只验「字体名能不能找到」的实现会把它选中 —— 那就等于没修。这条钉住
        「必须再验一次字形」这个决定。
        """
        assert _installed("DejaVu Sans"), "连 DejaVu Sans 都没有，环境不对"
        assert mpl_font._find_font(["DejaVu Sans"]) == (None, None)

    def test_bogus_family_is_not_silently_substituted(self):
        """findfont 默认会把找不到的名字静默换成 DejaVu Sans，必须关掉。

        不关的话每个候选都「找得到」，第一步的筛选形同虚设。
        """
        assert mpl_font._find_font(["绝无此字体 XYZ"]) == (None, None)
        assert mpl_font._find_font([]) == (None, None)

    def test_a_corrupt_font_file_does_not_crash_the_probe(self, monkeypatch):
        """字体文件坏了就跳过这个候选，不要把异常抛给调用方。

        探测是纯锦上添花的事，任何一个候选出问题都不该让界面起不来。
        """
        import matplotlib.ft2font as ft2font

        def _boom(path):
            raise RuntimeError("字体文件损坏 / 格式不支持")

        monkeypatch.setattr(ft2font, "FT2Font", _boom)
        assert mpl_font._find_font(["Microsoft YaHei"]) == (None, None)


# --------------------------------------------------------------------------
# 2b. 接线：stats_panel 真的调了它吗
# --------------------------------------------------------------------------
class TestStatsPanelIsWiredUp:
    """忘了在 stats_panel 里调用配置函数的话，本文件其它用例**照样全绿**，
    但界面上还是豆腐块 —— 所以接线本身必须单独钉一条。"""

    @pytest.mark.skipif(not _installed("Microsoft YaHei"),
                        reason="本机未装微软雅黑")
    def test_importing_stats_panel_has_already_configured_the_font(self):
        # ui/__init__.py 在收集阶段就导入了 stats_panel，那时它应当已经配好字体。
        # 每个用例跑完都会还原 rcParams，所以这里的顺序无关紧要。
        assert rcParams["font.sans-serif"][0] == "Microsoft YaHei"

    def test_stats_panel_calls_configure(self):
        assert "configure_cjk_font()" in STATS_PANEL.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 3. 配置行为本身
# --------------------------------------------------------------------------
class TestConfigure:
    def test_axes_unicode_minus_is_disabled(self):
        configure_cjk_font(force=True)
        assert rcParams["axes.unicode_minus"] is False

    @pytest.mark.skipif(not _installed("SimHei"),
                        reason="本机没有黑体，演示不了缺 U+2212 的场景")
    def test_unicode_minus_flag_saves_fonts_lacking_the_minus_sign(self):
        """关掉 unicode_minus 的真实理由，用黑体实测。

        黑体（以及宋体 / 等线 / 楷体）**没有 U+2212 字形**。若用 U+2212 画负号，
        这些字体下负刻度会变豆腐块 —— 修了中文、坏了负号。
        """
        rcParams["font.sans-serif"] = ["SimHei"]
        rcParams["axes.unicode_minus"] = True
        assert _missing_glyph_warnings(_draw_negative_ticks), \
            "黑体缺 U+2212，用 U+2212 画负号时应当报缺字"

        rcParams["axes.unicode_minus"] = False
        assert _missing_glyph_warnings(_draw_negative_ticks) == []

    def test_configure_is_idempotent(self):
        chosen = configure_cjk_font(force=True)
        assert chosen is not None
        after_first = list(rcParams["font.sans-serif"])

        configure_cjk_font()
        configure_cjk_font(force=True)

        assert rcParams["font.sans-serif"] == after_first
        assert after_first.count(chosen) == 1

    def test_configure_keeps_the_default_fonts_as_fallback(self):
        """前插而不是整个替换：中文字体缺的拉丁/数学符号还能落到 DejaVu 上。"""
        configure_cjk_font(force=True)
        assert "DejaVu Sans" in rcParams["font.sans-serif"][1:]

    def test_cached_family_is_returned_without_rescanning(self, monkeypatch):
        chosen = configure_cjk_font(force=True)

        calls = []
        monkeypatch.setattr(mpl_font, "_find_font",
                            lambda candidates: (calls.append(candidates), (None, None))[1])

        assert configure_cjk_font() == chosen
        assert calls == [], "第二次调用不该重新探测"

    def test_graceful_when_no_candidate_is_available(self, monkeypatch):
        """一个候选都没有时必须安静退化：不抛异常、不留痕迹。"""
        monkeypatch.setattr(mpl_font, "_CJK_CANDIDATES", ("绝无此字体 XYZ",))
        mpl_font._configured_family = None
        before_list = list(rcParams["font.sans-serif"])
        before_minus = rcParams["axes.unicode_minus"]

        assert configure_cjk_font(force=True) is None

        assert rcParams["font.sans-serif"] == before_list
        assert rcParams["axes.unicode_minus"] == before_minus


# --------------------------------------------------------------------------
# 4. 覆盖 stats_panel 里真正会画出来的每一个中文
# --------------------------------------------------------------------------
class TestPlottedLabels:
    def test_extraction_finds_the_known_labels(self):
        """抽取逻辑自身的哨兵 —— 抽空了的话下面那条会变成永远为真。"""
        strings = _plotting_strings()
        for label in KNOWN_LABELS:
            assert label in strings, f"没抽到 {label!r}，抽取逻辑坏了"

    def test_extraction_leaves_out_docstrings(self):
        """文档字符串不参与字形检查，否则写个生僻字就把测试搞红。"""
        strings = _plotting_strings()
        assert not any("Public API" in s for s in strings)

    def test_every_plotted_chinese_string_is_covered(self):
        chars = _plotted_cjk_chars()
        assert len(chars) > 30, f"只抽到 {len(chars)} 个汉字，抽取逻辑可疑"

        family = configure_cjk_font(force=True)
        assert family is not None
        missing = _missing_glyphs(family, chars)
        assert missing == [], f"{family} 缺这些字形: {''.join(missing)}"

    def test_the_glyph_check_can_actually_fail(self):
        """拿 DejaVu Sans 跑同一套检查必须报缺失。

        否则说明上面那条检查是空转的 —— 它永远返回空列表，什么也没守住。
        """
        assert _missing_glyphs("DejaVu Sans", _plotted_cjk_chars())
