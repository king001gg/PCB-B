"""matplotlib 中文字体配置。

**这个文件解决的问题**：matplotlib 的默认字体是 DejaVu Sans，它**一个汉字字形都
没有**。本仓库此前没有任何字体配置，于是 ``ui/stats_panel.py`` 里所有中文标签
（标题「质量评分趋势」、轴标签「检测序号」/「综合评分」、图例「评分」/「OK 阈值」、
饼图标签、雷达图分类名）在界面上**整片渲染成空心方块**（豆腐块）。数字和字母有
字形，所以只有中文坏掉 —— 这个「中文全坏、数值正常」的组合正是该缺陷的典型外观。

``rcParams`` 是**进程级全局状态**，所以在创建第一个 ``Figure`` 之前设一次即可，
之后所有图（包括本模块之外的）都继承。

**刻意不把字体文件放进仓库**：思源黑体一类体积 10~20 MB，且带许可证问题。改为
探测操作系统已安装的字体；一个都探不到就保持原样（继续显示豆腐块），不抛异常 ——
这是纯展示层的问题，不该让检测流程挂掉。
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# 候选字体，按优先级排列，第一个命中的胜出。
#
# 顺序不是随手排的：
#   - 微软雅黑放第一是明确选定的：屏幕可读性最好，且自带拉丁字形，
#     中英混排（如图例「OK 阈值」）风格统一。
#   - 后面几个是跨平台兜底。**必须探测而不是硬编码单个名字** —— 硬编码
#     SimHei 之类，在一台没装它的机器上会原样复现豆腐块缺陷。
_CJK_CANDIDATES: Tuple[str, ...] = (
    "Microsoft YaHei",      # 微软雅黑（Windows）
    "SimHei",               # 黑体（Windows）
    "Noto Sans SC",         # 思源黑体（Linux 常见）
    "Source Han Sans SC",   # 思源黑体（同一套字的 Adobe 命名）
    "WenQuanYi Micro Hei",  # 文泉驿微米黑（Linux）
    "PingFang SC",          # 苹方（macOS）
    "Hiragino Sans GB",     # 冬青黑体简体中文（macOS）
    "SimSun",               # 宋体（Windows，兜底：衬线在小字号下发虚）
)

# 用来验证「这个字体真的有中文字形」的探针字。
# 直接取自 stats_panel 实际会画出来的字，避免只验了字体名却没验字形。
_PROBE_CHARS = "质量评分趋势检测序号综合氧化斑磨料嵌入未粗化缺陷分布雷达图无"

_configured_family: Optional[str] = None


def _find_font(candidates: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """在候选里找第一个「装了、而且真的有中文字形」的字体。

    Returns:
        ``(family, 字体文件路径)``；都没命中则 ``(None, None)``。

    **两步缺一不可**：

    1. ``findfont(..., fallback_to_default=False)`` —— 不加这个参数，任何找不到的
       名字都会被静默替换成 DejaVu Sans，于是每个候选都「找得到」，这一步形同虚设。
    2. ``FT2Font(path).get_charmap()`` 验字形 —— 字体名对得上但字形缺失（别名、
       占位字体）是真的会发生的，只验名字等于没修。
    """
    try:
        from matplotlib import font_manager
        from matplotlib.ft2font import FT2Font
    except ImportError:  # pragma: no cover - 没装 matplotlib 时无从谈起
        return None, None

    for family in candidates:
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family),
                fallback_to_default=False,
            )
        except Exception:
            continue            # 这台机器没装这个字体
        try:
            charmap = FT2Font(path).get_charmap()
        except Exception:
            continue            # 字体文件损坏 / 格式不支持
        if all(ord(ch) in charmap for ch in _PROBE_CHARS):
            return family, path
        logger.debug("字体 %s 存在但缺少中文字形，跳过", family)
    return None, None


def configure_cjk_font(force: bool = False) -> Optional[str]:
    """把中文字体装进 matplotlib 的全局 rcParams，返回实际选中的字体名。

    找不到任何可用中文字体时返回 ``None``，并**保持 rcParams 原样** —— 界面会继续
    显示豆腐块，但程序照常运行。

    Args:
        force: 忽略缓存重新探测。给测试用（替换候选列表后需要重跑）。
    """
    global _configured_family

    if _configured_family is not None and not force:
        return _configured_family

    try:
        import matplotlib
    except ImportError:  # pragma: no cover
        return None

    family, path = _find_font(_CJK_CANDIDATES)
    if family is None:
        logger.warning(
            "没找到任何带中文字形的字体，图表上的中文将显示为方块。"
            "候选列表见 ui/mpl_font.py 的 _CJK_CANDIDATES。"
        )
        return None

    # **前插**而不是整个替换：保留 DejaVu Sans 等原条目做逐字形回退
    # （matplotlib ≥3.6 支持按字符逐个往下找）。这样中文字体里缺的拉丁符号、
    # 数学符号还能落到 DejaVu 上，不至于为了中文反而丢掉别的字形。
    # 先剔除同名项再前插，重复调用就不会把列表越撑越长。
    existing = [f for f in matplotlib.rcParams["font.sans-serif"] if f != family]
    matplotlib.rcParams["font.sans-serif"] = [family] + existing

    # 关掉「用 U+2212 渲染负号」。**这条不是走过场**：实测候选列表里
    # SimHei / SimSun / DengXian / KaiTi **都没有** U+2212 字形
    # （雅黑和 Noto Sans SC 有）。不关的话，一旦在某台机器上落到黑体或宋体，
    # 负刻度标签会变成豆腐块 —— 修了中文、坏了负号。改用 ASCII 连字符，
    # 任何字体都有这个字形。
    matplotlib.rcParams["axes.unicode_minus"] = False

    _configured_family = family
    logger.info("matplotlib 中文字体已设为 %s (%s)", family, path)
    return family
