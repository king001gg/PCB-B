"""全项目共用的测试夹具。

这里只放**跨模块复用**的东西：合成图像、配置工厂、临时目录。
单个测试文件自己的夹具留在该文件里，不要往这里堆。

合成图像一律用**带种子的随机数**生成，不用 data/samples 里的真实图：
    - 可复现：同一个种子永远得到同一张图，回归失败能稳定重现
    - 不依赖外部文件：真实图没进版本库 / 被替换时测试不会莫名挂掉
    - 快：不用读盘、不用解码

图像内容刻意贴近喷砂后的阻焊面：中等灰度底 + 细密随机纹理（喷砂粗化
留下的微观起伏）。缺陷类夹具在此基础上叠加氧化斑 / 磨料嵌入。
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


# ============================================================================
# 路径与配置
# ============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """仓库根目录。"""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def default_config(project_root: Path) -> dict:
    """config/default.yaml 的真实内容。

    读真文件而不是在测试里另写一份 —— 配置文件的字段名改了、
    层级挪了，这里会第一时间炸，这正是我们要的。
    """
    with open(project_root / "config" / "default.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def config_factory(default_config: dict):
    """在真实配置基础上做深拷贝 + 局部覆盖。

    用法::

        cfg = config_factory(camera={"driver": "mvs"})
        cfg = config_factory(**{"system.mode": "online"})   # 不支持，用下面的

    嵌套覆盖请直接传完整子树，或拿到返回值后自己改 —— 刻意不做
    「点号路径深合并」，那种魔法一旦键名写错会静默新建一个没人读的键。
    """
    import copy

    def _make(**overrides) -> dict:
        cfg = copy.deepcopy(default_config)
        for key, value in overrides.items():
            cfg[key] = copy.deepcopy(value)
        return cfg

    return _make


@pytest.fixture
def tmp_report_dir(tmp_path: Path) -> Path:
    """报告输出的临时目录。"""
    d = tmp_path / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================================
# 合成图像
# ============================================================================

def make_surface(
    height: int = 256,
    width: int = 256,
    seed: int = 0,
    base: int = 128,
    texture: float = 18.0,
    oxidation_blobs: int = 0,
    embedded_specks: int = 0,
) -> np.ndarray:
    """合成一张喷砂表面灰度图。

    Args:
        height, width: 图像尺寸。
        seed: 随机种子。相同种子 + 相同参数 → 完全相同的图像。
        base: 底色灰度。喷砂后阻焊面大致在中等灰。
        texture: 微观纹理的标准差。喷砂粗化留下的随机起伏，
            0 表示完全平坦（会被判定为「未粗化」）。
        oxidation_blobs: 叠加的暗色圆形团块数量（氧化斑）。
        embedded_specks: 叠加的亮色小点数量（磨料嵌入）。

    Returns:
        uint8 单通道灰度图。
    """
    rng = np.random.default_rng(seed)
    img = np.full((height, width), float(base), dtype=np.float32)

    if texture > 0:
        img += rng.normal(0.0, texture, (height, width)).astype(np.float32)

    # 氧化斑：暗色团块，边缘用高斯模糊软化，避免出现现实中不存在的硬边
    for _ in range(oxidation_blobs):
        cy = int(rng.integers(0, height))
        cx = int(rng.integers(0, width))
        r = int(rng.integers(8, max(9, min(height, width) // 6)))
        layer = np.zeros((height, width), dtype=np.float32)
        cv2.circle(layer, (cx, cy), r, 1.0, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), r / 2.0)
        img -= layer * 55.0

    # 磨料嵌入：亮色小点，尺寸远小于氧化斑
    for _ in range(embedded_specks):
        cy = int(rng.integers(0, height))
        cx = int(rng.integers(0, width))
        cv2.circle(img, (cx, cy), int(rng.integers(1, 3)), 235.0, -1)

    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture
def surface_factory():
    """拿到 ``make_surface`` 工厂，便于在一个用例里造多张图。"""
    return make_surface


@pytest.fixture
def sandblasted_image() -> np.ndarray:
    """典型喷砂面：中等灰度 + 细密纹理。正常工艺下的基准图。"""
    return make_surface(seed=42)


@pytest.fixture
def flat_image() -> np.ndarray:
    """完全平坦的表面 —— 喷砂粗化没做到，应被判为不合格。"""
    return make_surface(texture=0.0, seed=1)


@pytest.fixture
def coarse_image() -> np.ndarray:
    """纹理明显偏强的表面 —— 喷砂过度。"""
    return make_surface(texture=45.0, seed=2)


@pytest.fixture
def oxidized_image() -> np.ndarray:
    """带氧化斑的表面。"""
    return make_surface(seed=3, oxidation_blobs=4)


@pytest.fixture
def embedded_image() -> np.ndarray:
    """带磨料嵌入的表面。"""
    return make_surface(seed=4, embedded_specks=12)


@pytest.fixture
def rgb_surface() -> np.ndarray:
    """三通道版本的典型喷砂面。

    颜色用的是低饱和度偏色：真实阻焊面在彩色相机下并不是纯灰，
    若代码错误地把 RGB 当 BGR 处理，灰度转换权重对调后特征值会漂移。
    """
    gray = make_surface(seed=5)
    img = np.stack([gray, np.clip(gray * 0.92, 0, 255), np.clip(gray * 1.05, 0, 255)], axis=-1)
    return img.astype(np.uint8)


# ============================================================================
# 便捷断言
# ============================================================================

def assert_valid_gray_image(img: np.ndarray, height: int = None, width: int = None) -> None:
    """断言这是一张合法的单通道灰度图。

    各测试文件里重复写 ``assert img.dtype == np.uint8`` / ``ndim == 2``
    既啰嗦又容易漏，统一在这里。
    """
    assert isinstance(img, np.ndarray), f"期望 ndarray，实际 {type(img)}"
    assert img.ndim == 2, f"期望单通道，实际 {img.ndim} 维 shape={img.shape}"
    assert img.dtype == np.uint8, f"期望 uint8，实际 {img.dtype}"
    if height is not None:
        assert img.shape[0] == height, f"高度应为 {height}，实际 {img.shape[0]}"
    if width is not None:
        assert img.shape[1] == width, f"宽度应为 {width}，实际 {img.shape[1]}"


@pytest.fixture
def gray_assert():
    """把 ``assert_valid_gray_image`` 作为夹具注入。"""
    return assert_valid_gray_image
