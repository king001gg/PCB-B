"""输入验证与类型检查工具。"""

import os
import numpy as np
from typing import Any, Tuple


def validate_image(image: np.ndarray, name: str = "image") -> None:
    """验证是否为有效图像 numpy 数组。

    Args:
        image: 待验证的 numpy 数组。
        name: 参数名（用于错误消息）。

    Raises:
        TypeError: 不是 numpy 数组。
        ValueError: ndim 不是 2 或 3，或 dtype 不支持，或尺寸 ≤ 0。
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} 必须是 numpy.ndarray，收到 {type(image).__name__}")
    if image.ndim not in (2, 3):
        raise ValueError(f"{name} 维度必须为 2 或 3，收到 {image.ndim}")
    if image.size == 0:
        raise ValueError(f"{name} 尺寸不能为零")
    if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise ValueError(
            f"{name} dtype 不支持: {image.dtype}（期望 uint8/uint16/float32/float64）"
        )


def validate_file_path(path: str, must_exist: bool = True) -> str:
    """验证并规范化文件路径。

    Args:
        path: 文件路径字符串。
        must_exist: 是否要求文件必须存在。

    Returns:
        规范化后的绝对路径。

    Raises:
        FileNotFoundError: must_exist=True 且文件不存在。
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"路径不能为空: {path!r}")
    abs_path = os.path.abspath(path)
    if must_exist and not os.path.isfile(abs_path):
        raise FileNotFoundError(f"文件不存在: {abs_path}")
    return abs_path


def validate_config(config: dict) -> dict:
    """验证配置字典的基本结构。

    不会修改配置，只检查必需键是否存在。

    Args:
        config: 配置字典。

    Returns:
        未经修改的配置字典（通过验证后）。

    Raises:
        ValueError: 缺少必需键或值无效。
    """
    if not isinstance(config, dict):
        raise TypeError(f"config 必须是 dict，收到 {type(config).__name__}")

    # 检查顶层键
    required_top = ["system", "inspection", "output"]
    for key in required_top:
        if key not in config:
            raise ValueError(f"config 缺少顶层键: '{key}'")

    # 检查 inspection 子键
    insp = config["inspection"]
    for key in ["preprocessing", "texture", "defects", "classifier", "quality"]:
        if key not in insp:
            raise ValueError(f"config.inspection 缺少键: '{key}'")

    return config


def validate_config_path(path: str) -> str:
    """验证配置文件路径。

    与 validate_file_path 相同但提供更有意义的错误消息。
    """
    if not path.lower().endswith((".yaml", ".yml")):
        raise ValueError(f"配置文件必须是 YAML 格式: {path}")
    return validate_file_path(path, must_exist=True)
