"""图像采集模块。

支持离线（文件系统）和在线（相机）两种采集模式。
通过 ImageAcquisition 抽象基类统一接口。
"""

import os
import glob
import cv2
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ImageFrame:
    """采集帧数据结构。

    包含图像数据及其元信息。
    """
    image: np.ndarray              # BGR 或灰度图像
    timestamp: str                 # 采集时间 ISO 格式
    source_id: str = ""            # 文件路径 或 相机 ID
    frame_index: int = 0           # 帧序号
    metadata: dict = None           # 额外元数据

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============================================================================
# 抽象基类
# ============================================================================

class ImageAcquisition(ABC):
    """图像采集抽象基类。

    子类必须实现 acquire() 方法，支持上下文管理器协议。
    """

    def __init__(self, config: dict):
        self.config = config
        self.mode = config.get("system", {}).get("mode", "offline")
        self._frame_counter = 0

    @abstractmethod
    def acquire(self) -> Optional[ImageFrame]:
        """采集一帧图像。

        Returns:
            ImageFrame 或 None（无更多图像或采集失败）。
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """重置采集状态（如返回文件列表开头）。"""
        ...

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self) -> None:
        """释放资源（默认空实现）。"""
        pass


# ============================================================================
# 文件采集（离线模式）
# ============================================================================

class FileAcquisition(ImageAcquisition):
    """从文件系统加载图像（离线模式）。

    支持：
        - 单文件加载
        - 文件夹顺序遍历
        - 文件夹循环遍历（模拟连续产线）
    """

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, config: dict):
        super().__init__(config)
        self.image_dir = config.get("system", {}).get(
            "image_dir", "data/samples"
        )
        self.loop = config.get("system", {}).get("offline_loop", False)
        self._file_list: List[str] = []
        self._current_index: int = 0
        self._load_file_list()

    def _load_file_list(self) -> None:
        """扫描目录，收集支持的图像文件。"""
        path = Path(self.image_dir)
        if path.is_file():
            self._file_list = [str(path)]
        elif path.is_dir():
            self._file_list = sorted([
                str(f) for f in path.iterdir()
                if f.suffix.lower() in self.SUPPORTED_EXTS
            ])
        else:
            self._file_list = []

        print(f"[Acquisition] 发现 {len(self._file_list)} 个图像文件 "
              f"({self.image_dir})")

    def acquire(self) -> Optional[ImageFrame]:
        """加载下一张图像。"""
        if not self._file_list:
            print("[Acquisition] 无可用图像文件")
            return None

        if self._current_index >= len(self._file_list):
            if self.loop:
                self._current_index = 0  # 循环
            else:
                return None

        file_path = self._file_list[self._current_index]
        image = cv2.imread(file_path)

        if image is None:
            print(f"[Acquisition] 警告: 无法读取 {file_path}，跳过")
            self._current_index += 1
            return self.acquire()  # 递归尝试下一张

        frame = ImageFrame(
            image=image,
            timestamp=datetime.now().isoformat(),
            source_id=file_path,
            frame_index=self._frame_counter,
            metadata={"filename": os.path.basename(file_path)},
        )

        self._frame_counter += 1
        self._current_index += 1
        return frame

    def reset(self) -> None:
        """重置到文件列表开头。"""
        self._current_index = 0
        self._frame_counter = 0

    @property
    def n_files(self) -> int:
        """文件总数。"""
        return len(self._file_list)


# ============================================================================
# 相机采集（在线模式）
# ============================================================================

class CameraAcquisition(ImageAcquisition):
    """从工业相机采集图像（在线模式）。

    封装 CameraBase 接口，提供 ImageFrame 标准化输出。
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._camera = None

        # 延迟导入避免循环依赖
        from hardware.camera import create_camera
        self._camera = create_camera(config)

        # 尝试打开相机
        if not self._camera.open():
            print("[Acquisition] 警告: 相机打开失败")
        else:
            print("[Acquisition] 相机已就绪")

    def acquire(self) -> Optional[ImageFrame]:
        """从相机抓取一帧。"""
        if self._camera is None or not self._camera.is_open:
            return None

        image = self._camera.acquire()
        if image is None:
            return None

        # 设备编号与 hardware/camera.py 保持一致：先读 camera.device.index，
        # 再回退到旧键 system.camera_id。
        camera_cfg = self.config.get("camera", {}) or {}
        cam_index = (camera_cfg.get("device", {}) or {}).get(
            "index", self.config.get("system", {}).get("camera_id", 0)
        )
        frame = ImageFrame(
            image=image,
            timestamp=datetime.now().isoformat(),
            source_id=f"camera_{cam_index}",
            frame_index=self._frame_counter,
        )

        self._frame_counter += 1
        return frame

    def reset(self) -> None:
        self._frame_counter = 0

    def close(self) -> None:
        if self._camera is not None:
            self._camera.release()
            self._camera = None


# ============================================================================
# 工厂函数
# ============================================================================

def create_acquisition(config: dict) -> ImageAcquisition:
    """根据配置创建采集器。

    配置键 system.mode:
        - "offline" → FileAcquisition
        - "online"  → CameraAcquisition
    """
    mode = config.get("system", {}).get("mode", "offline")
    if mode == "online":
        return CameraAcquisition(config)
    else:
        return FileAcquisition(config)
