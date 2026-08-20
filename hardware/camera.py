"""工业相机抽象接口。

支持两种驱动模式：
    - OpenCV (cv2.VideoCapture) — 通用 USB/GigE 相机
    - GenICam (harvesters)  — 工业 GigE Vision / USB3 Vision 相机

使用策略模式，通过驱动标识符选择后端实现。
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional
import time


# ============================================================================
# 抽象基类
# ============================================================================

class CameraBase(ABC):
    """工业相机抽象基类。

    所有相机后端必须实现 acquire() 和 release() 方法。
    """

    def __init__(self, config: dict):
        self.config = config
        self._is_open = False

    @abstractmethod
    def open(self) -> bool:
        """打开相机并配置参数。"""
        ...

    @abstractmethod
    def acquire(self) -> Optional[np.ndarray]:
        """抓取一帧图像。

        Returns:
            numpy 数组 (H, W, 3) BGR 或 (H, W) 灰度，失败返回 None。
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """释放相机资源。"""
        ...

    @abstractmethod
    def set_exposure(self, exposure_us: float) -> None:
        """设置曝光时间（微秒）。"""
        ...

    @abstractmethod
    def set_gain(self, gain: float) -> None:
        """设置模拟增益。"""
        ...

    @property
    def is_open(self) -> bool:
        """相机是否已打开。"""
        return self._is_open

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()


# ============================================================================
# OpenCV 后端
# ============================================================================

class OpenCVCamera(CameraBase):
    """基于 OpenCV VideoCapture 的相机后端。

    适用于：
        - USB 工业相机（UVC 协议）
        - 部分 GigE Vision 相机（通过 DirectShow 或 V4L2）
        - 笔记本电脑内置摄像头（开发调试用）

    支持的属性：
        - 分辨率（width × height）
        - 曝光时间（us，仅部分相机支持手动设置）
        - 增益（仅部分相机支持）
    """

    def __init__(self, config: dict):
        super().__init__(config)
        camera_cfg = config.get("camera", {})
        self.camera_id = config.get("system", {}).get("camera_id", 0)
        self.width = camera_cfg.get("width", 2448)
        self.height = camera_cfg.get("height", 2048)
        self.exposure_us = camera_cfg.get("exposure_us", 5000)
        self.gain = camera_cfg.get("gain", 1.0)
        self._capture = None

    def open(self) -> bool:
        """打开相机并设置分辨率和曝光参数。"""
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV (cv2) 未安装")

        self._capture = cv2.VideoCapture(self.camera_id)
        if not self._capture.isOpened():
            self._is_open = False
            return False

        # 设置分辨率
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # 设置曝光（OpenCV 的 CAP_PROP_EXPOSURE 单位因平台而异）
        # Windows: -log2(秒)，需转换
        exposure_val = self.exposure_us / 1_000_000.0  # 转为秒
        if exposure_val > 0:
            self._capture.set(cv2.CAP_PROP_EXPOSURE, -np.log2(exposure_val))

        # 设置增益
        self._capture.set(cv2.CAP_PROP_GAIN, self.gain)

        # 预热：丢弃前几帧（某些相机需要稳定时间）
        for _ in range(5):
            self._capture.read()
            time.sleep(0.05)

        self._is_open = True
        actual_w = self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[Camera] OpenCV 相机已打开 "
              f"({actual_w:.0f}×{actual_h:.0f})")
        return True

    def acquire(self) -> Optional[np.ndarray]:
        """抓取一帧。"""
        if not self._is_open or self._capture is None:
            return None

        ret, frame = self._capture.read()
        if not ret:
            return None

        return frame  # BGR

    def release(self) -> None:
        """释放相机。"""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._is_open = False
        print("[Camera] 相机已释放")

    def set_exposure(self, exposure_us: float) -> None:
        """设置曝光时间。"""
        self.exposure_us = exposure_us
        if self._capture is not None and self._is_open:
            exposure_val = exposure_us / 1_000_000.0
            if exposure_val > 0:
                import cv2
                self._capture.set(
                    cv2.CAP_PROP_EXPOSURE,
                    -np.log2(exposure_val),
                )

    def set_gain(self, gain: float) -> None:
        """设置增益。"""
        self.gain = gain
        if self._capture is not None and self._is_open:
            import cv2
            self._capture.set(cv2.CAP_PROP_GAIN, gain)


# ============================================================================
# GenICam 后端（harvesters）
# ============================================================================

class GenICamCamera(CameraBase):
    """基于 GenICam 标准的工业相机后端。

    使用 harvesters 库，支持：
        - GigE Vision 相机
        - USB3 Vision 相机
        - 完整的 GenICam 特性树访问

    安装：
        pip install harvesters

    用法：
        camera = GenICamCamera(config)
        camera.open()
        frame = camera.acquire()
    """

    def __init__(self, config: dict):
        super().__init__(config)
        camera_cfg = config.get("camera", {})
        self.width = camera_cfg.get("width", 2448)
        self.height = camera_cfg.get("height", 2048)
        self.exposure_us = camera_cfg.get("exposure_us", 5000)
        self.gain = camera_cfg.get("gain", 1.0)
        self.pixel_format = camera_cfg.get("pixel_format", "Mono8")
        self.trigger_source = camera_cfg.get("trigger_source", "Line0")

        self._harvester = None
        self._device = None
        self._node_map = None

    def open(self) -> bool:
        """发现并打开第一个 GenICam 设备。"""
        try:
            from harvesters.core import Harvester
        except ImportError:
            raise ImportError(
                "harvesters 未安装。安装: pip install harvesters"
            )

        try:
            self._harvester = Harvester()

            # 加载 GenTL Producer
            # 默认搜索 /opt/dalsa, /opt/baumer 等路径
            # 也可通过环境变量 GENICAM_GENTL64_PATH 指定
            cti_files = self._find_cti_files()
            if not cti_files:
                raise RuntimeError(
                    "未找到 GenTL Producer (.cti 文件)。\n"
                    "请安装相机厂商的 GenICam 驱动并设置 "
                    "GENICAM_GENTL64_PATH 环境变量。"
                )
            for cti in cti_files:
                self._harvester.add_file(cti)

            self._harvester.update()

            # 获取第一个可用设备
            device_list = self._harvester.device_info_list
            if not device_list:
                raise RuntimeError("未发现 GenICam 设备")

            self._device = self._harvester.create({"id_": 0})

            if self._device is None:
                raise RuntimeError("无法创建设备实例")

            self._node_map = self._device.remote_device.node_map

            # 配置参数
            self._set_node("Width", self.width)
            self._set_node("Height", self.height)
            self._set_node("ExposureTime", self.exposure_us)
            self._set_node("Gain", self.gain)
            self._set_node("PixelFormat", self.pixel_format)

            # 启动采集
            self._device.start()
            self._is_open = True

            print(f"[Camera] GenICam 相机已打开 "
                  f"({self.width}×{self.height})")
            return True

        except Exception as e:
            print(f"[Camera] GenICam 初始化失败: {e}")
            self._is_open = False
            if self._harvester:
                self._harvester.reset()
            return False

    def acquire(self) -> Optional[np.ndarray]:
        """获取一帧 GenICam 图像。"""
        if not self._is_open or self._device is None:
            return None

        try:
            import cv2

            with self._device.fetch() as buffer:
                component = buffer.payload.components[0]
                # 根据像素格式转换为 numpy
                width = component.width
                height = component.height

                if "Mono8" in str(component.data_format):
                    img = component.data.reshape(height, width)
                elif "Mono10" in str(component.data_format) or \
                     "Mono12" in str(component.data_format):
                    # 10/12bit → 16bit → 8bit
                    raw = component.data.view(np.uint16).reshape(
                        height, width
                    )
                    img = (raw >> 2).astype(np.uint8)
                elif "RGB8" in str(component.data_format):
                    img = component.data.reshape(height, width, 3)
                else:
                    # 回退：视为 BGR
                    img = component.data.reshape(height, width, 3)

                return img.astype(np.uint8)

        except Exception as e:
            print(f"[Camera] 帧抓取失败: {e}")
            return None

    def release(self) -> None:
        """释放设备。"""
        if self._device is not None:
            self._device.stop()
            self._device.destroy()
            self._device = None
        if self._harvester is not None:
            self._harvester.reset()
            self._harvester = None
        self._is_open = False
        print("[Camera] GenICam 相机已释放")

    def set_exposure(self, exposure_us: float) -> None:
        self.exposure_us = exposure_us
        self._set_node("ExposureTime", exposure_us)

    def set_gain(self, gain: float) -> None:
        self.gain = gain
        self._set_node("Gain", gain)

    def _set_node(self, name: str, value) -> None:
        """设置 GenICam 节点值。"""
        if self._node_map is None:
            return
        try:
            node = self._node_map.get_node(name)
            node.value = value
        except Exception as e:
            print(f"[Camera] 设置 {name}={value} 失败: {e}")

    @staticmethod
    def _find_cti_files() -> list:
        """搜索系统中的 GenTL Producer (.cti) 文件。"""
        import glob
        import os

        search_paths = [
            os.environ.get("GENICAM_GENTL64_PATH", ""),
            "/opt/dalsa/*/bin",
            "/opt/baumer/*/bin",
            "/opt/pylon*/*/bin",
            "C:/Program Files/Basler/pylon*/Runtime/x64",
            "C:/Program Files/DAHENG*/Runtime/x64",
        ]
        cti_files = []
        for path in search_paths:
            if path and os.path.exists(path):
                cti_files.extend(glob.glob(os.path.join(path, "*.cti")))
        return cti_files


# ============================================================================
# 工厂函数
# ============================================================================

def create_camera(config: dict) -> CameraBase:
    """根据配置创建对应的相机实例。

    配置键 system.camera.driver:
        - "opencv" → OpenCVCamera
        - "harvesters" → GenICamCamera
    """
    driver = config.get("camera", {}).get("driver", "opencv")
    if driver == "harvesters":
        return GenICamCamera(config)
    else:
        return OpenCVCamera(config)
