"""工业相机抽象接口。

支持两种驱动模式：
    - OpenCV (cv2.VideoCapture) — 通用 USB/GigE 相机
    - GenICam (harvesters)  — 工业 GigE Vision / USB3 Vision 相机

使用策略模式，通过驱动标识符选择后端实现。

海康机器人 MVS（MV-系列工业相机）走 GenICam 这条路：MVS 安装后会在
Runtime 目录下提供 GenTL producer（MvProducerGEV.cti / MvProducerU3V.cti），
harvesters 作为 GenTL consumer 加载它即可，无需海康自家的 Python 绑定。
前置条件是**必须装 MVS**（含网卡驱动），否则枚举不到设备。
用 tools/check_camera.py 可逐项排查环境。

颜色约定：
    CameraBase.acquire() 统一返回 BGR（三通道）或灰度（单通道），与 OpenCV
    习惯一致。转换为项目内部使用的 RGB 由调用方负责 —— GUI 侧统一在
    ui/camera_worker.py 的边界处转换，不要在多处各转一次。
"""

from abc import ABC, abstractmethod
import glob
import os
import struct
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# GenTL producer 探测
# ============================================================================

#: 各厂商 GenTL producer 的常见安装位置。
#: 海康 MVS 在 64 位 Windows 上装到 Program Files (x86)，新版可能装到
#: Program Files，故两处都搜。Runtime 下按平台分子目录，用 * 通配以兼容版本差异。
#:
#: 注意海康的实际安装路径是 `Common Files\MVS\Runtime\Win64_x64\`，
#: 而不是 `MVS\Runtime\` —— 后者在老版本或某些组件组合下才存在。
#: 实测机器上 `MVS\Runtime\` 目录根本不存在，只有 Common Files 下那份。
_VENDOR_CTI_GLOBS: Tuple[str, ...] = (
    "C:/Program Files*/Common Files/MVS/Runtime/*/*.cti",  # 海康 MVS 实际路径
    "C:/Program Files*/MVS/Runtime/*/*.cti",               # 老版本/部分组件
    "C:/Program Files*/MVS/Development/**/*.cti",          # 部分版本放这里
    "C:/Program Files*/Basler/pylon*/Runtime/x64/*.cti",
    "C:/Program Files*/DAHENG*/Runtime/x64/*.cti",
    "C:/Program Files*/Common Files/GenICam/*/*.cti",
    "/opt/dalsa/*/bin/*.cti",
    "/opt/baumer/*/bin/*.cti",
    "/opt/pylon*/*/bin/*.cti",
)

#: 海康官方 Python 示例（MvImport）的常见位置。
#: 同样只是兜底 —— 主路径走注册表，见 find_mvimport_dirs()。
_VENDOR_MVIMPORT_GLOBS: Tuple[str, ...] = (
    "C:/Program Files*/MVS/Development/Samples/Python/MvImport",
    "C:/Program Files*/MVS/Development/**/MvImport",
    "/opt/MVS/Development/Samples/Python/MvImport",
)

#: 是否加载采集卡（Frame Grabber）的 producer。
#:
#: 海康的 MvFG*Producer*.cti 面向 CameraLink / CoaXPress / XoF 采集卡，
#: 本项目用的是 GigE 面阵相机，加载它们只会拖慢启动、并让 GenTL 层面的
#: 字符串解析 bug 多几个触发点（MVS 的 producer 返回非 UTF-8 字符串，
#: 见下方 GenICamCamera 的说明），扫不出任何设备。
#: 若将来真要接采集卡，把这个开关打开即可。
INCLUDE_FRAMEGRABBER_PRODUCERS: bool = False

#: 路径中的位数标识。判定顺序为先 64 位后 32 位 ——
#: "x86_64" 含有子串 "x86"，必须先判 64 位，否则 Linux 下的 64 位
#: producer 会被误判成 32 位剔除。
_ARCH_TOKENS_64: Tuple[str, ...] = ("Win64_x64", "Win64", "x86_64", "amd64", "x64")
_ARCH_TOKENS_32: Tuple[str, ...] = ("Win32_i86", "Win32_x86", "Win32", "i86", "x86", "ia32")

#: 可能指向 GenTL producer 目录的环境变量。
#: GENICAM_GENTL64_PATH 是 GenICam 标准变量（分号分隔的目录列表），
#: MVS 安装后通常会设置它；MVS_GENICAM_GENTL64_PATH 是海康自己的变量名。
_GENTL_ENV_VARS: Tuple[str, ...] = (
    "GENICAM_GENTL64_PATH",
    "GENICAM_GENTL32_PATH",
    "MVS_GENICAM_GENTL64_PATH",
    "MVS_ROOT",
)


def python_bitness() -> int:
    """当前 Python 解释器的位数（32 或 64）。

    GenTL producer 的位数必须与 Python 一致：64 位 Python 加载 Win32 的
    .cti 会失败，反之亦然。用 32/64 位 Python 调 64 位 MVS 是很常见的坑。
    """
    return struct.calcsize("P") * 8


def _expand_cti_paths(raw: str) -> List[str]:
    """把环境变量值展开为 .cti 文件列表。

    环境变量的值可能是目录（MVS 的风格），也可能是分号分隔的目录列表，
    也可能是直接指向某个 .cti。三种都处理。
    """
    found: List[str] = []

    # 分号与冒号都当分隔符：Windows 环境变量用分号，POSIX 用冒号。
    # 注意 Windows 路径里的盘符冒号（C:\...）不能当分隔符，故先按分号切，
    # 只有在没切出分号且不是 Windows 绝对路径时才按冒号切。
    parts = [raw]
    if os.pathsep in raw:
        parts = raw.split(os.pathsep)
    elif ":" in raw and not (len(raw) > 1 and raw[1] == ":"):
        parts = raw.split(":")

    for part in parts:
        part = part.strip().strip('"')
        if not part:
            continue
        if part.lower().endswith(".cti"):
            if os.path.isfile(part):
                found.append(os.path.normpath(part))
        elif os.path.isdir(part):
            found.extend(
                os.path.normpath(p) for p in glob.glob(os.path.join(part, "*.cti"))
            )

    return found


def find_gentl_producers(verbose: bool = False) -> List[str]:
    """搜索系统中的 GenTL producer（.cti）文件。

    按以下顺序收集并去重：
        1. 环境变量指向的目录（GENICAM_GENTL64_PATH 等）
        2. 各厂商的常见安装位置

    位数不匹配的 producer 会被剔除 —— 例如 64 位 Python 不使用
    Win32_x86 下的 .cti，否则加载时会报难以理解的错误。

    Args:
        verbose: 为 True 时打印搜索过程，供 tools/check_camera.py 诊断用。

    Returns:
        .cti 文件的绝对路径列表（已去重、已排序，64 位路径优先）。
    """
    bitness = python_bitness()
    candidates: List[str] = []

    for var in _GENTL_ENV_VARS:
        value = os.environ.get(var)
        if not value:
            continue
        hits = _expand_cti_paths(value)
        if verbose:
            print(f"  环境变量 {var} = {value!r} → 找到 {len(hits)} 个 .cti")
        candidates.extend(hits)

    for pattern in _cti_search_globs():
        hits = glob.glob(pattern, recursive=True)
        if verbose and hits:
            print(f"  路径模式 {pattern} → 找到 {len(hits)} 个 .cti")
        candidates.extend(os.path.normpath(p) for p in hits)

    # 去重并保持稳定顺序
    seen = set()
    unique: List[str] = []
    for path in candidates:
        key = os.path.normcase(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)

    # 按位数过滤：剔除明显是另一个位数的 producer
    kept: List[str] = []
    rejected_arch: List[str] = []
    for path in unique:
        if _path_arch(path) in (None, bitness):
            kept.append(path)
        else:
            rejected_arch.append(path)

    if verbose and rejected_arch:
        print(f"  跳过 {len(rejected_arch)} 个位数不匹配的 producer "
              f"（当前 Python 为 {bitness} 位）:")
        for p in rejected_arch:
            print(f"    - {p}")

    # 剔除采集卡 producer（GigE 面阵相机用不上，只会拖慢启动）
    if not INCLUDE_FRAMEGRABBER_PRODUCERS:
        fg = [p for p in kept if _is_framegrabber_producer(p)]
        if fg:
            kept = [p for p in kept if not _is_framegrabber_producer(p)]
            if verbose:
                print(f"  跳过 {len(fg)} 个采集卡 producer（GigE 相机不需要）:")
                for p in fg:
                    print(f"    - {os.path.basename(p)}")

    # 同厂商的 producer 按名称排序，保证多次运行顺序一致
    return sorted(kept, key=lambda p: (os.path.basename(p).lower(), p))


def _registry_mvs_roots() -> List[str]:
    """从注册表读出 MVS 的安装根目录。

    海康的安装器**允许改安装路径**。本机就把 MVS 和 SDK Development 装在
    `E:\\APP\\MVS`，而 Runtime 仍在 `C:\\Program Files (x86)\\Common Files\\MVS`
    —— 同一套 MVS 的两个组件可以落在不同盘上。

    只按默认路径硬编码搜索会整个漏掉这类安装（本机就是这么漏的，
    还误判成「开发组件没装」，让人白跑一趟去重装）。
    注册表里的 UninstallString 指向安装根，是唯一可靠的线索。
    """
    try:
        import winreg
    except ImportError:
        return []          # 非 Windows 平台没有注册表

    uninstall = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
    views = (0, getattr(winreg, "KEY_WOW64_64KEY", 0),
             getattr(winreg, "KEY_WOW64_32KEY", 0))

    roots: List[str] = []
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for view in views:
            try:
                key = winreg.OpenKey(hive, uninstall, 0,
                                     winreg.KEY_READ | view)
            except OSError:
                continue
            for i in range(winreg.QueryInfoKey(key)[0]):
                try:
                    sub = winreg.OpenKey(key, winreg.EnumKey(key, i))
                    name = str(winreg.QueryValueEx(sub, "DisplayName")[0])
                    if "MVS" not in name.upper():
                        continue
                    target = str(winreg.QueryValueEx(sub, "UninstallString")[0])
                except OSError:
                    continue
                root = os.path.dirname(target.strip().strip('"'))
                if root and os.path.isdir(root):
                    roots.append(os.path.normpath(root))

    # 去重且保持顺序（同一台机器上四个 MVS 条目常指向同一个根）
    seen = set()
    unique: List[str] = []
    for root in roots:
        key = os.path.normcase(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _cti_search_globs() -> List[str]:
    """汇总所有要搜索的 .cti 路径模式：内置默认位置 + 注册表实际安装位置。"""
    patterns = list(_VENDOR_CTI_GLOBS)
    for root in _registry_mvs_roots():
        # Runtime 与 Development 是 producer 可能出现的两处；用 ** 兜住版本差异
        patterns.append(os.path.join(root, "**", "*.cti"))
    return patterns


def find_mvs_runtime_dirs() -> List[str]:
    """查找含 ``MvCameraControl.dll`` 的运行时目录。

    供 hardware/mvs_camera.py 加载 DLL 用（该后端不用 GenTL，直接用原生 SDK）。

    注意 Runtime 与 Development 可能装在不同盘 —— 本机 Runtime 在
    `C:\\Program Files (x86)\\Common Files\\MVS`，Development 在 `E:\\APP\\MVS`。
    所以逐处探测并验证 DLL 是否真的存在，而不是猜一个路径。
    """
    bitness = python_bitness()
    plat_dir = "Win64_x64" if bitness == 64 else "Win32_i86"
    bin_dir = "win64" if bitness == 64 else "win32"

    candidates: List[str] = []

    # 1) .cti 所在目录 —— 同一份 Runtime 里 .cti 与 .dll 同处
    candidates.extend(os.path.dirname(p) for p in find_gentl_producers(False))

    # 2) 注册表安装根下的几种布局
    for root in _registry_mvs_roots():
        candidates.append(os.path.join(root, "Runtime", plat_dir))
        candidates.append(os.path.join(root, "Development", "Bin", bin_dir))
        candidates.append(os.path.join(root, "Runtime"))
        candidates.append(os.path.join(root, "Development", "Bin"))

    # 3) 默认位置兜底（注册表读不到时，例如绿色解压版）
    for base in ("C:/Program Files (x86)/Common Files/MVS/Runtime",
                 "C:/Program Files/Common Files/MVS/Runtime"):
        candidates.append(os.path.join(base, plat_dir))

    seen = set()
    found: List[str] = []
    for path in candidates:
        key = os.path.normcase(os.path.normpath(path))
        if key in seen:
            continue
        seen.add(key)
        if os.path.isfile(os.path.join(path, "MvCameraControl.dll")):
            found.append(os.path.normpath(path))
    return found


def find_mvimport_dirs() -> List[str]:
    """查找海康官方 Python 示例（MvImport）目录。

    与 .cti 同理，不能假设默认路径 —— 本机在
    `E:\\APP\\MVS\\Development\\Samples\\Python\\MvImport`。

    Returns:
        存在的 MvImport 目录列表（已去重）。
    """
    candidates: List[str] = []

    # 已经在 sys.path 上（用户自己加过）优先
    for entry in sys.path:
        if entry and os.path.isfile(
                os.path.join(entry, "MvCameraControl_class.py")):
            candidates.append(entry)

    for root in _registry_mvs_roots():
        candidates.append(os.path.join(
            root, "Development", "Samples", "Python", "MvImport"))
        candidates.append(os.path.join(root, "Development", "MVFG",
                                       "Samples", "Python", "MvImport"))

    for pattern in _VENDOR_MVIMPORT_GLOBS:
        candidates.extend(glob.glob(pattern, recursive=True))

    seen = set()
    unique: List[str] = []
    for path in candidates:
        key = os.path.normcase(os.path.normpath(path))
        if key not in seen and os.path.isfile(
                os.path.join(path, "MvCameraControl_class.py")):
            seen.add(key)
            unique.append(os.path.normpath(path))
    return unique


def _path_arch(path: str) -> Optional[int]:
    """从路径推断 producer 的位数，判断不出时返回 None。

    先判 64 位再判 32 位：`x86_64` 里含有子串 `x86`，顺序反了会把
    64 位的 producer 误判成 32 位丢掉。
    """
    for token in _ARCH_TOKENS_64:
        if token in path:
            return 64
    for token in _ARCH_TOKENS_32:
        if token in path:
            return 32
    return None


def _is_framegrabber_producer(path: str) -> bool:
    """是否为海康采集卡 producer（MvFGProducer*.cti）。"""
    return os.path.basename(path).upper().startswith("MVFG")


#: GenTL 设备访问状态（DEVICE_ACCESS_STATUS_LIST）。
_ACCESS_STATUS_LABELS: Dict[int, str] = {
    0: "未知",
    1: "可读写（正常）",
    2: "只读",
    3: "无访问权限",
    4: "被占用",
}

#: 明确表示「打不开」的状态码。其余一律不报警 —— 宁可漏报也不要误报，
#: 恒亮的警告灯和恒亮的报警灯一样没有信息量。
_ACCESS_STATUS_BAD: frozenset = frozenset({3, 4})


def _describe_access_status(raw) -> Tuple[str, bool]:
    """把设备访问状态转成（可读文本, 是否可正常打开）。

    harvesters 的 ``access_status`` 是 IntEnum，其 ``str()`` 在 Python 3.11
    起变成纯数字（``"1"``），3.10 及以前是 ``"DeviceAccessStatus.ReadWrite"``。
    早先的代码直接子串匹配 ``"Available"`` —— 两种形态都对不上，
    于是每台设备都被误报成「可能被其他程序占用」。

    这里改成按状态码判断，并且只在明确是「无权限 / 被占用」时才返回 False。
    """
    try:
        code = int(raw)
    except (TypeError, ValueError):
        # 判不出来时不给结论，避免又造一个误报源
        return (str(raw) if raw is not None else "未知"), True

    label = _ACCESS_STATUS_LABELS.get(code, f"未知状态({code})")
    return label, code not in _ACCESS_STATUS_BAD


def describe_producers(paths: List[str]) -> List[Dict[str, str]]:
    """把 producer 路径整理成便于展示的信息（不加载，仅看文件名）。"""
    result = []
    for p in paths:
        name = os.path.basename(p)
        upper = name.upper()
        if "MVPRODUCER" in upper or "MVS" in p.upper():
            vendor = "海康机器人 (MVS)"
        elif "PYLON" in upper.upper():
            vendor = "Basler (pylon)"
        elif "DAHENG" in p.upper() or "GALAXY" in upper:
            vendor = "大恒图像"
        else:
            vendor = "未知厂商"

        if "GEV" in upper:
            transport = "GigE Vision"
        elif "U3V" in upper:
            transport = "USB3 Vision"
        else:
            transport = "未知"

        result.append({
            "path": p,
            "file": name,
            "vendor": vendor,
            "transport": transport,
        })
    return result


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
        #: 最近一次 open() 失败的具体原因，空串表示没有失败。
        #: 后端的失败细节（比如「序列号 XXX 不存在，当前枚举到 [...]」）
        #: 早先只 print 到 stdout —— GUI 程序里用户根本看不到，
        #: 界面上只剩一句泛泛的「请检查相机是否上电」。调用方读这个字段
        #: 就能把真正的原因显示给用户。
        self.last_error: str = ""

    @abstractmethod
    def open(self) -> bool:
        """打开相机并配置参数。

        失败时应返回 False 并把原因写进 ``last_error``。
        """
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
        """设置增益（dB）。"""
        ...

    @property
    def is_open(self) -> bool:
        """相机是否已打开。"""
        return self._is_open

    def get_info(self) -> Dict[str, str]:
        """返回相机信息（型号、序列号等），供界面展示。

        默认实现返回空字典，由各后端按能力覆盖。
        """
        return {}

    def get_actual_params(self) -> Dict[str, object]:
        """读回相机实际生效的参数。

        与配置里「请求的值」不同 —— 相机可能因量程、自动模式或带宽限制
        而未能按请求生效。界面上应展示实际值，否则用户会以为改成功了。
        """
        return {}

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

    注意：UVC 相机的曝光/增益多数需要先关闭自动模式才可手动设置，
    OpenCV 没有暴露 auto 开关，故这两个参数在部分相机上会静默无效。
    """

    def __init__(self, config: dict):
        super().__init__(config)
        camera_cfg = config.get("camera", {})
        # 设备编号统一从 camera.device.index 读；旧配置写在 system.camera_id，
        # 保留回退以免老配置文件失效。
        device_cfg = camera_cfg.get("device", {}) or {}
        self.camera_id = int(
            device_cfg.get("index",
                          config.get("system", {}).get("camera_id", 0)) or 0
        )
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
            self.last_error = (
                f"打不开设备编号 {self.camera_id}。"
                f"该编号下没有可用摄像头，或被其它程序占用。")
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

    def get_actual_params(self) -> Dict[str, object]:
        """读回 OpenCV 上报的实际分辨率。"""
        if self._capture is None or not self._is_open:
            return {}
        import cv2
        return {
            "width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }


# ============================================================================
# GenICam 后端（harvesters）
# ============================================================================

#: 根据触发模式需要下发的 GenICam 节点组合。
#: TriggerMode=Off 时相机自由运行（连续采集）；On 时由 TriggerSource 指定
#: 的信号触发。海康相机的节点名遵循 GenICam SFNC 标准，故无需特殊适配。
_TRIGGER_MODES: Dict[str, Dict[str, object]] = {
    "continuous": {
        "TriggerMode": "Off",
        "AcquisitionMode": "Continuous",
    },
    "software": {
        "TriggerMode": "On",
        "TriggerSource": "Software",
        "AcquisitionMode": "Continuous",
    },
    "external": {
        "TriggerMode": "On",
        "TriggerSource": None,        # 由 trigger_source 配置决定，见 open()
        "TriggerActivation": "RisingEdge",
        "AcquisitionMode": "Continuous",
    },
}


class GenICamCamera(CameraBase):
    """基于 GenICam 标准的工业相机后端。

    使用 harvesters 库，支持：
        - GigE Vision 相机（海康 MV-系列走这条）
        - USB3 Vision 相机
        - 完整的 GenICam 特性树访问

    安装：
        pip install harvesters
        并安装相机厂商的运行时（海康为 MVS），以提供 GenTL producer。

    用法：
        camera = GenICamCamera(config)
        camera.open()
        frame = camera.acquire()

    关于自动曝光/增益：
        海康相机出厂默认为 ExposureAuto=Continuous，此时写 ExposureTime
        会被相机忽略（值能写进去，但不生效）。因此本类在设置曝光与增益前
        一律先关闭对应的自动模式 —— 这是最容易误判「设置成功了」的地方。
    """

    def __init__(self, config: dict):
        super().__init__(config)
        camera_cfg = config.get("camera", {})
        self.width = camera_cfg.get("width", 2448)
        self.height = camera_cfg.get("height", 2048)
        self.exposure_us = camera_cfg.get("exposure_us", 5000)
        self.gain = camera_cfg.get("gain", 1.0)
        self.pixel_format = camera_cfg.get("pixel_format", "Mono8")

        # 触发配置。旧版配置把这几项散在 system.trigger_mode / camera.trigger_source
        # 两处且从未下发到相机（见 docs 实施记录），此处统一从 camera.trigger 读，
        # 并兼容旧的 camera.trigger_source 键。
        trigger_cfg = camera_cfg.get("trigger", {}) or {}
        self.trigger_mode = trigger_cfg.get(
            "mode", camera_cfg.get("trigger_mode", "continuous")
        )
        self.trigger_source = trigger_cfg.get(
            "source", camera_cfg.get("trigger_source", "Line0")
        )

        # 设备选择：留空则取枚举到的第一台。
        # 多相机场景下必须按序列号绑定 —— 网口插拔顺序会变，靠「第一个设备」
        # 会拍错工位。
        device_cfg = camera_cfg.get("device", {}) or {}
        self.device_serial = str(device_cfg.get("serial_number", "") or "")
        self.device_index = int(device_cfg.get("index", 0) or 0)

        self.acquire_timeout_s = float(camera_cfg.get("acquire_timeout_ms", 2000)) / 1000.0

        self._harvester = None
        self._device = None
        self._node_map = None
        self._device_info: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 打开 / 关闭
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """发现并打开目标 GenICam 设备。"""
        try:
            from harvesters.core import Harvester
        except ImportError:
            raise ImportError(
                "harvesters 未安装。安装: pip install harvesters"
            )

        try:
            self._harvester = Harvester()

            cti_files = find_gentl_producers(verbose=False)
            if not cti_files:
                raise RuntimeError(
                    "未找到 GenTL Producer (.cti 文件)。\n"
                    "海康 MV-系列相机需要先安装 MVS（含网卡驱动）。\n"
                    "安装后若仍找不到，请设置环境变量 GENICAM_GENTL64_PATH "
                    "指向 MVS 的 Runtime/Win64_x64 目录。\n"
                    "可运行 tools/check_camera.py 查看详细排查信息。"
                )
            for cti in cti_files:
                self._harvester.add_file(cti, check_existence=True)

            self._harvester.update()

            device_list = self._harvester.device_info_list
            if not device_list:
                raise RuntimeError(
                    "GenTL producer 已加载但未发现相机。\n"
                    "请检查: 相机是否上电、网线是否插好、"
                    "相机与网卡是否在同一网段（海康默认 192.168.1.x）。\n"
                    "可用 MVS 客户端先确认能连上，再回来重试。"
                )

            search_key = self._build_search_key(device_list)
            self._device = self._harvester.create(search_key)
            if self._device is None:
                raise RuntimeError("无法创建设备实例")

            self._node_map = self._device.remote_device.node_map
            self._device_info = self._read_device_info(device_list, search_key)

            # 配置参数（顺序有讲究：先宽高与像素格式，再曝光增益，最后触发）
            self._set_node("Width", self.width)
            self._set_node("Height", self.height)
            self._set_node("PixelFormat", self.pixel_format)
            self._apply_exposure_gain()
            self._apply_trigger()

            # 启动采集
            self._device.start()
            self._is_open = True

            actual = self.get_actual_params()
            print(
                f"[Camera] GenICam 相机已打开 "
                f"({self._device_info.get('model', '?')} "
                f"SN:{self._device_info.get('serial_number', '?')}) "
                f"{actual.get('width', '?')}×{actual.get('height', '?')} "
                f"{actual.get('pixel_format', '?')}"
            )
            return True

        except UnicodeDecodeError as e:
            # 单独识别：这是海康 MVS 的 GenTL producer 不符合规范导致的，
            # 与「相机没插好」「被占用」完全无关，混在通用异常里会把人
            # 引到错误的排查方向上去。
            self.last_error = (
                "harvesters 解码 MVS 的 GenTL producer 时崩溃"
                f"（{e}）。\n"
                "GenTL 规范要求字符串寄存器用 UTF-8，海康的 producer 用的是"
                "系统 ANSI 代码页，属于 producer 侧的兼容性问题。\n"
                "相机与网络本身没问题。请把相机驱动改成 mvs（海康官方 SDK）。"
            )
            print(f"[Camera] GenICam 初始化失败：{self.last_error}")
            self._is_open = False
            self._cleanup()
            return False

        except Exception as e:
            print(f"[Camera] GenICam 初始化失败: {e}")
            self.last_error = str(e)
            self._is_open = False
            self._cleanup()
            return False

    def _build_search_key(self, device_list) -> object:
        """决定打开哪台设备。

        指定了序列号就按序列号找（多相机时必须这样做）；否则按索引取。
        序列号找不到时明确报错，而不是悄悄退回第一台 —— 否则会拍错工位
        却毫无提示。
        """
        if self.device_serial:
            for idx, info in enumerate(device_list):
                if str(getattr(info, "serial_number", "")) == self.device_serial:
                    return idx
            available = ", ".join(
                str(getattr(i, "serial_number", "?")) for i in device_list
            )
            raise RuntimeError(
                f"未找到序列号为 {self.device_serial!r} 的相机。\n"
                f"当前枚举到的序列号: {available}\n"
                f"请核对配置中的 camera.device.serial_number。"
            )

        if self.device_index >= len(device_list):
            raise RuntimeError(
                f"相机索引 {self.device_index} 超出范围，"
                f"当前只枚举到 {len(device_list)} 台设备"
            )
        return self.device_index

    @staticmethod
    def _read_device_info(device_list, search_key) -> Dict[str, str]:
        """读取所选设备的信息，供界面展示与日志。"""
        idx = search_key if isinstance(search_key, int) else 0
        try:
            info = device_list[idx]
        except (IndexError, TypeError):
            return {}
        return {
            "vendor": str(getattr(info, "vendor", "")),
            "model": str(getattr(info, "model", "")),
            "serial_number": str(getattr(info, "serial_number", "")),
            "tl_type": str(getattr(info, "tl_type", "")),
            "display_name": str(getattr(info, "display_name", "")),
        }

    def acquire(self) -> Optional[np.ndarray]:
        """获取一帧 GenICam 图像。

        Returns:
            BGR 三通道或灰度单通道数组；超时或失败返回 None。
        """
        if not self._is_open or self._device is None:
            return None

        try:
            buffer = self._device.fetch(timeout=self.acquire_timeout_s)
        except Exception as e:
            # 取帧超时在连续采集下偶发，交由调用方决定是否计入丢帧
            print(f"[Camera] 帧抓取失败: {e}")
            return None

        if buffer is None:
            return None

        try:
            with buffer:
                component = buffer.payload.components[0]
                return self._to_numpy(component)
        except Exception as e:
            print(f"[Camera] 帧解析失败: {e}")
            return None

    @staticmethod
    def _to_numpy(component) -> Optional[np.ndarray]:
        """把 GenICam component 转成 numpy 数组（BGR 或灰度）。

        统一在此处完成色彩空间归一，保证 acquire() 的输出契约稳定：
        彩图一律 BGR 三通道，黑白一律单通道灰度。
        """
        import cv2

        width = int(component.width)
        height = int(component.height)
        fmt = str(getattr(component, "data_format", "") or "")
        data = component.data

        # Mono8 / 8 位单通道
        if "Mono8" == fmt or ("Mono" in fmt and "8" in fmt and "Packed" not in fmt):
            return data.reshape(height, width).astype(np.uint8)

        # 高位深单通道：截断到 8 位。Mono10/12 存放在 16 位容器里，
        # 右移 2 位即可落到 0~255（Mono10 值域 0~1023，同样适用）。
        if "Mono10" in fmt or "Mono12" in fmt or "Mono16" in fmt:
            raw = data.view(np.uint16).reshape(height, width)
            return (raw >> 2).astype(np.uint8)

        # Bayer：需要去马赛克才能得到彩色
        for bayer, code in (
            ("BayerRG", cv2.COLOR_BayerRG2BGR),
            ("BayerGR", cv2.COLOR_BayerGR2BGR),
            ("BayerGB", cv2.COLOR_BayerGB2BGR),
            ("BayerBG", cv2.COLOR_BayerBG2BGR),
        ):
            if bayer in fmt:
                raw = data.reshape(height, width).astype(np.uint8)
                return cv2.cvtColor(raw, code)

        # RGB8：相机给的是 RGB，而本类的契约是 BGR，必须转换。
        # 这里若不转，下游会拿到红蓝互换的图 —— 且灰度图看不出来，
        # 只有彩色场景才暴露。
        if "RGB8" in fmt:
            img = data.reshape(height, width, 3).astype(np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if "BGR8" in fmt:
            return data.reshape(height, width, 3).astype(np.uint8)

        # 未识别的格式：按通道数猜测，并打印以便补充适配
        if len(data) == height * width:
            print(f"[Camera] 未识别的像素格式 {fmt!r}，按灰度处理")
            return data.reshape(height, width).astype(np.uint8)
        if len(data) == height * width * 3:
            print(f"[Camera] 未识别的像素格式 {fmt!r}，按 BGR 处理")
            return data.reshape(height, width, 3).astype(np.uint8)

        raise ValueError(
            f"无法解析像素格式 {fmt!r}（数据长度 {len(data)}，"
            f"分辨率 {width}×{height}）"
        )

    def release(self) -> None:
        """释放设备。"""
        self._cleanup()
        self._is_open = False
        print("[Camera] GenICam 相机已释放")

    def _cleanup(self) -> None:
        """按依赖顺序释放：先停设备，再重置 harvester。"""
        if self._device is not None:
            try:
                self._device.stop()
            except Exception:
                pass
            try:
                self._device.destroy()
            except Exception:
                pass
            self._device = None
        if self._harvester is not None:
            try:
                self._harvester.reset()
            except Exception:
                pass
            self._harvester = None
        self._node_map = None

    # ------------------------------------------------------------------
    # 参数设置
    # ------------------------------------------------------------------

    def set_exposure(self, exposure_us: float) -> None:
        """设置曝光时间（微秒）。会先关闭自动曝光。"""
        self.exposure_us = exposure_us
        self._apply_exposure_gain()

    def set_gain(self, gain: float) -> None:
        """设置增益。会先关闭自动增益。"""
        self.gain = gain
        self._apply_exposure_gain()

    def _apply_exposure_gain(self) -> None:
        """下发曝光与增益。必须先关自动模式，否则写入被相机忽略。"""
        if self._node_map is None:
            return

        # 顺序不能反：自动模式开着的时候写 ExposureTime 是无效的
        self._set_node("ExposureAuto", "Off")
        self._set_node("ExposureTime", float(self.exposure_us))
        self._set_node("GainAuto", "Off")
        self._set_node("Gain", float(self.gain))

    def _apply_trigger(self) -> None:
        """下发触发模式配置。

        连续采集（continuous）已实测可用；软件触发与外部硬触发本次未接
        信号源验证，节点写入逻辑按 GenICam SFNC 标准实现。
        """
        if self._node_map is None:
            return

        plan = _TRIGGER_MODES.get(self.trigger_mode)
        if plan is None:
            print(f"[Camera] 未知触发模式 {self.trigger_mode!r}，"
                  f"按连续采集处理。可选: {list(_TRIGGER_MODES)}")
            plan = _TRIGGER_MODES["continuous"]

        for name, value in plan.items():
            if name == "TriggerSource" and value is None:
                value = self.trigger_source
            self._set_node(name, value)

    # ------------------------------------------------------------------
    # 节点读写
    # ------------------------------------------------------------------

    def _set_node(self, name: str, value) -> bool:
        """设置 GenICam 节点值。

        Returns:
            是否成功。失败不抛异常 —— 相机型号不同可能缺少某些节点
            （例如无 Gain 的型号），不应因此中断初始化。
        """
        if self._node_map is None:
            return False
        try:
            node = self._node_map.get_node(name)
        except Exception:
            print(f"[Camera] 相机不支持节点 {name}，跳过")
            return False

        try:
            node.value = value
            return True
        except Exception as e:
            print(f"[Camera] 设置 {name}={value!r} 失败: {e}")
            return False

    def _get_node(self, name: str):
        """读取 GenICam 节点值，失败返回 None。"""
        if self._node_map is None:
            return None
        try:
            return self._node_map.get_node(name).value
        except Exception:
            return None

    def get_info(self) -> Dict[str, str]:
        """返回相机型号与序列号等信息。"""
        return dict(self._device_info)

    def get_actual_params(self) -> Dict[str, object]:
        """读回实际生效的参数。

        与配置里请求的值可能不同：相机受量程限制会夹取，自动模式未关会
        覆盖，GigE 带宽不足会降帧率。界面应展示这里的值。
        """
        if self._node_map is None:
            return {}

        params: Dict[str, object] = {}
        for key, node_name in (
            ("width", "Width"),
            ("height", "Height"),
            ("pixel_format", "PixelFormat"),
            ("exposure_us", "ExposureTime"),
            ("gain", "Gain"),
            ("exposure_auto", "ExposureAuto"),
            ("gain_auto", "GainAuto"),
            ("trigger_mode", "TriggerMode"),
            ("trigger_source", "TriggerSource"),
            ("acquisition_mode", "AcquisitionMode"),
        ):
            value = self._get_node(node_name)
            if value is not None:
                params[key] = value
        return params


# ============================================================================
# 设备枚举（供相机设置对话框与诊断脚本使用）
# ============================================================================

def list_genicam_devices(verbose: bool = False) -> List[Dict[str, str]]:
    """枚举当前可用的 GenICam 设备。

    独立于 GenICamCamera 的一次性探测：加载 producer、扫描设备、立即释放。
    供「扫描设备」按钮与 tools/check_camera.py 使用。

    Args:
        verbose: 打印搜索过程。

    Returns:
        设备信息字典列表；找不到 producer 或无设备时返回空列表。
    """
    try:
        from harvesters.core import Harvester
    except ImportError:
        if verbose:
            print("  harvesters 未安装")
        return []

    harvester = Harvester()
    try:
        cti_files = find_gentl_producers(verbose=verbose)
        if not cti_files:
            return []
        for cti in cti_files:
            harvester.add_file(cti, check_existence=True)

        harvester.update()

        devices = []
        for idx, info in enumerate(harvester.device_info_list):
            access_label, access_ok = _describe_access_status(
                getattr(info, "access_status", None))
            devices.append({
                "index": str(idx),
                "vendor": str(getattr(info, "vendor", "")),
                "model": str(getattr(info, "model", "")),
                "serial_number": str(getattr(info, "serial_number", "")),
                "tl_type": str(getattr(info, "tl_type", "")),
                "display_name": str(getattr(info, "display_name", "")),
                "access_status": access_label,
                "access_ok": access_ok,
            })
        return devices
    except Exception as e:
        if verbose:
            print(f"  枚举设备失败: {e}")
        return []
    finally:
        try:
            harvester.reset()
        except Exception:
            pass


# ============================================================================
# 工厂函数
# ============================================================================

#: 走 GenTL 的别名（harvesters 作为消费端）。
#: 注意：海康相机**不推荐**走这条路 —— MVS 的 producer 不符合 GenTL 的
#: UTF-8 要求，实测 create 阶段就抛 UnicodeDecodeError，取回的帧也全是 0。
#: 保留它是为了兼容 Basler / 大恒等其它厂商的相机。
GENICAM_DRIVER_ALIASES = frozenset({"harvesters", "genicam", "hikvision", "hikrobot"})

#: 走海康官方 SDK（MvCameraControl.dll，经 MvImport 封装）的别名。
#: 海康相机应当用这个 —— 不过 GenTL，没有编码问题。
MVS_SDK_DRIVER_ALIASES = frozenset({"mvs", "mvsdk", "mvs_sdk", "hik_sdk", "hikrobot_sdk"})


def create_camera(config: dict) -> CameraBase:
    """根据配置创建对应的相机实例。

    配置键 camera.driver:
        - "mvs" / "mvsdk"                       → MvsCamera（海康官方 SDK，默认）
        - "opencv"                              → OpenCVCamera
        - "harvesters" / "genicam" /
          "hikvision" / "hikrobot"              → GenICamCamera（通用 GenTL）

    未配置时按 mvs 处理。构造本身不加载 SDK —— MvsCamera 只在 open() 里
    才去找 MvCameraControl.dll，所以没装 MVS 的机器也能正常启动，
    失败点推迟到真正打开相机时，并带着 last_error 的具体原因。

    Args:
        config: 完整配置字典。

    Returns:
        CameraBase 实例（尚未 open()）。
    """
    driver = str(config.get("camera", {}).get("driver", "mvs")).lower()

    if driver in MVS_SDK_DRIVER_ALIASES:
        # 延迟导入：没装 MVS 的机器上不应因为 import 就失败
        from hardware.mvs_camera import MvsCamera
        return MvsCamera(config)

    if driver in GENICAM_DRIVER_ALIASES:
        return GenICamCamera(config)

    return OpenCVCamera(config)
