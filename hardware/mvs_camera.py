"""海康机器人 MV 系列相机后端 —— 走官方 Python 封装（MvImport）。

为什么不用 harvesters / GenTL
-----------------------------
MVS 的 GenTL producer 不把字符串寄存器按 UTF-8 编码（GenTL 规范要求 UTF-8），
harvesters 作为消费端在解码时抛 ``UnicodeDecodeError``，``create()`` 阶段就失败；
即便绕过那一处，取回的帧也全是 0（实测均值 0.0、标准差 0.0，缓冲区长度却是对的）。
打补丁是在跟一个 C 扩展玩打地鼠。

官方封装直接调 ``MvCameraControl.dll``，完全不过 GenTL，没有这个问题。
实测同一台相机：8/8 帧成功、0 丢帧、无编码报错。

线程约定
--------
``MvCamera`` 句柄不是线程安全的。与其它后端一致：
``open`` / ``acquire`` / ``release`` 必须在同一个线程里调用
（见 ui/camera_worker.py，整个生命周期都在采集线程内）。

像素格式
--------
请求什么由 ``camera.pixel_format`` 决定，**本驱动不做假设**：
彩色输入（含 Bayer / YUV）统一经 SDK 转成 BGR8 三通道，灰度输入转成 Mono8，
以符合 ``CameraBase.acquire()`` 的契约：``(H, W, 3) BGR`` 或 ``(H, W)`` 灰度。

换成彩色相机时**本文件无需改动** —— 相机给什么格式都能落到 BGR。要动的是配置：
``camera.pixel_format`` 得填该型号真正支持的枚举名（很多海康彩色机只出 Bayer，
不出 RGB8/BGR8 打包格式），并且色度指标的绝对值基准需要重新标定。
"""

import os
import sys
from ctypes import POINTER, byref, cast, c_ubyte, memset, sizeof
from typing import Dict, List, Optional, Tuple

import numpy as np

from hardware.camera import (
    CameraBase,
    find_mvimport_dirs,
    find_mvs_runtime_dirs,
    python_bitness,
)


# ============================================================================
# SDK 加载
# ============================================================================

class _Sdk:
    """已加载的海康 SDK 命名空间。

    把三个模块和几个常用常量收在一处，避免到处写 ``global``。
    """

    def __init__(self, mv_camera, header, const, pixel_type):
        self.MvCamera = mv_camera
        self.H = header          # CameraParams_header：结构体与枚举常量
        self.C = const           # CameraParams_const：接口类型、访问模式
        self.PX = pixel_type     # PixelType_header：像素格式常量

        # 像素格式 数值 → 名称，用于回读时显示可读名字
        self.pixel_names: Dict[int, str] = {}
        for name in dir(pixel_type):
            if name.startswith("PixelType_Gvsp_"):
                value = getattr(pixel_type, name)
                if isinstance(value, int):
                    self.pixel_names.setdefault(value, name[len("PixelType_Gvsp_"):])

    @property
    def layered_types(self) -> int:
        """枚举设备时用的接口类型掩码：网口 + USB。"""
        return self.const("MV_GIGE_DEVICE") | self.const("MV_USB_DEVICE")

    def const(self, name: str) -> int:
        """按名字取常量，两个模块都找。

        海康把常量在 ``CameraParams_header`` 与 ``CameraParams_const``
        之间挪过版本 —— 例如 ``MV_TRIGGER_MODE_OFF`` 在 header 里而不是
        const，``MV_GIGE_DEVICE`` 又在 const 里。写死模块名换版本就会炸，
        所以两处都查。
        """
        for module in (self.C, self.H):
            value = getattr(module, name, None)
            if value is not None:
                return value
        raise AttributeError(
            f"海康 SDK 里找不到常量 {name}（已查 CameraParams_const 与 "
            f"CameraParams_header），可能是 MVS 版本差异。")

    def pixel_name(self, value: int) -> str:
        return self.pixel_names.get(int(value), "未知(0x%X)" % int(value))


_SDK: Optional[_Sdk] = None
#: add_dll_directory 返回的句柄必须保持存活，否则目录会立刻失效
_DLL_HANDLES: List[object] = []


def _ensure_sdk() -> _Sdk:
    """定位并加载海康官方 Python 封装，进程内只做一次。

    Raises:
        RuntimeError: 找不到 MvImport 或 DLL 时，附带可操作的排查提示。
    """
    global _SDK
    if _SDK is not None:
        return _SDK

    mvimport_dirs = find_mvimport_dirs()
    if not mvimport_dirs:
        raise RuntimeError(
            "未找到海康官方 Python 示例（MvImport）。\n"
            "它由 MVS 安装包的「SDK / 开发组件」提供，默认可能不勾选。\n"
            "装好后示例位于 <MVS安装目录>\\Development\\Samples\\Python\\MvImport。\n"
            "可运行 tools/check_camera.py 查看当前状态。"
        )

    runtime_dirs = find_mvs_runtime_dirs()
    if not runtime_dirs:
        raise RuntimeError(
            "未找到 MvCameraControl.dll。\n"
            "请确认 MVS 的运行时组件已安装（安装包里的 Runtime）。\n"
            "可运行 tools/check_camera.py 查看当前状态。"
        )

    # DLL 搜索路径必须先于导入设置好 —— MvCameraControl_class 在模块级
    # 就执行 WinDLL("MvCameraControl.dll")，导入时找不到就直接失败。
    for path in runtime_dirs:
        try:
            _DLL_HANDLES.append(os.add_dll_directory(path))
        except (AttributeError, OSError):
            # Python < 3.8 没有 add_dll_directory；此时只能靠 PATH
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")

    mvimport = mvimport_dirs[0]
    if mvimport not in sys.path:
        sys.path.insert(0, mvimport)

    try:
        from MvCameraControl_class import MvCamera
        import CameraParams_const as const
        import CameraParams_header as header
        import PixelType_header as pixel_type
    except ImportError as e:
        raise RuntimeError(
            f"MvImport 目录已找到（{mvimport}）但导入失败：{e}\n"
            f"当前 Python 为 {python_bitness()} 位，"
            f"请确认与 MVS 的位数一致。"
        ) from e

    _SDK = _Sdk(MvCamera, header, const, pixel_type)
    return _SDK


# ============================================================================
# 错误码
# ============================================================================

#: 常见错误码。SDK 不提供错误码转字符串的接口（MvErrorDefine_const 里
#: 只有常量定义），所以这里自己维护一份常用的，其余的按十六进制原样打印。
_ERROR_HINTS: Dict[int, str] = {
    0x80000001: "无效句柄",
    0x80000003: "接口调用顺序错误",
    0x80000004: "无数据",
    0x80000005: "缓冲区已满",
    0x80000006: "句柄无效",
    0x80000007: "不支持的功能",
    0x80000046: "设备访问被拒绝（常被 MVS 客户端等程序独占）",
    0x80000203: "访问被拒绝",
    0x80000302: "USB 设备异常",
    0x80000500: "不支持的像素格式",
    0x80000501: "参数越界",
    0x80000533: "不支持的像素格式",
    0x80000106: "GenICam 访问失败",
}


def _err(code: int) -> str:
    """把错误码转成「十六进制 + 中文说明」。"""
    code = int(code) & 0xFFFFFFFF
    hint = _ERROR_HINTS.get(code)
    return "0x%08X（%s）" % (code, hint) if hint else "0x%08X" % code


# ============================================================================
# 相机后端
# ============================================================================

#: 触发模式名 → (TriggerMode, 触发的 TriggerSource 名或 None)
#: 相机侧的 TriggerSource 用字符串下发，省去维护一张常量表。
_TRIGGER_MODES: Dict[str, Tuple[str, Optional[str]]] = {
    "continuous": ("off", None),
    "off": ("off", None),
    "software": ("on", "Software"),
    "external": ("on", "Line0"),
    "line0": ("on", "Line0"),
    "line1": ("on", "Line1"),
    "line2": ("on", "Line2"),
    "line3": ("on", "Line3"),
}


#: 界面友好名 → 相机侧的 PixelFormat 枚举名。
#: 相机只认 SDK 的枚举字面量，而界面上写 "RGB8" 更自然 ——
#: 不加这张表，``MV_CC_SetEnumValueByString("PixelFormat", "RGB8")``
#: 会返回失败，且失败只打印一行警告，表现出来就是「选了彩色却是黑白」。
_PIXEL_FORMAT_ALIASES: Dict[str, str] = {
    "mono8": "Mono8",
    "mono10": "Mono10",
    "mono12": "Mono12",
    "rgb8": "RGB8_Packed",
    "bgr8": "BGR8_Packed",
    "rgb8_packed": "RGB8_Packed",
    "bgr8_packed": "BGR8_Packed",
    "yuv422": "YUV422_Packed",
}


def _resolve_pixel_format(name: str) -> str:
    """把界面上的像素格式名解析成相机认识的枚举名。

    表里没有的原样返回 —— 用户可能直接填了 SDK 的枚举名（如 ``BayerRG8``），
    那种情况不该被这张表挡住。
    """
    return _PIXEL_FORMAT_ALIASES.get(name.strip().lower(), name)


class MvsCamera(CameraBase):
    """海康 MV 系列相机（GigE / USB3），基于官方 MvImport 封装。"""

    def __init__(self, config: dict):
        super().__init__(config)
        camera_cfg = config.get("camera", {})

        self.width = int(camera_cfg.get("width", 2448))
        self.height = int(camera_cfg.get("height", 2048))
        self.exposure_us = float(camera_cfg.get("exposure_us", 5000))
        self.gain = float(camera_cfg.get("gain", 1.0))
        self.pixel_format = str(camera_cfg.get("pixel_format", "Mono8"))

        # 触发配置：与 GenICamCamera 保持一致，统一从 camera.trigger 读，
        # 并兼容旧的 camera.trigger_source 键。
        trigger_cfg = camera_cfg.get("trigger", {}) or {}
        self.trigger_mode = str(trigger_cfg.get(
            "mode", camera_cfg.get("trigger_mode", "continuous"))).lower()
        self.trigger_source = str(trigger_cfg.get(
            "source", camera_cfg.get("trigger_source", "Line0")))

        # 设备选择：留空按索引取第一台；多相机必须按序列号绑定
        device_cfg = camera_cfg.get("device", {}) or {}
        self.device_serial = str(device_cfg.get("serial_number", "") or "")
        self.device_index = int(device_cfg.get("index", 0) or 0)

        self.acquire_timeout_ms = int(
            camera_cfg.get("acquire_timeout_ms", 2000))

        self._cam = None
        self._sdk: Optional[_Sdk] = None
        self._device_info: Dict[str, str] = {}
        self._frame_out = None      # 复用的 MV_FRAME_OUT，避免每帧分配
        self._convert_buf = None    # 像素格式转换的临时缓冲
        self._convert_buf_len = 0

    # ------------------------------------------------------------------
    # 打开 / 关闭
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """枚举、打开并配置相机。"""
        try:
            sdk = _ensure_sdk()
        except RuntimeError as e:
            print(f"[Camera] {e}")
            return False

        self._sdk = sdk
        try:
            device_list = sdk.H.MV_CC_DEVICE_INFO_LIST()
            ret = sdk.MvCamera.MV_CC_EnumDevices(sdk.layered_types, device_list)
            if ret != 0:
                raise RuntimeError(f"枚举设备失败，错误码 {_err(ret)}")
            if device_list.nDeviceNum == 0:
                raise RuntimeError(
                    "未发现相机。请检查：相机是否上电、网线是否插好、"
                    "相机与网卡是否在同一网段（海康默认 192.168.1.x）。")

            index = self._select_device(device_list)
            dev_info = cast(device_list.pDeviceInfo[index],
                            POINTER(sdk.H.MV_CC_DEVICE_INFO)).contents
            self._device_info = self._describe_device(dev_info)

            self._cam = sdk.MvCamera()
            ret = self._cam.MV_CC_CreateHandle(dev_info)
            if ret != 0:
                raise RuntimeError(f"创建句柄失败，错误码 {_err(ret)}")

            ret = self._cam.MV_CC_OpenDevice()
            if ret != 0:
                raise RuntimeError(
                    f"打开设备失败，错误码 {_err(ret)}。"
                    "相机可能正被 MVS 客户端或其它程序独占。")

            # GigE 相机：探测最佳包大小（巨帧）。不做这一步会频繁丢包。
            if int(dev_info.nTLayerType) in (sdk.const("MV_GIGE_DEVICE"),
                                             sdk.const("MV_GENTL_GIGE_DEVICE")):
                packet = self._cam.MV_CC_GetOptimalPacketSize()
                if int(packet) > 0:
                    ret = self._cam.MV_CC_SetIntValue(
                        "GevSCPSPacketSize", packet)
                    if ret != 0:
                        print(f"[Camera] 设置 GevSCPSPacketSize 失败 "
                              f"（不影响取流，但可能丢包）: {_err(ret)}")
                else:
                    print(f"[Camera] 获取最佳包大小失败: {_err(packet)}")

            self._apply_params()

            ret = self._cam.MV_CC_StartGrabbing()
            if ret != 0:
                raise RuntimeError(f"启动取流失败，错误码 {_err(ret)}")

            self._frame_out = sdk.H.MV_FRAME_OUT()
            self._is_open = True

            actual = self.get_actual_params()
            print(f"[Camera] 海康相机已打开 "
                  f"({self._device_info.get('model', '?')} "
                  f"SN:{self._device_info.get('serial_number', '?')}) "
                  f"{actual.get('width', '?')}×{actual.get('height', '?')} "
                  f"{actual.get('pixel_format', '?')}")
            return True

        except Exception as e:
            print(f"[Camera] 海康相机初始化失败: {e}")
            self.last_error = str(e)
            self._is_open = False
            self._cleanup()
            return False

    def _select_device(self, device_list) -> int:
        """决定打开哪台设备。

        指定了序列号就按序列号找 —— 网口插拔顺序会变，靠「第一个设备」
        在多相机场景下会拍错工位。找不到就明确报错，不悄悄退回第一台。
        """
        count = int(device_list.nDeviceNum)

        if not self.device_serial:
            if not 0 <= self.device_index < count:
                raise RuntimeError(
                    f"配置的设备索引 {self.device_index} 超出范围"
                    f"（枚举到 {count} 台）。")
            return self.device_index

        for i in range(count):
            info = cast(device_list.pDeviceInfo[i],
                        POINTER(self._sdk.H.MV_CC_DEVICE_INFO)).contents
            if self._device_serial_of(info) == self.device_serial:
                return i

        found = [self._device_serial_of(
            cast(device_list.pDeviceInfo[i],
                 POINTER(self._sdk.H.MV_CC_DEVICE_INFO)).contents)
            for i in range(count)]
        raise RuntimeError(
            f"未找到序列号为 {self.device_serial} 的相机。"
            f"当前枚举到：{found}")

    def _device_serial_of(self, dev_info) -> str:
        """从设备信息结构体里取序列号（GigE 与 USB 的布局不同）。"""
        info = _special_info(self._sdk, dev_info)
        return _decode(info[1].chSerialNumber) if info else ""

    def _describe_device(self, dev_info) -> Dict[str, str]:
        """整理型号 / 序列号 / IP 等信息，供界面展示。"""
        kind, spec = _special_info(self._sdk, dev_info)
        info: Dict[str, str] = {"tlayer": kind}
        info["vendor"] = _decode(spec.chManufacturerName)
        info["model"] = _decode(spec.chModelName)
        info["serial_number"] = _decode(spec.chSerialNumber)
        info["user_name"] = _decode(spec.chUserDefinedName)
        info["ip"] = (_ip_to_str(int(spec.nCurrentIp)) if kind == "GEV" else "")
        return info

    def release(self) -> None:
        """停止取流并释放设备。可重复调用。"""
        self._cleanup()
        self._is_open = False

    def _cleanup(self) -> None:
        """按 SDK 要求的顺序释放：停流 → 关设备 → 销毁句柄。"""
        cam = self._cam
        if cam is not None:
            try:
                cam.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                cam.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                cam.MV_CC_DestroyHandle()
            except Exception:
                pass
            self._cam = None

        self._frame_out = None
        self._convert_buf = None
        self._convert_buf_len = 0

    # ------------------------------------------------------------------
    # 参数下发
    # ------------------------------------------------------------------

    def _apply_params(self) -> None:
        """按正确顺序下发参数。

        顺序有讲究：先几何（宽高）再像素格式，然后曝光增益与白平衡，最后触发。
        曝光/增益/白平衡都必须先关自动模式，否则手动值会被自动算法立刻覆盖。
        （白平衡还有第二重理由，见 ``_apply_white_balance``。）
        """
        # 1) 宽高：相机对对齐有要求（常为 8 的倍数），失败不致命，
        #    后面 get_actual_params() 会把真实生效值读回来给用户看。
        if not self._set_int("Width", self.width):
            print(f"[Camera] 设置宽度 {self.width} 失败，沿用相机当前值")
        if not self._set_int("Height", self.height):
            print(f"[Camera] 设置高度 {self.height} 失败，沿用相机当前值")

        # 2) 像素格式（先过别名表，界面上写 RGB8 也能落到 RGB8_Packed）
        pixel = _resolve_pixel_format(self.pixel_format)
        if not self._set_enum_str("PixelFormat", pixel):
            print(f"[Camera] 设置像素格式 {self.pixel_format}（{pixel}）失败，"
                  f"沿用相机当前值")

        # 3) 曝光与增益
        self._apply_exposure_gain()

        # 4) 白平衡（彩色相机才有）：和曝光增益同属"成像亮度/色彩"一组，
        #    紧挨着下发。灰度相机没这个节点，失败是预期的。
        self._apply_white_balance()

        # 5) 采集模式：连续
        self._set_enum_str("AcquisitionMode", "Continuous")

        # 6) 触发
        self._apply_trigger()

    def _apply_exposure_gain(self) -> None:
        """下发曝光与增益。

        必须先设 ``ExposureAuto=Off`` / ``GainAuto=Off``：
        自动模式开着的时候，写进去的 ExposureTime / Gain 会被立刻覆盖，
        参数看着设成功了、实际完全没生效 —— 这是最费时间的一类问题。
        """
        self._set_enum_str("ExposureAuto", "Off")
        if self._set_float("ExposureTime", self.exposure_us):
            pass
        else:
            print(f"[Camera] 设置曝光 {self.exposure_us}μs 失败"
                  f"（可能超出相机量程）")

        self._set_enum_str("GainAuto", "Off")
        if not self._set_float("Gain", self.gain):
            print(f"[Camera] 设置增益 {self.gain} 失败（可能超出相机量程）")

    def _apply_white_balance(self) -> None:
        """关掉自动白平衡（只有彩色相机有这个节点）。

        理由和上面曝光/增益那条同源（自动模式会把写进去的值立刻覆盖），但对本
        项目更致命：**自动白平衡的工作目标就是「让画面不偏色」，而色度指标要测的
        恰恰是偏色**。两者直接对着干 —— 一块均匀氧化的板会被算法主动校回中性色，
        ΔH 与 ΔS 一起被抹平，指标恒报正常，比不装这个功能还糟。它还会让同一块板
        相邻两帧的色相漂移，污染自适应判据的 MAD 基线。

        这里**只设 Off，不设 Once**：``Once`` 是拿当前视野去算白平衡，而当前视野
        就是正在检测的板子 —— 等于把"被测对象的色偏"当成白平衡基准，同样会把要
        测的量消掉。白平衡标定是开机前对着标准白板做的一次性操作，在 MVS 客户端
        里做完会存进相机，不需要（也不应该）每帧由本程序代劳。
        """
        if not self._set_enum_str("BalanceWhiteAuto", "Off"):
            # 写失败有两种可能，必须分开对待：
            #   节点不存在（灰度相机）—— 完全正常，天天都会发生，不该刷警告；
            #   节点存在却写不进去 —— 那才是问题，色度指标会不可重复。
            if self._get_enum("BalanceWhiteAuto") is not None:
                print("[Camera] 关闭自动白平衡失败，色度指标可能帧间不可重复")

    def _apply_trigger(self) -> None:
        """下发触发模式与触发源。

        continuous → TriggerMode=Off（相机自由跑）
        software   → TriggerMode=On + TriggerSource=Software
        external   → TriggerMode=On + TriggerSource=Line<N>
        """
        sdk = self._sdk
        spec = _TRIGGER_MODES.get(self.trigger_mode)
        if spec is None:
            print(f"[Camera] 未知的触发模式 {self.trigger_mode!r}，"
                  f"按连续采集处理。可选：{sorted(_TRIGGER_MODES)}")
            spec = ("off", None)

        mode, source = spec
        if mode == "off":
            self._set_enum("TriggerMode", sdk.const("MV_TRIGGER_MODE_OFF"))
            return

        self._set_enum("TriggerMode", sdk.const("MV_TRIGGER_MODE_ON"))
        if source:
            # 外部触发时允许配置覆盖触发源（例如接在 Line2 上）
            if self.trigger_mode == "external":
                source = self.trigger_source
            if not self._set_enum_str("TriggerSource", source):
                print(f"[Camera] 设置触发源 {source} 失败，"
                      f"相机将一直等不到触发信号")

    # ------------------------------------------------------------------
    # 节点读写
    # ------------------------------------------------------------------

    def _set_int(self, name: str, value: int) -> bool:
        ret = self._cam.MV_CC_SetIntValue(name, int(value))
        return ret == 0

    def _set_float(self, name: str, value: float) -> bool:
        ret = self._cam.MV_CC_SetFloatValue(name, float(value))
        return ret == 0

    def _set_enum(self, name: str, value: int) -> bool:
        ret = self._cam.MV_CC_SetEnumValue(name, int(value))
        return ret == 0

    def _set_enum_str(self, name: str, value: str) -> bool:
        """按字符串下发枚举值。

        用字符串而不是数值常量，省掉一张容易写错的常量表。
        """
        ret = self._cam.MV_CC_SetEnumValueByString(name, str(value))
        return ret == 0

    def _get_float(self, name: str) -> Optional[float]:
        sdk = self._sdk
        st = sdk.H.MVCC_FLOATVALUE()
        memset(byref(st), 0, sizeof(st))
        if self._cam.MV_CC_GetFloatValue(name, st) != 0:
            return None
        return float(st.fCurValue)

    def _get_int(self, name: str) -> Optional[int]:
        """读整数节点。

        必须用 ``MV_CC_GetIntValueEx`` 配 ``MVCC_INTVALUE_EX``：
        非 Ex 的 ``MV_CC_GetIntValue`` 接收的是 ``MVCC_INTVALUE``，
        两者 ``nCurValue`` 的宽度不同（4 字节 vs 8 字节）。
        配错不会报错，只会把相邻字段一起读进来 —— 实测宽高读出来是
        ``17282948401552×13039520712704`` 这种荒唐值。
        """
        sdk = self._sdk
        st = sdk.H.MVCC_INTVALUE_EX()
        memset(byref(st), 0, sizeof(st))
        if self._cam.MV_CC_GetIntValueEx(name, st) != 0:
            return None
        return int(st.nCurValue)

    def _get_enum(self, name: str) -> Optional[int]:
        sdk = self._sdk
        st = sdk.H.MVCC_ENUMVALUE()
        memset(byref(st), 0, sizeof(st))
        if self._cam.MV_CC_GetEnumValue(name, st) != 0:
            return None
        return int(st.nCurValue)

    def _get_string(self, name: str) -> Optional[str]:
        sdk = self._sdk
        st = sdk.H.MVCC_STRINGVALUE()
        memset(byref(st), 0, sizeof(st))
        if self._cam.MV_CC_GetStringValue(name, st) != 0:
            return None
        return _decode(st.chCurValue)

    # ------------------------------------------------------------------
    # 取帧
    # ------------------------------------------------------------------

    def acquire(self) -> Optional[np.ndarray]:
        """抓取一帧。

        Returns:
            ``(H, W)`` uint8 灰度，或 ``(H, W, 3)`` uint8 BGR；失败返回 None。
        """
        if not self._is_open or self._cam is None:
            return None

        sdk = self._sdk
        frame = self._frame_out
        memset(byref(frame), 0, sizeof(frame))

        ret = self._cam.MV_CC_GetImageBuffer(frame, self.acquire_timeout_ms)
        if ret != 0:
            # 超时（无数据）是最常见的，不作为异常抛出，交由上层统计
            return None

        try:
            info = frame.stFrameInfo
            return self._extract(frame.pBufAddr, int(info.nWidth),
                                 int(info.nHeight), int(info.enPixelType),
                                 int(info.nFrameLen))
        finally:
            # 无论转换成功与否都必须归还缓冲，否则几次之后就没缓冲可用了
            self._cam.MV_CC_FreeImageBuffer(frame)

    def _extract(self, p_buf, width: int, height: int,
                 pixel_type: int, data_len: int) -> Optional[np.ndarray]:
        """把 SDK 缓冲转成 numpy 数组。

        注意 ``free_image_buffer`` 之后指针就失效了，所以这里必须 copy，
        不能返回共享内存的视图。
        """
        sdk = self._sdk

        if pixel_type == sdk.PX.PixelType_Gvsp_Mono8:
            raw = np.frombuffer(
                cast(p_buf, POINTER(c_ubyte * data_len)).contents,
                dtype=np.uint8, count=data_len)
            return raw.reshape(height, width).copy()

        # 其它格式统一转一次：彩色转 BGR8，非 8 位灰度转 Mono8
        is_color = _is_color_pixel_type(sdk, pixel_type)
        dst_type = (sdk.PX.PixelType_Gvsp_BGR8_Packed if is_color
                    else sdk.PX.PixelType_Gvsp_Mono8)
        dst_len = width * height * (3 if is_color else 1)

        if self._convert_buf_len < dst_len:
            self._convert_buf = (c_ubyte * dst_len)()
            self._convert_buf_len = dst_len

        param = sdk.H.MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(param), 0, sizeof(param))
        param.nWidth = width
        param.nHeight = height
        param.enSrcPixelType = pixel_type
        param.pSrcData = p_buf
        param.nSrcDataLen = data_len
        param.enDstPixelType = dst_type
        param.pDstBuffer = cast(self._convert_buf, POINTER(c_ubyte))
        param.nDstBufferSize = dst_len

        ret = self._cam.MV_CC_ConvertPixelType(param)
        if ret != 0:
            print(f"[Camera] 像素格式转换失败 "
                  f"({sdk.pixel_name(pixel_type)} → "
                  f"{sdk.pixel_name(dst_type)})，错误码 {_err(ret)}")
            return None

        got = int(param.nDstLen)
        raw = np.frombuffer(self._convert_buf, dtype=np.uint8, count=got)
        if is_color:
            return raw.reshape(height, width, 3).copy()
        return raw.reshape(height, width).copy()

    # ------------------------------------------------------------------
    # 运行时改参数
    # ------------------------------------------------------------------

    def set_exposure(self, exposure_us: float) -> None:
        """运行中修改曝光。"""
        self.exposure_us = float(exposure_us)
        if self._cam is not None:
            self._apply_exposure_gain()

    def set_gain(self, gain: float) -> None:
        """运行中修改增益。"""
        self.gain = float(gain)
        if self._cam is not None:
            self._apply_exposure_gain()

    # ------------------------------------------------------------------
    # 信息回读
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, str]:
        """返回相机型号、序列号、IP 等信息。"""
        return dict(self._device_info)

    def get_actual_params(self) -> Dict[str, object]:
        """读回相机实际生效的参数。

        与配置里「请求的值」可能不同：宽高会被相机按对齐要求夹取，
        曝光增益会受量程限制。界面必须展示这里的值 —— 否则用户会以为
        改成功了。
        """
        if self._cam is None:
            return {}

        params: Dict[str, object] = {}
        for key, node in (("width", "Width"), ("height", "Height")):
            value = self._get_int(node)
            if value is not None:
                params[key] = value
        for key, node in (("exposure_us", "ExposureTime"), ("gain", "Gain")):
            value = self._get_float(node)
            if value is not None:
                params[key] = round(value, 3)

        pixel = self._get_enum("PixelFormat")
        if pixel is not None:
            params["pixel_format"] = self._sdk.pixel_name(pixel)

        for key, node in (("exposure_auto", "ExposureAuto"),
                          ("gain_auto", "GainAuto"),
                          # 灰度相机没这个节点，_get_enum 返回 None 会被跳过
                          ("balance_white_auto", "BalanceWhiteAuto"),
                          ("trigger_mode", "TriggerMode"),
                          ("trigger_source", "TriggerSource"),
                          ("acquisition_mode", "AcquisitionMode")):
            value = self._get_enum(node)
            if value is not None:
                params[key] = value

        return params


# ============================================================================
# 辅助
# ============================================================================

def _special_info(sdk: _Sdk, dev_info):
    """取出设备信息里的传输层专属结构体。

    Returns:
        ``("GEV", MV_GIGE_DEVICE_INFO)`` 或 ``("U3V", MV_USB3_DEVICE_INFO)``；
        传输层不认识时返回 ``None``。

    注意：**不要**对 ``SpecialInfo`` 的字段做 ``ctypes.cast``。
    ``cast(dev_info.SpecialInfo.stGigEInfo, POINTER(MV_GIGE_DEVICE_INFO))``
    会抛 ``ArgumentError: argument 1: TypeError: wrong type`` ——
    ctypes 不接受 Union 内嵌的 Structure 作为 cast 的源对象，
    尽管它确实是 Structure 实例。
    而这个字段本身就已经是目标类型的实例（与 Union 共享内存），
    直接用即可，cast 纯属多余。
    """
    tlayer = int(dev_info.nTLayerType)
    if tlayer in (sdk.const("MV_GIGE_DEVICE"), sdk.const("MV_GENTL_GIGE_DEVICE")):
        return "GEV", dev_info.SpecialInfo.stGigEInfo
    if tlayer == sdk.const("MV_USB_DEVICE"):
        return "U3V", dev_info.SpecialInfo.stUsb3VInfo
    return None


def _decode(raw) -> str:
    """把 SDK 里的定长字符数组转成字符串。

    海康的字符数组是补零的 ASCII/UTF-8，用 ``rstrip(b"\\x00")`` 去掉尾部填充。
    """
    try:
        data = bytes(bytearray(raw))
    except TypeError:
        return str(raw)
    return data.split(b"\x00")[0].decode("utf-8", errors="replace").strip()


def _ip_to_str(value: int) -> str:
    """把 SDK 的 32 位整数 IP 转成点分十进制。"""
    value = int(value) & 0xFFFFFFFF
    return "%d.%d.%d.%d" % ((value >> 24) & 0xFF, (value >> 16) & 0xFF,
                            (value >> 8) & 0xFF, value & 0xFF)


def _is_color_pixel_type(sdk: _Sdk, pixel_type: int) -> bool:
    """该像素格式是不是彩色的。

    按名字判断（PixelType 名称里带 RGB/BGR/YUV/Bayer 的都是彩色），
    比维护一张常量表更耐版本变化。
    """
    name = sdk.pixel_names.get(int(pixel_type), "")
    return any(tag in name for tag in ("RGB", "BGR", "YUV", "Bayer", "YCbCr"))


def list_mvs_devices(verbose: bool = False) -> List[Dict[str, str]]:
    """枚举海康相机，供「扫描设备」按钮与诊断脚本使用。

    Args:
        verbose: 为 True 时把失败原因打到 stderr。默认静默 ——
            界面上「没扫到设备」是正常情况，不该弹错误；
            但排查时必须能看到原因，所以留了这个开关。

    Returns:
        设备信息字典列表；SDK 不可用或没枚举到时返回空列表。
    """
    try:
        sdk = _ensure_sdk()
    except RuntimeError as e:
        if verbose:
            print(f"[Camera] {e}", file=sys.stderr)
        return []

    try:
        device_list = sdk.H.MV_CC_DEVICE_INFO_LIST()
        ret = sdk.MvCamera.MV_CC_EnumDevices(sdk.layered_types, device_list)
        if ret != 0:
            if verbose:
                print(f"[Camera] 枚举设备失败，错误码 {_err(ret)}",
                      file=sys.stderr)
            return []

        devices: List[Dict[str, str]] = []
        for i in range(int(device_list.nDeviceNum)):
            dev = cast(device_list.pDeviceInfo[i],
                       POINTER(sdk.H.MV_CC_DEVICE_INFO)).contents
            spec = _special_info(sdk, dev)
            if spec is None:
                continue
            kind, detail = spec
            devices.append({
                "index": str(i),
                "tl_type": "GigE" if kind == "GEV" else "USB3",
                "vendor": _decode(detail.chManufacturerName),
                "model": _decode(detail.chModelName),
                "serial_number": _decode(detail.chSerialNumber),
                "ip": _ip_to_str(int(detail.nCurrentIp)) if kind == "GEV" else "",
                "user_name": _decode(detail.chUserDefinedName),
            })
        return devices
    except Exception as e:
        # 不再无声吞掉 —— 之前这里吞掉了一个 ctypes 的 cast 错误，
        # 表现为「一台设备都扫不到」，排查时多绕了一大圈
        if verbose:
            import traceback
            print(f"[Camera] 枚举设备异常: {e}", file=sys.stderr)
            traceback.print_exc()
        return []
