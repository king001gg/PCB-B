"""海康 MV 系列相机 SDK 后端（hardware/mvs_camera.py）单元测试。

本文件把海康的 ``MvCameraControl.dll`` **整体替换成假实现**：假的
``MvCamera`` 句柄、假的 ``CameraParams_header`` / ``CameraParams_const`` /
``PixelType_header`` 模块、以及照抄真实 SDK 字段宽度的 ctypes 结构体。
测试不 ``import MvCameraControl_class``、不加载 DLL、不联网、不连相机，
因此在没装 MVS 的机器上也能跑 —— 这是硬性要求，装了 MVS 的机器上跑出来
的结果也必须一模一样。

假 SDK 的注入点是模块级缓存 ``hardware.mvs_camera._SDK``：``_ensure_sdk()``
见到它就原样返回，所以 ``open()`` 走的是假实现，不会去磁盘上找 MVS。

结构体定义照抄真实 SDK 的字段与**宽度**，这一点是刻意的：
``MVCC_INTVALUE`` 的 ``nCurValue`` 是 4 字节而 ``MVCC_INTVALUE_EX`` 是 8 字节，
只有宽度对了，配置错接口时才会复现出「荒唐宽高」那个实测症状（见
``TestGetIntUsesExApi``）。

用例分组按源码的分节来：SDK 加载 → 常量/像素格式 → 设备枚举与选择 →
参数下发 → 节点读写 → 取帧 → 释放 → 信息回读 → 设备扫描。
"""

import ctypes
import os
import subprocess
import sys
import types
from ctypes import POINTER, byref, cast, c_ubyte, sizeof
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

import hardware.mvs_camera as mvs
from hardware.mvs_camera import (
    _ERROR_HINTS,
    _PIXEL_FORMAT_ALIASES,
    _TRIGGER_MODES,
    MvsCamera,
    _Sdk,
    _decode,
    _ensure_sdk,
    _err,
    _ip_to_str,
    _is_color_pixel_type,
    _resolve_pixel_format,
    _special_info,
    list_mvs_devices,
)


# ============================================================================
# 假 SDK：常量
# ============================================================================

#: 传输层类型。真实 SDK 里 MV_GIGE_DEVICE / MV_USB_DEVICE 在
#: CameraParams_const 模块，MV_GENTL_GIGE_DEVICE 也在那边。
GIGE = 0x00000001
USB = 0x00000004
GENTL_GIGE = 0x00000008
UNKNOWN_TLAYER = 0x00001000          # 假 SDK 专用：代码不认识的传输层

#: 触发模式常量。真实 SDK 把这两个放在 CameraParams_header（不是 const）——
#: ``_Sdk.const()`` 必须两个模块都查，这两个常量就是活证据。
TRIGGER_MODE_OFF = 0
TRIGGER_MODE_ON = 1

#: 像素格式。数值只是标识，测试只依赖「假 SDK 里名字↔数值的对应关系」。
MONO8 = 0x01080001
MONO10 = 0x01100003
MONO12 = 0x01100005
BAYER_RG8 = 0x01080009
RGB8_PACKED = 0x02140019
BGR8_PACKED = 0x0214001C
YUV422_PACKED = 0x0210001F

#: 常见错误码（与源码 _ERROR_HINTS 里的一致，测试里各取所需）。
ERR_NO_DATA = 0x80000004
ERR_ACCESS_DENIED = 0x80000046
ERR_UNSUPPORTED = 0x80000007


# ============================================================================
# 假 SDK：结构体（字段与宽度照抄真实 SDK）
# ============================================================================

class MV_GIGE_DEVICE_INFO(ctypes.Structure):
    """MV_GIGE_DEVICE_INFO（GigE 相机的传输层结构体）。"""

    _fields_ = [
        ("nVersion", ctypes.c_uint),
        ("nMacAddrHigh", ctypes.c_uint),
        ("nMacAddrLow", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 4),
        ("chManufacturerName", ctypes.c_char * 32),
        ("chModelName", ctypes.c_char * 32),
        ("chDeviceVersion", ctypes.c_char * 32),
        ("chManufacturerSpecificInfo", ctypes.c_char * 48),
        ("chSerialNumber", ctypes.c_char * 16),
        ("chUserDefinedName", ctypes.c_char * 16),
        ("nIpConfigOptions", ctypes.c_uint),
        ("nIpConfigCurrent", ctypes.c_uint),
        ("nCurrentIp", ctypes.c_uint),
        ("nCurrentSubNetMask", ctypes.c_uint),
        ("nDefultGateWay", ctypes.c_uint),
    ]


class MV_USB3_DEVICE_INFO(ctypes.Structure):
    """MV_USB3_DEVICE_INFO（USB3 相机的传输层结构体）。"""

    _fields_ = [
        ("nCrtlInEndPoint", ctypes.c_uint),
        ("nCrtlOutEndPoint", ctypes.c_uint),
        ("nStreamEndPoint", ctypes.c_uint),
        ("nEventEndPoint", ctypes.c_uint),
        ("chDeviceGUID", ctypes.c_char * 64),
        ("chVendorName", ctypes.c_char * 64),
        ("chModelName", ctypes.c_char * 64),
        ("chFamilyName", ctypes.c_char * 64),
        ("chDeviceVersion", ctypes.c_char * 64),
        ("chManufacturerName", ctypes.c_char * 64),
        ("chSerialNumber", ctypes.c_char * 64),
        ("chUserDefinedName", ctypes.c_char * 64),
        ("nNbcdUSB", ctypes.c_uint),
        ("nDeviceNumber", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 3),
    ]


class MV_SPECIAL_INFO(ctypes.Union):
    """MV_SPECIAL_INFO。

    这里必须用真正的 ``ctypes.Union``、内嵌真正的 ``Structure`` ——
    「不能对 SpecialInfo 的字段做 cast」那个坑只在 Union 内嵌 Structure
    时才会触发，用普通对象模拟是测不出来的。
    """

    _fields_ = [
        ("stGigEInfo", MV_GIGE_DEVICE_INFO),
        ("stUsb3VInfo", MV_USB3_DEVICE_INFO),
        ("stReserved", c_ubyte * 512),
    ]


class MV_CC_DEVICE_INFO(ctypes.Structure):
    """MV_CC_DEVICE_INFO。"""

    _fields_ = [
        ("nMajorVer", ctypes.c_ushort),
        ("nMinorVer", ctypes.c_ushort),
        ("nMacAddrHigh", ctypes.c_uint),
        ("nMacAddrLow", ctypes.c_uint),
        ("nTLayerType", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 4),
        ("SpecialInfo", MV_SPECIAL_INFO),
    ]


MV_MAX_DEVICE_NUM = 256


class MV_CC_DEVICE_INFO_LIST(ctypes.Structure):
    """MV_CC_DEVICE_INFO_LIST。"""

    _fields_ = [
        ("nDeviceNum", ctypes.c_uint),
        ("pDeviceInfo", POINTER(MV_CC_DEVICE_INFO) * MV_MAX_DEVICE_NUM),
    ]


class MV_FRAME_OUT_INFO_EX(ctypes.Structure):
    """MV_FRAME_OUT_INFO_EX（只保留代码会用到的字段）。"""

    _fields_ = [
        ("nWidth", ctypes.c_ushort),
        ("nHeight", ctypes.c_ushort),
        ("enPixelType", ctypes.c_uint),
        ("nFrameNum", ctypes.c_uint),
        ("nFrameLen", ctypes.c_uint),
        ("nTimeStamp", ctypes.c_longlong),
        ("nLostPacket", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 8),
    ]


class MV_FRAME_OUT(ctypes.Structure):
    """MV_FRAME_OUT。"""

    _fields_ = [
        ("pBufAddr", POINTER(c_ubyte)),
        ("stFrameInfo", MV_FRAME_OUT_INFO_EX),
        ("nReserved", ctypes.c_uint * 16),
    ]


class MV_CC_PIXEL_CONVERT_PARAM(ctypes.Structure):
    """MV_CC_PIXEL_CONVERT_PARAM。"""

    _fields_ = [
        ("nWidth", ctypes.c_ushort),
        ("nHeight", ctypes.c_ushort),
        ("enSrcPixelType", ctypes.c_uint),
        ("pSrcData", POINTER(c_ubyte)),
        ("nSrcDataLen", ctypes.c_uint),
        ("enDstPixelType", ctypes.c_uint),
        ("pDstBuffer", POINTER(c_ubyte)),
        ("nDstBufferSize", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 4),
        ("nDstLen", ctypes.c_uint),
    ]


class MVCC_INTVALUE(ctypes.Structure):
    """MVCC_INTVALUE —— nCurValue 是 **4 字节**。"""

    _fields_ = [
        ("nCurValue", ctypes.c_uint),
        ("nMax", ctypes.c_uint),
        ("nMin", ctypes.c_uint),
        ("nInc", ctypes.c_uint),
    ]


class MVCC_INTVALUE_EX(ctypes.Structure):
    """MVCC_INTVALUE_EX —— nCurValue 是 **8 字节**（与上面差一倍）。"""

    _fields_ = [
        ("nCurValue", ctypes.c_longlong),
        ("nMax", ctypes.c_longlong),
        ("nMin", ctypes.c_longlong),
        ("nInc", ctypes.c_longlong),
    ]


class MVCC_FLOATVALUE(ctypes.Structure):
    """MVCC_FLOATVALUE。"""

    _fields_ = [
        ("fCurValue", ctypes.c_float),
        ("fMax", ctypes.c_float),
        ("fMin", ctypes.c_float),
    ]


class MVCC_ENUMVALUE(ctypes.Structure):
    """MVCC_ENUMVALUE。"""

    _fields_ = [
        ("nCurValue", ctypes.c_uint),
        ("nSupportedNum", ctypes.c_uint),
        ("nSupportValue", ctypes.c_uint * 64),
        ("nReserved", ctypes.c_uint * 4),
    ]


class MVCC_STRINGVALUE(ctypes.Structure):
    """MVCC_STRINGVALUE。"""

    _fields_ = [
        ("chCurValue", ctypes.c_char * 256),
        ("nMaxLength", ctypes.c_longlong),
        ("nReserved", ctypes.c_longlong * 2),
    ]


#: 假 header 模块里的结构体部分。
_HEADER_STRUCTS: Dict[str, type] = {
    "MV_GIGE_DEVICE_INFO": MV_GIGE_DEVICE_INFO,
    "MV_USB3_DEVICE_INFO": MV_USB3_DEVICE_INFO,
    "MV_CC_DEVICE_INFO": MV_CC_DEVICE_INFO,
    "MV_CC_DEVICE_INFO_LIST": MV_CC_DEVICE_INFO_LIST,
    "MV_FRAME_OUT": MV_FRAME_OUT,
    "MV_FRAME_OUT_INFO_EX": MV_FRAME_OUT_INFO_EX,
    "MV_CC_PIXEL_CONVERT_PARAM": MV_CC_PIXEL_CONVERT_PARAM,
    "MVCC_INTVALUE": MVCC_INTVALUE,
    "MVCC_INTVALUE_EX": MVCC_INTVALUE_EX,
    "MVCC_FLOATVALUE": MVCC_FLOATVALUE,
    "MVCC_ENUMVALUE": MVCC_ENUMVALUE,
    "MVCC_STRINGVALUE": MVCC_STRINGVALUE,
}

#: 假 CameraParams_header 模块里的常量。
_HEADER_CONSTS: Dict[str, int] = {
    "MV_TRIGGER_MODE_OFF": TRIGGER_MODE_OFF,
    "MV_TRIGGER_MODE_ON": TRIGGER_MODE_ON,
}

#: 假 CameraParams_const 模块里的常量。
_CONST_CONSTS: Dict[str, int] = {
    "MV_GIGE_DEVICE": GIGE,
    "MV_USB_DEVICE": USB,
    "MV_GENTL_GIGE_DEVICE": GENTL_GIGE,
}

#: 假 PixelType_header 模块里的像素格式常量。
_PIXEL_CONSTS: Dict[str, int] = {
    "PixelType_Gvsp_Mono8": MONO8,
    "PixelType_Gvsp_Mono10": MONO10,
    "PixelType_Gvsp_Mono12": MONO12,
    "PixelType_Gvsp_BayerRG8": BAYER_RG8,
    "PixelType_Gvsp_RGB8_Packed": RGB8_PACKED,
    "PixelType_Gvsp_BGR8_Packed": BGR8_PACKED,
    "PixelType_Gvsp_YUV422_Packed": YUV422_PACKED,
}


# ============================================================================
# 假 SDK：数据载荷
# ============================================================================

def _pattern(count: int, step: int = 7, offset: int = 3) -> bytes:
    """确定性字节序列 —— 不用随机数，回归失败能稳定重现。"""
    return bytes((i * step + offset) % 256 for i in range(count))


def _bytes_per_pixel(pixel_type: int) -> int:
    """该像素格式单像素占几个字节（仅用于造出长度合理的假缓冲）。"""
    if pixel_type in (MONO8, BAYER_RG8):
        return 1
    if pixel_type in (MONO10, MONO12):
        return 2
    return 3


def _convert_payload(count: int) -> bytes:
    """假 SDK 的像素格式转换输出。"""
    return _pattern(count, step=11, offset=5)


def _ip_to_int(text: str) -> int:
    """点分十进制 → SDK 的 32 位整数。"""
    a, b, c, d = (int(p) for p in text.split("."))
    return (a << 24) | (b << 16) | (c << 8) | d


def make_device(
    serial: str = "SN-0001",
    model: str = "MV-CE050-10GM",
    vendor: str = "Hikrobot",
    user_name: str = "",
    ip: str = "192.168.1.100",
    tlayer: int = GIGE,
) -> Dict[str, object]:
    """一台假设备的描述。"""
    return {"serial": serial, "model": model, "vendor": vendor,
            "user_name": user_name, "ip": ip, "tlayer": tlayer}


def _build_device_info(spec: Dict[str, object]) -> MV_CC_DEVICE_INFO:
    """按描述造一个 MV_CC_DEVICE_INFO（GigE 与 USB 的联合体分支不同）。"""
    dev = MV_CC_DEVICE_INFO()
    tlayer = int(spec["tlayer"])  # type: ignore[arg-type]
    dev.nTLayerType = tlayer
    if tlayer in (GIGE, GENTL_GIGE):
        info = dev.SpecialInfo.stGigEInfo
        info.nCurrentIp = _ip_to_int(str(spec["ip"]))
    else:
        info = dev.SpecialInfo.stUsb3VInfo
    info.chManufacturerName = str(spec["vendor"]).encode("ascii")
    info.chModelName = str(spec["model"]).encode("ascii")
    info.chSerialNumber = str(spec["serial"]).encode("ascii")
    info.chUserDefinedName = str(spec["user_name"]).encode("ascii")
    return dev


# ============================================================================
# 假 SDK：句柄与场景参数
# ============================================================================

class FakeSdkState:
    """假 SDK 的场景参数 —— 测试直接改这里的字段来构造各种分支。

    所有 SDK 调用都会记进 ``calls``（形如 ``("SetEnumValueByString",
    "ExposureAuto", "Off")``），顺序敏感的逻辑（曝光增益、释放流程）靠它断言。
    """

    def __init__(self, devices: Optional[List[Dict[str, object]]] = None) -> None:
        self.devices: List[Dict[str, object]] = (
            list(devices) if devices else [make_device()])
        self.calls: List[tuple] = []

        # 各接口的返回码，非 0 即失败
        self.ret_enum = 0
        self.ret_create = 0
        self.ret_open = 0
        self.ret_start = 0
        self.ret_get_image = ERR_NO_DATA
        self.ret_convert = 0

        #: 按节点名覆盖写接口的返回码（例如 {"Width": 0x80000007}）
        self.set_ret: Dict[str, int] = {}
        #: 按节点名覆盖读接口的返回码
        self.get_ret: Dict[str, int] = {}

        # 节点当前值
        self.ints: Dict[str, int] = {"Width": 2448, "Height": 2048}
        self.floats: Dict[str, float] = {"ExposureTime": 5000.0, "Gain": 1.0}
        self.enums: Dict[str, int] = {
            "PixelFormat": MONO8,
            "ExposureAuto": 0,
            "GainAuto": 0,
            "TriggerMode": TRIGGER_MODE_OFF,
            "TriggerSource": 0,
            "AcquisitionMode": 2,
        }
        self.strings: Dict[str, str] = {"DeviceUserID": "cam-1"}

        self.packet_size = 8164
        self.frame: Optional[Dict[str, object]] = None

        #: 让某个接口抛异常，用来测「异常路径也必须释放缓冲」这类契约
        self.raise_on: Dict[str, BaseException] = {}

        #: 只读：最近一次像素格式转换的入参
        self.last_convert: Dict[str, int] = {}
        #: 只读：最近一帧的原始缓冲（必须保持存活，否则指针悬垂）
        self.frame_buf = None
        #: 只读：枚举出来的设备结构体
        self.device_structs: List[MV_CC_DEVICE_INFO] = []

    # ------------------------------------------------------------------

    def _record(self, name: str, *args) -> None:
        self.calls.append((name,) + args)

    def names(self) -> List[str]:
        """按顺序取出所有被调用的接口名。"""
        return [c[0] for c in self.calls]

    def set_calls(self) -> List[tuple]:
        """按顺序取出所有「写节点」的调用。"""
        return [c for c in self.calls if c[0].startswith("Set")]

    def count(self, name: str) -> int:
        return self.names().count(name)


class _FakeMvCameraBase:
    """假的 MvCamera 句柄。

    ``MV_CC_EnumDevices`` 在真实 SDK 里是静态方法（代码用
    ``sdk.MvCamera.MV_CC_EnumDevices(...)`` 调用），这里用 classmethod
    以便拿到场景参数。
    """

    _state: FakeSdkState

    def __init__(self) -> None:
        pass

    # ---- 内部 ----

    def _call(self, name: str, *args) -> FakeSdkState:
        state = type(self)._state
        state._record(name, *args)
        exc = state.raise_on.get(name)
        if exc is not None:
            raise exc
        return state

    def _ret(self, name: str, attr: str) -> int:
        return int(getattr(self._call(name), attr))

    # ---- 枚举 / 生命周期 ----

    @classmethod
    def MV_CC_EnumDevices(cls, layered_type, device_list) -> int:
        state = cls._state
        state._record("EnumDevices", layered_type)
        if state.ret_enum != 0:
            return state.ret_enum
        structs = [_build_device_info(d) for d in state.devices]
        state.device_structs = structs
        device_list.nDeviceNum = len(structs)
        for i, info in enumerate(structs):
            device_list.pDeviceInfo[i] = ctypes.pointer(info)
        return 0

    def MV_CC_CreateHandle(self, dev_info) -> int:
        return self._ret("CreateHandle", "ret_create")

    def MV_CC_OpenDevice(self) -> int:
        return self._ret("OpenDevice", "ret_open")

    def MV_CC_StartGrabbing(self) -> int:
        return self._ret("StartGrabbing", "ret_start")

    def MV_CC_StopGrabbing(self) -> int:
        self._call("StopGrabbing")
        return 0

    def MV_CC_CloseDevice(self) -> int:
        self._call("CloseDevice")
        return 0

    def MV_CC_DestroyHandle(self) -> int:
        self._call("DestroyHandle")
        return 0

    def MV_CC_GetOptimalPacketSize(self) -> int:
        return int(self._call("GetOptimalPacketSize").packet_size)

    # ---- 写节点 ----

    def MV_CC_SetIntValue(self, name, value) -> int:
        state = self._call("SetIntValue", str(name), int(value))
        return state.set_ret.get(str(name), 0)

    def MV_CC_SetFloatValue(self, name, value) -> int:
        state = self._call("SetFloatValue", str(name), float(value))
        return state.set_ret.get(str(name), 0)

    def MV_CC_SetEnumValue(self, name, value) -> int:
        state = self._call("SetEnumValue", str(name), int(value))
        return state.set_ret.get(str(name), 0)

    def MV_CC_SetEnumValueByString(self, name, value) -> int:
        state = self._call("SetEnumValueByString", str(name), str(value))
        return state.set_ret.get(str(name), 0)

    # ---- 读节点 ----

    def MV_CC_GetIntValueEx(self, name, st) -> int:
        """Ex 接口：``MVCC_INTVALUE_EX``（8 字节字段）。"""
        state = self._call("GetIntValueEx", str(name))
        ret = state.get_ret.get(str(name), 0)
        if ret != 0:
            return ret
        st.nCurValue = int(state.ints.get(str(name), 0))
        return 0

    def MV_CC_GetIntValue(self, name, st) -> int:
        """非 Ex 接口：内部按 ``MVCC_INTVALUE``（4 字节字段）写入这块内存。

        这里刻意照真实 SDK 的写入宽度实现 —— 调用方若传的是
        ``MVCC_INTVALUE_EX``，ctypes **不会报错**，但 EX 的 8 字节
        ``nCurValue`` 会由「低 4 字节 = 真值、高 4 字节 = 紧随其后的 nMax」
        拼起来，读出来就是荒唐值。这正是实测踩到的那个坑。
        """
        state = self._call("GetIntValue", str(name))
        ret = state.get_ret.get(str(name), 0)
        if ret != 0:
            return ret
        stage = MVCC_INTVALUE()
        stage.nCurValue = int(state.ints.get(str(name), 0))
        stage.nMax = 65535
        ctypes.memmove(byref(st), byref(stage), sizeof(stage))
        return 0

    def MV_CC_GetFloatValue(self, name, st) -> int:
        state = self._call("GetFloatValue", str(name))
        ret = state.get_ret.get(str(name), 0)
        if ret != 0:
            return ret
        st.fCurValue = float(state.floats.get(str(name), 0.0))
        return 0

    def MV_CC_GetEnumValue(self, name, st) -> int:
        state = self._call("GetEnumValue", str(name))
        ret = state.get_ret.get(str(name), 0)
        if ret != 0:
            return ret
        st.nCurValue = int(state.enums.get(str(name), 0))
        return 0

    def MV_CC_GetStringValue(self, name, st) -> int:
        state = self._call("GetStringValue", str(name))
        ret = state.get_ret.get(str(name), 0)
        if ret != 0:
            return ret
        st.chCurValue = str(state.strings.get(str(name), "")).encode("ascii")
        return 0

    # ---- 取帧 ----

    def MV_CC_GetImageBuffer(self, frame, timeout_ms) -> int:
        state = self._call("GetImageBuffer", int(timeout_ms))
        spec = state.frame
        if spec is None:
            return state.ret_get_image

        data = bytes(spec["data"])  # type: ignore[arg-type]
        buf = (c_ubyte * len(data))(*data)
        state.frame_buf = buf          # 保持存活，否则指针悬垂
        frame.pBufAddr = cast(buf, POINTER(c_ubyte))
        frame.stFrameInfo.nWidth = int(spec["width"])       # type: ignore[arg-type]
        frame.stFrameInfo.nHeight = int(spec["height"])     # type: ignore[arg-type]
        frame.stFrameInfo.enPixelType = int(spec["pixel_type"])  # type: ignore[arg-type]
        frame.stFrameInfo.nFrameLen = len(data)
        return 0

    def MV_CC_FreeImageBuffer(self, frame) -> int:
        self._call("FreeImageBuffer")
        # 真实 SDK 归还后指针即失效，这里同样清掉 —— 防止有人误用悬垂指针
        frame.pBufAddr = None
        return 0

    def MV_CC_ConvertPixelType(self, param) -> int:
        state = self._call("ConvertPixelType", int(param.enSrcPixelType),
                           int(param.enDstPixelType))
        state.last_convert = {
            "width": int(param.nWidth),
            "height": int(param.nHeight),
            "src": int(param.enSrcPixelType),
            "dst": int(param.enDstPixelType),
            "src_len": int(param.nSrcDataLen),
            "buf_size": int(param.nDstBufferSize),
        }
        if state.ret_convert != 0:
            return state.ret_convert
        data = _convert_payload(int(param.nDstBufferSize))
        ctypes.memmove(param.pDstBuffer, data, len(data))
        param.nDstLen = len(data)
        return 0


def make_fake_sdk(devices: Optional[List[Dict[str, object]]] = None) -> _Sdk:
    """造一套假 SDK 并包装成 ``_Sdk``（真实代码读到的就是它）。

    Returns:
        ``_Sdk`` 实例，额外挂了 ``state`` 属性供测试改场景 / 查调用记录。
    """
    state = FakeSdkState(devices)
    handle_class = type("FakeMvCamera", (_FakeMvCameraBase,), {"_state": state})

    header = types.SimpleNamespace(**_HEADER_STRUCTS, **_HEADER_CONSTS)
    const = types.SimpleNamespace(**_CONST_CONSTS)
    pixel = types.SimpleNamespace(**_PIXEL_CONSTS)

    sdk = _Sdk(handle_class, header, const, pixel)
    sdk.state = state            # type: ignore[attr-defined]
    return sdk


def set_frame(sdk, width: int = 8, height: int = 6, pixel_type: int = MONO8,
              data: Optional[bytes] = None) -> None:
    """安排下一帧的内容。"""
    if data is None:
        data = _pattern(width * height * _bytes_per_pixel(pixel_type))
    sdk.state.frame = {"width": width, "height": height,
                       "pixel_type": pixel_type, "data": data}


class _ExplodingModule(types.ModuleType):
    """一个「一取属性就抛 ImportError」的模块。

    用它占住 ``sys.modules["MvCameraControl_class"]``，可以稳定复现
    「MvImport 找到了但导入失败」这条分支，同时**保证绝不加载真实 DLL**。
    """

    def __getattr__(self, name):
        raise ImportError(f"模拟导入失败：{name}")


def inject_fake_sdk_modules(monkeypatch) -> None:
    """把假的四个 SDK 模块塞进 sys.modules。

    ``sys.modules`` 比文件系统优先，所以即便这台机器装了 MVS，
    ``_ensure_sdk()`` 也只会导入这里的假模块 —— 绝不加载真实 DLL。
    """
    monkeypatch.setitem(sys.modules, "MvCameraControl_class",
                        types.SimpleNamespace(MvCamera=_FakeMvCameraBase))
    monkeypatch.setitem(sys.modules, "CameraParams_const",
                        types.SimpleNamespace(**_CONST_CONSTS))
    monkeypatch.setitem(sys.modules, "CameraParams_header",
                        types.SimpleNamespace(**_HEADER_STRUCTS, **_HEADER_CONSTS))
    monkeypatch.setitem(sys.modules, "PixelType_header",
                        types.SimpleNamespace(**_PIXEL_CONSTS))


# ============================================================================
# 夹具
# ============================================================================

@pytest.fixture
def sdk(monkeypatch) -> _Sdk:
    """假 SDK 注入模块级缓存 —— open() 便不会再去找真实 MVS。"""
    fake = make_fake_sdk()
    monkeypatch.setattr(mvs, "_SDK", fake)
    return fake


@pytest.fixture
def make_camera(default_config):
    """相机工厂：在 config/default.yaml 的真实内容上覆盖 camera 节。

    读真配置而不是另写一份 —— 字段名改了、层级挪了这里会第一时间炸。
    """
    import copy

    def _make(**camera_overrides) -> MvsCamera:
        cfg = copy.deepcopy(default_config)
        cfg["camera"].update(camera_overrides)
        return MvsCamera(cfg)

    return _make


@pytest.fixture
def cam(sdk, make_camera) -> MvsCamera:
    """已接上假 SDK 与假句柄、但没走 open() 的相机。"""
    camera = make_camera()
    camera._sdk = sdk
    camera._cam = sdk.MvCamera()
    return camera


@pytest.fixture
def live_cam(cam, sdk) -> MvsCamera:
    """「已打开」的相机：句柄与帧缓冲都就绪，供取帧相关用例使用。"""
    cam._is_open = True
    cam._frame_out = sdk.H.MV_FRAME_OUT()
    return cam


def enumerated(sdk) -> MV_CC_DEVICE_INFO_LIST:
    """跑一遍假的枚举，拿到填好的设备列表。"""
    lst = sdk.H.MV_CC_DEVICE_INFO_LIST()
    assert sdk.MvCamera.MV_CC_EnumDevices(sdk.layered_types, lst) == 0
    return lst


# ============================================================================
# SDK 加载（_ensure_sdk）
# ============================================================================

class TestEnsureSdk:
    """_ensure_sdk 的定位、缓存与失败提示（全程不碰磁盘上的真实 MVS）。"""

    def test_returns_cached_sdk_without_probing(self, sdk, monkeypatch):
        """已有缓存时直接返回，不再去找 MvImport / DLL。"""
        def _boom(*a, **kw):
            raise AssertionError("有缓存时不该再探测文件系统")

        monkeypatch.setattr(mvs, "find_mvimport_dirs", _boom)
        monkeypatch.setattr(mvs, "find_mvs_runtime_dirs", _boom)
        assert mvs._ensure_sdk() is sdk

    def test_missing_mvimport_gives_actionable_hint(self, monkeypatch):
        """找不到 MvImport 时，提示里要写清去哪装、用什么工具查。"""
        monkeypatch.setattr(mvs, "_SDK", None)
        monkeypatch.setattr(mvs, "find_mvimport_dirs", lambda: [])
        with pytest.raises(RuntimeError) as ei:
            mvs._ensure_sdk()
        msg = str(ei.value)
        assert "MvImport" in msg
        assert "check_camera.py" in msg

    def test_missing_runtime_dll_gives_hint(self, monkeypatch, tmp_path):
        """MvImport 找到了但没有 MvCameraControl.dll —— 提示要指向 Runtime 组件。"""
        monkeypatch.setattr(mvs, "_SDK", None)
        monkeypatch.setattr(mvs, "find_mvimport_dirs", lambda: [str(tmp_path)])
        monkeypatch.setattr(mvs, "find_mvs_runtime_dirs", lambda: [])
        with pytest.raises(RuntimeError) as ei:
            mvs._ensure_sdk()
        assert "MvCameraControl.dll" in str(ei.value)

    def test_import_failure_mentions_python_bitness(self, monkeypatch, tmp_path):
        """导入失败时要点出「位数不一致」这个最常见原因。

        用「爆炸模块」占住 sys.modules 里的名字 —— 即便这台机器装了 MVS，
        也绝不会去加载真实的 MvCameraControl.dll。
        """
        monkeypatch.setattr(mvs, "_SDK", None)
        monkeypatch.setattr(mvs, "find_mvimport_dirs", lambda: [str(tmp_path)])
        monkeypatch.setattr(mvs, "find_mvs_runtime_dirs", lambda: [str(tmp_path)])
        monkeypatch.setattr(mvs, "_DLL_HANDLES", [])
        monkeypatch.setitem(sys.modules, "MvCameraControl_class",
                            _ExplodingModule("MvCameraControl_class"))

        with pytest.raises(RuntimeError) as ei:
            mvs._ensure_sdk()
        msg = str(ei.value)
        assert "导入失败" in msg
        assert "位" in msg
        assert mvs.python_bitness() is not None

    def test_loads_and_caches_sdk(self, monkeypatch, tmp_path):
        """走通加载链路：找到 MvImport 与 DLL → 导入四个模块 → 缓存 _Sdk。"""
        monkeypatch.setattr(mvs, "_SDK", None)
        monkeypatch.setattr(mvs, "find_mvimport_dirs", lambda: [str(tmp_path)])
        monkeypatch.setattr(mvs, "find_mvs_runtime_dirs", lambda: [str(tmp_path)])
        monkeypatch.setattr(mvs, "_DLL_HANDLES", [])
        inject_fake_sdk_modules(monkeypatch)

        sdk = _ensure_sdk()
        assert isinstance(sdk, _Sdk)
        assert sdk.MvCamera is _FakeMvCameraBase
        assert sdk.PX.PixelType_Gvsp_Mono8 == MONO8
        assert sdk.const("MV_TRIGGER_MODE_OFF") == TRIGGER_MODE_OFF
        assert sdk.layered_types == GIGE | USB

        # 缓存住了：第二次调用不再探测文件系统
        def _boom(*a, **kw):
            raise AssertionError("SDK 已缓存，不该再探测文件系统")

        monkeypatch.setattr(mvs, "find_mvimport_dirs", _boom)
        assert _ensure_sdk() is sdk

    def test_path_fallback_when_add_dll_directory_unavailable(self, monkeypatch,
                                                              tmp_path):
        """Python < 3.8 没有 add_dll_directory，退回把 Runtime 目录塞进 PATH。"""
        monkeypatch.setattr(mvs, "_SDK", None)
        monkeypatch.setattr(mvs, "find_mvimport_dirs", lambda: [str(tmp_path)])
        monkeypatch.setattr(mvs, "find_mvs_runtime_dirs", lambda: [str(tmp_path)])
        monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
        inject_fake_sdk_modules(monkeypatch)

        def _no_such_api(path):
            raise AttributeError("add_dll_directory")

        monkeypatch.setattr(mvs.os, "add_dll_directory", _no_such_api,
                            raising=False)
        _ensure_sdk()
        assert str(tmp_path) in os.environ["PATH"]

    def test_dll_search_dirs_are_kept_alive(self, monkeypatch, tmp_path):
        """add_dll_directory 返回的句柄必须留住，否则目录立刻失效。"""
        monkeypatch.setattr(mvs, "_SDK", None)
        monkeypatch.setattr(mvs, "find_mvimport_dirs", lambda: [str(tmp_path)])
        monkeypatch.setattr(mvs, "find_mvs_runtime_dirs", lambda: [str(tmp_path)])
        handles: List[object] = []
        monkeypatch.setattr(mvs, "_DLL_HANDLES", handles)
        monkeypatch.setitem(sys.modules, "MvCameraControl_class",
                            _ExplodingModule("MvCameraControl_class"))

        with pytest.raises(RuntimeError):
            mvs._ensure_sdk()

        if hasattr(mvs.os, "add_dll_directory"):
            assert len(handles) == 1, "DLL 搜索目录的句柄没有被保存"
        else:
            assert str(tmp_path) in os.environ.get("PATH", "")


# ============================================================================
# _Sdk 命名空间
# ============================================================================

class TestSdkNamespace:
    """_Sdk 的常量查找与像素格式反查表。"""

    def test_sdk_constructs_without_touching_camera_handle(self):
        """构造 _Sdk 只需要三个模块对象，不需要任何真实 SDK 对象。"""
        sdk = _Sdk(None, types.SimpleNamespace(), types.SimpleNamespace(),
                   types.SimpleNamespace())
        assert sdk.pixel_names == {}

    def test_const_finds_value_in_const_module(self, sdk):
        """MV_GIGE_DEVICE 在 CameraParams_const 里。"""
        assert sdk.const("MV_GIGE_DEVICE") == GIGE
        assert sdk.const("MV_USB_DEVICE") == USB

    def test_const_finds_value_in_header_module(self, sdk):
        """MV_TRIGGER_MODE_OFF 在 CameraParams_header 里（不在 const）。

        写死模块名换 MVS 版本就会炸，所以两个模块都要查。
        """
        assert sdk.const("MV_TRIGGER_MODE_OFF") == TRIGGER_MODE_OFF
        assert sdk.const("MV_TRIGGER_MODE_ON") == TRIGGER_MODE_ON

    def test_const_finds_constant_only_in_one_module(self, sdk, monkeypatch):
        """只存在于某一个模块的常量都必须能取到。"""
        monkeypatch.setattr(sdk.C, "MV_ONLY_IN_CONST", 111, raising=False)
        monkeypatch.setattr(sdk.H, "MV_ONLY_IN_HEADER", 222, raising=False)
        assert sdk.const("MV_ONLY_IN_CONST") == 111
        assert sdk.const("MV_ONLY_IN_HEADER") == 222

    def test_const_prefers_const_module_on_conflict(self, sdk, monkeypatch):
        """两处都定义时取 CameraParams_const 的那份（源码里的查找顺序）。"""
        monkeypatch.setattr(sdk.C, "MV_CONFLICT", 111, raising=False)
        monkeypatch.setattr(sdk.H, "MV_CONFLICT", 222, raising=False)
        assert sdk.const("MV_CONFLICT") == 111

    def test_const_missing_raises_with_constant_name(self, sdk):
        """两边都没有时必须抛 AttributeError，且错误信息里带常量名。"""
        with pytest.raises(AttributeError) as ei:
            sdk.const("MV_NOT_A_REAL_CONST")
        msg = str(ei.value)
        assert "MV_NOT_A_REAL_CONST" in msg
        assert "CameraParams_const" in msg
        assert "CameraParams_header" in msg

    def test_layered_types_covers_gige_and_usb(self, sdk):
        """枚举掩码必须同时覆盖网口与 USB —— 只填一个会漏设备。"""
        assert sdk.layered_types == GIGE | USB
        assert sdk.layered_types & GIGE
        assert sdk.layered_types & USB

    def test_layered_types_raises_when_constant_missing(self, sdk, monkeypatch):
        """常量缺失时明确报错，而不是静默按 0 枚举（会一台设备都扫不到）。"""
        monkeypatch.delattr(sdk.C, "MV_USB_DEVICE")
        with pytest.raises(AttributeError, match="MV_USB_DEVICE"):
            sdk.layered_types

    def test_pixel_names_strips_prefix(self, sdk):
        assert sdk.pixel_names[MONO8] == "Mono8"
        assert sdk.pixel_names[RGB8_PACKED] == "RGB8_Packed"

    def test_pixel_name_known_and_unknown(self, sdk):
        assert sdk.pixel_name(MONO8) == "Mono8"
        unknown = sdk.pixel_name(0xDEADBEEF)
        assert "0xDEADBEEF" in unknown          # 十六进制原样打印，便于查文档

    def test_duplicate_pixel_values_keep_first_name(self):
        """多个枚举名共用一个数值时保留先出现的（setdefault 语义）。"""
        px = types.SimpleNamespace(PixelType_Gvsp_Aaa8=0x100,
                                  PixelType_Gvsp_Bbb8=0x100)
        sdk = _Sdk(None, types.SimpleNamespace(), types.SimpleNamespace(), px)
        assert sdk.pixel_names[0x100] == "Aaa8"


# ============================================================================
# 像素格式别名表
# ============================================================================

class TestResolvePixelFormat:
    """界面名 → 相机枚举名。"""

    @pytest.mark.parametrize("given, expected", [
        ("RGB8", "RGB8_Packed"),
        ("BGR8", "BGR8_Packed"),
        ("yuv422", "YUV422_Packed"),
        ("mono8", "Mono8"),
        ("MONO10", "Mono10"),
        ("mono12", "Mono12"),
        ("rgb8_packed", "RGB8_Packed"),
        ("bgr8_packed", "BGR8_Packed"),
    ])
    def test_aliases_map_to_sdk_enum_names(self, given, expected):
        """界面友好名必须落到 SDK 认的枚举字面量上。

        不落的话 ``MV_CC_SetEnumValueByString`` 会返回失败，而失败只打印
        一行警告 —— 表现出来就是「选了彩色却是黑白」。
        """
        assert _resolve_pixel_format(given) == expected

    @pytest.mark.parametrize("given, expected", [
        ("  RGB8  ", "RGB8_Packed"),
        ("\tBgr8\n", "BGR8_Packed"),
        ("yUV422", "YUV422_Packed"),
    ])
    def test_case_and_whitespace_tolerated(self, given, expected):
        assert _resolve_pixel_format(given) == expected

    @pytest.mark.parametrize("given", [
        "BayerRG8",          # 用户直接填了 SDK 枚举名
        "BayerGB12",
        "Mono16",
        "RGB8_Planar",
        "YUV422_YUYV_Packed",
    ])
    def test_unknown_names_pass_through_unchanged(self, given):
        """表里没有的必须原样透传，不能被这张表挡住。"""
        assert _resolve_pixel_format(given) == given

    def test_alias_table_keys_are_normalized(self):
        """表键必须是小写去空格的，否则 .strip().lower() 后查不到。"""
        for key in _PIXEL_FORMAT_ALIASES:
            assert key == key.strip().lower(), f"别名表键不规范: {key!r}"

    def test_alias_values_are_unique_per_target(self):
        """同义写法可以多对一（rgb8 / rgb8_packed），但值都得是 SDK 枚举名。"""
        assert _PIXEL_FORMAT_ALIASES["rgb8"] == _PIXEL_FORMAT_ALIASES["rgb8_packed"]


# ============================================================================
# 错误码
# ============================================================================

class TestErr:
    """错误码 → 十六进制 + 中文说明。"""

    @pytest.mark.parametrize("code", sorted(_ERROR_HINTS))
    def test_known_codes_include_hex_and_hint(self, code):
        text = _err(code)
        assert "0x%08X" % code in text
        assert _ERROR_HINTS[code] in text

    def test_unknown_code_prints_hex_only(self):
        text = _err(0x12345678)
        assert text == "0x12345678"
        assert "（" not in text

    def test_code_is_masked_to_32_bits(self):
        """SDK 会把错误码当有符号整数返回，负数也要能对上表。"""
        assert _err(0xFFFFFFFF80000046 - 0xFFFFFFFF00000000) == _err(ERR_ACCESS_DENIED)
        assert _ERROR_HINTS[ERR_ACCESS_DENIED] in _err(-2147483578)

    def test_never_raises_on_odd_input(self):
        for value in (0, 1, -1, 2 ** 63, 0xDEADBEEF):
            assert isinstance(_err(value), str)


# ============================================================================
# 触发模式映射表
# ============================================================================

class TestTriggerModeTable:
    """_TRIGGER_MODES 的映射内容。"""

    def test_full_mapping(self):
        """逐项断言 mode → (TriggerMode, TriggerSource)。"""
        assert _TRIGGER_MODES == {
            "continuous": ("off", None),
            "off": ("off", None),
            "software": ("on", "Software"),
            "external": ("on", "Line0"),
            "line0": ("on", "Line0"),
            "line1": ("on", "Line1"),
            "line2": ("on", "Line2"),
            "line3": ("on", "Line3"),
        }

    @pytest.mark.parametrize("mode, expected_source", [
        ("software", "Software"),
        ("external", "Line0"),
        ("line0", "Line0"),
        ("line1", "Line1"),
        ("line2", "Line2"),
        ("line3", "Line3"),
    ])
    def test_triggered_modes_are_on_with_source(self, mode, expected_source):
        trig_mode, source = _TRIGGER_MODES[mode]
        assert trig_mode == "on"
        assert source == expected_source

    @pytest.mark.parametrize("mode", ["continuous", "off"])
    def test_free_run_modes_have_no_source(self, mode):
        assert _TRIGGER_MODES[mode] == ("off", None)


# ============================================================================
# 构造（不碰 SDK）
# ============================================================================

class TestMvsCameraInit:
    """__init__ 只读配置 —— 保证没装 MVS 的机器也能启动。"""

    def test_init_never_touches_sdk(self, monkeypatch, default_config):
        """构造过程绝不能加载 SDK：失败点必须推迟到 open()。"""
        def _boom(*a, **kw):
            raise AssertionError("__init__ 不应加载海康 SDK")

        monkeypatch.setattr(mvs, "_ensure_sdk", _boom)
        monkeypatch.setattr(mvs, "find_mvimport_dirs", _boom)
        monkeypatch.setattr(mvs, "find_mvs_runtime_dirs", _boom)
        camera = MvsCamera(default_config)          # 不抛即通过
        assert camera._cam is None
        assert camera._sdk is None
        assert camera.is_open is False
        assert camera.last_error == ""

    def test_importing_module_does_not_import_sdk(self, project_root):
        """模块级 import 不能拉进 MvCameraControl_class（否则没装 MVS 就起不来）。

        在**干净的子进程**里验证，而不是查当前进程的 sys.modules ——
        别的测试文件可能往 sys.modules 里塞假模块，直接查会受用例顺序影响。
        """
        code = (
            "import sys\n"
            "import hardware.mvs_camera as m\n"
            "assert 'MvCameraControl_class' not in sys.modules, "
            "'导入 hardware.mvs_camera 时不该加载海康 SDK'\n"
            "assert m._SDK is None, '导入时不该已经有 SDK 缓存'\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(project_root), capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    def test_reads_real_config_values(self, make_camera):
        """默认值来自 config/default.yaml，不是写死在代码里的。"""
        camera = make_camera()
        assert (camera.width, camera.height) == (2448, 2048)
        assert camera.exposure_us == 5000.0
        assert camera.gain == 1.0
        # 由 Mono8 改为 BGR8：色度/饱和度指标需要彩色输入。选 BGR8 而不是
        # RGB8，是因为 CameraBase 的契约就是 BGR，上层 _to_rgb 会再转成 RGB。
        assert camera.pixel_format == "BGR8"
        assert camera.trigger_mode == "continuous"
        assert camera.trigger_source == "Line0"
        assert camera.device_serial == ""
        assert camera.device_index == 0
        assert camera.acquire_timeout_ms == 2000

    def test_missing_camera_section_uses_defaults(self):
        camera = MvsCamera({})
        assert (camera.width, camera.height) == (2448, 2048)
        assert camera.pixel_format == "Mono8"
        assert camera.trigger_mode == "continuous"
        assert camera.device_index == 0

    def test_trigger_mode_is_lowercased(self, make_camera):
        camera = make_camera(trigger={"mode": "Software", "source": "Line2"})
        assert camera.trigger_mode == "software"
        assert camera.trigger_source == "Line2"

    def test_legacy_trigger_keys_still_honoured(self, make_camera):
        """旧配置（没有 camera.trigger 节）里的 trigger_mode / trigger_source 仍要生效。"""
        camera = make_camera(trigger=None, trigger_mode="external",
                             trigger_source="Line3")
        assert camera.trigger_mode == "external"
        assert camera.trigger_source == "Line3"

    def test_new_trigger_section_wins_over_legacy(self, make_camera):
        camera = make_camera(trigger={"mode": "software", "source": "Line1"},
                             trigger_mode="external", trigger_source="Line3")
        assert camera.trigger_mode == "software"
        assert camera.trigger_source == "Line1"

    def test_empty_serial_number_becomes_empty_string(self, make_camera):
        camera = make_camera(device={"index": 1, "serial_number": None})
        assert camera.device_serial == ""
        assert camera.device_index == 1

    def test_numeric_config_is_coerced(self, make_camera):
        camera = make_camera(width="1280", height="960", exposure_us="2000",
                             gain="2", acquire_timeout_ms="500")
        assert camera.width == 1280 and isinstance(camera.width, int)
        assert camera.height == 960
        assert camera.exposure_us == 2000.0
        assert camera.gain == 2.0
        assert camera.acquire_timeout_ms == 500


# ============================================================================
# open()
# ============================================================================

class TestOpen:
    """打开流程：枚举 → 选设备 → 建句柄 → 开设备 → 下发参数 → 取流。"""

    def test_open_success_sets_state_and_device_info(self, sdk, make_camera):
        camera = make_camera()
        assert camera.open() is True
        assert camera.is_open is True
        assert camera.last_error == ""
        info = camera.get_info()
        assert info["model"] == "MV-CE050-10GM"
        assert info["serial_number"] == "SN-0001"
        assert info["vendor"] == "Hikrobot"
        assert info["tlayer"] == "GEV"
        assert info["ip"] == "192.168.1.100"
        assert camera._frame_out is not None
        assert sdk.state.count("StartGrabbing") == 1

    def test_open_enumerates_configured_transport_layers(self, sdk, make_camera):
        """枚举必须同时请求网口与 USB，否则 USB 相机会被漏掉。"""
        camera = make_camera()
        camera.open()
        assert ("EnumDevices", GIGE | USB) in sdk.state.calls

    def test_open_uses_real_config_geometry(self, sdk, make_camera, default_config):
        """下发的宽高就是配置里的值 —— 配置改了这里会跟着变。"""
        camera = make_camera()
        camera.open()
        assert ("SetIntValue", "Width", default_config["camera"]["width"]) \
            in sdk.state.calls
        assert ("SetIntValue", "Height", default_config["camera"]["height"]) \
            in sdk.state.calls

    def test_open_applies_params_before_start_grabbing(self, sdk, make_camera):
        camera = make_camera()
        camera.open()
        names = sdk.state.names()
        assert names.index("SetEnumValueByString") < names.index("StartGrabbing")
        assert names.index("SetFloatValue") < names.index("StartGrabbing")
        assert names.index("StartGrabbing") < names.index("GetIntValueEx")

    def test_open_gige_probes_optimal_packet_size(self, sdk, make_camera):
        """GigE 必须探一次最佳包大小（巨帧），否则频繁丢包。"""
        camera = make_camera()
        camera.open()
        assert sdk.state.count("GetOptimalPacketSize") == 1
        assert ("SetIntValue", "GevSCPSPacketSize", 8164) in sdk.state.calls

    def test_open_usb_skips_packet_size_probe(self, sdk, make_camera):
        """USB 相机没有这个 GigE 专属节点，不该去探。"""
        sdk.state.devices = [make_device(serial="SN-USB", tlayer=USB)]
        camera = make_camera()
        assert camera.open() is True
        assert sdk.state.count("GetOptimalPacketSize") == 0

    def test_open_tolerates_packet_size_write_failure(self, sdk, make_camera):
        """包大小设不进去不影响取流（只是可能丢包），不能因此打不开相机。"""
        sdk.state.set_ret["GevSCPSPacketSize"] = ERR_UNSUPPORTED
        camera = make_camera()
        assert camera.open() is True
        assert camera.is_open is True

    def test_open_tolerates_bad_packet_size(self, sdk, make_camera, capsys):
        """探测结果为负（失败）时不崩，也不去写节点。"""
        sdk.state.packet_size = -1
        camera = make_camera()
        assert camera.open() is True
        assert ("SetIntValue", "GevSCPSPacketSize", 1) not in sdk.state.calls
        assert "最佳包大小" in capsys.readouterr().out

    def test_unsigned_error_code_packet_size_is_not_fatal(self, sdk, make_camera,
                                                          capsys):
        """SDK 把错误码当无符号数返回时也只是一条警告，不影响打开相机。

        0x80000007 > 0，会被当成「探到的包大小」，写下去自然失败；
        但那只是可能丢包，相机该能开还是能开。
        """
        sdk.state.packet_size = ERR_UNSUPPORTED
        sdk.state.set_ret["GevSCPSPacketSize"] = ERR_UNSUPPORTED
        camera = make_camera()
        assert camera.open() is True
        assert camera.is_open is True
        assert "GevSCPSPacketSize" in capsys.readouterr().out

    @pytest.mark.xfail(
        reason="缺陷：hardware/mvs_camera.py:266-270 的 _ensure_sdk 失败分支只 "
               "print 不写 last_error，GUI 弹窗（ui/camera_worker.py:116）因此 "
               "拿不到「MVS 没装」这个最可操作的原因，只剩泛泛的通用提示。",
        strict=False)
    def test_open_returns_false_when_sdk_unavailable(self, monkeypatch, make_camera):
        """SDK 装不上时返回 False，且要把可操作的原因写进 last_error。

        last_error 是 GUI 弹窗显示真实原因的唯一依据
        （ui/camera_worker.py:116），这里必须被填上 ——
        CameraBase.open 的契约也是这么写的（hardware/camera.py:451）。
        """
        def _boom():
            raise RuntimeError("未找到海康官方 Python 示例（MvImport）。")

        monkeypatch.setattr(mvs, "_ensure_sdk", _boom)
        camera = make_camera()
        assert camera.open() is False
        assert camera.is_open is False
        assert "MvImport" in camera.last_error

    def test_open_fails_when_enumeration_fails(self, sdk, make_camera):
        sdk.state.ret_enum = ERR_ACCESS_DENIED
        camera = make_camera()
        assert camera.open() is False
        assert "枚举设备失败" in camera.last_error
        assert "0x80000046" in camera.last_error
        assert _ERROR_HINTS[ERR_ACCESS_DENIED] in camera.last_error

    def test_open_fails_when_no_device_found(self, sdk, make_camera):
        sdk.state.devices = []
        camera = make_camera()
        assert camera.open() is False
        assert "未发现相机" in camera.last_error
        assert "192.168.1.x" in camera.last_error

    def test_open_fails_when_create_handle_fails(self, sdk, make_camera):
        sdk.state.ret_create = ERR_UNSUPPORTED
        camera = make_camera()
        assert camera.open() is False
        assert "创建句柄失败" in camera.last_error
        assert "0x80000007" in camera.last_error

    def test_open_fails_when_device_busy(self, sdk, make_camera):
        sdk.state.ret_open = ERR_ACCESS_DENIED
        camera = make_camera()
        assert camera.open() is False
        assert "打开设备失败" in camera.last_error
        assert "独占" in camera.last_error

    def test_open_fails_when_start_grabbing_fails(self, sdk, make_camera):
        sdk.state.ret_start = ERR_NO_DATA
        camera = make_camera()
        assert camera.open() is False
        assert "启动取流失败" in camera.last_error
        assert camera.is_open is False

    @pytest.mark.parametrize("attr", ["ret_create", "ret_open", "ret_start"])
    def test_failed_open_releases_handle(self, sdk, make_camera, attr):
        """句柄一旦建出来，失败路径就必须把它还回去，否则相机一直被占着。"""
        setattr(sdk.state, attr, ERR_UNSUPPORTED)
        camera = make_camera()
        assert camera.open() is False
        assert camera._cam is None
        assert sdk.state.count("StopGrabbing") == 1
        assert sdk.state.count("CloseDevice") == 1
        assert sdk.state.count("DestroyHandle") == 1

    def test_failed_enumeration_creates_no_handle(self, sdk, make_camera):
        """枚举就失败时压根没建句柄，不能去 Stop/Close/Destroy 一个空句柄。"""
        sdk.state.ret_enum = ERR_UNSUPPORTED
        camera = make_camera()
        assert camera.open() is False
        assert camera._cam is None
        assert sdk.state.names() == ["EnumDevices"]

    def test_open_fails_on_unexpected_exception(self, sdk, make_camera):
        """非 RuntimeError 的意外异常同样要落到 last_error 并清理句柄。"""
        sdk.state.raise_on["SetFloatValue"] = RuntimeError("驱动层炸了")
        camera = make_camera()
        assert camera.open() is False
        assert "驱动层炸了" in camera.last_error
        assert camera._cam is None
        assert sdk.state.count("DestroyHandle") == 1

    def test_open_after_failure_can_be_retried(self, sdk, make_camera):
        """一次失败后，条件恢复还能再打开（句柄不能留在半开状态）。"""
        sdk.state.ret_open = ERR_ACCESS_DENIED
        camera = make_camera()
        assert camera.open() is False
        sdk.state.ret_open = 0
        assert camera.open() is True
        assert camera.is_open is True


# ============================================================================
# 设备选择
# ============================================================================

class TestSelectDevice:
    """选设备 —— 多相机场景下选错就是拍错工位。"""

    def test_index_used_when_no_serial_configured(self, sdk, make_camera):
        sdk.state.devices = [make_device(serial="SN-A"), make_device(serial="SN-B")]
        camera = make_camera(device={"index": 1, "serial_number": ""})
        camera._sdk = sdk
        assert camera._select_device(enumerated(sdk)) == 1

    def test_index_out_of_range_raises(self, sdk, make_camera):
        sdk.state.devices = [make_device(serial="SN-A")]
        camera = make_camera(device={"index": 3, "serial_number": ""})
        camera._sdk = sdk
        with pytest.raises(RuntimeError) as ei:
            camera._select_device(enumerated(sdk))
        assert "3" in str(ei.value)
        assert "1" in str(ei.value)

    def test_serial_finds_matching_device(self, sdk, make_camera):
        sdk.state.devices = [make_device(serial="SN-A"), make_device(serial="SN-B"),
                             make_device(serial="SN-C")]
        camera = make_camera(device={"index": 0, "serial_number": "SN-B"})
        camera._sdk = sdk
        assert camera._select_device(enumerated(sdk)) == 1

    def test_missing_serial_raises_with_name_and_found_list(self, sdk, make_camera):
        """指定的序列号找不到时必须明确报错，**不能**悄悄退回第一台。

        错误信息里要同时有「没找到的序列号」与「实际枚举到的序列号」，
        否则现场只能靠猜。
        """
        sdk.state.devices = [make_device(serial="SN-A"), make_device(serial="SN-B")]
        camera = make_camera(device={"index": 0, "serial_number": "SN-WRONG"})
        camera._sdk = sdk
        with pytest.raises(RuntimeError) as ei:
            camera._select_device(enumerated(sdk))
        msg = str(ei.value)
        assert "SN-WRONG" in msg
        assert "SN-A" in msg and "SN-B" in msg

    def test_serial_match_is_exact(self, sdk, make_camera):
        """序列号必须精确匹配，不能被前缀/子串蒙混过去。"""
        sdk.state.devices = [make_device(serial="SN-10")]
        camera = make_camera(device={"index": 0, "serial_number": "SN-1"})
        camera._sdk = sdk
        with pytest.raises(RuntimeError):
            camera._select_device(enumerated(sdk))

    def test_serial_of_usb_device_is_read_from_union_member(self, sdk, make_camera):
        """USB 设备信息在联合体的另一个分支上，也要能取到序列号。"""
        sdk.state.devices = [make_device(serial="SN-GE", tlayer=GIGE),
                             make_device(serial="SN-USB", tlayer=USB)]
        camera = make_camera(device={"index": 0, "serial_number": "SN-USB"})
        camera._sdk = sdk
        assert camera._select_device(enumerated(sdk)) == 1

    def test_open_binds_to_configured_serial(self, sdk, make_camera):
        """整条 open 链路上序列号绑定同样生效。"""
        sdk.state.devices = [make_device(serial="SN-A", model="MODEL-A"),
                             make_device(serial="SN-B", model="MODEL-B")]
        camera = make_camera(device={"index": 0, "serial_number": "SN-B"})
        assert camera.open() is True
        assert camera.get_info()["model"] == "MODEL-B"

    def test_open_fails_loudly_on_unknown_serial(self, sdk, make_camera):
        sdk.state.devices = [make_device(serial="SN-A")]
        camera = make_camera(device={"index": 0, "serial_number": "SN-NOPE"})
        assert camera.open() is False
        assert "SN-NOPE" in camera.last_error
        assert "SN-A" in camera.last_error


# ============================================================================
# 设备信息（_special_info / _decode / _ip_to_str）
# ============================================================================

class TestSpecialInfo:
    """传输层专属结构体的取值。"""

    def test_gige_device(self, sdk):
        dev = _build_device_info(make_device(serial="SN-GE", model="MV-GE",
                                             user_name="line-1"))
        kind, spec = _special_info(sdk, dev)
        assert kind == "GEV"
        assert _decode(spec.chSerialNumber) == "SN-GE"
        assert _decode(spec.chModelName) == "MV-GE"
        assert int(spec.nCurrentIp) == _ip_to_int("192.168.1.100")

    def test_usb_device(self, sdk):
        dev = _build_device_info(make_device(serial="SN-USB", model="MV-USB",
                                             tlayer=USB))
        kind, spec = _special_info(sdk, dev)
        assert kind == "U3V"
        assert _decode(spec.chSerialNumber) == "SN-USB"

    def test_genTL_gige_is_treated_as_gige(self, sdk):
        dev = _build_device_info(make_device(serial="SN-GTL", tlayer=GENTL_GIGE))
        kind, spec = _special_info(sdk, dev)
        assert kind == "GEV"
        assert _decode(spec.chSerialNumber) == "SN-GTL"

    def test_unknown_transport_returns_none(self, sdk):
        dev = _build_device_info(make_device(tlayer=UNKNOWN_TLAYER))
        assert _special_info(sdk, dev) is None

    def test_returned_struct_shares_memory_with_union_member(self, sdk):
        """返回的就是 Union 里的那个结构体，与 dev_info 共享内存。

        所以**不需要**（也不允许）对它做 ctypes.cast ——
        见下面那条回归守卫。
        """
        dev = _build_device_info(make_device(serial="SN-1"))
        _, spec = _special_info(sdk, dev)
        # Union 的各个分支共享内存：改一处，另一处就变
        dev.SpecialInfo.stGigEInfo.chSerialNumber = b"SN-CHANGED"
        assert _decode(spec.chSerialNumber) == "SN-CHANGED"

    def test_never_casts_union_member(self, sdk, monkeypatch):
        """回归守卫：不得对 SpecialInfo 的字段做 ctypes.cast。

        ``cast(dev_info.SpecialInfo.stGigEInfo, POINTER(MV_GIGE_DEVICE_INFO))``
        会抛 ``ArgumentError: argument 1: TypeError: wrong type`` ——
        ctypes 不接受 Union 内嵌的 Structure 作为 cast 源对象，
        而那个字段本身就已经是目标类型的实例，cast 纯属多余。
        这里把 cast 换成严格版本：一旦有人加回去，测试立刻红。
        """
        real_cast = ctypes.cast

        def _strict_cast(obj, typ):
            if isinstance(obj, (MV_GIGE_DEVICE_INFO, MV_USB3_DEVICE_INFO)):
                raise AssertionError(
                    "不要对 SpecialInfo 的字段做 ctypes.cast（会抛 ArgumentError）")
            return real_cast(obj, typ)

        monkeypatch.setattr(mvs, "cast", _strict_cast)
        for tlayer in (GIGE, USB):
            dev = _build_device_info(make_device(tlayer=tlayer))
            _special_info(sdk, dev)                 # 不抛即通过
            camera = MvsCamera({})
            camera._sdk = sdk
            assert camera._describe_device(dev)["serial_number"] == "SN-0001"

    def test_describe_device_gige_has_ip(self, sdk):
        dev = _build_device_info(make_device(ip="10.0.0.7"))
        camera = MvsCamera({"camera": {}})
        camera._sdk = sdk
        info = camera._describe_device(dev)
        assert info["tlayer"] == "GEV"
        assert info["ip"] == "10.0.0.7"

    def test_describe_device_usb_has_no_ip(self, sdk):
        dev = _build_device_info(make_device(tlayer=USB))
        camera = MvsCamera({"camera": {}})
        camera._sdk = sdk
        info = camera._describe_device(dev)
        assert info["tlayer"] == "U3V"
        assert info["ip"] == ""


class TestDecodeHelpers:
    """_decode / _ip_to_str / _is_color_pixel_type。"""

    def test_decode_strips_zero_padding(self):
        raw = (ctypes.c_char * 16)(*b"SN-0001\x00\x00\x00\x00\x00\x00\x00\x00\x00")
        assert _decode(raw) == "SN-0001"

    def test_decode_accepts_bytes_and_bytearray(self):
        assert _decode(b"MV-CE050\x00\x00") == "MV-CE050"
        assert _decode(bytearray(b"MV-CE050\x00")) == "MV-CE050"

    def test_decode_strips_surrounding_whitespace(self):
        assert _decode(b"  MV-CE050  \x00") == "MV-CE050"

    def test_decode_never_raises_on_non_ascii(self):
        """海康的字符串偶尔不是纯 ASCII —— 不能因此崩掉。"""
        assert isinstance(_decode(b"\xd6\xd0\xce\xc4\x00"), str)

    def test_decode_falls_back_on_non_buffer_input(self):
        """ctypes 的字符数组之外的类型（如 str）走 str() 兜底，不抛。"""
        assert _decode("already-a-string") == "already-a-string"

    @pytest.mark.parametrize("value, expected", [
        (0xC0A80164, "192.168.1.100"),
        (0x00000000, "0.0.0.0"),
        (0xFFFFFFFF, "255.255.255.255"),
        (0x0A000007, "10.0.0.7"),
    ])
    def test_ip_to_str(self, value, expected):
        assert _ip_to_str(value) == expected

    def test_ip_to_str_masks_to_32_bits(self):
        """SDK 把 IP 当有符号整数返回时也要能转对。"""
        assert _ip_to_str(0xC0A80164 - 2 ** 32) == "192.168.1.100"

    @pytest.mark.parametrize("pixel_type, expected", [
        (MONO8, False),
        (MONO10, False),
        (MONO12, False),
        (RGB8_PACKED, True),
        (BGR8_PACKED, True),
        (YUV422_PACKED, True),
        (BAYER_RG8, True),
    ])
    def test_is_color_pixel_type(self, sdk, pixel_type, expected):
        """按名字判断：带 RGB/BGR/YUV/Bayer 的都算彩色。"""
        assert _is_color_pixel_type(sdk, pixel_type) is expected

    def test_unknown_pixel_type_is_not_color(self, sdk):
        """认不出的格式按灰度处理（转 Mono8 比乱猜彩色安全）。"""
        assert _is_color_pixel_type(sdk, 0xDEADBEEF) is False


# ============================================================================
# 参数下发
# ============================================================================

class TestApplyExposureGain:
    """曝光 / 增益的写入顺序。"""

    def test_auto_modes_are_turned_off_first(self, cam, sdk):
        """必须先关自动模式，否则写进去的值会被相机立刻覆盖。

        相机出厂默认 ExposureAuto=Continuous，此时写 ExposureTime 看着成功、
        实际完全没生效 —— 顺序反了就是白设。
        """
        cam._apply_exposure_gain()
        assert sdk.state.set_calls() == [
            ("SetEnumValueByString", "ExposureAuto", "Off"),
            ("SetFloatValue", "ExposureTime", 5000.0),
            ("SetEnumValueByString", "GainAuto", "Off"),
            ("SetFloatValue", "Gain", 1.0),
        ]

    def test_auto_off_precedes_its_value(self, cam, sdk):
        cam._apply_exposure_gain()
        calls = sdk.state.set_calls()
        assert calls.index(("SetEnumValueByString", "ExposureAuto", "Off")) < \
            calls.index(("SetFloatValue", "ExposureTime", 5000.0))
        assert calls.index(("SetEnumValueByString", "GainAuto", "Off")) < \
            calls.index(("SetFloatValue", "Gain", 1.0))

    def test_values_come_from_config(self, cam, sdk, make_camera):
        camera = make_camera(exposure_us=1234, gain=3.5)
        camera._sdk = sdk
        camera._cam = sdk.MvCamera()
        camera._apply_exposure_gain()
        assert ("SetFloatValue", "ExposureTime", 1234.0) in sdk.state.calls
        assert ("SetFloatValue", "Gain", 3.5) in sdk.state.calls

    def test_write_failures_do_not_raise(self, cam, sdk, capsys):
        """量程超了只是警告，不能中断初始化。"""
        sdk.state.set_ret["ExposureTime"] = ERR_UNSUPPORTED
        sdk.state.set_ret["Gain"] = ERR_UNSUPPORTED
        cam._apply_exposure_gain()
        out = capsys.readouterr().out
        assert "曝光" in out and "增益" in out


class TestApplyParams:
    """_apply_params 的整体顺序与别名解析。"""

    def test_full_call_order(self, cam, sdk, make_camera):
        """先几何（宽高）→ 像素格式 → 曝光增益 → 采集模式 → 触发。

        顺序有讲究：曝光/增益必须先关自动模式，像素格式必须在曝光之前
        （改格式会重置部分参数），触发最后（它决定相机会不会立刻跑起来）。
        """
        camera = make_camera()
        camera._sdk = sdk
        camera._cam = sdk.MvCamera()
        camera._apply_params()

        assert sdk.state.set_calls() == [
            ("SetIntValue", "Width", 2448),
            ("SetIntValue", "Height", 2048),
            # 配置里写的是 BGR8，落到 SDK 时经 _PIXEL_FORMAT_ALIASES 变成
            # BGR8_Packed。少了这一步 SDK 会返回失败、且只打一行警告，
            # 表现为「选了彩色却是黑白」。
            ("SetEnumValueByString", "PixelFormat", "BGR8_Packed"),
            ("SetEnumValueByString", "ExposureAuto", "Off"),
            ("SetFloatValue", "ExposureTime", 5000.0),
            ("SetEnumValueByString", "GainAuto", "Off"),
            ("SetFloatValue", "Gain", 1.0),
            ("SetEnumValueByString", "AcquisitionMode", "Continuous"),
            ("SetEnumValue", "TriggerMode", TRIGGER_MODE_OFF),
        ]

    def test_pixel_format_goes_through_alias_table(self, sdk, make_camera):
        """配置写 RGB8 也要落到 SDK 认的 RGB8_Packed。"""
        camera = make_camera(pixel_format="RGB8")
        camera._sdk = sdk
        camera._cam = sdk.MvCamera()
        camera._apply_params()
        assert ("SetEnumValueByString", "PixelFormat", "RGB8_Packed") \
            in sdk.state.calls

    def test_unknown_pixel_format_passes_through(self, sdk, make_camera):
        camera = make_camera(pixel_format="BayerRG8")
        camera._sdk = sdk
        camera._cam = sdk.MvCamera()
        camera._apply_params()
        assert ("SetEnumValueByString", "PixelFormat", "BayerRG8") \
            in sdk.state.calls

    def test_acquisition_mode_is_continuous(self, sdk, make_camera):
        camera = make_camera()
        camera._sdk = sdk
        camera._cam = sdk.MvCamera()
        camera._apply_params()
        assert ("SetEnumValueByString", "AcquisitionMode", "Continuous") \
            in sdk.state.calls

    def test_geometry_failure_is_not_fatal(self, sdk, make_camera, capsys):
        """相机对宽高对齐有要求，设不进去就沿用相机当前值。"""
        sdk.state.set_ret["Width"] = ERR_UNSUPPORTED
        sdk.state.set_ret["Height"] = ERR_UNSUPPORTED
        camera = make_camera()
        camera._sdk = sdk
        camera._cam = sdk.MvCamera()
        camera._apply_params()          # 不抛
        out = capsys.readouterr().out
        assert "宽度" in out and "高度" in out

    def test_pixel_format_failure_is_not_fatal(self, sdk, make_camera, capsys):
        sdk.state.set_ret["PixelFormat"] = ERR_UNSUPPORTED
        camera = make_camera()
        camera._sdk = sdk
        camera._cam = sdk.MvCamera()
        camera._apply_params()
        assert "像素格式" in capsys.readouterr().out


class TestApplyTrigger:
    """触发模式下发。"""

    def test_continuous_writes_trigger_mode_off(self, cam, sdk):
        cam._apply_trigger()
        assert sdk.state.set_calls() == [
            ("SetEnumValue", "TriggerMode", TRIGGER_MODE_OFF)]
        assert "SetEnumValueByString" not in sdk.state.names()

    @pytest.mark.parametrize("mode, source", [
        ("software", "Software"),
        ("line1", "Line1"),
        ("line3", "Line3"),
    ])
    def test_triggered_modes_write_on_plus_source(self, sdk, make_camera, mode, source):
        camera = make_camera(trigger={"mode": mode})
        camera._sdk = sdk
        camera._cam = sdk.MvCamera()
        camera._apply_trigger()
        assert sdk.state.set_calls() == [
            ("SetEnumValue", "TriggerMode", TRIGGER_MODE_ON),
            ("SetEnumValueByString", "TriggerSource", source),
        ]

    def test_external_uses_configured_source(self, sdk, make_camera):
        """外部触发允许配置覆盖触发源（例如接在 Line2 上）。"""
        camera = make_camera(trigger={"mode": "external", "source": "Line2"})
        camera._sdk = sdk
        camera._cam = sdk.MvCamera()
        camera._apply_trigger()
        assert ("SetEnumValueByString", "TriggerSource", "Line2") \
            in sdk.state.calls

    def test_external_defaults_to_line0(self, sdk, make_camera):
        camera = make_camera(trigger={"mode": "external"})
        camera._sdk = sdk
        camera._cam = sdk.MvCamera()
        camera._apply_trigger()
        assert ("SetEnumValueByString", "TriggerSource", "Line0") \
            in sdk.state.calls

    def test_unknown_mode_falls_back_to_continuous(self, sdk, make_camera, capsys):
        """未知模式按连续采集处理，并且要把模式名打出来便于排查。"""
        camera = make_camera(trigger={"mode": "hardware-sync"})
        camera._sdk = sdk
        camera._cam = sdk.MvCamera()
        camera._apply_trigger()
        assert sdk.state.set_calls() == [
            ("SetEnumValue", "TriggerMode", TRIGGER_MODE_OFF)]
        out = capsys.readouterr().out
        assert "未知的触发模式" in out
        assert "hardware-sync" in out

    def test_trigger_source_failure_is_warned(self, sdk, make_camera, capsys):
        """触发源设失败要提示「相机将一直等不到触发信号」。"""
        sdk.state.set_ret["TriggerSource"] = ERR_UNSUPPORTED
        camera = make_camera(trigger={"mode": "software"})
        camera._sdk = sdk
        camera._cam = sdk.MvCamera()
        camera._apply_trigger()
        assert "触发" in capsys.readouterr().out

    def test_trigger_constants_come_from_header_module(self, sdk, cam):
        """TriggerMode 用的常量在 CameraParams_header 里，也得取到。"""
        cam._apply_trigger()
        value = [c for c in sdk.state.calls if c[0] == "SetEnumValue"][-1][2]
        assert value == sdk.const("MV_TRIGGER_MODE_OFF")


# ============================================================================
# 节点读写
# ============================================================================

class TestNodeSetters:
    """_set_int / _set_float / _set_enum / _set_enum_str。"""

    def test_set_int_passes_int(self, cam, sdk):
        assert cam._set_int("Width", 2448) is True
        assert ("SetIntValue", "Width", 2448) in sdk.state.calls
        cam._set_int("Width", 2448.7)
        assert ("SetIntValue", "Width", 2448) in sdk.state.calls

    def test_set_float_passes_float(self, cam, sdk):
        assert cam._set_float("Gain", 2) is True
        assert ("SetFloatValue", "Gain", 2.0) in sdk.state.calls

    def test_set_enum_passes_int(self, cam, sdk):
        assert cam._set_enum("TriggerMode", TRIGGER_MODE_ON) is True
        assert ("SetEnumValue", "TriggerMode", TRIGGER_MODE_ON) in sdk.state.calls

    def test_set_enum_str_passes_string(self, cam, sdk):
        assert cam._set_enum_str("PixelFormat", "Mono8") is True
        assert ("SetEnumValueByString", "PixelFormat", "Mono8") in sdk.state.calls

    @pytest.mark.parametrize("method, args, node", [
        ("_set_int", ("Width", 1), "Width"),
        ("_set_float", ("Gain", 1.0), "Gain"),
        ("_set_enum", ("TriggerMode", 0), "TriggerMode"),
        ("_set_enum_str", ("PixelFormat", "Mono8"), "PixelFormat"),
    ])
    def test_failures_return_false_without_raising(self, cam, sdk, method, args, node):
        """按节点报错必须走返回值，不能抛异常打断初始化。"""
        sdk.state.set_ret[node] = ERR_UNSUPPORTED
        assert getattr(cam, method)(*args) is False


class TestGetIntUsesExApi:
    """读整数必须用 Ex 接口 —— 本文件要守住的核心坑。"""

    def test_returns_sane_size(self, cam, sdk):
        """宽高读出来必须落在合理范围。

        配错接口时实测读成 17282948401552×13039520712704 这种荒唐值 ——
        那种值不会有任何异常，只会让下游莫名其妙。
        """
        width = cam._get_int("Width")
        height = cam._get_int("Height")
        assert width == 2448
        assert height == 2048
        assert 0 < width <= 65535
        assert 0 < height <= 65535

    def test_calls_ex_api_only(self, cam, sdk):
        cam._get_int("Width")
        cam._get_int("Height")
        assert sdk.state.names() == ["GetIntValueEx", "GetIntValueEx"]
        assert "GetIntValue" not in sdk.state.names()

    def test_passes_eight_byte_struct(self, cam, sdk, monkeypatch):
        """传给 Ex 接口的必须是 MVCC_INTVALUE_EX（8 字节字段）。"""
        seen: Dict[str, object] = {}
        real = sdk.MvCamera.MV_CC_GetIntValueEx

        def spy(self, name, st):
            seen["type"] = type(st).__name__
            seen["size"] = sizeof(st)
            seen["field_type"] = st._fields_[0][1]
            return real(self, name, st)

        monkeypatch.setattr(sdk.MvCamera, "MV_CC_GetIntValueEx", spy)
        cam._get_int("Width")

        assert seen["type"] == "MVCC_INTVALUE_EX"
        assert seen["size"] == 4 * 8            # 4 个 8 字节字段
        assert seen["field_type"] is ctypes.c_longlong

    def test_wrong_api_would_yield_absurd_value(self, cam, sdk):
        """（反证）走非 Ex 接口读出来的就是荒唐值 —— 说明上面几条守得住。

        假 SDK 里的非 Ex 接口照真实宽度（4 字节）写入，配错结构体时
        8 字节的 nCurValue 会被「真值 + nMax 左移 32 位」拼起来。
        """
        st = sdk.H.MVCC_INTVALUE_EX()
        assert cam._cam.MV_CC_GetIntValue("Width", st) == 0
        assert st.nCurValue > 10 ** 10

    def test_struct_is_zeroed_before_use(self, cam, sdk, monkeypatch):
        """读之前必须先 memset —— 否则相机不填的字段会带进脏数据。"""
        seen: Dict[str, int] = {}
        real = sdk.MvCamera.MV_CC_GetIntValueEx

        def spy(self, name, st):
            st.nMax = 0x7FFFFFFF            # 先塞脏值
            seen["before"] = int(st.nMax)
            return real(self, name, st)

        monkeypatch.setattr(sdk.MvCamera, "MV_CC_GetIntValueEx", spy)

        # _get_int 内部 memset 在前、调用在后，所以脏值必然被清掉
        cam._get_int("Width")
        assert seen["before"] == 0x7FFFFFFF

    def test_returns_none_on_failure(self, cam, sdk):
        sdk.state.get_ret["Width"] = ERR_UNSUPPORTED
        assert cam._get_int("Width") is None


class TestNodeGetters:
    """_get_float / _get_enum / _get_string。"""

    def test_get_float(self, cam, sdk):
        assert cam._get_float("ExposureTime") == pytest.approx(5000.0)
        assert ("GetFloatValue", "ExposureTime") in sdk.state.calls

    def test_get_float_failure(self, cam, sdk):
        sdk.state.get_ret["Gain"] = ERR_UNSUPPORTED
        assert cam._get_float("Gain") is None

    def test_get_enum(self, cam, sdk):
        assert cam._get_enum("PixelFormat") == MONO8

    def test_get_enum_failure(self, cam, sdk):
        sdk.state.get_ret["PixelFormat"] = ERR_UNSUPPORTED
        assert cam._get_enum("PixelFormat") is None

    def test_get_string(self, cam, sdk):
        assert cam._get_string("DeviceUserID") == "cam-1"

    def test_get_string_failure(self, cam, sdk):
        sdk.state.get_ret["DeviceUserID"] = ERR_UNSUPPORTED
        assert cam._get_string("DeviceUserID") is None


# ============================================================================
# 取帧
# ============================================================================

class TestAcquire:
    """acquire 的缓冲管理与格式归一。"""

    def test_returns_none_when_not_open(self, cam, sdk):
        assert cam._is_open is False
        assert cam.acquire() is None
        assert sdk.state.calls == []

    def test_returns_none_when_handle_missing(self, cam, sdk):
        cam._is_open = True
        cam._cam = None
        assert cam.acquire() is None

    def test_returns_none_on_timeout(self, live_cam, sdk):
        """超时（无数据）不作为异常，交由上层统计丢帧。"""
        sdk.state.frame = None
        sdk.state.ret_get_image = ERR_NO_DATA
        assert live_cam.acquire() is None

    def test_timeout_does_not_free_buffer(self, live_cam, sdk):
        """没拿到缓冲就谈不上归还 —— 乱 Free 会污染 SDK 的缓冲池。"""
        sdk.state.frame = None
        live_cam.acquire()
        assert "FreeImageBuffer" not in sdk.state.names()

    def test_passes_configured_timeout(self, live_cam, sdk):
        set_frame(sdk)
        live_cam.acquire()
        assert ("GetImageBuffer", 2000) in sdk.state.calls

    def test_custom_timeout_from_config(self, sdk, make_camera):
        camera = make_camera(acquire_timeout_ms=250)
        camera._sdk = sdk
        camera._cam = sdk.MvCamera()
        camera._is_open = True
        camera._frame_out = sdk.H.MV_FRAME_OUT()
        set_frame(sdk)
        camera.acquire()
        assert ("GetImageBuffer", 250) in sdk.state.calls

    def test_mono8_frame_shape_and_values(self, live_cam, sdk):
        set_frame(sdk, width=8, height=6, pixel_type=MONO8)
        frame = live_cam.acquire()
        expected = np.frombuffer(
            _pattern(8 * 6), dtype=np.uint8).reshape(6, 8)
        assert frame is not None
        assert frame.shape == (6, 8)
        assert frame.dtype == np.uint8
        np.testing.assert_array_equal(frame, expected)

    def test_frees_buffer_on_success(self, live_cam, sdk):
        set_frame(sdk)
        live_cam.acquire()
        assert sdk.state.count("FreeImageBuffer") == 1

    def test_frees_buffer_when_extract_raises(self, live_cam, sdk, monkeypatch):
        """取帧途中抛异常也必须归还缓冲，否则几次之后就没缓冲可用了。"""
        set_frame(sdk)

        def _boom(*a, **kw):
            raise RuntimeError("转换炸了")

        monkeypatch.setattr(live_cam, "_extract", _boom)
        with pytest.raises(RuntimeError):
            live_cam.acquire()
        assert sdk.state.count("FreeImageBuffer") == 1

    def test_frees_buffer_when_conversion_fails(self, live_cam, sdk, capsys):
        set_frame(sdk, pixel_type=MONO10)
        sdk.state.ret_convert = ERR_UNSUPPORTED
        assert live_cam.acquire() is None
        assert sdk.state.count("FreeImageBuffer") == 1
        assert "像素格式转换失败" in capsys.readouterr().out

    def test_result_is_a_copy_not_a_view(self, live_cam, sdk):
        """归还缓冲后指针就失效，返回的数组绝不能是共享内存的视图。"""
        set_frame(sdk, width=8, height=6, pixel_type=MONO8)
        frame = live_cam.acquire()
        source = np.frombuffer(sdk.state.frame_buf, dtype=np.uint8)
        assert frame is not None
        assert not np.shares_memory(frame, source)

    def test_convert_result_is_a_copy(self, live_cam, sdk):
        """转换缓冲是复用的，结果也必须拷走。"""
        set_frame(sdk, width=8, height=6, pixel_type=MONO10)
        frame = live_cam.acquire()
        first = frame.copy()
        sdk.state.frame = None
        assert live_cam.acquire() is None
        np.testing.assert_array_equal(frame, first)

    def test_frame_out_is_reused(self, live_cam, sdk):
        """MV_FRAME_OUT 每帧 memset 复用，不重新分配。"""
        frame_out = live_cam._frame_out
        set_frame(sdk)
        live_cam.acquire()
        live_cam.acquire()
        assert live_cam._frame_out is frame_out


class TestAcquirePixelFormats:
    """非 Mono8 格式一律经 MV_CC_ConvertPixelType 归一。"""

    def test_color_frame_becomes_bgr(self, live_cam, sdk):
        set_frame(sdk, width=8, height=6, pixel_type=RGB8_PACKED)
        frame = live_cam.acquire()
        assert frame is not None
        assert frame.shape == (6, 8, 3)
        assert sdk.state.last_convert["src"] == RGB8_PACKED
        assert sdk.state.last_convert["dst"] == BGR8_PACKED
        assert sdk.state.last_convert["buf_size"] == 8 * 6 * 3

    @pytest.mark.parametrize("pixel_type", [RGB8_PACKED, BGR8_PACKED,
                                            YUV422_PACKED, BAYER_RG8])
    def test_all_color_formats_convert_to_bgr8(self, live_cam, sdk, pixel_type):
        set_frame(sdk, width=4, height=3, pixel_type=pixel_type)
        frame = live_cam.acquire()
        assert frame is not None and frame.shape == (3, 4, 3)
        assert sdk.state.last_convert["dst"] == BGR8_PACKED

    @pytest.mark.parametrize("pixel_type", [MONO10, MONO12])
    def test_deep_mono_converts_to_mono8(self, live_cam, sdk, pixel_type):
        """非 8 位灰度转成 Mono8，符合 CameraBase 的输出契约。"""
        set_frame(sdk, width=8, height=6, pixel_type=pixel_type)
        frame = live_cam.acquire()
        assert frame is not None
        assert frame.shape == (6, 8)
        assert frame.dtype == np.uint8
        assert sdk.state.last_convert["dst"] == MONO8
        assert sdk.state.last_convert["buf_size"] == 8 * 6

    def test_convert_payload_is_reshaped_correctly(self, live_cam, sdk):
        set_frame(sdk, width=8, height=6, pixel_type=RGB8_PACKED)
        frame = live_cam.acquire()
        expected = np.frombuffer(
            _convert_payload(8 * 6 * 3), dtype=np.uint8).reshape(6, 8, 3)
        np.testing.assert_array_equal(frame, expected)

    def test_src_len_is_frame_length(self, live_cam, sdk):
        data = _pattern(8 * 6 * 2)
        set_frame(sdk, width=8, height=6, pixel_type=MONO10, data=data)
        live_cam.acquire()
        assert sdk.state.last_convert["src_len"] == len(data)

    def test_convert_param_carries_geometry(self, live_cam, sdk):
        set_frame(sdk, width=7, height=5, pixel_type=MONO10)
        live_cam.acquire()
        assert sdk.state.last_convert["width"] == 7
        assert sdk.state.last_convert["height"] == 5

    def test_convert_buffer_reused_for_same_size(self, live_cam, sdk):
        set_frame(sdk, width=8, height=6, pixel_type=MONO10)
        live_cam.acquire()
        first = live_cam._convert_buf
        live_cam.acquire()
        assert live_cam._convert_buf is first
        assert live_cam._convert_buf_len == 8 * 6

    def test_convert_buffer_grows_for_larger_frame(self, live_cam, sdk):
        set_frame(sdk, width=8, height=6, pixel_type=MONO10)
        live_cam.acquire()
        first = live_cam._convert_buf
        set_frame(sdk, width=64, height=48, pixel_type=MONO10)
        live_cam.acquire()
        assert live_cam._convert_buf is not first
        assert live_cam._convert_buf_len == 64 * 48

    def test_mono8_does_not_convert(self, live_cam, sdk):
        """Mono8 是零拷贝直通，不该白跑一次格式转换。"""
        set_frame(sdk, pixel_type=MONO8)
        live_cam.acquire()
        assert "ConvertPixelType" not in sdk.state.names()
        assert live_cam._convert_buf is None


# ============================================================================
# 释放
# ============================================================================

class TestRelease:
    """release / _cleanup 的顺序与幂等。"""

    def test_release_order_is_stop_close_destroy(self, cam, sdk):
        """顺序有讲究：先停流、再关设备、最后销毁句柄。"""
        cam.release()
        assert sdk.state.names() == ["StopGrabbing", "CloseDevice", "DestroyHandle"]

    def test_release_is_idempotent(self, cam, sdk):
        cam.release()
        calls_after_first = list(sdk.state.calls)
        cam.release()
        cam.release()
        assert sdk.state.calls == calls_after_first

    def test_release_without_handle_is_safe(self, sdk, make_camera):
        camera = make_camera()
        camera._sdk = sdk
        camera.release()                # _cam 还是 None
        assert sdk.state.calls == []
        assert camera.is_open is False

    def test_release_clears_state(self, cam, sdk):
        cam._is_open = True
        cam._frame_out = sdk.H.MV_FRAME_OUT()
        cam._convert_buf = (c_ubyte * 16)()
        cam._convert_buf_len = 16
        cam.release()
        assert cam.is_open is False
        assert cam._cam is None
        assert cam._frame_out is None
        assert cam._convert_buf is None
        assert cam._convert_buf_len == 0

    @pytest.mark.parametrize("step", ["StopGrabbing", "CloseDevice", "DestroyHandle"])
    def test_release_swallows_sdk_errors(self, cam, sdk, step):
        """某一步失败不能中断后续步骤，也不能把异常抛给调用方。"""
        sdk.state.raise_on[step] = OSError("设备已掉线")
        cam.release()                   # 不抛
        assert cam._cam is None
        for name in ("StopGrabbing", "CloseDevice", "DestroyHandle"):
            assert sdk.state.count(name) == 1

    def test_acquire_after_release_returns_none(self, live_cam, sdk):
        set_frame(sdk)
        assert live_cam.acquire() is not None
        live_cam.release()
        assert live_cam.acquire() is None

    def test_context_manager_releases(self, sdk, make_camera):
        """with 块退出时必须释放，异常退出也一样。"""
        camera = make_camera()
        camera._sdk = sdk
        camera._cam = sdk.MvCamera()
        try:
            with camera:
                raise RuntimeError("块内异常")
        except RuntimeError:
            pass
        assert sdk.state.count("DestroyHandle") == 1
        assert camera.is_open is False


# ============================================================================
# 信息回读
# ============================================================================

class TestGetInfo:
    def test_empty_before_open(self, cam):
        assert cam.get_info() == {}

    def test_returns_device_info_copy(self, sdk, make_camera):
        camera = make_camera()
        camera.open()
        info = camera.get_info()
        assert info["serial_number"] == "SN-0001"
        assert info["tlayer"] == "GEV"
        info["model"] = "被改过了"
        assert camera.get_info()["model"] == "MV-CE050-10GM"


class TestGetActualParams:
    """回读实际生效值 —— 界面必须显示它，否则用户以为改成功了。"""

    def test_empty_without_handle(self, sdk, make_camera):
        camera = make_camera()
        camera._sdk = sdk
        assert camera.get_actual_params() == {}

    def test_reads_geometry_through_ex_api(self, cam, sdk):
        params = cam.get_actual_params()
        assert params["width"] == 2448
        assert params["height"] == 2048
        assert sdk.state.count("GetIntValueEx") == 2
        assert sdk.state.count("GetIntValue") == 0

    def test_reads_float_params_rounded(self, cam, sdk):
        """浮点参数四舍五入到三位小数 —— 界面不该显示 1234.5678901234567。"""
        sdk.state.floats["ExposureTime"] = 1234.56789
        sdk.state.floats["Gain"] = 1.23456
        params = cam.get_actual_params()
        assert params["exposure_us"] == pytest.approx(1234.568, abs=1e-3)
        assert params["gain"] == pytest.approx(1.235, abs=1e-3)

    def test_pixel_format_is_translated_to_name(self, cam, sdk):
        sdk.state.enums["PixelFormat"] = RGB8_PACKED
        assert cam.get_actual_params()["pixel_format"] == "RGB8_Packed"

    def test_enum_params_are_reported(self, cam, sdk):
        params = cam.get_actual_params()
        for key in ("exposure_auto", "gain_auto", "trigger_mode",
                    "trigger_source", "acquisition_mode"):
            assert key in params

    def test_unreadable_nodes_are_omitted(self, cam, sdk):
        """读不到的节点不能塞一个假值进去 —— 宁可缺键。"""
        sdk.state.get_ret["Width"] = ERR_UNSUPPORTED
        sdk.state.get_ret["Gain"] = ERR_UNSUPPORTED
        sdk.state.get_ret["TriggerSource"] = ERR_UNSUPPORTED
        params = cam.get_actual_params()
        assert "width" not in params
        assert "gain" not in params
        assert "trigger_source" not in params
        assert "height" in params

    def test_open_reads_back_actual_values(self, sdk, make_camera):
        camera = make_camera()
        assert camera.open() is True
        params = camera.get_actual_params()
        assert params["width"] == 2448
        assert params["pixel_format"] == "Mono8"


# ============================================================================
# 运行时改参数
# ============================================================================

class TestRuntimeSetters:
    def test_set_exposure_before_open_only_records(self, sdk, make_camera):
        camera = make_camera()
        camera._sdk = sdk
        camera.set_exposure(1500)
        assert camera.exposure_us == 1500.0
        assert sdk.state.calls == []

    def test_set_exposure_after_open_reapplies_with_auto_off(self, cam, sdk):
        cam.set_exposure(1500)
        assert cam.exposure_us == 1500.0
        calls = sdk.state.set_calls()
        assert calls.index(("SetEnumValueByString", "ExposureAuto", "Off")) < \
            calls.index(("SetFloatValue", "ExposureTime", 1500.0))

    def test_set_gain_after_open_reapplies(self, cam, sdk):
        cam.set_gain(4.0)
        assert cam.gain == 4.0
        calls = sdk.state.set_calls()
        assert ("SetFloatValue", "Gain", 4.0) in calls
        assert calls.index(("SetEnumValueByString", "GainAuto", "Off")) < \
            calls.index(("SetFloatValue", "Gain", 4.0))

    def test_set_exposure_accepts_int(self, cam):
        cam.set_exposure(800)
        assert isinstance(cam.exposure_us, float)
        assert cam.exposure_us == 800.0


# ============================================================================
# 设备枚举（list_mvs_devices）
# ============================================================================

class TestListMvsDevices:
    """扫描设备按钮与诊断脚本用的枚举接口。"""

    def test_returns_empty_and_stays_silent_when_sdk_missing(self, monkeypatch, capsys):
        """界面上「没扫到设备」是正常情况，不该弹错误、也不该往 stderr 喷。"""
        def _boom():
            raise RuntimeError("未找到海康官方 Python 示例（MvImport）。")

        monkeypatch.setattr(mvs, "_ensure_sdk", _boom)
        assert list_mvs_devices() == []
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_verbose_reports_why_sdk_missing(self, monkeypatch, capsys):
        """排查时必须能看到原因 —— verbose 就是为这个留的。"""
        def _boom():
            raise RuntimeError("未找到 MvCameraControl.dll。")

        monkeypatch.setattr(mvs, "_ensure_sdk", _boom)
        assert list_mvs_devices(verbose=True) == []
        assert "MvCameraControl.dll" in capsys.readouterr().err

    def test_returns_empty_when_enumeration_fails(self, sdk, capsys):
        sdk.state.ret_enum = ERR_ACCESS_DENIED
        assert list_mvs_devices() == []
        assert list_mvs_devices(verbose=True) == []
        assert "0x80000046" in capsys.readouterr().err

    def test_gige_device_fields(self, sdk):
        sdk.state.devices = [make_device(
            serial="SN-GE", model="MV-CE050-10GM", vendor="Hikrobot",
            user_name="line-1", ip="192.168.1.100", tlayer=GIGE)]
        devices = list_mvs_devices()
        assert len(devices) == 1
        assert devices[0] == {
            "index": "0",
            "tl_type": "GigE",
            "vendor": "Hikrobot",
            "model": "MV-CE050-10GM",
            "serial_number": "SN-GE",
            "ip": "192.168.1.100",
            "user_name": "line-1",
        }

    def test_usb_device_has_no_ip(self, sdk):
        sdk.state.devices = [make_device(serial="SN-USB", tlayer=USB)]
        devices = list_mvs_devices()
        assert devices[0]["tl_type"] == "USB3"
        assert devices[0]["ip"] == ""

    def test_index_follows_enumeration_order(self, sdk):
        sdk.state.devices = [make_device(serial="SN-A"), make_device(serial="SN-B")]
        devices = list_mvs_devices()
        assert [d["index"] for d in devices] == ["0", "1"]
        assert [d["serial_number"] for d in devices] == ["SN-A", "SN-B"]

    def test_unknown_transport_is_skipped(self, sdk):
        """认不出的传输层跳过而不是塞一堆空字段。"""
        sdk.state.devices = [make_device(serial="SN-A"),
                             make_device(serial="SN-X", tlayer=UNKNOWN_TLAYER),
                             make_device(serial="SN-B")]
        devices = list_mvs_devices()
        assert [d["serial_number"] for d in devices] == ["SN-A", "SN-B"]

    def test_no_devices_returns_empty_list(self, sdk):
        sdk.state.devices = []
        assert list_mvs_devices() == []

    def test_internal_exception_is_swallowed(self, sdk, capsys, monkeypatch):
        """枚举途中出异常返回空列表而不抛；verbose 时要把堆栈打出来。

        早先这里静默吞掉了一个 ctypes cast 错误，表现为「一台设备都扫不到」，
        排查时多绕了一大圈 —— 所以 verbose 分支必须看得见。
        """
        def _boom():
            raise RuntimeError("结构体构造失败")

        monkeypatch.setattr(sdk.H, "MV_CC_DEVICE_INFO_LIST", _boom)
        assert list_mvs_devices() == []
        assert list_mvs_devices(verbose=True) == []
        assert "结构体构造失败" in capsys.readouterr().err


# ============================================================================
# 真实硬件（默认不跑）
# ============================================================================

@pytest.mark.hardware
@pytest.mark.skipif(
    not os.environ.get("PCB_MVS_HARDWARE_TEST"),
    reason="需要真实海康相机与 MVS 运行时；设 PCB_MVS_HARDWARE_TEST=1 才运行",
)
class TestRealMvsCamera:
    """真机冒烟用例 —— 默认被环境变量挡住，CI 上不会执行。

    这一组是唯一允许碰真实 SDK 的地方；它不在默认套件里，
    所以「换台没装 MVS 的机器也能跑」这条不受影响。
    """

    def test_enumerate_open_acquire_release(self):
        """枚举 → 打开 → 取一帧 → 释放。"""
        devices = list_mvs_devices(verbose=True)
        if not devices:
            pytest.skip("未枚举到海康相机")

        camera = MvsCamera({"camera": {"width": 640, "height": 480,
                                       "exposure_us": 2000, "gain": 1.0,
                                       "pixel_format": "Mono8"}})
        assert camera.open() is True, camera.last_error
        try:
            frame = camera.acquire()
            assert frame is not None, "取帧超时"
            assert frame.dtype == np.uint8
            assert frame.ndim in (2, 3)
        finally:
            camera.release()
