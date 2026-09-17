#!/usr/bin/env python
"""海康（及通用 GenICam）相机连通性诊断脚本。

不开 GUI，逐项检查从「MVS 装没装」到「能不能取到一帧图」的全链路，
每一步失败都给可操作的中文提示。

用法::

    python tools/check_camera.py                  # 默认走海康官方 SDK
    python tools/check_camera.py --driver harvesters  # 通用 GenTL 路线
    python tools/check_camera.py --driver opencv  # 检查 OpenCV 相机
    python tools/check_camera.py --frames 50      # 多抓几帧测帧率
    python tools/check_camera.py --list-only      # 只枚举设备，不打开
    python tools/check_camera.py --config config/default.yaml

两条海康路线（默认 mvs）：
    mvs        — 海康官方 SDK（MvCameraControl.dll，经 MvImport 封装）。
                 推荐。不过 GenTL，没有编码问题。
    harvesters — 通用 GenTL。海康的 producer 不符合规范的 UTF-8 要求，
                 create 阶段就会抛 UnicodeDecodeError，取回的帧也是全黑的。
                 保留它是给 Basler / 大恒等其它厂商的相机用的。

典型排查顺序：
    1. 报「未找到 MvImport」→ 重跑 MVS 安装包勾上「SDK / 开发组件」
    2. 报「未发现相机」→ 查网线/网段/供电
    3. 能枚举但打不开 → 相机被 MVS 客户端或别的程序占用
    4. 能打开但取不到帧 → 调大曝光、查网卡巨帧设置
    5. 帧率远低于预期 → 查 GigE 带宽与包大小（用 MVS 的网卡配置工具）

另：harvesters 加载 producer 时会往 stderr 刷若干行
「GenTL producer does not implement DSGetNumFlows」—— 那是 GenTL 可选
接口缺失，由 C 扩展直接输出，Python 层拦不住，对诊断结论无影响。
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# 允许从仓库根目录直接运行（python tools/check_camera.py）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# harvesters 在加载 producer 时会刷一串 "does not implement DSGetNumFlows"
# 之类的 WARNING —— 那是 GenTL 可选接口缺失，对诊断结论没有影响，
# 但会把真正有用的输出顶出屏幕。
logging.getLogger("harvesters").setLevel(logging.ERROR)

import numpy as np

from hardware.camera import (
    GENICAM_DRIVER_ALIASES,
    MVS_SDK_DRIVER_ALIASES,
    create_camera,
    describe_producers,
    find_gentl_producers,
    find_mvimport_dirs,
    list_genicam_devices,
    python_bitness,
)


# ============================================================================
# 输出小工具
# ============================================================================

def title(text: str) -> None:
    print()
    print("=" * 68)
    print(f"  {text}")
    print("=" * 68)


def step(text: str) -> None:
    print(f"\n[检查] {text}")


def ok(text: str) -> None:
    print(f"  [OK] {text}")


def warn(text: str) -> None:
    print(f"  [!]  {text}")


def bad(text: str) -> None:
    print(f"  [X]  {text}")


def hint(*lines: str) -> None:
    for line in lines:
        print(f"      {line}")


# ============================================================================
# 各项检查
# ============================================================================

def check_environment() -> bool:
    """检查 Python 与依赖库。"""
    step("运行环境")
    print(f"  Python {sys.version.split()[0]} ({python_bitness()} 位)")
    print(f"  解释器 {sys.executable}")

    healthy = True
    for name in ("numpy", "cv2", "harvesters"):
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "?")
            ok(f"{name} {version}")
        except ImportError:
            if name == "harvesters":
                bad(f"{name} 未安装")
                hint("pip install harvesters")
                healthy = False
            else:
                bad(f"{name} 未安装")
                healthy = False

    if python_bitness() == 32:
        warn("当前是 32 位 Python —— 无法加载 64 位 MVS 的 GenTL producer。")
        hint("请改用 64 位 Python，或安装 32 位 MVS 运行时。")

    return healthy


def check_mvimport() -> bool:
    """检查海康官方 Python 示例（MvImport）是否就位，并实际加载一次 SDK。

    这是 mvs 后端的硬性前置条件，缺了就没法用。
    """
    step("海康官方 Python 示例（MvImport）")

    required = [
        "MvCameraControl_class.py",
        "MvErrorDefine_const.py",
        "CameraParams_header.py",
        "CameraParams_const.py",
        "PixelType_header.py",
    ]

    found_dirs = find_mvimport_dirs()
    if not found_dirs:
        bad("未找到 MvImport")
        hint("")
        hint("MVS 安装器里「SDK / 开发组件」默认可能不勾。重跑安装包勾上即可，")
        hint("示例会在 <MVS安装目录>\\Development\\Samples\\Python\\MvImport\\。")
        hint("安装目录可以改（本机就装在 E:\\APP\\MVS），程序按注册表查找，")
        hint("不限于 C 盘。")
        hint("")
        hint("需要这几个文件:")
        for f in required:
            hint(f"    {f}")
        return False

    ok(f"找到 MvImport 目录 {len(found_dirs)} 处")
    for d in found_dirs:
        present = [f for f in required if os.path.isfile(os.path.join(d, f))]
        missing = [f for f in required if f not in present]
        print(f"      {d}")
        print(f"        已有 {len(present)}/{len(required)} 个文件")
        if missing:
            warn(f"        缺少: {', '.join(missing)}")
    print("      不需要手动加 sys.path —— hardware/mvs_camera.py 会自己定位。")

    # 真正加载一次。DLL 加载失败只有实际 import 才会暴露，
    # 光看文件在不在是不够的（位数不匹配、缺 VC 运行库都会在这里现形）。
    step("实际加载 SDK")
    try:
        from hardware.mvs_camera import _ensure_sdk
        sdk = _ensure_sdk()
        ok("MvCameraControl.dll 加载成功")
        print("      SDK 版本: 0x%08X" % sdk.MvCamera.MV_CC_GetSDKVersion())
        print("      已识别像素格式 %d 种" % len(sdk.pixel_names))
    except Exception as e:
        bad(f"加载失败: {e}")
        hint("")
        hint("常见原因：")
        hint("  - Python 位数与 MVS 不一致（32 位 Python 加载不了 64 位 DLL）")
        hint("  - 缺少 VC 运行库（装一下 MVS 自带的运行时组件）")
        return False
    return True


def check_mvs_enumeration() -> bool:
    """用海康官方 SDK 枚举设备。"""
    step("枚举海康相机（官方 SDK）")
    from hardware.mvs_camera import list_mvs_devices

    devices = list_mvs_devices(verbose=True)
    if not devices:
        bad("一台相机都没枚举到")
        hint("")
        hint("按顺序排查：")
        hint("  1. 相机是否上电（看相机指示灯）")
        hint("  2. 网线是否插好，网口灯是否闪")
        hint("  3. 相机与网卡是否在同一网段")
        hint(r"     海康相机出厂默认 IP 多为 192.168.1.x，")
        hint(r"     你的网卡需要配成同网段，例如 192.168.1.100 / 255.255.255.0")
        hint("  4. 先用 MVS 客户端确认能连上 —— 若 MVS 也连不上，是网络/供电问题，")
        hint("     与本项目代码无关")
        hint("  5. 防火墙可能拦截 GigE 发现包（UDP 广播）")
        return False

    ok(f"枚举到 {len(devices)} 台设备")
    for dev in devices:
        print(f"      [{dev['index']}] {dev['vendor']} {dev['model']}")
        print(f"          序列号: {dev['serial_number']}   "
              f"接口: {dev['tl_type']}   IP: {dev.get('ip') or '-'}")
        ip = dev.get("ip", "")
        if ip.startswith("169.254."):
            warn(f"          IP 是 169.254.x.x（链路本地地址）—— ")
            hint("             说明相机没拿到 DHCP 地址，且与网卡不在同一网段。")
            hint("             能用，但建议把相机和网卡都配成固定 IP（如 192.168.1.x），")
            hint("             否则换机器/重启后可能连不上。")
    return True


def check_producers() -> bool:
    """搜索 GenTL producer，未找到时给出安装指引。"""
    step("GenTL Producer（.cti）搜索")
    producers = find_gentl_producers(verbose=True)

    if not producers:
        bad("未找到任何 .cti 文件")
        hint("")
        hint("海康 MV-系列相机必须安装 MVS 才能被识别：")
        hint("  1. 到海康机器人官网下载 MVS 安装包（选与 Python 位数一致的版本）")
        hint("  2. 安装时勾选「GenICam」/「运行时」组件")
        hint("  3. 安装后重启本终端（环境变量需要重新加载）")
        hint("")
        hint("默认安装位置应为（注意在 Common Files 下）：")
        hint(r"  C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64\MvProducerGEV.cti")
        hint("")
        hint("若已安装但仍找不到，手动指定环境变量：")
        hint(r"  set GENICAM_GENTL64_PATH=C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64")
        return False

    ok(f"找到 {len(producers)} 个 producer")
    for info in describe_producers(producers):
        print(f"      [{info['vendor']} / {info['transport']}] {info['file']}")
        print(f"        {info['path']}")

    has_hik = any("海康" in i["vendor"] for i in describe_producers(producers))
    if has_hik:
        ok("其中包含海康 MVS 的 producer")
    else:
        warn("未发现海康 MVS 的 producer —— 若相机是海康的，请确认 MVS 已安装")
    return True


def check_enumeration() -> bool:
    """枚举设备。"""
    step("枚举 GenICam 设备")
    # verbose=False：上一步已经详细列过 producer 搜索过程，这里再列一遍纯属刷屏
    devices = list_genicam_devices(verbose=False)

    if not devices:
        bad("producer 已加载，但一台相机都没枚举到")
        hint("")
        hint("按顺序排查：")
        hint("  1. 相机是否上电（看相机指示灯）")
        hint("  2. 网线是否插好，网口灯是否闪")
        hint("  3. 相机与网卡是否在同一网段")
        hint(r"     海康相机出厂默认 IP 多为 192.168.1.x，")
        hint(r"     你的网卡需要配成同网段，例如 192.168.1.100 / 255.255.255.0")
        hint("  4. 先用 MVS 客户端确认能连上 —— 若 MVS 也连不上，是网络/供电问题，")
        hint("     与本项目代码无关")
        hint("  5. 防火墙可能拦截 GigE 发现包（UDP 广播）")
        return False

    ok(f"枚举到 {len(devices)} 台设备")
    for dev in devices:
        print(f"      [{dev['index']}] {dev['vendor']} {dev['model']}")
        print(f"          序列号: {dev['serial_number']}   接口: {dev['tl_type']}")
        if not dev.get("access_ok", True):
            warn(f"          访问状态: {dev['access_status']} "
                 f"（可能被其他程序占用）")
        else:
            print(f"          访问状态: {dev['access_status']}")
    return True


def check_open_and_grab(camera, frames: int, save_frame: bool) -> bool:
    """打开相机、读回参数、连拍若干帧。"""
    step("打开相机")
    t0 = time.perf_counter()
    try:
        opened = camera.open()
    except Exception as e:
        bad(f"打开时抛出异常: {e}")
        return False
    elapsed = (time.perf_counter() - t0) * 1000

    if not opened:
        bad(f"打开失败（耗时 {elapsed:.0f} ms）")
        hint("")
        hint("先看上面 [Camera] 开头的行 —— 具体原因在那里。")
        hint("")
        hint("若那行提到 UTF-8 / 解码：")
        hint("  这是海康 GenTL producer 不符合规范（返回非 UTF-8 字符串）导致，")
        hint("  相机和网络没问题，换海康官方 Python 封装（MvImport）即可绕过。")
        hint("")
        hint("否则按常见原因排查：")
        hint("  - 相机已被 MVS 客户端或另一个进程独占（先关掉它们）")
        hint("  - 分辨率/像素格式不被支持（改小一些再试）")
        hint("  - GigE 带宽不足，先降低分辨率")
        return False

    ok(f"打开成功（耗时 {elapsed:.0f} ms）")

    info = camera.get_info()
    if info:
        print(f"      型号: {info.get('model', '?')}    "
              f"序列号: {info.get('serial_number', '?')}")

    # 实际生效参数 —— 与配置里请求的值可能不同
    step("实际生效的参数（非配置文件里的请求值）")
    params = camera.get_actual_params()
    if not params:
        warn("该后端未提供参数回读")
    else:
        for key, value in params.items():
            print(f"      {key}: {value}")

        if str(params.get("exposure_auto", "")).lower() not in ("", "off"):
            warn(f"曝光仍处于自动模式 ({params['exposure_auto']}) —— "
                 f"此时手动设置的曝光时间不会生效")
        if str(params.get("gain_auto", "")).lower() not in ("", "off"):
            warn(f"增益仍处于自动模式 ({params['gain_auto']})")

    # 连拍测帧率
    step(f"连续抓取 {frames} 帧")
    images = []
    failures = 0
    t0 = time.perf_counter()
    for i in range(frames):
        img = camera.acquire()
        if img is None:
            failures += 1
        else:
            images.append(img)
    elapsed = time.perf_counter() - t0

    if not images:
        bad("一帧都没取到")
        hint("")
        hint("可能原因：")
        hint("  - 曝光时间过长（第一次调试建议先设 5000 μs 左右）")
        hint("  - 触发模式设成了外部触发但没接信号源 —— 相机在等触发信号，")
        hint("    自然一帧都不出。连续采集请把 trigger.mode 设为 continuous")
        hint("  - GigE 包大小/巨帧未配置，传输超时")
        hint("  - 网线质量差或交换机带宽不足")
        return False

    if failures:
        warn(f"{failures}/{frames} 帧抓取失败（超时）")
    else:
        ok(f"{len(images)}/{frames} 帧全部成功")

    fps = len(images) / elapsed if elapsed > 0 else 0
    shape = images[0].shape
    dtype = images[0].dtype
    print(f"      分辨率/通道: {shape}   数据类型: {dtype}")
    print(f"      耗时 {elapsed:.2f} s  平均帧率 {fps:.1f} FPS")

    # 图像内容自检：全黑/全白说明曝光或镜头有问题
    step("图像内容自检")
    sample = images[-1]
    mean = float(np.mean(sample))
    std = float(np.std(sample))
    print(f"      灰度均值 {mean:.1f}   标准差 {std:.1f}")

    if mean < 5:
        warn("图像接近全黑 —— 检查镜头盖是否取下、曝光是否过短、光源是否打开")
    elif mean > 250:
        warn("图像接近全白 —— 曝光过长或光源过强，考虑调小曝光时间")
    elif std < 2:
        warn("图像几乎无变化 —— 可能对着白墙或镜头未对焦")
    else:
        ok("图像内容正常（有明暗分布）")

    if save_frame:
        import cv2
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "camera_check.png"
        cv2.imwrite(str(out_path), sample)
        ok(f"已保存一帧到 {out_path}")

    return True


# ============================================================================
# 主流程
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="海康/GenICam 相机连通性诊断",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--driver", default="mvs",
                        help="相机驱动: mvs（海康官方 SDK，推荐）| "
                             "harvesters（通用 GenTL）| opencv")
    parser.add_argument("--frames", type=int, default=30,
                        help="连拍帧数，用于测帧率（默认 30）")
    parser.add_argument("--config", default=None,
                        help="配置文件路径，用于读取分辨率/曝光等参数")
    parser.add_argument("--list-only", action="store_true",
                        help="只枚举设备，不打开相机")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存测试帧")
    args = parser.parse_args()

    title("海康 / GenICam 相机连通性诊断")

    if not check_environment():
        bad("依赖检查未通过，先解决上面的问题")
        return 1

    # 默认驱动是 mvs；若它不可用，明确提示怎么退回 GenICam 路线
    # （具体的 MvImport 检查在下面的驱动分支里做）

    # 组装配置：默认值 + 配置文件覆盖
    config = {
        "system": {},
        "camera": {
            "driver": args.driver,
            "width": 2448,
            "height": 2048,
            "exposure_us": 5000,
            "gain": 1.0,
            "pixel_format": "Mono8",
            "device": {"index": 0, "serial_number": ""},
            "trigger": {"mode": "continuous", "source": "Line0"},
            "acquire_timeout_ms": 2000,
        },
    }
    if args.config:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            file_config = yaml.safe_load(f) or {}
        config["system"].update(file_config.get("system", {}))
        config["camera"].update(file_config.get("camera", {}))
        ok(f"已读取配置 {args.config}")
        print(f"      驱动={config['camera'].get('driver')} "
              f"分辨率={config['camera'].get('width')}×{config['camera'].get('height')} "
              f"曝光={config['camera'].get('exposure_us')}μs")

    driver = args.driver.lower()
    if driver in MVS_SDK_DRIVER_ALIASES:
        # 海康官方 SDK 路线：不过 GenTL，需要的是 MvImport + DLL，不是 .cti
        if not check_mvimport():
            title("结论")
            bad("缺少海康官方 Python 示例（MvImport）—— 请按上面的提示安装")
            return 1
        if not check_mvs_enumeration():
            title("结论")
            bad("找不到相机 —— 请先按上面的提示检查连接")
            return 1
    elif driver in GENICAM_DRIVER_ALIASES:
        if not check_producers():
            title("结论")
            bad("环境不完整 —— 请先按上面的提示安装 MVS")
            return 1
        if not check_enumeration():
            title("结论")
            bad("找不到相机 —— 请先按上面的提示检查连接")
            return 1
    else:
        step("跳过 GenTL 检查（当前驱动为 opencv）")

    if args.list_only:
        title("结论")
        ok("设备枚举正常（--list-only 未打开相机）")
        return 0

    camera = create_camera(config)
    grab_ok = check_open_and_grab(camera, args.frames, not args.no_save)

    step("释放相机")
    try:
        camera.release()
        ok("已释放")
    except Exception as e:
        warn(f"释放时出错: {e}")

    title("结论")
    if grab_ok:
        ok(f"相机链路全部打通，可以在软件里把 camera.driver 设为 {args.driver} 了")
        print()
        print(f"  下一步：启动主程序 → 菜单「相机」→ 驱动选 {args.driver} → 实时预览")
        return 0

    bad("打开或取帧失败 —— 请按上面的提示逐项排查")
    return 1


if __name__ == "__main__":
    sys.exit(main())
