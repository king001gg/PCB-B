"""hardware/camera.py（相机抽象层与驱动路由）单元测试。

覆盖四块：
    1. CameraBase 抽象契约 —— 后端必须实现哪些方法、last_error 的语义
    2. create_camera 驱动路由 —— 别名表、默认值、非法值去向
    3. OpenCVCamera —— 设备编号来源、打开失败、取帧、释放幂等
    4. 设备发现工具函数与 GenICamCamera 的纯逻辑部分

两条贯穿全篇的原则：

* **不碰真实硬件**。不开摄像头、不枚举网络相机、不读本机的 MVS 安装。
  凡是会落到真实设备 / 真实注册表 / 真实 Program Files 的分支，
  一律用 monkeypatch 隔离成常量输入。需要真实硬件的用例加
  ``@pytest.mark.hardware`` 且默认不跑（本文件当前没有这类用例）。
* **不断言实现细节的偶然值**。期望值尽量由输入算出（例如曝光换算），
  硬编码的数字只用于真正的契约（分辨率的配置默认值等）。

已知缺陷用 ``@pytest.mark.xfail(reason=...)``（非 strict）标注：
行为若被修好，用例会变成 XPASS 而不会让套件变红。
"""

import builtins
import struct
import sys
import types
from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np
import pytest

import hardware.camera as cam
from hardware.camera import (
    _ACCESS_STATUS_BAD,
    _ACCESS_STATUS_LABELS,
    GENICAM_DRIVER_ALIASES,
    MVS_SDK_DRIVER_ALIASES,
    CameraBase,
    GenICamCamera,
    OpenCVCamera,
    create_camera,
)


# ============================================================================
# 通用假对象
# ============================================================================

class FakeVideoCapture:
    """替换 cv2.VideoCapture 的假对象。

    只实现 OpenCVCamera 用到的那几个方法，并记录调用，
    以便断言「参数确实下发给了设备」而不是只改了 Python 属性。
    """

    def __init__(self, opened: bool = True, read_ok: bool = True,
                 width: float = 2448.0, height: float = 2048.0):
        self._opened = opened
        self._read_ok = read_ok
        self._props = {
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
        }
        self.set_calls = []
        self.read_count = 0
        self.release_count = 0

    # --- VideoCapture 接口 ---
    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def get(self, prop):
        return self._props.get(prop, 0.0)

    def read(self):
        self.read_count += 1
        if not self._read_ok:
            return False, None
        return True, np.full((4, 4, 3), 7, dtype=np.uint8)

    def release(self):
        self.release_count += 1


class FakeNode:
    """GenICam 节点：只存一个值。"""

    def __init__(self, value=None):
        self.value = value


class _ReadOnlyNode:
    """赋值即抛异常的节点 —— 模拟只读节点 / 超出量程。"""

    def __init__(self):
        self._value = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, _value):
        raise RuntimeError("node is read-only")


class FakeNodeMap:
    """假节点树。

    ``missing`` 里的节点在 get_node 时抛异常 —— 模拟「这台相机没有该节点」
    （例如无 Gain 的型号），用来验证 _set_node 的容错分支。
    ``broken`` 里的节点赋值时抛异常 —— 模拟只读节点 / 量程不符。
    """

    def __init__(self, missing=(), broken=(), values=None):
        self.missing = set(missing)
        self.broken = set(broken)
        self.nodes = {}
        #: 子类用来记录节点的写入顺序
        self.write_log = []
        for name, value in (values or {}).items():
            self.nodes[name] = FakeNode(value)

    def get_node(self, name):
        if name in self.missing:
            raise KeyError(name)
        if name in self.broken:
            return self.nodes.setdefault(name, _ReadOnlyNode())
        return self.nodes.setdefault(name, FakeNode())


class _RecordingNodeMap(FakeNodeMap):
    """额外记录写入顺序 —— 曝光与自动模式的先后顺序是有讲究的。"""

    def get_node(self, name):
        node = super().get_node(name)
        return _RecordingNode(node, name, self)


class _RecordingNode:
    def __init__(self, node, name, node_map):
        self._node = node
        self._name = name
        self._map = node_map

    @property
    def value(self):
        return self._node.value

    @value.setter
    def value(self, value):
        self._map.write_log.append((self._name, value))
        self._node.value = value


class FakeDeviceInfo:
    """harvesters 的设备信息对象（只有几个属性）。"""

    def __init__(self, serial_number="SN001", model="MV-CA050-10GC",
                 vendor="Hikrobot", tl_type="GEV",
                 display_name="Hikrobot MV-CA050-10GC", access_status=None):
        self.serial_number = serial_number
        self.model = model
        self.vendor = vendor
        self.tl_type = tl_type
        self.display_name = display_name
        self.access_status = access_status


class FakeComponent:
    """GenICam 的 buffer payload component。"""

    def __init__(self, data, width, height, data_format):
        self.data = data
        self.width = width
        self.height = height
        self.data_format = data_format


class FakeBuffer:
    """可作上下文管理器的缓冲区。"""

    def __init__(self, component):
        self.payload = types.SimpleNamespace(components=[component])
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False


class FakeDevice:
    """harvesters 的设备实例。"""

    def __init__(self, node_map=None, frame=None, fetch_error=None):
        self.remote_device = types.SimpleNamespace(node_map=node_map)
        self._frame = frame
        self._fetch_error = fetch_error
        self.started = 0
        self.stopped = 0
        self.destroyed = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def destroy(self):
        self.destroyed += 1

    def fetch(self, timeout=None):
        self.last_timeout = timeout
        if self._fetch_error is not None:
            raise self._fetch_error
        return self._frame


class FakeHarvester:
    """harvesters.core.Harvester 的假实现。

    构造时不加载任何 producer —— 这正是被测代码的调用姿势，
    同时保证测试不依赖本机 MVS 安装。
    """

    def __init__(self, device_infos=(), device_factory=None,
                 add_file_error=None, update_error=None):
        self._infos = list(device_infos)
        self._factory = device_factory
        self._add_file_error = add_file_error
        self._update_error = update_error
        self.added_files = []
        self.updated = 0
        self.reset_count = 0
        self.created_keys = []

    @property
    def device_info_list(self):
        return list(self._infos)

    def add_file(self, path, check_existence=False):
        if self._add_file_error is not None:
            raise self._add_file_error
        self.added_files.append(path)

    def update(self):
        self.updated += 1
        if self._update_error is not None:
            raise self._update_error

    def create(self, search_key):
        self.created_keys.append(search_key)
        if self._factory is None:
            return None
        return self._factory(search_key)

    def reset(self):
        self.reset_count += 1


@pytest.fixture
def fake_harvester_factory(monkeypatch):
    """把 harvesters.core.Harvester 换成可控的假类。

    本机确实装了 harvesters，若不隔离就会去扫真实的 GenTL producer，
    测试将随「这台机器装没装 MVS」而变 —— 那是不确定性的来源。

    用法::

        holder = fake_harvester_factory(infos=[FakeDeviceInfo()])
        cam.GenICamCamera(cfg).open()
        assert holder.last.added_files == [...]
    """

    class Holder:
        def __init__(self):
            self.last = None

        def __call__(self, *args, **kwargs):
            self.last = args[0] if args else FakeHarvester()
            return self.last

    def _install(harvester=None):
        holder = Holder()
        if harvester is None:
            module = types.ModuleType("harvesters.core")
            module.Harvester = holder
        else:
            module = types.ModuleType("harvesters.core")
            module.Harvester = lambda *a, **kw: harvester
            holder.last = harvester
        monkeypatch.setitem(sys.modules, "harvesters", types.ModuleType("harvesters"))
        monkeypatch.setitem(sys.modules, "harvesters.core", module)
        return holder

    return _install


@pytest.fixture
def fake_capture_factory(monkeypatch):
    """把 cv2.VideoCapture 换成返回 FakeVideoCapture 的工厂。

    返回值是「已被创建的 capture 列表」，测试里取 ``created[-1]`` 断言即可。
    """
    created = []

    def _install(**kwargs):
        def _factory(_index):
            cap = FakeVideoCapture(**kwargs)
            cap.index = _index
            created.append(cap)
            return cap
        monkeypatch.setattr(cv2, "VideoCapture", _factory)
        return created

    return _install


@pytest.fixture
def no_sleep(monkeypatch):
    """干掉 OpenCVCamera 预热阶段的 time.sleep。"""
    monkeypatch.setattr(cam, "time", types.SimpleNamespace(sleep=lambda _s: None))


@pytest.fixture
def isolated_producers(monkeypatch):
    """让 find_gentl_producers 完全不看真实磁盘。

    - 清掉所有 GenTL 相关环境变量
    - _cti_search_globs 置空（它内部会读注册表 + 遍历 Program Files）
    """
    for var in cam._GENTL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cam, "_cti_search_globs", lambda: [])
    return monkeypatch


@pytest.fixture
def isolated_mvs_roots(monkeypatch):
    """让注册表探测返回空 —— 不读本机真实安装信息。"""
    monkeypatch.setattr(cam, "_registry_mvs_roots", lambda: [])
    return monkeypatch


@pytest.fixture
def arch_neutral_tmp(tmp_path):
    """一个路径里不含位数标识的临时目录。

    位数过滤是按**整个路径**做子串匹配的，若临时目录名恰好含 "x64" / "Win32"
    之类，过滤结果就不再由被测的目录名决定。这里明确守卫一下，
    免得换台机器就变成看不懂的失败。
    """
    if cam._path_arch(str(tmp_path)) is not None:
        pytest.skip(f"临时目录路径含位数标识，无法用于位数过滤测试: {tmp_path}")
    return tmp_path


# ============================================================================
# 抽象基类契约
# ============================================================================

class TestCameraBaseContract:
    """CameraBase 的抽象契约与默认实现。"""

    def test_cannot_instantiate_abstract_base(self):
        """抽象基类不可直接实例化 —— 后端漏实现方法必须在这里就炸。"""
        with pytest.raises(TypeError, match="abstract"):
            CameraBase({})

    def test_missing_one_method_still_abstract(self):
        """只差一个抽象方法也必须是抽象的。"""
        class Partial(CameraBase):
            def open(self):
                return True

            def acquire(self):
                return None

            def release(self):
                return None

            def set_exposure(self, exposure_us):
                return None

            # set_gain 故意漏掉

        with pytest.raises(TypeError, match="abstract"):
            Partial({})

    def test_full_subclass_defaults(self):
        """实现全部抽象方法后可实例化，默认状态为「未打开、无错误」。"""
        class Complete(CameraBase):
            def open(self):
                self._is_open = True
                return True

            def acquire(self):
                return None

            def release(self):
                self._is_open = False

            def set_exposure(self, exposure_us):
                pass

            def set_gain(self, gain):
                pass

        instance = Complete({})
        assert instance.is_open is False
        assert instance.last_error == ""
        assert instance.config == {}

    def test_get_info_and_actual_params_default_to_empty(self):
        """默认实现返回空字典，而不是抛 NotImplementedError。"""
        class Complete(CameraBase):
            def open(self):
                return True

            def acquire(self):
                return None

            def release(self):
                pass

            def set_exposure(self, exposure_us):
                pass

            def set_gain(self, gain):
                pass

        instance = Complete({})
        assert instance.get_info() == {}
        assert instance.get_actual_params() == {}

    def test_context_manager_opens_and_releases(self):
        """with 语句应调用 open() 并在退出时 release()。"""
        events = []

        class Complete(CameraBase):
            def open(self):
                events.append("open")
                self._is_open = True
                return True

            def acquire(self):
                return None

            def release(self):
                events.append("release")
                self._is_open = False

            def set_exposure(self, exposure_us):
                pass

            def set_gain(self, gain):
                pass

        with Complete({}) as camera:
            assert camera.is_open is True
        assert events == ["open", "release"]

    def test_last_error_is_per_instance(self):
        """last_error 是实例属性，不因类属性共享而串台。"""
        first = OpenCVCamera({})
        second = OpenCVCamera({})
        first.last_error = "boom"
        assert second.last_error == ""


# ============================================================================
# 工厂与驱动路由
# ============================================================================

class TestCreateCameraRouting:
    """create_camera 的驱动别名路由。"""

    @pytest.mark.parametrize("driver", sorted(MVS_SDK_DRIVER_ALIASES))
    def test_mvs_sdk_aliases(self, driver, config_factory):
        """海康官方 SDK 的全部别名都应落到 MvsCamera。"""
        from hardware.mvs_camera import MvsCamera

        camera = create_camera(config_factory(camera={"driver": driver}))
        assert isinstance(camera, MvsCamera)

    @pytest.mark.parametrize("driver", sorted(GENICAM_DRIVER_ALIASES))
    def test_genicam_aliases(self, driver, config_factory):
        """GenTL 全部别名都应落到 GenICamCamera。"""
        camera = create_camera(config_factory(camera={"driver": driver}))
        assert isinstance(camera, GenICamCamera)

    @pytest.mark.parametrize("driver", ["opencv", "OpenCV", "OPENCV", "oPeNcV"])
    def test_opencv_alias_is_case_insensitive(self, driver, config_factory):
        """opencv 别名大小写不敏感。"""
        assert isinstance(
            create_camera(config_factory(camera={"driver": driver})), OpenCVCamera
        )

    @pytest.mark.parametrize("driver, expected_name", [
        ("MVS", "MvsCamera"),
        ("MvSDK", "MvsCamera"),
        ("HIK_SDK", "MvsCamera"),
        ("Harvesters", "GenICamCamera"),
        ("GenICam", "GenICamCamera"),
        ("HIKVision", "GenICamCamera"),
        ("HikRobot", "GenICamCamera"),
    ])
    def test_other_aliases_are_case_insensitive(self, driver, expected_name,
                                                config_factory):
        """其余别名同样大小写不敏感。"""
        camera = create_camera(config_factory(camera={"driver": driver}))
        assert type(camera).__name__ == expected_name

    def test_missing_camera_section_defaults_to_mvs(self, config_factory):
        """完全没有 camera 节时默认走 mvs（官方 SDK）。"""
        from hardware.mvs_camera import MvsCamera

        config = config_factory()
        config.pop("camera")
        assert isinstance(create_camera(config), MvsCamera)

    def test_empty_camera_section_defaults_to_mvs(self):
        """camera 节存在但没有 driver 键时同样默认 mvs。"""
        from hardware.mvs_camera import MvsCamera

        assert isinstance(create_camera({"camera": {}}), MvsCamera)

    def test_empty_config_defaults_to_mvs(self):
        """连配置都没有时默认 mvs。"""
        from hardware.mvs_camera import MvsCamera

        assert isinstance(create_camera({}), MvsCamera)

    def test_real_default_config_routes_to_mvs(self, default_config):
        """真实 config/default.yaml 的 driver 当前是 mvs。

        读取真文件而不是在测试里另写一份 —— 配置文件被改动时会第一时间暴露。
        """
        from hardware.mvs_camera import MvsCamera

        assert default_config["camera"]["driver"] == "mvs"
        assert isinstance(create_camera(default_config), MvsCamera)

    @pytest.mark.parametrize("driver", [
        "mvs_sdk_v2",      # 拼错的官方 SDK 别名
        "mvs-camera",      # 用了横线
        "opencv2",         # 多写了版本号
        "",                # 空字符串
        "garbage",         # 完全无关
        None,              # YAML 里 `driver:` 留空 → None
        " mvs ",           # 前后带空格的 YAML 字符串
    ])
    def test_unrecognized_driver_lands_on_opencv(self, driver, config_factory):
        """记录实际行为：无法识别的 driver 静默回退到 OpenCVCamera。

        这是**现状记录**，不是期望。见下面那条 xfail。
        """
        assert isinstance(
            create_camera(config_factory(camera={"driver": driver})), OpenCVCamera
        )

    @pytest.mark.xfail(reason=(
        "配置写错（driver 拼错 / 留空 / 带空格）时静默回退到 OpenCVCamera，"
        "而 driver 缺失时的默认值是 mvs —— 同一件事两种默认，且没有任何提示。"
        "现场表现是「打开的是笔记本摄像头」或「打开失败」，排查时极易被带偏。"
        "期望：未知 driver 抛 ValueError，或明确回退到与缺键一致的 mvs。"
        "hardware/camera.py:1166-1176"))
    @pytest.mark.parametrize("driver", ["garbage", "", None, " mvs "])
    def test_unrecognized_driver_should_not_silently_fall_back(self, driver,
                                                               config_factory):
        """非法 driver 应报错，而不是悄悄换成另一个后端。"""
        with pytest.raises(ValueError):
            create_camera(config_factory(camera={"driver": driver}))

    def test_construction_does_not_load_any_sdk(self, monkeypatch, config_factory):
        """构造任何后端都不得触发 SDK / DLL 加载。

        这是「没装 MVS 的机器也能启动」这条保证的守门用例：
        MvsCamera 只在 open() 里才去加载 MvCameraControl.dll，
        GenICamCamera 只在 open() 里才 import harvesters。
        """
        import hardware.mvs_camera as mvs

        def _boom():
            raise AssertionError("构造阶段不应加载海康 SDK")

        monkeypatch.setattr(mvs, "_ensure_sdk", _boom)

        real_import = builtins.__import__
        forbidden = ("harvesters", "MvImport", "MvCameraControl_class")

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in forbidden:
                raise AssertionError(f"构造阶段不应导入 {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        # 先确认这道岗哨本身是有效的，否则下面的断言可能只是在空跑
        with pytest.raises(AssertionError):
            __import__("harvesters.core")

        for driver in ("mvs", "mvsdk", "hik_sdk", "opencv",
                       "genicam", "harvesters", "hikrobot"):
            camera = create_camera(config_factory(camera={"driver": driver}))
            assert isinstance(camera, CameraBase)
            assert camera.is_open is False
            assert camera.last_error == ""

    def test_created_camera_is_not_open(self, config_factory):
        """工厂返回的是未打开实例 —— open() 由调用方显式触发。"""
        camera = create_camera(config_factory(camera={"driver": "genicam"}))
        assert camera.is_open is False
        assert camera.acquire() is None


# ============================================================================
# OpenCV 后端 —— 配置来源
# ============================================================================

class TestOpenCVCameraConfig:
    """OpenCVCamera 读取配置的方式。"""

    def test_device_index_from_camera_section(self, config_factory):
        """设备编号优先取 camera.device.index。"""
        camera = OpenCVCamera(config_factory(
            camera={"device": {"index": 3}}, system={"camera_id": 9}
        ))
        assert camera.camera_id == 3

    def test_falls_back_to_legacy_system_camera_id(self, config_factory):
        """camera.device 缺失时回退到旧的 system.camera_id。

        注意必须把整个 camera 节换掉 —— 真实配置里已经有
        camera.device.index，留着它就走不到回退分支了。
        """
        camera = OpenCVCamera(config_factory(camera={}, system={"camera_id": 7}))
        assert camera.camera_id == 7

    def test_falls_back_when_device_has_no_index(self, config_factory):
        """camera.device 存在但没有 index 键时同样回退。"""
        camera = OpenCVCamera(config_factory(
            camera={"device": {"serial_number": "SN1"}}, system={"camera_id": 5}
        ))
        assert camera.camera_id == 5

    def test_device_none_is_tolerated(self, config_factory):
        """camera.device 为 None（YAML 留空）时不应抛异常。

        此时 index 键不存在 → 回退到 system.camera_id。
        """
        camera = OpenCVCamera(config_factory(
            camera={"device": None}, system={"camera_id": 4}
        ))
        assert camera.camera_id == 4

    def test_index_none_becomes_zero(self, config_factory):
        """index 显式为 None 时取 0，而不是抛 TypeError。"""
        camera = OpenCVCamera(config_factory(
            camera={"device": {"index": None}}, system={"camera_id": 3}
        ))
        assert camera.camera_id == 0

    def test_no_config_at_all_defaults_to_zero(self):
        """完全没有相机相关配置时设备编号为 0。"""
        assert OpenCVCamera({}).camera_id == 0

    def test_imaging_defaults(self):
        """图像参数的兜底默认值。"""
        camera = OpenCVCamera({})
        assert camera.width == 2448
        assert camera.height == 2048
        assert camera.exposure_us == 5000
        assert camera.gain == 1.0

    def test_imaging_params_from_real_config(self, default_config):
        """真实配置中的图像参数被正确读出。"""
        camera = OpenCVCamera(default_config)
        assert camera.camera_id == default_config["camera"]["device"]["index"]
        assert camera.width == default_config["camera"]["width"]
        assert camera.height == default_config["camera"]["height"]
        assert camera.exposure_us == default_config["camera"]["exposure_us"]
        assert camera.gain == default_config["camera"]["gain"]

    def test_string_index_is_coerced(self, config_factory):
        """YAML 里带引号的编号应被转成 int。"""
        camera = OpenCVCamera(config_factory(camera={"device": {"index": "2"}}))
        assert camera.camera_id == 2
        assert isinstance(camera.camera_id, int)


# ============================================================================
# OpenCV 后端 —— 生命周期
# ============================================================================

class TestOpenCVCameraLifecycle:
    """OpenCVCamera 的 open / acquire / release。"""

    def test_open_success_sets_state_and_params(self, config_factory, fake_capture_factory,
                                                no_sleep):
        """打开成功后：状态置位、last_error 清空、分辨率与曝光下发到设备。"""
        created = fake_capture_factory(opened=True, width=2448.0, height=2048.0)
        camera = OpenCVCamera(config_factory())

        assert camera.open() is True
        assert camera.is_open is True
        assert camera.last_error == ""

        capture = created[-1]
        assert capture.index == camera.camera_id
        props = dict(capture.set_calls)
        assert props[cv2.CAP_PROP_FRAME_WIDTH] == camera.width
        assert props[cv2.CAP_PROP_FRAME_HEIGHT] == camera.height
        assert props[cv2.CAP_PROP_GAIN] == camera.gain

    def test_open_converts_exposure_to_log2_seconds(self, config_factory,
                                                    fake_capture_factory, no_sleep):
        """Windows 下 OpenCV 的曝光是 -log2(秒)，必须做单位换算。

        直接把微秒数写进去是这里的经典错误：5000us 会被当成一个
        天文数字的秒数，相机要么拒收要么取到极短曝光。
        """
        created = fake_capture_factory()
        camera = OpenCVCamera(config_factory(camera={"exposure_us": 5000}))

        camera.open()

        props = dict(created[-1].set_calls)
        expected = -np.log2(5000 / 1_000_000.0)
        assert props[cv2.CAP_PROP_EXPOSURE] == pytest.approx(expected)

    def test_open_warms_up_by_discarding_frames(self, config_factory,
                                                fake_capture_factory, no_sleep):
        """预热阶段应丢弃前几帧，避免拿到未稳定的第一帧。"""
        created = fake_capture_factory()
        OpenCVCamera(config_factory()).open()
        assert created[-1].read_count == 5

    def test_open_failure_sets_last_error(self, config_factory, fake_capture_factory,
                                          no_sleep):
        """打不开时应返回 False、is_open 为 False，并在 last_error 里给出编号。"""
        created = fake_capture_factory(opened=False)
        camera = OpenCVCamera(config_factory(camera={"device": {"index": 6}}))

        assert camera.open() is False
        assert camera.is_open is False
        assert camera.last_error != ""
        assert "6" in camera.last_error
        # 失败时不应继续配置参数
        assert created[-1].set_calls == []

    def test_open_raises_when_cv2_missing(self, config_factory, monkeypatch):
        """cv2 未安装时应明确抛 ImportError。

        sys.modules 里把 cv2 置为 None，等价于「这个模块不存在」。
        """
        monkeypatch.setitem(sys.modules, "cv2", None)
        with pytest.raises(ImportError, match="OpenCV"):
            OpenCVCamera(config_factory()).open()

    def test_acquire_without_open_returns_none(self):
        """未打开时 acquire() 返回 None，而不是抛异常。"""
        assert OpenCVCamera({}).acquire() is None

    def test_acquire_after_failed_open_returns_none(self, fake_capture_factory,
                                                    no_sleep, config_factory):
        """打开失败后取帧同样是 None —— 调用方只靠返回值判空。"""
        fake_capture_factory(opened=False)
        camera = OpenCVCamera(config_factory())
        camera.open()
        assert camera.acquire() is None

    def test_acquire_returns_frame_on_success(self, fake_capture_factory, no_sleep,
                                              config_factory):
        """取帧成功时原样返回 BGR 数组（契约见模块 docstring）。"""
        fake_capture_factory()
        camera = OpenCVCamera(config_factory())
        camera.open()

        frame = camera.acquire()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (4, 4, 3)

    def test_acquire_returns_none_when_read_fails(self, config_factory, no_sleep):
        """read() 返回 False 时应返回 None（掉线 / 丢帧）。"""
        camera = OpenCVCamera(config_factory())

        class FailingCapture(FakeVideoCapture):
            def read(self):
                return False, None

        camera._capture = FailingCapture()
        camera._is_open = True
        assert camera.acquire() is None

    def test_release_is_idempotent(self, config_factory, fake_capture_factory, no_sleep):
        """release() 可重复调用：第二次不应再碰已释放的 capture。"""
        created = fake_capture_factory()
        camera = OpenCVCamera(config_factory())
        camera.open()
        capture = created[-1]

        camera.release()
        assert camera.is_open is False
        assert camera._capture is None
        assert capture.release_count == 1

        camera.release()          # 不应抛异常
        assert capture.release_count == 1

    def test_release_without_open_is_safe(self):
        """从未打开过就 release() 不应抛异常。"""
        camera = OpenCVCamera({})
        camera.release()
        assert camera.is_open is False

    def test_acquire_after_release_returns_none(self, config_factory,
                                                fake_capture_factory, no_sleep):
        """释放后不应再取帧。"""
        fake_capture_factory()
        camera = OpenCVCamera(config_factory())
        camera.open()
        camera.release()
        assert camera.acquire() is None

    def test_set_exposure_updates_attribute_and_device(self, config_factory,
                                                       fake_capture_factory, no_sleep):
        """打开状态下设置曝光应同时更新属性与设备。"""
        created = fake_capture_factory()
        camera = OpenCVCamera(config_factory())
        camera.open()

        camera.set_exposure(1000)
        assert camera.exposure_us == 1000
        props = dict(created[-1].set_calls)
        assert props[cv2.CAP_PROP_EXPOSURE] == pytest.approx(
            -np.log2(1000 / 1_000_000.0)
        )

    def test_set_exposure_before_open_only_stores(self, config_factory):
        """未打开时设置曝光只记属性，不应崩在 _capture 为 None 上。"""
        camera = OpenCVCamera(config_factory())
        camera.set_exposure(2500)
        assert camera.exposure_us == 2500
        assert camera._capture is None

    def test_set_gain_updates_attribute_and_device(self, config_factory,
                                                   fake_capture_factory, no_sleep):
        """打开状态下设置增益应下发到设备。"""
        created = fake_capture_factory()
        camera = OpenCVCamera(config_factory())
        camera.open()

        camera.set_gain(6.0)
        assert camera.gain == 6.0
        assert dict(created[-1].set_calls)[cv2.CAP_PROP_GAIN] == 6.0

    def test_set_gain_before_open_only_stores(self, config_factory):
        """未打开时设置增益同样只记属性。"""
        camera = OpenCVCamera(config_factory())
        camera.set_gain(3.0)
        assert camera.gain == 3.0

    def test_get_actual_params_empty_before_open(self):
        """未打开时读回参数为空字典。"""
        assert OpenCVCamera({}).get_actual_params() == {}

    def test_get_actual_params_reports_device_resolution(self, config_factory,
                                                         fake_capture_factory, no_sleep):
        """打开后读回的是设备上报的分辨率，而非配置里的请求值。"""
        fake_capture_factory(width=1280.0, height=720.0)
        camera = OpenCVCamera(config_factory())
        camera.open()

        actual = camera.get_actual_params()
        assert actual == {"width": 1280, "height": 720}
        assert isinstance(actual["width"], int)

    def test_get_info_defaults_to_empty(self):
        """OpenCV 后端不提供型号信息，get_info() 返回空字典。"""
        assert OpenCVCamera({}).get_info() == {}


# ============================================================================
# 设备发现 —— 位数判定
# ============================================================================

class TestPathArch:
    """_path_arch：从路径推断 producer 位数。"""

    @pytest.mark.parametrize("path", [
        "C:/Program Files (x86)/Common Files/MVS/Runtime/Win64_x64/MvProducerGEV.cti",
        "C:/x/Win64/MvProducerGEV.cti",
        "/opt/pylon/x86_64/bin/tl.cti",
        "/usr/lib/amd64/lib.cti",
        "/usr/local/x64/tl.cti",
    ])
    def test_64_bit_paths(self, path):
        """含 64 位标识的路径判为 64。"""
        assert cam._path_arch(path) == 64

    @pytest.mark.parametrize("path", [
        "C:/x/Win32_i86/MvProducerGEV.cti",
        "C:/x/Win32_x86/MvProducerGEV.cti",
        "C:/x/Win32/tl.cti",
        "/usr/lib/i86/lib.cti",
        "/usr/lib/x86/lib.cti",
        "/usr/lib/ia32/lib.cti",
    ])
    def test_32_bit_paths(self, path):
        """含 32 位标识的路径判为 32。"""
        assert cam._path_arch(path) == 32

    def test_x86_64_is_not_misread_as_x86(self):
        """关键回归：``x86_64`` 里含子串 ``x86``，必须判成 64 位。

        判定顺序若反过来（先查 32 位 token），Linux 下 64 位生产者的
        路径会命中 ``x86`` 被当作 32 位剔除，表现为「producer 明明装了却
        一个都找不到」，且没有任何提示。
        """
        assert "x86" in "/opt/dalsa/x86_64/bin/tl.cti"      # 前提：确实含子串
        assert cam._path_arch("/opt/dalsa/x86_64/bin/tl.cti") == 64

    def test_win64_wins_over_win32(self):
        """同时含 32/64 标识时以 64 位为准（顺序即优先级）。"""
        assert cam._path_arch("/mnt/Win32/Win64_x64/tl.cti") == 64

    @pytest.mark.parametrize("path", [
        "C:/Program Files/Common Files/MVS/Runtime/tl.cti",
        "/opt/mvs/lib/tl.cti",
        "",
    ])
    def test_unknown_arch_returns_none(self, path):
        """判不出位数时返回 None —— 表示「不过滤」，不是「32 位」。"""
        assert cam._path_arch(path) is None

    def test_64_token_list_is_checked_first(self):
        """顺序契约：64 位 token 表在 32 位之前被检查。

        这条防的是「有人把两个常量元组的顺序调换」这种改动。
        """
        assert cam._ARCH_TOKENS_64.index("x86_64") >= 0
        assert "x86" in cam._ARCH_TOKENS_32
        assert "x86_64" not in cam._ARCH_TOKENS_32


class TestPythonBitness:
    """python_bitness 与解释器一致。"""

    def test_matches_struct_calcsize(self):
        """返回值必须与 struct.calcsize 一致 —— 位数判断错会加载错版本的 .cti。"""
        assert cam.python_bitness() == struct.calcsize("P") * 8

    def test_is_32_or_64(self):
        """只可能是 32 或 64。"""
        assert cam.python_bitness() in (32, 64)


# ============================================================================
# 设备发现 —— 采集卡与访问状态
# ============================================================================

class TestFramegrabberProducer:
    """_is_framegrabber_producer：采集卡 producer 的识别。"""

    @pytest.mark.parametrize("name", [
        "MvFGProducerGEV.cti",
        "MvFGProducerU3V.cti",
        "mvfgproducercl.cti",
        "MVFGProducerXoF.cti",
    ])
    def test_framegrabber_producers(self, name):
        """MVFG*Producer*.cti 是采集卡（CameraLink / CoaXPress / XoF）。"""
        assert cam._is_framegrabber_producer(f"C:/MVS/Runtime/Win64_x64/{name}") is True

    @pytest.mark.parametrize("name", [
        "MvProducerGEV.cti",
        "MvProducerU3V.cti",
        "MvCameraControl.cti",
        "ProducerGEV.cti",
        "MvProducerGEV_v2.cti",
    ])
    def test_normal_camera_producers(self, name):
        """MvProducer*.cti 是普通相机 producer，不能被误杀。

        GigE 面阵相机全靠它 —— 误判成采集卡会让设备列表恒为空。
        """
        assert cam._is_framegrabber_producer(f"C:/MVS/Runtime/Win64_x64/{name}") is False

    def test_judgement_is_on_basename_only(self):
        """只看文件名，不看目录 —— 目录名里带 MVFG 不算采集卡。"""
        assert cam._is_framegrabber_producer("C:/MVS/MVFG/Win64_x64/MvProducerGEV.cti") is False


class TestAccessStatus:
    """_describe_access_status：设备访问状态的判定。"""

    @pytest.mark.parametrize("code, expected_ok", [
        (0, True),    # 未知：不给结论，宁可不报警
        (1, True),    # 可读写（正常）
        (2, True),    # 只读：仍可打开
        (3, False),   # 无访问权限
        (4, False),   # 被占用
    ])
    def test_known_codes(self, code, expected_ok):
        """只有明确「无权限 / 被占用」才判为打不开。"""
        label, ok = cam._describe_access_status(code)
        assert ok is expected_ok
        assert label == _ACCESS_STATUS_LABELS[code]

    def test_intenum_readwrite_is_not_reported_as_busy(self):
        """关键回归：合法可读写状态不得被误报为「被占用」。

        harvesters 的 access_status 是 IntEnum，其 str() 在 Python 3.11+
        退化为纯数字（"1"）。老实现靠子串匹配 "Available"，两种形态都对不上，
        于是每台设备都被贴上「可能被其他程序占用」的标签。
        """
        class DeviceAccessStatus(IntEnum):
            Unknown = 0
            ReadWrite = 1
            ReadOnly = 2
            NoAccess = 3
            Busy = 4

        label, ok = cam._describe_access_status(DeviceAccessStatus.ReadWrite)
        assert ok is True, f"可读写设备被判成了不可用：{label!r}"
        assert label == "可读写（正常）"

    def test_intenum_noaccess_is_reported(self):
        """真被占用 / 无权限时仍要报出来，不能为了不误报而漏报。"""
        class DeviceAccessStatus(IntEnum):
            NoAccess = 3
            Busy = 4

        assert cam._describe_access_status(DeviceAccessStatus.Busy)[1] is False
        assert cam._describe_access_status(DeviceAccessStatus.NoAccess)[1] is False

    def test_numeric_string_is_understood(self):
        """字符串形态的数字也应能解析（某些版本会给 str）。"""
        assert cam._describe_access_status("1") == ("可读写（正常）", True)
        assert cam._describe_access_status("4") == ("被占用", False)

    @pytest.mark.parametrize("raw", [None, "Available", object(), []])
    def test_unparsable_input_never_raises_alarm(self, raw):
        """判不出来时不报警，也不能抛异常。

        恒亮的报警灯和恒亮的警告灯一样没有信息量。
        """
        label, ok = cam._describe_access_status(raw)
        assert ok is True
        assert isinstance(label, str) and label != ""

    def test_unknown_code_is_labelled_with_number(self):
        """未知状态码也要带上号码，便于现场对照文档。"""
        label, ok = cam._describe_access_status(99)
        assert "99" in label
        assert ok is True

    def test_bad_status_set_matches_documented_intent(self):
        """_ACCESS_STATUS_BAD 只含「无权限」与「被占用」两个码。"""
        assert _ACCESS_STATUS_BAD == frozenset({3, 4})
        assert set(_ACCESS_STATUS_LABELS) >= _ACCESS_STATUS_BAD


class TestDescribeProducers:
    """describe_producers：供界面展示的厂商 / 传输层分类。"""

    def test_hikvision_producer(self):
        """海康的 producer 判为海康，GEV 后缀判为 GigE Vision。"""
        info = cam.describe_producers(
            ["C:/Program Files/Common Files/MVS/Runtime/Win64_x64/MvProducerGEV.cti"]
        )[0]
        assert "海康" in info["vendor"]
        assert info["transport"] == "GigE Vision"
        assert info["file"] == "MvProducerGEV.cti"

    def test_u3v_transport(self):
        """U3V 后缀判为 USB3 Vision。"""
        info = cam.describe_producers(["/opt/MVS/MvProducerU3V.cti"])[0]
        assert info["transport"] == "USB3 Vision"

    def test_daheng_vendor_from_path(self):
        """大恒按**路径**识别厂商。"""
        info = cam.describe_producers(["/opt/DAHENG/Runtime/x64/ProducerGEV.cti"])[0]
        assert "大恒" in info["vendor"]

    def test_basler_vendor_from_filename(self):
        """文件名里带 pylon 时能认出 Basler。

        与下面被 xfail 的那条合起来看：识别口径是文件名，而内置搜索
        glob 命中的 pylon producer 文件名是 ProducerGEV.cti（不含 pylon），
        两者对不上 —— 这才是缺陷所在。
        """
        info = cam.describe_producers(["/opt/vendor/pylonProducerGEV.cti"])[0]
        assert "Basler" in info["vendor"]

    def test_unknown_transport(self):
        """既不是 GEV 也不是 U3V 的 producer 标为未知传输层。"""
        info = cam.describe_producers(["/opt/DAHENG/Runtime/x64/SomeProducer.cti"])[0]
        assert info["transport"] == "未知"

    def test_empty_list(self):
        """空输入返回空列表。"""
        assert cam.describe_producers([]) == []

    @pytest.mark.xfail(reason=(
        "厂商判定口径不一致：大恒按整个路径匹配，Basler 只按文件名匹配 "
        "`\"PYLON\" in os.path.basename(path)`。而内置搜索 glob 给出的 pylon "
        "producer 文件名是 ProducerGEV.cti（不含 pylon 字样），"
        "于是 Basler 相机在界面上被标成「未知厂商」。"
        "hardware/camera.py:404-409"))
    def test_basler_vendor_from_path(self):
        """Basler 的 producer 应被认出来（按路径即可）。"""
        info = cam.describe_producers(
            ["C:/Program Files/Basler/pylon 7/Runtime/x64/ProducerGEV.cti"]
        )[0]
        assert "Basler" in info["vendor"]


# ============================================================================
# 设备发现 —— 环境变量与注册表
# ============================================================================

class TestExpandCtiPaths:
    """_expand_cti_paths：环境变量值 → .cti 列表。"""

    def test_directory_yields_its_cti_files(self, tmp_path):
        """值是目录时，展开为该目录下的 .cti。"""
        (tmp_path / "A.cti").write_bytes(b"")
        (tmp_path / "B.cti").write_bytes(b"")
        (tmp_path / "readme.txt").write_text("x", encoding="utf-8")

        found = cam._expand_cti_paths(str(tmp_path))
        assert sorted(Path(p).name for p in found) == ["A.cti", "B.cti"]

    def test_direct_cti_path(self, tmp_path):
        """值直接指向某个 .cti 时取它本身。"""
        cti = tmp_path / "MvProducerGEV.cti"
        cti.write_bytes(b"")
        assert cam._expand_cti_paths(str(cti)) == [str(cti)]

    def test_missing_direct_cti_is_ignored(self, tmp_path):
        """指向不存在的 .cti 时返回空，而不是把死路径带下去。"""
        assert cam._expand_cti_paths(str(tmp_path / "nope.cti")) == []

    def test_nonexistent_directory(self, tmp_path):
        """不存在的目录返回空列表，不抛异常。"""
        assert cam._expand_cti_paths(str(tmp_path / "no_such_dir")) == []

    def test_empty_and_blank_values(self):
        """空字符串 / 纯空白 / 纯分号都应安全返回空列表。"""
        for raw in ("", "   ", ";", ";;", '""'):
            assert cam._expand_cti_paths(raw) == []

    def test_semicolon_separated_list(self, tmp_path):
        """分号分隔的目录列表（GenICam 标准形态）应逐项展开。"""
        first = tmp_path / "one"
        second = tmp_path / "two"
        for d, name in ((first, "A.cti"), (second, "B.cti")):
            d.mkdir()
            (d / name).write_bytes(b"")

        found = cam._expand_cti_paths(f"{first};{second}")
        assert sorted(Path(p).name for p in found) == ["A.cti", "B.cti"]

    def test_bare_path_without_quotes_is_unaffected(self, tmp_path):
        """不带引号、不带多余空白的路径正常展开。"""
        (tmp_path / "A.cti").write_bytes(b"")
        assert cam._expand_cti_paths(str(tmp_path)) == [str(tmp_path / "A.cti")]

    @pytest.mark.xfail(reason=(
        "环境变量的值带引号或前后有空白时，盘符冒号的保护失效："
        "判定 `raw[1] == \":\"` 只看第二个字符，而 `\"C:\\...` / `  C:\\...` "
        "的第二个字符是引号或空格，于是整个 Windows 路径被按冒号切碎，"
        "切出来的碎片都不是目录 → 静默返回空列表。"
        "而 GENICAM_GENTL64_PATH 里恰恰是最需要引号的形式"
        "（MVS 默认装在 'C:\\Program Files (x86)\\Common Files\\...'）。"
        "hardware/camera.py:109-113"))
    @pytest.mark.parametrize("template", ['"{}"', "  {}  ", '" {} "'])
    def test_quoted_or_padded_windows_path_survives(self, template, tmp_path):
        """带引号 / 前后有空白的 Windows 路径不应被冒号切碎。"""
        (tmp_path / "A.cti").write_bytes(b"")
        assert cam._expand_cti_paths(template.format(tmp_path)) == [
            str(tmp_path / "A.cti")
        ]

    def test_windows_drive_colon_is_not_a_separator(self, tmp_path):
        """盘符里的冒号不能被当成分隔符，否则 C:\\... 会被切碎。"""
        assert cam._expand_cti_paths("C:/definitely/not/here") == []

    @pytest.mark.parametrize("raw", [
        "/opt/dalsa/one:/opt/dalsa/two",   # POSIX 风格的冒号分隔列表
        "/opt/MVS/Runtime",                 # 单个 POSIX 路径
        "C:/definitely/not/here",           # 单个 Windows 路径
    ])
    def test_nonexistent_paths_return_empty_without_raising(self, raw):
        """路径不存在时一律返回空列表 —— 环境变量配错不应让程序崩在启动阶段。"""
        assert cam._expand_cti_paths(raw) == []

    def test_result_paths_are_normalized(self, tmp_path):
        """展开结果按 os.path.normpath 归一。"""
        (tmp_path / "A.cti").write_bytes(b"")
        raw = str(tmp_path / "sub" / "..")
        assert cam._expand_cti_paths(raw) == [str(tmp_path / "A.cti")]


class _FakeRegKey:
    """注册表键的假对象：底层就是一个 名字 → 值 的字典。"""

    def __init__(self, entries):
        self.entries = entries


class FakeWinreg:
    """winreg 模块的最小假实现。

    只支持被测代码用到的那几个函数：OpenKey / QueryInfoKey / EnumKey /
    QueryValueEx。HKLM 有内容，HKCU 一律 OSError（模拟读不到）。
    """

    HKEY_LOCAL_MACHINE = "HKLM"
    HKEY_CURRENT_USER = "HKCU"
    KEY_READ = 0x20019
    KEY_WOW64_64KEY = 0x0100
    KEY_WOW64_32KEY = 0x0200

    def __init__(self, hklm_entries):
        self._hklm = hklm_entries

    def OpenKey(self, hive, path, reserved=0, access=0):
        """OpenKey 会被调用两次：一次开 Uninstall 根，一次开下面的子键。"""
        if hive == self.HKEY_LOCAL_MACHINE:
            return _FakeRegKey(self._hklm)
        if isinstance(hive, _FakeRegKey):
            values = hive.entries.get(path)
            if values is None:
                raise OSError(f"no such subkey {path}")
            return _FakeRegKey(values)
        raise OSError("no such hive")

    def QueryInfoKey(self, key):
        return (len(key.entries), 0, 0)

    def EnumKey(self, key, index):
        return list(key.entries)[index]

    def QueryValueEx(self, key, name):
        if name not in key.entries:
            raise OSError(f"no value {name}")
        return (key.entries[name], 1)


class TestRegistryMvsRoots:
    """_registry_mvs_roots：从注册表找 MVS 安装根。"""

    def test_non_windows_returns_empty(self, monkeypatch):
        """非 Windows 平台（winreg 不存在）应返回空列表而不是崩。"""
        monkeypatch.setitem(sys.modules, "winreg", None)
        assert cam._registry_mvs_roots() == []

    def test_roots_from_uninstall_entries(self, monkeypatch, tmp_path):
        """从 UninstallString 反推出安装根，并过滤非 MVS 条目。

        海康允许改安装路径，只按默认路径硬编码会整个漏掉这类安装 ——
        注册表是唯一可靠线索。
        """
        root = tmp_path / "APP" / "MVS"
        root.mkdir(parents=True)
        monkeypatch.setitem(sys.modules, "winreg", FakeWinreg({
            "MVS Runtime": {
                "DisplayName": "MVS Runtime",
                "UninstallString": f'"{root / "uninstall.exe"}"',
            },
            "MVS Development": {
                "DisplayName": "hikrobot MVS Development Components",
                "UninstallString": f'"{root / "uninstall.exe"}"',
            },
            "Google Chrome": {
                "DisplayName": "Google Chrome",
                "UninstallString": f'"{root / "chrome.exe"}"',
            },
        }))

        roots = cam._registry_mvs_roots()
        assert roots == [str(root)]

    def test_nonexistent_install_root_is_dropped(self, monkeypatch, tmp_path):
        """卸载残留（注册表有条目但目录已删）不应进入结果。"""
        monkeypatch.setitem(sys.modules, "winreg", FakeWinreg({
            "MVS Ghost": {
                "DisplayName": "MVS",
                "UninstallString": f'"{tmp_path / "gone" / "uninstall.exe"}"',
            },
        }))
        assert cam._registry_mvs_roots() == []

    def test_missing_values_are_skipped(self, monkeypatch, tmp_path):
        """条目缺 DisplayName / UninstallString 时跳过，不影响其它条目。"""
        root = tmp_path / "MVS"
        root.mkdir()
        monkeypatch.setitem(sys.modules, "winreg", FakeWinreg({
            "Broken Entry": {},                       # 两个值都没有
            "Only DisplayName": {"DisplayName": "MVS"},  # 缺 UninstallString
            "Good": {
                "DisplayName": "MVS",
                "UninstallString": f'"{root / "uninstall.exe"}"',
            },
        }))
        assert cam._registry_mvs_roots() == [str(root)]

    def test_uninstall_string_without_quotes(self, monkeypatch, tmp_path):
        """UninstallString 不带引号时同样能取出目录。"""
        root = tmp_path / "MVS"
        root.mkdir()
        monkeypatch.setitem(sys.modules, "winreg", FakeWinreg({
            "MVS": {
                "DisplayName": "MVS",
                "UninstallString": str(root / "uninstall.exe"),
            },
        }))
        assert cam._registry_mvs_roots() == [str(root)]


# ============================================================================
# 设备发现 —— 目录搜索
# ============================================================================

class TestFindGentlProducers:
    """find_gentl_producers：.cti 搜索与过滤。"""

    def test_empty_environment_returns_empty(self, isolated_producers):
        """无环境变量、无内置路径时返回空列表，不抛异常。"""
        assert cam.find_gentl_producers() == []

    def test_empty_environment_verbose_does_not_crash(self, isolated_producers, capsys):
        """verbose 模式下空结果也应正常返回（诊断脚本会开这个开关）。"""
        assert cam.find_gentl_producers(verbose=True) == []
        assert capsys.readouterr().out == ""

    def test_finds_via_environment_variable(self, isolated_producers, arch_neutral_tmp):
        """环境变量指向的目录里的 .cti 应被收进来。"""
        cti = arch_neutral_tmp / "MvProducerGEV.cti"
        cti.write_bytes(b"")
        isolated_producers.setenv("GENICAM_GENTL64_PATH", str(arch_neutral_tmp))

        assert cam.find_gentl_producers() == [str(cti)]

    @staticmethod
    def _two_bitness_dirs(base: Path):
        """造出 64 位与 32 位两个平台目录，各放一份 .cti，并返回环境变量值。

        环境变量按平台分号列表给出 —— MVS 的实际布局就是每个位数一个目录。
        """
        dirs = []
        for sub in ("Win64_x64", "Win32_i86"):
            d = base / sub
            d.mkdir()
            (d / "MvProducerGEV.cti").write_bytes(b"")
            dirs.append(d)
        return cam.os.pathsep.join(str(d) for d in dirs)

    def test_wrong_bitness_producer_is_rejected(self, isolated_producers,
                                                arch_neutral_tmp):
        """位数不匹配的 producer 必须剔除。

        64 位 Python 加载 Win32 的 .cti 会失败，而报错信息与「相机没插好」
        毫无关系，容易把人引到错误方向。
        """
        value = self._two_bitness_dirs(arch_neutral_tmp)
        isolated_producers.setenv("GENICAM_GENTL64_PATH", value)
        assert cam.python_bitness() == 64        # 本套用例的前提

        found = cam.find_gentl_producers()
        assert [Path(p).parent.name for p in found] == ["Win64_x64"]

    def test_bitness_filter_follows_interpreter(self, isolated_producers,
                                                arch_neutral_tmp, monkeypatch):
        """换到 32 位解释器时保留的应是 Win32 那一份。"""
        value = self._two_bitness_dirs(arch_neutral_tmp)
        isolated_producers.setenv("GENICAM_GENTL64_PATH", value)
        monkeypatch.setattr(cam, "python_bitness", lambda: 32)

        found = cam.find_gentl_producers()
        assert [Path(p).parent.name for p in found] == ["Win32_i86"]

    def test_arch_neutral_paths_are_kept(self, isolated_producers, arch_neutral_tmp):
        """路径里没有位数标识时不参与过滤（判不出 ≠ 判成 32 位）。"""
        (arch_neutral_tmp / "MvProducerGEV.cti").write_bytes(b"")
        isolated_producers.setenv("GENICAM_GENTL64_PATH", str(arch_neutral_tmp))
        assert len(cam.find_gentl_producers()) == 1

    def test_framegrabber_producers_are_skipped(self, isolated_producers,
                                                arch_neutral_tmp):
        """默认不加载采集卡 producer（GigE 面阵相机用不上）。"""
        assert cam.INCLUDE_FRAMEGRABBER_PRODUCERS is False
        for name in ("MvProducerGEV.cti", "MvFGProducerGEV.cti"):
            (arch_neutral_tmp / name).write_bytes(b"")
        isolated_producers.setenv("GENICAM_GENTL64_PATH", str(arch_neutral_tmp))

        found = cam.find_gentl_producers()
        assert [Path(p).name for p in found] == ["MvProducerGEV.cti"]

    def test_framegrabber_switch_can_include_them(self, isolated_producers,
                                                  arch_neutral_tmp, monkeypatch):
        """开关打开时应把采集卡 producer 一起收进来。"""
        for name in ("MvProducerGEV.cti", "MvFGProducerGEV.cti"):
            (arch_neutral_tmp / name).write_bytes(b"")
        isolated_producers.setenv("GENICAM_GENTL64_PATH", str(arch_neutral_tmp))
        monkeypatch.setattr(cam, "INCLUDE_FRAMEGRABBER_PRODUCERS", True)

        found = cam.find_gentl_producers()
        assert sorted(Path(p).name for p in found) == [
            "MvFGProducerGEV.cti", "MvProducerGEV.cti"
        ]

    def test_results_are_deduplicated(self, isolated_producers, arch_neutral_tmp):
        """同一个 .cti 被环境变量与搜索模式同时命中时只出现一次。"""
        cti = arch_neutral_tmp / "MvProducerGEV.cti"
        cti.write_bytes(b"")
        isolated_producers.setenv("GENICAM_GENTL64_PATH", str(arch_neutral_tmp))
        isolated_producers.setattr(
            cam, "_cti_search_globs",
            lambda: [str(arch_neutral_tmp / "*.cti")],
        )

        assert cam.find_gentl_producers() == [str(cti)]

    def test_results_are_sorted_by_basename(self, isolated_producers, arch_neutral_tmp):
        """结果按文件名排序，保证多次运行顺序一致（设备索引才有意义）。"""
        for name in ("ZetaProducerGEV.cti", "alphaProducerGEV.cti"):
            (arch_neutral_tmp / name).write_bytes(b"")
        isolated_producers.setenv("GENICAM_GENTL64_PATH", str(arch_neutral_tmp))

        names = [Path(p).name for p in cam.find_gentl_producers()]
        assert names == sorted(names, key=str.lower)

    def test_search_globs_include_registry_install_root(self, monkeypatch, tmp_path):
        """搜索模式必须包含注册表安装根 —— 否则改过安装路径的机器会漏搜。"""
        monkeypatch.setattr(cam, "_registry_mvs_roots", lambda: [str(tmp_path / "MVS")])
        patterns = cam._cti_search_globs()

        assert list(cam._VENDOR_CTI_GLOBS) == patterns[:len(cam._VENDOR_CTI_GLOBS)]
        assert str(tmp_path / "MVS" / "**" / "*.cti") in patterns

    def test_search_globs_work_end_to_end(self, monkeypatch, arch_neutral_tmp):
        """注册表根下的 .cti 应能被真实搜索到（连带覆盖模式匹配分支）。

        这里不能用 isolated_producers 夹具 —— 它按设计把 _cti_search_globs
        整个置空了，本用例要验证的恰恰是它到磁盘这一段。
        """
        for var in cam._GENTL_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(cam, "_VENDOR_CTI_GLOBS", ())
        cti = arch_neutral_tmp / "Deep" / "Nested" / "MvProducerGEV.cti"
        cti.parent.mkdir(parents=True)
        cti.write_bytes(b"")
        monkeypatch.setattr(cam, "_registry_mvs_roots", lambda: [str(arch_neutral_tmp)])

        assert cam.find_gentl_producers(verbose=True) == [str(cti)]

    def test_verbose_reports_glob_hits(self, isolated_producers, monkeypatch,
                                      arch_neutral_tmp, capsys):
        """verbose 应把命中数量打出来，便于确认搜索路径确实生效。"""
        cti = arch_neutral_tmp / "MvProducerGEV.cti"
        cti.write_bytes(b"")
        monkeypatch.setattr(cam, "_registry_mvs_roots", lambda: [])
        isolated_producers.setattr(cam, "_VENDOR_CTI_GLOBS", ())
        isolated_producers.setattr(
            cam, "_cti_search_globs", lambda: [str(arch_neutral_tmp / "*.cti")]
        )

        cam.find_gentl_producers(verbose=True)
        assert "路径模式" in capsys.readouterr().out

    def test_verbose_reports_framegrabber_rejections(self, isolated_producers,
                                                     arch_neutral_tmp, capsys):
        """verbose 应说明哪些采集卡 producer 被跳过了。"""
        (arch_neutral_tmp / "MvFGProducerGEV.cti").write_bytes(b"")
        isolated_producers.setenv("GENICAM_GENTL64_PATH", str(arch_neutral_tmp))

        assert cam.find_gentl_producers(verbose=True) == []
        out = capsys.readouterr().out
        assert "采集卡" in out
        assert "MvFGProducerGEV.cti" in out

    def test_verbose_reports_arch_rejections(self, isolated_producers,
                                             arch_neutral_tmp, capsys):
        """verbose 模式应把被剔除的 producer 打出来（诊断靠它）。

        现场最常见的一类误判就是「装了 MVS 却一个 producer 都找不到」，
        而原因只是位数不匹配 —— 不说出来就无从排查。
        """
        old = arch_neutral_tmp / "Win32_i86"
        old.mkdir()
        (old / "OldProducerGEV.cti").write_bytes(b"")
        isolated_producers.setenv("GENICAM_GENTL64_PATH", str(old))

        cam.find_gentl_producers(verbose=True)
        out = capsys.readouterr().out
        assert "位数不匹配" in out
        assert "OldProducerGEV.cti" in out


class TestFindMvsRuntimeDirs:
    """find_mvs_runtime_dirs：含 MvCameraControl.dll 的目录。"""

    @staticmethod
    def _file_shim(monkeypatch, existing):
        """把 camera.py 里的 os 换成「只在 existing 集合里认文件」的替身。

        被测函数末尾有几条写死的 Program Files 兜底路径 —— 本机真的装了
        MVS 时它们会命中，测试结果就会随机器而变。替身让这几条必然落空。
        集合里的路径统一用 normcase 归一，与函数内部的比较口径一致。
        """
        keys = {cam.os.path.normcase(str(p)) for p in existing}

        class _PathShim:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def isfile(self, path):
                return cam.os.path.normcase(str(path)) in keys

        class _OsShim:
            def __init__(self, real):
                self._real = real
                self.path = _PathShim(real.path)

            def __getattr__(self, name):
                return getattr(self._real, name)

        import os as real_os
        monkeypatch.setattr(cam, "os", _OsShim(real_os))
        return keys

    def test_no_candidate_exists_returns_empty(self, isolated_mvs_roots, monkeypatch):
        """一处都找不到时返回空列表，不抛异常。"""
        monkeypatch.setattr(cam, "find_gentl_producers", lambda *a, **kw: [])
        self._file_shim(monkeypatch, [])
        assert cam.find_mvs_runtime_dirs() == []

    def test_dll_directory_from_registry_root(self, isolated_mvs_roots, monkeypatch,
                                              tmp_path):
        """注册表安装根下的 Runtime/<平台> 里若真有 DLL，应被找到。"""
        target = tmp_path / "MVS" / "Runtime" / "Win64_x64"
        target.mkdir(parents=True)
        monkeypatch.setattr(cam, "_registry_mvs_roots", lambda: [str(tmp_path / "MVS")])
        monkeypatch.setattr(cam, "find_gentl_producers", lambda *a, **kw: [])
        self._file_shim(monkeypatch, [target / "MvCameraControl.dll"])

        assert cam.find_mvs_runtime_dirs() == [str(target)]

    def test_development_bin_layout(self, isolated_mvs_roots, monkeypatch, tmp_path):
        """开发组件的 Development/Bin/win64 布局也应被覆盖。"""
        target = tmp_path / "MVS" / "Development" / "Bin" / "win64"
        target.mkdir(parents=True)
        monkeypatch.setattr(cam, "_registry_mvs_roots", lambda: [str(tmp_path / "MVS")])
        monkeypatch.setattr(cam, "find_gentl_producers", lambda *a, **kw: [])
        self._file_shim(monkeypatch, [target / "MvCameraControl.dll"])

        assert cam.find_mvs_runtime_dirs() == [str(target)]

    def test_directory_without_dll_is_rejected(self, isolated_mvs_roots, monkeypatch,
                                              tmp_path):
        """目录存在但没有 DLL 时不算数 —— 不能靠猜路径。"""
        target = tmp_path / "MVS" / "Runtime" / "Win64_x64"
        target.mkdir(parents=True)
        monkeypatch.setattr(cam, "_registry_mvs_roots", lambda: [str(tmp_path / "MVS")])
        monkeypatch.setattr(cam, "find_gentl_producers", lambda *a, **kw: [])
        self._file_shim(monkeypatch, [])

        assert cam.find_mvs_runtime_dirs() == []

    def test_32_bit_interpreter_uses_win32_dirs(self, isolated_mvs_roots, monkeypatch,
                                                tmp_path):
        """32 位解释器找的是 Win32_i86 / win32 布局。"""
        target = tmp_path / "MVS" / "Runtime" / "Win32_i86"
        target.mkdir(parents=True)
        monkeypatch.setattr(cam, "python_bitness", lambda: 32)
        monkeypatch.setattr(cam, "_registry_mvs_roots", lambda: [str(tmp_path / "MVS")])
        monkeypatch.setattr(cam, "find_gentl_producers", lambda *a, **kw: [])
        self._file_shim(monkeypatch, [target / "MvCameraControl.dll"])

        assert cam.find_mvs_runtime_dirs() == [str(target)]

    def test_cti_directory_is_a_candidate(self, isolated_mvs_roots, monkeypatch,
                                         tmp_path):
        """.cti 与 .dll 同处一个 Runtime 目录，故 producer 目录也是候选。"""
        target = tmp_path / "Runtime" / "Win64_x64"
        target.mkdir(parents=True)
        monkeypatch.setattr(cam, "_registry_mvs_roots", lambda: [])
        monkeypatch.setattr(
            cam, "find_gentl_producers",
            lambda *a, **kw: [str(target / "MvProducerGEV.cti")],
        )
        self._file_shim(monkeypatch, [target / "MvCameraControl.dll"])

        assert cam.find_mvs_runtime_dirs() == [str(target)]

    def test_results_are_deduplicated(self, isolated_mvs_roots, monkeypatch, tmp_path):
        """多处线索指向同一目录时只出现一次。"""
        target = tmp_path / "MVS" / "Runtime" / "Win64_x64"
        target.mkdir(parents=True)
        monkeypatch.setattr(cam, "_registry_mvs_roots", lambda: [str(tmp_path / "MVS")])
        monkeypatch.setattr(
            cam, "find_gentl_producers",
            lambda *a, **kw: [str(target / "MvProducerGEV.cti")],
        )
        self._file_shim(monkeypatch, [target / "MvCameraControl.dll"])

        assert cam.find_mvs_runtime_dirs() == [str(target)]


class TestFindMvimportDirs:
    """find_mvimport_dirs：海康官方 Python 封装的目录。"""

    @pytest.fixture
    def isolated(self, monkeypatch):
        """不读真实磁盘 / 真实 sys.path。"""
        monkeypatch.setattr(cam, "_registry_mvs_roots", lambda: [])
        monkeypatch.setattr(cam, "_VENDOR_MVIMPORT_GLOBS", ())
        return monkeypatch

    @staticmethod
    def _make_mvimport(root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "MvCameraControl_class.py").write_text("", encoding="utf-8")
        return root

    def test_empty_environment_returns_empty(self, isolated):
        """sys.path 上没有、注册表读不到、内置路径为空 → 返回空列表。"""
        isolated.setattr(cam, "sys", types.SimpleNamespace(path=[]))
        assert cam.find_mvimport_dirs() == []

    def test_found_on_sys_path(self, isolated, tmp_path):
        """已在 sys.path 上的 MvImport 目录优先被认出。"""
        mvimport = self._make_mvimport(tmp_path / "MvImport")
        isolated.setattr(cam, "sys", types.SimpleNamespace(path=[str(mvimport)]))
        assert cam.find_mvimport_dirs() == [str(mvimport)]

    def test_directory_without_marker_file_is_rejected(self, isolated, tmp_path):
        """目录里没有 MvCameraControl_class.py 就不算 MvImport。"""
        empty = tmp_path / "MvImport"
        empty.mkdir()
        isolated.setattr(cam, "sys", types.SimpleNamespace(path=[str(empty)]))
        assert cam.find_mvimport_dirs() == []

    def test_results_are_deduplicated(self, isolated, tmp_path):
        """sys.path 上重复出现时只报一次。"""
        mvimport = self._make_mvimport(tmp_path / "MvImport")
        isolated.setattr(
            cam, "sys", types.SimpleNamespace(path=[str(mvimport), str(mvimport)])
        )
        assert cam.find_mvimport_dirs() == [str(mvimport)]

    def test_found_under_registry_root(self, isolated, tmp_path):
        """注册表安装根下的 Development/Samples/Python/MvImport 应被找到。

        本机就把 Development 装在别的盘上，硬编码默认路径会整个漏掉。
        """
        mvimport = self._make_mvimport(
            tmp_path / "MVS" / "Development" / "Samples" / "Python" / "MvImport"
        )
        isolated.setattr(cam, "sys", types.SimpleNamespace(path=[]))
        isolated.setattr(cam, "_registry_mvs_roots", lambda: [str(tmp_path / "MVS")])
        assert cam.find_mvimport_dirs() == [str(mvimport)]

    def test_blank_sys_path_entries_are_ignored(self, isolated, tmp_path):
        """sys.path 里的空串（某些环境会插入）不应导致崩溃。"""
        mvimport = self._make_mvimport(tmp_path / "MvImport")
        isolated.setattr(cam, "sys", types.SimpleNamespace(path=["", str(mvimport)]))
        assert cam.find_mvimport_dirs() == [str(mvimport)]

    def test_found_via_builtin_glob_pattern(self, isolated, monkeypatch, tmp_path):
        """内置 glob 模式也必须真的被展开（注册表读不到时的兜底路径）。"""
        mvimport = self._make_mvimport(tmp_path / "Vendor" / "MvImport")
        isolated.setattr(cam, "sys", types.SimpleNamespace(path=[]))
        isolated.setattr(cam, "_VENDOR_MVIMPORT_GLOBS",
                         (str(tmp_path / "**" / "MvImport"),))

        assert cam.find_mvimport_dirs() == [str(mvimport)]

    def test_builtin_glob_with_no_match_returns_empty(self, isolated, tmp_path):
        """兜底 glob 指向不存在的路径时只返回空列表，不抛异常。"""
        isolated.setattr(cam, "sys", types.SimpleNamespace(path=[]))
        isolated.setattr(cam, "_VENDOR_MVIMPORT_GLOBS",
                         (str(tmp_path / "no_such" / "*"),))
        assert cam.find_mvimport_dirs() == []


# ============================================================================
# 设备枚举
# ============================================================================

class TestListGenicamDevices:
    """list_genicam_devices：一次性设备枚举。"""

    def test_missing_harvesters_returns_empty(self, monkeypatch):
        """harvesters 未安装时返回空列表（而不是抛异常）。"""
        monkeypatch.setitem(sys.modules, "harvesters.core", None)
        assert cam.list_genicam_devices() == []

    def test_missing_harvesters_verbose(self, monkeypatch, capsys):
        """verbose 模式下应能看出是 harvesters 没装。"""
        monkeypatch.setitem(sys.modules, "harvesters.core", None)
        assert cam.list_genicam_devices(verbose=True) == []
        assert "harvesters" in capsys.readouterr().out

    def test_no_producers_returns_empty(self, fake_harvester_factory, monkeypatch):
        """找不到 .cti 时返回空列表，且 harvester 仍被回收。"""
        holder = fake_harvester_factory(FakeHarvester())
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: [])

        assert cam.list_genicam_devices() == []
        assert holder.last.reset_count == 1         # finally 里必须 reset

    def test_device_fields_are_reported(self, fake_harvester_factory, monkeypatch):
        """枚举到的设备字段与访问状态应完整给出。"""
        infos = [
            FakeDeviceInfo(serial_number="SN-A", model="MV-CA050-10GC",
                           access_status=1),
            FakeDeviceInfo(serial_number="SN-B", model="MV-CU060-10GM",
                           access_status=4),
        ]
        fake_harvester_factory(FakeHarvester(device_infos=infos))
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        devices = cam.list_genicam_devices()

        assert [d["index"] for d in devices] == ["0", "1"]
        assert [d["serial_number"] for d in devices] == ["SN-A", "SN-B"]
        assert [d["model"] for d in devices] == ["MV-CA050-10GC", "MV-CU060-10GM"]
        assert all(d["tl_type"] == "GEV" for d in devices)
        assert devices[0]["access_ok"] is True
        assert devices[1]["access_ok"] is False
        assert devices[1]["access_status"] == "被占用"

    def test_missing_attributes_do_not_crash(self, fake_harvester_factory, monkeypatch):
        """设备信息对象缺属性时用空串兜底。"""
        fake_harvester_factory(FakeHarvester(device_infos=[types.SimpleNamespace()]))
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        devices = cam.list_genicam_devices()
        assert devices[0]["serial_number"] == ""
        assert devices[0]["access_ok"] is True      # access_status 取不到 → 不报警

    def test_update_failure_is_swallowed(self, fake_harvester_factory, monkeypatch):
        """producer 加载 / 扫描抛异常时返回空列表，不向上冒泡。"""
        holder = fake_harvester_factory(
            FakeHarvester(update_error=RuntimeError("tl 崩了"))
        )
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        assert cam.list_genicam_devices() == []
        assert holder.last.reset_count == 1

    def test_update_failure_verbose_reports_reason(self, fake_harvester_factory,
                                                   monkeypatch, capsys):
        """verbose 下要能看出枚举失败的原因（诊断脚本靠它）。

        界面上「没扫到设备」是正常情况，但排查时必须能看到原因。
        """
        fake_harvester_factory(FakeHarvester(update_error=RuntimeError("tl 崩了")))
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        assert cam.list_genicam_devices(verbose=True) == []
        out = capsys.readouterr().out
        assert "tl 崩了" in out

    def test_add_file_failure_is_swallowed(self, fake_harvester_factory, monkeypatch):
        """.cti 加载失败（位数不符 / 文件损坏）同样只返回空列表。"""
        fake_harvester_factory(
            FakeHarvester(add_file_error=RuntimeError("wrong bitness"))
        )
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])
        assert cam.list_genicam_devices() == []

    def test_reset_failure_does_not_mask_result(self, fake_harvester_factory,
                                                monkeypatch):
        """harvester.reset() 抛异常不应影响已枚举到的结果。"""
        class BadReset(FakeHarvester):
            def reset(self):
                raise RuntimeError("reset 失败")

        fake_harvester_factory(BadReset(device_infos=[FakeDeviceInfo(access_status=1)]))
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        assert len(cam.list_genicam_devices()) == 1


# ============================================================================
# GenICamCamera —— 配置
# ============================================================================

class TestGenICamConfig:
    """GenICamCamera 读取配置的方式（兼容旧键）。"""

    def test_defaults(self):
        """空配置下的兜底默认值。"""
        camera = GenICamCamera({})
        assert camera.width == 2448
        assert camera.height == 2048
        assert camera.exposure_us == 5000
        assert camera.gain == 1.0
        assert camera.pixel_format == "Mono8"
        assert camera.trigger_mode == "continuous"
        assert camera.trigger_source == "Line0"
        assert camera.device_serial == ""
        assert camera.device_index == 0
        assert camera.acquire_timeout_s == pytest.approx(2.0)

    def test_values_from_real_config(self, default_config):
        """真实配置的每个相机字段都被读到。"""
        camera = GenICamCamera(default_config)
        expected = default_config["camera"]
        assert camera.width == expected["width"]
        assert camera.height == expected["height"]
        assert camera.exposure_us == expected["exposure_us"]
        assert camera.gain == expected["gain"]
        assert camera.pixel_format == expected["pixel_format"]
        assert camera.trigger_mode == expected["trigger"]["mode"]
        assert camera.trigger_source == expected["trigger"]["source"]
        assert camera.device_index == expected["device"]["index"]
        assert camera.acquire_timeout_s == pytest.approx(
            expected["acquire_timeout_ms"] / 1000.0
        )

    def test_legacy_trigger_keys_are_honoured(self, config_factory):
        """旧配置把触发模式散在 camera.trigger_mode 下，仍要能读。"""
        camera = GenICamCamera(config_factory(camera={
            "trigger_mode": "software", "trigger_source": "Line2",
        }))
        assert camera.trigger_mode == "software"
        assert camera.trigger_source == "Line2"

    def test_new_trigger_section_wins_over_legacy(self, config_factory):
        """新旧键同时存在时以 camera.trigger 为准。"""
        camera = GenICamCamera(config_factory(camera={
            "trigger": {"mode": "external", "source": "Line1"},
            "trigger_mode": "software", "trigger_source": "Line9",
        }))
        assert camera.trigger_mode == "external"
        assert camera.trigger_source == "Line1"

    def test_trigger_section_none_is_tolerated(self, config_factory):
        """camera.trigger 显式为 None（YAML 留空）时不应崩。"""
        camera = GenICamCamera(config_factory(camera={"trigger": None}))
        assert camera.trigger_mode == "continuous"
        assert camera.trigger_source == "Line0"

    def test_timeout_is_converted_to_seconds(self, config_factory):
        """acquire_timeout_ms 换算成秒 —— harvesters 的 fetch 收的是秒。"""
        camera = GenICamCamera(config_factory(camera={"acquire_timeout_ms": 1500}))
        assert camera.acquire_timeout_s == pytest.approx(1.5)

    def test_serial_number_none_becomes_empty_string(self, config_factory):
        """序列号留空（None）应归一成空串，而不是字符串 "None"。"""
        camera = GenICamCamera(config_factory(
            camera={"device": {"serial_number": None, "index": 2}}
        ))
        assert camera.device_serial == ""
        assert camera.device_index == 2

    def test_device_section_none_is_tolerated(self, config_factory):
        """camera.device 为 None 时取默认值。"""
        camera = GenICamCamera(config_factory(camera={"device": None}))
        assert camera.device_serial == ""
        assert camera.device_index == 0

    def test_not_open_initially(self):
        """构造后即为未打开、无错误、无设备。"""
        camera = GenICamCamera({})
        assert camera.is_open is False
        assert camera.last_error == ""
        assert camera.get_info() == {}
        assert camera._device is None
        assert camera._harvester is None


# ============================================================================
# GenICamCamera —— 设备选择
# ============================================================================

class TestGenICamDeviceSelection:
    """_build_search_key：多相机场景下的设备绑定。"""

    def test_serial_number_wins_over_index(self, config_factory):
        """配置了序列号时按序列号定位，索引只作兜底。"""
        camera = GenICamCamera(config_factory(
            camera={"device": {"serial_number": "SN-B", "index": 0}}
        ))
        infos = [FakeDeviceInfo(serial_number="SN-A"),
                 FakeDeviceInfo(serial_number="SN-B")]
        assert camera._build_search_key(infos) == 1

    def test_missing_serial_raises_with_available_list(self, config_factory):
        """序列号找不到时必须报错，且列出当前枚举到的序列号。

        悄悄退回第一台会拍错工位却毫无提示 —— 这是本项目明确要避免的。
        """
        camera = GenICamCamera(config_factory(
            camera={"device": {"serial_number": "SN-X"}}
        ))
        infos = [FakeDeviceInfo(serial_number="SN-A"),
                 FakeDeviceInfo(serial_number="SN-B")]

        with pytest.raises(RuntimeError) as excinfo:
            camera._build_search_key(infos)
        message = str(excinfo.value)
        assert "SN-X" in message
        assert "SN-A" in message and "SN-B" in message
        assert "serial_number" in message

    def test_index_used_when_no_serial(self, config_factory):
        """未配置序列号时按索引取。"""
        camera = GenICamCamera(config_factory(camera={"device": {"index": 1}}))
        assert camera._build_search_key([FakeDeviceInfo(), FakeDeviceInfo()]) == 1

    def test_index_out_of_range_raises(self, config_factory):
        """索引越界应明确报错并给出实际数量。"""
        camera = GenICamCamera(config_factory(camera={"device": {"index": 5}}))
        with pytest.raises(RuntimeError) as excinfo:
            camera._build_search_key([FakeDeviceInfo()])
        assert "5" in str(excinfo.value)
        assert "1" in str(excinfo.value)

    def test_empty_device_list_raises(self, config_factory):
        """设备列表为空时同样越界报错，而不是返回 0 让下游崩。"""
        camera = GenICamCamera(config_factory(camera={"device": {"index": 0}}))
        with pytest.raises(RuntimeError):
            camera._build_search_key([])

    def test_serial_attribute_missing_is_skipped(self, config_factory):
        """设备信息对象没有 serial_number 属性时不应崩在查找上。"""
        camera = GenICamCamera(config_factory(
            camera={"device": {"serial_number": "SN-A"}}
        ))
        infos = [types.SimpleNamespace(), FakeDeviceInfo(serial_number="SN-A")]
        assert camera._build_search_key(infos) == 1

    def test_read_device_info_fields(self):
        """_read_device_info 应把界面要展示的字段都取出来。"""
        info = FakeDeviceInfo(serial_number="SN-9", model="MV-CA050-10GC",
                              vendor="Hikrobot", tl_type="GEV",
                              display_name="cam0")
        result = GenICamCamera._read_device_info([info], 0)
        assert result["serial_number"] == "SN-9"
        assert result["model"] == "MV-CA050-10GC"
        assert result["vendor"] == "Hikrobot"
        assert result["tl_type"] == "GEV"
        assert result["display_name"] == "cam0"

    def test_read_device_info_out_of_range_returns_empty(self):
        """索引越界（或 search_key 不是 int）时返回空字典，不抛异常。"""
        assert GenICamCamera._read_device_info([], 0) == {}
        assert GenICamCamera._read_device_info([], "not-an-int") == {}


# ============================================================================
# GenICamCamera —— 打开与释放
# ============================================================================

class TestGenICamOpen:
    """GenICamCamera.open() 的成功与各类失败路径。"""

    @pytest.fixture
    def node_map(self):
        return FakeNodeMap()

    def test_missing_harvesters_raises(self, monkeypatch, config_factory):
        """harvesters 未安装时应抛 ImportError 并给出安装提示。"""
        monkeypatch.setitem(sys.modules, "harvesters.core", None)
        with pytest.raises(ImportError, match="harvesters"):
            GenICamCamera(config_factory()).open()

    def test_no_genicam_producer_sets_last_error(self, fake_harvester_factory,
                                                 monkeypatch, config_factory):
        """找不到 .cti 时返回 False，并把「要装 MVS」写进 last_error。

        这句提示是现场唯一的线索来源 —— GUI 里 print 是看不到的。
        """
        fake_harvester_factory(FakeHarvester())
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: [])

        camera = GenICamCamera(config_factory())
        assert camera.open() is False
        assert camera.is_open is False
        assert "GenTL Producer" in camera.last_error
        assert "MVS" in camera.last_error
        assert camera._harvester is None            # 失败后必须回收

    def test_no_device_found_sets_last_error(self, fake_harvester_factory,
                                             monkeypatch, config_factory):
        """producer 加载了但没扫到相机时，提示应指向供电 / 网线 / 网段。"""
        fake_harvester_factory(FakeHarvester(device_infos=[]))
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        camera = GenICamCamera(config_factory())
        assert camera.open() is False
        assert "未发现相机" in camera.last_error

    def test_create_returning_none_sets_last_error(self, fake_harvester_factory,
                                                   monkeypatch, config_factory):
        """create() 返回 None 时报「无法创建设备实例」，而不是继续往下走。"""
        fake_harvester_factory(FakeHarvester(
            device_infos=[FakeDeviceInfo()], device_factory=lambda key: None
        ))
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        camera = GenICamCamera(config_factory())
        assert camera.open() is False
        assert "无法创建设备实例" in camera.last_error

    def test_unicode_decode_error_gets_dedicated_message(self, fake_harvester_factory,
                                                         monkeypatch, config_factory):
        """MVS producer 的编码崩溃要有专门提示，并指明改用 mvs 驱动。

        把它混进通用异常里会把人引到「相机没插好」的方向上白折腾。
        """
        fake_harvester_factory(FakeHarvester(
            add_file_error=UnicodeDecodeError("utf-8", b"\xd0\xd1", 0, 1, "invalid")
        ))
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        camera = GenICamCamera(config_factory())
        assert camera.open() is False
        assert "UTF-8" in camera.last_error
        assert "mvs" in camera.last_error

    def test_generic_failure_records_exception_text(self, fake_harvester_factory,
                                                    monkeypatch, config_factory):
        """其它异常原样记进 last_error（界面直接展示给用户）。"""
        fake_harvester_factory(FakeHarvester(update_error=RuntimeError("TL 初始化失败")))
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        camera = GenICamCamera(config_factory())
        assert camera.open() is False
        assert camera.last_error == "TL 初始化失败"
        assert camera._harvester is None
        assert camera._device is None

    def test_missing_serial_failure_records_details(self, fake_harvester_factory,
                                                    monkeypatch, config_factory):
        """序列号对不上时 last_error 要带上配置值与实际枚举值。"""
        fake_harvester_factory(FakeHarvester(
            device_infos=[FakeDeviceInfo(serial_number="SN-A")]
        ))
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        camera = GenICamCamera(config_factory(
            camera={"device": {"serial_number": "SN-WRONG"}}
        ))
        assert camera.open() is False
        assert "SN-WRONG" in camera.last_error
        assert "SN-A" in camera.last_error

    def test_open_success_configures_nodes(self, fake_harvester_factory, monkeypatch,
                                           config_factory):
        """打开成功时各节点被依次下发，且设备已启动。

        节点组合与顺序就是这台相机实际生效的配置，改错任何一项都会
        让「界面显示的值」与「相机真正的值」不一致。
        """
        node_map = FakeNodeMap()
        device = FakeDevice(node_map=node_map)
        fake_harvester_factory(FakeHarvester(
            device_infos=[FakeDeviceInfo(serial_number="SN-A")],
            device_factory=lambda key: device,
        ))
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        camera = GenICamCamera(config_factory(camera={
            "width": 1280, "height": 1024, "pixel_format": "Mono8",
            "exposure_us": 8000, "gain": 2.0,
        }))
        assert camera.open() is True
        assert camera.is_open is True
        assert camera.last_error == ""
        assert device.started == 1

        written = {name: node.value for name, node in node_map.nodes.items()}
        assert written["Width"] == 1280
        assert written["Height"] == 1024
        assert written["PixelFormat"] == "Mono8"
        assert written["ExposureAuto"] == "Off"
        assert written["ExposureTime"] == pytest.approx(8000.0)
        assert written["GainAuto"] == "Off"
        assert written["Gain"] == pytest.approx(2.0)
        assert written["TriggerMode"] == "Off"
        assert written["AcquisitionMode"] == "Continuous"

    def test_exposure_auto_is_cleared_before_writing_exposure(
            self, fake_harvester_factory, monkeypatch, config_factory):
        """顺序硬约束：先关自动模式，再写曝光 / 增益。

        海康相机出厂默认 ExposureAuto=Continuous，自动模式开着时写
        ExposureTime 会被相机忽略 —— 值写得进去但不生效，最容易误判成
        「设置成功了」。
        """
        node_map = _RecordingNodeMap()
        fake_harvester_factory(FakeHarvester(
            device_infos=[FakeDeviceInfo()],
            device_factory=lambda key: FakeDevice(node_map=node_map),
        ))
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        camera = GenICamCamera(config_factory())
        assert camera.open() is True

        order = [name for name, _ in node_map.write_log]
        assert order.index("ExposureAuto") < order.index("ExposureTime")
        assert order.index("GainAuto") < order.index("Gain")

    def test_open_populates_device_info(self, fake_harvester_factory, monkeypatch,
                                       config_factory):
        """打开成功后 get_info() 应给出型号与序列号。"""
        info = FakeDeviceInfo(serial_number="SN-42", model="MV-CA050-10GC",
                              vendor="Hikrobot")
        fake_harvester_factory(FakeHarvester(
            device_infos=[info],
            device_factory=lambda key: FakeDevice(node_map=FakeNodeMap()),
        ))
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        camera = GenICamCamera(config_factory())
        camera.open()

        reported = camera.get_info()
        assert reported["serial_number"] == "SN-42"
        assert reported["model"] == "MV-CA050-10GC"
        # 返回的是副本，调用方改不坏内部状态
        reported["model"] = "tampered"
        assert camera.get_info()["model"] == "MV-CA050-10GC"

    def test_open_uses_configured_serial_number(self, fake_harvester_factory,
                                                monkeypatch, config_factory):
        """配置了序列号时应按索引定位到那一台。"""
        holder = fake_harvester_factory(FakeHarvester(
            device_infos=[FakeDeviceInfo(serial_number="SN-A"),
                          FakeDeviceInfo(serial_number="SN-B")],
            device_factory=lambda key: FakeDevice(node_map=FakeNodeMap()),
        ))
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        camera = GenICamCamera(config_factory(
            camera={"device": {"serial_number": "SN-B"}}
        ))
        assert camera.open() is True
        assert holder.last.created_keys == [1]

    def test_all_producers_are_registered(self, fake_harvester_factory, monkeypatch,
                                          config_factory):
        """找到的每个 .cti 都要注册进 harvester，缺一个就可能少枚举一台相机。"""
        holder = fake_harvester_factory(FakeHarvester(device_infos=[]))
        monkeypatch.setattr(cam, "find_gentl_producers",
                            lambda **kw: ["/a/A.cti", "/b/B.cti"])

        GenICamCamera(config_factory()).open()
        assert holder.last.added_files == ["/a/A.cti", "/b/B.cti"]

    def test_release_is_idempotent(self, fake_harvester_factory, monkeypatch,
                                   config_factory):
        """release() 可重复调用：设备 stop/destroy 各一次，不重复销毁。"""
        device = FakeDevice(node_map=FakeNodeMap())
        holder = fake_harvester_factory(FakeHarvester(
            device_infos=[FakeDeviceInfo()],
            device_factory=lambda key: device,
        ))
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        camera = GenICamCamera(config_factory())
        camera.open()

        camera.release()
        assert camera.is_open is False
        assert device.stopped == 1
        assert device.destroyed == 1
        assert holder.last.reset_count == 1

        camera.release()                            # 不应抛异常
        assert device.stopped == 1
        assert device.destroyed == 1

    def test_cleanup_survives_harvester_reset_error(self, fake_harvester_factory,
                                                    monkeypatch, config_factory):
        """harvester.reset() 抛异常时也要把引用清干净，不能让释放半途而废。"""
        class AngryHarvester(FakeHarvester):
            def reset(self):
                raise RuntimeError("reset 失败")

        fake_harvester_factory(AngryHarvester(
            device_infos=[FakeDeviceInfo()],
            device_factory=lambda key: FakeDevice(node_map=FakeNodeMap()),
        ))
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        camera = GenICamCamera(config_factory())
        camera.open()
        camera.release()

        assert camera._harvester is None
        assert camera._node_map is None
        assert camera.is_open is False

    def test_release_without_open_is_safe(self, config_factory):
        """从未打开过就 release() 不应抛异常。"""
        camera = GenICamCamera(config_factory())
        camera.release()
        assert camera.is_open is False

    def test_cleanup_survives_device_errors(self, fake_harvester_factory, monkeypatch,
                                            config_factory):
        """stop()/destroy() 抛异常时仍要把引用清干净。"""
        class AngryDevice(FakeDevice):
            def stop(self):
                raise RuntimeError("stop 失败")

            def destroy(self):
                raise RuntimeError("destroy 失败")

        fake_harvester_factory(FakeHarvester(
            device_infos=[FakeDeviceInfo()],
            device_factory=lambda key: AngryDevice(node_map=FakeNodeMap()),
        ))
        monkeypatch.setattr(cam, "find_gentl_producers", lambda **kw: ["/x/MvProducerGEV.cti"])

        camera = GenICamCamera(config_factory())
        camera.open()
        camera.release()

        assert camera._device is None
        assert camera._harvester is None
        assert camera._node_map is None


# ============================================================================
# GenICamCamera —— 取帧与像素格式
# ============================================================================

class TestGenICamAcquire:
    """GenICamCamera.acquire() 的返回契约。"""

    def test_acquire_without_open_returns_none(self):
        """未打开时返回 None。"""
        assert GenICamCamera({}).acquire() is None

    def test_fetch_exception_returns_none(self, config_factory):
        """取帧抛异常（超时最常见）应返回 None，由调用方决定是否计丢帧。"""
        camera = GenICamCamera(config_factory())
        camera._is_open = True
        camera._device = FakeDevice(fetch_error=TimeoutError("timeout"))

        assert camera.acquire() is None

    def test_fetch_none_returns_none(self, config_factory):
        """fetch 返回 None 时返回 None。"""
        camera = GenICamCamera(config_factory())
        camera._is_open = True
        camera._device = FakeDevice(frame=None)

        assert camera.acquire() is None

    def test_acquire_returns_bgr_frame(self, config_factory):
        """取帧成功时返回 BGR 三通道。"""
        data = np.arange(4 * 4 * 3, dtype=np.uint8)
        component = FakeComponent(data, 4, 4, "BGR8")

        camera = GenICamCamera(config_factory())
        camera._is_open = True
        camera._device = FakeDevice(frame=FakeBuffer(component))

        frame = camera.acquire()
        assert frame.shape == (4, 4, 3)
        assert np.array_equal(frame, data.reshape(4, 4, 3))

    def test_buffer_is_released_as_context_manager(self, config_factory):
        """buffer 必须用 with 管理 —— 否则缓冲区泄漏，几帧后就取不到图。"""
        component = FakeComponent(np.zeros(4 * 4, dtype=np.uint8), 4, 4, "Mono8")
        buffer = FakeBuffer(component)

        camera = GenICamCamera(config_factory())
        camera._is_open = True
        camera._device = FakeDevice(frame=buffer)
        camera.acquire()

        assert buffer.entered is True
        assert buffer.exited is True

    def test_payload_parse_failure_returns_none(self, config_factory):
        """payload 结构异常时返回 None 而不是往上抛。"""
        buffer = FakeBuffer(None)
        buffer.payload = types.SimpleNamespace(components=[])

        camera = GenICamCamera(config_factory())
        camera._is_open = True
        camera._device = FakeDevice(frame=buffer)

        assert camera.acquire() is None

    def test_fetch_uses_configured_timeout(self, config_factory):
        """取帧超时应取自配置。"""
        device = FakeDevice(frame=None)
        camera = GenICamCamera(config_factory(camera={"acquire_timeout_ms": 750}))
        camera._is_open = True
        camera._device = device

        camera.acquire()
        assert device.last_timeout == pytest.approx(0.75)


class TestGenICamToNumpy:
    """_to_numpy：像素格式 → BGR / 灰度的归一。"""

    @staticmethod
    def _component(data, width, height, fmt):
        return FakeComponent(np.asarray(data), width, height, fmt)

    def test_mono8_stays_gray(self):
        """Mono8 → 单通道灰度，原样返回。"""
        data = np.arange(16, dtype=np.uint8)
        frame = GenICamCamera._to_numpy(self._component(data, 4, 4, "Mono8"))
        assert frame.shape == (4, 4)
        assert frame.dtype == np.uint8
        assert np.array_equal(frame, data.reshape(4, 4))

    def test_mono12_is_truncated_to_8_bit(self):
        """Mono10/12 存在 16 位容器里，右移 2 位落到 0~255。"""
        raw = (np.arange(16, dtype=np.uint16) * 4).reshape(4, 4)
        frame = GenICamCamera._to_numpy(
            self._component(raw.view(np.uint8), 4, 4, "Mono12")
        )
        assert frame.shape == (4, 4)
        assert np.array_equal(frame, (raw >> 2).astype(np.uint8))

    def test_mono10_packed_is_not_treated_as_mono8(self):
        """Mono8 的判定不能靠「名字里有 8」误伤 —— Mono10Packed 走别的分支。

        这里用长度匹配 16 位容器来确认它没被 Mono8 分支截走。
        """
        raw = (np.arange(4, dtype=np.uint16) * 8).reshape(2, 2)
        frame = GenICamCamera._to_numpy(
            self._component(raw.view(np.uint8), 2, 2, "Mono10")
        )
        assert np.array_equal(frame, (raw >> 2).astype(np.uint8))

    @pytest.mark.parametrize("bayer, code", [
        ("BayerRG8", cv2.COLOR_BayerRG2BGR),
        ("BayerGR8", cv2.COLOR_BayerGR2BGR),
        ("BayerGB8", cv2.COLOR_BayerGB2BGR),
        ("BayerBG8", cv2.COLOR_BayerBG2BGR),
    ])
    def test_bayer_is_demosaiced(self, bayer, code):
        """Bayer 格式必须去马赛克成 BGR 三通道。"""
        raw = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        frame = GenICamCamera._to_numpy(self._component(raw, 2, 2, bayer))

        assert frame.shape == (2, 2, 3)
        assert np.array_equal(frame, cv2.cvtColor(raw, code))

    def test_rgb8_is_converted_to_bgr(self):
        """关键契约：相机给 RGB，本类必须返回 BGR。

        这里若不转换，下游会拿到红蓝互换的图 —— 灰度图看不出来，
        只有彩色场景才暴露。
        """
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[..., 0] = 10      # R
        rgb[..., 1] = 20      # G
        rgb[..., 2] = 30      # B

        frame = GenICamCamera._to_numpy(self._component(rgb.reshape(-1), 2, 2, "RGB8"))

        assert frame[..., 0].tolist() == [[30, 30], [30, 30]]   # B 在前
        assert frame[..., 1].tolist() == [[20, 20], [20, 20]]
        assert frame[..., 2].tolist() == [[10, 10], [10, 10]]   # R 在后
        assert np.array_equal(frame, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def test_bgr8_is_passed_through(self):
        """BGR8 已是本类的契约格式，原样返回。"""
        bgr = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        frame = GenICamCamera._to_numpy(self._component(bgr.reshape(-1), 2, 2, "BGR8"))
        assert np.array_equal(frame, bgr)

    def test_unknown_gray_format_falls_back_by_length(self, capsys):
        """未识别格式按数据长度猜通道数，并打印以便补充适配。"""
        data = np.arange(4, dtype=np.uint8)
        frame = GenICamCamera._to_numpy(self._component(data, 2, 2, "Mystery8"))

        assert frame.shape == (2, 2)
        assert "未识别的像素格式" in capsys.readouterr().out

    def test_unknown_color_format_falls_back_by_length(self, capsys):
        """长度为 h*w*3 的未知格式按 BGR 处理。"""
        data = np.arange(12, dtype=np.uint8)
        frame = GenICamCamera._to_numpy(self._component(data, 2, 2, "Mystery24"))
        assert frame.shape == (2, 2, 3)
        assert "未识别的像素格式" in capsys.readouterr().out

    def test_empty_format_string_uses_length_guess(self):
        """data_format 缺失（空串）时走长度猜测，不抛异常。"""
        data = np.arange(4, dtype=np.uint8)
        frame = GenICamCamera._to_numpy(self._component(data, 2, 2, ""))
        assert frame.shape == (2, 2)

    def test_unparsable_length_raises(self):
        """数据长度与分辨率对不上时必须报错，而不是硬 reshape 出垃圾。"""
        data = np.arange(5, dtype=np.uint8)
        with pytest.raises(ValueError, match="无法解析像素格式"):
            GenICamCamera._to_numpy(self._component(data, 2, 2, "Mystery"))


# ============================================================================
# GenICamCamera —— 参数读写
# ============================================================================

class TestGenICamParams:
    """节点读写与参数回读。"""

    def test_set_node_without_node_map(self, config_factory):
        """未打开（无节点树）时 _set_node 返回 False，不抛异常。"""
        camera = GenICamCamera(config_factory())
        assert camera._set_node("Width", 100) is False
        assert camera._get_node("Width") is None

    def test_set_node_missing_node_is_skipped(self, config_factory):
        """相机没有该节点时返回 False —— 型号差异不应中断初始化。"""
        camera = GenICamCamera(config_factory())
        camera._node_map = FakeNodeMap(missing=["Gain"])
        assert camera._set_node("Gain", 1.0) is False

    def test_set_node_write_failure_is_reported_not_raised(self, config_factory):
        """节点只读 / 超量程时返回 False，不抛异常。

        相机型号不同可能缺少某些节点、或量程比配置更窄 ——
        不能因此中断整个初始化流程。
        """
        camera = GenICamCamera(config_factory())
        camera._node_map = FakeNodeMap(broken=["Width"])
        assert camera._set_node("Width", 100) is False

    def test_get_node_returns_value(self, config_factory):
        """节点可读时返回其值。"""
        camera = GenICamCamera(config_factory())
        camera._node_map = FakeNodeMap(values={"Width": 1280})
        assert camera._get_node("Width") == 1280

    def test_get_node_missing_returns_none(self, config_factory):
        """节点不存在时返回 None。"""
        camera = GenICamCamera(config_factory())
        camera._node_map = FakeNodeMap(missing=["Nope"])
        assert camera._get_node("Nope") is None

    def test_get_actual_params_empty_without_node_map(self, config_factory):
        """未打开时回读为空字典。"""
        assert GenICamCamera(config_factory()).get_actual_params() == {}

    def test_get_actual_params_reports_supported_nodes(self, config_factory):
        """回读只包含相机真正支持的节点。"""
        camera = GenICamCamera(config_factory())
        camera._node_map = FakeNodeMap(values={
            "Width": 2448,
            "Height": 2048,
            "PixelFormat": "Mono8",
            "ExposureTime": 5000.0,
            "ExposureAuto": "Off",
        })

        actual = camera.get_actual_params()
        assert actual["width"] == 2448
        assert actual["height"] == 2048
        assert actual["pixel_format"] == "Mono8"
        assert actual["exposure_us"] == 5000.0
        assert actual["exposure_auto"] == "Off"
        # 相机没有 Gain / TriggerMode 等节点 → 不出现在结果里
        assert "gain" not in actual
        assert "trigger_mode" not in actual

    def test_set_exposure_before_open_does_not_crash(self, config_factory):
        """未打开时设置曝光只改属性 —— 不应崩在没有节点树上。"""
        camera = GenICamCamera(config_factory())
        camera.set_exposure(1234.0)
        assert camera.exposure_us == 1234.0

    def test_set_exposure_writes_auto_off_first(self, config_factory):
        """再次设置曝光时仍要先关自动模式。"""
        camera = GenICamCamera(config_factory())
        node_map = _RecordingNodeMap()
        camera._node_map = node_map

        camera.set_exposure(2000.0)

        order = [name for name, _ in node_map.write_log]
        assert order.index("ExposureAuto") < order.index("ExposureTime")
        assert node_map.nodes["ExposureTime"].value == pytest.approx(2000.0)

    def test_set_gain_writes_auto_off_first(self, config_factory):
        """设置增益同理：GainAuto 必须先关。"""
        camera = GenICamCamera(config_factory())
        node_map = _RecordingNodeMap()
        camera._node_map = node_map

        camera.set_gain(4.0)

        order = [name for name, _ in node_map.write_log]
        assert order.index("GainAuto") < order.index("Gain")
        assert node_map.nodes["Gain"].value == pytest.approx(4.0)

    @pytest.mark.parametrize("mode, expected", [
        ("continuous", {"TriggerMode": "Off", "AcquisitionMode": "Continuous"}),
        ("software", {"TriggerMode": "On", "TriggerSource": "Software",
                      "AcquisitionMode": "Continuous"}),
    ])
    def test_apply_trigger_modes(self, mode, expected, config_factory):
        """连续采集与软触发的节点组合。"""
        camera = GenICamCamera(config_factory(camera={"trigger": {"mode": mode}}))
        node_map = FakeNodeMap()
        camera._node_map = node_map

        camera._apply_trigger()

        for name, value in expected.items():
            assert node_map.nodes[name].value == value

    def test_apply_external_trigger_uses_configured_source(self, config_factory):
        """外部硬触发：TriggerSource 取自配置，且要求上升沿。"""
        camera = GenICamCamera(config_factory(camera={
            "trigger": {"mode": "external", "source": "Line3"}
        }))
        node_map = FakeNodeMap()
        camera._node_map = node_map

        camera._apply_trigger()

        assert node_map.nodes["TriggerMode"].value == "On"
        assert node_map.nodes["TriggerSource"].value == "Line3"
        assert node_map.nodes["TriggerActivation"].value == "RisingEdge"

    def test_unknown_trigger_mode_falls_back_to_continuous(self, config_factory,
                                                           capsys):
        """未知触发模式退回连续采集，并打印可选值（不静默）。"""
        camera = GenICamCamera(config_factory(camera={"trigger": {"mode": "magic"}}))
        node_map = FakeNodeMap()
        camera._node_map = node_map

        camera._apply_trigger()

        assert node_map.nodes["TriggerMode"].value == "Off"
        out = capsys.readouterr().out
        assert "magic" in out
        assert "continuous" in out

    def test_apply_trigger_without_node_map_is_noop(self, config_factory):
        """未打开时下发触发配置不应抛异常。"""
        GenICamCamera(config_factory())._apply_trigger()

    @pytest.mark.parametrize("mode", ["continuous", "software", "external"])
    def test_all_trigger_modes_are_covered_by_plan(self, mode):
        """_TRIGGER_MODES 覆盖文档里承诺的三种模式。"""
        assert mode in cam._TRIGGER_MODES

    def test_trigger_plan_uses_sfnc_node_names(self):
        """触发相关节点名遵循 GenICam SFNC，海康相机无需特殊适配。"""
        for plan in cam._TRIGGER_MODES.values():
            assert set(plan) <= {
                "TriggerMode", "TriggerSource", "TriggerActivation",
                "AcquisitionMode",
            }
