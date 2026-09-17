"""core/acquisition.py 单元测试。

覆盖采集抽象基类、文件采集（离线）与相机采集（在线）三条路径。
相机路径**不接触真实硬件**：通过替换 hardware.camera.create_camera
注入 FakeCamera，把「相机没打开」「拿不到帧」「关掉后再取帧」等
分支逐条逼出来。需要真机的用例单独标 @pytest.mark.hardware。
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

import hardware.camera as hw_camera
from core.acquisition import (
    CameraAcquisition,
    FileAcquisition,
    ImageAcquisition,
    ImageFrame,
    create_acquisition,
)


# ============================================================================
# 夹具与小工具
# ============================================================================

def _write_image(path: Path, seed: int = 0, size: int = 24) -> None:
    """写一张可复现的随机灰度图。"""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (size, size), dtype=np.uint8)
    assert cv2.imwrite(str(path), img), f"写入失败: {path}"


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """三张受支持格式的图 + 两个干扰文件（.txt / .gif）。

    文件名刻意让字典序与创建序不一致，用来验证遍历按名称排序、
    而不是按文件系统返回顺序（后者在不同机器上不一致）。
    """
    d = tmp_path / "imgs"
    d.mkdir()
    _write_image(d / "b_second.png", seed=1)
    _write_image(d / "a_first.jpg", seed=2)
    _write_image(d / "c_third.bmp", seed=3)
    (d / "notes.txt").write_text("不是图像", encoding="utf-8")
    (d / "anim.gif").write_bytes(b"GIF89a")
    return d


@pytest.fixture
def image_dir_config(image_dir: Path) -> dict:
    """指向 image_dir 的离线配置。"""
    return {"system": {"mode": "offline", "image_dir": str(image_dir)}}


class FakeCamera:
    """最小可用的假相机，行为完全由测试摆布。

    只实现 CameraAcquisition 真正用到的四个成员：
    open / acquire / release / is_open。
    """

    def __init__(self, config: dict, open_ok: bool = True, frames=None):
        self.config = config
        self.last_error = ""
        self.open_calls = 0
        self.release_calls = 0
        self._open_ok = open_ok
        self._is_open = False
        self._frames = list(frames or [])

    def open(self) -> bool:
        self.open_calls += 1
        self._is_open = self._open_ok
        return self._open_ok

    def acquire(self):
        return self._frames.pop(0) if self._frames else None

    def release(self) -> None:
        self.release_calls += 1
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open


class FakeCameraInstaller:
    """把 create_camera 换掉，并记录被造出来的相机。"""

    def __init__(self):
        self.created = []

    def __call__(self, config: dict) -> FakeCamera:
        cam = FakeCamera(config)
        self.created.append(cam)
        return cam

    @property
    def last(self) -> FakeCamera:
        assert self.created, "create_camera 尚未被调用"
        return self.created[-1]


@pytest.fixture
def fake_camera(monkeypatch) -> FakeCameraInstaller:
    """注入假相机工厂，返回安装器（.last 拿到最近一台）。

    CameraAcquisition 在 __init__ 内部 `from hardware.camera import
    create_camera`，所以换掉 hardware.camera 模块上的属性即可生效。
    """
    installer = FakeCameraInstaller()
    monkeypatch.setattr(hw_camera, "create_camera", installer)
    return installer


def _online_config(camera: dict = None, system: dict = None) -> dict:
    """构造在线模式配置。"""
    cfg = {"system": {"mode": "online"}}
    if system:
        cfg["system"].update(system)
    if camera is not None:
        cfg["camera"] = camera
    return cfg


def _make_frame(cam: FakeCamera, seed: int = 0) -> None:
    """给假相机塞一帧可复现的图像。"""
    cam._frames.append(np.random.default_rng(seed).integers(0, 256, (8, 8, 3), dtype=np.uint8))


# ============================================================================
# ImageFrame
# ============================================================================

class TestImageFrame:
    """采集帧数据结构的构造与约束。"""

    def test_minimal_construction_uses_defaults(self):
        """只给必需字段时，其余字段取默认值。"""
        img = np.zeros((4, 4), dtype=np.uint8)
        frame = ImageFrame(image=img, timestamp="2026-01-01T00:00:00")
        assert frame.image is img
        assert frame.source_id == ""
        assert frame.frame_index == 0
        assert frame.metadata == {}

    def test_metadata_none_is_normalized_to_empty_dict(self):
        """metadata 显式传 None 时由 __post_init__ 归一化为空字典。"""
        frame = ImageFrame(image=None, timestamp="t", metadata=None)
        assert frame.metadata == {}

    def test_metadata_default_is_not_shared_between_instances(self):
        """metadata 的默认值不能是共享对象 —— 否则一帧写元数据会污染所有帧。

        这是 dataclass 可变默认值的经典坑，这里把行为钉住。
        """
        a = ImageFrame(image=None, timestamp="t")
        b = ImageFrame(image=None, timestamp="t")
        a.metadata["filename"] = "a.png"
        assert b.metadata == {}

    def test_metadata_is_passed_through_unchanged(self):
        """调用方给的 metadata 原样保留。"""
        meta = {"filename": "x.png", "exposure_us": 5000}
        frame = ImageFrame(image=None, timestamp="t", metadata=meta)
        assert frame.metadata == meta

    def test_metadata_type_is_not_validated(self):
        """实测行为记录：metadata 声明为 dict，但列表、字符串同样放行。

        __post_init__ 只做 None → {} 的归一化，不做类型检查。
        """
        frame = ImageFrame(image=None, timestamp="t", metadata=["不是字典"])
        assert frame.metadata == ["不是字典"]

    def test_image_is_not_validated(self):
        """实测行为记录：image 字段不做任何校验，None 也能构造成功。

        校验发生在别处（InspectorPipeline / utils.validators.validate_image），
        所以构造 ImageFrame 本身拦不住脏数据。
        """
        frame = ImageFrame(image=None, timestamp="t")
        assert frame.image is None

    @pytest.mark.parametrize("missing", ["image", "timestamp"])
    def test_missing_required_field_raises_type_error(self, missing: str):
        """缺少必需字段抛 TypeError。"""
        kwargs = {"image": None, "timestamp": "t"}
        kwargs.pop(missing)
        with pytest.raises(TypeError, match=missing):
            ImageFrame(**kwargs)

    def test_unknown_field_raises_type_error(self):
        """多余的字段名抛 TypeError，而不是静默丢弃。"""
        with pytest.raises(TypeError):
            ImageFrame(image=None, timestamp="t", camera_index=3)


# ============================================================================
# 抽象基类
# ============================================================================

class TestImageAcquisitionBase:
    """抽象基类的契约。"""

    def test_cannot_instantiate_directly(self):
        """抽象基类不能直接实例化。"""
        with pytest.raises(TypeError):
            ImageAcquisition({"system": {}})

    def test_subclass_without_acquire_cannot_be_instantiated(self):
        """只实现 reset 的子类仍不可实例化。"""

        class OnlyReset(ImageAcquisition):
            def reset(self):
                pass

        with pytest.raises(TypeError):
            OnlyReset({})

    def test_mode_is_read_from_config(self):
        """mode 从 system.mode 读取，缺省为 offline。"""
        assert FileAcquisition({"system": {"mode": "offline"}}).mode == "offline"
        assert FileAcquisition({"system": {}}).mode == "offline"
        assert FileAcquisition({}).mode == "offline"

    def test_default_close_is_noop(self, image_dir_config):
        """默认 close() 是空实现，不会抛异常。"""
        assert FileAcquisition(image_dir_config).close() is None


# ============================================================================
# FileAcquisition（离线）
# ============================================================================

class TestFileAcquisition:
    """文件采集。"""

    def test_counts_only_supported_extensions(self, image_dir: Path, image_dir_config):
        """只统计受支持的图像扩展名，.txt / .gif 不计入。"""
        acq = FileAcquisition(image_dir_config)
        assert acq.n_files == 3
        assert all(Path(f).suffix.lower() != ".txt" for f in acq._file_list)

    def test_file_list_is_sorted_by_name(self, image_dir: Path, image_dir_config):
        """遍历顺序按文件名排序，与文件系统返回顺序无关。"""
        acq = FileAcquisition(image_dir_config)
        names = [os.path.basename(f) for f in acq._file_list]
        assert names == ["a_first.jpg", "b_second.png", "c_third.bmp"]

    def test_image_dir_can_be_a_single_file(self, image_dir: Path, tmp_path: Path):
        """image_dir 指向单个文件时只装载这一张。"""
        one = image_dir / "a_first.jpg"
        acq = FileAcquisition({"system": {"image_dir": str(one)}})
        assert acq.n_files == 1
        assert Path(acq._file_list[0]) == one

    def test_nonexistent_dir_yields_empty_list(self, tmp_path: Path):
        """目录不存在时不报错，只是没有可用图像。"""
        acq = FileAcquisition({"system": {"image_dir": str(tmp_path / "no_such_dir")}})
        assert acq.n_files == 0
        assert acq.acquire() is None

    def test_empty_dir_returns_none(self, tmp_path: Path):
        """空目录：n_files 为 0，acquire 返回 None。"""
        d = tmp_path / "empty"
        d.mkdir()
        acq = FileAcquisition({"system": {"image_dir": str(d)}})
        assert acq.n_files == 0
        assert acq.acquire() is None

    def test_dir_without_images_returns_none(self, tmp_path: Path):
        """只有非图像文件的目录同样没有可用帧。"""
        d = tmp_path / "docs"
        d.mkdir()
        (d / "readme.md").write_text("x", encoding="utf-8")
        acq = FileAcquisition({"system": {"image_dir": str(d)}})
        assert acq.acquire() is None

    def test_acquire_iterates_in_order_and_exhausts(self, image_dir_config):
        """顺序取出全部文件，耗尽后返回 None 且不自动回头。"""
        acq = FileAcquisition(image_dir_config)
        frames = [acq.acquire() for _ in range(acq.n_files)]
        names = [os.path.basename(f.source_id) for f in frames]
        assert names == ["a_first.jpg", "b_second.png", "c_third.bmp"]
        assert [f.frame_index for f in frames] == [0, 1, 2]
        assert acq.acquire() is None
        assert acq.acquire() is None

    def test_frame_carries_metadata_and_source_id(self, image_dir_config):
        """帧上带有源文件路径与文件名元数据。"""
        acq = FileAcquisition(image_dir_config)
        frame = acq.acquire()
        assert isinstance(frame, ImageFrame)
        assert os.path.basename(frame.source_id) == "a_first.jpg"
        assert frame.metadata["filename"] == "a_first.jpg"
        assert frame.image is not None
        assert frame.image.ndim == 3  # cv2.imread 默认按彩色读取

    def test_timestamp_is_iso_format(self, image_dir_config):
        """时间戳是合法的 ISO 格式字符串。"""
        from datetime import datetime

        frame = FileAcquisition(image_dir_config).acquire()
        datetime.fromisoformat(frame.timestamp)

    def test_reset_rewinds_to_first_file(self, image_dir_config):
        """reset 后从第一张重新开始，且帧序号归零。"""
        acq = FileAcquisition(image_dir_config)
        acq.acquire()
        acq.acquire()
        acq.reset()
        frame = acq.acquire()
        assert os.path.basename(frame.source_id) == "a_first.jpg"
        assert frame.frame_index == 0

    def test_reset_on_empty_dir_is_safe(self, tmp_path: Path):
        """空目录上 reset 不应抛异常。"""
        acq = FileAcquisition({"system": {"image_dir": str(tmp_path / "none")}})
        acq.reset()
        assert acq.n_files == 0

    def test_loop_wraps_around(self, image_dir_config):
        """offline_loop=True 时耗尽后回到第一张继续。"""
        cfg = {"system": dict(image_dir_config["system"], offline_loop=True)}
        acq = FileAcquisition(cfg)
        names = [os.path.basename(acq.acquire().source_id) for _ in range(4)]
        assert names == [
            "a_first.jpg", "b_second.png", "c_third.bmp", "a_first.jpg",
        ]
        # 帧计数器持续累加，不随回绕归零
        assert acq.acquire().frame_index == 4

    def test_loop_disabled_returns_none_at_end(self, image_dir_config):
        """offline_loop 默认为 False，耗尽即停。"""
        acq = FileAcquisition(image_dir_config)
        for _ in range(acq.n_files):
            acq.acquire()
        assert acq.acquire() is None

    def test_unreadable_file_is_skipped(self, tmp_path: Path):
        """坏文件被跳过，返回的是下一张能读的图。"""
        d = tmp_path / "mixed"
        d.mkdir()
        (d / "a_broken.png").write_bytes(b"not a png at all")
        _write_image(d / "b_good.png", seed=7)
        acq = FileAcquisition({"system": {"image_dir": str(d)}})
        frame = acq.acquire()
        assert frame is not None
        assert os.path.basename(frame.source_id) == "b_good.png"
        assert frame.frame_index == 0  # 坏文件不占用帧序号
        assert acq.acquire() is None

    def test_only_unreadable_files_returns_none(self, tmp_path: Path):
        """全是坏文件时（非循环模式）返回 None 而非抛异常。"""
        d = tmp_path / "broken"
        d.mkdir()
        for name in ("a.png", "b.png"):
            (d / name).write_bytes(b"broken image data")
        acq = FileAcquisition({"system": {"image_dir": str(d)}})
        assert acq.acquire() is None
        assert acq.acquire() is None

    def test_acquire_after_exhaustion_then_reset_works(self, image_dir_config):
        """耗尽 → reset → 又能取到帧。"""
        acq = FileAcquisition(image_dir_config)
        for _ in range(acq.n_files + 1):
            acq.acquire()
        acq.reset()
        assert acq.acquire() is not None

    def test_context_manager_returns_self(self, image_dir_config):
        """with 语句返回采集器本身，退出时不报错。"""
        with FileAcquisition(image_dir_config) as acq:
            assert isinstance(acq, FileAcquisition)
            assert acq.acquire() is not None

    @pytest.mark.xfail(
        reason="FileAcquisition.acquire() 用递归跳过坏文件：offline_loop=true 且目录内"
               "全是读不出来的图时，回绕与递归互相放大，直接 RecursionError（栈溢出），"
               "而不是优雅返回 None",
        strict=False,
    )
    def test_loop_with_only_unreadable_files_terminates(self, tmp_path: Path):
        """循环模式 + 全坏目录应能停下来，不该递归到栈溢出。"""
        d = tmp_path / "broken_loop"
        d.mkdir()
        (d / "only.png").write_bytes(b"broken image data")
        acq = FileAcquisition(
            {"system": {"image_dir": str(d), "offline_loop": True}}
        )
        assert acq.acquire() is None


# ============================================================================
# CameraAcquisition（在线，全部走假相机）
# ============================================================================

class TestCameraAcquisition:
    """相机采集 —— 用假相机覆盖各分支，不连真机。"""

    def test_opens_camera_exactly_once_on_init(self, fake_camera):
        """构造时就打开相机，且只调一次 open()。"""
        CameraAcquisition(_online_config())
        assert fake_camera.last.open_calls == 1
        assert fake_camera.last.is_open is True

    def test_camera_is_created_from_config(self, fake_camera):
        """create_camera 收到的是完整配置对象。"""
        cfg = _online_config(camera={"driver": "opencv"})
        CameraAcquisition(cfg)
        assert fake_camera.last.config is cfg

    def test_source_id_uses_camera_device_index(self, fake_camera):
        """source_id 取自 camera.device.index（新键）。"""
        acq = CameraAcquisition(_online_config(camera={"device": {"index": 3}}))
        _make_frame(fake_camera.last)
        assert acq.acquire().source_id == "camera_3"

    def test_source_id_falls_back_to_legacy_camera_id(self, fake_camera):
        """camera.device.index 缺失时回退到旧键 system.camera_id。"""
        acq = CameraAcquisition(
            _online_config(camera={"driver": "opencv"}, system={"camera_id": 2})
        )
        _make_frame(fake_camera.last)
        assert acq.acquire().source_id == "camera_2"

    def test_source_id_defaults_to_zero_without_any_index_config(self, fake_camera):
        """新旧两处都没配时默认 0。"""
        acq = CameraAcquisition(_online_config())
        _make_frame(fake_camera.last)
        assert acq.acquire().source_id == "camera_0"

    @pytest.mark.parametrize(
        "camera_section",
        [None, {}, {"device": None}, {"device": {}}],
        ids=["camera_none", "no_device", "device_none", "device_empty"],
    )
    def test_source_id_fallback_survives_degenerate_camera_section(
        self, fake_camera, camera_section
    ):
        """camera 节为 None / 缺 device / device 为 None 时都应安全回退。"""
        acq = CameraAcquisition(
            _online_config(camera=camera_section, system={"camera_id": 2})
        )
        _make_frame(fake_camera.last)
        assert acq.acquire().source_id == "camera_2"

    def test_new_key_wins_over_legacy_key(self, fake_camera):
        """新旧键同时存在时，新键优先（重构后的口径）。"""
        acq = CameraAcquisition(
            _online_config(camera={"device": {"index": 5}}, system={"camera_id": 2})
        )
        _make_frame(fake_camera.last)
        assert acq.acquire().source_id == "camera_5"

    def test_acquire_returns_none_when_camera_failed_to_open(self, monkeypatch):
        """相机打开失败时，acquire 返回 None 而不是抛异常。"""
        installer = FakeCameraInstaller()
        monkeypatch.setattr(
            hw_camera, "create_camera", lambda cfg: FakeCamera(cfg, open_ok=False)
        )
        acq = CameraAcquisition(_online_config())
        assert acq._camera.is_open is False
        assert acq.acquire() is None

    def test_acquire_returns_none_when_camera_yields_nothing(self, fake_camera):
        """相机返回 None（超时/丢帧）时，acquire 返回 None 且不推进帧计数。"""
        acq = CameraAcquisition(_online_config())
        assert acq.acquire() is None
        assert acq.acquire() is None
        _make_frame(fake_camera.last, seed=1)
        assert acq.acquire().frame_index == 0

    def test_frame_index_increments(self, fake_camera):
        """连续取帧时 frame_index 递增。"""
        acq = CameraAcquisition(_online_config())
        for i in range(3):
            _make_frame(fake_camera.last, seed=i)
        assert [acq.acquire().frame_index for _ in range(3)] == [0, 1, 2]

    def test_frame_carries_image_and_timestamp(self, fake_camera):
        """帧上带图像本体与 ISO 时间戳。"""
        from datetime import datetime

        acq = CameraAcquisition(_online_config())
        _make_frame(fake_camera.last)
        frame = acq.acquire()
        assert frame.image is not None
        assert frame.image.ndim == 3
        datetime.fromisoformat(frame.timestamp)
        assert frame.metadata == {}

    def test_reset_restarts_frame_counter(self, fake_camera):
        """reset 把帧序号归零，且不影响相机连接。"""
        acq = CameraAcquisition(_online_config())
        _make_frame(fake_camera.last)
        acq.acquire()
        acq.reset()
        _make_frame(fake_camera.last, seed=9)
        assert acq.acquire().frame_index == 0
        assert fake_camera.last.release_calls == 0

    def test_close_releases_camera(self, fake_camera):
        """close 释放相机并把内部引用置空。"""
        acq = CameraAcquisition(_online_config())
        cam = fake_camera.last
        acq.close()
        assert cam.release_calls == 1
        assert acq._camera is None

    def test_close_is_idempotent(self, fake_camera):
        """重复 close 不应重复 release，也不应抛异常。"""
        acq = CameraAcquisition(_online_config())
        cam = fake_camera.last
        acq.close()
        acq.close()
        assert cam.release_calls == 1

    def test_acquire_after_close_returns_none(self, fake_camera):
        """关掉之后再取帧返回 None，而不是访问已释放的设备。"""
        acq = CameraAcquisition(_online_config())
        _make_frame(fake_camera.last)
        acq.close()
        assert acq.acquire() is None

    def test_context_manager_releases_on_exit(self, fake_camera):
        """with 块退出时自动释放相机。"""
        with CameraAcquisition(_online_config()) as acq:
            _make_frame(fake_camera.last)
            assert acq.acquire() is not None
            cam = fake_camera.last
        assert cam.release_calls == 1
        assert acq._camera is None

    def test_reset_on_closed_camera_is_safe(self, fake_camera):
        """已关闭的采集器上 reset 不应抛异常。"""
        acq = CameraAcquisition(_online_config())
        acq.close()
        acq.reset()
        assert acq.acquire() is None

    @pytest.mark.hardware
    def test_real_camera_roundtrip(self, default_config: dict):
        """真机冒烟：默认配置能建出采集器，能打开就抓一帧。

        无相机 / 无 SDK 的机器上 skip，绝不因缺硬件而失败。
        """
        acq = CameraAcquisition(default_config)
        try:
            if acq._camera is None or not acq._camera.is_open:
                pytest.skip(
                    f"相机不可用: {getattr(acq._camera, 'last_error', '未打开')}"
                )
            frame = acq.acquire()
            if frame is None:
                pytest.skip("相机已打开但本次未取到帧（超时）")
            assert isinstance(frame, ImageFrame)
            assert frame.source_id.startswith("camera_")
        finally:
            acq.close()


# ============================================================================
# 工厂函数
# ============================================================================

class TestCreateAcquisition:
    """create_acquisition 的模式分发。"""

    def test_offline_mode_returns_file_acquisition(self, tmp_path: Path):
        """offline → FileAcquisition。"""
        acq = create_acquisition({"system": {"mode": "offline", "image_dir": str(tmp_path)}})
        assert isinstance(acq, FileAcquisition)
        assert not isinstance(acq, CameraAcquisition)

    def test_online_mode_returns_camera_acquisition(self, fake_camera):
        """online → CameraAcquisition。"""
        acq = create_acquisition(_online_config())
        assert isinstance(acq, CameraAcquisition)
        assert fake_camera.last.open_calls == 1

    def test_missing_mode_defaults_to_offline(self, tmp_path: Path):
        """未配 mode 时默认离线。"""
        acq = create_acquisition({"system": {"image_dir": str(tmp_path)}})
        assert isinstance(acq, FileAcquisition)

    def test_empty_config_defaults_to_offline(self, tmp_path: Path, monkeypatch):
        """空配置走离线分支（image_dir 回退到默认值，用 chdir 兜住）。"""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "samples").mkdir(parents=True)
        acq = create_acquisition({})
        assert isinstance(acq, FileAcquisition)
        assert acq.n_files == 0

    def test_unknown_mode_falls_back_to_offline(self, tmp_path: Path):
        """实测行为记录：mode 拼错（如 'onilne'）不报错，静默按离线处理。

        工厂只有 online / 其它 两个分支。生产线上一个字母打错，系统不会
        报错，而是拿样本图片继续跑 —— 见报告中的风险观察项。
        """
        acq = create_acquisition(
            {"system": {"mode": "onilne", "image_dir": str(tmp_path)}}
        )
        assert isinstance(acq, FileAcquisition)

    def test_returned_object_satisfies_base_interface(self, tmp_path: Path):
        """返回值满足抽象基类契约（可 with、可 reset、可 close）。"""
        with create_acquisition(
            {"system": {"mode": "offline", "image_dir": str(tmp_path)}}
        ) as acq:
            assert isinstance(acq, ImageAcquisition)
            acq.reset()
