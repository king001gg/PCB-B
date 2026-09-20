"""主窗口的模式分派与抓拍逻辑测试（无头 Qt，不依赖真实相机）。

只覆盖不碰设备的逻辑：按钮按模式分派、抓拍的前置条件与抓拍后的状态、
检测期间按钮的启停。相机采集线程本身不在这里测。

用 offscreen 平台跑，CI 上无需显示器。
"""

import os
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication  # noqa: E402

import ui.main_window as mw_module  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    """整个模块共用一个 QApplication —— Qt 不允许同进程建多个。"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def window(qapp, monkeypatch):
    """构造主窗口，并把 _stop_live 换成不动相机的假实现。

    真实 _stop_live 会去 wait/terminate 采集线程，测试里没有相机线程。
    """
    w = MainWindow()
    stopped = []

    def fake_stop_live():
        stopped.append(True)
        w._live_mode = False

    monkeypatch.setattr(w, "_stop_live", fake_stop_live)
    w._stop_live_calls = stopped
    yield w
    w.close()


@pytest.fixture
def frozen_frame():
    """一张可辨认的假帧（左上角为 255，便于验证拿到的是同一份数据）。"""
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[0, 0] = 255
    return frame


# ---------------------------------------------------------------------------
# 按钮分派（修复前：文案是「拍照」但弹的是文件选择框）
# ---------------------------------------------------------------------------

class TestLoadButtonDispatch:

    def test_offline_mode_opens_file_dialog(self, window, monkeypatch):
        """离线模式：按钮走文件加载。"""
        called = []
        monkeypatch.setattr(window, "load_image", lambda: called.append("file"))
        monkeypatch.setattr(window, "_capture_frame",
                            lambda: called.append("capture") or True)
        window.mode = "offline"

        window.load_btn.click()

        assert called == ["file"]

    def test_online_mode_captures_from_preview(self, window, monkeypatch):
        """在线模式：按钮走抓拍，不再弹文件选择框。"""
        called = []
        monkeypatch.setattr(window, "load_image", lambda: called.append("file"))
        monkeypatch.setattr(window, "_capture_frame",
                            lambda: called.append("capture") or True)
        window.mode = "online"

        window.load_btn.click()

        assert called == ["capture"]

    def test_label_follows_mode(self, window):
        """按钮文案随模式切换，且与分派逻辑同源。"""
        window.mode = "online"
        window._sync_mode_controls()
        assert window.load_btn.text() == "📷 拍照"

        window.mode = "offline"
        window._sync_mode_controls()
        assert window.load_btn.text() == "📂 加载图像"

    def test_set_mode_online_switches_label(self, window):
        """菜单切到在线模式后，文案应立刻变成「拍照」。"""
        window._set_mode("online")
        assert window.mode == "online"
        assert window.load_btn.text() == "📷 拍照"

    def test_set_mode_offline_stops_preview(self, window):
        """切回离线模式必须停掉预览，否则文案与实际状态不一致。"""
        window._live_mode = True
        window._set_mode("offline")

        assert window._stop_live_calls == [True]
        assert window.load_btn.text() == "📂 加载图像"


# ---------------------------------------------------------------------------
# 抓拍
# ---------------------------------------------------------------------------

class TestCaptureFrame:

    def test_refused_when_not_previewing(self, window):
        """没开预览时抓拍应被拒绝，并给出可操作的提示。"""
        window._live_mode = False

        assert window._capture_frame() is False
        assert "实时预览" in window.status_bar.currentMessage()

    def test_refused_before_first_frame(self, window, monkeypatch):
        """预览已开但还没收到帧时抓拍应被拒绝。"""
        window._live_mode = True
        window.current_raw_image = None

        assert window._capture_frame() is False
        assert "尚未取到画面" in window.status_bar.currentMessage()

    def test_freezes_the_frame_and_leaves_live_mode(self, window, frozen_frame):
        """成功抓拍：停预览、留下冻结副本、给出板号、解锁检测。"""
        window._live_mode = True
        window.current_raw_image = frozen_frame
        window.current_image_path = "旧文件.png"

        assert window._capture_frame() is True

        assert window._live_mode is False
        assert window._stop_live_calls == [True]
        assert window.current_image_path is None
        assert window._snapshot_id.startswith("抓拍-")
        assert window.detect_btn.isEnabled()
        assert "已抓拍" in window.status_bar.currentMessage()

    def test_frozen_copy_is_not_the_live_buffer(self, window, frozen_frame):
        """必须是副本：预览缓冲随后会被下一帧覆盖，而检测在后台线程里读它。"""
        window._live_mode = True
        window.current_raw_image = frozen_frame

        window._capture_frame()

        assert window.current_raw_image is not frozen_frame
        assert np.array_equal(window.current_raw_image, frozen_frame)
        frozen_frame[:] = 7          # 模拟下一帧覆盖缓冲
        assert window.current_raw_image[0, 0, 0] == 255

    def test_clears_previous_result_views(self, window, frozen_frame):
        """抓拍后要清掉上一块板的结果，避免误读。"""
        window._live_mode = True
        window.current_raw_image = frozen_frame
        window.result_text.setText("上一块板的缺陷列表")
        window.score_label.setText("88.8")

        window._capture_frame()

        assert window.result_text.toPlainText() == ""
        assert window.score_label.text() == "--"


# ---------------------------------------------------------------------------
# 板号
# ---------------------------------------------------------------------------

class TestBoardId:

    def test_from_file_name(self, window):
        window.current_image_path = r"C:\data\PCB-0007.png"
        window._snapshot_id = "抓拍-101010"
        assert window._current_board_id() == "PCB-0007.png"

    def test_from_snapshot_when_no_file(self, window):
        window.current_image_path = None
        window._snapshot_id = "抓拍-101010"
        assert window._current_board_id() == "抓拍-101010"

    def test_empty_when_nothing_loaded(self, window):
        window.current_image_path = None
        window._snapshot_id = None
        assert window._current_board_id() == ""

    def test_loading_a_file_clears_snapshot_id(self, window, tmp_path):
        """先抓拍再打开文件，板号不能还留着上一次抓拍的时间。"""
        ok, buf = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
        assert ok
        path = tmp_path / "board.png"
        path.write_bytes(buf.tobytes())

        window._snapshot_id = "抓拍-101010"
        window._set_image(str(path))

        assert window._snapshot_id is None
        assert window._current_board_id() == "board.png"


# ---------------------------------------------------------------------------
# 检测与预览的互斥
# ---------------------------------------------------------------------------

class TestDetectionAndPreviewExclusion:

    @pytest.fixture
    def worker_spy(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(mw_module, "DetectionWorker", spy)
        return spy

    def test_live_detection_captures_before_detecting(self, window, worker_spy,
                                                      monkeypatch):
        """在线模式点检测：先抓拍，再走离线路径。"""
        captured = []
        monkeypatch.setattr(window, "_capture_frame",
                            lambda: captured.append(True) or True)
        window._live_mode = True
        window.current_raw_image = np.zeros((8, 8, 3), dtype=np.uint8)
        window._snapshot_id = "抓拍-120000"

        window.start_detection()

        assert captured == [True]
        assert worker_spy.call_count == 1
        assert worker_spy.call_args.kwargs["board_id"] == "抓拍-120000"

    def test_aborts_when_capture_fails(self, window, worker_spy, monkeypatch):
        """抓拍失败绝不能拿预览帧继续检测。"""
        monkeypatch.setattr(window, "_capture_frame", lambda: False)
        window._live_mode = True
        window.current_raw_image = np.zeros((8, 8, 3), dtype=np.uint8)

        window.start_detection()

        assert worker_spy.call_count == 0

    def test_preview_button_locked_while_detecting(self, window, worker_spy,
                                                   monkeypatch):
        """检测期间禁止进预览：两条路径都会写工艺面板。"""
        monkeypatch.setattr(window, "_capture_frame", lambda: True)
        window._live_mode = True
        window.current_raw_image = np.zeros((8, 8, 3), dtype=np.uint8)

        window.start_detection()
        assert not window.live_btn.isEnabled()
        assert not window.detect_btn.isEnabled()

    def test_error_reenables_preview_button(self, window, worker_spy,
                                            monkeypatch):
        """检测报错也必须解锁预览，否则用户再也进不去预览。"""
        monkeypatch.setattr(window, "_capture_frame", lambda: True)
        monkeypatch.setattr(mw_module.QMessageBox, "critical",
                            lambda *a, **k: None)
        window._live_mode = True
        window.current_raw_image = np.zeros((8, 8, 3), dtype=np.uint8)
        window.start_detection()

        window._on_detection_error("模拟失败")

        assert window.live_btn.isEnabled()
        assert window.detect_btn.isEnabled()
        assert window.load_btn.isEnabled()


# ---------------------------------------------------------------------------
# 在途帧不得覆盖冻结图
# ---------------------------------------------------------------------------

class TestStaleFrameDropped:

    def test_frame_ignored_when_preview_stopped(self, window):
        """预览停止后到达的帧必须丢弃，否则会盖掉刚抓拍的冻结图。"""
        window._live_mode = False
        frozen = np.full((8, 8, 3), 255, dtype=np.uint8)
        window.current_raw_image = frozen

        window._on_frame(np.zeros((8, 8, 3), dtype=np.uint8))

        assert window.current_raw_image is frozen
        assert window.current_raw_image[0, 0, 0] == 255

    def test_frame_accepted_while_previewing(self, window):
        """预览进行中，帧要正常显示。"""
        window._live_mode = True
        window.process_monitor.enabled = False

        frame = np.full((8, 8, 3), 42, dtype=np.uint8)
        window._on_frame(frame)

        assert window.current_raw_image is frame
