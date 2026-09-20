"""相机设置对话框测试（``ui/camera_dialog.py``）。

**重点是配置往返保真**：对话框读到的值必须能原样写回去 ——
``ui/main_window.py:532`` 保存的就是 ``dialog.camera_config``，
而它在点「确定」时由 ``_form_config()`` 覆盖（见 ``_save_and_close``）。

踩过的坑：``QComboBox.findText`` 找不到时返回 **-1**，配合
``setCurrentIndex(max(idx, 0))`` 就静默落到第 0 项。于是一个不在候选表里的像素格式
（如海康彩色机原生的 ``BayerRG8``）会被显示成、并在确定时**写回** ``Mono8``。
症状是「换了彩色相机却还是黑白」，而且全程不报任何错 —— 所以只能靠测试守。
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication

from ui.camera_dialog import CameraDialog, _PIXEL_FORMATS


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def dialog(qapp):
    """对话框工厂。

    照 ``test_main_window_ui`` 的既有写法：用完只 ``close()``，**不保留引用、
    也不 ``deleteLater()``**。多留一层引用（哪怕只是为了让 fixture 统一清理）
    会让 PySide6 的 meta-property 跟 Python 对象结成环，解释器退出时回收不掉、
    报一行 ResourceWarning 盖在测试输出里。
    """
    made = []

    def _make(**camera_cfg) -> CameraDialog:
        dlg = CameraDialog(camera_cfg)
        made.append(dlg)
        return dlg

    yield _make
    for dlg in made:
        dlg.close()
    made.clear()


class TestPixelFormatRoundTrip:
    """配置里写的像素格式，过一遍对话框必须原样出来。"""

    @pytest.mark.parametrize("value", [
        "Mono8", "Mono10", "Mono12", "RGB8", "BGR8",
        "BayerRG8", "BayerGB8", "BayerGR8", "BayerBG8", "YUV422",
    ])
    def test_listed_values_round_trip(self, dialog, value):
        assert dialog(pixel_format=value).pixel_combo.currentText() == value

    @pytest.mark.parametrize("value", [
        # 直接填的 SDK 枚举名（config/default.yaml 的注释里就让人这么填）
        "RGB8_Packed", "BGR8_Packed", "YUV422_Packed",
        # 候选表没覆盖到的型号
        "BayerRG10", "BayerRG12",
        # 大小写不同 —— 驱动的 _resolve_pixel_format 会 lower()，不该被对话框改写
        "bgr8",
    ])
    def test_unlisted_values_survive_instead_of_becoming_mono8(self, dialog, value):
        """候选表里没有的值必须原样保留 —— 这条就是本文件存在的理由。"""
        assert dialog(pixel_format=value).pixel_combo.currentText() == value

    def test_saving_preserves_a_bayer_format(self, dialog):
        """走一遍确定按钮，确认落到 camera_config（= 真正被保存的那份）。"""
        dlg = dialog(pixel_format="BayerRG8")
        dlg._save_and_close()
        assert dlg.camera_config["pixel_format"] == "BayerRG8"

    def test_saving_an_unlisted_value_does_not_fall_back_to_mono8(self, dialog):
        """回归用例：这正是「换了彩色相机却还是黑白」的直接原因。"""
        dlg = dialog(pixel_format="RGB8_Packed")
        dlg._save_and_close()
        assert dlg.camera_config["pixel_format"] == "RGB8_Packed"

    def test_combo_is_editable_so_new_formats_can_be_typed(self, dialog):
        """跟触发源下拉框一样的处理 —— 表列不全也不至于把用户堵死。"""
        assert dialog().pixel_combo.isEditable()

    def test_a_hand_typed_format_is_what_gets_saved(self, dialog):
        """用户手打的值必须真的落地，而不是被下拉框的当前选择盖掉。"""
        dlg = dialog(pixel_format="Mono8")
        dlg.pixel_combo.setEditText("BayerRG12")
        dlg._save_and_close()
        assert dlg.camera_config["pixel_format"] == "BayerRG12"


class TestPixelFormatChoices:
    def test_missing_pixel_format_falls_back_to_mono8(self, dialog):
        assert dialog().pixel_combo.currentText() == "Mono8"

    def test_bayer_formats_are_offered(self):
        """很多海康彩色机原生只出 Bayer，不提供 RGB8/BGR8 打包输出。

        表里没有它们，用户会以为「这相机不支持彩色」而跑去换驱动。
        """
        for fmt in ("BayerRG8", "BayerGB8", "BayerGR8", "BayerBG8"):
            assert fmt in _PIXEL_FORMATS

    def test_choices_do_not_contain_duplicates(self):
        assert len(set(_PIXEL_FORMATS)) == len(_PIXEL_FORMATS)


class TestOtherFieldsStillWork:
    """顺带守一下：改像素格式的下拉框不该碰坏别的字段。"""

    def test_geometry_and_trigger_survive_a_round_trip(self, dialog):
        dlg = dialog(
            width=1280, height=1024, pixel_format="BayerRG8",
            exposure_us=8000, gain=2.0,
            device={"index": 1, "serial_number": "SN123"},
            trigger={"mode": "software", "source": "Line2"},
        )
        dlg._save_and_close()
        cfg = dlg.camera_config
        assert cfg["width"] == 1280
        assert cfg["height"] == 1024
        assert cfg["pixel_format"] == "BayerRG8"
        assert cfg["device"]["serial_number"] == "SN123"
        assert cfg["trigger"]["mode"] == "software"
        assert cfg["trigger"]["source"] == "Line2"

    def test_test_connection_uses_the_form_value_not_the_saved_one(self, dialog):
        """「测试连接」要验用户刚填的格式，否则等于没测。"""
        dlg = dialog(pixel_format="Mono8")
        dlg.pixel_combo.setEditText("BayerRG8")
        assert dlg._form_config()["camera"]["pixel_format"] == "BayerRG8"
