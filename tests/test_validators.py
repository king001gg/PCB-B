"""utils/validators.py 单元测试。

本文件覆盖图像 / 路径 / 配置三类校验函数。其中 validate_config 是配置结构
重构（触发与设备键从 system.trigger_mode、camera.trigger_source 收口到
camera.trigger.mode、camera.device.index）后新旧配置的兼容守门人，因此
旧版配置、缺键配置、非法值配置的处理方式逐条记录在案 —— 记录的是**实际
行为**，而非期望行为；与文档承诺不符的地方用 xfail 标注，不做静默美化。
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from utils.validators import (
    validate_config,
    validate_config_path,
    validate_file_path,
    validate_image,
)


# ============================================================================
# 夹具
# ============================================================================

@pytest.fixture
def legacy_config(default_config: dict) -> dict:
    """重构前的旧版配置：相机相关键散落在 system 与 camera 两处。

    旧版形态（本次重构要消除的）：
        - system.camera_id      → 现 camera.device.index
        - system.trigger_mode   → 现 camera.trigger.mode
        - camera.trigger_source → 现 camera.trigger.source

    这里刻意**同时**保留旧键并删掉新键，模拟升级后用户手上那份没改过的
    配置文件 —— 校验层必须能明确表态（接受 / 拒绝 / 迁移），不能含糊。
    """
    cfg = copy.deepcopy(default_config)
    cfg["system"]["camera_id"] = 2
    cfg["system"]["trigger_mode"] = "software"
    cfg["camera"]["trigger_source"] = "Line0"
    cfg["camera"].pop("device", None)
    cfg["camera"].pop("trigger", None)
    return cfg


@pytest.fixture
def yaml_file(tmp_path: Path) -> Path:
    """磁盘上一个真实存在的最小 YAML 文件。"""
    p = tmp_path / "conf.yaml"
    p.write_text("system:\n  mode: offline\n", encoding="utf-8")
    return p


# ============================================================================
# validate_image
# ============================================================================

class TestValidateImage:
    """图像数组校验。"""

    def test_accepts_uint8_gray_image(self):
        """uint8 单通道灰度图应通过。"""
        validate_image(np.zeros((32, 32), dtype=np.uint8))

    def test_accepts_uint8_color_image(self, rgb_surface):
        """uint8 三通道彩色图应通过。"""
        validate_image(rgb_surface)

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
    def test_accepts_supported_dtypes(self, dtype):
        """四种受支持的 dtype 均应通过。"""
        validate_image(np.zeros((8, 8), dtype=dtype))

    @pytest.mark.parametrize(
        "dtype", [np.float16, np.int8, np.int32, np.int64, np.complex64]
    )
    def test_rejects_unsupported_dtypes(self, dtype):
        """白名单外的数值 dtype 应被拒（float16 / 有符号整数尤其危险）。"""
        with pytest.raises(ValueError, match="dtype"):
            validate_image(np.zeros((8, 8), dtype=dtype))

    def test_rejects_bool_dtype(self):
        """bool 数组应被拒 —— 它看起来像二值掩膜，但不是灰度图。"""
        with pytest.raises(ValueError, match="dtype"):
            validate_image(np.zeros((8, 8), dtype=bool))

    def test_rejects_none(self):
        """None 应抛 TypeError 而非 AttributeError。"""
        with pytest.raises(TypeError, match="numpy.ndarray"):
            validate_image(None)

    @pytest.mark.parametrize("value", [[[1, 2], [3, 4]], "image", 42, 3.14, {"a": 1}])
    def test_rejects_non_ndarray(self, value):
        """非 numpy 输入一律 TypeError，且消息里带上实际类型名。"""
        with pytest.raises(TypeError, match="numpy.ndarray"):
            validate_image(value)

    @pytest.mark.parametrize("shape", [(0, 0), (0, 16), (16, 0)])
    def test_rejects_empty_arrays(self, shape):
        """size 为 0 的二维数组应被拒。"""
        with pytest.raises(ValueError, match="尺寸不能为零"):
            validate_image(np.zeros(shape, dtype=np.uint8))

    @pytest.mark.parametrize("shape", [(16,), (4, 4, 4, 4)])
    def test_rejects_wrong_ndim(self, shape):
        """1 维与 4 维输入应被拒，ndim 必须为 2 或 3。"""
        with pytest.raises(ValueError, match="维度"):
            validate_image(np.zeros(shape, dtype=np.uint8))

    def test_empty_1d_reports_ndim_not_size(self):
        """(0,) 会先在维度上被拦下，报的是维度错而非尺寸错。

        校验顺序：类型 → 维度 → 尺寸 → dtype。这里把顺序钉住，
        免得日后有人调换判断次序、把错误消息改得对不上调用方的日志分析。
        """
        with pytest.raises(ValueError, match="维度"):
            validate_image(np.zeros((0,), dtype=np.uint8))

    def test_error_message_uses_given_name(self):
        """name 参数应出现在错误消息里，便于定位是哪个入参出问题。"""
        with pytest.raises(ValueError, match="frame_gray"):
            validate_image(np.zeros((4,), dtype=np.uint8), name="frame_gray")

    def test_accepts_1x1_image(self):
        """1x1 图像当前**通过**校验。

        实测行为记录：只拦 size == 0，没有最小尺寸（例如 16x16）检查。
        下游算法（GLCM / Gabor）在 1x1 上要么抛异常要么给出无意义的数，
        因此这是校验层的一个已知空档，见报告中的观察项。
        """
        validate_image(np.zeros((1, 1), dtype=np.uint8))

    def test_accepts_huge_image(self):
        """超大图当前**不设上限**，也不做内存/分辨率合理性检查。

        用广播视图构造一张 16 亿像素的图，零内存占用，只验证「没有上限
        判断」这一事实本身。
        """
        huge = np.broadcast_to(np.zeros((1, 1), dtype=np.uint8), (40000, 40000))
        assert huge.size == 1_600_000_000
        validate_image(huge)

    def test_accepts_any_channel_count_in_3d(self):
        """3 维输入不校验通道数：7 通道同样放行。

        实测行为记录：只要求 ndim == 3。下游按 BGR/RGB 3 通道处理，
        7 通道图会一路带病流到 cvtColor 才炸，错误点远离入口。
        """
        validate_image(np.zeros((8, 8, 7), dtype=np.uint8))


# ============================================================================
# validate_file_path
# ============================================================================

class TestValidateFilePath:
    """文件路径校验。"""

    def test_existing_file_returns_absolute_path(self, yaml_file: Path):
        """存在的文件返回绝对路径。"""
        got = validate_file_path(str(yaml_file))
        assert Path(got).is_absolute()
        assert Path(got) == yaml_file

    def test_relative_path_is_normalized(self, tmp_path: Path, monkeypatch):
        """相对路径被规范化为绝对路径。"""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.txt").write_text("x", encoding="utf-8")
        assert Path(validate_file_path("a.txt")) == tmp_path / "a.txt"

    def test_missing_file_raises(self, tmp_path: Path):
        """文件不存在时抛 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            validate_file_path(str(tmp_path / "nope.txt"))

    def test_missing_file_ok_when_must_exist_false(self, tmp_path: Path):
        """must_exist=False 时不检查存在性，直接返回绝对路径。"""
        target = tmp_path / "not_yet" / "out.json"
        assert Path(validate_file_path(str(target), must_exist=False)) == target

    def test_directory_is_rejected(self, tmp_path: Path):
        """目录不是文件，必须被拒。

        注意消息是「文件不存在: ...」，实际是「这是个目录」——
        信息量偏弱（见报告观察项），但异常类型正确。
        """
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            validate_file_path(str(tmp_path))

    def test_directory_ok_when_must_exist_false(self, tmp_path: Path):
        """must_exist=False 时目录也会放行 —— 本函数不区分文件与目录。"""
        assert Path(validate_file_path(str(tmp_path), must_exist=False)) == tmp_path

    @pytest.mark.parametrize("value", ["", "   ", "\t", "\n"])
    def test_blank_string_raises_value_error(self, value):
        """空串与纯空白串应抛 ValueError（与 FileNotFoundError 区分开）。"""
        with pytest.raises(ValueError, match="不能为空"):
            validate_file_path(value)

    def test_blank_string_raises_even_when_must_exist_false(self):
        """must_exist=False 也不能放过空串 —— 空路径无论如何都无意义。"""
        with pytest.raises(ValueError, match="不能为空"):
            validate_file_path("", must_exist=False)

    def test_non_str_input_raises_value_error(self, yaml_file: Path):
        """非字符串输入（如 pathlib.Path）会被当成空路径拒绝。

        实测行为记录：pathlib.Path 是合法且常见的路径表达，却撞上
        「路径不能为空: WindowsPath('...')」这种自相矛盾的消息。
        异常类型（ValueError）尚可接受，消息误导性较强，见报告观察项。
        """
        with pytest.raises(ValueError, match="不能为空"):
            validate_file_path(yaml_file)

    @pytest.mark.parametrize("name", ["a.txt", "b.png", "c.yaml", "d", "e.tar.gz"])
    def test_extension_is_not_checked(self, tmp_path: Path, name: str):
        """本函数只认「是个已存在的文件」，不限制扩展名。"""
        p = tmp_path / name
        p.write_text("x", encoding="utf-8")
        assert Path(validate_file_path(str(p))) == p

    def test_message_contains_absolute_path_of_missing_file(self, tmp_path: Path):
        """找不到文件时消息里应带绝对路径，便于日志排查。"""
        missing = tmp_path / "ghost.bin"
        with pytest.raises(FileNotFoundError) as exc:
            validate_file_path(str(missing))
        assert str(missing) in str(exc.value)


# ============================================================================
# validate_config
# ============================================================================

class TestValidateConfig:
    """配置字典校验。"""

    def test_real_config_passes(self, default_config: dict):
        """config/default.yaml 的真实内容必须通过校验。"""
        assert validate_config(default_config) is default_config

    def test_does_not_modify_config(self, default_config: dict):
        """校验只读：返回的配置与调用前逐字节一致。"""
        before = copy.deepcopy(default_config)
        validate_config(default_config)
        assert default_config == before

    @pytest.mark.parametrize("key", ["system", "inspection", "output"])
    def test_missing_top_level_key_raises(self, default_config: dict, key: str):
        """缺少任一必需顶层键都应报错，且消息点名是哪个键。"""
        cfg = copy.deepcopy(default_config)
        cfg.pop(key)
        with pytest.raises(ValueError, match=key):
            validate_config(cfg)

    @pytest.mark.parametrize(
        "key", ["preprocessing", "texture", "defects", "classifier", "quality"]
    )
    def test_missing_inspection_key_raises(self, default_config: dict, key: str):
        """inspection 下的五个必需子键缺一不可。"""
        cfg = copy.deepcopy(default_config)
        cfg["inspection"].pop(key)
        with pytest.raises(ValueError, match=key):
            validate_config(cfg)

    def test_empty_dict_raises(self):
        """空字典在第一关（system）就被拦下。"""
        with pytest.raises(ValueError, match="system"):
            validate_config({})

    @pytest.mark.parametrize("value", [None, [], "config", 42, 3.14, (1, 2)])
    def test_non_dict_input_raises_type_error(self, value):
        """非字典输入抛 TypeError，且消息带实际类型名。"""
        with pytest.raises(TypeError, match="dict"):
            validate_config(value)

    def test_optional_sections_are_not_required(self, default_config: dict):
        """camera / plc / roi / process_monitor 等节不是必需键。

        实测行为记录：校验层只认 3 个顶层键 + 5 个 inspection 子键，
        其余一概不管。也就是说删掉整个 camera 节照样「校验通过」。
        """
        cfg = copy.deepcopy(default_config)
        for key in ("camera", "plc", "data_processing"):
            cfg.pop(key, None)
        cfg["inspection"].pop("roi", None)
        cfg["inspection"].pop("process_monitor", None)
        assert validate_config(cfg) is cfg

    def test_inspection_none_raises_type_error(self):
        """inspection 为 None 时抛 TypeError（不是 ValueError）。

        实测行为记录：`key not in insp` 对 None 不可迭代，直接漏出
        Python 内置的 TypeError: argument of type 'NoneType' is not iterable。
        类型上说得通，但错误消息没有指明是 config.inspection，见报告观察项。
        """
        with pytest.raises(TypeError):
            validate_config({"system": {}, "inspection": None, "output": {}})

    def test_inspection_empty_list_raises_value_error(self):
        """inspection 为空列表时报「缺少键」，而不是「类型应为 dict」。"""
        with pytest.raises(ValueError, match="缺少键"):
            validate_config({"system": {}, "inspection": [], "output": {}})

    def test_legacy_config_is_accepted_unchanged(self, legacy_config: dict):
        """旧版配置（system.camera_id / system.trigger_mode / camera.trigger_source）
        能通过校验，且**原样返回、不做任何迁移**。

        实测行为记录（本次重构最关心的点）：校验层完全不看 camera 节，
        既不拒绝旧键、也不把它们翻译成新键，更不提示已废弃。
        调用方拿到旧配置后，读 camera.trigger.mode 的地方会各自走默认值 ——
        触发模式被静默降级，而不是在启动时报错。
        """
        got = validate_config(legacy_config)
        assert got is legacy_config
        assert got["system"]["camera_id"] == 2
        assert got["system"]["trigger_mode"] == "software"
        assert got["camera"]["trigger_source"] == "Line0"
        assert "device" not in got["camera"]

    def test_legacy_config_without_camera_section_passes(self, default_config: dict):
        """整节 camera 缺失（旧版配置更早的形态）同样通过。"""
        cfg = copy.deepcopy(default_config)
        del cfg["camera"]
        assert validate_config(cfg) is cfg

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda c: c["camera"].__setitem__("exposure_us", -5000),
                         id="negative_exposure"),
            pytest.param(lambda c: c["camera"].__setitem__("width", 0),
                         id="zero_width"),
            pytest.param(lambda c: c["camera"].__setitem__("height", -2048),
                         id="negative_height"),
            pytest.param(lambda c: c["camera"].__setitem__("gain", -1.0),
                         id="negative_gain"),
            pytest.param(lambda c: c["camera"].__setitem__("driver", "totally-bogus"),
                         id="unknown_driver"),
            pytest.param(lambda c: c["camera"]["trigger"].__setitem__("mode", "banana"),
                         id="invalid_trigger_mode"),
            pytest.param(lambda c: c["system"].__setitem__("resolution_mm_per_pixel", 0),
                         id="zero_resolution"),
            pytest.param(lambda c: c["system"].__setitem__("resolution_mm_per_pixel", -0.01),
                         id="negative_resolution"),
            pytest.param(lambda c: c["inspection"]["quality"].__setitem__("ok_score_threshold", 999),
                         id="out_of_range_score_threshold"),
        ],
    )
    def test_illegal_values_pass_validation(self, default_config: dict, mutate):
        """非法取值当前**全部放行** —— 校验层只查键是否存在，不查值。

        实测行为记录：负曝光、分辨率为 0、驱动名为乱码、触发模式拼错、
        判定阈值超范围，都会「校验通过」并原样返回。守门人只数门牌号，
        不看进门的车里装了什么。
        """
        cfg = copy.deepcopy(default_config)
        mutate(cfg)
        assert validate_config(cfg) is cfg

    @pytest.mark.xfail(
        reason="validate_config 的 docstring 承诺「值无效」抛 ValueError，"
               "实现只检查键存在性；camera.exposure_us = -5000 可通过校验",
        strict=False,
    )
    def test_should_reject_negative_exposure(self, default_config: dict):
        """文档承诺与实际实现不符：负曝光应被拒。"""
        cfg = copy.deepcopy(default_config)
        cfg["camera"]["exposure_us"] = -5000
        with pytest.raises(ValueError, match="exposure"):
            validate_config(cfg)

    @pytest.mark.xfail(
        reason="validate_config 不识别旧键 system.trigger_mode / camera.trigger_source，"
               "既不拒绝也不迁移，触发模式静默降级到默认值",
        strict=False,
    )
    def test_should_flag_deprecated_trigger_keys(self, legacy_config: dict):
        """废弃键存在时应报错或迁移，而不是静默接受。"""
        with pytest.raises(ValueError, match="trigger"):
            validate_config(legacy_config)

    def test_extra_unknown_keys_are_ignored(self, default_config: dict):
        """多余的未知键不会引起报错（宽松策略，便于向前兼容）。"""
        cfg = copy.deepcopy(default_config)
        cfg["system"]["totally_new_key"] = {"nested": [1, 2, 3]}
        assert validate_config(cfg) is cfg


# ============================================================================
# validate_config_path
# ============================================================================

class TestValidateConfigPath:
    """配置文件路径校验。"""

    @pytest.mark.parametrize("suffix", [".yaml", ".yml", ".YAML", ".YML", ".Yaml"])
    def test_accepts_yaml_extensions_case_insensitively(self, tmp_path: Path, suffix: str):
        """yaml / yml 扩展名大小写不敏感。"""
        p = tmp_path / f"conf{suffix}"
        p.write_text("system: {}\n", encoding="utf-8")
        assert Path(validate_config_path(str(p))) == p

    @pytest.mark.parametrize(
        "name", ["conf.json", "conf.txt", "conf", "conf.yaml.bak", "conf.yaml.txt"]
    )
    def test_rejects_non_yaml_extension(self, tmp_path: Path, name: str):
        """非 YAML 扩展名应报错，且消息里说明要求 YAML。"""
        p = tmp_path / name
        p.write_text("system: {}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML"):
            validate_config_path(str(p))

    def test_missing_yaml_file_raises_filenotfound(self, tmp_path: Path):
        """扩展名对但文件不存在 → FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            validate_config_path(str(tmp_path / "ghost.yaml"))

    def test_existing_yaml_returns_absolute_path(self, yaml_file: Path):
        """正常路径返回绝对路径。"""
        got = validate_config_path(str(yaml_file))
        assert Path(got).is_absolute()

    def test_extension_check_happens_before_existence_check(self, tmp_path: Path):
        """扩展名判断在前：扩展名不对时，即使文件不存在也报格式错。"""
        with pytest.raises(ValueError, match="YAML"):
            validate_config_path(str(tmp_path / "ghost.json"))

    def test_empty_string_reports_format_error(self):
        """空串报的是「必须是 YAML 格式」，而不是「路径不能为空」。

        实测行为记录：下划线做了 .lower()，空串在扩展名判断处就被拦下，
        调用方若按「空路径」分支处理这条异常会走错分支，见报告观察项。
        """
        with pytest.raises(ValueError, match="YAML"):
            validate_config_path("")

    @pytest.mark.xfail(
        reason="validate_config_path 对非字符串输入先做 .lower()，"
               "None 会抛 AttributeError 而非文档承诺的 ValueError",
        strict=False,
    )
    def test_none_input_raises_value_error(self):
        """None 输入应走 validate_file_path 那条 ValueError 分支。"""
        with pytest.raises(ValueError, match="路径不能为空"):
            validate_config_path(None)

    def test_directory_with_yaml_suffix_is_rejected(self, tmp_path: Path):
        """名字以 .yaml 结尾的目录仍要被拒（校验的是文件）。"""
        d = tmp_path / "dir.yaml"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            validate_config_path(str(d))


# ============================================================================
# 真实配置文件的端到端校验
# ============================================================================

class TestRealConfigFile:
    """用磁盘上真实的 config/default.yaml 走一遍完整链路。"""

    def test_real_config_path_and_content_pass(self, project_root: Path):
        """真实配置文件的路径与内容都应通过校验。"""
        path = validate_config_path(str(project_root / "config" / "default.yaml"))
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert validate_config(cfg) is cfg

    def test_real_config_uses_new_camera_keys(self, default_config: dict):
        """真实配置已按新结构组织：旧键不在，新键在位。

        这条用例把「重构结果」本身钉住 —— 若有人把 system.trigger_mode
        写回 default.yaml，测试立刻失败。
        """
        assert "camera_id" not in default_config["system"]
        assert "trigger_mode" not in default_config["system"]
        assert "trigger_source" not in default_config["camera"]
        assert "index" in default_config["camera"]["device"]
        assert "mode" in default_config["camera"]["trigger"]
