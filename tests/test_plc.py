"""PLC 通信层测试（纯 mock，不碰真实串口 / 网络）。

`hardware/plc.py` 是整个仓库里**唯一从未被执行过**的模块：没有任何模块
import 它。所以这里的测试有两个目的：

    1. 把 `PLCClient` 与配置之间的契约钉死。这个模块没被跑过，意味着
       ``coil_ok`` / ``register_quality`` 一旦和配置里写的地址对不上，
       没有任何东西会告诉你。
    2. 把它真正接进检测流程之前，先把已经存在的缺陷暴露出来。下面的
       ``xfail`` 用例全部是**已核实的缺陷**，不是猜测 —— 修好后它们会
       自己变成 XPASS，提示把标记删掉。

不依赖 pymodbus：所有传输层都换成假对象。pymodbus 在 requirements.txt
里本来就是注释掉的（可选依赖），用例必须在这两种机器上都成立。
"""

import re
import sys
import types
from pathlib import Path

import pytest

from hardware.plc import (
    ModbusRTUClient,
    ModbusTCPClient,
    PLCClient,
    create_plc_client,
)


# ============================================================================
# 测试替身
# ============================================================================

class RecordingPLC(PLCClient):
    """记录调用序列的假 PLC。

    直接继承被测的 ``PLCClient``，这样 ``signal_ok`` / ``signal_ng`` 等
    高级接口走的是**真实实现**，只是下面的传输原语被换掉了。
    """

    def __init__(self, config, results=None, raise_on=()):
        super().__init__(config)
        self.calls = []
        self.results = dict(results or {})
        self.raise_on = set(raise_on)

    def _record(self, name, *args, default=True):
        self.calls.append((name,) + args)
        if name in self.raise_on:
            raise IOError(f"{name} 故意失败")
        return self.results.get(name, default)

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def write_coil(self, address, value) -> bool:
        return self._record("write_coil", address, value)

    def write_register(self, address, value) -> bool:
        return self._record("write_register", address, value)

    def read_register(self, address):
        return self._record("read_register", address, default=1234)

    # 便捷断言用
    @property
    def coils(self):
        return [(a, v) for name, a, v in self.calls if name == "write_coil"]

    @property
    def registers(self):
        return [(a, v) for name, a, v in self.calls
                if name == "write_register"]


class FakeResponse:
    """pymodbus 响应对象的最小替身。"""

    def __init__(self, error: bool = False, registers=None):
        self._error = error
        self.registers = list(registers or [])

    def isError(self) -> bool:
        return self._error


class FakeModbusTransport:
    """pymodbus client 的替身：只记录参数、返回预设响应。"""

    def __init__(self, connect_result=True, response=None, raises=None):
        self.connect_result = connect_result
        self.response = response if response is not None else FakeResponse()
        self.raises = raises
        self.coil_writes = []
        self.register_writes = []
        self.reads = []
        self.closed = False

    def connect(self):
        return self.connect_result

    def close(self):
        self.closed = True

    def _maybe_raise(self):
        if self.raises is not None:
            raise self.raises

    def write_coil(self, address, value, slave=None):
        self._maybe_raise()
        self.coil_writes.append((address, value, slave))
        return self.response

    def write_register(self, address, value, slave=None):
        self._maybe_raise()
        self.register_writes.append((address, value, slave))
        return self.response

    def read_holding_registers(self, address, count=1, slave=None):
        self._maybe_raise()
        self.reads.append((address, count, slave))
        return self.response


# ============================================================================
# 夹具
# ============================================================================

def plc_config(**overrides) -> dict:
    """一份启用了 PLC 的最小配置，可局部覆盖。"""
    cfg = {
        "plc": {
            "enabled": True,
            "slave_id": 1,
            "timeout_ms": 1000,
            "coil_ok": 0,
            "coil_ng": 1,
            "coil_alarm": 2,
            "coil_busy": 3,
            "register_quality": 100,
            "register_defect_count": 101,
            "register_process_status": 102,
        }
    }
    cfg["plc"].update(overrides)
    return cfg


@pytest.fixture
def plc():
    """启用状态下的 RecordingPLC。"""
    return RecordingPLC(plc_config())


@pytest.fixture
def disabled_plc():
    """禁用状态（出货默认）下的 RecordingPLC。"""
    return RecordingPLC(plc_config(enabled=False))


@pytest.fixture
def fake_pymodbus(monkeypatch):
    """把 pymodbus 换成一个可注入的假包。

    返回 ``(serial_cls, tcp_cls)`` 两个记录型工厂。不这么做的话，
    「pymodbus 没装」和「pymodbus 装了」两台机器上跑出的结果不一样。
    """
    created = {}

    def make(kind):
        def factory(**kwargs):
            transport = FakeModbusTransport(
                connect_result=created.get(f"{kind}_connect", True))
            transport.kwargs = kwargs
            created[kind] = transport
            return transport
        return factory

    pkg = types.ModuleType("pymodbus")
    client_mod = types.ModuleType("pymodbus.client")
    client_mod.ModbusSerialClient = make("serial")
    client_mod.ModbusTcpClient = make("tcp")
    pkg.client = client_mod

    monkeypatch.setitem(sys.modules, "pymodbus", pkg)
    monkeypatch.setitem(sys.modules, "pymodbus.client", client_mod)
    return created


def hide_pymodbus(monkeypatch):
    """让 ``import pymodbus`` 抛 ImportError，模拟未安装。"""
    monkeypatch.setitem(sys.modules, "pymodbus", None)
    monkeypatch.setitem(sys.modules, "pymodbus.client", None)


# ============================================================================
# 配置解析 —— 地址映射必须和配置一致
# ============================================================================

class TestConfigMapping:

    def test_reads_all_addresses_from_config(self):
        client = RecordingPLC(plc_config())
        assert (client.coil_ok, client.coil_ng) == (0, 1)
        assert (client.coil_alarm, client.coil_busy) == (2, 3)
        assert client.register_quality == 100
        assert client.register_defect_count == 101
        assert client.register_process_status == 102
        assert client.slave_id == 1
        assert client.timeout_ms == 1000
        assert client.enabled is True

    def test_defaults_when_plc_section_missing(self):
        """没有 plc 节时用代码里的默认值，而不是崩掉。"""
        client = RecordingPLC({})
        assert client.enabled is False
        assert client.slave_id == 1
        assert client.timeout_ms == 1000
        assert client.coil_ok == 0
        assert client.coil_ng == 1
        assert client.coil_alarm == 2
        assert client.coil_busy == 3
        assert client.register_quality == 100
        assert client.register_defect_count == 101
        assert client.register_process_status == 102

    def test_default_yaml_addresses_match_code_defaults(self, default_config):
        """config/default.yaml 只写了 3 个地址，其余走代码默认。

        这个用例把两边的对应关系钉死：以后有人在 yaml 里改了线圈号、
        或改了代码里的默认值，这里会立刻炸 —— 而 PLC 地址错位在产线上
        的表现是「分拣机构不动作」，没有任何报错可看。
        """
        client = RecordingPLC(default_config)
        assert client.coil_ok == default_config["plc"]["coil_ok"] == 0
        assert client.coil_ng == default_config["plc"]["coil_ng"] == 1
        assert (client.register_quality
                == default_config["plc"]["register_quality"] == 100)
        # 下面这些 yaml 里没有，必须落在代码默认上
        assert client.coil_alarm == 2
        assert client.coil_busy == 3
        assert client.register_defect_count == 101
        assert client.register_process_status == 102
        assert client.slave_id == 1
        assert client.timeout_ms == 1000
        assert client.enabled is False, "出货默认必须是关闭的"

    @pytest.mark.xfail(reason=(
        "缺陷 #5：config['plc'] 存在但为 None 时（yaml 里写了空的 `plc:` 节），"
        "config.get('plc', {}) 返回的是 None 而不是 {}，"
        "紧接着的 .get() 直接 AttributeError。"
        "同项目其它地方是这么写的：camera_dialog.py 用 "
        "`self.camera_config.get('device', {}) or {}`，说明这个坑已经踩过一次。"))
    def test_empty_plc_section_falls_back_to_defaults(self, default_config):
        cfg = dict(default_config)
        cfg["plc"] = None
        client = RecordingPLC(cfg)
        assert client.enabled is False

    @pytest.mark.xfail(reason=(
        "缺陷 #5（同一处，工厂函数侧）：plc 节为空时 create_plc_client 抛 "
        "AttributeError，而不是退回 RTU 默认值。"))
    def test_factory_survives_empty_plc_section(self, default_config):
        cfg = dict(default_config)
        cfg["plc"] = None
        assert isinstance(create_plc_client(cfg), PLCClient)


# ============================================================================
# 禁用模式（出货默认）
# ============================================================================

class TestDisabledMode:
    """plc.enabled = False 时，所有信号接口都应「假装成功」且不碰传输层。

    这是刻意的设计（``模拟模式``）：没有 PLC 的机器上检测流程照样能跑完。
    """

    def test_enabled_flag_is_false(self, disabled_plc):
        assert disabled_plc.enabled is False

    def test_signal_ok_returns_true_without_touching_transport(self, disabled_plc):
        assert disabled_plc.signal_ok(95.0) is True
        assert disabled_plc.calls == [], "禁用状态下不该有任何总线操作"

    def test_signal_ng_returns_true_without_touching_transport(self, disabled_plc):
        assert disabled_plc.signal_ng(12.5, defect_count=7) is True
        assert disabled_plc.calls == []

    def test_signal_busy_returns_true_without_touching_transport(self, disabled_plc):
        assert disabled_plc.signal_busy() is True
        assert disabled_plc.calls == []

    def test_signal_alarm_returns_true_without_touching_transport(self, disabled_plc):
        assert disabled_plc.signal_alarm(3) is True
        assert disabled_plc.calls == []

    def test_reset_signals_returns_true_without_touching_transport(self, disabled_plc):
        assert disabled_plc.reset_signals() is True
        assert disabled_plc.calls == []

    def test_real_transport_connect_returns_false_when_disabled(self):
        """真实传输层在禁用时 connect() 返回 False，不建连接。

        这里有个语义不一致值得记一笔：同一个「禁用」状态，
        connect() 说失败，signal_* 说成功。调用方若拿 connect() 的返回值
        当前置条件，就永远走不到 signal_*；反过来若只看 signal_*，
        就分不清「真发了信号」和「模拟模式假装发了」。
        """
        for cls in (ModbusRTUClient, ModbusTCPClient):
            c = cls(plc_config(enabled=False))
            assert c.connect() is False
            assert c.connected is False
            assert c._client is None


# ============================================================================
# OK 信号
# ============================================================================

class TestSignalOk:

    def test_exact_sequence(self, plc):
        """确切的操作序列 —— 顺序错了分拣机构会误动作。"""
        assert plc.signal_ok(85.6) is True
        assert plc.calls == [
            ("write_coil", 1, False),        # 先复位 NG
            ("write_register", 100, 856),    # 再写质量分数
            ("write_coil", 0, True),         # 最后置位 OK
            ("write_coil", 3, False),        # 清 Busy
        ]

    @pytest.mark.parametrize("score,expected", [
        (100.0, 1000),
        (85.6, 856),
        (0.0, 0),
        (60.0, 600),
        (99.9, 999),
    ])
    def test_quality_score_encoded_times_ten(self, score, expected):
        """寄存器里存的是 ×10 整数 —— 现场 PLC 按这个口径解释。"""
        plc = RecordingPLC(plc_config())
        plc.signal_ok(score)
        assert plc.registers == [(100, expected)]

    def test_score_is_truncated_not_rounded(self):
        """85.67 → 856（截断）。已知口径，写在这里免得以后被"修"成四舍五入。"""
        plc = RecordingPLC(plc_config())
        plc.signal_ok(85.67)
        assert plc.registers == [(100, 856)]

    def test_ok_signal_is_latched_after_data(self, plc):
        """OK 线圈必须是最后一个动作：先置位再写分数，分拣机构会读到上一片的分数。"""
        plc.signal_ok(77.0)
        score_at = plc.calls.index(("write_register", 100, 770))
        ok_at = plc.calls.index(("write_coil", 0, True))
        assert ok_at > score_at, (
            f"OK 线圈在第 {ok_at} 步就置位了，而分数在第 {score_at} 步才写 —— "
            "PLC 会在分数还没更新时就读到 OK")
        assert plc.calls[-1] == ("write_coil", 3, False)

    def test_returns_false_when_transport_raises(self):
        """传输层抛异常时返回 False（这是唯一被处理的失败路径）。"""
        plc = RecordingPLC(plc_config(), raise_on={"write_coil"})
        assert plc.signal_ok(90.0) is False

    @pytest.mark.xfail(reason=(
        "缺陷 #4：write_coil / write_register 把异常吞成返回值 False，"
        "而 signal_ok 完全丢弃这些返回值、只在抛异常时才返回 False。"
        "两者叠加的结果是「写失败」永远表现为「成功」："
        "线圈没动、寄存器没写，函数返回 True，日志还打一行「OK 信号已发送」，"
        "调用方无从察觉。产线上这意味着不良品被判 OK 后直接流入下一工序。"))
    def test_returns_false_when_writes_fail(self):
        plc = RecordingPLC(plc_config(), results={
            "write_coil": False, "write_register": False})
        assert plc.signal_ok(90.0) is False

    @pytest.mark.xfail(reason=(
        "缺陷 #4（更常见的一面）：从未 connect() 过，_client 为 None，"
        "write_coil 静默返回 False —— signal_ok 照样返回 True，"
        "并打日志说信号已发送。"))
    def test_returns_false_when_never_connected(self):
        """没连上 PLC 就发信号，必须失败而不是谎报成功。"""
        plc = RecordingPLC(plc_config())
        # 不调用 connect()，直接发信号
        assert plc.signal_ok(90.0) is False

    def test_nan_score_returns_false_and_leaves_partial_state(self):
        """NaN 评分：int(nan*10) 抛 ValueError，被兜住返回 False。

        但此时 NG 线圈已经被复位了，而 OK 线圈还没置位 —— PLC 停在
        「既非 OK 也非 NG」的中间态。这是真实风险：上游算法给出 NaN
        并不罕见（除零、空 ROI）。
        """
        plc = RecordingPLC(plc_config())
        assert plc.signal_ok(float("nan")) is False
        assert plc.calls == [("write_coil", 1, False)]
        assert ("write_coil", 0, True) not in plc.calls


# ============================================================================
# NG 信号
# ============================================================================

class TestSignalNg:

    def test_exact_sequence(self, plc):
        assert plc.signal_ng(12.5, defect_count=7) is True
        assert plc.calls == [
            ("write_coil", 0, False),        # 先复位 OK
            ("write_register", 100, 125),    # 质量分数 ×10
            ("write_register", 101, 7),      # 缺陷数
            ("write_coil", 1, True),         # 置位 NG
            ("write_coil", 3, False),        # 清 Busy
        ]

    def test_defect_count_defaults_to_zero(self, plc):
        plc.signal_ng(30.0)
        assert plc.registers == [(100, 300), (101, 0)]

    def test_ok_and_ng_are_mutually_exclusive(self, plc):
        """一次信号里 OK 与 NG 不能同时为 True。"""
        plc.signal_ng(10.0, defect_count=3)
        ok_vals = [v for a, v in plc.coils if a == 0]
        ng_vals = [v for a, v in plc.coils if a == 1]
        assert ok_vals == [False]
        assert ng_vals == [True]

    def test_returns_false_when_transport_raises(self):
        plc = RecordingPLC(plc_config(), raise_on={"write_register"})
        assert plc.signal_ng(10.0, 5) is False

    @pytest.mark.xfail(reason="缺陷 #4（同上）：写失败被静默吞掉，NG 信号谎报成功。")
    def test_returns_false_when_writes_fail(self):
        plc = RecordingPLC(plc_config(), results={
            "write_coil": False, "write_register": False})
        assert plc.signal_ng(10.0, 5) is False

    def test_quality_register_is_written_before_ng_coil(self, plc):
        """NG 线圈置位前，分数必须已经就位。"""
        plc.signal_ng(42.0, 9)
        assert plc.calls.index(("write_register", 100, 420)) < \
            plc.calls.index(("write_coil", 1, True))


# ============================================================================
# Busy / Alarm / Reset
# ============================================================================

class TestBusyAlarmReset:

    def test_busy_clears_both_result_coils(self, plc):
        assert plc.signal_busy() is True
        assert plc.calls == [
            ("write_coil", 0, False),
            ("write_coil", 1, False),
            ("write_coil", 3, True),
        ]

    def test_busy_does_not_touch_alarm(self, plc):
        plc.signal_busy()
        assert all(a != 2 for a, _ in plc.coils)

    def test_alarm_writes_code_and_sets_coil(self, plc):
        assert plc.signal_alarm(7) is True
        assert plc.calls == [
            ("write_register", 102, 7),
            ("write_coil", 2, True),
        ]

    def test_alarm_default_code_is_one(self, plc):
        plc.signal_alarm()
        assert plc.registers == [(102, 1)]

    def test_alarm_leaves_busy_untouched(self, plc):
        """报警时不清 Busy —— 报警期间设备仍在运行，这是刻意的。"""
        plc.signal_alarm(2)
        assert all(a != 3 for a, _ in plc.coils)

    def test_reset_clears_all_four_coils(self, plc):
        assert plc.reset_signals() is True
        assert plc.calls == [
            ("write_coil", 0, False),
            ("write_coil", 1, False),
            ("write_coil", 2, False),
            ("write_coil", 3, False),
        ]

    def test_reset_does_not_touch_registers(self, plc):
        plc.reset_signals()
        assert plc.registers == []

    def test_reset_returns_false_when_transport_raises(self):
        plc = RecordingPLC(plc_config(), raise_on={"write_coil"})
        assert plc.reset_signals() is False

    @pytest.mark.xfail(reason="缺陷 #4（同上）：复位失败谎报成功，线圈停在原位。")
    def test_reset_returns_false_when_writes_fail(self):
        plc = RecordingPLC(plc_config(), results={"write_coil": False})
        assert plc.reset_signals() is False


# ============================================================================
# 上下文管理器
# ============================================================================

class TestContextManager:

    def test_enter_connects_and_returns_self(self):
        plc = RecordingPLC(plc_config())
        with plc as entered:
            assert entered is plc
            assert plc.connected is True

    def test_exit_disconnects(self):
        plc = RecordingPLC(plc_config())
        with plc:
            pass
        assert plc.connected is False

    def test_exit_disconnects_on_exception(self):
        plc = RecordingPLC(plc_config())
        with pytest.raises(ValueError):
            with plc:
                raise ValueError("检测过程中炸了")
        assert plc.connected is False, "异常路径也必须放开 PLC"


# ============================================================================
# Modbus RTU / TCP 传输层
# ============================================================================

class TestModbusTransport:

    def test_rtu_reads_port_and_baudrate(self):
        c = ModbusRTUClient(plc_config(port="COM3", baudrate=19200))
        assert c.port == "COM3"
        assert c.baudrate == 19200

    def test_rtu_defaults(self):
        c = ModbusRTUClient(plc_config())
        assert c.port == "COM1"
        assert c.baudrate == 9600

    def test_tcp_reads_host_and_port(self):
        c = ModbusTCPClient(plc_config(host="10.0.0.5", tcp_port=5020))
        assert c.host == "10.0.0.5"
        assert c.tcp_port == 5020

    def test_tcp_defaults(self):
        c = ModbusTCPClient(plc_config())
        assert c.host == "192.168.1.100"
        assert c.tcp_port == 502

    def test_timeout_converted_from_ms_to_seconds(self, fake_pymodbus):
        """配置里是毫秒，pymodbus 要秒 —— 差 1000 倍，接错会立刻超时。"""
        c = ModbusRTUClient(plc_config(timeout_ms=2500))
        assert c.connect() is True
        assert fake_pymodbus["serial"].kwargs["timeout"] == 2.5

    def test_connect_when_disabled_does_not_create_transport(self):
        c = ModbusRTUClient(plc_config(enabled=False))
        assert c.connect() is False
        assert c._client is None
        assert c.connected is False

    def test_connect_without_pymodbus_returns_false(self, monkeypatch):
        """pymodbus 是可选依赖，没装时必须是「连接失败」而不是抛异常。"""
        hide_pymodbus(monkeypatch)
        for cls in (ModbusRTUClient, ModbusTCPClient):
            c = cls(plc_config())
            assert c.connect() is False
            assert c.connected is False

    def test_connect_success_sets_connected(self, fake_pymodbus):
        c = ModbusRTUClient(plc_config())
        assert c.connect() is True
        assert c.connected is True

    def test_connect_failure_leaves_disconnected(self, fake_pymodbus):
        """底层 connect() 返回 False（对端不在）时不能谎报连接成功。"""
        fake_pymodbus["serial_connect"] = False
        c = ModbusRTUClient(plc_config())
        assert c.connect() is False
        assert c.connected is False

    def test_connect_transport_exception_returns_false(self, fake_pymodbus,
                                                       monkeypatch):
        """构造 client 时抛异常（串口被占用）不能往上冒。"""
        def boom(**kwargs):
            raise OSError("COM1 已被占用")
        fake_pymodbus_mod = sys.modules["pymodbus.client"]
        fake_pymodbus_mod.ModbusSerialClient = boom
        c = ModbusRTUClient(plc_config())
        assert c.connect() is False
        assert c.connected is False

    def test_disconnect_closes_and_clears(self, fake_pymodbus):
        c = ModbusRTUClient(plc_config())
        c.connect()
        transport = c._client
        c.disconnect()
        assert transport.closed is True
        assert c._client is None
        assert c.connected is False

    def test_disconnect_is_idempotent(self, fake_pymodbus):
        """重复断开不能炸：上层会在 finally 里无脑调一次。"""
        c = ModbusRTUClient(plc_config())
        c.disconnect()
        c.disconnect()
        assert c._client is None

    def test_tcp_disconnect_clears_client(self, fake_pymodbus):
        c = ModbusTCPClient(plc_config())
        c.connect()
        transport = c._client
        c.disconnect()
        assert transport.closed is True
        assert c._client is None


# ============================================================================
# 读写原语
# ============================================================================

class TestPrimitives:

    @pytest.mark.parametrize("cls", [ModbusRTUClient, ModbusTCPClient])
    def test_write_coil_without_client_returns_false(self, cls):
        """没有 client 时必须返回 False 而不是抛 —— 但见缺陷 #4。"""
        c = cls(plc_config())
        assert c.write_coil(0, True) is False

    @pytest.mark.parametrize("cls", [ModbusRTUClient, ModbusTCPClient])
    def test_write_register_without_client_returns_false(self, cls):
        c = cls(plc_config())
        assert c.write_register(100, 500) is False

    @pytest.mark.parametrize("cls", [ModbusRTUClient, ModbusTCPClient])
    def test_read_register_without_client_returns_none(self, cls):
        c = cls(plc_config())
        assert c.read_register(100) is None

    @pytest.mark.parametrize("cls", [ModbusRTUClient, ModbusTCPClient])
    def test_write_coil_passes_slave_id(self, cls, fake_pymodbus):
        """从站地址必须带上 —— 总线上有多台设备时写错站等于写错设备。"""
        c = cls(plc_config(slave_id=9))
        c.connect()
        assert c.write_coil(4, True) is True
        assert c._client.coil_writes == [(4, True, 9)]

    @pytest.mark.parametrize("cls", [ModbusRTUClient, ModbusTCPClient])
    def test_write_register_passes_slave_id(self, cls, fake_pymodbus):
        c = cls(plc_config(slave_id=5))
        c.connect()
        assert c.write_register(100, 856) is True
        assert c._client.register_writes == [(100, 856, 5)]

    @pytest.mark.parametrize("cls", [ModbusRTUClient, ModbusTCPClient])
    def test_error_response_yields_false(self, cls, fake_pymodbus):
        c = cls(plc_config())
        c.connect()
        c._client.response = FakeResponse(error=True)
        assert c.write_coil(0, True) is False
        assert c.write_register(100, 1) is False

    @pytest.mark.parametrize("cls", [ModbusRTUClient, ModbusTCPClient])
    def test_transport_exception_yields_false(self, cls, fake_pymodbus):
        """总线异常（断线、超时）被吞成 False —— 这是缺陷 #4 的源头。"""
        c = cls(plc_config())
        c.connect()
        c._client.raises = IOError("总线超时")
        assert c.write_coil(0, True) is False
        assert c.write_register(100, 1) is False
        assert c.read_register(100) is None

    @pytest.mark.parametrize("cls", [ModbusRTUClient, ModbusTCPClient])
    def test_read_register_returns_first_register(self, cls, fake_pymodbus):
        c = cls(plc_config())
        c.connect()
        c._client.response = FakeResponse(registers=[4321])
        assert c.read_register(100) == 4321
        assert c._client.reads == [(100, 1, 1)]

    @pytest.mark.parametrize("cls", [ModbusRTUClient, ModbusTCPClient])
    def test_read_register_error_response_returns_none(self, cls, fake_pymodbus):
        c = cls(plc_config())
        c.connect()
        c._client.response = FakeResponse(error=True, registers=[1])
        assert c.read_register(100) is None

    @pytest.mark.parametrize("cls", [ModbusRTUClient, ModbusTCPClient])
    def test_read_register_empty_response_returns_none(self, cls, fake_pymodbus):
        """响应没报错但寄存器列表是空的 —— 不能 IndexError 冒到上层。"""
        c = cls(plc_config())
        c.connect()
        c._client.response = FakeResponse(registers=[])
        assert c.read_register(100) is None

    def test_rtu_timeout_used_for_client(self, fake_pymodbus):
        ModbusRTUClient(plc_config(timeout_ms=500)).connect()
        assert fake_pymodbus["serial"].kwargs["timeout"] == 0.5

    def test_tcp_timeout_used_for_client(self, fake_pymodbus):
        ModbusTCPClient(plc_config(timeout_ms=3000)).connect()
        assert fake_pymodbus["tcp"].kwargs["timeout"] == 3.0


# ============================================================================
# 工厂函数
# ============================================================================

class TestFactory:

    def test_tcp_protocol(self):
        assert isinstance(create_plc_client(plc_config(protocol="modbus_tcp")),
                          ModbusTCPClient)

    def test_rtu_protocol(self):
        assert isinstance(create_plc_client(plc_config(protocol="modbus_rtu")),
                          ModbusRTUClient)

    def test_missing_protocol_defaults_to_rtu(self):
        assert isinstance(create_plc_client(plc_config()), ModbusRTUClient)

    def test_unknown_protocol_falls_back_to_rtu(self):
        """拼错协议名时退回 RTU，不抛异常 —— 静默退回值得商榷，但先钉住行为。"""
        assert isinstance(create_plc_client(plc_config(protocol="modbus_zzz")),
                          ModbusRTUClient)

    def test_factory_reads_real_config(self, default_config):
        """用真实 default.yaml 走一遍：协议是 modbus_rtu。"""
        assert isinstance(create_plc_client(default_config), ModbusRTUClient)

    def test_factory_result_is_disabled_by_default(self, default_config):
        c = create_plc_client(default_config)
        assert c.enabled is False
        assert c.connect() is False


# ============================================================================
# 缺陷 #6 —— 模块整体未接入检测流程
# ============================================================================

#: 生产代码目录（不含 tests / tools / scripts）
_PRODUCTION_ROOTS = ("core", "ui", "utils", "hardware")

_PLC_REF = re.compile(
    r"(from\s+hardware\.plc\s+import|import\s+hardware\.plc|"
    r"from\s+\.plc\s+import|create_plc_client|PLCClient)")


def find_plc_references(project_root: Path):
    """扫描生产代码里对 PLC 模块的引用，返回 {文件: 命中行}。"""
    hits = {}
    targets = [project_root / "main.py"]
    for name in _PRODUCTION_ROOTS:
        targets.extend((project_root / name).rglob("*.py"))

    for path in targets:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        found = [ln.strip() for ln in text.splitlines()
                 if _PLC_REF.search(ln)]
        if found:
            # as_posix: Windows 上 relative_to 给的是反斜杠，
            # 会让下面「排除模块自身」的比较静默失配
            hits[path.relative_to(project_root).as_posix()] = found
    return hits


class TestPlcIsWired:

    def test_scanner_finds_the_module_itself(self, project_root):
        """先验证扫描器本身有效：它必须能看见 hardware/plc.py 里的定义。

        否则「没找到引用」可能只是正则写错了，而不是真的没接。
        """
        hits = find_plc_references(project_root)
        assert "hardware/plc.py" in hits, (
            f"扫描器失效，只在这些文件里找到了引用: {sorted(hits)}")
        assert any("create_plc_client" in ln for ln in hits["hardware/plc.py"])

    def test_scanner_can_see_a_reference(self, project_root, tmp_path):
        """再验证正反两面：往 core/ 里放一个引用，扫描器必须报出来。"""
        probe = project_root / "core" / "_plc_probe_tmp.py"
        probe.write_text("from hardware.plc import create_plc_client\n",
                         encoding="utf-8")
        try:
            hits = find_plc_references(project_root)
            assert "core/_plc_probe_tmp.py" in hits
        finally:
            probe.unlink()

    @pytest.mark.xfail(reason=(
        "缺陷 #6：hardware/plc.py（411 行）在生产代码里**零引用**。"
        "没有任何模块 import 它，config['plc'] 也没有任何代码读。"
        "后果：把 config/default.yaml 里的 plc.enabled 改成 true，"
        "系统行为完全不变 —— 分选机构永不动作，而且不报错、不打日志。"
        "现场把「没接 PLC」误认为「PLC 配置好了」是很容易发生的。"
        "要么把 PLC 接进检测流程（检测完调用 signal_ok/signal_ng），"
        "要么把模块和配置节一起删掉，不要留着一个看起来能用的空壳。"))
    def test_pipeline_references_plc(self, project_root):
        """检测流程里必须真的用到 PLC —— 否则 plc 配置节是摆设。"""
        hits = find_plc_references(project_root)
        consumers = {k: v for k, v in hits.items() if k != "hardware/plc.py"}
        assert consumers, (
            "PLC 模块没有任何消费方：config 里的 plc 节改了也不会生效。\n"
            f"全部命中（含模块自身）: {sorted(hits)}")
