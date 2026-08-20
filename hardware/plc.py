"""PLC 通信模块。

通过 Modbus 协议与喷砂设备 PLC 通信，实现：
    - OK/NG 信号输出（控制分拣机构）
    - 质量分数传输（寄存器写入）
    - 工艺参数读取（可选）
    - 报警信号发送

支持 Modbus RTU（串口）和 Modbus TCP（以太网）两种模式。
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 抽象基类
# ============================================================================

class PLCClient(ABC):
    """PLC 通信抽象基类。

    封装 Modbus 读写操作，提供统一的 OK/NG 信号和
    质量分数传输接口。
    """

    def __init__(self, config: dict):
        plc_cfg = config.get("plc", {})
        self.enabled = plc_cfg.get("enabled", False)
        self.slave_id = plc_cfg.get("slave_id", 1)
        self.timeout_ms = plc_cfg.get("timeout_ms", 1000)

        # 线圈地址映射
        self.coil_ok = plc_cfg.get("coil_ok", 0)
        self.coil_ng = plc_cfg.get("coil_ng", 1)
        self.coil_alarm = plc_cfg.get("coil_alarm", 2)
        self.coil_busy = plc_cfg.get("coil_busy", 3)

        # 保持寄存器地址映射
        self.register_quality = plc_cfg.get("register_quality", 100)
        self.register_defect_count = plc_cfg.get("register_defect_count", 101)
        self.register_process_status = plc_cfg.get("register_process_status", 102)

        self._connected = False

    @abstractmethod
    def connect(self) -> bool:
        """建立 PLC 连接。"""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """断开 PLC 连接。"""
        ...

    @abstractmethod
    def write_coil(self, address: int, value: bool) -> bool:
        """写单个线圈。

        Args:
            address: 线圈地址。
            value: True = ON, False = OFF。

        Returns:
            True 如果写入成功。
        """
        ...

    @abstractmethod
    def write_register(self, address: int, value: int) -> bool:
        """写单个保持寄存器（16 位）。"""
        ...

    @abstractmethod
    def read_register(self, address: int) -> Optional[int]:
        """读单个保持寄存器。"""
        ...

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # 高级信号接口
    # ------------------------------------------------------------------

    def signal_ok(self, quality_score: float = 100.0) -> bool:
        """发送 OK 信号，同时写入质量分数。

        序列：
            1. 复位 NG 线圈
            2. 写入质量分数到保持寄存器
            3. 置位 OK 线圈
            4. 复位 Busy 线圈
        """
        if not self.enabled:
            logger.info("[PLC] OK (模拟模式)")
            return True

        try:
            self.write_coil(self.coil_ng, False)
            self.write_register(
                self.register_quality,
                int(quality_score * 10),  # 存储为 ×10 整数（如 85.6 → 856）
            )
            self.write_coil(self.coil_ok, True)
            self.write_coil(self.coil_busy, False)
            logger.info(f"[PLC] OK 信号已发送 (评分={quality_score:.1f})")
            return True
        except Exception as e:
            logger.error(f"[PLC] OK 信号发送失败: {e}")
            return False

    def signal_ng(self, quality_score: float = 0.0, defect_count: int = 0) -> bool:
        """发送 NG 信号。

        序列：
            1. 复位 OK 线圈
            2. 写入质量分数和缺陷数
            3. 置位 NG 线圈
            4. 复位 Busy 线圈
        """
        if not self.enabled:
            logger.info(f"[PLC] NG (模拟模式) 评分={quality_score:.1f}")
            return True

        try:
            self.write_coil(self.coil_ok, False)
            self.write_register(
                self.register_quality,
                int(quality_score * 10),
            )
            self.write_register(
                self.register_defect_count,
                defect_count,
            )
            self.write_coil(self.coil_ng, True)
            self.write_coil(self.coil_busy, False)
            logger.info(
                f"[PLC] NG 信号已发送 "
                f"(评分={quality_score:.1f}, 缺陷数={defect_count})"
            )
            return True
        except Exception as e:
            logger.error(f"[PLC] NG 信号发送失败: {e}")
            return False

    def signal_busy(self) -> bool:
        """发送 Busy（检测中）信号。"""
        if not self.enabled:
            return True
        try:
            self.write_coil(self.coil_ok, False)
            self.write_coil(self.coil_ng, False)
            self.write_coil(self.coil_busy, True)
            return True
        except Exception as e:
            logger.error(f"[PLC] Busy 信号发送失败: {e}")
            return False

    def signal_alarm(self, alarm_code: int = 1) -> bool:
        """发送报警信号。

        Args:
            alarm_code: 报警代码（写入过程状态寄存器）。
        """
        if not self.enabled:
            logger.info(f"[PLC] 报警 (模拟模式) 代码={alarm_code}")
            return True
        try:
            self.write_register(self.register_process_status, alarm_code)
            self.write_coil(self.coil_alarm, True)
            logger.info(f"[PLC] 报警信号已发送 (代码={alarm_code})")
            return True
        except Exception as e:
            logger.error(f"[PLC] 报警信号发送失败: {e}")
            return False

    def reset_signals(self) -> bool:
        """复位所有输出信号。"""
        if not self.enabled:
            return True
        try:
            self.write_coil(self.coil_ok, False)
            self.write_coil(self.coil_ng, False)
            self.write_coil(self.coil_alarm, False)
            self.write_coil(self.coil_busy, False)
            return True
        except Exception as e:
            logger.error(f"[PLC] 信号复位失败: {e}")
            return False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()


# ============================================================================
# Modbus RTU（串口）实现
# ============================================================================

class ModbusRTUClient(PLCClient):
    """基于 pymodbus 的 Modbus RTU 客户端。

    通过串口（RS-232/RS-485）与 PLC 通信。

    依赖：
        pip install pymodbus
    """

    def __init__(self, config: dict):
        super().__init__(config)
        plc_cfg = config.get("plc", {})
        self.port = plc_cfg.get("port", "COM1")
        self.baudrate = plc_cfg.get("baudrate", 9600)
        self._client = None

    def connect(self) -> bool:
        if not self.enabled:
            logger.info("[PLC] Modbus RTU 已禁用")
            self._connected = False
            return False

        try:
            from pymodbus.client import ModbusSerialClient

            self._client = ModbusSerialClient(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout_ms / 1000.0,
            )
            self._connected = self._client.connect()
            if self._connected:
                logger.info(
                    f"[PLC] Modbus RTU 已连接 {self.port} "
                    f"@{self.baudrate}"
                )
            else:
                logger.error(
                    f"[PLC] Modbus RTU 连接失败: {self.port}"
                )
            return self._connected
        except ImportError:
            logger.error("pymodbus 未安装。安装: pip install pymodbus")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"[PLC] 连接异常: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._connected = False
        logger.info("[PLC] Modbus RTU 已断开")

    def write_coil(self, address: int, value: bool) -> bool:
        if self._client is None:
            return False
        try:
            result = self._client.write_coil(address, value, slave=self.slave_id)
            return not result.isError()
        except Exception as e:
            logger.error(f"[PLC] 线圈写入失败 ({address}={value}): {e}")
            return False

    def write_register(self, address: int, value: int) -> bool:
        if self._client is None:
            return False
        try:
            result = self._client.write_register(
                address, value, slave=self.slave_id,
            )
            return not result.isError()
        except Exception as e:
            logger.error(f"[PLC] 寄存器写入失败 ({address}={value}): {e}")
            return False

    def read_register(self, address: int) -> Optional[int]:
        if self._client is None:
            return None
        try:
            result = self._client.read_holding_registers(
                address, 1, slave=self.slave_id,
            )
            return result.registers[0] if not result.isError() else None
        except Exception as e:
            logger.error(f"[PLC] 寄存器读取失败 ({address}): {e}")
            return None


# ============================================================================
# Modbus TCP（以太网）实现
# ============================================================================

class ModbusTCPClient(PLCClient):
    """基于 pymodbus 的 Modbus TCP 客户端。

    通过以太网与 PLC 通信（默认端口 502）。

    依赖：
        pip install pymodbus
    """

    def __init__(self, config: dict):
        super().__init__(config)
        plc_cfg = config.get("plc", {})
        self.host = plc_cfg.get("host", "192.168.1.100")
        self.tcp_port = plc_cfg.get("tcp_port", 502)
        self._client = None

    def connect(self) -> bool:
        if not self.enabled:
            logger.info("[PLC] Modbus TCP 已禁用")
            self._connected = False
            return False

        try:
            from pymodbus.client import ModbusTcpClient

            self._client = ModbusTcpClient(
                host=self.host,
                port=self.tcp_port,
                timeout=self.timeout_ms / 1000.0,
            )
            self._connected = self._client.connect()
            if self._connected:
                logger.info(
                    f"[PLC] Modbus TCP 已连接 {self.host}:{self.tcp_port}"
                )
            else:
                logger.error(
                    f"[PLC] Modbus TCP 连接失败: {self.host}"
                )
            return self._connected
        except ImportError:
            logger.error("pymodbus 未安装。安装: pip install pymodbus")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"[PLC] 连接异常: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._connected = False
        logger.info("[PLC] Modbus TCP 已断开")

    def write_coil(self, address: int, value: bool) -> bool:
        if self._client is None:
            return False
        try:
            result = self._client.write_coil(address, value, slave=self.slave_id)
            return not result.isError()
        except Exception as e:
            logger.error(f"[PLC] 线圈写入失败 ({address}={value}): {e}")
            return False

    def write_register(self, address: int, value: int) -> bool:
        if self._client is None:
            return False
        try:
            result = self._client.write_register(
                address, value, slave=self.slave_id,
            )
            return not result.isError()
        except Exception as e:
            logger.error(f"[PLC] 寄存器写入失败 ({address}={value}): {e}")
            return False

    def read_register(self, address: int) -> Optional[int]:
        if self._client is None:
            return None
        try:
            result = self._client.read_holding_registers(
                address, 1, slave=self.slave_id,
            )
            return result.registers[0] if not result.isError() else None
        except Exception as e:
            logger.error(f"[PLC] 寄存器读取失败 ({address}): {e}")
            return None


# ============================================================================
# 工厂函数
# ============================================================================

def create_plc_client(config: dict) -> PLCClient:
    """根据配置创建 PLC 客户端。

    配置键 plc.protocol:
        - "modbus_rtu" → ModbusRTUClient
        - "modbus_tcp" → ModbusTCPClient
    """
    protocol = config.get("plc", {}).get("protocol", "modbus_rtu")
    if protocol == "modbus_tcp":
        return ModbusTCPClient(config)
    else:
        return ModbusRTUClient(config)
