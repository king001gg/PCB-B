"""相机采集工作线程。

把取流从 GUI 线程挪出来。原因：GenICam 相机的 fetch() 是带超时的阻塞调用，
留在 GUI 线程会直接卡死界面；OpenCV 的 read() 虽然通常很快，但相机拔线、
带宽不足时会长时间阻塞，同样会冻结界面。因此两种后端统一走本线程。

线程边界上的两个约定：
    1. 相机在本线程内 open()/release()，GUI 线程不碰设备句柄。
    2. 本线程只吐 **RGB 三通道** 帧。BGR→RGB 的转换只在这里做一次，
       下游（图像显示、工艺监测、检测流程）一律按 RGB 处理。
       之前转换散落在多处，已经因此出过一次红蓝互换的 bug。
"""

import time
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from hardware.camera import CameraBase


#: 取帧失败时的退避时间（秒）。避免相机掉线后空转烧 CPU。
RETRY_BACKOFF_S = 0.05


class CameraGrabWorker(QThread):
    """持续取流并逐帧投递到 GUI 线程。

    Usage:
        worker = CameraGrabWorker(camera)
        worker.frame_ready.connect(on_frame)      # 参数为 RGB 三通道 ndarray
        worker.failed.connect(on_error)           # 参数为错误描述
        worker.opened.connect(on_opened)          # 相机打开成功
        worker.start()
        ...
        worker.stop()          # 请求停止
        worker.wait(3000)      # 等待线程真正退出

    停止不是瞬时的：acquire() 可能正阻塞在超时等待中，最坏要等一个
    acquire 超时周期。调用方应 wait() 后再释放相机。
    """

    #: 一帧就绪（RGB 三通道）
    frame_ready = Signal(np.ndarray)
    #: 相机打开成功
    opened = Signal()
    #: 发生错误（打开失败、连续取帧失败等），参数为可展示的中文描述
    failed = Signal(str)
    #: 周期性状态（帧率、取帧失败计数），供状态栏显示
    stats = Signal(float, int)

    def __init__(self, camera: CameraBase, parent=None):
        """
        Args:
            camera: 已构造但尚未 open() 的相机实例。
            parent: Qt 父对象。
        """
        super().__init__(parent)
        self._camera = camera
        self._running = False
        self._gui_busy = False
        self._frames = 0
        self._failures = 0
        self._consecutive_failures = 0

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """请求停止采集。返回后线程可能还在收尾，需 wait() 确认。"""
        self._running = False

    def notify_gui_busy(self, busy: bool) -> None:
        """告知本线程 GUI 是否还在处理上一帧。

        GUI 处理不过来时跳过投递，避免帧在信号队列里堆积 —— 12MP 的
        Mono8 帧每帧 5MB，堆积几帧就会吃掉大量内存。

        这是**握手**而非状态查询：``run()`` 每投出一帧就把标志置 True，
        由 GUI 在处理完那一帧后调 ``notify_gui_busy(False)`` 解除。
        在途帧因此恒定为 1。

        不能反过来由 GUI 在槽函数首尾置位 —— 槽函数在事件循环里是原子
        执行的，置位期间 Qt 根本不会派发新事件，标志永远是 False，
        拦不住已经排进队列的帧。

        用普通 bool 而非锁：最坏情况是多投或漏投一帧，没有正确性问题。
        """
        self._gui_busy = busy

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # 线程主体
    # ------------------------------------------------------------------

    def run(self) -> None:
        """采集主循环。"""
        self._running = True
        self._frames = 0
        self._failures = 0
        self._consecutive_failures = 0
        # 上一轮若在「已投递未确认」状态被停止，标志会留在 True 上，
        # 不复位则重启后一帧都投不出来
        self._gui_busy = False

        try:
            if not self._camera.open():
                # 后端的失败原因（如「序列号 XXX 不存在」）比通用提示有用得多，
                # 优先显示它 —— 否则用户按通用提示查一圈也找不到问题。
                detail = getattr(self._camera, "last_error", "")
                msg = "相机打开失败。"
                if detail:
                    msg += f"\n\n原因：{detail}"
                msg += ("\n\n请检查：相机是否上电、网线是否插好、"
                        "是否被其它程序占用。\n"
                        "可运行 tools/check_camera.py 查看详细原因。")
                self.failed.emit(msg)
                return
        except Exception as e:
            self.failed.emit(f"相机初始化异常：{e}")
            return

        self.opened.emit()
        stats_t0 = time.perf_counter()

        try:
            while self._running:
                frame = self._acquire_safely()

                if frame is None:
                    # 连续失败到一定次数说明相机掉了，不必再空转
                    if self._consecutive_failures >= 30:
                        self.failed.emit(
                            "连续 30 次取帧失败，已停止预览。\n"
                            "请检查相机连接与网络带宽。"
                        )
                        break
                    time.sleep(RETRY_BACKOFF_S)
                else:
                    self._consecutive_failures = 0
                    if not self._gui_busy:
                        self.frame_ready.emit(frame)
                        # 投出即置忙，等 GUI 处理完回调 notify_gui_busy(False)
                        # 解除。详见该方法的说明。
                        self._gui_busy = True

                # 每秒报一次状态
                now = time.perf_counter()
                if now - stats_t0 >= 1.0:
                    fps = self._frames / (now - stats_t0)
                    self.stats.emit(fps, self._failures)
                    self._frames = 0
                    self._failures = 0
                    stats_t0 = now
        finally:
            # 无论正常退出还是异常，都在本线程内释放设备
            try:
                self._camera.release()
            except Exception:
                pass
            self._running = False

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _acquire_safely(self) -> Optional[np.ndarray]:
        """取一帧并转成 RGB 三通道。失败返回 None。"""
        try:
            frame = self._camera.acquire()
        except Exception:
            self._failures += 1
            self._consecutive_failures += 1
            return None

        if frame is None:
            self._failures += 1
            self._consecutive_failures += 1
            return None

        self._frames += 1
        return self._to_rgb(frame)

    @staticmethod
    def _to_rgb(frame: np.ndarray) -> np.ndarray:
        """把相机输出统一转成 RGB 三通道。

        CameraBase 的契约是「BGR 三通道或灰度单通道」。灰度扩成三通道
        再交给下游，这样 current_raw_image 与离线路径的形状一致，
        图像显示与工艺监测都不必再判断通道数。
        """
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
