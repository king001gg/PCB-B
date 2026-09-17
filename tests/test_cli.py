"""命令行入口的端到端测试。

走**真实子进程**而不是 import 后调函数：`main.py` 的问题大多出在进程边界上
（退出码、stdout 编码、未捕获异常），进程内调用测不到这些。

最关键的一条是 `TestConsoleEncoding` —— 它复现并守住一个真实故障：
Windows 控制台是 GBK(cp936)，`report.summary()` 里的 ✓✗⚠ 编不出来，
`UnicodeEncodeError` 会在第一条输出处打死整个 CLI 入口。修复方式是在输出
边界加降级层（`_ConsoleSafeStream`），这里用 `PYTHONIOENCODING=gbk`
把那个环境强制复现出来。
"""

import os
import subprocess
import sys
from pathlib import Path

import cv2
import pytest
import yaml


# ============================================================================
# 子进程辅助
# ============================================================================

#: 子进程超时。冷启动要 import cv2 / sklearn / PySide6，给足余量。
TIMEOUT_S = 180


def run_cli(project_root: Path, *args, env_extra: dict = None,
            timeout: int = TIMEOUT_S) -> subprocess.CompletedProcess:
    """跑一次 `main.py`，返回已解码的 CompletedProcess。

    stdout/stderr 先按字节收，再自行解码：被测的正是编码行为本身，
    交给 subprocess 用固定 encoding 解会把要测的东西掩盖掉。

    默认**不设** PYTHONIOENCODING，让子进程用控制台的真实编码；
    要复现特定编码环境的用例通过 ``env_extra`` 显式指定。
    """
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    if env_extra:
        env.update(env_extra)

    proc = subprocess.run(
        [sys.executable, str(project_root / "main.py"), *args],
        cwd=str(project_root),
        capture_output=True,
        timeout=timeout,
        env=env,
    )
    return subprocess.CompletedProcess(
        proc.args, proc.returncode,
        decode(proc.stdout), decode(proc.stderr),
    )


def decode(raw: bytes) -> str:
    """按 Windows 控制台的真实编码解码子进程输出。

    不假定 utf-8：本机 stdout 走的是 cp936，硬解成 utf-8 会把中文变成
    乱码，断言「输出里有没有这句话」就全失效了。
    """
    for enc in ("utf-8", "gbk", "cp936"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def write_config(project_root: Path, tmp_path: Path, **camera_overrides) -> Path:
    """基于真实 default.yaml 写一份临时配置，把输出目录指到 tmp_path。

    必须改 result_dir：否则每跑一次 CLI 就往仓库的 results/ 里丢一张图。
    """
    with open(project_root / "config" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("output", {})
    cfg["output"]["result_dir"] = str(tmp_path / "cli_out")
    cfg["output"]["save_result_image"] = True
    for key, value in camera_overrides.items():
        cfg["camera"][key] = value

    path = tmp_path / "cli_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return path


@pytest.fixture
def sample_image(tmp_path: Path, surface_factory) -> Path:
    """一张合成的喷砂面图，写成 PNG 供子进程读取。"""
    img = surface_factory(seed=42)
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), img)
    return path


# ============================================================================
# 参数解析
# ============================================================================

class TestArgumentParsing:

    def test_help_exits_zero(self, project_root):
        """--help 应正常退出，不启动 GUI。"""
        result = run_cli(project_root, "--help")
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()

    def test_help_mentions_cli_option(self, project_root):
        result = run_cli(project_root, "--help")
        assert "--cli" in result.stdout
        assert "--config" in result.stdout


# ============================================================================
# 正常检测流程
# ============================================================================

class TestCliDetection:

    def test_valid_image_exits_zero(self, project_root, tmp_path, sample_image):
        cfg = write_config(project_root, tmp_path)
        result = run_cli(project_root, "--cli", str(sample_image),
                         "--config", str(cfg))
        assert result.returncode == 0, (
            f"退出码 {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    def test_reports_the_image_being_processed(self, project_root, tmp_path,
                                               sample_image):
        cfg = write_config(project_root, tmp_path)
        result = run_cli(project_root, "--cli", str(sample_image),
                         "--config", str(cfg))
        assert "处理图像" in result.stdout
        assert sample_image.name in result.stdout

    def test_prints_process_monitor_section(self, project_root, tmp_path,
                                            sample_image):
        """「实时分析数据」的 6 个 GLCM 特征都应出现在输出里。"""
        cfg = write_config(project_root, tmp_path)
        result = run_cli(project_root, "--cli", str(sample_image),
                         "--config", str(cfg))
        for label in ("对比度", "相关性", "能量值", "差异性", "同质性", "ASM值"):
            assert label in result.stdout, f"输出里缺少「{label}」"

    def test_result_image_written_to_configured_dir(self, project_root, tmp_path,
                                                    sample_image):
        """结果图必须落在配置指定的目录，不能污染仓库的 results/。"""
        cfg = write_config(project_root, tmp_path)
        run_cli(project_root, "--cli", str(sample_image), "--config", str(cfg))
        out_dir = tmp_path / "cli_out"
        assert out_dir.is_dir(), "配置的输出目录没被创建"
        assert list(out_dir.glob("result_*.jpg")), "没有生成结果图"

    def test_same_input_gives_same_verdict(self, project_root, tmp_path,
                                           sample_image):
        """同一张图跑两次，判定结论必须一致 —— 在线检测的基本要求。"""
        cfg = write_config(project_root, tmp_path)
        a = run_cli(project_root, "--cli", str(sample_image), "--config", str(cfg))
        b = run_cli(project_root, "--cli", str(sample_image), "--config", str(cfg))
        assert a.returncode == b.returncode == 0
        # 特征值是浮点，可能有末位差异；比对判定行而不是整段输出
        assert verdict_line(a.stdout) == verdict_line(b.stdout)

    @pytest.mark.xfail(reason=(
        "缺陷 #1：OpenCV 在 Windows 上用 ANSI fopen，处理不了非 ASCII 路径。"
        "cv2.imread('喷砂试样_甲.png') 返回 None（文件确实存在且是合法 PNG），"
        "于是 main.py:150 走到「无法读取图像」分支、以退出码 1 结束。"
        "同样的坑还在 ui/main_window.py:491、core/acquisition.py:129、"
        "utils/detector.py:70。修法是改用 np.fromfile + cv2.imdecode。"))
    def test_chinese_filename(self, project_root, tmp_path, surface_factory):
        """中文文件名不应该让检测失败 —— 中文产线上这是常态。"""
        img = surface_factory(seed=8)
        path = tmp_path / "喷砂试样_甲.png"
        # 刻意不走 cv2.imwrite：它本身也写不了中文路径（同一缺陷的另一面），
        # 用它落盘会让用例在「还没测到 CLI」时就挂掉，测不到真正要测的东西。
        ok, buf = cv2.imencode(".png", img)
        assert ok
        path.write_bytes(buf.tobytes())
        assert path.exists() and path.stat().st_size > 0

        cfg = write_config(project_root, tmp_path)
        result = run_cli(project_root, "--cli", str(path), "--config", str(cfg))
        assert result.returncode == 0, (
            f"中文名图像读取失败\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


class TestNonAsciiOutputPath:
    """结果图写到非 ASCII 目录时的行为 —— 同一缺陷的输出侧。"""

    @pytest.mark.xfail(reason=(
        "缺陷 #1（输出侧）：cv2.imwrite 遇到非 ASCII 路径返回 False 且不抛异常，"
        "main.py:210 没有检查返回值，结果图静默不保存，"
        "用户只在事后找文件时才发现。「静默失败」比报错更糟。"))
    def test_result_image_written_to_chinese_dir(self, project_root,
                                                 tmp_path, surface_factory):
        """输出目录含中文时，结果图也应真的写出来。"""
        img = surface_factory(seed=9)
        src = tmp_path / "src.png"
        ok, buf = cv2.imencode(".png", img)
        assert ok
        src.write_bytes(buf.tobytes())

        cfg = write_config(project_root, tmp_path)
        out_dir = tmp_path / "中文输出"
        with open(cfg, encoding="utf-8") as f:
            c = yaml.safe_load(f)
        c["output"]["result_dir"] = str(out_dir)
        with open(cfg, "w", encoding="utf-8") as f:
            yaml.safe_dump(c, f, allow_unicode=True)

        run_cli(project_root, "--cli", str(src), "--config", str(cfg))
        assert out_dir.is_dir() and list(out_dir.glob("result_*.jpg")), \
            "结果图没有写到中文输出目录（cv2.imwrite 静默失败）"


def verdict_line(stdout: str) -> str:
    """从输出里抽出判定结论那一行，用于比对。"""
    for line in stdout.splitlines():
        if "工艺判定" in line or "判定" in line:
            return line.strip()
    return ""


# ============================================================================
# 错误路径
# ============================================================================

class TestCliErrorHandling:

    def test_missing_image_exits_nonzero(self, project_root, tmp_path):
        """图像不存在应给出明确提示并非零退出，而不是抛栈。"""
        cfg = write_config(project_root, tmp_path)
        result = run_cli(project_root, "--cli", str(tmp_path / "nope.png"),
                         "--config", str(cfg))
        assert result.returncode != 0
        assert "无法读取图像" in result.stdout

    def test_missing_image_does_not_traceback(self, project_root, tmp_path):
        """走的是 sys.exit(1) 的干净路径，不该出现 Python 栈。"""
        cfg = write_config(project_root, tmp_path)
        result = run_cli(project_root, "--cli", str(tmp_path / "nope.png"),
                         "--config", str(cfg))
        assert "Traceback" not in result.stderr

    def test_missing_config_fails_loudly(self, project_root, tmp_path,
                                         sample_image):
        """配置文件不存在：可以失败，但不能假装成功。"""
        result = run_cli(project_root, "--cli", str(sample_image),
                         "--config", str(tmp_path / "nope.yaml"))
        assert result.returncode != 0

    def test_directory_as_image_fails_cleanly(self, project_root, tmp_path):
        """把目录当图像传进来，应非零退出。"""
        cfg = write_config(project_root, tmp_path)
        result = run_cli(project_root, "--cli", str(tmp_path), "--config", str(cfg))
        assert result.returncode != 0


# ============================================================================
# 控制台编码 —— 守住已修复的真实故障
# ============================================================================

class TestConsoleEncoding:
    """Windows 控制台是 GBK(cp936)，而报告里有 ✓✗⚠ 这类字符。

    修复前：`UnicodeEncodeError` 在第一条 print 处抛出，整个 CLI 入口不可用。
    修复后：输出层把编不出的字符降级成 ASCII，其余照常输出。
    """

    def test_gbk_console_does_not_crash(self, project_root, tmp_path,
                                        sample_image):
        """强制 GBK stdout —— 这是修复前必崩的场景。"""
        cfg = write_config(project_root, tmp_path)
        result = run_cli(project_root, "--cli", str(sample_image),
                         "--config", str(cfg),
                         env_extra={"PYTHONIOENCODING": "gbk"})
        assert result.returncode == 0, (
            "GBK 控制台下 CLI 崩溃了 —— 输出层降级失效\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        assert "UnicodeEncodeError" not in result.stderr

    def test_gbk_console_still_prints_chinese(self, project_root, tmp_path,
                                              sample_image):
        """降级只该影响编不出的符号，中文本身在 GBK 里是有的，必须照常显示。"""
        cfg = write_config(project_root, tmp_path)
        result = run_cli(project_root, "--cli", str(sample_image),
                         "--config", str(cfg),
                         env_extra={"PYTHONIOENCODING": "gbk"})
        assert "处理图像" in result.stdout
        assert "对比度" in result.stdout

    def test_utf8_console_also_works(self, project_root, tmp_path, sample_image):
        """UTF-8 控制台上不能因为降级层反而出问题。"""
        cfg = write_config(project_root, tmp_path)
        result = run_cli(project_root, "--cli", str(sample_image),
                         "--config", str(cfg),
                         env_extra={"PYTHONIOENCODING": "utf-8"})
        assert result.returncode == 0, result.stderr

    def test_ascii_console_does_not_crash(self, project_root, tmp_path,
                                          sample_image):
        """最严苛的情况：纯 ASCII 控制台，所有中文都编不出来。"""
        cfg = write_config(project_root, tmp_path)
        result = run_cli(project_root, "--cli", str(sample_image),
                         "--config", str(cfg),
                         env_extra={"PYTHONIOENCODING": "ascii"})
        assert result.returncode == 0, (
            f"ASCII 控制台下崩溃\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        assert "UnicodeEncodeError" not in result.stderr
