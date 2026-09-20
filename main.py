"""PCB 阻焊前喷砂质量在线检测系统 — 启动入口。

用法：
    python main.py                          # 使用默认配置启动 GUI
    python main.py --config config/custom.yaml  # 使用自定义配置
    python main.py --cli image.jpg           # 命令行模式（单张图像检测）
"""

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="PCB 阻焊前喷砂质量在线检测系统"
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="配置文件路径（默认: config/default.yaml）"
    )
    parser.add_argument(
        "--cli", default=None,
        help="命令行模式：指定单张图像的路径进行检测"
    )
    args = parser.parse_args()

    config_path = str(PROJECT_ROOT / args.config) \
        if not os.path.isabs(args.config) else args.config

    if args.cli:
        # 命令行模式
        run_cli(args.cli, config_path)
    else:
        # GUI 模式
        run_gui(config_path)


def run_gui(config_path: str):
    """启动 PySide6 图形界面。"""
    from PySide6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QMainWindow { background: #f0f0f0; }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ccc;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            padding: 6px 14px;
            border-radius: 3px;
            border: 1px solid #bbb;
            background: #fff;
        }
        QPushButton:hover { background: #e0e0e0; }
        QPushButton:pressed { background: #d0d0d0; }
        QTextEdit {
            border: 1px solid #ccc;
            border-radius: 3px;
        }
        QProgressBar {
            border: 1px solid #ccc;
            border-radius: 3px;
            text-align: center;
        }
        QProgressBar::chunk {
            background: #3498db;
            border-radius: 2px;
        }
    """)

    window = MainWindow(config_path)
    window.show()
    sys.exit(app.exec())


class _ConsoleSafeStream:
    """给控制台输出套一层字符降级。

    Windows 控制台的默认编码是 GBK，而检测报告里的 ✓ ✗ ⚠ ² 都不在 GBK
    字符集里。直接 print 会抛 UnicodeEncodeError —— 之前 CLI 模式第一条
    输出 `report.summary()` 就崩，整个命令行入口不可用。

    这里在写出前把已知符号换成 ASCII 等价物，换不掉的再交给
    errors="replace" 兜底，保证结果一定打得出来。
    只包 CLI 路径：GUI 与写出的报告文件仍用原字符，不受影响。
    """

    _FALLBACKS = (
        ("✓", "[v]"), ("✗", "[x]"), ("⚠", "[!]"), ("✅", "[v]"),
        ("❌", "[x]"), ("²", "2"), ("→", "->"), ("×", "x"),
    )

    def __init__(self, stream):
        self._stream = stream
        # 控制台编码；取不到时按 UTF-8 处理（多数 CI/管道场景）
        self.encoding = getattr(stream, "encoding", None) or "utf-8"

    def write(self, text: str):
        try:
            text.encode(self.encoding)
        except (UnicodeEncodeError, LookupError):
            for bad, good in self._FALLBACKS:
                text = text.replace(bad, good)
            text = text.encode(self.encoding, "replace") \
                       .decode(self.encoding, "replace")
        return self._stream.write(text)

    def __getattr__(self, name):
        # flush / isatty / fileno 等一律透传给真正的流
        return getattr(self._stream, name)


def run_cli(image_path: str, config_path: str):
    """命令行模式：对单张图像执行检测并输出结果。"""
    import cv2
    import yaml
    import numpy as np

    # 见 _ConsoleSafeStream 的说明：不加这层，GBK 控制台上第一条输出就崩
    if not isinstance(sys.stdout, _ConsoleSafeStream):
        sys.stdout = _ConsoleSafeStream(sys.stdout)

    from core.preprocessing import Preprocessor
    from core.texture import TextureAnalyzer
    from core.defects import DefectDetector
    from core.quality import QualityAssessor
    from core.process_monitor import ProcessMonitor

    # 加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        sys.exit(1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 初始化模块
    preprocessor = Preprocessor(config)
    texture_analyzer = TextureAnalyzer(config)
    defect_detector = DefectDetector(config)
    quality_assessor = QualityAssessor(config)

    # 执行检测管道
    print(f"处理图像: {image_path}")
    gray = preprocessor.process(image_rgb)
    texture_vec = texture_analyzer.analyze(gray)
    cv_heatmap = texture_analyzer.compute_cv_heatmap(gray)
    dci = texture_analyzer.direction_consistency(
        gray, texture_vec.gabor_orientation_energies
    )
    defects = defect_detector.detect_all(image_rgb, texture_vec)
    report = quality_assessor.assess(defects, cv_heatmap, dci)

    # 输出结果
    print(report.summary())
    if defects:
        print(f"\n缺陷列表 ({len(defects)} 处):")
        for d in defects:
            print(f"  [{d.type}] 面积={d.area_mm2:.4f}mm2 "
                  f"位置=({d.bbox[0]},{d.bbox[1]}→{d.bbox[2]},{d.bbox[3]}) "
                  f"严重度={d.severity:.2f}")
    if report.warnings:
        print("\n预警:")
        for w in report.warnings:
            print(f"  [!] {w}")

    # 工艺参数监测（8 级口径 GLCM，与上面的 texture_vec 是两套独立口径）。
    # 注意用 image_rgb 而非 preprocessor 的输出：Retinex + CLAHE 会改变灰度
    # 分布使特征漂移，详见 core/process_monitor.py 的 R2。
    process_monitor = ProcessMonitor(config)
    if process_monitor.enabled:
        try:
            features = process_monitor.extractor.compute(image_rgb)
            verdict = process_monitor.evaluate(features)
            print("\n工艺参数监测（实时分析数据）:")
            for key, label in (("contrast", "对比度"), ("correlation", "相关性"),
                               ("energy", "能量值"), ("dissimilarity", "差异性"),
                               ("homogeneity", "同质性"), ("asm", "ASM值")):
                print(f"  {label}: {getattr(features, key):.4f}")
            print(f"  工艺判定: {verdict.level}")
            if verdict.alarm:
                print("  报警状态: [!] 报警")
            print(f"  智能维护建议: {verdict.suggestion}")
        except Exception as e:
            print(f"\n工艺参数监测失败，已跳过: {e}")

    # 保存标注结果
    if config.get("output", {}).get("save_result_image", True):
        output_dir = Path(config.get("output", {}).get("result_dir", "results"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"result_{Path(image_path).stem}.jpg"
        annotated = defect_detector.draw_defects(image_rgb, defects)
        cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"\n结果图像已保存: {output_path}")


if __name__ == "__main__":
    main()