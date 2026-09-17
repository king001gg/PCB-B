"""喷砂工艺参数监测模块。

移植自电磁继电器动触簧表面缺陷识别系统的 GLCM 纹理分析与工艺判定逻辑，
将纹理特征翻译成操作工可执行的工艺调节建议（提高/降低喷砂速度）。

本模块与 core/texture.py 的 GLCMExtractor 是**两套并存的口径**，互不影响：

    core/texture.GLCMExtractor          ProcessGLCMExtractor（本模块）
    ------------------------------      ----------------------------------
    256 级灰度量化                       8 级灰度量化
    距离 [1, 3, 5]                       距离 1
    输出 12 组特征（3 距离 × 4 角度）      输出 1 组（4 个角度取平均）
    供缺陷检测 / SVM 分类器使用            供工艺面板与报警判定使用

为什么不能共用：继电器系统的报警阈值（contrast 0.3 / 0.5 / 2.0）是在 8 级量化下
标定的。8 级量化的步长为 32 个灰度级，相邻像素常相差 1 级，(i-j)² = 1；而 256 级
量化下相邻像素常相差约 10 个灰度级，(i-j)² ≈ 100。两套口径的 contrast 相差 1~2 个
数量级，阈值互换会导致永远报警或永远不报警。

数值口径四条硬规则（改动本模块前务必先读）：

    R1  灰度量化 8 级 / 距离 1 / 4 个角度取平均
    R2  只喂原始灰度，不经过 Preprocessor
        （Retinex + CLAHE 会改变灰度分布，使特征漂移）
    R3  离线与在线统一降采样至长边 resize_long_side
        （两条路径分辨率不同则同一块板算出不同的值，共用阈值将自相矛盾）
    R4  三通道输入默认按 RGB 解释，离线与在线必须传相同的 color_order
        （本项目内部约定为 RGB；R/B 权重对调会让两条路径的结果相差最大约 27%，
         详见 _to_gray）

阈值说明（重要）：
    本模块 DEFAULT_THRESHOLDS 里的 0.3 / 0.5 / 2.0 是继电器系统的原值，来自动触簧
    图像，**不适用于 PCB 喷砂图**。实测 PCB 样本的 contrast 仅 0.084 ~ 0.182，
    沿用原值会让全部板卡落入报警区间、报警灯恒亮。当前实际取值见
    config/default.yaml 的 inspection.process_monitor.thresholds（临时标定值，
    上线前必须用真实良品/不良品图像重新标定）。

参考文献:
    Haralick et al., "Textural Features for Image Classification", IEEE SMC, 1973.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from skimage.feature import graycomatrix, graycoprops
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ============================================================================
# 内置默认值（配置缺失时回退使用）
# ============================================================================

DEFAULT_LEVELS = 8
DEFAULT_DISTANCE = 1
DEFAULT_ANGLES: Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
DEFAULT_RESIZE_LONG_SIDE = 640

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "contrast_low_alarm": 0.3,   # 低于此值触发闪烁报警
    "contrast_low_warn": 0.5,    # 低于此值判「偏小」
    "contrast_high": 2.0,        # 高于此值判「偏大」
    "speed_base": 3.0,           # 建议速度公式基数
}

DEFAULT_ADVICE: Dict[str, str] = {
    "alarm": "提高速度到 {speed:.2f}",
    "low": "提高速度到 {speed:.2f}",
    "high": "降低速度到 {speed:.2f}",
    "normal": "纹理清晰，检测环境理想",
}

# 建议速度的合法区间下界。上界取 speed_base。
SPEED_MIN = 0.1

# 状态显示色（浅色主题下使用）
COLOR_ALARM = "#e74c3c"   # 红
COLOR_WARN = "#f39c12"    # 橙
COLOR_OK = "#2ecc71"      # 绿


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ProcessGLCMFeatures:
    """工艺监测口径的 6 个 GLCM 特征。

    注意 energy 与 asm 是两个独立的量，二者为平方关系：

        energy = sqrt(asm) = sqrt(sum(p²))

    core/texture.py 的 GLCMFeatures.energy 字段名虽为 energy，实际存的是
    sum(glcm²)，即本类的 asm。此处刻意不复用该命名，避免语义混淆。
    """
    contrast: float = 0.0        # 对比度
    correlation: float = 0.0     # 相关性
    energy: float = 0.0          # 能量值 = sqrt(ASM)
    dissimilarity: float = 0.0   # 差异性
    homogeneity: float = 0.0     # 同质性
    asm: float = 0.0             # ASM 值 = sum(p²)

    def as_dict(self) -> Dict[str, float]:
        """转换为字典（用于日志与报表导出）。"""
        return {
            "contrast": self.contrast,
            "correlation": self.correlation,
            "energy": self.energy,
            "dissimilarity": self.dissimilarity,
            "homogeneity": self.homogeneity,
            "asm": self.asm,
        }


@dataclass
class ProcessVerdict:
    """工艺判定结果。"""
    level: str = "正常"          # "正常" | "偏小" | "偏大" | "异常"
    suggestion: str = ""         # 维护建议文本
    alarm: bool = False          # 是否触发闪烁报警
    color: str = COLOR_OK        # 显示色（十六进制）


# ============================================================================
# 8 级 GLCM 提取器（继电器口径）
# ============================================================================

class ProcessGLCMExtractor:
    """工艺监测口径的 GLCM 特征提取器。

    与继电器系统保持数值一致：8 级量化、距离 1、4 个角度取平均。

    Attributes:
        levels: 灰度量化级数（R1，须为 8，阈值依赖此口径）。
        distance: 像素距离。
        angles: 角度列表（度），输出取其平均。
        resize_long_side: 降采样长边目标（R3）。
    """

    #: 从 GLCM 提取的特征名，顺序固定
    FEATURE_KEYS: Tuple[str, ...] = (
        "contrast", "correlation", "energy",
        "dissimilarity", "homogeneity", "asm",
    )

    def __init__(
        self,
        levels: int = DEFAULT_LEVELS,
        distance: int = DEFAULT_DISTANCE,
        angles: Optional[Sequence[float]] = None,
        resize_long_side: int = DEFAULT_RESIZE_LONG_SIDE,
    ):
        if not HAS_SKIMAGE:
            # 刻意不提供 numpy 回退实现：手写公式极易把 energy（= sqrt(ASM)）
            # 写成 asm（= sum(p²)），造成与继电器数值不可比、阈值静默失效。
            raise RuntimeError(
                "工艺监测模块依赖 scikit-image（graycomatrix / graycoprops），"
                "请先安装: pip install scikit-image"
            )

        self.levels = int(levels)
        self.distance = int(distance)
        self.angles = tuple(angles) if angles else DEFAULT_ANGLES
        self.resize_long_side = int(resize_long_side)

        self._angles_rad: List[float] = [np.deg2rad(a) for a in self.angles]

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def compute(
        self,
        img: np.ndarray,
        color_order: str = "rgb",
    ) -> ProcessGLCMFeatures:
        """计算 8 级口径的 GLCM 特征（4 个角度取平均）。

        Args:
            img: 输入图像。三通道或单通道灰度均可，内部会转为灰度（R2）并降采样（R3）。
            color_order: 三通道输入的通颜色顺序，'rgb' 或 'bgr'。
                默认 'rgb' —— 与本项目内部约定一致：main.py 与 MainWindow._set_image()
                都把图像转成 RGB 后才送入流水线，DefectDetector 也按 RGB 处理。
                传错会让 RGB2GRAY 与 BGR2GRAY 的 R/B 权重对调，彩色图上产生不同的
                灰度、进而得到不同的特征值，破坏 R3 要求的离线/在线一致性。

        Returns:
            ProcessGLCMFeatures 六个特征的平均值。
        """
        gray = self._to_gray(img, color_order)  # R2
        gray = self._resize(gray)              # R3
        quantized = self._quantize(gray)       # R1

        glcm = graycomatrix(
            quantized,
            distances=[self.distance],
            angles=self._angles_rad,
            levels=self.levels,
            symmetric=True,
            normed=True,
        )  # 形状: (levels, levels, 1, n_angles)

        props = self._props_from_glcm(glcm)  # {特征名: (n_angles,)}

        # 对角度取平均
        averaged = {
            key: float(np.mean(np.nan_to_num(
                props[key], nan=0.0, posinf=0.0, neginf=0.0
            )))
            for key in self.FEATURE_KEYS
        }
        return ProcessGLCMFeatures(**averaged)

    # ------------------------------------------------------------------
    # 特征提取
    # ------------------------------------------------------------------

    @staticmethod
    def _props_from_glcm(glcm: np.ndarray) -> Dict[str, np.ndarray]:
        """从 4 维 GLCM 提取各角度的 6 个特征。

        必须用 graycoprops 而非手写公式：其 'energy' 返回的是 sqrt(ASM)，
        手写时极易写成 sum(p²) 而静默改变数值口径。

        注意 graycoprops 要求输入为 4 维 (levels, levels, n_dist, n_angle)，
        传入二维切片会抛 ValueError —— 继电器原实现正是栽在这里（详见模块
        顶部的口径说明）。

        Returns:
            {特征名: 长度为 n_angles 的一维数组}
        """
        return {
            "contrast": graycoprops(glcm, "contrast")[0],          # (n_angles,)
            "correlation": graycoprops(glcm, "correlation")[0],
            "energy": graycoprops(glcm, "energy")[0],              # = sqrt(ASM)
            "dissimilarity": graycoprops(glcm, "dissimilarity")[0],
            "homogeneity": graycoprops(glcm, "homogeneity")[0],
            "asm": graycoprops(glcm, "ASM")[0],                    # = sum(p²)
        }

    # ------------------------------------------------------------------
    # 输入规范化
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(img: np.ndarray, color_order: str = "rgb") -> np.ndarray:
        """转灰度（R2：不做任何预处理）。

        刻意不调用 core.preprocessing.Preprocessor —— Retinex + CLAHE
        会改变灰度分布，使 GLCM 特征漂移、与继电器数值不可比。
        """
        if img is None:
            raise ValueError("输入图像不能为 None")
        if img.ndim == 2:
            return img
        if img.ndim != 3:
            raise ValueError(f"不支持的图像维度: {img.ndim}（期望 2 或 3）")

        channels = img.shape[2]
        if channels not in (3, 4):
            return img[:, :, 0]

        if color_order == "rgb":
            code = cv2.COLOR_RGB2GRAY if channels == 3 else cv2.COLOR_RGBA2GRAY
        elif color_order == "bgr":
            code = cv2.COLOR_BGR2GRAY if channels == 3 else cv2.COLOR_BGRA2GRAY
        else:
            raise ValueError(
                f"color_order 必须为 'rgb' 或 'bgr'，收到 {color_order!r}"
            )
        return cv2.cvtColor(img, code)

    def _resize(self, gray: np.ndarray) -> np.ndarray:
        """按长边降采样（R3），保证离线与在线口径一致。"""
        h, w = gray.shape[:2]
        long_side = max(h, w)
        if self.resize_long_side <= 0 or long_side <= self.resize_long_side:
            return gray

        scale = self.resize_long_side / float(long_side)
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        return cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)

    def _quantize(self, gray: np.ndarray) -> np.ndarray:
        """量化到 levels 级（R1）。与继电器实现一致：floor(gray * levels / 256)。"""
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        quantized = np.floor(
            gray.astype(np.float64) * (self.levels / 256.0)
        ).astype(np.uint8)
        return np.clip(quantized, 0, self.levels - 1)


# ============================================================================
# 工艺判定
# ============================================================================

class ProcessMonitor:
    """工艺参数监测器 —— 将 GLCM 特征翻译为工艺调节建议。

    判定完全基于 contrast 一个维度，与继电器系统一致。

    Attributes:
        enabled: 是否启用。
        extractor: GLCM 特征提取器。
        contrast_low_alarm: 低于此值触发闪烁报警。
        contrast_low_warn: 低于此值判「偏小」。
        contrast_high: 高于此值判「偏大」。
        speed_base: 建议速度公式基数。
        advice: 各状态的建议文本模板。
    """

    def __init__(self, config: Optional[dict] = None):
        """从配置构造。

        Args:
            config: 完整配置字典，读取 config["inspection"]["process_monitor"]。
                    该段整体缺失时回退到内置默认值，保证旧配置文件仍可加载。
        """
        cfg = (
            (config or {})
            .get("inspection", {})
            .get("process_monitor", {})
        ) or {}

        self.enabled = bool(cfg.get("enabled", True))

        glcm_cfg = cfg.get("glcm", {}) or {}
        self.extractor = ProcessGLCMExtractor(
            levels=glcm_cfg.get("levels", DEFAULT_LEVELS),
            distance=glcm_cfg.get("distance", DEFAULT_DISTANCE),
            angles=glcm_cfg.get("angles", DEFAULT_ANGLES),
            resize_long_side=cfg.get("resize_long_side", DEFAULT_RESIZE_LONG_SIDE),
        )

        thresholds = {**DEFAULT_THRESHOLDS, **(cfg.get("thresholds", {}) or {})}
        self.contrast_low_alarm = float(thresholds["contrast_low_alarm"])
        self.contrast_low_warn = float(thresholds["contrast_low_warn"])
        self.contrast_high = float(thresholds["contrast_high"])
        self.speed_base = float(thresholds["speed_base"])

        self.advice = {**DEFAULT_ADVICE, **(cfg.get("advice", {}) or {})}

    # ------------------------------------------------------------------
    # 判定
    # ------------------------------------------------------------------

    def evaluate(self, features: ProcessGLCMFeatures) -> ProcessVerdict:
        """根据 GLCM 特征给出工艺判定。

        Args:
            features: ProcessGLCMExtractor.compute() 的输出。

        Returns:
            ProcessVerdict 含状态、建议、报警标志与显示色。
        """
        contrast = features.contrast
        speed = self._suggest_speed(contrast)

        if contrast < self.contrast_low_alarm:
            # 修正：继电器原实现此分支显示「正常」却同时点红灯，自相矛盾，
            # 判定为笔误，此处改为「异常」。
            level, color, alarm, key = "异常", COLOR_ALARM, True, "alarm"
        elif contrast < self.contrast_low_warn:
            level, color, alarm, key = "偏小", COLOR_WARN, False, "low"
        elif contrast > self.contrast_high:
            level, color, alarm, key = "偏大", COLOR_ALARM, False, "high"
        else:
            level, color, alarm, key = "正常", COLOR_OK, False, "normal"

        return ProcessVerdict(
            level=level,
            suggestion=self._render(self.advice[key], speed),
            alarm=alarm,
            color=color,
        )

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _suggest_speed(self, contrast: float) -> float:
        """由 contrast 反推建议速度。

        修正：继电器原公式 speed_base - contrast 在 contrast ≥ speed_base 时
        产出零或负速度，无物理意义，此处钳制到 [SPEED_MIN, speed_base]。
        """
        speed = self.speed_base - contrast
        return float(max(SPEED_MIN, min(speed, self.speed_base)))

    @staticmethod
    def _render(template: str, speed: float) -> str:
        """渲染建议模板。模板不含 {speed} 占位符时原样返回。"""
        try:
            return template.format(speed=speed)
        except (KeyError, IndexError, ValueError):
            # 用户自定义模板语法有误，退回原文而非崩溃
            return template
