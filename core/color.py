"""色度 / 饱和度指标（监测用，不进总分）。

为什么需要它
------------
现有 `oxidation_percentage` 是**面积法**：HSV 阈值 → 连通域 → 面积占比。
它有结构性盲区——**均匀的薄氧化膜不产生连通域**，面积比接近 0，会被判为良品；
而这层膜恰恰影响后续焊接 / 层压的附着力。色相偏移 ΔH 与饱和度下降 ΔS 是
**全局面统计量**，对均匀变色敏感、不需要连通域，正好补这个盲区。

次用途：油污 / 水渍 / 指纹一类污染在现有缺陷体系里没有对应检测器，而这类问题
灰度特征很弱、色彩特征很强，属净增能力。

口径（改这个模块前先读这七条）
------------------------------
C1  只吃**原始彩色图**，绝不经过 `Preprocessor`。预处理链会灰度化 / Retinex，
    色彩量从此不可恢复。
C2  `color_order` 由**调用方显式声明**，永不猜。用 `image.shape[2] == 3` 判不出
    通道顺序——BGR 与 RGB 都是三通道。
C3  S、V 与通道顺序**无关**；H 强相关。OpenCV 的 `BGR2HSV` 与 `RGB2HSV` 只交换
    R/B 的角色，恒等式：``cvtColor(img, BGR2HSV) == cvtColor(cvtColor(img,
    BGR2RGB), RGB2HSV)``，且 S、V 逐位相同。顺序喂错**不会报错**，只会静默把
    色相转掉 172°（实测金色铜面 H=17 → 103）。
C4  灰度输入 → ``available=False`` + ``note``，**不抛异常、不静默填 0**。灰度是
    本项目的合法输入（xflux 分支），填 0 会被读成"色度完美"即满分。
C5  色相统计只在 ``S >= hue_s_min`` 的像素上做。近中性像素的 H 是噪声——R≈G≈B
    时 H 完全由最低位的抖动决定。
C6  绝对判定与自适应判定**互相独立**，各自出结论。绝对阈值抓整板均匀变色，
    自适应阈值抓局部异常，两者同时命中时各出一条区域记录，不合并。
C7  输出**永不进** `weights` / `detail["scores"]`。这是刻意的：`weights` 五项精确
    加和 1.0，加维度必然重排权重、改动既有样本总分。所以本模块只监测、只展示、
    只导出，不影响 OK/NG。（`QualityAssessor.assess()` 的加权总分只遍历
    ``self.weights`` 的键，不遍历 ``scores`` 的键，所以不进 weights 就天然隔离。）

阈值全部是**占位值**，尚未用现场彩色样本标定。第一步的目标正是拿到真实分布。

色相是圆统计量
--------------
不能用线性均值：半圈 H=179 加半圈 H=1，线性均值给 90（=180°，恰好是最远的点），
圆均值给 0°。用 Mardia(1972)：``C=mean(cosθ)``、``S=mean(sinθ)``、
``R=hypot(C,S)``、均值 ``atan2(S,C)``、圆标准差 ``sqrt(-2·lnR)``。
R→0 时标准差发散（纯噪声色相会给出 425.9°），故必须夹在 ``CIRC_STD_MAX_DEG``。

圆中位数无闭式解，但 H 是整数，走 180 桶直方图穷举即可（180×180 = 32400 次
运算）。代价是量化下限 2°，所以自适应色相容差不可能低于约 4°。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ============================================================================
# 常量
# ============================================================================

HUE_UNIT_TO_DEG = 2.0      # OpenCV 的 H∈[0,179] 覆盖 0–360°
HUE_BINS = 180             # 直方图桶数 = H 的整数取值数
R_UNDEFINED = 1e-6         # 集中度低于此值视为"方向完全分散"
CIRC_STD_MAX_DEG = 180.0   # 圆标准差的物理上限
MAD_TO_SIGMA = 1.4826      # 正态下 MAD → σ 的一致性系数，与 SPC 3σ 同源

_VALID_ORDERS = ("bgr", "rgb")


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ColorRegion:
    """一处越界区域的定位信息。

    Attributes:
        basis: ``"absolute"`` 或 ``"adaptive"``，标明由哪套判据判出。
        reason: 人可读的越界原因（如"色相偏离 41°"）。
        area_pixels / area_mm2 / area_pct: 面积的三种口径。面积百分比的分母是
            **整幅图像**，与 `oxidation_percentage` 同口径，便于对照。
        centroid / bbox: 质心 ``(cx, cy)`` 与包围盒 ``(x1, y1, x2, y2)``。
        mean_hue_deg / mean_sat: 区域内的圆均值色相与算术均值饱和度。
        hue_deviation_deg: 区域内圆均值色相与所用基线的环形距离。
        severity: 越界倍数，``1.0`` 表示恰好压在阈值上，越大越离谱。
    """

    basis: str = "absolute"
    reason: str = ""
    area_pixels: int = 0
    area_mm2: float = 0.0
    area_pct: float = 0.0
    centroid: Tuple[float, float] = (0.0, 0.0)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    mean_hue_deg: Optional[float] = None
    mean_sat: float = 0.0
    hue_deviation_deg: Optional[float] = None
    severity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "basis": self.basis,
            "reason": self.reason,
            "area_pixels": self.area_pixels,
            "area_mm2": round(self.area_mm2, 6),
            "area_pct": round(self.area_pct, 4),
            "centroid": [round(self.centroid[0], 2), round(self.centroid[1], 2)],
            "bbox": list(self.bbox),
            "mean_hue_deg": (None if self.mean_hue_deg is None
                             else round(self.mean_hue_deg, 2)),
            "mean_sat": round(self.mean_sat, 2),
            "hue_deviation_deg": (None if self.hue_deviation_deg is None
                                  else round(self.hue_deviation_deg, 2)),
            "severity": round(self.severity, 3),
        }


@dataclass
class ColorFeatures:
    """一幅彩色图像的色彩特征与越界结论。

    色相类字段是 ``Optional``：当有效像素不足时置 ``None`` 而不是填一个假的
    数字。饱和度 / 明度类字段恒有值（只要图像是彩色的），因为 S 的分母是 V、
    对整体明暗不敏感，是最稳健的一路。
    """

    # --- 可用性 ---
    available: bool = False
    note: str = ""
    color_order: str = ""
    pixel_count: int = 0
    valid_hue_pixels: int = 0

    # --- 色相（圆统计量，单位：度） ---
    hue_mean_deg: Optional[float] = None
    hue_circ_std_deg: Optional[float] = None
    hue_r: Optional[float] = None          # 集中度，1=完全一致，0=完全分散
    hue_median_deg: Optional[float] = None

    # --- 饱和度 / 明度（OpenCV 0–255 量纲） ---
    sat_mean: Optional[float] = None
    sat_std: Optional[float] = None
    sat_p05: Optional[float] = None
    sat_p95: Optional[float] = None
    val_mean: Optional[float] = None

    # --- 越界统计 ---
    oor_abs_pixels: int = 0
    oor_abs_pct: float = 0.0
    oor_adaptive_pixels: int = 0
    oor_adaptive_pct: float = 0.0

    # --- 自适应基线（None = 该套未启用或像素不足） ---
    adaptive_hue_baseline_deg: Optional[float] = None
    adaptive_hue_tol_deg: Optional[float] = None
    adaptive_sat_baseline: Optional[float] = None
    adaptive_sat_tol: Optional[float] = None

    # --- 相对绝对基线的色相偏移（报告主指标） ---
    hue_deviation_deg: Optional[float] = None

    # --- 定位 ---
    regions: List[ColorRegion] = field(default_factory=list)

    @classmethod
    def unavailable(cls, note: str, color_order: str = "") -> "ColorFeatures":
        """构造一个"拿不到色彩量"的占位结果（C4）。"""
        return cls(available=False, note=note, color_order=color_order)

    @property
    def oor_count(self) -> int:
        """两套判据的区域条数之和（同一区域被两套都判出时算两条）。"""
        return len(self.regions)

    def to_dict(self) -> dict:
        """标量部分序列化。``regions`` 有意排除，与 `QualityReport.to_dict()`
        排除 ``detail`` 的写法对齐。"""
        def _r(v, n):
            return None if v is None else round(v, n)

        return {
            "available": self.available,
            "note": self.note,
            "color_order": self.color_order,
            "pixel_count": self.pixel_count,
            "valid_hue_pixels": self.valid_hue_pixels,
            "hue_mean_deg": _r(self.hue_mean_deg, 2),
            "hue_circ_std_deg": _r(self.hue_circ_std_deg, 2),
            "hue_r": _r(self.hue_r, 4),
            "hue_median_deg": _r(self.hue_median_deg, 2),
            "hue_deviation_deg": _r(self.hue_deviation_deg, 2),
            "sat_mean": _r(self.sat_mean, 2),
            "sat_std": _r(self.sat_std, 2),
            "sat_p05": _r(self.sat_p05, 2),
            "sat_p95": _r(self.sat_p95, 2),
            "val_mean": _r(self.val_mean, 2),
            "oor_abs_pixels": self.oor_abs_pixels,
            "oor_abs_pct": round(self.oor_abs_pct, 4),
            "oor_adaptive_pixels": self.oor_adaptive_pixels,
            "oor_adaptive_pct": round(self.oor_adaptive_pct, 4),
            "adaptive_hue_baseline_deg": _r(self.adaptive_hue_baseline_deg, 2),
            "adaptive_hue_tol_deg": _r(self.adaptive_hue_tol_deg, 2),
            "adaptive_sat_baseline": _r(self.adaptive_sat_baseline, 2),
            "adaptive_sat_tol": _r(self.adaptive_sat_tol, 2),
            "regions": [r.to_dict() for r in self.regions],
        }


# ============================================================================
# 分析器
# ============================================================================

class ColorAnalyzer:
    """色度 / 饱和度分析器。

    无状态、可复用。构造时读配置并校验 ``color_order``（早失败好过每帧静默算错）。
    """

    def __init__(self, config: dict):
        cfg = config.get("inspection", {}).get("color", {}) or {}

        self.enabled = cfg.get("enabled", True)
        self.color_order = str(cfg.get("color_order", "rgb")).lower()
        self.hue_s_min = float(cfg.get("hue_s_min", 30))
        self.use_roi = bool(cfg.get("use_roi", False))
        self.area_min_px = int(cfg.get("area_min_px", 100))
        self.morph_kernel = int(cfg.get("morph_kernel", 5))
        self.max_regions = int(cfg.get("max_regions", 50))

        abs_cfg = cfg.get("absolute", {}) or {}
        self.abs_enabled = bool(abs_cfg.get("enabled", True))
        self.abs_hue_baseline = float(abs_cfg.get("hue_baseline_deg", 34.0))
        self.abs_hue_tol = float(abs_cfg.get("hue_tolerance_deg", 25.0))
        self.abs_sat_baseline = float(abs_cfg.get("sat_baseline", 178.0))
        self.abs_sat_tol = float(abs_cfg.get("sat_tolerance", 50.0))

        ad_cfg = cfg.get("adaptive", {}) or {}
        self.ad_enabled = bool(ad_cfg.get("enabled", True))
        self.mad_k = float(ad_cfg.get("mad_k", 3.0))
        self.min_hue_tol = float(ad_cfg.get("min_hue_tol_deg", 8.0))
        self.min_sat_tol = float(ad_cfg.get("min_sat_tol", 20.0))
        self.min_valid_pixels = int(ad_cfg.get("min_valid_pixels", 1000))

        self.mm_per_pixel = float(
            config.get("system", {}).get("resolution_mm_per_pixel", 0.01)
        )

        self._validate_order(self.color_order)

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    def analyze(self, image: np.ndarray,
                color_order: Optional[str] = None) -> ColorFeatures:
        """分析一幅**原始彩色图**。

        Args:
            image: BGR 或 RGB 三通道图（uint8）。二维灰度图会返回
                ``available=False`` 的占位结果，不抛异常。
            color_order: ``"bgr"`` 或 ``"rgb"``。省略则用配置里的值。
                **必须与喂进来的字节顺序一致**，见模块 docstring 的 C2/C3。

        Returns:
            ColorFeatures。
        """
        order = self.color_order if color_order is None else str(color_order).lower()
        self._validate_order(order)

        if not self.enabled:
            return ColorFeatures.unavailable("色度指标未启用", order)
        if image is None:
            return ColorFeatures.unavailable("输入图像为空", order)

        hsv = self._to_hsv(image, order)
        if hsv is None:
            return ColorFeatures.unavailable(
                "灰度输入无色彩信息（相机像素格式需切到彩色）", order
            )

        hue_u = hsv[:, :, 0].astype(np.float64)   # OpenCV 单位 0–179
        sat = hsv[:, :, 1].astype(np.float64)
        val = hsv[:, :, 2].astype(np.float64)

        sel = self._selection_mask(image, order)
        pixel_count = int(np.count_nonzero(sel))
        if pixel_count == 0:
            return ColorFeatures.unavailable("ROI 内无有效像素", order)

        # C5：色相统计只在饱和度足够高的像素上做
        valid = sel & (sat >= self.hue_s_min)
        valid_hue_pixels = int(np.count_nonzero(valid))

        feats = ColorFeatures(
            available=True,
            color_order=order,
            pixel_count=pixel_count,
            valid_hue_pixels=valid_hue_pixels,
        )

        # --- 饱和度 / 明度（恒可算，最稳健的一路） ---
        s_sel = sat[sel]
        feats.sat_mean = float(np.mean(s_sel))
        feats.sat_std = float(np.std(s_sel))
        feats.sat_p05 = float(np.percentile(s_sel, 5))
        feats.sat_p95 = float(np.percentile(s_sel, 95))
        feats.val_mean = float(np.mean(val[sel]))

        # --- 色相圆统计量 ---
        notes = []
        if valid_hue_pixels > 0:
            hue_deg = hue_u[valid] * HUE_UNIT_TO_DEG
            mean_deg, std_deg, r = self._circular_stats(hue_deg)
            feats.hue_mean_deg = mean_deg
            feats.hue_circ_std_deg = std_deg
            feats.hue_r = r
            feats.hue_median_deg = (
                self._circular_median_units(hue_u[valid]) * HUE_UNIT_TO_DEG
            )
        else:
            notes.append(
                f"无饱和度≥{self.hue_s_min:g} 的像素，色相不可测"
                "（整板近中性灰，可能是严重氧化或光照不足）"
            )

        # --- 绝对判据 ---
        if self.abs_enabled:
            if feats.hue_mean_deg is not None:
                feats.hue_deviation_deg = self._circular_distance_deg(
                    feats.hue_mean_deg, self.abs_hue_baseline
                )
            mask_abs = np.zeros(hue_u.shape, dtype=np.uint8)
            sat_dev_map = np.abs(sat - self.abs_sat_baseline)
            mask_abs[sel & (sat_dev_map > self.abs_sat_tol)] = 255
            if valid_hue_pixels > 0:
                hue_dev_map = self._hue_deviation_map(hue_u, self.abs_hue_baseline)
                mask_abs[valid & (hue_dev_map > self.abs_hue_tol)] = 255

            feats.oor_abs_pixels = int(np.count_nonzero(mask_abs))
            feats.oor_abs_pct = feats.oor_abs_pixels / pixel_count * 100
            feats.regions.extend(self._localize(
                mask_abs, "absolute", hue_u, sat, valid, pixel_count,
                baseline_deg=self.abs_hue_baseline,
                hue_tol=self.abs_hue_tol,
                sat_baseline=self.abs_sat_baseline,
                sat_tol=self.abs_sat_tol,
            ))

        # --- 自适应判据 ---
        if self.ad_enabled:
            thr = self._adaptive_thresholds(hue_u, sat, valid, valid_hue_pixels)
            if thr is None:
                notes.append(
                    f"有效色相像素 {valid_hue_pixels} < {self.min_valid_pixels}，"
                    "自适应基线不可信，本帧只出绝对判据"
                )
            else:
                ad_hue_base, ad_hue_tol, ad_sat_base, ad_sat_tol = thr
                feats.adaptive_hue_baseline_deg = ad_hue_base
                feats.adaptive_hue_tol_deg = ad_hue_tol
                feats.adaptive_sat_baseline = ad_sat_base
                feats.adaptive_sat_tol = ad_sat_tol

                mask_ad = np.zeros(hue_u.shape, dtype=np.uint8)
                mask_ad[sel & (np.abs(sat - ad_sat_base) > ad_sat_tol)] = 255
                hue_dev_map = self._hue_deviation_map(hue_u, ad_hue_base)
                mask_ad[valid & (hue_dev_map > ad_hue_tol)] = 255

                feats.oor_adaptive_pixels = int(np.count_nonzero(mask_ad))
                feats.oor_adaptive_pct = (
                    feats.oor_adaptive_pixels / pixel_count * 100
                )
                feats.regions.extend(self._localize(
                    mask_ad, "adaptive", hue_u, sat, valid, pixel_count,
                    baseline_deg=ad_hue_base,
                    hue_tol=ad_hue_tol,
                    sat_baseline=ad_sat_base,
                    sat_tol=ad_sat_tol,
                ))

        # 区域按面积降序、截断
        feats.regions.sort(key=lambda r: r.area_pixels, reverse=True)
        if len(feats.regions) > self.max_regions:
            notes.append(
                f"越界区域 {len(feats.regions)} 处，仅保留面积最大的 "
                f"{self.max_regions} 处"
            )
            feats.regions = feats.regions[:self.max_regions]

        feats.note = "；".join(notes)
        return feats

    # ------------------------------------------------------------------
    # 色彩空间
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_order(order: str) -> None:
        if order not in _VALID_ORDERS:
            raise ValueError(
                f"color_order 必须是 {_VALID_ORDERS} 之一，收到 {order!r}。"
                "通道顺序无法从数组形状推断，必须显式声明。"
            )

    @staticmethod
    def _to_hsv(image: np.ndarray, order: str) -> Optional[np.ndarray]:
        """转 HSV。灰度 / 非法输入返回 None（由调用方转成 unavailable）。"""
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            return None
        ch = image.shape[2]
        if ch == 4:                      # BGRA / RGBA：丢掉 alpha
            image = image[:, :, :3]
        elif ch != 3:
            return None
        code = cv2.COLOR_BGR2HSV if order == "bgr" else cv2.COLOR_RGB2HSV
        return cv2.cvtColor(image, code)

    def _selection_mask(self, image: np.ndarray, order: str) -> np.ndarray:
        """参与统计的像素范围。默认全幅（``use_roi=False``）。

        ROI 默认关的三个理由：
          1. 配置里 ROI 本来就是关的；
          2. 铜面 ROI 的判据本身是色相——而这正是本指标要测的量，均匀氧化会被
             ROI 直接排除掉，恰好废掉主用途；
          3. 全幅分母与 `oxidation_percentage` 同口径，两者才好对照。

        启用时也不复用 `Preprocessor.get_roi_mask`——它把 ``COLOR_BGR2HSV``
        写死了，与这里的 C2（顺序由调用方声明）冲突。
        """
        if not self.use_roi:
            return np.ones(image.shape[:2], dtype=bool)
        hsv = self._to_hsv(image, order)
        if hsv is None:
            # 防御性分支：`analyze()` 在调这里之前已经做过同样的 `_to_hsv`，
            # 灰度输入根本走不到这一行（返回的是 unavailable）。留着是因为
            # `_selection_mask` 是个独立方法，直接调它时不该炸。
            return np.ones(image.shape[:2], dtype=bool)
        # ROI 判据：偏向暖色的低饱和以上像素（铜面特征的粗略代理）
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        return ((h <= 30) | (h >= 150)) & (s >= 40)

    # ------------------------------------------------------------------
    # 圆统计
    # ------------------------------------------------------------------

    @staticmethod
    def _circular_stats(hue_deg: np.ndarray) -> Tuple[float, float, float]:
        """Mardia 圆统计量。返回 ``(圆均值°, 圆标准差°, 集中度 R)``。

        圆标准差 ``sqrt(-2·lnR)`` 在 ``R→0`` 时发散（纯噪声色相会算出 425.9°），
        必须夹在 ``CIRC_STD_MAX_DEG``。
        """
        if hue_deg.size == 0:
            return 0.0, CIRC_STD_MAX_DEG, 0.0
        theta = np.deg2rad(hue_deg)
        c_bar = float(np.mean(np.cos(theta)))
        s_bar = float(np.mean(np.sin(theta)))
        # 上界必须夹：hypot 的浮点误差会让 R 略大于 1，于是 log(R)>0、
        # 根号下为负 → NaN，会顺着报告一路污染到数据库。R 是合向量长度，
        # 数学上不可能超过 1。
        r = min(1.0, float(np.hypot(c_bar, s_bar)))
        mean_deg = float(np.rad2deg(np.arctan2(s_bar, c_bar))) % 360.0
        if r <= R_UNDEFINED:
            std_deg = CIRC_STD_MAX_DEG
        else:
            # 下界也夹：R=1 时 log(1)=0，-2·0 会给出 -0.0，序列化后是刺眼的
            # "-0.0"。圆标准差按定义非负。
            std_deg = max(0.0, min(
                float(np.rad2deg(np.sqrt(-2.0 * np.log(r)))), CIRC_STD_MAX_DEG
            ))
        return mean_deg, std_deg, r

    @staticmethod
    def _circular_median_units(hue_units: np.ndarray) -> float:
        """圆中位数（OpenCV 单位，0–179）。

        无闭式解，用 180 桶直方图穷举：取使 ``Σ hist[k]·arc(k, b)`` 最小的桶。
        环形距离按 ``min(d, 180-d)`` 算。量化下限 1 个单位 = 2°。
        """
        if hue_units.size == 0:
            return 0.0
        idx = np.clip(hue_units.astype(np.int32), 0, HUE_BINS - 1)
        hist = np.bincount(idx, minlength=HUE_BINS).astype(np.float64)
        bins = np.arange(HUE_BINS, dtype=np.float64)
        d = np.abs(bins[:, None] - bins[None, :])
        d = np.minimum(d, HUE_BINS - d)
        cost = hist @ d
        return float(np.argmin(cost))

    @staticmethod
    def _circular_distance_deg(a_deg: float, b_deg: float) -> float:
        """两个角度之间的环形距离，落在 [0, 180]。"""
        d = abs(float(a_deg) - float(b_deg)) % 360.0
        return min(d, 360.0 - d)

    @classmethod
    def _hue_deviation_map(cls, hue_units: np.ndarray,
                           baseline_deg: float) -> np.ndarray:
        """逐像素到基线的环形偏差，**返回度**。

        返回度而不是 OpenCV 单位，是因为调用方拿去比的是配置里的
        ``*_tolerance_deg``（度）。曾经这里返回单位（0–90）而直接与度比较，
        等于把容差悄悄放大一倍——色相是 2°/单位，25° 的容差被当成 50° 用。
        这种错不报错，只是阈值失真。

        全图只在 ``[0,179]`` 上取值，故与基线的偏差取 ``d`` 与 ``180-d``
        的较小者——这里的 180 是**单位**跨度（=360° 的一半）。
        """
        base_u = (baseline_deg / HUE_UNIT_TO_DEG) % 180.0
        d = np.abs(hue_units - base_u)
        return np.minimum(d, 180.0 - d) * HUE_UNIT_TO_DEG

    @staticmethod
    def _robust_sigma(x: np.ndarray) -> float:
        """MAD → σ。``MAD_TO_SIGMA = 1.4826`` 是正态下的一致性系数。"""
        if x.size == 0:
            return 0.0
        med = float(np.median(x))
        return MAD_TO_SIGMA * float(np.median(np.abs(x - med)))

    def _adaptive_thresholds(self, hue_u, sat, valid, valid_hue_pixels):
        """用本帧自身的分布估计基线。返回 4 元组或 None（像素不足）。"""
        if valid_hue_pixels < self.min_valid_pixels:
            return None

        hue_deg = hue_u[valid] * HUE_UNIT_TO_DEG
        hue_base, _, _ = self._circular_stats(hue_deg)
        hue_dev = self._circular_dev_array_deg(hue_deg, hue_base)
        hue_tol = max(self.min_hue_tol, self.mad_k * self._robust_sigma(hue_dev))

        s_sel = sat[valid]
        sat_base = float(np.median(s_sel))
        sat_tol = max(self.min_sat_tol, self.mad_k * self._robust_sigma(s_sel))
        return hue_base, hue_tol, sat_base, sat_tol

    @staticmethod
    def _circular_dev_array_deg(values_deg: np.ndarray,
                                center_deg: float) -> np.ndarray:
        """逐元素到中心的环形偏差（度），落在 [0, 180]。"""
        d = np.abs(values_deg - center_deg) % 360.0
        return np.minimum(d, 360.0 - d)

    # ------------------------------------------------------------------
    # 区域定位
    # ------------------------------------------------------------------

    def _localize(self, mask: np.ndarray, basis: str, hue_u, sat, valid,
                  pixel_count: int, baseline_deg: float, hue_tol: float,
                  sat_baseline: float, sat_tol: float) -> List[ColorRegion]:
        """把越界掩膜拆成连通域，逐域算几何与色彩统计。

        这里**刻意不复用** ``DefectDetector._extract_connected_components``：
        那是实例私有方法，且返回带 ``type``/``severity`` 的 `Defect` 对象——对
        色彩区域而言这两个字段没有意义，硬套会把色彩区域混进缺陷体系。
        两边在形态学处理上的等价性由 ``tests/test_color_contract.py`` 的契约
        测试钉住，避免日后各自漂移。
        """
        if not mask.any():
            return []

        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
        )
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        if not m.any():
            return []

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            m, connectivity=8
        )
        mm2_per_px = self.mm_per_pixel ** 2
        regions: List[ColorRegion] = []

        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.area_min_px:
                continue

            x1 = int(stats[i, cv2.CC_STAT_LEFT])
            y1 = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = float(centroids[i][0]), float(centroids[i][1])

            sel = labels == i
            mean_sat = float(np.mean(sat[sel]))
            sat_dev = abs(mean_sat - sat_baseline)

            in_valid = sel & valid
            mean_hue_deg = None
            hue_dev = None
            if np.count_nonzero(in_valid) > 0:
                hd = hue_u[in_valid] * HUE_UNIT_TO_DEG
                mean_hue_deg, _, _ = self._circular_stats(hd)
                hue_dev = self._circular_distance_deg(mean_hue_deg, baseline_deg)

            reasons = []
            if hue_dev is not None:
                reasons.append(f"色相偏离 {hue_dev:.0f}°(容差 {hue_tol:.0f}°)")
            reasons.append(f"饱和度偏离 {sat_dev:.0f}(容差 {sat_tol:.0f})")

            sev_parts = []
            if hue_dev is not None and hue_tol > 0:
                sev_parts.append(hue_dev / hue_tol)
            if sat_tol > 0:
                sev_parts.append(sat_dev / sat_tol)

            regions.append(ColorRegion(
                basis=basis,
                reason="；".join(reasons),
                area_pixels=area,
                area_mm2=area * mm2_per_px,
                area_pct=area / pixel_count * 100,
                centroid=(cx, cy),
                bbox=(x1, y1, x1 + w, y1 + h),
                mean_hue_deg=mean_hue_deg,
                mean_sat=mean_sat,
                hue_deviation_deg=hue_dev,
                severity=max(sev_parts) if sev_parts else 0.0,
            ))

        return regions
