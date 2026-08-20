"""喷砂铜面图像预处理模块。

处理流程：
    原始图像 → ROI 提取（铜面颜色阈值） → 多尺度 Retinex 光照校正 → CLAHE 纹理增强

参考文献:
    - Jobson et al., "A Multiscale Retinex for Bridging the Gap Between Color
      Images and the Human Observation of Scenes", IEEE TIP, 1997.
    - Zuiderveld, "Contrast Limited Adaptive Histogram Equalization", 1994.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class Preprocessor:
    """喷砂铜面图像预处理流水线。

    封装 ROI 提取、Retinex 光照校正和 CLAHE 纹理增强三个步骤，
    每个步骤可独立启用/禁用。

    Attributes:
        config: 来自 config/default.yaml 的 preprocessing + roi 配置字典。
    """

    def __init__(self, config: dict):
        roi_cfg = config.get("inspection", {}).get("roi", {})
        pp_cfg = config.get("inspection", {}).get("preprocessing", {})

        # --- ROI ---
        self.roi_enabled = roi_cfg.get("enabled", False)
        self.copper_lower = np.array(roi_cfg.get("copper_hsv_lower", [0, 30, 60]))
        self.copper_upper = np.array(roi_cfg.get("copper_hsv_upper", [25, 255, 255]))

        # --- Retinex ---
        retinex = pp_cfg.get("retinex", {})
        self.retinex_enabled = retinex.get("enabled", True)
        self.sigmas: List[float] = retinex.get("sigma", [15, 80, 250])
        self.gain: float = retinex.get("gain", 128)
        self.offset: float = retinex.get("offset", 128)

        # --- CLAHE ---
        clahe = pp_cfg.get("clahe", {})
        self.clahe_enabled = clahe.get("enabled", True)
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe.get("clip_limit", 2.0),
            tileGridSize=tuple(clahe.get("tile_size", [8, 8])),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray) -> np.ndarray:
        """运行完整预处理流水线。

        Args:
            image: 输入 BGR 或灰度图像 (H, W, 3) 或 (H, W)。

        Returns:
            预处理后的灰度图像 (H, W)，取值范围 [0, 255]。
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        # 转灰度（如果尚未是灰度）
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Step 1: ROI 提取（可选）
        if self.roi_enabled and image.ndim == 3:
            gray = self.extract_roi(image)

        # Step 2: Retinex 光照校正（可选）
        if self.retinex_enabled:
            # Retinex 需要 RGB 输入，但我们在灰度域也能工作
            if image.ndim == 3:
                gray_rgb = self.retinex_correct(image)
                gray = cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2GRAY)
            else:
                # 灰度输入：构造伪彩色，处理后取灰度
                pseudo = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                gray_rgb = self.retinex_correct(pseudo)
                gray = cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2GRAY)

        # Step 3: CLAHE 增强（可选，需要灰度）
        if self.clahe_enabled:
            gray = self.enhance_texture(gray)

        # 确保输出是 uint8
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray

    # ------------------------------------------------------------------
    # Step 1: ROI 提取
    # ------------------------------------------------------------------

    def extract_roi(self, image: np.ndarray) -> np.ndarray:
        """基于铜面颜色特征的 ROI 提取。

        将 BGR 图像转换到 HSV 空间，根据铜面颜色范围生成掩膜，
        返回掩膜内的灰度图像（仅铜面区域），用于后续纹理分析。

        Args:
            image: BGR 图像 (H, W, 3)。

        Returns:
            只保留铜面区域的灰度图像 (H, W)。
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.copper_lower, self.copper_upper)

        # 形态学清理：去除噪点，填补孔洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.bitwise_and(gray, gray, mask=mask)

    # ------------------------------------------------------------------
    # Step 2: 多尺度 Retinex（MSRCR）
    # ------------------------------------------------------------------

    def retinex_correct(self, image: np.ndarray) -> np.ndarray:
        """多尺度 Retinex 带色彩恢复（MSRCR）光照校正。

        使用对数域减法消除光照不均匀性，多尺度高斯模糊覆盖不同空间频率，
        色彩恢复函数减轻灰度化效应。

        公式：
            R_i = Σ w_k · [log I_i(x) - log(F_k * I_i(x))]
            MSRCR_i = G · [R_i · CR_i + b]

        Args:
            image: RGB 图像 (H, W, 3), dtype uint8 或 float32。

        Returns:
            光照校正后的 RGB 图像 (H, W, 3), uint8。
        """
        if image.dtype != np.float32:
            img = image.astype(np.float32) + 1.0  # +1 避免 log(0)
        else:
            img = image + 1.0

        # 多尺度 Retinex
        retinex = np.zeros_like(img)
        weight = 1.0 / len(self.sigmas)

        for sigma in self.sigmas:
            blurred = self._gaussian_blur(img, sigma)
            retinex += weight * (np.log(img) - np.log(blurred + 1e-8))

        # 色彩恢复
        if image.ndim == 3:
            retinex = self._color_restoration(retinex, img)

        # 增益/偏置映射到显示范围
        retinex = self.gain * retinex + self.offset
        retinex = np.clip(retinex, 0, 255).astype(np.uint8)
        return retinex

    def _gaussian_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """生成可分离的高斯核并进行滤波。

        核半径取 ceil(3*sigma)，以确保覆盖 99.7% 的能量。
        """
        ksize = int(np.ceil(3 * sigma))
        ksize = max(1, ksize)
        # 确保 ksize 为奇数
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    def _color_restoration(
        self, retinex: np.ndarray, original: np.ndarray
    ) -> np.ndarray:
        """色彩恢复函数：按通道归一化以保持颜色比率。

        CR_i = log(α I_i) - log(Σ_c I_c)
        """
        alpha = 125.0
        sum_img = np.sum(original, axis=2, keepdims=True) + 1e-8
        cr = np.log(alpha * original + 1e-8) - np.log(sum_img)
        return retinex * cr

    # ------------------------------------------------------------------
    # Step 3: CLAHE 纹理增强
    # ------------------------------------------------------------------

    def enhance_texture(self, image: np.ndarray) -> np.ndarray:
        """使用 CLAHE 增强纹理对比度。

        适用于低对比度喷砂铜面图像，可有效突显锚纹微观结构。

        Args:
            image: 灰度图像 (H, W), uint8。

        Returns:
            CLAHE 增强后的灰度图像 (H, W), uint8。
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if gray.dtype == np.float32 or gray.dtype == np.float64:
            gray = (gray * 255).astype(np.uint8)

        return self.clahe.apply(gray)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def get_roi_mask(image: np.ndarray, hsv_lower: np.ndarray,
                      hsv_upper: np.ndarray) -> np.ndarray:
        """静态方法：生成铜面 ROI 二值掩膜。

        Args:
            image: BGR 图像 (H, W, 3)。
            hsv_lower: HSV 下界 [H, S, V]。
            hsv_upper: HSV 上界 [H, S, V]。

        Returns:
            二值掩膜 (H, W), dtype uint8。
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask
