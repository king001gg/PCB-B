"""纹理特征提取模块。

实现了申报书中描述的三类纹理分析方法：

    GLCM（灰度共生矩阵）
        - 对比度（Contrast）
        - 能量（Energy / Angular Second Moment）
        - 熵（Entropy）
        - 均匀性（Homogeneity）
        - 相关性（Correlation）

    LBP（局部二值模式）
        - 旋转不变 uniform LBP
        - 多半径直方图拼接

    Gabor 滤波器组
        - 多方向、多尺度、多频率
        - 方向一致性指数
        - 纹理方向各向异性评估

    CV 均匀性热力图
        - 子区域变异系数（局部标准差 / 局部均值）
        - 滑动窗口实现

参考文献:
    - Haralick et al., "Textural Features for Image Classification", IEEE SMC, 1973.
    - Ojala et al., "Multiresolution Gray-Scale and Rotation Invariant Texture
      Classification with Local Binary Patterns", IEEE PAMI, 2002.
    - Jain & Farrokhnia, "Unsupervised Texture Segmentation Using Gabor Filters",
      Pattern Recognition, 1991.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class GLCMFeatures:
    """单个 GLCM 的特征值集合。"""
    contrast: float = 0.0          # 对比度：值越大对比度越高
    energy: float = 0.0            # 能量 / ASM：纹理均匀性的度量
    entropy: float = 0.0           # 熵：纹理的随机性/复杂度
    homogeneity: float = 0.0       # 均匀性：GLCM 元素集中在对角线附近的程度
    correlation: float = 0.0       # 相关性：灰度值沿行/列方向的线性依赖
    dissimilarity: float = 0.0     # 相异性：|i - j| 加权平均

    def as_vector(self) -> np.ndarray:
        """转换为 6 维特征向量。"""
        return np.array([
            self.contrast, self.energy, self.entropy,
            self.homogeneity, self.correlation, self.dissimilarity,
        ], dtype=np.float64)


@dataclass
class TextureFeatureVector:
    """多维度纹理特征向量。

    融合了 GLCM、LBP、Gabor 三方面的特征，用于分类器输入。
    """
    # GLCM 特征：每个 (距离, 角度) 组合一个 GLCMFeatures 对象
    glcm: List[GLCMFeatures] = field(default_factory=list)

    # LBP 特征：拼接的多半径直方图
    lbp_histogram: Optional[np.ndarray] = None

    # Gabor 特征：每个滤波器响应的均值和标准差
    gabor_features: Optional[np.ndarray] = None

    # 局部熵图（用于评估粗糙度分布均匀性）
    local_entropy_map: Optional[np.ndarray] = None

    # CV 均匀性热力图
    cv_heatmap: Optional[np.ndarray] = None

    # Gabor 各 (频率, 尺度) 组内各方向的响应能量，供 DCI 复用同一趟卷积。
    # 只有 9 组 × 8 方向 = 72 个标量 —— 刻意不保留响应图本身：
    # 2448×2048 下 72 张 float64 响应图共约 2.9 GB。
    gabor_orientation_energies: Optional[List[List[float]]] = None

    def flatten(self) -> np.ndarray:
        """将所有特征拼接为一维向量（用于分类器）。"""
        parts = []
        for g in self.glcm:
            parts.append(g.as_vector())
        if self.lbp_histogram is not None:
            parts.append(self.lbp_histogram.ravel())
        if self.gabor_features is not None:
            parts.append(self.gabor_features.ravel())
        if parts:
            return np.concatenate(parts)
        return np.array([], dtype=np.float64)


# ============================================================================
# GLCM 特征提取器
# ============================================================================

class GLCMExtractor:
    """灰度共生矩阵（GLCM）特征提取器。

    计算多个距离和角度下的 GLCM，提取 Haralick 纹理特征。

    Attributes:
        distances: 像素距离列表。
        angles: 角度列表（度），0/45/90/135 等。
        levels: 灰度量化级数。
        symmetric: 是否生成对称 GLCM。
        normalize: 是否归一化。
    """

    def __init__(
        self,
        distances: List[int] = None,
        angles: List[float] = None,
        levels: int = 256,
        symmetric: bool = True,
        normalize: bool = True,
    ):
        self.distances = distances or [1, 3, 5]
        self.angles = angles or [0, 45, 90, 135]
        self.levels = levels
        self.symmetric = symmetric
        self.normalize = normalize

        # 将角度转为弧度
        self._angles_rad = [np.deg2rad(a) for a in self.angles]

    def compute(self, image: np.ndarray) -> List[GLCMFeatures]:
        """计算所有 (距离, 角度) 组合的 GLCM 特征。

        Args:
            image: 灰度图像 (H, W), uint8。

        Returns:
            GLCMFeatures 列表，长度为 len(distances) × len(angles)。
        """
        self._validate_input(image)

        if HAS_SKIMAGE:
            return self._compute_skimage(image)
        else:
            return self._compute_numpy(image)

    # ------------------------------------------------------------------
    # scikit-image 实现（快速）
    # ------------------------------------------------------------------

    def _compute_skimage(self, image: np.ndarray) -> List[GLCMFeatures]:
        features = []

        for d in self.distances:
            glcm = graycomatrix(
                image,
                distances=[d],
                angles=self._angles_rad,
                levels=self.levels,
                symmetric=self.symmetric,
                normed=self.normalize,
            )

            # 提取各角度下的特征
            n_angles = len(self._angles_rad)
            for a in range(n_angles):
                g = glcm[:, :, 0, a]  # shape: (levels, levels)
                features.append(self._extract_features_from_glcm(g))

        return features

    def _extract_features_from_glcm(self, glcm: np.ndarray) -> GLCMFeatures:
        """从一个 GLCM 矩阵提取 6 个特征。"""
        eps = 1e-10
        # 行/列索引
        I, J = np.mgrid[0:self.levels, 0:self.levels]

        contrast = float(np.sum(glcm * (I - J) ** 2))
        energy = float(np.sum(glcm ** 2))
        entropy = float(-np.sum(glcm * np.log(glcm + eps)))
        homogeneity = float(np.sum(glcm / (1 + (I - J) ** 2)))

        # 相关性
        p_i = glcm.sum(axis=1)  # 行和
        p_j = glcm.sum(axis=0)  # 列和
        mu_i = float(np.sum(I[:, 0] * p_i))
        mu_j = float(np.sum(J[0, :] * p_j))
        sigma_i = float(np.sqrt(np.sum((I[:, 0] - mu_i) ** 2 * p_i)))
        sigma_j = float(np.sqrt(np.sum((J[0, :] - mu_j) ** 2 * p_j)))
        if sigma_i > eps and sigma_j > eps:
            correlation = float(
                np.sum(glcm * (I - mu_i) * (J - mu_j)) / (sigma_i * sigma_j)
            )
        else:
            correlation = 0.0

        dissimilarity = float(np.sum(glcm * np.abs(I - J)))

        return GLCMFeatures(
            contrast=round(contrast, 6),
            energy=round(energy, 6),
            entropy=round(entropy, 6),
            homogeneity=round(homogeneity, 6),
            correlation=round(correlation, 6),
            dissimilarity=round(dissimilarity, 6),
        )

    # ------------------------------------------------------------------
    # 纯 NumPy 实现（回退）
    # ------------------------------------------------------------------

    def _compute_numpy(self, image: np.ndarray) -> List[GLCMFeatures]:
        features = []

        # 量化灰度
        if self.levels < 256:
            scale = self.levels / 256.0
            img_q = (image.astype(np.float64) * scale).astype(np.int32)
        else:
            img_q = image.astype(np.int32)

        for d in self.distances:
            for angle_deg in self.angles:
                glcm = self._build_glcm_numpy(img_q, d, angle_deg)
                features.append(self._extract_features_from_glcm(glcm))

        return features

    def _build_glcm_numpy(
        self, image: np.ndarray, distance: int, angle_deg: float
    ) -> np.ndarray:
        """用纯 NumPy 构建 GLCM。

        对较大图像使用步长优化以保持速度。
        """
        angle_rad = np.deg2rad(angle_deg)
        dy = int(round(distance * np.sin(angle_rad)))
        dx = int(round(distance * np.cos(angle_rad)))

        h, w = image.shape
        L = self.levels

        # 有效像素掩膜
        y_start, y_end = max(0, -dy), min(h, h - dy)
        x_start, x_end = max(0, -dx), min(w, w - dx)

        ref = image[y_start:y_end, x_start:x_end]
        shifted = image[y_start + dy:y_end + dy, x_start + dx:x_end + dx]

        glcm = np.zeros((L, L), dtype=np.float64)
        np.add.at(glcm, (ref.ravel(), shifted.ravel()), 1)

        if self.symmetric:
            glcm = glcm + glcm.T

        if self.normalize:
            total = glcm.sum()
            if total > 0:
                glcm /= total

        return glcm

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def mean_features(self, image: np.ndarray) -> GLCMFeatures:
        """计算所有 (距离, 角度) 组合特征的平均值。"""
        all_features = self.compute(image)
        keys = [
            "contrast", "energy", "entropy",
            "homogeneity", "correlation", "dissimilarity",
        ]
        avg = {}
        for key in keys:
            avg[key] = np.mean([getattr(f, key) for f in all_features])
        return GLCMFeatures(**avg)

    @staticmethod
    def _validate_input(image: np.ndarray) -> None:
        if image is None:
            raise ValueError("输入图像不能为 None")
        if image.ndim != 2:
            raise ValueError(f"GLCM 需要灰度图像，收到 {image.ndim} 维")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)


# ============================================================================
# LBP 特征提取器
# ============================================================================

class LBPExtractor:
    """局部二值模式（LBP）特征提取器。

    支持多半径旋转不变 uniform LBP，提取局部纹理微观结构特征。

    Attributes:
        radius_list: 半径值列表。
        n_points_list: 每半径的采样点数量列表。
        method: LBP 变体（'uniform', 'default', 'var'）。
    """

    def __init__(
        self,
        radius_list: List[int] = None,
        n_points_list: List[int] = None,
        method: str = "uniform",
    ):
        self.radius_list = radius_list or [1, 2, 3]
        self.n_points_list = n_points_list or [8, 16, 24]

        if len(self.radius_list) != len(self.n_points_list):
            raise ValueError(
                f"radius_list 和 n_points_list 长度必须相同: "
                f"{len(self.radius_list)} vs {len(self.n_points_list)}"
            )

        self.method = method

    def compute(self, image: np.ndarray) -> np.ndarray:
        """计算 LBP 图像。

        Args:
            image: 灰度图像 (H, W), uint8。

        Returns:
            LBP 编码图像（每像素一个编码值）。
        """
        self._validate_input(image)

        if HAS_SKIMAGE:
            return self._compute_skimage(image)
        else:
            return self._compute_numpy(image)

    def histogram(self, lbp_image: np.ndarray) -> np.ndarray:
        """计算归一化 LBP 直方图。

        Args:
            lbp_image: LBP 编码图像（来自 compute() 的输出）。

        Returns:
            归一化直方图，形状取决于 method：
            - 'uniform': (n_points + 2,) 个 bin
            - 'default': (2^n_points,) 个 bin
        """
        n_bins = int(lbp_image.max()) + 1
        hist, _ = np.histogram(
            lbp_image.ravel(), bins=n_bins, range=(0, n_bins)
        )
        hist = hist.astype(np.float64)
        hist /= (hist.sum() + 1e-10)
        return hist

    def multi_radius_histogram(self, image: np.ndarray) -> np.ndarray:
        """计算多半径 LBP 直方图（拼接各半径的直方图）。

        Args:
            image: 灰度图像 (H, W), uint8。

        Returns:
            拼接的归一化直方图向量。
        """
        histograms = []
        for r, p in zip(self.radius_list, self.n_points_list):
            lbp = self._compute_skimage_single(image, r, p)
            hist = self.histogram(lbp)
            histograms.append(hist)
        return np.concatenate(histograms)

    # ------------------------------------------------------------------
    # scikit-image 实现
    # ------------------------------------------------------------------

    def _compute_skimage(self, image: np.ndarray) -> np.ndarray:
        # 默认使用最大半径
        r = self.radius_list[0]
        p = self.n_points_list[0]
        return self._compute_skimage_single(image, r, p)

    def _compute_skimage_single(
        self, image: np.ndarray, radius: int, n_points: int
    ) -> np.ndarray:
        if self.method == "uniform":
            method = "uniform"
        elif self.method == "var":
            method = "uniform"  # skimage 不支持 var，回退
        else:
            method = "default"

        return local_binary_pattern(image, n_points, radius, method=method)

    # ------------------------------------------------------------------
    # 纯 NumPy 实现（回退）
    # ------------------------------------------------------------------

    def _compute_numpy(self, image: np.ndarray) -> np.ndarray:
        r = self.radius_list[0]
        p = self.n_points_list[0]
        return self._lbp_numpy(image, r, p)

    def _lbp_numpy(
        self, image: np.ndarray, radius: int, n_points: int
    ) -> np.ndarray:
        """纯 NumPy LBP 实现。"""
        h, w = image.shape
        lbp = np.zeros((h, w), dtype=np.uint8)

        angles = 2 * np.pi * np.arange(n_points) / n_points
        dy = np.round(radius * np.sin(angles)).astype(int)
        dx = np.round(radius * np.cos(angles)).astype(int)

        for i in range(n_points):
            shifted = np.roll(image, (dy[i], dx[i]), axis=(0, 1))
            lbp += ((shifted >= image) << i).astype(np.uint8)

        # 边界修复
        lbp[:radius, :] = 0
        lbp[-radius:, :] = 0
        lbp[:, :radius] = 0
        lbp[:, -radius:] = 0

        if self.method == "uniform":
            lbp = self._to_uniform(lbp, n_points)

        return lbp

    @staticmethod
    def _to_uniform(lbp: np.ndarray, n_points: int) -> np.ndarray:
        """将 LBP 编码映射为 uniform 编码（跳变数 ≤ 2）。"""
        # 计算跳变数
        transitions = np.zeros_like(lbp, dtype=np.int32)
        encoded = np.zeros_like(lbp, dtype=np.uint8)

        for i in range(n_points):
            bit_i = (lbp >> i) & 1
            bit_next = (lbp >> ((i + 1) % n_points)) & 1
            transitions += (bit_i ^ bit_next).astype(np.int32)

        # uniform 编码：跳变 ≤ 2 的保留 popcount，其余标记为 n_points+1
        uniform_mask = transitions <= 2
        for i in range(n_points):
            encoded[uniform_mask] += ((lbp[uniform_mask] >> i) & 1)

        encoded[~uniform_mask] = n_points + 1
        return encoded

    @staticmethod
    def _validate_input(image: np.ndarray) -> None:
        if image is None:
            raise ValueError("输入图像不能为 None")
        if image.ndim != 2:
            raise ValueError(f"LBP 需要灰度图像，收到 {image.ndim} 维")


# ============================================================================
# Gabor 滤波器组
# ============================================================================

class GaborFilterBank:
    """多方向多尺度 Gabor 滤波器组。

    用于分析喷砂锚纹的方向一致性（各向异性）和纹理周期。
    方向一致性高的表明喷砂均匀；方向分散则表明存在区域缺陷。

    Attributes:
        orientations: 滤波器方向数（均匀分布在 [0, π)）。
        scales: 滤波器核尺寸列表（像素）。
        frequencies: 中心频率列表（周期/像素）。
    """

    def __init__(
        self,
        orientations: int = 8,
        scales: List[int] = None,
        frequencies: List[float] = None,
    ):
        self.orientations = orientations
        self.scales = scales or [3, 5, 7]
        self.frequencies = frequencies or [0.1, 0.3, 0.5]

        self._kernels: List[np.ndarray] = []
        self._build_kernels()

    def _build_kernels(self) -> None:
        """预生成所有 Gabor 滤波核，并记录每个核所属的 (freq, scale) 组号。

        组号必须显式记录：``direction_consistency`` 要按组比较各方向的能量，
        而「连续 orientations 个核就是一组」只是当前嵌套顺序下的巧合，
        靠位置约定会在顺序一改时静默算错。
        """
        self._kernels = []
        self._kernel_group = []
        thetas = np.linspace(0, np.pi, self.orientations, endpoint=False)

        group = 0
        for freq in self.frequencies:
            for scale in self.scales:
                for theta in thetas:
                    kernel = self._gabor_kernel(scale, theta, freq)
                    self._kernels.append(kernel)
                    self._kernel_group.append(group)
                group += 1

    def _gabor_kernel(
        self, size: int, theta: float, frequency: float,
        sigma_x: float = None, sigma_y: float = None,
    ) -> np.ndarray:
        """生成 Gabor 核。"""
        if sigma_x is None:
            sigma_x = size / 3.0
        if sigma_y is None:
            sigma_y = sigma_x * 1.5

        # 确保 ksize 为奇数
        ksize = 2 * size + 1
        y, x = np.mgrid[-size:size + 1, -size:size + 1]

        # 旋转坐标
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        # Gabor 函数
        gb = np.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 +
                             y_theta ** 2 / sigma_y ** 2))
        gb *= np.cos(2 * np.pi * frequency * x_theta)

        # 零均值归一化
        gb -= gb.mean()
        gb /= np.sqrt(np.sum(gb ** 2)) + 1e-10

        return gb.astype(np.float64)

    @property
    def n_kernels(self) -> int:
        """滤波器总数。"""
        return len(self._kernels)

    @property
    def n_groups(self) -> int:
        """(频率, 尺度) 组合数，即方向一致性的分组数。"""
        return len(self.frequencies) * len(self.scales)

    @staticmethod
    def _as_float(image: np.ndarray) -> np.ndarray:
        """统一成 [0, 1] 的 float64，与原有口径一致。"""
        if image.dtype != np.float64:
            return image.astype(np.float64) / 255.0
        return image

    @staticmethod
    def _convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """用 cv2 复现 ``scipy.signal.convolve2d(img, kernel, mode="same", boundary="symm")``。

        两处都要对齐，否则结果是错的而不是慢的：

        * ``filter2D`` 算的是**相关**，``convolve2d`` 算的是**卷积**，
          故核要翻转。
        * scipy 的 ``boundary="symm"``（``d c b a | a b c d | d c b a``，
          边缘样本重复）对应 cv2 的 ``BORDER_REFLECT``；
          少一像素的 ``BORDER_REFLECT_101`` 是 scipy 的 ``reflect``，不是它。

        实测三个尺寸的核实测与 scipy 逐点最大差 4.2e-16（机器精度量级），
        边界带同样对齐。换掉的原因：``convolve2d`` 在 2448×2048 上单核
        耗时 0.27～1.0 s，整条流水线 144 次卷积占 97 s 中的 93.8 s；
        ``filter2D`` 快 12～21 倍。
        """
        return cv2.filter2D(img, -1, kernel[::-1, ::-1].copy(),
                            borderType=cv2.BORDER_REFLECT)

    def filter(self, image: np.ndarray) -> List[np.ndarray]:
        """对图像应用所有 Gabor 滤波器。

        Args:
            image: 灰度图像 (H, W), dtype float64。

        Returns:
            响应图列表，每个元素形状 (H, W)，长度 = n_kernels。

        注意本方法会同时持有全部响应图 —— 2448×2048 下约 2.9 GB。
        只需要逐图标量时请用 ``scan()``，它逐张归约后即丢弃。
        """
        img = self._as_float(image)
        return [self._convolve(img, kernel) for kernel in self._kernels]

    def scan(self, image: np.ndarray) -> Tuple[np.ndarray, List[List[float]]]:
        """单趟扫描全部滤波器，同时产出特征向量与各方向响应能量。

        ``feature_vector`` 与 ``direction_consistency`` 都要遍历全部核，
        原本各跑一趟、输入完全相同，第二趟是纯重算。这里合并为一趟，
        并**逐张**把响应图归约成标量后丢弃。

        Returns:
            ``(特征向量, 各 (freq, scale) 组的 mean(响应²) 列表)``。
            特征向量为 2 × n_kernels 维（每个核的均值、标准差交替排列）。

        两次归约都作用在同一张响应图上，故与分两趟跑的结果逐位相同；
        响应图不再累积，峰值内存从 n_kernels 张降到 1 张。
        """
        img = self._as_float(image)
        groups: List[List[float]] = [[] for _ in range(self.n_groups)]
        features: List[float] = []

        for kernel, group in zip(self._kernels, self._kernel_group):
            r = self._convolve(img, kernel)
            features.append(np.mean(r))
            features.append(np.std(r))
            groups[group].append(np.mean(r ** 2))

        return np.array(features, dtype=np.float64), groups

    def energy_map(self, image: np.ndarray) -> np.ndarray:
        """计算 Gabor 能量图（所有滤波器响应平方的平均值）。

        Returns:
            能量图 (H, W)。

        逐张累加而非先收集全部响应图，避免 n_kernels 张响应图同时驻留。
        """
        img = self._as_float(image)
        energy = None
        for kernel in self._kernels:
            r = self._convolve(img, kernel)
            energy = r ** 2 if energy is None else energy + r ** 2
        return energy / len(self._kernels)

    def direction_consistency(
        self,
        image: np.ndarray,
        orientation_energies: Optional[List[List[float]]] = None,
    ) -> float:
        """评估锚纹方向一致性。

        方向一致性指数 (Direction Consistency Index, DCI)：
        DCI = max_orientation_response / sum(all_orientation_responses)

        值越接近 1/n_orientations 表示各向同性（均匀喷砂）；
        值接近 1 表示强方向性（喷砂不均匀）。

        Args:
            image: 灰度图像 (H, W)。
            orientation_energies: 已由 ``scan()`` 算出的各组方向能量。
                给定则直接使用，不再重跑一趟卷积 —— 调用方在 ``analyze()``
                之后调用本方法时，两处遍历的是同一批核、同一张图，
                重跑纯属浪费。不传则本方法自行扫描一遍（原有调用方式照旧可用）。

        Returns:
            DCI 值 [0, 1]（归一化后 [0, 1] 化：0 = 完全各向同性，1 = 最强方向性）。
        """
        if orientation_energies is None:
            orientation_energies = self.scan(image)[1]

        orientation_responses = []
        for orient_energy in orientation_energies:
            ori_arr = np.array(orient_energy)
            orientation_responses.append(
                float(np.max(ori_arr) / (np.sum(ori_arr) + 1e-10))
            )

        # 返回平均 DCI（归一化到 [0, 1]）
        raw = np.mean(orientation_responses)
        # 1/n_orientations → 0, 1 → 1
        baseline = 1.0 / self.orientations
        normalized = (raw - baseline) / (1.0 - baseline + 1e-10)
        return max(0.0, min(1.0, normalized))

    def feature_vector(self, image: np.ndarray) -> np.ndarray:
        """从 Gabor 响应提取特征向量。

        对每个滤波器响应计算均值和标准差作为特征。

        Returns:
            特征向量 (2 * n_kernels,)。
        """
        return self.scan(image)[0]


# ============================================================================
# 纹理分析主入口
# ============================================================================

class TextureAnalyzer:
    """纹理分析主入口 — 组合 GLCM、LBP、Gabor 三种方法。

    提供一站式纹理特征提取接口，输出多维特征向量、
    局部熵图、CV 均匀性热力图等。

    Attributes:
        glcm_extractor: GLCM 特征提取器。
        lbp_extractor: LBP 特征提取器。
        gabor_bank: Gabor 滤波器组。
        cv_window: CV 滑动窗口大小。
        local_entropy_window: 局部熵窗口大小。
    """

    def __init__(self, config: dict):
        tex_cfg = config.get("inspection", {}).get("texture", {})

        # GLCM
        glcm_cfg = tex_cfg.get("glcm", {})
        self.glcm_extractor = GLCMExtractor(
            distances=glcm_cfg.get("distances", [1, 3, 5]),
            angles=glcm_cfg.get("angles", [0, 45, 90, 135]),
            levels=glcm_cfg.get("levels", 256),
            symmetric=glcm_cfg.get("symmetric", True),
            normalize=glcm_cfg.get("normalize", True),
        )

        # LBP
        lbp_cfg = tex_cfg.get("lbp", {})
        self.lbp_extractor = LBPExtractor(
            radius_list=lbp_cfg.get("radius_list", [1, 2, 3]),
            n_points_list=lbp_cfg.get("n_points_list", [8, 16, 24]),
            method=lbp_cfg.get("method", "uniform"),
        )

        # Gabor
        gabor_cfg = tex_cfg.get("gabor", {})
        self.gabor_bank = GaborFilterBank(
            orientations=gabor_cfg.get("orientations", 8),
            scales=gabor_cfg.get("scales", [3, 5, 7]),
            frequencies=gabor_cfg.get("frequencies", [0.1, 0.3, 0.5]),
        )

        # CV
        self.cv_window = tex_cfg.get("cv_window", 32)
        self.local_entropy_window = tex_cfg.get("local_entropy_window", 9)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def analyze(self, image: np.ndarray) -> TextureFeatureVector:
        """运行完整的纹理分析流水线。

        Args:
            image: 灰度图像 (H, W), uint8。

        Returns:
            TextureFeatureVector 包含所有提取的特征。
        """
        # GLCM
        glcm_features = self.glcm_extractor.compute(image)

        # LBP（多半径直方图）
        lbp_hist = self.lbp_extractor.multi_radius_histogram(image)

        # Gabor —— 一趟拿到特征向量与各方向能量，后者随结果带走供 DCI 复用
        gabor_features, gabor_energies = self.gabor_bank.scan(image)

        # 局部熵图
        entropy_map = self.compute_local_entropy(image)

        result = TextureFeatureVector(
            glcm=glcm_features,
            lbp_histogram=lbp_hist,
            gabor_features=gabor_features,
            local_entropy_map=entropy_map,
            gabor_orientation_energies=gabor_energies,
        )

        return result

    # ------------------------------------------------------------------
    # CV 均匀性热力图
    # ------------------------------------------------------------------

    def compute_cv_heatmap(self, image: np.ndarray) -> np.ndarray:
        """生成表面粗糙度均匀性热力图。

        使用滑动窗口计算局部变异系数（CV = std / mean），
        表征表面粗化程度的空间分布均匀性。

        CV 值高表示该区域粗糙度波动大（潜在质量问题）；
        CV 值低表示该区域粗糙度均匀。

        Args:
            image: 灰度图像 (H, W), uint8。

        Returns:
            CV 热力图 (H, W), float64，取值 [0, 1]。
        """
        if image.dtype == np.uint8:
            img = image.astype(np.float64)
        else:
            img = image.astype(np.float64)
        img = img / 255.0

        # 局部均值
        kernel = np.ones((self.cv_window, self.cv_window),
                          dtype=np.float64) / (self.cv_window ** 2)
        local_mean = cv2.filter2D(img, -1, kernel)

        # 局部标准差
        local_sq = cv2.filter2D(img ** 2, -1, kernel)
        local_var = local_sq - local_mean ** 2
        local_var = np.maximum(local_var, 0)
        local_std = np.sqrt(local_var)

        # CV = std / mean
        cv_map = np.zeros_like(local_mean)
        valid = local_mean > 1e-6
        cv_map[valid] = local_std[valid] / local_mean[valid]

        # 归一化到 [0, 1]
        cv_max = np.percentile(cv_map[valid], 95) if valid.any() else 1.0
        if cv_max > 1e-6:
            cv_map = np.clip(cv_map / cv_max, 0, 1)

        return cv_map

    # ------------------------------------------------------------------
    # 局部熵图
    # ------------------------------------------------------------------

    def compute_local_entropy(
        self, image: np.ndarray, window_size: int = None
    ) -> np.ndarray:
        """计算局部熵图。

        局部熵反映区域内纹理的复杂度和随机性。
        高熵区 → 纹理丰富（良好粗化），低熵区 → 纹理贫乏（可能存在未粗化）。

        Args:
            image: 灰度图像 (H, W), uint8。
            window_size: 窗口大小，默认使用 self.local_entropy_window。

        Returns:
            局部熵图 (H, W), float64。
        """
        ws = window_size or self.local_entropy_window
        if image.dtype != np.float64:
            img = image.astype(np.float64)
        else:
            img = image

        # 使用 cv2 的积分图加速局部直方图熵计算
        # 简化实现：高斯滤波近似
        kernel = np.ones((ws, ws), dtype=np.float64) / (ws * ws)
        local_mean = cv2.filter2D(img, -1, kernel)

        # 局部熵近似：-p * log(p) 的局部均值
        eps = 1e-10
        p = (img - local_mean) ** 2  # 近似概率密度
        p = p / (p.sum() + eps)
        local_entropy = -p * np.log(p + eps)
        entropy_map = cv2.filter2D(local_entropy, -1, kernel)

        # 归一化
        entropy_map -= entropy_map.min()
        e_max = entropy_map.max()
        if e_max > 1e-6:
            entropy_map /= e_max

        return entropy_map

    # ------------------------------------------------------------------
    # 纹理方向一致性
    # ------------------------------------------------------------------

    def direction_consistency(
        self,
        image: np.ndarray,
        orientation_energies: Optional[List[List[float]]] = None,
    ) -> float:
        """评估喷砂锚纹方向一致性。

        Args:
            image: 灰度图像 (H, W)。
            orientation_energies: ``analyze()`` 返回值里的
                ``gabor_orientation_energies``。传入即复用那一趟卷积，
                省掉一次全量 Gabor（在相机分辨率下约占总耗时的四成）。

        Returns:
            DCI 归一化值 [0, 1]，0 = 完全各向同性（均匀），1 = 最强方向性（不均匀）。
        """
        return self.gabor_bank.direction_consistency(image, orientation_energies)
