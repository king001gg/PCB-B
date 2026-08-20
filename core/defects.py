"""喷砂表面缺陷检测模块。

实现对申报书所列三类缺陷的自动检测：

    (1) 氧化斑/污渍检测（oxidation）
        - HSV 颜色空间分割，利用铜面金色到暗褐色的色相变化
        - 连通域分析，过滤噪点，输出面积和位置

    (2) 磨料嵌入检测（abrasive_embedding）
        - 形态学白顶帽变换（White Top-hat）
        - 提取比背景亮的小型突起 → 磨料颗粒嵌入
        - 计数和尺寸分布统计

    (3) 未粗化区域检测（unroughened）
        - 基于 GLCM 纹理能量的阈值分割
        - 低能量区域 → 光滑区域 → 铜面未充分粗化
        - 面积百分比和空间分布

    (4) 划痕/沟槽检测（scratch）
        - 扩展检测：基于 Gabor 方向响应的异常线性结构

检测结果以 Defect dataclass 统一输出。
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


# ============================================================================
# 缺陷数据结构
# ============================================================================

@dataclass
class Defect:
    """单个缺陷的信息。

    Attributes:
        type: 缺陷类型字符串。
        mask: 二值掩膜 (H, W)，标记该缺陷的像素级位置。
        area_pixels: 缺陷面积（像素数）。
        area_mm2: 缺陷面积（mm²），基于 resolution_mm_per_pixel 计算。
        centroid: 质心坐标 (cx, cy)。
        bbox: 边界框 (x1, y1, x2, y2)。
        severity: 严重度 [0.0, 1.0]，1.0 最严重。
        confidence: 检测置信度 [0.0, 1.0]。
    """
    type: str
    mask: np.ndarray
    area_pixels: int = 0
    area_mm2: float = 0.0
    centroid: Tuple[int, int] = (0, 0)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    severity: float = 0.0
    confidence: float = 0.0


# ============================================================================
# 缺陷检测器
# ============================================================================

class DefectDetector:
    """喷砂铜面缺陷检测器。

    组合三种检测方法（HSV 颜色分割、形态学 Top-hat、纹理能量阈值），
    对喷砂后 PCB 铜面进行全面的缺陷检测。

    Attributes:
        config: 缺陷检测配置字典。
        resolution_mm_per_pixel: 图像空间分辨率，用于像素→物理尺寸转换。
    """

    # 缺陷类型常量
    OXIDATION = "oxidation"
    EMBEDDING = "embedding"
    UNROUGHENED = "unroughened"
    SCRATCH = "scratch"

    def __init__(self, config: dict):
        defects_cfg = config.get("inspection", {}).get("defects", {})
        self.resolution_mm_per_pixel = config.get(
            "system", {}
        ).get("resolution_mm_per_pixel", 0.01)

        # --- 氧化斑参数 ---
        ox = defects_cfg.get("oxidation", {})
        self.ox_enabled = ox.get("enabled", True)
        self.ox_h_low, self.ox_h_high = ox.get("h_range", [10, 30])
        self.ox_s_min = ox.get("s_min", 25)
        self.ox_v_min = ox.get("v_min", 35)
        self.ox_area_min = ox.get("area_min_px", 50)

        # --- 磨料嵌入参数 ---
        emb = defects_cfg.get("embedding", {})
        self.emb_enabled = emb.get("enabled", True)
        self.emb_kernel = tuple(emb.get("kernel_size", [15, 15]))
        self.emb_thresh = emb.get("threshold_binary", 30)
        self.emb_area_min = emb.get("area_min_px", 5)

        # --- 未粗化参数 ---
        unrough = defects_cfg.get("unroughened", {})
        self.unrough_enabled = unrough.get("enabled", True)
        self.unrough_energy_threshold = unrough.get("energy_threshold", 0.12)
        self.unrough_area_min = unrough.get("area_min_px", 100)

    # ------------------------------------------------------------------
    # 主检测入口
    # ------------------------------------------------------------------

    def detect_all(
        self,
        image: np.ndarray,
        texture_features=None,
    ) -> List[Defect]:
        """运行全部可用的缺陷检测器。

        Args:
            image: 输入图像（BGR，RGB，或灰度均可，方法内部自动转换）。
            texture_features: 可选的预计算 TextureFeatureVector（含局部熵图）。

        Returns:
            所有检测到的缺陷的列表。
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        all_defects: List[Defect] = []

        # 确保有灰度图像
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)

        # (1) 氧化斑检测（需要彩色图像）
        if self.ox_enabled and image.ndim == 3:
            oxidation_defects = self.detect_oxidation(image)
            all_defects.extend(oxidation_defects)

        # (2) 磨料嵌入检测
        if self.emb_enabled:
            embedding_defects = self.detect_abrasive_embedding(gray)
            all_defects.extend(embedding_defects)

        # (3) 未粗化区域检测
        if self.unrough_enabled:
            # 如果有预计算的纹理特征则使用，否则需要计算
            entropy_map = None
            if texture_features is not None:
                entropy_map = texture_features.local_entropy_map
            unrough_defects = self.detect_unroughened(gray, entropy_map)
            all_defects.extend(unrough_defects)

        return all_defects

    # ------------------------------------------------------------------
    # (1) 氧化斑 / 污渍检测
    # ------------------------------------------------------------------

    def detect_oxidation(self, image: np.ndarray) -> List[Defect]:
        """HSV 颜色空间分割检测氧化斑和污渍。

        原理：
            正常喷砂铜面呈金黄色（H ≈ 15°-30°），氧化后颜色
            变深变暗（向红/褐/灰偏移），且饱和度降低。

        流程：
            RGB → HSV → 颜色阈值（H范围 + Smin + Vmin）
            → 形态学清理 → 连通域分析 → 面积筛选

        Args:
            image: BGR 或 RGB 图像 (H, W, 3)。

        Returns:
            氧化斑缺陷列表。
        """
        if image.ndim != 3:
            raise ValueError("氧化斑检测需要彩色图像（3 通道）")

        # 根据通道顺序判断，统一转 RGB 再转 HSV
        # OpenCV 默认是 BGR
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 \
            else image
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # 颜色阈值
        h_mask_low = cv2.inRange(hsv, np.array([self.ox_h_low, self.ox_s_min, self.ox_v_min]),
                                  np.array([self.ox_h_high, 255, 255]))

        # 氧化区域还可能呈现暗色（低 V）或低饱和度
        h_mask_dark = cv2.inRange(hsv,
                                   np.array([0, 0, 0]),
                                   np.array([180, self.ox_s_min, self.ox_v_min]))
        mask = cv2.bitwise_or(h_mask_low, h_mask_dark)

        # 形态学清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 连通域分析
        defects = self._extract_connected_components(
            mask, self.OXIDATION, self.ox_area_min,
        )

        return defects

    # ------------------------------------------------------------------
    # (2) 磨料嵌入检测（形态学 Top-hat）
    # ------------------------------------------------------------------

    def detect_abrasive_embedding(self, gray: np.ndarray) -> List[Defect]:
        """形态学白顶帽变换检测磨料嵌入。

        原理：
            磨料颗粒嵌入铜面后形成微小的突起/亮点，
            与周围粗糙纹理形成对比。白顶帽变换
            (I - opening(I)) 可突出这些小型明亮结构。

        流程：
            灰度图 → 高斯去噪 → White Top-hat → 二值化
            → 连通域分析 → 计数统计

        Args:
            gray: 灰度图像 (H, W), uint8。

        Returns:
            磨料嵌入缺陷列表。
        """
        # 轻量去噪
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        # 白顶帽变换：原图 - 开运算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.emb_kernel)
        tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)

        # 二值化
        _, binary = cv2.threshold(
            tophat, self.emb_thresh, 255, cv2.THRESH_BINARY
        )

        # 连通域分析
        defects = self._extract_connected_components(
            binary, self.EMBEDDING, self.emb_area_min,
        )

        # 磨料嵌入的特点：面积小、数量多 → severity 与数量相关
        # 这里不做额外标记，留待 QualityAssessor 综合评估
        return defects

    # ------------------------------------------------------------------
    # (3) 未粗化区域检测
    # ------------------------------------------------------------------

    def detect_unroughened(
        self,
        gray: np.ndarray,
        entropy_map: Optional[np.ndarray] = None,
    ) -> List[Defect]:
        """纹理能量阈值分割检测未粗化区域。

        原理：
            喷砂粗化后铜面形成微米级锚纹，纹理能量高；
            未粗化区域保持轧制/蚀刻后的光滑状态，纹理能量低。

            使用 GLCM 能量（ASM）的局部映射作为纹理能量指标，
            低于阈值的区域被判定为"未粗化"。

            如果提供了预计算的局部熵图（entropy_map），则用其替代
            实时计算的纹理能量图。

        流程：
            灰度图 → 局部熵/能量计算 → 阈值分割
            → 连通域分析 → 面积筛选 → 严重度评估

        Args:
            gray: 灰度图像 (H, W), uint8。
            entropy_map: 预计算的局部熵图 (H, W)。

        Returns:
            未粗化区域缺陷列表。
        """
        if entropy_map is not None:
            texture_energy = entropy_map
        else:
            # 快速版局部能量：使用标准差滤波器
            kernel_size = 9
            local_mean = cv2.blur(gray.astype(np.float64),
                                   (kernel_size, kernel_size))
            local_sq = cv2.blur((gray.astype(np.float64)) ** 2,
                                (kernel_size, kernel_size))
            local_var = np.maximum(local_sq - local_mean ** 2, 0)
            local_std = np.sqrt(local_var)
            # 归一化
            s_max = np.percentile(local_std[local_std > 0], 95) \
                if np.any(local_std > 0) else 1.0
            texture_energy = local_std / (s_max + 1e-10)

        # 纹理能量阈值：低能量 → 未粗化
        binary = (texture_energy < self.unrough_energy_threshold).astype(np.uint8) * 255

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 连通域分析
        defects = self._extract_connected_components(
            binary, self.UNROUGHENED, self.unrough_area_min,
        )

        # 未粗化区域严重度：面积越大越严重
        total_pixels = gray.size
        for d in defects:
            d.severity = min(1.0, d.area_pixels / max(total_pixels * 0.01, 1))

        return defects

    # ------------------------------------------------------------------
    # (4) 划痕检测（扩展）
    # ------------------------------------------------------------------

    def detect_scratches(
        self,
        gray: np.ndarray,
        direction_map: Optional[np.ndarray] = None,
    ) -> List[Defect]:
        """检测喷砂表面的线性划痕/沟槽。

        使用 Canny 边缘检测 + 霍夫线检测，寻找不自然的直线结构。
        天然喷砂纹理不会产生规则的长直线。

        Args:
            gray: 灰度图像。
            direction_map: 可选的 Gabor 方向响应图。

        Returns:
            划痕缺陷列表（可能为空）。
        """
        # 边缘检测
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50,
            minLineLength=30, maxLineGap=10,
        )

        defects = []
        if lines is None:
            return defects

        # 为每条检测到的线创建缺陷条目
        mask = np.zeros(gray.shape, dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)

        if np.any(mask):
            defects = self._extract_connected_components(
                mask, self.SCRATCH, area_min_px=30,
            )

        return defects

    # ------------------------------------------------------------------
    # 通用连通域分析
    # ------------------------------------------------------------------

    def _extract_connected_components(
        self, binary: np.ndarray, defect_type: str, area_min_px: int,
    ) -> List[Defect]:
        """从二值掩膜提取连通域并创建 Defect 对象列表。

        Args:
            binary: 二值图像（白色=缺陷区域）。
            defect_type: 缺陷类型字符串。
            area_min_px: 最小面积过滤阈值（像素）。

        Returns:
            过滤后的缺陷列表。
        """
        if binary.dtype != np.uint8:
            binary = binary.astype(np.uint8)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8,
        )

        defects = []
        for i in range(1, n_labels):  # 跳过背景（标签 0）
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < area_min_px:
                continue

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[i]

            # 生成该连通域的二值掩膜
            component_mask = (labels == i).astype(np.uint8) * 255

            # 计算面积（mm²）
            area_mm2 = area * (self.resolution_mm_per_pixel ** 2)

            # 估计严重度（基于面积百分比）
            total_pixels = binary.size
            severity = min(1.0, area / max(total_pixels * 0.001, 1))

            bbox = (x, y, x + w, y + h)

            defects.append(Defect(
                type=defect_type,
                mask=component_mask,
                area_pixels=area,
                area_mm2=round(area_mm2, 4),
                centroid=(int(cx), int(cy)),
                bbox=bbox,
                severity=round(severity, 4),
                confidence=0.8,  # 基础置信度，分类器可更新
            ))

        return defects

    # ------------------------------------------------------------------
    # 缺陷可视化
    # ------------------------------------------------------------------

    def draw_defects(
        self,
        image: np.ndarray,
        defects: List[Defect],
    ) -> np.ndarray:
        """在图像上叠加缺陷标注。

        Args:
            image: 输入图像（RGB 或 BGR）。
            defects: 缺陷列表。

        Returns:
            带缺陷标注的图像。
        """
        vis = image.copy()
        if vis.dtype == np.float64 or vis.dtype == np.float32:
            vis = (vis * 255).astype(np.uint8)

        colors = {
            self.OXIDATION: (0, 0, 255),     # 红色 — 氧化斑
            self.EMBEDDING: (0, 255, 255),    # 青色 — 磨料嵌入
            self.UNROUGHENED: (255, 0, 0),    # 蓝色 — 未粗化区域
            self.SCRATCH: (255, 0, 255),      # 品红 — 划痕
        }

        for d in defects:
            color = colors.get(d.type, (128, 128, 128))
            x1, y1, x2, y2 = d.bbox

            # 边界框
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # 半透明掩膜叠加
            if d.mask is not None and d.mask.shape == vis.shape[:2]:
                overlay = vis.copy()
                overlay[d.mask > 0] = color
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

            # 标签
            label = f"{d.type}"
            if d.area_mm2 > 0:
                label += f" {d.area_mm2:.1f}mm²"
            cv2.putText(
                vis, label, (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

        return vis

    def generate_defect_heatmap(
        self,
        image_shape: Tuple[int, int],
        defects: List[Defect],
    ) -> np.ndarray:
        """生成缺陷分布热力图。

        将各缺陷的掩膜累加生成热力图，用于可视化缺陷密集区域。

        Args:
            image_shape: 图像尺寸 (H, W)。
            defects: 缺陷列表。

        Returns:
            热力图 (H, W), float64，取值 [0, 1]。
        """
        heatmap = np.zeros(image_shape, dtype=np.float64)
        if not defects:
            return heatmap

        for d in defects:
            if d.mask is not None and d.mask.shape == image_shape:
                heatmap[d.mask > 0] += d.severity

        # 归一化
        h_max = heatmap.max()
        if h_max > 1e-6:
            heatmap /= h_max

        return heatmap
