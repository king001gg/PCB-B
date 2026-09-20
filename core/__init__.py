"""PCB 阻焊前喷砂质量检测 — 核心算法模块。

Public API:
    Preprocessor          — ROI 提取 + Retinex 光照校正 + CLAHE 增强
    TextureAnalyzer       — GLCM / LBP / Gabor 纹理特征提取
    DefectDetector        — 氧化斑 / 磨料嵌入 / 未粗化区域检测
    DefectClassifier      — SVM / MobileNet 缺陷分类
    QualityAssessor       — 多维度质量评分 + OK/NG 判定
    ColorAnalyzer         — 色度 / 饱和度（只监测，不进总分）
    InspectionPipeline    — 主流水线编排
    ImageAcquisition      — 图像采集（文件 / 相机）
    InspectionReport      — 检测报告生成
"""

from core.preprocessing import Preprocessor
from core.texture import (
    TextureAnalyzer,
    GLCMExtractor,
    LBPExtractor,
    GaborFilterBank,
    GLCMFeatures,
    TextureFeatureVector,
)
from core.defects import DefectDetector, Defect
from core.classifier import DefectClassifier, SVMClassifier, MobileNetClassifier
from core.quality import QualityAssessor, QualityReport
from core.color import ColorAnalyzer, ColorFeatures, ColorRegion
from core.pipeline import InspectionPipeline, InspectionResult
from core.acquisition import ImageAcquisition, FileAcquisition, CameraAcquisition
from core.reporter import InspectionReport

__all__ = [
    # Preprocessing
    "Preprocessor",
    # Texture
    "TextureAnalyzer",
    "GLCMExtractor",
    "LBPExtractor",
    "GaborFilterBank",
    "GLCMFeatures",
    "TextureFeatureVector",
    # Defects
    "DefectDetector",
    "Defect",
    # Classifier
    "DefectClassifier",
    "SVMClassifier",
    "MobileNetClassifier",
    # Quality
    "QualityAssessor",
    "QualityReport",
    # Color（只监测，不进总分）
    "ColorAnalyzer",
    "ColorFeatures",
    "ColorRegion",
    # Pipeline
    "InspectionPipeline",
    "InspectionResult",
    # Acquisition
    "ImageAcquisition",
    "FileAcquisition",
    "CameraAcquisition",
    # Reporter
    "InspectionReport",
]
