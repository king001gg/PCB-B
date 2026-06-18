"""Image preprocessing utilities for PCB defect detection.

PCB images often have low contrast, making subtle defects (spurs, open circuits)
hard to detect.  The functions here can be applied before inference to improve
detection accuracy on difficult images.
"""

import cv2
import numpy as np
from typing import Tuple


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement (useful for low-contrast PCB images).

    Args:
        image: RGB image as uint8 numpy array (H, W, 3).

    Returns:
        Contrast-enhanced RGB image (uint8).
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def denoise(image: np.ndarray) -> np.ndarray:
    """Lightweight denoising (Gaussian + median blur).

    Args:
        image: RGB image as uint8 numpy array (H, W, 3).

    Returns:
        Denoised RGB image (uint8).
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.medianBlur(image, 3)
    return image


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
) -> np.ndarray:
    """Resize an image to the target size (RGB, uint8 or float32)."""
    return cv2.resize(image, target_size)