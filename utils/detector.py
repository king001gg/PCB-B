import os
import cv2
import numpy as np
import yaml
import torch
from typing import List, Dict, Tuple, Optional

from utils.patches import apply_patches
apply_patches()

from ultralytics import YOLO
from utils.preprocess import enhance_contrast

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PCBDefectDetector:
    """PCB defect detector using YOLOv8.

    Loads a trained YOLO model and runs inference on PCB images,
    returning an annotated image and a list of detected defects.
    """

    def __init__(self, config_path: str = "config.yaml"):
        if not os.path.isabs(config_path):
            config_path = os.path.join(PROJECT_ROOT, config_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # --- Model loading ---
        model_path = self.config['model_path']
        if not os.path.isabs(model_path):
            model_path = os.path.join(PROJECT_ROOT, model_path)

        self.device = self._resolve_device(
            self.config.get('device', 'auto')
        )
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # --- Config ---
        self.class_names = self.config['class_names']
        self.conf_threshold = self.config.get('confidence_threshold', 0.5)
        input_size = self.config.get('input_size', {})
        self.imgsz = (
            input_size.get('width', 640),
            input_size.get('height', 640),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, image_path: str, preprocess: bool = False
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Run detection on a single image.

        Args:
            image_path: Path to the PCB image.
            preprocess: If True, apply CLAHE contrast enhancement before
                        inference (useful for low-contrast PCB images).

        Returns:
            (annotated_image_rgb, detections_list)
        """
        # --- Preprocessing (optional) ---
        source = image_path
        if preprocess:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = enhance_contrast(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            source = img

        # --- Inference ---
        results = self.model(
            source,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        # Reuse YOLO's decoded image instead of re-reading from disk
        result = results[0]
        image = result.orig_img.copy()                     # BGR (numpy)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # → RGB

        # --- Parse detections ---
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Safety: skip out-of-range class indices
                if cls >= len(self.class_names):
                    continue

                class_name = self.class_names[cls]

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': round(conf, 4),
                    'class_name': class_name,
                    'class_id': cls,
                })

                self._draw_box(image, x1, y1, x2, y2, cls, class_name, conf)

        return image, detections

    def detect_batch(
        self, image_paths: List[str]
    ) -> List[Tuple[np.ndarray, List[Dict]]]:
        """Run detection on multiple images (batched)."""
        return [self.detect(p) for p in image_paths]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_box(
        self, image: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        cls: int, class_name: str, conf: float
    ) -> None:
        """Draw a single bounding box + label on the image (in-place)."""
        color = self._get_color(cls)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name}: {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        # Filled background rectangle for text readability
        cv2.rectangle(
            image,
            (x1, y1 - th - baseline - 6),
            (x1 + tw, y1),
            color, -1,
        )
        cv2.putText(
            image, label, (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
        )

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Return a distinct BGR colour for each defect class."""
        colors = [
            (255, 0, 0),    # red
            (0, 255, 0),    # green
            (0, 0, 255),    # blue
            (255, 255, 0),  # yellow
            (255, 0, 255),  # magenta
            (0, 255, 255),  # cyan
            (128, 0, 0),    # maroon
            (0, 128, 0),    # dark green
        ]
        return colors[class_id % len(colors)]

    @staticmethod
    def _resolve_device(device_setting: str) -> str:
        """Resolve 'auto'/'cpu'/'cuda'/'0' to a torch device string."""
        setting = str(device_setting).strip().lower()
        if setting in ('', 'auto'):
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        if setting == 'cuda' and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available — falling back to CPU")
            return 'cpu'
        return setting