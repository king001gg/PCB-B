"""Training script for PCB defect detection (YOLOv8).

Reads hyperparameters from config.yaml so they can be tuned without
editing source code.  Run:  python train.py
"""

import os
import sys
import yaml
import torch

# Ensure project root on path so shared patch can be imported
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.patches import apply_patches
apply_patches()

from ultralytics import YOLO


def _load_config(path: str = "config.yaml") -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _resolve_device(device: str) -> str:
    """Resolve 'auto' → 'cpu' or 'cuda' (ultralytics requires explicit device)."""
    device = str(device).strip().lower()
    if device in ('', 'auto'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available — falling back to CPU")
        return 'cpu'
    return device


def train_model(config: dict | None = None):
    if config is None:
        config = _load_config()



    train_cfg = config.get('training', {})

    model = YOLO(train_cfg.get('pretrained_model', 'yolov8n.pt'))

    results = model.train(
        data=train_cfg.get('data_yaml', 'data.yaml'),
        epochs=train_cfg.get('epochs', 100),
        imgsz=train_cfg.get('imgsz', 640),
        batch=train_cfg.get('batch', 8),
        lr0=train_cfg.get('lr0', 0.01),
        device=_resolve_device(train_cfg.get('device', 'auto')),
        project=train_cfg.get('project', 'PCB_defect_detection'),
        name=train_cfg.get('name', 'yolov8n_pcb'),
        exist_ok=train_cfg.get('exist_ok', True),
        mosaic=train_cfg.get('mosaic', 0.0),
        patience=train_cfg.get('patience', 50),
    )

    # Validation after training
    model.val()

    # Export best model path back to config for detection
    best_pt = os.path.join(
        train_cfg.get('project', 'PCB_defect_detection'),
        train_cfg.get('name', 'yolov8n_pcb'),
        'weights', 'best.pt',
    )
    print(f"\nTraining complete. Best weights saved to: {best_pt}")
    return results


if __name__ == "__main__":
    train_model()