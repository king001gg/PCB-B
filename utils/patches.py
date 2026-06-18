"""Shared monkey-patches for PyTorch 2.6+ compatibility.

PyTorch 2.6 changed torch.load default to weights_only=True, which breaks
ultralytics YOLO model loading. Apply once at import time.
"""

import torch
import ultralytics.nn.tasks as tasks

_original_torch_safe_load = tasks.torch_safe_load


def _patched_torch_safe_load(file):
    """Patched loader that forces weights_only=False for YOLO model files."""
    return torch.load(file, map_location="cpu", weights_only=False), file


def apply_patches():
    """Apply all monkey-patches. Idempotent — safe to call multiple times."""
    if tasks.torch_safe_load is not _patched_torch_safe_load:
        tasks.torch_safe_load = _patched_torch_safe_load