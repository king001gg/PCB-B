"""
Generate synthetic PCB images with annotated defects for demo/testing.

Creates realistic-looking PCB images with the 6 defect types:
  missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper

Each generated image is saved with a matching annotation file describing
the ground-truth bounding boxes.
"""

import os
import sys
import json
import random
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Board colours (BGR)
BOARD_GREEN_LIGHT = (160, 210, 120)
BOARD_GREEN_DARK = (90, 150, 60)
COPPER_COLOR = (180, 160, 100)       # gold-copper tone
PAD_COLOR = (170, 150, 80)
HOLE_COLOR = (40, 40, 30)            # dark hole
MASK_COLOR = (70, 130, 50)           # solder mask green
SILK_COLOR = (240, 240, 235)         # white silkscreen

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

IMAGE_SIZE = (800, 600)  # W, H


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _add_noise(img: np.ndarray, intensity: float = 8.0) -> None:
    """Add subtle Gaussian noise to simulate real-camera grain."""
    noise = np.random.normal(0, intensity, img.shape).astype(np.int16)
    img_i16 = img.astype(np.int16)
    np.add(img_i16, noise, out=img_i16)
    np.clip(img_i16, 0, 255, out=img_i16)
    img[:] = img_i16.astype(np.uint8)


def _draw_trace(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                thickness: int = 6, color: tuple = COPPER_COLOR) -> None:
    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def _draw_pad(img: np.ndarray, cx: int, cy: int, radius: int = 14,
              hole_radius: int = 5) -> None:
    """Draw a circular pad with a centre hole."""
    cv2.circle(img, (cx, cy), radius, COPPER_COLOR, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), radius, PAD_COLOR, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), hole_radius, HOLE_COLOR, -1, cv2.LINE_AA)


def _draw_ic_package(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """Draw an IC package footprint (rectangle + pins)."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), SILK_COLOR, 1)
    # Pins on left & right
    pin_count = max(2, h // 16)
    for i in range(pin_count):
        py = y + 10 + i * (h - 20) // (pin_count - 1) if pin_count > 1 else y + h // 2
        cv2.line(img, (x - 8, py), (x, py), COPPER_COLOR, 3, cv2.LINE_AA)
        cv2.line(img, (x + w, py), (x + w + 8, py), COPPER_COLOR, 3, cv2.LINE_AA)


def _draw_silkscreen_text(img: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, SILK_COLOR, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Board background
# ---------------------------------------------------------------------------

def make_board() -> np.ndarray:
    """Create a base PCB board image with solder-mask texture."""
    board = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3),
                    BOARD_GREEN_DARK, dtype=np.uint8)

    # Add subtle colour variation across the board
    for y in range(IMAGE_SIZE[1]):
        gradient = 1.0 - 0.15 * (y / IMAGE_SIZE[1])
        row_color = tuple(int(c * gradient) for c in BOARD_GREEN_LIGHT)
        board[y, :] = row_color

    _add_noise(board, intensity=5)
    return board


# ---------------------------------------------------------------------------
# Defect injectors  (each returns (x1, y1, x2, y2) bounding box)
# ---------------------------------------------------------------------------

def inject_missing_hole(img: np.ndarray, cx: int, cy: int) -> tuple:
    """Draw a pad WITHOUT its centre hole."""
    r = random.randint(12, 16)
    cv2.circle(img, (cx, cy), r, COPPER_COLOR, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, PAD_COLOR, 1, cv2.LINE_AA)
    # deliberately skip the hole
    return (cx - r, cy - r, cx + r, cy + r)


def inject_mouse_bite(img: np.ndarray, x1: int, y1: int,
                      x2: int, y2: int) -> tuple:
    """Cut a semi-circular notch into an existing trace edge."""
    # Pick a point on the trace and cut a semicircle
    t = random.uniform(0.25, 0.75)
    cx = int(x1 + t * (x2 - x1))
    cy = int(y1 + t * (y2 - y1))
    radius = random.randint(6, 10)
    # Draw board-coloured circle over the trace
    cv2.circle(img, (cx, cy), radius, BOARD_GREEN_DARK, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), radius, BOARD_GREEN_LIGHT, 1, cv2.LINE_AA)
    return (cx - radius, cy - radius, cx + radius, cy + radius)


def inject_open_circuit(img: np.ndarray, x1: int, y1: int,
                        x2: int, y2: int) -> tuple:
    """Erase a segment of a trace to simulate a broken connection."""
    t = random.uniform(0.35, 0.65)
    cx = int(x1 + t * (x2 - x1))
    cy = int(y1 + t * (y2 - y1))
    gap = random.randint(6, 10)
    # Overpaint with board green
    cv2.circle(img, (cx, cy), gap, BOARD_GREEN_DARK, -1, cv2.LINE_AA)
    return (cx - gap, cy - gap, cx + gap, cy + gap)


def inject_short(img: np.ndarray, tx1: int, ty1: int, tx2: int, ty2: int,
                 thickness: int = 6) -> tuple:
    """Draw an unintended copper bridge between two nearby traces."""
    # Place the short at the midpoint of the gap between traces
    mx = (tx1 + tx2) // 2
    my = (ty1 + ty2) // 2
    # Draw a small copper blob connecting them
    blob_r = thickness + random.randint(2, 5)
    cv2.circle(img, (mx, my), blob_r, COPPER_COLOR, -1, cv2.LINE_AA)
    return (mx - blob_r, my - blob_r, mx + blob_r, my + blob_r)


def inject_spur(img: np.ndarray, x1: int, y1: int,
                x2: int, y2: int) -> tuple:
    """Add a small copper protrusion from a trace edge."""
    t = random.uniform(0.3, 0.7)
    cx = int(x1 + t * (x2 - x1))
    cy = int(y1 + t * (y2 - y1))
    length = random.randint(10, 18)
    angle = random.choice([-1, 1]) * random.uniform(0.5, 1.2)  # perpendicular-ish
    ex = cx + int(length * np.cos(angle))
    ey = cy + int(length * np.sin(angle))
    cv2.line(img, (cx, cy), (ex, ey), COPPER_COLOR, 3, cv2.LINE_AA)
    return (min(cx, ex) - 2, min(cy, ey) - 2,
            max(cx, ex) + 2, max(cy, ey) + 2)


def inject_spurious_copper(img: np.ndarray, margin: int = 60) -> tuple:
    """Place a floating copper island in an empty area."""
    cx = random.randint(margin, IMAGE_SIZE[0] - margin)
    cy = random.randint(margin, IMAGE_SIZE[1] - margin)
    r = random.randint(8, 16)
    # Irregular blob via random polygon
    pts = []
    for i in range(8):
        a = 2 * np.pi * i / 8
        rr = r * random.uniform(0.6, 1.4)
        pts.append((int(cx + rr * np.cos(a)), int(cy + rr * np.sin(a))))
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [pts_arr], COPPER_COLOR)
    cv2.polylines(img, [pts_arr], True, PAD_COLOR, 1, cv2.LINE_AA)
    return (cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4)


# ---------------------------------------------------------------------------
# Layout generators — each returns the board + list of defect annotations
# ---------------------------------------------------------------------------

Annotation = dict  # {"bbox": (x1,y1,x2,y2), "class_name": str}


def generate_board_01() -> tuple[np.ndarray, list[Annotation]]:
    """Simple board with horizontal traces + pads.  Contains all 6 defects."""
    img = make_board()
    annotations = []

    # --- Healthy traces + pads ---
    trace_y_positions = [80, 140, 200, 260, 320, 380, 440, 500]
    for i, y in enumerate(trace_y_positions):
        _draw_trace(img, 50, y, 750, y, thickness=6)
        # Pads at both ends
        _draw_pad(img, 60, y, 14)
        _draw_pad(img, 740, y, 14)
        # Extra vias along the trace
        for vx in range(200, 700, 120):
            _draw_pad(img, vx, y, 10, 4)

    # --- Vertical bus traces ---
    vx_positions = [120, 680]
    for vx in vx_positions:
        _draw_trace(img, vx, 40, vx, 560, thickness=6)
        _draw_pad(img, vx, 40, 14)
        _draw_pad(img, vx, 560, 14)

    # IC packages
    _draw_ic_package(img, 280, 160, 60, 80)
    _draw_ic_package(img, 460, 340, 60, 80)
    _draw_silkscreen_text(img, "U1", 290, 152)
    _draw_silkscreen_text(img, "U2", 470, 332)

    # --- Inject defects ---
    # 1. missing_hole — pad at (200, 380) without hole
    b = inject_missing_hole(img, 320, 380)
    annotations.append({"bbox": b, "class_name": "missing_hole"})

    # 2. mouse_bite — notch on trace at y=200
    b = inject_mouse_bite(img, 200, 200, 500, 200)
    annotations.append({"bbox": b, "class_name": "mouse_bite"})

    # 3. open_circuit — broken trace at y=320
    b = inject_open_circuit(img, 100, 320, 700, 320)
    annotations.append({"bbox": b, "class_name": "open_circuit"})

    # 4. short — bridge between trace y=440 and a nearby pad
    b = inject_short(img, 300, 440, 300, 380)
    annotations.append({"bbox": b, "class_name": "short"})

    # 5. spur — protrusion from the trace at y=140
    b = inject_spur(img, 300, 140, 600, 140)
    annotations.append({"bbox": b, "class_name": "spur"})

    # 6. spurious_copper — floating copper blob
    b = inject_spurious_copper(img)
    annotations.append({"bbox": b, "class_name": "spurious_copper"})

    # Extra spurs for variety
    b2 = inject_spur(img, 50, 500, 750, 500)
    annotations.append({"bbox": b2, "class_name": "spur"})

    return img, annotations


def generate_board_02() -> tuple[np.ndarray, list[Annotation]]:
    """Denser board with diagonal routing + more vias."""
    img = make_board()
    annotations = []

    # Grid of pads (like a BGA area)
    for row in range(4):
        for col in range(6):
            px = 150 + col * 100
            py = 100 + row * 120
            _draw_pad(img, px, py, 12, 5)

    # Horizontal traces connecting pads
    for row in range(4):
        py = 100 + row * 120
        _draw_trace(img, 120, py, 680, py, thickness=5)
        _draw_pad(img, 120, py, 10, 4)
        _draw_pad(img, 680, py, 10, 4)

    # IC packages
    _draw_ic_package(img, 300, 520, 80, 50)

    # --- Defects ---
    b = inject_missing_hole(img, 350, 220)
    annotations.append({"bbox": b, "class_name": "missing_hole"})

    b = inject_mouse_bite(img, 150, 340, 600, 340)
    annotations.append({"bbox": b, "class_name": "mouse_bite"})

    b = inject_open_circuit(img, 120, 460, 680, 460)
    annotations.append({"bbox": b, "class_name": "open_circuit"})

    b = inject_short(img, 250, 100, 250, 220)
    annotations.append({"bbox": b, "class_name": "short"})

    b = inject_spur(img, 250, 460, 550, 460)
    annotations.append({"bbox": b, "class_name": "spur"})

    b = inject_spurious_copper(img)
    annotations.append({"bbox": b, "class_name": "spurious_copper"})

    return img, annotations


def generate_board_03() -> tuple[np.ndarray, list[Annotation]]:
    """Vertical-heavy layout with power traces."""
    img = make_board()
    annotations = []

    # Two thick power rails
    _draw_trace(img, 80, 50, 80, 550, thickness=12)
    _draw_trace(img, 720, 50, 720, 550, thickness=12)

    # Thinner signal traces
    for x in range(160, 680, 60):
        _draw_trace(img, x, 50, x, 550, thickness=4)
        _draw_pad(img, x, 50, 8, 3)
        _draw_pad(img, x, 550, 8, 3)

    # Horizontal jumpers
    for y in [120, 260, 400]:
        _draw_trace(img, 80, y, 720, y, thickness=4)

    _draw_ic_package(img, 340, 250, 120, 100)
    _draw_silkscreen_text(img, "MCU", 370, 310)

    # --- Defects ---
    b = inject_missing_hole(img, 400, 400)
    annotations.append({"bbox": b, "class_name": "missing_hole"})

    b = inject_mouse_bite(img, 80, 50, 80, 550)
    annotations.append({"bbox": b, "class_name": "mouse_bite"})

    b = inject_open_circuit(img, 280, 50, 280, 550)
    annotations.append({"bbox": b, "class_name": "open_circuit"})

    # short between two adjacent vertical traces
    b = inject_short(img, 520, 300, 580, 300)
    annotations.append({"bbox": b, "class_name": "short"})

    b = inject_spur(img, 80, 260, 720, 260)
    annotations.append({"bbox": b, "class_name": "spur"})

    b = inject_spurious_copper(img)
    annotations.append({"bbox": b, "class_name": "spurious_copper"})

    return img, annotations


def generate_board_04() -> tuple[np.ndarray, list[Annotation]]:
    """Mixed signal board — analog + digital sections."""
    img = make_board()
    annotations = []

    # Analog section (left) — curved-ish traces
    for i, y in enumerate([100, 180, 260, 340, 420]):
        _draw_trace(img, 40, y, 360, y, thickness=5)
        _draw_pad(img, 50, y, 10, 4)
        _draw_pad(img, 350, y, 10, 4)

    # Digital section (right) — dense parallel traces
    for x in range(430, 760, 35):
        _draw_trace(img, x, 60, x, 540, thickness=3)
        _draw_pad(img, x, 60, 7, 3)
        _draw_pad(img, x, 540, 7, 3)

    # Divider
    _draw_trace(img, 395, 60, 395, 540, thickness=10)

    _draw_ic_package(img, 140, 500, 80, 60)
    _draw_ic_package(img, 500, 280, 80, 80)
    _draw_silkscreen_text(img, "A1", 155, 530)
    _draw_silkscreen_text(img, "D1", 515, 325)

    # --- Defects ---
    b = inject_missing_hole(img, 200, 260)
    annotations.append({"bbox": b, "class_name": "missing_hole"})

    b = inject_mouse_bite(img, 40, 340, 360, 340)
    annotations.append({"bbox": b, "class_name": "mouse_bite"})

    b = inject_open_circuit(img, 430, 60, 430, 540)
    annotations.append({"bbox": b, "class_name": "open_circuit"})

    b = inject_short(img, 570, 300, 605, 300)
    annotations.append({"bbox": b, "class_name": "short"})

    b = inject_spur(img, 40, 420, 360, 420)
    annotations.append({"bbox": b, "class_name": "spur"})

    b = inject_spurious_copper(img)
    annotations.append({"bbox": b, "class_name": "spurious_copper"})

    # Extra missing hole for variety
    b2 = inject_missing_hole(img, 640, 540)
    annotations.append({"bbox": b2, "class_name": "missing_hole"})

    return img, annotations


def generate_board_05() -> tuple[np.ndarray, list[Annotation]]:
    """Board with large copper pours + thermal relief pads."""
    img = make_board()
    annotations = []

    # Copper pour (large filled area with clearance)
    pour_pts = np.array([
        (30, 30), (770, 30), (770, 570), (30, 570),
    ], dtype=np.int32)
    cv2.fillPoly(img, [pour_pts], BOARD_GREEN_DARK)

    # Thermal relief pads on the pour
    for px in range(100, 750, 80):
        for py in [80, 160, 240, 320, 400, 480]:
            if random.random() < 0.6:
                _draw_pad(img, px, py, 10, 4)

    # Thick power traces
    _draw_trace(img, 60, 300, 740, 300, thickness=14)
    _draw_trace(img, 400, 40, 400, 560, thickness=10)

    _draw_ic_package(img, 500, 80, 100, 100)
    _draw_silkscreen_text(img, "PWR", 525, 135)

    # --- Defects ---
    b = inject_missing_hole(img, 300, 300)
    annotations.append({"bbox": b, "class_name": "missing_hole"})

    b = inject_mouse_bite(img, 60, 300, 740, 300)
    annotations.append({"bbox": b, "class_name": "mouse_bite"})

    b = inject_open_circuit(img, 400, 40, 400, 560)
    annotations.append({"bbox": b, "class_name": "open_circuit"})

    b = inject_short(img, 400, 200, 480, 200)
    annotations.append({"bbox": b, "class_name": "short"})

    b = inject_spur(img, 60, 300, 740, 300)
    annotations.append({"bbox": b, "class_name": "spur"})

    b = inject_spurious_copper(img)
    annotations.append({"bbox": b, "class_name": "spurious_copper"})

    return img, annotations


def generate_board_06() -> tuple[np.ndarray, list[Annotation]]:
    """High-density board with many thin traces and small vias (SMD-style)."""
    img = make_board()
    annotations = []

    # Fine-pitch parallel traces
    for i, x in enumerate(range(40, 770, 18)):
        _draw_trace(img, x, 40, x, 560, thickness=2)
        if i % 3 == 0:
            _draw_pad(img, x, 50, 6, 2)
            _draw_pad(img, x, 550, 6, 2)

    # Horizontal bus
    for y in [100, 200, 300, 400, 500]:
        _draw_trace(img, 40, y, 760, y, thickness=3)

    # Small SMD pads
    for sx in range(100, 700, 50):
        for sy in [150, 350]:
            if random.random() < 0.5:
                cv2.rectangle(img, (sx - 8, sy - 4), (sx + 8, sy + 4),
                              COPPER_COLOR, -1)

    _draw_ic_package(img, 320, 60, 140, 40)
    _draw_silkscreen_text(img, "QFP", 360, 45)

    # --- Defects ---
    b = inject_missing_hole(img, 200, 300)
    annotations.append({"bbox": b, "class_name": "missing_hole"})

    b = inject_mouse_bite(img, 40, 400, 760, 400)
    annotations.append({"bbox": b, "class_name": "mouse_bite"})

    b = inject_open_circuit(img, 100, 200, 700, 200)
    annotations.append({"bbox": b, "class_name": "open_circuit"})

    b = inject_short(img, 400, 300, 418, 300)
    annotations.append({"bbox": b, "class_name": "short"})

    b = inject_spur(img, 40, 500, 760, 500)
    annotations.append({"bbox": b, "class_name": "spur"})

    b = inject_spurious_copper(img)
    annotations.append({"bbox": b, "class_name": "spurious_copper"})

    # Extra short
    b2 = inject_short(img, 580, 200, 598, 200)
    annotations.append({"bbox": b2, "class_name": "short"})

    return img, annotations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BOARDS = [
    ("board_01_simple", generate_board_01),
    ("board_02_grid", generate_board_02),
    ("board_03_vertical", generate_board_03),
    ("board_04_mixed_signal", generate_board_04),
    ("board_05_power", generate_board_05),
    ("board_06_dense", generate_board_06),
]


def main():
    # Fix encoding on Windows consoles
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"Generating {len(BOARDS)} synthetic PCB samples -> {OUTPUT_DIR}\n")

    summary = []

    for name, gen_func in BOARDS:
        img, annotations = gen_func()

        # Save image
        img_path = OUTPUT_DIR / f"{name}.jpg"
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [OK] {img_path.name}  ({len(annotations)} defects)")

        # Save ground-truth annotations as JSON
        ann_path = OUTPUT_DIR / f"{name}.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump({
                "image": f"{name}.jpg",
                "image_size": {"width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1]},
                "detections": annotations,
            }, f, indent=2, ensure_ascii=False)

        # Count defect types for summary
        counts = {}
        for a in annotations:
            counts[a["class_name"]] = counts.get(a["class_name"], 0) + 1
        summary.append((name, len(annotations), counts))

    # Print summary table
    print(f"\n{'='*62}")
    print(f"{'Sample':<26} {'Defects':>8}  {'Breakdown'}")
    print(f"{'='*62}")
    for name, total, counts in summary:
        parts = [f"{k}({v})" for k, v in sorted(counts.items())]
        print(f"  {name:<24} {total:>8}  {'  '.join(parts)}")
    print(f"{'='*62}")
    print(f"\nDone — {len(BOARDS)} images + {len(BOARDS)} annotation files written.")


if __name__ == "__main__":
    main()