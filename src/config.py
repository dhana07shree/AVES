
"""
AVES configuration.

This version is tuned for an edge-AI hackathon demo: strong visual recovery,
minimal clutter, CPU-friendly defaults, and clean output videos.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

DAY_VIDEO = os.path.join(DATA_DIR, "day_samples", "sample.mp4")
NIGHT_VIDEO = os.path.join(DATA_DIR, "night_samples", "sample.mp4")
DEFAULT_VIDEO = NIGHT_VIDEO

WINDOW_NAME = "AVES - Adaptive Vision Enhancement System"
EXIT_KEY = 27

PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 360

SAVE_COMPARISON_VIDEO = True
SAVE_ENHANCED_VIDEO = True
SHOW_PREVIEW = True

OUTPUT_COMPARISON = os.path.join(OUTPUT_DIR, "comparison.mp4")
OUTPUT_ENHANCED = os.path.join(OUTPUT_DIR, "enhanced.mp4")

YOLO_MODEL = os.path.join(BASE_DIR, "yolov8n.pt")
CONFIDENCE_THRESHOLD = 0.35
ENABLE_DETECTION = True
DETECT_EVERY_N_FRAMES = 2

EDGE_MODE = True

CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
GLARE_VALUE_THRESHOLD = 228
HEADLIGHT_VALUE_THRESHOLD = 238
DARK_THRESHOLD = 75