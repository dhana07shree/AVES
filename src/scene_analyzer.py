

"""
Scene analysis for adaptive driving-video enhancement.
"""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class SceneInfo:
    mode: str
    brightness: float
    contrast: float
    saturation: float
    glare_percent: float
    dark_percent: float
    exposure: str


class SceneAnalyzer:
    def __init__(self):
        self.day_threshold = 82
        self.dark_threshold = 72

    def analyze(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        saturation = float(np.mean(hsv[:, :, 1]))
        glare_percent = float(np.mean(hsv[:, :, 2] > 225) * 100.0)
        dark_percent = float(np.mean(gray < 45) * 100.0)

        if brightness > 205 or glare_percent > 10:
            exposure = "OVEREXPOSED"
        elif brightness < self.dark_threshold or dark_percent > 45:
            exposure = "UNDEREXPOSED"
        else:
            exposure = "NORMAL"

        if brightness >= self.day_threshold or saturation > 38:
            mode = "DAY"
        else:
            mode = "NIGHT"

        return SceneInfo(
            mode=mode,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            glare_percent=glare_percent,
            dark_percent=dark_percent,
            exposure=exposure,
        )