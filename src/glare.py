
"""
Glare analysis helper.
"""

import cv2
import numpy as np


class GlareAnalyzer:
    def __init__(self, threshold=230):
        self.threshold = threshold

    def analyze(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        mask = ((value > self.threshold) & (saturation < 130)).astype(np.uint8) * 255
        glare_percentage = float(np.mean(mask > 0) * 100.0)
        return {"mask": mask, "percentage": glare_percentage}