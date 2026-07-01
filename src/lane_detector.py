
"""
Optional simple lane overlay.

It is intentionally not used in the main pipeline because too many overlays can
make a judge demo look messy. Keep this for future experiments.
"""

import cv2
import numpy as np


class LaneDetector:
    def region_of_interest(self, image):
        height, width = image.shape
        mask = np.zeros_like(image)
        polygon = np.array(
            [[
                (0, height),
                (width, height),
                (int(width * 0.58), int(height * 0.55)),
                (int(width * 0.42), int(height * 0.55)),
            ]],
            np.int32,
        )
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(image, mask)

    def detect(self, frame):
        output = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 160)
        roi = self.region_of_interest(edges)

        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=55,
            minLineLength=55,
            maxLineGap=120,
        )

        if lines is not None:
            overlay = frame.copy()
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 3)
            output = cv2.addWeighted(output, 0.82, overlay, 0.18, 0)

        return output