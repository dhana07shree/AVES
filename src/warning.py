
"""
Compatibility wrapper for older code that imports WarningSystem.
"""

from src.hazard import HazardDetector


class WarningSystem(HazardDetector):
    def evaluate(self, detections, frame_shape):
        self.warning = ""
        h, w = frame_shape[:2]
        center_x = w // 2

        for obj in detections["objects"]:
            label = obj["label"]
            x1, _, x2, _ = obj["box"]
            obj_center = (x1 + x2) // 2
            box_width = x2 - x1

            if label in {"car", "truck", "bus", "motorcycle"}:
                if abs(obj_center - center_x) < w * 0.18 and box_width > w * 0.22:
                    self.warning = "COLLISION RISK"

            if label == "person" and abs(obj_center - center_x) < w * 0.22:
                self.warning = "PEDESTRIAN AHEAD"

        return self.warning

    def draw(self, frame):
        return self.draw_warning(frame)


