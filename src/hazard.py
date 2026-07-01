
"""
Simple collision and pedestrian warning logic.
"""

import cv2


class HazardDetector:
    def __init__(self):
        self.warning = ""

    def process(self, frame, detections):
        self.warning = ""
        h, w = frame.shape[:2]
        center_x = w // 2

        for obj in detections["objects"]:
            label = obj["label"]
            x1, y1, x2, y2 = obj["box"]
            obj_center = (x1 + x2) // 2
            box_width = x2 - x1
            distance = obj.get("distance", 999.0)

            if label in {"car", "truck", "bus", "motorcycle"}:
                in_lane = abs(obj_center - center_x) < w * 0.18
                close = distance < 9.0 or box_width > w * 0.28
                if in_lane and close:
                    self.warning = "COLLISION RISK"

            if label == "person":
                in_path = abs(obj_center - center_x) < w * 0.24
                if in_path:
                    self.warning = "PEDESTRIAN AHEAD"

            cv2.putText(
                frame,
                f"{distance:.1f} m",
                (x1, min(h - 8, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 240, 255),
                2,
                cv2.LINE_AA,
            )

        return frame

    def draw_warning(self, frame):
        if not self.warning:
            return frame

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 58), (w, h), (0, 0, 190), -1)
        cv2.putText(
            frame,
            self.warning,
            (22, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame