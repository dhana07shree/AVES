
"""
Optional YOLO object detection for AVES.

The enhancer must still work if YOLO or model weights are unavailable, because
edge demos often run on different laptops.
"""

import os
import cv2

import src.config as config


class ObjectDetector:
    def __init__(self, model_path=None, confidence=None):
        self.confidence = confidence or config.CONFIDENCE_THRESHOLD
        self.model_path = model_path or config.YOLO_MODEL
        self.model = None
        self.available = False
        self.allowed_classes = {
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "person",
            "traffic light",
            "stop sign",
        }

        self._load_model()

    def _load_model(self):
        if not config.ENABLE_DETECTION:
            return

        try:
            from ultralytics import YOLO

            path = self.model_path
            if not os.path.exists(path):
                path = "yolov8n.pt"

            self.model = YOLO(path)
            self.available = True
            print("YOLO detector loaded.")
        except Exception as exc:
            self.available = False
            print(f"YOLO detector disabled: {exc}")

    def empty_result(self, frame):
        return {
            "frame": frame,
            "objects": [],
            "vehicles": 0,
            "persons": 0,
            "traffic_lights": 0,
        }

    def estimate_distance(self, box_height):
        if box_height <= 0:
            return 999.0
        return round(1900.0 / box_height, 1)

    def detect(self, frame):
        if not self.available:
            return self.empty_result(frame)

        annotated = frame.copy()
        objects = []
        vehicle_count = 0
        person_count = 0
        traffic_light_count = 0

        results = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=640,
            device="cpu",
            verbose=False,
        )

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]
                if label not in self.allowed_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                box_height = y2 - y1
                distance = self.estimate_distance(box_height)

                if label in {"car", "truck", "bus", "motorcycle", "bicycle"}:
                    vehicle_count += 1
                    color = (0, 210, 255)
                elif label == "person":
                    person_count += 1
                    color = (0, 120, 255)
                else:
                    traffic_light_count += 1
                    color = (0, 255, 120)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                text = f"{label.title()} {conf:.2f}"
                cv2.putText(
                    annotated,
                    text,
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                objects.append(
                    {
                        "label": label,
                        "box": (x1, y1, x2, y2),
                        "distance": distance,
                        "confidence": conf,
                    }
                )

        return {
            "frame": annotated,
            "objects": objects,
            "vehicles": vehicle_count,
            "persons": person_count,
            "traffic_lights": traffic_light_count,
        }