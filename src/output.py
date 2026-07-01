
"""
Video writing and clean AVES dashboard.
"""

import os
import cv2
import numpy as np

import src.config as config


class OutputManager:
    def __init__(self):
        self.comparison_writer = None
        self.enhanced_writer = None

    def initialize_writer(self, width, height, fps):
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if config.SAVE_COMPARISON_VIDEO:
            self.comparison_writer = cv2.VideoWriter(
                config.OUTPUT_COMPARISON,
                fourcc,
                fps,
                (width * 2, height),
            )

        if config.SAVE_ENHANCED_VIDEO:
            self.enhanced_writer = cv2.VideoWriter(
                config.OUTPUT_ENHANCED,
                fourcc,
                fps,
                (width, height),
            )

    def save(self, enhanced, comparison):
        if self.enhanced_writer is not None:
            self.enhanced_writer.write(enhanced)
        if self.comparison_writer is not None:
            self.comparison_writer.write(comparison)

    def release(self):
        if self.comparison_writer is not None:
            self.comparison_writer.release()
        if self.enhanced_writer is not None:
            self.enhanced_writer.release()

    def comparison(self, original, enhanced):
        if original.shape[:2] != enhanced.shape[:2]:
            enhanced = cv2.resize(
                enhanced,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        return np.hstack((original, enhanced))

    def draw_dashboard(self, frame, scene, detection, fps):
        overlay = frame.copy()
        panel_w = 240
        panel_h = 188
        x = frame.shape[1] - panel_w - 12
        y = 12

        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (28, 28, 28), -1)
        frame = cv2.addWeighted(overlay, 0.58, frame, 0.42, 0)

        white = (255, 255, 255)
        green = (0, 255, 120)
        cyan = (0, 220, 255)

        cv2.putText(
            frame,
            "AVES",
            (x + 14, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            cyan,
            2,
            cv2.LINE_AA,
        )

        data = [
            ("Mode", scene.mode),
            ("Brightness", f"{scene.brightness:.0f}"),
            ("Exposure", scene.exposure),
            ("Vehicles", str(detection["vehicles"])),
            ("Persons", str(detection["persons"])),
            ("Traffic", str(detection["traffic_lights"])),
            ("FPS", f"{fps:.1f}"),
        ]

        yy = y + 58
        for key, value in data:
            cv2.putText(
                frame,
                key,
                (x + 14, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                white,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                str(value),
                (x + 126, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                green,
                2,
                cv2.LINE_AA,
            )
            yy += 23

        return frame

    def draw_labels(self, comparison):
        h, w = comparison.shape[:2]
        half = w // 2

        cv2.rectangle(comparison, (0, 0), (half, 42), (35, 35, 35), -1)
        cv2.rectangle(comparison, (half, 0), (w, 42), (35, 35, 35), -1)

        cv2.putText(
            comparison,
            "ORIGINAL",
            (20, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            comparison,
            "ENHANCED (AVES)",
            (half + 20, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 230, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.line(comparison, (half, 0), (half, h), (255, 255, 255), 2)
        return comparison