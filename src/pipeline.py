
"""
Main AVES video pipeline.
"""

import time
import cv2

import src.config as config
from src.scene_analyzer import SceneAnalyzer
from src.enhancement import ImageEnhancer
from src.detect import ObjectDetector
from src.output import OutputManager
from src.hazard import HazardDetector


class VideoPipeline:
    def __init__(self, source=config.DEFAULT_VIDEO):
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise FileNotFoundError(
                f"Unable to open video source: {source}. "
                "Put your test video in data/day_samples/sample.mp4 or change DEFAULT_VIDEO."
            )

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps <= 0:
            self.video_fps = 30

        self.scene = SceneAnalyzer()
        self.enhancer = ImageEnhancer()
        self.detector = ObjectDetector()
        self.hazard = HazardDetector()
        self.output = OutputManager()

        self.prev_time = time.time()
        self.fps = 0.0
        self.frame_index = 0
        self.last_detection = None

        self.output.initialize_writer(config.PROCESS_WIDTH, config.PROCESS_HEIGHT, self.video_fps)

    def calculate_fps(self):
        now = time.time()
        delta = max(1e-6, now - self.prev_time)
        instant = 1.0 / delta
        self.fps = instant if self.fps == 0 else (0.85 * self.fps + 0.15 * instant)
        self.prev_time = now

    def resize_frame(self, frame):
        return cv2.resize(
            frame,
            (config.PROCESS_WIDTH, config.PROCESS_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    def process_frame(self, frame):
        scene = self.scene.analyze(frame)
        enhanced = self.enhancer.enhance(frame, scene)

        run_detection = (
            config.ENABLE_DETECTION
            and self.frame_index % max(1, config.DETECT_EVERY_N_FRAMES) == 0
        )

        if run_detection or self.last_detection is None:
            detection = self.detector.detect(enhanced)
            self.last_detection = detection
        else:
            detection = self.last_detection
            detection = {**detection, "frame": enhanced.copy()}

        enhanced = detection["frame"]
        enhanced = self.hazard.process(enhanced, detection)
        enhanced = self.hazard.draw_warning(enhanced)
        enhanced = self.output.draw_dashboard(enhanced, scene, detection, self.fps)

        comparison = self.output.comparison(frame, enhanced)
        comparison = self.output.draw_labels(comparison)
        self.output.save(enhanced, comparison)
        return comparison

    def prepare_preview(self, frame):
        return cv2.resize(
            frame,
            (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.calculate_fps()
            frame = self.resize_frame(frame)
            output = self.process_frame(frame)
            self.frame_index += 1

            if config.SHOW_PREVIEW:
                cv2.imshow(config.WINDOW_NAME, self.prepare_preview(output))
                key = cv2.waitKey(1) & 0xFF
                if key == config.EXIT_KEY:
                    break

        self.cap.release()
        self.output.release()
        if config.SHOW_PREVIEW:
            cv2.destroyAllWindows()

        print(f"Enhanced video saved to: {config.OUTPUT_ENHANCED}")
        print(f"Comparison video saved to: {config.OUTPUT_COMPARISON}")