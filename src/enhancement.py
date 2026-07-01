
"""
AVES adaptive enhancement.

This version protects over-bright headlights and glare before improving
midtone contrast. The important idea is: do not sharpen or CLAHE the white
patches. Recover road/object detail around them instead.
"""

import cv2
import numpy as np

import src.config as config


class ImageEnhancer:
    def __init__(self):
        self.clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_GRID_SIZE,
        )

    def automatic_white_balance(self, frame):
        result = frame.astype(np.float32)
        b, g, r = cv2.split(result)

        mean_b = np.mean(b)
        mean_g = np.mean(g)
        mean_r = np.mean(r)

        gray_mean = (mean_b + mean_g + mean_r) / 3.0

        b *= gray_mean / (mean_b + 1e-6)
        g *= gray_mean / (mean_g + 1e-6)
        r *= gray_mean / (mean_r + 1e-6)

        return np.clip(cv2.merge((b, g, r)), 0, 255).astype(np.uint8)

    def gamma(self, frame, gamma_value):
        gamma_value = max(0.25, min(3.0, float(gamma_value)))
        inv_gamma = 1.0 / gamma_value

        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype(np.uint8)

        return cv2.LUT(frame, table)

    def bright_protection_mask(
        self,
        frame,
        value_threshold=225,
        saturation_threshold=135
    ):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        mask = (
            (v > value_threshold) &
            (s < saturation_threshold)
        ).astype(np.uint8) * 255

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        )

        mask = cv2.dilate(
            mask,
            np.ones((7, 7), np.uint8),
            iterations=1
        )

        return mask

    def compress_bright_regions(self, frame, mask, strength=0.58):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)

        compressed_v = v.astype(np.float32)
        compressed_v[mask > 0] = compressed_v[mask > 0] * strength
        compressed_v = np.clip(compressed_v, 0, 255).astype(np.uint8)

        compressed = cv2.cvtColor(
            cv2.merge((h, s, compressed_v)),
            cv2.COLOR_HSV2BGR
        )

        soft = cv2.GaussianBlur(mask, (0, 0), 5).astype(np.float32) / 255.0
        soft = soft[:, :, None]

        blended = (
            compressed.astype(np.float32) * soft +
            frame.astype(np.float32) * (1.0 - soft)
        )

        return np.clip(blended, 0, 255).astype(np.uint8)

    def luminance_clahe(self, frame, clip_limit=None, protect_mask=None):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        original_l = l.copy()

        if clip_limit is None:
            enhanced_l = self.clahe.apply(l)
        else:
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=(8, 8)
            )
            enhanced_l = clahe.apply(l)

        if protect_mask is not None:
            soft = cv2.GaussianBlur(
                protect_mask,
                (0, 0),
                7
            ).astype(np.float32) / 255.0

            l = (
                enhanced_l.astype(np.float32) * (1.0 - soft) +
                original_l.astype(np.float32) * soft
            )

            l = np.clip(l, 0, 255).astype(np.uint8)
        else:
            l = enhanced_l

        return cv2.cvtColor(
            cv2.merge((l, a, b)),
            cv2.COLOR_LAB2BGR
        )

    def denoise(self, frame, strength=2):
        return cv2.fastNlMeansDenoisingColored(
            frame,
            None,
            strength,
            strength,
            7,
            21
        )

    def unsharp_mask(
        self,
        frame,
        amount=0.55,
        radius=1.0,
        protect_mask=None
    ):
        blurred = cv2.GaussianBlur(frame, (0, 0), radius)

        sharp = cv2.addWeighted(
            frame,
            1.0 + amount,
            blurred,
            -amount,
            0
        )

        sharp = np.clip(sharp, 0, 255).astype(np.uint8)

        if protect_mask is None:
            return sharp

        soft = cv2.GaussianBlur(
            protect_mask,
            (0, 0),
            5
        ).astype(np.float32) / 255.0

        soft = soft[:, :, None]

        protected = (
            sharp.astype(np.float32) * (1.0 - soft) +
            frame.astype(np.float32) * soft
        )

        return np.clip(protected, 0, 255).astype(np.uint8)

    def improve_shadow_detail(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)

        shadow = v < 95

        boosted = self.gamma(
            cv2.merge((v, v, v)),
            1.35
        )[:, :, 0]

        v[shadow] = cv2.addWeighted(
            v,
            0.35,
            boosted,
            0.65,
            0
        )[shadow]

        return cv2.cvtColor(
            cv2.merge((h, s, v)),
            cv2.COLOR_HSV2BGR
        )

    def natural_color_limit(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)

        s = np.clip(
            s.astype(np.float32) * 0.95,
            0,
            255
        ).astype(np.uint8)

        return cv2.cvtColor(
            cv2.merge((h, s, v)),
            cv2.COLOR_HSV2BGR
        )

    def enhance_day(self, frame, scene=None):
        enhanced = self.automatic_white_balance(frame)

        protect_mask = self.bright_protection_mask(
            enhanced,
            value_threshold=config.GLARE_VALUE_THRESHOLD,
            saturation_threshold=130
        )

        enhanced = self.compress_bright_regions(
            enhanced,
            protect_mask,
            strength=0.62
        )

        enhanced = self.improve_shadow_detail(enhanced)

        enhanced = self.luminance_clahe(
            enhanced,
            clip_limit=1.8,
            protect_mask=protect_mask
        )

        if scene and scene.exposure == "OVEREXPOSED":
            enhanced = self.gamma(enhanced, 0.88)
        else:
            enhanced = self.gamma(enhanced, 1.02)

        enhanced = cv2.bilateralFilter(
            enhanced,
            5,
            28,
            28
        )

        enhanced = self.unsharp_mask(
            enhanced,
            amount=0.58,
            radius=0.9,
            protect_mask=protect_mask
        )

        return self.natural_color_limit(enhanced)

    def enhance_night(self, frame, scene=None):
        protect_mask = self.bright_protection_mask(
            frame,
            value_threshold=config.HEADLIGHT_VALUE_THRESHOLD,
            saturation_threshold=145
        )

        enhanced = self.compress_bright_regions(
            frame,
            protect_mask,
            strength=0.50
        )

        enhanced = self.gamma(enhanced, 1.28)

        enhanced = self.luminance_clahe(
            enhanced,
            clip_limit=1.9,
            protect_mask=protect_mask
        )

        enhanced = self.denoise(enhanced, strength=2)

        enhanced = cv2.bilateralFilter(
            enhanced,
            5,
            24,
            24
        )

        enhanced = self.unsharp_mask(
            enhanced,
            amount=0.70,
            radius=0.85,
            protect_mask=protect_mask
        )

        return self.natural_color_limit(enhanced)

    def enhance(self, frame, scene):
        if scene.mode == "DAY":
            return self.enhance_day(frame, scene)

        return self.enhance_night(frame, scene)