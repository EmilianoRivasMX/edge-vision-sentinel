"""
Synthetic perturbation engine for Edge Vision Sentinel.
Simulates real-world physical and hardware failure modes:
1. Optical defocus / dirty lens (Gaussian blur)
2. Sudden lighting failure / underexposure
3. Extreme industrial glare / specular reflection
4. Sensor occlusion / physical obstruction
5. Edge compute thermal throttling / latency jitter
"""

import time

import cv2
import numpy as np
from pydantic import BaseModel, Field


class PerturbationConfig(BaseModel):
    """Specification for injecting controlled visual or hardware anomalies."""

    blur: bool = Field(default=False, description="Simulate dirty or defocused lens")
    blur_kernel_size: int = Field(
        default=35, description="Odd kernel size for Gaussian blur"
    )
    blur_sigma: float = Field(
        default=15.0, description="Standard deviation for Gaussian blur"
    )

    lighting_drop: bool = Field(
        default=False, description="Simulate plant lighting outage or blackout"
    )
    lighting_factor: float = Field(
        default=0.20, description="Multiplier for pixel luminance (0.0 to 1.0)"
    )

    glare: bool = Field(
        default=False, description="Simulate reflective glare or welding wash"
    )
    glare_intensity: int = Field(
        default=220, description="Added pixel intensity for specular spot"
    )

    occlusion: bool = Field(
        default=False, description="Simulate physical lens obstruction"
    )
    occlusion_ratio: float = Field(
        default=0.45, description="Fraction of frame obscured"
    )

    hardware_jitter_ms: float = Field(
        default=0.0, description="Added processing delay in ms"
    )
    start_frame: int = Field(
        default=0, description="Frame index from which perturbations activate"
    )


class PerturbationEngine:
    """Applies controlled physical and computing perturbations to raw video frames."""

    def __init__(self, config: PerturbationConfig | None = None) -> None:
        self.config = config or PerturbationConfig()

    def apply(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        """Apply active perturbations if current frame meets or exceeds start_frame."""
        if frame_id < self.config.start_frame:
            return frame

        modified = frame.copy()

        # 1. Optical Defocus / Dirt
        if self.config.blur:
            k = self.config.blur_kernel_size
            if k % 2 == 0:
                k += 1
            modified = cv2.GaussianBlur(modified, (k, k), self.config.blur_sigma)

        # 2. Lighting Drop / Underexposure
        if self.config.lighting_drop:
            modified = np.clip(
                modified.astype(np.float32) * self.config.lighting_factor, 0, 255
            ).astype(np.uint8)

        # 3. Specular Glare / Flash
        if self.config.glare:
            h, w = modified.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (int(w * 0.5), int(h * 0.5))
            radius = int(min(h, w) * 0.35)
            cv2.circle(mask, center, radius, 255, -1)
            mask_blurred = cv2.GaussianBlur(mask, (51, 51), 25)
            glare_layer = np.full_like(modified, self.config.glare_intensity)
            norm_mask = mask_blurred[:, :, np.newaxis] / 255.0
            modified = np.clip(
                modified * (1.0 - norm_mask) + glare_layer * norm_mask, 0, 255
            ).astype(np.uint8)

        # 4. Lens Occlusion
        if self.config.occlusion:
            h, w = modified.shape[:2]
            occ_h = int(h * self.config.occlusion_ratio)
            occ_w = int(w * self.config.occlusion_ratio)
            modified[0:occ_h, 0:occ_w] = 10  # Dark obstruction (grease/tape)

        # 5. Hardware Bottleneck / Latency Jitter
        if self.config.hardware_jitter_ms > 0:
            time.sleep(self.config.hardware_jitter_ms / 1000.0)

        return modified
