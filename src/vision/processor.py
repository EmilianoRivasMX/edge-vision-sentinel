"""
High-performance headless OpenCV edge vision processing engine.
Extracts per-frame hardware and visual telemetry without GUI overhead:
- Processing latency (ms) & moving average FPS
- Focus / Sharpness metric (Laplacian variance: Var(nabla^2 I))
- Luminance distribution (mean & std of pixel intensities)
- Downstream inference confidence proxy
"""

import time
from collections import deque
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np

from src.agent.schemas import TelemetryBatch, TelemetryFrame
from src.config import settings
from src.vision.perturbations import PerturbationConfig, PerturbationEngine


class VisionProcessor:
    """Headless video stream analyzer that extracts real-time edge telemetry."""

    def __init__(
        self,
        fps_window_size: int = 15,
        target_fps: float = settings.TARGET_FPS,
    ) -> None:
        self.fps_window_size = fps_window_size
        self.target_fps = target_fps
        self._timestamp_history: deque[float] = deque(maxlen=fps_window_size)

    @staticmethod
    def compute_sharpness(frame_gray: np.ndarray) -> float:
        """
        Calculates image sharpness using the variance of the Laplacian operator:
        Var(nabla^2 I) = (1/N) * sum((L(x, y) - mu_L)^2)
        High values indicate sharp, in-focus edges; low values signify blur or dirty optics.
        """
        if frame_gray.size == 0:
            return 0.0
        laplacian = cv2.Laplacian(frame_gray, cv2.CV_64F)
        return float(laplacian.var())

    @staticmethod
    def compute_luminance(frame_gray: np.ndarray) -> tuple[float, float]:
        """Calculates the mean and standard deviation of grayscale pixel intensities (0-255)."""
        if frame_gray.size == 0:
            return 0.0, 0.0
        mean_val = float(np.mean(frame_gray))
        std_val = float(np.std(frame_gray))
        return mean_val, std_val

    def compute_confidence_proxy(
        self,
        sharpness: float,
        luminance_mean: float,
        frame_gray: np.ndarray,
    ) -> tuple[float, int]:
        """
        Calculates the downstream object detection confidence proxy.
        Models how optical degradation (severe blur or blackout) damages model predictions.
        """
        # Nominal baseline confidence of healthy model
        base_confidence = 0.92

        # Degradation penalty from blur
        blur_penalty = 0.0
        if sharpness < settings.SHARPNESS_BLUR_THRESHOLD:
            # Drop confidence proportionally as sharpness drops towards zero
            deficit = (
                settings.SHARPNESS_BLUR_THRESHOLD - sharpness
            ) / settings.SHARPNESS_BLUR_THRESHOLD
            blur_penalty = min(0.65, deficit * 0.70)

        # Degradation penalty from lighting failure or extreme glare
        lighting_penalty = 0.0
        if luminance_mean < settings.LUMINANCE_LOW_THRESHOLD:
            deficit = (
                settings.LUMINANCE_LOW_THRESHOLD - luminance_mean
            ) / settings.LUMINANCE_LOW_THRESHOLD
            lighting_penalty = min(0.60, deficit * 0.65)
        elif luminance_mean > settings.LUMINANCE_HIGH_THRESHOLD:
            excess = (luminance_mean - settings.LUMINANCE_HIGH_THRESHOLD) / (
                255.0 - settings.LUMINANCE_HIGH_THRESHOLD
            )
            lighting_penalty = min(0.55, excess * 0.60)

        final_confidence = float(
            np.clip(base_confidence - blur_penalty - lighting_penalty, 0.15, 0.99)
        )

        # Detect coarse motion / contours as proxy for active object count
        detections_count = 1
        if final_confidence < 0.40:
            detections_count = 0
        else:
            # Estimate foreground activity via edge energy
            edges = cv2.Canny(frame_gray, 50, 150)
            edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
            if edge_ratio > 0.03:
                detections_count = min(6, int(edge_ratio * 70))

        return final_confidence, detections_count

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
        start_time: float,
    ) -> TelemetryFrame:
        """Processes a single BGR frame and returns structured telemetry metrics."""
        # Convert to grayscale for optical calculations
        if len(frame_bgr.shape) == 3 and frame_bgr.shape[2] == 3:
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame_bgr

        # Compute optical metrics
        sharpness = self.compute_sharpness(frame_gray)
        lum_mean, lum_std = self.compute_luminance(frame_gray)

        # Compute confidence proxy
        confidence, detections = self.compute_confidence_proxy(
            sharpness, lum_mean, frame_gray
        )

        # Measure elapsed processing latency
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0

        # Update FPS moving window
        now = time.time()
        self._timestamp_history.append(now)
        if len(self._timestamp_history) >= 2:
            dt = self._timestamp_history[-1] - self._timestamp_history[0]
            calculated_fps = (
                (len(self._timestamp_history) - 1) / dt if dt > 0 else self.target_fps
            )
            fps = float(min(120.0, max(0.5, calculated_fps)))
        else:
            fps = self.target_fps

        return TelemetryFrame(
            frame_id=frame_id,
            timestamp=now,
            latency_ms=round(latency_ms, 2),
            fps=round(fps, 1),
            sharpness=round(sharpness, 2),
            brightness_mean=round(lum_mean, 2),
            brightness_std=round(lum_std, 2),
            confidence_score=round(confidence, 3),
            detections_count=detections,
        )

    def stream_video(
        self,
        video_source: str | Path | int,
        max_frames: int | None = None,
        perturbation: PerturbationConfig | None = None,
    ) -> Generator[TelemetryFrame, None, None]:
        """
        Streams and profiles frames from a video file, CCTV stream, or synthetic feed.
        Yields individual TelemetryFrame instances lazily.
        """
        engine = PerturbationEngine(perturbation)
        cap = cv2.VideoCapture(str(video_source))

        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open video source: {video_source}")

        frame_id = 0
        try:
            while True:
                if max_frames is not None and frame_id >= max_frames:
                    break

                t_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Apply synthetic perturbations if configured
                perturbed_frame = engine.apply(frame, frame_id)

                # Process frame telemetry
                telemetry = self.process_frame(perturbed_frame, frame_id, t_start)
                yield telemetry
                frame_id += 1
        finally:
            cap.release()

    def process_batch(
        self,
        video_source: str | Path | int,
        batch_size: int = settings.WINDOW_SIZE,
        start_frame: int = 0,
        perturbation: PerturbationConfig | None = None,
        batch_id: str = "batch-001",
    ) -> TelemetryBatch:
        """Captures a window of frames and aggregates them into a TelemetryBatch."""
        frames: list[TelemetryFrame] = []
        cap = cv2.VideoCapture(str(video_source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open video source: {video_source}")

        engine = PerturbationEngine(perturbation)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        try:
            while len(frames) < batch_size:
                t_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Loop video if reached end to satisfy batch_size
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                perturbed = engine.apply(frame, frame_idx)
                tf = self.process_frame(perturbed, frame_idx, t_start)
                frames.append(tf)
                frame_idx += 1
        finally:
            cap.release()

        if not frames:
            return TelemetryBatch(batch_id=batch_id, frames=[])

        latencies = [f.latency_ms for f in frames]
        fps_vals = [f.fps for f in frames]
        sharp_vals = [f.sharpness for f in frames]
        bright_vals = [f.brightness_mean for f in frames]
        conf_vals = [f.confidence_score for f in frames]

        return TelemetryBatch(
            node_id=settings.NODE_ID,
            batch_id=batch_id,
            frames=frames,
            mean_latency_ms=round(float(np.mean(latencies)), 2),
            p95_latency_ms=round(float(np.percentile(latencies, 95)), 2),
            mean_fps=round(float(np.mean(fps_vals)), 1),
            mean_sharpness=round(float(np.mean(sharp_vals)), 2),
            mean_brightness=round(float(np.mean(bright_vals)), 2),
            mean_confidence=round(float(np.mean(conf_vals)), 3),
        )
