"""
Unit tests for Edge Vision Processor and Perturbation Engine.
"""

import numpy as np
import pytest

from src.config import settings
from src.vision.perturbations import PerturbationConfig, PerturbationEngine
from src.vision.processor import VisionProcessor


def test_compute_sharpness(synthetic_sharp_frame, synthetic_blank_frame):
    vp = VisionProcessor()
    sharp_gray = np.mean(synthetic_sharp_frame, axis=2).astype(np.uint8)
    blank_gray = np.mean(synthetic_blank_frame, axis=2).astype(np.uint8)

    sharp_val = vp.compute_sharpness(sharp_gray)
    blank_val = vp.compute_sharpness(blank_gray)

    assert sharp_val > 1000.0, "High contrast grid should have high Laplacian variance"
    assert blank_val == 0.0, "Uniform gray image must have zero variance"


def test_compute_luminance(synthetic_blank_frame):
    vp = VisionProcessor()
    blank_gray = np.mean(synthetic_blank_frame, axis=2).astype(np.uint8)
    mean_val, std_val = vp.compute_luminance(blank_gray)

    assert mean_val == 128.0
    assert std_val == 0.0


def test_perturbation_blur(synthetic_sharp_frame):
    engine = PerturbationEngine(
        PerturbationConfig(blur=True, blur_kernel_size=31, blur_sigma=10.0)
    )
    perturbed = engine.apply(synthetic_sharp_frame, frame_id=0)

    vp = VisionProcessor()
    orig_sharp = vp.compute_sharpness(
        np.mean(synthetic_sharp_frame, axis=2).astype(np.uint8)
    )
    blur_sharp = vp.compute_sharpness(np.mean(perturbed, axis=2).astype(np.uint8))

    assert blur_sharp < orig_sharp * 0.1, (
        "Gaussian blur must reduce Laplacian variance by >90%"
    )


def test_perturbation_lighting_drop(synthetic_blank_frame):
    engine = PerturbationEngine(
        PerturbationConfig(lighting_drop=True, lighting_factor=0.20)
    )
    perturbed = engine.apply(synthetic_blank_frame, frame_id=0)

    vp = VisionProcessor()
    mean_lum, _ = vp.compute_luminance(np.mean(perturbed, axis=2).astype(np.uint8))
    assert 20.0 <= mean_lum <= 30.0, "Lighting drop must scale 128 down to ~25"


def test_perturbation_glare(synthetic_blank_frame):
    engine = PerturbationEngine(PerturbationConfig(glare=True, glare_intensity=240))
    perturbed = engine.apply(synthetic_blank_frame, frame_id=0)

    vp = VisionProcessor()
    mean_lum, _ = vp.compute_luminance(np.mean(perturbed, axis=2).astype(np.uint8))
    assert mean_lum > 130.0, "Glare must increase overall luminance"


def test_perturbation_start_frame_delay(synthetic_sharp_frame):
    engine = PerturbationEngine(PerturbationConfig(blur=True, start_frame=5))
    frame_0 = engine.apply(synthetic_sharp_frame, frame_id=0)
    frame_5 = engine.apply(synthetic_sharp_frame, frame_id=5)

    assert np.array_equal(frame_0, synthetic_sharp_frame)
    assert not np.array_equal(frame_5, synthetic_sharp_frame)


def test_stream_video_generator():
    vp = VisionProcessor()
    video_path = settings.DEFAULT_VIDEO_PATH
    if not video_path.exists():
        pytest.skip(f"Video asset not present at {video_path}")

    frames = list(vp.stream_video(video_path, max_frames=5))
    assert len(frames) == 5
    assert frames[0].frame_id == 0
    assert frames[0].sharpness > 0.0
    assert frames[0].fps > 0.0


def test_process_batch():
    vp = VisionProcessor()
    video_path = settings.DEFAULT_VIDEO_PATH
    if not video_path.exists():
        pytest.skip(f"Video asset not present at {video_path}")

    batch = vp.process_batch(video_path, batch_size=10, batch_id="test-batch-01")
    assert batch.batch_id == "test-batch-01"
    assert len(batch.frames) == 10
    assert batch.mean_latency_ms > 0.0
    assert batch.mean_fps > 0.0
    assert batch.mean_sharpness > 0.0
