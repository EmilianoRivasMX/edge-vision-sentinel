"""
Shared Pytest fixtures for Edge Vision Sentinel test suite.
"""

import numpy as np
import pytest

from src.agent.schemas import TelemetryBatch, TelemetryFrame
from src.mlops.baseline_profile import BaselineProfile, get_default_golden_baseline


@pytest.fixture
def synthetic_sharp_frame() -> np.ndarray:
    """Creates a high-contrast synthetic image with sharp geometric edges."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add high frequency black and white checkerboard patterns
    for i in range(0, 480, 40):
        for j in range(0, 640, 40):
            if (i // 40 + j // 40) % 2 == 0:
                img[i : i + 40, j : j + 40] = 255
    return img


@pytest.fixture
def synthetic_blank_frame() -> np.ndarray:
    """Creates a completely uniform gray image (zero sharpness / edges)."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def golden_baseline() -> BaselineProfile:
    """Returns standard golden baseline."""
    return get_default_golden_baseline()


@pytest.fixture
def mock_healthy_batch() -> TelemetryBatch:
    """Creates a mock TelemetryBatch representing healthy nominal conditions."""
    frames = []
    for i in range(30):
        frames.append(
            TelemetryFrame(
                frame_id=i,
                timestamp=1700000000.0 + i * 0.033,
                latency_ms=3.1 + (i % 3) * 0.2,
                fps=30.0,
                sharpness=72.0 + (i % 5) * 1.5,
                brightness_mean=115.0 + (i % 4) * 1.0,
                brightness_std=12.0,
                confidence_score=0.85,
                detections_count=2,
            )
        )
    return TelemetryBatch(
        node_id="test-node",
        batch_id="batch-mock-healthy",
        frames=frames,
        mean_latency_ms=3.3,
        p95_latency_ms=3.6,
        mean_fps=30.0,
        mean_sharpness=74.0,
        mean_brightness=116.0,
        mean_confidence=0.85,
    )
