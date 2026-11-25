"""
Golden Baseline Profiler for Edge Vision Sentinel.
Constructs, saves, and loads reference statistical distributions representing
a healthy edge camera and nominal factory operating environment.
"""

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from src.agent.schemas import TelemetryBatch


class BaselineProfile(BaseModel):
    """Statistical distribution profile of a calibrated, healthy edge camera stream."""

    profile_name: str = "golden-cctv-baseline"
    sample_count: int = 60

    # Sharpness / Focus distribution
    sharpness_mean: float = 72.0
    sharpness_std: float = 8.5
    sharpness_samples: list[float] = Field(default_factory=list)

    # Luminance / Lighting distribution
    brightness_mean: float = 120.0
    brightness_std: float = 15.0
    brightness_samples: list[float] = Field(default_factory=list)

    # Hardware & Performance statistics
    latency_mean_ms: float = 3.5
    latency_std_ms: float = 0.8
    latency_p95_ms: float = 4.8
    fps_mean: float = 30.0
    fps_std: float = 1.2

    # Model Confidence distribution
    confidence_mean: float = 0.82
    confidence_std: float = 0.05
    confidence_samples: list[float] = Field(default_factory=list)

    def save(self, filepath: Path) -> None:
        """Persists baseline profile to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, filepath: Path) -> "BaselineProfile":
        """Loads baseline profile from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_batch(
        cls, batch: TelemetryBatch, profile_name: str = "calibrated-baseline"
    ) -> "BaselineProfile":
        """Builds an empirical baseline from an ingested TelemetryBatch."""
        if not batch.frames:
            return cls(profile_name=profile_name)

        sharpness_list = [f.sharpness for f in batch.frames]
        brightness_list = [f.brightness_mean for f in batch.frames]
        latency_list = [f.latency_ms for f in batch.frames]
        fps_list = [f.fps for f in batch.frames]
        confidence_list = [f.confidence_score for f in batch.frames]

        return cls(
            profile_name=profile_name,
            sample_count=len(batch.frames),
            sharpness_mean=round(float(np.mean(sharpness_list)), 2),
            sharpness_std=round(float(np.std(sharpness_list)) or 1.0, 2),
            sharpness_samples=sharpness_list,
            brightness_mean=round(float(np.mean(brightness_list)), 2),
            brightness_std=round(float(np.std(brightness_list)) or 1.0, 2),
            brightness_samples=brightness_list,
            latency_mean_ms=round(float(np.mean(latency_list)), 2),
            latency_std_ms=round(float(np.std(latency_list)) or 0.5, 2),
            latency_p95_ms=round(float(np.percentile(latency_list, 95)), 2),
            fps_mean=round(float(np.mean(fps_list)), 1),
            fps_std=round(float(np.std(fps_list)) or 0.5, 2),
            confidence_mean=round(float(np.mean(confidence_list)), 3),
            confidence_std=round(float(np.std(confidence_list)) or 0.02, 3),
            confidence_samples=confidence_list,
        )


def get_default_golden_baseline() -> BaselineProfile:
    """Returns a synthetic golden reference baseline based on calibrated factory footage."""
    # Generate 50 synthetic baseline samples matching calibrated factory CCTV conditions
    np.random.seed(42)
    sharp_samples = (
        np.random.normal(loc=72.0, scale=6.0, size=50).clip(55.0, 95.0).tolist()
    )
    bright_samples = (
        np.random.normal(loc=115.0, scale=8.0, size=50).clip(90.0, 140.0).tolist()
    )
    conf_samples = (
        np.random.normal(loc=0.82, scale=0.03, size=50).clip(0.72, 0.95).tolist()
    )

    return BaselineProfile(
        profile_name="factory-cctv-golden-baseline",
        sample_count=50,
        sharpness_mean=72.0,
        sharpness_std=6.0,
        sharpness_samples=[round(x, 2) for x in sharp_samples],
        brightness_mean=115.0,
        brightness_std=8.0,
        brightness_samples=[round(x, 2) for x in bright_samples],
        latency_mean_ms=3.2,
        latency_std_ms=0.6,
        latency_p95_ms=4.5,
        fps_mean=30.0,
        fps_std=0.8,
        confidence_mean=0.82,
        confidence_std=0.03,
        confidence_samples=[round(x, 3) for x in conf_samples],
    )
