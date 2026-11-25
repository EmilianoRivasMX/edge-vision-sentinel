"""
Unit tests for MLOps Statistical Drift Detector and Baseline Profiler.
"""

from src.agent.schemas import TelemetryBatch, TelemetryFrame
from src.mlops.baseline_profile import BaselineProfile, get_default_golden_baseline
from src.mlops.drift_detector import DriftDetector


def test_golden_baseline_defaults():
    baseline = get_default_golden_baseline()
    assert baseline.sample_count > 0
    assert len(baseline.sharpness_samples) == baseline.sample_count
    assert len(baseline.brightness_samples) == baseline.sample_count
    assert len(baseline.confidence_samples) == baseline.sample_count
    assert baseline.sharpness_mean > 0.0


def test_baseline_from_batch(mock_healthy_batch):
    baseline = BaselineProfile.from_batch(mock_healthy_batch, profile_name="test-prof")
    assert baseline.profile_name == "test-prof"
    assert baseline.sample_count == len(mock_healthy_batch.frames)
    assert 70.0 <= baseline.sharpness_mean <= 78.0


def test_drift_detector_nominal_batch(mock_healthy_batch):
    detector = DriftDetector()
    report = detector.analyze_batch(mock_healthy_batch)

    assert not report.sharpness_drift
    assert not report.brightness_drift
    assert not report.confidence_drift
    assert not report.latency_anomaly
    assert not report.fps_drop
    assert report.overall_health_score >= 0.80


def test_drift_detector_sharpness_degradation(mock_healthy_batch):
    # Alter frames to simulate severe optical blur
    degraded_frames = []
    for f in mock_healthy_batch.frames:
        degraded_frames.append(
            TelemetryFrame(
                frame_id=f.frame_id,
                timestamp=f.timestamp,
                latency_ms=f.latency_ms,
                fps=f.fps,
                sharpness=1.2,  # Very blurry
                brightness_mean=f.brightness_mean,
                brightness_std=f.brightness_std,
                confidence_score=0.25,
                detections_count=0,
            )
        )
    degraded_batch = TelemetryBatch(batch_id="degraded-blur", frames=degraded_frames)

    detector = DriftDetector()
    report = detector.analyze_batch(degraded_batch)

    assert report.sharpness_drift is True
    assert report.confidence_drift is True
    assert "OPTICAL_SHARPNESS_DEGRADED" in report.summary_flags
    assert report.overall_health_score < 0.60


def test_drift_detector_latency_spike(mock_healthy_batch):
    spiked_frames = []
    for i, f in enumerate(mock_healthy_batch.frames):
        spiked_frames.append(
            TelemetryFrame(
                frame_id=f.frame_id,
                timestamp=f.timestamp,
                latency_ms=45.0,  # 45ms is 10x normal
                fps=10.0,  # severe FPS drop
                sharpness=f.sharpness,
                brightness_mean=f.brightness_mean,
                brightness_std=f.brightness_std,
                confidence_score=f.confidence_score,
                detections_count=f.detections_count,
            )
        )
    spiked_batch = TelemetryBatch(batch_id="latency-spike", frames=spiked_frames)

    detector = DriftDetector()
    report = detector.analyze_batch(spiked_batch)

    assert report.latency_anomaly is True
    assert report.fps_drop is True
    assert report.overall_health_score < 0.80


def test_drift_detector_empty_batch():
    empty_batch = TelemetryBatch(batch_id="empty-batch", frames=[])
    detector = DriftDetector()
    report = detector.analyze_batch(empty_batch)

    assert report.overall_health_score == 0.0
    assert "EMPTY_TELEMETRY_BATCH" in report.summary_flags
