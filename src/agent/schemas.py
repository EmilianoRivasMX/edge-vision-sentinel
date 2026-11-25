"""
Pydantic v2 data models and schemas for Edge Vision Sentinel.
Defines telemetry structures, drift reports, and structured diagnostic outcomes.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeHealthStatus(str, Enum):
    """Overall operational health state of the edge vision node."""

    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AnomalyCategory(str, Enum):
    """Categorized root causes for vision telemetry deviations."""

    NORMAL = "NORMAL"
    OPTICAL_DEGRADATION = "OPTICAL_DEGRADATION"  # Blur, dirty lens, defocus
    LIGHTING_ANOMALY = "LIGHTING_ANOMALY"  # Glare, underexposure, outage
    HARDWARE_BOTTLENECK = (
        "HARDWARE_BOTTLENECK"  # Thermal throttle, FPS drop, latency spikes
    )
    COVARIATE_DATA_DRIFT = "COVARIATE_DATA_DRIFT"  # Model input distribution shift


class RemediationAction(str, Enum):
    """Prescribed autonomous action for the edge node or MLOps cloud orchestrator."""

    NO_ACTION = "NO_ACTION"
    CLEAN_OPTICS_HARDWARE_ALERT = "CLEAN_OPTICS_HARDWARE_ALERT"
    EDGE_CAMERA_RECALIBRATE = "EDGE_CAMERA_RECALIBRATE"
    THROTTLE_STREAM_ALERT = "THROTTLE_STREAM_ALERT"
    TRIGGER_GCP_VERTEX_RETRAIN_PIPELINE = "TRIGGER_GCP_VERTEX_RETRAIN_PIPELINE"


class TelemetryFrame(BaseModel):
    """Telemetry data extracted from an individual video frame."""

    frame_id: int = Field(..., description="Sequential index of the frame")
    timestamp: float = Field(..., description="Epoch timestamp of frame ingestion")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")
    fps: float = Field(..., description="Instantaneous moving average FPS")
    sharpness: float = Field(..., description="Laplacian variance focus metric")
    brightness_mean: float = Field(..., description="Mean pixel intensity (0-255)")
    brightness_std: float = Field(
        ..., description="Standard deviation of pixel intensity"
    )
    confidence_score: float = Field(
        ..., description="Detection model confidence proxy (0.0-1.0)"
    )
    detections_count: int = Field(default=0, description="Number of detected objects")


class TelemetryBatch(BaseModel):
    """Batch window of collected edge telemetry frames for statistical evaluation."""

    node_id: str = Field(default="edge-node-01")
    batch_id: str = Field(...)
    frames: list[TelemetryFrame] = Field(default_factory=list)
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    mean_fps: float = 0.0
    mean_sharpness: float = 0.0
    mean_brightness: float = 0.0
    mean_confidence: float = 0.0


class StatisticalDriftReport(BaseModel):
    """Outcome of 2-sample statistical hypothesis testing and outlier detection."""

    sharpness_drift: bool = False
    sharpness_pvalue: float = 1.0
    brightness_drift: bool = False
    brightness_pvalue: float = 1.0
    confidence_drift: bool = False
    confidence_pvalue: float = 1.0
    latency_anomaly: bool = False
    latency_zscore_max: float = 0.0
    fps_drop: bool = False
    min_fps_observed: float = 30.0
    overall_health_score: float = Field(
        1.0, description="1.0 (Optimal) to 0.0 (Severe Anomaly)"
    )
    summary_flags: list[str] = Field(default_factory=list)


class DiagnosticReport(BaseModel):
    """Structured Pydantic decision returned by the Diagnostic Agent."""

    timestamp: float = Field(...)
    node_id: str = Field(...)
    batch_id: str = Field(...)
    health_status: NodeHealthStatus = Field(...)
    anomaly_category: AnomalyCategory = Field(...)
    recommended_action: RemediationAction = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str = Field(...)
    root_cause_analysis: str = Field(...)
    remediation_payload: dict[str, Any] = Field(default_factory=dict)
    statistical_evidence: StatisticalDriftReport | None = None
