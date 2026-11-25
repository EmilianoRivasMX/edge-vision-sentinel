"""
Diagnostic Agent Tools for Edge Vision Sentinel.
Provides modular, decorated @tool functions for evaluating edge telemetry,
identifying root causes (optical, hardware, data drift), and formulating remediation actions.
Compatible with Strands Agents and standalone deterministic execution.
"""

from collections.abc import Callable
from typing import Any

from src.agent.schemas import AnomalyCategory, RemediationAction
from src.config import settings


def tool(func: Callable) -> Callable:
    """Decorator to mark functions as agent-executable tools (compatible with Strands)."""
    func.__tool_metadata__ = {
        "name": func.__name__,
        "description": func.__doc__ or "",
    }
    return func


@tool
def analyze_optical_health(
    mean_sharpness: float,
    mean_brightness: float,
    sharpness_drift: bool,
    brightness_drift: bool,
) -> dict[str, Any]:
    """
    Evaluates physical lens and lighting conditions.
    Detects blur (dirty lens/defocus), severe underexposure, or extreme glare.
    """
    is_anomaly = False
    anomaly_subtype = "NORMAL"
    severity = "NONE"
    reasons = []

    if sharpness_drift or mean_sharpness < settings.SHARPNESS_BLUR_THRESHOLD:
        is_anomaly = True
        anomaly_subtype = "OPTICAL_BLUR_OR_CONTAMINATION"
        severity = (
            "CRITICAL"
            if mean_sharpness < (settings.SHARPNESS_BLUR_THRESHOLD * 0.5)
            else "WARNING"
        )
        reasons.append(
            f"Sharpness metric ({mean_sharpness}) fell below threshold ({settings.SHARPNESS_BLUR_THRESHOLD}). "
            "Indicates dirty optics, lens smudge, or focus drift."
        )

    if (
        brightness_drift
        or mean_brightness < settings.LUMINANCE_LOW_THRESHOLD
        or mean_brightness > settings.LUMINANCE_HIGH_THRESHOLD
    ):
        is_anomaly = True
        if mean_brightness < settings.LUMINANCE_LOW_THRESHOLD:
            anomaly_subtype = "UNDEREXPOSURE_LIGHTING_FAILURE"
            severity = "CRITICAL"
            reasons.append(
                f"Luminance ({mean_brightness}) below lower bound ({settings.LUMINANCE_LOW_THRESHOLD})."
            )
        else:
            anomaly_subtype = "SPECULAR_GLARE_OR_WASHOUT"
            severity = "WARNING"
            reasons.append(
                f"Luminance ({mean_brightness}) exceeded glare bound ({settings.LUMINANCE_HIGH_THRESHOLD})."
            )

    return {
        "is_optical_anomaly": is_anomaly,
        "anomaly_subtype": anomaly_subtype,
        "severity": severity,
        "details": " | ".join(reasons)
        if reasons
        else "Optical and illumination profiles nominal.",
    }


@tool
def analyze_hardware_health(
    p95_latency_ms: float,
    mean_fps: float,
    latency_anomaly: bool,
    fps_drop: bool,
) -> dict[str, Any]:
    """
    Evaluates edge hardware computational health.
    Detects thermal throttling, OS thread starvation, and sensor frame drop.
    """
    is_anomaly = False
    severity = "NONE"
    reasons = []

    if fps_drop or mean_fps < settings.FPS_MIN_THRESHOLD:
        is_anomaly = True
        severity = (
            "CRITICAL" if mean_fps < (settings.FPS_MIN_THRESHOLD * 0.7) else "WARNING"
        )
        reasons.append(
            f"FPS ({mean_fps}) dropped below minimum acceptable threshold ({settings.FPS_MIN_THRESHOLD})."
        )

    if latency_anomaly or p95_latency_ms > 20.0:
        is_anomaly = True
        severity = "WARNING" if severity == "NONE" else severity
        reasons.append(
            f"P95 per-frame latency ({p95_latency_ms}ms) elevated beyond nominal baseline."
        )

    return {
        "is_hardware_anomaly": is_anomaly,
        "severity": severity,
        "details": " | ".join(reasons)
        if reasons
        else "Edge compute latency and frame rates nominal.",
    }


@tool
def assess_covariate_data_drift(
    confidence_drift: bool,
    mean_confidence: float,
    baseline_confidence: float,
    is_optical_issue: bool,
) -> dict[str, Any]:
    """
    Evaluates statistical covariate shift in model output confidence.
    Distinguishes physical lens degradation from genuine data distribution shift.
    """
    is_drift = False
    reasons = []

    if confidence_drift or (
        mean_confidence < baseline_confidence - settings.CONFIDENCE_DROP_THRESHOLD
    ):
        is_drift = True
        reasons.append(
            f"Mean prediction confidence dropped from {baseline_confidence} to {mean_confidence}. "
            f"Statistically significant distribution shift confirmed."
        )

    # Distinguish root causes:
    # If optical issue is present, confidence drop is a symptom of physical blur/lighting.
    # If optics are healthy but confidence dropped, it indicates genuine covariate data drift!
    requires_cloud_retrain = bool(is_drift and not is_optical_issue)

    return {
        "is_covariate_drift": is_drift,
        "requires_cloud_retraining": requires_cloud_retrain,
        "confidence_delta": round(mean_confidence - baseline_confidence, 3),
        "details": " | ".join(reasons)
        if reasons
        else "Prediction confidence matches reference distribution.",
    }


@tool
def generate_remediation_action(
    anomaly_category: AnomalyCategory,
    severity: str,
    node_id: str,
) -> dict[str, Any]:
    """
    Synthesizes the specific actionable remediation payload for the edge node or cloud MLOps orchestrator.
    """
    if anomaly_category == AnomalyCategory.OPTICAL_DEGRADATION:
        return {
            "recommended_action": RemediationAction.CLEAN_OPTICS_HARDWARE_ALERT,
            "target": "edge_maintenance_gateway",
            "payload": {
                "node_id": node_id,
                "dispatch_priority": "P1" if severity == "CRITICAL" else "P2",
                "instruction": "Inspect optical sensor. Clean lens element or adjust mechanical autofocus ring.",
            },
        }

    if anomaly_category == AnomalyCategory.LIGHTING_ANOMALY:
        return {
            "recommended_action": RemediationAction.EDGE_CAMERA_RECALIBRATE,
            "target": "camera_firmware_controller",
            "payload": {
                "node_id": node_id,
                "auto_exposure_recalibrate": True,
                "target_ev_compensation": "+1.5"
                if "UNDEREXPOSURE" in severity
                else "-1.5",
            },
        }

    if anomaly_category == AnomalyCategory.HARDWARE_BOTTLENECK:
        return {
            "recommended_action": RemediationAction.THROTTLE_STREAM_ALERT,
            "target": "edge_daemon_supervisor",
            "payload": {
                "node_id": node_id,
                "action": "reduce_capture_resolution_to_720p",
                "cool_down_requested": True,
            },
        }

    if anomaly_category == AnomalyCategory.COVARIATE_DATA_DRIFT:
        return {
            "recommended_action": RemediationAction.TRIGGER_GCP_VERTEX_RETRAIN_PIPELINE,
            "target": "gcp_vertex_ai_orchestrator",
            "payload": {
                "project_id": settings.GCP_PROJECT_ID,
                "region": settings.GCP_REGION,
                "pipeline_name": settings.VERTEX_PIPELINE_NAME,
                "trigger_reason": "Covariate data drift detected on edge node video feed",
                "dataset_snapshot_uri": f"gs://{settings.GCP_PROJECT_ID}-drift-snapshots/{node_id}/latest.parquet",
            },
        }

    return {
        "recommended_action": RemediationAction.NO_ACTION,
        "target": "telemetry_logger",
        "payload": {
            "status": "Nominal operational state. No corrective action required."
        },
    }
