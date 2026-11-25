"""
Unit tests for Diagnostic Tools and Diagnostic Agent.
"""

from src.agent.diagnostic_agent import DiagnosticAgent
from src.agent.schemas import (
    AnomalyCategory,
    NodeHealthStatus,
    RemediationAction,
    StatisticalDriftReport,
    TelemetryBatch,
)
from src.agent.tools import (
    analyze_hardware_health,
    analyze_optical_health,
    assess_covariate_data_drift,
    generate_remediation_action,
)


def test_analyze_optical_health_blur():
    res = analyze_optical_health(
        mean_sharpness=10.0,
        mean_brightness=110.0,
        sharpness_drift=True,
        brightness_drift=False,
    )
    assert res["is_optical_anomaly"] is True
    assert res["anomaly_subtype"] == "OPTICAL_BLUR_OR_CONTAMINATION"
    assert res["severity"] == "CRITICAL"


def test_analyze_optical_health_lighting():
    res = analyze_optical_health(
        mean_sharpness=65.0,
        mean_brightness=20.0,
        sharpness_drift=False,
        brightness_drift=True,
    )
    assert res["is_optical_anomaly"] is True
    assert res["anomaly_subtype"] == "UNDEREXPOSURE_LIGHTING_FAILURE"
    assert res["severity"] == "CRITICAL"


def test_analyze_hardware_health_jitter():
    res = analyze_hardware_health(
        p95_latency_ms=35.0,
        mean_fps=12.0,
        latency_anomaly=True,
        fps_drop=True,
    )
    assert res["is_hardware_anomaly"] is True
    assert res["severity"] in ["WARNING", "CRITICAL"]


def test_assess_covariate_drift_distinction():
    # If optical is present, retrain should NOT be triggered (physical issue)
    res_opt = assess_covariate_data_drift(
        confidence_drift=True,
        mean_confidence=0.30,
        baseline_confidence=0.82,
        is_optical_issue=True,
    )
    assert res_opt["is_covariate_drift"] is True
    assert res_opt["requires_cloud_retraining"] is False

    # If optical is NOT present, retrain MUST be triggered (genuine data drift)
    res_drift = assess_covariate_data_drift(
        confidence_drift=True,
        mean_confidence=0.45,
        baseline_confidence=0.82,
        is_optical_issue=False,
    )
    assert res_drift["is_covariate_drift"] is True
    assert res_drift["requires_cloud_retraining"] is True


def test_remediation_actions_mapping():
    act_opt = generate_remediation_action(
        AnomalyCategory.OPTICAL_DEGRADATION, "CRITICAL", "node-01"
    )
    assert (
        act_opt["recommended_action"] == RemediationAction.CLEAN_OPTICS_HARDWARE_ALERT
    )

    act_light = generate_remediation_action(
        AnomalyCategory.LIGHTING_ANOMALY, "CRITICAL", "node-01"
    )
    assert act_light["recommended_action"] == RemediationAction.EDGE_CAMERA_RECALIBRATE

    act_hw = generate_remediation_action(
        AnomalyCategory.HARDWARE_BOTTLENECK, "WARNING", "node-01"
    )
    assert act_hw["recommended_action"] == RemediationAction.THROTTLE_STREAM_ALERT

    act_drift = generate_remediation_action(
        AnomalyCategory.COVARIATE_DATA_DRIFT, "WARNING", "node-01"
    )
    assert (
        act_drift["recommended_action"]
        == RemediationAction.TRIGGER_GCP_VERTEX_RETRAIN_PIPELINE
    )


def test_diagnostic_agent_healthy(mock_healthy_batch):
    agent = DiagnosticAgent()
    drift_rep = StatisticalDriftReport()
    report = agent.evaluate_batch(mock_healthy_batch, drift_rep)

    assert report.health_status == NodeHealthStatus.HEALTHY
    assert report.anomaly_category == AnomalyCategory.NORMAL
    assert report.recommended_action == RemediationAction.NO_ACTION
    assert report.confidence >= 0.90


def test_diagnostic_agent_covariate_drift(mock_healthy_batch):
    agent = DiagnosticAgent()
    # Batch has nominal optics and nominal latency, but confidence dropped
    batch_drift = TelemetryBatch(
        batch_id="covariate-drift-batch",
        frames=mock_healthy_batch.frames,
        mean_sharpness=72.0,
        mean_brightness=115.0,
        mean_latency_ms=3.0,
        p95_latency_ms=4.0,
        mean_fps=30.0,
        mean_confidence=0.55,  # Significant drop from 0.82
    )
    drift_rep = StatisticalDriftReport(confidence_drift=True)
    report = agent.evaluate_batch(batch_drift, drift_rep)

    assert report.health_status == NodeHealthStatus.WARNING
    assert report.anomaly_category == AnomalyCategory.COVARIATE_DATA_DRIFT
    assert (
        report.recommended_action
        == RemediationAction.TRIGGER_GCP_VERTEX_RETRAIN_PIPELINE
    )
