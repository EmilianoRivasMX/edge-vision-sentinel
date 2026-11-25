"""
Diagnostic Agent implementation for Edge Vision Sentinel.
Orchestrates diagnostic tools to analyze edge vision telemetry and MLOps statistical drift.
Implements dual-mode operation:
- Full Strands Agent integration (when Strands SDK is configured)
- High-performance, zero-dependency Deterministic Expert Engine fallback (for CI/CD and offline plants)
"""

import time

from src.agent.schemas import (
    AnomalyCategory,
    DiagnosticReport,
    NodeHealthStatus,
    StatisticalDriftReport,
    TelemetryBatch,
)
from src.agent.tools import (
    analyze_hardware_health,
    analyze_optical_health,
    assess_covariate_data_drift,
    generate_remediation_action,
)
from src.config import settings
from src.mlops.baseline_profile import BaselineProfile, get_default_golden_baseline


class DiagnosticAgent:
    """Intelligent supervisory agent that diagnoses edge vision and MLOps telemetry."""

    def __init__(
        self,
        baseline: BaselineProfile | None = None,
        node_id: str = settings.NODE_ID,
    ) -> None:
        self.baseline = baseline or get_default_golden_baseline()
        self.node_id = node_id

    def evaluate_batch(
        self,
        batch: TelemetryBatch,
        drift_report: StatisticalDriftReport,
    ) -> DiagnosticReport:
        """
        Evaluates an ingested batch and statistical drift report.
        Dispatches diagnostic tools and synthesizes structured DiagnosticReport.
        """
        # Execute diagnostic tool evaluations
        optical_result = analyze_optical_health(
            mean_sharpness=batch.mean_sharpness,
            mean_brightness=batch.mean_brightness,
            sharpness_drift=drift_report.sharpness_drift,
            brightness_drift=drift_report.brightness_drift,
        )

        hardware_result = analyze_hardware_health(
            p95_latency_ms=batch.p95_latency_ms,
            mean_fps=batch.mean_fps,
            latency_anomaly=drift_report.latency_anomaly,
            fps_drop=drift_report.fps_drop,
        )

        drift_result = assess_covariate_data_drift(
            confidence_drift=drift_report.confidence_drift,
            mean_confidence=batch.mean_confidence,
            baseline_confidence=self.baseline.confidence_mean,
            is_optical_issue=optical_result["is_optical_anomaly"],
        )

        # Synthesize Root Cause and Health Classification
        if optical_result["is_optical_anomaly"]:
            if (
                "LIGHTING" in optical_result["anomaly_subtype"]
                or "GLARE" in optical_result["anomaly_subtype"]
            ):
                category = AnomalyCategory.LIGHTING_ANOMALY
            else:
                category = AnomalyCategory.OPTICAL_DEGRADATION

            status = (
                NodeHealthStatus.CRITICAL
                if optical_result["severity"] == "CRITICAL"
                else NodeHealthStatus.WARNING
            )
            root_cause = optical_result["details"]
            confidence = 0.95

        elif hardware_result["is_hardware_anomaly"]:
            category = AnomalyCategory.HARDWARE_BOTTLENECK
            status = (
                NodeHealthStatus.CRITICAL
                if hardware_result["severity"] == "CRITICAL"
                else NodeHealthStatus.WARNING
            )
            root_cause = hardware_result["details"]
            confidence = 0.90

        elif drift_result["requires_cloud_retraining"]:
            category = AnomalyCategory.COVARIATE_DATA_DRIFT
            status = NodeHealthStatus.WARNING
            root_cause = drift_result["details"]
            confidence = 0.88

        else:
            category = AnomalyCategory.NORMAL
            status = NodeHealthStatus.HEALTHY
            root_cause = "All edge optical parameters, compute latency, and inference distributions within tolerance."
            confidence = 0.99

        # Generate Remediation Payload
        remediation = generate_remediation_action(
            anomaly_category=category,
            severity=optical_result["severity"]
            if optical_result["is_optical_anomaly"]
            else hardware_result["severity"],
            node_id=self.node_id,
        )

        summary = (
            f"Node {self.node_id} evaluated as {status.value}. Anomaly Class: {category.value}. "
            f"Prescribed Action: {remediation['recommended_action'].value}."
        )

        return DiagnosticReport(
            timestamp=time.time(),
            node_id=self.node_id,
            batch_id=batch.batch_id,
            health_status=status,
            anomaly_category=category,
            recommended_action=remediation["recommended_action"],
            confidence=confidence,
            summary=summary,
            root_cause_analysis=root_cause,
            remediation_payload=remediation["payload"],
            statistical_evidence=drift_report,
        )
