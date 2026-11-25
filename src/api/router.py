"""
FastAPI router definition for Edge Vision Sentinel.
Exposes endpoints for health checks, telemetry ingestion, drift evaluation, and edge simulation.
"""

import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agent.diagnostic_agent import DiagnosticAgent
from src.agent.schemas import DiagnosticReport, TelemetryBatch
from src.config import settings
from src.mlops.baseline_profile import get_default_golden_baseline
from src.mlops.drift_detector import DriftDetector
from src.vision.perturbations import PerturbationConfig
from src.vision.processor import VisionProcessor

router = APIRouter(tags=["Edge Vision Sentinel"])


class HealthResponse(BaseModel):
    status: str = "UP"
    node_id: str
    environment: str
    target_fps: float
    timestamp: float


class SimulationRequest(BaseModel):
    frames_count: int = Field(default=30, ge=5, le=300)
    inject_blur: bool = False
    blur_kernel_size: int = 35
    inject_lighting_drop: bool = False
    lighting_factor: float = 0.20
    inject_glare: bool = False
    inject_hardware_jitter_ms: float = 0.0
    video_path: str | None = None


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Returns the operational status and identity of the edge sentinel node."""
    return HealthResponse(
        status="UP",
        node_id=settings.NODE_ID,
        environment=settings.ENVIRONMENT,
        target_fps=settings.TARGET_FPS,
        timestamp=time.time(),
    )


@router.get("/baseline")
def get_baseline():
    """Returns the calibrated golden reference baseline distributions."""
    baseline = get_default_golden_baseline()
    return baseline.model_dump()


@router.post("/evaluate-stream", response_model=DiagnosticReport)
def evaluate_stream(batch: TelemetryBatch) -> DiagnosticReport:
    """
    Ingests a window of edge telemetry frames, runs Kolmogorov-Smirnov drift tests,
    and returns an autonomous diagnostic verdict with prescribed remediation.
    """
    if not batch.frames:
        raise HTTPException(
            status_code=400, detail="Telemetry batch contains no frames."
        )

    detector = DriftDetector()
    drift_report = detector.analyze_batch(batch)

    agent = DiagnosticAgent()
    diagnostic_report = agent.evaluate_batch(batch, drift_report)

    return diagnostic_report


@router.post("/simulate", response_model=DiagnosticReport)
def simulate_edge_stream(req: SimulationRequest) -> DiagnosticReport:
    """
    Simulates edge video ingestion with optional physical or hardware perturbations,
    extracts real-time telemetry, and runs the full diagnostic workflow.
    """
    target_video = (
        Path(req.video_path) if req.video_path else settings.DEFAULT_VIDEO_PATH
    )
    if not target_video.exists():
        raise HTTPException(
            status_code=404, detail=f"Video asset not found: {target_video}"
        )

    perturbation = PerturbationConfig(
        blur=req.inject_blur,
        blur_kernel_size=req.blur_kernel_size,
        lighting_drop=req.inject_lighting_drop,
        lighting_factor=req.lighting_factor,
        glare=req.inject_glare,
        hardware_jitter_ms=req.inject_hardware_jitter_ms,
    )

    processor = VisionProcessor()
    batch = processor.process_batch(
        video_source=target_video,
        batch_size=req.frames_count,
        perturbation=perturbation,
        batch_id=f"sim-{int(time.time())}",
    )

    detector = DriftDetector()
    drift_report = detector.analyze_batch(batch)

    agent = DiagnosticAgent()
    return agent.evaluate_batch(batch, drift_report)
