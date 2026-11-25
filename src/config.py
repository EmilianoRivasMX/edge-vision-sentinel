"""
Configuration module for Edge Vision Sentinel.
Defines edge camera profile tolerances, statistical drift thresholds, and paths.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class SentinelSettings(BaseSettings):
    """System-wide configuration settings with environment variable override support."""

    model_config = SettingsConfigDict(env_prefix="SENTINEL_", case_sensitive=False)

    # General & Edge Node Identity
    NODE_ID: str = "edge-node-line-01"
    ENVIRONMENT: str = "production"

    # Base Paths
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent
    ASSETS_DIR: Path = ROOT_DIR / "assets"
    DEFAULT_VIDEO_PATH: Path = ASSETS_DIR / "videos" / "people-detection.mp4"
    DEFAULT_IMAGE_PATH: Path = ASSETS_DIR / "images" / "cctv_image.jpg"

    # Vision Processing & Telemetry Settings
    TARGET_FPS: float = 30.0
    FPS_MIN_THRESHOLD: float = 18.0
    WINDOW_SIZE: int = 30  # Number of frames in sliding window analysis

    # Optical Quality Thresholds
    SHARPNESS_BLUR_THRESHOLD: float = (
        45.0  # Laplacian variance threshold (<45 indicates defocus/blur)
    )
    LUMINANCE_LOW_THRESHOLD: float = 40.0  # Underexposure threshold (0-255)
    LUMINANCE_HIGH_THRESHOLD: float = 210.0  # Overexposure / glare threshold (0-255)

    # Statistical Drift & Outlier Thresholds
    KS_DRIFT_ALPHA: float = 0.05  # Kolmogorov-Smirnov test significance level
    LATENCY_ZSCORE_THRESHOLD: float = 2.5  # Z-score outlier cutoff for latency
    CONFIDENCE_DROP_THRESHOLD: float = (
        0.15  # Drop in model confidence compared to baseline
    )

    # Cloud & MLOps Trigger Parameters
    GCP_PROJECT_ID: str = "edge-sentinel-mlops-prod"
    GCP_REGION: str = "us-central1"
    VERTEX_PIPELINE_NAME: str = "edge-vision-retrain-pipeline"
    ENABLE_CLOUD_DISPATCH: bool = False

    # Strands Agent Settings
    LLM_PROVIDER: str = "deterministic"  # Options: "deterministic", "strands_bedrock", "openai", "gemini"


settings = SentinelSettings()
