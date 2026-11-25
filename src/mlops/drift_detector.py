"""
MLOps Data Drift and Statistical Anomaly Engine.
Evaluates streaming edge telemetry against reference golden baselines using:
- Kolmogorov-Smirnov (KS) two-sample non-parametric hypothesis testing
- Z-score outlier detection for latency and frame rate jitter
- Composite Edge Health Index scoring
"""

import numpy as np
from scipy import stats

from src.agent.schemas import StatisticalDriftReport, TelemetryBatch
from src.config import settings
from src.mlops.baseline_profile import BaselineProfile, get_default_golden_baseline


class DriftDetector:
    """Statistical evaluation engine comparing live edge telemetry batches with baseline."""

    def __init__(
        self,
        baseline: BaselineProfile | None = None,
        alpha: float = settings.KS_DRIFT_ALPHA,
        z_thresh: float = settings.LATENCY_ZSCORE_THRESHOLD,
    ) -> None:
        self.baseline = baseline or get_default_golden_baseline()
        self.alpha = alpha
        self.z_thresh = z_thresh

    def analyze_batch(self, batch: TelemetryBatch) -> StatisticalDriftReport:
        """Runs statistical hypothesis testing and outlier analysis over an ingested batch."""
        if not batch.frames:
            return StatisticalDriftReport(
                overall_health_score=0.0,
                summary_flags=["EMPTY_TELEMETRY_BATCH"],
            )

        sharp_current = np.array([f.sharpness for f in batch.frames])
        bright_current = np.array([f.brightness_mean for f in batch.frames])
        conf_current = np.array([f.confidence_score for f in batch.frames])
        lat_current = np.array([f.latency_ms for f in batch.frames])
        fps_current = np.array([f.fps for f in batch.frames])

        sharp_base = np.array(self.baseline.sharpness_samples)
        bright_base = np.array(self.baseline.brightness_samples)
        conf_base = np.array(self.baseline.confidence_samples)

        summary_flags: list[str] = []

        # 1. Optical Sharpness Drift (Kolmogorov-Smirnov 2-sample test)
        ks_sharp = stats.ks_2samp(sharp_current, sharp_base)
        sharp_mean_diff = float(np.mean(sharp_current)) - self.baseline.sharpness_mean
        sharp_drift = bool(ks_sharp.pvalue < self.alpha and sharp_mean_diff < -15.0)
        if (
            sharp_drift
            or float(np.mean(sharp_current)) < settings.SHARPNESS_BLUR_THRESHOLD * 0.7
        ):
            sharp_drift = True
            summary_flags.append("OPTICAL_SHARPNESS_DEGRADED")

        # 2. Luminance / Lighting Shift (KS test & threshold boundaries)
        ks_bright = stats.ks_2samp(bright_current, bright_base)
        mean_lum = float(np.mean(bright_current))
        bright_drift = bool(
            (
                ks_bright.pvalue < self.alpha
                and abs(mean_lum - self.baseline.brightness_mean) > 25.0
            )
            or mean_lum < settings.LUMINANCE_LOW_THRESHOLD
            or mean_lum > settings.LUMINANCE_HIGH_THRESHOLD
        )
        if bright_drift:
            summary_flags.append("LUMINANCE_DISTRIBUTION_DRIFT")

        # 3. Covariate Model Confidence Degradation (KS test)
        ks_conf = stats.ks_2samp(conf_current, conf_base)
        conf_mean_diff = float(np.mean(conf_current)) - self.baseline.confidence_mean
        conf_drift = bool(
            ks_conf.pvalue < self.alpha
            and conf_mean_diff < -settings.CONFIDENCE_DROP_THRESHOLD
        )
        if conf_drift:
            summary_flags.append("COVARIATE_CONFIDENCE_DRIFT")

        # 4. Latency Spikes (Z-Score Outlier Analysis, excluding frame 0 cold start)
        base_lat_mean = self.baseline.latency_mean_ms
        base_lat_std = max(0.2, self.baseline.latency_std_ms)
        eval_lat = lat_current[1:] if len(lat_current) > 3 else lat_current
        z_scores = (eval_lat - base_lat_mean) / base_lat_std
        max_zscore = float(np.max(z_scores)) if len(z_scores) > 0 else 0.0
        p95_lat = float(np.percentile(eval_lat, 95)) if len(eval_lat) > 0 else 0.0
        latency_anomaly = bool(
            max_zscore > self.z_thresh and p95_lat > (base_lat_mean * 2.0)
        )
        if latency_anomaly:
            summary_flags.append(f"LATENCY_SPIKE_ZSCORE_{round(max_zscore, 1)}")

        # 5. FPS Drop (Moving Average & Minimum observed)
        min_fps = float(np.min(fps_current)) if len(fps_current) > 0 else 0.0
        mean_fps = float(np.mean(fps_current)) if len(fps_current) > 0 else 0.0
        fps_drop = bool(
            min_fps < settings.FPS_MIN_THRESHOLD
            or mean_fps < settings.FPS_MIN_THRESHOLD
        )
        if fps_drop:
            summary_flags.append("FPS_BOTTLENECK_DETECTED")

        # 6. Overall Health Score Calculation (1.0 = Optimal, 0.0 = Degraded)
        penalties = 0.0
        if sharp_drift:
            penalties += 0.35
        if bright_drift:
            penalties += 0.25
        if conf_drift:
            penalties += 0.30
        if latency_anomaly:
            penalties += 0.20
        if fps_drop:
            penalties += 0.20

        health_score = round(float(np.clip(1.0 - penalties, 0.0, 1.0)), 2)

        return StatisticalDriftReport(
            sharpness_drift=sharp_drift,
            sharpness_pvalue=round(float(ks_sharp.pvalue), 4),
            brightness_drift=bright_drift,
            brightness_pvalue=round(float(ks_bright.pvalue), 4),
            confidence_drift=conf_drift,
            confidence_pvalue=round(float(ks_conf.pvalue), 4),
            latency_anomaly=latency_anomaly,
            latency_zscore_max=round(max_zscore, 2),
            fps_drop=fps_drop,
            min_fps_observed=round(min_fps, 1),
            overall_health_score=health_score,
            summary_flags=summary_flags,
        )
