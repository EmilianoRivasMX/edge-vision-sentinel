"""
Interactive Command Line Interface (CLI) for Edge Vision Sentinel.
Provides commands to run edge profiling, simulate optical/hardware drift,
benchmark per-frame processing latency, and start the FastAPI service.
"""

import argparse
import sys
import time
from pathlib import Path

import uvicorn

from src.agent.diagnostic_agent import DiagnosticAgent
from src.agent.schemas import NodeHealthStatus
from src.config import settings
from src.mlops.drift_detector import DriftDetector
from src.vision.perturbations import PerturbationConfig
from src.vision.processor import VisionProcessor


def print_banner() -> None:
    banner = """
========================================================================
           EDGE VISION SENTINEL - AUTONOMOUS DRIFT SUPERVISOR          
========================================================================
"""
    print(banner)


def run_stream_command(args: argparse.Namespace) -> None:
    """Processes video stream, prints real-time ASCII telemetry table, and executes agent diagnosis."""
    print_banner()
    video_path = Path(args.video) if args.video else settings.DEFAULT_VIDEO_PATH
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    perturbation = PerturbationConfig(
        blur=args.blur,
        lighting_drop=args.lighting_drop,
        glare=args.glare,
        hardware_jitter_ms=args.jitter_ms,
    )

    print(f"[*] Initializing Edge Vision Processor on: {video_path.name}")
    print(f"[*] Batch Window: {args.frames} frames")
    if args.blur:
        print("[!] Injected Perturbation: OPTICAL BLUR (Dirty Lens)")
    if args.lighting_drop:
        print("[!] Injected Perturbation: LIGHTING FAILURE (Underexposure)")
    if args.jitter_ms > 0:
        print(
            f"[!] Injected Perturbation: HARDWARE BOTTLENECK ({args.jitter_ms}ms jitter)"
        )

    processor = VisionProcessor()
    t0 = time.time()
    batch = processor.process_batch(
        video_source=video_path,
        batch_size=args.frames,
        perturbation=perturbation,
        batch_id=f"cli-{int(time.time())}",
    )
    elapsed = time.time() - t0

    print("\n" + "-" * 75)
    print(
        f"{'FRAME':<8}{'LATENCY':<12}{'FPS':<10}{'SHARPNESS':<14}{'BRIGHTNESS':<14}{'CONF':<10}"
    )
    print("-" * 75)
    for f in batch.frames[:12]:
        print(
            f"{f.frame_id:<8}{f.latency_ms:<12.2f}{f.fps:<10.1f}{f.sharpness:<14.2f}"
            f"{f.brightness_mean:<14.1f}{f.confidence_score:<10.3f}"
        )
    if len(batch.frames) > 12:
        print(f"... [{len(batch.frames) - 12} additional frames processed]")
    print("-" * 75)

    print("\n[BATCH AGGREGATION]")
    print(
        f" - Processed Frames: {len(batch.frames)} in {elapsed:.2f}s ({len(batch.frames) / elapsed:.1f} effective FPS)"
    )
    print(
        f" - Mean Latency:     {batch.mean_latency_ms} ms (P95: {batch.p95_latency_ms} ms)"
    )
    print(f" - Mean Sharpness:   {batch.mean_sharpness}")
    print(f" - Mean Luminance:   {batch.mean_brightness}")
    print(f" - Mean Confidence:  {batch.mean_confidence}")

    # Run MLOps Drift Detection
    detector = DriftDetector()
    drift_report = detector.analyze_batch(batch)

    print("\n[STATISTICAL DRIFT EVALUATION (KS-Test & Z-Score)]")
    print(
        f" - Sharpness Drift:    {drift_report.sharpness_drift} (p-value: {drift_report.sharpness_pvalue})"
    )
    print(
        f" - Luminance Drift:    {drift_report.brightness_drift} (p-value: {drift_report.brightness_pvalue})"
    )
    print(
        f" - Confidence Drift:   {drift_report.confidence_drift} (p-value: {drift_report.confidence_pvalue})"
    )
    print(
        f" - Latency Anomaly:    {drift_report.latency_anomaly} (max Z-score: {drift_report.latency_zscore_max})"
    )
    print(f" - Overall Health:     {drift_report.overall_health_score * 100:.0f}%")

    # Run Diagnostic Agent
    agent = DiagnosticAgent()
    diag = agent.evaluate_batch(batch, drift_report)

    status_color = (
        "🟢"
        if diag.health_status == NodeHealthStatus.HEALTHY
        else ("🟡" if diag.health_status == NodeHealthStatus.WARNING else "🔴")
    )
    print("\n[AGENTIC DIAGNOSTIC VERDICT]")
    print(f" - Health Status:      {status_color} {diag.health_status.value}")
    print(f" - Anomaly Class:      {diag.anomaly_category.value}")
    print(f" - Recommended Action: {diag.recommended_action.value}")
    print(f" - Agent Confidence:   {diag.confidence * 100:.1f}%")
    print(f" - Root Cause:         {diag.root_cause_analysis}")
    print(f" - Action Payload:     {diag.remediation_payload}")


def run_benchmark_command(args: argparse.Namespace) -> None:
    """Profiles per-frame latency and throughput over 120 frames."""
    print_banner()
    video_path = Path(args.video) if args.video else settings.DEFAULT_VIDEO_PATH
    print(f"[*] Running Edge Profiling Benchmark (120 frames) on {video_path.name}...")

    processor = VisionProcessor()
    batch = processor.process_batch(video_source=video_path, batch_size=120)

    latencies = [f.latency_ms for f in batch.frames]
    import numpy as np

    print("\n" + "=" * 50)
    print("           EDGE PROFILING BENCHMARK RESULTS       ")
    print("=" * 50)
    print(f"Total Frames Analyzed:   {len(latencies)}")
    print(f"Latency Mean:            {np.mean(latencies):.2f} ms")
    print(f"Latency Median (P50):    {np.median(latencies):.2f} ms")
    print(f"Latency 95th Percentile: {np.percentile(latencies, 95):.2f} ms")
    print(f"Latency 99th Percentile: {np.percentile(latencies, 99):.2f} ms")
    print(f"Min Latency:             {np.min(latencies):.2f} ms")
    print(f"Max Latency:             {np.max(latencies):.2f} ms")
    print(
        f"SLA Compliance (<15ms):  {100.0 * np.mean(np.array(latencies) < 15.0):.1f}%"
    )
    print("=" * 50)


def run_serve_command(args: argparse.Namespace) -> None:
    """Starts the FastAPI Sentinel server using Uvicorn."""
    print_banner()
    print(f"[*] Starting Edge Vision Sentinel API on {args.host}:{args.port}...")
    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Edge Vision Sentinel: Autonomous Visual Quality & Data Drift Platform"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run-stream
    stream_parser = subparsers.add_parser(
        "run-stream", help="Stream video and run live diagnosis"
    )
    stream_parser.add_argument(
        "--video", type=str, default=None, help="Path to video file"
    )
    stream_parser.add_argument(
        "--frames", type=int, default=30, help="Batch frame count"
    )
    stream_parser.add_argument(
        "--blur", action="store_true", help="Inject optical blur"
    )
    stream_parser.add_argument(
        "--lighting-drop", action="store_true", help="Inject lighting blackout"
    )
    stream_parser.add_argument(
        "--glare", action="store_true", help="Inject specular glare"
    )
    stream_parser.add_argument(
        "--jitter-ms", type=float, default=0.0, help="Inject hardware latency jitter"
    )

    # benchmark
    bench_parser = subparsers.add_parser(
        "benchmark", help="Benchmark edge latency and throughput"
    )
    bench_parser.add_argument(
        "--video", type=str, default=None, help="Path to video file"
    )

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI service")
    serve_parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Binding host"
    )
    serve_parser.add_argument("--port", type=int, default=8000, help="Binding port")
    serve_parser.add_argument("--reload", action="store_true", help="Enable hot reload")

    args = parser.parse_args()

    if args.command == "run-stream":
        run_stream_command(args)
    elif args.command == "benchmark":
        run_benchmark_command(args)
    elif args.command == "serve":
        run_serve_command(args)


if __name__ == "__main__":
    main()
