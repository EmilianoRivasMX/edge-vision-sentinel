"""
Smoke tests for CLI entrypoint.
"""

from unittest.mock import patch

from src.cli import main


def test_cli_benchmark_smoke():
    with patch("sys.argv", ["sentinel", "benchmark"]):
        main()


def test_cli_run_stream_smoke():
    with patch("sys.argv", ["sentinel", "run-stream", "--frames", "10"]):
        main()
