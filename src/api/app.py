"""
FastAPI application factory and middleware setup for Edge Vision Sentinel.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.router import router


def create_app() -> FastAPI:
    """Instantiates and configures the production FastAPI sentinel application."""
    app = FastAPI(
        title="Edge Vision Sentinel API",
        description=(
            "Autonomous Visual Quality Supervision & Real-Time Data Drift Platform. "
            "Integrates Edge Computer Vision telemetry, MLOps statistical drift observability, "
            "and Agentic root-cause remediation."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable CORS for industrial dashboards and edge HMIs
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")
    app.include_router(router)  # Also expose at root for convenience

    return app


app = create_app()
