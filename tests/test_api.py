"""
Integration tests for FastAPI endpoints.
"""

from fastapi.testclient import TestClient

from src.agent.schemas import NodeHealthStatus, RemediationAction
from src.api.app import app

client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UP"
    assert "node_id" in data
    assert "environment" in data


def test_api_baseline_endpoint():
    response = client.get("/baseline")
    assert response.status_code == 200
    data = response.json()
    assert "sharpness_mean" in data
    assert "brightness_mean" in data


def test_api_evaluate_stream(mock_healthy_batch):
    payload = mock_healthy_batch.model_dump()
    response = client.post("/evaluate-stream", json=payload)
    assert response.status_code == 200
    report = response.json()
    assert report["health_status"] == NodeHealthStatus.HEALTHY.value
    assert report["recommended_action"] == RemediationAction.NO_ACTION.value


def test_api_evaluate_stream_empty_fails():
    payload = {"node_id": "test", "batch_id": "b-empty", "frames": []}
    response = client.post("/evaluate-stream", json=payload)
    assert response.status_code == 400


def test_api_simulate_normal():
    response = client.post("/simulate", json={"frames_count": 15, "inject_blur": False})
    assert response.status_code == 200
    data = response.json()
    assert data["health_status"] == NodeHealthStatus.HEALTHY.value


def test_api_simulate_blur():
    response = client.post("/simulate", json={"frames_count": 15, "inject_blur": True})
    assert response.status_code == 200
    data = response.json()
    assert data["health_status"] in [
        NodeHealthStatus.WARNING.value,
        NodeHealthStatus.CRITICAL.value,
    ]
    assert (
        data["recommended_action"]
        == RemediationAction.CLEAN_OPTICS_HARDWARE_ALERT.value
    )


def test_api_simulate_invalid_video():
    response = client.post(
        "/simulate", json={"frames_count": 10, "video_path": "non_existent_file.mp4"}
    )
    assert response.status_code == 404
