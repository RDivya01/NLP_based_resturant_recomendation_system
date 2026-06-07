"""Tests for the thin API layer."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_health_check() -> None:
    """Health endpoint should confirm API startup."""

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
