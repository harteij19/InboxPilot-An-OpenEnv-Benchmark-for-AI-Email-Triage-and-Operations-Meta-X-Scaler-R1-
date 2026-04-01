"""
Tests for the InboxPilot FastAPI server.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestLandingPage:
    def test_landing_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "InboxPilot" in response.text


class TestTasksEndpoint:
    def test_tasks_returns_list(self, client):
        response = client.get("/tasks")
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert len(data["tasks"]) >= 3


class TestResetEndpoint:
    def test_reset_empty_body(self, client):
        """POST /reset with {} must return 200."""
        response = client.post("/reset", json={})
        assert response.status_code == 200
        data = response.json()
        assert "observation" in data
        assert data["done"] is False

    def test_reset_with_task_id(self, client):
        response = client.post("/reset", json={"task_id": "medium_customer_resolution"})
        assert response.status_code == 200
        data = response.json()
        assert data["observation"]["task_id"] == "medium_customer_resolution"

    def test_reset_invalid_task(self, client):
        response = client.post("/reset", json={"task_id": "nonexistent"})
        assert response.status_code == 404


class TestStepEndpoint:
    def test_step_after_reset(self, client):
        client.post("/reset", json={})
        response = client.post("/step", json={
            "action_type": "open_email",
            "email_id": "email_001",
            "payload": {}
        })
        assert response.status_code == 200
        data = response.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data


class TestStateEndpoint:
    def test_state_after_reset(self, client):
        client.post("/reset", json={})
        response = client.get("/state")
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "emails" in data
