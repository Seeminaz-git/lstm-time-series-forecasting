from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_forecast_endpoint_returns_requested_steps():
    response = client.post("/forecast", json={"steps": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["steps"] == 5
    assert len(body["forecast"]) == 5
