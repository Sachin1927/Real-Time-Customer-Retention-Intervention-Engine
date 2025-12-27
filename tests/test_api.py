from fastapi.testclient import TestClient
from src.api.app import app

# Create a test client (simulates a browser)
client = TestClient(app)

def test_read_main():
    """Check if the home page is active."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "active", "model_version": "v1"}

def test_predict_churn_risky():
    """Check if the model correctly flags a risky customer."""
    payload = {
        "age": 30,
        "tenure_months": 1,
        "monthly_charges": 120.0,
        "avg_daily_usage_min": 5,
        "payment_fails_last_3m": 3,
        "last_interaction_sentiment": -0.9
    }
    response = client.post("/predict_churn", json=payload)
    assert response.status_code == 200
    assert response.json()["risk_label"] == "High Risk"

def test_predict_churn_safe():
    """Check if the model correctly identifies a safe customer."""
    payload = {
        "age": 45,
        "tenure_months": 60,
        "monthly_charges": 50.0,
        "avg_daily_usage_min": 200,
        "payment_fails_last_3m": 0,
        "last_interaction_sentiment": 0.8
    }
    response = client.post("/predict_churn", json=payload)
    assert response.status_code == 200
    assert response.json()["risk_label"] == "Safe"