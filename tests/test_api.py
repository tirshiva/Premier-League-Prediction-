import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import numpy as np

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint_valid_input():
    """Test prediction endpoint with valid input"""
    test_input = {
        "features": [2023.0]  # Example feature (year)
    }
    response = client.post("/predict", json=test_input)
    assert response.status_code == 200
    assert "predicted_score" in response.json()
    assert isinstance(response.json()["predicted_score"], float)

def test_predict_endpoint_invalid_input():
    """Test prediction endpoint with invalid input"""
    test_input = {
        "features": "invalid"  # Invalid input type
    }
    response = client.post("/predict", json=test_input)
    assert response.status_code == 422  # Validation error

def test_predict_endpoint_empty_input():
    """Test prediction endpoint with empty input"""
    test_input = {
        "features": []
    }
    response = client.post("/predict", json=test_input)
    assert response.status_code == 422  # Validation error

def test_predict_endpoint_missing_features():
    """Test prediction endpoint with missing features"""
    test_input = {}
    response = client.post("/predict", json=test_input)
    assert response.status_code == 422  # Validation error
