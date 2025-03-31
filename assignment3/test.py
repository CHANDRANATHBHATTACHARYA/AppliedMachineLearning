import joblib
from score import score
import pytest
import os
import time
import requests


@pytest.fixture
def model():
    try:
        return joblib.load("spam_model.joblib")
    except Exception as e:
        pytest.fail(f"Failed to load model: {str(e)}")

def test_smoke(model):
    try:
        prediction, propensity = score("test", model, 0.5)
        assert isinstance(prediction, bool)
        assert isinstance(propensity, float)
    except Exception as e:
        pytest.fail(f"Smoke test failed: {str(e)}")

def test_threshold_0(model):
    prediction, _ = score("Test", model, 0.0)
    assert prediction == True

def test_threshold_1(model):
    prediction, _ = score("Test", model, 1.0)
    assert prediction == False

def test_obvious_spam(model):
    prediction, _ = score("WIN A FREE IPHONE!", model, 0.5)
    assert prediction == True

# In test.py
def test_flask():
    import subprocess
    import time
    import requests
    
    # Start Flask with explicit host/port
    flask_process = subprocess.Popen(
        ["python", "app.py", "--host", "127.0.0.1", "--port", "5000"]
    )
    time.sleep(3)  # Increased wait time
    
    try:
        response = requests.post(
            "http://127.0.0.1:5000/score",
            json={"text": "TEST"},
            timeout=5
        )
        assert response.status_code == 200
    finally:
        flask_process.terminate()
        time.sleep(1)  # Cleanup time