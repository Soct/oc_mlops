"""Tests unitaires pour l'API FastAPI."""

import json
import importlib
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app" / "api"))

from fastapi.testclient import TestClient

# Fixer les variables d'environnement AVANT le chargement du module API
# pour éviter la création de /app/logs (permission denied hors Docker)
_test_tmp = tempfile.mkdtemp()
os.environ.setdefault("MODELS_DIR", _test_tmp)
os.environ.setdefault("LOGS_DIR", _test_tmp)

# Charger app/api/main.py sous un nom unique pour éviter le conflit avec le main.py racine
_spec = importlib.util.spec_from_file_location("api_main", ROOT / "app" / "api" / "main.py")
api_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(api_module)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture()
def mock_model():
    """Mock du modèle qui retourne une probabilité fixe."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.4, 0.6]])
    return model


@pytest.fixture()
def mock_metadata():
    """Métadonnées de modèle simulées."""
    return {
        "model_type": "LightGBM",
        "n_features": 5,
        "features": [
            "EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1",
            "DAYS_BIRTH", "AMT_ANNUITY",
        ],
        "threshold_default": 0.5,
        "threshold_business": 0.4,
        "business_cost_fn": 10,
    }


@pytest.fixture()
def client(mock_model, mock_metadata, tmp_path):
    """TestClient avec modèle et métadonnées mockés."""
    api_module._model = mock_model
    api_module._metadata = mock_metadata
    api_module.PREDICTIONS_LOG = tmp_path / "predictions.jsonl"

    return TestClient(api_module.app)


# ============================================================================
# Tests /health
# ============================================================================

class TestHealth:

    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["model_type"] == "LightGBM"

    def test_health_degraded(self, mock_metadata, tmp_path):
        api_module._model = None
        api_module._metadata = None
        api_module.PREDICTIONS_LOG = tmp_path / "predictions.jsonl"
        c = TestClient(api_module.app)

        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"


# ============================================================================
# Tests /model-info
# ============================================================================

class TestModelInfo:

    def test_model_info(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_type"] == "LightGBM"
        assert "features" in data

    def test_model_info_no_model(self, tmp_path):
        api_module._model = None
        api_module._metadata = None
        api_module.PREDICTIONS_LOG = tmp_path / "predictions.jsonl"
        c = TestClient(api_module.app)

        resp = c.get("/model-info")
        assert resp.status_code == 503


# ============================================================================
# Tests /predict
# ============================================================================

class TestPredict:

    def test_predict_success(self, client):
        resp = client.post("/predict", json={
            "customer_id": "test_001",
            "features": {
                "EXT_SOURCE_3": 0.52,
                "EXT_SOURCE_2": 0.61,
                "EXT_SOURCE_1": 0.44,
                "DAYS_BIRTH": -12000,
                "AMT_ANNUITY": 18000.0,
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert 0 <= data["probability"] <= 1
        assert data["prediction_default"] in (0, 1)
        assert data["prediction_business"] in (0, 1)
        assert data["processing_time_ms"] >= 0

    def test_predict_missing_features(self, client):
        """Les features manquantes doivent être remplacées par 0.0."""
        resp = client.post("/predict", json={
            "features": {"EXT_SOURCE_3": 0.5},
        })
        assert resp.status_code == 200

    def test_predict_logs_features(self, client, tmp_path):
        log_path = api_module.PREDICTIONS_LOG
        client.post("/predict", json={
            "customer_id": "log_test",
            "features": {"EXT_SOURCE_3": 0.5},
        })

        assert log_path.exists()
        with open(log_path) as f:
            entry = json.loads(f.readline())
        assert "features" in entry
        assert entry["customer_id"] == "log_test"

    def test_predict_no_model(self, tmp_path):
        api_module._model = None
        api_module._metadata = None
        api_module.PREDICTIONS_LOG = tmp_path / "predictions.jsonl"
        c = TestClient(api_module.app)

        resp = c.post("/predict", json={"features": {"a": 1}})
        assert resp.status_code == 503


# ============================================================================
# Tests /metrics
# ============================================================================

class TestMetrics:

    def test_metrics_empty(self, client, tmp_path):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.json()["total_predictions"] == 0

    def test_metrics_with_data(self, client):
        # Générer quelques prédictions
        for i in range(3):
            client.post("/predict", json={
                "customer_id": f"m_{i}",
                "features": {"EXT_SOURCE_3": 0.3 + i * 0.1},
            })
        resp = client.get("/metrics")
        data = resp.json()
        assert data["total_predictions"] == 3
        assert "score_distribution" in data
        assert "response_time_ms" in data


# ============================================================================
# Tests /prediction-logs
# ============================================================================

class TestPredictionLogs:

    def test_empty_logs(self, client):
        resp = client.get("/prediction-logs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_logs_with_features(self, client):
        client.post("/predict", json={
            "customer_id": "pl_1",
            "features": {"EXT_SOURCE_3": 0.5, "DAYS_BIRTH": -10000},
        })
        resp = client.get("/prediction-logs")
        data = resp.json()
        assert len(data) == 1
        assert "features" in data[0]
