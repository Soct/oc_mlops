"""
Home Credit Default Risk - API FastAPI
Endpoints :
  GET  /health       -> statut de l'API et du modele
  GET  /model-info   -> metadonnees du modele (features, seuils, metriques)
  POST /predict      -> prediction pour un client
  GET  /metrics      -> metriques de monitoring en temps reel
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
LOGS_DIR   = Path(os.getenv("LOGS_DIR",   "/app/logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"

# ============================================================================
# CHARGEMENT DU MODELE (au demarrage)
# ============================================================================

_model    = None
_metadata = None


def load_model() -> None:
    global _model, _metadata
    model_path    = MODELS_DIR / "model.pkl"
    metadata_path = MODELS_DIR / "model_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Modele introuvable : {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadonnees introuvables : {metadata_path}")

    with open(model_path, "rb") as f:
        _model = pickle.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        _metadata = json.load(f)

    print(f"[startup] Modele charge : {_metadata['model_type']}")
    print(f"[startup] Features      : {_metadata['n_features']}")
    print(f"[startup] Seuil default : {_metadata['threshold_default']}")
    print(f"[startup] Seuil business: {_metadata['threshold_business']}")


# ============================================================================
# APPLICATION
# ============================================================================

app = FastAPI(
    title="Home Credit Default Risk API",
    description=(
        "API de prediction de risque de defaut de paiement. "
        "Modele : LightGBM + StandardScaler (20 features selectionnees par SHAP). "
        "Expose deux seuils de decision : standard (0.5) et business-optimal (FN=10xFP)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"[startup] AVERTISSEMENT : {e}")
        print("[startup] L'API demarre sans modele. Lancez le notebook d'export en premier.")


# ============================================================================
# SCHEMAS
# ============================================================================

class PredictionRequest(BaseModel):
    features:    Dict[str, float]
    customer_id: Optional[str] = None

    model_config = {"json_schema_extra": {
        "example": {
            "customer_id": "client_001",
            "features": {
                "EXT_SOURCE_3": 0.52,
                "EXT_SOURCE_2": 0.61,
                "EXT_SOURCE_1": 0.44,
                "DAYS_BIRTH": -12000,
                "AMT_ANNUITY": 18000.0,
            }
        }
    }}


class PredictionResponse(BaseModel):
    customer_id:         Optional[str]
    probability:         float
    prediction_default:  int    # seuil 0.5
    prediction_business: int    # seuil business-optimal
    threshold_default:   float
    threshold_business:  float
    processing_time_ms:  float
    decision_label:      str    # texte lisible pour l'UI


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", tags=["Infrastructure"])
def health() -> dict:
    """Verifie que l'API tourne et que le modele est charge."""
    return {
        "status":        "ok" if _model is not None else "degraded",
        "model_loaded":  _model is not None,
        "model_type":    _metadata["model_type"] if _metadata else None,
        "n_features":    _metadata["n_features"] if _metadata else None,
        "timestamp":     datetime.utcnow().isoformat(),
    }


@app.get("/model-info", tags=["Modele"])
def model_info() -> dict:
    """Retourne les metadonnees completes du modele deploye."""
    if _metadata is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    return _metadata


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predit le risque de defaut pour un client.

    - **features** : dictionnaire {nom_feature: valeur}. Les features manquantes sont
      remplacees par 0.0 (imputation par la mediane d'entrainement approximee).
    - **customer_id** : identifiant optionnel pour le tracking.

    Retourne :
    - **probability** : probabilite brute de defaut [0, 1]
    - **prediction_default** : decision au seuil 0.5 (0=OK, 1=DEFAUT)
    - **prediction_business** : decision au seuil business-optimal (minimise FN*10+FP)
    """
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Modele non charge. Relancez l'API.")

    start_ns = time.perf_counter()

    # Construction du vecteur de features dans l'ordre attendu
    expected_features = _metadata["features"]
    row = {feat: float(request.features.get(feat, 0.0)) for feat in expected_features}
    df  = pd.DataFrame([row])[expected_features]

    # Inference
    probability = float(_model.predict_proba(df)[0, 1])

    threshold_default  = float(_metadata["threshold_default"])
    threshold_business = float(_metadata["threshold_business"])

    prediction_default  = int(probability >= threshold_default)
    prediction_business = int(probability >= threshold_business)

    elapsed_ms = (time.perf_counter() - start_ns) * 1_000

    # Label lisible
    if prediction_business == 1:
        decision_label = f"DEFAUT (proba={probability:.1%})"
    else:
        decision_label = f"ACCORD   (proba={probability:.1%})"

    # Persistance du log
    log_entry = {
        "timestamp":          datetime.utcnow().isoformat(),
        "customer_id":        request.customer_id,
        "probability":        round(probability, 6),
        "prediction_default":  prediction_default,
        "prediction_business": prediction_business,
        "processing_time_ms": round(elapsed_ms, 3),
    }
    with open(PREDICTIONS_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_entry) + "\n")

    return PredictionResponse(
        customer_id=request.customer_id,
        probability=probability,
        prediction_default=prediction_default,
        prediction_business=prediction_business,
        threshold_default=threshold_default,
        threshold_business=threshold_business,
        processing_time_ms=elapsed_ms,
        decision_label=decision_label,
    )


@app.get("/metrics", tags=["Monitoring"])
def get_metrics() -> dict:
    """
    Metriques de monitoring en temps reel calculees sur l'historique des predictions.

    - Distribution des scores (min, mean, std, p50, p90, p95, max)
    - Latences (mean, p50, p95, p99)
    - Taux de defaut (seuil 0.5 et business)
    - 10 dernieres predictions
    """
    if not PREDICTIONS_LOG.exists():
        return {"total_predictions": 0, "message": "Aucune prediction enregistree."}

    logs: list[dict] = []
    with open(PREDICTIONS_LOG, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not logs:
        return {"total_predictions": 0, "message": "Aucune prediction enregistree."}

    probs           = np.array([l["probability"]         for l in logs])
    times_ms        = np.array([l["processing_time_ms"]  for l in logs])
    preds_default   = np.array([l["prediction_default"]  for l in logs])
    preds_business  = np.array([l["prediction_business"] for l in logs])

    return {
        "total_predictions": len(logs),
        "score_distribution": {
            "mean": round(float(np.mean(probs)),              4),
            "std":  round(float(np.std(probs)),               4),
            "min":  round(float(np.min(probs)),               4),
            "p50":  round(float(np.percentile(probs, 50)),    4),
            "p90":  round(float(np.percentile(probs, 90)),    4),
            "p95":  round(float(np.percentile(probs, 95)),    4),
            "max":  round(float(np.max(probs)),               4),
        },
        "response_time_ms": {
            "mean": round(float(np.mean(times_ms)),            2),
            "p50":  round(float(np.percentile(times_ms, 50)), 2),
            "p95":  round(float(np.percentile(times_ms, 95)), 2),
            "p99":  round(float(np.percentile(times_ms, 99)), 2),
        },
        "prediction_rates": {
            "default_rate_05":       round(float(np.mean(preds_default)),  4),
            "default_rate_business": round(float(np.mean(preds_business)), 4),
        },
        "recent_predictions": logs[-10:],
    }
