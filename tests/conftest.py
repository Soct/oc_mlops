"""Fixtures partagées pour les tests."""

import json
import pytest
import numpy as np
import pandas as pd


@pytest.fixture()
def sample_features():
    """20 features simulant le format attendu par le modèle."""
    return {
        "EXT_SOURCE_3": 0.52,
        "EXT_SOURCE_2": 0.61,
        "EXT_SOURCE_1": 0.44,
        "DAYS_BIRTH": -12000.0,
        "AMT_ANNUITY": 18000.0,
        "AMT_CREDIT": 200000.0,
        "AMT_INCOME_TOTAL": 150000.0,
        "AMT_GOODS_PRICE": 180000.0,
        "DAYS_EMPLOYED": -2000.0,
        "DAYS_ID_PUBLISH": -3000.0,
        "DAYS_REGISTRATION": -5000.0,
        "DAYS_LAST_PHONE_CHANGE": -500.0,
        "REGION_POPULATION_RELATIVE": 0.02,
        "HOUR_APPR_PROCESS_START": 10.0,
        "OWN_CAR_AGE": 5.0,
        "FLAG_DOCUMENT_3": 1.0,
        "CNT_FAM_MEMBERS": 2.0,
        "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
        "PAYMENT_RATE": 0.09,
        "INCOME_CREDIT_PERC": 0.75,
    }


@pytest.fixture()
def reference_df():
    """DataFrame de référence pour le drift (50 lignes, 5 features)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "feat_a": rng.normal(0, 1, 50),
        "feat_b": rng.normal(5, 2, 50),
        "feat_c": rng.uniform(0, 1, 50),
        "feat_d": rng.exponential(1, 50),
        "feat_e": rng.normal(10, 0.5, 50),
    })


@pytest.fixture()
def current_df_no_drift(reference_df):
    """Données courantes SANS drift (même distribution que la référence)."""
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "feat_a": rng.normal(0, 1, 30),
        "feat_b": rng.normal(5, 2, 30),
        "feat_c": rng.uniform(0, 1, 30),
        "feat_d": rng.exponential(1, 30),
        "feat_e": rng.normal(10, 0.5, 30),
    })


@pytest.fixture()
def current_df_with_drift(reference_df):
    """Données courantes AVEC drift (distributions décalées)."""
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "feat_a": rng.normal(5, 1, 30),     # shift +5 en moyenne
        "feat_b": rng.normal(50, 2, 30),    # shift +45 en moyenne
        "feat_c": rng.uniform(0.8, 1, 30),  # distribution compressée
        "feat_d": rng.exponential(10, 30),   # scale x10
        "feat_e": rng.normal(100, 0.5, 30),  # shift +90
    })


@pytest.fixture()
def predictions_log_lines(sample_features):
    """Lignes JSON simulant predictions.jsonl avec features."""
    entries = []
    for i in range(5):
        entry = {
            "timestamp": f"2026-03-11T09:00:0{i}.000000",
            "customer_id": f"test_{i}",
            "probability": 0.3 + i * 0.1,
            "prediction_default": int((0.3 + i * 0.1) >= 0.5),
            "prediction_business": int((0.3 + i * 0.1) >= 0.4),
            "processing_time_ms": 3.0 + i,
            "features": sample_features,
        }
        entries.append(json.dumps(entry))
    return entries
