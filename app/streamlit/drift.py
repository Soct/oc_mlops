"""
Module de monitoring du data drift avec Evidently.

Compare les distributions des features entre les données de référence
(sous-ensemble du holdout) et les données courantes (features reçues par l'API).

Compatible avec Evidently >= 0.7.
"""

from __future__ import annotations

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
):
    """Génère un rapport Evidently de data drift.

    Parameters
    ----------
    reference : DataFrame avec les colonnes features (même noms dans les deux DF).
    current   : DataFrame des données courantes (features loguées par l'API).

    Returns
    -------
    Snapshot Evidently (retour de report.run()).
    """
    # Aligner les colonnes (ne garder que celles communes)
    common_cols = sorted(set(reference.columns) & set(current.columns))
    ref = reference[common_cols]
    cur = current[common_cols]

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref, current_data=cur)
    return snapshot


def drift_summary(snapshot) -> dict:
    """Extrait un résumé du rapport de drift.

    Parameters
    ----------
    snapshot : objet retourné par report.run() (evidently.core.report.Snapshot).

    Returns
    -------
    dict avec :
      - n_features        : nombre de features analysées
      - n_drifted         : nombre de features en drift
      - share_drifted     : proportion de features en drift
      - dataset_drift     : True si drift global détecté
    """
    n_features = 0
    n_drifted = 0
    share_drifted = 0.0

    for _key, value in snapshot.metric_results.items():
        display_name = value.dict().get("display_name", "")
        if "Count of Drifted Columns" in display_name:
            # count et share sont des SingleValue contenant un .value
            n_drifted = int(value.count.value)
            share_drifted = float(value.share.value)
        elif "Value drift for" in display_name:
            n_features += 1

    # n_features = nombre total de colonnes testées (chaque colonne a un "Value drift for ...")
    dataset_drift = share_drifted > 0.5 if n_features > 0 else False

    return {
        "n_features": n_features,
        "n_drifted": n_drifted,
        "share_drifted": share_drifted,
        "dataset_drift": dataset_drift,
    }
