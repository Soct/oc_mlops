"""Tests unitaires pour le module de drift monitoring."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app" / "streamlit"))

from drift import build_drift_report, drift_summary


# ============================================================================
# build_drift_report
# ============================================================================

class TestBuildDriftReport:

    def test_returns_snapshot(self, reference_df, current_df_no_drift):
        snapshot = build_drift_report(reference_df, current_df_no_drift)
        assert snapshot is not None

    def test_snapshot_has_html(self, reference_df, current_df_no_drift):
        snapshot = build_drift_report(reference_df, current_df_no_drift)
        html = snapshot.get_html_str(as_iframe=False)
        assert "<div" in html

    def test_snapshot_has_metric_results(self, reference_df, current_df_no_drift):
        snapshot = build_drift_report(reference_df, current_df_no_drift)
        assert hasattr(snapshot, "metric_results")
        assert len(snapshot.metric_results) > 0

    def test_handles_extra_columns(self, reference_df, current_df_no_drift):
        """Le rapport ne garde que les colonnes communes."""
        ref = reference_df.copy()
        ref["extra_col"] = 0
        snapshot = build_drift_report(ref, current_df_no_drift)
        assert snapshot is not None


# ============================================================================
# drift_summary
# ============================================================================

class TestDriftSummary:

    def test_no_drift(self, reference_df, current_df_no_drift):
        snapshot = build_drift_report(reference_df, current_df_no_drift)
        summary = drift_summary(snapshot)

        assert "n_features" in summary
        assert "n_drifted" in summary
        assert "share_drifted" in summary
        assert "dataset_drift" in summary
        assert summary["n_features"] == 5

    def test_with_drift(self, reference_df, current_df_with_drift):
        snapshot = build_drift_report(reference_df, current_df_with_drift)
        summary = drift_summary(snapshot)

        # Avec un shift aussi important, toutes les features devraient être en drift
        assert summary["n_drifted"] > 0
        assert summary["dataset_drift"] is True

    def test_summary_values_consistent(self, reference_df, current_df_no_drift):
        snapshot = build_drift_report(reference_df, current_df_no_drift)
        summary = drift_summary(snapshot)

        assert 0 <= summary["share_drifted"] <= 1.0
        assert summary["n_drifted"] <= summary["n_features"]


# ============================================================================
# split_holdout
# ============================================================================

class TestSplitHoldout:

    def test_split_creates_two_files(self, tmp_path):
        sys.path.insert(0, str(ROOT / "app" / "scripts"))
        from split_holdout import split_holdout

        # Créer un holdout factice
        df = pd.DataFrame({
            "feat_a": np.random.randn(100),
            "feat_b": np.random.randn(100),
            "TARGET": np.random.randint(0, 2, 100),
        })
        source = tmp_path / "holdout_sample.parquet"
        df.to_parquet(source, index=False)

        ref_path, test_path = split_holdout(
            source=source, dest_dir=tmp_path, reference_frac=0.5
        )

        assert ref_path.exists()
        assert test_path.exists()

        ref = pd.read_parquet(ref_path)
        test = pd.read_parquet(test_path)

        assert len(ref) + len(test) == 100
        assert len(ref) == 50
