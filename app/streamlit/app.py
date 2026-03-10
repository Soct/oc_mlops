"""
Home Credit Default Risk - Dashboard Streamlit
Deux pages :
  - Test Modele    : tester le modele sur des exemples du holdout
  - Monitoring     : distribution des scores, latences, predictions recentes
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================================
# CONFIGURATION
# ============================================================================

API_URL  = os.getenv("API_URL",  "http://localhost:8000")
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))

st.set_page_config(
    page_title="Home Credit Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# HELPERS
# ============================================================================

@st.cache_data
def load_holdout() -> pd.DataFrame | None:
    path = DATA_DIR / "holdout_sample.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def call_predict(features: dict, customer_id: str | None = None) -> dict | None:
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"features": {k: float(v) for k, v in features.items()}, "customer_id": customer_id},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Impossible de joindre l'API. Verifiez que le conteneur `api` est demarré.")
    except Exception as e:
        st.error(f"Erreur API : {e}")
    return None


def fetch_metrics() -> dict | None:
    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Impossible de joindre l'API. Verifiez que le conteneur `api` est demarré.")
    except Exception as e:
        st.error(f"Erreur API : {e}")
    return None


def fetch_model_info() -> dict | None:
    try:
        resp = requests.get(f"{API_URL}/model-info", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def score_color(prob: float) -> str:
    if prob < 0.3:
        return "green"
    if prob < 0.6:
        return "orange"
    return "red"


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🏦 Home Credit Risk")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Test Modele", "Monitoring"], label_visibility="collapsed")

# Statut API dans la sidebar
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    api_ok = health.get("status") == "ok"
    st.sidebar.success("API : connectee") if api_ok else st.sidebar.warning("API : degradee (modele absent)")
except Exception:
    st.sidebar.error("API : hors ligne")

st.sidebar.markdown("---")
st.sidebar.caption(f"API : `{API_URL}`")

# ============================================================================
# PAGE 1 : TEST MODELE
# ============================================================================

if page == "Test Modele":
    st.title("Test du Modele sur le Holdout")
    st.markdown(
        "Selectionnez un client du jeu holdout (jamais vu pendant l'entrainement), "
        "soumettez ses features a l'API et comparez la prediction au vrai label."
    )

    df = load_holdout()
    if df is None:
        st.error(
            "Donnees holdout introuvables (`/app/data/holdout_sample.parquet`). "
            "Executez la derniere cellule du notebook `modelisation_undersampling.ipynb`."
        )
        st.stop()

    model_info = fetch_model_info()
    feature_cols = [c for c in df.columns if c != "TARGET"]

    # ---- Filtres ----
    col_filter, col_slider = st.columns([1, 2])
    with col_filter:
        filter_class = st.selectbox(
            "Filtrer par vrai label",
            ["Tous", "Non-defaut (0)", "Defaut (1)"],
        )

    df_filtered = df.copy()
    if filter_class == "Non-defaut (0)":
        df_filtered = df[df["TARGET"] == 0]
    elif filter_class == "Defaut (1)":
        df_filtered = df[df["TARGET"] == 1]

    with col_slider:
        idx = st.slider(
            f"Client ({len(df_filtered)} disponibles)",
            0, max(0, len(df_filtered) - 1), 0
        )

    row = df_filtered.iloc[idx]
    true_label = int(row["TARGET"])
    features   = row[feature_cols].to_dict()

    # ---- Layout principal ----
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Informations client")
        label_str   = "DEFAUT" if true_label == 1 else "NON-DEFAUT"
        label_color = "red" if true_label == 1 else "green"
        st.markdown(f"**Vrai label :** :{label_color}[{label_str}]")
        st.markdown(f"**Index dans holdout :** `{idx}`")

        if model_info:
            st.markdown(
                f"**Seuil business :** `{model_info['threshold_business']}`  "
                f"(FN = {model_info['business_cost_fn']}x FP)"
            )

        st.markdown("---")
        if st.button("Obtenir la prediction", type="primary", use_container_width=True):
            with st.spinner("Appel API en cours..."):
                result = call_predict(features, customer_id=f"holdout_{idx}")

            if result:
                st.session_state["last_result"] = result
                st.session_state["last_true"]   = true_label

    # Affichage du resultat (persiste entre les reruns)
    if "last_result" in st.session_state:
        result     = st.session_state["last_result"]
        true_label_saved = st.session_state["last_true"]
        prob       = result["probability"]
        pred_def   = result["prediction_default"]
        pred_biz   = result["prediction_business"]

        with left:
            st.markdown("---")
            st.subheader("Resultat")

            # Jauge de probabilite
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = prob * 100,
                title = {"text": "Proba de defaut (%)"},
                gauge = {
                    "axis":  {"range": [0, 100]},
                    "bar":   {"color": score_color(prob)},
                    "steps": [
                        {"range": [0, 30],  "color": "#d4edda"},
                        {"range": [30, 60], "color": "#fff3cd"},
                        {"range": [60, 100],"color": "#f8d7da"},
                    ],
                },
                number={"suffix": "%", "valueformat": ".1f"},
            ))
            fig_gauge.update_layout(height=220, margin=dict(t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric(
                "Decision seuil 0.5",
                "DEFAUT" if pred_def else "ACCORD",
                delta=None,
            )
            c2.metric(
                f"Decision seuil {result['threshold_business']}",
                "DEFAUT" if pred_biz else "ACCORD",
                delta=None,
            )

            is_correct = pred_biz == true_label_saved
            if is_correct:
                st.success(f"Prediction correcte (business threshold)")
            else:
                kind = "Faux Positif" if pred_biz == 1 else "Faux Negatif (FN x10)"
                st.error(f"Prediction incorrecte : {kind}")

            st.caption(f"Latence API : {result['processing_time_ms']:.1f} ms")

    with right:
        st.subheader("Features du client")
        feat_df = (
            pd.Series(features, name="Valeur")
            .rename_axis("Feature")
            .reset_index()
        )
        feat_df["Valeur"] = feat_df["Valeur"].apply(lambda x: round(x, 4))

        # Graphique barre pour visualiser les features
        fig_feat = px.bar(
            feat_df,
            x="Valeur", y="Feature",
            orientation="h",
            title="Valeurs des features (normalisees apres scaling interne)",
            height=max(300, len(feature_cols) * 28),
        )
        fig_feat.update_layout(margin=dict(l=150))
        st.plotly_chart(fig_feat, use_container_width=True)

# ============================================================================
# PAGE 2 : MONITORING
# ============================================================================

elif page == "Monitoring":
    st.title("Dashboard Monitoring Production")
    st.markdown(
        "Vue en temps reel des predictions enregistrees par l'API : "
        "distribution des scores, latences, taux de defaut."
    )

    auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=False)
    if auto_refresh:
        import time
        time.sleep(10)
        st.rerun()

    if st.button("Rafraichir maintenant"):
        st.rerun()

    metrics = fetch_metrics()
    if metrics is None:
        st.stop()

    total = metrics.get("total_predictions", 0)

    if total == 0:
        st.info(
            "Aucune prediction enregistree pour l'instant.  \n"
            "Allez sur la page **Test Modele** et soumettez quelques clients."
        )
        st.stop()

    # ---- KPI Row ----
    st.subheader("Indicateurs cles")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    dist = metrics["score_distribution"]
    rt   = metrics["response_time_ms"]
    pr   = metrics["prediction_rates"]

    kpi1.metric("Total predictions",    total)
    kpi2.metric("Score moyen",           f"{dist['mean']:.3f}")
    kpi3.metric("Taux defaut (0.5)",     f"{pr['default_rate_05']:.1%}")
    kpi4.metric("Taux defaut (business)",f"{pr['default_rate_business']:.1%}")
    kpi5.metric("Latence mediane",       f"{rt['p50']:.1f} ms")

    st.markdown("---")

    # ---- Graphiques ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution des scores")
        percentiles = ["min", "p50", "mean", "p90", "p95", "max"]
        values      = [dist[k] for k in percentiles]
        colors      = ["#2ecc71", "#3498db", "#3498db", "#e67e22", "#e74c3c", "#c0392b"]

        fig_scores = go.Figure(go.Bar(
            x=percentiles,
            y=values,
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        ))
        fig_scores.update_layout(
            yaxis=dict(range=[0, 1], title="Probabilite de defaut"),
            xaxis_title="Percentile",
            showlegend=False,
            height=350,
        )
        st.plotly_chart(fig_scores, use_container_width=True)

        # Distribution des predictions recentes sous forme d'histogramme approxime
        recent = metrics.get("recent_predictions", [])
        if recent:
            probs_recent = [r["probability"] for r in recent]
            fig_hist = px.histogram(
                x=probs_recent,
                nbins=20,
                title=f"Histogramme des {len(recent)} dernieres probabilites",
                labels={"x": "Probabilite", "y": "Nb predictions"},
                color_discrete_sequence=["#3498db"],
            )
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="gray",
                               annotation_text="seuil 0.5")
            fig_hist.update_layout(height=280)
            st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("Latences API (ms)")
        rt_labels = ["Moyenne", "P50", "P95", "P99"]
        rt_values = [rt["mean"], rt["p50"], rt["p95"], rt["p99"]]
        rt_colors = ["#3498db", "#3498db", "#e67e22", "#e74c3c"]

        fig_rt = go.Figure(go.Bar(
            x=rt_labels,
            y=rt_values,
            marker_color=rt_colors,
            text=[f"{v:.1f}" for v in rt_values],
            textposition="outside",
        ))
        fig_rt.update_layout(
            yaxis_title="ms",
            showlegend=False,
            height=350,
        )
        st.plotly_chart(fig_rt, use_container_width=True)

        # Taux de defaut comparatif
        st.subheader("Taux de defaut par seuil")
        fig_rates = go.Figure(go.Bar(
            x=["Seuil 0.5", "Seuil business"],
            y=[pr["default_rate_05"], pr["default_rate_business"]],
            marker_color=["#e67e22", "#e74c3c"],
            text=[f"{v:.1%}" for v in [pr["default_rate_05"], pr["default_rate_business"]]],
            textposition="outside",
        ))
        fig_rates.update_layout(
            yaxis=dict(range=[0, 1], title="Taux"),
            showlegend=False,
            height=280,
        )
        st.plotly_chart(fig_rates, use_container_width=True)

    # ---- Predictions recentes ----
    st.markdown("---")
    st.subheader("10 dernieres predictions")
    if recent:
        df_recent = pd.DataFrame(recent)

        # Colonne lisible pour la probabilite
        df_recent["prob_pct"] = (df_recent["probability"] * 100).round(1).astype(str) + "%"
        df_recent["decision_05"]  = df_recent["prediction_default"].map({0: "ACCORD", 1: "DEFAUT"})
        df_recent["decision_biz"] = df_recent["prediction_business"].map({0: "ACCORD", 1: "DEFAUT"})

        display_cols = ["timestamp", "customer_id", "prob_pct",
                        "decision_05", "decision_biz", "processing_time_ms"]
        display_cols = [c for c in display_cols if c in df_recent.columns]

        st.dataframe(
            df_recent[display_cols].rename(columns={
                "timestamp":          "Horodatage",
                "customer_id":        "Client",
                "prob_pct":           "Proba defaut",
                "decision_05":        "Decision (0.5)",
                "decision_biz":       "Decision (business)",
                "processing_time_ms": "Latence (ms)",
            }),
            use_container_width=True,
            hide_index=True,
        )
