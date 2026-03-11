"""
Home Credit Default Risk - Dashboard Streamlit
Deux pages :
  - Test Modele    : tester le modele sur des exemples du holdout
  - Monitoring     : distribution des scores, latences, predictions recentes
"""

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

# ============================================================================
# CONFIGURATION
# ============================================================================

API_URL  = os.getenv("API_URL",  "http://localhost:8000")
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))

st.set_page_config(
    page_title="Home Credit Risk",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- CSS global ----
st.markdown("""
<style>
/* Badges de risque */
.risk-low    {background:#d4edda;border-left:5px solid #28a745;padding:12px 18px;
              border-radius:6px;color:#155724;font-size:1.15rem;font-weight:700;}
.risk-medium {background:#fff3cd;border-left:5px solid #ffc107;padding:12px 18px;
              border-radius:6px;color:#856404;font-size:1.15rem;font-weight:700;}
.risk-high   {background:#f8d7da;border-left:5px solid #dc3545;padding:12px 18px;
              border-radius:6px;color:#721c24;font-size:1.15rem;font-weight:700;}
/* Carte info client */
.info-card   {background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;
              padding:14px 18px;margin-bottom:12px;line-height:1.9;}
.info-card b {color:#495057;}
/* Séparateur léger */
hr.soft      {border:none;border-top:1px solid #dee2e6;margin:1rem 0;}
</style>
""", unsafe_allow_html=True)

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
        st.error("Impossible de joindre l'API. Vérifiez que le conteneur `api` est démarré.")
    except Exception as e:
        st.error(f"Erreur API : {e}")
    return None


def fetch_metrics() -> dict | None:
    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Impossible de joindre l'API. Vérifiez que le conteneur `api` est démarré.")
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


def risk_badge(prob: float) -> str:
    """Retourne le HTML du badge de niveau de risque."""
    if prob < 0.3:
        return f'<div class="risk-low">Risque FAIBLE — {prob*100:.1f}%</div>'
    if prob < 0.6:
        return f'<div class="risk-medium">Risque MODÉRÉ — {prob*100:.1f}%</div>'
    return f'<div class="risk-high">Risque ÉLEVÉ — {prob*100:.1f}%</div>'


def score_color(prob: float) -> str:
    if prob < 0.3:
        return "#28a745"
    if prob < 0.6:
        return "#ffc107"
    return "#dc3545"


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("Home Credit Risk")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Test Modele", "Monitoring"], label_visibility="collapsed")

# Statut API dans la sidebar
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    api_ok = health.get("status") == "ok"
    st.sidebar.success("API : connectée") if api_ok else st.sidebar.warning("API : dégradée — modèle non chargé")
except Exception:
    st.sidebar.error("API : hors ligne — vérifiez le conteneur")

st.sidebar.markdown("---")

# Infos modèle dans la sidebar
_minfo = fetch_model_info()
if _minfo:
    st.sidebar.markdown("**Infos modèle**")
    st.sidebar.caption(
        f"Seuil de décision business : **{_minfo['threshold_business']}**  \n"
        f"Un faux négatif (client défaillant non détecté) coûte **{_minfo['business_cost_fn']}×** "
        f"plus qu'un faux positif (bon client refusé)."
    )

st.sidebar.markdown("---")
st.sidebar.caption(f"Endpoint API : `{API_URL}`")

# ============================================================================
# PAGE 1 : TEST MODELE
# ============================================================================

if page == "Test Modele":
    st.title("Test du Modèle sur le Holdout")
    st.markdown(
        """
        Cette page vous permet de tester le modèle de scoring sur des clients du jeu **holdout** —
        c'est-à-dire des clients mis de côté avant l'entraînement, que le modèle n'a jamais vus.

        **Comment utiliser cette page :**
        1. Filtrez les clients par vrai label si vous souhaitez tester un profil spécifique (défaut ou non-défaut).
        2. Faites glisser le curseur pour choisir un client.
        3. Cliquez sur **Obtenir la prédiction** pour envoyer les données à l'API.
        4. Comparez la décision du modèle (colonne de gauche) avec les valeurs des features (colonne de droite).
        """
    )

    df = load_holdout()
    if df is None:
        st.error(
            "Données holdout introuvables (`/app/data/holdout_sample.parquet`). "
            "Exécutez la dernière cellule du notebook `modelisation_undersampling.ipynb`."
        )
        st.stop()

    model_info = fetch_model_info()
    feature_cols = [c for c in df.columns if c != "TARGET"]

    # ---- Filtres ----
    col_filter, col_slider = st.columns([1, 2])
    with col_filter:
        filter_class = st.selectbox(
            "Filtrer par vrai label",
            ["Tous", "Non-défaut (0)", "Défaut (1)"],
        )

    df_filtered = df.copy()
    if filter_class == "Non-défaut (0)":
        df_filtered = df[df["TARGET"] == 0]
    elif filter_class == "Défaut (1)":
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

        label_str   = "DÉFAUT"   if true_label == 1 else "NON-DÉFAUT"
        label_color = "#721c24" if true_label == 1 else "#155724"
        label_bg    = "#f8d7da" if true_label == 1 else "#d4edda"

        threshold_line = (
            f"Seuil de décision business : <b>{model_info['threshold_business']}</b><br>"
            f"Au-dessus de ce seuil, le client est classé en défaut. "
            f"Un faux négatif coûte <b>{model_info['business_cost_fn']}×</b> plus qu'un faux positif, "
            f"ce qui justifie un seuil plus bas que 0.5."
            if model_info else "Seuil business : non disponible"
        )

        st.markdown(
            f'<div class="info-card">'
            f'<b>Vrai label (réalité terrain) :</b> <span style="color:{label_color};background:{label_bg};'
            f'padding:2px 8px;border-radius:4px;font-weight:700;">{label_str}</span><br>'
            f'<b>Index dans le holdout :</b> {idx}<br>'
            f'<b>Nombre de features envoyées à l\'API :</b> {len(feature_cols)}<br>'
            f'<hr class="soft">'
            f'{threshold_line}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.button("Obtenir la prédiction", type="primary", use_container_width=True):
            with st.spinner("Appel API en cours..."):
                result = call_predict(features, customer_id=f"holdout_{idx}")

            if result:
                st.session_state["last_result"] = result
                st.session_state["last_true"]   = true_label

    # Affichage du resultat (persiste entre les reruns)
    if "last_result" in st.session_state:
        result           = st.session_state["last_result"]
        true_label_saved = st.session_state["last_true"]
        prob             = result["probability"]
        pred_def         = result["prediction_default"]
        pred_biz         = result["prediction_business"]

        with left:
            st.markdown('<hr class="soft">', unsafe_allow_html=True)
            st.subheader("Résultat de la prédiction")

            # Badge de niveau de risque
            st.markdown(risk_badge(prob), unsafe_allow_html=True)
            st.caption(
                "Faible : probabilité < 30 % — le modèle estime que le client remboursera son crédit. "
                "Modéré : 30–60 % — profil incertain, à surveiller. "
                "Élevé : > 60 % — risque important de défaut de paiement."
            )
            st.markdown("")  # espaceur

            # Jauge de probabilité
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = prob * 100,
                title = {"text": "Probabilité de défaut", "font": {"size": 14}},
                gauge = {
                    "axis":  {"range": [0, 100], "ticksuffix": "%"},
                    "bar":   {"color": score_color(prob), "thickness": 0.3},
                    "steps": [
                        {"range": [0, 30],  "color": "#d4edda"},
                        {"range": [30, 60], "color": "#fff3cd"},
                        {"range": [60, 100],"color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line":  {"color": "#6c757d", "width": 2},
                        "thickness": 0.75,
                        "value": result.get("threshold_business", 0.5) * 100,
                    },
                },
                number={"suffix": "%", "valueformat": ".1f", "font": {"size": 28}},
            ))
            fig_gauge.update_layout(height=230, margin=dict(t=50, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Décisions côte à côte
            st.markdown("**Décisions selon les deux seuils :**")
            c1, c2 = st.columns(2)
            c1.metric(
                "Seuil standard (0.5)",
                "DÉFAUT" if pred_def else "ACCORD",
                help="Seuil neutre : le client est classé en défaut si la probabilité dépasse 50 %.",
            )
            c2.metric(
                f"Seuil business ({result['threshold_business']})",
                "DÉFAUT" if pred_biz else "ACCORD",
                help=(
                    f"Seuil ajusté au coût métier : abaissé à {result['threshold_business']} "
                    "pour réduire les faux négatifs, plus coûteux que les faux positifs."
                ),
            )

            # Verdict
            is_correct = pred_biz == true_label_saved
            if is_correct:
                st.success("Prédiction correcte au seuil business.")
            else:
                if pred_biz == 1:
                    st.error(
                        "Faux Positif : le modèle prédit un défaut alors que le client est solvable. "
                        "Conséquence : refus abusif d'un bon client."
                    )
                else:
                    st.error(
                        "Faux Négatif : le modèle accorde un crédit à un client qui fera défaut. "
                        f"Ce type d'erreur coûte {result.get('threshold_business', '?')} fois plus cher qu'un faux positif."
                    )

            st.caption(f"Temps de traitement API : {result['processing_time_ms']:.1f} ms")

    with right:
        st.subheader("Features du client")
        st.markdown(
            """
            Ce graphique montre les **valeurs de toutes les features** envoyées au modèle pour ce client.
            Les features sont triées par **valeur absolue décroissante** : les plus éloignées de zéro
            (en haut) ont généralement plus de poids dans la décision du modèle.

            Les valeurs ont été normalisées lors du prétraitement (centrage-réduction).
            Une valeur positive (bleue) signifie que le client est au-dessus de la moyenne sur cet indicateur ;
            une valeur négative (rouge) signifie qu'il est en dessous.
            """
        )

        feat_series = row[feature_cols].sort_values(key=abs, ascending=True)
        colors = ["#dc3545" if v < 0 else "#0d6efd" for v in feat_series.values]

        fig_feat = go.Figure(go.Bar(
            x=feat_series.values,
            y=feat_series.index,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in feat_series.values],
            textposition="outside",
        ))
        fig_feat.add_vline(x=0, line_color="#adb5bd", line_width=1)
        fig_feat.update_layout(
            title="Valeurs des features (triées par valeur absolue)",
            xaxis_title="Valeur normalisée",
            yaxis_title="",
            height=max(350, len(feature_cols) * 26),
            margin=dict(l=10, r=60),
            plot_bgcolor="#fafafa",
            paper_bgcolor="#ffffff",
        )
        fig_feat.update_xaxes(showgrid=True, gridcolor="#dee2e6")
        st.plotly_chart(fig_feat, use_container_width=True)

# ============================================================================
# PAGE 2 : MONITORING
# ============================================================================

# ============================================================================
# PAGE 2 : MONITORING
# ============================================================================

elif page == "Monitoring":
    st.title("Monitoring — Prédictions en production")
    st.markdown(
        """
        Ce tableau de bord agrège les prédictions en temps réel effectuées par l'API depuis son démarrage.
        Il permet de surveiller la **santé du modèle en production** : dérive des scores, taux de défaut,
        performances de l'infrastructure (latences).

        Les données sont lues directement depuis le fichier de logs de l'API (`logs/predictions.jsonl`).
        Cliquez sur **Rafraîchir maintenant** pour recharger les métriques, ou activez le rafraîchissement
        automatique toutes les 10 secondes dans la barre latérale.
        """
    )

    auto_refresh = st.sidebar.checkbox("Auto-refresh toutes les 10 s", value=False)
    if auto_refresh:
        import time
        time.sleep(10)
        st.rerun()

    if st.button("Rafraîchir maintenant"):
        st.rerun()

    metrics = fetch_metrics()
    if metrics is None:
        st.stop()

    total = metrics.get("total_predictions", 0)

    if total == 0:
        st.info(
            "Aucune prédiction enregistrée pour l'instant.  \n"
            "Allez sur la page **Test Modele** et soumettez quelques clients pour voir les métriques apparaître ici."
        )
        st.stop()

    # ---- KPI Row ----
    st.subheader("Indicateurs clés")
    st.caption(
        "Ces cinq métriques donnent une vue instantanée de l'état de la production. "
        "Le score moyen et les taux de défaut permettent de détecter une dérive du modèle ; "
        "la latence médiane mesure la santé de l'infrastructure."
    )
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    dist = metrics["score_distribution"]
    rt   = metrics["response_time_ms"]
    pr   = metrics["prediction_rates"]

    dr05  = pr["default_rate_05"]
    dr_biz = pr["default_rate_business"]

    kpi1.metric(
        "Total prédictions",
        total,
        help="Nombre total de requêtes de prédiction traitées depuis le démarrage de l'API.",
    )
    kpi2.metric(
        "Score moyen",
        f"{dist['mean']:.3f}",
        help="Probabilité de défaut moyenne sur l'ensemble des prédictions. Une valeur élevée peut indiquer une dérive.",
    )
    kpi3.metric(
        "Taux défaut (seuil 0.5)",
        f"{dr05:.1%}",
        delta=f"{'Elevé' if dr05 > 0.3 else ''}",
        delta_color="inverse" if dr05 > 0.3 else "off",
        help="Part des clients classés en défaut avec le seuil standard de 0.5.",
    )
    kpi4.metric(
        "Taux défaut (seuil business)",
        f"{dr_biz:.1%}",
        delta=f"{'Elevé' if dr_biz > 0.3 else ''}",
        delta_color="inverse" if dr_biz > 0.3 else "off",
        help="Part des clients classés en défaut avec le seuil business ajusté au coût métier.",
    )
    kpi5.metric(
        "Latence médiane",
        f"{rt['p50']:.1f} ms",
        help="Temps médian de traitement d'une requête par l'API. L'objectif est d'être sous 100 ms.",
    )

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ---- Graphiques ligne 1 ----
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Distribution des scores de risque")
        st.caption(
            "Chaque barre représente le nombre de clients dont la probabilité de défaut tombe dans cet intervalle. "
            "Un histogramme concentré à gauche (zone verte) indique que la majorité des demandeurs "
            "présentent un faible risque. Une accumulation vers la droite (zone rouge) mérite attention."
        )

        # Histogramme des prédictions récentes
        recent = metrics.get("recent_predictions", [])
        if recent:
            probs_recent = [r["probability"] for r in recent]
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=probs_recent,
                nbinsx=20,
                marker_color="#0d6efd",
                opacity=0.8,
                name="Prédictions",
            ))
            fig_hist.add_vline(x=0.5,
                               line_dash="dash", line_color="#6c757d", line_width=1.5,
                               annotation_text="Seuil 0.5",
                               annotation_position="top right",
                               annotation_font_color="#6c757d")
            biz_thr = metrics.get("threshold_business")
            if biz_thr and biz_thr != 0.5:
                fig_hist.add_vline(x=biz_thr,
                                   line_dash="dot", line_color="#dc3545", line_width=1.5,
                                   annotation_text=f"Seuil business ({biz_thr})",
                                   annotation_position="top left",
                                   annotation_font_color="#dc3545")
            # Zones colorées
            fig_hist.add_vrect(x0=0,    x1=0.3, fillcolor="#d4edda", opacity=0.15, line_width=0)
            fig_hist.add_vrect(x0=0.3,  x1=0.6, fillcolor="#fff3cd", opacity=0.25, line_width=0)
            fig_hist.add_vrect(x0=0.6,  x1=1,   fillcolor="#f8d7da", opacity=0.25, line_width=0)
            fig_hist.update_layout(
                xaxis=dict(title="Probabilité de défaut", range=[0, 1], tickformat=".0%"),
                yaxis_title="Nombre de prédictions",
                legend_title="",
                height=320,
                plot_bgcolor="#fafafa",
                paper_bgcolor="#ffffff",
                margin=dict(t=20, b=40),
            )
            fig_hist.update_xaxes(showgrid=True, gridcolor="#dee2e6")
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption(
                "Zone verte : risque faible (probabilité < 30 %). "
                "Zone jaune : risque modéré (30–60 %). "
                "Zone rouge : risque élevé (> 60 %). "
                "Les lignes pointillées indiquent les seuils de décision."
            )
        else:
            # Percentiles si pas de données récentes
            percentiles = ["min", "p50", "mean", "p90", "p95", "max"]
            values      = [dist[k] for k in percentiles]
            labels      = ["Min", "Médiane", "Moyenne", "P90", "P95", "Max"]
            bar_colors  = ["#28a745","#0d6efd","#0d6efd","#ffc107","#fd7e14","#dc3545"]
            fig_scores  = go.Figure(go.Bar(
                x=labels, y=values,
                marker_color=bar_colors,
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
            ))
            fig_scores.update_layout(
                yaxis=dict(range=[0, 1.1], title="Probabilité de défaut"),
                height=320, plot_bgcolor="#fafafa",
            )
            st.plotly_chart(fig_scores, use_container_width=True)

    with col2:
        st.markdown("#### Latences API (ms)")
        st.caption(
            "Temps de traitement d'une requête de prédiction, mesuré côté API. "
            "La moyenne et la médiane (P50) reflètent le comportement habituel. "
            "P95 et P99 mesurent les cas les plus lents : ils doivent idéalement rester sous 100 ms."
        )
        rt_labels = ["Moyenne", "Médiane (P50)", "P95", "P99"]
        rt_values = [rt["mean"], rt["p50"], rt["p95"], rt["p99"]]
        rt_colors = ["#0d6efd", "#0d6efd", "#ffc107", "#dc3545"]

        fig_rt = go.Figure(go.Bar(
            x=rt_labels,
            y=rt_values,
            marker_color=rt_colors,
            text=[f"{v:.1f} ms" for v in rt_values],
            textposition="outside",
        ))
        fig_rt.add_hline(y=100, line_dash="dash", line_color="#adb5bd", line_width=1,
                         annotation_text="Objectif < 100 ms", annotation_font_color="#6c757d")
        fig_rt.update_layout(
            yaxis_title="Millisecondes (ms)",
            showlegend=False,
            height=240,
            plot_bgcolor="#fafafa",
            paper_bgcolor="#ffffff",
            margin=dict(t=20, b=10),
        )
        fig_rt.update_yaxes(showgrid=True, gridcolor="#dee2e6")
        st.plotly_chart(fig_rt, use_container_width=True)

        # Taux de défaut comparatif
        st.markdown("#### Taux de défaut par seuil")
        st.caption(
            "Comparaison du taux de défaut selon le seuil utilisé. "
            "Le seuil business est volontairement plus bas que 0.5 : "
            "il classe davantage de clients en défaut pour limiter les faux négatifs, "
            "dont le coût métier est bien supérieur à celui des faux positifs."
        )
        fig_rates = go.Figure(go.Bar(
            x=["Seuil standard (0.5)", f"Seuil business ({metrics.get('threshold_business','?')})"],
            y=[dr05, dr_biz],
            marker_color=["#fd7e14", "#dc3545"],
            text=[f"{v:.1%}" for v in [dr05, dr_biz]],
            textposition="outside",
        ))
        fig_rates.update_layout(
            yaxis=dict(range=[0, 1.1], title="Taux de défaut", tickformat=".0%"),
            showlegend=False,
            height=240,
            plot_bgcolor="#fafafa",
            paper_bgcolor="#ffffff",
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig_rates, use_container_width=True)

    # ---- Prédictions récentes ----
    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    st.subheader("Dernières prédictions enregistrées")
    st.caption(
        "Tableau des prédictions les plus récentes reçues par l'API. "
        "La colonne 'Proba défaut' affiche une barre de progression proportionnelle au risque estimé. "
        "Les colonnes 'Décision' indiquent la conclusion du modèle selon chacun des deux seuils."
    )
    recent = metrics.get("recent_predictions", [])
    if recent:
        df_recent = pd.DataFrame(recent)
        df_recent["prob_float"] = df_recent["probability"]
        df_recent["decision_05"]  = df_recent["prediction_default"].map({0: "Accord", 1: "Défaut"})
        df_recent["decision_biz"] = df_recent["prediction_business"].map({0: "Accord", 1: "Défaut"})

        display_cols = ["timestamp", "customer_id", "prob_float",
                        "decision_05", "decision_biz", "processing_time_ms"]
        display_cols = [c for c in display_cols if c in df_recent.columns]

        st.dataframe(
            df_recent[display_cols].rename(columns={
                "timestamp":          "Horodatage",
                "customer_id":        "Client",
                "prob_float":         "Proba défaut",
                "decision_05":        "Décision (0.5)",
                "decision_biz":       "Décision (business)",
                "processing_time_ms": "Latence (ms)",
            }),
            column_config={
                "Proba défaut": st.column_config.ProgressColumn(
                    "Proba défaut",
                    help="Probabilité de défaut estimée par le modèle",
                    format="%.1f%%",
                    min_value=0,
                    max_value=1,
                ),
                "Latence (ms)": st.column_config.NumberColumn(
                    "Latence (ms)",
                    format="%.1f ms",
                ),
            },
            use_container_width=True,
            hide_index=True,
        )
