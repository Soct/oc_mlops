# Home Credit Default Risk — MLOps

Projet MLOps de scoring de risque de défaut de paiement (Home Credit).  
Modèle **LightGBM** entraîné avec sélection de features SHAP, exposé via une **API FastAPI** et un **dashboard Streamlit**, déployable sur **Fly.io**.

## Structure du projet

```
├── notebooks/               # Exploration, modélisation, analyse du drift
├── app/
│   ├── api/                 # API FastAPI (prédiction + monitoring)
│   ├── streamlit/           # Dashboard Streamlit (test modèle + drift)
│   ├── scripts/             # Scripts utilitaires (split holdout)
│   ├── models/              # Modèle sérialisé + métadonnées
│   ├── data/                # Données holdout (référence + test)
│   ├── logs/                # Logs de prédictions (predictions.jsonl)
│   └── docker-compose.yml   # Orchestration locale
├── tests/                   # Tests unitaires et d'intégration
├── .github/workflows/       # Pipeline CI/CD (GitHub Actions)
├── mlruns/                  # Artefacts MLflow (tracking expérimentations)
└── pyproject.toml           # Dépendances et configuration projet
```

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/Soct/oc_mlops.git
cd oc_mlops

# Créer l'environnement avec uv
uv venv && source .venv/bin/activate
uv pip install -e ".[test]"
```

## Lancement local (Docker)

```bash
cd app
docker compose up --build
# API      : http://localhost:8000
# Streamlit: http://localhost:8501
```

## Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

## CI/CD

Le pipeline GitHub Actions (`.github/workflows/ci.yml`) s'exécute sur chaque `push` / `pull_request` sur `main` :
1. Installation des dépendances
2. Exécution des tests avec couverture
3. Build de l'image Docker de l'API (vérification de déployabilité)

## Déploiement

Déploiement sur **Fly.io** :
```bash
cd app
fly deploy --config fly-api.toml        # API
fly deploy --config fly-streamlit.toml   # Dashboard
```

## Monitoring & Data Drift

- **Logging prod** : chaque prédiction est enregistrée dans `predictions.jsonl` (timestamp, features, probabilité, décision, latence).
- **Data drift** : le dashboard Streamlit intègre une page « Data Drift » utilisant **Evidently** pour comparer les distributions des features entre les données de référence et les données reçues par l'API.
- **Métriques temps réel** : endpoint `/metrics` (distribution des scores, latences, taux de défaut).

## Endpoints API

| Méthode | Route              | Description                          |
|---------|--------------------|--------------------------------------|
| GET     | `/health`          | Statut de l'API et du modèle         |
| GET     | `/model-info`      | Métadonnées du modèle déployé        |
| POST    | `/predict`         | Prédiction pour un client            |
| GET     | `/metrics`         | Métriques de monitoring temps réel   |
| GET     | `/prediction-logs` | Historique des prédictions (+ features) |
