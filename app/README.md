# Home Credit Default Risk — Stack applicative

> API FastAPI + Dashboard Streamlit, Docker-ready.

---

## Architecture

```
app/
├── docker-compose.yml       # Orchestre les deux services
├── README.md                # Ce fichier
│
├── api/                     # Service 1 : API de prediction
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py              # FastAPI (predict, health, metrics)
│
├── streamlit/               # Service 2 : Dashboard interactif
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py               # Streamlit (test holdout + monitoring)
│
├── models/                  # Artefacts ML (generes par le notebook)
│   ├── model.pkl            # Pipeline LightGBM + StandardScaler
│   └── model_metadata.json  # Features, seuils, metriques holdout
│
├── data/                    # Donnees holdout (generees par le notebook)
│   └── holdout_sample.parquet
│
└── logs/                    # Logs de predictions (volume Docker)
    └── predictions.jsonl    # 1 ligne JSON par prediction
```

---

## Plan d'action

### Etape 1 — Generer les artefacts (notebook)

Ouvrir `notebooks/modelisation_undersampling.ipynb` et **executer toutes les cellules** dans l'ordre.

La derniere cellule export :
- `app/models/model.pkl` — pipeline LightGBM + StandardScaler (20 features)
- `app/models/model_metadata.json` — features, seuils, metriques
- `app/data/holdout_sample.parquet` — 1000 exemples equilbres (500 defauts / 500 OK)

> **Pourquoi 20 features ?** Le modele reduit (SHAP top-20) atteint des performances
> quasi-identiques au modele complet (795 features) avec 93.7% de reduction de complexite.
> Avantage majeur en production : predictions plus rapides, modele plus leger, moins de
> risques de data drift sur des features inutiles.

### Etape 2 — Lancer l'application

```bash
cd app
docker-compose up --build
```

- **API** disponible sur http://localhost:8000
- **Dashboard** disponible sur http://localhost:8501
- **Docs API** (Swagger) sur http://localhost:8000/docs

### Etape 3 — Verifier le deploiement

```bash
# Sante de l'API
curl http://localhost:8000/health

# Informations sur le modele
curl http://localhost:8000/model-info

# Test de prediction (exemple)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "test_001", "features": {"EXT_SOURCE_3": 0.52}}'
```

---

## API FastAPI — Endpoints

| Methode | Route          | Description                                      |
|---------|---------------|--------------------------------------------------|
| GET     | `/health`      | Statut de l'API et du modele                     |
| GET     | `/model-info`  | Metadonnees (features, seuils, metriques holdout)|
| POST    | `/predict`     | Prediction pour un client                        |
| GET     | `/metrics`     | Metriques de monitoring temps reel               |

### Schema `/predict`

**Request :**
```json
{
  "customer_id": "client_001",
  "features": {
    "EXT_SOURCE_3": 0.52,
    "EXT_SOURCE_2": 0.61,
    "DAYS_BIRTH": -12000
  }
}
```

**Response :**
```json
{
  "customer_id":          "client_001",
  "probability":          0.234,
  "prediction_default":   0,
  "prediction_business":  0,
  "threshold_default":    0.5,
  "threshold_business":   0.23,
  "processing_time_ms":   4.7,
  "decision_label":       "ACCORD   (proba=23.4%)"
}
```

> **Deux seuils de decision :**
> - `threshold_default = 0.5` — seuil standard, aucun biais metier
> - `threshold_business` — calcule sur le holdout pour minimiser `FN*10 + FP*1`
>   (un faux negatif = accorder un credit qui fera defaut = 10x plus couteux qu'un faux positif)

---

## Dashboard Streamlit

### Page "Test Modele"

- Selecteur de client parmi les 1000 exemples holdout
- Filtre par vrai label (defaut / non-defaut)
- Soumission a l'API en un clic
- Jauge de probabilite + deux decisions (seuil 0.5 et seuil business)
- Verdict : prediction correcte / faux positif / faux negatif

### Page "Monitoring"

Metriques calculees en temps reel sur l'historique des predictions :

| Metrique                     | Description                                  |
|-----------------------------|----------------------------------------------|
| Distribution des scores      | min, p50, mean, p90, p95, max                |
| Histogramme des probabilites | 20 bins sur les 10 dernieres predictions     |
| Latences API                 | mean, p50, p95, p99 en ms                    |
| Taux de defaut               | Comparaison seuil 0.5 vs seuil business      |
| Predictions recentes         | Tableau des 10 dernieres predictions         |

> **En production**, ces metriques permettent de detecter :
> - Un **data drift** : le score moyen derive significativement des metriques holdout
> - Une **degradation de performance** : le taux de defaut s'eloigne des attentes
> - Des **problemes de latence** : les P95/P99 augmentent anormalement

---

## Monitoring des logs

Chaque prediction est ecrite dans `/app/logs/predictions.jsonl` (1 JSON par ligne) :

```json
{
  "timestamp":           "2026-03-10T14:22:01.123456",
  "customer_id":         "holdout_42",
  "probability":         0.312456,
  "prediction_default":  0,
  "prediction_business": 1,
  "processing_time_ms":  3.841
}
```

Ce fichier est persiste dans un volume Docker nomme (`logs_data`) et survit aux redemarrages.

---

## Variables d'environnement

| Variable      | Service    | Defaut              | Description                       |
|--------------|-----------|---------------------|-----------------------------------|
| `MODELS_DIR`  | api        | `/app/models`       | Chemin vers model.pkl + metadata  |
| `LOGS_DIR`    | api        | `/app/logs`         | Chemin pour predictions.jsonl     |
| `API_URL`     | streamlit  | `http://api:8000`   | URL de l'API FastAPI              |
| `DATA_DIR`    | streamlit  | `/app/data`         | Chemin vers holdout_sample.parquet|

---

## Prerequis

- Docker >= 20.10
- Docker Compose >= 2.0
- Le notebook `modelisation_undersampling.ipynb` doit avoir ete execute jusqu'a la fin
  pour que `app/models/` et `app/data/` contiennent les fichiers necessaires.
