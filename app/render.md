# Deploiement sur Render

> Deploiement des deux services (API FastAPI + Dashboard Streamlit) sur Render.
> Region : Frankfurt (la plus proche de Paris). Free tier compatible.

---

## Prerequis

### 1. Creer un compte Render

Aller sur [render.com](https://render.com) et creer un compte gratuit.

### 2. Connecter le depot Git

Dans le dashboard Render, connecter votre compte GitHub/GitLab au depot.

### 3. Generer les artefacts ML (obligatoire avant le deploiement)

Ouvrir `notebooks/modelisation_undersampling.ipynb` et **executer toutes les cellules**.
La derniere cellule genere :

```
app/models/model.pkl
app/models/model_metadata.json
app/data/holdout_sample.parquet
```

Ces fichiers sont **bakes dans les images Docker** lors du build Render.
**Committer et pousser ces fichiers sur le depot Git avant de deployer.**

---

## Structure des fichiers Render

```
app/
├── render.yaml                  # Blueprint Render (ce fichier configure les deux services)
├── Dockerfile.api.render        # Image Docker de l'API (modele bake)
├── Dockerfile.streamlit.render  # Image Docker du dashboard (donnees holdout bakes)
```

---

## Deploiement

### Option A — Blueprint (recommande)

Le fichier `render.yaml` definit les deux services en une seule operation.

1. Dans le dashboard Render, cliquer sur **"New +"** → **"Blueprint"**
2. Selectionner le depot Git
3. Indiquer le chemin du blueprint : `app/render.yaml`
4. Cliquer sur **"Apply"** — Render cree et deploie l'API et le Streamlit

### Option B — Services manuels

Creer chaque service manuellement via **"New +" → "Web Service"** :

**API :**

| Champ              | Valeur                          |
|--------------------|---------------------------------|
| Runtime            | Docker                          |
| Dockerfile Path    | `app/Dockerfile.api.render`     |
| Docker Context     | `app`                           |
| Plan               | Free (ou Starter pour les logs) |
| Region             | Frankfurt                       |
| Env MODELS_DIR     | `/app/models`                   |
| Env LOGS_DIR       | `/app/logs`                     |

**Streamlit :**

| Champ              | Valeur                                         |
|--------------------|------------------------------------------------|
| Runtime            | Docker                                         |
| Dockerfile Path    | `app/Dockerfile.streamlit.render`              |
| Docker Context     | `app`                                          |
| Plan               | Free                                           |
| Region             | Frankfurt                                      |
| Env API_URL        | `https://homecredit-api.onrender.com` (voir §) |
| Env DATA_DIR       | `/app/data`                                    |

---

## Mettre a jour l'URL de l'API apres deploiement

Render assigne automatiquement l'URL `https://<nom-du-service>.onrender.com`.
Si le service API s'appelle `homecredit-api`, l'URL est :

```
https://homecredit-api.onrender.com
```

Si Render a assigne un nom different (ex. `homecredit-api-xxxx`), mettre a jour
la variable d'environnement `API_URL` dans le service Streamlit :

1. Dashboard Render → Service `homecredit-streamlit` → **Environment**
2. Modifier `API_URL` avec la bonne URL
3. Cliquer sur **"Save Changes"** — Render redeploit automatiquement

---

## Persistence des logs (optionnel, plan paye)

Sur le **free tier**, les logs de predictions (`predictions.jsonl`) sont **perdus
a chaque redemarrage**. Le service fonctionne normalement, seul l'historique
des logs est efface.

Pour persister les logs sur un **Persistent Disk** ($0.25/GB/mois, plan Starter
minimum a $7/mois) :

1. Dashboard Render → Service `homecredit-api` → **Disks**
2. Ajouter un disque : nom `homecredit-logs`, mount path `/app/logs`, taille 1 GB
3. Le bloc `disk` dans `render.yaml` est deja configure pour cela

---

## Differences avec Fly.io

| Aspect              | Fly.io                              | Render                                  |
|---------------------|-------------------------------------|-----------------------------------------|
| Config              | `fly-api.toml` + `fly-streamlit.toml` | `render.yaml` (blueprint unique)      |
| Reseau prive        | `.internal` DNS                     | URLs publiques HTTPS                    |
| Scale to zero       | Oui (free tier)                     | Oui (free tier, apres 15 min)           |
| Volumes             | Inclus dans le free tier            | Disks payants ($0.25/GB/mois)           |
| RAM free tier       | 256 MB                              | 512 MB                                  |
| Region Paris        | `cdg`                               | `frankfurt` (le plus proche)            |
| Deploiement         | CLI `fly deploy`                    | Push Git ou dashboard                   |
