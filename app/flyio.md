# Deploiement sur Fly.io

> Deploiement des deux services (API FastAPI + Dashboard Streamlit) sur Fly.io.
> Region : Paris (`cdg`). Free tier suffisant pour ce projet.

---

## Prerequis

### 1. Installer flyctl

```bash
# macOS / Linux
curl -L https://fly.io/install.sh | sh

# Windows (PowerShell)
pwsh -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://fly.io/install.ps1'))"

# macOS avec Homebrew
brew install flyctl
```

Verifier :
```bash
fly version
```

### 2. Creer un compte et se connecter

```bash
fly auth signup   # premiere fois
# ou
fly auth login    # si vous avez deja un compte
```

### 3. Generer les artefacts ML (obligatoire avant le deploiement)

Ouvrir `notebooks/modelisation_undersampling.ipynb` et **executer toutes les cellules**.
La derniere cellule genere :
```
app/models/model.pkl
app/models/model_metadata.json
app/data/holdout_sample.parquet
```

Ces fichiers sont **bakes dans les images Docker** lors du build Fly.io.

---

## Structure des fichiers Fly.io

```
app/
├── fly-api.toml            # Config Fly.io pour l'API
├── fly-streamlit.toml      # Config Fly.io pour Streamlit
├── Dockerfile.api.fly      # Dockerfile API (build context = app/)
├── Dockerfile.streamlit.fly# Dockerfile Streamlit (build context = app/)
├── models/                 # Bake dans l'image API
│   ├── model.pkl
│   └── model_metadata.json
└── data/                   # Bake dans l'image Streamlit
    └── holdout_sample.parquet
```

> **Pourquoi des Dockerfiles separes ?**
> Les `Dockerfile.*.fly` ont pour build context `app/` (lance depuis ce dossier),
> ce qui leur donne acces a `models/` et `data/`. Les Dockerfiles originaux
> dans `api/` et `streamlit/` ont un build context plus etroit, adapte a docker-compose local.

---

## Deploiement — etape par etape

**Toutes les commandes se lancent depuis le dossier `app/`.**

```bash
cd app
```

### Etape 1 — Creer l'application API

```bash
fly apps create homecredit-api --org personal
```

### Etape 2 — Creer le volume de logs (persistance des predictions)

```bash
fly volumes create homecredit_logs \
  --app homecredit-api \
  --region cdg \
  --size 1
```

> 1 GB de logs = environ 5 millions de predictions. Largement suffisant.

### Etape 3 — Deployer l'API

```bash
fly deploy --config fly-api.toml
```

Le build :
1. Construit l'image depuis `Dockerfile.api.fly` (context = `app/`)
2. Copie `models/` dans l'image
3. Pousse l'image vers le registry Fly.io
4. Lance la machine dans la region `cdg`
5. Monte le volume `homecredit_logs` sur `/app/logs`

Verifier que l'API repond :
```bash
fly status --app homecredit-api
curl https://homecredit-api.fly.dev/health
```

### Etape 4 — Creer l'application Streamlit

```bash
fly apps create homecredit-streamlit --org personal
```

### Etape 5 — Deployer Streamlit

```bash
fly deploy --config fly-streamlit.toml
```

Le Streamlit communique avec l'API via le **reseau prive Fly.io** :
`http://homecredit-api.internal:8000`
(resolu uniquement depuis l'interieur du reseau Fly, pas expose publiquement)

Verifier :
```bash
fly status --app homecredit-streamlit
```

Ouvrir le dashboard :
```bash
fly open --app homecredit-streamlit
```

---

## URLs finales

| Service    | URL publique                              |
|-----------|-------------------------------------------|
| API        | `https://homecredit-api.fly.dev`          |
| Swagger    | `https://homecredit-api.fly.dev/docs`     |
| Dashboard  | `https://homecredit-streamlit.fly.dev`    |

---

## Mettre a jour le modele

Quand vous re-entrainez le modele et relancez la cellule d'export :

```bash
# Rebuilder et redeployer uniquement l'API (le modele est bake dans l'image)
cd app
fly deploy --config fly-api.toml
```

Le Streamlit n'a pas besoin d'etre redeploy sauf si les donnees holdout changent.

---

## Commandes utiles

### Logs en temps reel
```bash
fly logs --app homecredit-api
fly logs --app homecredit-streamlit
```

### SSH dans un conteneur
```bash
fly ssh console --app homecredit-api
# Une fois dans le shell :
cat /app/logs/predictions.jsonl | tail -20
```

### Voir les metriques
```bash
fly status --app homecredit-api
fly status --app homecredit-streamlit
```

### Relancer une machine bloquee
```bash
fly machines restart --app homecredit-api
```

### Voir les machines et leur etat
```bash
fly machines list --app homecredit-api
```

### Scaler manuellement (si besoin)
```bash
# Passer a 512 MB si OOM (hors free tier, ~$2/mois)
fly scale memory 512 --app homecredit-api
```

### Arreter / reprendre
```bash
fly scale count 0 --app homecredit-api      # arreter
fly scale count 1 --app homecredit-api      # reprendre
```

### Supprimer tout
```bash
fly apps destroy homecredit-api
fly apps destroy homecredit-streamlit
```

---

## Free tier — limites et ce que ca couvre

| Ressource              | Inclus (free)    | Ce projet          | Commentaire                   |
|-----------------------|------------------|--------------------|-------------------------------|
| Machines shared-cpu   | 3 VMs            | 2 VMs              | OK                            |
| RAM par machine       | 256 MB           | 256 MB x 2         | Suffisant pour LightGBM leger |
| Volumes               | 3 GB total       | 1 GB (logs)        | OK                            |
| Bande passante sortante| 160 GB/mois     | < 1 GB/mois        | Largement OK                  |
| Regions               | Toutes           | cdg (Paris)        | OK                            |

> **Cold start :** avec `auto_stop_machines = "stop"`, les machines s'arretent apres ~5 min
> d'inactivite et redemarrent en ~2-3 secondes a la premiere requete.
> Pour eviter ca : `min_machines_running = 1` (mais sort du free tier, ~$1.94/mois/machine).

---

## Depannage

### "Error: app name already taken"
Les noms Fly.io sont globaux. Changez `homecredit-api` par `homecredit-api-VOTRENOM` dans
`fly-api.toml` et `fly-streamlit.toml` (et mettez a jour `API_URL` en consequence).

### "OOM killed" (Out of Memory)
```bash
fly scale memory 512 --app homecredit-api
```
Cout : ~$1.94/mois pour 512 MB (hors free tier).

### Streamlit ne joint pas l'API
Verifier que les deux apps sont dans la meme organisation Fly.io :
```bash
fly orgs list
fly status --app homecredit-api   # verifier qu'elle tourne
```
Le reseau `.internal` ne fonctionne qu'entre apps de la meme org.

### Volume non monte
```bash
fly volumes list --app homecredit-api
# Si vide : recreer le volume (etape 2) puis redeployer
```
