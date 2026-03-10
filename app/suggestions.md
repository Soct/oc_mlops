# Suggestions de deploiement — Home Credit Risk API

> Comparatif des plateformes pour deployer 2 conteneurs Docker :
> - **API FastAPI** (~256 MB RAM, CPU only, LightGBM inference)
> - **Dashboard Streamlit** (~128 MB RAM, web app legere)
> - **Volume persistant** pour les logs de predictions (`predictions.jsonl`)
>
> _Prix indicatifs au moment de la redaction (mars 2026). Verifier les tarifs actuels
> avant toute decision._

---

## TL;DR — Recommandation selon le contexte

| Contexte              | Recommandation     | Prix mensuel estimé |
|----------------------|--------------------|---------------------|
| Demo / portfolio      | Railway ou Render  | 0 – 5 €             |
| Production legere     | Fly.io             | 5 – 15 €            |
| Production scalable   | Google Cloud Run   | 0 – 10 €            |
| Hebergement FR/RGPD   | Scaleway           | 5 – 15 €            |
| Budget illimite       | AWS App Runner     | 15 – 40 €           |

---

## 1. Railway

**Lien :** https://railway.app

### Points forts
- Deploiement Docker en 2 minutes : `railway up`
- Detection automatique du `docker-compose.yml`
- Interface tres simple, pas de YAML Kubernetes a ecrire
- Volumes persistants inclus (meme sur le free tier)
- Logs en temps reel integres

### Points faibles
- Free tier limite a $5 de credit/mois (environ 500h de compute shared)
- Pas de region EU sur le free tier (US par defaut), EU disponible en payant
- Moins adapte si le trafic grossit beaucoup

### Tarification
| Offre       | Prix          | CPU          | RAM     | Remarques                        |
|-------------|---------------|--------------|---------|----------------------------------|
| Free        | $5 credit/mois| Shared       | 512 MB  | Sleeps apres inactivite          |
| Hobby       | $5/mois fixe  | Shared       | 512 MB  | Toujours actif, volumes inclus   |
| Pro         | Usage-based   | Shared/Dedi  | variable| ~$0.000463/vCPU-s + $0.000231/GB |

**Estimation pour ce projet (Hobby) :** ~$5/mois pour les 2 services (API + Streamlit)

### Deploiement
```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

---

## 2. Render

**Lien :** https://render.com

### Points forts
- Support Docker natif, tres populaire dans la communaute ML
- Disques persistants disponibles (pour les logs)
- Region EU (Frankfurt) disponible
- Interface claire, deploys automatiques via GitHub

### Points faibles
- Free tier : **le service s'arrete apres 15 min d'inactivite** (cold start ~30s)
- Les volumes persistants ne sont pas disponibles sur le free tier
- Pas de docker-compose : chaque service se deploie separement

### Tarification
| Offre             | Prix/service/mois | CPU      | RAM     | Remarques                    |
|-------------------|-------------------|----------|---------|------------------------------|
| Free              | 0 €               | Shared   | 512 MB  | Sleeps + pas de volume       |
| Starter           | ~7 €              | Shared   | 512 MB  | Toujours actif               |
| Standard          | ~21 €             | Shared   | 2 GB    | Pour prod serieuse           |
| Disque persistant | +1 € / GB / mois  | -        | -       | Necessaire pour les logs     |

**Estimation pour ce projet (2x Starter + 1 disque 1GB) :** ~15-16 €/mois

### Deploiement
Depuis le dashboard Render :
1. "New Web Service" > connecter repo GitHub > choisir `Dockerfile`
2. Definir les variables d'environnement (`API_URL`, `MODELS_DIR`, etc.)
3. Pour les volumes : "New Disk" > monter sur `/app/logs`

---

## 3. Fly.io

**Lien :** https://fly.io

### Points forts
- **Excellent free tier** : 3 VMs shared (256 MB chacune), 3 GB volumes
- Support Docker natif (`fly deploy`)
- Volumes persistants inclus dans le free tier
- Reseau prive entre les apps (parfait pour API + Streamlit)
- Regions EU disponibles (Amsterdam, Paris, Frankfurt)
- Tres bonnes performances pour le prix

### Points faibles
- CLI `flyctl` a prendre en main (un peu plus de courbe d'apprentissage que Railway)
- Pas de support `docker-compose` direct : fichier `fly.toml` par service
- Monitoring basique en free tier

### Tarification
| Ressource        | Free                | Paye                        |
|-----------------|--------------------|-----------------------------|
| Compute (shared)| 3 VMs 256MB inclus | ~$0.0000008/s par 256MB     |
| Volumes         | 3 GB inclus         | $0.15/GB/mois au-dela       |
| Bandwidth       | 160 GB/mois inclus  | $0.02/GB au-dela            |

**Estimation pour ce projet (free tier) :** **0 €/mois** tant que la RAM reste < 512 MB

**Estimation si depassement :** ~5-10 €/mois

### Deploiement
```bash
brew install flyctl   # ou winget install flyctl
fly auth login
fly launch           # dans app/api/
fly launch           # dans app/streamlit/
fly volumes create logs_data --size 1   # pour les logs
```

---

## 4. Google Cloud Run

**Lien :** https://cloud.google.com/run

### Points forts
- **Pay-per-request** : 0 € si personne ne l'utilise (parfait pour un projet de portfolio)
- Scale to zero automatique
- Free tier genereux : 2 millions de requetes/mois incluses
- Support Docker complet (via Google Container Registry ou Artifact Registry)
- Regions EU (europe-west1/Paris disponible)
- Tres bonnes performances a la demande

### Points faibles
- **Cold start** : ~1-3 secondes si le conteneur est a zero (acceptable pour une API ML)
- Pas de volumes persistants natifs → logs a envoyer vers Cloud Storage ou BigQuery
- Courbe d'apprentissage GCP (IAM, projets, billing)
- Le docker-compose ne fonctionne pas directement : deployer service par service

### Tarification (apres free tier)
| Ressource      | Prix                          |
|---------------|-------------------------------|
| CPU           | $0.00002400 / vCPU-seconde    |
| RAM           | $0.00000250 / GB-seconde      |
| Requetes      | $0.40 / million               |
| Egress EU     | $0.08 / GB                    |

**Estimation pour ce projet (faible trafic < 10 000 req/mois) :** ~0-2 €/mois

**Estimation pour ce projet (trafic moyen ~100 000 req/mois) :** ~3-8 €/mois

### Deploiement
```bash
gcloud run deploy homecredit-api \
  --image europe-west1-docker.pkg.dev/PROJECT/repo/api:latest \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --set-env-vars MODELS_DIR=/app/models

gcloud run deploy homecredit-streamlit \
  --image europe-west1-docker.pkg.dev/PROJECT/repo/streamlit:latest \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars API_URL=https://homecredit-api-XXXX-ew.a.run.app
```

> **Note logs :** Remplacer le fichier JSONL local par des logs vers **Cloud Logging**
> (gratuit jusqu'a 50 GB/mois) ou **Cloud Storage** (~$0.02/GB/mois).

---

## 5. Scaleway (hebergeur francais)

**Lien :** https://www.scaleway.com

### Points forts
- **Hebergeur francais, donnees en France** (Paris, Amsterdam) → RGPD natif
- **Serverless Containers** : similaire a Cloud Run, pay-per-use
- Bonne documentation en francais
- Offre Student / Startup accessible
- Prix competitifs vs AWS/GCP

### Points faibles
- Ecosysteme moins mature que GCP/AWS
- Pas de free tier perenne (credits de bienvenue de 100€ pour les nouveaux comptes)
- Volumes persistants : via Object Storage S3-compatible

### Tarification Serverless Containers
| Ressource      | Prix                           |
|---------------|--------------------------------|
| CPU           | €0.000000083 / vCPU-ms         |
| RAM           | €0.000000017 / Mo-ms           |
| Requetes      | €0.40 / million                |
| Premiers 400k | **Gratuits** / mois            |

**Estimation pour ce projet (faible trafic) :** ~0-3 €/mois

**Estimation pour ce projet (trafic moyen) :** ~5-12 €/mois

### Alternative : Instances DEV1-S
- 2 vCPU, 2 GB RAM, 20 GB SSD
- **€3.99/mois** (facture a l'heure)
- Heberger API + Streamlit sur la meme instance avec Docker Compose
- Volumes persistants sur disque local

---

## 6. OVHcloud (hebergeur francais)

**Lien :** https://www.ovhcloud.com/fr

### Points forts
- Hebergeur historique francais, tres present en Europe
- **VPS Starter : €3.50/mois** (1 vCPU, 2 GB RAM, 20 GB SSD)
- Donnees hebergees en France/Europe
- Docker disponible sur tous les VPS

### Points faibles
- Pas d'offre serverless mature
- Pas de scaling automatique sur les VPS
- Interface moins moderne que les concurrents
- Support moins reactif

### Tarification VPS
| Offre      | Prix/mois | vCPU | RAM   | Stockage |
|-----------|-----------|------|-------|----------|
| Starter   | €3.50     | 1    | 2 GB  | 20 GB    |
| Value     | €6.00     | 2    | 4 GB  | 40 GB    |
| Essential | €12.00    | 4    | 8 GB  | 80 GB    |

**Recommandation :** VPS Starter suffit largement pour ce projet (API legere + Streamlit).

### Deploiement sur VPS
```bash
# Sur le VPS OVH (Ubuntu 22.04)
apt install docker.io docker-compose-plugin
git clone <repo>
cd app
docker compose up -d
```

---

## 7. Hugging Face Spaces

**Lien :** https://huggingface.co/spaces

### Points forts
- **Gratuit** pour les apps Streamlit publiques
- Zero configuration : push le code, c'est deploye
- Ideal pour portfolio / demo visible publiquement
- Support Docker (Spaces SDK = Docker)
- Tres connu dans la communaute ML

### Points faibles
- **API FastAPI separee impossible** en free tier (1 seul Space = 1 seul service)
- Le Streamlit devra appeler l'API ailleurs (pas de reseau interne)
- Pas de volumes persistants en free tier
- Les Spaces publics mettent les modeles en cache 72h d'inactivite (puis endormi)

### Tarification Spaces
| Offre           | Prix/mois | CPU        | RAM    |
|----------------|-----------|------------|--------|
| Free (public)  | 0 €       | 2 vCPU     | 16 GB  |
| Persistent     | $5        | 2 vCPU     | 16 GB  |
| T4 GPU Small   | $60       | GPU NVIDIA | 16 GB  |

> **Usage recommande :** Deployer le Streamlit en Space gratuit, avec l'API sur
> Google Cloud Run ou Fly.io. Lien entre les deux via variable d'environnement `API_URL`.

---

## Tableau comparatif global

| Plateforme        | Prix/mois (projet) | Free tier      | Docker Compose | Volume logs | Region FR/EU | Facilite |
|------------------|--------------------|----------------|----------------|-------------|--------------|----------|
| Railway           | 0-5 €              | Oui (limite)   | Oui natif      | Oui         | EU (payant)  | ★★★★★    |
| Render            | 14-16 €            | Oui (sleeps)   | Non            | Oui (paye)  | EU           | ★★★★☆    |
| Fly.io            | 0-10 €             | Oui (genereux) | Non (fly.toml) | Oui         | EU (Paris)   | ★★★★☆    |
| Google Cloud Run  | 0-8 €              | Oui (2M req)   | Non            | Cloud Logging| EU (Paris)  | ★★★☆☆    |
| Scaleway          | 0-12 €             | Credits 100€   | Oui (VPS)      | Oui (VPS)   | FR natif     | ★★★☆☆    |
| OVHcloud VPS      | 3.50-6 €           | Non            | Oui natif      | Oui natif   | FR natif     | ★★★☆☆    |
| HF Spaces         | 0-5 €              | Oui            | Non (1 service)| Non         | EU           | ★★★★★    |

---

## Recommandation detaillee selon l'usage

### Pour un projet OpenClassrooms / portfolio
> Objectif : montrer que ca tourne, zero budget

**Option A (zero euro) :** Fly.io free tier
- API FastAPI sur Fly.io (free)
- Streamlit sur Fly.io (free)
- Volume logs sur Fly.io (3 GB free)
- Region Amsterdam ou Paris

**Option B (simple) :** Railway Hobby ($5/mois)
- Un seul compte, tout au meme endroit
- Support docker-compose natif
- URL propre incluse

---

### Pour une production legere (startup, POC interne)
> Objectif : disponibilite 99%, cout maitrise

**Recommandation : Google Cloud Run + Scaleway Object Storage**
- API et Streamlit en Cloud Run (pay-per-use, scale to zero)
- Logs vers Scaleway Object Storage (~$0.01/mois pour quelques MB)
- Cout total estimé : **2-8 €/mois**

---

### Pour une production RGPD stricte (donnees sensibles)
> Objectif : donnees en France, conformite legale

**Recommandation : Scaleway Serverless Containers ou OVHcloud VPS**
- Hebergement France garanti
- OVHcloud VPS Starter a €3.50/mois est suffisant pour ce workload
- Scaleway Serverless est plus elegant mais necessite d'adapter le logging (S3)

---

## Adapter le logging pour le cloud

Le fichier `predictions.jsonl` actuel fonctionne parfaitement en local/Docker.
En production cloud sans volume persistant (Cloud Run, HF Spaces), deux alternatives :

### Option 1 : Remplacer par un service de logging cloud
```python
# Dans api/main.py, remplacer l'ecriture fichier par :
import logging
logger = logging.getLogger("predictions")
logger.info(json.dumps(log_entry))
# Cloud Run, Fly.io, Railway capturent les logs stdout automatiquement
```

### Option 2 : Ecrire vers S3/Object Storage
```python
import boto3  # deja dans pyproject.toml !
s3 = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT"))
s3.put_object(
    Bucket=os.getenv("S3_BUCKET"),
    Key=f"logs/{datetime.utcnow().date()}/predictions.jsonl",
    Body="\n".join(...)
)
```
boto3 est deja dans le `pyproject.toml` du projet — cette option est donc gratuite a implementer.
Scaleway Object Storage et MinIO (deja dans votre docker-compose.yml racine !) sont 100% compatibles S3.
