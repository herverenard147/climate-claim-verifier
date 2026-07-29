 🌍 TERRAVA-AI : Plateforme d'Intelligence et de Fact-Checking Climatique

**TERRAVA-AI** (anciennement ClimaCheck) est un outil de vérification des faits (fact-checking) propulsé par l'Intelligence Artificielle. Conçu spécifiquement pour les journalistes, chercheurs et décideurs climatiques, il permet de confronter instantanément une déclaration aux données officielles de la littérature scientifique (GIEC, OMM, Banque Mondiale) afin de lutter contre la désinformation climatique.


  Fonctionnalités Principales

- **Détection Anti-Désinformation :** Évalue si une affirmation est `CONFIRMÉE`, `RÉFUTÉE` ou `NON VÉRIFIABLE` par la science.
- **Architecture Zéro-GPU :** Modèle hybride ultra-optimisé combinant une base vectorielle (FAISS) et un classifieur de Machine Learning (Régression Logistique) capable de tourner sur un simple ordinateur CPU local.
- **Traçabilité Totale :** Les sources institutionnelles exactes ayant servi à la décision sont toujours affichées à l'utilisateur (Citations, Liens, Années).
- **Analyse de Documents (PDF) :** Importez un document par Glisser-Déposer pour extraire instantanément le texte et lancer l'analyse.
- **Filtre Régional :** Sélecteur de zone géographique (Global / Afrique de l'Ouest / Côte d'Ivoire) côté interface. Le corpus contient un focus institutionnel sur cette région (GIEC, OMM, Banque Mondiale), mais **le filtre n'est pas encore appliqué côté backend** : `zone_geo` est transmis à l'API mais actuellement ignoré par `check_claim()` dans `main.py` — la recherche FAISS reste globale quelle que soit la zone sélectionnée. *(Limitation connue, pas un bug caché : à implémenter si le filtrage réel par région est souhaité.)*


 Déploiement en production (Render)

- **Frontend (interface) :** https://terrava-ai-frontend.onrender.com
- **Backend (API) :** https://terrava-ai-backend.onrender.com (documentation interactive : `/docs`)
- **Dépôt utilisé pour le déploiement :** [`herverenard147/climate-claim-verifier`](https://github.com/herverenard147/climate-claim-verifier) — un fork du dépôt d'origine. Nécessaire car l'installation de la GitHub App de Render est limitée aux dépôts appartenant directement au compte GitHub connecté ; un accès collaborateur/push sur un dépôt tiers ne suffit pas (même contrainte déjà rencontrée sur un autre projet, NouanKanyAI).
- **Configuration :** `render.yaml` déclare les deux services (backend `Docker`, frontend site statique) avec injection automatique des URLs publiques croisées (`CORS_ORIGINS` / `VITE_API_BASE_URL`) — voir `DOCUMENTATION_TECHNIQUE.md` section 12 pour le détail.

**Limites connues de cet environnement de déploiement** (plan gratuit Render, vérifiées par des tests réels sur les URLs ci-dessus, pas supposées) :

- **Démarrage à froid (cold start) :** le plan gratuit met le service en veille après une période d'inactivité. Le premier appel après une veille peut prendre 30 à 60 secondes (redémarrage du conteneur + rechargement du modèle `all-MiniLM-L6-v2` en mémoire), contre une réponse quasi instantanée une fois le service « chaud ».
- **Disque éphémère (historique et feedback perdus à chaque redéploiement) :** `history.db` (SQLite) est recréé vide à chaque redémarrage/redéploiement du conteneur — **aucune persistance entre déploiements** sur le plan gratuit. Un disque persistant Render existe pour lever cette limite mais est une option **payante**, non activée dans ce déploiement.
- **Upload de PDF volumineux/scanné : non fiable sur le plan gratuit.** Testé en conditions réelles avec un vrai rapport GIEC AR6 (18 pages, 6,9 Mo) : le traitement fait planter le processus backend (probable dépassement de RAM du plan gratuit) et rend **tout le service indisponible** pendant son redémarrage automatique (30 à 60 secondes, toutes les routes). Les fichiers légers ou à texte natif (sans recours au fallback OCR) fonctionnent normalement et n'ont provoqué aucun incident lors des tests. Cause probable : le pipeline `pdf2image`/`pytesseract` (rendu d'image à 200 DPI par page) s'ajoute à la mémoire déjà occupée par le modèle chargé, ce qui peut dépasser la RAM allouée par le plan gratuit. **Non corrigé dans le cadre de ce déploiement** : la correction (limiter/désactiver l'OCR en production, ou passer à un plan payant avec plus de RAM) implique un arbitrage de coût ou de fonctionnalité qui n'a pas été tranché.
- **Seuil anti-hallucination (0.20) :** confirmé fonctionnel comme documenté (section 4 de `DOCUMENTATION_TECHNIQUE.md`) pour un texte réellement dénué de sens (ex. `"???!!!###"`). En revanche, un claim hors-sujet en langage naturel « normal » (ex. un fait sur le Bitcoin ou le football) ne déclenche généralement **pas** ce seuil : son score cosinus avec le corpus climatique se situe typiquement entre 0.30 et 0.37, au-dessus du seuil, à cause du plancher naturel de similarité des embeddings de phrases (une paire de phrases sans aucun rapport a rarement une similarité cosinus proche de 0). Limite pré-existante du modèle d'embeddings, révélée ici en testant contre l'instance de production avec plusieurs exemples réels.

 Architecture Technique (SaaS)

Le projet a été refondu pour adopter un standard industriel **Full-Stack** :

 1. Le "Cerveau" : Back-End (Python / FastAPI)
L'API REST est exposée via `main.py` et orchestre :
- L'encodeur de similarité sémantique (`all-MiniLM-L6-v2`).
- La base de connaissances vectorielle (`FAISS`).
- L'algorithme de logique de vérité scientifique (`Joblib / Régression Logistique scikit-learn`, `class_weight='balanced'`, `max_iter=1000`).
- Le parseur natif de PDF (`PyPDF2`).

### 2. Le "Visage" : Front-End (React / Tailwind CSS)
Une interface moderne "Scientific Workbench" gérée dans le dossier `/frontend` :
- Créée avec **Vite + React + TypeScript**.
- Composants stylisés sur-mesure via **Tailwind CSS**.
- Design épuré, accessible et totalement exempt de jargon technique.


 Guide d'Installation et d'Exécution

 Prérequis
- **Python 3.9+** (Pour l'API IA)
- **Node.js 18+** (Pour l'interface React)
- **tesseract-ocr** (dépendance système, pas un paquet Python) : requis pour l'OCR de repli sur les PDF scannés (pages sans couche texte). Sans ce binaire, l'upload de PDF continue de fonctionner normalement pour les documents à texte natif — seules les pages scannées restent vides, comme avant l'ajout de l'OCR. Installation (Debian/Ubuntu) :
  ```bash
  sudo apt install tesseract-ocr tesseract-ocr-fra
  ```
  `poppler-utils` (nécessaire à `pdf2image` pour convertir une page PDF en image) est généralement déjà présent sur ces distributions ; sinon : `sudo apt install poppler-utils`.

 Étape 1 : Lancer le Serveur IA (Back-End)

```bash
# Dans le dossier principal du projet :
pip install -r requirements.txt

# Démarrer l'API sur le port 8000
uvicorn main:app --host 127.0.0.1 --port 8000
```
*L'API est désormais disponible sur `http://localhost:8000/docs`.*

> ⚠️ **Le port 8000 est parfois déjà utilisé par un autre service local** (observé sur certaines machines : un autre projet écoutant sur ce port). Si `uvicorn` refuse de démarrer avec une erreur `address already in use`, relancez-le sur un autre port (ex. `--port 8001`) et configurez le frontend en conséquence — voir ci-dessous.

### Étape 2 : Lancer l'Interface Graphique (Front-End)

```bash
# Dans un NOUVEAU terminal, se rendre dans le dossier frontend :
cd frontend

# Installer les dépendances
npm install

# (Optionnel) Si le backend ne tourne pas sur le port 8000 par défaut :
cp .env.example .env
# puis éditez frontend/.env et définissez VITE_API_BASE_URL sur l'URL réelle
# de votre backend, ex. VITE_API_BASE_URL=http://localhost:8001

# Lancer le serveur de développement React
npm run dev
```
*Le portail TERRAVA-AI s'ouvrira sur `http://localhost:5173`.*

L'URL de l'API backend appelée par le frontend est configurable via la variable d'environnement `VITE_API_BASE_URL` (fichier `frontend/.env`, non commité — voir `frontend/.env.example`). Par défaut, en l'absence de ce fichier, le frontend appelle `http://localhost:8000`. **`frontend/.env` doit être adapté à chaque machine** si le port 8000 y est indisponible ; ne codez jamais l'URL du backend en dur dans le code source (`frontend/src/config.ts` centralise cette valeur).

---

 Les Scripts de Modélisation (MLOps)
Ce dépôt inclut également les scripts ayant servi à la conception de l'IA (idéal pour la mise à jour des rapports). **L'ordre d'exécution ci-dessous est important** :

| Ordre | Script | Rôle |
|---|---|---|
| 1 | `1_prepare_data.py` | Télécharge le dataset Climate-FEVER (HuggingFace `datasets`) et génère `data/train.csv`, `data/val.csv`, `data/test.csv`, ainsi qu'un `data/corpus.csv` **brut** (une seule colonne `evidence`, sans métadonnées). |
| 2 | `migrate_csv.py` | Ajoute les colonnes `institution` / `title` / `year` / `url` au corpus brut. **Étape obligatoire avant l'étape 4** — sans elle, les preuves issues de Climate-FEVER s'affichent sans source exploitable. |
| 3 | `2_build_retrieval.py` | Encode `data/corpus.csv` avec `all-MiniLM-L6-v2` et construit l'index vectoriel FAISS (`models_saved/faiss_index.bin`). |
| 4 | `4_ingest_documents.py` | Ajoute les documents institutionnels (GIEC AR6, OMM, Banque Mondiale) au corpus avec leurs métadonnées, puis reconstruit l'index FAISS avec l'ensemble enrichi. |
| 5 *(optionnel)* | `update_corpus.py` | Fusionne `data/corpus_additionnel.csv` (affirmations régionales additionnelles) dans le corpus et reconstruit l'index FAISS. |
| — | `3_train_classifier.py` | Entraîne la Régression Logistique (`class_weight='balanced'`, `C=0.1`, `max_iter=1000`) sur `train.csv` + `val.csv`, évalue une seule fois sur `test.csv`, et sauvegarde `models_saved/classifier.joblib`. Indépendant du corpus/FAISS : peut être lancé à tout moment après l'étape 1. |
| — *(optionnel)* | `scripts/veille_giec_omm.py` | Vérifie si de nouveaux rapports GIEC/OMM sont apparus en ligne et le signale (log + rapport dans `veille_reports/`) — **détection uniquement, n'ingère rien automatiquement**. Voir `DOCUMENTATION_TECHNIQUE.md` section 11 pour le détail et un exemple de planification cron. |

> ⚠️ **Attention — ne pas casser le corpus enrichi.** `data/corpus.csv` est déjà fourni dans ce dépôt en version enrichie (Climate-FEVER **+** documents institutionnels **+** métadonnées `institution/title/year/url`). **Ne relance pas `1_prepare_data.py` isolément** : ce script écrase `data/corpus.csv` par une version brute sans métadonnées ni documents institutionnels. Si tu dois régénérer les données depuis zéro, exécute bien la séquence complète `1 → migrate_csv → 2 → 4 → (5)` pour reconstruire un corpus équivalent, sans quoi les sources GIEC/OMM/Banque Mondiale disparaîtront de l'application et les preuves Climate-FEVER afficheront une institution manquante.

---

 Performance mesurée du classificateur

Chiffres obtenus en exécutant réellement `3_train_classifier.py` sur `data/test.csv` (1040 paires claim-evidence, jamais vues pendant l'entraînement ni le réglage des hyperparamètres) :

| Métrique | Valeur mesurée |
|---|---|
| **Macro-F1** (global) | **0.53** |
| F1 — SUPPORTS | 0.64 |
| F1 — NOT_ENOUGH_INFO | 0.52 |
| F1 — REFUTES | 0.44 |
| Accuracy | 0.56 |

**Méthodologie et honnêteté sur ce chiffre :**
- Le jeu de données provient à 100 % de Climate-FEVER (Wikipedia). Le corpus institutionnel (GIEC/OMM/Banque Mondiale) ne représente que 3 chunks sur ~4870 dans `data/corpus.csv` et ne participe à aucune paire claim-evidence étiquetée — il sert uniquement à la récupération FAISS, jamais à l'entraînement ou à l'évaluation du classificateur.
- Réglage : `C=0.1` (régularisation L2 renforcée par rapport à `C=1.0` par défaut, qui surapprend sur 1536 features pour ~4830 exemples) sélectionné par grid search sur `data/val.csv`, jamais sur `data/test.csv`. Le modèle final est ensuite réentraîné sur `train.csv + val.csv` et évalué **une seule fois** sur `test.csv`.
- Pistes testées et **écartées** car elles n'ont pas amélioré le Macro-F1 mesuré : calibration Platt (`CalibratedClassifierCV`, sigmoid) — fait baisser le Macro-F1 test à 0.46 ; rééquilibrage manuel des poids de classes au-delà de `class_weight='balanced'` — aucune configuration testée ne bat `'balanced'` ; repondération post-hoc du seuil de décision (argmax pondéré) — gain apparent sur validation qui ne se généralise pas sur test (signe de surapprentissage au set de validation, taille limitée à 1035 exemples).
- REFUTES reste la classe la plus difficile (F1=0.44) : le classificateur confond souvent une affirmation qui *contredit* une preuve avec une affirmation qu'elle *confirme*, quand les deux portent sur le même sujet — limite connue d'une approche par similarité d'embeddings sans modélisation explicite de la négation.
- Ce chiffre (Macro-F1 ≈ 0.53) remplace toute estimation antérieure ~0.72 qui n'apparaissait dans aucun fichier de ce dépôt (README, code, frontend) — elle provenait uniquement du cahier des charges initial du hackathon, pas d'une mesure réelle sur ce jeu de données.

---
*Ce projet a été développé dans le cadre du Hackathon "TTA W3" pour proposer une solution frugale (Zéro-GPU) et à fort impact dans la lutte contre la désinformation climatique.*
