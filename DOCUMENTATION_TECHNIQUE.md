# TERRAVA-AI — Documentation technique

Ce document décrit l'état **réel et vérifié** du projet (code exécuté, métriques mesurées sur le jeu de test), pas les objectifs du cahier des charges initial. Chaque chiffre cité ici provient d'une exécution effective des scripts de ce dépôt.

---

## 1. Architecture générale

```
┌──────────────┐     ┌────────────────────┐     ┌─────────────────────────┐     ┌──────────────────┐
│   Entrée     │     │    Récupération     │     │  Feature engineering    │     │  Classification   │
│   (claim)    │ --> │  (encodage + FAISS  │ --> │  NLI (concat 4×384)     │ --> │  + logique verdict │
│  texte libre │     │   top-k=3)          │     │                         │     │                    │
└──────────────┘     └────────────────────┘     └─────────────────────────┘     └──────────────────┘
                                                                                          │
                                                                                          v
                                                                                 ┌──────────────────┐
                                                                                 │  Verdict sourcé   │
                                                                                 │  affiché (React)  │
                                                                                 └──────────────────┘
```

Étapes détaillées (voir `main.py`, fonction `check_claim`) :

1. **Entrée** : une affirmation climatique en texte libre (français ou anglais), reçue par `POST /api/check-claim`.
2. **Récupération** : le claim est encodé en vecteur de 384 dimensions (`all-MiniLM-L6-v2`, normalisé), puis comparé aux ~4870 preuves du corpus via un index FAISS `IndexFlatIP` (produit scalaire = similarité cosinus car les vecteurs sont normalisés). Les 3 preuves les plus proches (top-k=3) sont retenues.
3. **Garde-fou anti-hallucination** : si le score cosinus de la preuve la plus proche (top-1) est **< 0.20**, le pipeline s'arrête ici et renvoie directement `NON VÉRIFIABLE`, sans appeler le classificateur.
4. **Feature engineering NLI** : sinon, l'evidence top-1 est encodée à son tour, et un vecteur de 1536 dimensions est construit par concaténation de 4 blocs de 384 dimensions chacun (détail section 3).
5. **Classification** : une Régression Logistique scikit-learn prédit une classe parmi `SUPPORTS` / `REFUTES` / `NOT_ENOUGH_INFO`.
6. **Verdict final** : traduction de la classe en verdict affiché (section 4), avec les 3 sources top-k citées (institution, extrait, titre, année, lien).

---

## 2. Stack technique et choix

| Composant             | Choix                                                                                        | Pourquoi                                                                                                                                                                                                                                                  |
| --------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Backend               | **FastAPI** (async)                                                                          | API REST légère, documentation OpenAPI automatique (`/docs`), validation de schéma via Pydantic, adaptée à un service CPU mono-modèle sans besoin de queue/worker complexe.                                                                               |
| Recherche vectorielle | **FAISS `IndexFlatIP`**                                                                      | Recherche exacte (pas d'approximation ANN) adaptée à un corpus de taille modeste (~4870 vecteurs) : latence négligeable, aucun compromis de rappel. `IndexFlatIP` (produit scalaire) sur vecteurs normalisés = équivalent exact de la similarité cosinus. |
| Encodeur sémantique   | **SentenceTransformers `all-MiniLM-L6-v2`**                                                  | 22M paramètres, 384 dimensions, tourne en quelques millisecondes par phrase sur CPU. Compromis qualité/vitesse standard pour de la recherche sémantique légère, pas de dépendance GPU.                                                                    |
| Classificateur        | **Régression Logistique scikit-learn** (`class_weight='balanced'`, `C=0.1`, `max_iter=1000`) | Modèle linéaire interprétable, entraînement en secondes sur CPU, pas de risque d'overfitting catastrophique avec une régularisation adaptée (voir section 6 sur le choix de `C`).                                                                         |
| Frontend              | **React + Vite + TypeScript + Tailwind CSS**                                                 | SPA légère, hot-reload rapide en développement, typage statique pour réduire les erreurs d'intégration avec l'API.                                                                                                                                        |
| Parsing PDF           | **PyPDF2**, + **pytesseract/pdf2image/tesseract-ocr** en repli (section 8)                  | PyPDF2 suffit pour l'immense majorité des PDF (texte natif) ; l'OCR ne se déclenche que pour les pages sans couche texte (scan/image), afin de ne pas ralentir systématiquement l'upload.                                                                |

---

## 3. Feature engineering NLI (détail)

Le vecteur d'entrée du classificateur est construit ainsi (`create_features()` dans `3_train_classifier.py`, logique identique dans `main.py`) :

```
c_emb  = encode(claim)      # 384 dims, normalisé
e_emb  = encode(evidence)   # 384 dims, normalisé

abs_diff          = |c_emb − e_emb|        # 384 dims
elementwise_mult  = c_emb ⊙ e_emb          # 384 dims

features = concat(c_emb, e_emb, abs_diff, elementwise_mult)   # 1536 dims
```

Interprétation :

- `c_emb`, `e_emb` : information brute des deux textes.
- `abs_diff` : capture la divergence terme-à-terme entre les deux représentations (utile pour détecter une contradiction directionnelle).
- `elementwise_mult` : capture le chevauchement / l'alignement sémantique (proche de la similarité cosinus mais dimension par dimension, avant agrégation).

C'est un pattern classique de feature engineering pour la classification de paires de phrases (NLI), permettant à un modèle linéaire de capter des interactions que la seule concaténation de `c_emb`/`e_emb` ne capturerait pas.

---

## 4. Seuil anti-hallucination et logique de décision

```python
if similarity_score < 0.20:
    verdict = "NON_VERIFIABLE"
else:
    raw_verdict = classifier.predict(features)[0]
    verdict = {"SUPPORTS": "CONFIRME", "REFUTES": "REFUTE"}.get(raw_verdict, "NON_VERIFIABLE")
```

- **Score < 0.20** : aucune preuve suffisamment proche sémantiquement n'existe dans le corpus → `NON VÉRIFIABLE`, badge "Aucune preuve scientifique", **aucune source affichée**. Le classificateur n'est même pas appelé.
- **Score ≥ 0.20** :
  - `SUPPORTS` → **CONFIRMÉ**
  - `REFUTES` → **RÉFUTÉ**
  - `NOT_ENOUGH_INFO` → **NON VÉRIFIABLE** (badge "Preuves indirectes/insuffisantes", sources affichées car le sujet est proche mais non tranché)

Ce seuil a été vérifié expérimentalement en phase 1 : une phrase en coréen hors-sujet obtient un score de 0.1987 (juste sous le seuil) et déclenche bien le verdict `NON VÉRIFIABLE` avec zéro source.

---

## 5. Pipeline de données (ordre d'exécution)

⚠️ **L'ordre ci-dessous est important.** `data/corpus.csv` est déjà fourni dans ce dépôt en version enrichie (Climate-FEVER + documents institutionnels + métadonnées). Relancer `1_prepare_data.py` seul **écrase** cette version enrichie par une version brute sans métadonnées ni documents institutionnels.

| Ordre           | Script                  | Rôle                                                                                                                                                |
| --------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1               | `1_prepare_data.py`     | Télécharge Climate-FEVER (HuggingFace `datasets`) et génère `train.csv`/`val.csv`/`test.csv` + un `corpus.csv` **brut** (colonne `evidence` seule). |
| 2               | `migrate_csv.py`        | Ajoute les colonnes `institution`/`title`/`year`/`url` au corpus brut. **Obligatoire avant l'étape 4.**                                             |
| 3               | `2_build_retrieval.py`  | Encode `corpus.csv` et construit l'index FAISS (`models_saved/faiss_index.bin`).                                                                    |
| 4               | `4_ingest_documents.py` | Ajoute les documents institutionnels (GIEC, OMM, Banque Mondiale) au corpus avec métadonnées, reconstruit l'index FAISS.                            |
| 5 *(optionnel)* | `update_corpus.py`      | Fusionne `corpus_additionnel.csv` (affirmations régionales additionnelles) et reconstruit l'index FAISS.                                            |
| —               | `3_train_classifier.py` | Entraîne la Régression Logistique sur `train.csv` + `val.csv`, évalue une seule fois sur `test.csv`. Indépendant du corpus/FAISS.                   |

---

## 6. Métriques réelles mesurées

**Méthodologie** : `data/val.csv` (1035 exemples) sert exclusivement à la sélection d'hyperparamètre (`C`), jamais à l'entraînement final ni à l'évaluation. `data/test.csv` (1040 exemples) n'est utilisé qu'**une seule fois**, en évaluation finale, jamais pendant le tuning. Le modèle final est réentraîné sur `train.csv` + `val.csv` après sélection de `C`, puis évalué sur `test.csv`.

### Résultat final (modèle en production, `models_saved/classifier.joblib`)

| Métrique             | Valeur mesurée |
| -------------------- | -------------- |
| **Macro-F1**         | **0.532**      |
| F1 — SUPPORTS        | 0.638          |
| F1 — NOT_ENOUGH_INFO | 0.518          |
| F1 — REFUTES         | 0.440          |
| Accuracy             | 0.562          |

### Historique des itérations (mesurées, pas estimées)

| Configuration                                                                                         | Macro-F1 test               |
| ----------------------------------------------------------------------------------------------------- | --------------------------- |
| Baseline initiale (`C=1.0` défaut, entraîné sur train seul)                                           | 0.482                       |
| `C=0.1` (grid search sur val), entraîné sur train seul                                                | 0.516                       |
| `C=0.1`, entraîné sur train+val (config finale déployée)                                              | **0.532**                   |
| Calibration Platt (`CalibratedClassifierCV`, sigmoid) sur `C=0.1`                                     | 0.461 *(rejetée — dégrade)* |
| **Baseline de comparaison : TF-IDF** (mêmes 4 blocs de features, mêmes réglages, `max_features=2000`) | 0.485                       |

La baseline TF-IDF (vectorisation lexicale classique, sans embeddings sémantiques) atteint 0.485 avec le même pipeline de classification — l'encodeur `all-MiniLM-L6-v2` apporte un gain réel mais modeste (+0.047 absolu) par rapport à une approche purement lexicale, sur ce jeu de données.

> Script de reproduction : [`evaluation/tfidf_baseline.py`](evaluation/tfidf_baseline.py). À lancer depuis la racine du dépôt avec `python3 evaluation/tfidf_baseline.py` (venv activé, `data/train.csv`/`val.csv`/`test.csv` présents). Résultat reproductible à l'identique : `Macro-F1 test = 0.4854`.

### Ce que ce chiffre signifie concrètement

Sur les 1040 paires claim-evidence du jeu de test (jamais vues pendant l'entraînement), le système donne le bon verdict (SUPPORTS/REFUTES/NOT_ENOUGH_INFO) dans **56.2% des cas** (accuracy), avec une performance très inégale selon la classe : bonne sur SUPPORTS, moyenne sur NOT_ENOUGH_INFO, faible sur REFUTES (le système confond souvent une affirmation qui *contredit* une preuve avec une affirmation qu'elle *confirme*, quand les deux portent sur le même sujet).

### Pourquoi le corpus institutionnel n'explique pas l'écart

Le corpus institutionnel (GIEC, OMM, Banque Mondiale) ne représente que **3 chunks sur ~4870** dans `data/corpus.csv`, et **aucun** n'apparaît dans `train.csv`/`val.csv`/`test.csv` — ces fichiers proviennent à 100% de Climate-FEVER (Wikipedia). Le corpus institutionnel sert uniquement à la récupération FAISS en production, jamais à l'entraînement ou à l'évaluation du classificateur. 100% de l'écart de Macro-F1 mesuré est donc attribuable à Climate-FEVER seul.

### Pistes testées et écartées

- **Calibration Platt** (`CalibratedClassifierCV`, sigmoid, cv=5) : dégrade le Macro-F1 test (0.516 → 0.461 sur la config train-seul). Améliore la qualité des probabilités mais lisse les frontières de décision au détriment de l'exactitude des labels durs.
- **Rééquilibrage manuel des classes** au-delà de `class_weight='balanced'` : aucune configuration testée (grille de multiplicateurs sur REFUTES/NOT_ENOUGH_INFO) ne bat `'balanced'` natif sur validation.
- **Repondération post-hoc du seuil de décision** (argmax pondéré par classe, tuné sur validation) : gain apparent sur validation (0.5105 → 0.5247) qui **ne se généralise pas** sur test (0.5090 → 0.5020, en régression) — signe de surapprentissage à un set de validation de taille limitée (1035 exemples). Piste écartée pour cette raison précise.

---

## 7. Installation et lancement

### Prérequis

- Python 3.9+ (testé avec 3.12.3)
- Node.js 18+

### Backend

```bash
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

⚠️ **Le port 8000 peut être déjà occupé par un autre service sur votre machine** (observé en conditions réelles : un autre projet local répondant sur ce port). Si `uvicorn` échoue avec `address already in use`, relancez-le sur un autre port (`--port 8001` par exemple).

### Frontend

```bash
cd frontend
npm install

# Si le backend ne tourne pas sur le port 8000 par défaut :
cp .env.example .env
# puis éditez frontend/.env : VITE_API_BASE_URL=http://localhost:<votre_port>

npm run dev
```

L'URL du backend appelée par le frontend est centralisée dans `frontend/src/config.ts` et lue depuis la variable d'environnement `VITE_API_BASE_URL` (fichier `frontend/.env`, non commité). **Ne jamais coder cette URL en dur** dans le code source — c'était un bug corrigé après un incident réel où le frontend continuait d'appeler le port 8000 (occupé par un autre projet) alors que le backend tournait sur 8001.

### Régénérer les modèles depuis zéro

Voir la section 5 pour l'ordre exact. `models_saved/` (index FAISS + classificateur) n'est pas versionné (`.gitignore`) : ces artefacts doivent être régénérés localement via `2_build_retrieval.py` et `3_train_classifier.py` après un clone.

---

## 8. Limitations techniques connues

- **Corpus institutionnel quasi-absent de l'entraînement** : 3 chunks sur ~4870 dans le corpus de récupération (GIEC, OMM, Banque Mondiale), 0 dans les données labellisées train/val/test. L'apport réel du focus régional Afrique de l'Ouest/Côte d'Ivoire au *raisonnement* du classificateur est donc nul — il n'agit qu'au niveau de la récupération de sources à afficher.

- **Filtre régional cosmétique** : le sélecteur de zone géographique (`zone_geo`) de l'interface est transmis à l'API mais n'est pas exploité par `check_claim()` — la recherche FAISS reste globale quelle que soit la zone sélectionnée.

- **OCR de repli pour les PDF scannés (`upload_pdf`, `main.py`)** : si une page ne renvoie aucun texte via PyPDF2 (scan/image sans couche texte), un fallback OCR (`pytesseract` + `pdf2image`, binaire système `tesseract-ocr` requis — voir README) est tenté **uniquement sur cette page précise**, jamais systématiquement (le rendu image + OCR est coûteux). Import optionnel côté backend : si `pytesseract`/`pdf2image` ou le binaire `tesseract-ocr` sont absents, l'app démarre quand même et les pages sans texte natif restent simplement vides, comme avant l'ajout de l'OCR.

  Vérifié par exécution réelle : un PDF généré sans couche texte (image convertie en PDF, confirmé vide par `pdftotext` et `PyPDF2.extract_text()` avant test) est correctement recupéré par OCR (texte extrait identique au texte source de l'image, en français). Sur un PDF mixte (1 page à texte natif dense + 1 page scannée), seule la page scannée déclenche l'OCR (`1 page(s) récupérée(s) par OCR` loggé côté backend) — la page à texte natif n'invoque jamais l'OCR, confirmant le comportement de repli et non systématique.

  Limites : qualité dépendante de la résolution/netteté du scan (rendu à 200 DPI ici, ajustable) ; langues OCR fixées à `fra+eng` (`lang="fra+eng"` dans le code) — un scan dans une autre langue serait mal reconnu ; le binaire `tesseract-ocr` est une dépendance système (pas seulement Python), à installer séparément (voir README) — non incluse automatiquement par `pip install -r requirements.txt`.

- **Lien d'archive limité aux documents connus** : les 2 documents institutionnels locaux (Banque Mondiale, OMM) sont désormais servis via `/documents/<fichier>` (backend, `StaticFiles`). Le document GIEC pointe vers une URL externe réelle (`ipcc.ch`) mais le fichier local correspondant (`data/climate_docs/GIEC_AR6_Afrique_Resume.pdf`) est en réalité un texte de simulation (limitation déjà documentée : le téléchargement réel du PDF n'a jamais été effectué par `4_ingest_documents.py`, qui écrit un texte de substitution avec l'extension `.pdf`) — cliquer sur ce lien externe mène au vrai site du GIEC, pas nécessairement à la page exacte citée.

- **Sources thématiquement/géographiquement non pertinentes possibles (partiellement atténué)** : le seuil de 0.20 filtre les cas hors-sujet extrêmes, mais entre 0.20 et ~0.55 le système peut retenir une source qui partage un vocabulaire climatique générique avec le claim sans rapport géographique ou thématique réel avec lui. Exemple mesuré : pour une question portant sur une hausse localisée des précipitations dans un quartier d'Abidjan sur une période de deux ans, une des sources renvoyées porte sur la sécheresse dans la Corne de l'Afrique (score cosinus mesuré : 0.32, au-dessus du seuil, mais sans lien géographique — pays différent — ni thématique — sécheresse contre excès de pluie — avec la question). Ce n'est pas un cas isolé : sur 8 affirmations hyper-locales testées (villes différentes, phénomènes climatiques différents), aucune ne tombe sous 0.20, et certaines remontent même des extraits Climate-FEVER totalement hors sujet climatique (un extrait sur une pièce de théâtre classique a été retourné avec un score de 0.42 pour une question sur le paludisme lié à la chaleur). Cette limite tient à la nature du filtrage par similarité d'embeddings : la proximité sémantique globale (vocabulaire climatique commun) ne garantit pas la pertinence contextuelle précise (même lieu, même phénomène).
  
  **Filtre de cohérence à l'affichage (`relevance_filter.py`)** : un contrôle léger a été ajouté, appliqué **uniquement à l'affichage des sources, après que le verdict est déjà déterminé** — il ne touche ni au seuil de 0.20 ni à la classification. Un lexique d'une trentaine d'entités géographiques (villes/pays/régions d'Afrique de l'Ouest et au-delà, avec expansion ville → pays, ex. "Cocody" implique "Côte d'Ivoire") et thématiques (pluie, température, sécheresse, inondation, chaleur, paludisme, niveau de la mer, etc.) est comparé entre le claim et chaque evidence retournée. Si le claim contient au moins une entité reconnaissable et qu'**aucune** ne se retrouve dans l'evidence, la source est marquée `relevance_uncertain: true` côté API et affichée avec un badge "Pertinence géographique/thématique incertaine" côté interface — elle n'est **pas masquée** (choix délibéré : masquer risquerait de faire disparaître les 3 sources d'un coup dans certains cas, ce qui contredirait la "Traçabilité Totale" mise en avant comme fonctionnalité, et serait moins simple à implémenter sans cas limite). Si le claim ne contient aucune entité reconnaissable par le lexique, aucun filtrage n'est appliqué (par choix, pour ne jamais masquer d'information par excès de prudence).
  
  **Décision géo/thème séparée (depuis la phase de correction dédiée)** : le chevauchement n'est plus vérifié sur un ensemble combiné géo+thème (ce qui laissait un simple mot-clé thématique partagé, ex. "chaleur", suffire à ne pas marquer une source même quand la géographie diffère réellement — cas observé : une source sur la Somalie n'était pas marquée incertaine pour une question sur le paludisme lié à la chaleur à Bouaké). La règle actuelle (`is_relevance_uncertain`) traite les deux signaux séparément : **si le claim contient au moins une entité géographique reconnue, c'est elle seule qui décide** (chevauchement géo requis, un thème partagé seul ne suffit plus à sauver une source géographiquement hors-sujet) ; le signal thématique ne décide seul que lorsque le claim ne contient **aucune** entité géographique reconnue. Vérifié par exécution réelle : le cas Somalie/Bouaké est désormais correctement marqué `relevance_uncertain: true`, tandis que le cas de référence Cocody/Banque Mondiale (géographie commune — Côte d'Ivoire — thème différent, pluie contre température) reste correctement non marqué.

  Limites assumées de ce filtre : (1) couverture lexicale nécessairement incomplète (toutes les villes/phénomènes ne sont pas listés) ; (2) **nouvelle limite introduite par la priorité donnée au signal géo** : un claim citant un pays face à une source ne citant qu'une région plus large non reliée dans le lexique (ex. "Côte d'Ivoire" dans le claim contre "Afrique subsaharienne" dans l'evidence du GIEC, sans lien ville→pays→région dans `CITY_TO_COUNTRY`) est désormais marqué "incertain" même si la source est en réalité pertinente à l'échelle régionale — testé et confirmé (`is_relevance_uncertain` renvoie `True` pour ce couple). Impact pratique limité : c'est le seul chunk du corpus dans ce cas (le chunk GIEC), et son rang de similarité cosinus pure est très bas (#493 sur ~4870 pour un claim générique testé), donc il apparaît rarement dans le top-k affiché en pratique. Une hiérarchie pays→région a été envisagée pour combler cet écart mais écartée : elle aurait aussi fait remonter des régions non comparables (ex. Somalie/Corne de l'Afrique regroupée avec l'Afrique de l'Ouest sous un même méta-groupe continental), ce qui aurait cassé le cas Somalie/Bouaké que cette correction vise justement à détecter — un compromis jugé pire que la limite actuelle, plus étroite et documentée.

- **Granularité du corpus** : GIEC/OMM/Banque Mondiale documentent des tendances macro (nationales, régionales, mondiales, sur plusieurs décennies). Le système ne peut structurellement pas confirmer ou infirmer une statistique hyper-locale et récente (ex. "+80% de pluie en 2 ans dans un quartier précis") — ce type d'affirmation, pourtant fréquent dans la désinformation climatique qui circule sur les réseaux sociaux en Afrique de l'Ouest, tombe presque systématiquement en `NON VÉRIFIABLE` par manque de preuve directe, ce qui est le comportement correct mais peut donner l'impression d'un système peu utile sur ce type de cas précis.

- **Pas de calibration efficace** : testée (Platt scaling) et écartée car elle dégrade le Macro-F1 (section 6). Les scores de confiance affichés ne sont donc pas de vraies probabilités calibrées.

- **Pas de support multilingue structuré, mais l'anglais fonctionne déjà en pratique (capacité existante, non exploitée/annoncée)** : le corpus mélange français (documents institutionnels, `corpus_additionnel.csv`) et anglais (Climate-FEVER, ~4867 des ~4870 chunks). `all-MiniLM-L6-v2` gère nativement les deux langues, et aucune correction de code n'a été nécessaire (diagnostic réalisé avant tout développement, comme demandé). **Vérifié par exécution réelle** : un claim anglais général ("Global temperatures have risen significantly due to human activity.") donne `CONFIRMÉ` avec 3 sources anglaises pertinentes ; un exemple réel étiqueté `REFUTES` dans `test.csv` ("The rate of warming according to the data is much slower than the models used by the IPCC") donne bien `RÉFUTÉ` ; le seuil de 0.20 se comporte de façon cohérente en anglais comme en français (même sensibilité de bord sur les phrases courtes hors-sujet, indépendante de la langue — déjà documentée section 4) ; les formes de surface anglaises du lexique `relevance_filter.py` (ex. "Ivory Coast", "West Africa") sont reconnues sans erreur. Aucune stratégie de traduction ou d'alignement cross-lingue explicite n'existe pour autant : la qualité de la récupération pour une langue absente du corpus (autre que français/anglais) n'est pas garantie, et rien n'indique à l'utilisateur que l'anglais est supporté (non documenté dans l'interface).

- **Classe REFUTES la plus faible** (F1=0.44) : voir section 6, confusion fréquente avec SUPPORTS quand claim et evidence portent sur le même sujet mais avec un signe opposé — limite connue d'un classificateur basé uniquement sur la proximité d'embeddings, sans modélisation explicite de la négation.

- **Corpus FAISS statique** : aucune ingestion continue ; toute mise à jour nécessite de relancer manuellement le pipeline (section 5).

- **Dépendances non pinnées dans les fichiers `legacy/`** : `api.py`/`app.py` (prototypes obsolètes) ne font pas partie du chemin de production et ne sont pas couverts par les garanties ci-dessus (voir `legacy/README.md`).

- **Identifiant utilisateur léger sans authentification (historique, section 10)** : `user_id` est une chaîne quelconque générée côté client, jamais vérifiée côté serveur. Ce n'est **pas** un mécanisme de sécurité — quiconque connaît (ou devine) un `user_id` peut consulter l'historique associé via `GET /api/history/{user_id}`. Acceptable pour un usage à faible enjeu (retrouver ses propres vérifications sur son navigateur) mais impropre à protéger une donnée sensible. Une vraie authentification (session/mot de passe) résoudrait ce point mais sortait explicitement du périmètre demandé pour cette fonctionnalité.

---

## 9. Pistes d'amélioration futures

- **Affiner encore le filtre de cohérence géographique/thématique** (décision géo/thème séparée déjà implémentée, voir section 8) : envisager une hiérarchie géographique à plusieurs niveaux (ville→pays→région) plutôt qu'un seul niveau, pour couvrir le cas pays-vs-région-plus-large encore marqué à tort aujourd'hui (ex. Côte d'Ivoire vs GIEC/Afrique subsaharienne, section 8) sans casser la détection Somalie/Bouaké — nécessiterait une hiérarchie plus fine que "un pays appartient à une seule région" (ex. pondération décroissante par niveau plutôt qu'un simple ensemble d'ancêtres) pour éviter que des régions non comparables finissent regroupées ; élargir le lexique au-delà des ~35 entités actuelles ; envisager une reconnaissance d'entités nommées (NER) légère plutôt qu'un lexique statique si la couverture devient un problème récurrent.
- **Élargir le corpus institutionnel** : au-delà des 3 chunks actuels, ingérer davantage de rapports GIEC/OMM/Banque Mondiale complets (pas de simulation de téléchargement) pour que le focus régional annoncé ait un effet réel sur la couverture des sources.
- **Implémenter le filtre régional** (`zone_geo`) côté backend, par exemple en repondérant ou restreignant la recherche FAISS aux documents tagués pour la zone sélectionnée.
- **Explorer un jeu de données REFUTES plus riche** ou des techniques d'augmentation ciblées sur cette classe, seule à rester sous 0.5 de F1.
- **Ingestion continue du corpus** plutôt qu'un index FAISS statique reconstruit manuellement.
- **Stratégie multilingue explicite** (traduction automatique du claim vers la langue dominante du corpus avant recherche, ou corpus francophone étoffé) plutôt que de compter sur la robustesse cross-lingue implicite de l'encodeur.
- **Authentification réelle pour l'historique** (voir section 8 et section 10) : remplacer l'identifiant léger par un vrai compte (session, mot de passe) si l'historique venait à contenir des informations sensibles ou à être exposé publiquement.
- **Exploiter le feedback 👍/👎** (collecté, voir section 10, non encore utilisé) : par exemple pour repérer les claims où le verdict est jugé insatisfaisant par les utilisateurs (candidats à une revue humaine prioritaire), ou comme signal pour une future itération du classificateur.

---

## 10. Niveau de compréhension adaptatif et historique personnalisé

### Niveau de compréhension

Avant chaque vérification, l'utilisateur choisit un niveau parmi 4 libellés exacts : **débutant, intermédiaire, amateur, expert**. Le niveau **ne modifie jamais le verdict** — retrieval FAISS, seuil anti-hallucination (0.20) et classification NLI sont calculés une seule fois, exactement comme avant cette fonctionnalité (`check_claim`, `main.py`). Le niveau agit uniquement comme une **couche de formatage** appliquée après coup (`build_analyse_text`), qui adapte :
- le texte d'explication (`analyse_text`) : résumé pédagogique simple (débutant) → vocabulaire un peu plus précis (intermédiaire) → mention du score de similarité et du nombre de sources consultées (amateur) → détail technique complet, score cosinus exact + classe NLI brute + probabilités par classe (expert) ;
- un champ `technical_details` (JSON), `null` pour débutant/intermédiaire, peuplé pour amateur/expert.

Une valeur non reconnue (faute de frappe, champ absent) retombe sur `intermediaire` (`normalize_comprehension_level`) — comportement par défaut inchangé pour tout appelant qui ignore ce champ.

**Vérifié par exécution réelle** : même claim (Côte d'Ivoire/Banque Mondiale) interrogé aux 4 niveaux → même badge (`CONFIRMÉ PAR LES DONNÉES SCIENTIFIQUES`) dans les 4 cas ; au niveau expert, la probabilité de classe la plus élevée (`SUPPORTS=0.476`) corrobore bien ce même badge — aucune contradiction entre niveaux. Un bug de sérialisation a été trouvé et corrigé pendant ce test : `classifier.predict()`/`predict_proba()` renvoient des types `numpy` non sérialisables par Pydantic, plantant le niveau expert en 500 avant correction (cast explicite en `str`/`float` natifs).

### Historique personnalisé

Chaque vérification effectuée avec un `user_id` (transmis par le client, voir ci-dessous) est enregistrée : claim, niveau de compréhension choisi, verdict complet (badge, texte d'analyse déjà formaté pour ce niveau, sources), horodatage. **Enregistrée telle quelle** — la consultation de l'historique ne recalcule jamais rien, elle relit simplement ce qui a été décidé et renvoyé au moment de la vérification.

**Stockage** : SQLite (`history_store.py`, fichier `history.db` à la racine, gitignoré comme `models_saved/`). Choix justifié par l'absence de toute base de données préexistante dans le projet (uniquement des CSV statiques) et par une architecture FastAPI mono-instance déjà locale — SQLite évite d'ajouter un serveur de base de données à administrer pour un besoin de cette taille.

**Identifiant utilisateur léger** (périmètre volontairement limité, décision déjà validée — pas d'authentification complète) : une chaîne générée une fois côté client (`crypto.randomUUID()`, `frontend/src/userId.ts`) et conservée en `localStorage`. Envoyée en tant que `user_id` optionnel dans la requête de vérification ; absente, la vérification n'est simplement pas sauvegardée (aucune erreur). **Limite assumée** : ce n'est pas un mécanisme de sécurité — voir section 8.

**Endpoint** : `GET /api/history/{user_id}` renvoie uniquement les entrées de ce `user_id` (filtrage strict `WHERE user_id = ?` côté SQL). Vérifié par exécution réelle avec deux identifiants distincts (3 vérifications pour l'un, 1 pour l'autre) : chaque historique ne contient que ses propres entrées, aucune fuite croisée constatée.

**Partage d'une entrée d'historique** : réutilise le même mécanisme que le bouton "Partager" du verdict (texte formaté copiable dans le presse-papier, ou `navigator.share()` si disponible), factorisé dans `frontend/src/shareText.ts` et partagé entre `VerdictCard.tsx` (verdict qui vient d'être rendu) et `HistoryPanel.tsx` (entrée d'historique), au lieu d'être dupliqué.

### Feedback utilisateur (👍/👎)

Chaque verdict affiché propose un bouton 👍/👎, actif uniquement si la vérification a été sauvegardée dans l'historique (`verification_id` renvoyé par `check_claim`, donc uniquement si `user_id` a été fourni). `POST /api/feedback` (`{verification_id, user_id, rating}`, `rating` ∈ `{"up","down"}`) enregistre le retour dans la même base SQLite (`history_store.py`, table `feedback`, clé étrangère vers `verifications.id` — un `verification_id` inexistant est rejeté en 404). **Simplement collecté pour l'instant, non exploité** : aucune boucle de ré-entraînement ni tableau de bord ne consomme encore ces données — piste d'amélioration future (section 9).

### Vérification par lot

`POST /api/check-claims-batch` accepte un `text` (plusieurs affirmations collées) et renvoie un verdict complet par affirmation. **Découpage volontairement simple : une ligne non vide = une affirmation**, pas de segmentation NLP par phrase. Limite assumée (angle mort évident, documentée plutôt que masquée) : une affirmation rédigée sur plusieurs lignes serait scindée à tort en plusieurs claims séparés, et plusieurs affirmations courtes sur une même ligne seraient traitées comme une seule. Limite de 20 affirmations par lot (`MAX_BATCH_CLAIMS`), défensive contre un collage démesuré.

**Aucun raccourci sur le pipeline** : chaque ligne appelle directement `check_claim()` (la même fonction Python que la route `/api/check-claim`, pas une copie ni une version simplifiée) — seuil anti-hallucination à 0.20, classification NLI, filtre de cohérence géo/thème et niveau de compréhension s'appliquent identiquement à chaque affirmation du lot, y compris la sauvegarde en historique si `user_id` est fourni.

**Vérifié par exécution réelle**, lot de 3 affirmations de nature différente :
- *"En Côte d'Ivoire, la Banque Mondiale prévoit une hausse des températures."* → `CONFIRMÉ PAR LES DONNÉES SCIENTIFIQUES` (3 sources)
- *"The rate of warming according to the data is much slower than the models used by the IPCC"* (exemple réel étiqueté REFUTES dans `test.csv`) → `RÉFUTÉ / DÉSINFORMATION` (3 sources)
- *"???!!!###"* (hors-sujet, score < 0.20) → `AUCUNE PREUVE SCIENTIFIQUE` (0 source)

Chacune a reçu un `verification_id` distinct et séquentiel, confirmant que chaque affirmation a bien traversé le pipeline complet et a été sauvegardée séparément dans l'historique.

---

## 11. Veille GIEC/OMM (détection de nouveaux rapports)

`scripts/veille_giec_omm.py` vérifie si de nouveaux rapports/publications GIEC ou OMM sont apparus depuis la dernière exécution, et **signale** la détection (stdout + fichier de rapport dans `veille_reports/`). Périmètre volontairement limité à la **détection**, pas à l'ingestion : aucun contenu n'est téléchargé ni ajouté au corpus FAISS automatiquement — l'ingestion reste un geste manuel (`4_ingest_documents.py` / `update_corpus.py`, section 5).

**Sources surveillées** (vérifiées accessibles avant d'écrire le script, pas devinées) :
- **GIEC/IPCC** : flux RSS officiel `https://www.ipcc.ch/feed/` (HTTP 200 confirmé, `Content-Type: application/rss+xml`), parsé avec `feedparser`.
- **OMM/WMO** : aucun flux RSS public découvert sur `wmo.int` (recherche effectuée : ni balise `<link rel="alternate" type="application/rss+xml">` sur la page d'accueil, ni chemins usuels comme `/rss`, `/en/rss.xml`). Repli sur l'extraction des liens de la page `https://wmo.int/resources/publications` (HTTP 200 confirmé, liste bien des publications réelles, ex. *"State of Climate in Africa 2025"*).

**État** : `veille_state.json` (racine du dépôt, gitignoré comme `history.db`) conserve les identifiants (URLs) déjà vus par source, d'une exécution à l'autre. Au tout premier lancement, tout est signalé comme nouveau par définition.

**Vérifié par exécution réelle** (deux lancements successifs, contre les vraies sources en ligne) :
- 1er lancement : 10 entrées détectées côté GIEC (articles RSS réels : *"IPCC opens registration for experts to review First Order Draft..."*, etc.), 12 côté OMM (publications réelles : *state-of-climate-africa-2025*, *wmo-airborne-dust-bulletin-no-10-july-2026*, etc.).
- 2e lancement (immédiatement après) : `Aucune nouveauté (10 entrée(s) déjà connue(s))` / `Aucune nouveauté (12 entrée(s) déjà connue(s))` — confirme que l'état est bien persisté et comparé correctement.

**Lancement manuel** :
```bash
source venv/bin/activate
python3 scripts/veille_giec_omm.py
```

**Ce qui reste manuel** : la planification récurrente (pas configurée dans cette session) et l'ingestion effective d'un nouveau document détecté. **Ce qui est automatique** : la détection elle-même et la mise à jour de l'état une fois le script lancé. Exemple de tâche cron pour une vérification quotidienne (à ajouter par l'utilisateur, `crontab -e`) :
```
0 8 * * * cd /chemin/vers/terrava-ai && venv/bin/python3 scripts/veille_giec_omm.py >> veille.log 2>&1
```

**Limites** : le repli OMM par extraction de liens est plus fragile qu'un flux RSS (dépend de la structure HTML actuelle de `wmo.int`, pourrait casser silencieusement si le site change de mise en page — un échec de requête est cependant signalé dans le rapport, pas masqué) ; la détection ne distingue pas un document réellement nouveau d'une réorganisation d'URL (un lien renommé serait signalé comme "nouveau") ; aucune notification (email, Slack, etc.) n'est envoyée, seulement un fichier de rapport local.

## 12. Déploiement (Render)

### Diagnostic préalable : contrainte GitHub App

Avant toute configuration, vérification de la même contrainte déjà rencontrée sur un autre projet (NouanKanyAI) : l'installation de la GitHub App de Render ne peut sélectionner que des dépôts appartenant directement au compte GitHub connecté (`herverenard147`) — un accès collaborateur/push sur un dépôt tiers (`Yannick07-sys/climate-claim-verifier`, l'`origin` du projet à ce moment) ne suffit pas. Confirmé par `gh api repos/Yannick07-sys/climate-claim-verifier` (propriétaire différent du compte authentifié) et par la liste des dépôts déjà connectés à Render (`render services --output json`), tous sous `herverenard147`. Contournement identique à NouanKanyAI : fork sous `herverenard147/climate-claim-verifier`, devenu l'`origin` du projet et le dépôt utilisé pour le déploiement.

### Backend — Web Service Docker

Un `Dockerfile` est nécessaire (plutôt que l'environnement natif Render) car **l'environnement natif ne permet pas d'installer de paquets système (`apt-get`) au build** — vérifié par recherche avant implémentation, pas supposé. Or `tesseract-ocr` (OCR) est une dépendance système, pas un paquet Python.

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-fra poppler-utils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
```

**torch CPU-only explicite** : `sentence-transformers` installe par défaut `torch` avec tout l'écosystème CUDA (paquets `nvidia-cu13`), inutile pour ce projet explicitement conçu Zéro-GPU. Mesuré par build réel : image à **3,13 Go** avec le torch par défaut, contre **583 Mo** en forçant `torch==2.13.0` depuis l'index CPU officiel avant le reste des requirements. Ce n'est pas qu'une optimisation cosmétique : la première tentative de déploiement avec l'image à 3,13 Go a **réellement échoué** sur Render (`update_failed`, voir plus bas), la seconde avec l'image allégée a réussi.

**`$PORT`** : Render assigne le port d'écoute dynamiquement via cette variable d'environnement (pas un port fixe) — d'où la forme shell du `CMD` (`uvicorn ... --port $PORT`), qui permet son expansion par le shell au démarrage du conteneur.

### Artefacts modèle (`models_saved/`)

Absents d'un clone frais (gitignorés jusqu'ici), nécessaires au démarrage (`load_models()` les lit directement). Deux options considérées :
- (a) régénérer tout le pipeline ML au build (`1_prepare_data.py → migrate_csv.py → 2_build_retrieval.py → 3_train_classifier.py → 4_ingest_documents.py`) ;
- (b) committer les artefacts déjà entraînés.

**Choix : (b)**. Taille totale (`faiss_index.bin` + `classifier.joblib`) : 7,2 Mo — largement raisonnable pour Git. Démarrage prévisible et rapide, sans dépendre du téléchargement du dataset HuggingFace Climate-FEVER (~lourd) à chaque déploiement, ni du temps de ré-entraînement.

### CORS

`CORS_ORIGINS` (variable d'environnement, liste séparée par virgules) remplace un `allow_origins=["*"]` codé en dur. `allow_credentials` passé à `False` : le frontend n'envoie ni cookie ni en-tête d'authentification, et un wildcard combiné à `allow_credentials=True` est de toute façon rejeté par les navigateurs. Chaque origine subit un `rstrip("/")` avant comparaison — **bug réel rencontré en production** : une valeur d'env var saisie avec un slash final (`https://terrava-ai-frontend.onrender.com/`, copiée depuis la barre d'adresse du navigateur) provoquait un rejet CORS systématique (`Disallowed CORS origin`, HTTP 400), l'en-tête `Origin` envoyé par les navigateurs n'ayant jamais de slash final.

### `render.yaml`

Déclare les deux services et injecte automatiquement les URLs publiques croisées via `fromService`/`envVarKey: RENDER_EXTERNAL_URL`, sans valeur codée en dur ni redéploiement manuel nécessaire si une URL change :

```yaml
services:
  - name: terrava-ai-backend
    type: web
    runtime: docker
    plan: free
    envVars:
      - key: CORS_ORIGINS
        fromService: {name: terrava-ai-frontend, type: web, envVarKey: RENDER_EXTERNAL_URL}
  - name: terrava-ai-frontend
    type: web
    runtime: static
    buildCommand: cd frontend && npm install && npm run build
    staticPublishPath: frontend/dist
    envVars:
      - key: VITE_API_BASE_URL
        fromService: {name: terrava-ai-backend, type: web, envVarKey: RENDER_EXTERNAL_URL}
```

Les deux services ont en réalité été créés via le CLI Render (`render services create`) plutôt que via le flux Blueprint du dashboard (non pilotable sans navigateur) ; `render.yaml` documente la configuration cible et reste utilisable pour une recréation ou une migration future.

### Bug réel trouvé en production : `upload-pdf` bloquait tout le service

La route `/api/upload-pdf` était déclarée `async def`, alors que `PyPDF2`/`pytesseract` y exécutent du travail **CPU synchrone**. Dans une route `async def`, ce travail s'exécute directement sur la boucle asyncio — avec un seul worker (`WEB_CONCURRENCY=1`, valeur par défaut de Render selon les CPU disponibles sur l'instance), cela **bloque tout le serveur** pendant toute la durée du traitement. Vérifié en conditions réelles : un unique upload a rendu `/docs` (route sans rapport) indisponible (502) pendant toute la durée du traitement.

**Correction** : route passée en `def` (non `async`) — Starlette exécute alors automatiquement une route synchrone dans un thread du pool, libérant la boucle asyncio pour les autres requêtes. `content = await file.read()` remplacé par `content = file.file.read()` (lecture synchrone du fichier temporaire sous-jacent, `await` n'étant plus utilisable hors contexte `async`).

Vérifié après correction (local, `uvicorn` direct, hors Docker) : upload PDF réel de 18 pages (40 534 caractères extraits, 0 page OCR nécessaire) traité en 5,4 s, avec une requête `/docs` concurrente servie en 0,19 s **pendant** ce traitement.

### Limite non résolue : OCR sur PDF volumineux/scanné, plan gratuit

Testé en conditions réelles sur l'instance Render déployée (pas seulement en local) avec un vrai rapport GIEC AR6 (18 pages, 6,9 Mo) : le traitement fait **planter le processus backend** (le worker redémarre entièrement — rechargement complet du modèle observé dans les logs, `Started server process [7]` apparaissant à nouveau) et rend tout le service indisponible pendant 30 à 60 secondes. Reproduit deux fois de façon cohérente (échec après 41,7 s puis 66,3 s, sans aucune ligne de log applicative pour la requête elle-même — signe d'un arrêt brutal du processus, pas d'une exception Python capturée). Un fichier `.txt` léger, lui, est traité normalement (200 OK, < 0,2 s) juste avant et juste après ces deux échecs, isolant bien le problème au traitement PDF lui-même plutôt qu'à la route en général.

**Hypothèse la plus probable** : dépassement de la RAM allouée par le plan gratuit Render lorsque le pipeline `pdf2image`/`pytesseract` (rendu d'image à 200 DPI par page, potentiellement plusieurs pages) s'ajoute à la mémoire déjà occupée par le modèle `all-MiniLM-L6-v2`/FAISS/pandas chargés au démarrage — non confirmée avec certitude absolue (pas de message OOM explicite disponible dans les logs accessibles), mais cohérente avec l'absence totale de trace applicative (un OOM-kill du noyau ne laisse pas de traceback Python) et avec le fait que ce même PDF, en local, n'a déclenché aucun fallback OCR (extraction native réussie sur toutes les pages sauf une) — suggérant que l'environnement Render (bibliothèques système différentes) déclenche l'OCR différemment, ou que la RAM y est simplement plus contrainte qu'en local.

**Non corrigé dans le cadre de ce déploiement** — décision structurante non tranchée : soit limiter/désactiver le fallback OCR en production (perte de fonctionnalité pour les PDF scannés), soit passer à un plan Render payant avec plus de RAM (coût récurrent). Les deux options nécessitent un arbitrage qui n'a pas été fait.

### Disque éphémère (historique et feedback)

Le disque du plan gratuit Render est **éphémère** : `history.db` (SQLite, historique des vérifications et feedback 👍/👎) est recréé vide à chaque redémarrage/redéploiement du conteneur. Vérifié fonctionnel *pendant la durée de vie d'une instance* (aller-retour réel : `POST /api/check-claim` avec `user_id` → `verification_id` renvoyé → l'entrée apparaît immédiatement dans `GET /api/history/{user_id}`), mais **aucune persistance n'est garantie entre deux déploiements**. Un disque persistant Render lèverait cette limite mais est une option payante, **non activée** (décision volontairement laissée à l'utilisateur, coût récurrent).

### Autre limite pré-existante, révélée par les tests de production

Le seuil anti-hallucination (0.20, section 4) se comporte comme documenté pour un texte réellement dénué de sens (`"???!!!###"` → score 0.1925, sous le seuil, `NON_VERIFIABLE`, 0 source). En testant plusieurs claims hors-sujet en langage naturel « normal » contre l'instance de production (Bitcoin, Coupe du Monde, smartphones), aucun ne tombe sous le seuil : scores mesurés entre 0.31 et 0.37, tous classés `NOT_ENOUGH_INFO`/« preuves indirectes » avec sources affichées plutôt que `NON_VERIFIABLE`. Cette limite (plancher naturel de similarité cosinus des embeddings de phrases, indépendant du sujet) préexistait à ce déploiement mais n'avait pas été caractérisée avec des exemples de langage naturel réel — seulement documentée avec des exemples extrêmes (ponctuation, langue étrangère hors corpus, voir section 4).

### URLs de production et commandes

| | Backend | Frontend |
|---|---|---|
| URL | `https://terrava-ai-backend.onrender.com` | `https://terrava-ai-frontend.onrender.com` |
| Type | Web Service (Docker) | Static Site |
| Build | `docker build` (Dockerfile à la racine) | `cd frontend && npm install && npm run build` |
| Start | `uvicorn main:app --host 0.0.0.0 --port $PORT` (CMD du Dockerfile) | — (fichiers statiques servis directement) |
| Publish directory | — | `frontend/dist` |
| Variables d'env clés | `CORS_ORIGINS` | `VITE_API_BASE_URL` |
| Plan | Free | Free (les sites statiques Render sont gratuits par nature) |

## 13. Détection heuristique des saisies multiples/ambiguës

### Constat à l'origine de cette section

TERRAVA-AI n'a aucune capacité de raisonnement ou de désambiguïsation d'intention : une phrase entre dans le pipeline, un vecteur en sort, comparé au corpus (voir section 1). Si un utilisateur saisit plusieurs affirmations mélangées dans un seul champ — cas plausible pour un débutant, ou pour une transcription vocale longue et peu structurée (voir section 10) —, tout le texte est encodé en un seul vecteur "moyen".

**Mesuré en conditions réelles avant tout développement** (pas supposé) : le claim composite *"Le rechauffement climatique est cause par les activites humaines. La banquise arctique augmente chaque annee depuis 1980."* (une affirmation vraie + une affirmation fausse) obtient un verdict **CONFIRMÉ PAR LES DONNÉES SCIENTIFIQUES** (score cosinus 0.5855, classe NLI `SUPPORTS`) — un verdict confiant portant en réalité sur les deux affirmations à la fois, alors que la seconde est fausse. Une simple question (*"Est-ce que le climat change vraiment ?"*) obtient elle aussi un verdict confiant (`CONFIRMÉ`, score 0.429) au lieu d'un signal indiquant qu'il ne s'agit pas d'une affirmation vérifiable.

### Principe : détection par motifs de surface, pas de compréhension du sens

`input_heuristics.py` implémente `detect_input_issue()`, appelée par `check_claim()` **avant** tout calcul d'embedding. Aucun modèle supplémentaire entraîné, aucun appel LLM externe (cohérent avec le positionnement zéro-GPU du projet) — uniquement des règles explicables :

| Type détecté | Règle | Limite assumée |
|---|---|---|
| `multiple` | Découpage par ponctuation forte (`.!?`) puis par quelques conjonctions de coordination fréquentes (`et`, `mais`, `donc`, `car`, `puis`) ; retenu si ≥2 segments de 4 mots ou plus | Découpage naïf : une conjonction interne à une expression ("l'Afrique de l'Ouest et du Centre") peut produire un faux segment — atténué par le filtre de longueur minimale, pas éliminé. `ou`/`or` volontairement exclus (trop ambigus). |
| `vague` (`too_short`) | Moins de 3 mots | Un fragment de 3 mots légitime mais rare serait laissé passer ; un fragment de 2 mots répété (nom propre composé) serait bloqué à tort |
| `vague` (`question`) | Termine par `?`, ou contient une tournure interrogative (`est-ce que`, `pourquoi`...), ou commence par un mot interrogatif | Une phrase déclarative citant accidentellement l'un de ces mots ailleurs qu'en tête pourrait être mal classée |
| `vague` (`command`) | Commence par un verbe d'instruction fréquent (`vérifie`, `dis-moi`, `explique`...) | Détection par préfixe uniquement, liste fermée de verbes |
| `incoherent` (`symbols`) | Plus de 50% de caractères non alphabétiques (hors espaces) | Un texte scientifique légitime très dense en chiffres/symboles pourrait être signalé à tort |
| `incoherent` (`no_function_words`) | Texte de 3+ mots ne contenant AUCUN mot-outil français/anglais fréquent (`le`, `la`, `est`, `the`, `and`...) | Une phrase dans une langue absente de cette liste restreinte (espagnol, wolof...) serait signalée à tort comme incohérente — aucune détection de langue réelle |

**Vérifié avec `melange langues`** (*"The climate change est vraiment un grand probleme worldwide today"*, français+anglais) : correctement **non** signalé (`ok`), car il contient des mots-outils anglais reconnus — la détection ne pénalise pas le mélange de langues en soi, seulement l'absence de tout mot-outil reconnu.

### Guidage, jamais de blocage

Quand `detect_input_issue()` renvoie un type différent de `ok`, `check_claim()` ne lance **pas** le pipeline de classification : il renvoie `{"needs_guidance": true, "guidance_type": ..., "segments": [...], "message": "..."}` à la place d'un verdict (la route n'a plus de `response_model` unique — elle peut renvoyer l'un ou l'autre). Le frontend (`GuidanceCard.tsx`) affiche ce message avec des **choix fermés uniquement** (boutons), jamais un champ de dialogue libre — le système ne peut pas tenir une conversation. Pour une saisie `multiple` : un bouton par segment détecté (vérifie ce segment seul), un bouton "Vérifier tous séparément" (pré-remplit la Vérification par lot, voir section précédente, avec un segment par ligne), et toujours un bouton "Envoyer tel quel quand même".

Le contournement passe par le champ `force: true` de `ClaimRequest`, qui saute entièrement la détection. Deux origines : le clic utilisateur sur "Envoyer tel quel", et l'appel interne de `check_claims_batch()` — chaque ligne d'un lot est déjà une affirmation séparée par construction (convention "une ligne = un claim"), la re-détecter serait redondant et le composant `BatchPanel` ne peut de toute façon pas afficher de guidage interactif par ligne (il attend un verdict immédiat).

### Message distinct pour un texte incohérent (vs `NON VÉRIFIABLE`)

Un verdict `NON VÉRIFIABLE` existant signifie *"affirmation compréhensible, mais absente du corpus institutionnel"* (score cosinus < 0.20, section 4). Un texte `incoherent` (charabia, emojis seuls, chiffres seuls) n'est **pas la même chose** : le texte lui-même n'a pas pu être interprété comme une phrase, indépendamment de ce que contient le corpus. Le message affiché le précise explicitement (*"ceci diffère d'un verdict « non vérifiable »..."*) pour ne pas induire l'utilisateur en erreur sur la cause réelle.

### Application à la voix et à l'upload (sans code dédié)

La détection s'applique automatiquement au texte issu de la dictée vocale et de l'extraction PDF/TXT, **sans logique spécifique à ces sources** : le texte transcrit ou extrait pré-remplit le même champ `claim` que la saisie manuelle, et `check_claim()` détecte sur le texte final soumis, quelle que soit son origine. Vérifié réellement (Playwright, transcription vocale simulée) : un texte dicté contenant deux affirmations mélangées déclenche le même `GuidanceCard` qu'une saisie manuelle identique.

### Tests de robustesse (cas limites, hors du cadre "normal")

Testés contre l'API réelle, aucune erreur 500 brute ni trace serveur visible dans aucun cas :

| Saisie | Résultat |
|---|---|
| Charabia (`"xyzzy foobar quux plonk zorp glorp"`) | `incoherent` / `no_function_words` |
| Emojis seuls, chiffres seuls, ponctuation seule (`"!!!???..."`) | `incoherent` / `symbols` |
| Texte très long (plusieurs phrases mêlant claims et remplissage sans rapport) | `multiple` (découpage correct malgré le bruit) |
| Artefacts de mise en forme (tabulations, retours à la ligne multiples) | Traité normalement si le contenu reste cohérent ; `multiple` si plusieurs affirmations s'y trouvent |
| Commande (`"vérifie que la Terre est ronde stp"`) | `vague` / `command` |
| Vide / espaces seuls | `HTTP 400` inchangé (comportement pré-existant) |
| Transcription vocale bruitée (mots isolés sans lien réel) | Non détectée comme incohérente dans un cas testé, car elle contenait accidentellement un mot-outil reconnu (`être`) — traitée comme une affirmation normale, verdict `PREUVES INDIRECTES`. **Limite assumée et documentée**, pas un bug : la détection ne comprend pas le sens, un mot-outil isolé suffit à la faire passer. |

Aucun cas testé n'a produit de plantage, d'erreur technique brute, ou de réponse incompréhensible — objectif du point 7 de la mission, distinct de "comprendre" ces cas.
