from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import joblib
import PyPDF2
import io
import os
import unicodedata
from typing import List, Optional
from relevance_filter import (
    is_relevance_uncertain,
    extract_entities,
    extract_geo_entities,
    extract_theme_entities,
    describe_entities,
)
from input_heuristics import detect_input_issue, build_guidance_message
import history_store

# OCR de repli pour les pages PDF sans couche texte (scan/image). Import
# optionnel : si pytesseract/pdf2image ou le binaire système tesseract-ocr
# ne sont pas installés, l'app démarre quand même — l'OCR est simplement
# indisponible et les pages sans texte natif restent vides comme avant
# (voir upload_pdf). Ne jamais bloquer tout le backend pour cette dépendance
# optionnelle système (documentée dans README.md).
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Poids additif appliqué au score cosinus d'une source candidate quand sa zone
# géographique (déduite du même lexique que relevance_filter.py) recoupe la
# zone demandée par l'utilisateur. Calibré empiriquement : les chunks
# institutionnels pertinents mais peu représentés dans le corpus (3 sur ~4870)
# se classent souvent autour de 0.10-0.15 sous les meilleurs chunks Climate-FEVER
# génériques (score cosinus) ; ce boost suffit à les faire remonter dans le
# top-k affiché sans écraser un score sémantique nettement supérieur.
ZONE_GEO_BOOST = 0.15

app = FastAPI(
    title="TERRAVA-AI API",
    description="API Back-End pour la vérification climatique",
    version="2.0"
)

# Configuration CORS. En local (variable CORS_ORIGINS absente), autorise les
# ports Vite habituels. En production (Render), CORS_ORIGINS doit contenir
# l'URL exacte du frontend déployé (ex. https://terrava-ai-frontend.onrender.com),
# en variable d'environnement plutôt qu'en dur, pour ne pas avoir à modifier
# le code (ni redéployer l'image Docker) si cette URL change. Plusieurs
# origines peuvent être séparées par une virgule. allow_credentials=False :
# le frontend n'envoie ni cookie ni en-tête d'authentification (fetch simples
# avec corps JSON), donc pas besoin de credentials côté CORS.
_default_origins = "http://localhost:5173,http://localhost:3000"
# rstrip("/") : l'en-tête Origin envoyé par les navigateurs n'a jamais de
# slash final (format scheme://host[:port]) — une valeur d'env var saisie
# avec un slash final (erreur de saisie courante en copiant une URL) ferait
# sinon échouer silencieusement toute comparaison d'origine.
allowed_origins = [o.strip().rstrip("/") for o in os.environ.get("CORS_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sert les documents institutionnels ingérés localement (data/climate_docs/) pour
# que le lien "Consulter l'archive" des sources sans URL externe pointe vers un
# contenu réel au lieu d'un placeholder "local" mal interprété par le frontend.
if os.path.isdir("data/climate_docs"):
    app.mount("/documents", StaticFiles(directory="data/climate_docs"), name="documents")

COMPREHENSION_LEVELS = {"debutant", "intermediaire", "amateur", "expert"}


def normalize_comprehension_level(level: str) -> str:
    """Normalise un niveau de compréhension libre vers l'un des 4 niveaux
    canoniques ; retombe sur "intermediaire" si non reconnu (comportement
    par défaut inchangé, jamais d'erreur pour une valeur inattendue)."""
    if not level:
        return "intermediaire"
    normalized = unicodedata.normalize("NFKD", level.strip().lower()).encode("ascii", "ignore").decode("ascii")
    return normalized if normalized in COMPREHENSION_LEVELS else "intermediaire"


# Verbe/tournure reliant le claim à l'evidence top-1, par niveau, pour les
# trois buckets où une source top-1 existe réellement (AUCUNE_PREUVE est
# volontairement exclu : par construction, aucune source n'est retenue sous
# le seuil anti-hallucination, donc aucun raisonnement claim/evidence n'a de
# sens à ce niveau - voir check_claim).
_VERDICT_CONNECTORS = {
    "CONFIRME": {
        "debutant": "va dans le même sens que",
        "intermediaire": "corrobore",
        "amateur": "corrobore",
    },
    "REFUTE": {
        "debutant": "dit le contraire de",
        "intermediaire": "contredit",
        "amateur": "contredit",
    },
    "INSUFFISANT": {
        "debutant": "parle d'un sujet proche, sans confirmer ni contredire clairement",
        "intermediaire": "aborde un sujet connexe sans confirmer ni infirmer explicitement",
        "amateur": "aborde un sujet connexe sans confirmer ni infirmer explicitement",
    },
}


def _build_reasoning(claim: str, evidence: str, institution: str, verdict_bucket: str,
                      similarity_score: float, relevance_uncertain: bool) -> dict:
    """
    Construit, pour chacun des 4 niveaux, une phrase reliant explicitement le
    claim à l'evidence top-1 (pas seulement le verdict et le nom de la
    source) — à partir des entités géo/thème détectées dans le claim et
    l'evidence (voir relevance_filter.py), JAMAIS en recopiant le texte de
    l'evidence : celui-ci est déjà affiché tel quel juste en dessous (citation
    dans SourcesAccordion), un raisonnement qui le recopierait n'apporterait
    rien de nouveau à l'utilisateur.

    Si relevance_uncertain est vrai (le claim vise une entité géographique
    précise absente de cette evidence), le rapprochement est explicitement
    présenté comme large/régional plutôt que direct — ne jamais laisser
    croire à une correspondance exacte qui n'existe pas.
    """
    if verdict_bucket not in _VERDICT_CONNECTORS:
        return {}

    claim_geo = extract_geo_entities(claim)
    evidence_geo = extract_geo_entities(evidence)
    evidence_theme = extract_theme_entities(evidence)
    claim_theme = extract_theme_entities(claim)

    theme_desc = describe_entities(evidence_theme) or describe_entities(claim_theme) or "le sujet climatique évoqué"

    if relevance_uncertain and claim_geo:
        evidence_scope = describe_entities(evidence_geo) if evidence_geo else "une zone plus large, non précisée par cette source"
        claim_scope = describe_entities(claim_geo)
        scope_clause = f"à l'échelle de {evidence_scope} plutôt que spécifiquement de {claim_scope}"
        scope_clause_short = f"portée régionale ({evidence_scope}), pas nationale/locale ({claim_scope})"
        directness = "indirect"
    elif evidence_geo:
        scope_clause = f"portant sur {describe_entities(evidence_geo)}"
        scope_clause_short = f"portée géographique alignée ({describe_entities(evidence_geo)})"
        directness = "direct"
    else:
        scope_clause = "sans préciser de zone géographique particulière"
        scope_clause_short = "portée géographique non précisée par la source"
        directness = "direct"

    connector = _VERDICT_CONNECTORS[verdict_bucket]

    debutant = (
        f"Le rapport de {institution} porte sur {theme_desc}, {scope_clause} : "
        f"cela {connector['debutant']} ce que vous avez affirmé."
    )
    intermediaire = (
        f"La source la plus proche ({institution}) porte sur {theme_desc}, {scope_clause}. "
        f"Ce contenu {connector['intermediaire']} votre affirmation."
    )
    amateur = (
        f"Avec un score de similarité de {similarity_score:.2f} entre votre affirmation et la source la plus "
        f"proche ({institution}, {theme_desc}), le rapprochement est {directness} : {scope_clause_short}."
    )
    expert = f"le rapprochement s'appuie sur {theme_desc} ({scope_clause_short})"

    return {"debutant": debutant, "intermediaire": intermediaire, "amateur": amateur, "expert": expert}


def build_analyse_text(verdict_bucket: str, level: str, similarity_score: float,
                        raw_verdict: Optional[str] = None,
                        probabilities: Optional[dict] = None,
                        nb_sources: int = 0,
                        claim: str = "", evidence: str = "", institution: str = "",
                        relevance_uncertain: bool = False):
    """
    Formate le texte d'analyse (et, pour amateur/expert, des détails
    techniques) selon le niveau de compréhension, à partir d'un verdict DÉJÀ
    décidé — cette fonction ne relance JAMAIS la classification ni le calcul
    du score : verdict_bucket/similarity_score/raw_verdict/probabilities lui
    sont passés tels quels par check_claim(), qui ne les calcule qu'une
    seule fois. Un seul verdict, quatre présentations — chacune incluant
    désormais un raisonnement reliant explicitement le claim à la source
    top-1 (voir _build_reasoning), pas seulement le verdict brut.
    """
    templates = {
        "CONFIRME": {
            "debutant": "✅ Cette affirmation est confirmée ! {reasoning} Cette information vient de sources fiables (comme le GIEC ou l'OMM), donc vous pouvez lui faire confiance.",
            "intermediaire": "L'information soumise est exacte et validée par le consensus scientifique actuel. {reasoning} Ces observations soulignent la nécessité d'intégrer ces risques dans les plans d'adaptation locaux et les politiques de résilience.",
            "amateur": "L'information soumise est exacte et validée par le consensus scientifique actuel. {reasoning} Le système a consulté {nb_sources} source(s) institutionnelle(s) au total (le seuil minimal pour une correspondance valide est de 0.20).",
            "expert": "Verdict : SUPPORTS (classification NLI). Score cosinus top-1 : {score:.4f} (seuil anti-hallucination : 0.20, franchi). Correspondance claim/evidence : {reasoning}. Probabilités par classe : {proba_str}. Nombre de sources retenues pour l'affichage : {nb_sources}.",
        },
        "REFUTE": {
            "debutant": "❌ Cette affirmation est fausse. {reasoning} Attention à ne pas partager cette information sans la corriger.",
            "intermediaire": "L'information soumise est inexacte ou trompeuse. {reasoning} Il est crucial de corriger cette communication afin de ne pas fausser l'évaluation des vulnérabilités climatiques.",
            "amateur": "L'information soumise est inexacte ou trompeuse. {reasoning} Le système a consulté {nb_sources} source(s) institutionnelle(s) au total.",
            "expert": "Verdict : REFUTES (classification NLI). Score cosinus top-1 : {score:.4f} (seuil anti-hallucination : 0.20, franchi). Correspondance claim/evidence : {reasoning}. Probabilités par classe : {proba_str}. Nombre de sources retenues pour l'affichage : {nb_sources}.",
        },
        "INSUFFISANT": {
            "debutant": "⚠️ On ne peut pas dire si c'est vrai ou faux avec certitude. {reasoning} Regardez les sources ci-dessous pour vous faire votre propre idée.",
            "intermediaire": "Les documents institutionnels (GIEC, OMM, etc.) traitent de sujets connexes, mais ils ne permettent pas de confirmer ou de réfuter explicitement et directement cette affirmation précise. {reasoning} Une analyse humaine des documents sourcés ci-dessous est recommandée.",
            "amateur": "Les documents institutionnels traitent de sujets connexes sans trancher explicitement. {reasoning} Le système a consulté {nb_sources} source(s) au total (score au-dessus du seuil de 0.20, mais le classificateur n'a pas identifié de confirmation ou de réfutation nette).",
            "expert": "Verdict : NOT_ENOUGH_INFO (classification NLI). Score cosinus top-1 : {score:.4f} (seuil anti-hallucination : 0.20, franchi). Correspondance claim/evidence : {reasoning}. Probabilités par classe : {proba_str}. Nombre de sources retenues pour l'affichage : {nb_sources}.",
        },
        "AUCUNE_PREUVE": {
            "debutant": "⚠️ Aucune source scientifique ne parle de ce sujet précis. On ne peut donc pas vérifier cette affirmation avec les documents disponibles — méfiance.",
            "intermediaire": "Aucune source institutionnelle ne mentionne ou ne justifie cette affirmation. En l'absence de données fiables et directes issues de la littérature scientifique officielle (GIEC, OMM, rapports nationaux), cette déclaration est considérée comme totalement infondée.",
            "amateur": "Aucune source institutionnelle suffisamment proche n'a été trouvée : le score de similarité sémantique le plus élevé obtenu ({score:.2f}) reste sous le seuil minimal de 0.20, en-deçà duquel le système refuse de statuer plutôt que de risquer une réponse non fondée.",
            "expert": "Verdict : NON_VERIFIABLE (seuil anti-hallucination). Score cosinus top-1 : {score:.4f}, sous le seuil de 0.20 — le classificateur n'est pas invoqué dans ce cas (aucune probabilité de classe disponible). Nombre de sources retenues pour l'affichage : 0 (par construction, sous le seuil).",
        },
    }

    proba_str = "non disponible (classificateur non invoqué)"
    if probabilities:
        proba_str = ", ".join(f"{cls}={p:.3f}" for cls, p in probabilities.items())

    reasoning_by_level = _build_reasoning(claim, evidence, institution, verdict_bucket, similarity_score, relevance_uncertain)

    text = templates[verdict_bucket][level].format(
        score=similarity_score, nb_sources=nb_sources, proba_str=proba_str,
        reasoning=reasoning_by_level.get(level, ""),
    )

    technical_details = None
    if level in ("amateur", "expert"):
        technical_details = {
            "similarity_score": round(similarity_score, 4),
            "nb_sources_consulted": nb_sources,
        }
        if level == "expert":
            technical_details["raw_nli_class"] = raw_verdict
            technical_details["class_probabilities"] = (
                {cls: round(p, 4) for cls, p in probabilities.items()} if probabilities else None
            )

    return text, technical_details


# Modèles de données
class ClaimRequest(BaseModel):
    claim: str
    zone_geo: str = "Global (International)"
    comprehension_level: str = "intermediaire"
    # Identifiant léger généré et conservé côté client (pas de compte, pas de
    # mot de passe - voir history_store.py et DOCUMENTATION_TECHNIQUE.md pour
    # les limites assumées). Optionnel : si absent, la vérification n'est pas
    # sauvegardée dans l'historique (aucune régression pour un client qui
    # n'envoie pas ce champ).
    user_id: Optional[str] = None
    # True quand l'utilisateur a explicitement choisi d'ignorer le guidage
    # heuristique ("envoyer tel quel") ou quand l'appel vient du traitement
    # par lot (voir check_claims_batch : chaque ligne y est déjà séparée par
    # construction, la re-détecter serait redondant et casserait le contrat
    # "un résultat immédiat par ligne" du batch, qui ne peut pas afficher de
    # guidage interactif). Ne bloque jamais rien d'autre : ignorer la
    # détection heuristique en amont, pas une clé d'administration.
    force: bool = False

class Source(BaseModel):
    institution: str
    evidence: str
    title: str
    year: str
    url: str
    # Filtre d'affichage uniquement (post-classification, n'affecte ni le
    # seuil de 0.20 ni le verdict) : True si le claim contient une entité
    # géographique/thématique reconnue qui ne se retrouve pas dans cette
    # evidence précise. Voir relevance_filter.py.
    relevance_uncertain: bool = False

class VerificationResponse(BaseModel):
    badge_class: str
    badge_icon: str
    badge_text: str
    analyse_text: str
    sources: List[Source]
    # Détails techniques (score cosinus, classe NLI brute, probabilités par
    # classe) : uniquement peuplé pour les niveaux "amateur"/"expert" (voir
    # build_analyse_text). None pour "debutant"/"intermediaire".
    technical_details: Optional[dict] = None
    # Id de l'entrée d'historique créée (uniquement si user_id a été fourni,
    # voir plus bas) - permet au frontend d'associer un feedback 👍/👎 à
    # cette vérification précise via POST /api/feedback. None si non
    # sauvegardée (pas de user_id fourni).
    verification_id: Optional[int] = None

# Variables globales pour l'IA
embedding_model = None
index = None
classifier = None
corpus_df = None

@app.on_event("startup")
def load_models():
    global embedding_model, index, classifier, corpus_df
    print("Démarrage du moteur TERRAVA-AI...")
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        index = faiss.read_index("models_saved/faiss_index.bin")
        classifier = joblib.load("models_saved/classifier.joblib")
        corpus_df = pd.read_csv("data/corpus.csv")
        history_store.init_db()
        print("Moteur d'IA chargé avec succès.")
    except Exception as e:
        print(f"Erreur fatale lors du chargement des modèles : {e}")

@app.post("/api/check-claim")
def check_claim(request: ClaimRequest):
    if not request.claim.strip():
        raise HTTPException(status_code=400, detail="La déclaration est vide.")

    # Détection heuristique (voir input_heuristics.py) AVANT tout calcul
    # d'embedding : évite un verdict silencieusement trompeur sur une saisie
    # multiple/vague/incohérente - vérifié en conditions réelles avant ce
    # correctif qu'un mélange affirmation vraie + affirmation fausse pouvait
    # produire un badge CONFIRMÉ (score 0.59) portant sur les deux à la fois.
    # `force` contourne ce guidage (bouton "envoyer tel quel" côté frontend,
    # ou appel interne depuis check_claims_batch) - jamais un blocage total.
    # response_model retiré du décorateur : cette route peut renvoyer soit un
    # verdict (VerificationResponse), soit une réponse de guidage (forme
    # différente), FastAPI sérialise les deux sans schéma de réponse unique.
    if not request.force:
        issue = detect_input_issue(request.claim)
        if issue["type"] != "ok":
            return {
                "needs_guidance": True,
                "guidance_type": issue["type"],
                "reason": issue["reason"],
                "segments": issue["segments"],
                "message": build_guidance_message(issue),
            }

    try:
        c_emb = embedding_model.encode([request.claim], normalize_embeddings=True)
        k = 3
        distances, indices = index.search(c_emb, k)
        
        top_evidence_row = corpus_df.iloc[indices[0][0]]
        top_evidence = top_evidence_row['evidence']
        similarity_score = float(distances[0][0])
        
        # Filtre anti-hallucination (Seuil de tolérance à 0.20 comme décidé)
        raw_verdict = None
        probabilities = None
        if similarity_score < 0.20:
            verdict = "NON_VERIFIABLE"
        else:
            e_emb = embedding_model.encode([top_evidence], normalize_embeddings=True)
            features = np.hstack((c_emb, e_emb, np.abs(c_emb - e_emb), c_emb * e_emb))
            raw_verdict = str(classifier.predict(features)[0])
            # str()/float() : predict()/predict_proba() renvoient des types
            # numpy (numpy.str_, numpy.float32), non sérialisables tels quels
            # par Pydantic dans la réponse JSON (technical_details, niveau
            # "expert") — convertis en types Python natifs dès leur calcul.
            probabilities = {
                str(cls): float(p)
                for cls, p in zip(classifier.classes_, classifier.predict_proba(features)[0])
            }

            if raw_verdict == "SUPPORTS":
                verdict = "CONFIRME"
            elif raw_verdict == "REFUTES":
                verdict = "REFUTE"
            else:
                verdict = "NON_VERIFIABLE"

        # Détermination du badge et du "bucket" de verdict (utilisé ensuite
        # par build_analyse_text pour choisir le bon texte selon le niveau de
        # compréhension). Le verdict lui-même (CONFIRME/REFUTE/NON_VERIFIABLE)
        # est déjà figé ci-dessus et ne dépend en rien du niveau demandé.
        if verdict == "CONFIRME":
            badge_class = "badge-confirmed"
            badge_icon = "✅"
            badge_text = "CONFIRMÉ PAR LES DONNÉES SCIENTIFIQUES"
            verdict_bucket = "CONFIRME"
        elif verdict == "REFUTE":
            badge_class = "badge-refuted"
            badge_icon = "❌"
            badge_text = "RÉFUTÉ / DÉSINFORMATION"
            verdict_bucket = "REFUTE"
        else:
            badge_class = "badge-insufficient"
            badge_icon = "⚠️"
            if similarity_score >= 0.20:
                badge_text = "PREUVES INDIRECTES / INSUFFISANTES"
                verdict_bucket = "INSUFFISANT"
            else:
                badge_text = "AUCUNE PREUVE SCIENTIFIQUE"
                verdict_bucket = "AUCUNE_PREUVE"

        # Préparation des sources
        sources = []
        if similarity_score >= 0.20:
            # Sélection des sources à AFFICHER (top-k). Par défaut, identique à
            # avant : les k meilleurs candidats du retrieval sémantique déjà
            # effectué ci-dessus (indices/distances), sans aucun changement.
            #
            # Si zone_geo contient au moins une entité géographique reconnue
            # (même lexique que relevance_filter.py), on repondère un pool de
            # candidats plus large en ajoutant ZONE_GEO_BOOST au score cosinus
            # des candidats dont l'evidence mentionne cette zone, puis on
            # retrie et reprend les k meilleurs. C'est un second passage
            # strictement APRÈS le choix du top-1 utilisé pour le seuil
            # anti-hallucination et la classification (inchangés ci-dessus) :
            # zone_geo ne peut donc jamais faire basculer le verdict, il
            # affecte uniquement quelles sources complémentaires sont mises
            # en avant. Si zone_geo n'est pas reconnu (valeur par défaut,
            # faute de frappe, zone hors lexique), aucune repondération n'a
            # lieu et le comportement reste strictement identique à avant.
            source_indices = [int(i) for i in indices[0][:k]]
            zone_entities = extract_entities(request.zone_geo)
            if zone_entities:
                pool_size = len(corpus_df)  # corpus institutionnel réduit (~4870 lignes) : un re-scan complet reste négligeable en coût
                pool_distances, pool_indices = index.search(c_emb, pool_size)
                candidates = []
                for idx, score in zip(pool_indices[0], pool_distances[0]):
                    idx = int(idx)
                    evidence_text = str(corpus_df.iloc[idx]['evidence'])
                    boosted_score = float(score)
                    if zone_entities & extract_entities(evidence_text):
                        boosted_score += ZONE_GEO_BOOST
                    candidates.append((boosted_score, idx))
                candidates.sort(key=lambda pair: pair[0], reverse=True)
                source_indices = [idx for _, idx in candidates[:k]]

            for idx in source_indices:
                row = corpus_df.iloc[idx]
                evidence_text = str(row['evidence'])
                sources.append(Source(
                    institution=str(row.get('institution', 'Source Inconnue')),
                    evidence=evidence_text,
                    title=str(row.get('title', 'Document officiel')),
                    year=str(row.get('year', 'N/A')),
                    url=str(row.get('url', '#')),
                    # Filtre de cohérence géo/thématique appliqué uniquement à
                    # l'affichage, une fois le verdict déjà déterminé.
                    relevance_uncertain=is_relevance_uncertain(request.claim, evidence_text)
                ))

        # Couche de formatage selon le niveau de compréhension : le verdict
        # (verdict_bucket), le score et les probabilités sont déjà figés
        # ci-dessus, calculés une seule fois quel que soit le niveau demandé.
        # Le raisonnement claim/evidence porte sur la source top-1 D'ORIGINE
        # (top_evidence_row, avant repondération zone_geo éventuelle des
        # sources AFFICHÉES ci-dessus) : c'est elle qui a réellement servi au
        # seuil anti-hallucination et à la classification, donc la seule
        # dont le rapprochement avec le claim a un sens à expliquer.
        top_institution = str(top_evidence_row.get('institution', 'une source institutionnelle'))
        top_relevance_uncertain = is_relevance_uncertain(request.claim, top_evidence) if similarity_score >= 0.20 else False
        level = normalize_comprehension_level(request.comprehension_level)
        analyse_text, technical_details = build_analyse_text(
            verdict_bucket, level, similarity_score,
            raw_verdict=raw_verdict, probabilities=probabilities, nb_sources=len(sources),
            claim=request.claim, evidence=str(top_evidence), institution=top_institution,
            relevance_uncertain=top_relevance_uncertain,
        )

        # Sauvegarde dans l'historique personnel : TELLE QUELLE, la réponse
        # déjà décidée ci-dessus (verdict, texte de niveau, sources) - jamais
        # recalculée à la consultation de l'historique (voir GET /api/history).
        # Uniquement si un user_id a été fourni ; sinon la vérification n'est
        # simplement pas persistée (pas d'erreur, comportement anonyme normal).
        verification_id = None
        if request.user_id:
            try:
                verification_id = history_store.save_verification(
                    user_id=request.user_id,
                    claim=request.claim,
                    comprehension_level=level,
                    badge_class=badge_class,
                    badge_icon=badge_icon,
                    badge_text=badge_text,
                    analyse_text=analyse_text,
                    sources=[s.model_dump() for s in sources],
                )
            except Exception as history_err:
                # Un échec de sauvegarde de l'historique ne doit jamais faire
                # échouer la vérification elle-même (fonctionnalité annexe).
                print(f"[check-claim] Échec de sauvegarde de l'historique: {history_err}")

        return VerificationResponse(
            badge_class=badge_class,
            badge_icon=badge_icon,
            badge_text=badge_text,
            analyse_text=analyse_text,
            sources=sources,
            technical_details=technical_details,
            verification_id=verification_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class HistoryEntry(BaseModel):
    id: int
    claim: str
    comprehension_level: str
    badge_class: str
    badge_icon: str
    badge_text: str
    analyse_text: str
    sources: List[Source]
    created_at: str


@app.get("/api/history/{user_id}", response_model=List[HistoryEntry])
def get_history(user_id: str):
    """Retourne UNIQUEMENT les vérifications appartenant à ce user_id (filtrage
    strict côté SQL, voir history_store.get_history). Aucune authentification :
    quiconque connaît le user_id peut consulter l'historique associé — limite
    assumée du modèle d'identifiant léger, documentée dans
    DOCUMENTATION_TECHNIQUE.md."""
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="Identifiant utilisateur manquant.")
    return history_store.get_history(user_id)


class FeedbackRequest(BaseModel):
    verification_id: int
    user_id: str
    rating: str  # "up" ou "down"


@app.post("/api/feedback")
def submit_feedback(request: FeedbackRequest):
    """Enregistre un feedback 👍/👎 lié à une vérification existante
    (verification_id renvoyé par check-claim, voir VerificationResponse).
    Simplement collecté pour exploitation future - aucune logique de
    classification n'en dépend."""
    if request.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating doit être 'up' ou 'down'.")
    if not history_store.verification_exists(request.verification_id):
        raise HTTPException(status_code=404, detail="Vérification introuvable pour cet identifiant.")
    feedback_id = history_store.save_feedback(request.verification_id, request.user_id, request.rating)
    return {"feedback_id": feedback_id}


MAX_BATCH_CLAIMS = 20  # limite défensive : évite qu'un collage géant ne rende une seule requête trop lente


class BatchClaimRequest(BaseModel):
    text: str  # plusieurs affirmations, une par ligne (voir check_claims_batch)
    zone_geo: str = "Global (International)"
    comprehension_level: str = "intermediaire"
    user_id: Optional[str] = None


class BatchResultItem(BaseModel):
    claim: str
    result: VerificationResponse


@app.post("/api/check-claims-batch", response_model=List[BatchResultItem])
def check_claims_batch(request: BatchClaimRequest):
    """
    Vérifie plusieurs affirmations en une requête. Découpage volontairement
    simple : une ligne non vide = une affirmation (pas de segmentation NLP
    par phrase). Limite connue et assumée : une affirmation rédigée sur
    plusieurs lignes serait scindée à tort en plusieurs claims séparés, et
    plusieurs affirmations courtes sur une même ligne seraient traitées comme
    une seule - à l'utilisateur de coller un texte avec une affirmation par
    ligne pour un découpage correct.

    Chaque ligne suit EXACTEMENT le même pipeline complet qu'une vérification
    simple : cette fonction appelle check_claim() ligne par ligne (même
    fonction, pas de logique dupliquée ni de raccourci) - seuil
    anti-hallucination, classification, filtre de pertinence et niveau de
    compréhension s'appliquent identiquement à chaque claim du lot.
    """
    lines = [line.strip() for line in request.text.split("\n") if line.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="Aucune affirmation détectée (texte vide).")
    if len(lines) > MAX_BATCH_CLAIMS:
        raise HTTPException(
            status_code=400,
            detail=f"Trop d'affirmations dans le lot ({len(lines)}) : maximum {MAX_BATCH_CLAIMS} par requête."
        )

    results = []
    for line in lines:
        claim_request = ClaimRequest(
            claim=line,
            zone_geo=request.zone_geo,
            comprehension_level=request.comprehension_level,
            user_id=request.user_id,
            # Chaque ligne du lot est déjà une affirmation séparée par
            # construction (convention "une ligne = un claim" du batch) :
            # la re-détecter comme "saisie multiple" serait redondant, et le
            # batch ne peut de toute façon pas afficher de guidage
            # interactif par ligne (il attend un verdict immédiat).
            force=True,
        )
        verdict = check_claim(claim_request)
        results.append(BatchResultItem(claim=line, result=verdict))
    return results


# Limites anti-crash sur l'upload - valeurs déterminées par des mesures
# réelles contre l'instance Render déployée (plan gratuit, RAM limitée), pas
# choisies par prudence arbitraire. Méthodologie et données complètes dans
# DOCUMENTATION_TECHNIQUE.md section 14 ; résumé ci-dessous.
#
# MAX_UPLOAD_SIZE_BYTES (8 Mo) : bissection réelle entre un point qui
# fonctionne de façon répétée (6,6-6,9 Mo, ~20-25s, plusieurs fois) et un
# point qui fait systématiquement planter le service (9,85 Mo, 502 en 16s,
# process redémarré, AUCUNE ligne de log applicative - même signature OOM
# que le crash à 33 Mo déjà documenté). 8 Mo laisse une marge sous 9,85 Mo
# tout en restant au-dessus du cas à 7 Mo qui fonctionnait déjà (pas de
# régression). Fait notable : le fichier à 9,85 Mo plante SANS que l'OCR
# soit impliqué (la page volumineuse en cause a du texte natif extractible,
# donc pas de fallback OCR) - la taille du fichier reste un facteur de
# crash indépendant de MAX_OCR_PAGES, pas seulement corrélé à lui.
#
# MAX_OCR_PAGES (0, OCR désactivé) : à l'origine fixé à 5 par estimation,
# retesté avec de vrais PDF 100% scannés (aucun texte natif, donc OCR
# obligatoire sur chaque page) directement contre Render. Résultat sans
# ambiguïté : MÊME 1 SEULE page OCR, sur un fichier de 23 Ko, fait planter
# le service en ~5s (502, process redémarré, même signature OOM). Testé
# aussi à 5 et 12 pages OCR (mêmes résultats, ~5s à chaque fois - le
# nombre de pages OCR au-delà de 1 ne change rien, c'est le premier appel
# à pdf2image/Tesseract qui est déjà fatal sur ce plan). Le rendu d'image
# 200dpi + le sous-processus Tesseract dépassent visiblement la RAM
# disponible dès qu'ils s'ajoutent à la mémoire déjà occupée par les
# modèles ML chargés en permanence (SentenceTransformer/FAISS/pandas).
# Conclusion : l'OCR n'est pas seulement "à plafonner", il n'est pas
# viable du tout sur ce plan Render - décision structurante (fonctionnalité
# perdue pour les PDF scannés en production) signalée explicitement,
# pas prise silencieusement. Réversible instantanément si le plan Render
# change (aucune autre modification de code nécessaire).
MAX_UPLOAD_SIZE_BYTES = 8 * 1024 * 1024  # 8 Mo
MAX_OCR_PAGES = 0


@app.post("/api/upload-pdf")
def upload_pdf(request: Request, file: UploadFile = File(...)):
    # def (pas async def) : PyPDF2/OCR ci-dessous sont du travail CPU
    # synchrone. Dans une route async def, ce travail s'exécute directement
    # sur la boucle asyncio et bloque TOUT le serveur (un seul worker,
    # WEB_CONCURRENCY=1 sur Render) pendant toute sa durée — vérifié en
    # conditions réelles sur Render : un unique upload de PDF volumineux a
    # rendu le service entièrement indisponible (502 sur toutes les routes,
    # y compris /docs) le temps du traitement. Une route def synchrone est
    # automatiquement exécutée par Starlette dans un thread du pool, ce qui
    # libère la boucle asyncio pour les autres requêtes pendant ce temps.
    # La casse de l'extension n'est pas fiable comme signal de format : de
    # nombreux PDF (export Windows/macOS, scanners) sont nommés ".PDF" ou
    # ".Pdf". Un filtre sensible à la casse rejetait ces fichiers, pourtant
    # valides, avec un message trompeur ("format non accepté").
    filename_lower = file.filename.lower()
    is_pdf = filename_lower.endswith('.pdf')
    is_txt = filename_lower.endswith('.txt')
    if not is_pdf and not is_txt:
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF et TXT sont acceptés.")

    # Rejet précoce via l'en-tête Content-Length quand le navigateur le
    # fournit (quasi toujours pour un upload de fichier) : évite de lire
    # inutilement un fichier énorme en mémoire avant de le rejeter. Filet de
    # sécurité en profondeur, pas la seule protection : un client qui
    # mentirait sur cet en-tête est quand même bloqué par le contrôle sur
    # len(content) juste après la lecture.
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_SIZE_BYTES + 10_000:  # marge pour l'overhead multipart
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop volumineux (max {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} Mo)."
        )

    try:
        content = file.file.read()
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Fichier trop volumineux ({len(content) / (1024 * 1024):.1f} Mo, max {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} Mo)."
            )
        text = ""

        if is_pdf:
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            except Exception:
                # Erreur de format côté client (PDF corrompu/invalide), pas une
                # panne serveur : 400 plutôt que 500, avec un message actionnable.
                raise HTTPException(
                    status_code=400,
                    detail="Le fichier PDF est illisible ou corrompu. Vérifiez qu'il s'agit bien d'un PDF valide et réessayez."
                )

            # Parcourt TOUTES les pages (auparavant limité aux 5 premières, ce
            # qui tronquait silencieusement tout document plus long). Une page
            # individuellement illisible par PyPDF2 (scan/image sans couche
            # texte, page corrompue) déclenche un fallback OCR SUR CETTE SEULE
            # PAGE avant d'être comptée comme vide — jamais systématique : le
            # rendu image + OCR est coûteux en temps, donc réservé aux pages
            # où l'extraction native a échoué.
            page_texts = []
            failed_pages = 0
            ocr_pages = 0
            ocr_skipped_by_cap = 0  # pages sans texte natif, non tentées en OCR car MAX_OCR_PAGES déjà atteint
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                try:
                    page_text = page.extract_text()
                except Exception:
                    page_text = None

                # Plafonné à MAX_OCR_PAGES : le rendu image 200dpi + Tesseract par
                # page est le poste le plus coûteux en RAM de tout ce pipeline
                # (reproduit en conditions réelles sur Render) - au-delà de la
                # limite, une page sans texte natif est comptée comme "échouée"
                # plutôt que de continuer à tenter l'OCR indéfiniment.
                if not page_text and OCR_AVAILABLE and ocr_pages >= MAX_OCR_PAGES:
                    ocr_skipped_by_cap += 1
                if not page_text and OCR_AVAILABLE and ocr_pages < MAX_OCR_PAGES:
                    try:
                        images = convert_from_bytes(content, first_page=page_num, last_page=page_num, dpi=200)
                        if images:
                            ocr_text = pytesseract.image_to_string(images[0], lang="fra+eng")
                            if ocr_text and ocr_text.strip():
                                page_text = ocr_text
                                ocr_pages += 1
                    except Exception as ocr_err:
                        print(f"[upload-pdf] OCR échoué sur la page {page_num} de {file.filename}: {ocr_err}")

                if page_text:
                    page_texts.append(page_text)
                else:
                    failed_pages += 1
            text = " ".join(page_texts)
            pages_total = len(pdf_reader.pages)
            ocr_capped = ocr_skipped_by_cap > 0
            print(f"[upload-pdf] {file.filename}: {pages_total} page(s), "
                  f"{failed_pages} page(s) sans texte extrait, {ocr_pages} page(s) récupérée(s) par OCR, "
                  f"{ocr_skipped_by_cap} page(s) non tentée(s) en OCR (limite {MAX_OCR_PAGES} atteinte), "
                  f"{len(text)} caractères extraits au total")

            if not text:
                raise HTTPException(
                    status_code=400,
                    detail="Aucun texte n'a pu être extrait de ce PDF (pages scannées/images sans OCR ?)."
                )
        else:
            text = content.decode('utf-8')
            # Pas de notion de page/OCR pour un .txt : champs à None pour que
            # le frontend puisse distinguer "pas concerné" de "0 problème".
            pages_total = failed_pages = ocr_pages = None
            ocr_capped = False

        # Le champ extracted_text préremplit la barre de claim (pas un
        # visualiseur de document) : on renvoie un aperçu court plutôt que le
        # texte intégral, mais en échantillonnant début ET fin du document
        # extrait (pas seulement les premiers caractères) pour rester
        # représentatif d'un document long.
        # `truncated` permet au frontend de prévenir l'utilisateur quand
        # l'aperçu inséré n'est PAS le texte intégral : seul ce texte
        # tronqué est réellement envoyé à /api/check-claim ensuite (le texte
        # complet extrait ici n'est jamais conservé côté serveur).
        HEAD, TAIL = 350, 150
        truncated = len(text) > HEAD + TAIL
        if not truncated:
            extracted = text
        else:
            extracted = text[:HEAD].rstrip() + " [...] " + text[-TAIL:].lstrip()

        return {
            "extracted_text": extracted.strip(),
            "truncated": truncated,
            "pages_total": pages_total,
            "pages_failed": failed_pages,
            "ocr_pages_used": ocr_pages,
            "ocr_capped": ocr_capped,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de lecture du document: {str(e)}")
