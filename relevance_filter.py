"""
Filtre de cohérence géographique/thématique appliqué UNIQUEMENT à l'affichage
des sources, jamais au seuil anti-hallucination (0.20) ni à la classification
(voir main.py: ce module n'est appelé qu'après que le verdict est décidé).

Principe : un petit lexique de ~35 entités géographiques (pays/régions/villes
pertinentes pour le climat en Afrique de l'Ouest et au-delà) et thématiques
(phénomènes climatiques) est comparé entre le claim et chaque evidence
retournée par FAISS. Si le claim contient au moins une entité reconnaissable
et qu'aucune ne se retrouve dans l'evidence, la source est marquée comme
"pertinence incertaine" (elle reste affichée - voir DOCUMENTATION_TECHNIQUE.md
pour la justification de ce choix plutôt que de la masquer).

Décision géo/thème (séparés, pas un simple ensemble combiné) : quand le claim
contient au moins une entité GÉOGRAPHIQUE reconnue, c'est ELLE qui décide
(un chevauchement thématique seul, ex. "chaleur" partagé entre un claim sur
la Côte d'Ivoire et une source sur la Somalie, ne suffit plus à couvrir un
vrai écart géographique). Le thème ne décide seul que si le claim ne contient
AUCUNE entité géographique reconnue. Voir DOCUMENTATION_TECHNIQUE.md section 8
pour la justification et les limites de ce choix (notamment le cas d'un pays
face à une source décrivant seulement une région plus large non reliée dans
le lexique).

Limites assumées (documentées) : couverture lexicale forcément incomplète
(toutes les villes/phénomènes ne sont pas listés) ; un claim qui ne contient
aucune entité reconnaissable par ce lexique n'est jamais filtré, par choix
délibéré pour ne pas masquer de l'information par excès de prudence.
"""
import re
import unicodedata

# Chaque entrée : (identifiant canonique, [formes de surface à détecter]).
# Formes de surface données sans accents/apostrophes : la normalisation du
# texte (voir normalize()) retire accents et apostrophes avant comparaison,
# donc "Côte d'Ivoire" et "Ivory Coast" sont bien deux formes du même groupe.
GEO_ENTITIES = [
    ("cote_ivoire", ["cote d ivoire", "ivory coast", "ivoirien", "ivoirienne"]),
    ("abidjan", ["abidjan"]),
    ("cocody", ["cocody"]),
    ("bouake", ["bouake"]),
    ("korhogo", ["korhogo"]),
    ("yamoussoukro", ["yamoussoukro"]),
    ("grand_bassam", ["grand bassam"]),
    ("san_pedro", ["san pedro"]),
    ("afrique_ouest", ["afrique de l ouest", "west africa"]),
    ("afrique_subsaharienne", ["afrique subsaharienne", "sub saharan africa", "sub-saharan africa"]),
    ("senegal", ["senegal"]),
    ("dakar", ["dakar"]),
    ("somalie", ["somalie", "somalia"]),
    ("corne_afrique", ["corne de l afrique", "horn of africa"]),
    ("kenya", ["kenya"]),
    ("ghana", ["ghana"]),
    ("kumasi", ["kumasi"]),
    ("nigeria", ["nigeria"]),
    ("mali", ["mali"]),
    ("burkina_faso", ["burkina faso"]),
    ("guinee", ["guinee", "guinea"]),
    ("liberia", ["liberia"]),
    ("congo", ["congo", "rdc", "drc"]),
    ("goma", ["goma"]),
]

THEME_ENTITIES = [
    ("pluie", ["pluie", "pluviometrie", "rain", "precipitation", "precipitations"]),
    ("temperature", ["temperature", "temperatures"]),
    ("secheresse", ["secheresse", "drought"]),
    ("inondation", ["inondation", "inondations", "flood", "flooding"]),
    ("chaleur", ["chaleur", "vague de chaleur", "heat", "heatwave"]),
    ("niveau_mer", ["niveau de la mer", "sea level", "sea-level"]),
    ("rechauffement", ["rechauffement climatique", "changement climatique", "global warming", "climate change"]),
    ("paludisme", ["paludisme", "malaria"]),
    ("recolte", ["recolte", "harvest"]),
    ("cacao", ["cacao", "cocoa"]),
]

ALL_ENTITIES = GEO_ENTITIES + THEME_ENTITIES

# Une ville implique son pays (ex. une source nationale sur la Côte d'Ivoire
# est pertinente pour une question sur Abidjan, même si "Abidjan" n'est pas
# mot pour mot dans l'evidence) - sans quoi une source Banque Mondiale sur la
# Côte d'Ivoire serait marquée "incertaine" pour une question sur Cocody, ce
# qui serait un faux positif du filtre lui-même. Une seule strate
# ville -> pays est utilisée volontairement (pas de remontée pays -> région)
# pour rester simple et éviter de trop diluer la spécificité géographique.
CITY_TO_COUNTRY = {
    "abidjan": "cote_ivoire",
    "cocody": "cote_ivoire",
    "bouake": "cote_ivoire",
    "korhogo": "cote_ivoire",
    "yamoussoukro": "cote_ivoire",
    "grand_bassam": "cote_ivoire",
    "san_pedro": "cote_ivoire",
    "dakar": "senegal",
    "kumasi": "ghana",
    "goma": "congo",
}


def _normalize(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[’'`]", " ", text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _match_entities(normalized_text: str, lexicon: list) -> set:
    found = set()
    for canonical_id, surface_forms in lexicon:
        for form in surface_forms:
            if re.search(r"\b" + re.escape(form) + r"\b", normalized_text):
                found.add(canonical_id)
                break
    return found


def extract_geo_entities(text: str) -> set:
    """Entités géographiques uniquement (avec expansion ville -> pays, voir CITY_TO_COUNTRY)."""
    if not text:
        return set()
    found = _match_entities(_normalize(text), GEO_ENTITIES)
    for city_id in list(found):
        if city_id in CITY_TO_COUNTRY:
            found.add(CITY_TO_COUNTRY[city_id])
    return found


def extract_theme_entities(text: str) -> set:
    """Entités thématiques (phénomènes climatiques) uniquement."""
    if not text:
        return set()
    return _match_entities(_normalize(text), THEME_ENTITIES)


def extract_entities(text: str) -> set:
    """Union géo + thème (utilisée par main.py pour la repondération zone_geo,
    qui n'a besoin que de savoir si une entité quelconque est partagée)."""
    return extract_geo_entities(text) | extract_theme_entities(text)


# Libellés français lisibles des identifiants canoniques ci-dessus, utilisés
# pour générer un raisonnement claim/evidence en langage naturel (voir
# build_analyse_text dans main.py) SANS jamais recopier le texte brut de
# l'evidence : on ne réutilise que les entités déjà détectées, jamais la
# citation elle-même.
ENTITY_LABELS = {
    "cote_ivoire": "la Côte d'Ivoire",
    "abidjan": "Abidjan",
    "cocody": "Cocody",
    "bouake": "Bouaké",
    "korhogo": "Korhogo",
    "yamoussoukro": "Yamoussoukro",
    "grand_bassam": "Grand-Bassam",
    "san_pedro": "San-Pédro",
    "afrique_ouest": "l'Afrique de l'Ouest",
    "afrique_subsaharienne": "l'Afrique subsaharienne",
    "senegal": "le Sénégal",
    "dakar": "Dakar",
    "somalie": "la Somalie",
    "corne_afrique": "la Corne de l'Afrique",
    "kenya": "le Kenya",
    "ghana": "le Ghana",
    "kumasi": "Kumasi",
    "nigeria": "le Nigeria",
    "mali": "le Mali",
    "burkina_faso": "le Burkina Faso",
    "guinee": "la Guinée",
    "liberia": "le Liberia",
    "congo": "le Congo",
    "goma": "Goma",
    "pluie": "les précipitations",
    "temperature": "les températures",
    "secheresse": "la sécheresse",
    "inondation": "les inondations",
    "chaleur": "les vagues de chaleur",
    "niveau_mer": "le niveau de la mer",
    "rechauffement": "le réchauffement climatique",
    "paludisme": "le paludisme",
    "recolte": "les récoltes",
    "cacao": "la production de cacao",
}


def describe_entities(ids: set) -> str:
    """Rend un ensemble d'identifiants canoniques en une liste lisible en
    français ("X, Y et Z"). Chaîne vide si l'ensemble est vide."""
    labels = sorted(ENTITY_LABELS.get(i, i) for i in ids)
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " et " + labels[-1]


def is_relevance_uncertain(claim: str, evidence: str) -> bool:
    """
    Décision en deux signaux séparés (géo, thème), pas un ensemble combiné :

    - Si le claim contient au moins une entité GÉOGRAPHIQUE reconnue, c'est ce
      signal qui décide seul : la source est "incertaine" si aucune entité
      géographique du claim ne se retrouve dans l'evidence, même si un
      thème est partagé (ex. "chaleur" commun à une source sur la Somalie et
      un claim sur la Côte d'Ivoire ne suffit plus à la sauver).
    - Sinon (claim sans entité géo reconnue), c'est le signal THÉMATIQUE qui
      décide : "incertaine" si aucun thème du claim ne se retrouve dans
      l'evidence.
    - Si le claim ne contient ni entité géo ni entité thème reconnue, jamais
      filtré (choix délibéré, inchangé : on ne masque pas d'information par
      excès de prudence sur un claim que le lexique ne sait pas interpréter).
    """
    claim_geo = extract_geo_entities(claim)
    if claim_geo:
        evidence_geo = extract_geo_entities(evidence)
        return claim_geo.isdisjoint(evidence_geo)

    claim_theme = extract_theme_entities(claim)
    if not claim_theme:
        return False
    evidence_theme = extract_theme_entities(evidence)
    return claim_theme.isdisjoint(evidence_theme)
