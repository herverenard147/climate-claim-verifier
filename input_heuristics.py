"""
Détection heuristique légère des saisies à risque, appliquée par check_claim()
AVANT le calcul d'embedding et la classification (voir main.py).

Contexte (voir DOCUMENTATION_TECHNIQUE.md) : TERRAVA-AI n'a aucune capacité de
raisonnement ou de désambiguïsation - une phrase entre dans le pipeline, un
vecteur en sort, comparé au corpus. Si un utilisateur saisit plusieurs
affirmations mélangées dans un seul champ, elles sont encodées en un seul
vecteur "moyen" : le verdict rendu peut sembler confiant (score élevé, badge
CONFIRMÉ) alors qu'une seule des affirmations mélangées est vraie - vérifié en
conditions réelles avant d'écrire ce module (voir résumé de session), pas
supposé.

CE MODULE NE COMPREND PAS LE SENS DU TEXTE. Il repère des motifs de surface
(ponctuation, mots-outils fréquents, longueur, ratio de caractères non
alphabétiques) - aucune analyse syntaxique ou sémantique réelle. Chaque
heuristique ci-dessous documente sa propre limite connue. Objectif : détecter
les cas les plus francs et guider l'utilisateur par des choix fermés (jamais
un dialogue libre, le système ne peut pas tenir une conversation), sans jamais
bloquer complètement - un utilisateur peut toujours forcer l'envoi tel quel.
"""
import re
import unicodedata

# Mots-outils très fréquents en français et en anglais : un texte de plusieurs
# mots qui n'en contient AUCUN est un signal de suspicion (pas une preuve)
# qu'il ne s'agit probablement pas d'une phrase naturelle reconnaissable -
# une vraie phrase française ou anglaise de 3+ mots contient presque
# toujours un article, une préposition ou un auxiliaire. Limite assumée :
# une langue absente de cette liste (ex. une phrase 100% en espagnol ou en
# wolof) serait à tort signalée comme incohérente - aucune détection de
# langue réelle ici, juste ce lexique restreint.
COMMON_FUNCTION_WORDS = {
    "le", "la", "les", "l", "un", "une", "des", "de", "du", "et", "est", "sont",
    "que", "qui", "quoi", "dans", "pour", "avec", "sur", "par", "ne", "pas",
    "au", "aux", "ce", "cette", "ces", "son", "sa", "ses", "il", "elle", "ils",
    "elles", "nous", "vous", "je", "tu", "en", "a", "ont", "etre", "avoir",
    "plus", "moins", "tres", "selon", "depuis", "the", "is", "are", "and",
    "of", "in", "for", "with", "on", "by", "not", "this", "that", "to", "it",
    "its", "was", "were", "has", "have", "as", "at", "from", "or",
}

# Tournures interrogatives fréquentes (français). Reconnaissance par motif de
# surface uniquement : une phrase déclarative contenant accidentellement l'un
# de ces mots ailleurs qu'en tête ("on se demande pourquoi" dans un rapport)
# pourrait être classée à tort comme une question - limite assumée.
QUESTION_MARKERS = ["est-ce que", "est ce que", "qu'est-ce", "qu est ce"]
QUESTION_WORD_STARTS = {
    "pourquoi", "comment", "combien", "quel", "quelle", "quels", "quelles",
    "qui", "quand", "ou est", "où est",
}

# Verbes d'instruction fréquents en tête de phrase ("vérifie que...", "dis-moi
# si...") : signale une formulation de commande plutôt qu'une affirmation
# déclarative à vérifier telle quelle. Détection par préfixe uniquement.
IMPERATIVE_STARTERS = [
    "verifie", "vérifie", "verifiez", "vérifiez", "dis moi", "dis-moi",
    "dites moi", "dites-moi", "explique", "expliquez", "trouve", "trouvez",
    "cherche", "cherchez", "donne moi", "donne-moi", "montre", "montrez",
    "confirme", "confirmez", "dis si", "dis-si",
]

# Conjonctions de coordination utilisées pour scinder un texte en segments
# candidats, en plus de la ponctuation forte (. ! ?). "ou"/"or" est
# volontairement exclu : trop ambigu (connecteur logique rare mais aussi
# fragment de noms propres/anglicismes), risque de faux positifs jugé trop
# élevé pour une heuristique "simple et explicable".
COORDINATING_CONJUNCTIONS = r"\s+(?:et|mais|donc|car|puis)\s+"

MIN_WORDS_FOR_CLAIM = 3  # sous ce seuil, un texte ne peut pas former une affirmation déclarative testable
MIN_WORDS_PER_SEGMENT = 4  # heuristique "contient un verbe" : pas de détection réelle, juste une longueur minimale après découpage
MAX_SEGMENTS_RETURNED = 5


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKD", text.strip().lower())


def _symbol_ratio(text: str) -> float:
    """Part de caractères non alphabétiques (hors espaces) dans le texte.
    Un texte dominé par des chiffres/symboles/emojis n'est pas une phrase."""
    stripped = text.replace(" ", "")
    if not stripped:
        return 0.0
    non_alpha = sum(1 for c in stripped if not c.isalpha())
    return non_alpha / len(stripped)


def _has_recognizable_function_word(text: str) -> bool:
    words = set(re.findall(r"[a-zà-ÿ]+", _normalize(text)))
    return bool(words & COMMON_FUNCTION_WORDS)


def _is_too_short(text: str) -> bool:
    return len(text.split()) < MIN_WORDS_FOR_CLAIM


def _looks_like_question(text: str) -> bool:
    t = _normalize(text)
    if t.endswith("?"):
        return True
    if any(marker in t for marker in QUESTION_MARKERS):
        return True
    first_word = t.split()[0].rstrip(",") if t.split() else ""
    return first_word in QUESTION_WORD_STARTS


def _looks_like_command(text: str) -> bool:
    t = _normalize(text)
    return any(t.startswith(starter) for starter in IMPERATIVE_STARTERS)


def _split_segments(text: str) -> list:
    """Découpe par ponctuation forte puis par quelques conjonctions de
    coordination fréquentes. Découpage NAÏF par motifs de surface, pas une
    analyse syntaxique : une conjonction interne à une expression ("l'Afrique
    de l'Ouest et du Centre") produirait un faux découpage en 2 segments -
    limite assumée, atténuée par le filtre de longueur minimale ensuite
    (un faux segment issu d'un tel découpage est rarement assez long pour
    être retenu comme "déclaratif")."""
    segments = []
    for raw in re.split(r"[.!?]+", text):
        raw = raw.strip()
        if not raw:
            continue
        segments.extend(p.strip() for p in re.split(COORDINATING_CONJUNCTIONS, raw, flags=re.IGNORECASE) if p.strip())
    return segments


def _looks_like_declarative_segment(segment: str) -> bool:
    """Heuristique volontairement basique ('assez long après découpage') :
    aucune détection réelle de verbe/sujet (nécessiterait un analyseur
    syntaxique, hors périmètre "heuristique simple"). Un segment court issu
    du découpage (ex. "chaleur" après un découpage sur "et") est ignoré."""
    return len(segment.split()) >= MIN_WORDS_PER_SEGMENT


def detect_input_issue(text: str) -> dict:
    """
    Retourne un diagnostic heuristique du texte, calculé AVANT tout appel au
    modèle d'embedding : {"type": "ok"|"multiple"|"vague"|"incoherent",
    "reason": str|None, "segments": [str, ...]}.

    Priorité de détection (un texte peut techniquement déclencher plusieurs
    signaux ; on retient le plus spécifique/le plus tôt dans le pipeline
    utilisateur) : incohérent > vague (trop court/question/commande) >
    saisie multiple > ok. Un texte vide n'est pas traité ici (déjà rejeté en
    amont par check_claim avec un 400 dédié).
    """
    stripped = text.strip()
    if not stripped:
        return {"type": "ok", "reason": None, "segments": []}

    if _symbol_ratio(stripped) > 0.5:
        return {"type": "incoherent", "reason": "symbols", "segments": []}

    word_count = len(stripped.split())
    if word_count >= MIN_WORDS_FOR_CLAIM and not _has_recognizable_function_word(stripped):
        return {"type": "incoherent", "reason": "no_function_words", "segments": []}

    if _is_too_short(stripped):
        return {"type": "vague", "reason": "too_short", "segments": []}

    if _looks_like_question(stripped):
        return {"type": "vague", "reason": "question", "segments": []}

    if _looks_like_command(stripped):
        return {"type": "vague", "reason": "command", "segments": []}

    segments = _split_segments(stripped)
    declarative_segments = [s for s in segments if _looks_like_declarative_segment(s)]
    if len(declarative_segments) >= 2:
        return {"type": "multiple", "reason": None, "segments": declarative_segments[:MAX_SEGMENTS_RETURNED]}

    return {"type": "ok", "reason": None, "segments": []}


def build_guidance_message(issue: dict) -> str:
    """Message d'aide en français, à choix fermés côté frontend (jamais un
    champ de dialogue libre - le système ne peut pas tenir une conversation).
    Le message d'un texte 'incoherent' est volontairement distinct de celui
    d'un verdict NON VÉRIFIABLE existant : NON VÉRIFIABLE signifie une
    affirmation compréhensible mais absente du corpus ; ici, c'est le texte
    lui-même qui n'a pas pu être interprété comme une phrase."""
    if issue["type"] == "multiple":
        return "Votre texte semble contenir plusieurs affirmations distinctes. Que voulez-vous vérifier ?"

    if issue["type"] == "vague":
        if issue["reason"] == "question":
            return (
                "Votre texte ressemble à une question plutôt qu'à une affirmation à vérifier. "
                "Essayez une phrase déclarative, par exemple : « Les températures ont augmenté de X en Y »."
            )
        if issue["reason"] == "command":
            return (
                "Votre texte ressemble à une instruction plutôt qu'à une affirmation. "
                "Reformulez en phrase déclarative, par exemple : « La Terre est ronde » plutôt que « Vérifie que la Terre est ronde »."
            )
        return (
            "Votre texte est trop court pour constituer une affirmation vérifiable. "
            "Essayez une phrase complète, par exemple : « Le niveau de la mer a augmenté de 20 cm depuis 1900 »."
        )

    if issue["type"] == "incoherent":
        return (
            "Ce texte ne ressemble pas à une phrase reconnaissable : il ne peut pas être analysé comme une affirmation "
            "(ceci diffère d'un verdict « non vérifiable », qui signifie une affirmation compréhensible mais absente de nos sources). "
            "Reformulez avec une phrase complète, par exemple : « Les émissions de CO2 ont augmenté depuis 1990 »."
        )

    return ""
