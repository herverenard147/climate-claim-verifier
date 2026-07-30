# TERRAVA-AI — Présentation du projet

*Plateforme de fact-checking climatique par intelligence artificielle*

---

## Le problème

La désinformation climatique circule vite, en particulier sur les réseaux sociaux, et elle touche directement des régions comme l'Afrique de l'Ouest et la Côte d'Ivoire — pourtant parmi les plus exposées aux impacts du changement climatique (vagues de chaleur, perturbation des saisons des pluies, montée du niveau de la mer sur les côtes d'Abidjan et de Dakar).

Vérifier une affirmation climatique "à la main" demande du temps : retrouver un rapport du GIEC, de l'Organisation Météorologique Mondiale ou de la Banque Mondiale, l'ouvrir, chercher le bon passage, comparer. Pour un journaliste, un décideur ou un citoyen curieux, ce temps de vérification est souvent un frein — et le champ reste libre pour la désinformation.

**TERRAVA-AI** répond à un besoin simple : donner en quelques secondes un premier avis sourcé sur une affirmation climatique, avec les documents officiels à l'appui, pour que la vérification humaine qui suit soit plus rapide et mieux informée.

---

## Ce que fait TERRAVA-AI

L'utilisateur saisit (ou dicte, ou importe via un PDF) une affirmation climatique. Le système répond avec un verdict :

- ✅ **CONFIRMÉ** — l'affirmation est cohérente avec la littérature scientifique officielle disponible.
- ❌ **RÉFUTÉ** — l'affirmation contredit les données scientifiques officielles.
- ⚠️ **NON VÉRIFIABLE** — le système n'a pas trouvé, dans son corpus, de quoi confirmer ou infirmer l'affirmation de façon fiable.

Chaque verdict est accompagné des extraits de documents (GIEC, OMM, Banque Mondiale, ou base de données scientifique Climate-FEVER) qui ont servi à la décision, avec institution, titre, année et lien vers la source quand elle est disponible en ligne.

**Comment ça marche, en une phrase :** le système compare le sens de l'affirmation à une base de milliers d'extraits scientifiques, retient les plus proches, puis un modèle entraîné juge si ces extraits confirment, contredisent, ou ne permettent pas de trancher l'affirmation.

---

## Pourquoi cette approche est pertinente

- **Zéro-GPU** : tout tourne sur un ordinateur CPU standard, sans matériel spécialisé coûteux — un choix délibéré pour rester déployable à faible coût, y compris dans des contextes à ressources limitées.
- **Rapide** : réponse en une fraction de seconde par affirmation.
- **Sourcé** : jamais de verdict "boîte noire" — chaque décision est appuyée par les extraits de documents réellement utilisés, consultables.
- **Transparent sur ses limites** : quand le système ne trouve rien de pertinent dans son corpus, il le dit clairement ("NON VÉRIFIABLE") au lieu d'inventer une réponse — c'est un garde-fou volontaire contre l'hallucination, testé et vérifié en conditions réelles.
- **Ancré régionalement** : le corpus inclut des données spécifiques à la Côte d'Ivoire et à l'Afrique de l'Ouest (température, montée des eaux, événements extrêmes), en complément de la base scientifique internationale.

---

## Résultats obtenus

Sur un jeu de 1040 affirmations de test, jamais vues pendant l'entraînement, **le système donne le bon verdict dans un peu plus d'un cas sur deux (56%)**, avec un niveau de fiabilité qui varie selon le type de verdict :

- Il est **le plus fiable** quand l'affirmation est confirmée par la science.
- Il est **correct un peu moins de la moitié du temps** quand il s'agit de repérer une affirmation qui contredit clairement les faits (RÉFUTÉ) — c'est aujourd'hui le point le plus fragile du système : distinguer "confirme" de "contredit" quand les deux portent sur le même sujet reste un exercice difficile pour ce type de modèle.

C'est un système d'**aide à la décision utile mais encore imparfait** — comparable à un premier filtre qui doit être complété par un regard humain sur les cas ambigus, pas un arbitre définitif.

---

## Exemple concret

**Affirmation testée :** *"Les températures moyennes en Côte d'Ivoire ont augmenté depuis 1960."*

**Réponse de TERRAVA-AI :**
- Verdict : ✅ **CONFIRMÉ PAR LES DONNÉES SCIENTIFIQUES**
- Source citée en premier : **Banque Mondiale (CCKP)** — *"Les données de la Banque Mondiale indiquent que les températures moyennes annuelles en Côte d'Ivoire ont augmenté d'environ 1°C depuis 1960..."*
- Deux sources complémentaires issues de la base scientifique internationale sont également citées.

En quelques secondes, l'utilisateur obtient un verdict et peut remonter directement à la source pour vérifier par lui-même.

---

## Une plateforme pour apprendre, pas seulement pour vérifier

TERRAVA-AI ne se limite pas à donner un verdict brut : la façon dont l'explication est présentée s'adapte à qui pose la question, pour que l'outil serve aussi de porte d'entrée pédagogique sur le climat, pas seulement de "juge" binaire.

Avant chaque vérification, on choisit un niveau — **débutant, intermédiaire, amateur ou expert**. Le verdict lui-même (confirmé, réfuté, non vérifiable) ne change jamais selon le niveau choisi : c'est uniquement l'explication qui s'adapte.
- En **débutant**, la réponse est formulée en langage courant, sans jargon, avec une explication simple du pourquoi.
- En **intermédiaire**, un peu plus de vocabulaire précis, tout en restant accessible.
- En **amateur**, des éléments de méthode apparaissent (combien de sources ont été consultées, à quel point la correspondance est forte).
- En **expert**, le détail technique complet est affiché (score de similarité exact, classe de classification, probabilités par catégorie).

Chaque utilisateur retrouve aussi ses propres vérifications passées dans un **historique personnel** (sans compte ni mot de passe à créer — un simple identifiant conservé sur son navigateur), avec la possibilité de repartager n'importe quelle vérification déjà effectuée. L'idée : permettre de suivre sa propre progression, de revenir sur une affirmation vérifiée plus tôt, et de partager ce qu'on a appris — pas seulement d'obtenir une réponse ponctuelle.

Ce volet pédagogique inclut aussi un **guidage sur la formulation de la saisie** : si le texte saisi (tapé ou dicté) semble contenir plusieurs affirmations mélangées, ressembler à une question plutôt qu'à une affirmation, ou être trop court ou incompréhensible pour être vérifié, TERRAVA-AI ne devine pas à la place de l'utilisateur et ne rend jamais un verdict global potentiellement trompeur sur un mélange d'affirmations — il propose des choix simples (vérifier chaque affirmation séparément, ou envoyer tel quel malgré tout) et un exemple concret de reformulation, sans jamais bloquer l'utilisateur pressé qui veut continuer tel quel.

---

## Limites actuelles, présentées honnêtement

TERRAVA-AI est un outil d'aide à la décision, pas une autorité absolue. Quelques limites importantes à connaître :

- **Le système est fiable sur les grandes tendances climatiques (nationales, régionales, mondiales, sur plusieurs années ou décennies) — mais il ne peut pas encore vérifier des chiffres très précis et très locaux.** Par exemple, une affirmation du type "la pluviométrie a augmenté de 80% en deux ans dans un quartier précis d'Abidjan" ne pourra pas être confirmée ni réfutée par le système, tout simplement parce qu'aucun rapport officiel ne descend à ce niveau de détail géographique et temporel. Le système répond alors "NON VÉRIFIABLE" — ce qui est la bonne réponse honnête, mais peut donner l'impression que l'outil "ne sait pas répondre" sur exactement le type d'affirmation qui circule le plus sur les réseaux sociaux dans les grandes villes africaines. Ce n'est pas un défaut caché : c'est une limite de portée, propre à la nature des sources disponibles (rapports officiels macro), qu'il est important de faire connaître aux utilisateurs.
- **Les sources affichées ne sont pas toujours parfaitement pertinentes.** Le système sélectionne les extraits les plus proches par le sens général du texte ; il arrive qu'un extrait sur un pays ou un phénomène différent (par exemple une sécheresse ailleurs sur le continent, pour une question portant sur la pluie en Côte d'Ivoire) soit cité alors qu'il n'a pas de lien direct avec la question posée. Une amélioration ciblée est à l'étude pour filtrer ce type de source hors-sujet avant affichage.
- **Le filtre de zone géographique** (Global / Afrique de l'Ouest / Côte d'Ivoire) présent dans l'interface **n'influence pas encore réellement la recherche** — c'est une fonctionnalité prévue mais pas encore active.
- **Couverture des documents institutionnels encore limitée** : la base de données actuelle contient surtout des extraits d'un grand jeu de données scientifique international (Climate-FEVER) ; les documents propres au GIEC, à l'OMM et à la Banque Mondiale ne représentent aujourd'hui qu'une infime partie du corpus (quelques extraits sur près de 4900).
- **Le score de confiance affiché est indicatif**, pas une probabilité statistiquement calibrée.
- **L'historique personnel n'est pas protégé par un mot de passe.** L'identifiant qui permet de retrouver ses vérifications est simplement conservé sur l'appareil utilisé, sans création de compte — pratique et sans friction, mais ce n'est pas un système de sécurité : à ne pas utiliser pour des informations sensibles.

Ces limites sont documentées pour que TERRAVA-AI soit utilisé pour ce qu'il est aujourd'hui : un premier filtre rapide et sourcé, pas un jugement final.

---

## Perspectives d'évolution

- Filtrer plus finement les sources affichées pour écarter celles qui, bien que sémantiquement proches, ne concernent pas la même zone géographique ou le même phénomène climatique que l'affirmation posée.
- Étoffer significativement le corpus de documents institutionnels (rapports complets du GIEC, de l'OMM, de la Banque Mondiale) au-delà des quelques extraits actuels.
- Activer réellement le filtre par zone géographique.
- Améliorer la détection des affirmations qui contredisent la science (RÉFUTÉ), aujourd'hui le point le plus fragile du système.
- Étendre la couverture linguistique au-delà du français et de l'anglais.
- Mettre en place une mise à jour continue du corpus plutôt qu'une base figée reconstruite manuellement.

---

*Projet développé dans le cadre du Hackathon "TTA W3", avec pour objectif une solution frugale (zéro-GPU) et à fort impact dans la lutte contre la désinformation climatique.*
