# Script — version courte, à lire mot pour mot

Objectif réel : 2min45 à 3min15 parlées (cible finale 3-3min30 — vise
en dessous, le débit réel à l'oral rallonge toujours par rapport à une
estimation mécanique. Chronomètre-toi à voix haute avant de figer
cette version).

Nom du produit : **TERRAVA-AI** (nom de code technique : ClimaCheck).

Structure alignée sur le PowerPoint à 5 slides : Titre / Le Défi &
L'Approche / Démonstration / Personnalisation & Résultats / Conclusion.

---

## SLIDE 1 — TITRE

[ACTION : slide de titre]

**TU DIS :**
« Bonjour. Je vous présente TERRAVA-AI, une plateforme de fact-checking
climatique par intelligence artificielle. »

---

## SLIDE 2 — LE DÉFI & L'APPROCHE

[ACTION : slide "Le Défi & L'Approche" — reste sur cette slide pour tout ce bloc]

**TU DIS, partie Défi :**
« Environ 60 % des affirmations climatiques virales sur les réseaux
sociaux sont inexactes, et un fact-check humain prend 2 à 4 heures —
trop lent, surtout en Côte d'Ivoire où ces contenus circulent d'abord
sur WhatsApp. TERRAVA-AI répond en quelques secondes, sans GPU ni
modèle coûteux. »

**TU DIS, partie Approche :**
« Chaque affirmation est comparée aux preuves du GIEC, de l'OMM et de
la Banque Mondiale par recherche sémantique, puis un classificateur
léger rend un verdict : confirmé, réfuté, ou preuves insuffisantes.
Deux garde-fous protègent ce verdict : si aucune preuve ne dépasse un
seuil de similarité minimal, le système refuse de trancher plutôt que
d'inventer une réponse — et un second filtre vérifie que les sources
affichées sont vraiment pertinentes, pas juste proches par leur
vocabulaire. »

---

## SLIDE 3 — DÉMONSTRATION

[ACTION : bascule sur l'app]

**TU DIS :**
« Voici ma réponse en pratique. »

[ACTION : saisir un claim dans l'interface]

**TU DIS, saisie :**
« Je saisis : "les températures moyennes en Côte d'Ivoire ont augmenté
depuis 1960." »

**TU DIS, résultat affiché :**
« Verdict en moins de 200 millisecondes : confirmé, avec la source
réelle citée — ici, la Banque Mondiale. »

[ACTION : montrer un claim hors-sujet]

**TU DIS :**
« Et si je pose une affirmation totalement hors du corpus — sur un
tout autre sujet — il ne force pas un verdict : il répond honnêtement
"non vérifiable". »

[ACTION : coller plusieurs affirmations d'un coup]

**TU DIS :**
« On peut aussi soumettre plusieurs affirmations d'un coup, utile pour
vérifier un article entier plutôt qu'une phrase à la fois — chacune
suit exactement le même pipeline, avec le même seuil. »

---

## SLIDE 4 — PERSONNALISATION & RÉSULTATS

[ACTION : slide "Personnalisation & Résultats", puis retour sur l'app si le temps le permet]

**TU DIS, partie Personnalisation :**
« Avant chaque vérification, on choisit un niveau : débutant,
intermédiaire, amateur ou expert. Le verdict est calculé une seule
fois, mais expliqué différemment selon qui le lit. Chaque vérification
reste dans un historique personnel, consultable et partageable — et un
simple retour, pouce en haut ou en bas, permet de signaler un
désaccord avec le verdict. »

**TU DIS, partie Résultats :**
« Le classificateur atteint un Macro-F1 réel de 0,53, contre 0,485
pour une baseline plus simple — un gain mesuré, pas estimé, sur des cas
jamais vus pendant les réglages. Prochaine étape : affiner encore la
pertinence géographique des sources, et faire de TERRAVA-AI la brique
technique d'une proposition plus large d'éducation aux médias. »

---

## SLIDE 5 — CONCLUSION

[ACTION : slide conclusion]

**TU DIS :**
« Un pipeline complet, zéro-GPU, entièrement mesuré, avec une réponse
adaptée à chacun. Merci. »

---

# NOTES

1. **Ce script est volontairement resserré** : l'OCR, l'ingestion
   continue des rapports, le détail des 4 niveaux, et la hiérarchie
   géographique ne sont pas mentionnés — ce sont des réponses prêtes en
   cas de question du jury (voir point 4), pas du contenu à réciter.
2. **Le garde-fou anti-hallucination (slide 2, partie Approche)** reste
   le point le plus différenciant — ne le coupe jamais, même si tu dois
   raccourcir ailleurs.
3. **Le chiffre 0,53 (slide 4, partie Résultats)** : dis-le sans
   hésitation. Un chiffre réel et mesuré vaut mieux qu'un chiffre
   optimiste non vérifié.
4. **Si le jury pose une question technique**, vocabulaire de secours :
   "RAG" = recherche de preuves avant de répondre · "seuil de
   similarité" = mesure de proximité de sens entre l'affirmation et la
   preuve · "Macro-F1" = une note globale qui traite les 3 verdicts à
   égalité, sans favoriser le plus fréquent.
5. **Si tu dépasses 3min30 au chronométrage réel**, coupe dans cet
   ordre : d'abord la phrase sur la vérification par lot (slide 3),
   puis la phrase sur le feedback pouce haut/bas (slide 4) — jamais la
   démo du verdict confirmé, jamais le garde-fou anti-hallucination,
   jamais le chiffre 0,53.
6. **Slide 2 et Slide 4 contiennent chacune deux temps de parole**
   (Défi/Approche, puis Personnalisation/Résultats) sur une seule
   slide fixe à l'écran — pas besoin de changer de slide entre les
   deux, une légère pause entre les deux suffit à marquer la
   transition à l'oral.
