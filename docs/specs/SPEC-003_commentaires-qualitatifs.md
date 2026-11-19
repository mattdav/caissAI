---
type: Spec
id: SPEC-003
title: "Commentaires qualitatifs sur les moments clés d'une partie"
description: "Produire, pour 3 à 5 moments choisis par partie, un commentaire explicatif du niveau de ceux de la partie 316, en un seul appel LLM par partie."
status: draft
superseded-by:
work-item:
tags: [python, chess, llm, commentaires, maia3]
timestamp: 2026-11-19
perimeter: project
audience: []
---

# Commentaires qualitatifs sur les moments clés d'une partie

## Objectif

Les commentaires de la partie 316 sur 9…Cxd5 et 10…b6 ont été jugés excellents par
le professeur de Matthieu. Ils ne viennent pas de caissAI, dont `get_comment` produit
des gabarits chiffrés (« Espérance de gain pour les noirs : -8.49% »). Ils ont été
écrits par un LLM (Fable 2) à qui Matthieu a donné les FEN en disant ne pas comprendre
ces coups : le LLM a rejoué les lignes avec Stockfish 16, puis reconstruit
l'explication à partir des FEN et des évaluations. Pour 9…Cxd5, le seul appui était
la phrase du cours Chessable « L'échange des cavaliers est avantageux pour les
noirs ». Ce procédé est reproductible dans caissAI : Stockfish y tourne déjà. caissAI DOIT produire des commentaires de cette nature sur
quelques moments par partie, sans multiplier les appels API : le budget visé est
**le même nombre d'appels qu'aujourd'hui, soit un par partie**.

## Ce qui fait la qualité des commentaires de la partie 316

Analyse des deux commentaires de référence (variante 9…Cxd5, coup 10…b6 et
variante 10…De7) :

1. **Une question directrice** tirée de la structure : « est-ce que le coup aide ou
   gêne …d4 ? », « qui contrôle d4, et qui attaque d5 ? ». Le coup est jugé contre
   le plan, pas contre un chiffre.
2. **Des faits concrets de pièces et de cases** : le Cc6 perd son soutien b7, le Fc5
   perd sa retraite b6, les cases a6/b5/c6 sont trouées, Ca4 revient dans toutes les
   lignes. Ce sont des faits calculables (défenseurs, cases de retraite, trous).
3. **Des lignes moteur vérifiées**, citées avec leur évaluation, pour le coup joué et
   pour l'alternative.
4. **Le contexte personnel** : la note de répertoire (Schandorff), une partie
   antérieure (#274, même forme d'erreur), et le lieu réel de la défaite
   (« ta partie s'est perdue aux coups 10 et 13, pas au coup 9 »).
5. **Des nuances honnêtes** : « plus simple » ne veut pas dire « bien meilleur »
   (+0,14 contre +0,32).
6. **Un commentaire sur un coup non fautif** : 9…exd5 n'a reçu aucun NAG ; le moment
   instructif était un choix de structure entre deux coups jouables. Les critères
   actuels (`get_nags`) ne le détectent pas.

Un LLM sait rédiger 1, 4 et 5 ; il calcule mal 2 et 3. Pour la partie 316, c'est
Stockfish qui a fourni 2 et 3, à la demande du LLM. Le principe de la feature est de
faire ce travail à l'avance : **tout ce qui se calcule est calculé localement
(Stockfish, python-chess, Maia), le LLM ne fait que rédiger à partir de faits
fournis.** Un seul appel suffit alors, au lieu des allers-retours d'une conversation.

## Périmètre

### Inclus

- Sélection déterministe de K moments clés par partie (défaut 4).
- Un « dossier de faits » par moment, calculé localement, sans appel API.
- Un appel LLM unique par partie, qui rédige les K commentaires et le résumé actuel.
- Une vérification locale des coups et évaluations cités par le LLM.
- Un commentaire posé sur le coup joué ou sur la variante alternative.

### Non-objectifs

Modifier les NAGs ou leurs seuils ; remplacer les commentaires chiffrés existants
(ils restent sur les autres coups) ; boucle d'agent avec le moteur (plusieurs
allers-retours LLM). Une passe de vérification par le LLM reste une évolution
ultérieure possible, hors de ce périmètre.

## Spécification fonctionnelle

### 1. Sélection des moments (local, gratuit)

Chaque demi-coup du joueur analysé (`--player`, sinon les deux camps) reçoit un score.
Signaux, tous disponibles après `evaluate_game` et `get_move_probs` :

| Signal | Définition | Exemple 316 |
| --- | --- | --- |
| Erreur | NAG `?!`, `?`, `??` ; poids croissant avec \|ΔES\| | 10…b6, 13…Fd6 |
| Choix de reprise | L'adversaire vient de prendre en s ; le joueur a au moins deux reprises légales en s, dont une de pion et une de pièce | 9…exd5 (choix entre exd5, Cxd5, Dxd5) |
| Écart de niveau (Maia-3, à valider) | Le meilleur coup Stockfish devient nettement plus probable quand l'Elo de Maia monte de 400 : coup « de connaissance » | à mesurer |
| Coup naturel | Le coup le plus probable selon le modèle de coups humains à l'Elo du joueur perd ≥ 0,10 d'ES (voir « Coup naturel ») | 10…b6 si c'est le coup le plus probable |
| Tournant | Plus forte variation d'ES de la partie, même sans NAG | — |

Pourquoi des signaux sans évaluation : 9…exd5 coûte environ 0,2 pion (+0,32 contre
+0,14 selon Stockfish 16), soit un écart d'ES trop faible pour une règle fondée sur
l'évaluation. Ce qui le rend instructif est un choix de plan (liquider le pion isolé
par …d4), invisible dans le chiffre. Il faut donc repérer la *nature* de la décision.

Mesure du signal « choix de reprise » sur `mes parties.pgn` (300 parties de
Matthieu, 2026-11-19) : 1,55 occurrence par partie en moyenne, dont 0,82 avec un
choix entre pion et pièce. La partie 316 est détectée au coup 9 (choix entre Dxd5,
Cxd5 et exd5). Le volume est compatible avec K = 4. Le signal est déterministe et ne dépend pas de
Maia.

Écarté (2026-11-19) : un signal « sortie du répertoire personnel ». caissAI doit
rester un annotateur générique, utilisable par n'importe quel joueur ; le répertoire
reste l'affaire de chess_coach.

- Le sélecteur DOIT retenir au plus K moments, et au plus un par fenêtre de 3 coups
  du même joueur (sinon 10…b6, 12…Tc8 et 13…Fd6 mangent tout le budget).
- La MultiPV 2 et l'analyse « menace » (coup nul) NE DOIVENT être lancées que sur les
  positions candidates, pas sur toute la partie (coût Stockfish).
- K DOIT être réglable : `--deep-comments K` et `CAISSAI_DEEP_COMMENTS`, `0`
  désactive la feature.

### 2. Dossier de faits par moment (local, gratuit)

Pour chaque moment retenu, un bloc compact (cible : 600 à 900 jetons) :

- FEN, trait, coup joué, historique des 4 derniers coups en SAN.
- Stockfish MultiPV 3 sur la position avant le coup : coup, évaluation, PV en SAN
  (8 demi-coups), profondeur. Évaluation de la ligne réellement jouée.
- Maia : probabilité du coup joué, 3 coups les plus probables à l'Elo du joueur, et
  le plus probable à Elo + 400 (« ce que jouerait un joueur plus fort »).
- Structure, coup joué contre meilleur coup : pions isolés, doublés, arriérés,
  passés, îlots, colonnes ouvertes et semi-ouvertes, cases trouées (non attaquables
  par un pion du camp) dans le camp du joueur.
- Cases clés : pour la case devant chaque pion isolé ou passé et pour les cases de
  rupture (ex. d4), nombre d'attaquants et de défenseurs après le coup joué et après
  le meilleur coup, puis au bout de la PV.
- Pièces : pièces qui perdent un défenseur, cases de retraite perdues, fou bloqué
  par ses propres pions, pièces non défendues.
- Menace adverse : meilleur coup adverse si le joueur passait (coup nul).
- Position du moment dans la courbe d'ES (« la partie s'est perdue aux coups N et
  M »).

Ces fonctions DOIVENT être pures, testées par doctests ou tests unitaires, dans un
module dédié (`bin/position_facts.py`).

### 3. Appel LLM unique par partie

- Un seul appel `client.responses.create` par partie, contenant toutes les positions
  à commenter de la partie, qui remplace l'appel actuel de
  `summarise_game` et produit à la fois les K commentaires et le résumé en 3 phrases.
- Sortie structurée (JSON imposé par schéma) : pour chaque moment, `ply`,
  `placement` (`coup` ou `variante`), `commentaire`, `idee_cle` (12 mots maximum),
  `theme` (étiquette parmi une liste fermée, ex. `pion-isole`, `echange`,
  `cases-faibles`, `developpement`, `calcul`) ; puis `resume`.
- Le préfixe du prompt (consignes, guide de style, exemple de référence tiré de la
  partie 316, liste des thèmes) DOIT être stable d'une partie à l'autre et placé en
  tête, pour bénéficier du cache de prompt automatique de l'API.
- Consignes de rédaction imposées : question directrice d'abord ; ce que demandait la
  position ; ce que le coup change, en pièces et en cases ; une ligne moteur citée ;
  une nuance si l'écart est faible. Français, tutoiement, 80 à 180 mots par moment.
- Interdiction explicite : citer un coup, une ligne ou une évaluation absents du
  dossier de faits.
- Option `--batch`, indépendante du point précédent : au lieu d'un appel immédiat par
  partie, toutes les parties sélectionnées sont envoyées par l'API Batch (tarif réduit,
  résultat sous 24 h). Utile pour annoter un tournoi entier ; le mode par défaut reste
  l'appel immédiat.

### 4. Vérification locale (gratuit)

- Chaque séquence de coups citée DOIT être rejouée avec python-chess depuis la
  position du moment ; une séquence illégale ou absente du dossier entraîne le rejet
  du commentaire, remplacé par le commentaire chiffré actuel, avec un
  `logger.warning`.
- Chaque évaluation citée DOIT correspondre (±0,05) à une évaluation du dossier.
- Les commentaires générés DOIVENT être mis en cache par hachage (PGN de la partie +
  K + version du prompt) : relancer caissAI sur une partie déjà traitée ne rappelle
  pas l'API.

## NAGs positifs : critères proposés

Les NAGs négatifs reposent sur des règles d'évaluation fiables. Les NAGs positifs
(`!!`, `!?`) dépendent de la probabilité Maia, aujourd'hui le point faible. Ce
chapitre propose des critères qui s'appuient sur la distribution complète des coups
et sur des signaux indépendants de Maia, pour moins dépendre de sa calibration. Les
seuils sont des valeurs de départ, à calibrer sur `mes parties.pgn`.

### `!?` — coup piégeux

Définition retenue : un coup pas forcément le meilleur ni le plus évident, qui rend la
position adverse périlleuse parce qu'il est piégeux.

- `P_err(position)` = somme des probabilités Maia (à l'Elo de l'adversaire) des
  réponses adverses qui perdent ≥ 0,10 d'ES par rapport à la meilleure réponse.
  C'est la probabilité que l'adversaire se trompe. Aujourd'hui seule la réponse la
  plus probable est regardée : trois réponses perdantes à 25 % chacune (75 % de
  risque d'erreur) passent inaperçues si la réponse la plus probable, à 30 %, est
  bonne.
- Conditions :
  1. Le coup n'est pas une erreur : perte d'ES ≤ 0,05 par rapport au meilleur coup.
  2. Le piège vient du coup : `P_err(après le coup) − P_err(après le meilleur coup)`
     ≥ 0,25. Sans cette comparaison, un coup quelconque dans une position déjà
     difficile serait noté `!?`. La différence neutralise aussi en partie le biais
     de Maia-3, qui prédit trop d'erreurs (voir Toby ci-dessous).
  3. `P_err(après le coup)` ≥ 0,40 (seuil absolu).
  4. Seuil relatif, décidé le 2026-11-19 : la réponse adverse la plus probable est
     perdante et sa probabilité vaut au moins 2 fois celle de la 2e réponse. Le
     modèle doit être net, pas seulement au-dessus d'un seuil.
  5. Filtre de plausibilité : une réponse qui perd ≥ 0,30 d'ES en un coup (pièce en
     prise, mat en un) n'entre pas dans `P_err`. Ce sont les coups « loufoques »
     hérités du blitz.
- Signal complémentaire, sans Maia : nombre de réponses adverses qui tiennent (à
  moins de 0,05 d'ES de la meilleure). 1 ou 2 réponses = position étroite.
- Coût : une évaluation Stockfish de toutes les réponses légales sur deux positions,
  seulement pour les coups candidats (condition 1 remplie).

### `!!` — coup brillant

Définition retenue : un coup très bon et particulièrement difficile à voir ou à jouer.

- Très bon (toutes requises) :
  1. Le coup joué est le meilleur, ou à ≤ 0,02 d'ES du meilleur.
  2. C'est presque un coup unique : écart d'ES ≥ 0,15 avec le 2e meilleur coup
     (MultiPV 2). Un coup brillant n'a pas d'équivalent.
  3. Résultat : ES ≥ 0,5 après le coup, ou gain ≥ 0,20 sur le 2e meilleur coup (un
     coup unique qui sauve une position perdue compte aussi).
- Difficile (au moins une condition) :
  4. sacrifice (le matériel baisse après le coup et
     l'échange statique sur la case est perdant) ; coup tranquille (ni prise ni échec)
     alors que des prises ou échecs existent ; coup trouvé tard par le moteur (absent
     en tête de l'analyse rapide à 0,3 s que `get_position_depth` fait déjà, donc
     gratuit).
- Cette règle remplace la condition actuelle `is_zeroing` (prise ou coup de pion),
  qui exclut les coups tranquilles et ne mesure pas la difficulté.
- Décision du 2026-11-19 : pas de critère Maia pour `!!`. La probabilité Maia2 n'est
  pas assez fiable pour mesurer la difficulté ; le critère humain (probabilité ≤ 0,10)
  sera réévalué après la comparaison des modèles.

### Défaut constaté dans le `!?` actuel (2026-11-19)

Dans `get_nags`, `es_next_likeliest_before_delta` vaut `es_after_likeliest_move −
es_before_move` : il compare l'ES après le coup le plus probable *du joueur* à l'ES
avant son coup, et ne regarde pas la réponse adverse. `es_next_likeliest_move` est
calculée (à partir de `after_likeliest_move_score`, pas de `next_likeliest_move_score`)
puis jamais utilisée. Seule la condition sur l'avantage regarde la réponse adverse,
mais en valeur absolue : une réponse probable qui *améliore* la position adverse
déclenche aussi le `!?`. Le chiffre affiché par `get_comment` pour un `!?` est donc
faux. Le nouveau critère remplace cette règle ; le défaut est à signaler (règle 5 de
`CLAUDE.md`).

Le seul garde-fou actuel contre une probabilité aberrante est le seuil absolu de
0,33 sur la réponse la plus probable. Il n'y a pas de seuil relatif aux autres coups.
Le nouveau critère ajoute un seuil relatif et un filtre de plausibilité (conditions 4
et 5 ci-dessus).

### `!?` sans Maia (variante de repli)

Le critère `!?` ci-dessus, validé le 2026-11-19, reste le seul NAG qui dépend de
Maia (`P_err`). Si la comparaison des modèles montre que les probabilités ne sont pas
fiables, la variante de repli est la seule mesure Stockfish : nombre de réponses
adverses qui tiennent ≤ 2 et part des réponses légales perdantes ≥ 0,5. Elle est
moins juste : elle ne dit pas si les réponses perdantes sont *tentantes*, ce qui est
précisément ce que mesure Maia.

### Coup naturel : rôle principal du modèle de coups humains

Hors `!?`, le modèle de coups humains (Maia2, Maia-3 ou Toby) ne sert plus aux NAGs.
Son rôle devient d'identifier le **coup naturel** dans une position compliquée : le
meilleur coup de Stockfish y est souvent sophistiqué, et l'explication utile au joueur
est « pourquoi le coup qui vient à l'esprit est moins bon ».

- Coup naturel : coup le plus probable selon le modèle à l'Elo du joueur.
- Position compliquée (toutes requises) :
  1. Le coup naturel perd ≥ 0,10 d'ES par rapport au meilleur coup.
  2. Le meilleur coup n'est pas le coup naturel.
- Ce signal alimente la sélection des moments (ligne « Coup naturel »). Il
  s'applique que le joueur ait joué le coup naturel (erreur à expliquer) ou non.
- Dossier de faits : coup naturel, sa ligne de réfutation et son évaluation. caissAI
  évalue déjà le coup le plus probable à chaque demi-coup (`likeliest_move_eval` dans
  `process_row`) : la ligne de réfutation est disponible sans calcul supplémentaire.
- Consigne de rédaction : expliquer ce que le coup naturel néglige (menace, case,
  pièce), en s'appuyant sur la ligne de réfutation, puis ce que le meilleur coup fait
  de plus.

### Choix du modèle de coups humains

Le choix entre Maia2, Maia-3 et Toby DEVRAIT se faire sur les parties de Matthieu,
selon deux mesures liées aux deux usages restants :

- Coup naturel : part des positions où le coup le plus probable est le coup réellement
  joué, en particulier dans les positions compliquées au sens ci-dessus. C'est la
  mesure principale.
- `!?` : taux d'erreurs prédit contre taux réel, qui conditionne directement `P_err`.
- Coups « loufoques » : parmi les positions où le coup le plus probable du modèle
  perd ≥ 0,20 d'ES, part où le joueur a réellement joué ce coup. C'est la mesure
  directe du défaut reproché à Maia2.
- Modèles comparés : Maia2 `blitz` (actuel), Maia-3, Toby. Maia2 `rapid` est écarté :
  il est entraîné sur beaucoup moins de parties que le modèle `blitz`, et un même Elo
  Lichess rapide correspond à un niveau nettement plus faible qu'en blitz. C'est
  pourquoi caissAI charge le modèle `blitz`. Le décalage entre parties en ligne
  rapides et parties de club à cadence lente est précisément ce que Toby annonce
  corriger (entrées cadence, en ligne ou sur l'échiquier, Elo FIDE équivalent).

Toby ne serait appelé que sur les positions compliquées, ce qui limite son coût
(plusieurs secondes de Stockfish par position). À environ 12 ms par position pour Maia, 300 parties représentent
quelques minutes de calcul.

Toby (MF Yosha Iglesias, AGPL v3, github.com/Yosha87/Toby) corrige les probabilités
de Maia-3 par un petit réseau qui prend en entrée la perte d'ES de chaque coup. Faits
relevés dans `docs/METHOD.md` du dépôt (2026-11-19) :

- Mode « Fast » (parties lentes) : Stockfish évalue chaque coup légal à 150 000
  nœuds, puis les meilleurs candidats à 250 000 ; Leela ne fournit qu'une évaluation
  de politique brute, sans recherche. Ce mode n'exige pas de carte NVIDIA ; seul le
  mode « Accurate » en exige une et retombe sur « Fast » sans elle. Le paquet
  distribué embarque une version CUDA de Leela : l'usage sur CPU suppose son propre
  binaire `lc0`, non testé par l'auteur.
- Rapide et blitz : correction par Stockfish seul, sans Leela.
- Gain mesuré : faible sous 1700 Elo, croissant avec le niveau. Gain marqué sur les
  erreurs : Maia-3 prédit 7,1 % d'erreurs graves en partie lente contre 2,9 % chez les
  vrais joueurs, Toby Fast 3,2 %.
- Coût : évaluer tous les coups légaux à 150 000 nœuds représente plusieurs secondes
  de Stockfish par position, à comparer au coût actuel de `evaluate_game`.
- Maia-3 utilisé par Toby : `maia3-79m`, plus gros que le `maia3-5m` mesuré dans
  SPEC-002.

Conséquence pour caissAI : le biais d'erreurs de Maia-3 gonfle `P_err`, donc les `!?`.
La comparaison prévue au critère 2 de `!?` le compense en partie ; Toby le corrige à
la source. À trancher par la mesure sur les parties de Matthieu.

## Articulation avec Maia-3 (CAI-002)

La feature s'appuie sur Maia et DOIT être implémentée après CAI-002, pour ne pas
écrire deux fois le code des signaux. Ce qu'elle demande à CAI-002, pour un coût
négligeable (environ 12 ms par position, inférence déjà faite) :

- `get_move_probs` DEVRAIT conserver la distribution complète (au moins les 3 coups
  les plus probables et leurs probabilités), pas seulement le coup le plus probable
  et la probabilité du coup joué.
- La sortie de la tête de valeur (`wdl`) DEVRAIT être conservée : l'écart entre
  l'évaluation humaine de Maia et celle de Stockfish mesure la difficulté pratique
  d'une position, signal utile au sélecteur.
- L'Elo continu de Maia-3 permet le signal « Elo + 400 » sans modèle supplémentaire.
- Les étiquettes `theme` produites PEUVENT alimenter chess_coach (axes de travail).

## Coût

| | Aujourd'hui | Avec SPEC-003 (K = 4) |
| --- | --- | --- |
| Appels API par partie | 1 (résumé) | 1 (commentaires + résumé) |
| Jetons d'entrée | PGN complet | ~2 500 (préfixe en cache) + ~3 000 (dossiers) |
| Jetons de sortie | ~150 | ~1 500 à 2 000 |
| Stockfish en plus | — | MultiPV 2-3 et coup nul sur ~8 positions candidates |

Le coût par partie augmente avec la sortie, mais le nombre d'appels ne change pas.
`--batch` et le cache de prompt le réduisent ; `--deep-comments 0` revient au
comportement actuel.

## Choix et contraintes

- Le LLM rédige, il ne calcule pas : c'est ce qui rend un appel unique suffisant et
  ce qui empêche les lignes inventées.
- Après CAI-002, pour bâtir les signaux sur la sortie Maia-3 définitive.
- Changement du comportement d'annotation : à signaler explicitement (règle 5 de
  `CLAUDE.md`), même si les NAGs ne changent pas.

## Critères d'acceptation

- [ ] Sur la partie 316, le sélecteur retient 9…exd5 (choix de reprise) et 10…b6
      parmi ses moments, avec K = 4.
- [ ] Une partie traitée déclenche exactement un appel API (compteur dans les logs) ;
      une seconde exécution sur la même partie n'en déclenche aucun.
- [ ] Aucune séquence de coups illégale ou absente du dossier de faits dans le PGN de
      sortie (test avec réponse LLM simulée contenant une ligne inventée).
- [ ] `--deep-comments 0` produit un PGN identique à la version actuelle.
- [ ] Le professeur juge au moins 3 commentaires sur 4 « utiles » sur 3 parties de
      novembre 2026 (évaluation humaine, à consigner dans `log.md`).
- [ ] `uv run inv lint` passe ; couverture ≥ 60 %.
