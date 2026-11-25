---
type: Spec
id: SPEC-004
title: "Diagnostic de Toby sur des parties réelles"
description: "Mesurer si Toby, appelé en UCI, tourne sur CPU sous Linux et prédit mieux que Maia2 les coups d'un joueur de club."
status: draft
superseded-by:
work-item:
tags: [python, chess, toby, maia2, diagnostic]
timestamp: 2026-11-25
perimeter: project
audience: []
---

# Diagnostic de Toby sur des parties réelles

## Objectif

SPEC-003 retient Toby, appelé comme un moteur UCI séparé, pour désigner le
coup naturel d'une position et estimer la probabilité d'erreur adverse du NAG
`!?`. Cette Spec vérifie la prémisse avant d'écrire du code : Toby tourne-t-il
sur CPU sous Linux, à quel coût, comment lire sa sortie, et prédit-il mieux que
Maia2 sur de vraies parties de club ? Elle ne modifie ni le code ni les
dépendances de caissAI.

## Périmètre

### Inclus

- Installation d'essai de Toby 1.0 sur CPU, pilotage en UCI par python-chess.
- Mesures M1 à M4 pour Maia2 `blitz`, Maia-3 brut et Toby.
- Valeurs de départ pour les seuils de `!?` de SPEC-003.

### Non-objectifs

- Aucune intégration dans caissAI (c'est l'objet de CAI-002).
- Aucune mesure de Toby 2.0 : ses poids ne sont publiés que sur tribychess.org,
  hors des réseaux autorisés au diagnostic. Tout ce qui suit vaut pour Toby 1.0.

## Constats

### Versions essayées

- Toby 1.0, dépôt `Yosha87/Toby`, commit `b3b4e0f`, `toby_uci.py`.
- Toby 2.0 (HEAD du 2026-11-23) : **non mesurable**, voir Non-objectifs. Les
  conclusions ne s'étendent pas à cette version.
- Maia-3 : `CSSLab/maia3` au commit `1e13597c42d4858b7cfd7cfdae01e297263364b2`,
  poids `UofTCSSLab/Maia3-79M` (Hugging Face).
- `lc0` v0.32.1 compilé depuis les sources, backend BLAS (CPU), réseau
  `BT3-768x15x24h-swa-2790000`.
- Stockfish 17.1 (paquet Debian). Toby est validé avec Stockfish 19 et émet un
  avertissement (`info string Toby: WARNING Stockfish 'Stockfish 17.1' is older
  than the tested version`). Les chiffres de Toby ne sont donc pas ceux
  d'une installation validée.
- Maia2 0.9, modèle `blitz`, avec le décalage `MAIA2_ELO_OFFSET = 200` de caissAI.

### Installation sur CPU

Le calculateur Stockfish de Toby est utilisé : `lc0` est compilé mais n'est pas
lancé pendant l'analyse, ce qui évite tout GPU. Variables d'environnement
nécessaires (chemins propres à l'installation) :

```text
TOBY_LC0=<lc0 compilé>  TOBY_NET=<BT3 .pb.gz>  TOBY_MAIA_REPO=<clone de maia3>
TOBY_MAIA_WEIGHTS=<maia3-79m.pt>  TOBY_DEVICE=cpu  TOBY_BACKEND=blas
TOBY_STOCKFISH=<stockfish>  HF_HOME=<cache>  HF_HUB_OFFLINE=1
```

Un environnement Python séparé porte Maia-3 ; un autre porte Maia2 (dépendances
`gdown`, `pyzstd` ajoutées, poids récupérés par `gdown`). Un premier clone de
Maia-3 a échoué et a dû être refait.

### Appel UCI avec python-chess

- `engine.configure({"MultiPV": …})` est **refusé** par python-chess (l'option
  est gérée par la bibliothèque). Il faut passer `analyse(board, limit,
  multipv=500)`.
- Options utilisées : `White Rating`, `Black Rating` (Elo réels des deux
  joueurs), `Rating Source` = `FIDE`, `Time Control` = `Classical`, précision
  `Accurate`. Le moteur reste ouvert d'une position à l'autre.

### Sortie brute, affichage `Nibbler`

Le premier appel dure environ 9 s (chargement). Chaque position émet plusieurs
blocs ; on garde le **dernier**. `N` est la probabilité de Toby en
cent-millièmes (÷ 100 000), `P` la probabilité de Maia-3 avant correction.
Seuls 7 à 12 coups sont listés ; les autres sont négligeables.

```text
g1f3  (1   ) N:   37049 (+ 0) (P: 38.49%) (WL:  0.39697) (D: 0.255) (M:  60.0) (Q:  0.39697) (U: 0.01000) (S:  0.40697) (V:  0.3970)
b1c3  (2   ) N:   24969 (+ 0) (P: 26.02%) (WL:  0.35030) (D: 0.268) (M:  60.0) (Q:  0.35030) (U: 0.01000) (S:  0.36030) (V:  0.3503)
f1c4  (3   ) N:   12793 (+ 0) (P: 12.84%) (WL:  0.35205) (D: 0.248) (M:  60.0) (Q:  0.35205) (U: 0.01000) (S:  0.36205) (V:  0.3520)
node  (  1) N:  100000 (+ 0) (P:  0.00%) (WL:  0.35788) (D: 0.262) (M:  60.0) (Q:  0.35788) (V:  0.3579)
```

(Position initiale, extrait de 3 des 13 lignes.) Ici, Toby et Maia-3 diffèrent
déjà : `c2c4` passe de 0,44 % (P) à 1 381 / 100 000 de visites.

### Sortie brute, affichage `ChessBase`

Lignes `info … multipv` ; `score cp` porte le **pourcentage** de probabilité
(et non des centipions) :

```text
info depth 10 seldepth 10 multipv 1 score cp 65 nodes 1 nps 10 time 100 pv f3a8
info depth 10 seldepth 10 multipv 2 score cp 34 nodes 1 nps 10 time 100 pv f1e1
info depth 10 seldepth 10 multipv 3 score cp 0 nodes 1 nps 10 time 100 pv c1g5
info string Toby (provisional): Qxa8 65.3% Re1 33.6% Bg5 0.4% Nc3 0.3%
```

(Position du milieu de jeu, premier bloc « provisoire » ; la distribution
finale arrive dans le dernier bloc.)

### Effet des options

Mesuré sur une seule position (blancs 1199, noirs 1099), donc indicatif :

| Option | Valeur | Qxa8 | Re1 | Autres | Temps |
| --- | --- | ---: | ---: | --- | ---: |
| Classement (les deux joueurs) | 1000 | 44 % | 52 % | Bg5 2 % | 0,9 s |
| | 1500 | 68 % | 32 % | | 0,5 s |
| | 2000 | 69 % | 30 % | | 0,6 s |
| | 2500 | 70 % | 30 % | | 0,5 s |
| Cadence | Classical | 52 % | 46 % | Bg5 1 % | 0,0 s |
| | Rapid | 36 % | 50 % | Bg5 8 %, Bf4 2 % | 0,9 s |
| | Blitz | 32 % | 44 % | Bg5 13 %, Bf4 3 % | 1,0 s |
| Source du classement | FIDE | 52 % | 46 % | | 0,0 s |
| | Lichess Blitz | 62 % | 37 % | | 0,5 s |

Le classement et la cadence déplacent nettement la distribution : plus la
cadence est rapide, plus Toby spread vers des coups moins bons. Les temps de
ce tableau sont des mesures de latence sur une position déjà vue (mise en
cache possible), à ne pas confondre avec M4.

La précision `Fast` rend en 0,01 s (contre 2,02 s en `Accurate`) : ce temps
est trop court pour qu'un calculateur ait évalué les coups ; on la tient pour
non exploitable sans vérification supplémentaire, et on ne l'a pas utilisée.

### Lecture recommandée pour caissAI

1. Piloter Toby avec python-chess, `analyse(..., multipv=500)`.
2. Lire le **dernier** bloc `info string` (affichage `Nibbler`), pas le
   premier.
3. Probabilité de Toby = `N / 100000` ; probabilité de Maia-3 brut = `P / 100`.
   Le coup naturel est le coup de plus forte `N`.
4. Renseigner les deux classements (Elo de chaque joueur), la source `FIDE` et
   la cadence `Classical`, et documenter ce choix : il change les résultats.
5. Traiter les coups absents de la liste comme de probabilité nulle (liste
   tronquée à 7–12 coups).

## Jeu de données

- **Origine** : PGN personnel de parties de club (324 parties), copié hors du
  dépôt ; aucun nom de joueur ni liste de parties ici.
- **Tirage** : 60 parties tirées parmi les 322 ayant les deux Elo, graine
  `20261125`.
- **Positions** : 3 693 positions (un coup joué par position). Toby, plus coûteux, est mesuré sur un sous-ensemble de
  **1 200 positions** tiré avec la même graine ; **tous les tableaux de
  Mesures portent sur ces 1 200 positions** et les trois modèles y sont
  évalués sur les mêmes.
- **Référence** : Stockfish 17.1, profondeur 14, MultiPV égal au nombre de coups
  légaux, 1 thread, 128 Mo de hash. Les mats valent 10 000 centipions. ES via
  `utils.get_es` avec l'Elo du joueur au trait.
- **Positions compliquées** (définition imposée : le 2e meilleur coup perd
  ≥ 0,10 d'ES) : 106 sur 1 200 (8,8 %).
- **Erreur grave** (≥ 0,20 d'ES perdu) ; **coup naturel** = coup le plus
  probable du modèle.
- **Répartition des 1 200 positions par tranche d'Elo** (3 693 au total entre
  parenthèses) : 1000-1199 : 95 (304) ; 1200-1399 : 26 (77) ;
  1400-1599 : 326 (935) ; 1600-1799 : 596 (1 824) ; 1800-1999 : 132 (452) ;
  2000-2199 : 22 (76) ; 2200-2399 : 3 (25).
- **Cadence** : les en-têtes ne la donnent que pour 175 positions (`3000+10` :
  107, `600+5` : 68) ; 3 518 n'en ont pas. La répartition par cadence n'est donc
  pas exploitable, et Toby a été lancé en `Classical` pour toutes.

## Mesures

M1 : part des positions où le coup naturel est le coup joué. M2 : part où le
coup naturel est une erreur grave, à comparer à la part des coups joués qui
sont des erreurs graves. M3 (« loufoques ») : parmi les positions où le coup
naturel est une erreur grave, part où le joueur l'a jouée. M4 : temps.
La dernière colonne est la somme moyenne des probabilités du modèle sur les
erreurs graves.

Les tranches ont des effectifs très faibles ; les lignes avec peu de
positions compliquées ne se lisent pas individuellement.

### Maia2 blitz

| Sous-ensemble | n | M1 coup naturel = coup joué | M2 coup naturel = erreur grave | coups joués = erreur grave | n erreurs graves naturelles | M3 « loufoques » | P(erreur grave) moyenne du modèle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Toutes positions | 1200 | 49.9 % | 3.2 % | 1.2 % | 38 | 8 % (3/38) | 5.5 % |
| Positions compliquées | 106 | 81.1 % | 9.4 % | 1.9 % | 10 | 0 % (0/10) | 17.0 % |
| Elo 1000-1199 | 95 | 50.5 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 3.0 % |
| Elo 1200-1399 | 26 | 61.5 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 1.2 % |
| Elo 1400-1599 | 326 | 46.0 % | 2.5 % | 1.2 % | 8 | 0 % (0/8) | 4.3 % |
| Elo 1600-1799 | 596 | 50.7 % | 4.2 % | 1.5 % | 25 | 8 % (2/25) | 6.2 % |
| Elo 1800-1999 | 132 | 52.3 % | 3.0 % | 0.8 % | 4 | 25 % (1/4) | 6.6 % |
| Elo 2000-2199 | 22 | 50.0 % | 4.5 % | 0.0 % | 1 | 0 % (0/1) | 12.4 % |
| Elo 2200-2399 | 3 | 100.0 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 0.3 % |
| Compliquées, Elo 1000-1199 | 6 | 66.7 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 11.7 % |
| Compliquées, Elo 1200-1399 | 1 | 100.0 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 0.9 % |
| Compliquées, Elo 1400-1599 | 26 | 88.5 % | 7.7 % | 0.0 % | 2 | 0 % (0/2) | 12.7 % |
| Compliquées, Elo 1600-1799 | 54 | 75.9 % | 13.0 % | 3.7 % | 7 | 0 % (0/7) | 18.9 % |
| Compliquées, Elo 1800-1999 | 17 | 88.2 % | 5.9 % | 0.0 % | 1 | 0 % (0/1) | 19.6 % |
| Compliquées, Elo 2000-2199 | 2 | 100.0 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 25.2 % |
| Compliquées, Elo 2200-2399 | 0 | - | - | - | - | - | - |

### Maia-3 brut

| Sous-ensemble | n | M1 coup naturel = coup joué | M2 coup naturel = erreur grave | coups joués = erreur grave | n erreurs graves naturelles | M3 « loufoques » | P(erreur grave) moyenne du modèle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Toutes positions | 1200 | 57.7 % | 1.4 % | 1.2 % | 17 | 6 % (1/17) | 2.7 % |
| Positions compliquées | 106 | 82.1 % | 5.7 % | 1.9 % | 6 | 0 % (0/6) | 10.8 % |
| Elo 1000-1199 | 95 | 60.0 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 1.6 % |
| Elo 1200-1399 | 26 | 73.1 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 0.4 % |
| Elo 1400-1599 | 326 | 52.5 % | 0.9 % | 1.2 % | 3 | 0 % (0/3) | 1.9 % |
| Elo 1600-1799 | 596 | 58.7 % | 1.3 % | 1.5 % | 8 | 0 % (0/8) | 3.1 % |
| Elo 1800-1999 | 132 | 61.4 % | 3.0 % | 0.8 % | 4 | 25 % (1/4) | 2.7 % |
| Elo 2000-2199 | 22 | 50.0 % | 9.1 % | 0.0 % | 2 | 0 % (0/2) | 9.1 % |
| Elo 2200-2399 | 3 | 100.0 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 0.3 % |
| Compliquées, Elo 1000-1199 | 6 | 83.3 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 9.2 % |
| Compliquées, Elo 1200-1399 | 1 | 100.0 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 1.8 % |
| Compliquées, Elo 1400-1599 | 26 | 84.6 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 5.4 % |
| Compliquées, Elo 1600-1799 | 54 | 81.5 % | 5.6 % | 3.7 % | 3 | 0 % (0/3) | 12.0 % |
| Compliquées, Elo 1800-1999 | 17 | 82.4 % | 11.8 % | 0.0 % | 2 | 0 % (0/2) | 11.6 % |
| Compliquées, Elo 2000-2199 | 2 | 50.0 % | 50.0 % | 0.0 % | 1 | 0 % (0/1) | 49.6 % |
| Compliquées, Elo 2200-2399 | 0 | - | - | - | - | - | - |

### Toby

| Sous-ensemble | n | M1 coup naturel = coup joué | M2 coup naturel = erreur grave | coups joués = erreur grave | n erreurs graves naturelles | M3 « loufoques » | P(erreur grave) moyenne du modèle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Toutes positions | 1200 | 58.3 % | 0.2 % | 1.2 % | 3 | 33 % (1/3) | 1.5 % |
| Positions compliquées | 106 | 91.5 % | 1.9 % | 1.9 % | 2 | 50 % (1/2) | 6.3 % |
| Elo 1000-1199 | 95 | 57.9 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 1.2 % |
| Elo 1200-1399 | 26 | 65.4 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 0.3 % |
| Elo 1400-1599 | 326 | 52.1 % | 0.0 % | 1.2 % | 0 | nan % (0/0) | 1.2 % |
| Elo 1600-1799 | 596 | 60.6 % | 0.5 % | 1.5 % | 3 | 33 % (1/3) | 1.8 % |
| Elo 1800-1999 | 132 | 62.1 % | 0.0 % | 0.8 % | 0 | nan % (0/0) | 1.0 % |
| Elo 2000-2199 | 22 | 54.5 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 2.2 % |
| Elo 2200-2399 | 3 | 100.0 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 0.3 % |
| Compliquées, Elo 1000-1199 | 6 | 83.3 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 7.9 % |
| Compliquées, Elo 1200-1399 | 1 | 100.0 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 2.1 % |
| Compliquées, Elo 1400-1599 | 26 | 88.5 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 3.1 % |
| Compliquées, Elo 1600-1799 | 54 | 92.6 % | 3.7 % | 3.7 % | 2 | 50 % (1/2) | 8.2 % |
| Compliquées, Elo 1800-1999 | 17 | 94.1 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 3.8 % |
| Compliquées, Elo 2000-2199 | 2 | 100.0 % | 0.0 % | 0.0 % | 0 | nan % (0/0) | 14.6 % |
| Compliquées, Elo 2200-2399 | 0 | - | - | - | - | - | - |


### M4 — temps

- **Toby** : 2,38 s en moyenne par position (médiane 1,72 s, p90 4,72 s, max
  9,6 s), hors chargement initial. Une partie de 40 coups, soit 40 positions
  du joueur analysé, coûte environ **95 s** ; avec les 80 demi-coups des deux
  camps, environ **190 s**.
- **Maia2** : 16,9 ms par position (2 threads), soit environ 0,7 s pour 40
  positions.
- **Maia-3 brut** : pas de temps séparé, c'est un sous-produit de Toby.
- La machine était partagée (charge 4 à 14) : les temps sont bruités et
  plutôt pessimistes.

### Lecture

- **M1** : Toby 58,3 % contre 49,9 % pour Maia2 sur toutes les positions ;
  91,5 % contre 81,1 % sur les compliquées. Maia-3 brut est entre les deux
  (57,7 % ; 82,1 %). Sur les compliquées, la correction de Toby apporte
  environ dix points sur Maia-3 brut ; l'écart avec Maia2 est cohérent d'une
  tranche à l'autre. Ces chiffres dépendent du jeu de référence : Toby est
  construit sur Stockfish, et les coups de ce joueur sont souvent proches de
  ceux de Stockfish.
- **M2** : le coup naturel de Maia2 est une erreur grave dans 3,2 % des
  positions (38 cas), contre 1,2 % de coups réellement joués qui en sont :
  Maia2 **surestime** les erreurs. Toby descend à 0,2 % (3 cas), sous le
  taux réel : il les **sous-estime** légèrement.
- **M3** : 3 cas sur 38 pour Maia2, 1 sur 17 pour Maia-3 brut, 1 sur 3 pour
  Toby, et 0 sur 10 / 6 / 2 sur les compliquées. Ces effectifs sont trop
  faibles pour conclure quoi que ce soit : aucun écart entre modèles n'est
  établi sur M3.

## Conclusion

**Prémisse confirmée**, avec réserves : Toby 1.0 tourne sur CPU sous Linux
sans GPU, à environ 2,4 s par position, et prédit mieux que Maia2 le coup joué
(M1) tout en proposant bien moins de coups naturels qui sont des erreurs graves
(M2). Les réserves sont Toby 2.0 non mesuré, Stockfish 17.1 au lieu de 19,
un sous-ensemble de 1 200 positions de 60 parties d'un même fonds personnel, 106
positions compliquées seulement, et un M3 inexploitable (1 à 3 cas).

### Ce que CAI-002 devra faire

- Appeler Toby en UCI comme décrit dans Constats, avec Stockfish ≥ 19 si
  possible pour retrouver la version validée, et refaire une mesure de contrôle.
- Prévoir environ 95 s par partie de 40 coups pour Toby seul, et le dire à
  l'utilisateur .
- Ne pas supposer que Toby remplace entièrement Maia2 pour les NAGs `?`/`??`
  tant que M3 n'est pas mesuré sur plus de cas : le mesurer sur davantage de
  parties avant de s'en servir comme filtre de « coups loufoques ».
- Rendre configurables (variables d'environnement, jamais en dur) le chemin de
  Toby, du réseau et du clone de Maia-3, et signaler l'avertissement de version
  Stockfish.

### Valeurs de départ pour les seuils de `!?` (SPEC-003)

Les probabilités de Toby sont bien plus basses que celles de Maia2 : la somme
des probabilités d'erreurs graves, `P_err`, dépasse 0,40 dans 46 positions sur
1 200 pour Maia2 mais dans 4 seulement pour Toby (3 sur 106 compliquées).
Le seuil absolu de SPEC-003 (`P_err` ≥ 0,40) serait quasi jamais atteint avec
Toby. Quantiles de `P_err` de Toby (toutes positions) : médiane 0,000,
p75 0,005, p90 0,031, p95 0,077, p99 0,279 ; sur les compliquées : médiane
0,009, p75 0,046, p90 0,181, p95 0,302.

- `P_err(après le coup)` : partir de **0,10** (atteint dans 50 positions sur
  1 200, soit 4,2 %, proche de la sélectivité de 0,40 avec Maia2 : 3,8 %).
- `ΔP_err` et ratio ≥ 2 : non mesurés ici ; garder le ratio de SPEC-003 et
  fixer `ΔP_err` à **0,05** comme valeur provisoire, à recaler sur CAI-002.
- Filtre de plausibilité de 0,30 : non mesuré ; à conserver provisoirement.

## Choix et contraintes

- Version 1.0 mesurée, par cohérence avec la fiche ; la 2.0 n'est pas
  téléchargeable dans ce cadre.
- Référence Stockfish à profondeur fixe 14, pour que les trois modèles soient
  comparés au même jeu de pertes d'ES.
- Aucune donnée personnelle : seuls des agrégats sont publiés.

## Critères d'acceptation

- [ ] Les trois modèles ont M1 à M4 dans `## Mesures`, ou la raison d'un
  manque (M4 de Maia-3 brut).
- [ ] La Conclusion dit « prémisse confirmée » ou « prémisse infirmée » et le
  coût par partie.
- [ ] Aucun fichier de `src`, `tests`, `pyproject.toml` ni `uv.lock` n'a changé.
