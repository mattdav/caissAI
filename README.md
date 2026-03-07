# CaïssAI

> Annotation automatique de parties d'échecs en PGN — ouvertures, NAGs, commentaires en français, résumé GPT.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-3670A0?logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Coverage 80%+](https://img.shields.io/badge/coverage-80%25%2B-brightgreen)](htmlcov/index.html)

---

## Ce que fait CaïssAI

CaïssAI prend un fichier PGN en entrée et produit une version annotée qui :

1. **Identifie l'ouverture** — code ECO et nom (base Lichess)
2. **Évalue chaque coup** — Stockfish adapte sa profondeur d'analyse à la complexité de la position
3. **Mesure la probabilité humaine** — Maia2 estime la probabilité qu'un joueur de niveau équivalent joue chaque coup
4. **Attribue des NAGs** — `!!` brillant, `!` bon coup, `?!` douteux, `?` erreur, `??` gaffe, `!?` intéressant
5. **Commente les coups annotés** — texte en français avec espérances de gain et alternatives
6. **Résume la partie** *(optionnel)* — trois phrases de synthèse générées par GPT

---

## Installation

### Prérequis

| Outil | Rôle |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Gestionnaire de paquets/environnements |
| [Stockfish](https://stockfishchess.org/download/) | Moteur d'analyse UCI |
| Clé API OpenAI | Résumé GPT *(optionnel)* |

### Installer le projet

```bash
git clone https://github.com/mattdav/caissAI.git
cd caissAI

# Créer l'environnement et installer les dépendances
uv sync

# Pour le développement (tests, linting, docs)
uv sync --all-extras
```

### Configurer

Éditez `src/caissAI/config/config.cfg` :

```ini
[ENGINE]
path = /usr/local/bin/stockfish   ; chemin absolu vers Stockfish

[OPENAI]
api_key = sk-...                  ; laisser vide pour désactiver le résumé GPT
```

> **Ne jamais committer `config.cfg` avec une vraie clé API.**

---

## Utilisation

### Lancer l'annotation

Placez vos fichiers PGN dans `src/caissAI/data/input/`, puis :

```bash
uv run caissAI
```

Les parties annotées sont écrites dans `src/caissAI/data/output/` sous le nom `<fichier>_vCaïssAI.pgn`.

### Exemple de sortie

```pgn
[Event "Rapid"]
[White "Alice"]
[Black "Bob"]
[WhiteElo "1500"]
[BlackElo "1480"]
[ECO "C50"]
[Opening "Italian Game"]
[Annotator "CaïssAI"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 { C50 Italian Game }
4. d3?! { Espérance de gain pour les blancs : -5.32%. }
  ( 4. c3 { Meilleure variante. } )
4... Nf6 5. O-O!! { Un coup brillant qui n'avait que 4.20% de probabilité
d'être joué et qui débouche sur une position avec une espérance de gain
de +18.50% pour les blancs par rapport au coup d3 qui était le plus
probable dans cette position pour un joueur de 1500 ELO. }
...
*  { Espérance de gain pour les blancs : 62.14% }
```

---

## Pipeline d'analyse

```
PGN brut
   │
   ▼
classify_opening()      Remonte la partie pour trouver la position ECO la plus profonde.
                        Écrit ECO + Opening dans les en-têtes, commente le dernier coup théorique.
   │
   ▼
extract_game_data()     Construit un DataFrame : nœud, coup joué, ELO joueur/adversaire,
                        coup forcé, coup zéro (capture ou avance de pion).
   │
   ▼
get_move_probs()        Maia2 calcule la probabilité de chaque coup pour un joueur de
                        ce niveau ELO. Identifie le coup le plus probable (likeliest).
   │
   ▼
evaluate_game()         ThreadPoolExecutor — un worker Stockfish par coup.
   └─ evaluate_position()
       ├─ get_position_depth()   2 analyses rapides (temps t1, t2) pour mesurer la
       │                         complexité : si le meilleur coup change ET l'évaluation
       │                         varie, on augmente la profondeur.
       └─ get_eval()             Analyse finale à la profondeur calculée.
   │
   ▼
_build_analysis_columns()  Aligne chaque ligne avec son contexte : évaluation du coup
                            précédent (shift+1), deux coups avant (shift+2),
                            prochain coup probable de l'adversaire (shift-1).
   │
   ▼
get_nags()              Compare les espérances de gain (formule Oracle) et les
                        magnitudes d'avantage pour attribuer les NAGs.
   │
   ▼
get_comment()           Génère le commentaire textuel en français.
   │
   ▼
df_to_pgn()             Reconstruit le chess.pgn.Game avec NAGs, commentaires
                        et variantes (meilleur coup pour gaffe/erreur/coup douteux).
   │
   ▼
summarise_game() ──(opt)── GPT génère une synthèse en 3 phrases ajoutée en
                            commentaire final.
   │
   ▼
PGN annoté (_vCaïssAI.pgn)
```

### Formule Oracle — Espérance de gain

L'espérance de gain (ES) mesure la probabilité de victoire pour un joueur d'ELO donné :

```
coefficient = elo × −0.00000274 + 0.00048
ES = 0.5 + 0.5 × (2 / (1 + exp(coefficient × centipawns)) − 1)
```

Plus l'ELO est élevé, plus la courbe est abrupte : un écart de 100 cp compte davantage pour un joueur de 2200 que pour un joueur de 1000.

### Classification des coups

| Delta ES | Delta avantage | NAG |
|---|---|---|
| ≤ −0.20 **et** delta ≥ 3 | ≥ 3 | `??` Gaffe |
| ≤ −0.20 **ou** (≤ −0.10 et delta ≥ 2) | ≥ 2 | `?` Erreur |
| ≤ −0.05 | ≥ 1 | `?!` Douteux |
| ES > 0.5, proba ≤ 20 %, coup zéro, +ES ≥ 0.20 vs likeliest | ≥ 3 | `!!` Brillant |
| +ES ≥ 0.10 sur 2 coups | ≥ 2 | `!` Bon coup |
| ES > 0.5, prochain coup adverse probable et mauvais | ≥ 2 | `!?` Intéressant |

---

## Développement

```bash
# Tests + couverture (seuil : 80 %)
uv run pytest

# Vérification des types
uv run mypy src/

# Linting
uv run ruff check src/

# Formatage
uv run ruff format src/
```

---

## Architecture

```
src/caissAI/
├── __main__.py              # Point d'entrée CLI : config, chargement modèles, boucle PGN
├── bin/
│   ├── game_analyzer.py     # Pipeline d'analyse (process_game et sous-fonctions)
│   └── utils.py             # I/O PGN, helpers d'évaluation (get_es, get_advantage, …)
├── config/
│   ├── config.cfg           # Configuration runtime (chemin moteur, clé API)
│   └── lichess_eco.parquet  # Base ECO Lichess (~3 500 ouvertures)
└── data/
    ├── input/               # Déposer les PGN à annoter ici
    └── output/              # PGN annotés générés ici
```

---

## Auteur

[@mattdav](https://github.com/mattdav)
