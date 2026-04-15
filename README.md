# CaïssAI

> Annotation automatique de parties d'échecs — ouvertures ECO, NAGs, commentaires en français, résumé GPT.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-3670A0?logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Ce que fait CaïssAI

CaïssAI prend un fichier PGN en entrée et produit une version annotée qui :

1. **Identifie l'ouverture** — code ECO et nom depuis la base Lichess (~3 500 ouvertures)
2. **Évalue chaque coup** — Stockfish adapte sa profondeur à la complexité de la position
3. **Mesure la probabilité humaine** — Maia2 estime la probabilité qu'un joueur de même niveau joue chaque coup
4. **Attribue des NAGs** — `!!` brillant · `!` bon · `?!` douteux · `?` erreur · `??` gaffe · `!?` intéressant
5. **Commente les coups annotés** — texte en français avec espérances de gain et alternatives
6. **Résume la partie** *(optionnel)* — synthèse en 3 phrases générée par GPT

---

## Prérequis

| Outil | Rôle |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Gestionnaire d'environnements Python |
| [Stockfish](https://stockfishchess.org/download/) | Moteur d'analyse UCI |
| Clé API OpenAI | Résumé GPT *(optionnel)* |

---

## Installation

```bash
git clone https://github.com/mattdav/caissAI.git
cd caissAI
uv sync
```

### Configurer

```bash
cp .env.example .env
# Éditer .env
```

| Variable | Obligatoire | Description |
|---|---|---|
| `OPENAI_API_KEY` | Oui | Clé API OpenAI |
| `STOCKFISH_PATH` | Oui | Chemin absolu vers l'exécutable Stockfish |
| `OPENAI_MODEL` | Non | Modèle GPT pour le résumé (défaut : `gpt-4.1`) |

> `.env` est ignoré par git — ne jamais y committer une vraie clé.

---

## Utilisation

### Mode classique — dossier `input/output`

Déposez vos PGN dans `src/caissAI/data/input/`, puis :

```bash
uv run caissAI
```

Les parties annotées sont écrites dans `src/caissAI/data/output/` sous `<fichier>_vCaïssAI.pgn`.

---

### Mode fichier — sélection précise dans un PGN multi-parties

```bash
# Lister toutes les parties avec leur index
uv run caissAI --pgn "C:/ChessBase/mes parties.pgn" --list

# Annoter la 3ème partie
uv run caissAI --pgn "C:/ChessBase/mes parties.pgn" --games 3

# Annoter plusieurs parties
uv run caissAI --pgn "C:/ChessBase/mes parties.pgn" --games 1 5 12

# Filtrer par joueur (correspondance partielle, insensible à la casse)
uv run caissAI --pgn "C:/ChessBase/mes parties.pgn" --player Dupont

# Choisir le fichier de sortie
uv run caissAI --pgn "C:/ChessBase/mes parties.pgn" --games 3 --output "C:/annotee.pgn"

# Sans résumé GPT (plus rapide)
uv run caissAI --pgn "C:/ChessBase/mes parties.pgn" --games 3 --no-comment
```

Par défaut, les parties annotées sont **écrites dans le fichier source** (mode append). Utilisez `--output` pour choisir un fichier séparé.

### Toutes les options

| Option | Description |
|---|---|
| `--pgn CHEMIN` | Fichier PGN source |
| `--output CHEMIN` | Fichier de sortie (défaut : fichier source) |
| `--games N [N …]` | Indices 1-basés des parties à annoter |
| `--player NOM` | Filtre par nom de joueur |
| `--list` | Liste numérotée des parties et quitte |
| `--no-comment` | Désactive le résumé GPT |
| `--workers N` | Workers CPU (défaut : moitié des CPUs) |

---

## Exemple de sortie

```pgn
[ECO "C50"]
[Opening "Italian Game"]
[Annotator "CaïssAI"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 { C50 Italian Game }
4. d3?! { Espérance de gain pour les blancs : -5.32%. }
  ( 4. c3 { Meilleure variante. } )
4... Nf6 5. O-O!! { Un coup brillant qui n'avait que 4.20% de probabilité
d'être joué… }
*  { Espérance de gain pour les blancs : 62.14% - Les blancs ont bien géré
l'ouverture… }
```

---

## Pipeline d'analyse

```
PGN brut
  │
  ▼  classify_opening()       → ECO + nom dans les headers, commentaire du dernier coup théorique
  ▼  extract_game_data()      → DataFrame : nœud, coup, ELO, coup forcé, coup zéro
  ▼  get_move_probs()         → Maia2 : probabilité de chaque coup pour ce niveau ELO
  ▼  evaluate_game()          → Stockfish en parallèle (ThreadPoolExecutor)
       └─ get_position_depth() → 2 analyses rapides pour mesurer la complexité
       └─ get_eval()           → analyse finale à la profondeur calculée
  ▼  _build_analysis_columns() → alignement contexte : coup précédent (shift+1), adversaire (shift-1)
  ▼  get_nags()               → classification par delta ES et magnitude d'avantage
  ▼  get_comment()            → commentaire textuel en français
  ▼  df_to_pgn()              → reconstruction du Game avec NAGs, commentaires, variantes
  ▼  summarise_game() (opt)   → synthèse GPT en 3 phrases
  │
  ▼
PGN annoté
```

### Formule Oracle — Espérance de gain

```
coefficient = elo × −0.00000274 + 0.00048
ES = 0.5 + 0.5 × (2 / (1 + exp(coefficient × centipawns)) − 1)
```

Plus l'ELO est élevé, plus la courbe est abrupte.

### Classification des coups

| Condition | NAG |
|---|---|
| Delta ES ≤ −0.20 **et** delta avantage ≥ 3 | `??` Gaffe |
| Delta ES ≤ −0.20 **ou** (≤ −0.10 et delta ≥ 2) | `?` Erreur |
| Delta ES ≤ −0.05 et delta ≥ 1 | `?!` Douteux |
| ES > 0.5, proba ≤ 20 %, coup zéro, +ES ≥ 0.20 vs likeliest | `!!` Brillant |
| +ES ≥ 0.10 sur 2 coups, delta ≥ 2 | `!` Bon coup |
| ES > 0.5, prochain coup adverse probable et mauvais | `!?` Intéressant |

---

## Architecture

```
src/caissAI/
├── __main__.py              # CLI argparse — modes input/output et --pgn
├── bin/
│   ├── game_analyzer.py     # Pipeline complet (process_game et sous-fonctions)
│   └── utils.py             # I/O PGN, helpers (get_es, get_advantage…)
├── config/
│   └── lichess_eco.parquet  # Base ECO Lichess (~3 500 ouvertures)
└── data/
    ├── input/               # PGN à annoter (mode classique)
    └── output/              # PGN annotés (mode classique)
```

---

## Développement

```bash
uv run pytest                 # Tests + couverture (seuil 80 %)
uv run mypy src/              # Vérification des types (strict)
uv run ruff check src/        # Linting
uv run ruff format src/       # Formatage
```

---

## Auteur

[@mattdav](https://github.com/mattdav)

---

## Changelog

### [Unreleased]

#### Ajouts
- **CLI `--pgn` / `--games` / `--player` / `--list` / `--no-comment`** : annotation directe dans n'importe quel fichier PGN multi-parties, sans passer par `data/input/`. `--list` affiche la liste numérotée sans nécessiter de clé API ; `--games` sélectionne les parties par index 1-basé ; `--player` filtre par nom de joueur ; `--output` choisit le fichier de sortie (défaut : fichier source en mode append).
- **`python-dotenv`** : dépendance runtime ajoutée ; `types-python-dotenv` ajouté dans les stubs de lint.

#### Modifications
- **Migration `config.cfg` → `.env`** : `configparser` supprimé ; `OPENAI_API_KEY`, `STOCKFISH_PATH` et `OPENAI_MODEL` lus via `os.environ` après `load_dotenv()`. Le chemin du `.env` est résolu par `find_dotenv(usecwd=True)` puis fallback sur la racine du package.
- **`OPENAI_MODEL` externalisé** : la constante `_OPENAI_MODEL` retirée de `game_analyzer.py` ; le modèle est propagé par injection dans `process_game()` et `summarise_game()`.
- **Corrections beartype** : `get_nags()` — `next_likeliest_move_score: PovScore | None` ; `get_comment()` — `next_likeliest_move: Move | None` avec guard sur `NAG_SPECULATIVE_MOVE`.
