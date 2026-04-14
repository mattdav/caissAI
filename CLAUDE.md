# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

CaïssAI is a Python tool that automatically annotates chess games in PGN format. It:
1. Classifies the opening using the Lichess ECO database
2. Uses Maia2 (a human-like chess model) to get the probability of each move being played
3. Uses Stockfish to evaluate positions and detect complexity
4. Assigns NAGs (Numeric Annotation Glyphs: `!!`, `!`, `?`, `??`, `!?`) to notable moves
5. Optionally calls OpenAI GPT-5 to generate a natural language game summary (in French)
6. Outputs annotated PGN files to `src/caissAI/data/output/`

## Running the program

```bash
cd src/caissAI
python __main__.py
```

Place input PGN files in `src/caissAI/data/input/`. Output is written to `src/caissAI/data/output/` as `<filename>_vCaïssAI.pgn`.

## Installation

```bash
pip install -e .
```

## Configuration

Edit `src/caissAI/config/config.cfg`:
- `[ENGINE] path` — absolute path to the Stockfish executable
- `[OPENAI] api_key` — OpenAI API key (never commit this file with a real key)

## Architecture

```
src/caissAI/
├── __main__.py          # Entry point: reads config, loads models, loops over input PGNs
├── bin/
│   ├── game_analyzer.py # Core analysis pipeline (process_game and all sub-functions)
│   └── utils.py         # PGN I/O, evaluation helpers (get_es, get_advantage, classify_epd)
├── config/
│   ├── config.cfg       # Runtime config (engine path, API key)
│   └── lichess_eco.parquet  # ECO opening database
└── data/
    ├── input/           # Drop PGN files here
    └── output/          # Annotated PGNs are written here
```

### Analysis pipeline (`process_game` in `game_analyzer.py`)

1. `classify_opening` — walks the game backward to find the deepest ECO match; sets `ECO`/`Opening` headers
2. `extract_game_data` — builds a DataFrame of nodes, ELOs, forced/zeroing flags
3. `get_move_probs` — runs Maia2 inference to get probability of played move vs. likeliest move
4. `evaluate_game` — parallelises Stockfish evaluation across moves using `ThreadPoolExecutor`; each worker calls `evaluate_position` which first calls `get_position_depth` to adapt depth to position complexity
5. `get_nags` — compares expected scores (ES via Oracle's formula) and advantage magnitudes to assign NAGs
6. `get_comment` — generates a French text comment for annotated moves
7. `df_to_pgn` — converts the annotated DataFrame back into a `chess.pgn.Game` with variations for blunders/mistakes
8. `summarise_game` — (optional) calls GPT-5 for a three-sentence game summary appended as final comment

### Key concepts

- **Expected Score (ES)**: computed via Oracle's formula (`get_es`) — maps centipawn evaluation + player ELO to a win probability [0, 1].
- **Advantage magnitude**: `get_advantage` maps centipawns to a 0–4 scale; used alongside ES delta to classify moves.
- **Maia2**: loaded once at startup (`model.from_pretrained(type="blitz", device="cpu")`); uses `inference.inference_each` per position.
- **Caching**: `@cached` (from `memoization`) is used on `evaluate_position`, `get_eval`, `get_position_depth`, `get_nags`, `get_comment`, `classify_epd`, `get_es`, `get_advantage` to avoid redundant computation.
- **Parallelism**: Stockfish evaluations run in parallel threads (`ThreadPoolExecutor`); each thread opens its own engine instance.

## Notes

- Comments and game summaries are generated in **French**.
- The `COMMENT = True` flag in `__main__.py` controls whether GPT-5 is called.
- `n_workers` is set to half the logical CPU count via `psutil`.
- The `src/caissAI/bin/resources/` directory contains reference implementations (third-party puzzle generators) that are not part of the active pipeline.

## Qualité du code

- `uv run mypy src/` → objectif : 0 erreurs
- `uv run ruff check src/` → objectif : 0 warnings  
- `uv run ruff format src/` avant tout commit
- beartype actif sur toutes les fonctions publiques

## Commandes disponibles

- `uv run caissAI` — analyse les parties présentes dans le dossier src\caissAI\data\input
- `uv run mypy src/` — vérification des types
- `uv run ruff check src/` — linting
- `uv run pytest` — tests avec coverage
```