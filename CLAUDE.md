---
type: ProjectStandards
project: caissAI
updated: 2026-08-24
tags: [python, chess, ai]
---

# CLAUDE.md

Ce fichier guide Claude Code lors du travail sur ce dépôt.

## Ce que fait caissAI

CaïssAI prend un fichier PGN en entrée et produit une version annotée qui :

1. Identifie l'ouverture (code ECO + nom) depuis la base Lichess (~3 500 ouvertures)
2. Évalue chaque coup avec Stockfish (profondeur adaptée à la complexité de la position)
3. Mesure la probabilité humaine de chaque coup avec Maia2 (modèle imitant un joueur du même niveau)
4. Attribue des NAGs (`!!` `!` `?!` `?` `??` `!?`) selon la formule d'espérance de gain d'Oracle
5. Commente les coups annotés en français (espérance de gain, alternatives)
6. Résume la partie en 3 phrases via GPT *(optionnel, désactivable avec `--no-comment`)*

## Commandes CLI

```bash
uv run caissAI                                    # mode data/input -> data/output
uv run caissAI --pgn "parties.pgn" --list         # liste numérotée, sans clé API/Stockfish requis
uv run caissAI --pgn "parties.pgn" --games 3
uv run caissAI --pgn "parties.pgn" --games 1 5 12
uv run caissAI --pgn "parties.pgn" --games 250:266    # plage inclusive
uv run caissAI --pgn "parties.pgn" --player Dupont
uv run caissAI --pgn "parties.pgn" --output "annotee.pgn"
uv run caissAI --pgn "parties.pgn" --no-comment
uv run caissAI --pgn "parties.pgn" --games 3 --moments-json moments.json
```

Sans `--pgn` (ni `CAISSAI_PGN`) : mode classique, traite tous les PGN de
`<CAISSAI_DATA_DIR>/input/`, écrit dans `<CAISSAI_DATA_DIR>/output/`.
Avec `--pgn` : traite les parties sélectionnées du fichier indiqué, écrit en
mode append dans le fichier source sauf `--output` explicite.

`--list` sort avant toute validation des variables d'environnement — ne
nécessite ni `OPENAI_API_KEY` ni `STOCKFISH_PATH`.

## Architecture

```text
src/caissAI/
├── __main__.py              # CLI argparse, résolution env, deux modes d'exécution
├── bin/
│   ├── annotator.py         # Orchestration haut niveau (les 2 modes d'exécution)
│   ├── game_analyzer.py     # Pipeline d'analyse (process_game et sous-fonctions)
│   ├── nags.py              # Critères `!?` (P_err) et `!!` (Stockfish seul), seuils nommés
│   ├── moments.py           # Moments clés (score, sélection, dossiers de faits)
│   ├── paths.py             # Résolution CAISSAI_DATA_DIR / CAISSAI_LOG_DIR / CAISSAI_PGN
│   ├── position_facts.py    # Faits de position purs (FEN/coups en entrée)
│   └── utils.py             # I/O PGN, parsing --games, helpers (get_es, get_advantage…)
└── config/
    └── lichess_eco.parquet  # Base ECO Lichess, ressource packagée (jamais écrite)
```

`data/` (`input/`/`output/`) et les journaux applicatifs vivent hors du
package installé, sous `%LOCALAPPDATA%/caissAI/` par défaut (voir
`CAISSAI_DATA_DIR`/`CAISSAI_LOG_DIR` plus bas) : une réinstallation ou un
`uv tool upgrade` ne les efface jamais. `config/lichess_eco.parquet` reste
packagé via `utils.get_folder_path("config")` — c'est une ressource de
référence livrée avec le code, jamais écrite à l'exécution.

## Pipeline d'exécution

```text
annotator.comment_games_from_file() / comment_game_from_input_dir()
  │
  ▼  game_analyzer.classify_opening()       -> Game   (headers ECO/Opening, commentaire dernier coup théorique)
  ▼  game_analyzer.extract_game_data()      -> DataFrame  (nœud, coup, ELO, coup forcé)
  ▼  game_analyzer.get_move_probs()         -> DataFrame  (Maia2 : proba du coup joué vs coup le plus probable)
  ▼  game_analyzer.evaluate_game()          -> DataFrame  (Stockfish en parallèle, ThreadPoolExecutor)
       └─ get_position_depth()              -> int        (2 analyses rapides pour mesurer la complexité)
       └─ get_eval()                        -> PovScore    (analyse finale à la profondeur calculée)
  ▼  _build_analysis_columns()              -> DataFrame  (alignement contexte : shift+1 coup précédent, shift-1 adversaire)
  ▼  game_analyzer.get_nags()               -> list[int]  (classification par delta ES + magnitude d'avantage)
  ▼  moments.build_moments() (--moments-json) -> list[dict] (dossiers de faits ; PGN inchangé)
  ▼  game_analyzer.get_comment()            -> str        (commentaire textuel en français)
  ▼  game_analyzer.df_to_pgn()              -> Game       (reconstruction : NAGs, commentaires, variations)
       └─ comment_end()                     -> str        (commentaire sur le dernier nœud)
  ▼  game_analyzer.summarise_game() (opt.)  -> Game       (appel GPT, synthèse 3 phrases en commentaire final)
  │
  ▼
PGN annoté
```

### Formule Oracle — Espérance de gain (`utils.get_es`)

```text
coefficient = elo × −0.00000274 + 0.00048
ES = 0.5 + 0.5 × (2 / (1 + exp(coefficient × centipawns)) − 1)
```

Plus l'ELO est élevé, plus la courbe est abrupte.

### Classification des coups (`game_analyzer.get_nags`)

| Condition | NAG |
| --- | --- |
| Delta ES ≤ −0.20 **et** delta avantage ≥ 3 | `??` Gaffe |
| Delta ES ≤ −0.20 **ou** (≤ −0.10 et delta ≥ 2) | `?` Erreur |
| Delta ES ≤ −0.05 et delta ≥ 1 | `?!` Douteux |
| Stockfish seul (`nags.is_brilliant`) : coup à ≤ 0.02 d'ES du meilleur, écart ≥ 0.15 avec le 2e coup (MultiPV 2), ES ≥ 0.5 (ou gain ≥ 0.20 sur le 2e coup), et sacrifice, coup tranquille ou coup absent de l'analyse rapide | `!!` Brillant |
| +ES ≥ 0.10 sur 2 coups, delta ≥ 2 | `!` Bon coup |
| `P_err` (`nags.is_speculative`) : coup à ≤ 0.05 d'ES du meilleur, `P_err` après le coup − `P_err` après le meilleur coup ≥ 0.25, `P_err` ≥ 0.40, réponse adverse la plus probable perdante ⇒ ≥ 2× la 2e, réponses perdant ≥ 0.30 d'ES écartées | `!?` Piégeux |

`P_err` est la somme des probabilités Maia des réponses adverses qui perdent au
moins 0.10 d'ES (et moins de 0.30) par rapport à la meilleure réponse. Les seuils
sont des constantes nommées de `bin/nags.py`. La colonne `move_probs` (distribution
complète de Maia2) doit être conservée par `get_move_probs`.

## Modules — détail

### `__main__.py`

- `main()` — parse les arguments, charge `.env` (`find_dotenv(usecwd=True)`,
  repli sur la racine du package), résout `config/`, `CAISSAI_LOG_DIR`,
  `CAISSAI_MAIA2_DIR`, puis bascule entre mode `--pgn` et mode dossier
  `input/output`.
- `--list` sort avant toute validation d'env (ni clé API ni Stockfish requis).
- Résolution `OPENAI_MODEL` → `CAISSAI_OPENAI_MODEL` : si la variable
  historique `OPENAI_MODEL` est renseignée, un `logger.warning` est émis ;
  `CAISSAI_OPENAI_MODEL` est prioritaire, défaut `"gpt-4.1"`.

### `bin/paths.py`

- `default_state_dir() -> Path` — `%LOCALAPPDATA%/caissAI` (Windows) ou
  `~/.local/share/caissAI` ailleurs.
- `resolve_path(cli_value, env_var, default) -> Path` — précédence
  CLI > variable d'environnement > défaut ; ne crée ni ne vérifie rien.
- `resolve_data_dir()`, `resolve_log_dir()`, `resolve_pgn_path()` —
  spécialisations pour `CAISSAI_DATA_DIR`, `CAISSAI_LOG_DIR`, `CAISSAI_PGN`.
  `resolve_pgn_path()` seul peut renvoyer `None` (mode dossier classique).
- `resolve_maia2_dir()` — `CAISSAI_MAIA2_DIR`, passé en `save_root` à
  `maia2.model.from_pretrained`. Le défaut de maia2 (`./maia2_models`) est
  relatif au répertoire courant : sans cette variable, lancer caissAI depuis
  un autre dossier retélécharge les poids (plusieurs centaines de Mo).

### `bin/utils.py`

- `read_pgn`, `write_game_to_pgn` — I/O bas niveau sur fichiers PGN.
- `read_games_from_file(pgn_path) -> list[Game]` — lit toutes les parties
  d'un fichier PGN multi-parties (tolérant aux erreurs d'encodage).
- `parse_game_indices(tokens) -> list[int]` — parse `--games`, accepte
  indices isolés (`"184"`) et plages inclusives (`"250:266"`), mélangeables ;
  lève `ValueError` sur token invalide ou plage décroissante. Résultat trié
  et dédupliqué. Pattern identique à
  `chess_coach/bin/pgn_collector.py::parse_game_indices` (dupliqué
  volontairement, pas de package partagé pour l'instant — voir Points
  d'attention).
- `select_games(games, indices, player) -> list[tuple[int, Game]]` — filtre
  par indices 1-basés et/ou nom de joueur (correspondance partielle,
  insensible à la casse, White ou Black).
- `list_games(games) -> None` — affiche la liste numérotée (`--list`).
- `get_es`, `get_advantage`, `get_nag_advantage`, `classify_epd` —
  helpers de classification, décorés `@cached` (mémoïsation via `memoization`).
- `get_folder_path(folder_name) -> Path` — résout un sous-dossier packagé
  via `__file__` ; réservé aux ressources jamais écrites à l'exécution
  (`config/`). Les dossiers d'état runtime (`data/`, `log/`) passent par
  `bin/paths.py`, pas par cette fonction.
- `clean_game`, `clean_previous_variations`, `set_game_length` — helpers
  de manipulation de `chess.pgn.Game`/`GameNode`.

### `bin/game_analyzer.py`

- `MAIA2_ELO_OFFSET: int = 200` — décalage appliqué à l'ELO avant inférence
  Maia2.
- `process_game(...)` — orchestre le pipeline complet décrit ci-dessus.
- `classify_opening`, `extract_game_data`, `get_move_probs`, `evaluate_game`,
  `get_position_depth`, `get_eval`, `get_nags`, `get_comment`, `df_to_pgn`,
  `comment_end`, `summarise_game` — voir le diagramme de pipeline.
- Évaluation Stockfish parallélisée via `ThreadPoolExecutor` : chaque worker
  ouvre sa propre instance moteur.
- Commentaires numérotés (`#4`–`#19`) dans le fichier : renvoient à des
  correctifs/issues résolus historiquement — utile pour comprendre le
  "pourquoi" d'un choix d'implémentation avant de le modifier.

### `bin/position_facts.py`

Fonctions pures (FEN et coups UCI en entrée, dictionnaires en sortie) :
`pawn_structure` (isolés, doublés, arriérés, passés, îlots, colonnes ouvertes et
semi-ouvertes, cases trouées), `square_control` (attaquants/défenseurs),
`recapture_choice` (choix de reprise), `lost_support` (défenseurs et cases de
retraite perdus après un coup), `piece_facts` (fous bloqués, pièces non défendues), `space` (cases de la moitié
adverse contrôlées par chaque camp, avantage d'espace).

### `bin/moments.py`

- `MOMENTS_K = 6` ; `annotated_plies` (NAGs `?!` `?` `??` du PGN source, lus avant
  `clean_game`, signal `annotation`) ; `score_plies` (signaux : annotation, erreur,
  choix de reprise, coup naturel, tournant ; erreurs tactiques `is_tactical` et parties pliées
  `is_decided` écartées, bonus d'ouverture, de structure et d'espace `is_space_exchange`), `select_moments` (au
  plus K, un par fenêtre de 3 coups du même joueur, sauf en ouverture),
  `build_moments` (3K candidats, `punition` dans le dossier d'une erreur) (dossiers de faits ; Stockfish MultiPV 3 et menace
  par coup nul uniquement sur les moments retenus), `write_moments_json`.
  **Modifie la sélection des moments, pas l'annotation du PGN.**
- Le coup naturel est `likeliest_move`, sa réfutation `likeliest_move_eval` :
  aucune évaluation supplémentaire. `process_game(moments_out=...)` ne modifie
  pas le PGN produit.

### `bin/annotator.py`

- `comment_games_from_file(pgn_path, output_path, config_path, engine_path,
  maia2_model, n_workers, openai_client, comment, openai_model, indices,
  player)` — mode `--pgn` : lit un PGN multi-parties, filtre via
  `select_games`, traite chaque partie retenue avec un `try/except` par
  partie (avertissement et passage à la suivante en cas d'échec), écrit en
  mode append.
- `comment_game_from_input_dir(data_path, config_path, engine_path,
  maia2_model, n_workers, openai_client, comment, openai_model)` — mode
  classique `data/input/` → `data/output/`, sans filtrage, sans gestion
  d'exception par partie.
- `__all__ = ["comment_game_from_input_dir", "comment_games_from_file",
  "list_games"]`.
- Les deux fonctions sont décorées `@beartype` et `@time_logger`
  (profilage cProfile loggé en debug, voir `utils.time_logger`).

## Variables d'environnement

| Variable | Obligatoire | Usage |
| --- | --- | --- |
| `OPENAI_API_KEY` | Oui (sauf `--list`) | Clé API OpenAI |
| `STOCKFISH_PATH` | Oui (sauf `--list`) | Chemin absolu vers l'exécutable Stockfish |
| `CAISSAI_OPENAI_MODEL` | Non | Modèle GPT pour le résumé (défaut : `gpt-4.1`) |
| `OPENAI_MODEL` | Non *(dépréciée)* | Ancien nom de `CAISSAI_OPENAI_MODEL`, lue en repli avec `logger.warning` |
| `CAISSAI_PGN` | Non | Fichier PGN par défaut, équivaut à `--pgn` |
| `CAISSAI_DATA_DIR` | Non | Dossier `data/` (mode classique). Défaut : `%LOCALAPPDATA%/caissAI/data` |
| `CAISSAI_LOG_DIR` | Non | Dossier des journaux applicatifs. Défaut : `%LOCALAPPDATA%/caissAI/log` |
| `CAISSAI_MAIA2_DIR` | Non | Dossier des poids Maia2 (`blitz_model.pt`). Défaut : `./maia2_models`, **relatif au répertoire de lancement** |

`.env` est chargé via `find_dotenv(usecwd=True)` avec repli sur la racine du
package ; il est ignoré par git — ne jamais y committer une vraie clé.

## Points d'attention

1. **`parse_game_indices` dupliqué avec chess_coach** — même fonction dans
   les deux dépôts, sans package partagé. Décision explicite (voir
   `.claude/DECISIONS.md`) : à reconsidérer seulement si un troisième cas
   d'usage apparaît.
2. **`data/`/`log/` hors du package** — ne jamais résoudre ces chemins via
   `get_folder_path`/`__file__` ; toujours passer par `bin/paths.py`. Une
   régression ici efface les PGN traités et les logs à la prochaine
   réinstallation.
3. **`--list` doit rester utilisable sans configuration** — ne pas déplacer
   la validation `OPENAI_API_KEY`/`STOCKFISH_PATH` avant le bloc `--list` de
   `__main__.py`.
4. **`.env.example` non synchronisé avec le code** — n'a pas encore été mis
   à jour pour refléter `CAISSAI_OPENAI_MODEL`/`CAISSAI_PGN`/
   `CAISSAI_DATA_DIR`/`CAISSAI_LOG_DIR`. À corriger manuellement.
5. **Pas de rotation des logs** — un fichier horodaté par exécution dans
   `CAISSAI_LOG_DIR`, comportement conservé intentionnellement.
6. **Intégration chess_coach** — chess_coach lit les PGN déjà annotés par
   caissAI et résout `config/lichess_eco.parquet` via `CAISSAI_CONFIG_PATH`
   côté chess_coach ; ne pas dupliquer ce fichier.

## Standards de qualité

- `uv run inv lint` délègue entièrement à `pre-commit run --all-files`
  (ruff, mypy strict, markdownlint, okflint) → objectif 0 erreur.
- `beartype` actif sur toutes les fonctions publiques.
- Docstrings Google sur toutes les fonctions publiques ; doctests exécutés
  via `--doctest-modules` (comptent dans la couverture).
- Couverture mesurée au dernier réalignement : 65,86 % ; seuil configuré
  `fail_under = 60`.
- Commentaires et résumés générés par le pipeline sont en **français**.

## Règles importantes pour Claude Code

1. Ne jamais modifier `.env`, `maia2_models/`, ou `src/caissAI/config/`
   sans demande explicite.
2. Ne jamais committer `.env` ni une clé API en clair.
3. Avant de toucher `bin/paths.py` ou la résolution `CAISSAI_DATA_DIR`/
   `CAISSAI_LOG_DIR`, relire le point d'attention 2 ci-dessus.
4. En cas de duplication constatée avec chess_coach (au-delà de
   `parse_game_indices`), signaler plutôt que de créer un package partagé
   sans accord préalable.
5. Toute modification du comportement d'annotation (NAGs, ES, pipeline)
   doit être signalée explicitement, jamais silencieuse.
