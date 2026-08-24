---
type: ProjectJournal
project: caissAI
updated: 2026-08-24
tags: [python, chess, ai]
---

# Décisions

## 2026-04-XX — Migration `config.cfg` vers `.env`

`configparser` supprimé ; `OPENAI_API_KEY`, `STOCKFISH_PATH` et
`OPENAI_MODEL` (devenu `CAISSAI_OPENAI_MODEL`, voir plus bas) sont lus via
`os.environ` après `load_dotenv()`. Le chemin du `.env` est résolu par
`find_dotenv(usecwd=True)` puis repli sur la racine du package.

**Pourquoi** : cohérence avec le reste de l'écosystème (`chess_coach`
utilise déjà `.env`) et suppression d'un fichier de config au format
`.cfg` propriétaire au profit d'un standard partagé.

**Comment l'appliquer** : toute nouvelle valeur configurable côté caissAI
passe par une variable d'environnement, jamais par un nouveau fichier
`config.cfg`-like.

## 2026-04-XX — CLI directe sur un PGN multi-parties (`--pgn`/`--games`/`--player`/`--list`)

Ajout d'un mode d'exécution alternatif au dossier `data/input/` classique :
`--pgn` cible un fichier PGN arbitraire, `--games`/`--player` filtrent les
parties, `--list` les énumère sans configuration requise.

**Pourquoi** : le mode dossier oblige à copier chaque fichier à annoter
dans `data/input/` ; le besoin réel (annoter une partie précise dans une
base ChessBase volumineuse) nécessitait une sélection directe.

**Comment l'appliquer** : le mode dossier reste le comportement par
défaut (sans `--pgn`) pour ne pas casser les usages existants — les deux
modes cohabitent dans `__main__.py` plutôt que de remplacer l'un par
l'autre.

## 2026-08 — Externalisation de `data/`/`log/` hors du package installé (`CAISSAI_*`)

`bin/paths.py` ajouté sur le modèle de `chess_coach/bin/paths.py` :
`CAISSAI_PGN`, `CAISSAI_DATA_DIR`, `CAISSAI_LOG_DIR` remplacent la
résolution via `__file__` pour `data/` et `log/`. `config/lichess_eco.parquet`
reste packagé via `get_folder_path`, inchangé.

**Pourquoi** : le projet doit rester exécutable via
`uv tool install git+https://github.com/mattdav/caissAI` — un dossier
`data/`/`log/` résolu à l'intérieur du package installé serait effacé à
chaque réinstallation/upgrade.

**Comment l'appliquer** : ne jamais résoudre un chemin d'état runtime
(tout ce qui est écrit à l'exécution) via `get_folder_path`/`__file__` ;
toujours passer par `bin/paths.py`. Pas de rotation des logs implémentée
(un fichier horodaté par exécution) — comportement conservé
intentionnellement, à ne pas changer en passant.

## 2026-08 — Pas de package partagé avec chess_coach pour `parse_game_indices`

`parse_game_indices()` (syntaxe `--games N:M`) est dupliquée à l'identique
dans caissAI (`bin/utils.py`) et chess_coach (`bin/pgn_collector.py`),
plutôt que d'être extraite dans un package commun.

**Pourquoi** : un seul point de duplication ne justifie pas la charge
(publication, versionnement, dépendance croisée) d'un package partagé
entre les deux dépôts.

**Comment l'appliquer** : à reconsidérer si un troisième cas d'usage
identique apparaît dans l'un des deux projets.

## 2026-08 — `OPENAI_MODEL` → `CAISSAI_OPENAI_MODEL` avec repli déprécié

Renommage pour la cohérence de préfixage avec `CAISSAI_PGN`/
`CAISSAI_DATA_DIR`/`CAISSAI_LOG_DIR`. `OPENAI_API_KEY` et `STOCKFISH_PATH`
restent inchangées (partagées telles quelles avec d'autres outils comme
Stockfish CLI). `OPENAI_MODEL` reste lue en repli avec un
`logger.warning`.

**Pourquoi** : un `.env` existant ne doit pas se retrouver silencieusement
en échec après la migration — d'où le repli au lieu d'une suppression
sèche.

**Comment l'appliquer** : ne pas supprimer le repli `OPENAI_MODEL` sans
une période de dépréciation explicite communiquée à l'utilisateur.

## 2026-08 — `.env.example` non réécrit dans le cadre du réalignement template

Ce fichier reste non synchronisé avec les variables `CAISSAI_*` ajoutées
(`CAISSAI_OPENAI_MODEL`, `CAISSAI_PGN`, `CAISSAI_DATA_DIR`,
`CAISSAI_LOG_DIR`).

**Pourquoi** : le fichier est inaccessible en lecture et en écriture dans
l'environnement d'exécution (deny du système de permissions), et non
contournable.

**Comment l'appliquer** : à traiter manuellement par l'utilisateur — voir
`.claude/progress.log` pour le contenu attendu.

## 2026-08 — Alignement `pyproject.toml`/`tasks.py` sur `project_template`

`[tool.commitizen]` et `[dependency-groups]` ajoutés (fusion des extras
dev/lint existants, puisqu'il s'agit d'outillage local jamais installé par
un utilisateur PyPI). `line-ending = "lf"`, `fail_under = 60` (couverture
réelle mesurée : 65,86 %). `tasks.py` remplacé par la version courante du
template (clean/lint/precommit-install/test/docs/build/release via
pre-commit et commitizen).

**Pourquoi** : `inv lint` doit déléguer entièrement à
`pre-commit run --all-files`, condition posée par les instructions
globales du dépôt Claude Code.

**Comment l'appliquer** : ne pas réintroduire de logique de lint propre à
`tasks.py` qui dupliquerait `pre-commit`.
