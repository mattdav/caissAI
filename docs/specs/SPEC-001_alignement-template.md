---
type: Spec
id: SPEC-001
title: "Alignement de caissAI sur project_template et externalisation de l'état runtime"
description: "caissAI doit se conformer aux standards actuels de project_template et devenir exécutable sans code source local après uv tool install."
status: implemented
superseded-by:
work-item:
tags: [python, chess, template-alignment]
timestamp: 2026-08-24
perimeter: project
audience: []
---

# Alignement de caissAI sur project_template et externalisation de l'état runtime

## Objectif

caissAI est resté sur une génération antérieure de `project_template` alors que `chess_coach`, son projet sœur, a déjà été entièrement réaligné et a vu son état runtime externalisé. Deux objectifs motivent ce chantier : la conformité aux standards actuels du template, et l'exécutabilité sans code source local — le programme doit tourner après `uv tool install git+https://github.com/mattdav/caissAI`, les sources locales étant destinées à être supprimées. `chess_coach` et caissAI partagent un même répertoire de travail (`Chess coach/`) et un même `.env`.

## Périmètre

### Inclus

- Rattachement au template via `cruft link` (`.cruft.json`).
- Outillage manquant : pre-commit (ruff, mypy, markdownlint-cli2, prettier, hooks génériques, commitizen, okflint), `.markdownlint.jsonc`, `.prettierrc`/`.prettierignore`, `.gitattributes`, `CHANGELOG.md`, `index.md`/`log.md`.
- Migration `pyproject.toml` (`dependency-groups`, `[tool.commitizen]`, `line-ending = "lf"`, seuil de couverture atteignable) et `tasks.py` (reprise intégrale de la version actuelle du template).
- Migration de la documentation Sphinx `docs/source/` → `docs/code/` (génération par `sphinx-apidoc`), création de `docs/specs/`, `docs/fixes/`, `docs/plans/`, `docs/_templates/`.
- Remplacement de `lint.yml` par `ci.yml` (avec le correctif de plage commitizen déjà présent dans le template), adaptation de `docs.yml`.
- Externalisation de l'état runtime (`src/caissAI/bin/paths.py`, sur le modèle de `chess_coach`) : logs et `data/` résolus hors du package installé (précédence CLI > env > défaut), avec rotation ou fichier unique pour les logs.
- Support de la syntaxe `--games N:M` (plages inclusives, mélangeables avec des indices isolés), sur le modèle de `parse_game_indices()` de `chess_coach`.
- `.env.example` partagé avec `chess_coach` : préfixe `CAISSAI_*` pour les variables propres au projet, `OPENAI_MODEL` renommé `CAISSAI_OPENAI_MODEL` avec repli sur `OPENAI_MODEL`, `OPENAI_API_KEY` et `STOCKFISH_PATH` inchangés.
- `CLAUDE.md` réécrit depuis les sources réelles, complété par `.claude/rules/`, `.claude/commands/`, `.claude/settings.json`, `.claude/DECISIONS.md`/`LESSONS.md` reconstitués à partir du git log et du code.
- Tests : `tests/unit/test_paths.py`, `tests/conftest.py` avec fixture `autouse` neutralisant les variables `CAISSAI_*`.

### Non-objectifs

- Aucune modification du comportement d'annotation des parties (Maia2 + Stockfish + GPT) sans signalement explicite préalable.
- Pas de package partagé entre caissAI et `chess_coach` — la duplication de `parse_game_indices()`/`select_games()` entre les deux dépôts est assumée, pas résolue par une extraction commune.
- Ne couvre pas la correction du bug `src/caissAI/bin/annotator.py` non commité, ni la remise en cohérence de `.github/workflows/test.yml` (supprimé sur disque mais tracké) — constats découverts pendant le diagnostic Phase 0, signalés séparément et hors périmètre de cette spec.

## Spécification fonctionnelle

- Le programme DOIT rester utilisable après `uv tool install git+https://github.com/mattdav/caissAI`, sans dossier source local.
- `config/lichess_eco.parquet` DOIT rester packagé (ressource en lecture seule livrée avec le code), à la différence de `log/` et `data/` qui DOIVENT être résolus hors du package.
- `--pgn` DOIT être alimenté par `CAISSAI_PGN` quand l'argument CLI est absent ; DOIT distinguer l'erreur « aucun PGN fourni » de l'erreur « fichier introuvable » (validation via `.is_file()`).
- `--list` DOIT fonctionner sans `--pgn` dès lors que `CAISSAI_PGN` est défini.
- `--games` NE DOIT PLUS être typé `int` dans argparse ; le parsing des plages `N:M` DOIT se faire après coup, avec `parser.error()` sur format invalide ou plage décroissante.
- `maia2_models/blitz_model.pt` DEVRAIT être évalué pour un téléchargement à la demande via `maia2.model.from_pretrained` plutôt qu'une dépendance à un fichier local préexistant, si le code actuel en dépend au démarrage.
- Le fichier `.env` unique du répertoire de travail partagé DOIT continuer à servir les deux projets sans renommage des variables déjà communes (`OPENAI_API_KEY`).

## Choix et contraintes

- Linters à périmètres strictement disjoints (ruff → `*.py`, markdownlint-cli2 → `*.md`, prettier → `*.json`/`*.jsonc`/`*.yml`/`*.yaml`) — déjà arbitré au niveau du template, non renégociable ici.
- `inv lint` délègue entièrement à `pre-commit run --all-files` — source unique de vérité, cohérence avec `chess_coach`.
- Python figé à 3.13, comme le reste de l'écosystème `project_template`.
- `chess_coach` sert de référence directe d'implémentation (déjà migré) plutôt que de réinterpréter le template à froid — réduit le risque de divergence entre les deux projets sœurs.

## Critères d'acceptation

- [ ] `.cruft.json` présent, `cruft check` ne signale pas d'écart non expliqué.
- [ ] `uv run inv lint` passe à zéro (deux `pre-commit run --all-files` consécutifs, le second sans modification).
- [ ] `uv run inv test` et `uv run inv docs` passent sans erreur.
- [ ] `uv run caissAI --list` fonctionne sans `--pgn` si `CAISSAI_PGN` est défini dans `.env`.
- [ ] `uv run caissAI --pgn <test> --games 1:3 --no-comment` fonctionne (plage `N:M`).
- [ ] Aucun chemin runtime (`log/`, `data/`) résolu à l'intérieur du package installé.
- [ ] `CLAUDE.md` et `.claude/DECISIONS.md`/`LESSONS.md` reflètent l'état réel du code, sans contenu inventé.
