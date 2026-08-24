---
type: Plan
id: PLAN-001
title: "Alignement de caissAI sur project_template et externalisation de l'état runtime"
description: "Exécution en 11 phases de SPEC-001 : rattachement au template, outillage, docs, CI, externalisation de l'état runtime, nettoyage git, syntaxe --games N:M, .env partagé, CLAUDE.md, tests."
status: draft
implements: SPEC-001
tags: [python, chess, template-alignment]
timestamp: 2026-08-24
perimeter: project
audience: []
---

# Alignement de caissAI sur project_template et externalisation de l'état runtime

## Périmètre

Ce plan couvre les phases 1 à 11 de SPEC-001. La phase 0 (diagnostic, comparaison template/caissAI, quatre arbitrages "poids mort", rédaction de SPEC-001 et du présent plan) est terminée et constitue le point d'arrêt obligatoire : aucune étape ci-dessous ne s'exécute avant validation explicite de l'utilisateur.

Chaque phase produit un commit atomique, conventionnel, en français. `.claude/progress.log` est mis à jour en fin de chantier ; `.claude/DECISIONS.md` et `.claude/LESSONS.md` sont alimentés au fil des phases concernées. Le statut de ce plan passe à `done` uniquement après la vérification finale (phase 11) réussie.

## Étapes

### 1. Rattachement au template

**Fichiers cibles** : `.cruft.json`

**Action** : `cruft link` vers `project_template` (préférer `cruft link` à une écriture manuelle). Résoudre les variables Jinja : `project_name=caissAI`, `author=Matthieu Daviaud`, `email=matthieu.daviaud@gmail.com`, `license=MIT license`, `publish_pypi=no`, `use_wiki=no`, `_python_version=3.13`.

**Vérification** : `cruft check` et `cruft diff` exécutés, écarts reportés à l'utilisateur avant de continuer.

### 2. Outillage manquant

**Fichiers cibles** : `.pre-commit-config.yaml`, `.markdownlint.jsonc`, `.prettierrc`, `.prettierignore`, `.gitattributes`, `CHANGELOG.md`, `index.md`, `log.md`, `CONTRIBUTING.md`

**Action** : créer `.pre-commit-config.yaml` (ruff, mypy via `uv run`, markdownlint-cli2, prettier, hooks génériques, commitizen sur commit-msg, okflint en hook local ; pas de bloc wiki-check). Créer `.markdownlint.jsonc`, `.prettierrc`, `.prettierignore`, `.gitattributes` (avec `merge=union` sur `.claude/progress.log`), `CHANGELOG.md` (OKF `ProjectLifeCycle`), `index.md`, `log.md`. `CONTRIBUTING.md` existe déjà : l'aligner sur le template plutôt que l'écraser, reporter les deltas substantiels.

**Vérification** : `pre-commit run --all-files` s'exécute (échecs de contenu attendus à ce stade, seule l'absence d'erreur de configuration est vérifiée ici).

### 3. `pyproject.toml` / `tasks.py`

**Fichiers cibles** : `pyproject.toml`, `tasks.py`

**Action** : migrer les outils de dev vers `[dependency-groups]` ; ajouter `[tool.commitizen]` (`version_files` sur `pyproject.toml` et `src/caissAI/__init__.py:__version__`, `tag_format = "v$version"`) ; `[tool.ruff.format] line-ending = "lf"` ; fixer `fail_under` de couverture à une valeur immédiatement atteignable (proposer la valeur à l'utilisateur avant de l'appliquer, avec une cible de progression future). Reprendre `tasks.py` de `chess_coach` tel quel (délégation pre-commit, `htmlcov`/`docs/build`/`docs/code/api` dans `CLEAN_DIRS`, `.coverage` dans `CLEAN_FILES`, `sphinx-apidoc` dans la tâche `docs`).

**Vérification** : `uv sync` réussit, `uv run inv --list` affiche les tâches attendues.

### 4. Documentation Sphinx

**Fichiers cibles** : `docs/source/` → `docs/code/`, `docs/specs/`, `docs/fixes/`, `docs/plans/`, `docs/_templates/`, `docs/index.md`, `docs/Makefile`, `docs/make.bat`

**Action** : migrer la config Sphinx sur le modèle de `chess_coach` (`sys.path` sur `src/`, extensions `napoleon` + `viewcode` + `sphinx_autodoc_typehints`, `index.rst` avec toctree vers `api/modules`, génération via `sphinx-apidoc` dans `inv docs`, `docs/build/` et `docs/code/api/` gitignorés). Une fois la génération automatique vérifiée, supprimer les `.rst` écrits à la main (`game_analyzer.rst`, `utils.rst`) devenus obsolètes.

**Vérification** : `uv run inv docs` construit sans erreur. Confirmer avec l'utilisateur avant de supprimer `docs/Makefile`/`docs/make.bat` si `inv docs` les remplace intégralement.

### 5. CI GitHub

**Fichiers cibles** : `.github/workflows/lint.yml` → `ci.yml`, `.github/workflows/docs.yml`

**Action** : remplacer `lint.yml` par le `ci.yml` du template tel quel (déjà porteur du correctif de plage commitizen via `base_ref`/`github.event.before`, ne pas le réécrire). Adapter `docs.yml` au chemin `docs/code/`.

**Vérification** : workflows valides (`yamllint` ou vérification manuelle de la syntaxe), déclenchement observé sur le prochain push/PR.

### 6. Externalisation de l'état runtime

**Fichiers cibles** : `src/caissAI/bin/paths.py` (nouveau), `src/caissAI/__main__.py`

**Action** : créer `paths.py` sur le modèle exact de `chess_coach/bin/paths.py` (`default_state_dir()`, `resolve_path(cli_value, env_var, default)`, variables `CAISSAI_PGN`, `CAISSAI_DATA_DIR`, `CAISSAI_LOG_DIR`). Dans `__main__.py` : `--pgn` alimenté par `CAISSAI_PGN` si absent ; distinguer « aucun PGN fourni » de « fichier introuvable » (`.is_file()`) ; `--list` fonctionnel sans `--pgn` si `CAISSAI_PGN` est défini (corrige un bug actuel) ; logs et `data/` résolus hors du package. `config/lichess_eco.parquet` reste packagé, ne pas y toucher.

**Vérification** : exécution manuelle confirmant la résolution CLI > env > défaut pour chaque chemin. Avant d'implémenter la rotation des logs, proposer à l'utilisateur le choix entre rétention des N derniers fichiers (N configurable) ou `RotatingFileHandler` unique, et attendre confirmation.

### 7. Poids mort en git — arbitrage requis

**Fichiers cibles** : aucun par défaut — dépend de l'arbitrage utilisateur

**Action** : ne rien supprimer sans validation explicite préalable. Présenter les quatre points (logs `src/caissAI/log/`, `.coverage` racine, `maia2_models/blitz_model.pt`, `resources/`) avec leur état réel vérifié (voir constat Phase 0 : aucun des quatre n'est actuellement tracké par git — tous déjà exclus via `.gitignore`). Si l'utilisateur confirme, appliquer l'action retenue pour chaque point individuellement.

**Vérification** : `git ls-files` ne référence plus aucun des quatre chemins après action (si action il y a) ; sinon, aucune modification.

### 8. Sélection de parties par plage `N:M`

**Fichiers cibles** : `src/caissAI/bin/utils.py`, `src/caissAI/__main__.py`

**Action** : vérifier d'abord l'état réel de `--games` dans `utils.py` (ne pas présumer l'absence totale de la fonctionnalité). Reprendre `parse_game_indices()` de `chess_coach/src/chess_coach/bin/pgn_collector.py` (docstring/doctests portables, erreurs explicites sur plage décroissante et token malformé) à côté de `select_games()`. `--games` n'est plus typé `int` dans argparse ; le parsing se fait après coup via `parser.error()` sur format invalide.

**Vérification** : `--games 250:266`, `--games 3 250:266 300` et un token invalide produisent le comportement attendu. Signaler à l'utilisateur que cette fonction est dupliquée entre les deux dépôts — ne pas créer de package partagé de sa propre initiative.

### 9. `.env` partagé avec `chess_coach`

**Fichiers cibles** : `.env.example`, `README.md`

**Action** : préfixer toutes les variables propres à caissAI de `CAISSAI_`. Ne jamais renommer `OPENAI_API_KEY` (commune aux deux projets). Renommer `OPENAI_MODEL` en `CAISSAI_OPENAI_MODEL` avec repli documenté sur `OPENAI_MODEL` (déprécié). `STOCKFISH_PATH` garde son nom (chemin système, pas une variable de projet). Réécrire `.env.example` sur le modèle de `chess_coach` (sections commentées, distinction requis/optionnel, défauts explicites). Ajouter une section « Utilisation conjointe avec chess_coach » au `README.md`.

**Vérification** : `find_dotenv(usecwd=True)` résout le même `.env` depuis le répertoire de travail partagé pour les deux projets.

### 10. `CLAUDE.md` et cohérence

**Fichiers cibles** : `CLAUDE.md`, `.claude/rules/python-style.md`, `.claude/rules/docs-standard.md`, `.claude/commands/doc-new.md`, `.claude/settings.json`, `.claude/DECISIONS.md`, `.claude/DECISIONS-archive.md`, `.claude/LESSONS.md`, `.claude/agents/.gitkeep`, `.claude/skills/.gitkeep`

**Action** : relire `CLAUDE.md` intégralement contre le code réel et le réécrire depuis les sources selon la structure du template (Structure/Commandes/Architecture/Dépendances notables/Mémoire, imports `@.claude/DECISIONS.md` et `@.claude/LESSONS.md`). Compléter les fichiers `.claude/` manquants. Si `.claude/DECISIONS.md` est vide ou squelettique, reconstituer les décisions réelles depuis le git log et le code (pool de moteurs Stockfish pour la parallélisation, `ThreadPoolExecutor` plutôt que `ProcessPoolExecutor` sous Windows, abandon de `@cached` sur l'analyse Stockfish, `Limit(depth=X)` contre les artefacts de `lower_bound`, remplacement de `importlib.resources.path` par `__file__`). N'inventer aucun contenu ; signaler ce qui reste incertain.

**Vérification** : chaque affirmation de `CLAUDE.md` est vérifiable dans le code ou le git log au moment de la relecture finale.

### 11. Tests et validation

**Fichiers cibles** : `tests/unit/test_paths.py`, `tests/conftest.py`

**Action** : ajouter `test_paths.py` et une fixture `autouse` dans `conftest.py` neutralisant les variables `CAISSAI_*` (précédence CLI/env/défaut, `parse_game_indices`, résolution du PGN).

**Vérification** :
- `uv sync`
- `uv run inv precommit-install`
- `uv run inv lint` → zéro erreur
- `uv run inv test` → tous les tests passent
- `uv run inv docs` → Sphinx construit sans erreur
- Deux `pre-commit run --all-files` consécutifs : le second ne modifie rien (sinon, corriger le chevauchement de périmètre des linters plutôt que d'arbitrer au cas par cas)
- `uv run caissAI --pgn "<PGN de test>" --list`
- `uv run caissAI --pgn "<PGN de test>" --games 1:3 --no-comment`

## Mise à jour documentaire

- [ ] `README.md` : section « Utilisation conjointe avec chess_coach » (phase 9)
- [ ] `CLAUDE.md` réécrit depuis les sources (phase 10)
- [ ] `.claude/DECISIONS.md` et `.claude/LESSONS.md` alimentés au fil des phases
- [ ] `.claude/progress.log` : entrée de fin de chantier
- [ ] `CHANGELOG.md` mis à jour si la version est bumpée

## Vérification finale

- [ ] Les 8 vérifications de l'étape 11 passent toutes
- [ ] `SPEC-001` passe à `status: implemented`
- [ ] Ce plan passe à `status: done`
- [ ] Un commit atomique par phase existe dans l'historique git, format conventionnel
