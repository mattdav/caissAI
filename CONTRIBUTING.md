---
type: ProjectStandards
project: caissAI
updated: 2026-08-24
tags: [python, chess, ai]
---

# Guide de contribution — caissAI

>**Merci de contribuer à ce projet. Ce document décrit les conventions et
processus à respecter pour maintenir une base de code cohérente et de qualité.**

---

## Environnement de développement

### Prérequis

- Python 3.13
- [uv](https://docs.astral.sh/uv/) — gestionnaire de paquets
- [invoke](https://www.pyinvoke.org/) — runner de tâches

### Installation

```bash
git clone <url-du-repo>
cd caissAI
uv sync
cp .env.example .env
```

---

## Configuration et variables d'environnement

**Toute valeur susceptible de changer selon l'environnement doit être externalisée.**
Ce projet charge sa configuration via [python-dotenv](https://github.com/theskumar/python-dotenv)
(`.env`), lu depuis le répertoire de travail (`find_dotenv(usecwd=True)`) — partagé
avec le projet sœur `chess_coach` lorsqu'ils s'exécutent depuis le même dossier.

### Règles

- Les secrets et valeurs locales vont dans `.env` (jamais commité, dans `.gitignore`)
- `.env.example` liste toutes les variables attendues, sans valeurs sensibles (commité)
- Les variables propres à caissAI sont préfixées `CAISSAI_` pour éviter toute
  collision avec `chess_coach` dans le `.env` partagé

### Ce qui doit être variabilisé

- Credentials et secrets (API keys, mots de passe, tokens)
- URLs et endpoints (base URLs, hosts, ports)
- Paramètres d'environnement (timeouts, chemins, feature flags)

### Ce qui ne doit jamais apparaître dans le code

```python
# ❌ Interdit
API_KEY = "sk-1234abcd"
BASE_URL = "https://api.monservice.com"
TIMEOUT = 30

# ✅ Correct
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
```

### Mise à jour de .env.example

Toute nouvelle variable d'environnement doit être ajoutée dans `.env.example`
avec une description en commentaire :

```bash
# URL de l'API externe (obligatoire)
API_BASE_URL=

# Timeout des requêtes en secondes (défaut : 30)
REQUEST_TIMEOUT=30
```

---

## Qualité du code

### Outils

| Outil | Rôle |
| --- | --- |
| [ruff](https://docs.astral.sh/ruff/) | Linting et formatting |
| [mypy](https://mypy.readthedocs.io/) | Vérification des types statiques |

### Commande unique

```bash
uv run inv lint
```

Délègue entièrement à `pre-commit run --all-files` (ruff, mypy, markdownlint-cli2,
prettier, commitizen, okflint). **Doit passer à zéro avant tout commit.**

### Règles strictes

- Aucun `# noqa` sans commentaire justificatif sur la même ligne
- Aucun `# type: ignore` sans commentaire justificatif sur la même ligne
- Type hints obligatoires sur toutes les fonctions et méthodes publiques
- Éviter `Any` sauf cas justifié

---

## Tests

```bash
uv run inv test
```

### Conventions

- Un fichier de test par module : `tests/unit/test_<module>.py`
- Les fixtures partagées vont dans `tests/conftest.py`
- Nommer les tests de façon explicite : `test_<fonction>_<scenario>_<resultat_attendu>`
- Couvrir les cas limites : `None`, liste vide, valeurs invalides, erreurs

### Règle fondamentale

Les tests ne doivent jamais être supprimés pour faire passer la CI.
Un test qui échoue signale un vrai problème à corriger.

---

## Workflow git

### Branches

| Branche | Usage |
| --- | --- |
| `main` | Code stable, prêt pour la production |
| `feat/<nom>` | Nouvelle fonctionnalité |
| `fix/<nom>` | Correction de bug |
| `chore/<nom>` | Maintenance, refactoring, CI |

### Format des commits

```text
feat: ajouter le chargement de configuration depuis yaml
fix: corriger le timeout des requêtes HTTP
chore: mettre à jour les dépendances
docs: compléter la section installation du README
```

Types acceptés : `feat`, `fix`, `chore`, `docs`, `test`, `refactor`

### Avant de committer

```bash
uv run inv lint   # doit passer à zéro
uv run inv test   # tous les tests doivent passer
```

---

## Structure du projet

```text
src/caissAI/
├── bin/        ← scripts exécutables, points d'entrée CLI et logique métier
├── config/     ← ressources de configuration en lecture seule (ex. lichess_eco.parquet)
├── log/        ← configuration du logging
├── __init__.py
└── __main__.py ← point d'entrée principal
```

Chaque sous-module a sa propre responsabilité. Ne pas mélanger logique métier
et configuration dans le même fichier. L'état d'exécution (logs, données
produites) est résolu hors du package installé — voir `bin/paths.py`.

---

## Documentation

```bash
# Générer la documentation Sphinx
uv run inv docs
```

Les docstrings suivent le format [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

---

## Checklist avant une Pull Request

- [ ] `uv run inv lint` passe à zéro
- [ ] `uv run inv test` passe à zéro
- [ ] Les nouvelles fonctions publiques ont des type hints et une docstring
- [ ] Aucune valeur hardcodée (URL, clé, chemin absolu)
- [ ] `.env.example` mis à jour si de nouvelles variables ont été ajoutées
- [ ] Le CHANGELOG ou les notes de commit reflètent les changements
