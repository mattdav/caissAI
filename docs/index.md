# Documentation du projet

Ce dossier regroupe la documentation fonctionnelle et de pilotage du projet,
en complément de la documentation technique générée dans `docs/code/`
(Sphinx, à partir des docstrings du code).

## Les trois types de documents

| Type | Emplacement | Nommage | Rôle |
| --- | --- | --- | --- |
| Spec | `docs/specs/` | `SPEC-<id>_<nom>.md` | Décrit un besoin métier avant implémentation |
| Fix | `docs/fixes/` | `FIX-<id>_<nom>.md` | Décrit un dysfonctionnement, sa cause et sa résolution |
| Plan | `docs/plans/` | `PLAN-<id>_<nom>.md` | Décrit le mode opératoire d'implémentation d'une Spec ou d'un Fix |

Chaque document démarre depuis son modèle dans `docs/_templates/` (`spec.md`,
`fix.md`, `plan.md`), qui fixe le frontmatter et les sections attendues.

Le standard complet (frontmatter, statuts, règles d'usage) est documenté dans
`okf-base.yaml`.
