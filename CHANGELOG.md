---
type: ProjectLifeCycle
project: caissAI
status: active
updated: 2026-08-24
tags: [python, chess, ai]
---

# Changelog

All notable changes to caissAI are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/).

This file is maintained automatically by [commitizen](https://commitizen-tools.github.io/commitizen/).
Do not manually edit the generated sections.

---

## [Unreleased]

### Feat

- NAG `!?` (coup piégeux) fondé sur `P_err`, la somme des probabilités des réponses adverses perdantes, avec ses cinq conditions ; `!!` (coup brillant) fondé sur Stockfish seul (coup quasi unique, sacrifice, coup tranquille ou trouvé tard), sans probabilité Maia ni condition sur les prises et coups de pion ; le commentaire de `!?` et de `!!` affiche les nouveaux chiffres (SPEC-003, CAI-004). Change les annotations produites.
- `get_move_probs` ajoute la colonne `move_probs` (distribution complète de Maia2) (CAI-004)
- option `--moments-json` : sélection des moments clés et dossiers de faits (SPEC-003, CAI-005)
- moments clés : signal `annotation` (NAG du PGN source), tactique mesurée sur 6 demi-coups, fait d'espace et `echange_en_avantage_d_espace` dans le dossier, K = 6 (CAI-005)
- sélection des moments : écarte les erreurs tactiques et les parties pliées, favorise l'ouverture et la structure, dossier d'erreur avec `punition` (CAI-005)

### Fix

- NAG `!?` : l'ancien critère comparait l'ES après le coup le plus probable du joueur à l'ES avant son coup, ne regardait pas la réponse adverse et déclenchait aussi pour une réponse qui améliore la position adverse ; les chiffres du commentaire étaient faux (CAI-004)
