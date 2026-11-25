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

- option `--moments-json` : sélection des moments clés et dossiers de faits (SPEC-003, CAI-005)
- sélection des moments : écarte les erreurs tactiques et les parties pliées, favorise l'ouverture et la structure, dossier d'erreur avec `punition` (CAI-005)
