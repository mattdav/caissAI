---
type: ProjectJournal
project: caissAI
updated: 2026-08-24
tags: [python, chess, ai]
---

# Leçons apprises

## Un hook ruff `--fix` post-édition retire un import ajouté avant son usage

**Symptôme** : ajouter un import (ex. `parse_game_indices` dans
`__main__.py`) avant que la fonction ne soit réellement utilisée dans le
fichier fait que le hook `PostToolUse` (ruff `--fix`) le retire
silencieusement comme import inutilisé, dans le même tour d'édition.

**Fausse piste** : re-tenter d'ajouter l'import une seconde fois de la
même façon, en pensant à une erreur transitoire du hook.

**Cause racine** : le hook applique `ruff --fix` immédiatement après
chaque `Write`/`Edit`. Tant que l'usage n'existe pas encore dans le
fichier, l'import est objectivement mort à cet instant précis — ruff a
raison localement, même si la séquence d'édition prévue le rendrait
utile ensuite.

**Fix** : séquencer les éditions dans l'autre ordre — ajouter d'abord le
code qui *utilise* le nouveau symbole (au prix d'un `F821 Undefined name`
transitoire signalé par le hook), puis ajouter/réintroduire l'import en
dernier. Le hook laisse alors l'import tranquille car il est déjà
consommé au moment où il apparaît.

## `.env.example` peut être hors de portée du système de permissions même en lecture

**Symptôme** : `Read` sur `.env.example` renvoie "File is in a directory
that is denied by your permission settings", de façon reproductible sur
plusieurs tentatives à des moments différents de la session.

**Fausse piste** : supposer qu'il s'agit d'un denial ponctuel lié à un
état de session particulier et retenter plus tard dans la même session.

**Cause racine** : la restriction est une politique de permission stable
de l'environnement (probablement liée au nom `.env*`, traité comme
sensible par défaut), pas un état transitoire.

**Fix** : traiter ce type de denial comme une frontière d'environnement
définitive pour la session en cours — ne pas retenter, documenter le
contenu attendu dans le rapport/commit pour que l'utilisateur applique le
changement lui-même, plutôt que de chercher un contournement (édition via
un autre outil, écriture d'un fichier différent puis renommage, etc.).
