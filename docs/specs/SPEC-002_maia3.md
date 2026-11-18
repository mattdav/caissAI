---
type: Spec
id: SPEC-002
title: "Diagnostic d'intégration de Maia-3 dans caissAI"
description: "Établir comment obtenir de Maia-3 la probabilité de chaque coup joué, à quel coût, et ce que change sa licence AGPL-3.0."
status: draft
superseded-by:
work-item:
tags: [python, chess, maia3, diagnostic]
timestamp: 2026-11-18
perimeter: project
audience: []
---

# Diagnostic d'intégration de Maia-3 dans caissAI

## Objectif

Matthieu a décidé le 2026-11-16 de remplacer Maia2 par Maia-3. caissAI a besoin,
pour chaque position, de la probabilité du coup joué (même improbable) et du coup
le plus probable, pour le niveau des deux joueurs : c'est le contrat de
`get_move_probs` (`src/caissAI/bin/game_analyzer.py`). Ce diagnostic vérifie que
Maia-3 sait le fournir, à quel coût, et sous quelle licence. Il ne modifie ni code
ni dépendances.

## Périmètre

### Inclus

- Installation d'essai de Maia-3 dans un environnement jetable sous `/tmp`.
- Constats sur l'API, les Elo, le cache des poids ; mesures de temps ; faits de licence.

### Non-objectifs

Remplacer Maia2 dans le code (CAI-002), tester Toby, changer les NAGs ou les
commentaires.

## Constats

### Commit essayé

`https://github.com/CSSLab/maia3` au commit `1e13597c42d4858b7cfd7cfdae01e297263364b2`
(2026-05-25), version `maia3 0.1.0`. Poids `UofTCSSLab/Maia3-5M`, révision Hugging Face
`b6559de2398d7140b985f28fd2c19fb5e47ddabe` (fichier `maia3-5m.pt`, environ 21 Mo).

### Versions exigées

- Python : `>=3.10` (`pyproject.toml`). Essayé avec 3.13.5, la version de caissAI.
- torch : aucune borne (`"torch"` sans version). Essayé avec `2.15.0+cpu`, ce qui
  rend compatible le plancher `torch>=2.10.0` de caissAI.
- Autres dépendances : `huggingface-hub>=0.23`, `numpy`, `python-chess`. Aucune
  n'est nouvelle pour caissAI sauf `huggingface-hub`.
- Pas de paquet PyPI : l'installation se fait depuis le dépôt git. Pour caissAI,
  une dépendance `maia3 @ git+https://github.com/CSSLab/maia3@<commit>` est
  nécessaire.

### Fonction rendant la distribution complète : elle n'existe pas

Le paquet `maia3` n'expose **aucune** fonction publique « FEN + deux Elo →
distribution des coups ». Il fournit seulement les briques (`maia3.models.MAIA3Model`,
`maia3.dataset.tokenize_board`, `get_historical_tokens`, `get_legal_moves_mask`,
`maia3.utils.get_all_possible_moves`, `mirror_move`,
`maia3.model_registry.resolve_model_spec`, `apply_model_config`,
`resolve_checkpoint_path`). La seule logique d'inférence complète est la méthode
`Maia3UCIEngine.score_moves` de `maia3/uci.py`, qui ne rend que le coup échantillonné
et les `MultiPV` coups les plus probables.

**Le moteur UCI ne peut pas rendre la probabilité d'un coup donné.** Vérifié en
exécutant `maia3-uci --model maia3-5m` sur une position à 40 coups légaux avec
`setoption name MultiPV value 50` :

- `MultiPV` est plafonné à 20 (`min 1 max 20`) : 20 coups sur 40 seulement sont émis.
- Les lignes `info` portent `score cp`, `wdl` (issus de la tête de valeur, après le
  coup) et `pv` ; la probabilité du coup (`policy`) calculée en interne n'est **pas
  imprimée**. Seul le rang (`multipv N`) la reflète.
- Un coup joué hors des 20 premiers est donc invisible, et aucun coup n'a de
  probabilité chiffrée.

Le chemin UCI ne couvre donc pas le besoin. Il faut appeler le modèle en Python avec un
adaptateur d'environ 40 lignes, reprenant `score_moves` sans l'échantillonnage.
Prototype essayé (`/tmp/cai001/work/m3api.py`), qui a produit les mesures ci-dessous :

```python
from collections import deque
from types import SimpleNamespace

import chess
import torch
from maia3.dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
from maia3.model_registry import apply_model_config, resolve_checkpoint_path, resolve_model_spec
from maia3.models import MAIA3Model
from maia3.utils import get_all_possible_moves, mirror_move


def load(name="maia3-5m", cache_dir=None, device="cpu"):
    spec = resolve_model_spec(name)
    cfg = SimpleNamespace()
    apply_model_config(cfg, spec)  # architecture de la taille choisie
    path = resolve_checkpoint_path(spec, cache_dir=cache_dir)  # téléchargement HF + cache
    model = MAIA3Model(cfg).to(device)
    sd = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict({k.replace("smolgen", "gab"): v for k, v in sd.items()}, strict=False)
    model.eval()
    moves = get_all_possible_moves()
    return SimpleNamespace(model=model, cfg=cfg, device=device,
                           idx={m: i for i, m in enumerate(moves)}, moves=moves)


@torch.no_grad()
def move_probs(m, boards, self_elo, oppo_elo):
    """boards : historique de chess.Board (ancien -> courant)."""
    board = boards[-1]
    hist = deque((tokenize_board(b) for b in boards[-m.cfg.history:]), maxlen=m.cfg.history)
    tokens = get_historical_tokens(hist, m.cfg, 0.0, 0.0, 0.0, 0.0).unsqueeze(0).to(m.device)
    se = torch.tensor([self_elo], dtype=torch.long, device=m.device)
    oe = torch.tensor([oppo_elo], dtype=torch.long, device=m.device)
    logits, _, _ = m.model(tokens, se, oe)
    mask = get_legal_moves_mask(board, m.idx).to(m.device)
    probs = torch.softmax(logits[0].float().masked_fill(~mask, float("-inf")), dim=-1)
    out = {}
    for i in torch.nonzero(mask).flatten().tolist():
        u = m.moves[i]
        out[mirror_move(u) if board.turn == chess.BLACK else u] = float(probs[i])
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
```

Le résultat est une table `{coup UCI: probabilité}` sur tous les coups légaux (somme
égale à 1), de même forme que le `move_probs` de `maia2.inference.inference_each` :
`list(p.items())[0]` donne le coup le plus probable, `p[played.uci()]` la probabilité
du coup joué. Le miroir du plateau pour les noirs est géré par `tokenize_board`,
`get_legal_moves_mask` et `mirror_move` ; le code appelant n'a rien à faire.

Le modèle attend jusqu'à 8 positions d'historique (`history=8`). L'historique réel
(`node.parent`, `node.parent.parent`…) est disponible dans caissAI ; sans lui
(`boards[-1:]`), la position courante est répétée. Les deux variantes sont mesurées
plus bas.

### Elo : sens et décalage

- Entrées : `self_elo` = Elo du joueur **au trait**, `oppo_elo` = Elo de l'adversaire
  (même ordre que Maia2). Ce sont des entiers continus : `MAIA3Model.interpolate_elo`
  borne à `[0, 5000]` et interpole linéairement entre deux embeddings. Il n'y a pas de
  catégories de 100 points comme dans Maia2.
- Plage annoncée par la fiche : 600 à 2600, Elo Lichess blitz. Le README du dépôt ne
  la précise pas ; non revérifiée ici. Hors de cette plage le modèle accepte
  la valeur mais son comportement n'est pas documenté.
- `MAIA2_ELO_OFFSET` : le dépôt Maia-3 ne documente aucun décalage équivalent. Ne pas
  reporter le `200` sans raison. Sensibilité mesurée sur la partie de la section
  Mesures (143 positions, Maia-3 5M) : décalage `-200` / `0` / `+200` donne une
  probabilité moyenne du coup joué de `0,415` / `0,426` / `0,433`. L'effet est
  faible mais non nul ; le choix reste à arbitrer (voir Conclusion).

### Cache des poids

- Par défaut : cache Hugging Face, `~/.cache/huggingface/hub/models--UofTCSSLab--Maia3-5M/`.
  Le standard Hugging Face le déplace avec `HF_HOME` ou `HF_HUB_CACHE`.
- Explicite : `resolve_checkpoint_path(spec, cache_dir="<dossier>")` (équivalent de
  `save_root` de Maia2), ou `maia3-uci`/`maia3-cache --cache-dir`. C'est ce qu'il
  faut brancher sur un `CAISSAI_MAIA3_DIR`, par analogie avec `CAISSAI_MAIA2_DIR`.
  Essayé : `maia3-cache --model maia3-5m --cache-dir /tmp/cai001/hf-cache` crée
  `models--UofTCSSLab--Maia3-5M/snapshots/<révision>/maia3-5m.pt`.
- `local_files_only=True` (`--local-files-only`) évite tout accès réseau une fois
  les poids en cache ; `revision=` épingle la révision des poids.
  Sans cela, chaque appel à `hf_hub_download` contacte le Hub (avertissement
  « unauthenticated requests »).
- Le défaut de Maia2 (`./maia2_models`, relatif au répertoire courant) ne se retrouve
  pas ici : le défaut Hugging Face est global à l'utilisateur.

## Mesures

Machine : Linux, 8 cœurs logiques, CPU uniquement, torch `2.15.0+cpu` (4 threads par
défaut), même environnement jetable pour Maia2 (`maia2 0.9`, modèle `blitz`) et
Maia-3 (`maia3-5m`). Une seule exécution de chaque, mesures répétées 3 fois.

**Écart à la fiche** : `tests/` ne contient aucune partie, et aucune partie publique de
60 coups n'a pu être récupérée (le téléchargement depuis Lichess a été refusé par
l'environnement d'exécution). Les mesures portent sur une **partie synthétique légale**,
`game.pgn`, de 72 coups (143 demi-coups), générée avec une graine fixe : les blancs
jouent par tirage selon Maia2, les noirs par tirage selon Maia-3, Elo 1600 / 1450. Ce
n'est pas une partie humaine : les probabilités ne mesurent donc pas la qualité de
prédiction (le test de l'équipe de Toronto, 57,1 % contre 52,0 %, n'est pas
reproduit ici). Les temps, eux, sont indépendants de la provenance de la partie. À
refaire sur une partie réelle avant CAI-002 si la fiche l'exige.

### Temps (143 positions, une inférence par position, taille de lot 1)

| Variante | Essais successifs (s) | Par position (meilleur essai) |
| --- | --- | --- |
| Maia2 (`inference_each`, `prepare()` hors mesure) | 12,75 / 2,40 / 1,60 | 11,2 ms |
| Maia-3 5M, avec historique de 8 positions | 2,23 / 1,87 / 1,71 | 12,0 ms |
| Maia-3 5M, position courante seule | 1,60 / 1,61 / 1,62 | 11,2 ms |

Chargement du modèle : Maia2 1,31 s, Maia-3 5M 1,00 s (poids déjà en cache). Le premier
essai de Maia2 (12,75 s) est un démarrage à froid. À régime établi, les deux modèles sont
équivalents (environ 12 ms par position, soit moins de 2 s pour une partie de 70 coups),
négligeable devant Stockfish.

### Probabilité du coup joué, 10 positions

Maia2 reçoit `Elo + 200` (comme `get_move_probs`) ; Maia-3 reçoit l'Elo brut, avec
historique. Numéro = demi-coup de la partie.

| Demi-coup | Trait | Coup | Maia2 | Maia-3 5M |
| --- | --- | --- | --- | --- |
| 1 | B | Nf3 | 0,557 | 0,024 |
| 9 | B | g3 | 0,234 | 0,119 |
| 17 | B | Re1 | 0,071 | 0,060 |
| 31 | B | Na3 | 0,107 | 0,058 |
| 46 | N | Nxb3 | 0,291 | 0,233 |
| 61 | B | exf5 | 0,479 | 0,751 |
| 81 | B | Kf2 | 0,685 | 0,799 |
| 101 | B | Rb5 | 0,212 | 0,195 |
| 121 | B | Ke3 | 0,335 | 0,501 |
| 141 | B | Rc7+ | 0,927 | 0,967 |

Sur les 143 positions : probabilité moyenne du coup joué `0,430` (Maia2) contre `0,426`
(Maia-3, avec historique) et `0,383` (Maia-3, sans historique) ; coup joué égal au plus
probable dans `58,7 %`, `58,0 %` et `60,1 %` des cas. Écarts sans valeur statistique sur
une seule partie synthétique ; ils montrent seulement que les deux sorties sont de même
nature et de même ordre de grandeur. L'ouverture (demi-coup 1) diffère fortement : `Nf3`
vaut 0,557 pour Maia2 et 0,024 pour Maia-3.

## Licence

Faits, sans conclusion juridique.

- `maia2` : MIT (classifier PyPI et section License du README du paquet `maia2 0.9`).
  caissAI : MIT (`LICENSE`, `pyproject.toml`), dépôt public.
- `maia3` : GNU AGPL-3.0 (fichier `LICENSE` du dépôt, `license = { file = "LICENSE" }`
  dans `pyproject.toml`). Les poids Hugging Face `UofTCSSLab/Maia3-5M` n'ont aucune
  métadonnée de licence ; leur README indique « CC BY 4.0 (paper); see repo for
  code/weights license ». La licence des poids n'est donc pas établie ici.
- L'AGPL-3.0 est une licence de copyleft : toute œuvre dérivée distribuée doit l'être
  sous AGPL-3.0, avec le code source ; la section 13 l'étend à l'usage par un utilisateur
  distant à travers un réseau.
- Importer `maia3` comme bibliothèque (`import maia3`, adaptateur ci-dessus dans
  caissAI) : le code de caissAI et `maia3` s'exécutent dans un même processus et le
  paquet est une dépendance du projet. La question de savoir si caissAI devient ainsi
  une œuvre combinée soumise à l'AGPL, et donc si sa licence MIT reste applicable
  en l'état, est celle que Matthieu doit trancher (avis juridique si besoin).
  Un adaptateur qui copie du code de `uci.py` (comme le prototype) reprend du code AGPL.
- Appeler Maia-3 comme programme séparé (`maia3-uci`, via python-chess, comme
  Stockfish pour caissAI, GPL-3.0) : les deux programmes échangent du texte par un
  protocole standard et restent des exécutables distincts. caissAI ne contient alors
  pas de code `maia3`. Mais ce chemin ne donne pas la probabilité d'un coup (voir
  Constats). Un petit programme séparé, écrit pour exposer la distribution, serait lui
  un programme AGPL qui importe `maia3`, appelé par caissAI en sous-processus (JSON sur
  stdin/stdout) : c'est la même question de frontière, déplacée.
- Maintenir Maia2 (MIT) ne change rien à la licence de caissAI.
- Cette section expose des faits ; **la décision revient à Matthieu**.

## Conclusion

**Constat : prémisse confirmée**, avec une réserve de forme : l'intégration est possible et
la distribution complète est obtenue en Python, au même coût que Maia2 (environ 12 ms
par position sur CPU), mais **seulement par un adaptateur maison au-dessus des briques
de `maia3`**, car le paquet n'a pas de fonction d'inférence documentée. Le moteur UCI ne
convient pas : il ne rend aucune probabilité et se limite à 20 coups.

Option d'intégration recommandée (technique) : appel en processus (bibliothèque), via
un module d'adaptateur dans `src/caissAI/bin/`, avec la signature de `get_move_probs`
inchangée. C'est la seule option sans surcoût de sous-processus et sans duplication
de position. **Elle est conditionnée à la décision de licence de Matthieu** (section
Licence) : s'il refuse l'AGPL dans le dépôt MIT, les repli sont le sous-processus
dédié (même question de frontière, à instruire) ou rester sur Maia2.

CAI-002 devra :

- Ajouter la dépendance `maia3` épinglée sur un commit (git), retirer `maia2` et
  ses dépendances (`gdown`, etc.) de `pyproject.toml` et `uv.lock`, ajouter
  `huggingface-hub`, mettre à jour le bloc mypy `module = ["maia2", …]`.
- Remplacer `from maia2 import inference`, le type `MAIA2Model` et `model.from_pretrained`
  (`__main__.py`, `game_analyzer.py`, `annotator.py`) par l'adaptateur (chargement une
  fois ; `get_move_probs` doit fournir l'historique des 8 dernières positions
  depuis `node`).
- Traiter `MAIA2_ELO_OFFSET` : le supprimer ou le remplacer par une constante
  configurable à `0` par défaut, avec borne `[600, 2600]` à décider ; l'Elo des
  en-têtes PGN n'est pas un Elo Lichess blitz, ce que la fiche CAI-002 doit trancher.
- Remplacer `CAISSAI_MAIA2_DIR` / `resolve_maia2_dir` par un équivalent passé en
  `cache_dir` (voir « Cache des poids ») ; documenter `.env.example`, `README.md`,
  `CLAUDE.md` ; choisir `maia3-5m` ou `maia3-23m` (non mesuré ici).
- Adapter les tests (`tests/unit/test_game_analyzer.py` et `test_paths.py`), revérifier
  les seuils des NAGs : les probabilités changent (ex. ouverture), donc le critère
  `!!` (proba ≤ 20 %) est un changement de comportement d'annotation à signaler.
- Mesurer sur une partie humaine réelle de 60 coups ou plus.

## Spécification fonctionnelle

Sans objet : ce document est un diagnostic. L'exigence fonctionnelle visée par CAI-002
est que `get_move_probs` DOIT rendre, pour chaque position, le coup le plus probable et la
probabilité du coup joué au niveau des deux joueurs.

## Choix et contraintes

- Diagnostic seul : ni `pyproject.toml`, `uv.lock`, ni `src`, `tests` ne changent.
- Maia-3 est essayé dans des environnements jetables sous `/tmp`, jamais dans le dépôt.
- La licence AGPL-3.0 de `maia3` est un point de décision de Matthieu, pas du diagnostic.

## Critères d'acceptation

- [ ] La Spec contient ses constats, ses mesures, les faits de licence et une conclusion explicite.
- [ ] `git diff main -- pyproject.toml uv.lock src tests` est vide et `uv run inv lint` passe.
