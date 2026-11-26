"""Commentaires qualitatifs : un appel LLM par partie, vérifié et mis en cache.

Le LLM rédige à partir des dossiers de faits (`moments.build_moments`) ; il ne
calcule rien. Sa réponse est vérifiée localement avant d'être posée dans le PGN.
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.pgn
from beartype import beartype
from openai import OpenAI

_log = logging.getLogger(__name__)

# À incrémenter à chaque changement du prompt ou du schéma : invalide le cache.
PROMPT_VERSION: str = "1"
REFERENCE_FILE: str = "commentaire_reference.txt"
THEMES: tuple[str, ...] = (
    "pion-isole",
    "echange",
    "cases-faibles",
    "developpement",
    "calcul",
    "structure",
    "espace",
    "activite-des-pieces",
    "securite-du-roi",
    "initiative",
)
# Écart toléré entre une évaluation citée et celles du dossier (en pions).
EVAL_TOLERANCE: float = 0.05
IDEE_CLE_MAX_WORDS: int = 12
DEEP_COMMENTS_ENV: str = "CAISSAI_DEEP_COMMENTS"
DEEP_COMMENTS_DEFAULT: int = 4
BATCH_POLL_ENV: str = "CAISSAI_BATCH_POLL_SECONDS"
BATCH_POLL_DEFAULT: float = 60.0
_BATCH_DONE = {"completed", "failed", "expired", "cancelled"}

OUTPUT_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "commentaires_partie",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["moments", "resume"],
        "properties": {
            "moments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "ply",
                        "placement",
                        "commentaire",
                        "idee_cle",
                        "theme",
                    ],
                    "properties": {
                        "ply": {"type": "integer"},
                        "placement": {"type": "string", "enum": ["coup", "variante"]},
                        "commentaire": {"type": "string"},
                        "idee_cle": {"type": "string"},
                        "theme": {"type": "string", "enum": list(THEMES)},
                    },
                },
            },
            "resume": {"type": "string"},
        },
    },
}

_TO_FRENCH = str.maketrans("KQRBN", "RDTFC")
_TO_ENGLISH = str.maketrans("RDTFC", "KQRBN")
_SAN = r"(?:O-O(?:-O)?|[RDTFC]?[a-h]?[1-8]?x?[a-h][1-8](?:=[DTFC])?)[+#]?[!?]*(?!\w)"
_MARK = r"\d{1,3}(?:\.\.\.|…|\.)"
_SEQUENCE = re.compile(rf"{_MARK}{_SAN}(?:\s+(?:{_MARK})?{_SAN})*")
_EVALUATION = re.compile(r"(?<![\w.,])[+\-−–]\d+(?:[.,]\d+)?(?![\w.,]*\d)(?!\s*%)")

_STYLE = """Tu es un coach d'échecs. Tu commentes quelques moments clés d'une partie \
pour le joueur, que tu tutoies, en français.

Tout ce qui se calcule l'a été avant toi (Stockfish, python-chess, Maia) et figure \
dans le dossier de faits de chaque moment. Tu rédiges à partir de ces faits ; tu ne \
calcules rien.

Pour chaque moment, 80 à 180 mots, dans cet ordre :
1. une question directrice tirée de la structure (le coup aide-t-il ou gêne-t-il \
le plan ?), jamais un simple chiffre ;
2. ce que demandait la position ;
3. ce que le coup change, en pièces et en cases (pions, défenseurs, cases de retraite, \
cases trouées) ;
4. une ligne moteur citée, pour le coup joué ou l'alternative ;
5. une nuance honnête si l'écart est faible : « plus simple » ne veut pas dire \
« bien meilleur ».
Si le dossier contient un `coup_naturel`, explique ce qu'il néglige avant de dire ce \
que le meilleur coup fait de plus. Si le dossier contient une `punition`, dis si \
l'adversaire l'a jouée.

Règles strictes, vérifiées ensuite par un programme :
- Les coups du dossier sont en notation française (R roi, D dame, T tour, F fou, \
C cavalier). Écris-les tels quels.
- Une ligne citée est une suite de coups du dossier, écrite avec ses numéros collés \
au coup : `11.O-O Fe6 12.Tfd1`, `10…b6`. Ne cite jamais un coup, une ligne ou une \
évaluation absents du dossier. Une pièce ou une case seule (`le Fc5`, `d4`) est libre.
- Une évaluation citée est en pions, avec son signe et une virgule décimale \
(`+0,91`), et figure dans le dossier. Pas de pourcentage.
- `placement` : `coup` pose le commentaire sur le coup joué, `variante` sur la \
meilleure ligne du dossier (utile quand le coup joué n'est pas le meilleur).
- `idee_cle` : 12 mots au plus. `theme` : un seul, dans la liste fermée.
- `resume` : trois phrases (moments clés ; blancs : réussites, erreurs, à améliorer ; \
noirs : idem), sans coup ni évaluation chiffrés.
- Un objet de `moments` par moment du dossier, `ply` = son `demi_coup`.

Thèmes autorisés : {themes}.

Exemple de référence (commentaire de 10…b6, partie 316) :
{reference}
"""


@dataclass
class PendingGame:
    """Requête Batch d'une partie, en attente de sa réponse."""

    #: Identifiant de la requête dans le lot.
    custom_id: str
    #: Corps de la requête `responses.create`.
    body: dict[str, Any]
    #: Dossiers de faits des moments, pour la vérification.
    dossiers: list[dict[str, Any]]
    #: Fichier de cache de la partie, ou None.
    cache_path: Path | None


def _norm(san: str) -> str:
    """SAN anglais ou français sans suffixe, en notation française."""
    return san.translate(_TO_FRENCH).rstrip("+#!?")


# Clés du dossier portant des coups en SAN (voir `moments._dossier`).
_SAN_KEYS = {"coup", "coup_joue", "derniers_coups", "pv", "ligne_de_refutation"}


def _french(value: Any, key: str = "") -> Any:
    """Copie de `value` où les SAN du dossier passent en notation française."""
    if isinstance(value, dict):
        return {k: _french(item, k) for k, item in value.items()}
    if key in _SAN_KEYS and isinstance(value, str):
        return value.translate(_TO_FRENCH)
    if isinstance(value, list):
        return [_french(item, key) for item in value]
    return value


@beartype
def resolve_deep_comments(cli_value: int | None = None) -> int:
    """Résout le nombre de moments commentés : CLI > `CAISSAI_DEEP_COMMENTS` > 4.

    Args:
        cli_value: Valeur de `--deep-comments`, ou None.

    Returns:
        Nombre de moments (`0` : comportement historique).

    Raises:
        ValueError: Si la valeur n'est pas un entier positif ou nul.

    Examples:
        >>> resolve_deep_comments(2)
        2
        >>> resolve_deep_comments()
        4
    """
    raw = cli_value if cli_value is not None else os.environ.get(DEEP_COMMENTS_ENV)
    if raw is None or raw == "":
        return DEEP_COMMENTS_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{DEEP_COMMENTS_ENV} doit être un entier : {raw!r}") from None
    if value < 0:
        raise ValueError("Le nombre de moments commentés doit être >= 0.")
    return value


@beartype
def cache_key(pgn: str, k: int, color: bool | None) -> str:
    """Clé de cache d'une partie.

    Args:
        pgn: PGN de la partie avant analyse.
        k: Nombre de moments demandés.
        color: Camp analysé (`None` pour les deux) : il change les moments.

    Returns:
        Hachage hexadécimal du PGN, de K, du camp et de `PROMPT_VERSION`.
    """
    raw = json.dumps([pgn, k, color, PROMPT_VERSION], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@beartype
def load_cache(path: Path | None) -> dict[str, Any] | None:
    """Lit le résultat mis en cache d'une partie.

    Args:
        path: Fichier de cache, ou None.

    Returns:
        Le résultat, ou None s'il n'existe pas ou est illisible.
    """
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


@beartype
def save_cache(path: Path | None, result: dict[str, Any]) -> None:
    """Écrit le résultat vérifié d'une partie.

    Args:
        path: Fichier de cache, ou None (aucun cache).
        result: Résultat de `parse_response`.
    """
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


@beartype
def build_request(
    game: chess.pgn.Game,
    dossiers: list[dict[str, Any]],
    model: str,
    config_path: Path,
) -> dict[str, Any]:
    """Construit le corps de la requête : préfixe stable, puis la partie.

    Args:
        game: Partie nettoyée (sans commentaires).
        dossiers: Dossiers de faits des moments (peut être vide : résumé seul).
        model: Modèle OpenAI.
        config_path: Dossier `config/` (exemple de référence).

    Returns:
        Arguments de `client.responses.create`, aussi le corps d'une requête Batch.
    """
    reference = (config_path / REFERENCE_FILE).read_text(encoding="utf-8").strip()
    instructions = _STYLE.format(themes=", ".join(THEMES), reference=reference)
    entree = (
        f"Partie (notation anglaise, pour le résumé) :\n{game}\n\n"
        "Dossiers de faits des moments à commenter :\n"
        + json.dumps([_french(d) for d in dossiers], ensure_ascii=False, indent=1)
    )
    return {
        "model": model,
        "instructions": instructions,
        "input": entree,
        "text": {"format": OUTPUT_FORMAT},
    }


def _lines(dossier: dict[str, Any]) -> list[tuple[chess.Board | None, list[str]]]:
    """Lignes du dossier avec leur position de départ (`None` : non rejouable)."""
    board = chess.Board(dossier["fen"])
    stock = dossier["stockfish"]
    lines: list[tuple[chess.Board | None, list[str]]] = [
        (board, ligne["pv"]) for ligne in stock["lignes"]
    ]
    lines.append((board, stock["ligne_jouee"]["pv"]))
    natural = dossier["coup_naturel"]
    after = board.copy()
    after.push_san(natural["coup"])
    lines.append((board, [natural["coup"], *natural["ligne_de_refutation"]]))
    lines.append((after, natural["ligne_de_refutation"]))
    if dossier.get("menace_si_passe"):
        passed = board.copy()
        passed.push(chess.Move.null())
        lines.append((passed, dossier["menace_si_passe"]["pv"]))
    if dossier.get("punition"):
        after_played = board.copy()
        after_played.push_san(dossier["coup_joue"])
        lines.append((after_played, [dossier["punition"]["coup"]]))
    lines.append((None, dossier["derniers_coups"]))
    return lines


def _replay(start: chess.Board | None, before: list[str], cited: list[str]) -> bool:
    """Rejoue `before` puis `cited` (notation française) depuis `start`."""
    if start is None:
        return True
    board = start.copy()
    try:
        for san in before:
            board.push_san(san)
        for token in cited:
            board.push_san(token.translate(_TO_ENGLISH))
    except ValueError:
        return False
    return True


def _line_is_known(cited: list[str], dossier: dict[str, Any]) -> bool:
    """Vrai si `cited` est une suite de coups d'une ligne du dossier, légale."""
    for start, sans in _lines(dossier):
        normed = [_norm(san) for san in sans]
        for i in range(len(normed) - len(cited) + 1):
            if normed[i : i + len(cited)] == cited and _replay(start, sans[:i], cited):
                return True
    return False


def _pawn_evals(value: Any) -> list[float]:
    """Évaluations en pions (`cp` / 100) de tout le dossier."""
    found: list[float] = []
    if isinstance(value, dict):
        if isinstance(value.get("cp"), int):
            found.append(value["cp"] / 100)
        for item in value.values():
            found.extend(_pawn_evals(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_pawn_evals(item))
    return found


def _text_problem(text: str, dossiers: list[dict[str, Any]]) -> str | None:
    """Premier défaut d'un texte : ligne ou évaluation absente des dossiers."""
    for match in _SEQUENCE.finditer(text):
        cited = [
            token.rstrip("+#!?") for token in re.sub(_MARK, " ", match.group()).split()
        ]
        if not any(_line_is_known(cited, d) for d in dossiers):
            return f"ligne absente ou illégale : {match.group()}"
    known = [e for d in dossiers for e in _pawn_evals(d)]
    for match in _EVALUATION.finditer(text):
        value = float(
            match.group().replace(",", ".").replace("−", "-").replace("–", "-")
        )
        if not any(abs(abs(value) - abs(e)) <= EVAL_TOLERANCE for e in known):
            return f"évaluation absente du dossier : {match.group()}"
    return None


def _moment_problem(item: dict[str, Any], dossier: dict[str, Any]) -> str | None:
    """Premier défaut d'un commentaire de moment, ou None s'il est valide."""
    if item.get("theme") not in THEMES or item.get("placement") not in (
        "coup",
        "variante",
    ):
        return "thème ou placement hors liste"
    if len(str(item.get("idee_cle", "")).split()) > IDEE_CLE_MAX_WORDS:
        return "idée clé trop longue"
    text = f"{item.get('commentaire', '')} {item.get('idee_cle', '')}"
    if not item.get("commentaire"):
        return "commentaire vide"
    return _text_problem(text, [dossier])


@beartype
def parse_response(text: str, dossiers: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Vérifie la réponse du LLM contre les dossiers de faits.

    Un moment dont une ligne est illégale ou absente du dossier, ou dont une
    évaluation s'écarte de plus de `EVAL_TOLERANCE`, est rejeté (avertissement) :
    il garde son commentaire chiffré.

    Args:
        text: Texte JSON de la réponse.
        dossiers: Dossiers de faits envoyés au LLM.

    Returns:
        `{"moments": [...], "resume": str}` des seuls éléments valides, ou None
        si la réponse est inexploitable.
    """
    try:
        data = json.loads(text)
        items = list(data["moments"])
        resume = str(data["resume"])
    except (ValueError, KeyError, TypeError) as exc:
        _log.warning("Réponse LLM inexploitable : %s", exc)
        return None
    by_ply = {d["demi_coup"]: d for d in dossiers}
    accepted: list[dict[str, Any]] = []
    for item in items:
        dossier = by_ply.get(item.get("ply") if isinstance(item, dict) else None)
        if dossier is None:
            _log.warning("Commentaire rejeté : demi-coup absent des dossiers")
            continue
        problem = _moment_problem(item, dossier)
        if problem:
            _log.warning(
                "Commentaire du demi-coup %s rejeté : %s", item["ply"], problem
            )
            continue
        best = dossier["stockfish"]["lignes"][0]
        variante = best["pv"] if best["coup"] != dossier["coup_joue"] else None
        accepted.append(
            {
                **item,
                "placement": item["placement"] if variante else "coup",
                "variante": variante if item["placement"] == "variante" else None,
            }
        )
    problem = _text_problem(resume, dossiers)
    if problem:
        _log.warning("Résumé rejeté : %s", problem)
        resume = ""
    return {"moments": accepted, "resume": resume}


@beartype
def extract_output_text(body: dict[str, Any]) -> str:
    """Texte de sortie d'un corps de réponse de l'API Responses (mode Batch).

    Args:
        body: Corps `response.body` d'une ligne du fichier de résultats.

    Returns:
        Texte concaténé des parties `output_text`.
    """
    return "".join(
        part.get("text", "")
        for item in body.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


@beartype
def request_comments(client: OpenAI, body: dict[str, Any], game_label: str = "") -> str:
    """Envoie l'appel unique de la partie.

    Args:
        client: Client OpenAI.
        body: Corps de `build_request`.
        game_label: Libellé de la partie pour le journal.

    Returns:
        Texte JSON de la réponse.
    """
    _log.info("Appel API OpenAI (commentaires qualitatifs) : %s", game_label)
    return str(client.responses.create(**body).output_text)


@beartype
def apply_comments(game: chess.pgn.Game, result: dict[str, Any]) -> None:
    """Pose les commentaires vérifiés dans la partie annotée.

    Args:
        game: Partie annotée par `df_to_pgn`, modifiée en place.
        result: Résultat de `parse_response`.
    """
    nodes = {node.ply(): node for node in game.mainline()}
    for item in result["moments"]:
        node = nodes.get(item["ply"])
        if node is None or node.parent is None:
            continue
        if item["placement"] == "variante" and item.get("variante"):
            board = node.parent.board()
            moves = []
            for san in item["variante"]:
                moves.append(board.push_san(san))
            existing = next(
                (v for v in node.parent.variations if v.move == moves[0]), None
            )
            if existing is not None:
                existing.starting_comment = item["commentaire"]
            else:
                node.parent.add_line(moves, starting_comment=item["commentaire"])
        else:
            node.comment = item["commentaire"]
    if result["resume"]:
        game.end().comment += f" - {result['resume']}"


@beartype
def batch_line(custom_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Ligne du fichier d'entrée de l'API Batch pour une partie.

    Args:
        custom_id: Identifiant de la requête dans le lot.
        body: Corps de `build_request`.

    Returns:
        Requête au format Batch (`custom_id`, `method`, `url`, `body`).
    """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


@beartype
def run_batch(client: OpenAI, pendings: list[PendingGame]) -> dict[str, str]:
    """Envoie une requête par partie par l'API Batch et attend les réponses.

    Args:
        client: Client OpenAI.
        pendings: Requêtes en attente, une par partie ; leur `custom_id` est
            numéroté ici pour rester unique dans le lot.

    Returns:
        Texte de réponse par `custom_id` (les requêtes en erreur sont absentes).

    Raises:
        RuntimeError: Si le lot n'aboutit pas.
    """
    for number, pending in enumerate(pendings, 1):
        pending.custom_id = f"partie-{number}"
    payload = "\n".join(
        json.dumps(batch_line(p.custom_id, p.body), ensure_ascii=False)
        for p in pendings
    )
    sent = client.files.create(
        file=("caissai_batch.jsonl", payload.encode("utf-8")), purpose="batch"
    )
    batch = client.batches.create(
        input_file_id=sent.id, endpoint="/v1/responses", completion_window="24h"
    )
    _log.info("Lot OpenAI %s : %d partie(s)", batch.id, len(pendings))
    poll = float(os.environ.get(BATCH_POLL_ENV, BATCH_POLL_DEFAULT))
    while batch.status not in _BATCH_DONE:
        time.sleep(poll)
        batch = client.batches.retrieve(batch.id)
    if batch.status != "completed" or not batch.output_file_id:
        raise RuntimeError(f"Lot OpenAI {batch.id} : statut {batch.status}")
    texts: dict[str, str] = {}
    for line in client.files.content(batch.output_file_id).text.splitlines():
        record = json.loads(line)
        response = record.get("response") or {}
        if response.get("status_code") == 200:
            texts[record["custom_id"]] = extract_output_text(response["body"])
        else:
            _log.warning(
                "Requête %s en erreur : %s", record["custom_id"], record.get("error")
            )
    return texts


@beartype
def finish_pending(
    game: chess.pgn.Game, pending: PendingGame, text: str | None
) -> None:
    """Applique la réponse d'un lot à sa partie et la met en cache.

    Args:
        game: Partie annotée, modifiée en place.
        pending: Requête de la partie.
        text: Texte de réponse, ou None si la requête a échoué.
    """
    result = None if text is None else parse_response(text, pending.dossiers)
    if result is None:
        return
    save_cache(pending.cache_path, result)
    apply_comments(game, result)
