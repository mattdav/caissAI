"""Sélection des moments clés d'une partie et dossiers de faits (SPEC-003 § 1-2).

Aucun appel à un LLM : la sélection est locale et gratuite, Stockfish n'est
appelé que sur les positions des moments retenus.
"""

import json
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn
import pandas as pd
from beartype import beartype
from chess import Board, Move
from chess.engine import PovScore, SimpleEngine
from chess.pgn import Game

from .position_facts import (
    lost_support,
    pawn_structure,
    piece_facts,
    recapture_choice,
    square_control,
)
from .utils import get_es

# Nombre maximal de moments retenus par partie.
MOMENTS_K: int = 4
# Deux moments du même joueur doivent être séparés d'au moins ce nombre de coups.
WINDOW_MOVES: int = 3
# Poids des signaux (valeurs de départ, à calibrer sur de vraies parties).
# Poids de l'erreur : maximal jusqu'à |ΔES| = ERROR_PLATEAU (`?!`), décroissant au-delà.
ERROR_WEIGHT: float = 1.0
ERROR_PLATEAU: float = 0.10
RECAPTURE_WEIGHT: float = 1.25
NATURAL_WEIGHT: float = 0.8  # + perte d'ES du coup naturel
TURNING_WEIGHT: float = 0.5  # + |ΔES| du tournant
# Bonus : phase d'ouverture (coups 1 à OPENING_MOVES) et différence de structure ou
# de pièces entre le coup joué et le meilleur coup.
OPENING_MOVES: int = 15
OPENING_BONUS: float = 0.5
STRUCTURE_BONUS: float = 0.5
# Au plus CANDIDATES_FACTOR × K positions sont analysées par Stockfish.
CANDIDATES_FACTOR: int = 3
# Erreur tactique : écart matériel (P=1, C=F=3, T=5, D=9) entre la meilleure ligne
# et la ligne jouée, ou mat dans l'une des deux.
TACTICAL_GAP: int = 2
# Profondeur maximale (en demi-coups) de la résolution des prises en suspens.
SETTLE_PLIES: int = 8
_PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}
# Partie pliée : ES du meilleur coup, vue par le camp au trait, hors de ]LOSS ; WIN[.
DECIDED_WIN: float = 0.80
DECIDED_LOSS: float = 0.20
# Perte d'ES minimale du coup naturel pour compter comme signal.
NATURAL_LOSS: float = 0.10
# Analyse Stockfish des positions retenues.
DOSSIER_DEPTH: int = 20
MULTIPV: int = 3
PV_PLIES: int = 8
LAST_MOVES: int = 4
_ENGINE_OPTIONS: dict[str, int] = {"Threads": 2, "Hash": 256}
_MATE_SCORE: int = 100000
_ERROR_NAGS = (
    chess.pgn.NAG_DUBIOUS_MOVE,
    chess.pgn.NAG_MISTAKE,
    chess.pgn.NAG_BLUNDER,
)


def _es(score: PovScore, elo: int, color: bool) -> float:
    """Espérance de gain du point de vue de `color`."""
    es = get_es(elo, score.white().score(mate_score=_MATE_SCORE))
    return es if color else 1 - es


def _eval(score: PovScore, elo: int, color: bool) -> dict[str, float | int]:
    """Évaluation : centipions côté blancs et espérance de gain du joueur."""
    return {
        "cp": score.white().score(mate_score=_MATE_SCORE),
        "es": round(_es(score, elo, color), 3),
    }


def _san_line(board: Board, moves: list[Move], limit: int = PV_PLIES) -> list[str]:
    """Convertit une ligne en SAN, depuis `board`, sur `limit` demi-coups."""
    board = board.copy()
    line = []
    for move in moves[:limit]:
        line.append(board.san(move))
        board.push(move)
    return line


def _move_number(ply: int) -> int:
    """Numéro de coup (1-basé) du demi-coup `ply` (1 = premier coup blanc)."""
    return (ply + 1) // 2


def _is_opening(ply: int) -> bool:
    """Vrai si le demi-coup `ply` est dans la phase d'ouverture."""
    return _move_number(ply) <= OPENING_MOVES


def _label(ply: int) -> str:
    """Étiquette de coup : `9...` pour les noirs, `10.` pour les blancs."""
    return f"{_move_number(ply)}{'.' if ply % 2 else '...'}"


def _balance(board: Board) -> int:
    """Bilan matériel du point de vue du camp au trait."""
    total = 0
    for piece_type, value in _PIECE_VALUES.items():
        total += value * (
            len(board.pieces(piece_type, board.turn))
            - len(board.pieces(piece_type, not board.turn))
        )
    return total


def _capture_order(board: Board, move: Move) -> int:
    """Clé de tri des prises : victime la plus chère, attaquant le moins cher."""
    victim = board.piece_type_at(move.to_square) or chess.PAWN
    attacker = board.piece_type_at(move.from_square) or chess.PAWN
    return 10 * _PIECE_VALUES.get(attacker, 0) - 100 * _PIECE_VALUES.get(victim, 1)


def _settle(board: Board, alpha: int, beta: int, depth: int = SETTLE_PLIES) -> int:
    """Bilan matériel du camp au trait une fois les prises en suspens résolues."""
    stand_pat = _balance(board)
    if depth == 0 or stand_pat >= beta:
        return stand_pat
    alpha = max(alpha, stand_pat)
    captures = sorted(
        board.generate_legal_captures(), key=lambda m: _capture_order(board, m)
    )
    for move in captures:
        board.push(move)
        value = -_settle(board, -beta, -alpha, depth - 1)
        board.pop()
        if value >= beta:
            return value
        alpha = max(alpha, value)
    return alpha


def _line_balance(board: Board, moves: list[Move], color: bool) -> int:
    """Bilan matériel de `color` au bout de `moves`, prises en suspens résolues."""
    board = board.copy()
    for move in moves:
        board.push(move)
    value = _settle(board, -1000, 1000)
    return value if board.turn == color else -value


@beartype
def is_tactical(
    board: Board,
    best_line: list[Move],
    best_score: PovScore,
    played_line: list[Move],
    played_score: PovScore,
) -> bool:
    """Indique si l'écart entre le meilleur coup et le coup joué est tactique.

    L'écart est tactique quand il se lit en matériel : bilan matériel (prises en
    suspens résolues) des deux variantes complètes de Stockfish, comparées à
    longueur égale, celle de la plus courte ; l'avantage de la meilleure ligne sur
    la ligne jouée atteint `TACTICAL_GAP` points, ou mat dans l'une des deux.
    Comparer à longueur égale évite de créditer la meilleure ligne des gains
    qu'elle réalise au-delà de l'horizon de la ligne jouée.

    Args:
        board: Position avant le coup joué.
        best_line: Meilleure variante complète, depuis `board`.
        best_score: Évaluation de la meilleure variante.
        played_line: Coup joué suivi de la variante complète de Stockfish.
        played_score: Évaluation de la ligne jouée.

    Returns:
        `True` si l'écart est tactique.

    Examples:
        >>> start = Board()
        >>> e4, nf3 = Move.from_uci("e2e4"), Move.from_uci("g1f3")
        >>> score = PovScore(chess.engine.Cp(20), chess.WHITE)
        >>> is_tactical(start, [e4], score, [nf3], score)
        False
    """
    if best_score.is_mate() or played_score.is_mate():
        return True
    color = board.turn
    length = min(len(best_line), len(played_line))
    gap = _line_balance(board, best_line[:length], color) - _line_balance(
        board, played_line[:length], color
    )
    return gap >= TACTICAL_GAP


@beartype
def is_decided(score: PovScore, elo: int, color: bool) -> bool:
    """Indique si la partie est pliée : ES du meilleur coup hors de ]0,20 ; 0,80[.

    Args:
        score: Évaluation du meilleur coup (position avant le coup joué).
        elo: ELO du joueur.
        color: Camp au trait (`True` = blancs).

    Returns:
        `True` si l'ES du meilleur coup, vue par `color`, est au moins
        `DECIDED_WIN` ou au plus `DECIDED_LOSS`.

    Examples:
        >>> is_decided(PovScore(chess.engine.Cp(20), chess.WHITE), 1500, True)
        False
        >>> is_decided(PovScore(chess.engine.Cp(2000), chess.WHITE), 1500, True)
        True
    """
    es = _es(score, elo, color)
    return es >= DECIDED_WIN or es <= DECIDED_LOSS


@beartype
def player_color(game: Game, player: str | None) -> bool | None:
    """Camp du joueur analysé dans la partie.

    Args:
        game: La partie.
        player: Nom filtré par `--player` (correspondance partielle), ou None.

    Returns:
        `True` (blancs) ou `False` (noirs) si le nom ne correspond qu'à un camp,
        `None` (les deux camps) sinon.

    Examples:
        >>> g = Game(headers={"White": "Dupont, A", "Black": "Martin, B"})
        >>> player_color(g, "dupont"), player_color(g, "martin"), player_color(g, None)
        (True, False, None)
    """
    if player is None:
        return None
    needle = player.lower()
    is_white = needle in game.headers.get("White", "").lower()
    is_black = needle in game.headers.get("Black", "").lower()
    if is_white == is_black:
        return None
    return is_white


@beartype
def score_plies(df: pd.DataFrame, color: bool | None = None) -> list[dict[str, Any]]:
    """Attribue un score aux demi-coups d'une partie analysée.

    Signaux : erreur (NAG `?!` `?` `??`, poids maximal pour `?!` puis décroissant),
    choix de reprise entre pion et pièce, coup naturel qui perd au moins
    `NATURAL_LOSS` d'ES, tournant (plus forte variation d'ES). Les scores d'un même
    demi-coup s'additionnent, avec un bonus d'ouverture. Les erreurs tactiques et
    les positions pliées sont écartées, quel que soit le signal.

    Args:
        df: DataFrame de `process_game` après `get_nags` (avec `nag_dict`).
        color: Camp à examiner (`True` = blancs), `None` pour les deux.

    Returns:
        Une entrée `{"ply", "color", "row", "score", "signals"}` par demi-coup
        portant au moins un signal (`row` : position dans `df`).
    """
    rows = list(enumerate(df.to_dict("records")))
    mover_rows = [
        (i, r) for i, r in rows if color is None or bool(r["node"].ply() % 2) == color
    ]
    turning = max(
        mover_rows,
        key=lambda ir: abs(ir[1]["nag_dict"]["es_after_before_delta"]),
        default=None,
    )
    candidates = []
    for i, row in mover_rows:
        node = row["node"]
        mover = bool(node.ply() % 2)
        elo = row["player_elo"]
        delta = row["nag_dict"]["es_after_before_delta"]
        signals: dict[str, float] = {}
        nags = row["nag_dict"]["nag"]
        if nags and nags[0] in _ERROR_NAGS:
            signals["erreur"] = ERROR_WEIGHT * min(
                1.0, ERROR_PLATEAU / max(abs(delta), 1e-9)
            )
        grandparent = node.parent.parent if node.parent is not None else None
        if grandparent is not None and node.parent.move is not None:
            choice = recapture_choice(grandparent.board().fen(), node.parent.move.uci())
            if choice is not None and choice["pion_et_piece"]:
                signals["choix_de_reprise"] = RECAPTURE_WEIGHT
        best = row["best_move_variation"][0]
        loss = _es(row["before_move_score"], elo, mover) - _es(
            row["likeliest_move_eval.score"], elo, mover
        )
        if row["likeliest_move"] != best and loss >= NATURAL_LOSS:
            signals["coup_naturel"] = NATURAL_WEIGHT + loss
        if turning is not None and i == turning[0] and delta != 0:
            signals["tournant"] = TURNING_WEIGHT + abs(delta)
        if not signals or is_decided(row["before_move_score"], elo, mover):
            continue
        assert node.parent is not None
        if is_tactical(
            node.parent.board(),
            list(row["best_move_variation"]),
            row["before_move_score"],
            [row["played_move"], *row["played_move_eval.variation"]],
            row["played_move_eval.score"],
        ):
            continue
        bonus = ["ouverture"] if _is_opening(node.ply()) else []
        candidates.append(
            {
                "ply": node.ply(),
                "color": mover,
                "row": i,
                "score": sum(signals.values()) + OPENING_BONUS * len(bonus),
                "signals": sorted(signals),
                "bonus": bonus,
            }
        )
    return candidates


@beartype
def select_moments(
    candidates: list[dict[str, Any]], k: int = MOMENTS_K
) -> list[dict[str, Any]]:
    """Retient au plus `k` moments, au plus un par fenêtre de `WINDOW_MOVES` coups
    du même joueur, hors de la phase d'ouverture (coups 1 à `OPENING_MOVES`).

    Args:
        candidates: Demi-coups scorés (`ply`, `color`, `score`).
        k: Nombre maximal de moments.

    Returns:
        Les moments retenus, dans l'ordre de la partie.

    Examples:
        >>> cands = [
        ...     {"ply": 60, "color": False, "score": 1.5},
        ...     {"ply": 64, "color": False, "score": 1.4},
        ...     {"ply": 66, "color": False, "score": 1.2},
        ... ]
        >>> [c["ply"] for c in select_moments(cands, 4)]
        [60, 66]
    """
    chosen: list[dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: (-c["score"], c["ply"])):
        if len(chosen) >= k:
            break
        if any(
            c["color"] == cand["color"]
            and not (_is_opening(c["ply"]) and _is_opening(cand["ply"]))
            and abs(_move_number(c["ply"]) - _move_number(cand["ply"])) < WINDOW_MOVES
            for c in chosen
        ):
            continue
        chosen.append(cand)
    return sorted(chosen, key=lambda c: c["ply"])


def _recent_moves(node: chess.pgn.GameNode) -> list[str]:
    """SAN des `LAST_MOVES` demi-coups précédant le coup de `node`."""
    moves: list[str] = []
    current = node.parent
    while current is not None and current.parent is not None:
        if len(moves) >= LAST_MOVES:
            break
        assert current.move is not None
        moves.insert(0, current.parent.board().san(current.move))
        current = current.parent
    return moves


def _key_squares(structure: dict[str, dict[str, Any]]) -> list[str]:
    """Cases devant les pions isolés ou passés des deux camps."""
    squares: list[str] = []
    for name, step in (("blancs", 1), ("noirs", -1)):
        pawns: list[str] = structure[name]["isoles"] + structure[name]["passes"]
        for pawn in pawns:
            ahead = chess.parse_square(pawn) + 8 * step
            if 0 <= ahead <= 63 and chess.square_name(ahead) not in squares:
                squares.append(chess.square_name(ahead))
    return squares


def _control(fens: dict[str, str], squares: list[str], color: bool) -> dict[str, Any]:
    """Attaquants et défenseurs de chaque case pour chaque position de `fens`."""
    return {
        square: {
            label: square_control(fen, square, color) for label, fen in fens.items()
        }
        for square in squares
    }


def _dossier(
    df: pd.DataFrame,
    moment: dict[str, Any],
    engine: SimpleEngine,
) -> dict[str, Any]:
    """Construit le dossier de faits d'un moment (Stockfish MultiPV et menace)."""
    row = df.iloc[moment["row"]]
    node = row["node"]
    assert node.parent is not None
    board = node.parent.board()
    color: bool = board.turn
    elo: int = int(row["player_elo"])
    played: Move = row["played_move"]
    infos = engine.analyse(
        board, chess.engine.Limit(depth=DOSSIER_DEPTH), multipv=MULTIPV
    )
    lines = [
        {
            "coup": board.san(info["pv"][0]),
            "eval": _eval(info["score"], elo, color),
            "pv": _san_line(board, info["pv"]),
            "profondeur": info.get("depth"),
        }
        for info in infos
    ]
    best: Move = infos[0]["pv"][0]
    after_played = board.copy()
    after_played.push(played)
    after_best = board.copy()
    after_best.push(best)
    end_pv = board.copy()
    for move in infos[0]["pv"][:PV_PLIES]:
        end_pv.push(move)
    fens = {
        "coup_joue": after_played.fen(),
        "meilleur_coup": after_best.fen(),
        "fin_pv": end_pv.fen(),
    }
    structures = {
        "avant": pawn_structure(board.fen()),
        "apres_coup_joue": pawn_structure(fens["coup_joue"]),
        "apres_meilleur_coup": pawn_structure(fens["meilleur_coup"]),
    }
    likeliest: Move = row["likeliest_move"]
    after_likeliest = board.copy()
    after_likeliest.push(likeliest)
    threat = None
    if not board.is_check():
        passed = board.copy()
        passed.push(Move.null())
        info = engine.analyse(passed, chess.engine.Limit(depth=DOSSIER_DEPTH))
        threat = {
            "coup": passed.san(info["pv"][0]),
            "eval": _eval(info["score"], elo, color),
            "pv": _san_line(passed, info["pv"]),
        }
    mover_rows = df[df["node"].map(lambda n: bool(n.ply() % 2) == color)]
    drops = sorted(
        (
            (r["nag_dict"]["es_after_before_delta"], r["node"].ply())
            for _, r in mover_rows.iterrows()
        )
    )[:2]
    dossier = {
        "coup": _label(moment["ply"]),
        "demi_coup": moment["ply"],
        "signaux": moment["signals"],
        "score": round(moment["score"], 3),
        "fen": board.fen(),
        "trait": "blancs" if color else "noirs",
        "coup_joue": board.san(played),
        "derniers_coups": _recent_moves(node),
        "stockfish": {
            "lignes": lines,
            "ligne_jouee": {
                "eval": _eval(row["played_move_eval.score"], elo, color),
                "pv": _san_line(board, [played, *row["played_move_eval.variation"]]),
            },
        },
        "maia": {"proba_coup_joue": round(float(row["played_move_proba"]), 3)},
        "coup_naturel": {
            "coup": board.san(likeliest),
            "proba": round(float(row["likeliest_move_proba"]), 3),
            "joue": likeliest == played,
            "eval": _eval(row["likeliest_move_eval.score"], elo, color),
            "ligne_de_refutation": _san_line(
                after_likeliest, list(row["likeliest_move_eval.variation"])
            ),
        },
        "structure": structures,
        "cases_cles": _control(
            fens, _key_squares(structures["apres_coup_joue"]), color
        ),
        "pieces": {
            "coup_joue": lost_support(board.fen(), played.uci()),
            "meilleur_coup": lost_support(board.fen(), best.uci()),
            "apres_coup_joue": piece_facts(fens["coup_joue"], color),
        },
        "menace_si_passe": threat,
        "courbe_es": {
            "avant": round(_es(row["before_move_score"], elo, color), 3),
            "apres": round(_es(row["played_move_eval.score"], elo, color), 3),
            "plus_fortes_chutes": [_label(ply) for _, ply in drops],
        },
    }
    if "erreur" in moment["signals"]:
        dossier["punition"] = _punishment(row, board, elo, color)
    return dossier


def _punishment(
    row: pd.Series, board: Board, elo: int, color: bool
) -> dict[str, Any] | None:
    """Meilleure réponse adverse au coup joué, et si l'adversaire l'a jouée."""
    variation = list(row["played_move_eval.variation"])
    if not variation:
        return None
    after_played = board.copy()
    after_played.push(row["played_move"])
    node = row["node"]
    reply = node.variations[0].move if node.variations else None
    return {
        "coup": after_played.san(variation[0]),
        "eval": _eval(row["played_move_eval.score"], elo, color),
        "jouee": None if reply is None else reply == variation[0],
    }


def _structure_differs(dossier: dict[str, Any]) -> bool:
    """Vrai si la structure ou les pièces diffèrent entre coup joué et meilleur coup."""
    structure = dossier["structure"]
    for camp in ("blancs", "noirs"):
        played = structure["apres_coup_joue"][camp]
        best = structure["apres_meilleur_coup"][camp]
        for key in ("isoles", "doubles", "arrieres", "cases_trouees"):
            if set(played[key]) != set(best[key]):
                return True
    pieces = dossier["pieces"]
    return bool(pieces["coup_joue"] != pieces["meilleur_coup"])


@beartype
def build_moments(
    df: pd.DataFrame,
    engine_path: Path,
    color: bool | None = None,
    k: int = MOMENTS_K,
) -> list[dict[str, Any]]:
    """Sélectionne les moments clés et construit leurs dossiers de faits.

    Args:
        df: DataFrame de `process_game` après `get_nags`.
        engine_path: Chemin de l'exécutable Stockfish.
        color: Camp du joueur analysé (`True` = blancs), `None` pour les deux.
        k: Nombre maximal de moments (`0` : aucun, Stockfish n'est pas lancé).

    Returns:
        Un dossier de faits par moment retenu, dans l'ordre de la partie.
        Stockfish analyse au plus `CANDIDATES_FACTOR` × `k` positions ; le bonus
        de structure est ajouté après analyse, puis les `k` meilleures sont gardées.
    """
    candidates = select_moments(score_plies(df, color), CANDIDATES_FACTOR * k)
    if not candidates or k == 0:
        return []
    engine = SimpleEngine.popen_uci(str(engine_path))
    try:
        engine.configure(_ENGINE_OPTIONS)
        dossiers = {c["ply"]: _dossier(df, c, engine) for c in candidates}
    finally:
        engine.quit()
    for cand in candidates:
        if _structure_differs(dossiers[cand["ply"]]):
            cand["score"] += STRUCTURE_BONUS
            cand["bonus"] = [*cand["bonus"], "structure"]
        dossiers[cand["ply"]]["score"] = round(cand["score"], 3)
        dossiers[cand["ply"]]["bonus"] = cand["bonus"]
    return [dossiers[m["ply"]] for m in select_moments(candidates, k)]


@beartype
def write_moments_json(path: Path, games: list[dict[str, Any]]) -> None:
    """Écrit les dossiers de moments des parties traitées.

    Args:
        path: Fichier JSON de sortie.
        games: Une entrée par partie (identité de la partie et `moments`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"parties": games}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
