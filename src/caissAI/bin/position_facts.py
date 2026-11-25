"""Faits de position calculés localement (fonctions pures, FEN et coups en entrée).

Chaque fonction reçoit une FEN (et éventuellement un coup en UCI) et renvoie
un dictionnaire sérialisable en JSON. Aucune ne fait appel à un moteur.
"""

from typing import Any

import chess
from beartype import beartype
from chess import BLACK, WHITE, Board, Color, Move

_COLOR_NAMES: dict[Color, str] = {WHITE: "blancs", BLACK: "noirs"}
# Rangs relatifs (1 = rang de départ des pièces) où une case trouée est signalée.
_HOLE_RANKS = (2, 3, 4)
# Écart minimal de cases contrôlées chez l'adversaire pour parler d'avantage d'espace.
SPACE_MARGIN: int = 2
# Une pièce "retraite" sur ces types de case : les pièces légères et lourdes.
_RETREATING_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def _relative_rank(square: chess.Square, color: Color) -> int:
    """Rang de `square` du point de vue de `color` (0 = rang de départ)."""
    rank = chess.square_rank(square)
    return rank if color == WHITE else 7 - rank


def _is_ahead(square: chess.Square, reference: chess.Square, color: Color) -> bool:
    """Vrai si `square` est strictement devant `reference` pour `color`."""
    return _relative_rank(square, color) > _relative_rank(reference, color)


def _pawns(board: Board, color: Color) -> list[chess.Square]:
    """Cases des pions de `color`."""
    return list(board.pieces(chess.PAWN, color))


def _adjacent_files(file: int) -> list[int]:
    """Colonnes voisines de `file`, bornées à l'échiquier."""
    return [f for f in (file - 1, file + 1) if 0 <= f <= 7]


def _is_isolated(square: chess.Square, own: list[chess.Square]) -> bool:
    """Aucun pion allié sur les colonnes voisines."""
    adjacent = _adjacent_files(chess.square_file(square))
    return not any(chess.square_file(p) in adjacent for p in own)


def _is_passed(square: chess.Square, color: Color, enemy: list[chess.Square]) -> bool:
    """Aucun pion adverse devant sur la colonne ou les colonnes voisines."""
    files = [chess.square_file(square), *_adjacent_files(chess.square_file(square))]
    return not any(
        chess.square_file(p) in files and _is_ahead(p, square, color) for p in enemy
    )


def _is_backward(
    square: chess.Square,
    color: Color,
    own: list[chess.Square],
    enemy: list[chess.Square],
) -> bool:
    """Pion non isolé dont les voisins sont tous plus avancés, et dont la case
    d'avance est contrôlée par un pion adverse."""
    adjacent = _adjacent_files(chess.square_file(square))
    neighbours = [p for p in own if chess.square_file(p) in adjacent]
    if not neighbours or any(not _is_ahead(p, square, color) for p in neighbours):
        return False
    stop = square + (8 if color == WHITE else -8)
    if not 0 <= stop <= 63:
        return False
    stop_file = chess.square_file(stop)
    return any(
        abs(chess.square_file(p) - stop_file) == 1
        and _relative_rank(p, not color) == _relative_rank(stop, not color) - 1
        for p in enemy
    )


def _holes(color: Color, own: list[chess.Square]) -> list[str]:
    """Cases du camp qu'aucun pion allié ne pourra plus jamais attaquer."""
    holes = []
    for square in chess.SQUARES:
        if _relative_rank(square, color) not in _HOLE_RANKS:
            continue
        adjacent = _adjacent_files(chess.square_file(square))
        guards = [
            p
            for p in own
            if chess.square_file(p) in adjacent and _is_ahead(square, p, color)
        ]
        if not guards and square not in own:
            holes.append(chess.square_name(square))
    return holes


@beartype
def pawn_structure(fen: str) -> dict[str, dict[str, Any]]:
    """Décrit la structure de pions de chaque camp.

    Args:
        fen: Position à analyser.

    Returns:
        Un dictionnaire par camp (`blancs`, `noirs`) : pions `isoles`, `doubles`,
        `arrieres`, `passes` (noms de cases), nombre d'`ilots`, colonnes
        `colonnes_ouvertes` (sans pion) et `colonnes_semi_ouvertes` (sans pion du
        camp, avec un pion adverse), et `cases_trouees` du camp.

    Examples:
        >>> fen = "r1bq1rk1/pp3ppp/2n5/2bp4/8/2N1PN2/PP2BPPP/R2QK2R b KQ - 0 9"
        >>> pawn_structure(fen)["noirs"]["isoles"]
        ['d5']
        >>> pawn_structure(fen)["noirs"]["colonnes_semi_ouvertes"]
        ['e']
    """
    board = Board(fen)
    pawns = {color: _pawns(board, color) for color in (WHITE, BLACK)}
    result: dict[str, dict[str, Any]] = {}
    for color in (WHITE, BLACK):
        own, enemy = pawns[color], pawns[not color]
        own_files = {chess.square_file(p) for p in own}
        enemy_files = {chess.square_file(p) for p in enemy}
        files_with_two = [
            f for f in own_files if sum(chess.square_file(p) == f for p in own) > 1
        ]
        islands = sum(1 for f in sorted(own_files) if f - 1 not in own_files)
        result[_COLOR_NAMES[color]] = {
            "isoles": [chess.square_name(p) for p in own if _is_isolated(p, own)],
            "doubles": [
                chess.square_name(p)
                for p in own
                if chess.square_file(p) in files_with_two
            ],
            "arrieres": [
                chess.square_name(p) for p in own if _is_backward(p, color, own, enemy)
            ],
            "passes": [
                chess.square_name(p) for p in own if _is_passed(p, color, enemy)
            ],
            "ilots": islands,
            "colonnes_ouvertes": [
                chess.FILE_NAMES[f]
                for f in range(8)
                if f not in own_files and f not in enemy_files
            ],
            "colonnes_semi_ouvertes": [
                chess.FILE_NAMES[f]
                for f in range(8)
                if f not in own_files and f in enemy_files
            ],
            "cases_trouees": _holes(color, own),
        }
    return result


@beartype
def square_control(fen: str, square: str, color: bool) -> dict[str, int]:
    """Compte les attaquants et les défenseurs d'une case.

    Args:
        fen: Position à analyser.
        square: Case en notation algébrique (`"d4"`).
        color: Camp de référence (`True` = blancs) : ses pièces sont les
            défenseurs, celles de l'adversaire les attaquants.

    Returns:
        `{"attaquants": n, "defenseurs": m}`.

    Examples:
        >>> square_control(chess.STARTING_FEN, "e3", True)
        {'attaquants': 0, 'defenseurs': 2}
    """
    board = Board(fen)
    target = chess.parse_square(square)
    return {
        "attaquants": len(board.attackers(not color, target)),
        "defenseurs": len(board.attackers(color, target)),
    }


@beartype
def recapture_choice(fen_before: str, capture_uci: str) -> dict[str, Any] | None:
    """Détecte un choix de reprise après une prise adverse.

    Args:
        fen_before: Position avant la prise de l'adversaire.
        capture_uci: La prise de l'adversaire (UCI).

    Returns:
        `None` si le coup n'est pas une prise ou s'il y a moins de deux reprises
        légales ; sinon `{"case", "reprises" (SAN), "pion_et_piece"}`, où
        `pion_et_piece` vaut `True` si une reprise de pion et une reprise de
        pièce existent.

    Examples:
        >>> start = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/2PP4/2N1PN2/PP2BPPP/R1BQK2R w - - 0 8"
        >>> recapture_choice(start, "c4d5") is not None
        True
    """
    board = Board(fen_before)
    move = Move.from_uci(capture_uci)
    if not board.is_capture(move):
        return None
    board.push(move)
    recaptures = [m for m in board.legal_moves if m.to_square == move.to_square]
    if len(recaptures) < 2:
        return None
    kinds = {board.piece_type_at(m.from_square) for m in recaptures}
    return {
        "case": chess.square_name(move.to_square),
        "reprises": [board.san(m) for m in recaptures],
        "pion_et_piece": chess.PAWN in kinds and len(kinds) > 1,
    }


def _defenders(board: Board, square: chess.Square, color: Color) -> set[int]:
    """Cases des pièces de `color` qui défendent `square`."""
    return set(board.attackers(color, square))


def _retreats(board: Board, square: chess.Square, color: Color) -> set[int]:
    """Cases où la pièce en `square` peut aller (vide ou prise adverse)."""
    own = board.occupied_co[color]
    return set(board.attacks(square)) - set(chess.scan_forward(own))


@beartype
def lost_support(fen: str, move_uci: str) -> dict[str, list[dict[str, str]]]:
    """Liste les pièces qui perdent un défenseur ou une case de retraite.

    Seules les pièces du camp qui joue sont examinées.

    Args:
        fen: Position avant le coup.
        move_uci: Coup joué (UCI).

    Returns:
        `defenseurs_perdus` : `{"piece", "case", "defenseur"}` (le défenseur
        est désigné par sa case avant le coup) ;
        `retraites_perdues` : `{"piece", "case", "retraite"}`.

    Examples:
        >>> fen = "r1bq1rk1/pp3ppp/2n5/2bp4/8/1QN1PN2/PP2BPPP/R3K2R b KQ - 1 10"
        >>> {"piece": "cavalier", "case": "c6", "defenseur": "b7"} in lost_support(
        ...     fen, "b7b6"
        ... )["defenseurs_perdus"]
        True
    """
    board = Board(fen)
    move = Move.from_uci(move_uci)
    color = board.turn
    after = board.copy()
    after.push(move)
    names = {
        chess.KNIGHT: "cavalier",
        chess.BISHOP: "fou",
        chess.ROOK: "tour",
        chess.QUEEN: "dame",
        chess.PAWN: "pion",
    }
    lost_defenders: list[dict[str, str]] = []
    lost_retreats: list[dict[str, str]] = []
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        new_square = move.to_square if square == move.from_square else square
        after_defenders = _defenders(after, new_square, color)
        for defender in _defenders(board, square, color):
            if defender == square:
                continue
            moved_to = move.to_square if defender == move.from_square else defender
            if moved_to not in after_defenders:
                lost_defenders.append(
                    {
                        "piece": names[piece.piece_type],
                        "case": chess.square_name(new_square),
                        "defenseur": chess.square_name(defender),
                    }
                )
        if piece.piece_type in _RETREATING_PIECES and square != move.from_square:
            lost = _retreats(board, square, color) - _retreats(after, square, color)
            lost_retreats.extend(
                {
                    "piece": names[piece.piece_type],
                    "case": chess.square_name(square),
                    "retraite": chess.square_name(r),
                }
                for r in sorted(lost)
            )
    return {
        "defenseurs_perdus": lost_defenders,
        "retraites_perdues": lost_retreats,
    }


@beartype
def piece_facts(fen: str, color: bool) -> dict[str, list[str]]:
    """Signale les fous bloqués par leurs pions et les pièces non défendues.

    Args:
        fen: Position à analyser.
        color: Camp examiné (`True` = blancs).

    Returns:
        `fous_bloques` : fous qui ne disposent plus que d'une case au plus, avec
        un pion allié sur une diagonale voisine ; `pieces_non_defendues` :
        cases des pièces (hors roi et pions) sans défenseur.

    Examples:
        >>> piece_facts(chess.STARTING_FEN, True)["fous_bloques"]
        ['c1', 'f1']
    """
    board = Board(fen)
    blocked = []
    undefended = []
    for square in board.pieces(chess.BISHOP, color):
        neighbours = [
            square + step
            for step in (-9, -7, 7, 9)
            if 0 <= square + step <= 63
            and chess.square_distance(square, square + step) == 1
        ]
        own_pawns = [
            n for n in neighbours if board.piece_at(n) == chess.Piece(chess.PAWN, color)
        ]
        if own_pawns and len(_retreats(board, square, color)) <= 1:
            blocked.append(chess.square_name(square))
    for piece_type in _RETREATING_PIECES:
        for square in board.pieces(piece_type, color):
            if not _defenders(board, square, color):
                undefended.append(chess.square_name(square))
    return {
        "fous_bloques": sorted(blocked),
        "pieces_non_defendues": sorted(undefended),
    }


@beartype
def space(fen: str) -> dict[str, Any]:
    """Compare l'espace des deux camps : cases de la moitié adverse qu'ils contrôlent.

    Args:
        fen: Position à analyser.

    Returns:
        `blancs` et `noirs` : nombre de cases de la moitié adverse attaquées par au
        moins une pièce ou un pion du camp ; `avantage` : le camp qui en contrôle au
        moins `SPACE_MARGIN` de plus, ou `None`.

    Examples:
        >>> space(chess.STARTING_FEN)["avantage"] is None
        True
    """
    board = Board(fen)
    counts: dict[str, int] = {}
    for color in (WHITE, BLACK):
        counts[_COLOR_NAMES[color]] = sum(
            1
            for square in chess.SQUARES
            if _relative_rank(square, color) >= 4 and board.attackers(color, square)
        )
    gap = counts["blancs"] - counts["noirs"]
    advantage = None
    if abs(gap) >= SPACE_MARGIN:
        advantage = "blancs" if gap > 0 else "noirs"
    return {**counts, "avantage": advantage}
