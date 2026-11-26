"""Critères des NAGs positifs `!?` et `!!` (SPEC-003, « NAGs positifs »).

Les fonctions de décision sont pures (évaluations et distributions en entrée).
Les fonctions qui interrogent Stockfish (`reply_es`, `brilliance_facts`) ne sont
appelées que pour les coups candidats.
"""

from dataclasses import dataclass

import chess
import chess.engine
from beartype import beartype
from chess import Board, Move

from .utils import get_es

# ---------------------------------------------------------------------------
# Seuils de SPEC-003, à calibrer sur `mes parties.pgn` après CAI-003
# ---------------------------------------------------------------------------

# `!?` : le coup n'est pas une erreur (perte d'ES maximale vs le meilleur coup).
SPECULATIVE_LOSS_MAX: float = 0.05
# `!?` : `P_err(après le coup) − P_err(après le meilleur coup)` minimal.
SPECULATIVE_P_ERR_DELTA_MIN: float = 0.25
# `!?` : seuil absolu sur `P_err(après le coup)`.
SPECULATIVE_P_ERR_MIN: float = 0.40
# `!?` : la réponse la plus probable, si elle perd, vaut ce multiple de la 2e.
SPECULATIVE_RATIO_MIN: float = 2.0
# Une réponse est perdante si elle perd au moins cet ES vs la meilleure réponse…
REPLY_LOSS_MIN: float = 0.10
# … et loufoque (écartée de `P_err`) si elle perd cet ES ou plus.
REPLY_LOSS_SILLY: float = 0.30
# `!!` : écart d'ES maximal avec le meilleur coup.
BRILLIANT_BEST_TOLERANCE: float = 0.02
# `!!` : écart d'ES minimal avec le 2e meilleur coup (coup quasi unique).
BRILLIANT_UNIQUE_GAP_MIN: float = 0.15
# `!!` : ES minimal après le coup…
BRILLIANT_RESULT_MIN: float = 0.5
# … ou gain minimal sur le 2e meilleur coup (position sauvée).
BRILLIANT_RESCUE_GAIN_MIN: float = 0.20
# Profondeur des analyses Stockfish des coups candidats.
CANDIDATE_DEPTH: int = 16

_MATE_SCORE: int = 100000
_EPSILON: float = 1e-9
_PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


@dataclass(frozen=True)
class Replies:
    """Réponses adverses après un coup.

    Attributes:
        probs: Distribution Maia `{coup UCI: probabilité}` de l'adversaire.
        es: ES de l'adversaire après chacune de ses réponses légales
            `{coup UCI: ES}`.
    """

    probs: dict[str, float]
    es: dict[str, float]


@dataclass(frozen=True)
class Trap:
    """Réponses adverses après le coup joué et après le meilleur coup."""

    played: Replies
    best: Replies


@dataclass(frozen=True)
class Brilliance:
    """Faits Stockfish sur un coup joué proche du meilleur.

    Attributes:
        es_best: ES du meilleur coup (MultiPV 2, point de vue du joueur).
        es_alternative: ES du meilleur autre coup que le coup joué.
        is_sacrifice: L'échange statique sur la case d'arrivée est perdant.
        is_quiet: Ni prise ni échec, alors que des prises ou échecs existent.
        found_late: Le coup n'est pas en tête de l'analyse rapide.
    """

    es_best: float
    es_alternative: float
    is_sacrifice: bool
    is_quiet: bool
    found_late: bool


@beartype
def losing_replies(replies: Replies) -> dict[str, float]:
    """Réponses plausibles qui perdent de l'espérance de gain.

    Args:
        replies: Réponses adverses d'une position.

    Returns:
        dict[str, float]: `{coup UCI: perte d'ES}` des réponses qui perdent au
        moins `REPLY_LOSS_MIN` et strictement moins que `REPLY_LOSS_SILLY`.
    """
    if not replies.es:
        return {}
    best = max(replies.es.values())
    losses = {move: best - es for move, es in replies.es.items()}
    return {
        move: loss
        for move, loss in losses.items()
        if loss >= REPLY_LOSS_MIN - _EPSILON and loss < REPLY_LOSS_SILLY - _EPSILON
    }


@beartype
def p_err(replies: Replies) -> float:
    """Probabilité que l'adversaire se trompe.

    Args:
        replies: Réponses adverses d'une position.

    Returns:
        float: Somme des probabilités Maia des réponses perdantes plausibles.
    """
    losing = losing_replies(replies)
    return sum((prob for move, prob in replies.probs.items() if move in losing), 0.0)


@beartype
def passes_p_err_thresholds(replies: Replies) -> bool:
    """Conditions 3 (seuil absolu) et 4 (seuil relatif) du `!?`.

    La condition 4 ne joue que si la réponse la plus probable est perdante :
    elle exige alors au moins `SPECULATIVE_RATIO_MIN` fois la probabilité de la
    2e réponse. Si la plus probable est une bonne réponse, elle ne contraint rien.

    Args:
        replies: Réponses adverses après le coup joué.

    Returns:
        bool: True si les deux seuils sont respectés.
    """
    if p_err(replies) < SPECULATIVE_P_ERR_MIN - _EPSILON:
        return False
    ranked = sorted(replies.probs.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][0] not in losing_replies(replies):
        return True
    return len(ranked) == 1 or ranked[0][1] >= SPECULATIVE_RATIO_MIN * ranked[1][1]


@beartype
def is_speculative(es_after_before_delta: float, trap: Trap) -> bool:
    """Critère `!?` : le coup piégeux (SPEC-003, cinq conditions).

    Args:
        es_after_before_delta: ES après le coup moins ES avant, pour le joueur.
        trap: Réponses adverses après le coup joué et après le meilleur coup.

    Returns:
        bool: True si le coup mérite `!?`.
    """
    return (
        es_after_before_delta >= -SPECULATIVE_LOSS_MAX - _EPSILON
        and passes_p_err_thresholds(trap.played)
        and p_err(trap.played) - p_err(trap.best)
        >= SPECULATIVE_P_ERR_DELTA_MIN - _EPSILON
    )


@beartype
def is_brilliant(
    es_before_move: float, es_after_move: float, brilliance: Brilliance
) -> bool:
    """Critère `!!` : le coup brillant, sans aucune probabilité Maia.

    Args:
        es_before_move: ES du meilleur coup avant le coup, pour le joueur.
        es_after_move: ES après le coup joué, pour le joueur.
        brilliance: Faits Stockfish du coup.

    Returns:
        bool: True si le coup mérite `!!`.
    """
    near_best = es_before_move - es_after_move <= BRILLIANT_BEST_TOLERANCE + _EPSILON
    gap = brilliance.es_best - brilliance.es_alternative
    unique = gap >= BRILLIANT_UNIQUE_GAP_MIN - _EPSILON
    result = (
        es_after_move >= BRILLIANT_RESULT_MIN - _EPSILON
        or es_after_move - brilliance.es_alternative
        >= BRILLIANT_RESCUE_GAIN_MIN - _EPSILON
    )
    difficult = brilliance.is_sacrifice or brilliance.is_quiet or brilliance.found_late
    return near_best and unique and result and difficult


@beartype
def static_exchange(board: Board, move: Move) -> int:
    """Bilan matériel de l'échange statique sur la case d'arrivée du coup.

    Args:
        board: Position avant le coup.
        move: Coup du joueur.

    Returns:
        int: Gain net du joueur en points de matériel (négatif : sacrifice).

    Examples:
        >>> static_exchange(chess.Board(), chess.Move.from_uci("g1f3"))
        0
        >>> b = chess.Board("4k3/8/2p5/3p4/8/8/3R4/4K3 w - - 0 1")
        >>> static_exchange(b, chess.Move.from_uci("d2d5"))
        -4
    """
    captured = board.piece_at(move.to_square)
    gain = _PIECE_VALUES[captured.piece_type] if captured is not None else 0
    after = board.copy(stack=False)
    after.push(move)
    return gain - _best_recapture(after, move.to_square)


@beartype
def _best_recapture(board: Board, square: int) -> int:
    """Meilleur gain du camp au trait en reprenant sur `square` (0 s'il passe)."""
    recaptures = [m for m in board.legal_moves if m.to_square == square]
    victim = board.piece_at(square)
    if not recaptures or victim is None:
        return 0
    move = min(
        recaptures,
        key=lambda m: _PIECE_VALUES[board.piece_type_at(m.from_square) or chess.PAWN],
    )
    value = _PIECE_VALUES[victim.piece_type]
    board = board.copy(stack=False)
    board.push(move)
    return max(0, value - _best_recapture(board, square))


@beartype
def _mover_es(info: chess.engine.InfoDict, turn: bool, elo: int) -> float:
    """ES du camp `turn` d'après un résultat d'analyse Stockfish."""
    return get_es(elo, info["score"].pov(turn).score(mate_score=_MATE_SCORE))


@beartype
def reply_es(
    board: Board, engine: chess.engine.SimpleEngine, elo: int
) -> dict[str, float]:
    """Évalue toutes les réponses légales d'une position.

    Args:
        board: Position où l'adversaire est au trait.
        engine: Moteur UCI.
        elo: ELO de l'adversaire.

    Returns:
        dict[str, float]: `{coup UCI: ES de l'adversaire après ce coup}`.
    """
    count = board.legal_moves.count()
    if count == 0:
        return {}
    infos = engine.analyse(
        board, chess.engine.Limit(depth=CANDIDATE_DEPTH), multipv=count
    )
    return {
        info["pv"][0].uci(): _mover_es(info, board.turn, elo)
        for info in infos
        if "pv" in info
    }


@beartype
def brilliance_facts(
    board_before: Board,
    move: Move,
    quick_best_move: Move,
    engine: chess.engine.SimpleEngine,
    elo: int,
) -> Brilliance | None:
    """Mesure l'unicité et la difficulté d'un coup (MultiPV 2 avant le coup).

    Args:
        board_before: Position avant le coup.
        move: Coup joué.
        quick_best_move: Meilleur coup de l'analyse rapide de la position.
        engine: Moteur UCI.
        elo: ELO du joueur.

    Returns:
        Brilliance | None: Les faits, ou None si le joueur n'a aucune
        alternative (coup unique légal).
    """
    infos = engine.analyse(
        board_before, chess.engine.Limit(depth=CANDIDATE_DEPTH), multipv=2
    )
    lines = [
        (info["pv"][0], _mover_es(info, board_before.turn, elo))
        for info in infos
        if "pv" in info
    ]
    alternatives = [es for line_move, es in lines if line_move != move]
    if not lines or not alternatives:
        return None
    legal = list(board_before.legal_moves)
    forcing_exists = any(
        board_before.is_capture(m) or board_before.gives_check(m) for m in legal
    )
    return Brilliance(
        es_best=lines[0][1],
        es_alternative=max(alternatives),
        is_sacrifice=static_exchange(board_before, move) < 0,
        is_quiet=forcing_exists
        and not board_before.is_capture(move)
        and not board_before.gives_check(move),
        found_late=move != quick_best_move,
    )
