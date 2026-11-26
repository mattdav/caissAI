"""Tests des critères `!?` et `!!` (caissAI.bin.nags), évaluations simulées."""

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import chess
import chess.engine
import chess.pgn
import pandas as pd
import pytest
from chess import Board, Move
from chess.engine import Cp, PovScore
from maia2.main import MAIA2Model

from caissAI.bin import nags
from caissAI.bin.game_analyzer import evaluate_candidates, get_nags
from caissAI.bin.nags import (
    Brilliance,
    Replies,
    Trap,
    brilliance_facts,
    losing_replies,
    p_err,
    passes_p_err_thresholds,
    static_exchange,
)

_ELO = 1500
_GOOD, _LOSING, _SILLY = 0.60, 0.40, 0.20  # ES adverse : bonne / perdante / loufoque


def _replies(*moves: tuple[str, float, float]) -> Replies:
    """Réponses `(coup, probabilité Maia, ES adverse)`."""
    return Replies(
        probs={move: prob for move, prob, _ in moves},
        es={move: es for move, _, es in moves},
    )


def _nags(trap: Trap | None = None, brilliance: Brilliance | None = None) -> list[int]:
    """NAGs d'un coup d'équilibre (avant = après), avec les faits donnés."""
    nag_list: list[int] = get_nags(
        is_forced=False,
        two_moves_before_score=PovScore(Cp(0), chess.WHITE),
        before_move_score=PovScore(Cp(0), chess.WHITE),
        after_move_score=PovScore(Cp(0), chess.WHITE),
        player_elo=_ELO,
        trap=trap,
        brilliance=brilliance,
    )["nag"]
    return nag_list


def _engine(analyse: Callable[..., Any]) -> MagicMock:
    engine = MagicMock(spec=chess.engine.SimpleEngine)
    engine.analyse.side_effect = analyse
    return engine


def _lines_engine(lines: list[tuple[str, int]]) -> MagicMock:
    """Moteur dont l'analyse MultiPV rend les lignes `(coup UCI, centipions)`."""

    def analyse(board: Board, limit: Any, multipv: int | None = None) -> Any:
        return [
            {"pv": [Move.from_uci(uci)], "score": PovScore(Cp(cp), board.turn)}
            for uci, cp in lines[:multipv]
        ]

    return _engine(analyse)


def _after_e4_d5() -> Board:
    board = Board()
    board.push_uci("e2e4")
    board.push_uci("d7d5")
    return board


# ---------------------------------------------------------------------------
# !? — coup piégeux
# ---------------------------------------------------------------------------

_CALM = _replies(("a", 0.9, _GOOD), ("b", 0.1, _LOSING))  # P_err = 0.10


def test_speculative_somme_reponses_perdantes() -> None:
    # Trois réponses perdantes à 25 % et une bonne réponse à 30 % (distribution
    # simulée, non normalisée) : P_err = 0,75 alors que la plus probable est bonne.
    played = _replies(
        ("a", 0.25, _LOSING),
        ("b", 0.25, _LOSING),
        ("c", 0.25, _LOSING),
        ("d", 0.30, _GOOD),
    )
    assert p_err(played) == pytest.approx(0.75)
    assert chess.pgn.NAG_SPECULATIVE_MOVE in _nags(trap=Trap(played, _CALM))


def test_speculative_seuil_relatif() -> None:
    # La plus probable est perdante (34 %) mais pas 2 fois la suivante (30 %).
    played = _replies(("a", 0.34, _LOSING), ("b", 0.30, _LOSING), ("c", 0.1, _GOOD))
    assert p_err(played) >= nags.SPECULATIVE_P_ERR_MIN
    assert not passes_p_err_thresholds(played)
    assert chess.pgn.NAG_SPECULATIVE_MOVE not in _nags(trap=Trap(played, _CALM))
    # Au moins 2 fois la 2e réponse : le seuil relatif est respecté.
    net = _replies(("a", 0.60, _LOSING), ("b", 0.30, _GOOD))
    assert passes_p_err_thresholds(net)


def test_speculative_filtre_loufoque() -> None:
    # Une réponse qui perd >= 0,30 d'ES (pièce en prise) n'entre pas dans P_err.
    played = _replies(
        ("a", 0.50, _SILLY),
        ("b", 0.30, _LOSING),
        ("c", 0.20, _GOOD),
    )
    assert "a" not in losing_replies(played)
    assert p_err(played) == pytest.approx(0.30)
    assert chess.pgn.NAG_SPECULATIVE_MOVE not in _nags(trap=Trap(played, _CALM))


def test_speculative_piege_vient_du_coup() -> None:
    # Même P_err après le coup joué et après le meilleur coup : pas de piège du coup.
    played = _replies(("a", 0.55, _LOSING), ("b", 0.45, _GOOD))
    assert p_err(played) >= nags.SPECULATIVE_P_ERR_MIN
    assert chess.pgn.NAG_SPECULATIVE_MOVE not in _nags(trap=Trap(played, played))


def test_speculative_reponse_favorable_adverse() -> None:
    # Non-régression : la réponse la plus probable (60 %) est la meilleure, elle
    # améliore la position adverse. Elle ne compte pas dans P_err.
    played = _replies(("a", 0.60, 0.70), ("b", 0.10, 0.55), ("c", 0.30, 0.65))
    assert p_err(played) == pytest.approx(0.10)
    assert chess.pgn.NAG_SPECULATIVE_MOVE not in _nags(trap=Trap(played, _CALM))


def test_speculative_coup_pas_une_erreur() -> None:
    played = _replies(("a", 0.25, _LOSING), ("b", 0.25, _LOSING), ("c", 0.5, _LOSING))
    result = get_nags(
        is_forced=False,
        two_moves_before_score=PovScore(Cp(0), chess.WHITE),
        before_move_score=PovScore(Cp(0), chess.WHITE),
        after_move_score=PovScore(Cp(-60), chess.WHITE),  # perd ~0,09 d'ES
        player_elo=_ELO,
        trap=Trap(played, _CALM),
    )
    assert chess.pgn.NAG_SPECULATIVE_MOVE not in result["nag"]


def test_speculative_chiffres_du_dict() -> None:
    played = _replies(("a", 0.6, _LOSING), ("b", 0.2, _LOSING), ("c", 0.2, _GOOD))
    result = get_nags(
        is_forced=False,
        two_moves_before_score=PovScore(Cp(0), chess.WHITE),
        before_move_score=PovScore(Cp(0), chess.WHITE),
        after_move_score=PovScore(Cp(0), chess.WHITE),
        player_elo=_ELO,
        trap=Trap(played, _CALM),
    )
    assert result["p_err"] == pytest.approx(0.8)
    assert result["p_err_best"] == pytest.approx(0.1)
    assert result["trap_reply"] == "a"
    assert result["trap_reply_loss"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# !! — coup brillant
# ---------------------------------------------------------------------------


def _quiet_unique(**overrides: Any) -> Brilliance:
    board = _after_e4_d5()
    engine = _lines_engine([("g1f3", 300), ("b1c3", overrides.pop("alt_cp", -100))])
    brilliance = brilliance_facts(
        board, Move.from_uci("g1f3"), Move.from_uci("g1f3"), engine, _ELO
    )
    assert brilliance is not None
    return brilliance


def _after_300() -> list[int]:
    nag_list: list[int] = get_nags(
        is_forced=False,
        two_moves_before_score=PovScore(Cp(300), chess.WHITE),
        before_move_score=PovScore(Cp(300), chess.WHITE),
        after_move_score=PovScore(Cp(300), chess.WHITE),
        player_elo=_ELO,
        brilliance=_quiet_unique(),
    )["nag"]
    return nag_list


def test_brilliant_coup_tranquille() -> None:
    # Nf3 : ni prise ni échec alors que exd5 existe ; seul coup à +300 cp.
    board = _after_e4_d5()
    assert any(board.is_capture(m) for m in board.legal_moves)
    brilliance = _quiet_unique()
    assert brilliance.is_quiet
    assert not brilliance.is_sacrifice
    assert not brilliance.found_late
    assert chess.pgn.NAG_BRILLIANT_MOVE in _after_300()


def test_brilliant_independant_de_maia() -> None:
    # Le coup a 90 % de probabilité Maia : `!!` quand même, rien de Maia n'entre.
    game = chess.pgn.Game()
    game.headers["WhiteElo"] = game.headers["BlackElo"] = str(_ELO)
    node = game.add_variation(Move.from_uci("e2e4"))
    node = node.add_variation(Move.from_uci("d7d5"))
    node = node.add_variation(Move.from_uci("g1f3"))
    played = Move.from_uci("g1f3")
    scores = {
        "before_move_score": PovScore(Cp(300), chess.WHITE),
        "played_move_eval.score": PovScore(Cp(300), chess.WHITE),
    }
    for proba in (0.05, 0.90):
        df = pd.DataFrame(
            [
                {
                    "node": node,
                    "played_move": played,
                    "is_forced": False,
                    "player_elo": _ELO,
                    "opponent_elo": _ELO,
                    "best_move_variation": [played],
                    "quick_best_move_before": played,
                    "next_move_probs": None,
                    "played_move_proba": proba,
                    "move_probs": {"g1f3": proba},
                    **scores,
                }
            ]
        )
        with patch.object(
            chess.engine.SimpleEngine,
            "popen_uci",
            return_value=_lines_engine([("g1f3", 300), ("b1c3", -100)]),
        ):
            df = evaluate_candidates(df, Path("stockfish"), MAIA2Model(), 1)
        row = df.iloc[0]
        assert row["trap"] is None
        result = get_nags(
            is_forced=False,
            two_moves_before_score=scores["before_move_score"],
            before_move_score=scores["before_move_score"],
            after_move_score=scores["played_move_eval.score"],
            player_elo=_ELO,
            brilliance=row["brilliance"],
        )
        assert chess.pgn.NAG_BRILLIANT_MOVE in result["nag"]
    for function in (nags.is_brilliant, nags.brilliance_facts):
        assert not [p for p in inspect.signature(function).parameters if "prob" in p]
    assert not [f for f in Brilliance.__dataclass_fields__ if "prob" in f]


def test_brilliant_pas_coup_unique() -> None:
    # Le 2e coup (+250 cp) est à moins de 0,15 d'ES du meilleur (+300 cp).
    board = _after_e4_d5()
    brilliance = brilliance_facts(
        board,
        Move.from_uci("g1f3"),
        Move.from_uci("g1f3"),
        _lines_engine([("g1f3", 300), ("b1c3", 250)]),
        _ELO,
    )
    assert brilliance is not None
    assert brilliance.es_best - brilliance.es_alternative < 0.15
    result = get_nags(
        is_forced=False,
        two_moves_before_score=PovScore(Cp(300), chess.WHITE),
        before_move_score=PovScore(Cp(300), chess.WHITE),
        after_move_score=PovScore(Cp(300), chess.WHITE),
        player_elo=_ELO,
        brilliance=brilliance,
    )
    assert chess.pgn.NAG_BRILLIANT_MOVE not in result["nag"]


def test_brilliant_sans_difficulte() -> None:
    # Coup unique et gagnant, mais ni sacrifice, ni tranquille, ni tardif.
    brilliance = Brilliance(0.75, 0.40, False, False, False)
    assert not nags.is_brilliant(0.75, 0.75, brilliance)
    assert nags.is_brilliant(0.75, 0.75, Brilliance(0.75, 0.40, False, False, True))


def test_brilliant_position_sauvee() -> None:
    # ES après le coup < 0,5 mais gain >= 0,20 sur le 2e coup : résultat accepté.
    assert nags.is_brilliant(0.45, 0.45, Brilliance(0.45, 0.10, False, False, True))
    assert not nags.is_brilliant(0.45, 0.45, Brilliance(0.45, 0.30, False, False, True))


def test_brilliant_coup_loin_du_meilleur() -> None:
    assert not nags.is_brilliant(0.75, 0.70, Brilliance(0.75, 0.40, True, True, True))


def test_sacrifice_echange_statique_perdant() -> None:
    board = Board("4k3/8/2p5/3p4/8/8/3R4/4K3 w - - 0 1")
    assert static_exchange(board, Move.from_uci("d2d5")) == -4
    brilliance = brilliance_facts(
        board,
        Move.from_uci("d2d5"),
        Move.from_uci("d2d5"),
        _lines_engine([("d2d5", 300), ("e1e2", -100)]),
        _ELO,
    )
    assert brilliance is not None
    assert brilliance.is_sacrifice


# ---------------------------------------------------------------------------
# Coups candidats : évaluations Stockfish seulement pour eux
# ---------------------------------------------------------------------------


def test_candidat_piege_evalue_apres_le_meilleur_coup() -> None:
    # Les Noirs jouent e5 au lieu du meilleur d5. Après e5, 90 % des réponses
    # perdent (le 1er coup légal est la seule bonne réponse) ; après d5, la
    # distribution Maia est concentrée sur la bonne réponse.
    game = chess.pgn.Game()
    game.headers["WhiteElo"] = game.headers["BlackElo"] = str(_ELO)
    node = game.add_variation(Move.from_uci("e2e4")).add_variation(
        Move.from_uci("e7e5")
    )
    after_played = node.board()
    first = next(iter(after_played.legal_moves)).uci()
    others = [m.uci() for m in after_played.legal_moves if m.uci() != first]
    played_probs = {first: 0.1, **{m: 0.9 / len(others) for m in others}}

    def analyse(board: Board, limit: Any, multipv: int | None = None) -> Any:
        legal = list(board.legal_moves)
        return [
            {"pv": [m], "score": PovScore(Cp(0 if i == 0 else -200), board.turn)}
            for i, m in enumerate(legal[:multipv])
        ]

    best_dist = {next(iter(Board().legal_moves)).uci(): 1.0}
    row = {
        "node": node,
        "played_move": Move.from_uci("e7e5"),
        "is_forced": False,
        "player_elo": _ELO,
        "opponent_elo": _ELO,
        "best_move_variation": [Move.from_uci("d7d5")],
        "quick_best_move_before": Move.from_uci("d7d5"),
        "next_move_probs": played_probs,
        "before_move_score": PovScore(Cp(0), chess.BLACK),
        "played_move_eval.score": PovScore(Cp(0), chess.BLACK),
    }
    with (
        patch.object(
            chess.engine.SimpleEngine, "popen_uci", return_value=_engine(analyse)
        ),
        patch(
            "caissAI.bin.game_analyzer.inference.inference_each",
            return_value=(best_dist, 0.5),
        ),
    ):
        df = evaluate_candidates(
            pd.DataFrame([row]), Path("stockfish"), MAIA2Model(), 1
        )
    trap = df.iloc[0]["trap"]
    assert isinstance(trap, Trap)
    assert p_err(trap.played) == pytest.approx(0.9)
    assert p_err(trap.best) == pytest.approx(0.0)
    result = get_nags(
        is_forced=False,
        two_moves_before_score=PovScore(Cp(0), chess.BLACK),
        before_move_score=PovScore(Cp(0), chess.BLACK),
        after_move_score=PovScore(Cp(0), chess.BLACK),
        player_elo=_ELO,
        trap=trap,
    )
    assert chess.pgn.NAG_SPECULATIVE_MOVE in result["nag"]
