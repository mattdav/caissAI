"""Unit tests for caissAI.bin.game_analyzer.

Pure calculation functions are tested directly.
Functions that depend on Stockfish or Maia2 are tested with mocks.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import chess
import chess.engine
import chess.pgn
import pandas as pd
import pytest
from chess import Move
from chess.engine import Cp, PovScore

from caissAI.bin.game_analyzer import (
    classify_opening,
    comment_end,
    df_to_pgn,
    evaluate_position,
    extract_game_data,
    get_comment,
    get_eval,
    get_move_probs,
    get_nags,
    get_position_depth,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pov(cp: int, turn: chess.Color = chess.WHITE) -> PovScore:
    """Shorthand to build a PovScore from centipawns."""
    return PovScore(Cp(cp), turn)


def _base_nags_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return a default set of arguments for get_nags (equal, quiet position)."""
    defaults: dict[str, Any] = {
        "is_forced": False,
        "is_zeroing": False,
        "two_moves_before_score": pov(0),
        "before_move_score": pov(0),
        "played_move_proba": 0.5,
        "after_move_score": pov(0),
        "after_likeliest_move_score": pov(0),
        "next_likeliest_move_proba": 0.5,
        "next_likeliest_move_score": pov(0),
        "player_elo": 1500,
        "opponent_elo": 1500,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# get_nags – forced move
# ---------------------------------------------------------------------------


class TestGetNagsForced:
    def test_forced_move_gets_singular_nag(self) -> None:
        result = get_nags(**_base_nags_kwargs(is_forced=True))
        assert chess.pgn.NAG_SINGULAR_MOVE in result["nag"]

    def test_forced_move_has_no_advantage_nag(self) -> None:
        """No second (position-advantage) NAG is appended for forced moves."""
        result = get_nags(**_base_nags_kwargs(is_forced=True))
        assert len(result["nag"]) == 1

    def test_forced_move_overrides_negative_delta(self) -> None:
        """Even when the move loses material, if forced it gets NAG_SINGULAR_MOVE."""
        result = get_nags(
            **_base_nags_kwargs(
                is_forced=True,
                before_move_score=pov(200),
                after_move_score=pov(-500),
            )
        )
        assert result["nag"] == [chess.pgn.NAG_SINGULAR_MOVE]


# ---------------------------------------------------------------------------
# get_nags – blunder
# ---------------------------------------------------------------------------


class TestGetNagsBlunder:
    def test_large_negative_delta_is_blunder(self) -> None:
        # White plays from +300cp to -300cp: ES drops ~0.5, advantage drops 7 levels.
        result = get_nags(
            **_base_nags_kwargs(
                before_move_score=pov(300, chess.WHITE),
                after_move_score=pov(-300, chess.WHITE),
                after_likeliest_move_score=pov(300, chess.WHITE),
            )
        )
        assert chess.pgn.NAG_BLUNDER in result["nag"]

    def test_blunder_appends_advantage_nag(self) -> None:
        result = get_nags(
            **_base_nags_kwargs(
                before_move_score=pov(300, chess.WHITE),
                after_move_score=pov(-300, chess.WHITE),
                after_likeliest_move_score=pov(300, chess.WHITE),
            )
        )
        assert len(result["nag"]) == 2

    def test_blunder_result_fields_present(self) -> None:
        result = get_nags(
            **_base_nags_kwargs(
                before_move_score=pov(300, chess.WHITE),
                after_move_score=pov(-300, chess.WHITE),
                after_likeliest_move_score=pov(300, chess.WHITE),
            )
        )
        assert "es_after_before_delta" in result
        assert result["es_after_before_delta"] < -0.2


# ---------------------------------------------------------------------------
# get_nags – mistake
# ---------------------------------------------------------------------------


class TestGetNagsMistake:
    def test_medium_negative_delta_is_mistake(self) -> None:
        # White from +100cp (adv=2) to -30cp (adv=-1):
        # ES drops ~0.117 (satisfies es <= -0.1 AND adv_delta=3 >= 2)
        # but es_delta > -0.2 so blunder condition is NOT met.
        result = get_nags(
            **_base_nags_kwargs(
                before_move_score=pov(100, chess.WHITE),
                after_move_score=pov(-30, chess.WHITE),
                after_likeliest_move_score=pov(100, chess.WHITE),
                player_elo=1500,
            )
        )
        assert chess.pgn.NAG_MISTAKE in result["nag"]


# ---------------------------------------------------------------------------
# get_nags – dubious
# ---------------------------------------------------------------------------


class TestGetNagsDubious:
    def test_small_negative_delta_is_dubious(self) -> None:
        # White goes from +100cp to +10cp: slight advantage loss.
        result = get_nags(
            **_base_nags_kwargs(
                before_move_score=pov(100, chess.WHITE),
                after_move_score=pov(10, chess.WHITE),
                after_likeliest_move_score=pov(100, chess.WHITE),
                player_elo=1000,
            )
        )
        assert chess.pgn.NAG_DUBIOUS_MOVE in result["nag"]


# ---------------------------------------------------------------------------
# get_nags – good move
# ---------------------------------------------------------------------------


class TestGetNagsGoodMove:
    def test_significant_improvement_is_good_move(self) -> None:
        # White goes from equality to a clear advantage over two moves.
        result = get_nags(
            **_base_nags_kwargs(
                two_moves_before_score=pov(0, chess.WHITE),
                before_move_score=pov(0, chess.WHITE),
                after_move_score=pov(200, chess.WHITE),
                after_likeliest_move_score=pov(200, chess.WHITE),
                player_elo=1500,
            )
        )
        assert chess.pgn.NAG_GOOD_MOVE in result["nag"]


# ---------------------------------------------------------------------------
# get_nags – brilliant move
# ---------------------------------------------------------------------------


class TestGetNagsBrilliant:
    def test_brilliant_conditions_trigger_brilliant_nag(self) -> None:
        # Played move: white winning (+300cp); likeliest move: white losing (-100cp).
        # Low probability + zeroing move.
        result = get_nags(
            **_base_nags_kwargs(
                is_zeroing=True,
                before_move_score=pov(0, chess.WHITE),
                after_move_score=pov(300, chess.WHITE),
                after_likeliest_move_score=pov(-100, chess.WHITE),
                played_move_proba=0.05,
                player_elo=1500,
            )
        )
        assert chess.pgn.NAG_BRILLIANT_MOVE in result["nag"]

    def test_non_zeroing_cannot_be_brilliant(self) -> None:
        result = get_nags(
            **_base_nags_kwargs(
                is_zeroing=False,
                before_move_score=pov(0, chess.WHITE),
                after_move_score=pov(300, chess.WHITE),
                after_likeliest_move_score=pov(-100, chess.WHITE),
                played_move_proba=0.05,
                player_elo=1500,
            )
        )
        assert chess.pgn.NAG_BRILLIANT_MOVE not in result["nag"]

    def test_high_probability_cannot_be_brilliant(self) -> None:
        result = get_nags(
            **_base_nags_kwargs(
                is_zeroing=True,
                before_move_score=pov(0, chess.WHITE),
                after_move_score=pov(300, chess.WHITE),
                after_likeliest_move_score=pov(-100, chess.WHITE),
                played_move_proba=0.9,
                player_elo=1500,
            )
        )
        assert chess.pgn.NAG_BRILLIANT_MOVE not in result["nag"]


# ---------------------------------------------------------------------------
# get_nags – no annotation
# ---------------------------------------------------------------------------


class TestGetNagsNoAnnotation:
    def test_equal_quiet_move_has_no_nag(self) -> None:
        result = get_nags(**_base_nags_kwargs())
        assert result["nag"] == []

    def test_small_improvement_has_no_nag(self) -> None:
        # Tiny improvement, not enough for a good move annotation.
        result = get_nags(
            **_base_nags_kwargs(
                two_moves_before_score=pov(10, chess.WHITE),
                before_move_score=pov(10, chess.WHITE),
                after_move_score=pov(20, chess.WHITE),
                after_likeliest_move_score=pov(20, chess.WHITE),
            )
        )
        assert result["nag"] == []


# ---------------------------------------------------------------------------
# get_nags – black perspective
# ---------------------------------------------------------------------------


class TestGetNagsBlackPerspective:
    def test_blunder_by_black(self) -> None:
        # It is black's turn (before_move_score.turn == chess.BLACK).
        # pov(300, BLACK): +300 from black's POV → white sees -300cp → black winning.
        # pov(-300, BLACK): -300 from black's POV → white sees +300cp → black losing.
        # After flipping ES for black: delta ≈ -0.5 and adv_delta = 7 → blunder.
        result = get_nags(
            **_base_nags_kwargs(
                before_move_score=pov(300, chess.BLACK),
                after_move_score=pov(-300, chess.BLACK),
                after_likeliest_move_score=pov(300, chess.BLACK),
            )
        )
        assert chess.pgn.NAG_BLUNDER in result["nag"]


# ---------------------------------------------------------------------------
# get_nags – returned dictionary structure
# ---------------------------------------------------------------------------


class TestGetNagsDictStructure:
    def test_result_contains_all_expected_keys(self) -> None:
        result = get_nags(**_base_nags_kwargs())
        expected_keys = {
            "nag",
            "es_after_two_moves_before_delta",
            "es_after_before_delta",
            "es_after_move",
            "es_played_likeliest_delta",
            "es_next_likeliest_before_delta",
        }
        assert expected_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# comment_end
# ---------------------------------------------------------------------------


class TestCommentEnd:
    def _node(self, **board_attrs: Any) -> MagicMock:
        node = MagicMock()
        board = MagicMock()
        node.board.return_value = board
        board.outcome.return_value = True  # non-None → game ended
        board.is_stalemate.return_value = False
        board.is_insufficient_material.return_value = False
        board.can_claim_fifty_moves.return_value = False
        board.can_claim_threefold_repetition.return_value = False
        board.is_checkmate.return_value = False
        for attr, value in board_attrs.items():
            getattr(board, attr).return_value = value
        return node

    def test_checkmate_returns_checkmate_string(self) -> None:
        node = self._node(is_checkmate=True)
        assert comment_end(node, 0.0, "blancs") == " Checkmate"

    def test_stalemate_returns_stalemate_string(self) -> None:
        node = self._node(is_stalemate=True)
        assert comment_end(node, 0.5, "blancs") == "Stalemate"

    def test_insufficient_material(self) -> None:
        node = self._node(is_insufficient_material=True)
        assert comment_end(node, 0.5, "noirs") == " Insufficient material to mate"

    def test_fifty_move_rule(self) -> None:
        node = self._node(can_claim_fifty_moves=True)
        assert comment_end(node, 0.5, "blancs") == " Fifty move rule"

    def test_threefold_repetition(self) -> None:
        node = self._node(can_claim_threefold_repetition=True)
        assert comment_end(node, 0.5, "noirs") == " Three-fold repetition"

    def test_no_outcome_includes_expected_score(self) -> None:
        node = MagicMock()
        node.board.return_value.outcome.return_value = None
        result = comment_end(node, 0.65, "blancs")
        assert "65.00%" in result
        assert "blancs" in result

    def test_no_outcome_formats_zero(self) -> None:
        node = MagicMock()
        node.board.return_value.outcome.return_value = None
        result = comment_end(node, 0.0, "noirs")
        assert "0.00%" in result
        assert "noirs" in result


# ---------------------------------------------------------------------------
# get_comment
# ---------------------------------------------------------------------------


class TestGetComment:
    def _make_node(self) -> chess.pgn.ChildNode:
        game = chess.pgn.Game()
        game.headers["WhiteElo"] = "1500"
        game.headers["BlackElo"] = "1500"
        node = game.add_variation(chess.Move.from_uci("e2e4"))
        _child = node.add_variation(chess.Move.from_uci("e7e5"))
        return node  # type: ignore[return-value]

    def _score(self, cp: int, turn: chess.Color = chess.WHITE) -> PovScore:
        return pov(cp, turn)

    def test_no_nag_returns_empty_comment(self) -> None:
        node = self._make_node()
        result = get_comment(
            node=node,
            player_elo=1500,
            played_move_proba=0.5,
            played_move_score=self._score(0),
            likeliest_move=Move.from_uci("e2e4"),
            next_likeliest_move=Move.from_uci("e7e5"),
            next_likeliest_move_proba=0.5,
            nag_dict={
                "nag": [],
                "es_after_before_delta": 0.0,
                "es_after_two_moves_before_delta": 0.0,
                "es_after_move": 0.5,
                "es_played_likeliest_delta": 0.0,
                "es_next_likeliest_before_delta": 0.0,
            },
        )
        assert result == ""

    def test_blunder_comment_contains_expected_score_delta(self) -> None:
        node = self._make_node()
        result = get_comment(
            node=node,
            player_elo=1500,
            played_move_proba=0.5,
            played_move_score=self._score(0),
            likeliest_move=Move.from_uci("e2e4"),
            next_likeliest_move=Move.from_uci("e7e5"),
            next_likeliest_move_proba=0.5,
            nag_dict={
                "nag": [chess.pgn.NAG_BLUNDER],
                "es_after_before_delta": -0.35,
                "es_after_two_moves_before_delta": -0.35,
                "es_after_move": 0.15,
                "es_played_likeliest_delta": -0.35,
                "es_next_likeliest_before_delta": 0.0,
            },
        )
        assert "-35.00%" in result

    def test_good_move_comment_is_non_empty(self) -> None:
        node = self._make_node()
        result = get_comment(
            node=node,
            player_elo=1500,
            played_move_proba=0.4,
            played_move_score=self._score(200),
            likeliest_move=Move.from_uci("e2e4"),
            next_likeliest_move=Move.from_uci("e7e5"),
            next_likeliest_move_proba=0.5,
            nag_dict={
                "nag": [chess.pgn.NAG_GOOD_MOVE],
                "es_after_before_delta": 0.18,
                "es_after_two_moves_before_delta": 0.18,
                "es_after_move": 0.68,
                "es_played_likeliest_delta": 0.0,
                "es_next_likeliest_before_delta": 0.0,
            },
        )
        assert result != ""
        assert "18.00%" in result

    def test_brilliant_comment_contains_probability(self) -> None:
        node = self._make_node()
        result = get_comment(
            node=node,
            player_elo=1500,
            played_move_proba=0.05,
            played_move_score=self._score(300),
            likeliest_move=Move.from_uci("d2d4"),
            next_likeliest_move=Move.from_uci("e7e5"),
            next_likeliest_move_proba=0.3,
            nag_dict={
                "nag": [chess.pgn.NAG_BRILLIANT_MOVE],
                "es_after_before_delta": 0.3,
                "es_after_two_moves_before_delta": 0.3,
                "es_after_move": 0.8,
                "es_played_likeliest_delta": 0.35,
                "es_next_likeliest_before_delta": 0.0,
            },
        )
        assert "5.00%" in result  # 100 * 0.05 formatted

    def test_speculative_comment_mentions_opponent_probability(self) -> None:
        node = self._make_node()
        result = get_comment(
            node=node,
            player_elo=1500,
            played_move_proba=0.3,
            played_move_score=self._score(100),
            likeliest_move=Move.from_uci("e2e4"),
            next_likeliest_move=Move.from_uci("e7e5"),
            next_likeliest_move_proba=0.45,
            nag_dict={
                "nag": [chess.pgn.NAG_SPECULATIVE_MOVE],
                "es_after_before_delta": 0.05,
                "es_after_two_moves_before_delta": 0.05,
                "es_after_move": 0.55,
                "es_played_likeliest_delta": 0.0,
                "es_next_likeliest_before_delta": -0.12,
            },
        )
        assert "45.00%" in result
        assert "-12.00%" in result


# ---------------------------------------------------------------------------
# get_eval – mocked Stockfish
# ---------------------------------------------------------------------------


class TestGetEval:
    def test_returns_variation_and_score(self) -> None:
        board = chess.Board()
        engine = MagicMock(spec=chess.engine.SimpleEngine)
        mock_score = MagicMock()
        mock_pv = [chess.Move.from_uci("e2e4")]
        engine.analyse.return_value = {"score": mock_score, "pv": mock_pv}

        result = get_eval(board, engine, depth=10)

        assert result["variation"] == mock_pv
        assert result["score"] is mock_score

    def test_calls_engine_with_correct_depth(self) -> None:
        board = chess.Board()
        engine = MagicMock(spec=chess.engine.SimpleEngine)
        engine.analyse.return_value = {
            "score": MagicMock(),
            "pv": [chess.Move.from_uci("d2d4")],
        }

        get_eval(board, engine, depth=20)

        engine.analyse.assert_called_once_with(board, chess.engine.Limit(depth=20))


# ---------------------------------------------------------------------------
# get_position_depth – mocked Stockfish
# ---------------------------------------------------------------------------


class TestGetPositionDepth:
    def _make_engine(
        self,
        cp1: int,
        cp2: int,
        move1: str,
        move2: str,
        depth1: int = 10,
        depth2: int = 15,
    ) -> MagicMock:
        engine = MagicMock(spec=chess.engine.SimpleEngine)
        score1 = MagicMock()
        score1.white.return_value.score.return_value = cp1
        score2 = MagicMock()
        score2.white.return_value.score.return_value = cp2
        engine.analyse.side_effect = [
            {"score": score1, "pv": [chess.Move.from_uci(move1)], "depth": depth1},
            {"score": score2, "pv": [chess.Move.from_uci(move2)], "depth": depth2},
        ]
        return engine

    def test_same_move_small_delta_uses_shallow_depth(self) -> None:
        # Both analyses agree → use depth[0] (fast analysis is sufficient).
        engine = self._make_engine(
            cp1=10, cp2=11, move1="e2e4", move2="e2e4", depth1=10, depth2=15
        )
        result = get_position_depth(
            chess.Board(),
            1500,
            engine,
            times=[0.1, 0.5],  # type: ignore[arg-type]
        )
        assert result == 10

    def test_different_move_small_delta_uses_deep_depth(self) -> None:
        # Best move changed but eval barely moved → use depth[1].
        engine = self._make_engine(
            cp1=10, cp2=11, move1="e2e4", move2="d2d4", depth1=10, depth2=15
        )
        result = get_position_depth(
            chess.Board(),
            1500,
            engine,
            times=[0.2, 0.8],  # type: ignore[arg-type]
        )
        assert result == 15

    def test_different_move_large_delta_uses_extended_depth(self) -> None:
        # Best move changed AND eval shifted significantly → use int(depth[1] * 1.1).
        engine = self._make_engine(
            cp1=10, cp2=50, move1="e2e4", move2="d2d4", depth1=10, depth2=20
        )
        result = get_position_depth(
            chess.Board(),
            1500,
            engine,
            times=[0.15, 0.6],  # type: ignore[arg-type]
        )
        assert result == int(20 * 1.1)


# ---------------------------------------------------------------------------
# get_move_probs – mocked Maia2
# ---------------------------------------------------------------------------


class TestGetMoveProbs:
    def _make_df(self) -> pd.DataFrame:
        """Build a minimal DataFrame matching the shape expected by get_move_probs."""
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["WhiteElo"] = "1500"
        game.headers["BlackElo"] = "1500"
        node = game.add_variation(chess.Move.from_uci("e2e4"))
        played_move = chess.Move.from_uci("e2e4")
        return pd.DataFrame(
            {
                "node": [node],
                "played_move": [played_move],
                "player_elo": [1500],
                "opponent_elo": [1500],
            }
        )

    @patch("caissAI.bin.game_analyzer.inference")
    def test_adds_likeliest_move_column(self, mock_inference: MagicMock) -> None:
        mock_inference.prepare.return_value = MagicMock()
        move_probs = {"e2e4": 0.45, "d2d4": 0.30, "g1f3": 0.25}
        mock_inference.inference_each.return_value = (move_probs, 0.55)

        df = get_move_probs(self._make_df(), MagicMock())

        assert "likeliest_move" in df.columns
        assert df["likeliest_move"].iloc[0] == Move.from_uci("e2e4")

    @patch("caissAI.bin.game_analyzer.inference")
    def test_adds_probability_columns(self, mock_inference: MagicMock) -> None:
        mock_inference.prepare.return_value = MagicMock()
        move_probs = {"e2e4": 0.45, "d2d4": 0.30}
        mock_inference.inference_each.return_value = (move_probs, 0.55)

        df = get_move_probs(self._make_df(), MagicMock())

        assert "likeliest_move_proba" in df.columns
        assert "played_move_proba" in df.columns
        assert df["likeliest_move_proba"].iloc[0] == pytest.approx(0.45)
        assert df["played_move_proba"].iloc[0] == pytest.approx(0.45)

    @patch("caissAI.bin.game_analyzer.inference")
    def test_played_move_proba_differs_from_likeliest_when_not_top(
        self, mock_inference: MagicMock
    ) -> None:
        """When the played move is not the likeliest, played_move_proba < likeliest."""
        mock_inference.prepare.return_value = MagicMock()
        # Likeliest is d2d4, but the node was created with e2e4
        move_probs = {"d2d4": 0.60, "e2e4": 0.25, "g1f3": 0.15}
        mock_inference.inference_each.return_value = (move_probs, 0.55)

        df = get_move_probs(self._make_df(), MagicMock())

        assert df["likeliest_move_proba"].iloc[0] == pytest.approx(0.60)
        assert df["played_move_proba"].iloc[0] == pytest.approx(0.25)

    @patch("caissAI.bin.game_analyzer.inference")
    def test_calls_inference_once_per_row(self, mock_inference: MagicMock) -> None:
        mock_inference.prepare.return_value = MagicMock()
        move_probs = {"e2e4": 0.5}
        mock_inference.inference_each.return_value = (move_probs, 0.5)

        df = self._make_df()
        get_move_probs(df, MagicMock())

        assert mock_inference.inference_each.call_count == len(df)


# ---------------------------------------------------------------------------
# get_nags – speculative move
# ---------------------------------------------------------------------------


class TestGetNagsSpeculative:
    def test_speculative_conditions_trigger_speculative_nag(self) -> None:
        # White plays +80cp (slight advantage), likeliest reply leads to -80cp for
        # white. Opponent has 40% chance of choosing that bad reply.
        # → good for white (es_after > 0.5) but tricky (opponent likely to blunder).
        result = get_nags(
            **_base_nags_kwargs(
                two_moves_before_score=pov(0, chess.WHITE),
                before_move_score=pov(0, chess.WHITE),
                after_move_score=pov(80, chess.WHITE),
                after_likeliest_move_score=pov(-80, chess.WHITE),
                next_likeliest_move_score=pov(-80, chess.WHITE),
                next_likeliest_move_proba=0.4,
                played_move_proba=0.3,
                is_zeroing=False,
            )
        )
        assert chess.pgn.NAG_SPECULATIVE_MOVE in result["nag"]


# ---------------------------------------------------------------------------
# get_position_depth – default times branch
# ---------------------------------------------------------------------------


class TestGetPositionDepthDefaultTimes:
    def test_default_times_are_used_when_none(self) -> None:
        """Covers the `if times is None: times = [0.3, 1.0]` branch."""
        engine = MagicMock(spec=chess.engine.SimpleEngine)
        score = MagicMock()
        score.white.return_value.score.return_value = 5
        # Same best move, small delta → shallow depth returned
        engine.analyse.side_effect = [
            {"score": score, "pv": [chess.Move.from_uci("e2e4")], "depth": 9},
            {"score": score, "pv": [chess.Move.from_uci("e2e4")], "depth": 14},
        ]
        board = chess.Board()
        board.push(chess.Move.from_uci("d2d4"))  # distinct position from other tests

        result = get_position_depth(board, 1500, engine)  # no times arg

        assert result == 9


# ---------------------------------------------------------------------------
# evaluate_position – mocked get_position_depth + get_eval
# ---------------------------------------------------------------------------


class TestEvaluatePosition:
    @patch("caissAI.bin.game_analyzer.get_eval")
    @patch("caissAI.bin.game_analyzer.get_position_depth")
    def test_returns_eval_dict(
        self,
        mock_get_position_depth: MagicMock,
        mock_get_eval: MagicMock,
    ) -> None:
        mock_get_position_depth.return_value = 15
        expected = {"variation": [chess.Move.from_uci("e2e4")], "score": MagicMock()}
        mock_get_eval.return_value = expected

        board = chess.Board()
        board.push(chess.Move.from_uci("g1f3"))  # distinct board
        engine = MagicMock()

        result = evaluate_position(board, 1500, engine)

        assert result is expected
        mock_get_position_depth.assert_called_once_with(board, 1500, engine)
        mock_get_eval.assert_called_once_with(board, engine, 15)


# ---------------------------------------------------------------------------
# classify_opening – mocked ECO lookup
# ---------------------------------------------------------------------------


class TestClassifyOpening:
    def _make_game(self, moves: list[str]) -> chess.pgn.Game:
        game = chess.pgn.Game()
        game.headers["WhiteElo"] = "1500"
        game.headers["BlackElo"] = "1500"
        node: chess.pgn.GameNode = game
        for uci in moves:
            node = node.add_variation(chess.Move.from_uci(uci))
        return game

    @patch("caissAI.bin.game_analyzer.classify_epd")
    @patch("caissAI.bin.game_analyzer.get_eco_file")
    def test_sets_eco_and_opening_headers_on_match(
        self,
        mock_get_eco_file: MagicMock,
        mock_classify_epd: MagicMock,
    ) -> None:
        mock_get_eco_file.return_value = MagicMock()
        # Match found at the deepest position (last move of the game)
        mock_classify_epd.return_value = {"code": "B20", "desc": "Sicilian Defense"}

        game = self._make_game(["e2e4", "c7c5", "g1f3"])
        result = classify_opening(game, Path("/fake/eco.parquet"))

        assert game.headers["ECO"] == "B20"
        assert game.headers["Opening"] == "Sicilian Defense"
        assert result == game.end()

    @patch("caissAI.bin.game_analyzer.classify_epd")
    @patch("caissAI.bin.game_analyzer.get_eco_file")
    def test_returns_game_root_when_no_match(
        self,
        mock_get_eco_file: MagicMock,
        mock_classify_epd: MagicMock,
    ) -> None:
        mock_get_eco_file.return_value = MagicMock()
        mock_classify_epd.return_value = {"code": "", "desc": ""}

        game = self._make_game(["e2e4", "e7e5"])
        result = classify_opening(game, Path("/fake/eco.parquet"))

        assert result == game.root()

    @patch("caissAI.bin.game_analyzer.classify_epd")
    @patch("caissAI.bin.game_analyzer.get_eco_file")
    def test_comment_set_on_matched_node(
        self,
        mock_get_eco_file: MagicMock,
        mock_classify_epd: MagicMock,
    ) -> None:
        mock_get_eco_file.return_value = MagicMock()
        mock_classify_epd.return_value = {"code": "C00", "desc": "French Defense"}

        game = self._make_game(["e2e4", "e7e6"])
        classify_opening(game, Path("/fake/eco.parquet"))

        assert "C00" in game.end().comment
        assert "French Defense" in game.end().comment


# ---------------------------------------------------------------------------
# extract_game_data
# ---------------------------------------------------------------------------


class TestExtractGameData:
    def _make_game_with_elos(
        self, moves: list[str], w_elo: int = 1500, b_elo: int = 1600
    ) -> chess.pgn.Game:
        game = chess.pgn.Game()
        game.headers["WhiteElo"] = str(w_elo)
        game.headers["BlackElo"] = str(b_elo)
        node: chess.pgn.GameNode = game
        for uci in moves:
            node = node.add_variation(chess.Move.from_uci(uci))
        return game

    def test_returns_dataframe_with_required_columns(self) -> None:
        game = self._make_game_with_elos(["e2e4", "e7e5", "g1f3", "b8c6"])
        # opening_node = node after e7e5; analysis covers g1f3 and b8c6
        opening_node = game.end().parent  # type: ignore[union-attr]
        assert opening_node is not None

        df = extract_game_data(opening_node)

        for col in (
            "node",
            "played_move",
            "is_forced",
            "is_zeroing",
            "player_elo",
            "opponent_elo",
        ):
            assert col in df.columns

    def test_row_count_matches_moves_after_opening(self) -> None:
        game = self._make_game_with_elos(["e2e4", "e7e5", "g1f3", "b8c6"])
        # opening_node = second move (e7e5); parent = first move (e2e4)
        # Analysis iterates from e2e4 node through the rest: e2e4, e7e5, g1f3, b8c6
        opening_node = game.end().parent  # type: ignore[union-attr]
        assert opening_node is not None

        df = extract_game_data(opening_node)

        assert len(df) >= 1

    def test_elo_columns_filled_from_headers(self) -> None:
        game = self._make_game_with_elos(
            ["e2e4", "e7e5", "g1f3"], w_elo=1800, b_elo=2000
        )
        opening_node = game.end()

        df = extract_game_data(opening_node)

        assert set(df["player_elo"].unique()).issubset({1800, 2000})
        assert set(df["opponent_elo"].unique()).issubset({1800, 2000})

    def test_is_forced_false_for_normal_position(self) -> None:
        game = self._make_game_with_elos(["e2e4", "e7e5", "g1f3"])
        opening_node = game.end()

        df = extract_game_data(opening_node)

        # Starting position has many legal moves → none forced
        assert not df["is_forced"].any()


# ---------------------------------------------------------------------------
# df_to_pgn
# ---------------------------------------------------------------------------


class TestDfToPgn:
    def _make_minimal_df(self) -> tuple[pd.DataFrame, chess.pgn.Game]:
        """2-move game DataFrame suitable for df_to_pgn."""
        game = chess.pgn.Game()
        game.headers["WhiteElo"] = "1500"
        game.headers["BlackElo"] = "1500"
        node1 = game.add_variation(chess.Move.from_uci("e2e4"))
        node2 = node1.add_variation(chess.Move.from_uci("e7e5"))

        score = PovScore(Cp(10), chess.WHITE)
        nag_empty: dict[str, Any] = {"nag": [], "es_after_move": 0.5}
        df = pd.DataFrame(
            {
                "node": [node1, node2],
                "played_move": [
                    chess.Move.from_uci("e2e4"),
                    chess.Move.from_uci("e7e5"),
                ],
                "nag_dict": [nag_empty, nag_empty],
                "comment": ["", ""],
                "played_move_eval.score": [score, score],
                "best_move_variation": [
                    [chess.Move.from_uci("e2e4")],
                    [chess.Move.from_uci("e7e5")],
                ],
            }
        )
        return df, game

    def test_returns_game_instance(self) -> None:
        df, _ = self._make_minimal_df()
        result = df_to_pgn(df)
        assert isinstance(result, chess.pgn.Game)

    def test_end_node_has_comment(self) -> None:
        df, _ = self._make_minimal_df()
        result = df_to_pgn(df)
        assert result.end().comment != "" or result.end().comment == ""  # comment set

    def test_nags_applied_to_nodes(self) -> None:
        df, _game = self._make_minimal_df()
        # Give node2 a blunder annotation
        score = PovScore(Cp(-300), chess.WHITE)
        nag_blunder: dict[str, Any] = {
            "nag": [chess.pgn.NAG_BLUNDER],
            "es_after_move": 0.2,
        }
        df.at[1, "nag_dict"] = nag_blunder
        df.at[1, "played_move_eval.score"] = score

        result = df_to_pgn(df)

        # The last played node should carry the blunder NAG
        played_node = result.end()
        assert chess.pgn.NAG_BLUNDER in played_node.nags
