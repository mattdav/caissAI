"""Unit tests for caissAI.bin.utils pure and I/O functions."""

import io
import math
from pathlib import Path

import chess
import chess.pgn
import pandas as pd
import pytest

from caissAI.bin.utils import (
    classify_epd,
    clean_game,
    clean_previous_variations,
    get_advantage,
    get_eco_file,
    get_es,
    get_nag_advantage,
    read_pgn,
    set_game_length,
    time_logger,
    write_game_to_pgn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_game(moves: list[str]) -> chess.pgn.Game:
    """Build a chess.pgn.Game from a list of UCI move strings."""
    game = chess.pgn.Game()
    node: chess.pgn.GameNode = game
    for uci in moves:
        node = node.add_variation(chess.Move.from_uci(uci))
    return game


# ---------------------------------------------------------------------------
# get_es
# ---------------------------------------------------------------------------


class TestGetEs:
    def test_equal_position_returns_half(self) -> None:
        """At centipawns=0, expected score must be exactly 0.5."""
        assert get_es(1500, 0) == pytest.approx(0.5, abs=1e-9)

    def test_positive_eval_above_half(self) -> None:
        assert get_es(1500, 200) > 0.5

    def test_negative_eval_below_half(self) -> None:
        assert get_es(1500, -200) < 0.5

    def test_result_always_in_unit_interval(self) -> None:
        for cp in [-10000, -500, -1, 0, 1, 500, 10000]:
            result = get_es(2000, cp)
            assert 0.0 <= result <= 1.0

    def test_symmetry_sums_to_one(self) -> None:
        """get_es(elo, +cp) + get_es(elo, -cp) == 1."""
        assert get_es(1500, 300) + get_es(1500, -300) == pytest.approx(1.0, abs=1e-9)

    def test_higher_elo_steeper_curve(self) -> None:
        """Higher ELO → coefficient more negative → ES further from 0.5 for same eval.

        Oracle's coefficient = elo * -0.00000274 + 0.00048 becomes more negative as
        ELO grows past the inflection point (~1752), amplifying the exponent.
        """
        es_low = get_es(800, 300)
        es_high = get_es(2500, 300)
        assert abs(es_high - 0.5) > abs(es_low - 0.5)

    def test_matches_oracle_formula(self) -> None:
        elo, cp = 1500, 100
        coefficient = elo * -0.00000274 + 0.00048
        expected = 0.5 + 0.5 * (2 / (1 + math.exp(coefficient * cp)) - 1)
        assert get_es(elo, cp) == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# get_advantage
# ---------------------------------------------------------------------------


class TestGetAdvantage:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (-10000, -4),
            (-300, -4),
            (-301, -4),
            (-299, -3),
            (-150, -3),
            (-151, -3),
            (-149, -2),
            (-75, -2),
            (-76, -2),
            (-74, -1),
            (-25, -1),
            (-26, -1),
            (-24, 0),
            (0, 0),
            (25, 0),
            (26, 1),
            (75, 1),
            (76, 2),
            (150, 2),
            (151, 3),
            (300, 3),
            (301, 4),
            (10000, 4),
        ],
    )
    def test_boundary(self, score: int, expected: int) -> None:
        assert get_advantage(score) == expected


# ---------------------------------------------------------------------------
# get_nag_advantage
# ---------------------------------------------------------------------------


class TestGetNagAdvantage:
    @pytest.mark.parametrize(
        "advantage,expected_nag",
        [
            (-5, chess.pgn.NAG_BLACK_DECISIVE_ADVANTAGE),
            (-4, chess.pgn.NAG_BLACK_DECISIVE_ADVANTAGE),
            (-3, chess.pgn.NAG_BLACK_DECISIVE_ADVANTAGE),
            (-2, chess.pgn.NAG_BLACK_MODERATE_ADVANTAGE),
            (-1, chess.pgn.NAG_BLACK_SLIGHT_ADVANTAGE),
            (0, chess.pgn.NAG_DRAWISH_POSITION),
            (1, chess.pgn.NAG_WHITE_SLIGHT_ADVANTAGE),
            (2, chess.pgn.NAG_WHITE_MODERATE_ADVANTAGE),
            (3, chess.pgn.NAG_WHITE_DECISIVE_ADVANTAGE),
            (4, chess.pgn.NAG_WHITE_DECISIVE_ADVANTAGE),
        ],
    )
    def test_nag_mapping(self, advantage: int, expected_nag: int) -> None:
        assert get_nag_advantage(advantage) == expected_nag


# ---------------------------------------------------------------------------
# classify_epd
# ---------------------------------------------------------------------------


@pytest.fixture
def df_eco() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "epd": [
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq",
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq",
            ],
            "eco": ["B00", "C20"],
            "name": ["King's Pawn Opening", "King's Pawn Game"],
        }
    )


class TestClassifyEpd:
    def test_known_epd_returns_code_and_desc(self, df_eco: pd.DataFrame) -> None:
        result = classify_epd(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq", df_eco
        )
        assert result["code"] == "B00"
        assert result["desc"] == "King's Pawn Opening"

    def test_second_known_epd(self, df_eco: pd.DataFrame) -> None:
        result = classify_epd(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", df_eco
        )
        assert result["code"] == "C20"
        assert result["desc"] == "King's Pawn Game"

    def test_unknown_epd_returns_empty_strings(self, df_eco: pd.DataFrame) -> None:
        result = classify_epd("not_a_real_epd_xyz", df_eco)
        assert result["code"] == ""
        assert result["desc"] == ""

    def test_result_always_has_code_and_desc_keys(self, df_eco: pd.DataFrame) -> None:
        result = classify_epd("any_string", df_eco)
        assert "code" in result
        assert "desc" in result


# ---------------------------------------------------------------------------
# set_game_length
# ---------------------------------------------------------------------------


class TestSetGameLength:
    def test_empty_game_is_zero(self) -> None:
        game = chess.pgn.Game()
        assert set_game_length(game) == 0

    def test_one_move_game(self) -> None:
        game = _make_game(["e2e4"])
        assert set_game_length(game) == 1

    def test_four_ply_game(self) -> None:
        game = _make_game(["e2e4", "e7e5", "g1f3", "b8c6"])
        assert set_game_length(game) == 4

    def test_six_ply_game(self) -> None:
        game = _make_game(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"])
        assert set_game_length(game) == 6


# ---------------------------------------------------------------------------
# clean_game
# ---------------------------------------------------------------------------


class TestCleanGame:
    def _annotated_game(self) -> chess.pgn.Game:
        game = chess.pgn.Game()
        node = game.add_variation(chess.Move.from_uci("e2e4"))
        node.comment = "Excellent opening move"
        node.nags = {chess.pgn.NAG_GOOD_MOVE}
        child = node.add_variation(chess.Move.from_uci("e7e5"))
        child.comment = "Symmetric response"
        # Side variation on the first move
        node.add_variation(chess.Move.from_uci("d2d4"))
        return game

    def test_comments_are_removed(self) -> None:
        cleaned = clean_game(self._annotated_game())
        node = cleaned.next()
        assert node is not None
        assert node.comment == ""
        child = node.next()
        assert child is not None
        assert child.comment == ""

    def test_nags_are_removed(self) -> None:
        cleaned = clean_game(self._annotated_game())
        node = cleaned.next()
        assert node is not None
        assert len(node.nags) == 0

    def test_side_variations_are_removed(self) -> None:
        cleaned = clean_game(self._annotated_game())
        node = cleaned.next()
        assert node is not None
        # Only the main variation should remain
        assert len(node.variations) == 1

    def test_main_line_is_preserved(self) -> None:
        game = _make_game(["e2e4", "e7e5", "g1f3"])
        cleaned = clean_game(game)
        assert set_game_length(cleaned) == 3


# ---------------------------------------------------------------------------
# clean_previous_variations
# ---------------------------------------------------------------------------


class TestCleanPreviousVariations:
    def test_removes_side_variation_before_target_node(self) -> None:
        game = chess.pgn.Game()
        node1 = game.add_variation(chess.Move.from_uci("e2e4"))
        # Main variation must be added first so that next() follows it
        node2 = node1.add_variation(chess.Move.from_uci("e7e5"))
        # Side variation added after
        node1.add_variation(chess.Move.from_uci("c7c5"))
        assert len(node1.variations) == 2  # sanity check before clean

        result = clean_previous_variations(node2)

        assert len(node1.variations) == 1
        assert result == node2

    def test_returns_target_node_unchanged(self) -> None:
        game = chess.pgn.Game()
        node = game.add_variation(chess.Move.from_uci("d2d4"))
        result = clean_previous_variations(node)
        assert result == node

    def test_target_node_variations_are_untouched(self) -> None:
        """Side variations on the target node itself are not cleaned."""
        game = chess.pgn.Game()
        node = game.add_variation(chess.Move.from_uci("e2e4"))
        node.add_variation(chess.Move.from_uci("e7e5"))
        node.add_variation(chess.Move.from_uci("c7c5"))

        result = clean_previous_variations(node)

        # Variations on the target node are preserved
        assert len(result.variations) == 2


# ---------------------------------------------------------------------------
# read_pgn
# ---------------------------------------------------------------------------


class TestReadPgn:
    def test_single_game(self, tmp_path: Path) -> None:
        pgn_content = (
            '[Event "Test"]\n'
            '[White "Player1"]\n'
            '[Black "Player2"]\n'
            '[WhiteElo "1500"]\n'
            '[BlackElo "1500"]\n'
            "\n"
            "1. e4 e5 2. Nf3 Nc6 *\n"
            "\n"
        )
        pgn_file = tmp_path / "single.pgn"
        pgn_file.write_text(pgn_content, encoding="utf-8")

        result = read_pgn(pgn_file)

        assert len(result) == 1
        assert isinstance(result[0], io.StringIO)
        # The StringIO content should be parseable as PGN
        game = chess.pgn.read_game(result[0])
        assert game is not None
        assert game.headers["White"] == "Player1"

    def test_two_games_in_one_file(self, tmp_path: Path) -> None:
        pgn_content = '[Event "Game 1"]\n\n1. e4 *\n\n[Event "Game 2"]\n\n1. d4 *\n\n'
        pgn_file = tmp_path / "two_games.pgn"
        pgn_file.write_text(pgn_content, encoding="utf-8")

        result = read_pgn(pgn_file)

        assert len(result) == 2

    def test_returns_list_of_stringio(self, tmp_path: Path) -> None:
        pgn_content = '[Event "X"]\n\n1. Nf3 *\n\n'
        pgn_file = tmp_path / "minimal.pgn"
        pgn_file.write_text(pgn_content, encoding="utf-8")

        result = read_pgn(pgn_file)

        assert all(isinstance(item, io.StringIO) for item in result)

    def test_file_without_trailing_newline(self, tmp_path: Path) -> None:
        """Branch: last line is not \\n → a newline is appended before splitting."""
        pgn_content = '[Event "Y"]\n\n1. e4 *'  # no trailing newline
        pgn_file = tmp_path / "no_newline.pgn"
        pgn_file.write_text(pgn_content, encoding="utf-8")

        result = read_pgn(pgn_file)

        assert len(result) == 1


# ---------------------------------------------------------------------------
# time_logger
# ---------------------------------------------------------------------------


class TestTimeLogger:
    def test_wrapped_function_returns_correct_value(self) -> None:
        @time_logger
        def double(x: int) -> int:
            return x * 2

        assert double(7) == 14

    def test_wrapper_passes_arguments_through(self) -> None:
        @time_logger
        def add(a: int, b: int) -> int:
            return a + b

        assert add(3, 4) == 7


# ---------------------------------------------------------------------------
# write_game_to_pgn
# ---------------------------------------------------------------------------


class TestWriteGameToPgn:
    def test_creates_new_file_with_annotated_name(self, tmp_path: Path) -> None:
        write_game_to_pgn(tmp_path, "mygame.pgn", "1. e4 e5 *")

        expected = tmp_path / "mygame_vCaïssAI.pgn"
        assert expected.exists()
        assert "1. e4 e5 *" in expected.read_text(encoding="utf-8")

    def test_appends_to_existing_file(self, tmp_path: Path) -> None:
        write_game_to_pgn(tmp_path, "game.pgn", "1. d4 *")
        write_game_to_pgn(tmp_path, "game.pgn", "1. e4 *")

        content = (tmp_path / "game_vCaïssAI.pgn").read_text(encoding="utf-8")
        assert "1. d4 *" in content
        assert "1. e4 *" in content


# ---------------------------------------------------------------------------
# get_eco_file
# ---------------------------------------------------------------------------


class TestGetEcoFile:
    def test_returns_dataframe_from_parquet(self, tmp_path: Path) -> None:
        eco_path = tmp_path / "eco.parquet"
        df_in = pd.DataFrame(
            {"epd": ["abc", "def"], "eco": ["A00", "B00"], "name": ["Polish", "Owen"]}
        )
        df_in.to_parquet(eco_path)

        result = get_eco_file(eco_path)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["epd", "eco", "name"]
        assert len(result) == 2
