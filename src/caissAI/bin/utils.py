import bisect
import cProfile
import io
import logging
import math
import pstats
from collections.abc import Callable
from pathlib import Path
from typing import cast

import chess.pgn
import pandas as pd
from beartype import beartype
from chess.pgn import Game, GameNode
from memoization import cached

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lookup tables for advantage classification (#16)
# ---------------------------------------------------------------------------

# Upper bound (inclusive) of each advantage interval, sorted ascending.
# bisect_left(thresholds, score) gives the index in _ADV_VALUES.
_ADV_THRESHOLDS: list[int] = [-300, -150, -75, -25, 25, 75, 150, 300]
_ADV_VALUES: list[int] = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

# NAG mapping for the [-2, 2] range; extremes are handled by guards.
_NAG_ADVANTAGE_MAP: dict[int, int] = {
    -2: chess.pgn.NAG_BLACK_MODERATE_ADVANTAGE,
    -1: chess.pgn.NAG_BLACK_SLIGHT_ADVANTAGE,
    0: chess.pgn.NAG_DRAWISH_POSITION,
    1: chess.pgn.NAG_WHITE_SLIGHT_ADVANTAGE,
    2: chess.pgn.NAG_WHITE_MODERATE_ADVANTAGE,
}


def time_logger[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Log the execution time of a function.

    Args:
        func (function): The function to log.

    Returns:
        function: The resulting function.
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs().sort_stats("cumulative").print_stats(10)
        _log.debug(stream.getvalue())
        return result

    return wrapper


@beartype
def read_pgn(filepath: Path) -> list[io.StringIO]:
    """Read a Portable Game Notation (PGN) file and return a list with its games.

    Args:
        filepath (Path): Path to the PGN file.

    Returns:
        list[io.StringIO]: List of the games in PGN format.
    """
    with open(filepath, encoding="utf-8") as pgn:
        content_list = pgn.readlines()
    if content_list[-1] != "\n":
        content_list.append("\n")
    list_games: list[io.StringIO] = []
    game: list[str] = []
    j = 0
    for i in content_list:
        if i == "\n":
            j += 1
        if j == 2:
            j = 0
            list_games.append(io.StringIO("".join(game)))
            game = []
        game.append(i)
    return list_games


@beartype
def write_game_to_pgn(file_path: Path, filename: str, game: str) -> None:
    """Write a game to an output PGN file.

    Args:
        file_path (Path): Path to the output folder.
        filename (str): Filename of the input PGN file to rename.
        game (str): Game to write.
    """
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    analysed_path = file_path / f"{stem}_vCaïssAI{suffix}"
    mode = "a" if analysed_path.exists() else "w"
    with open(analysed_path, mode, encoding="utf-8") as f:
        f.write(game + "\n")


@beartype
def clean_game(game: Game) -> Game:
    """Takes a game and strips all comments and variations, returning the
    cleaned game.

    Args:
        game (Game): The game to clean.

    Returns:
        Game: The cleaned game.
    """
    node = game.end()
    while node != game.root():
        node.comment = ""
        node.nags = set()
        for variation in reversed(node.variations):
            if not variation.is_main_variation():
                node.remove_variation(variation)
        assert node.parent is not None
        node = node.parent
    return cast(Game, node.root())


@beartype
def clean_previous_variations(node: GameNode) -> GameNode:
    """Clean a all previous variations from a specific node.

    Args:
        node (GameNode): The node to clean.

    Returns:
        GameNode: The cleaned node.
    """
    new_node: GameNode = node.game()
    while new_node != node:
        for variation in reversed(new_node.variations):
            if not variation.is_main_variation():
                new_node.remove_variation(variation)
        next_node = new_node.next()
        assert next_node is not None
        new_node = next_node
    return new_node


@beartype
def set_game_length(game: Game) -> int:
    """Takes a game and compute its ply count.

    Args:
        game (Game): The game to process.

    Returns:
        int: Ply count.
    """
    ply_count = 0
    node = game.end()
    while not node == game.root():
        assert node.parent is not None
        node = node.parent
        ply_count += 1
    return ply_count


@beartype
def get_eco_file(eco_path: Path) -> pd.DataFrame:
    """Load the Lichess Encyclopaedia of Chess Openings (ECO) database in a dataframe.

    Args:
        eco_path (Path): Path to the eco file.

    Returns:
        pd.DataFrame: Dataframe containing the Lichess ECO database.
    """
    df_eco = pd.read_parquet(eco_path)
    return df_eco


@cached
def classify_epd(epd: str, df_eco: pd.DataFrame) -> dict[str, str]:
    """Searches the Lichess eco database to check if the given
    Extended Position Description (EPD) matches an existing opening record.

    Args:
        epd (str): The position to classify.
        df_eco (pd.DataFrame): Dataframe containing the Lichess ECO database.

    Returns:
        dict: A dictionary containing the following elements :
        "code": The ECO code of the matched opening.
        "desc": The long description of the matched opening.
    """

    classification = {}
    classification["code"] = ""
    classification["desc"] = ""
    opening = df_eco.loc[df_eco["epd"] == epd]
    if len(opening) > 0:
        classification["code"] = opening.iloc[0]["eco"]
        classification["desc"] = opening.iloc[0]["name"]
    return classification


@cached
def get_es(player_elo: int, centipawns: int) -> float:
    """Compute the expected score from the evaluation in centipawns
    and the player's rating according to Oracle's formula :
    https://yoshachess.com/fr/article/oracle-le-meilleur-module-dechecs-jouant-comme-un-humain/#evaluations-in-chess

    Args:
        player_elo (int): The player's ELO.
        centipawns (int): The evaluation in centipawns.

    Returns:
        float: The expected score.
    """
    coefficient = player_elo * -0.00000274 + 0.00048
    exponent = coefficient * centipawns
    expected_score = 0.5 + (0.5 * (2 / (1 + math.exp(exponent)) - 1))
    return expected_score


@cached
def get_advantage(score: int) -> int:
    """Return the degree of advantage according to the score
    as defined here :
    https://chessdream.app/blogs/advanced-stockfish-analysis

    Args:
        score (int): The current score from white's perspective in centipawns.

    Returns:
        int: The resulting advantage :
        0: almost equality
        1: slight advantage
        2: clear advantage
        3: decisive advantage
        4: winning position
        Negative value if black has the advantage.
    """
    return _ADV_VALUES[bisect.bisect_left(_ADV_THRESHOLDS, score)]


@beartype
def get_nag_advantage(advantage: int) -> int:
    """Return the NAG of according to the advantage magnitude.

    Args:
        advantage (int): The advantage magnitude from the
        get_advantage function.

    Returns:
        int: The according NAG.
    """
    if advantage <= -3:
        return chess.pgn.NAG_BLACK_DECISIVE_ADVANTAGE
    if advantage >= 3:
        return chess.pgn.NAG_WHITE_DECISIVE_ADVANTAGE
    return _NAG_ADVANTAGE_MAP[advantage]
