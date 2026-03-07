"""Main module used to gather all informations and launch the submodules."""

import configparser
import logging
import os
from datetime import datetime
from pathlib import Path

import psutil
from beartype import beartype
from chess.pgn import read_game
from maia2 import model
from maia2.main import MAIA2Model
from openai import OpenAI
from tqdm import tqdm

from .bin.game_analyzer import process_game
from .bin.utils import (
    clean_game,
    read_pgn,
    set_game_length,
    time_logger,
    write_game_to_pgn,
)

# Module-level logger: uses root configuration set up in main() (#2)
logger = logging.getLogger(__name__)


@beartype
def get_folder_path(folder_name: str) -> Path:
    """Get directory path of the package from its name.

    Uses __file__ to resolve sibling directories within the installed package,
    replacing the deprecated importlib.resources.path() API (#17).

    Args:
        folder_name (str): Name of the directory (e.g. "config", "data", "log").

    Raises:
        NameError: If the directory doesn't exist.

    Returns:
        Path: Path to the directory.
    """
    folder_path = Path(__file__).parent / folder_name
    if not folder_path.is_dir():
        logger.error("Le dossier %s n'existe pas.", folder_name)
        raise NameError(folder_name)
    return folder_path


@beartype
@time_logger
def comment_game(
    data_path: Path,
    config_path: Path,
    engine_path: Path,
    maia2_model: MAIA2Model,
    n_workers: int,
    openai_client: OpenAI,
    comment: bool,
) -> None:
    """Comment a clean version of the game, add infos to the header,
    classifying the opening and annotating and commenting each move.
    Write the commented game to an updated PGN file.

    Args:
        data_path (Path): Path to the data folder.
        config_path (Path): Path to the config folder.
        engine_path (Path): Path to the engine to use.
        maia2_model (MAIA2Model): The Maia2 model (https://www.maiachess.com/).
        n_workers (int): Number of CPU workers to use.
        openai_client (OpenAI): OpenAI API client to use.
        comment (bool): Whether to ask GPT for a comment of the game or not.
    """
    for filename in os.listdir(data_path / "input"):
        if filename.endswith(".pgn"):
            games = read_pgn(data_path / "input" / filename)
            for game_io in tqdm(games):
                if not game_io.errors:
                    pgn_game = read_game(game_io)
                    assert pgn_game is not None
                    pgn_game.headers["PlyCount"] = str(set_game_length(pgn_game))
                    pgn_game = clean_game(pgn_game)
                    commented_game = process_game(
                        pgn_game,
                        config_path,
                        engine_path,
                        maia2_model,
                        n_workers,
                        openai_client,
                        comment,
                    )
                    pgn_game.headers["Annotator"] = "CaïssAI"
                    write_game_to_pgn(
                        data_path / "output", filename, str(commented_game)
                    )


def main() -> None:  # pragma: no cover
    """Entry point for the caissAI CLI."""
    #: Path to the config directory
    config_path = get_folder_path("config")
    #: Path to the data directory
    data_path = get_folder_path("data")
    #: Path to the log directory
    log_path = get_folder_path("log")

    # Configure logging with a timestamped file so each run produces its own
    # log and previous runs are never overwritten (#timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename=log_path / f"app_{timestamp}.log",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    #: Loading config infos.
    config = configparser.ConfigParser()
    config.read(config_path / "config.cfg")
    engine_path = Path(config["ENGINE"]["path"])
    openai_client = OpenAI(api_key=config["OPENAI"]["api_key"])
    n_workers = int((psutil.cpu_count(logical=True) or 2) / 2)
    maia2_model = model.from_pretrained(type="blitz", device="cpu")

    COMMENT = True
    comment_game(
        data_path,
        config_path,
        engine_path,
        maia2_model,
        n_workers,
        openai_client,
        COMMENT,
    )


if __name__ == "__main__":
    main()
