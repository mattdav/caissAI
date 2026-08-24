"""Orchestration de l'annotation de parties d'échecs."""

import logging
import os
from pathlib import Path

from beartype import beartype
from chess.pgn import read_game
from maia2.main import MAIA2Model
from openai import OpenAI
from tqdm import tqdm

from caissAI.bin.game_analyzer import process_game
from caissAI.bin.utils import (
    clean_game,
    list_games,
    read_games_from_file,
    read_pgn,
    select_games,
    set_game_length,
    time_logger,
    write_game_to_pgn,
)

_log = logging.getLogger(__name__)


@beartype
@time_logger
def comment_games_from_file(
    pgn_path: Path,
    output_path: Path,
    config_path: Path,
    engine_path: Path,
    maia2_model: MAIA2Model,
    n_workers: int,
    openai_client: OpenAI,
    comment: bool,
    openai_model: str,
    indices: list[int] | None = None,
    player: str | None = None,
) -> None:
    """Annote des parties sélectionnées depuis un fichier PGN arbitraire.

    Args:
        pgn_path: Chemin vers le fichier PGN source (peut contenir N parties).
        output_path: Fichier de sortie PGN pour les parties annotées.
        config_path: Chemin vers le dossier config/ de caissAI.
        engine_path: Chemin vers l'exécutable Stockfish.
        maia2_model: Modèle Maia2 initialisé.
        n_workers: Nombre de workers CPU.
        openai_client: Client OpenAI.
        comment: Si True, demande un résumé GPT de chaque partie.
        openai_model: Modèle OpenAI à utiliser pour le résumé.
        indices: Indices 1-basés des parties à traiter. None = toutes.
        player: Filtre sur le nom du joueur (correspondance partielle).
    """
    all_games = read_games_from_file(pgn_path)
    if not all_games:
        print(f"Aucune partie trouvée dans {pgn_path}.")
        return

    selected = select_games(all_games, indices=indices, player=player)
    if not selected:
        print("Aucune partie ne correspond aux critères de sélection.")
        return

    print(
        f"{len(selected)}/{len(all_games)} partie(s) sélectionnée(s) "
        f"dans {pgn_path.name}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, game in tqdm(selected):
        white = game.headers.get("White", "?")
        black = game.headers.get("Black", "?")
        print(f"\n  [{idx}] {white} vs {black}")
        try:
            game.headers["PlyCount"] = str(set_game_length(game))
            cleaned = clean_game(game)
            annotated = process_game(
                cleaned,
                config_path,
                engine_path,
                maia2_model,
                n_workers,
                openai_client,
                comment,
                openai_model,
            )
            annotated.headers["Annotator"] = "CaïssAI"
            mode = "a" if output_path.exists() else "w"
            with output_path.open(mode, encoding="utf-8") as fh:
                fh.write(str(annotated) + "\n\n")
            print(f"       → écrite dans {output_path.name}")
        except Exception as exc:
            _log.warning("Partie [%d] ignorée : %s", idx, exc)
            print(f"       → ignorée ({exc})")

    print(f"\nTerminé. Parties annotées dans : {output_path}")


@beartype
@time_logger
def comment_game_from_input_dir(
    data_path: Path,
    config_path: Path,
    engine_path: Path,
    maia2_model: MAIA2Model,
    n_workers: int,
    openai_client: OpenAI,
    comment: bool,
    openai_model: str,
) -> None:
    """Mode classique : annote tous les PGN du répertoire input/.

    Args:
        data_path: Chemin vers le dossier data/ du package.
        config_path: Chemin vers le dossier config/ de caissAI.
        engine_path: Chemin vers l'exécutable Stockfish.
        maia2_model: Modèle Maia2 initialisé.
        n_workers: Nombre de workers CPU.
        openai_client: Client OpenAI.
        comment: Si True, demande un résumé GPT de chaque partie.
        openai_model: Modèle OpenAI à utiliser pour le résumé.
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
                        openai_model,
                    )
                    pgn_game.headers["Annotator"] = "CaïssAI"
                    write_game_to_pgn(
                        data_path / "output", filename, str(commented_game)
                    )


__all__ = ["comment_game_from_input_dir", "comment_games_from_file", "list_games"]
