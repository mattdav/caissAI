"""Orchestration de l'annotation de parties d'échecs."""

import logging
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from beartype import beartype
from chess.pgn import Game, read_game
from maia2.main import MAIA2Model
from openai import OpenAI
from tqdm import tqdm

from caissAI.bin import llm_comments
from caissAI.bin.game_analyzer import process_game
from caissAI.bin.moments import annotated_plies, player_color, write_moments_json
from caissAI.bin.paths import resolve_data_dir
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

_Deferred = list[tuple[Game, llm_comments.PendingGame | None, Callable[[Game], None]]]


def _appender(output_path: Path) -> Callable[[Game], None]:
    """Fonction qui ajoute une partie annotée à `output_path`."""

    def write(game: Game) -> None:
        mode = "a" if output_path.exists() else "w"
        with output_path.open(mode, encoding="utf-8") as fh:
            fh.write(str(game) + "\n\n")
        print(f"       → écrite dans {output_path.name}")

    return write


def _write_classic(output_dir: Path, filename: str, game: Game) -> None:
    """Écrit une partie annotée dans le dossier `output/` (mode classique)."""
    write_game_to_pgn(output_dir, filename, str(game))


def _flush_batch(client: OpenAI, deferred: _Deferred) -> None:
    """Envoie le lot, applique les réponses puis écrit les parties dans l'ordre."""
    pendings = [p for _, p, _ in deferred if p is not None]
    texts = llm_comments.run_batch(client, pendings) if pendings else {}
    for game, pending, write in deferred:
        if pending is not None:
            llm_comments.finish_pending(game, pending, texts.get(pending.custom_id))
        write(game)


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
    moments_json: Path | None = None,
    deep_comments: int = 0,
    batch: bool = False,
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
        moments_json: Si renseigné, fichier JSON où écrire les dossiers de faits
            des moments clés ; le PGN produit est inchangé.
        deep_comments: Nombre de moments commentés par le LLM (un appel par
            partie, résumé compris). 0 : résumé seul, comportement historique.
        batch: Si True, envoie les parties par l'API Batch d'OpenAI et écrit
            les résultats une fois le lot terminé.
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
    moments_games: list[dict[str, Any]] = []
    cache_dir = resolve_data_dir() / "cache"
    deferred: _Deferred = []
    write_game = _appender(output_path)

    for idx, game in tqdm(selected):
        white = game.headers.get("White", "?")
        black = game.headers.get("Black", "?")
        print(f"\n  [{idx}] {white} vs {black}")
        pending: list[llm_comments.PendingGame] | None = [] if batch else None
        try:
            game.headers["PlyCount"] = str(set_game_length(game))
            flagged = annotated_plies(game)
            cleaned = clean_game(game)
            moments: list[dict[str, Any]] | None = (
                [] if moments_json is not None else None
            )
            annotated = process_game(
                cleaned,
                config_path,
                engine_path,
                maia2_model,
                n_workers,
                openai_client,
                comment,
                openai_model,
                moments_out=moments,
                player_color=player_color(game, player),
                annotated=flagged,
                deep_comments=deep_comments,
                cache_dir=cache_dir,
                batch_out=pending,
            )
            if moments is not None:
                moments_games.append(
                    {"partie": idx, "blancs": white, "noirs": black, "moments": moments}
                )
            annotated.headers["Annotator"] = "CaïssAI"
            if batch:
                deferred.append(
                    (annotated, pending[-1] if pending else None, write_game)
                )
            else:
                write_game(annotated)
        except Exception as exc:
            _log.warning("Partie [%d] ignorée : %s", idx, exc)
            print(f"       → ignorée ({exc})")

    if batch:
        _flush_batch(openai_client, deferred)
    if moments_json is not None:
        write_moments_json(moments_json, moments_games)
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
    moments_json: Path | None = None,
    deep_comments: int = 0,
    batch: bool = False,
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
        moments_json: Si renseigné, fichier JSON où écrire les dossiers de faits
            des moments clés ; le PGN produit est inchangé.
        deep_comments: Nombre de moments commentés par le LLM (un appel par
            partie, résumé compris). 0 : résumé seul, comportement historique.
        batch: Si True, envoie les parties par l'API Batch d'OpenAI et écrit
            les résultats une fois le lot terminé.
    """
    moments_games: list[dict[str, Any]] = []
    deferred: _Deferred = []
    for filename in os.listdir(data_path / "input"):
        if filename.endswith(".pgn"):
            games = read_pgn(data_path / "input" / filename)
            for game_io in tqdm(games):
                if not game_io.errors:
                    pgn_game = read_game(game_io)
                    assert pgn_game is not None
                    pgn_game.headers["PlyCount"] = str(set_game_length(pgn_game))
                    flagged = annotated_plies(pgn_game)
                    pgn_game = clean_game(pgn_game)
                    moments: list[dict[str, Any]] | None = (
                        [] if moments_json is not None else None
                    )
                    pending: list[llm_comments.PendingGame] | None = (
                        [] if batch else None
                    )
                    commented_game = process_game(
                        pgn_game,
                        config_path,
                        engine_path,
                        maia2_model,
                        n_workers,
                        openai_client,
                        comment,
                        openai_model,
                        moments_out=moments,
                        annotated=flagged,
                        deep_comments=deep_comments,
                        cache_dir=data_path / "cache",
                        batch_out=pending,
                    )
                    if moments is not None:
                        moments_games.append(
                            {
                                "fichier": filename,
                                "blancs": pgn_game.headers.get("White", "?"),
                                "noirs": pgn_game.headers.get("Black", "?"),
                                "moments": moments,
                            }
                        )
                    pgn_game.headers["Annotator"] = "CaïssAI"
                    write_game = partial(_write_classic, data_path / "output", filename)
                    if batch:
                        deferred.append(
                            (
                                commented_game,
                                pending[-1] if pending else None,
                                write_game,
                            )
                        )
                    else:
                        write_game(commented_game)
    if batch:
        _flush_batch(openai_client, deferred)
    if moments_json is not None:
        write_moments_json(moments_json, moments_games)


__all__ = ["comment_game_from_input_dir", "comment_games_from_file", "list_games"]
