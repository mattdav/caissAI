"""Main module used to gather all informations and launch the submodules."""

import argparse
import io
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import psutil
from beartype import beartype
from chess.pgn import Game, read_game
from dotenv import load_dotenv
from maia2 import model
from maia2.main import MAIA2Model
from openai import OpenAI
from tqdm import tqdm

from caissAI.bin.game_analyzer import process_game
from caissAI.bin.utils import (
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
def read_games_from_file(pgn_path: Path) -> list[Game]:
    """Lit toutes les parties d'un fichier PGN multi-parties.

    Args:
        pgn_path: Chemin vers le fichier PGN.

    Returns:
        Liste des parties valides dans l'ordre de lecture.
    """
    games: list[Game] = []
    try:
        with pgn_path.open(encoding="utf-8", errors="replace") as fh:
            content = fh.read()
    except OSError as exc:
        logger.error("Impossible de lire %s : %s", pgn_path, exc)
        return games

    stream = io.StringIO(content)
    while True:
        game = read_game(stream)
        if game is None:
            break
        games.append(game)

    return games


@beartype
def select_games(
    games: list[Game],
    indices: list[int] | None = None,
    player: str | None = None,
) -> list[tuple[int, Game]]:
    """Filtre et sélectionne des parties depuis une liste.

    Args:
        games: Toutes les parties du fichier.
        indices: Indices (1-basés) des parties à traiter. None = toutes.
        player: Si fourni, ne retient que les parties où ce joueur apparaît
            (correspondance partielle, insensible à la casse) dans White ou Black.

    Returns:
        Liste de (index_1_basé, partie) pour les parties sélectionnées.
    """
    selected: list[tuple[int, Game]] = []
    for i, game in enumerate(games, start=1):
        if indices is not None and i not in indices:
            continue
        if player:
            white = game.headers.get("White", "").lower()
            black = game.headers.get("Black", "").lower()
            if player.lower() not in white and player.lower() not in black:
                continue
        selected.append((i, game))
    return selected


@beartype
def list_games(games: list[Game]) -> None:
    """Affiche la liste numérotée des parties d'un fichier.

    Args:
        games: Parties à lister.
    """
    print(f"{len(games)} partie(s) trouvée(s) :\n")
    for i, game in enumerate(games, start=1):
        white = game.headers.get("White", "?")
        black = game.headers.get("Black", "?")
        date = game.headers.get("Date", "?")
        event = game.headers.get("Event", "?")
        result = game.headers.get("Result", "?")
        opening = game.headers.get("Opening", "")
        opening_str = f"  {opening}" if opening else ""
        print(f"  [{i:3d}] {white} vs {black}  {result}  {date}  {event}{opening_str}")


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
            logger.warning("Partie [%d] ignorée : %s", idx, exc)
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


def main() -> None:  # pragma: no cover
    """Entry point for the caissAI CLI."""
    # Cherche le .env en remontant depuis le répertoire courant (comportement
    # standard de python-dotenv), puis depuis la racine du package en fallback.
    from dotenv import find_dotenv

    env_file = find_dotenv(usecwd=True) or str(
        Path(__file__).parent.parent.parent / ".env"
    )
    load_dotenv(env_file)

    config_path = get_folder_path("config")
    log_path = get_folder_path("log")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename=log_path / f"app_{timestamp}.log",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    # ── CLI — parsing en premier pour que --list n'exige pas de .env ──────
    parser = argparse.ArgumentParser(
        prog="caissAI",
        description=(
            "Annote des parties d'échecs avec Maia2 + Stockfish + GPT.\n\n"
            "Sans --pgn : traite tous les fichiers PGN du dossier data/input/.\n"
            "Avec --pgn  : traite les parties sélectionnées du fichier indiqué."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pgn",
        metavar="CHEMIN",
        default=None,
        help=(
            "Fichier PGN source (peut contenir plusieurs parties). "
            'Ex : --pgn "C:/ChessBase/mes parties.pgn"'
        ),
    )
    parser.add_argument(
        "--output",
        metavar="CHEMIN",
        default=None,
        help=(
            "Fichier PGN de sortie pour les parties annotées. "
            "Défaut : le fichier source --pgn lui-même (les parties annotées "
            "sont ajoutées à la suite). "
            "Les parties sont ajoutées à la suite si le fichier existe déjà."
        ),
    )
    parser.add_argument(
        "--games",
        metavar="N",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Indices (1-basés) des parties à annoter. "
            "Ex : --games 3  ou  --games 1 5 12. "
            "Utilisez --list pour connaître les indices."
        ),
    )
    parser.add_argument(
        "--player",
        metavar="NOM",
        default=None,
        help=(
            "Filtre les parties par nom de joueur (correspondance partielle, "
            "insensible à la casse). Ex : --player Dupont"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Affiche la liste numérotée des parties du fichier --pgn et quitte.",
    )
    parser.add_argument(
        "--no-comment",
        action="store_true",
        help="Désactive le résumé GPT (annotation Stockfish + Maia2 uniquement).",
    )
    parser.add_argument(
        "--workers",
        metavar="N",
        type=int,
        default=None,
        help="Nombre de workers CPU (défaut : moitié des CPUs logiques).",
    )
    args = parser.parse_args()

    # ── Mode --list : ne nécessite ni OpenAI ni Stockfish, on sort tôt ────
    if args.list:
        if not args.pgn:
            parser.error("--list requiert --pgn.")
        pgn_path = Path(args.pgn)
        if not pgn_path.exists():
            parser.error(f"Fichier introuvable : {pgn_path}")
        list_games(read_games_from_file(pgn_path))
        sys.exit(0)

    # ── Validation des variables d'environnement (inutile pour --list) ────
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    stockfish_path_str = os.environ.get("STOCKFISH_PATH", "")
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-4.1")

    missing = []
    if not openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not stockfish_path_str:
        missing.append("STOCKFISH_PATH")
    if missing:
        parser.error(
            f"Variables d'environnement manquantes dans .env : {', '.join(missing)}"
        )

    engine_path = Path(stockfish_path_str)
    if not engine_path.exists():
        parser.error(f"Stockfish introuvable : {engine_path}")

    # ── Initialisation des ressources lourdes ──────────────────────────────
    n_workers = args.workers or int((psutil.cpu_count(logical=True) or 2) / 2)
    openai_client = OpenAI(api_key=openai_api_key)
    maia2_model = model.from_pretrained(type="blitz", device="cpu")
    comment = not args.no_comment

    # ── Mode --pgn ─────────────────────────────────────────────────────────
    if args.pgn:
        pgn_path = Path(args.pgn)
        if not pgn_path.exists():
            parser.error(f"Fichier introuvable : {pgn_path}")

        output_path = Path(args.output) if args.output else pgn_path

        comment_games_from_file(
            pgn_path=pgn_path,
            output_path=output_path,
            config_path=config_path,
            engine_path=engine_path,
            maia2_model=maia2_model,
            n_workers=n_workers,
            openai_client=openai_client,
            comment=comment,
            openai_model=openai_model,
            indices=args.games,
            player=args.player,
        )

    # ── Mode classique input/output ────────────────────────────────────────
    else:
        data_path = get_folder_path("data")
        comment_game_from_input_dir(
            data_path=data_path,
            config_path=config_path,
            engine_path=engine_path,
            maia2_model=maia2_model,
            n_workers=n_workers,
            openai_client=openai_client,
            comment=comment,
            openai_model=openai_model,
        )


if __name__ == "__main__":
    main()
