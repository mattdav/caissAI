"""Entry point for the caissAI CLI."""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import psutil
from dotenv import find_dotenv, load_dotenv
from maia2 import model
from openai import OpenAI

from caissAI.bin.annotator import comment_game_from_input_dir, comment_games_from_file
from caissAI.bin.paths import resolve_data_dir, resolve_log_dir, resolve_pgn_path
from caissAI.bin.utils import get_folder_path, list_games, read_games_from_file

logger = logging.getLogger(__name__)


def main() -> None:  # pragma: no cover
    """Entry point for the caissAI CLI."""
    env_file = find_dotenv(usecwd=True) or str(
        Path(__file__).parent.parent.parent / ".env"
    )
    load_dotenv(env_file)

    config_path = get_folder_path("config")
    log_path = resolve_log_dir()
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename=log_path / f"app_{timestamp}.log",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

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
            'Ex : --pgn "C:/ChessBase/mes parties.pgn". '
            "Repli sur la variable CAISSAI_PGN si absent."
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

    pgn_path = resolve_pgn_path(args.pgn)

    # ── Mode --list : ne nécessite ni OpenAI ni Stockfish, on sort tôt ────
    if args.list:
        if pgn_path is None:
            parser.error("--list requiert --pgn (ou la variable CAISSAI_PGN).")
        if not pgn_path.is_file():
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

    # ── Mode --pgn (ou CAISSAI_PGN) ──────────────────────────────────────────
    if pgn_path is not None:
        if not pgn_path.is_file():
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
        data_path = resolve_data_dir()
        (data_path / "input").mkdir(parents=True, exist_ok=True)
        (data_path / "output").mkdir(parents=True, exist_ok=True)
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
