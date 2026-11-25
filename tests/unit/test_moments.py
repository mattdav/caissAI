"""Tests de la sélection des moments clés et de l'export JSON."""

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import chess
import chess.engine
import chess.pgn
import pandas as pd
from chess.engine import Cp, PovScore
from maia2.main import MAIA2Model
from openai import OpenAI

from caissAI.bin import annotator, game_analyzer, moments
from caissAI.bin.moments import (
    MOMENTS_K,
    annotated_plies,
    is_decided,
    is_space_exchange,
    is_tactical,
    select_moments,
)
from caissAI.bin.position_facts import space

_PGN_316 = """[Event "Test"]
[White "Blanc"]
[Black "Noir"]
[WhiteElo "1500"]
[BlackElo "1500"]
[Result "*"]

1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Nf3 Be7 5.Bf4 O-O 6.e3 c5 7.Be2 Nc6 8.dxc5 Bxc5
9.cxd5 exd5 10.Qb3 b6 11.O-O Be6 12.Rfd1 Qe7 13.Nb5 a6 14.Nc3 Rfd8 *
"""


class _FakeEngine:
    """Moteur UCI factice : première coups légaux, évaluation constante."""

    def configure(self, options: dict[str, Any]) -> None:
        return None

    def quit(self) -> None:
        return None

    def analyse(
        self, board: chess.Board, limit: Any, multipv: int | None = None
    ) -> Any:
        moves = list(board.legal_moves)

        def info(move: chess.Move) -> dict[str, Any]:
            return {
                "score": PovScore(Cp(20), board.turn),
                "pv": [move],
                "depth": 20,
            }

        if multipv is None:
            return info(moves[0])
        return [info(m) for m in moves[:multipv]]


def _fake_move_probs(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    return df.assign(
        likeliest_move=[next(iter(n.parent.board().legal_moves)) for n in df["node"]],
        likeliest_move_proba=0.3,
        played_move_proba=0.2,
    )


def _fake_evaluate(df: pd.DataFrame, engine_path: Path, workers: int) -> pd.DataFrame:
    def score(node: Any, i: int) -> PovScore:
        return PovScore(Cp((i * 37) % 200 - 100), node.board().turn)

    variations = [[next(iter(n.board().legal_moves))] for n in df["node"]]
    columns: dict[str, Any] = {
        "played_move_eval.score": [score(n, i) for i, n in enumerate(df["node"])],
        "played_move_eval.variation": variations,
        "likeliest_move_eval.score": [
            score(n, i + 5) for i, n in enumerate(df["node"])
        ],
        "likeliest_move_eval.variation": variations,
    }
    return df.assign(**columns)


def _run(
    tmp_path: Path,
    name: str,
    moments_json: Path | None,
    pgn: str = _PGN_316,
) -> Path:
    source = tmp_path / "source.pgn"
    source.write_text(pgn, encoding="utf-8")
    output = tmp_path / name
    with (
        patch.object(game_analyzer, "get_move_probs", _fake_move_probs),
        patch.object(game_analyzer, "evaluate_game", _fake_evaluate),
        patch.object(
            chess.engine.SimpleEngine, "popen_uci", return_value=_FakeEngine()
        ),
    ):
        annotator.comment_games_from_file(
            pgn_path=source,
            output_path=output,
            config_path=Path(game_analyzer.__file__).parent.parent / "config",
            engine_path=Path("stockfish"),
            maia2_model=MAIA2Model(),
            n_workers=1,
            openai_client=OpenAI(api_key="test"),
            comment=False,
            openai_model="gpt-4.1",
            moments_json=moments_json,
        )
    return output


def test_fenetre_trois_coups() -> None:
    # Noirs aux coups 30 et 32 : une seule erreur retenue ; coup 33 : hors fenêtre.
    candidates = [
        {"ply": 60, "color": False, "score": 1.5},
        {"ply": 64, "color": False, "score": 1.4},
        {"ply": 66, "color": False, "score": 1.2},
        {"ply": 63, "color": True, "score": 1.1},
    ]
    chosen = select_moments(candidates, 4)
    assert [c["ply"] for c in chosen] == [60, 63, 66]


def test_fenetre_ouverture() -> None:
    # Noirs aux coups 9 et 10 (ouverture) : la fenêtre ne s'applique pas.
    candidates = [
        {"ply": 18, "color": False, "score": 1.5},
        {"ply": 20, "color": False, "score": 1.4},
    ]
    assert [c["ply"] for c in select_moments(candidates, 4)] == [18, 20]


def _tactical(case: dict[str, Any]) -> bool:
    board = chess.Board(case["fen"])

    def line(uci: str) -> list[chess.Move]:
        return [chess.Move.from_uci(u) for u in uci.split()]

    def pov(cp: int) -> PovScore:
        return PovScore(Cp(cp), chess.WHITE)

    return is_tactical(
        board,
        line(case["best"]),
        pov(case["best_cp"]),
        line(case["played"]),
        pov(case["played_cp"]),
    )


_CAS_315: dict[str, dict[str, Any]] = {
    "24...g6": {
        "fen": "2rqrbk1/1b3pp1/p2p1n1p/1p2nN1N/1P1BP3/P6P/2B2PP1/2RQR1K1 b - - 8 24",
        "best": "f6h5 d1h5 g7g6 f5h6 g8h7 h6f7 g6h5 f7d8 e8d8 c2b1 c8c1 e1c1",
        "best_cp": -263,
        "played": "g7g6 d4b6 d8b6 h5f6 g8h8 f6e8 g6f5 e8f6 b6d8 f6d5 f5e4 e1e4",
        "played_cp": 261,
    },
    "25.Nxf6+": {
        "fen": "2rqrbk1/1b3p2/p2p1npp/1p2nN1N/1P1BP3/P6P/2B2PP1/2RQR1K1 w - - 0 25",
        "best": "d4b6 d8b6 h5f6 g8h8 f6e8 g6f5 e8f6 b6d8 f6d5 f5e4 e1e4 d8g5",
        "best_cp": 261,
        "played": "h5f6 d8f6 f5g3 f8g7 c2b3 e5f3 d1f3 f6f3 g2f3 g7d4 c1d1 d4e5",
        "played_cp": -29,
    },
    "29.a4": {
        "fen": "2rqr1k1/5pb1/p2p2pp/1p1Pn3/1P1B4/P6P/2B2PP1/2RQR1K1 w - - 1 29",
        "best": "d4e5 g7e5 c2d3 c8c1 d1c1 h6h5 e1e2 h5h4 g1f1",
        "best_cp": -69,
        "played": "a3a4 e5f3 d1f3 g7d4 e1e8 d8e8 f3d3 c8c4 c1d1 g8g7 a4b5 a6b5",
        "played_cp": -181,
    },
    "31...Qxb5": {
        "fen": "2r1r1k1/5pb1/p2p2pp/1P1Pn3/1q6/4B2P/2B2PP1/2RQR1K1 b - - 0 31",
        "best": "a6b5 e3d2 b4d4 c1b1 d4c4 c2b3 c4c7 d2f4",
        "best_cp": -144,
        "played": "b4b5 c1b1 b5c4 c2a4 e8f8 a4b3 c4b4 b3a4 b4a3 b1a1 a3d3 a1a2",
        "played_cp": -71,
    },
    "14.Bxd6": {  # partie 316
        "fen": "2rq1rk1/p4ppp/1pnbbn2/3p4/5B2/P1N1PN2/1PQ1BPPP/R4RK1 w - - 1 14",
        "best": "e2a6 c8b8 f4d6 d8d6 c3b5 d6c5 f1c1 e6d7",
        "best_cp": 189,
        "played": "f4d6 d8d6 e2a6 c8d8 c3b5 d6c5 f1c1 c6b8 c2a4",
        "played_cp": 99,
    },
}


def test_erreur_tactique_ecartee() -> None:
    assert _tactical(_CAS_315["24...g6"])
    assert _tactical(_CAS_315["25.Nxf6+"])
    assert not _tactical(_CAS_315["29.a4"])
    assert not _tactical(_CAS_315["31...Qxb5"])


def test_partie_pliee() -> None:
    # 27.Kg2 (partie 316) : ES du meilleur coup 0,84 pour les blancs.
    assert is_decided(PovScore(Cp(420), chess.WHITE), 1611, True)
    assert not is_decided(PovScore(Cp(100), chess.WHITE), 1611, True)
    assert is_decided(PovScore(Cp(420), chess.WHITE), 1611, False)


def test_budget_k() -> None:
    candidates = [
        {"ply": 2 * i, "color": False, "score": 1.0 + i / 100} for i in range(1, 40)
    ]
    assert len(select_moments(candidates)) == MOMENTS_K
    assert len(select_moments(candidates, 2)) == 2
    assert select_moments(candidates, 0) == []


def test_exd5_candidat_partie_316(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    real = moments.select_moments

    def spy(cands: list[dict[str, Any]], k: int = MOMENTS_K) -> Any:
        captured.extend(cands)
        return real(cands, k)

    with patch.object(moments, "select_moments", spy):
        _run(tmp_path, "out.pgn", tmp_path / "m.json")
    exd5 = [c for c in captured if c["ply"] == 18]
    assert exd5
    assert "choix_de_reprise" in exd5[0]["signals"]


def test_moments_json(tmp_path: Path) -> None:
    sans = _run(tmp_path, "sans.pgn", None)
    avec = _run(tmp_path, "avec.pgn", tmp_path / "m" / "moments.json")
    assert avec.read_text(encoding="utf-8") == sans.read_text(encoding="utf-8")
    data = json.loads((tmp_path / "m" / "moments.json").read_text(encoding="utf-8"))
    partie = data["parties"][0]
    assert partie["partie"] == 1
    assert 1 <= len(partie["moments"]) <= MOMENTS_K
    dossier = partie["moments"][0]
    for key in ("fen", "stockfish", "structure", "pieces", "coup_naturel"):
        assert key in dossier


def test_echange_non_tactique() -> None:
    # 14.Bxd6 Qxd6 : prise suivie de la reprise à valeur égale, pas une tactique.
    case = _CAS_315["14.Bxd6"]
    board = chess.Board(case["fen"])
    played = [chess.Move.from_uci(u) for u in case["played"].split()]
    assert board.is_capture(played[0])
    assert played[1].to_square == played[0].to_square
    assert not _tactical(case)


def test_annotation_du_joueur_candidate(tmp_path: Path) -> None:
    def candidates(pgn: str) -> list[dict[str, Any]]:
        captured: list[dict[str, Any]] = []
        real = moments.select_moments

        def spy(cands: list[dict[str, Any]], k: int = MOMENTS_K) -> Any:
            captured.extend(cands)
            return real(cands, k)

        with patch.object(moments, "select_moments", spy):
            _run(tmp_path, "out.pgn", tmp_path / "m.json", pgn)
        return captured

    annotated = _PGN_316.replace(" b6 ", " b6?! ")
    assert annotated != _PGN_316
    game = chess.pgn.read_game(io.StringIO(annotated))
    assert game is not None
    assert annotated_plies(game) == {20}
    avec = [c for c in candidates(annotated) if c["ply"] == 20]
    assert avec
    assert "annotation" in avec[0]["signals"]
    sans = [c for c in candidates(_PGN_316) if c["ply"] == 20]
    assert not sans or "annotation" not in sans[0]["signals"]


def test_fait_espace_echange() -> None:
    # Avant 14.Bxd6 : les blancs contrôlent 10 cases de la moitié noire, les noirs 8.
    case = _CAS_315["14.Bxd6"]
    assert space(case["fen"]) == {"blancs": 10, "noirs": 8, "avantage": "blancs"}
    board = chess.Board(case["fen"])
    capture = chess.Move.from_uci("f4d6")
    assert is_space_exchange(board, capture, chess.Move.from_uci("d8d6"))
    assert not is_space_exchange(board, capture, None)
    assert not is_space_exchange(board, chess.Move.from_uci("e2a6"), None)
    assert not is_space_exchange(chess.Board(), chess.Move.from_uci("e2e4"), None)
