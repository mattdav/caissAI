"""Tests de la sélection des moments clés et de l'export JSON."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import chess
import chess.engine
import pandas as pd
from chess.engine import Cp, PovScore
from maia2.main import MAIA2Model
from openai import OpenAI

from caissAI.bin import annotator, game_analyzer, moments
from caissAI.bin.moments import MOMENTS_K, select_moments

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


def _run(tmp_path: Path, name: str, moments_json: Path | None) -> Path:
    source = tmp_path / "source.pgn"
    source.write_text(_PGN_316, encoding="utf-8")
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
    # Noirs aux coups 10 et 12 : une seule erreur retenue ; coup 13 : hors fenêtre.
    candidates = [
        {"ply": 20, "color": False, "score": 1.5},
        {"ply": 24, "color": False, "score": 1.4},
        {"ply": 26, "color": False, "score": 1.2},
        {"ply": 23, "color": True, "score": 1.1},
    ]
    chosen = select_moments(candidates, 4)
    assert [c["ply"] for c in chosen] == [20, 23, 26]


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
