"""Tests des commentaires qualitatifs : un appel par partie, vérifié, en cache."""

import io
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import chess.engine
import chess.pgn
import pytest
from maia2.main import MAIA2Model
from openai import OpenAI

from caissAI.bin import annotator, game_analyzer, llm_comments
from tests.unit.test_moments import (
    _PGN_316,
    _fake_evaluate,
    _fake_move_probs,
    _FakeEngine,
)

_MARKER = "Dossiers de faits des moments à commenter :\n"
_GOLDEN = Path(__file__).parent / "golden_316_main.pgn"
_Edit = Callable[[int, dict[str, Any]], dict[str, Any]]


def _dossiers(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Dossiers (notation française) lus dans l'entrée de la requête."""
    loaded: list[dict[str, Any]] = json.loads(body["input"].split(_MARKER)[1])
    return loaded


def _answer(body: dict[str, Any], edit: _Edit | None = None) -> str:
    """Réponse valide : une ligne et une évaluation du dossier, par moment."""
    moments = []
    for i, dossier in enumerate(_dossiers(body)):
        best = dossier["stockfish"]["lignes"][0]
        cp = best["eval"]["cp"] / 100
        item = {
            "ply": dossier["demi_coup"],
            "placement": "variante" if i == 0 else "coup",
            "commentaire": (
                f"Que change ce coup ? Le moteur joue {dossier['coup']}{best['pv'][0]} "
                f"({cp:+.2f})".replace(".", ",")
            ),
            "idee_cle": "garder la structure",
            "theme": "structure",
        }
        moments.append(edit(i, item) if edit else item)
    return json.dumps({"moments": moments, "resume": "Trois phrases."})


class _Llm:
    """Client OpenAI simulé : compte les appels, répond avec `_answer`."""

    def __init__(self, edit: _Edit | None = None, raw: str | None = None) -> None:
        self.client = OpenAI(api_key="test")
        self.bodies: list[dict[str, Any]] = []
        self.edit = edit
        self.raw = raw

    def create(self, **body: Any) -> Any:
        self.bodies.append(body)
        if "text" not in body:  # résumé historique (`summarise_game`)
            return SimpleNamespace(output_text="Résumé.")
        return SimpleNamespace(output_text=self.raw or _answer(body, self.edit))


def _annotate(
    tmp_path: Path,
    llm: _Llm,
    name: str = "out.pgn",
    pgn: str = _PGN_316,
    **kwargs: Any,
) -> str:
    source = tmp_path / "source.pgn"
    source.write_text(pgn, encoding="utf-8")
    output = tmp_path / name
    with (
        patch.object(game_analyzer, "get_move_probs", _fake_move_probs),
        patch.object(game_analyzer, "evaluate_game", _fake_evaluate),
        patch.object(
            chess.engine.SimpleEngine, "popen_uci", return_value=_FakeEngine()
        ),
        patch.object(llm.client.responses, "create", side_effect=llm.create),
    ):
        annotator.comment_games_from_file(
            pgn_path=source,
            output_path=output,
            config_path=Path(game_analyzer.__file__).parent.parent / "config",
            engine_path=Path("stockfish"),
            maia2_model=MAIA2Model(),
            n_workers=1,
            openai_client=llm.client,
            openai_model="gpt-4.1",
            **{"comment": True, "deep_comments": 4, **kwargs},
        )
    return output.read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Cache et données dans le dossier temporaire du test."""
    path = tmp_path / "data"
    monkeypatch.setenv("CAISSAI_DATA_DIR", str(path))
    return path


def _comments(pgn: str) -> dict[int, str]:
    """Commentaire de chaque coup de la ligne principale, par demi-coup."""
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    return {node.ply(): node.comment for node in game.mainline()}


def test_un_seul_appel_par_partie(tmp_path: Path) -> None:
    llm = _Llm()
    pgn = _annotate(tmp_path, llm)
    assert len(llm.bodies) == 1
    assert len(_dossiers(llm.bodies[0])) == 4
    assert llm.bodies[0]["text"]["format"]["strict"] is True
    assert pgn.count("Que change ce coup ?") == 4
    assert "Trois phrases." in pgn
    # Le premier moment est posé en `variante` : sa variation porte le texte.
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    assert any(
        "Que change ce coup ?" in variation.starting_comment
        for node in game.mainline()
        for variation in node.variations[1:]
    )


def _rejet(tmp_path: Path, ajout: str) -> tuple[str, int]:
    """Annote avec `ajout` dans le premier commentaire ; rend le PGN et son ply."""
    seen: list[int] = []

    def edit(i: int, item: dict[str, Any]) -> dict[str, Any]:
        if i == 0:
            seen.append(item["ply"])
            item["commentaire"] += ajout
        return item

    return _annotate(tmp_path, _Llm(edit), "rejet.pgn"), seen[0]


def _chiffre(tmp_path: Path, ply: int) -> str:
    """Commentaire chiffré de `ply`, sans commentaires qualitatifs."""
    return _comments(_annotate(tmp_path, _Llm(), "base.pgn", deep_comments=0))[ply]


def test_ligne_inventee_rejetee(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pgn, ply = _rejet(tmp_path, " Après 12.Dh5 Cf6 13.Dxf7 tout s'effondre.")
    assert "Dh5" not in pgn
    assert pgn.count("Que change ce coup ?") == 3
    assert _comments(pgn)[ply] == _chiffre(tmp_path, ply)
    assert "rejeté" in caplog.text


def test_evaluation_inventee_rejetee(tmp_path: Path) -> None:
    pgn, ply = _rejet(tmp_path, " Le moteur donne +7,77.")
    assert "+7,77" not in pgn
    assert pgn.count("Que change ce coup ?") == 3
    assert _comments(pgn)[ply] == _chiffre(tmp_path, ply)


def test_cache_sans_appel(tmp_path: Path, data_dir: Path) -> None:
    llm = _Llm()
    premiere = _annotate(tmp_path, llm, "un.pgn")
    assert len(llm.bodies) == 1
    assert len(list((data_dir / "cache").glob("*.json"))) == 1
    seconde = _annotate(tmp_path, llm, "deux.pgn")
    assert len(llm.bodies) == 1
    assert seconde == premiere


def test_deep_comments_zero_identique(tmp_path: Path) -> None:
    llm = _Llm()
    pgn = _annotate(tmp_path, llm, deep_comments=0)
    assert pgn.rstrip() == _GOLDEN.read_text(encoding="utf-8").rstrip()
    assert len(llm.bodies) == 1  # le résumé historique : un appel, sans dossiers
    assert "Dossiers de faits" not in llm.bodies[0].get("input", "")


def test_no_comment_sans_appel(tmp_path: Path) -> None:
    llm = _Llm()
    _annotate(tmp_path, llm, comment=False)
    assert llm.bodies == []


def test_batch_requete(tmp_path: Path) -> None:
    llm = _Llm()
    sent: dict[str, Any] = {}

    def files_create(file: tuple[str, bytes], purpose: str) -> Any:
        sent["lines"] = [json.loads(x) for x in file[1].decode().splitlines()]
        sent["purpose"] = purpose
        return SimpleNamespace(id="file-in")

    def results(_: str) -> Any:
        lines = [
            {
                "custom_id": line["custom_id"],
                "response": {
                    "status_code": 200,
                    "body": {
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": _answer(line["body"]),
                                    }
                                ],
                            }
                        ]
                    },
                },
            }
            for line in sent["lines"]
        ]
        return SimpleNamespace(text="\n".join(json.dumps(x) for x in lines))

    batch = SimpleNamespace(id="batch-1", status="completed", output_file_id="file-out")
    with (
        patch.object(llm.client.files, "create", side_effect=files_create),
        patch.object(llm.client.files, "content", side_effect=results),
        patch.object(llm.client.batches, "create", return_value=batch) as create,
    ):
        pgn = _annotate(tmp_path, llm, pgn=_PGN_316 + "\n" + _PGN_316, batch=True)
    assert llm.bodies == []  # aucun appel immédiat
    assert sent["purpose"] == "batch"
    assert len(sent["lines"]) == 2
    assert len({line["custom_id"] for line in sent["lines"]}) == 2
    for line in sent["lines"]:
        assert line["method"] == "POST"
        assert line["url"] == "/v1/responses"
        assert line["body"]["model"] == "gpt-4.1"
        assert "instructions" in line["body"]
    create.assert_called_once_with(
        input_file_id="file-in", endpoint="/v1/responses", completion_window="24h"
    )
    assert pgn.count("Trois phrases.") == 2


def test_resolve_deep_comments(monkeypatch: pytest.MonkeyPatch) -> None:
    assert llm_comments.resolve_deep_comments() == 4
    monkeypatch.setenv("CAISSAI_DEEP_COMMENTS", "2")
    assert llm_comments.resolve_deep_comments() == 2
    assert llm_comments.resolve_deep_comments(0) == 0
    monkeypatch.setenv("CAISSAI_DEEP_COMMENTS", "x")
    with pytest.raises(ValueError):
        llm_comments.resolve_deep_comments()


def test_reponse_inexploitable_sans_cache(tmp_path: Path, data_dir: Path) -> None:
    pgn = _annotate(tmp_path, _Llm(raw="pas du JSON"))
    assert pgn == _annotate(tmp_path, _Llm(), "libre.pgn", comment=False)
    assert not list((data_dir / "cache").glob("*.json"))
