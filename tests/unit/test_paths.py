"""Tests de la résolution des chemins runtime (``caissAI.bin.paths``)."""

from pathlib import Path

import pytest

from caissAI.bin.paths import (
    default_state_dir,
    resolve_data_dir,
    resolve_log_dir,
    resolve_path,
    resolve_pgn_path,
)


def test_default_state_dir_utilise_localappdata_quand_defini(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert default_state_dir() == tmp_path / "caissAI"


def test_default_state_dir_repli_sur_home_sans_localappdata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert default_state_dir().parts[-3:] == (".local", "share", "caissAI")


def test_resolve_path_cli_prioritaire_sur_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CAISSAI_DATA_DIR", str(tmp_path / "depuis_env"))
    result = resolve_path(
        str(tmp_path / "depuis_cli"), "CAISSAI_DATA_DIR", tmp_path / "defaut"
    )
    assert result == tmp_path / "depuis_cli"


def test_resolve_path_env_prioritaire_sur_defaut(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CAISSAI_DATA_DIR", str(tmp_path / "depuis_env"))
    result = resolve_path(None, "CAISSAI_DATA_DIR", tmp_path / "defaut")
    assert result == tmp_path / "depuis_env"


def test_resolve_path_defaut_quand_rien_de_defini(tmp_path: Path) -> None:
    assert resolve_path(None, "CAISSAI_DATA_DIR", tmp_path / "d") == tmp_path / "d"


def test_resolve_path_ignore_une_valeur_env_vide(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CAISSAI_DATA_DIR", "   ")
    assert resolve_path(None, "CAISSAI_DATA_DIR", tmp_path / "d") == tmp_path / "d"


def test_resolve_data_dir_derive_de_default_state_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("CAISSAI_DATA_DIR", raising=False)
    assert resolve_data_dir() == tmp_path / "caissAI" / "data"


def test_resolve_log_dir_derive_de_default_state_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("CAISSAI_LOG_DIR", raising=False)
    assert resolve_log_dir() == tmp_path / "caissAI" / "log"


def test_resolve_pgn_path_utilise_cli_en_priorite(tmp_path: Path) -> None:
    cible = tmp_path / "parties.pgn"
    assert resolve_pgn_path(str(cible)) == cible


def test_resolve_pgn_path_utilise_env_a_defaut_du_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cible = tmp_path / "depuis_env.pgn"
    monkeypatch.setenv("CAISSAI_PGN", str(cible))
    assert resolve_pgn_path(None) == cible


def test_resolve_pgn_path_none_si_rien_de_defini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAISSAI_PGN", raising=False)
    assert resolve_pgn_path(None) is None
