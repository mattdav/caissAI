"""Tests des faits de position (fonctions pures)."""

import chess

from caissAI.bin.position_facts import (
    lost_support,
    pawn_structure,
    piece_facts,
    recapture_choice,
    square_control,
)

_COUPS_316 = "d4 d5 c4 e6 Nc3 Nf6 Nf3 Be7 Bf4 O-O e3 c5 Be2 Nc6 dxc5 Bxc5".split()


def _partie_316(*coups: str) -> chess.Board:
    """Position de la partie 316 après 8...Bxc5, puis `coups`."""
    board = chess.Board()
    for san in (*_COUPS_316, *coups):
        board.push_san(san)
    return board


def test_choix_de_reprise_partie_316() -> None:
    avant_prise = _partie_316().fen()
    choix = recapture_choice(avant_prise, "c4d5")
    assert choix is not None
    assert choix["case"] == "d5"
    assert set(choix["reprises"]) == {"exd5", "Nxd5", "Qxd5"}
    assert choix["pion_et_piece"] is True


def test_choix_de_reprise_une_seule_reprise() -> None:
    fen = "4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1"
    assert recapture_choice(fen, "e4d5") is None


def test_choix_de_reprise_sans_prise() -> None:
    assert recapture_choice(chess.STARTING_FEN, "e2e4") is None


def test_structure_pion_isole() -> None:
    apres_prise = _partie_316("cxd5")
    avec_pion = apres_prise.copy()
    avec_pion.push_san("exd5")
    avec_piece = apres_prise.copy()
    avec_piece.push_san("Nxd5")
    assert pawn_structure(avec_pion.fen())["noirs"]["isoles"] == ["d5"]
    assert pawn_structure(avec_piece.fen())["noirs"]["isoles"] == []


def test_defenseur_perdu() -> None:
    board = _partie_316("cxd5", "Nxd5", "Qb3")
    faits = lost_support(board.fen(), "b7b6")
    assert {
        "piece": "cavalier",
        "case": "c6",
        "defenseur": "b7",
    } in faits["defenseurs_perdus"]
    assert {"piece": "fou", "case": "c5", "retraite": "b6"} in faits[
        "retraites_perdues"
    ]


def test_structure_doubles_passes_arrieres_colonnes() -> None:
    # Blancs : pions c2 c3 (doublés), e4 ; noirs : pions d6, e5, f7.
    fen = "4k3/5p2/3p4/4p3/4P3/2P5/2P5/4K3 w - - 0 1"
    blancs = pawn_structure(fen)["blancs"]
    assert blancs["doubles"] == ["c2", "c3"] or blancs["doubles"] == ["c3", "c2"]
    assert blancs["ilots"] == 2
    assert "a" in blancs["colonnes_ouvertes"]
    assert "f" in blancs["colonnes_semi_ouvertes"]
    assert pawn_structure(fen)["noirs"]["arrieres"] == ["d6"]
    passe = pawn_structure("4k3/8/8/3P4/8/8/8/4K3 w - - 0 1")["blancs"]["passes"]
    assert passe == ["d5"]


def test_pion_arriere() -> None:
    # d3 est arriéré : c4 et e4 sont plus avancés, d4 est tenue par le pion noir e5.
    fen = "4k3/8/8/4p3/2P1P3/3P4/8/4K3 w - - 0 1"
    assert pawn_structure(fen)["blancs"]["arrieres"] == ["d3"]


def test_cases_trouees() -> None:
    fen = "4k3/8/8/8/8/8/P2P1PPP/4K3 w - - 0 1"
    trouees = pawn_structure(fen)["blancs"]["cases_trouees"]
    assert "d3" in trouees


def test_square_control() -> None:
    assert square_control(chess.STARTING_FEN, "e3", True) == {
        "attaquants": 0,
        "defenseurs": 2,
    }


def test_fou_bloque_et_pieces_non_defendues() -> None:
    faits = piece_facts(chess.STARTING_FEN, True)
    assert faits["fous_bloques"] == ["c1", "f1"]
    assert piece_facts("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", True)[
        "pieces_non_defendues"
    ] == ["a1"]
