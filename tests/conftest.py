"""File to define fixtures to be used by the tests."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

_PATH_ENV_VARS = (
    "CAISSAI_PGN",
    "CAISSAI_DATA_DIR",
    "CAISSAI_LOG_DIR",
    "CAISSAI_OPENAI_MODEL",
    "OPENAI_MODEL",
    "LOCALAPPDATA",
)


@pytest.fixture(autouse=True)
def clean_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise les variables de chemin héritées de l'environnement.

    Sans cela, un poste de développement où ces variables sont définies
    (via ``.env`` chargé dans le shell, ou l'environnement système) ferait
    passer ou échouer les tests selon la machine.
    """
    for var in _PATH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Stub out maia2 and its heavy ML dependencies (torch, einops, yaml, …)
# so that game_analyzer.py can be imported during tests without the full
# ML stack being installed.
# ---------------------------------------------------------------------------


def _make_stub(name: str) -> ModuleType:
    mod = ModuleType(name)
    mod.__spec__ = None
    return mod


_maia2 = _make_stub("maia2")
_maia2_inference = _make_stub("maia2.inference")
_maia2_main = _make_stub("maia2.main")

# Attributes accessed at import time or in function bodies
_maia2_inference.prepare = MagicMock()  # type: ignore[attr-defined]
_maia2_inference.inference_each = MagicMock()  # type: ignore[attr-defined]


class _FakeMAIA2Model:
    """Minimal stand-in for maia2.main.MAIA2Model."""


_maia2_main.MAIA2Model = _FakeMAIA2Model  # type: ignore[attr-defined]
_maia2.inference = _maia2_inference  # type: ignore[attr-defined]

for _name, _mod in [
    ("maia2", _maia2),
    ("maia2.inference", _maia2_inference),
    ("maia2.main", _maia2_main),
]:
    sys.modules.setdefault(_name, _mod)
