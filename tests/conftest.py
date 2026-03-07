"""File to define fixtures to be used by the tests."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub out maia2 and its heavy ML dependencies (torch, einops, yaml, …)
# so that game_analyzer.py can be imported during tests without the full
# ML stack being installed.
# ---------------------------------------------------------------------------


def _make_stub(name: str) -> ModuleType:
    mod = ModuleType(name)
    mod.__spec__ = None  # type: ignore[attr-defined]
    return mod


_maia2 = _make_stub("maia2")
_maia2_inference = _make_stub("maia2.inference")
_maia2_main = _make_stub("maia2.main")

# Attributes accessed at import time or in function bodies
_maia2_inference.prepare = MagicMock()
_maia2_inference.inference_each = MagicMock()


class _FakeMAIA2Model:
    """Minimal stand-in for maia2.main.MAIA2Model."""


_maia2_main.MAIA2Model = _FakeMAIA2Model
_maia2.inference = _maia2_inference

for _name, _mod in [
    ("maia2", _maia2),
    ("maia2.inference", _maia2_inference),
    ("maia2.main", _maia2_main),
]:
    sys.modules.setdefault(_name, _mod)
