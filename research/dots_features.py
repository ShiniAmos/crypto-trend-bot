"""Alias: the relationship features live in app/relfeatures.py.

They moved into app/ when the live scorer started needing them, so that research and production
cannot compute them differently. This file makes `import dots_features` return THAT module - not a
copy of its names - so that monkeypatching in tests and attribute lookups in research scripts all
land on the single implementation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import relfeatures as _impl  # noqa: E402

sys.modules[__name__] = _impl
