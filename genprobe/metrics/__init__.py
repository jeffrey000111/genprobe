"""Metric plugins. Import a module here to register it.

Plugins that need heavy packages (torch, t2v_metrics, open_clip, ...) import them inside
``setup()``, so importing this package stays cheap and works in every conda env.
"""

from .base import REGISTRY, Metric, register, validate_record  # noqa: F401

# Next step (M1): add clipscore.py and vqascore.py, then import them here, e.g.
# from . import clipscore, vqascore  # noqa: F401
