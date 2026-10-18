"""Metric plugins. Import a module here to register it.

Plugins that need heavy packages (torch, t2v_metrics, open_clip, ...) import them inside
``setup()``, so importing this package stays cheap and works in every env (and on the laptop).
"""

from .base import REGISTRY, Metric, register, validate_record  # noqa: F401
from . import clipscore, vqascore  # noqa: F401,E402  (registers "clipscore", "vqascore")
