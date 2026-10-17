"""Shared test fixtures. Tests use unittest so they run under both `python -m unittest` and pytest."""

from __future__ import annotations

from genprobe.backends import FakeBackend
from genprobe.config import GenConfig
from genprobe.samples import Prompt


def make_cfg(**overrides) -> GenConfig:
    base = dict(name="toy", repo="none/toy", dtype="fp16", steps=1, cfg=1.0,
                height=64, width=64, batch_size=3, seeds=[0, 1])
    base.update(overrides)
    return GenConfig(**base)


def make_prompts(n: int) -> list[Prompt]:
    return [Prompt(prompt_id=f"p{i:03d}", prompt=f"a photo of {i % 8 + 1} apples", tags=["counting"],
                   spec={"objects": [{"name": "apple", "count": i % 8 + 1}]}, split="dev")
            for i in range(n)]


def quiet(*_args, **_kwargs) -> None:
    pass


class SimulatedKill(KeyboardInterrupt):
    """BaseException, like a real SIGTERM/SIGKILL path: not swallowed by `except Exception`."""


class KillAfter(FakeBackend):
    """Fake backend that dies after generating `n` images."""

    def __init__(self, cfg, n: int, **kw):
        super().__init__(cfg, **kw)
        self.n = n

    def generate(self, batch):
        if len(self.calls) + len(batch) > self.n:
            raise SimulatedKill()
        return super().generate(batch)
