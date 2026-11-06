"""Settings for the scoring service, read from environment variables (D-040).

Everything has a default that works for local development against a Postgres on localhost;
docker-compose sets the rest. Only the standard library is used here so the module imports in
every environment.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return default if v in (None, "") else int(v)


def _float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return default if v in (None, "") else float(v)


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    if v.lower() in ("1", "true", "yes", "on"):
        return True
    if v.lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name}={v!r} is not a boolean")


def _list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    v = os.environ.get(name)
    if v is None:
        return default
    return tuple(x.strip() for x in v.split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    """API settings."""

    database_url: str = "postgresql://genprobe:genprobe@localhost:5432/genprobe"
    data_dir: str = "data/service"                 # uploaded images, content-addressed
    metrics: tuple[str, ...] = ("clipscore",)      # metrics the API accepts
    api_keys: tuple[str, ...] = ()                 # empty: no authentication (local use)
    max_image_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 4096 * 4096
    max_prompt_chars: int = 1000
    max_queue_depth: int = 1000                    # per metric; above it, submissions get 429
    max_attempts: int = 3                          # per task, including the first
    scorer_ttl_seconds: int = 120                  # a worker not seen for this long is gone
    pool_min: int = 1
    pool_max: int = 10
    auto_migrate: bool = False                     # docker-compose runs migrations once instead

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        s = cls(
            database_url=_str("GENPROBE_DATABASE_URL", d.database_url),
            data_dir=_str("GENPROBE_SERVICE_DATA", d.data_dir),
            metrics=_list("GENPROBE_SERVICE_METRICS", d.metrics),
            api_keys=_list("GENPROBE_API_KEYS", d.api_keys),
            max_image_bytes=_int("GENPROBE_MAX_IMAGE_BYTES", d.max_image_bytes),
            max_image_pixels=_int("GENPROBE_MAX_IMAGE_PIXELS", d.max_image_pixels),
            max_prompt_chars=_int("GENPROBE_MAX_PROMPT_CHARS", d.max_prompt_chars),
            max_queue_depth=_int("GENPROBE_MAX_QUEUE_DEPTH", d.max_queue_depth),
            max_attempts=_int("GENPROBE_MAX_ATTEMPTS", d.max_attempts),
            scorer_ttl_seconds=_int("GENPROBE_SCORER_TTL_SECONDS", d.scorer_ttl_seconds),
            pool_min=_int("GENPROBE_POOL_MIN", d.pool_min),
            pool_max=_int("GENPROBE_POOL_MAX", d.pool_max),
            auto_migrate=_bool("GENPROBE_AUTO_MIGRATE", d.auto_migrate),
        )
        s.check()
        return s

    def check(self) -> None:
        if not self.metrics:
            raise ValueError("GENPROBE_SERVICE_METRICS lists no metric")
        if self.max_attempts < 1 or self.max_queue_depth < 1:
            raise ValueError("max_attempts and max_queue_depth must be at least 1")


@dataclass(frozen=True)
class WorkerSettings:
    """Worker settings. One worker process serves one metric."""

    database_url: str = Settings.database_url
    data_dir: str = Settings.data_dir
    metric: str = "clipscore"
    device: str = "cpu"
    batch_size: int = 16                 # most tasks claimed and scored together
    lease_seconds: float = 60.0          # renewed every lease/3 while a batch is scored
    retry_backoff_seconds: float = 2.0   # doubled per attempt
    poll_min_seconds: float = 0.05       # idle polling starts here ...
    poll_max_seconds: float = 1.0        # ... and backs off to here
    housekeeping_seconds: float = 10.0   # reap expired leases, refresh the scorer row
    metrics_port: int = 0                # Prometheus endpoint; 0 = off
    worker_id: str = field(default_factory=lambda: f"{socket.gethostname()}-{os.getpid()}")

    @classmethod
    def from_env(cls, **overrides) -> "WorkerSettings":
        d = cls()
        s = cls(
            database_url=_str("GENPROBE_DATABASE_URL", d.database_url),
            data_dir=_str("GENPROBE_SERVICE_DATA", d.data_dir),
            metric=_str("GENPROBE_WORKER_METRIC", d.metric),
            device=_str("GENPROBE_WORKER_DEVICE", d.device),
            batch_size=_int("GENPROBE_WORKER_BATCH_SIZE", d.batch_size),
            lease_seconds=_float("GENPROBE_LEASE_SECONDS", d.lease_seconds),
            retry_backoff_seconds=_float("GENPROBE_RETRY_BACKOFF_SECONDS", d.retry_backoff_seconds),
            poll_min_seconds=_float("GENPROBE_POLL_MIN_SECONDS", d.poll_min_seconds),
            poll_max_seconds=_float("GENPROBE_POLL_MAX_SECONDS", d.poll_max_seconds),
            housekeeping_seconds=_float("GENPROBE_HOUSEKEEPING_SECONDS", d.housekeeping_seconds),
            metrics_port=_int("GENPROBE_WORKER_METRICS_PORT", d.metrics_port),
            worker_id=_str("GENPROBE_WORKER_ID", d.worker_id),
        )
        s = cls(**{**s.__dict__, **{k: v for k, v in overrides.items() if v is not None}})
        if s.batch_size < 1 or s.lease_seconds <= 0:
            raise ValueError("batch_size must be >= 1 and lease_seconds > 0")
        return s
