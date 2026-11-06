"""Prometheus metrics and JSON logs for the scoring service (D-040).

The API serves its metrics at ``/metrics``; each worker on its own port
(``GENPROBE_WORKER_METRICS_PORT``). Label values are bounded (route templates, metric names,
fixed outcome words) so the number of series stays small.
"""

from __future__ import annotations

import json
import logging
import sys
import time

from prometheus_client import Counter, Gauge, Histogram

_LATENCY = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)

# API
HTTP_SECONDS = Histogram("genprobe_http_request_seconds", "HTTP request latency",
                         ["method", "route", "status"], buckets=_LATENCY)
SUBMISSIONS = Counter("genprobe_submissions_total", "Job submissions by outcome",
                      ["outcome"])  # created, replayed, busy, invalid, conflict
CACHED_AT_SUBMIT = Counter("genprobe_cached_at_submit_total", "Tasks answered from the cache at submit", ["metric"])
QUEUE_TASKS = Gauge("genprobe_queue_tasks", "Tasks queued or running", ["metric", "status"])

# Workers
TASKS = Counter("genprobe_worker_tasks_total", "Task attempts by outcome",
                ["metric", "outcome"])  # done, cached, retry, failed, lost_lease
BATCH_SIZE = Histogram("genprobe_worker_batch_size", "Tasks scored per batch", ["metric"],
                       buckets=(1, 2, 4, 8, 16, 32, 64, 128))
BATCH_SECONDS = Histogram("genprobe_worker_batch_seconds", "Time to score one batch", ["metric"], buckets=_LATENCY)
TASK_SECONDS = Histogram("genprobe_task_seconds", "Submit-to-result time of completed tasks", ["metric"],
                         buckets=_LATENCY)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
             "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        for k, v in record.__dict__.items():
            if k.startswith("ctx_"):
                d[k[4:]] = v
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, default=str)


def setup_logging(level: str = "INFO") -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level)


def ctx(**kw) -> dict:
    """``log.info("...", extra=ctx(job_id=...))`` puts the fields into the JSON line."""
    return {f"ctx_{k}": v for k, v in kw.items()}
