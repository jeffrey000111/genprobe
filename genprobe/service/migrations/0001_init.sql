-- Scoring service schema (D-040). Postgres holds the jobs, the work queue and the score cache,
-- so a job and its tasks are created in one transaction and there is no second system to keep
-- in step with the database.

-- One request: one image, one prompt, one or more metrics.
CREATE TABLE jobs (
    id              uuid PRIMARY KEY,
    -- Client-chosen key (Idempotency-Key header). A retried request with the same key returns the
    -- first job instead of creating a second one. NULLs never conflict, so keyless requests work.
    idempotency_key text UNIQUE,
    -- sha256 of (image, prompt, metrics): the same key with a different request is a 409.
    request_hash    text NOT NULL,
    image_sha256    text NOT NULL,
    prompt          text NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- One metric for one job: the unit of work a worker claims.
CREATE TABLE tasks (
    id               bigserial PRIMARY KEY,
    job_id           uuid NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    metric           text NOT NULL,
    status           text NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed')),
    attempts         integer NOT NULL DEFAULT 0,
    max_attempts     integer NOT NULL CHECK (max_attempts >= 1),
    available_at     timestamptz NOT NULL DEFAULT now(),   -- retry backoff: not claimable before
    lease_owner      text,                                  -- worker holding the task
    lease_expires_at timestamptz,                           -- after this, another worker may take it
    value            double precision,
    extra            jsonb,
    metric_version   text,
    cached           boolean NOT NULL DEFAULT false,
    error            text,
    created_at       timestamptz NOT NULL DEFAULT now(),
    started_at       timestamptz,                           -- first claim
    finished_at      timestamptz,
    UNIQUE (job_id, metric),
    CHECK (status <> 'running' OR (lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)),
    CHECK (status <> 'done' OR value IS NOT NULL)
);

-- Claiming reads the oldest available queued tasks of one metric ...
CREATE INDEX tasks_queued_idx ON tasks (metric, available_at, id) WHERE status = 'queued';
-- ... and running tasks whose lease has expired (their worker died or stalled).
CREATE INDEX tasks_running_idx ON tasks (metric, lease_expires_at) WHERE status = 'running';
CREATE INDEX tasks_job_idx ON tasks (job_id);

-- Scores already computed. The metric version is part of the key, so upgrading a model
-- invalidates its old scores, the same rule as the research pipeline's resume (D-003).
CREATE TABLE score_cache (
    image_sha256   text NOT NULL,
    prompt_sha256  text NOT NULL,
    metric         text NOT NULL,
    metric_version text NOT NULL,
    value          double precision NOT NULL,
    extra          jsonb NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (image_sha256, prompt_sha256, metric, metric_version)
);

-- Live workers and the metric version each one serves. The API reads it to answer from the
-- cache at submit time and to list what can be scored.
CREATE TABLE scorers (
    worker_id      text PRIMARY KEY,
    metric         text NOT NULL,
    metric_version text NOT NULL,
    started_at     timestamptz NOT NULL DEFAULT now(),
    last_seen      timestamptz NOT NULL DEFAULT now()
);
