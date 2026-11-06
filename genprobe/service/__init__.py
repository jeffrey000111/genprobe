"""GenProbe's metrics as a service: an HTTP API, a Postgres work queue and scoring workers (D-040).

See docs/service.md. The modules that need the service dependencies (FastAPI, psycopg,
prometheus-client) import them themselves, so importing ``genprobe`` stays light.
"""
