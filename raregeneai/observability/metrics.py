"""Prometheus metrics for RareGeneAI.

Exposes application metrics for monitoring:
  - Request latency and counts
  - Pipeline execution times
  - Error rates
  - Active job counts
  - Resource usage
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class MetricsCollector:
    """Central metrics registry for RareGeneAI."""

    def __init__(self):
        # ── Application info ──────────────────────────────────────────
        self.app_info = Info("raregeneai", "RareGeneAI application info")

        # ── HTTP metrics ──────────────────────────────────────────────
        self.http_requests_total = Counter(
            "raregeneai_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
        )

        self.http_request_duration_seconds = Histogram(
            "raregeneai_http_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        # ── Pipeline metrics ─────────────────────────────────────────
        self.pipeline_runs_total = Counter(
            "raregeneai_pipeline_runs_total",
            "Total pipeline executions",
            ["status"],  # "success", "failure"
        )

        self.pipeline_duration_seconds = Histogram(
            "raregeneai_pipeline_duration_seconds",
            "Pipeline execution duration",
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
        )

        self.variants_processed_total = Counter(
            "raregeneai_variants_processed_total",
            "Total variants processed",
        )

        self.genes_analyzed_total = Counter(
            "raregeneai_genes_analyzed_total",
            "Total genes analyzed",
        )

        # ── Job metrics ──────────────────────────────────────────────
        self.active_jobs = Gauge(
            "raregeneai_active_jobs",
            "Currently running analysis jobs",
        )

        self.jobs_total = Counter(
            "raregeneai_jobs_total",
            "Total jobs created",
            ["status"],
        )

        # ── Auth metrics ─────────────────────────────────────────────
        self.auth_attempts_total = Counter(
            "raregeneai_auth_attempts_total",
            "Authentication attempts",
            ["result"],  # "success", "failure"
        )

        # ── Error metrics ────────────────────────────────────────────
        self.errors_total = Counter(
            "raregeneai_errors_total",
            "Total errors",
            ["type", "component"],
        )

    def set_app_info(self, version: str, build: str = "") -> None:
        """Set application info labels."""
        self.app_info.info({
            "version": version,
            "build": build,
        })


# Global metrics instance
metrics = MetricsCollector()


class PrometheusMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware to collect HTTP metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        path = request.url.path

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        metrics.http_requests_total.labels(
            method=method,
            endpoint=path,
            status_code=response.status_code,
        ).inc()

        metrics.http_request_duration_seconds.labels(
            method=method,
            endpoint=path,
        ).observe(duration)

        return response


def metrics_endpoint(request: Request) -> Response:
    """Endpoint to expose Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


def track_pipeline_run(func: Callable) -> Callable:
    """Decorator to track pipeline execution metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        metrics.active_jobs.inc()
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            metrics.pipeline_runs_total.labels(status="success").inc()
            return result
        except Exception:
            metrics.pipeline_runs_total.labels(status="failure").inc()
            raise
        finally:
            duration = time.perf_counter() - start
            metrics.pipeline_duration_seconds.observe(duration)
            metrics.active_jobs.dec()

    return wrapper
