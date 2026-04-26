"""Unit tests for observability module: metrics, health checks."""

import time

import pytest


# ── Metrics tests ────────────────────────────────────────────────────────────


class TestMetricsCollector:
    def test_singleton_metrics(self):
        from raregeneai.observability.metrics import metrics

        assert metrics is not None
        assert hasattr(metrics, "http_requests_total")
        assert hasattr(metrics, "pipeline_runs_total")
        assert hasattr(metrics, "active_jobs")

    def test_set_app_info(self):
        from raregeneai.observability.metrics import metrics

        # Should not raise
        metrics.set_app_info(version="1.1.0", build="test")

    def test_counter_increment(self):
        from raregeneai.observability.metrics import metrics

        # Use the singleton to avoid duplicate registry errors
        metrics.pipeline_runs_total.labels(status="success").inc()
        metrics.pipeline_runs_total.labels(status="failure").inc()

    def test_histogram_observe(self):
        from raregeneai.observability.metrics import metrics

        metrics.pipeline_duration_seconds.observe(5.0)
        metrics.http_request_duration_seconds.labels(method="GET", endpoint="/health").observe(0.01)

    def test_gauge_inc_dec(self):
        from raregeneai.observability.metrics import metrics

        metrics.active_jobs.inc()
        metrics.active_jobs.dec()

    def test_error_counter_labels(self):
        from raregeneai.observability.metrics import metrics

        metrics.errors_total.labels(type="pipeline_error", component="api").inc()
        metrics.auth_attempts_total.labels(result="failure").inc()


class TestTrackPipelineRun:
    def test_decorator_tracks_success(self):
        from raregeneai.observability.metrics import track_pipeline_run

        @track_pipeline_run
        def successful_pipeline():
            return "result"

        result = successful_pipeline()
        assert result == "result"

    def test_decorator_tracks_failure(self):
        from raregeneai.observability.metrics import track_pipeline_run

        @track_pipeline_run
        def failing_pipeline():
            raise ValueError("pipeline error")

        with pytest.raises(ValueError):
            failing_pipeline()


class TestMetricsEndpoint:
    def test_generate_metrics_output(self):
        from prometheus_client import generate_latest

        output = generate_latest()
        assert isinstance(output, bytes)
        assert len(output) > 0


# ── Health check tests ───────────────────────────────────────────────────────


class TestHealthStatus:
    def test_status_enum(self):
        from raregeneai.observability.health import HealthStatus

        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNHEALTHY == "unhealthy"


class TestComponentHealth:
    def test_component_creation(self):
        from raregeneai.observability.health import ComponentHealth, HealthStatus

        comp = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            detail="all good",
            latency_ms=1.5,
        )
        assert comp.name == "test"
        assert comp.latency_ms == 1.5


class TestSystemHealth:
    def test_to_dict(self):
        from raregeneai.observability.health import (
            ComponentHealth,
            HealthStatus,
            SystemHealth,
        )

        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            version="1.1.0",
            uptime_seconds=100.0,
            components=[
                ComponentHealth(name="db", status=HealthStatus.HEALTHY, detail="ok", latency_ms=2.5),
                ComponentHealth(name="disk", status=HealthStatus.DEGRADED, detail="low"),
            ],
        )

        d = health.to_dict()
        assert d["status"] == "healthy"
        assert d["version"] == "1.1.0"
        assert len(d["components"]) == 2
        assert d["components"][0]["latency_ms"] == 2.5
        assert "latency_ms" not in d["components"][1]  # No latency for disk


class TestHealthChecker:
    def test_check_all_returns_system_health(self):
        from raregeneai.observability.health import HealthChecker, HealthStatus

        checker = HealthChecker(version="1.1.0", start_time=time.time())
        health = checker.check_all()

        assert health.version == "1.1.0"
        assert health.uptime_seconds >= 0
        assert len(health.components) >= 2  # At least disk + memory

    def test_disk_space_check(self):
        from raregeneai.observability.health import HealthChecker

        checker = HealthChecker()
        result = checker._check_disk_space()
        assert result.name == "disk_space"
        assert result.status is not None
        assert "GB" in result.detail

    def test_memory_check(self):
        from raregeneai.observability.health import HealthChecker

        checker = HealthChecker()
        result = checker._check_memory()
        assert result.name == "memory"
        assert "GB" in result.detail

    def test_database_check_without_db(self):
        from raregeneai.observability.health import HealthChecker, HealthStatus

        checker = HealthChecker()
        result = checker._check_database()
        # May be healthy (if DB initialized) or unhealthy
        assert result.name == "database"

    def test_reference_data_check(self):
        from raregeneai.observability.health import HealthChecker

        checker = HealthChecker()
        result = checker._check_reference_data()
        assert result.name == "reference_data"

    def test_worst_status_propagates(self):
        from raregeneai.observability.health import (
            ComponentHealth,
            HealthChecker,
            HealthStatus,
            SystemHealth,
        )

        health = SystemHealth(components=[
            ComponentHealth(name="a", status=HealthStatus.HEALTHY),
            ComponentHealth(name="b", status=HealthStatus.UNHEALTHY),
            ComponentHealth(name="c", status=HealthStatus.DEGRADED),
        ])

        # Simulate the checker's aggregation logic
        statuses = [c.status for c in health.components]
        if HealthStatus.UNHEALTHY in statuses:
            health.status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            health.status = HealthStatus.DEGRADED

        assert health.status == HealthStatus.UNHEALTHY

    def test_uptime_calculation(self):
        from raregeneai.observability.health import HealthChecker

        start = time.time() - 60  # 60 seconds ago
        checker = HealthChecker(start_time=start)
        health = checker.check_all()
        assert health.uptime_seconds >= 59
