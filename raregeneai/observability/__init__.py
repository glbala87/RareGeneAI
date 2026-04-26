"""Observability module for RareGeneAI.

Provides:
  - Prometheus metrics
  - Comprehensive health checks
  - Alerting integration
"""

from raregeneai.observability.metrics import MetricsCollector, metrics
from raregeneai.observability.health import HealthChecker, HealthStatus

__all__ = [
    "MetricsCollector",
    "metrics",
    "HealthChecker",
    "HealthStatus",
]
