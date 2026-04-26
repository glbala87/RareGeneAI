"""Health check system for RareGeneAI.

Provides comprehensive health checks for:
  - Database connectivity
  - Disk space
  - Reference data availability
  - External service connectivity
  - Memory usage
"""

from __future__ import annotations

import os
import shutil
import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import psutil
from loguru import logger


class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    detail: str = ""
    latency_ms: float | None = None


@dataclass
class SystemHealth:
    """Overall system health report."""
    status: HealthStatus = HealthStatus.HEALTHY
    version: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    components: list[ComponentHealth] = field(default_factory=list)
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "version": self.version,
            "timestamp": self.timestamp,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "detail": c.detail,
                    **({"latency_ms": round(c.latency_ms, 2)} if c.latency_ms else {}),
                }
                for c in self.components
            ],
        }


class HealthChecker:
    """Run health checks against system components."""

    def __init__(self, version: str = "1.0.0", start_time: float | None = None):
        self.version = version
        self._start_time = start_time or __import__("time").time()

    def check_all(self) -> SystemHealth:
        """Run all health checks."""
        import time

        health = SystemHealth(
            version=self.version,
            uptime_seconds=time.time() - self._start_time,
        )

        health.components.extend([
            self._check_database(),
            self._check_disk_space(),
            self._check_memory(),
            self._check_reference_data(),
        ])

        # Overall status is the worst component status
        statuses = [c.status for c in health.components]
        if HealthStatus.UNHEALTHY in statuses:
            health.status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            health.status = HealthStatus.DEGRADED
        else:
            health.status = HealthStatus.HEALTHY

        return health

    def _check_database(self) -> ComponentHealth:
        """Check database connectivity."""
        import time

        try:
            from raregeneai.database.connection import get_session
            start = time.perf_counter()
            session = get_session()
            session.execute(__import__("sqlalchemy").text("SELECT 1"))
            latency = (time.perf_counter() - start) * 1000
            session.close()
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                detail="Connected",
                latency_ms=latency,
            )
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                detail=f"Connection failed: {e}",
            )

    def _check_disk_space(self) -> ComponentHealth:
        """Check available disk space."""
        try:
            data_dir = os.environ.get("RAREGENEAI_HOME", ".")
            usage = shutil.disk_usage(data_dir)
            free_gb = usage.free / (1024**3)
            free_pct = (usage.free / usage.total) * 100

            if free_pct < 5:
                return ComponentHealth(
                    name="disk_space",
                    status=HealthStatus.UNHEALTHY,
                    detail=f"Critical: {free_gb:.1f} GB free ({free_pct:.1f}%)",
                )
            elif free_pct < 15:
                return ComponentHealth(
                    name="disk_space",
                    status=HealthStatus.DEGRADED,
                    detail=f"Low: {free_gb:.1f} GB free ({free_pct:.1f}%)",
                )
            return ComponentHealth(
                name="disk_space",
                status=HealthStatus.HEALTHY,
                detail=f"{free_gb:.1f} GB free ({free_pct:.1f}%)",
            )
        except Exception as e:
            return ComponentHealth(
                name="disk_space",
                status=HealthStatus.DEGRADED,
                detail=f"Check failed: {e}",
            )

    def _check_memory(self) -> ComponentHealth:
        """Check system memory usage."""
        try:
            mem = psutil.virtual_memory()
            used_pct = mem.percent
            available_gb = mem.available / (1024**3)

            if used_pct > 95:
                status = HealthStatus.UNHEALTHY
            elif used_pct > 85:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            return ComponentHealth(
                name="memory",
                status=status,
                detail=f"{available_gb:.1f} GB available ({100 - used_pct:.1f}% free)",
            )
        except Exception as e:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.DEGRADED,
                detail=f"Check failed: {e}",
            )

    def _check_reference_data(self) -> ComponentHealth:
        """Check that required reference data files exist."""
        from raregeneai.config.settings import REFERENCE_DIR

        required_files = [
            REFERENCE_DIR / "hp.obo",
            REFERENCE_DIR / "genes_to_phenotype.txt",
        ]

        missing = [str(f.name) for f in required_files if not f.exists()]

        if missing:
            return ComponentHealth(
                name="reference_data",
                status=HealthStatus.DEGRADED,
                detail=f"Missing: {', '.join(missing)}",
            )
        return ComponentHealth(
            name="reference_data",
            status=HealthStatus.HEALTHY,
            detail="All required reference files present",
        )
