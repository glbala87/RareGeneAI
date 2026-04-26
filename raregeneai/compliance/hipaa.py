"""HIPAA compliance controls for RareGeneAI.

Implements the HIPAA Security Rule technical safeguards:
  - Access controls (§164.312(a))
  - Audit controls (§164.312(b))
  - Integrity controls (§164.312(c))
  - Transmission security (§164.312(e))
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class ComplianceCheck:
    """Result of a single compliance check."""
    name: str
    category: str  # "access", "audit", "integrity", "transmission"
    passed: bool
    detail: str
    severity: str = "critical"  # "critical", "warning", "info"


@dataclass
class ComplianceReport:
    """Full compliance assessment report."""
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    checks: list[ComplianceCheck] = field(default_factory=list)
    passed: bool = False
    critical_failures: int = 0
    warnings: int = 0

    def summary(self) -> str:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        return (
            f"HIPAA Compliance: {passed}/{total} checks passed | "
            f"{self.critical_failures} critical | {self.warnings} warnings"
        )


class HIPAAComplianceChecker:
    """Run HIPAA compliance checks against current configuration."""

    def run_all_checks(self) -> ComplianceReport:
        """Execute all compliance checks and return a report."""
        report = ComplianceReport()

        # Access controls (§164.312(a))
        report.checks.extend(self._check_access_controls())

        # Audit controls (§164.312(b))
        report.checks.extend(self._check_audit_controls())

        # Integrity controls (§164.312(c))
        report.checks.extend(self._check_integrity_controls())

        # Transmission security (§164.312(e))
        report.checks.extend(self._check_transmission_security())

        # Compute summary
        report.critical_failures = sum(
            1 for c in report.checks if not c.passed and c.severity == "critical"
        )
        report.warnings = sum(
            1 for c in report.checks if not c.passed and c.severity == "warning"
        )
        report.passed = report.critical_failures == 0

        logger.info(report.summary())
        return report

    def _check_access_controls(self) -> list[ComplianceCheck]:
        """§164.312(a) - Access controls."""
        checks = []

        # Check JWT secret key is configured
        secret = os.environ.get("RAREGENEAI_SECRET_KEY", "")
        checks.append(ComplianceCheck(
            name="JWT Secret Key",
            category="access",
            passed=bool(secret) and len(secret) >= 32,
            detail="RAREGENEAI_SECRET_KEY must be set (min 32 chars)" if not secret
                   else "JWT signing key is configured",
            severity="critical",
        ))

        # Check encryption key is configured
        enc_key = os.environ.get("RAREGENEAI_ENCRYPTION_KEY", "")
        checks.append(ComplianceCheck(
            name="PHI Encryption Key",
            category="access",
            passed=bool(enc_key),
            detail="RAREGENEAI_ENCRYPTION_KEY must be set for PHI encryption" if not enc_key
                   else "PHI encryption key is configured",
            severity="critical",
        ))

        # Check CORS is restricted
        origins = os.environ.get("RAREGENEAI_ALLOWED_ORIGINS", "*")
        checks.append(ComplianceCheck(
            name="CORS Restriction",
            category="access",
            passed=origins != "*",
            detail="CORS must not allow all origins (*)" if origins == "*"
                   else f"CORS restricted to: {origins}",
            severity="critical",
        ))

        # Check secure temp directory
        temp_dir = os.environ.get("RAREGENEAI_SECURE_TEMP_DIR", "")
        checks.append(ComplianceCheck(
            name="Secure Temp Directory",
            category="access",
            passed=bool(temp_dir),
            detail="RAREGENEAI_SECURE_TEMP_DIR should be set to a restricted directory"
                   if not temp_dir else f"Secure temp dir: {temp_dir}",
            severity="warning",
        ))

        return checks

    def _check_audit_controls(self) -> list[ComplianceCheck]:
        """§164.312(b) - Audit controls."""
        checks = []

        # Check database is configured (for audit log persistence)
        db_url = os.environ.get("RAREGENEAI_DATABASE_URL", "")
        checks.append(ComplianceCheck(
            name="Audit Log Database",
            category="audit",
            passed=bool(db_url) and "sqlite" not in db_url,
            detail="Production must use PostgreSQL for audit logs, not SQLite"
                   if not db_url or "sqlite" in db_url
                   else "Audit log database is configured",
            severity="critical",
        ))

        # Check log file is configured
        log_dir = os.environ.get("RAREGENEAI_LOG_DIR", "")
        checks.append(ComplianceCheck(
            name="Application Log Directory",
            category="audit",
            passed=bool(log_dir),
            detail="RAREGENEAI_LOG_DIR should be set for persistent application logs"
                   if not log_dir else f"Log directory: {log_dir}",
            severity="warning",
        ))

        return checks

    def _check_integrity_controls(self) -> list[ComplianceCheck]:
        """§164.312(c) - Integrity controls."""
        checks = []

        # Check database URL uses SSL for PostgreSQL
        db_url = os.environ.get("RAREGENEAI_DATABASE_URL", "")
        if db_url and "postgresql" in db_url:
            checks.append(ComplianceCheck(
                name="Database SSL",
                category="integrity",
                passed="sslmode=" in db_url,
                detail="PostgreSQL connection should use sslmode=require or sslmode=verify-full"
                       if "sslmode=" not in db_url
                       else "Database SSL is configured",
                severity="warning",
            ))

        return checks

    def _check_transmission_security(self) -> list[ComplianceCheck]:
        """§164.312(e) - Transmission security."""
        checks = []

        # Check allowed origins use HTTPS
        origins = os.environ.get("RAREGENEAI_ALLOWED_ORIGINS", "")
        if origins and origins != "*":
            non_https = [o for o in origins.split(",") if o.strip() and not o.strip().startswith("https://")]
            # Allow localhost for development
            non_https = [o for o in non_https if "localhost" not in o and "127.0.0.1" not in o]
            checks.append(ComplianceCheck(
                name="HTTPS Origins",
                category="transmission",
                passed=len(non_https) == 0,
                detail=f"Non-HTTPS origins found: {non_https}" if non_https
                       else "All non-local origins use HTTPS",
                severity="warning",
            ))

        return checks
