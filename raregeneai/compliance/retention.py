"""Data retention policies for RareGeneAI.

Implements configurable retention periods for different data types
with automatic purging of expired data.

HIPAA requires:
  - Audit logs: 6 years minimum
  - Clinical data: Per institutional policy (typically 7-10 years)
  - Temporary/working data: Purge after analysis completion
"""

from __future__ import annotations

import datetime
import os

from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import and_
from sqlalchemy.orm import Session

from raregeneai.database.models import AnalysisJob, AuditLog


class RetentionPolicy(BaseModel):
    """Data retention policy configuration."""
    # Analysis results retention (days)
    analysis_retention_days: int = Field(
        default=int(os.environ.get("RAREGENEAI_ANALYSIS_RETENTION_DAYS", "2555")),  # ~7 years
        description="Days to retain analysis results",
    )

    # Audit log retention (days) - HIPAA minimum is 6 years
    audit_log_retention_days: int = Field(
        default=int(os.environ.get("RAREGENEAI_AUDIT_RETENTION_DAYS", "2190")),  # 6 years
        description="Days to retain audit logs (HIPAA min: 6 years = 2190 days)",
    )

    # Temporary data (hours)
    temp_data_retention_hours: int = Field(
        default=24,
        description="Hours to retain temporary working data",
    )

    # Purged data handling
    purge_method: str = Field(
        default="anonymize",
        description="How to handle expired data: 'delete' or 'anonymize'",
    )


class RetentionService:
    """Service for enforcing data retention policies."""

    def __init__(self, policy: RetentionPolicy | None = None):
        self.policy = policy or RetentionPolicy()

    def set_retention_deadline(
        self,
        db: Session,
        job: AnalysisJob,
    ) -> None:
        """Set the retention expiration date for a new analysis job."""
        job.retention_expires_at = (
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=self.policy.analysis_retention_days)
        )
        db.add(job)

    def purge_expired_jobs(self, db: Session) -> int:
        """Purge analysis jobs past their retention deadline.

        Returns the number of purged records.
        """
        now = datetime.datetime.now(datetime.timezone.utc)

        expired_jobs = db.query(AnalysisJob).filter(
            and_(
                AnalysisJob.retention_expires_at <= now,
                AnalysisJob.is_purged == False,  # noqa: E712
            )
        ).all()

        count = 0
        for job in expired_jobs:
            if self.policy.purge_method == "anonymize":
                # Replace PHI with anonymized markers
                job.patient_id_encrypted = "PURGED"
                job.sample_id_encrypted = "PURGED"
                job.results_json = None
                job.is_purged = True
            else:
                db.delete(job)
            count += 1

        if count:
            db.commit()
            logger.info(f"Retention: purged {count} expired analysis jobs")

        return count

    def purge_expired_audit_logs(self, db: Session) -> int:
        """Purge audit logs past retention period.

        CAUTION: HIPAA requires 6 years minimum. This method enforces
        that the configured retention is at least 2190 days.
        """
        if self.policy.audit_log_retention_days < 2190:
            logger.warning(
                f"Audit log retention ({self.policy.audit_log_retention_days} days) "
                f"is below HIPAA minimum (2190 days). Enforcing minimum."
            )
            effective_days = 2190
        else:
            effective_days = self.policy.audit_log_retention_days

        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=effective_days)

        count = db.query(AuditLog).filter(AuditLog.timestamp < cutoff).delete()

        if count:
            db.commit()
            logger.info(f"Retention: purged {count} expired audit log entries")

        return count

    def get_retention_report(self, db: Session) -> dict:
        """Generate a report on data retention status."""
        now = datetime.datetime.now(datetime.timezone.utc)

        total_jobs = db.query(AnalysisJob).count()
        purged_jobs = db.query(AnalysisJob).filter(AnalysisJob.is_purged == True).count()  # noqa: E712
        expiring_soon = db.query(AnalysisJob).filter(
            and_(
                AnalysisJob.retention_expires_at <= now + datetime.timedelta(days=90),
                AnalysisJob.is_purged == False,  # noqa: E712
            )
        ).count()

        total_audit = db.query(AuditLog).count()
        oldest_audit = db.query(AuditLog).order_by(AuditLog.timestamp.asc()).first()

        return {
            "policy": self.policy.model_dump(),
            "analysis_jobs": {
                "total": total_jobs,
                "purged": purged_jobs,
                "expiring_within_90_days": expiring_soon,
            },
            "audit_logs": {
                "total": total_audit,
                "oldest_entry": oldest_audit.timestamp.isoformat() if oldest_audit else None,
            },
        }
