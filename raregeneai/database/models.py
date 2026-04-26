"""SQLAlchemy models for RareGeneAI.

Tables:
  - user_accounts: User authentication and roles
  - analysis_jobs: Pipeline job tracking and results
  - audit_logs: HIPAA-compliant audit trail
  - data_retention: Tracks data lifecycle for retention policies
"""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# ── User accounts ───────────────────────────────────────────────────────────

class UserAccount(Base):
    """User account for authentication and RBAC."""
    __tablename__ = "user_accounts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(128), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    role = Column(SAEnum("admin", "analyst", "viewer", name="user_role"), default="viewer")
    institution = Column(String(256), default="")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    last_login = Column(DateTime, nullable=True)

    # Relationships
    jobs = relationship("AnalysisJob", back_populates="user")
    audit_entries = relationship("AuditLog", back_populates="user")


# ── Analysis jobs ────────────────────────────────────────────────────────────

class AnalysisJob(Base):
    """Persistent analysis job tracking (replaces in-memory dict)."""
    __tablename__ = "analysis_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String(8), unique=True, nullable=False, index=True)

    # PHI fields - stored encrypted (see encryption.py)
    patient_id_encrypted = Column(Text, nullable=False)
    sample_id_encrypted = Column(Text, nullable=True)

    # Non-PHI metadata
    status = Column(
        SAEnum("pending", "running", "completed", "failed", name="job_status"),
        default="pending",
    )
    genome_build = Column(String(10), default="GRCh38")
    hpo_terms = Column(Text, default="")  # JSON array
    top_n = Column(Integer, default=20)

    # Results (non-PHI aggregated data)
    total_variants = Column(Integer, default=0)
    total_genes = Column(Integer, default=0)
    results_json = Column(Text, nullable=True)  # JSON serialized results
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user_id = Column(String(36), ForeignKey("user_accounts.id"), nullable=True)
    user = relationship("UserAccount", back_populates="jobs")

    # Data retention
    retention_expires_at = Column(DateTime, nullable=True)
    is_purged = Column(Boolean, default=False)

    __table_args__ = (
        Index("ix_analysis_jobs_status", "status"),
        Index("ix_analysis_jobs_created", "created_at"),
        Index("ix_analysis_jobs_retention", "retention_expires_at"),
    )


# ── Audit log ────────────────────────────────────────────────────────────────

class AuditLog(Base):
    """HIPAA-compliant audit trail.

    Records all access to PHI and significant system events.
    This table is append-only - rows are never updated or deleted.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False)

    # Who
    user_id = Column(String(36), ForeignKey("user_accounts.id"), nullable=True)
    username = Column(String(128), nullable=False)
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6
    user_agent = Column(String(512), nullable=True)

    # What
    action = Column(String(64), nullable=False)  # e.g., "phi_access", "analysis_start", "login"
    resource_type = Column(String(64), nullable=True)  # e.g., "analysis_job", "patient_data"
    resource_id = Column(String(128), nullable=True)

    # Details
    detail = Column(Text, nullable=True)  # Additional context (never contains PHI)
    success = Column(Boolean, default=True)

    # Relationships
    user = relationship("UserAccount", back_populates="audit_entries")

    __table_args__ = (
        Index("ix_audit_logs_timestamp", "timestamp"),
        Index("ix_audit_logs_action", "action"),
        Index("ix_audit_logs_user", "username"),
        Index("ix_audit_logs_resource", "resource_type", "resource_id"),
    )
