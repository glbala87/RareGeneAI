"""Audit logging service for HIPAA compliance.

Records all PHI access, authentication events, and significant
system actions to the audit_logs table.
"""

from __future__ import annotations

from fastapi import Request
from loguru import logger
from sqlalchemy.orm import Session

from raregeneai.database.models import AuditLog


class AuditService:
    """Service for recording audit trail events."""

    @staticmethod
    def log(
        db: Session,
        *,
        username: str,
        action: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        detail: str | None = None,
        success: bool = True,
        request: Request | None = None,
        user_id: str | None = None,
    ) -> None:
        """Record an audit event.

        Args:
            db: Database session
            username: Who performed the action
            action: Action type (e.g., "phi_access", "login", "analysis_start")
            resource_type: Type of resource accessed
            resource_id: ID of resource accessed
            detail: Additional non-PHI context
            success: Whether the action succeeded
            request: FastAPI request for IP/user-agent extraction
            user_id: User account ID (FK)
        """
        ip_address = None
        user_agent = None

        if request:
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent", "")[:512]

        entry = AuditLog(
            username=username,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            detail=detail,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        db.add(entry)
        # Flush immediately - audit logs must be persisted even if the
        # outer transaction rolls back
        db.flush()

        logger.info(
            f"AUDIT: {action} by {username} on {resource_type}:{resource_id} "
            f"success={success}"
        )

    @staticmethod
    def log_phi_access(
        db: Session,
        *,
        username: str,
        patient_id: str,
        job_id: str,
        access_type: str = "view",
        request: Request | None = None,
    ) -> None:
        """Record PHI access (HIPAA requirement)."""
        AuditService.log(
            db,
            username=username,
            action=f"phi_{access_type}",
            resource_type="patient_data",
            resource_id=job_id,
            detail=f"Accessed patient data (access_type={access_type})",
            request=request,
        )

    @staticmethod
    def log_auth_event(
        db: Session,
        *,
        username: str,
        action: str,
        success: bool,
        request: Request | None = None,
        detail: str | None = None,
    ) -> None:
        """Record authentication event."""
        AuditService.log(
            db,
            username=username,
            action=f"auth_{action}",
            resource_type="authentication",
            success=success,
            detail=detail,
            request=request,
        )
