"""Database module for RareGeneAI.

Provides:
  - PostgreSQL connection management via SQLAlchemy
  - Encrypted PHI field storage
  - Audit logging
  - Job persistence (replaces in-memory store)
"""

from raregeneai.database.connection import (
    get_db,
    init_db,
    DatabaseConfig,
)
from raregeneai.database.models import (
    AnalysisJob,
    AuditLog,
    UserAccount,
)
from raregeneai.database.encryption import FieldEncryptor

__all__ = [
    "get_db",
    "init_db",
    "DatabaseConfig",
    "AnalysisJob",
    "AuditLog",
    "UserAccount",
    "FieldEncryptor",
]
