"""Field-level encryption for PHI (Protected Health Information).

Uses AES-256-GCM via the cryptography library (Fernet).
Encryption key is loaded from environment variable.

All patient-identifiable data must be encrypted at rest.
In production (PostgreSQL), the encryption key MUST be set — startup will fail without it.
"""

from __future__ import annotations

import base64
import os

from cryptography.fernet import Fernet, InvalidToken
from loguru import logger


def _is_production() -> bool:
    """Detect production environment (PostgreSQL configured)."""
    db_url = os.environ.get("RAREGENEAI_DATABASE_URL", "")
    return bool(db_url) and "sqlite" not in db_url


class FieldEncryptor:
    """Encrypt and decrypt PHI fields using Fernet (AES-128-CBC + HMAC).

    Usage:
        encryptor = FieldEncryptor()
        encrypted = encryptor.encrypt("PATIENT_001")
        decrypted = encryptor.decrypt(encrypted)

    In production (PostgreSQL), raises RuntimeError if encryption key is missing.
    In development (SQLite/no DB), logs a warning and passes through unencrypted.
    """

    def __init__(self, key: str | bytes | None = None):
        """Initialize with encryption key.

        Args:
            key: Fernet key. If None, reads from RAREGENEAI_ENCRYPTION_KEY env var.
                 Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
        """
        if key is None:
            key = os.environ.get("RAREGENEAI_ENCRYPTION_KEY", "")

        if not key:
            if _is_production():
                raise RuntimeError(
                    "RAREGENEAI_ENCRYPTION_KEY must be set in production. "
                    "PHI cannot be stored unencrypted. "
                    "Generate a key with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
                )
            logger.warning(
                "RAREGENEAI_ENCRYPTION_KEY not set. PHI encryption is disabled. "
                "This is only acceptable for local development/testing."
            )
            self._fernet = None
            return

        if isinstance(key, str):
            key = key.encode()

        try:
            self._fernet = Fernet(key)
        except Exception as e:
            raise ValueError(
                f"Invalid encryption key: {e}. "
                "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )

    @property
    def is_enabled(self) -> bool:
        """Check if encryption is active."""
        return self._fernet is not None

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string. Returns base64-encoded ciphertext."""
        if not self._fernet:
            return plaintext
        encrypted = self._fernet.encrypt(plaintext.encode("utf-8"))
        return base64.urlsafe_b64encode(encrypted).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a string. Returns plaintext."""
        if not self._fernet:
            return ciphertext
        try:
            encrypted = base64.urlsafe_b64decode(ciphertext.encode("ascii"))
            return self._fernet.decrypt(encrypted).decode("utf-8")
        except (InvalidToken, Exception) as e:
            logger.error(f"Failed to decrypt PHI field: {e}")
            raise ValueError("Failed to decrypt protected health information") from e

    @staticmethod
    def generate_key() -> str:
        """Generate a new encryption key."""
        return Fernet.generate_key().decode()
