"""Security module for RareGeneAI.

Provides:
  - JWT/OAuth2 authentication
  - Input validation and sanitization
  - Secure temporary file handling
  - CORS configuration
"""

from raregeneai.security.auth import (
    AuthConfig,
    create_access_token,
    get_current_user,
    verify_password,
    hash_password,
)
from raregeneai.security.validation import (
    validate_vcf_upload,
    validate_hpo_terms,
    validate_patient_id,
    sanitize_filename,
)
from raregeneai.security.secure_temp import SecureTempFile

__all__ = [
    "AuthConfig",
    "create_access_token",
    "get_current_user",
    "verify_password",
    "hash_password",
    "validate_vcf_upload",
    "validate_hpo_terms",
    "validate_patient_id",
    "sanitize_filename",
    "SecureTempFile",
]
