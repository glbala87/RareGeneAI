"""Input validation and sanitization for RareGeneAI.

Validates:
  - VCF file uploads (size, type, format)
  - HPO term format
  - Patient/sample IDs
  - File names
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

# ── Constants ────────────────────────────────────────────────────────────────

MAX_VCF_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB for non-VCF files
ALLOWED_VCF_EXTENSIONS = {".vcf", ".vcf.gz", ".bcf"}
ALLOWED_VCF_MIME_TYPES = {
    "application/gzip",
    "application/x-gzip",
    "application/octet-stream",
    "text/plain",
    "text/x-vcard",  # Some systems misidentify VCF
}
HPO_PATTERN = re.compile(r"^HP:\d{7}$")
PATIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-\.]{1,128}$")
SAMPLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-\.]{1,128}$")
SAFE_FILENAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-\.]{1,255}$")

# VCF magic bytes
VCF_HEADER_PREFIX = b"##fileformat=VCF"
GZIP_MAGIC = b"\x1f\x8b"


# ── VCF validation ───────────────────────────────────────────────────────────

async def validate_vcf_upload(
    vcf_file: UploadFile,
    max_size: int = MAX_VCF_SIZE_BYTES,
) -> bytes:
    """Validate and read a VCF upload.

    Returns the file content bytes if valid.
    Raises HTTPException if validation fails.
    """
    if not vcf_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="VCF file must have a filename",
        )

    # Check extension
    filename_lower = vcf_file.filename.lower()
    valid_ext = any(filename_lower.endswith(ext) for ext in ALLOWED_VCF_EXTENSIONS)
    if not valid_ext:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension. Allowed: {', '.join(ALLOWED_VCF_EXTENSIONS)}",
        )

    # Read content with size limit
    content = await vcf_file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"VCF file exceeds maximum size of {max_size // (1024**3)} GB",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="VCF file is empty",
        )

    # Check magic bytes
    if content[:2] == GZIP_MAGIC:
        pass  # gzipped VCF - can't check header without decompressing
    elif not content[:16].startswith(VCF_HEADER_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File does not appear to be a valid VCF (missing ##fileformat=VCF header)",
        )

    return content


# ── HPO term validation ──────────────────────────────────────────────────────

def validate_hpo_terms(terms: list[str]) -> list[str]:
    """Validate a list of HPO terms.

    Returns deduplicated, validated terms.
    Raises HTTPException on invalid terms.
    """
    if not terms:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one HPO term is required",
        )

    if len(terms) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 HPO terms allowed per analysis",
        )

    validated = []
    invalid = []
    for term in terms:
        term = term.strip()
        if HPO_PATTERN.match(term):
            validated.append(term)
        else:
            invalid.append(term)

    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid HPO terms (expected HP:NNNNNNN format): {', '.join(invalid[:5])}",
        )

    return list(set(validated))


# ── ID validation ────────────────────────────────────────────────────────────

def validate_patient_id(patient_id: str) -> str:
    """Validate a patient ID string."""
    if not PATIENT_ID_PATTERN.match(patient_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Patient ID must be 1-128 characters: letters, numbers, underscore, hyphen, dot",
        )
    return patient_id


def validate_sample_id(sample_id: str | None) -> str | None:
    """Validate an optional sample ID."""
    if sample_id is None:
        return None
    if not SAMPLE_ID_PATTERN.match(sample_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sample ID must be 1-128 characters: letters, numbers, underscore, hyphen, dot",
        )
    return sample_id


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename, removing path traversal and special characters."""
    # Strip path components
    name = Path(filename).name
    # Replace unsafe characters
    name = re.sub(r"[^\w.\-]", "_", name)
    if not name or name.startswith("."):
        name = "upload" + name
    return name[:255]
