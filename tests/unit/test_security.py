"""Unit tests for security module: auth, validation, secure temp files."""

import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ── Auth tests ───────────────────────────────────────────────────────────────


class TestPasswordHashing:
    def test_hash_and_verify(self):
        from raregeneai.security.auth import hash_password, verify_password

        hashed = hash_password("TestPass123!")
        assert hashed != "TestPass123!"
        assert verify_password("TestPass123!", hashed)

    def test_wrong_password_fails(self):
        from raregeneai.security.auth import hash_password, verify_password

        hashed = hash_password("CorrectPass!")
        assert not verify_password("WrongPass!!", hashed)

    def test_hash_is_unique(self):
        from raregeneai.security.auth import hash_password

        h1 = hash_password("SamePass123!")
        h2 = hash_password("SamePass123!")
        assert h1 != h2  # bcrypt uses random salt


class TestTokenManagement:
    @patch.dict(os.environ, {"RAREGENEAI_SECRET_KEY": "test-secret-key-for-unit-tests-only"})
    def test_create_and_decode_token(self):
        from raregeneai.security.auth import create_access_token, decode_access_token

        token = create_access_token(data={"sub": "testuser", "role": "analyst"})
        assert isinstance(token, str)

        token_data = decode_access_token(token)
        assert token_data.sub == "testuser"
        assert token_data.role == "analyst"

    @patch.dict(os.environ, {"RAREGENEAI_SECRET_KEY": "test-secret-key-for-unit-tests-only"})
    def test_token_missing_subject(self):
        from fastapi import HTTPException
        from raregeneai.security.auth import create_access_token, decode_access_token

        # Create token without 'sub' field
        from jose import jwt
        token = jwt.encode(
            {"role": "admin"},
            "test-secret-key-for-unit-tests-only",
            algorithm="HS256",
        )
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(token)
        assert exc_info.value.status_code == 401

    @patch.dict(os.environ, {"RAREGENEAI_SECRET_KEY": "test-secret-key-for-unit-tests-only"})
    def test_invalid_token_rejected(self):
        from fastapi import HTTPException
        from raregeneai.security.auth import decode_access_token

        with pytest.raises(HTTPException) as exc_info:
            decode_access_token("invalid.token.value")
        assert exc_info.value.status_code == 401

    def test_missing_secret_key_raises(self):
        import importlib
        import raregeneai.security.auth as auth_mod

        with patch.dict(os.environ, {}, clear=True):
            # Reset module-level SECRET_KEY
            original = auth_mod.SECRET_KEY
            auth_mod.SECRET_KEY = ""
            try:
                with pytest.raises(RuntimeError, match="RAREGENEAI_SECRET_KEY"):
                    auth_mod._get_secret_key()
            finally:
                auth_mod.SECRET_KEY = original


class TestUserRoles:
    def test_role_enum_values(self):
        from raregeneai.security.auth import UserRole

        assert UserRole.ADMIN == "admin"
        assert UserRole.ANALYST == "analyst"
        assert UserRole.VIEWER == "viewer"

    def test_user_model(self):
        from raregeneai.security.auth import User, UserRole

        user = User(username="doc1", role=UserRole.ANALYST, institution="Hospital A")
        assert user.username == "doc1"
        assert user.role == UserRole.ANALYST


# ── Validation tests ─────────────────────────────────────────────────────────


class TestHPOValidation:
    def test_valid_hpo_terms(self):
        from raregeneai.security.validation import validate_hpo_terms

        result = validate_hpo_terms(["HP:0001250", "HP:0002878"])
        assert len(result) == 2
        assert "HP:0001250" in result

    def test_duplicate_removal(self):
        from raregeneai.security.validation import validate_hpo_terms

        result = validate_hpo_terms(["HP:0001250", "HP:0001250", "HP:0001250"])
        assert len(result) == 1

    def test_invalid_hpo_raises(self):
        from fastapi import HTTPException
        from raregeneai.security.validation import validate_hpo_terms

        with pytest.raises(HTTPException) as exc_info:
            validate_hpo_terms(["INVALID_TERM"])
        assert exc_info.value.status_code == 400

    def test_empty_list_raises(self):
        from fastapi import HTTPException
        from raregeneai.security.validation import validate_hpo_terms

        with pytest.raises(HTTPException):
            validate_hpo_terms([])

    def test_too_many_terms_raises(self):
        from fastapi import HTTPException
        from raregeneai.security.validation import validate_hpo_terms

        terms = [f"HP:{i:07d}" for i in range(101)]
        with pytest.raises(HTTPException) as exc_info:
            validate_hpo_terms(terms)
        assert exc_info.value.status_code == 400

    def test_mixed_valid_invalid(self):
        from fastapi import HTTPException
        from raregeneai.security.validation import validate_hpo_terms

        with pytest.raises(HTTPException):
            validate_hpo_terms(["HP:0001250", "bad_term"])

    def test_whitespace_stripped(self):
        from raregeneai.security.validation import validate_hpo_terms

        result = validate_hpo_terms(["  HP:0001250  "])
        assert "HP:0001250" in result


class TestPatientIDValidation:
    def test_valid_ids(self):
        from raregeneai.security.validation import validate_patient_id

        assert validate_patient_id("PATIENT_001") == "PATIENT_001"
        assert validate_patient_id("P-123.test") == "P-123.test"
        assert validate_patient_id("a") == "a"

    def test_invalid_ids(self):
        from fastapi import HTTPException
        from raregeneai.security.validation import validate_patient_id

        with pytest.raises(HTTPException):
            validate_patient_id("patient with spaces")

        with pytest.raises(HTTPException):
            validate_patient_id("patient;DROP TABLE")

        with pytest.raises(HTTPException):
            validate_patient_id("<script>alert(1)</script>")

        with pytest.raises(HTTPException):
            validate_patient_id("")

    def test_max_length(self):
        from fastapi import HTTPException
        from raregeneai.security.validation import validate_patient_id

        # 128 chars should pass
        assert validate_patient_id("A" * 128)

        # 129 chars should fail
        with pytest.raises(HTTPException):
            validate_patient_id("A" * 129)


class TestSampleIDValidation:
    def test_none_passes(self):
        from raregeneai.security.validation import validate_sample_id

        assert validate_sample_id(None) is None

    def test_valid_sample_id(self):
        from raregeneai.security.validation import validate_sample_id

        assert validate_sample_id("SAMPLE_01") == "SAMPLE_01"

    def test_invalid_sample_id(self):
        from fastapi import HTTPException
        from raregeneai.security.validation import validate_sample_id

        with pytest.raises(HTTPException):
            validate_sample_id("sample with spaces")


class TestFilenameSanitization:
    def test_basic_filename(self):
        from raregeneai.security.validation import sanitize_filename

        assert sanitize_filename("test.vcf") == "test.vcf"

    def test_path_traversal_stripped(self):
        from raregeneai.security.validation import sanitize_filename

        result = sanitize_filename("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_special_chars_replaced(self):
        from raregeneai.security.validation import sanitize_filename

        result = sanitize_filename("file name (1).vcf")
        assert " " not in result
        assert "(" not in result

    def test_dotfile_prefixed(self):
        from raregeneai.security.validation import sanitize_filename

        result = sanitize_filename(".hidden")
        assert result.startswith("upload")

    def test_max_length_truncated(self):
        from raregeneai.security.validation import sanitize_filename

        result = sanitize_filename("a" * 300)
        assert len(result) <= 255


# ── Secure temp file tests ───────────────────────────────────────────────────


class TestSecureTempFile:
    def test_creates_and_cleans_up(self):
        from raregeneai.security.secure_temp import SecureTempFile

        with SecureTempFile(suffix=".txt", content=b"test data") as path:
            assert path.exists()
            assert path.read_bytes() == b"test data"
            # Check restrictive permissions
            mode = oct(path.stat().st_mode)[-3:]
            assert mode == "600"
            saved_path = path

        # File should be deleted after exit
        assert not saved_path.exists()

    def test_async_context_manager(self):
        import asyncio
        from raregeneai.security.secure_temp import SecureTempFile

        async def _test():
            async with SecureTempFile(suffix=".vcf", content=b"##fileformat=VCF") as path:
                assert path.exists()
                saved = path
            assert not saved.exists()

        asyncio.run(_test())

    def test_empty_content(self):
        from raregeneai.security.secure_temp import SecureTempFile

        with SecureTempFile(suffix=".txt") as path:
            assert path.exists()
            assert path.stat().st_size == 0

    def test_secure_directory_permissions(self):
        from raregeneai.security.secure_temp import _ensure_secure_temp_dir

        secure_dir = _ensure_secure_temp_dir()
        mode = oct(secure_dir.stat().st_mode)[-3:]
        assert mode == "700"

    def test_cleanup_on_exception(self):
        from raregeneai.security.secure_temp import SecureTempFile

        saved_path = None
        try:
            with SecureTempFile(suffix=".txt", content=b"data") as path:
                saved_path = path
                raise ValueError("test error")
        except ValueError:
            pass

        assert saved_path is not None
        assert not saved_path.exists()
