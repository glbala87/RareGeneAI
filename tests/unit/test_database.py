"""Unit tests for database module: connection, encryption, audit, models."""

import os
from unittest.mock import patch

import pytest


# ── Encryption tests ─────────────────────────────────────────────────────────


class TestFieldEncryptor:
    def test_encrypt_decrypt_roundtrip(self):
        from raregeneai.database.encryption import FieldEncryptor

        key = FieldEncryptor.generate_key()
        enc = FieldEncryptor(key=key)
        assert enc.is_enabled

        plaintext = "PATIENT_001"
        ciphertext = enc.encrypt(plaintext)
        assert ciphertext != plaintext
        assert enc.decrypt(ciphertext) == plaintext

    def test_different_encryptions_differ(self):
        from raregeneai.database.encryption import FieldEncryptor

        key = FieldEncryptor.generate_key()
        enc = FieldEncryptor(key=key)

        c1 = enc.encrypt("PATIENT_001")
        c2 = enc.encrypt("PATIENT_001")
        # Fernet includes timestamp, so same plaintext produces different ciphertext
        assert c1 != c2

    def test_wrong_key_fails_decrypt(self):
        from raregeneai.database.encryption import FieldEncryptor

        enc1 = FieldEncryptor(key=FieldEncryptor.generate_key())
        enc2 = FieldEncryptor(key=FieldEncryptor.generate_key())

        ciphertext = enc1.encrypt("SECRET")
        with pytest.raises(ValueError, match="Failed to decrypt"):
            enc2.decrypt(ciphertext)

    def test_disabled_without_key(self):
        from raregeneai.database.encryption import FieldEncryptor

        with patch.dict(os.environ, {"RAREGENEAI_DATABASE_URL": ""}, clear=False):
            os.environ.pop("RAREGENEAI_ENCRYPTION_KEY", None)
            enc = FieldEncryptor(key="")
            assert not enc.is_enabled
            # Passthrough mode
            assert enc.encrypt("test") == "test"
            assert enc.decrypt("test") == "test"

    def test_production_fails_without_key(self):
        from raregeneai.database.encryption import FieldEncryptor

        with patch.dict(os.environ, {
            "RAREGENEAI_DATABASE_URL": "postgresql://user:pass@host/db",
            "RAREGENEAI_ENCRYPTION_KEY": "",
        }):
            with pytest.raises(RuntimeError, match="RAREGENEAI_ENCRYPTION_KEY must be set"):
                FieldEncryptor()

    def test_invalid_key_raises(self):
        from raregeneai.database.encryption import FieldEncryptor

        with pytest.raises(ValueError, match="Invalid encryption key"):
            FieldEncryptor(key="not-a-valid-fernet-key")

    def test_generate_key_format(self):
        from raregeneai.database.encryption import FieldEncryptor

        key = FieldEncryptor.generate_key()
        assert isinstance(key, str)
        assert len(key) == 44  # Base64-encoded 32 bytes

    def test_encrypt_empty_string(self):
        from raregeneai.database.encryption import FieldEncryptor

        enc = FieldEncryptor(key=FieldEncryptor.generate_key())
        ciphertext = enc.encrypt("")
        assert enc.decrypt(ciphertext) == ""

    def test_encrypt_unicode(self):
        from raregeneai.database.encryption import FieldEncryptor

        enc = FieldEncryptor(key=FieldEncryptor.generate_key())
        plaintext = "患者 αβγ 🧬"
        assert enc.decrypt(enc.encrypt(plaintext)) == plaintext


# ── Database connection tests ────────────────────────────────────────────────


class TestDatabaseConnection:
    def test_init_db_sqlite(self):
        from raregeneai.database.connection import DatabaseConfig, init_db, get_session

        config = DatabaseConfig(database_url="sqlite:///test_raregeneai.db")
        init_db(config)

        session = get_session()
        result = session.execute(__import__("sqlalchemy").text("SELECT 1")).scalar()
        assert result == 1
        session.close()

        # Cleanup
        import os
        os.unlink("test_raregeneai.db")

    def test_tables_created(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from raregeneai.database.models import Base, UserAccount, AnalysisJob, AuditLog

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Should be able to query empty tables without error
        assert session.query(UserAccount).count() == 0
        assert session.query(AnalysisJob).count() == 0
        assert session.query(AuditLog).count() == 0
        session.close()


# ── Model tests ──────────────────────────────────────────────────────────────


class TestModels:
    def _get_session(self):
        from raregeneai.database.connection import DatabaseConfig, init_db, get_session
        config = DatabaseConfig(database_url="sqlite:///:memory:")
        init_db(config)
        return get_session()

    def test_create_user(self):
        from raregeneai.database.models import UserAccount

        session = self._get_session()
        user = UserAccount(
            username="testuser",
            hashed_password="fakehash",
            role="analyst",
            institution="Test Hospital",
        )
        session.add(user)
        session.commit()

        loaded = session.query(UserAccount).filter_by(username="testuser").first()
        assert loaded is not None
        assert loaded.role == "analyst"
        assert loaded.is_active is True
        session.close()

    def test_create_analysis_job(self):
        from raregeneai.database.models import AnalysisJob

        session = self._get_session()
        job = AnalysisJob(
            job_id="abc12345",
            patient_id_encrypted="encrypted_data",
            status="running",
            genome_build="GRCh38",
        )
        session.add(job)
        session.commit()

        loaded = session.query(AnalysisJob).filter_by(job_id="abc12345").first()
        assert loaded is not None
        assert loaded.status == "running"
        assert loaded.is_purged is False
        session.close()

    def test_create_audit_log(self):
        from raregeneai.database.models import AuditLog

        session = self._get_session()
        log = AuditLog(
            username="admin",
            action="phi_access",
            resource_type="patient_data",
            resource_id="job123",
            success=True,
        )
        session.add(log)
        session.commit()

        loaded = session.query(AuditLog).first()
        assert loaded.action == "phi_access"
        assert loaded.username == "admin"
        assert loaded.timestamp is not None
        session.close()

    def test_user_job_relationship(self):
        from raregeneai.database.models import UserAccount, AnalysisJob

        session = self._get_session()

        user = UserAccount(username="doc1", hashed_password="hash", role="analyst")
        session.add(user)
        session.flush()

        job = AnalysisJob(
            job_id="rel12345",
            patient_id_encrypted="enc",
            user_id=user.id,
        )
        session.add(job)
        session.commit()

        loaded_user = session.query(UserAccount).filter_by(username="doc1").first()
        assert len(loaded_user.jobs) == 1
        assert loaded_user.jobs[0].job_id == "rel12345"
        session.close()


# ── Audit service tests ──────────────────────────────────────────────────────


class TestAuditService:
    def _get_session(self):
        from raregeneai.database.connection import DatabaseConfig, init_db, get_session
        config = DatabaseConfig(database_url="sqlite:///:memory:")
        init_db(config)
        return get_session()

    def test_log_event(self):
        from raregeneai.database.audit import AuditService
        from raregeneai.database.models import AuditLog

        session = self._get_session()
        AuditService.log(
            session,
            username="testuser",
            action="test_action",
            resource_type="test_resource",
            resource_id="123",
            detail="test detail",
        )
        session.commit()

        logs = session.query(AuditLog).all()
        assert len(logs) == 1
        assert logs[0].action == "test_action"
        assert logs[0].username == "testuser"
        session.close()

    def test_log_phi_access(self):
        from raregeneai.database.audit import AuditService
        from raregeneai.database.models import AuditLog

        session = self._get_session()
        AuditService.log_phi_access(
            session,
            username="analyst1",
            patient_id="P001",
            job_id="job123",
            access_type="view",
        )
        session.commit()

        log = session.query(AuditLog).first()
        assert log.action == "phi_view"
        assert log.resource_type == "patient_data"
        # Detail should NOT contain the actual patient ID
        assert "P001" not in (log.detail or "")
        session.close()

    def test_log_auth_event(self):
        from raregeneai.database.audit import AuditService
        from raregeneai.database.models import AuditLog

        session = self._get_session()
        AuditService.log_auth_event(
            session,
            username="attacker",
            action="login",
            success=False,
            detail="Invalid credentials",
        )
        session.commit()

        log = session.query(AuditLog).first()
        assert log.action == "auth_login"
        assert log.success is False
        session.close()

    def test_multiple_audit_entries(self):
        from raregeneai.database.audit import AuditService
        from raregeneai.database.models import AuditLog

        session = self._get_session()
        for i in range(10):
            AuditService.log(
                session, username="user", action=f"action_{i}",
            )
        session.commit()

        assert session.query(AuditLog).count() == 10
        session.close()
