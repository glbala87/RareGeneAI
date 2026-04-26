"""Unit tests for compliance module: HIPAA checks, data retention."""

import datetime
import os
from unittest.mock import patch

import pytest


# ── HIPAA compliance checker tests ───────────────────────────────────────────


class TestHIPAAComplianceChecker:
    def test_all_checks_fail_without_config(self):
        from raregeneai.compliance.hipaa import HIPAAComplianceChecker

        with patch.dict(os.environ, {}, clear=True):
            checker = HIPAAComplianceChecker()
            report = checker.run_all_checks()
            assert not report.passed
            assert report.critical_failures > 0

    def test_all_checks_pass_with_config(self):
        from raregeneai.compliance.hipaa import HIPAAComplianceChecker

        env = {
            "RAREGENEAI_SECRET_KEY": "a" * 64,
            "RAREGENEAI_ENCRYPTION_KEY": "test-key-32-chars-long-enough!!",
            "RAREGENEAI_ALLOWED_ORIGINS": "https://app.example.com",
            "RAREGENEAI_DATABASE_URL": "postgresql://user:pass@host/db?sslmode=require",
            "RAREGENEAI_LOG_DIR": "/var/log/raregeneai",
            "RAREGENEAI_SECURE_TEMP_DIR": "/secure/tmp",
        }
        with patch.dict(os.environ, env, clear=True):
            checker = HIPAAComplianceChecker()
            report = checker.run_all_checks()
            assert report.passed
            assert report.critical_failures == 0

    def test_cors_wildcard_fails(self):
        from raregeneai.compliance.hipaa import HIPAAComplianceChecker

        with patch.dict(os.environ, {"RAREGENEAI_ALLOWED_ORIGINS": "*"}, clear=False):
            checker = HIPAAComplianceChecker()
            report = checker.run_all_checks()
            cors_check = next(c for c in report.checks if c.name == "CORS Restriction")
            assert not cors_check.passed

    def test_short_secret_key_fails(self):
        from raregeneai.compliance.hipaa import HIPAAComplianceChecker

        with patch.dict(os.environ, {"RAREGENEAI_SECRET_KEY": "short"}, clear=False):
            checker = HIPAAComplianceChecker()
            report = checker.run_all_checks()
            key_check = next(c for c in report.checks if c.name == "JWT Secret Key")
            assert not key_check.passed

    def test_sqlite_audit_db_fails(self):
        from raregeneai.compliance.hipaa import HIPAAComplianceChecker

        with patch.dict(os.environ, {"RAREGENEAI_DATABASE_URL": "sqlite:///test.db"}, clear=False):
            checker = HIPAAComplianceChecker()
            report = checker.run_all_checks()
            db_check = next(c for c in report.checks if c.name == "Audit Log Database")
            assert not db_check.passed

    def test_report_summary_format(self):
        from raregeneai.compliance.hipaa import HIPAAComplianceChecker

        checker = HIPAAComplianceChecker()
        report = checker.run_all_checks()
        summary = report.summary()
        assert "HIPAA Compliance:" in summary
        assert "checks passed" in summary

    def test_check_categories(self):
        from raregeneai.compliance.hipaa import HIPAAComplianceChecker

        checker = HIPAAComplianceChecker()
        report = checker.run_all_checks()
        categories = {c.category for c in report.checks}
        assert "access" in categories
        assert "audit" in categories


# ── Retention policy tests ───────────────────────────────────────────────────


class TestRetentionPolicy:
    def test_default_policy_values(self):
        from raregeneai.compliance.retention import RetentionPolicy

        policy = RetentionPolicy()
        assert policy.analysis_retention_days >= 365  # At least 1 year
        assert policy.audit_log_retention_days >= 2190  # HIPAA minimum 6 years

    def test_custom_policy(self):
        from raregeneai.compliance.retention import RetentionPolicy

        policy = RetentionPolicy(
            analysis_retention_days=3650,
            audit_log_retention_days=3650,
        )
        assert policy.analysis_retention_days == 3650


class TestRetentionService:
    def _get_session(self):
        from raregeneai.database.connection import DatabaseConfig, init_db, get_session
        config = DatabaseConfig(database_url="sqlite:///:memory:")
        init_db(config)
        return get_session()

    def test_set_retention_deadline(self):
        from raregeneai.compliance.retention import RetentionService
        from raregeneai.database.models import AnalysisJob

        session = self._get_session()
        service = RetentionService()

        job = AnalysisJob(
            job_id="ret12345",
            patient_id_encrypted="enc",
        )
        service.set_retention_deadline(session, job)
        session.add(job)
        session.commit()

        loaded = session.query(AnalysisJob).first()
        assert loaded.retention_expires_at is not None
        assert loaded.retention_expires_at > datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        session.close()

    def test_purge_expired_jobs_anonymize(self):
        from raregeneai.compliance.retention import RetentionPolicy, RetentionService
        from raregeneai.database.models import AnalysisJob

        session = self._get_session()
        policy = RetentionPolicy(purge_method="anonymize")
        service = RetentionService(policy=policy)

        # Create an expired job
        job = AnalysisJob(
            job_id="exp12345",
            patient_id_encrypted="sensitive_data",
            results_json='[{"rank": 1}]',
            retention_expires_at=datetime.datetime(2020, 1, 1),
            is_purged=False,
        )
        session.add(job)
        session.commit()

        purged = service.purge_expired_jobs(session)
        assert purged == 1

        loaded = session.query(AnalysisJob).first()
        assert loaded.is_purged is True
        assert loaded.patient_id_encrypted == "PURGED"
        assert loaded.results_json is None
        session.close()

    def test_purge_expired_jobs_delete(self):
        from raregeneai.compliance.retention import RetentionPolicy, RetentionService
        from raregeneai.database.models import AnalysisJob

        session = self._get_session()
        policy = RetentionPolicy(purge_method="delete")
        service = RetentionService(policy=policy)

        job = AnalysisJob(
            job_id="del12345",
            patient_id_encrypted="data",
            retention_expires_at=datetime.datetime(2020, 1, 1),
        )
        session.add(job)
        session.commit()

        purged = service.purge_expired_jobs(session)
        assert purged == 1
        assert session.query(AnalysisJob).count() == 0
        session.close()

    def test_non_expired_jobs_not_purged(self):
        from raregeneai.compliance.retention import RetentionService
        from raregeneai.database.models import AnalysisJob

        session = self._get_session()
        service = RetentionService()

        job = AnalysisJob(
            job_id="future123",
            patient_id_encrypted="data",
            retention_expires_at=datetime.datetime(2099, 1, 1),
        )
        session.add(job)
        session.commit()

        purged = service.purge_expired_jobs(session)
        assert purged == 0
        assert session.query(AnalysisJob).count() == 1
        session.close()

    def test_hipaa_minimum_audit_retention(self):
        from raregeneai.compliance.retention import RetentionPolicy, RetentionService
        from raregeneai.database.models import AuditLog

        session = self._get_session()
        # Try to set audit retention below HIPAA minimum
        policy = RetentionPolicy(audit_log_retention_days=365)
        service = RetentionService(policy=policy)

        # Add an audit log from 2 years ago (below HIPAA min of 6 years)
        log = AuditLog(
            username="test",
            action="test",
            timestamp=datetime.datetime.now() - datetime.timedelta(days=730),
        )
        session.add(log)
        session.commit()

        # Should NOT be purged because HIPAA enforces 6 year minimum
        purged = service.purge_expired_audit_logs(session)
        assert purged == 0
        session.close()

    def test_retention_report(self):
        from raregeneai.compliance.retention import RetentionService

        session = self._get_session()
        service = RetentionService()

        report = service.get_retention_report(session)
        assert "policy" in report
        assert "analysis_jobs" in report
        assert "audit_logs" in report
        assert report["analysis_jobs"]["total"] == 0
        session.close()
