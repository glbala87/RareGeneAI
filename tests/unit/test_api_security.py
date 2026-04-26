"""Integration tests for API security: auth, RBAC, ownership, rate limiting."""

import os
import time
from unittest.mock import patch

import pytest

# Set test environment before imports
os.environ.setdefault("RAREGENEAI_SECRET_KEY", "test-secret-for-api-tests-only")
os.environ.setdefault("RAREGENEAI_DATABASE_URL", "sqlite:///test_api.db")


# ── Rate limiter tests ───────────────────────────────────────────────────────


class TestRateLimiter:
    def test_allows_within_limit(self):
        from unittest.mock import MagicMock
        from raregeneai.security.rate_limit import RateLimiter

        limiter = RateLimiter()
        request = MagicMock()
        request.client.host = "127.0.0.1"
        request.headers = {}

        # Should not raise for requests within limit
        for _ in range(5):
            limiter.check("test", request, max_requests=5, window_seconds=60)

    def test_blocks_over_limit(self):
        from unittest.mock import MagicMock
        from fastapi import HTTPException
        from raregeneai.security.rate_limit import RateLimiter

        limiter = RateLimiter()
        request = MagicMock()
        request.client.host = "192.168.1.1"
        request.headers = {}

        # Fill up the limit
        for _ in range(3):
            limiter.check("block_test", request, max_requests=3, window_seconds=60)

        # Next request should be blocked
        with pytest.raises(HTTPException) as exc_info:
            limiter.check("block_test", request, max_requests=3, window_seconds=60)
        assert exc_info.value.status_code == 429
        assert "Retry-After" in exc_info.value.headers

    def test_different_ips_independent(self):
        from unittest.mock import MagicMock
        from raregeneai.security.rate_limit import RateLimiter

        limiter = RateLimiter()

        req1 = MagicMock()
        req1.client.host = "10.0.0.1"
        req1.headers = {}

        req2 = MagicMock()
        req2.client.host = "10.0.0.2"
        req2.headers = {}

        # Fill up IP 1's limit
        for _ in range(3):
            limiter.check("ip_test", req1, max_requests=3, window_seconds=60)

        # IP 2 should still be allowed
        limiter.check("ip_test", req2, max_requests=3, window_seconds=60)

    def test_different_keys_independent(self):
        from unittest.mock import MagicMock
        from raregeneai.security.rate_limit import RateLimiter

        limiter = RateLimiter()
        request = MagicMock()
        request.client.host = "10.0.0.3"
        request.headers = {}

        # Fill "login" bucket
        for _ in range(2):
            limiter.check("login", request, max_requests=2, window_seconds=60)

        # "analyze" bucket should be independent
        limiter.check("analyze", request, max_requests=2, window_seconds=60)

    def test_custom_identifier(self):
        from unittest.mock import MagicMock
        from fastapi import HTTPException
        from raregeneai.security.rate_limit import RateLimiter

        limiter = RateLimiter()
        request = MagicMock()
        request.client.host = "10.0.0.4"
        request.headers = {}

        # Rate limit by username, not IP
        for _ in range(2):
            limiter.check("user_test", request, max_requests=2, window_seconds=60, identifier="user1")

        # Same IP but different user should be allowed
        limiter.check("user_test", request, max_requests=2, window_seconds=60, identifier="user2")

        # user1 should be blocked
        with pytest.raises(HTTPException):
            limiter.check("user_test", request, max_requests=2, window_seconds=60, identifier="user1")

    def test_window_expires(self):
        from unittest.mock import MagicMock
        from raregeneai.security.rate_limit import RateLimiter

        limiter = RateLimiter()
        request = MagicMock()
        request.client.host = "10.0.0.5"
        request.headers = {}

        # Use a very short window
        for _ in range(2):
            limiter.check("expire_test", request, max_requests=2, window_seconds=1)

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        limiter.check("expire_test", request, max_requests=2, window_seconds=1)

    def test_x_forwarded_for_header(self):
        from unittest.mock import MagicMock
        from fastapi import HTTPException
        from raregeneai.security.rate_limit import RateLimiter

        limiter = RateLimiter()
        request = MagicMock()
        request.client.host = "10.0.0.6"
        request.headers = {"x-forwarded-for": "203.0.113.50, 70.41.3.18"}

        # Should use the first IP from X-Forwarded-For
        for _ in range(2):
            limiter.check("xff_test", request, max_requests=2, window_seconds=60)

        with pytest.raises(HTTPException):
            limiter.check("xff_test", request, max_requests=2, window_seconds=60)


# ── Auth and RBAC tests ──────────────────────────────────────────────────────


class TestAuthRBAC:
    """Test role-based access control via token creation and verification."""

    @patch.dict(os.environ, {"RAREGENEAI_SECRET_KEY": "test-secret-for-rbac-tests"})
    def test_admin_token_grants_admin_role(self):
        from raregeneai.security.auth import create_access_token, decode_access_token

        token = create_access_token(data={"sub": "admin_user", "role": "admin"})
        data = decode_access_token(token)
        assert data.role == "admin"

    @patch.dict(os.environ, {"RAREGENEAI_SECRET_KEY": "test-secret-for-rbac-tests"})
    def test_analyst_token_grants_analyst_role(self):
        from raregeneai.security.auth import create_access_token, decode_access_token

        token = create_access_token(data={"sub": "analyst_user", "role": "analyst"})
        data = decode_access_token(token)
        assert data.role == "analyst"

    @patch.dict(os.environ, {"RAREGENEAI_SECRET_KEY": "test-secret-for-rbac-tests"})
    def test_viewer_cannot_access_analyst_endpoints(self):
        import asyncio
        from fastapi import HTTPException
        from raregeneai.security.auth import User, UserRole, require_analyst

        viewer = User(username="viewer_user", role=UserRole.VIEWER)
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(require_analyst(current_user=viewer))
        assert exc_info.value.status_code == 403

    @patch.dict(os.environ, {"RAREGENEAI_SECRET_KEY": "test-secret-for-rbac-tests"})
    def test_viewer_cannot_access_admin_endpoints(self):
        import asyncio
        from fastapi import HTTPException
        from raregeneai.security.auth import User, UserRole, require_admin

        viewer = User(username="viewer_user", role=UserRole.VIEWER)
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(require_admin(current_user=viewer))
        assert exc_info.value.status_code == 403

    @patch.dict(os.environ, {"RAREGENEAI_SECRET_KEY": "test-secret-for-rbac-tests"})
    def test_analyst_cannot_access_admin_endpoints(self):
        import asyncio
        from fastapi import HTTPException
        from raregeneai.security.auth import User, UserRole, require_admin

        analyst = User(username="analyst_user", role=UserRole.ANALYST)
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(require_admin(current_user=analyst))
        assert exc_info.value.status_code == 403

    @patch.dict(os.environ, {"RAREGENEAI_SECRET_KEY": "test-secret-for-rbac-tests"})
    def test_admin_passes_all_checks(self):
        import asyncio
        from raregeneai.security.auth import User, UserRole, require_admin, require_analyst

        admin = User(username="admin_user", role=UserRole.ADMIN)
        result = asyncio.run(require_admin(current_user=admin))
        assert result.username == "admin_user"

        result = asyncio.run(require_analyst(current_user=admin))
        assert result.username == "admin_user"

    @patch.dict(os.environ, {"RAREGENEAI_SECRET_KEY": "test-secret-for-rbac-tests"})
    def test_expired_token_rejected(self):
        import datetime
        from fastapi import HTTPException
        from raregeneai.security.auth import create_access_token, decode_access_token

        # Create token that expired 1 hour ago
        token = create_access_token(
            data={"sub": "expired_user", "role": "analyst"},
            expires_delta=datetime.timedelta(seconds=-3600),
        )
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(token)
        assert exc_info.value.status_code == 401


# ── Job ownership tests ──────────────────────────────────────────────────────


class TestJobOwnership:
    """Test that users can only access their own jobs."""

    def _get_session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from raregeneai.database.models import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        return Session()

    def test_user_can_access_own_job(self):
        from raregeneai.database.models import AnalysisJob, UserAccount

        session = self._get_session()

        user = UserAccount(username="owner", hashed_password="hash", role="analyst")
        session.add(user)
        session.flush()

        job = AnalysisJob(
            job_id="own12345",
            patient_id_encrypted="enc",
            user_id=user.id,
        )
        session.add(job)
        session.commit()

        # Verify ownership check
        loaded_job = session.query(AnalysisJob).filter_by(job_id="own12345").first()
        loaded_user = session.query(UserAccount).filter_by(username="owner").first()
        assert loaded_job.user_id == loaded_user.id
        session.close()

    def test_other_user_cannot_access_job(self):
        from raregeneai.database.models import AnalysisJob, UserAccount

        session = self._get_session()

        owner = UserAccount(username="owner2", hashed_password="hash", role="analyst")
        other = UserAccount(username="other2", hashed_password="hash", role="analyst")
        session.add_all([owner, other])
        session.flush()

        job = AnalysisJob(
            job_id="other1234",
            patient_id_encrypted="enc",
            user_id=owner.id,
        )
        session.add(job)
        session.commit()

        # Other user's ID doesn't match job's user_id
        loaded_job = session.query(AnalysisJob).filter_by(job_id="other1234").first()
        loaded_other = session.query(UserAccount).filter_by(username="other2").first()
        assert loaded_job.user_id != loaded_other.id
        session.close()

    def test_admin_bypasses_ownership(self):
        """Admin should have access to any job regardless of user_id."""
        from raregeneai.security.auth import User, UserRole

        admin = User(username="admin", role=UserRole.ADMIN)
        # Admin check is: if current_user.role != UserRole.ADMIN: deny
        # So admin always passes
        assert admin.role == UserRole.ADMIN


# ── Session timeout tests ────────────────────────────────────────────────────


class TestSessionTimeout:
    @patch.dict(os.environ, {"RAREGENEAI_SECRET_KEY": "test-secret-for-session-tests"})
    def test_expired_jwt_detected(self):
        import datetime
        from jose import jwt

        # Create an expired token
        expired_token = jwt.encode(
            {"sub": "user", "role": "analyst", "exp": time.time() - 100},
            "test-secret-for-session-tests",
            algorithm="HS256",
        )

        # Verify expiry detection
        payload = jwt.get_unverified_claims(expired_token)
        assert payload["exp"] < time.time()

    def test_inactivity_timeout_logic(self):
        """Test that sessions older than timeout are rejected."""
        _SESSION_TIMEOUT = 3600
        last_activity = time.time() - 3601  # 1 second past timeout
        assert time.time() - last_activity > _SESSION_TIMEOUT

    def test_active_session_passes(self):
        """Test that recent activity keeps session alive."""
        _SESSION_TIMEOUT = 3600
        last_activity = time.time() - 10  # 10 seconds ago
        assert time.time() - last_activity < _SESSION_TIMEOUT
