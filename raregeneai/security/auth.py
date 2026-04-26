"""JWT/OAuth2 authentication for RareGeneAI API.

Implements:
  - Password hashing with bcrypt
  - JWT token creation and verification
  - OAuth2 password bearer flow
  - Role-based access control (RBAC)
  - Database-backed user verification
"""

from __future__ import annotations

import os
import datetime
from enum import Enum

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import bcrypt as _bcrypt
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# ── Configuration ────────────────────────────────────────────────────────────

SECRET_KEY = os.environ.get("RAREGENEAI_SECRET_KEY", "")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("RAREGENEAI_TOKEN_EXPIRE_MINUTES", "60"))


class AuthConfig(BaseModel):
    """Authentication configuration."""
    secret_key: str = Field(default="", description="JWT signing key (set via RAREGENEAI_SECRET_KEY env var)")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    allowed_origins: list[str] = Field(
        default_factory=lambda: os.environ.get(
            "RAREGENEAI_ALLOWED_ORIGINS", "http://localhost:8501"
        ).split(",")
    )


def _get_secret_key() -> str:
    """Get the JWT secret key, raising an error if not configured."""
    key = os.environ.get("RAREGENEAI_SECRET_KEY", SECRET_KEY)
    if not key:
        raise RuntimeError(
            "RAREGENEAI_SECRET_KEY environment variable must be set. "
            "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
        )
    return key


# ── Password hashing ────────────────────────────────────────────────────────


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its bcrypt hash."""
    return _bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return _bcrypt.hashpw(
        password.encode("utf-8"),
        _bcrypt.gensalt(),
    ).decode("utf-8")


# ── Roles ────────────────────────────────────────────────────────────────────

class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class User(BaseModel):
    """Authenticated user."""
    user_id: str = ""
    username: str
    role: UserRole = UserRole.VIEWER
    institution: str = ""
    disabled: bool = False


class TokenData(BaseModel):
    """JWT token payload."""
    sub: str  # username
    role: str = "viewer"
    exp: datetime.datetime | None = None


# ── Token management ────────────────────────────────────────────────────────

def create_access_token(
    data: dict,
    expires_delta: datetime.timedelta | None = None,
) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.datetime.now(datetime.timezone.utc) + (
        expires_delta or datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, _get_secret_key(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> TokenData:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, _get_secret_key(), algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
            )
        return TokenData(
            sub=username,
            role=payload.get("role", "viewer"),
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI dependencies ─────────────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def _get_db_session():
    """Lazy import to avoid circular dependency."""
    from raregeneai.database.connection import get_db
    return Depends(get_db)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(lambda: __import__("raregeneai.database.connection", fromlist=["get_session"]).get_session()),
) -> User:
    """FastAPI dependency: extract, validate, and verify the current user.

    Performs a database lookup to ensure the user still exists, is active,
    and has the role claimed in the token.
    """
    from raregeneai.database.models import UserAccount

    token_data = decode_access_token(token)

    # Verify user exists and is active in the database
    db_user = db.query(UserAccount).filter(
        UserAccount.username == token_data.sub,
    ).first()

    if db_user is None:
        db.close()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not db_user.is_active:
        db.close()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is deactivated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Use the database role (authoritative), not the token role
    user = User(
        user_id=db_user.id,
        username=db_user.username,
        role=UserRole(db_user.role),
        institution=db_user.institution or "",
    )

    db.close()
    return user


async def require_analyst(current_user: User = Depends(get_current_user)) -> User:
    """Require at least analyst role."""
    if current_user.role == UserRole.VIEWER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Analyst or admin role required",
        )
    return current_user


async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin role."""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return current_user
