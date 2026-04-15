"""JWT authentication for the Legal QA API.

Configuration (set in .env):
    JWT_SECRET_KEY      — required; sign/verify tokens (use a long random string)
    AUTH_USERNAME       — optional; defaults to "admin"
    AUTH_PASSWORD       — required; plain-text password for the single admin user
    JWT_EXPIRE_MINUTES  — optional; token lifetime in minutes (default 60)

Usage:
    - POST /auth/token  with form fields username + password → returns Bearer token
    - Add `_: Annotated[str, Depends(require_auth)]` to any endpoint to protect it
"""

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# ── Settings ──────────────────────────────────────────────────────────────────

_SECRET_KEY:     str = os.getenv("JWT_SECRET_KEY", "")
_ALGORITHM:      str = "HS256"
_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
_AUTH_USERNAME:  str = os.getenv("AUTH_USERNAME", "admin")
_AUTH_PASSWORD:  str = os.getenv("AUTH_PASSWORD", "")

# tokenUrl tells Swagger UI where to POST credentials to get a token.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _assert_configured() -> None:
    """Raise a 500 at call time (not at import time) if env vars are missing."""
    if not _SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT_SECRET_KEY is not set. Add it to your .env file.",
        )
    if not _AUTH_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AUTH_PASSWORD is not set. Add it to your .env file.",
        )


# ── Public API ────────────────────────────────────────────────────────────────

def verify_credentials(username: str, password: str) -> bool:
    """Return True if username and password match the configured credentials.

    Uses secrets.compare_digest to prevent timing-based enumeration attacks.
    """
    _assert_configured()
    username_ok = secrets.compare_digest(username.encode(), _AUTH_USERNAME.encode())
    password_ok = secrets.compare_digest(password.encode(), _AUTH_PASSWORD.encode())
    return username_ok and password_ok


def create_access_token(username: str) -> str:
    """Create a signed JWT that expires after JWT_EXPIRE_MINUTES minutes."""
    _assert_configured()
    expire = datetime.now(timezone.utc) + timedelta(minutes=_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": username, "exp": expire},
        _SECRET_KEY,
        algorithm=_ALGORITHM,
    )


async def require_auth(token: Annotated[str, Depends(oauth2_scheme)]) -> str:
    """FastAPI dependency — validates the Bearer token on protected endpoints.

    Returns the username embedded in the token on success.
    Raises HTTP 401 on missing, invalid, or expired tokens.
    """
    _assert_configured()
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
        username: str | None = payload.get("sub")
        if not username:
            raise ValueError("sub claim missing")
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username
