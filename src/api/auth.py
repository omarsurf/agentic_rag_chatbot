"""Authentication module for GRI RAG API.

Provides Bearer token authentication with automatic toggle:
- Disabled when api_host == "127.0.0.1" (localhost dev)
- Enabled when api_host != "127.0.0.1" OR api_auth_enabled == True

Usage:
    from src.api.auth import verify_token

    @app.post("/protected")
    async def protected_endpoint(token: str = Depends(verify_token)):
        ...
"""

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.core.config import settings

security = HTTPBearer(auto_error=False)


def is_auth_required() -> bool:
    """Determine if authentication is required based on config.

    Returns True if:
    - api_auth_enabled is explicitly True AND api_bearer_token is set, OR
    - api_host is not localhost AND api_bearer_token is set

    If api_bearer_token is not configured, auth is disabled (with a warning logged).
    """
    # If no token is configured, auth cannot be enforced
    if not settings.api_bearer_token:
        return False

    # Explicit opt-in
    if settings.api_auth_enabled:
        return True

    # Auto-enable for non-localhost
    return settings.api_host not in ("127.0.0.1", "localhost")


def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str | None:
    """Verify Bearer token from Authorization header.

    Args:
        credentials: Extracted from Authorization: Bearer <token> header

    Returns:
        The validated token string, or None if auth is disabled

    Raises:
        HTTPException: 401 if token is missing or invalid
    """
    if not is_auth_required():
        return None

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(
        credentials.credentials.encode("utf-8"),
        settings.api_bearer_token.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials
