"""Authentication, role-based authorisation and rate limiting.

* Passwords: PBKDF2-HMAC-SHA256, 240k iterations, per-user salt (stdlib only).
* Sessions: opaque random bearer tokens; only their SHA-256 is stored, so a
  database leak does not leak usable tokens. Tokens expire (default 12 h).
* Roles are ordered: viewer < engineer < approver < admin. Production-changing
  actions additionally require *separation of duties*: whoever requested an
  approval cannot approve it.
* Rate limiting: in-process sliding window. Logins are limited per client IP *and* per
  username; authenticated requests per user (after the token is validated); invalid tokens
  per client IP. Adequate for a single API process; multiple replicas need Redis.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from evalforge.config import get_settings
from evalforge.db import get_db, session_scope
from evalforge.models import ApiToken, AuditEvent, User

log = logging.getLogger(__name__)

ROLES = ("viewer", "engineer", "approver", "admin")
_RANK = {r: i for i, r in enumerate(ROLES)}
ITERATIONS = 240_000


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    b64 = base64.b64encode
    return f"pbkdf2_sha256${ITERATIONS}${b64(salt).decode()}${b64(dk).decode()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iters, salt_b64, dk_b64 = encoded.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), base64.b64decode(salt_b64), int(iters)
        )
        return hmac.compare_digest(dk, base64.b64decode(dk_b64))
    except Exception:
        return False


_DUMMY_HASH: str | None = None


def burn_password_check(password: str) -> None:
    """Spend the same PBKDF2 work as a real check so unknown usernames are not
    distinguishable by response time (audit SEC-02)."""
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password(secrets.token_urlsafe(16))
    verify_password(password, _DUMMY_HASH)


def client_ip(request: Request) -> str:
    if get_settings().trust_forwarded_for:
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd.strip():
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def throttle(key: str, limit: int) -> None:
    ok, retry = limiter.allow(key, limit)
    if not ok:
        raise HTTPException(
            429, "Rate limit exceeded", headers={"Retry-After": str(int(retry) + 1)}
        )


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def issue_token(db: Session, user: User, name: str = "session") -> str:
    token = "efk_" + secrets.token_urlsafe(32)
    hours = get_settings().token_ttl_hours
    db.add(
        ApiToken(
            user_id=user.id,
            token_hash=token_digest(token),
            name=name,
            expires_at=datetime.now(UTC) + timedelta(hours=hours),
        )
    )
    return token


def ensure_bootstrap_users() -> list[str]:
    """Create initial accounts on an empty database.

    Accounts are created from ``EVALFORGE_BOOTSTRAP_PASSWORD``. In dev/test, if it is
    unset, the documented default ``evalforge-dev`` is used and a warning logged; in
    prod the API refuses to start without it.
    """
    s = get_settings()
    with session_scope() as db:
        if db.scalar(select(User).limit(1)) is not None:
            return []
        pw = s.bootstrap_password
        if not pw:
            if s.env == "prod":
                raise RuntimeError("EVALFORGE_BOOTSTRAP_PASSWORD must be set in prod")
            pw = "evalforge-dev"
            log.warning(
                "Created dev accounts admin/engineer/approver/viewer with the default "
                "password 'evalforge-dev'. Set EVALFORGE_BOOTSTRAP_PASSWORD to change."
            )
        created = []
        for name, role in (
            ("admin", "admin"),
            ("engineer", "engineer"),
            ("approver", "approver"),
            ("viewer", "viewer"),
        ):
            db.add(User(username=name, password_hash=hash_password(pw), role=role))
            created.append(name)
        db.add(
            AuditEvent(
                action="users.bootstrapped",
                entity_type="user",
                entity_id="*",
                data={"users": created},
            )
        )
        return created


class Principal:
    def __init__(self, user_id: str, username: str, role: str) -> None:
        self.user_id, self.username, self.role = user_id, username, role

    def has(self, role: str) -> bool:
        return _RANK[self.role] >= _RANK[role]


def current_principal(request: Request, db: Session = Depends(get_db)) -> Principal:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    tok = db.scalar(select(ApiToken).where(ApiToken.token_hash == token_digest(header[7:].strip())))
    now = datetime.now(UTC)
    if tok is None or tok.revoked or _aware(tok.expires_at) < now or not tok.user.active:
        # Invalid credentials are throttled per client IP, so rotating forged tokens does not
        # open new rate-limit buckets (audit SEC-01).
        throttle(f"authfail:{client_ip(request)}", get_settings().rate_limit_login_per_min * 6)
        raise HTTPException(401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    bucket = rate_limit_bucket(request.method, request.url.path)
    if bucket is not None:
        throttle(f"{bucket[0]}:{tok.user.id}", bucket[1])
    if tok.last_used_at is None or now - _aware(tok.last_used_at) > timedelta(minutes=1):
        # Best-effort bookkeeping: a busy database must not turn a read into a 500.
        tok.last_used_at = now
        try:
            db.commit()
        except OperationalError:
            db.rollback()
            log.warning("could not record token use (database busy); continuing")
            tok = db.scalar(select(ApiToken).where(ApiToken.id == tok.id))
            assert tok is not None
    p = Principal(tok.user.id, tok.user.username, tok.user.role)
    request.state.principal = p
    return p


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def require(role: str):  # type: ignore[no-untyped-def]
    def dep(p: Principal = Depends(current_principal)) -> Principal:
        if not p.has(role):
            raise HTTPException(403, f"Requires role '{role}' (you are '{p.role}')")
        return p

    return dep


# --------------------------------------------------------------------------- rate limiting


class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_s: float = 60.0) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= limit:
                return False, window_s - (now - q[0])
            q.append(now)
            return True, 0.0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = RateLimiter()


def rate_limit_bucket(method: str, path: str) -> tuple[str, int] | None:
    s = get_settings()
    if path == "/api/auth/login":
        return "login", s.rate_limit_login_per_min
    if path.startswith("/api/serve/"):
        return "serve", s.rate_limit_serve_per_min
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        return "write", s.rate_limit_write_per_min
    return None
