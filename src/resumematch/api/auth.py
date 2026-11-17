"""Sign-in and roles for the API.

Demo set-up: four fixed accounts, one per role, sharing one password
read from DEMO_PASSWORD (.env). With DEMO_PASSWORD_PUBLIC=1 (the default in
.env.example) the password is meant to be public: the README lists it and
the login page fills it in when a role is picked, so anyone can try the
system. Set it to 0 with a private password to close the demo. The password is never stored, only its
scrypt hash, computed when the module loads. Sessions are a signed token
(payload + HMAC-SHA256 with SESSION_SECRET) in an HttpOnly cookie, so
page scripts can't read it and the Next.js proxy passes it through with
no CORS set-up. Standard library only.

Roles:
    seeker     checks their own resume, explores roles
    recruiter  ranks candidates for a job description (blind review), explores roles
    advisor    checks students' resumes, explores roles
    admin      everything, plus the admin overview

Every endpoint states the roles it allows (require_role); the site only
hides links, the API is what enforces access.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass

from fastapi import Cookie, HTTPException

log = logging.getLogger("resumematch.auth")

COOKIE_NAME = "rm_session"
SESSION_SECONDS = 8 * 3600
MAX_FAILED = 5                 # failed sign-ins per email ...
FAILED_WINDOW_SECONDS = 300    # ... within this window, then refused until it passes

ROLES = ("seeker", "recruiter", "advisor", "admin")
ROLE_LABELS = {"seeker": "Job seeker", "recruiter": "Recruiter", "advisor": "Career advisor", "admin": "Admin"}
ROLE_DESCRIPTIONS = {
    "seeker": "Check a resume against postings, explore roles",
    "recruiter": "Rank candidates for a job, with blind review; explore roles",
    "advisor": "Check students' resumes, explore roles",
    "admin": "Everything, plus accounts and recent sign-ins",
}


@dataclass(frozen=True)
class User:
    email: str
    name: str
    role: str

    @property
    def role_label(self) -> str:
        return ROLE_LABELS[self.role]

    def public(self) -> dict:
        return {**asdict(self), "role_label": self.role_label}


# Fictional people on the reserved example.com domain.
DEMO_USERS = {
    u.email: u for u in (
        User("seeker@example.com", "Sam Seeker", "seeker"),
        User("recruiter@example.com", "Riley Recruiter", "recruiter"),
        User("advisor@example.com", "Ava Advisor", "advisor"),
        User("admin@example.com", "Adrian Admin", "admin"),
    )
}


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message, "soft": False})


# --- Passwords ----------------------------------------------------------------

def hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)


class PasswordBook:
    """Hashes of the demo password, one salt per account."""

    def __init__(self, password: str | None):
        self.configured = bool(password)
        self._hashes = {}
        if password:
            for email in DEMO_USERS:
                salt = secrets.token_bytes(16)
                self._hashes[email] = (salt, hash_password(password, salt))

    def verify(self, email: str, password: str) -> bool:
        entry = self._hashes.get(email)
        if entry is None:
            # Same work for unknown emails, so timing doesn't reveal which accounts exist.
            hash_password(password, b"0" * 16)
            return False
        salt, expected = entry
        return hmac.compare_digest(hash_password(password, salt), expected)


# --- Session tokens -------------------------------------------------------------

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign_session(user: User, secret: bytes, now: float | None = None) -> str:
    payload = {"sub": user.email, "role": user.role, "exp": int((now or time.time()) + SESSION_SECONDS)}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    return f"{body}.{_b64(hmac.new(secret, body.encode(), hashlib.sha256).digest())}"


def read_session(token: str | None, secret: bytes, now: float | None = None) -> User | None:
    """The signed-in user, or None for a missing, tampered or expired token."""
    if not token or token.count(".") != 1:
        return None
    body, signature = token.split(".")
    expected = _b64(hmac.new(secret, body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        payload = json.loads(_unb64(body))
    except (ValueError, json.JSONDecodeError):
        return None
    if payload.get("exp", 0) < (now or time.time()):
        return None
    user = DEMO_USERS.get(payload.get("sub"))
    return user if user and user.role == payload.get("role") else None


# --- Failed sign-in limit --------------------------------------------------------

class FailedLogins:
    def __init__(self):
        self._lock = threading.Lock()
        self._times: dict[str, deque] = {}

    def blocked(self, key: str, now: float | None = None) -> bool:
        now = now or time.time()
        with self._lock:
            times = self._times.get(key, deque())
            while times and times[0] < now - FAILED_WINDOW_SECONDS:
                times.popleft()
            return len(times) >= MAX_FAILED

    def record(self, key: str, now: float | None = None) -> None:
        with self._lock:
            self._times.setdefault(key, deque()).append(now or time.time())

    def clear(self, key: str) -> None:
        with self._lock:
            self._times.pop(key, None)


# --- Process-wide state ------------------------------------------------------------

def _secret() -> bytes:
    value = os.environ.get("SESSION_SECRET")
    if value:
        return value.encode()
    log.warning("SESSION_SECRET is not set; using a random one, so sign-ins end when the API restarts.")
    return secrets.token_bytes(32)


class AuthState:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Re-read DEMO_PASSWORD and SESSION_SECRET (tests change them)."""
        password = os.environ.get("DEMO_PASSWORD")
        self.passwords = PasswordBook(password)
        # Shown on the login page only when the operator says it's public.
        self.public_password = password if password and os.environ.get("DEMO_PASSWORD_PUBLIC") == "1" else None
        self.secret = _secret()
        self.failed = FailedLogins()
        self.recent: deque = deque(maxlen=20)   # recent sign-ins for the admin page: email, role, time


state = AuthState()


def login(email: str, password: str) -> tuple[User, str]:
    """(user, session token), or a plain error. Wrong email and wrong
    password get the same message."""
    email = email.strip().lower()
    if not state.passwords.configured:
        raise _error(503, "login_not_configured", "Sign-in isn't set up: add DEMO_PASSWORD to .env and restart the API.")
    if state.failed.blocked(email):
        raise _error(429, "too_many_attempts", "Too many failed attempts. Wait a few minutes and try again.")
    if not state.passwords.verify(email, password):
        state.failed.record(email)
        raise _error(401, "bad_credentials", "That email and password don't match a demo account.")
    state.failed.clear(email)
    user = DEMO_USERS[email]
    state.recent.appendleft({"email": user.email, "role": user.role, "at": int(time.time())})
    return user, sign_session(user, state.secret)


def current_user(rm_session: str | None = Cookie(default=None)) -> User:
    user = read_session(rm_session, state.secret)
    if user is None:
        raise _error(401, "not_signed_in", "Sign in to continue.")
    return user


def require_role(*roles: str):
    """FastAPI dependency: the signed-in user, if their role is allowed."""
    unknown = set(roles) - set(ROLES)
    if unknown:
        raise ValueError(f"unknown roles: {unknown}")

    def dependency(rm_session: str | None = Cookie(default=None)) -> User:
        user = current_user(rm_session)
        if user.role not in roles:
            raise _error(403, "wrong_role", f"This isn't available to the {user.role_label} role.")
        return user
    return dependency
