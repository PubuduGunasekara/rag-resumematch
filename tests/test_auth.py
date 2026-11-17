"""Sign-in, sessions and role checks (api/auth.py), through the real API."""

import pytest
from fastapi.testclient import TestClient

from resumematch.api import app as api
from resumematch.api import auth

PASSWORD = "test-only-demo-password"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DEMO_PASSWORD", PASSWORD)
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    auth.state.reset()
    monkeypatch.setattr(api, "run_pipeline", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("not reached")))
    api.app.dependency_overrides.clear()
    yield TestClient(api.app)
    auth.state.reset()


def sign_in(client, email):
    return client.post("/api/auth/login", json={"email": email, "password": PASSWORD})


def test_sign_in_sets_an_httponly_cookie_and_me_returns_the_user(client):
    r = sign_in(client, "Seeker@Example.com ")
    assert r.status_code == 200 and r.json()["user"]["role"] == "seeker"
    cookie = r.headers["set-cookie"]
    assert "rm_session=" in cookie and "HttpOnly" in cookie and "samesite=lax" in cookie.lower()
    assert client.get("/api/auth/me").json()["user"]["role_label"] == "Job seeker"


def test_wrong_password_and_unknown_email_get_the_same_answer(client):
    a = client.post("/api/auth/login", json={"email": "seeker@example.com", "password": "nope"})
    b = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "nope"})
    assert a.status_code == b.status_code == 401 and a.json() == b.json()


def test_endpoints_need_sign_in(client):
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/roles", params={"q": "data"}).status_code == 401
    assert client.post("/api/analyze", files={"file": ("cv.pdf", b"%PDF")}).status_code == 401
    assert client.post("/api/role-report", data={"role": "nurse"}).status_code == 401


def test_roles_are_enforced_by_the_api(client):
    sign_in(client, "recruiter@example.com")
    r = client.post("/api/analyze", files={"file": ("cv.pdf", b"%PDF")})
    assert r.status_code == 403 and "Recruiter" in r.json()["detail"]["message"]
    assert client.get("/api/admin/overview").status_code == 403
    tailor = client.post("/api/tailor", json={"kind": "letter", "job_id": "1", "resume_text": "x"})
    assert tailor.status_code == 403
    sign_in(client, "seeker@example.com")
    assert client.post("/api/rank", data={"job_text": "x"}, files=[("files", ("a.pdf", b"x"))]).status_code == 403
    sign_in(client, "admin@example.com")
    body = client.get("/api/admin/overview").json()
    assert {a["role"] for a in body["accounts"]} == set(auth.ROLES)
    assert body["recent_sign_ins"][0]["email"] == "admin@example.com"


def test_logout_clears_the_session(client):
    sign_in(client, "advisor@example.com")
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401


def test_repeated_failures_are_refused_for_a_while(client):
    for _ in range(auth.MAX_FAILED):
        client.post("/api/auth/login", json={"email": "admin@example.com", "password": "wrong"})
    assert sign_in(client, "admin@example.com").status_code == 429


def test_tampered_and_expired_tokens_are_rejected():
    secret = b"s"
    user = auth.DEMO_USERS["seeker@example.com"]
    token = auth.sign_session(user, secret, now=1000)
    assert auth.read_session(token, secret, now=1001) == user
    assert auth.read_session(token, secret, now=1000 + auth.SESSION_SECONDS + 1) is None
    assert auth.read_session(token, b"other", now=1001) is None
    body, sig = token.split(".")
    admin_body = auth._b64(b'{"sub":"admin@example.com","role":"admin","exp":99999999999}')
    assert auth.read_session(f"{admin_body}.{sig}", secret, now=1001) is None
    assert auth.read_session("garbage", secret) is None


def test_no_password_configured_means_no_sign_in(monkeypatch):
    monkeypatch.delenv("DEMO_PASSWORD", raising=False)
    auth.state.reset()
    try:
        r = TestClient(api.app).post("/api/auth/login", json={"email": "seeker@example.com", "password": ""})
        assert r.status_code == 503 and r.json()["detail"]["code"] == "login_not_configured"
    finally:
        auth.state.reset()


def test_unknown_role_name_is_a_programming_error():
    with pytest.raises(ValueError):
        auth.require_role("superuser")


def test_demo_accounts_show_the_password_only_when_public(monkeypatch):
    monkeypatch.setenv("DEMO_PASSWORD", PASSWORD)
    monkeypatch.setenv("DEMO_PASSWORD_PUBLIC", "1")
    auth.state.reset()
    try:
        body = TestClient(api.app).get("/api/auth/demo-accounts").json()
        assert body["password"] == PASSWORD and len(body["accounts"]) == 4
        assert {a["role_label"] for a in body["accounts"]} == set(auth.ROLE_LABELS.values())
        assert all(a["does"] for a in body["accounts"])
        monkeypatch.setenv("DEMO_PASSWORD_PUBLIC", "0")
        auth.state.reset()
        assert TestClient(api.app).get("/api/auth/demo-accounts").json()["password"] is None
    finally:
        auth.state.reset()
