"""Startup checks: each missing piece gives one line saying what to run."""

import requests

from resumematch import health


class FakeResponse:
    def __init__(self, models):
        self._models = models

    def json(self):
        return {"models": [{"name": m} for m in self._models]}


def codes(problems):
    return [(p.code, p.blocking) for p in problems]


def test_everything_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no data/processed files here

    def down(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(health.requests, "get", down)
    problems = health.check(str(tmp_path / "no_index"), host="http://localhost:11434", model="llama3.2:3b")
    assert codes(problems) == [("no_index", True), ("model_down", True), ("no_vocab", False), ("no_details", False)]
    assert all("\n" not in p.message for p in problems)  # one line each
    assert "ollama serve" in problems[1].message
    assert not (tmp_path / "no_index").exists()  # the check must not create an index folder


def test_model_not_pulled(tmp_path, monkeypatch):
    monkeypatch.setattr(health.requests, "get", lambda *a, **k: FakeResponse(["llama3.1:8b"]))
    problems = health.check(str(tmp_path), model="llama3.2:3b")
    assert ("no_model", True) in codes(problems)
    assert "ollama pull llama3.2:3b" in next(p.message for p in problems if p.code == "no_model")


def test_model_present_including_latest_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(health.requests, "get", lambda *a, **k: FakeResponse(["mistral:latest"]))
    assert "no_model" not in [p.code for p in health.check(str(tmp_path), model="mistral")]


def test_api_health_reports_ok_only_without_blocking_problems(monkeypatch):
    from fastapi.testclient import TestClient

    from resumematch.api import app as api
    monkeypatch.setattr(api, "check", lambda *_a, **_k: [health.Problem("no_vocab", "Typo correction is off.", False)])
    body = TestClient(api.app).get("/api/health").json()
    assert body["ok"] is True and body["problems"][0]["code"] == "no_vocab"
    monkeypatch.setattr(api, "check", lambda *_a, **_k: [health.Problem("model_down", "Ollama isn't running.", True)])
    assert TestClient(api.app).get("/api/health").json()["ok"] is False
