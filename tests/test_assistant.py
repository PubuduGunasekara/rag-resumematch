"""The assistant's routing (chat/assistant.py) and its endpoint."""

import pytest
from fastapi.testclient import TestClient

from resumematch.api import app as api
from resumematch.api import auth
from resumematch.chat.assistant import help_reply, pick_job, role_title, route

JOBS = [{"job_id": "7", "title": "Staff Accountant", "company": "IFG Chicago", "closest": True},
        {"job_id": "8", "title": "Accounting Supervisor", "company": "Truity Partners", "closest": False},
        {"job_id": "9", "title": "Staff Accountant", "company": "Robert Half", "closest": False}]


@pytest.mark.parametrize("question, title", [
    ("What do data engineer postings ask for?", "data engineer"),
    ("what does a registered nurse job usually require", "registered nurse"),
    ("Skills for a product manager role?", "product manager"),
    ("What do I need to become an accountant?", "accountant"),
    ("What do these postings ask for?", None),
    ("What do other postings usually require?", None),
])
def test_role_titles(question, title):
    assert role_title(question) == title


def test_pick_job_by_ordinal_name_or_closest():
    assert pick_job("draft a cover letter for the second posting", JOBS)["job_id"] == "8"
    assert pick_job("cover letter for Robert Half", JOBS)["job_id"] == "9"
    assert pick_job("cover letter for posting 3", JOBS)["job_id"] == "9"
    assert pick_job("cover letter for my closest match", JOBS)["job_id"] == "7"
    assert pick_job("cover letter please", JOBS)["job_id"] == "7"
    assert pick_job("cover letter for the fifth one", JOBS)["job_id"] == "7"   # only 3 postings: falls back
    assert pick_job("anything", []) is None


def test_before_a_resume_the_assistant_explains_and_asks_for_one():
    r = route("hi", "seeker", False, [])
    assert r.kind == "help" and any(a["type"] == "upload" for a in r.actions)
    assert "Upload your resume" in r.text[0]
    r = route("draft a cover letter", "seeker", False, [])
    assert r.kind == "need" and r.actions[0]["type"] == "upload"
    r = route("Why is my closest match a good fit?", "seeker", False, [])
    assert r.kind == "need"


def test_with_a_report_actions_name_their_posting():
    r = route("Draft a cover letter for the second posting", "seeker", True, JOBS)
    assert r.kind == "action" and r.actions == [{"type": "letter", "job_id": "8", "label": "Accounting Supervisor at Truity Partners"}]
    assert route("suggest resume edits for Robert Half", "seeker", True, JOBS).actions[0] == \
        {"type": "bullets", "job_id": "9", "label": "Staff Accountant at Robert Half"}
    assert route("prepare me for the interview", "advisor", True, JOBS).actions[0]["type"] == "interview"
    assert route("compare my revised resume", "seeker", True, JOBS).actions[0]["href"] == "/compare"


def test_report_questions_go_to_the_model_path():
    assert route("What do these postings say about month-end close?", "seeker", True, JOBS) is None


def test_role_questions_work_without_a_resume():
    r = route("What do data engineer postings ask for?", "seeker", False, [])
    assert r.kind == "role" and r.actions == [{"type": "role", "title": "data engineer"}]


def test_recruiters_get_recruiter_help_and_no_resume_actions():
    r = route("hello", "recruiter", False, [])
    assert [a["href"] for a in r.actions] == ["/recruit", "/role"]
    assert route("draft a cover letter", "recruiter", False, []).kind == "need"
    assert route("rank these candidates", "recruiter", False, []).actions[0]["href"] == "/recruit"


def test_help_never_offers_a_page_that_does_not_exist():
    pages = {"/", "/matches", "/resume", "/tailor", "/compare", "/role", "/recruit", "/admin"}
    for role in ("seeker", "recruiter", "advisor", "admin"):
        for has_report in (False, True):
            for a in help_reply(role, has_report).actions:
                assert a["type"] in ("upload", "open") and (a["type"] == "upload" or a["href"] in pages)


def test_privacy_question_is_answered_in_code():
    r = route("What happens to my resume after the check?", "seeker", False, [])
    assert r.kind == "help" and "deleted" in r.text[0]


# --- Endpoint ---------------------------------------------------------------

@pytest.fixture
def as_role():
    def sign_in(role):
        user = auth.DEMO_USERS[f"{role}@example.com"]
        api.app.dependency_overrides[api.ALL_ROLES] = lambda: user
        api.app.dependency_overrides[api.RESUME_CHECK_ROLES] = lambda: user
        return TestClient(api.app)
    yield sign_in
    api.app.dependency_overrides.clear()


def test_endpoint_help_and_role_summary(as_role, monkeypatch):
    c = as_role("seeker")
    body = c.post("/api/assistant", json={"message": "hi"}).json()
    assert body["kind"] == "help" and body["actions"][0]["type"] == "upload"
    monkeypatch.setattr(api, "_role_summary", lambda title, resume: {
        "title": "Data Engineer", "headline": "What Data Engineer postings ask for", "summary": "The terms ...",
        "analyzed": 331, "terms": [{"term": "SQL", "share": 0.65, "shown": None}]})
    body = c.post("/api/assistant", json={"message": "What do data engineer postings ask for?"}).json()
    assert body["kind"] == "role" and body["role"]["terms"][0]["term"] == "SQL"
    assert body["actions"][0]["href"] == "/role?title=Data Engineer"


def test_endpoint_sends_report_questions_through_chat(as_role, monkeypatch):
    seen = {}

    def fake_chat(chat_body, user):
        seen["q"] = chat_body.question
        return {"answer": [{"text": "x", "sources": [1]}], "message": None, "no_answer": False, "by_code": False,
                "sources": [], "removed": []}
    monkeypatch.setattr(api, "chat", fake_chat)
    body = as_role("seeker").post("/api/assistant", json={
        "message": "What do these postings say about reconciliations?", "resume_text": "James Okafor\nReconcile 40+ accounts.",
        "jobs": [{"job_id": "7", "title": "Staff Accountant", "company": "IFG", "closest": True, "requirements": []}]}).json()
    assert body["kind"] == "qa" and body["answer"]["answer"][0]["text"] == "x" and seen["q"].startswith("What do these")


def test_endpoint_recruiter_never_reaches_the_resume_path(as_role, monkeypatch):
    monkeypatch.setattr(api, "chat", lambda *a: (_ for _ in ()).throw(AssertionError("not for recruiters")))
    body = as_role("recruiter").post("/api/assistant", json={"message": "tell me something about the market"}).json()
    assert body["kind"] == "help" and body["actions"][0]["href"] == "/recruit"


# --- Before sign-in ------------------------------------------------------------

def test_guest_replies_explain_and_ask_to_sign_in():
    from resumematch.chat.assistant import guest_reply
    r = guest_reply("hello", password_public=True)
    assert "checks your resume against real LinkedIn job postings" in r.text[0] and "Sign in to access" in r.text[-1]
    assert '"Fill in"' in guest_reply("How do I sign in?", password_public=True).text[0]
    assert '"Fill in"' not in guest_reply("How do I sign in?", password_public=False).text[0]
    assert "deleted" in guest_reply("What happens to my resume?", True).text[0]
    other = guest_reply("Draft a cover letter for me", True)
    assert other.kind == "need" and other.text[0].startswith("Sign in to access that")


def test_guest_endpoint_is_public_and_never_touches_the_pipeline(monkeypatch):
    api.app.dependency_overrides.clear()
    monkeypatch.setattr(api, "chat", lambda *a: (_ for _ in ()).throw(AssertionError("no model before sign-in")))
    c = TestClient(api.app)
    body = c.post("/api/assistant/guest", json={"message": "What can ResumeMatch do?"}).json()
    assert body["kind"] == "help" and body["suggestions"] and body["actions"] == []
    assert c.post("/api/assistant", json={"message": "hi"}).status_code == 401   # the full assistant still needs sign-in
