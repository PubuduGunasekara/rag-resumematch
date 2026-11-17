"""API contract tests with the pipeline faked (no Ollama, no index)."""

import pytest
from fastapi.testclient import TestClient

from resumematch.api import app as api
from resumematch.api import auth
from resumematch.generation.gap_analysis import CitedClaim, GapAnalysisResult
from resumematch.generation.job_report import JobAnalysis, JobReport, LearnItem, Requirement
from resumematch.pipeline import PipelineResult
from resumematch.preprocess.resume_sections import ResumeSection

RESUME = "Maria Delgado, RN\nACLS, BLS, PALS"


@pytest.fixture(autouse=True)
def signed_in():
    """These tests are about the endpoints, not sign-in (tests/test_auth.py):
    every request runs as the demo job seeker."""
    user = auth.DEMO_USERS["seeker@example.com"]
    api.app.dependency_overrides[api.ALL_ROLES] = lambda: user
    api.app.dependency_overrides[api.RESUME_CHECK_ROLES] = lambda: user
    yield
    api.app.dependency_overrides.clear()


def fake_result() -> PipelineResult:
    reqs = [Requirement("ACLS", True, "required", False, "ACLS, BLS", None),
            Requirement("Siemens NX", False, "preferred", False, None, None)]
    job = JobAnalysis(1, "7", "RN ICU", "Southcoast Health", "https://x/7", {"location": "New Bedford, MA"},
                      reqs, tier="best", reason="1 requirement not found on your resume yet.")
    rejected = CitedClaim("Child care", "7", None, "", None, claimed_as="missing",
                          issue="an employer benefit, not a requirement")
    analysis = GapAnalysisResult([], [], "", "{}", unverified=[rejected])
    report = JobReport([job], [LearnItem("Siemens NX", 1, False)], "Your closest match is RN ICU", "These are the 1 postings closest to your resume.",
                       best=job, analysis=analysis)
    return PipelineResult(gap_analysis=analysis, corrections=[], sections=[ResumeSection("SKILLS", RESUME)],
                          retrieved=[{"title": "RN ICU", "company_name": "Southcoast Health", "matched_sections": ["SKILLS"]}],
                          report=report, resume_text=RESUME)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "run_pipeline", lambda *a, **k: fake_result())
    return TestClient(api.app)


def test_analyze_returns_the_report(client):
    r = client.post("/api/analyze", files={"file": ("cv.docx", b"x")}, data={"correction": "true"})
    assert r.status_code == 200
    body = r.json()
    assert body["best_job_id"] == "7" and body["headline"] == "Your closest match is RN ICU"
    job = body["jobs"][0]
    assert [q["status"] for q in job["requirements"]] == ["shown", "not_found"]
    assert job["highlight"] == [[RESUME.index("ACLS"), RESUME.index("ACLS") + len("ACLS, BLS")]]
    assert body["how"]["rejected"][0]["issue"] == "an employer benefit, not a requirement"


def test_wrong_file_type_is_refused_plainly(client):
    r = client.post("/api/analyze", files={"file": ("cv.txt", b"x")})
    assert r.status_code == 415 and r.json()["detail"]["message"] == "Upload a PDF or Word (.docx) file."


def test_oversized_file_is_refused(client):
    r = client.post("/api/analyze", files={"file": ("cv.pdf", b"x" * (api.MAX_UPLOAD_BYTES + 1))})
    assert r.status_code == 413


def test_second_request_while_busy_gets_429(client):
    api._busy.acquire()
    try:
        r = client.post("/api/analyze", files={"file": ("cv.pdf", b"x")})
        assert r.status_code == 429
    finally:
        api._busy.release()


def test_rejection_is_structured_and_soft_ones_can_be_overridden(monkeypatch):
    from resumematch.ingest.validate import ResumeRejected
    seen = {}

    def pipeline(*a, allow_unusual=False, **k):
        seen["allow_unusual"] = allow_unusual
        if not allow_unusual:
            raise ResumeRejected("not_resume", "This doesn't look like a resume.", soft=True)
        return fake_result()

    monkeypatch.setattr(api, "run_pipeline", pipeline)
    client = TestClient(api.app)
    r = client.post("/api/analyze", files={"file": ("cv.pdf", b"%PDF-1.4")})
    assert r.status_code == 422
    assert r.json()["detail"] == {"code": "not_resume", "message": "This doesn't look like a resume.", "soft": True}
    r = client.post("/api/analyze", files={"file": ("cv.pdf", b"%PDF-1.4")}, data={"allow_unusual": "true"})
    assert r.status_code == 200 and seen["allow_unusual"] is True


def test_unexpected_error_is_a_plain_message_not_a_trace(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(api, "run_pipeline", boom)
    r = TestClient(api.app).post("/api/analyze", files={"file": ("cv.pdf", b"%PDF-1.4")})
    assert r.status_code == 500 and r.json()["detail"]["code"] == "internal"
    assert "secret" not in r.text


def test_empty_file_is_refused(client):
    r = client.post("/api/analyze", files={"file": ("cv.pdf", b"")})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "empty"


def test_unexpected_error_never_logs_its_message(monkeypatch, caplog):
    # An exception message can quote the resume; the log gets type and stack only.
    resume_text = "Maria Delgado, maria.delgado.rn@example.com, ICU nurse"

    def boom(*a, **k):
        raise RuntimeError(resume_text)  # message built from data, as in real failures
    monkeypatch.setattr(api, "run_pipeline", boom)
    with caplog.at_level("ERROR", logger="resumematch.api"):
        TestClient(api.app).post("/api/analyze", files={"file": ("cv.pdf", b"%PDF-1.4")})
    logged = caplog.text
    assert "RuntimeError" in logged and "boom" in logged      # still debuggable
    assert "Maria" not in logged and "example.com" not in logged


def test_report_and_health_use_the_index_count(monkeypatch):
    monkeypatch.setattr(api, "run_pipeline", lambda *a, **k: fake_result())
    monkeypatch.setattr(api, "indexed_postings", lambda _dir: 2000)
    monkeypatch.setattr(api, "check", lambda *_a, **_k: [])
    c = TestClient(api.app)
    assert c.post("/api/analyze", files={"file": ("cv.docx", b"x")}).json()["total_postings"] == 2000
    assert c.get("/api/health").json()["postings"] == 2000


def _fake_role_data(monkeypatch):
    from resumematch.market import role_profile as rp
    titles = {str(i): ["Data Engineer", "Co", f"https://x/{i}"] for i in range(20)}
    texts = {str(i): "Requirements: SQL and Airflow." for i in range(20)}

    class Store:
        def get_postings_text(self, ids):
            return {j: [{"text": texts[j], "section": "description"}] for j in ids}
    monkeypatch.setattr(api, "_role_data", lambda: (Store(), titles, {"postings": 100, "df": {}}))
    return rp


def test_role_suggestions(monkeypatch):
    _fake_role_data(monkeypatch)
    body = TestClient(api.app).get("/api/roles", params={"q": "data"}).json()
    assert body["suggestions"] == [{"title": "Data Engineer", "postings": 20}]


def test_role_report_without_and_with_resume(monkeypatch):
    _fake_role_data(monkeypatch)
    monkeypatch.setattr(api, "_resume_text", lambda *a: "I write SQL every day.")
    c = TestClient(api.app)
    body = c.post("/api/role-report", data={"role": "data engineer"}).json()
    assert body["role"] == "Data Engineer" and body["analyzed"] == 20 and body["with_resume"] is False
    assert {t["term"] for t in body["terms"]} >= {"SQL", "Airflow"}
    body = c.post("/api/role-report", data={"role": "data engineer"}, files={"file": ("cv.pdf", b"%PDF")}).json()
    shown = {t["term"]: t["shown"] for t in body["terms"]}
    assert shown["SQL"] is True and shown["Airflow"] is False and "Airflow" in body["summary"]


def test_role_report_needs_a_role(monkeypatch):
    _fake_role_data(monkeypatch)
    r = TestClient(api.app).post("/api/role-report", data={"role": " - "})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "no_role"


def test_filters_reach_the_pipeline_and_come_back_in_the_report(monkeypatch):
    from resumematch.retrieval.filters import SearchFilters
    seen = {}

    def pipeline(*a, filters=None, indexed_ids=None, **k):
        seen["filters"], seen["indexed"] = filters, set(indexed_ids or [])
        result = fake_result()
        result.filters, result.allowed_count = filters, 42
        return result
    monkeypatch.setattr(api, "run_pipeline", pipeline)
    monkeypatch.setattr(api, "_role_data", lambda: (None, {"7": ["RN", None, None]}, {}))
    body = TestClient(api.app).post("/api/analyze", files={"file": ("cv.docx", b"x")},
                                    data={"remote_only": "true", "states": "MA", "min_salary": "70000"}).json()
    assert seen["filters"] == SearchFilters(True, ("MA",), (), (), 70000) and seen["indexed"] == {"7"}
    assert body["filters"]["applied"] == "Remote only; MA; at least $70,000 a year" and body["filters"]["postings"] == 42


def test_no_filters_means_no_filter_block(client):
    body = client.post("/api/analyze", files={"file": ("cv.docx", b"x")}).json()
    assert body["filters"] is None


def test_filters_that_match_nothing_are_a_plain_422(monkeypatch):
    from resumematch.pipeline import NoPostingsMatch

    def pipeline(*a, **k):
        raise NoPostingsMatch("x")
    monkeypatch.setattr(api, "run_pipeline", pipeline)
    monkeypatch.setattr(api, "_role_data", lambda: (None, {}, {}))
    r = TestClient(api.app).post("/api/analyze", files={"file": ("cv.docx", b"x")}, data={"remote_only": "true"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "no_matching_postings"


def test_role_report_applies_filters(monkeypatch):
    _fake_role_data(monkeypatch)
    details = {str(i): {"remote": i < 16, "location": "Boston, MA", "work_type": "Full-time"} for i in range(20)}
    monkeypatch.setattr(api, "_load_json", lambda path: details if "details" in path else None)
    c = TestClient(api.app)
    body = c.post("/api/role-report", data={"role": "data engineer", "remote_only": "true"}).json()
    assert body["matched"] == 16 and body["filters"]["applied"] == "Remote only"
    r = c.post("/api/role-report", data={"role": "data engineer", "work_types": "Contract"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "no_matching_postings"


def _fake_tailor_store(monkeypatch, chunks):
    class Store:
        def __init__(self, *a, **k):
            pass

        def get_posting_chunks(self, job_id):
            return chunks
    monkeypatch.setattr(api, "VectorStore", Store)


def test_tailor_bullets_returns_shown_and_rejected(monkeypatch):
    from resumematch.generation.tailor import Rewrite, TailorResult
    _fake_tailor_store(monkeypatch, [{"text": "Kafka required.", "title": "RN ICU", "company_name": "Acme", "section": "description"}])
    result = TailorResult("bullets", rewrites=[Rewrite("a b c", "x y z", "Kafka"), Rewrite("a", "b", "c", issue="adds b")])
    monkeypatch.setattr(api.tailor, "suggest_rewrites", lambda *a: result)
    body = TestClient(api.app).post("/api/tailor", json={"kind": "bullets", "job_id": "7", "resume_text": RESUME}).json()
    assert body["shown"] == [{"original": "a b c", "text": "x y z", "requirement": "Kafka"}]
    assert body["rejected"][0]["issue"] == "adds b" and body["message"] is None


def test_tailor_letter_cites_evidence_and_passes_it_on(monkeypatch):
    from resumematch.generation.tailor import Evidence, LetterSentence, TailorResult
    _fake_tailor_store(monkeypatch, [{"text": "ACLS required.", "title": "RN ICU", "company_name": "Acme", "section": "description"}])
    seen = {}

    def draft(resume, title, company, req, evidence, client):
        seen["evidence"] = evidence
        return TailorResult("letter", evidence=evidence, sentences=[
            LetterSentence("I am applying.", 1, 0), LetterSentence("I hold ACLS.", 2, 1)])
    monkeypatch.setattr(api.tailor, "draft_letter", draft)
    body = TestClient(api.app).post("/api/tailor", json={
        "kind": "letter", "job_id": "7", "resume_text": RESUME,
        "evidence": [{"requirement": "ACLS", "resume_quote": "ACLS, BLS"}]}).json()
    assert seen["evidence"] == [Evidence("ACLS", "ACLS, BLS")]
    assert body["paragraphs"][1] == [{"text": "I hold ACLS.", "resume_quote": "ACLS, BLS", "requirement": "ACLS"}]


def test_tailor_refuses_bad_requests(monkeypatch):
    _fake_tailor_store(monkeypatch, [])
    c = TestClient(api.app)
    assert c.post("/api/tailor", json={"kind": "poem", "job_id": "7", "resume_text": RESUME}).status_code == 400
    assert c.post("/api/tailor", json={"kind": "letter", "job_id": "7", "resume_text": " "}).status_code == 400
    assert c.post("/api/tailor", json={"kind": "letter", "job_id": "7", "resume_text": RESUME}).status_code == 404


def test_interview_returns_questions_with_hints(monkeypatch):
    from resumematch.generation.interview import InterviewPrep, PrepItem
    _fake_tailor_store(monkeypatch, [{"text": "ACLS required.", "title": "RN ICU", "company_name": "Acme", "section": "description"}])
    seen = {}

    def prepare(resume, title, company, req_text, reqs, client):
        seen["reqs"] = reqs
        return InterviewPrep([PrepItem("ACLS", "shown", resume_quote="ACLS, BLS", question="Tell me about ACLS?"),
                              PrepItem("ECMO", "not_found", question="Your ECMO experience?", generated=False, issue="x")],
                             [PrepItem("About the role", "general", question="Why us?")])
    monkeypatch.setattr(api.interview, "prepare", prepare)
    body = TestClient(api.app).post("/api/interview", json={
        "job_id": "7", "resume_text": RESUME, "requirements": [{"text": "ACLS", "resume_quote": "ACLS, BLS"}]}).json()
    assert seen["reqs"][0].text == "ACLS" and seen["reqs"][0].resume_quote == "ACLS, BLS"
    assert body["items"][0]["hint"].startswith('Draw on: "ACLS, BLS"') and body["templates"] == 1
    assert body["general"][0]["question"] == "Why us?" and body["message"] is None


def test_interview_with_nothing_checkable_says_so(monkeypatch):
    from resumematch.generation.interview import InterviewPrep
    _fake_tailor_store(monkeypatch, [{"text": "x", "title": "T", "company_name": None, "section": "description"}])
    monkeypatch.setattr(api.interview, "prepare", lambda *a: InterviewPrep([], []))
    body = TestClient(api.app).post("/api/interview", json={"job_id": "7", "resume_text": RESUME}).json()
    assert body["items"] == [] and "nothing to prepare" in body["message"]


def test_compare_reports_changes_between_versions(monkeypatch):
    import json as _json
    monkeypatch.setattr(api, "_resume_text", lambda *a: "Maria Delgado, RN\nCertifications: ACLS, BLS, PALS\nECMO support.")
    jobs = [{"job_id": "7", "title": "RN ICU", "company": "Acme", "requirements": ["ACLS certification", "ECMO experience"]}]
    body = TestClient(api.app).post("/api/compare", files={"file": ("v2.pdf", b"%PDF")},
                                    data={"old_resume_text": RESUME, "jobs": _json.dumps(jobs)}).json()
    assert (body["shown_before"], body["shown_after"]) == (1, 2)
    assert body["jobs"][0]["changes"] == [{"requirement": "ECMO experience", "before": "Not found", "after": "Shown",
                                           "better": True, "evidence": "ECMO support."}]
    assert body["summary"] == "Your revised resume shows 2 of 2 requirements in this posting, up from 1."


def test_compare_refuses_bad_input(client):
    r = client.post("/api/compare", files={"file": ("v2.pdf", b"%PDF")}, data={"old_resume_text": RESUME, "jobs": "{"})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "bad_jobs"
    r = client.post("/api/compare", files={"file": ("v2.txt", b"x")}, data={"old_resume_text": RESUME, "jobs": "[]"})
    assert r.status_code == 415
