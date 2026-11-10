"""API contract tests with the pipeline faked (no Ollama, no index)."""

import pytest
from fastapi.testclient import TestClient

from resumematch.api import app as api
from resumematch.generation.gap_analysis import CitedClaim, GapAnalysisResult
from resumematch.generation.job_report import JobAnalysis, JobReport, LearnItem, Requirement
from resumematch.pipeline import PipelineResult
from resumematch.preprocess.resume_sections import ResumeSection

RESUME = "Maria Delgado, RN\nACLS, BLS, PALS"


def fake_result() -> PipelineResult:
    reqs = [Requirement("ACLS", True, "required", False, "ACLS, BLS", None),
            Requirement("Siemens NX", False, "preferred", False, None, None)]
    job = JobAnalysis(1, "7", "RN ICU", "Southcoast Health", "https://x/7", {"location": "New Bedford, MA"},
                      reqs, tier="best", reason="1 requirement not found on your resume yet.")
    rejected = CitedClaim("Child care", "7", None, "", None, claimed_as="missing",
                          issue="an employer benefit, not a requirement")
    analysis = GapAnalysisResult([], [], "", "{}", unverified=[rejected])
    report = JobReport([job], [LearnItem("Siemens NX", 1, False)], "Your closest match is RN ICU", "We found 1.",
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
