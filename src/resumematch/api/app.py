"""JSON API over the same pipeline the Streamlit app uses, for the
Next.js site in web/.

Run with: uvicorn resumematch.api.app:app --port 8000

Endpoints:
    GET  /api/health       startup problems (index, Ollama, model), each with a one-line fix (public)
    POST /api/auth/login   JSON {"email", "password"}: sets the session cookie
    POST /api/auth/logout  clears it
    GET  /api/auth/me      the signed-in user
    GET  /api/auth/demo-accounts  the demo accounts, and the password when it's public (public)
    GET  /api/admin/overview  accounts, recent sign-ins, index size (admin only)
    POST /api/analyze      multipart "file" (.pdf/.docx), form "correction" (bool),
                           form "allow_unusual" (bool: "Check it anyway" after a soft warning)
    POST /api/tailor       JSON {"kind": "bullets"|"letter", "job_id", "resume_text", "evidence"}:
                           resume bullet rewrites or a cover letter for one posting, checked in code
    POST /api/interview    JSON {"job_id", "resume_text", "requirements"}: likely questions for one
                           posting, each with what on the resume to draw on
    POST /api/compare      multipart "file" (revised resume), form "old_resume_text" and "jobs"
                           (JSON from the report): which requirements the edit changed
    POST /api/rank         form "job_text", multipart "files" (up to 10 resumes), form "blind":
                           candidates ranked against the job's requirements (recruiter, admin)
    POST /api/chat         JSON {"question", "resume_text", "jobs", "history"}: a question about the
                           report, answered from the resume and its postings with sources
    POST /api/assistant    JSON {"message", "resume_text"?, "jobs"?, "history"}: the assistant that
                           works from sign-in; routes to help, a role report, an action or /api/chat
    POST /api/assistant/guest  JSON {"message"}: the assistant before sign-in (public; no model)
    GET  /api/filter-options  choices and counts for the search filters
    GET  /api/roles        ?q= title suggestions for the role report
    POST /api/role-report  form "role", optional multipart "file": what postings with
                           that title ask for, and which of those terms the resume uses

Both POST endpoints take the same optional filter fields: "remote_only" (bool),
"states", "work_types", "levels" (comma-separated) and "min_salary" (per year).

Every endpoint except health needs a signed-in user, and each names the
roles it allows (see api/auth.py).

Errors are JSON {"detail": {"code", "message", "soft"}} so the site can
show the message and, when soft is true, offer to continue.

One analysis runs at a time: a local 3B model on one GPU (or CPU) can't
serve two at once without both getting slow, so a second request gets
a clear 429 instead of waiting in a hidden queue.
"""

import json
import logging
import os
import tempfile
import threading
import traceback
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from functools import cache
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel

from resumematch.api import auth
from resumematch.chat import ask
from resumematch.chat import assistant as assistant_router
from resumematch.frontend.render import highlight_spans
from resumematch.generation import compare as version_compare
from resumematch.generation import interview, tailor
from resumematch.generation.job_report import requirement_excerpt
from resumematch.generation.ollama_client import OllamaClient
from resumematch.health import as_dicts, check
from resumematch.market import role_profile
from resumematch.pipeline import (
    IndexNotBuiltError,
    NoPostingsMatch,
    PipelineResult,
    ResumeRejected,
    indexed_postings,
    load_config,
    resolve_persist_dir,
    run_pipeline,
    validate_resume,
)
from resumematch.recruit import blind as blind_review
from resumematch.recruit import rank as recruit
from resumematch.retrieval import filters as search_filters
from resumematch.retrieval.vector_store import VectorStore

load_dotenv()
auth.state.reset()  # read DEMO_PASSWORD and SESSION_SECRET from .env, loaded just above

VOCAB_PATH = Path("data/processed/domain_vocab.json")
DETAILS_PATH = Path("data/processed/posting_details.json")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the role-report data in the background so the first role report
    # doesn't wait for it (about 9 s from the caches, 20 s without them).
    def warm():
        with suppress(Exception):  # a missing index is reported by /api/health instead
            _role_data()
    threading.Thread(target=warm, daemon=True).start()
    yield


app = FastAPI(title="ResumeMatch API", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
_busy = threading.Lock()
log = logging.getLogger("resumematch.api")


def _error(status: int, code: str, message: str, soft: bool = False) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message, "soft": soft})


@cache
def _load_json(path: str):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def report_to_json(result: PipelineResult, total_postings: int | None = None) -> dict:
    """Everything the results page shows, as plain JSON. All wording a
    user reads is decided in Python (job_report, render) so both front
    ends say exactly the same thing."""
    report = result.report
    jobs = []
    for job in report.jobs:
        quotes = [r.resume_quote for r in job.requirements if r.met and r.resume_quote]
        jobs.append({
            "rank": job.rank, "job_id": job.job_id, "title": job.title, "company": job.company,
            "url": job.url, "details": job.details, "tier": job.tier, "reason": job.reason,
            "met": job.met, "checked": job.checked, "partly": sum(bool(r.partly) for r in job.gaps),
            # Each requirement's own evidence span, so the site can highlight one at a time.
            "requirements": [{**asdict(r), "status": r.status,
                              "spans": highlight_spans(result.resume_text, [r.resume_quote]) if r.met and r.resume_quote else []}
                             for r in job.requirements],
            "highlight": highlight_spans(result.resume_text, quotes),
        })
    analysis = report.analysis
    return {
        "headline": report.headline, "summary": report.summary, "total_postings": total_postings,
        "best_job_id": report.best.job_id if report.best else None,
        "jobs": jobs,
        "learn_first": [asdict(i) for i in report.learn_first],
        "resume_text": result.resume_text,
        "filters": _filters_json(result.filters, result.allowed_count),
        "how": {
            "sections": [s.heading for s in result.sections],
            "retrieved": [{"title": p["title"], "company": p.get("company_name"),
                           "matched_sections": p.get("matched_sections", [])} for p in result.retrieved],
            "corrections": [{"original": c.original_token, "corrected": c.corrected_token} for c in result.corrections],
            "rejected": [{"requirement": c.requirement, "claimed_as": c.claimed_as, "issue": c.issue}
                         for c in (analysis.unverified if analysis else [])],
            "attempts": analysis.attempts if analysis else 0,
        },
    }


@app.get("/api/health")
def health() -> dict:
    """{"ok", "problems": [{"code", "message", "blocking"}], "busy"}. The
    site shows each problem's one-line message before anyone uploads."""
    persist_dir = resolve_persist_dir(load_config())
    problems = as_dicts(check(persist_dir))
    if not auth.state.passwords.configured:
        problems.append({"code": "login_not_configured", "blocking": True,
                         "message": "Sign-in isn't set up: add DEMO_PASSWORD to .env and restart the API."})
    return {"ok": not any(p["blocking"] for p in problems), "problems": problems, "busy": _busy.locked(),
            "postings": indexed_postings(persist_dir)}


# --- Sign-in -------------------------------------------------------------------

class LoginBody(BaseModel):
    email: str
    password: str


@app.post("/api/auth/login")
def login(body: LoginBody, response: Response) -> dict:
    user, token = auth.login(body.email, body.password)
    response.set_cookie(auth.COOKIE_NAME, token, max_age=auth.SESSION_SECONDS, httponly=True, samesite="lax",
                        secure=os.environ.get("COOKIE_SECURE") == "1", path="/")
    return {"user": user.public()}


@app.post("/api/auth/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(auth.COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/auth/demo-accounts")
def demo_accounts() -> dict:
    """For the login page: one entry per role, plus the shared password if
    DEMO_PASSWORD_PUBLIC=1, so picking a role fills in both fields."""
    return {
        "accounts": [{**u.public(), "does": auth.ROLE_DESCRIPTIONS[u.role]} for u in auth.DEMO_USERS.values()],
        "password": auth.state.public_password,
    }


@app.get("/api/auth/me")
def me(user: auth.User = Depends(auth.current_user)) -> dict:
    return {"user": user.public()}


@app.get("/api/admin/overview")
def admin_overview(_user: auth.User = Depends(auth.require_role("admin"))) -> dict:
    return {
        "accounts": [u.public() for u in auth.DEMO_USERS.values()],
        "recent_sign_ins": list(auth.state.recent),
        "postings": indexed_postings(resolve_persist_dir(load_config())),
        "busy": _busy.locked(),
    }


def _read_upload(file: UploadFile) -> tuple[str, bytes]:
    """(suffix, bytes) of an uploaded resume, or a plain error."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".pdf", ".docx"):
        raise _error(415, "wrong_type", "Upload a PDF or Word (.docx) file.")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise _error(413, "too_large", "That file is over 5 MB. A resume is usually well under 1 MB.")
    if not data:
        raise _error(400, "empty", "That file is empty.")
    return suffix, data


def _resume_text(suffix: str, data: bytes, allow_unusual: bool) -> str:
    """Validated resume text; the temporary file is deleted either way."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
    try:
        return validate_resume(tmp.name, allow_unusual=allow_unusual)[0]
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@cache
def _role_data() -> tuple[VectorStore, dict, dict]:
    """Store, posting titles and background term counts, loaded once.
    The first call builds the two caches if missing (about 20 s)."""
    persist_dir = resolve_persist_dir(load_config())
    store = VectorStore(persist_dir=persist_dir)
    if store.count() == 0:
        raise IndexNotBuiltError("empty index")
    cache_dir = DETAILS_PATH.parent
    titles = role_profile.load_titles(store, cache_dir)
    return store, titles, role_profile.load_background(store, titles, cache_dir)


def _filters_json(filters: search_filters.SearchFilters, allowed_count: int | None) -> dict | None:
    if not filters.active:
        return None
    return {"applied": filters.describe(), "caveats": filters.caveats(), "postings": allowed_count}


def _form_filters(remote_only: bool, states: str, work_types: str, levels: str, min_salary: str):
    return search_filters.parse_filters(remote_only, states, work_types, levels, min_salary)


NO_MATCH_MESSAGE = "No indexed postings match these filters. Remove one and try again."


def role_report_to_json(profile: role_profile.RoleProfile, display_role: str,
                        filters: search_filters.SearchFilters | None = None, allowed_count: int | None = None) -> dict:
    filters = filters or search_filters.SearchFilters()
    headline, summary = role_profile.summarize(profile, display_role, filtered=filters.active)
    return {
        "role": display_role, "headline": headline, "summary": summary,
        "filters": _filters_json(filters, allowed_count),
        "matched": profile.matched, "analyzed": profile.analyzed, "facts": profile.facts,
        "with_resume": any(t.shown is not None for t in profile.terms),
        "terms": [asdict(t) for t in profile.terms],
    }


ALL_ROLES = auth.require_role(*auth.ROLES)
RESUME_CHECK_ROLES = auth.require_role("seeker", "advisor", "admin")


@cache
def _filter_options() -> dict:
    _store, titles, _bg = _role_data()
    return search_filters.filter_options(_load_json(str(DETAILS_PATH)) or {}, titles.keys())


@app.get("/api/filter-options")
def filter_options(_user: auth.User = Depends(ALL_ROLES)) -> dict:
    try:
        return _filter_options()
    except IndexNotBuiltError:
        raise _error(503, "no_index", "The job postings index hasn't been built yet.") from None


@app.get("/api/roles")
def roles(q: str = "", _user: auth.User = Depends(ALL_ROLES)) -> dict:
    try:
        _store, titles, _bg = _role_data()
    except IndexNotBuiltError:
        raise _error(503, "no_index", "The job postings index hasn't been built yet.") from None
    return {"suggestions": role_profile.suggest_titles(titles, q)}


@app.post("/api/role-report")
def role_report(role: str = Form(...), file: UploadFile | None = File(None), allow_unusual: bool = Form(False),
                remote_only: bool = Form(False), states: str = Form(""), work_types: str = Form(""),
                levels: str = Form(""), min_salary: str = Form(""),
                _user: auth.User = Depends(ALL_ROLES)) -> dict:
    filters = _form_filters(remote_only, states, work_types, levels, min_salary)
    display_role = " ".join(role.split())[:80]
    if len(role_profile.normalize_role(display_role)) < 2:
        raise _error(400, "no_role", "Enter a job title, for example Data Engineer.")
    if display_role.islower():
        display_role = display_role.title()
    resume_text = None
    if file is not None and file.filename:
        suffix, data = _read_upload(file)
        try:
            resume_text = _resume_text(suffix, data, allow_unusual)
        except ResumeRejected as rejected:
            raise _error(422, rejected.code, rejected.message, rejected.soft) from None
    try:
        store, titles, background = _role_data()
        details = _load_json(str(DETAILS_PATH)) or {}
        allowed = search_filters.allowed_ids(details, filters, titles.keys()) if filters.active else None
        if allowed is not None and not allowed:
            raise _error(422, "no_matching_postings", NO_MATCH_MESSAGE)
        profile = role_profile.build_role_profile(store, titles, background, display_role, details=details,
                                                  resume_text=resume_text, allowed=allowed)
        return role_report_to_json(profile, display_role, filters, len(allowed) if allowed is not None else None)
    except HTTPException:
        raise
    except IndexNotBuiltError:
        raise _error(503, "no_index", "The job postings index hasn't been built yet.") from None
    except Exception as err:  # noqa: BLE001 - same rule as /api/analyze: no stack traces, no resume text in logs
        log.error("role report failed: %s\n%s", type(err).__name__, "".join(traceback.format_tb(err.__traceback__)))
        raise _error(500, "internal", "Something went wrong while building this report. Try again.") from None


class EvidenceItem(BaseModel):
    requirement: str = ""
    resume_quote: str


class TailorBody(BaseModel):
    kind: str
    job_id: str
    resume_text: str
    evidence: list[EvidenceItem] = []


MAX_RESUME_CHARS = 30_000


def tailor_to_json(result: tailor.TailorResult) -> dict:
    """What the report shows for a rewrite or letter request. Wording is decided here."""
    if result.kind == "bullets":
        shown = [r for r in result.rewrites if r.issue is None]
        rejected = [r for r in result.rewrites if r.issue is not None]
        return {
            "kind": "bullets",
            "message": None if result.rewrites else "No experience bullets were found on this resume to rewrite.",
            "shown": [{"original": r.original, "text": r.text, "requirement": r.requirement} for r in shown],
            "rejected": [{"original": r.original, "text": r.text, "issue": r.issue} for r in rejected],
        }
    paragraphs, left_out = tailor.arrange_letter(result.sentences)
    def cited(s: tailor.LetterSentence) -> dict:
        e = result.evidence[s.evidence - 1] if s.evidence else None
        return {"text": s.text, "resume_quote": e.resume_quote if e else None, "requirement": (e.requirement or None) if e else None}
    return {
        "kind": "letter",
        "message": None if result.evidence else "Nothing on this resume could be connected to this posting, so there is nothing grounded to write about.",
        "paragraphs": [[cited(s) for s in para] for para in paragraphs],
        "rejected": [{"text": s.text, "issue": s.issue} for s in result.sentences if s.issue],
        "left_out": left_out,
    }


def _posting(job_id: str) -> tuple[str, str | None, str]:
    """(title, company, requirement sentences) of one indexed posting, or a 404."""
    store = VectorStore(persist_dir=resolve_persist_dir(load_config()))
    chunks = store.get_posting_chunks(job_id)
    if not chunks:
        raise _error(404, "no_posting", "That posting isn't in the index.")
    return chunks[0]["title"], chunks[0].get("company_name"), "\n".join(c["text"] for c in requirement_excerpt(chunks))


@app.post("/api/tailor")
def tailor_endpoint(body: TailorBody, _user: auth.User = Depends(RESUME_CHECK_ROLES)) -> dict:
    """The resume text comes back from the report the browser already has
    (nothing is stored on the server); evidence items are re-checked."""
    if body.kind not in ("bullets", "letter"):
        raise _error(400, "bad_kind", "Ask for bullets or a letter.")
    if not body.resume_text.strip() or len(body.resume_text) > MAX_RESUME_CHARS:
        raise _error(400, "bad_resume", "Run the resume check again, then try this.")
    if not _busy.acquire(blocking=False):
        raise _error(429, "busy", "Another request is running right now. Try again in a minute.")
    try:
        title, company, requirement_text = _posting(body.job_id)
        client = OllamaClient()
        if body.kind == "bullets":
            result = tailor.suggest_rewrites(body.resume_text, title, company, requirement_text, client)
        else:
            evidence = [tailor.Evidence(e.requirement, e.resume_quote) for e in body.evidence[:20]]
            result = tailor.draft_letter(body.resume_text, title, company, requirement_text, evidence, client)
        return tailor_to_json(result)
    except HTTPException:
        raise
    except requests.ConnectionError:
        raise _error(503, "model_down", "The local model isn't running. Start Ollama and try again.") from None
    except ValueError:
        raise _error(502, "bad_model_output", "The model's answer couldn't be used, even after a retry. Try again.") from None
    except Exception as err:  # noqa: BLE001 - type and stack only, never the message (it could quote the resume)
        log.error("tailor failed: %s\n%s", type(err).__name__, "".join(traceback.format_tb(err.__traceback__)))
        raise _error(500, "internal", "Something went wrong. Try again.") from None
    finally:
        _busy.release()


class ReportRequirementItem(BaseModel):
    text: str
    resume_quote: str | None = None
    credential: bool = False


class InterviewBody(BaseModel):
    job_id: str
    resume_text: str
    requirements: list[ReportRequirementItem] = []


def interview_to_json(prep: interview.InterviewPrep) -> dict:
    def item(i: interview.PrepItem) -> dict:
        return {"requirement": i.requirement, "status": i.status, "question": i.question, "hint": i.hint,
                "generated": i.generated, "issue": i.issue}
    return {
        "message": None if prep.items else "This posting had no requirements that could be checked, so there is nothing to prepare from.",
        "items": [item(i) for i in prep.items],
        "general": [item(i) for i in prep.general],
        "templates": sum(not i.generated for i in prep.items),
    }


@app.post("/api/interview")
def interview_endpoint(body: InterviewBody, _user: auth.User = Depends(RESUME_CHECK_ROLES)) -> dict:
    """Requirements come from the report the browser has; their statuses are
    recomputed against the resume text, not trusted."""
    if not body.resume_text.strip() or len(body.resume_text) > MAX_RESUME_CHARS:
        raise _error(400, "bad_resume", "Run the resume check again, then try this.")
    if not _busy.acquire(blocking=False):
        raise _error(429, "busy", "Another request is running right now. Try again in a minute.")
    try:
        title, company, requirement_text = _posting(body.job_id)
        reqs = [interview.ReportRequirement(r.text, r.resume_quote, r.credential) for r in body.requirements[:30]]
        return interview_to_json(interview.prepare(body.resume_text, title, company, requirement_text, reqs, OllamaClient()))
    except HTTPException:
        raise
    except requests.ConnectionError:
        raise _error(503, "model_down", "The local model isn't running. Start Ollama and try again.") from None
    except Exception as err:  # noqa: BLE001 - type and stack only, never the message (it could quote the resume)
        log.error("interview failed: %s\n%s", type(err).__name__, "".join(traceback.format_tb(err.__traceback__)))
        raise _error(500, "internal", "Something went wrong. Try again.") from None
    finally:
        _busy.release()


def comparison_to_json(c: version_compare.Comparison) -> dict:
    labels = {"shown": "Shown", "partly": "Partly shown", "not_found": "Not found"}
    return {
        "summary": c.summary(),
        "total": c.total, "shown_before": c.shown_before, "shown_after": c.shown_after,
        "jobs": [{
            "job_id": j.job_id, "title": j.title, "company": j.company, "total": j.total,
            "shown_before": j.shown_before, "shown_after": j.shown_after,
            "changes": [{"requirement": ch.requirement, "before": labels[ch.before], "after": labels[ch.after],
                         "better": version_compare._RANK[ch.after] > version_compare._RANK[ch.before],
                         "evidence": ch.evidence_after or ch.evidence_before} for ch in j.changes],
        } for j in c.jobs],
        "added_lines": c.added_lines, "removed_lines": c.removed_lines,
        "note": "Both versions are checked the same way, by wording and without the model, so these counts can "
                "differ from the report above.",
    }


@app.post("/api/compare")
def compare_versions(file: UploadFile = File(...), old_resume_text: str = Form(...), jobs: str = Form(...),
                     allow_unusual: bool = Form(False), _user: auth.User = Depends(RESUME_CHECK_ROLES)) -> dict:
    """No model and no search: the report's requirements are checked against
    both versions, so it takes about a second and needs no busy lock."""
    if not old_resume_text.strip() or len(old_resume_text) > MAX_RESUME_CHARS:
        raise _error(400, "bad_resume", "Run the resume check again, then try this.")
    try:
        job_list = json.loads(jobs)
        if not isinstance(job_list, list):
            raise ValueError
    except ValueError:
        raise _error(400, "bad_jobs", "Run the resume check again, then try this.") from None
    suffix, data = _read_upload(file)
    try:
        new_text = _resume_text(suffix, data, allow_unusual)
    except ResumeRejected as rejected:
        raise _error(422, rejected.code, rejected.message, rejected.soft) from None
    return comparison_to_json(version_compare.compare(old_resume_text, new_text, job_list))


RECRUIT_ROLES = auth.require_role("recruiter", "admin")
MAX_CANDIDATES = 10
MAX_JOB_CHARS = 20_000


def ranking_to_json(requirements: list[recruit.JobRequirement], ranked: list[recruit.CandidateResult],
                    failed: list[dict], blind: bool) -> dict:
    kept = [r for r in requirements if r.issue is None and not r.soft]
    totals = {imp: sum(1 for r in kept if r.importance == imp) for imp in ("required", "unspecified", "preferred")}
    candidates = []
    for rank, c in enumerate(ranked, 1):
        def shown(text: str | None, name: str | None = c.name) -> str | None:
            return blind_review.blind_line(text, name) if blind else text
        candidates.append({
            "label": f"Candidate {rank}",
            # Sent so the recruiter can reveal names after ranking; the page hides them in blind mode.
            "identity": {"name": c.name, "file": c.file_name},
            "counts": {imp: {"shown": c.count(imp), "total": totals[imp]} for imp in totals},
            "partly": c.partly,
            "checklist": [{"requirement": r.text, "importance": r.importance, "soft": r.soft, "status": st,
                           "evidence": shown(ev)} for r, st, ev in c.checklist],
        })
    return {
        "blind": blind,
        "requirements": [{"text": r.text, "importance": r.importance, "soft": r.soft, "issue": r.issue} for r in requirements],
        "candidates": candidates,
        "failed": failed,
        "note": "Every resume is checked against the same requirements by the same wording rule, with no model, so "
                "a skill written differently counts as not shown. Ranked by required requirements shown, then "
                "others, then preferred, then partial matches.",
    }


@app.post("/api/rank")
def rank_candidates(job_text: str = Form(...), files: list[UploadFile] = File(...), blind: bool = Form(True),
                    _user: auth.User = Depends(RECRUIT_ROLES)) -> dict:
    """One model call (the job's requirements), then the same lexical check for
    every resume. Uploaded files are deleted as soon as their text is read."""
    job_text = job_text.strip()
    if len(job_text) < recruit.MIN_JOB_CHARS or len(job_text) > MAX_JOB_CHARS:
        raise _error(400, "bad_job", f"Paste the full job description ({recruit.MIN_JOB_CHARS:,} to {MAX_JOB_CHARS:,} characters).")
    if not files or len(files) > MAX_CANDIDATES:
        raise _error(400, "bad_files", f"Upload 1 to {MAX_CANDIDATES} resumes.")
    resumes, failed = [], []
    for n, f in enumerate(files, 1):
        label = f"File {n}" if blind else (f.filename or f"File {n}")
        try:
            suffix, data = _read_upload(f)
            text = _resume_text(suffix, data, allow_unusual=False)
            resumes.append((f.filename or f"File {n}", text, blind_review.candidate_name(text)))
        except HTTPException as err:
            failed.append({"file": label, "message": err.detail["message"]})
        except ResumeRejected as rejected:
            failed.append({"file": label, "message": rejected.message})
    if not resumes:
        raise _error(422, "no_readable_resumes", "None of the files could be read as a resume.")
    if not _busy.acquire(blocking=False):
        raise _error(429, "busy", "Another request is running right now. Try again in a minute.")
    try:
        requirements = recruit.extract_requirements(job_text, OllamaClient())
    except requests.ConnectionError:
        raise _error(503, "model_down", "The local model isn't running. Start Ollama and try again.") from None
    except ValueError:
        raise _error(502, "bad_model_output", "The model's answer couldn't be used, even after a retry. Try again.") from None
    finally:
        _busy.release()
    if not any(r.issue is None and not r.soft for r in requirements):
        raise _error(422, "no_requirements", "No checkable requirements were found in that job description.")
    return ranking_to_json(requirements, recruit.rank_candidates(requirements, resumes), failed, blind)


class ChatRequirement(BaseModel):
    text: str
    status: str = "not_found"
    resume_quote: str | None = None
    soft: bool = False


class ChatJob(BaseModel):
    job_id: str
    closest: bool = False
    requirements: list[ChatRequirement] = []


class ChatTurn(BaseModel):
    question: str
    answer: str


class ChatBody(BaseModel):
    question: str
    resume_text: str
    jobs: list[ChatJob] = []
    history: list[ChatTurn] = []


MAX_CHAT_JOBS = 6


@cache
def _embedder():
    from resumematch.retrieval.embedder import Embedder
    return Embedder(model_name=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))


def _chat_postings(jobs: list[ChatJob], resume_text: str, store: VectorStore) -> list[ask.ReportPosting]:
    """The report's postings rebuilt from the index. The browser's checks are
    kept only as far as they hold: "shown" needs its quote in the resume."""
    details = _load_json(str(DETAILS_PATH)) or {}
    resume = f" {ask._normalize(resume_text)} "
    out = []
    for job in jobs[:MAX_CHAT_JOBS]:
        chunks = store.get_posting_chunks(job.job_id)
        if not chunks:
            continue
        checks = []
        for r in job.requirements[:30]:
            if r.soft or not r.text.strip():
                continue
            quote = (r.resume_quote or "").strip()
            held = r.status == "shown" and quote and f" {ask._normalize(quote)} " in resume
            status = "shown" if held else ("partly" if r.status == "partly" else "not_found")
            checks.append((r.text.strip()[:150], status, quote if held else None))
        out.append(ask.ReportPosting(
            job.job_id, chunks[0]["title"], chunks[0].get("company_name"), chunks[0].get("job_posting_url"),
            "\n".join(c["text"] for c in requirement_excerpt(chunks)), details.get(job.job_id), job.closest, checks))
    return out


def chat_to_json(a: ask.ChatAnswer) -> dict:
    cited = {i for s in a.shown for i in s.sources}
    suffix = {"check": ", report check", "facts": ", facts"}
    def label(p: ask.Piece) -> str:
        return "Your resume" if p.kind == "resume" else f"{p.name or p.label}{suffix.get(p.kind, '')}"
    return {
        "answer": [{"text": s.text, "sources": s.sources} for s in a.shown],
        "message": ask.NO_ANSWER if a.no_answer and not a.shown else None,
        "no_answer": a.no_answer,
        "by_code": a.by_code,
        "sources": [{"id": p.id, "kind": p.kind, "label": label(p), "text": p.text, "url": p.url}
                    for p in a.pieces if p.id in cited],
        "removed": [{"text": s.text, "issue": s.issue} for s in a.sentences if s.issue],
    }


class AssistantJob(ChatJob):
    title: str = ""
    company: str | None = None


class AssistantBody(BaseModel):
    message: str
    resume_text: str = ""
    jobs: list[AssistantJob] = []
    history: list[ChatTurn] = []


def _role_summary(title: str, resume_text: str) -> dict:
    """The role report, shortened for a chat reply. Same code as /api/role-report."""
    store, titles, background = _role_data()
    profile = role_profile.build_role_profile(store, titles, background, title,
                                              details=_load_json(str(DETAILS_PATH)) or {},
                                              resume_text=resume_text or None)
    display = title.title() if title.islower() else title
    headline, summary = role_profile.summarize(profile, display)
    return {"title": display, "headline": headline, "summary": summary, "analyzed": profile.analyzed,
            "terms": [{"term": t.term, "share": t.share, "shown": t.shown} for t in profile.terms[:10]]}


class GuestBody(BaseModel):
    message: str


@app.post("/api/assistant/guest")
def assistant_guest(body: GuestBody) -> dict:
    """Public: what ResumeMatch is and how to sign in. Answers are written in
    chat/assistant.py; no model, no index, no user data."""
    message = " ".join(body.message.split())[:ask.MAX_QUESTION_CHARS] or "hello"
    reply = assistant_router.guest_reply(message, auth.state.public_password is not None)
    return {"kind": reply.kind, "text": reply.text, "actions": [], "suggestions": reply.suggestions}


@app.post("/api/assistant")
def assistant(body: AssistantBody, user: auth.User = Depends(ALL_ROLES)) -> dict:
    """One chat box for every role, from sign-in. Routing is code
    (chat/assistant.py); only report questions reach the model, through the
    same path and checks as /api/chat."""
    message = " ".join(body.message.split())
    if len(message) < 1 or len(message) > ask.MAX_QUESTION_CHARS:
        raise _error(400, "bad_question", f"Ask a question of up to {ask.MAX_QUESTION_CHARS} characters.")
    if len(body.resume_text) > MAX_RESUME_CHARS:
        raise _error(400, "bad_resume", "Run the resume check again, then try this.")
    jobs = [{"job_id": j.job_id, "title": j.title, "company": j.company, "closest": j.closest} for j in body.jobs]
    reply = assistant_router.route(message, user.role, bool(body.resume_text.strip()), jobs)
    if reply is not None:
        out = {"kind": reply.kind, "text": reply.text, "actions": reply.actions, "suggestions": reply.suggestions}
        if reply.kind == "role":
            try:
                summary = _role_summary(reply.actions[0]["title"], body.resume_text)
            except IndexNotBuiltError:
                raise _error(503, "no_index", "The job postings index hasn't been built yet.") from None
            out["role"] = summary
            out["text"] = [summary["headline"] + ".", summary["summary"]]
            out["actions"] = [{"type": "open", "href": f"/role?title={summary['title']}", "label": "Open the full role report"}]
        return out
    if user.role not in assistant_router.RESUME_ROLES:
        reply = assistant_router.help_reply(user.role, False)
        return {"kind": reply.kind, "text": reply.text, "actions": reply.actions, "suggestions": reply.suggestions}
    chat_body = ChatBody(question=message, resume_text=body.resume_text, jobs=body.jobs, history=body.history)
    return {"kind": "qa", "text": [], "actions": [], "suggestions": [], "answer": chat(chat_body, user)}


@app.post("/api/chat")
def chat(body: ChatBody, _user: auth.User = Depends(RESUME_CHECK_ROLES)) -> dict:
    """Nothing is stored: the browser sends the resume text, the report's
    postings and the previous turn with each question."""
    question = " ".join(body.question.split())
    if len(question) < 3 or len(question) > ask.MAX_QUESTION_CHARS:
        raise _error(400, "bad_question", f"Ask a question of up to {ask.MAX_QUESTION_CHARS} characters.")
    if not body.resume_text.strip() or len(body.resume_text) > MAX_RESUME_CHARS:
        raise _error(400, "bad_resume", "Run the resume check again, then try this.")
    if not _busy.acquire(blocking=False):
        raise _error(429, "busy", "Another request is running right now. Try again in a minute.")
    try:
        store = VectorStore(persist_dir=resolve_persist_dir(load_config()))
        postings = _chat_postings(body.jobs, body.resume_text, store)
        history = [(t.question, t.answer) for t in body.history[-ask.MAX_HISTORY:]]
        result = ask.answer(question, body.resume_text, postings, OllamaClient(), _embedder().embed,
                            search=lambda vector: store.query(vector, top_k=12), history=history)
        return chat_to_json(result)
    except requests.ConnectionError:
        raise _error(503, "model_down", "The local model isn't running. Start Ollama and try again.") from None
    except ValueError:
        raise _error(502, "bad_model_output", "The model's answer couldn't be used, even after a retry. Try again.") from None
    except Exception as err:  # noqa: BLE001 - type and stack only, never the message (it could quote the resume)
        log.error("chat failed: %s\n%s", type(err).__name__, "".join(traceback.format_tb(err.__traceback__)))
        raise _error(500, "internal", "Something went wrong. Try again.") from None
    finally:
        _busy.release()


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), correction: bool = Form(True), allow_unusual: bool = Form(False),
            remote_only: bool = Form(False), states: str = Form(""), work_types: str = Form(""),
            levels: str = Form(""), min_salary: str = Form(""),
            _user: auth.User = Depends(RESUME_CHECK_ROLES)) -> dict:
    suffix, data = _read_upload(file)
    filters = _form_filters(remote_only, states, work_types, levels, min_salary)
    if not _busy.acquire(blocking=False):
        raise _error(429, "busy", "Another resume is being checked right now. Try again in a minute.")
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        indexed_ids = _role_data()[1].keys() if filters.active else None
        result = run_pipeline(path, domain_vocabulary=_load_json(str(VOCAB_PATH)), apply_correction=correction,
                              details=_load_json(str(DETAILS_PATH)) or {}, allow_unusual=allow_unusual,
                              filters=filters, indexed_ids=indexed_ids)
        return report_to_json(result, indexed_postings(resolve_persist_dir(load_config())))
    except ResumeRejected as rejected:
        raise _error(422, rejected.code, rejected.message, rejected.soft) from None
    except NoPostingsMatch:
        raise _error(422, "no_matching_postings", NO_MATCH_MESSAGE) from None
    except IndexNotBuiltError:
        raise _error(503, "no_index", "The job postings index hasn't been built yet.") from None
    except requests.ConnectionError:
        raise _error(503, "model_down", "The local model isn't running. Start Ollama and try again.") from None
    except ValueError:
        raise _error(502, "bad_model_output", "The model's answer couldn't be used, even after a retry. Try the file again.") from None
    except Exception as err:  # noqa: BLE001 - last resort: log it, never show a stack trace to a user
        # Type and stack only. An exception message can quote its input,
        # and the input here is someone's resume, so it never goes to the log.
        log.error("analysis failed: %s\n%s", type(err).__name__, "".join(traceback.format_tb(err.__traceback__)))
        raise _error(500, "internal", "Something went wrong while checking this file. Try again, and if it keeps happening, "
                                      "try exporting your resume again.") from None
    finally:
        _busy.release()
        if path:
            Path(path).unlink(missing_ok=True)  # the resume is never kept
