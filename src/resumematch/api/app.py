"""JSON API over the same pipeline the Streamlit app uses, for the
Next.js site in web/.

Run with: uvicorn resumematch.api.app:app --port 8000

Endpoints:
    GET  /api/health       startup problems (index, Ollama, model), each with a one-line fix (public)
    POST /api/auth/login   JSON {"email", "password"}: sets the session cookie
    POST /api/auth/logout  clears it
    GET  /api/auth/me      the signed-in user
    GET  /api/admin/overview  accounts, recent sign-ins, index size (admin only)
    POST /api/analyze      multipart "file" (.pdf/.docx), form "correction" (bool),
                           form "allow_unusual" (bool: "Check it anyway" after a soft warning)
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
from resumematch.frontend.render import highlight_spans
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
            "requirements": [{**asdict(r), "status": r.status} for r in job.requirements],
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
