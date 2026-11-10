"""JSON API over the same pipeline the Streamlit app uses, for the
Next.js site in web/.

Run with: uvicorn resumematch.api.app:app --port 8000

Endpoints:
    GET  /api/health       startup problems (index, Ollama, model), each with a one-line fix
    POST /api/analyze      multipart "file" (.pdf/.docx), form "correction" (bool),
                           form "allow_unusual" (bool: "Check it anyway" after a soft warning)

Errors are JSON {"detail": {"code", "message", "soft"}} so the site can
show the message and, when soft is true, offer to continue.

One analysis runs at a time: a local 3B model on one GPU (or CPU) can't
serve two at once without both getting slow, so a second request gets
a clear 429 instead of waiting in a hidden queue.
"""

import json
import logging
import tempfile
import threading
import traceback
from dataclasses import asdict
from functools import cache
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from resumematch.frontend.render import highlight_spans
from resumematch.health import as_dicts, check
from resumematch.pipeline import (
    IndexNotBuiltError,
    PipelineResult,
    ResumeRejected,
    load_config,
    resolve_persist_dir,
    run_pipeline,
)

load_dotenv()

VOCAB_PATH = Path("data/processed/domain_vocab.json")
DETAILS_PATH = Path("data/processed/posting_details.json")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
TOTAL_POSTINGS = 107_528

app = FastAPI(title="ResumeMatch API", docs_url="/api/docs", openapi_url="/api/openapi.json")
_busy = threading.Lock()
log = logging.getLogger("resumematch.api")


def _error(status: int, code: str, message: str, soft: bool = False) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message, "soft": soft})


@cache
def _load_json(path: str):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def report_to_json(result: PipelineResult) -> dict:
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
        "headline": report.headline, "summary": report.summary, "total_postings": TOTAL_POSTINGS,
        "best_job_id": report.best.job_id if report.best else None,
        "jobs": jobs,
        "learn_first": [asdict(i) for i in report.learn_first],
        "resume_text": result.resume_text,
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
    problems = check(resolve_persist_dir(load_config()))
    return {"ok": not any(p.blocking for p in problems), "problems": as_dicts(problems), "busy": _busy.locked()}


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), correction: bool = Form(True), allow_unusual: bool = Form(False)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".pdf", ".docx"):
        raise _error(415, "wrong_type", "Upload a PDF or Word (.docx) file.")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise _error(413, "too_large", "That file is over 5 MB. A resume is usually well under 1 MB.")
    if not data:
        raise _error(400, "empty", "That file is empty.")
    if not _busy.acquire(blocking=False):
        raise _error(429, "busy", "Another resume is being checked right now. Try again in a minute.")
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        result = run_pipeline(path, domain_vocabulary=_load_json(str(VOCAB_PATH)), apply_correction=correction,
                              details=_load_json(str(DETAILS_PATH)) or {}, allow_unusual=allow_unusual)
        return report_to_json(result)
    except ResumeRejected as rejected:
        raise _error(422, rejected.code, rejected.message, rejected.soft) from None
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
