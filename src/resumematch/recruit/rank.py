"""Recruiter mode: rank several resumes against one job description.

Two steps, with the model used once:

1. Requirements. The model lists the job description's requirements
   (JSON schema). Code keeps only those really in the text (half their
   key terms in one sentence), drops benefits, employer statements and
   cut-off phrases (the gap analysis checks), and decides required or
   preferred from the description's own wording: the cue in the sentence
   ("must", "preferred"), else the nearest heading above it
   ("Preferred qualifications").

2. Candidates. Every resume is checked against the same requirements
   with the same lexical rule (generation/compare.check: one line holding
   the requirement's key terms is the evidence). No model per candidate,
   so the model answering differently can't favor one candidate, and ten
   resumes take a fraction of a second.

Ranking: required requirements shown, then ones of unspecified importance,
then preferred, then partial matches. The counts are shown, not hidden in
a score. General strengths ("strong communication") are listed but not
counted, as in the resume report.
"""

import json
import re
from dataclasses import dataclass, field

from resumematch.generation.compare import _lines, check, terms_of
from resumematch.generation.gap_analysis import (
    _BENEFIT,
    _EMPLOYER_STATEMENT,
    _INCOMPLETE,
    _found,
    _mentions,
    key_terms,
)
from resumematch.generation.job_report import classify_importance, is_soft
from resumematch.generation.ollama_client import OllamaClient
from resumematch.generation.tailor import _GENERIC_NOUNS, _clean
from resumematch.preprocess.boilerplate import split_sentences

# Too general to count as a partial match ("data", "systems").
_TOO_GENERAL = _GENERIC_NOUNS | {"data", "information", "technical", "technology", "technologies", "database",
                                 "databases", "framework", "frameworks", "application", "applications",
                                 "software", "development", "tools", "tool"}

MAX_REQUIREMENTS = 15
MIN_JOB_CHARS = 200
MAX_SOURCE_WORDS = 14


@dataclass
class JobRequirement:
    text: str
    importance: str              # "required" | "preferred" | "unspecified"
    soft: bool = False
    issue: str | None = None     # why it was dropped (None = kept)
    source: str | None = None    # the job description's own line, when short enough to check against

    @property
    def check_text(self) -> str:
        """What resumes are checked against: the employer's wording when it is a
        short line ("PostgreSQL or another relational database"), since the
        model's phrase can drop an "or" ("PostgreSQL relational database")."""
        return self.source or self.text


@dataclass
class CandidateResult:
    index: int                   # upload order, for the server only
    file_name: str
    name: str | None
    text: str
    checklist: list[tuple[JobRequirement, str, str | None]] = field(default_factory=list)  # (req, status, evidence)

    def count(self, importance: str, status: str = "shown") -> int:
        return sum(1 for r, s, _e in self.checklist if not r.soft and r.importance == importance and s == status)

    @property
    def partly(self) -> int:
        return sum(1 for r, s, _e in self.checklist if not r.soft and s == "partly")

    def sort_key(self) -> tuple:
        return (-self.count("required"), -self.count("unspecified"), -self.count("preferred"), -self.partly, self.index)


PROMPT = """List the requirements a candidate needs for this job, from the job description below.

Rules:
- Each requirement is a short phrase of 2 to 8 words, using the description's own words.
- Include skills, tools, experience, education, licenses and certifications.
- Leave out benefits, pay, schedules and statements about the employer.
- At most 15.

JOB DESCRIPTION:
{job}
"""

SCHEMA = {
    "type": "object",
    "properties": {"requirements": {"type": "array", "minItems": 1, "maxItems": MAX_REQUIREMENTS,
                                    "items": {"type": "string", "maxLength": 120}}},
    "required": ["requirements"],
}

_HEADING_PREFERRED = re.compile(r"\b(preferred|nice to have|bonus|plus|desired|desirable)\b", re.IGNORECASE)
_HEADING_REQUIRED = re.compile(r"\b(required|requirements|qualifications|must have|minimum|what you need|you have)\b",
                               re.IGNORECASE)


def _section_importance(requirement: str, job_text: str) -> str:
    """The cue of the heading above the line that matches the requirement best:
    "Preferred qualifications:" then "- Kubernetes". Best, not first: the title
    line "Senior Backend Engineer (Java)" also mentions Java."""
    terms = key_terms(requirement)
    heading, best, best_n = "unspecified", "unspecified", 0
    for line in job_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # A heading ends with a colon, or is a short line of capital words only.
        # Not title case ("- Kubernetes") and not "- AWS (EC2, S3, SQS)".
        is_heading = len(stripped.split()) <= 6 and (
            stripped.endswith(":") or bool(re.fullmatch(r"[A-Z][A-Z &/]{3,}", stripped)))
        if is_heading:
            if _HEADING_PREFERRED.search(stripped):
                heading = "preferred"
            elif _HEADING_REQUIRED.search(stripped):
                heading = "required"
            else:
                heading = "unspecified"
            continue
        n = len(_found(terms, stripped)) if terms else 0
        if n > best_n and _mentions(terms, stripped):
            best, best_n = heading, n
    return best


def _best_sentence(requirement: str, job_text: str) -> str | None:
    """The job description sentence that holds the most of the requirement's
    key terms (shortest on a tie). Shortest alone picked the title line
    "Senior Backend Engineer (Java)" for "5+ years of Java backend development"."""
    terms = key_terms(requirement)
    if not terms:
        return None
    candidates = [s for s in split_sentences(job_text) if _mentions(terms, s)]
    if not candidates:
        return None
    return max(candidates, key=lambda s: (len(_found(terms, s)), -len(s)))


def check_requirements(items: list[str], job_text: str) -> list[JobRequirement]:
    out: list[JobRequirement] = []
    seen = set()
    for raw in items:
        text = _clean(raw).strip(" .;:-")
        terms = key_terms(text)
        key = " ".join(sorted(terms))
        if not text or key in seen:
            continue
        seen.add(key)
        req = JobRequirement(text, "unspecified", soft=is_soft(text))
        sentence = _best_sentence(text, job_text)
        if not terms:
            req.issue = "too vague to check"
        elif _INCOMPLETE.search(text):
            req.issue = "an incomplete phrase"
        elif _EMPLOYER_STATEMENT.search(text):
            req.issue = "a statement about the employer"
        elif _BENEFIT.search(text):
            req.issue = "a benefit, not a requirement"
        elif sentence is None:
            req.issue = "not in the job description"
        else:
            importance = classify_importance(sentence, text)
            req.importance = importance if importance != "unspecified" else _section_importance(text, job_text)
            line = sentence.strip().lstrip("-*\u2022 ").strip()
            if len(line.split()) <= MAX_SOURCE_WORDS:
                req.source = line
        out.append(req)
    return out


def extract_requirements(job_text: str, client: OllamaClient) -> list[JobRequirement]:
    prompt = PROMPT.format(job=job_text[:8000])
    last: ValueError | None = None
    for _attempt in (1, 2):
        try:
            items = json.loads(client.generate(prompt, temperature=0.1, max_tokens=700, schema=SCHEMA))["requirements"]
            return check_requirements([str(i) for i in items], job_text)
        except (ValueError, KeyError, TypeError) as error:
            last = error if isinstance(error, ValueError) else ValueError(str(error))
    assert last is not None
    raise last


def rank_candidates(requirements: list[JobRequirement], resumes: list[tuple[str, str, str | None]]) -> list[CandidateResult]:
    """resumes: (file name, text, name or None), in upload order. Returns
    candidates best first."""
    kept = [r for r in requirements if r.issue is None]
    out = []
    for i, (file_name, text, name) in enumerate(resumes):
        c = CandidateResult(i, file_name, name, text)
        lines = _lines(text)
        for r in kept:
            status, evidence = check(r.check_text, text, lines)
            if status == "not_found":
                # For ranking, any specific word of the requirement on the resume
                # is a partial match ("Java concurrency" with Java on the resume).
                found = [t for t in _found(terms_of(r.check_text), text) if t not in _TOO_GENERAL and len(t) > 2]
                status = "partly" if found else status
            c.checklist.append((r, status, evidence))
        out.append(c)
    return sorted(out, key=CandidateResult.sort_key)

