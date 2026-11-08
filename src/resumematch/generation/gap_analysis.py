"""Builds the gap-analysis prompt from the resume + retrieved job-posting
chunks, and turns the LLM's answer into a structured, citable result.

Direction: resume is the query, retrieved chunks are candidate job
postings. The model says which requirements in those postings the
resume evidences and which it doesn't, citing excerpts by number; every
citation is resolved in code against the actual retrieved record
(job_id / job_posting_url), never taken from the model's own text.

Two guarantees, and which layer provides each:
  - SHAPE (valid JSON, at most 8 items per list, excerpt numbers in
    range, required fields present) comes from constrained decoding:
    the JSON schema below is compiled by Ollama into a grammar, so the
    model cannot emit anything else. Prompt instructions alone did not
    work: both local models ignored length/format instructions and the
    hand-parsed text format failed in 3 of 5 baseline runs.
  - TRUTH of a match comes from code: each matched item quotes the
    resume, and evidence_in_resume checks the quote really occurs in it.
    Unverifiable claims go to `unverified`, never `matched`.
  - GROUNDING of every claim on both sides also comes from code
    (check_claims): a requirement must appear in the posting it cites; a
    matched quote must mention the requirement; a "missing" requirement
    must not be something the resume plainly contains. Measured before
    these checks (llama3.2:3b, 6 runs): 73% of "matched" requirements
    were not in the cited posting at all -- the model listed the
    candidate's own skills and attached an arbitrary citation -- and 52%
    of "missing" items were skills the resume has.
These are lexical checks on key terms, so they can be fooled by
synonyms ("test-driven development" vs "unit testing" fails the
relevance check) -- they err toward "unverified", never toward a false
match. The UI shows each rejected claim with its reason.
"""

import json
import re
from dataclasses import dataclass, field

from resumematch.generation.ollama_client import OllamaClient

MAX_ITEMS = 8           # per list; enforced by the schema, not by asking
MAX_TEXT_CHARS = 150    # per requirement/quote; stops runaway strings


@dataclass
class CitedClaim:
    requirement: str
    job_id: str
    job_posting_url: str | None
    supporting_excerpt: str
    # Matched claims only: the phrase the model quoted from the resume.
    resume_evidence: str | None = None
    claimed_as: str = "matched"  # what the model said: "matched" or "missing"
    issue: str | None = None     # why a check rejected it (None = passed)
    recited: bool = False        # code moved the citation to the posting that mentions it


@dataclass
class GapAnalysisResult:
    matched: list[CitedClaim]      # passed every check in check_claims
    missing: list[CitedClaim]      # in the cited posting, and not plainly in the resume
    overall_summary: str
    raw_response: str  # keep raw output for debugging/grading evidence
    # Claims (matched OR missing) that failed a check, each with .issue --
    # shown separately, never silently dropped, never presented as fact.
    unverified: list[CitedClaim] = field(default_factory=list)
    attempts: int = 1  # model calls needed (2 = first answer was unusable)


PROMPT_TEMPLATE = """You are checking a candidate's resume against excerpts from job postings.

JOB POSTING EXCERPTS (numbered; these describe the JOBS, not the candidate):
{retrieved_chunks}

CANDIDATE RESUME (the only source of evidence about the candidate):
<<<RESUME
{resume_text}
RESUME>>>

Pick the most important requirements from the job posting excerpts and check each one against the CANDIDATE RESUME.
- matched: the resume clearly shows the requirement. resume_quote must be a short exact phrase
  (at most 6 words) copied from between <<<RESUME and RESUME>>> that proves it. Never quote a job posting excerpt.
- missing: an excerpt asks for the requirement and the resume does not show it.
If you cannot copy a phrase from the resume that proves a requirement, it belongs in missing, not matched.
Write each requirement using the excerpt's own words (e.g. "Kubernetes", "AWS Lambda"), never words from the resume.
Only list requirements that the cited excerpt actually mentions.
Do not list the same requirement twice. Keep each requirement under 8 words.
excerpt is the number of the job posting excerpt that asks for the requirement.
summary: a 2-3 sentence overall assessment.

Respond with JSON only."""

FORMAT_REMINDER = """

Your previous answer could not be used ({problem}). Respond again with JSON only."""


def response_schema(num_excerpts: int) -> dict:
    """JSON Schema for the answer, used for constrained decoding. The
    excerpt range is built from the actual number of retrieved chunks,
    so an out-of-range citation is impossible to generate."""
    excerpt = {"type": "integer", "minimum": 1, "maximum": max(num_excerpts, 1)}
    text = {"type": "string", "maxLength": MAX_TEXT_CHARS}
    return {
        "type": "object",
        "properties": {
            "matched": {
                "type": "array",
                "maxItems": MAX_ITEMS,
                "items": {
                    "type": "object",
                    "properties": {"requirement": text, "resume_quote": text, "excerpt": excerpt},
                    "required": ["requirement", "resume_quote", "excerpt"],
                },
            },
            "missing": {
                "type": "array",
                "maxItems": MAX_ITEMS,
                "items": {
                    "type": "object",
                    "properties": {"requirement": text, "excerpt": excerpt},
                    "required": ["requirement", "excerpt"],
                },
            },
            "summary": {"type": "string", "maxLength": 800},
        },
        "required": ["matched", "missing", "summary"],
    }


def build_prompt(resume_text: str, retrieved_chunks: list[dict]) -> str:
    chunks_text = "\n\n".join(
        f"[{i+1}] ({c.get('title', '')} at {c.get('company_name') or 'unknown company'}, job_id={c['job_id']}) {c['text']}"
        for i, c in enumerate(retrieved_chunks)
    )
    return PROMPT_TEMPLATE.format(resume_text=resume_text, retrieved_chunks=chunks_text)


def _normalize(text: str) -> str:
    """Lowercase, punctuation -> spaces, whitespace collapsed. Keeps + and #
    (C++, C#). PDF line wraps become single spaces, so a phrase that
    wraps across lines in the resume still matches."""
    return " ".join(re.sub(r"[^a-z0-9+#]+", " ", text.lower()).split())


def evidence_in_resume(evidence: str, resume_text: str) -> bool:
    """True if the quoted phrase occurs in the resume as whole words.
    This proves the quote is REAL, not that it's relevant to the
    requirement -- the UI shows the quote so a reader can judge that."""
    needle = _normalize(evidence)
    return bool(needle) and f" {needle} " in f" {_normalize(resume_text)} "


# Words that say nothing about WHICH skill a requirement is about.
_FILLER = set("""
a an the and or of in on with for to as at by from using use including such etc plus
experience experienced knowledge skills skill strong solid proven track record background
expertise exposure familiarity familiar proficiency proficient understanding ability able
years year hands working work excellent good basic advanced deep demonstrated
preferred required requirement nice have must services service tools tool concepts
development developer programming language languages environment environments
""".split())


def key_terms(requirement: str) -> list[str]:
    """The words that identify a requirement: "Experience with AWS Lambda"
    -> ["aws", "lambda"]. Empty for pure filler ("Strong skills")."""
    return [t for t in _normalize(requirement).split() if t not in _FILLER]


def _found(terms: list[str], text: str) -> list[str]:
    haystack = f" {_normalize(text)} "
    return [t for t in terms if f" {t} " in haystack]


def _mentions(terms: list[str], text: str) -> bool:
    """At least half of the key terms appear in text."""
    return len(_found(terms, text)) * 2 >= len(terms)


def check_claims(claims: list[CitedClaim], resume_text: str, retrieved_chunks: list[dict] | None = None) -> None:
    """Sets .issue on every claim that fails a grounding check (in place).

    Both lists:  the requirement must be in the posting it cites -- at
                 least half its key terms appear in that excerpt. If it
                 isn't but another retrieved posting mentions it, the
                 citation is moved there (.recited) instead of rejecting:
                 the model often cites the wrong number for a real
                 requirement ("TypeScript" cited to [2], asked for by [5]).
    matched:     the quote must be in the resume AND mention the
                 requirement (share a key term with it).
    missing:     the resume must not contain ALL of the requirement's
                 key terms ("AWS" listed as missing when the resume
                 says AWS is a contradiction, not a gap).
    """
    for claim in claims:
        terms = key_terms(claim.requirement)
        if not terms:
            claim.issue = "too vague to check against the posting"
            continue
        if not _mentions(terms, claim.supporting_excerpt):
            other = next((c for c in retrieved_chunks or [] if _mentions(terms, c["text"])), None)
            if other is None:
                claim.issue = "none of the retrieved postings mention this requirement"
                continue
            claim.job_id, claim.job_posting_url = other["job_id"], other.get("job_posting_url")
            claim.supporting_excerpt, claim.recited = other["text"], True
        if claim.claimed_as == "matched":
            if not claim.resume_evidence or not evidence_in_resume(claim.resume_evidence, resume_text):
                claim.issue = "the quoted resume text isn't in the resume"
            elif not _found(terms, claim.resume_evidence):
                claim.issue = "the quoted resume text doesn't mention this requirement"
        elif len(_found(terms, resume_text)) == len(terms):
            claim.issue = "listed as missing, but the resume mentions it"


def parse_response(raw_response: str, retrieved_chunks: list[dict], resume_text: str | None = None) -> GapAnalysisResult:
    """
    Parses the model's JSON answer, resolving each excerpt number back
    to retrieved_chunks[N-1] for the real job_id/job_posting_url/excerpt.

    Constrained decoding should make every check below redundant; they
    stay because a schema bug, an Ollama version without format support,
    or output cut off at max_tokens would otherwise fail silently.

    If resume_text is given, every claim goes through check_claims and
    failures move to `unverified` with a reason. (None skips the checks --
    tests of the citation logic only.) Exact-duplicate requirements are
    collapsed.

    Raises ValueError (raw response included) on invalid JSON, a missing
    field, or an out-of-range excerpt -- never returns an empty result
    that looks like "no gaps found".
    """
    try:
        data = json.loads(raw_response)
        matched_items, missing_items = data["matched"], data["missing"]
        summary = str(data.get("summary", "")).strip()
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Model response is not the expected JSON ({error}):\n{raw_response}") from error

    def to_claims(items: list, with_quote: bool) -> list[CitedClaim]:
        claimed_as = "matched" if with_quote else "missing"
        claims, seen = [], set()
        for item in items:
            try:
                requirement = str(item["requirement"]).strip()
                n = int(item["excerpt"])
                quote = str(item["resume_quote"]).strip() if with_quote else None
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Malformed item {item!r} ({error}):\n{raw_response}") from error
            if not (1 <= n <= len(retrieved_chunks)):
                raise ValueError(
                    f"Citation [{n}] is out of range (only {len(retrieved_chunks)} "
                    f"chunks were retrieved): {item!r}"
                )
            if not requirement or requirement.lower() in seen:
                continue
            seen.add(requirement.lower())
            chunk = retrieved_chunks[n - 1]
            claims.append(CitedClaim(
                requirement=requirement,
                job_id=chunk["job_id"],
                job_posting_url=chunk.get("job_posting_url"),
                supporting_excerpt=chunk["text"],
                resume_evidence=quote or None,
                claimed_as=claimed_as,
            ))
        return claims

    claimed = to_claims(matched_items, with_quote=True)
    missing = to_claims(missing_items, with_quote=False)

    if resume_text is not None:
        check_claims(claimed + missing, resume_text, retrieved_chunks)
    matched = [c for c in claimed if c.issue is None]
    unverified = [c for c in claimed + missing if c.issue is not None]
    missing = [c for c in missing if c.issue is None]

    return GapAnalysisResult(
        matched=matched,
        missing=missing,
        overall_summary=summary,
        raw_response=raw_response,
        unverified=unverified,
    )


def run_gap_analysis(
    resume_text: str,
    retrieved_chunks: list[dict],
    client: OllamaClient,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    num_ctx: int = 8192,
    max_attempts: int = 2,
) -> GapAnalysisResult:
    """
    Generates (schema-constrained) and parses the gap analysis. With
    constrained decoding the shape is guaranteed; the one retry remains
    for the cases that can still produce unusable output (cut off at
    max_tokens -> incomplete JSON, or an Ollama without schema support).
    Raises the last ValueError if every attempt fails.
    """
    base_prompt = build_prompt(resume_text, retrieved_chunks)
    schema = response_schema(len(retrieved_chunks))
    prompt = base_prompt
    last_error: ValueError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw = client.generate(
                prompt, temperature=temperature, max_tokens=max_tokens, num_ctx=num_ctx, schema=schema
            )
            result = parse_response(raw, retrieved_chunks, resume_text=resume_text)
            result.attempts = attempt
            return result
        except ValueError as error:  # includes GenerationTruncatedError
            last_error = error
            problem = str(error).splitlines()[0][:200]
            prompt = base_prompt + FORMAT_REMINDER.format(problem=problem)
    assert last_error is not None
    raise last_error
