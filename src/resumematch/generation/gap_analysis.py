"""Builds the gap-analysis prompt from the resume + retrieved job-posting
chunks, and parses the LLM's response into a structured, citable result.

Direction (corrected): resume is the query, retrieved chunks are
candidate job postings. The model's job is to say, per retrieved
posting, what the resume evidences and what it doesn't — grounded ONLY
in the retrieved text, with every claim traceable back to a specific
posting via job_id/job_posting_url (the output structure carries this
through; the LLM is never asked to invent a citation, only to reference
which numbered excerpt supports a claim).
"""

import re
from dataclasses import dataclass, field

from resumematch.generation.ollama_client import GenerationTruncatedError, OllamaClient

_HEADERS = ("MATCHED", "MISSING", "SUMMARY")
_EMPTY_MARKERS = {"none", "n/a", "na", "nothing"}
_EVIDENCE = re.compile(r'resume:\s*["“]([^"”]+)["”]', re.IGNORECASE)


@dataclass
class CitedClaim:
    requirement: str
    job_id: str
    job_posting_url: str | None
    supporting_excerpt: str
    # MATCHED claims only: the phrase the model quoted from the resume as
    # proof (None if it gave none).
    resume_evidence: str | None = None


@dataclass
class GapAnalysisResult:
    matched: list[CitedClaim]      # resume evidence quoted AND found in the resume
    missing: list[CitedClaim]      # job_id/url/excerpt identify WHICH posting requires it
    overall_summary: str
    raw_response: str  # keep raw output for debugging/grading evidence
    # Claimed as matched, but the quoted evidence isn't in the resume (or
    # no quote was given) -- shown separately, never silently dropped and
    # never presented as a match.
    unverified: list[CitedClaim] = field(default_factory=list)
    attempts: int = 1  # model calls needed (2 = first answer was unusable)
    # Output hit the token cap; only complete lines were kept, so later
    # lists (MISSING, SUMMARY) may be incomplete or empty.
    truncated: bool = False


PROMPT_TEMPLATE = """You are checking a candidate's resume against excerpts from job postings.

JOB POSTING EXCERPTS (numbered; these describe the JOBS, not the candidate):
{retrieved_chunks}

CANDIDATE RESUME (the only source of evidence about the candidate):
<<<RESUME
{resume_text}
RESUME>>>

Pick the most important requirements from the job posting excerpts and check each one against the CANDIDATE RESUME.
- MATCHED: the resume clearly shows it. Copy a short exact phrase (at most 6 words) from between <<<RESUME and RESUME>>> that proves it. Never copy evidence from a job posting excerpt.
- MISSING: an excerpt asks for it and the resume does not show it.
If you cannot copy a phrase from the resume that proves a requirement, it is MISSING, not MATCHED.
At most 8 lines per list. Each requirement at most 8 words. If a list is empty, write "- none".
[N] is the number of the excerpt that asks for the requirement.

Example lines (format only):
- Python programming | resume: "Languages: Java, Python" | [3]
- Kubernetes | [1]

Respond in exactly this format and nothing else:

MATCHED:
- <requirement> | resume: "<exact phrase copied from the resume>" | [N]

MISSING:
- <requirement> | [N]

SUMMARY:
<2-3 sentence overall assessment>
"""

FORMAT_REMINDER = """

Your previous answer could not be used ({problem}).
Answer again and follow the format exactly: every line under MATCHED and MISSING ends with an excerpt number like [2];
write "- none" for an empty list; at most 8 short lines per list."""


def build_prompt(resume_text: str, retrieved_chunks: list[dict]) -> str:
    chunks_text = "\n\n".join(
        f"[{i+1}] ({c.get('title', '')}, job_id={c['job_id']}) {c['text']}"
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


def parse_response(raw_response: str, retrieved_chunks: list[dict], resume_text: str | None = None) -> GapAnalysisResult:
    """
    Parses the MATCHED/MISSING/SUMMARY format, resolving each "[N]" back
    to retrieved_chunks[N-1] to attach the real job_id/job_posting_url/
    excerpt -- the citation is never just the LLM's own text.

    If resume_text is given, every MATCHED claim's quoted evidence is
    checked against it; claims whose quote isn't in the resume go to
    `unverified` instead of `matched`. (resume_text=None skips the
    check -- for tests of the citation logic only.)

    Raises ValueError (raw response included) on missing headers, a
    bullet with no [N], or an out-of-range [N] -- never returns an
    empty result that looks like "no gaps found".
    """
    sections: dict[str, list[str]] = {h: [] for h in _HEADERS}
    current: str | None = None

    for raw_line in raw_response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header = line.strip("*# ").rstrip(":").strip("*# ").upper()  # tolerate **MATCHED:** / ## MATCHED
        if header in _HEADERS:
            current = header
            continue
        if current is not None:
            sections[current].append(line)

    if not any(sections[h] for h in _HEADERS):
        raise ValueError(
            "Could not find MATCHED:/MISSING:/SUMMARY: headers in model "
            f"response:\n{raw_response}"
        )

    def parse_claims(bullet_lines: list[str], with_evidence: bool) -> list[CitedClaim]:
        claims = []
        for line in bullet_lines:
            text = line.lstrip("-*• ").strip()
            if not text or text.lower().strip(" .") in _EMPTY_MARKERS:
                continue  # "- none": the model's way of saying the list is empty
            match = re.search(r"\[(\d+)\]", text)
            if not match:
                raise ValueError(
                    f"No [N] citation found in line: {line!r}\nFull response:\n{raw_response}"
                )
            n = int(match.group(1))
            if not (1 <= n <= len(retrieved_chunks)):
                raise ValueError(
                    f"Citation [{n}] is out of range (only {len(retrieved_chunks)} "
                    f"chunks were retrieved): {line!r}"
                )
            chunk = retrieved_chunks[n - 1]
            requirement = re.split(r"\s+\|\s+|\s+--\s+|\s*\[\d+\]", text)[0].strip()
            evidence = None
            if with_evidence:
                found = _EVIDENCE.search(text)
                evidence = found.group(1).strip() if found else None
            claims.append(CitedClaim(
                requirement=requirement,
                job_id=chunk["job_id"],
                job_posting_url=chunk.get("job_posting_url"),
                supporting_excerpt=chunk["text"],
                resume_evidence=evidence,
            ))
        return claims

    claimed = parse_claims(sections["MATCHED"], with_evidence=True)
    missing = parse_claims(sections["MISSING"], with_evidence=False)
    summary = " ".join(sections["SUMMARY"]).strip()

    if resume_text is None:
        matched, unverified = claimed, []
    else:
        matched = [c for c in claimed if c.resume_evidence and evidence_in_resume(c.resume_evidence, resume_text)]
        unverified = [c for c in claimed if c not in matched]

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
    Generates and parses the gap analysis, retrying with a format
    reminder if the answer is unusable (unparseable, uncited, out-of-range
    citation). Local 3B models drifted from the format in 3 of 5 baseline
    runs; one retry is cheap next to showing the user an error. Raises
    the last ValueError if every attempt fails.

    Output cut off at the token cap is SALVAGED rather than retried: the
    incomplete last line is dropped and the complete lines are parsed,
    with result.truncated set. Both local models ignore the "at most 8
    lines" instruction at times and run into the cap; retrying just
    hits the cap again at twice the cost (llama3.1:8b: 3 of 8 runs failed
    this way, ~5 min each).
    """
    base_prompt = build_prompt(resume_text, retrieved_chunks)
    prompt = base_prompt
    last_error: ValueError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            truncated = False
            try:
                raw = client.generate(prompt, temperature=temperature, max_tokens=max_tokens, num_ctx=num_ctx)
            except GenerationTruncatedError as cut_off:
                raw = cut_off.partial_response.rsplit("\n", 1)[0]  # drop the incomplete line
                truncated = True
            result = parse_response(raw, retrieved_chunks, resume_text=resume_text)
            result.attempts = attempt
            result.truncated = truncated
            return result
        except ValueError as error:
            last_error = error
            problem = str(error).splitlines()[0][:200]
            prompt = base_prompt + FORMAT_REMINDER.format(problem=problem)
    assert last_error is not None
    raise last_error
