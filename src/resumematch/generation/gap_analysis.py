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
from dataclasses import dataclass
from resumematch.generation.ollama_client import OllamaClient

_HEADERS = ("MATCHED", "MISSING", "SUMMARY")


@dataclass
class CitedClaim:
    requirement: str
    job_id: str
    job_posting_url: str | None
    supporting_excerpt: str


@dataclass
class GapAnalysisResult:
    matched: list[CitedClaim]
    missing: list[CitedClaim]   # job_id/url/excerpt identify WHICH posting requires it
    overall_summary: str
    raw_response: str  # keep raw output for debugging/grading evidence


PROMPT_TEMPLATE = """You are analyzing how well a candidate's resume matches a set of job postings.

RESUME EXCERPTS:
{resume_text}

RETRIEVED JOB POSTING EXCERPTS (numbered, most relevant first):
{retrieved_chunks}

Based ONLY on the excerpts above (never assume a skill not shown), respond in exactly this format,
referencing excerpts by their number [N]:

MATCHED:
- <requirement> -- evidenced by resume, required by [N]

MISSING:
- <requirement from [N]> -- not evidenced anywhere in the resume excerpts

SUMMARY:
<2-3 sentence overall assessment>
"""


def build_prompt(resume_text: str, retrieved_chunks: list[dict]) -> str:
    chunks_text = "\n\n".join(
        f"[{i+1}] (job_id={c['job_id']}, {c['section']}) {c['text']}"
        for i, c in enumerate(retrieved_chunks)
    )
    return PROMPT_TEMPLATE.format(resume_text=resume_text, retrieved_chunks=chunks_text)


def parse_response(raw_response: str, retrieved_chunks: list[dict]) -> GapAnalysisResult:
    """
    Parses the strict MATCHED/MISSING/SUMMARY format, resolving each
    "[N]" reference back to retrieved_chunks[N-1] to attach the real
    job_id/job_posting_url/excerpt — the citation in the final output
    is never just the LLM's own text, it's resolved against the actual
    retrieved record.

    Don't silently swallow a malformed response or an out-of-range [N]
    — raise with raw_response included rather than returning an empty
    result that looks like "no gaps found."
    """
    sections: dict[str, list[str]] = {h: [] for h in _HEADERS}
    current: str | None = None

    for raw_line in raw_response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header = line.rstrip(":").upper()
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

    def parse_claims(bullet_lines: list[str]) -> list[CitedClaim]:
        claims = []
        for line in bullet_lines:
            text = line.lstrip("-").strip()
            if not text:
                continue
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
            requirement = text.split(" -- ")[0].strip()
            claims.append(CitedClaim(
                requirement=requirement,
                job_id=chunk["job_id"],
                job_posting_url=chunk.get("job_posting_url"),
                supporting_excerpt=chunk["text"],
            ))
        return claims

    matched = parse_claims(sections["MATCHED"])
    missing = parse_claims(sections["MISSING"])
    summary = " ".join(sections["SUMMARY"]).strip()

    return GapAnalysisResult(
        matched=matched,
        missing=missing,
        overall_summary=summary,
        raw_response=raw_response,
    )


def run_gap_analysis(resume_text: str, retrieved_chunks: list[dict], client: OllamaClient, temperature: float = 0.2, max_tokens: int = 800) -> GapAnalysisResult:
    prompt = build_prompt(resume_text, retrieved_chunks)
    raw = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    return parse_response(raw, retrieved_chunks)
