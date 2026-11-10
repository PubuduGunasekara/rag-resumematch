"""Splits resume text into sections that each fit the embedding model.

Why this exists: the embedder (bge-small) reads at most 512 tokens and
silently drops the rest. Embedding a whole resume as ONE query meant a
1,017-token test resume lost its entire EXPERIENCE and TECHNICAL SKILLS
sections -- the most job-relevant text -- without any error. Each
section here becomes its own query; retrieval results are fused per
posting afterwards (see retrieval/fusion.py).

Why sections and not plain fixed-size windows: a section is a coherent
unit of evidence ("what this person built", "what tools they know"), so
each query vector means one thing, and the UI can say WHICH part of the
resume matched a posting. Fixed windows are only the fallback for
resumes with no detectable headings.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

# Matched case-insensitively against a whole line. Covers the common
# headings; anything else is caught by the ALL-CAPS rule below.
_KNOWN_HEADINGS = {
    "summary", "professional summary", "profile", "objective", "about me",
    "education", "experience", "work experience", "professional experience",
    "employment", "employment history", "relevant experience",
    "projects", "personal projects", "academic projects", "selected projects",
    "skills", "technical skills", "core competencies", "technologies",
    "certifications", "certificates", "awards", "honors", "achievements",
    "publications", "research", "research experience", "leadership",
    "volunteer", "volunteering", "activities", "languages", "interests",
}
_CAPS_HEADING = re.compile(r"^[A-Z][A-Z &/]{2,40}$")  # "TECHNICAL SKILLS", "WORK & LEADERSHIP"

# Title Case headings outside the list ("Work History", "Certifications
# and Tools"): every word capitalized (connectors aside) AND one of these
# words present. Without the keyword, short Title Case lines are mostly
# job titles ("Manufacturing Engineer II"). Deliberately absent: words
# that also head job titles ("research", "project", "leadership").
_HEADING_KEYWORDS = {
    "experience", "history", "education", "skills", "summary", "competencies",
    "certifications", "certificates", "licenses", "licensure", "qualifications",
    "projects", "awards", "publications", "tools",
}
_CONNECTORS = {"and", "&", "/", "of", "or"}


def _is_title_case_heading(stripped: str) -> bool:
    words = stripped.split()
    if not all(w in _CONNECTORS or w[0].isupper() for w in words):
        return False
    return any(w.lower() in _HEADING_KEYWORDS for w in words)

# Inside a skills section, lines like "Languages", "LLM" or "Cloud" are
# category labels, not new sections. In a two-column PDF the sidebar's
# skill categories come out as short standalone lines and were read as
# headings (see docs/engineering-log.md, sidebar-headings). While inside a
# skills section, only one of these major headings starts a new section.
_SKILLS_HEADINGS = {"skills", "technical skills", "core competencies", "technologies", "skills and tools"}
_MAJOR_KEYWORDS = {
    "summary", "profile", "objective", "experience", "employment", "history", "education",
    "projects", "certifications", "certificates", "licenses", "licensure", "awards", "honors",
    "publications", "volunteer", "volunteering", "leadership", "activities", "qualifications", "research",
}


def _is_skills_heading(heading: str) -> bool:
    return heading.lower() in _SKILLS_HEADINGS or "skills" in heading.lower().split()


def _is_major_heading(line: str) -> bool:
    words = line.strip().rstrip(":").lower().replace("&", " ").split()
    return any(w in _MAJOR_KEYWORDS for w in words)


# Shorter sections than this are merged into the next one: a lone
# heading or a one-line fragment makes a noisy, low-information query.
MIN_SECTION_WORDS = 12


@dataclass
class ResumeSection:
    heading: str  # "EXPERIENCE", "HEADER" (text before the first heading), or "PART n" (fallback)
    text: str     # heading + body, ready to embed


def _is_heading(line: str) -> bool:
    stripped = line.strip().rstrip(":")
    if not stripped or len(stripped.split()) > 4:
        return False
    if stripped.endswith((".", ",", ";")):  # a sentence or bullet, not a heading
        return False
    return (
        stripped.lower() in _KNOWN_HEADINGS
        or bool(_CAPS_HEADING.match(stripped))
        or _is_title_case_heading(stripped)
    )


def _split_long_line(line: str, max_tokens: int, count_tokens: Callable[[str], int]) -> list[str]:
    """Last resort for a single line over budget: split it on words."""
    pieces: list[str] = []
    part: list[str] = []
    for word in line.split():
        if part and count_tokens(" ".join(part + [word])) > max_tokens:
            pieces.append(" ".join(part))
            part = []
        part.append(word)
    if part:
        pieces.append(" ".join(part))
    return pieces


def _pack_lines(lines: list[str], max_tokens: int, count_tokens: Callable[[str], int]) -> list[str]:
    """Greedily packs whole lines into pieces of <= max_tokens, so a
    section is only ever cut BETWEEN lines (bullets), never mid-line."""
    units: list[str] = []
    for line in lines:
        if count_tokens(line) > max_tokens:
            units.extend(_split_long_line(line, max_tokens, count_tokens))
        else:
            units.append(line)

    def greedy(limit: int) -> list[str]:
        pieces: list[str] = []
        current: list[str] = []
        for unit in units:
            if current and count_tokens("\n".join(current + [unit])) > limit:
                pieces.append("\n".join(current))
                current = []
            current.append(unit)
        if current:
            pieces.append("\n".join(current))
        return pieces

    # Pure greedy packing leaves a scrap at the end (477 + 26 tokens on
    # the test resume's PROJECTS). Aim for evenly sized pieces instead,
    # and fall back to the hard limit if that needs extra pieces.
    first_try = greedy(max_tokens)
    if len(first_try) == 1:
        return first_try
    total = count_tokens("\n".join(units))
    target = min(max_tokens, -(-total // len(first_try)) + 20)  # ceil + slack
    balanced = greedy(target)
    return balanced if len(balanced) == len(first_try) else first_try


def split_resume(
    text: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = 480,
) -> list[ResumeSection]:
    """
    Input:
        text: resume text from parse_resume (newlines preserved).
        count_tokens: the EMBEDDER's tokenizer length (Embedder.count_tokens)
            -- whitespace word counts undercount wordpieces ~2x on resumes
            (1,017 tokens vs 494 words on the test resume).
        max_tokens: budget per section. 480, not 512, leaves room for the
            [CLS]/[SEP] specials and tokenizer differences.
    Output:
        Sections in document order, every one <= max_tokens.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    # 1. Group lines under headings. The first line is never a heading:
    #    it's almost always the candidate's name, often in capitals.
    groups: list[tuple[str, list[str]]] = [("HEADER", [lines[0]])]
    for line in lines[1:]:
        in_skills = _is_skills_heading(groups[-1][0])
        if _is_heading(line) and (not in_skills or _is_major_heading(line)):
            groups.append((line.rstrip(":").upper(), [line]))
        else:
            groups[-1][1].append(line)

    if len(groups) < 3:  # HEADER + at least 2 real sections
        # Not a sectioned resume (or headings that can't be detected): fixed windows.
        pieces = _pack_lines(lines, max_tokens, count_tokens)
        return [ResumeSection(f"PART {i + 1}", piece) for i, piece in enumerate(pieces)]

    # 2. Merge tiny groups (a bare heading, a one-line fragment) forward.
    merged: list[tuple[str, list[str]]] = []
    carry: list[str] = []
    for heading, body in groups:
        body = carry + body
        if sum(len(line.split()) for line in body) < MIN_SECTION_WORDS:
            carry = body
            continue
        carry = []
        merged.append((heading, body))
    if carry:  # trailing fragment joins the last real section
        merged[-1] = (merged[-1][0], merged[-1][1] + carry) if merged else ("HEADER", carry)

    # 3. Enforce the token budget inside each section. Continuation
    #    pieces get the heading re-attached, so the budget reserves room.
    sections: list[ResumeSection] = []
    for heading, body in merged:
        budget = max_tokens - count_tokens(heading) - 1
        pieces = _pack_lines(body, budget, count_tokens)
        for i, piece in enumerate(pieces):
            if len(pieces) == 1:
                sections.append(ResumeSection(heading, piece))
            else:
                text_i = piece if i == 0 else f"{heading}\n{piece}"
                sections.append(ResumeSection(f"{heading} ({i + 1}/{len(pieces)})", text_i))
    return sections
