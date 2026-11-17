"""Blind review: remove what identifies a candidate, or hints at age, sex or
background, from resume text shown to a recruiter.

Applied to everything displayed (evidence lines), never to the text the
checks run on, so redaction can't change a result. Removed:

  - the name (the first line, when it looks like one) and every later use
    of its words ("Daniela" in "Daniela led ...")
  - email addresses, phone numbers and web links
  - the location in the header (first lines): "Austin, TX"
  - in an education section: school names and years (a graduation year
    is an age proxy)

What it can't catch: a name that isn't on the first line, or a school
named in a line outside the education section. The page says blind
review hides these details from display, not that it is complete.
"""

import re

from resumematch.preprocess.redact import redact_contacts
from resumematch.preprocess.resume_sections import _is_heading

NAME_MARK = "[name]"
LINK_MARK = "[link]"
PLACE_MARK = "[location]"
SCHOOL_MARK = "[school]"
YEAR_MARK = "[year]"

_URL = re.compile(r"\b(?:https?://|www\.)\S+|\b(?:linkedin\.com|github\.com|gitlab\.com)/\S*", re.IGNORECASE)
_CITY_STATE = re.compile(r"\b[A-Z][A-Za-z.]+(?: [A-Z][A-Za-z.]+)*, [A-Z]{2}\b(?: \d{5})?")
_SCHOOL = re.compile(
    r"(?:[A-Z][\w.&'-]*\s+){0,5}(?:University|College|Institute|School|Academy|Polytechnic)"
    r"(?:\s+(?:of|at|for)(?:\s+[A-Z][\w.&'-]*)+)?")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_EDUCATION = re.compile(r"\b(education|academic|degrees?)\b", re.IGNORECASE)


def candidate_name(text: str) -> str | None:
    """The first non-empty line, if it looks like a name: 2 to 4 capitalized
    words, letters only (plus . ' -), no colon. "Maria Delgado, RN" -> "Maria Delgado"."""
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    first = first.split(",")[0].split("|")[0].strip()
    words = first.split()
    if 2 <= len(words) <= 4 and all(re.fullmatch(r"[A-Z][A-Za-z.'-]*", w) for w in words):
        return first
    return None


def _name_sub(line: str, name: str | None) -> str:
    if not name:
        return line
    for word in sorted({w.strip(".,") for w in name.split() if len(w.strip(".,")) > 1}, key=len, reverse=True):
        line = re.sub(rf"\b{re.escape(word)}\b", NAME_MARK, line)
    return re.sub(rf"{re.escape(NAME_MARK)}(?:\s+{re.escape(NAME_MARK)})+", NAME_MARK, line)


def blind(text: str, name: str | None = None, header_lines: int = 4) -> str:
    """The text with identifying details replaced by markers. The name goes
    first, so "Maria Delgado, RN" isn't read as a city and state."""
    if name is None:
        name = candidate_name(text)
    out_lines = []
    in_education = False
    non_empty = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            non_empty += 1
            if _is_heading(stripped):
                in_education = bool(_EDUCATION.search(stripped))
        new = redact_contacts(_name_sub(line, name)) or ""
        new = _URL.sub(LINK_MARK, new)
        if non_empty <= header_lines:
            new = _CITY_STATE.sub(PLACE_MARK, new)
        if in_education:
            new = _YEAR.sub(YEAR_MARK, _SCHOOL.sub(SCHOOL_MARK, new))
        out_lines.append(new)
    return "\n".join(out_lines)


def blind_line(line: str | None, name: str | None, education_line: bool = False) -> str | None:
    """One evidence line, redacted the same way (used for checklist evidence)."""
    if line is None:
        return None
    new = _URL.sub(LINK_MARK, redact_contacts(_name_sub(line, name)) or "")
    if education_line or _SCHOOL.search(new):
        new = _YEAR.sub(YEAR_MARK, _SCHOOL.sub(SCHOOL_MARK, new))
    return new
