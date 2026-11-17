"""Compare two versions of a resume against the same postings.

The user edits their resume after a report and wants to know what the
edit changed. Running the whole pipeline again would search again and
may return other postings, so the two reports could not be compared.
Instead the report's postings and requirements stay fixed and both
versions are checked against them with one rule, no model:

    shown   one line of the resume (or two consecutive lines, for PDF
            wraps) holds all of the requirement's key terms when it has
            one or two, two thirds when it has more; that line is the
            evidence. (Half was too loose: "Charge nurse" counted as shown
            by "Staff nurse".)
    partly  at least two of its key terms appear somewhere (job_report rule)
    not found

The same rule for both versions means a change in status comes from the
edit, not from the model answering differently. It can differ from the
model-based report, and the page says so.
"""

from dataclasses import dataclass, field

from resumematch.generation.gap_analysis import _found, _normalize, key_terms
from resumematch.generation.job_report import _partly_shown, is_soft

MAX_JOBS = 10
MAX_REQUIREMENTS = 30
MAX_LINES = 12
_RANK = {"not_found": 0, "partly": 1, "shown": 2}


def _lines(text: str) -> list[str]:
    raw = [ln.strip().lstrip("-*\u2022\u25aa\u25cf ").strip() for ln in text.splitlines() if ln.strip()]
    return raw + [f"{a} {b}" for a, b in zip(raw, raw[1:], strict=False)]   # wrapped lines


def check(requirement: str, text: str, lines: list[str] | None = None) -> tuple[str, str | None]:
    """(status, evidence line) for one requirement against one resume version."""
    terms = key_terms(requirement)
    if not terms:
        return "not_found", None
    best, best_n = None, 0
    for line in lines if lines is not None else _lines(text):
        n = len(_found(terms, line))
        if n > best_n or (n == best_n and best is not None and len(line) < len(best)):
            best, best_n = line, n
    needed = len(terms) if len(terms) <= 2 else -(-2 * len(terms) // 3)
    if best_n >= needed:
        return "shown", best
    return ("partly" if _partly_shown(requirement, text) else "not_found"), None


@dataclass
class Change:
    requirement: str
    before: str
    after: str
    evidence_after: str | None = None
    evidence_before: str | None = None


@dataclass
class JobCompare:
    job_id: str
    title: str
    company: str | None
    total: int
    shown_before: int
    shown_after: int
    changes: list[Change] = field(default_factory=list)   # status changed, better first


@dataclass
class Comparison:
    jobs: list[JobCompare]
    added_lines: list[str]
    removed_lines: list[str]

    @property
    def total(self) -> int:
        return sum(j.total for j in self.jobs)

    @property
    def shown_before(self) -> int:
        return sum(j.shown_before for j in self.jobs)

    @property
    def shown_after(self) -> int:
        return sum(j.shown_after for j in self.jobs)

    def summary(self) -> str:
        """One sentence, decided here so every interface says the same."""
        if not self.total:
            return "These postings had no requirements that could be checked, so there is nothing to compare."
        before, after, n = self.shown_before, self.shown_after, sum(1 for j in self.jobs if j.total)
        where = "in this posting" if n == 1 else f"across these {n} postings"
        if after == before:
            return f"Both versions show {after} of {self.total} requirements {where}."
        direction = "up" if after > before else "down"
        return f"Your revised resume shows {after} of {self.total} requirements {where}, {direction} from {before}."


def changed_lines(old: str, new: str) -> tuple[list[str], list[str]]:
    """Lines added and removed, ignoring spacing and case."""
    def norm_lines(text):
        out, seen = [], set()
        for ln in text.splitlines():
            key = _normalize(ln)
            if key and key not in seen:
                seen.add(key)
                out.append((key, ln.strip()))
        return out
    a, b = norm_lines(old), norm_lines(new)
    a_keys, b_keys = {k for k, _ in a}, {k for k, _ in b}
    added = [ln for k, ln in b if k not in a_keys]
    removed = [ln for k, ln in a if k not in b_keys]
    return added[:MAX_LINES], removed[:MAX_LINES]


def compare(old_text: str, new_text: str, jobs: list[dict]) -> Comparison:
    """jobs: [{"job_id", "title", "company", "requirements": [str, ...]}] from the report.
    General strengths ("Strong communication") are skipped, as in the report's counts."""
    old_lines, new_lines = _lines(old_text), _lines(new_text)
    out = []
    for job in jobs[:MAX_JOBS]:
        reqs = []
        for r in job.get("requirements", [])[:MAX_REQUIREMENTS]:
            r = str(r).strip()
            if r and key_terms(r) and not is_soft(r) and r not in reqs:
                reqs.append(r)
        jc = JobCompare(str(job.get("job_id", "")), str(job.get("title", "")), job.get("company"), len(reqs), 0, 0)
        for r in reqs:
            before, ev_before = check(r, old_text, old_lines)
            after, ev_after = check(r, new_text, new_lines)
            jc.shown_before += before == "shown"
            jc.shown_after += after == "shown"
            if before != after:
                jc.changes.append(Change(r, before, after, ev_after, ev_before))
        jc.changes.sort(key=lambda c: _RANK[c.before] - _RANK[c.after])
        out.append(jc)
    added, removed = changed_lines(old_text, new_text)
    return Comparison(out, added, removed)

