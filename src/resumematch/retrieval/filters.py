"""Search filters: remote, state, work type, level and minimum salary.

The filter fields are not in the vector index; they live in
data/processed/posting_details.json (job_id -> location, work_type,
level, remote, salary). A filter is turned into the set of job ids that
pass it, and retrieval is limited to that set (see fusion.py for how).

Coverage in the full index (docs/engineering-log.md, search-filters):
work type and remote are known for every posting, level for 75%, a
salary for 31%, and 7% say only "United States" as their location. A
level or salary filter therefore leaves out postings that don't state
one; describe() says so to the user.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

# Pay periods in the salary strings written by job_postings_loader.format_salary.
_PER_YEAR = {"year": 1, "month": 12, "week": 52, "every two weeks": 26, "hour": 2080}
_SALARY = re.compile(r"^\$([\d,.]+)(?: to \$([\d,.]+))? (?:an? |)(year|month|week|every two weeks|hour)$")

NO_LOCATION = "United States"   # how a posting says it names no place


def annual_salary(text: str | None) -> tuple[float, float] | None:
    """("$17 to $20 an hour") -> (35360.0, 41600.0): the range per year,
    assuming 2,080 working hours. None for no salary or another currency."""
    if not text:
        return None
    m = _SALARY.match(text.strip())
    if not m:
        return None
    low = float(m.group(1).replace(",", ""))
    high = float(m.group(2).replace(",", "")) if m.group(2) else low
    factor = _PER_YEAR[m.group(3)]
    return low * factor, high * factor


def state_of(location: str | None) -> str | None:
    """"Boston, MA" -> "MA"; "United States" or anything else -> None."""
    if not location or location == NO_LOCATION:
        return None
    tail = location.rsplit(", ", 1)[-1]
    return tail if len(tail) == 2 and tail.isupper() else None


@dataclass(frozen=True)
class SearchFilters:
    remote_only: bool = False
    states: tuple[str, ...] = field(default_factory=tuple)
    work_types: tuple[str, ...] = field(default_factory=tuple)
    levels: tuple[str, ...] = field(default_factory=tuple)
    min_salary: int | None = None      # per year, US dollars

    @property
    def active(self) -> bool:
        return bool(self.remote_only or self.states or self.work_types or self.levels or self.min_salary)

    def matches(self, detail: dict | None) -> bool:
        if detail is None:
            return not self.active
        if self.remote_only and not detail.get("remote"):
            return False
        if self.states and state_of(detail.get("location")) not in self.states:
            return False
        if self.work_types and detail.get("work_type") not in self.work_types:
            return False
        if self.levels and detail.get("level") not in self.levels:
            return False
        if self.min_salary:
            pay = annual_salary(detail.get("salary"))
            if pay is None or pay[1] < self.min_salary:
                return False
        return True

    def describe(self) -> str:
        """Plain words for the report: "Remote only; MA; Full-time; at least $80,000 a year"."""
        parts = []
        if self.remote_only:
            parts.append("Remote only")
        if self.states:
            parts.append(", ".join(self.states))
        if self.work_types:
            parts.append(", ".join(self.work_types))
        if self.levels:
            parts.append(", ".join(self.levels))
        if self.min_salary:
            parts.append(f"at least ${self.min_salary:,} a year")
        return "; ".join(parts)

    def caveats(self) -> list[str]:
        out = []
        if self.levels:
            out.append("Postings that don't state a level are left out.")
        if self.min_salary:
            out.append("Postings that don't list a salary are left out; hourly pay is counted as 2,080 hours a year.")
        if self.states:
            out.append('Postings that give only "United States" as their location are left out.')
        return out


def parse_filters(remote_only: bool = False, states: str = "", work_types: str = "", levels: str = "",
                  min_salary: str | int | None = None) -> SearchFilters:
    """From form fields (comma-separated lists). Unknown values are kept:
    they simply match nothing, which the "no postings match" message covers."""
    def split(value: str) -> tuple[str, ...]:
        return tuple(sorted({v.strip() for v in (value or "").split(",") if v.strip()}))
    try:
        salary = int(float(min_salary)) if min_salary not in (None, "") else None
    except (TypeError, ValueError):
        salary = None
    return SearchFilters(bool(remote_only), split(states), split(work_types), split(levels),
                         salary if salary and salary > 0 else None)


def allowed_ids(details: dict[str, dict], filters: SearchFilters, job_ids=None) -> set[str]:
    """Job ids (from job_ids, default all in details) that pass the filters."""
    ids = details.keys() if job_ids is None else job_ids
    return {j for j in ids if filters.matches(details.get(j))}


def filter_options(details: dict[str, dict], job_ids=None) -> dict:
    """Choices for the filter form, with posting counts, from the indexed postings."""
    rows = [details[j] for j in (details if job_ids is None else job_ids) if j in details]
    states = Counter(s for s in (state_of(r.get("location")) for r in rows) if s)
    work = Counter(r["work_type"] for r in rows if r.get("work_type"))
    levels = Counter(r["level"] for r in rows if r.get("level"))
    return {
        "postings": len(rows),
        "states": [{"value": s, "postings": n} for s, n in sorted(states.items())],
        "work_types": [{"value": w, "postings": n} for w, n in work.most_common()],
        "levels": [{"value": v, "postings": n} for v, n in levels.most_common()],
        "remote": sum(1 for r in rows if r.get("remote")),
        "with_salary": sum(1 for r in rows if annual_salary(r.get("salary"))),
        "with_level": sum(levels.values()),
    }
