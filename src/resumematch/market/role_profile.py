"""Role report: what postings for one job title ask for, across all of them.

The resume report reads the 5 postings closest to a resume. This reads
every indexed posting whose title matches a role ("Data Engineer": 369
postings) and counts which terms their requirement sentences use, so a
user can see what a field asks for as a whole and which of those terms
their resume shows.

No language model: terms are word n-grams (1 to 4 words) from the
requirement sentences, kept when they appear in many of the role's
postings and much more often than in a fixed random sample of all
postings. That contrast drops words every posting uses ("team",
"experience") without a hand-made skill list. Each term carries the
postings it came from, so every number on the page can be opened.
"""

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from resumematch.generation.gap_analysis import _FILLER, _canonical, _normalize, _stem
from resumematch.generation.job_report import requirement_excerpt
from resumematch.retrieval.vector_store import VectorStore

TITLES_FILE = "posting_titles.json"
BACKGROUND_FILE = "background_terms.json"
BACKGROUND_SAMPLE = 3000      # postings in the background sample (fixed seed)
MAX_POSTINGS = 1000           # postings read per role (fixed-seed sample above this)
MIN_POSTINGS = 15             # fewer than this and the shares mean little
MIN_SHARE = 0.08              # a term must appear in at least 8% of the role's postings
MIN_LIFT = 3.0                # ... and at least 3x as often as in the background sample
MAX_NGRAM = 4

# Words that end or start no useful term, on top of the requirement filler list.
_EDGE_STOP = _FILLER | set("""
is are be been will our you your we us this that these those who all any other within across into over
per can may should would could not no more most well also both each etc i e g eg ie their its it
know understand like must able
""".split())

# Single words that pass the frequency contrast for many roles but name no skill.
_GENERIC = set("""
technologies technology platforms platform code processing large end prepare preparation accuracy
principles setting unit practice facility facilities families job requirements engineers nurses
team teams business company role position candidate candidates opportunity responsibilities duties
customers customer client clients systems system solutions solution tasks related field quality support
new high including include includes provide providing ensure ensuring perform performing assist assisting
maintain maintaining develop developing manage managing variety various daily day days hours week weeks
monthly weekly time full part shift duty duties area areas level levels office staff people
design general annual close account accounts scalable integration overall
user users features scale sources progress applications application
""".split())

# Spelling variants folded before matching a term to a resume.
_VARIANTS = [(re.compile(r"\bhealth care\b"), "healthcare"), (re.compile(r"\be mail\b"), "email")]

# Terms never cross a sentence, list item, clause or "and"/"or": "SQL and Airflow"
# is two terms, not one.
_SPLIT = re.compile(r"[.;:\n•·|/]|,(?=\s)|\s(?:and|or|&)\s", re.IGNORECASE)
_TOKEN = re.compile(r"[A-Za-z0-9+#]+")


@dataclass
class TermStat:
    term: str                 # display form, most common casing in the postings ("SQL", "journal entries")
    postings: int             # role postings that use it
    share: float              # postings / postings analyzed
    lift: float               # share / background share
    examples: list[dict] = field(default_factory=list)   # up to 3: job_id, title, company, url
    shown: bool | None = None  # on the resume? None when no resume was given


@dataclass
class RoleProfile:
    role: str
    matched: int              # postings whose title matches
    analyzed: int             # postings read (matched, capped at MAX_POSTINGS)
    terms: list[TermStat]
    facts: dict               # remote share, top locations, work types


# --- Posting titles -------------------------------------------------------

def load_titles(store: VectorStore, cache_dir: str | Path) -> dict[str, list]:
    """job_id -> [title, company, url]. Built from the index on first use
    and cached next to it, so a rebuilt index needs the cache deleted."""
    path = Path(cache_dir) / TITLES_FILE
    if path.exists():
        return json.loads(path.read_text())
    titles = {m["job_id"]: [m["title"], m.get("company_name") or None, m.get("job_posting_url") or None]
              for m in store.iter_metadata()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(titles))
    return titles


def normalize_role(role: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9+#]+", " ", role.lower()).split())


def _title_pattern(role: str) -> re.Pattern:
    words = [re.escape(w) for w in normalize_role(role).split()]
    return re.compile(r"\b" + r"\W+".join(words) + r"s?\b")


def match_postings(titles: dict[str, list], role: str) -> list[str]:
    """Job ids whose title contains the role as whole words, plural allowed:
    "data engineer" matches "Senior Data Engineer II" and "Data Engineers",
    not "Data Engineering Manager" (a different job)."""
    if not normalize_role(role):
        return []
    pattern = _title_pattern(role)
    return sorted(j for j, (title, *_rest) in titles.items() if title and pattern.search(title.lower()))


def suggest_titles(titles: dict[str, list], query: str, limit: int = 8) -> list[dict]:
    """The most common titles containing the query, for the role box."""
    q = normalize_role(query)
    if len(q) < 2:
        return []
    counts: Counter = Counter()
    surface: dict[str, Counter] = defaultdict(Counter)
    for title, *_rest in titles.values():
        if not title:
            continue
        key = normalize_role(title)
        if q in key:
            counts[key] += 1
            surface[key][title.strip()] += 1
    return [{"title": surface[k].most_common(1)[0][0], "postings": n} for k, n in counts.most_common(limit)]


# --- Terms ------------------------------------------------------------------

def candidate_terms(text: str) -> dict[str, str]:
    """normalized n-gram -> its surface form, for one posting's text. An
    n-gram may not start or end on a filler word or contain a number, and
    never crosses a sentence, clause, list or "and"/"or" boundary."""
    out: dict[str, str] = {}
    for part in _SPLIT.split(text):
        raw = _TOKEN.findall(part)
        low = [t.lower() for t in raw]
        for n in range(1, MAX_NGRAM + 1):
            for i in range(len(low) - n + 1):
                gram = low[i:i + n]
                if gram[0] in _EDGE_STOP or gram[-1] in _EDGE_STOP or len(gram[-1]) < 2:
                    continue
                if any(t.isdigit() for t in gram) or (n == 1 and gram[0] in _GENERIC):
                    continue
                if all(t in _GENERIC or t in _EDGE_STOP for t in gram):
                    continue
                out.setdefault(" ".join(gram), " ".join(raw[i:i + n]))
    return out


def _requirement_text(chunks: list[dict]) -> str:
    return " \n".join(c["text"] for c in requirement_excerpt(chunks, max_words=100_000))


def _document_frequency(store: VectorStore, job_ids: list[str]):
    """term -> postings using it, term -> surface-form counts, term -> job ids (first 3)."""
    df: Counter = Counter()
    surface: dict[str, Counter] = defaultdict(Counter)
    where: dict[str, list[str]] = defaultdict(list)
    for job_id, chunks in store.get_postings_text(job_ids).items():
        for term, form in candidate_terms(_requirement_text(chunks)).items():
            df[term] += 1
            surface[term][form] += 1
            if len(where[term]) < 3:
                where[term].append(job_id)
    return df, surface, where


def load_background(store: VectorStore, titles: dict[str, list], cache_dir: str | Path) -> dict:
    """{"postings": n, "df": {term: count}} for a fixed random sample of all
    postings. Cached: about 5 s to build on the full index."""
    path = Path(cache_dir) / BACKGROUND_FILE
    if path.exists():
        return json.loads(path.read_text())
    ids = sorted(titles)
    sample = random.Random(0).sample(ids, min(BACKGROUND_SAMPLE, len(ids)))
    df, _surface, _where = _document_frequency(store, sample)
    data = {"postings": len(sample), "df": {t: c for t, c in df.items() if c >= 2}}
    path.write_text(json.dumps(data))
    return data


def _role_words(role: str) -> set[str]:
    """The role's own words with plural and -ing forms, plus its
    abbreviation ("registered nurse" -> "rn")."""
    words = set(normalize_role(role).split()) | set(_canonical(normalize_role(role)).split())
    return words | {_stem(w) for w in words} | {w + "ing" for w in words} | {w + "s" for w in words}


def _stem_key(term: str) -> str:
    for pattern, replacement in _VARIANTS:
        term = pattern.sub(replacement, term)
    return " ".join(_stem(w) for w in term.split())


# "Degree in Computer Science" is met by "B.S. in Computer Science": match on the field.
_DEGREE_PREFIX = re.compile(r"^(?:(?:bachelor|master)s?(?: s)? )?degree (?:in|of) ", re.IGNORECASE)


def select_terms(df: Counter, surface: dict[str, Counter], analyzed: int, background: dict, role: str,
                 limit: int = 20) -> list[TermStat]:
    """Terms common in the role's postings and rare elsewhere, longest form
    kept when a shorter one is mostly part of it ("journal" inside
    "journal entries"). Terms made only of the role's own words ("data
    engineer" for Data Engineer) say nothing and are dropped."""
    bg_n, bg_df = background["postings"], background["df"]
    role_words = _role_words(role)
    scored = []
    for term, count in df.items():
        share = count / analyzed
        if share < MIN_SHARE or all(w in role_words for w in term.split()):
            continue
        lift = share / ((bg_df.get(term, 0) + 1) / (bg_n + 1))
        if lift >= MIN_LIFT:
            scored.append((share, lift, term))
    scored.sort(key=lambda s: (-s[0], s[2]))
    kept: list[tuple[float, float, str]] = []
    seen: set[str] = set()
    for share, lift, term in scored:
        longer = [s for s, _l, t in scored if t != term and f" {term} " in f" {t} " and s >= 0.6 * share]
        if longer or _stem_key(term) in seen:   # "patients" after "patient"
            continue
        seen.add(_stem_key(term))
        kept.append((share, lift, term))
    return [TermStat(surface[t].most_common(1)[0][0], df[t], round(s, 3), round(lift, 1)) for s, lift, t in kept[:limit]]


# --- Resume comparison ------------------------------------------------------

def _forms(text: str) -> tuple[str, str]:
    """The text stemmed twice: as written, and with written-out names folded
    to abbreviations. Both are needed: folding turns "Bachelor of Science in
    Nursing" into "bsn", so "nursing" is only found in the first form, while
    "Registered Nurse" only matches a resume that says "RN" in the second."""
    plain = _normalize(text)
    for pattern, replacement in _VARIANTS:
        plain = pattern.sub(replacement, plain)
    stem = lambda t: f" {' '.join(_stem(w) for w in t.split())} "  # noqa: E731
    return stem(plain), stem(_canonical(plain))


def term_on_resume(term: str, resume_text: str, _cache: dict | None = None) -> bool:
    """Same matching as the claim checks: lowercase, punctuation ignored,
    plurals folded, written-out names and abbreviations treated alike."""
    haystacks = _cache.setdefault("h", _forms(resume_text)) if _cache is not None else _forms(resume_text)
    needles = [n.strip() for n in _forms(_DEGREE_PREFIX.sub("", term))]
    return any(n and f" {n} " in h for n, h in zip(needles, haystacks, strict=True))


# --- Facts from posting metadata ------------------------------------------

def role_facts(job_ids: list[str], details: dict[str, dict]) -> dict:
    rows = [details[j] for j in job_ids if j in details]
    if not rows:
        return {}
    # "United States" is how a posting says it names no location, so it isn't a place to list.
    locations = Counter(r["location"] for r in rows if r.get("location") and r["location"] != "United States")
    work_types = Counter(r["work_type"] for r in rows if r.get("work_type"))
    with_salary = sum(1 for r in rows if r.get("salary"))
    return {
        "remote_share": round(sum(1 for r in rows if r.get("remote")) / len(rows), 3),
        "top_locations": [{"name": n, "postings": c} for n, c in locations.most_common(5)],
        "work_types": [{"name": n, "postings": c} for n, c in work_types.most_common()],
        "salary_listed_share": round(with_salary / len(rows), 3),
    }


# --- Whole report -------------------------------------------------------------

def build_role_profile(store: VectorStore, titles: dict[str, list], background: dict, role: str,
                       details: dict[str, dict] | None = None, resume_text: str | None = None,
                       limit: int = 20) -> RoleProfile:
    matched = match_postings(titles, role)
    ids = matched if len(matched) <= MAX_POSTINGS else sorted(random.Random(0).sample(matched, MAX_POSTINGS))
    if len(ids) < MIN_POSTINGS:
        return RoleProfile(role, len(matched), len(ids), [], role_facts(ids, details or {}))
    df, surface, where = _document_frequency(store, ids)
    terms = select_terms(df, surface, len(ids), background, role, limit)
    cache: dict = {}
    for stat in terms:
        key = normalize_role(stat.term)
        stat.examples = [{"job_id": j, "title": titles[j][0], "company": titles[j][1], "url": titles[j][2]}
                         for j in where.get(key, []) if j in titles]
        if resume_text is not None:
            stat.shown = term_on_resume(stat.term, resume_text, cache)
    return RoleProfile(role, len(matched), len(ids), terms, role_facts(ids, details or {}))


def summarize(profile: RoleProfile, display_role: str) -> tuple[str, str]:
    """Headline and one-line summary, decided here so every interface says the same."""
    if profile.analyzed < MIN_POSTINGS:
        return (f"Too few postings titled {display_role}",
                f"{profile.matched} indexed postings have that title. Try a broader title, for example without "
                f"the seniority level.")
    checked = [t for t in profile.terms if t.shown is not None]
    if not checked:
        return (f"What {display_role} postings ask for",
                f"The terms used most often across {profile.analyzed:,} {display_role} postings, compared with "
                f"postings in general.")
    shown = sum(t.shown for t in checked)
    missing = [t.term for t in checked if not t.shown][:3]
    summary = f"Your resume uses {shown} of the {len(checked)} terms most common in {profile.analyzed:,} {display_role} postings."
    if missing:
        summary += " The most common ones it doesn't use: " + ", ".join(missing) + "."
    return f"How your resume compares with {display_role} postings", summary
