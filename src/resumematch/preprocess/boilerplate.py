"""Sentence splitting for job postings, and boilerplate detection.

Boilerplate (equal-opportunity statements, accommodation notices,
benefits blurbs, "this job description is subject to change") says
nothing about what a job requires, but it is long and near-identical
across thousands of postings -- so it embeds close to everything and
wins retrieval slots from real requirements. On the first 2k postings,
13% of chunks contained equal-opportunity text.

Two rules, because each catches what the other misses:

1. FREQUENCY, learned from the corpus (find_boilerplate): a sentence is
   boilerplate if it appears verbatim in >= 20 postings from >= 10
   different companies. Cross-company is the key: a sentence repeated
   within ONE company is usually that company's real duties
   ("assists in the preparation of a work schedule for all nursing
   personnel": 6 postings, 1 company) and must be kept. Sentences that
   state a requirement are exempt even if frequent ("bachelor's degree
   in computer science, information technology, or related field" is in
   24 postings from 16 companies).
2. LEGAL PATTERNS (_LEGAL): EEO / non-discrimination / accommodation
   wording, which varies by company name and so slips past exact
   matching.

Both rules exempt any sentence that states a requirement (degree,
years of experience, "must be able to", ...): physical requirements
are often worded "must be able to lift 50 pounds with or without
reasonable accommodation", which the legal pattern alone would drop.
Losing a requirement is worse than keeping a stray legal sentence.

Known gap: one company's own "About us" paragraph repeated across its
postings is kept -- the frequency rule can't tell it from real duties.
"""

import re
from collections import defaultdict
from collections.abc import Iterable

# Bullets/sentences glued without a space ("oil baths.Use standard") are
# common (53% of descriptions); re-insert the space before splitting.
_GLUED = re.compile(r"([a-z0-9)][.!?:])(?=[A-Z])")
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|\s*[•·▪●]\s*")

_LEGAL = re.compile(
    r"equal (employment )?opportunity|without regard to|regardless of (race|age|sex|gender)|"
    r"reasonable accommodations?|protected veteran|affirmative action|"
    r"(race|color), (religion|creed)|sexual orientation|genetic information|"
    r"e-?verify|fair chance (ordinance|act)|arrest (and|or) conviction records",
    re.IGNORECASE,
)
_REQUIREMENT_CUE = re.compile(
    r"\b(degree|bachelor|master'?s|ph\.?d|diploma|certif\w*|licen[cs]\w*|"
    r"years? of (\w+ )?experience|experience (with|in)|proficien\w*|knowledge of|"
    r"familiar\w* with|must have|must be able|ability to|lift\w*|pounds|lbs|"
    r"responsibilities include)\b",
    re.IGNORECASE,
)

MIN_WORDS = 8            # frequency rule: shorter sentences are never boilerplate
MIN_LEGAL_WORDS = 5      # legal rule: the pattern is specific enough to judge shorter
                         # statements ("Delaware North is an equal opportunity employer.")
MIN_POSTINGS = 20
MIN_COMPANIES = 10


def split_sentences(text: str) -> list[str]:
    """Sentences/bullets/lines of a posting, in order, whitespace-tidied."""
    pieces = _SPLIT.split(_GLUED.sub(r"\1 ", text))
    return [" ".join(p.split()) for p in pieces if p and p.strip()]


def normalize_sentence(sentence: str) -> str:
    return " ".join(sentence.lower().split())


def find_boilerplate(
    company_descriptions: Iterable[tuple[str | None, str]],
    min_postings: int = MIN_POSTINGS,
    min_companies: int = MIN_COMPANIES,
) -> set[str]:
    """
    One pass over (company_name, description) pairs -> the set of
    normalized sentences that are boilerplate by the frequency rule.
    Feed it the same de-duplicated postings that get indexed, so a
    posting copied 50 times doesn't make its own text "boilerplate".
    """
    postings: dict[str, int] = defaultdict(int)
    companies: dict[str, set] = defaultdict(set)
    for company, description in company_descriptions:
        seen: set[str] = set()
        for sentence in split_sentences(description):
            key = normalize_sentence(sentence)
            if len(key.split()) < MIN_WORDS or key in seen:
                continue
            seen.add(key)
            postings[key] += 1
            if len(companies[key]) < min_companies:  # only need to know it reached the bar
                companies[key].add(company)
    return {
        key for key, count in postings.items()
        if count >= min_postings
        and len(companies[key]) >= min_companies
        and not _REQUIREMENT_CUE.search(key)
    }


def is_boilerplate(sentence: str, frequent: set[str]) -> bool:
    words = len(sentence.split())
    if words < MIN_LEGAL_WORDS:
        return False
    key = normalize_sentence(sentence)
    if _REQUIREMENT_CUE.search(key):
        return False
    return bool(_LEGAL.search(key)) or (words >= MIN_WORDS and key in frequent)
