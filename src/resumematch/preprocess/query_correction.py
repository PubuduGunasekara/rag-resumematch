"""Query-correction preprocessing: an edit-distance autocorrect for resume text.

Direction, now that the architecture is corrected: the RESUME is the
query, job postings are the indexed corpus. So this corrects typos in
the resume's own text (OCR artifacts from PDF extraction, e.g.
"Pyhton", "Kubernets") before it's used to retrieve matching postings:
a single corrupted token can tank embedding-similarity retrieval even
though a human reader wouldn't notice it.

Reuses edit-distance substitution logic I wrote earlier
(probable_substitutes / min_edit_distance) directly, with the
vocabulary built from the FULL job-postings corpus (so tech jargon like
"faiss", "pytorch", "redis" survives as "known") rather than a generic
English corpus or the resume's own text alone (a resume is too small a
sample to know what's jargon vs. typo).

On top of that sit resume-specific guard rails (skip URLs, acronyms,
brand-cased names, short words; edit-distance 2 only for long words),
because a resume, unlike a short typo-heavy query, is
mostly clean text full of legitimately unknown proper nouns.
"""

import re
import string
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

LETTERS = string.ascii_lowercase

# -----------------------------------------------------------------------
# Edit-distance autocorrect helpers, reused from my earlier code. Kept
# here rather than imported from another repository so this module has
# no dependency on that repository's directory structure.
# -----------------------------------------------------------------------


def delete_letter(word):
    """Yield all strings formed by deleting one character from word."""
    for i in range(len(word)):
        yield word[:i] + word[i + 1:]


def switch_letter(word):
    """Yield all strings formed by swapping two adjacent characters."""
    for i in range(len(word) - 1):
        chars = list(word)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        yield "".join(chars)


def replace_letter(word):
    """Yield all strings formed by replacing one character with a different letter."""
    for i in range(len(word)):
        for ch in LETTERS:
            if ch != word[i]:
                yield word[:i] + ch + word[i + 1:]


def insert_letter(word):
    """Yield all strings formed by inserting one letter at any position."""
    for i in range(len(word) + 1):
        for ch in LETTERS:
            yield word[:i] + ch + word[i:]


def edit_one_letter(word):
    """Return the SET of all strings exactly one edit away from word."""
    candidates = set()
    candidates.update(delete_letter(word))
    candidates.update(switch_letter(word))
    candidates.update(replace_letter(word))
    candidates.update(insert_letter(word))
    return candidates


def probable_substitutes(word, probs, maxret=10):
    """Candidates exactly edit-distance 2 from word, present in probs,
    sorted by probability descending (ties broken alphabetically)."""
    word = word.lower()
    edit1 = edit_one_letter(word)
    edit2 = set()
    for w1 in edit1:
        edit2.update(edit_one_letter(w1))
    edit2.discard(word)
    edit2 -= edit1
    matches = [(w, probs[w]) for w in edit2 if w in probs]
    matches.sort(key=lambda pair: (-pair[1], pair[0]))
    return matches[:maxret]


def _best_edit1_match(word: str, probs: dict[str, float]) -> tuple[str, float] | None:
    candidates = [(w, probs[w]) for w in edit_one_letter(word) if w in probs]
    if not candidates:
        return None
    candidates.sort(key=lambda pair: (-pair[1], pair[0]))
    return candidates[0]


def _match_case(correction: str, original: str) -> str:
    if original.isupper():
        return correction.upper()
    if original[:1].isupper():
        return correction.capitalize()
    return correction



# -----------------------------------------------------------------------
# Resume-specific guard rails (not in the original autocorrect).
#
# The original autocorrect assumed a short, typo-heavy query where every unknown
# word is probably a typo. A resume is the opposite: long, mostly clean,
# and full of proper nouns, acronyms, product names and URLs that are
# unknown to any vocabulary without being typos. Measured on a clean
# test resume, "correct every unknown word" changed 42 words and every
# change was wrong (Redis->Hedis, OpenAI->Openam). Each rule below
# exists because of a specific false positive seen on real text; see
# docs/design.md and scripts/eval_query_correction.py.
# -----------------------------------------------------------------------

# A whitespace-delimited span that looks like a URL, email or domain
# ("github.com/dwhitaker-demo", "x@example.com") is left alone entirely:
# its pieces are identifiers, not words.
_URLISH = re.compile(r"[@/]|[A-Za-z]\.[A-Za-z]")
_WORD = re.compile(r"[A-Za-z]+")


# Applied to the right-stripped text before the word on its line: line
# start, a bullet that opens the line, or sentence/list punctuation. A
# dash mid-line is a separator ("Process Engineer — Midstate ..."),
# which is exactly where company names sit.
_SENTENCE_START = re.compile(r"(^\s*[•\-–—*]?|[.!?:;|])$")


def _maybe_name(token: str, line_prefix: str) -> bool:
    """Capitalized and NOT the first word of a line, bullet or sentence."""
    return token[0].isupper() and not _SENTENCE_START.search(line_prefix.rstrip())


def _skip_reason(token: str, min_token_length: int) -> str | None:
    """Why `token` must not be corrected, or None if it's a candidate."""
    if len(token) < min_token_length:
        return "short"        # Lua, ESP, BT: every short string has a neighbour
    if token.isupper():
        return "acronym"      # NSBM, AWS, EXE
    if any(ch.isupper() for ch in token[1:]):
        return "mixed_case"   # OpenAI, PyTorch, PostgreSQL: brand names
    return None


@dataclass
class CorrectionResult:
    original_token: str
    corrected_token: str
    was_corrected: bool
    confidence: float | None  # probability of the chosen correction, if corrected
    edit_distance: int | None = None  # 1 or 2 when corrected


def correct_query(
    query: str,
    vocabulary_probs: dict[str, float],
    min_token_length: int = 4,
    edit2_min_length: int = 8,
) -> tuple[str, list[CorrectionResult]]:
    """
    Input:
        query: raw resume text.
        vocabulary_probs: word -> probability, from build_domain_vocabulary.
        min_token_length: unknown words shorter than this are never
            corrected (short strings almost always have a real word one
            edit away, so a "match" carries no evidence).
        edit2_min_length: edit-distance-2 candidates (the original autocorrect's
            probable_substitutes) are only considered for words at least
            this long. Two edits on a 6-letter word rewrites a third of
            it (Khoury -> Hours); on a 9-letter word it's a plausible
            double typo (kuberntes -> kubernetes).
            Capitalized words in mid-line ("at Midstate Metal Forming")
            never get edit-2 candidates whatever their length: they're
            often names, and Midstate -> Mistake (2 edits) was a real
            false correction. They still get edit-1 fixes, since skills
            lists are capitalized too ("Tools: Docker, Kubernets").
    Output:
        (corrected_query, list of CorrectionResult for every unknown
        token that was considered) -- kept separate from the string so
        the UI can show exactly what changed.
    """
    results: list[CorrectionResult] = []

    def correct_word(match: re.Match, line_prefix: str) -> str:
        original = match.group(0)
        lower = original.lower()

        if lower in vocabulary_probs:
            return original  # known word (incl. domain jargon) -- never touched
        if _skip_reason(original, min_token_length) is not None:
            return original

        best, distance = _best_edit1_match(lower, vocabulary_probs), 1
        if best is None and len(lower) >= edit2_min_length and not _maybe_name(original, line_prefix):
            edit2 = probable_substitutes(lower, vocabulary_probs, maxret=1)
            best, distance = (edit2[0] if edit2 else None), 2

        if best is None:
            results.append(CorrectionResult(original, original, False, None))
            return original

        candidate, candidate_prob = best
        corrected = _match_case(candidate, original)
        results.append(CorrectionResult(original, corrected, True, candidate_prob, distance))
        return corrected

    def correct_span(match: re.Match) -> str:
        span = match.group(0)
        if _URLISH.search(span):
            return span
        line_start = query.rfind("\n", 0, match.start()) + 1
        return _WORD.sub(
            lambda w: correct_word(w, query[line_start:match.start()] + span[:w.start()]), span
        )

    corrected_query = re.sub(r"\S+", correct_span, query)
    return corrected_query, results


def build_domain_vocabulary(texts: Iterable[str], min_count_threshold: int = 2) -> dict[str, float]:
    """
    Builds the vocabulary/probability dict used by correct_query from
    job-posting text.

    Feed it the FULL corpus, not a sample: one streaming pass over all
    ~124k postings takes ~15s and grows the vocabulary from ~14k to
    ~160k words, which on its own cut false corrections on a clean test
    resume from 42 to 8 (Redis, Vite, OpenAI, common first names are
    all "known" at full scale but not in a 2k-posting sample).

    Input:
        texts: posting description/skills/title text (any iterable, so
            it can stream without holding the corpus in memory).
        min_count_threshold: words below this frequency are excluded.
            2 drops one-off typos in the postings themselves ("kubernets"
            appears once); 5 was tested and drops real words like
            "retries" and "hexagonal", which then get mis-corrected.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(re.findall(r"[a-z]+", text.lower()))

    kept = {word: count for word, count in counts.items() if count >= min_count_threshold}
    total = sum(kept.values())
    if total == 0:
        return {}
    return {word: count / total for word, count in kept.items()}
