"""Query-correction preprocessing, adapted from Assignment 3's autocorrect.

Direction, now that the architecture is corrected: the RESUME is the
query, job postings are the indexed corpus. So this corrects typos in
the resume's own text (OCR artifacts from PDF extraction, e.g.
"Pyhton", "Kubernets") before it's used to retrieve matching postings —
a single corrupted token can tank embedding-similarity retrieval even
though a human reader wouldn't notice it.

Reuses the edit-distance substitution logic from Assignment 3
(probable_substitutes / min_edit_distance) directly, with the
vocabulary built from a SAMPLE of the job-postings corpus (so tech
jargon like "k8s", "faiss", "pytorch" survives as "known") rather than
a generic English corpus or the resume's own text alone (a resume is
too small a sample to know what's jargon vs. typo).
"""

import re
import string
from collections import Counter
from dataclasses import dataclass

LETTERS = string.ascii_lowercase

# -----------------------------------------------------------------------
# Reused verbatim from Assignment 3 (edit-distance autocorrect). Kept
# here rather than imported from the coursework repo so this module has
# no dependency on that assignment's directory structure.
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


@dataclass
class CorrectionResult:
    original_token: str
    corrected_token: str
    was_corrected: bool
    confidence: float | None  # probability of the chosen correction, if corrected


def correct_query(query: str, vocabulary_probs: dict[str, float], min_confidence_margin: float = 0.0) -> tuple[str, list[CorrectionResult]]:
    """
    Input:
        query: raw resume text (or an extracted query phrase from it).
        vocabulary_probs: word -> probability, from build_domain_vocabulary.
        min_confidence_margin: only replace a token if the top
            edit-distance candidate's probability exceeds the raw
            token's own frequency (if any) by at least this margin —
            guards against "correcting" a rare-but-real word (a real
            surname, a niche tool name not in the sample).
    Output:
        (corrected_query, list of CorrectionResult for tokens actually
        looked at) — kept separate from the string so this is testable
        and debuggable, not a black box.
    """
    results: list[CorrectionResult] = []

    def replace(match: re.Match) -> str:
        original = match.group(0)
        lower = original.lower()

        if lower in vocabulary_probs:
            return original  # known word (incl. domain jargon) -- never touched

        raw_prob = vocabulary_probs.get(lower, 0.0)

        best = _best_edit1_match(lower, vocabulary_probs)
        if best is None:
            edit2 = probable_substitutes(lower, vocabulary_probs, maxret=1)
            best = edit2[0] if edit2 else None

        if best is None:
            results.append(CorrectionResult(original, original, False, None))
            return original

        candidate, candidate_prob = best
        if candidate_prob < raw_prob + min_confidence_margin:
            results.append(CorrectionResult(original, original, False, None))
            return original

        corrected = _match_case(candidate, original)
        results.append(CorrectionResult(original, corrected, True, candidate_prob))
        return corrected

    corrected_query = re.sub(r"[A-Za-z]+", replace, query)
    return corrected_query, results


def build_domain_vocabulary(job_posting_sample_texts: list[str], min_count_threshold: int = 2) -> dict[str, float]:
    """
    Builds the vocabulary/probability dict used by correct_query, from a
    SAMPLE of job-posting description/skills_desc text (not the full
    124k-row corpus — a few thousand postings' worth of text is plenty
    to establish "faiss" and "k8s" are real words; building this from
    the full corpus on every run would be wasted work).

    Input:
        job_posting_sample_texts: list of raw text (e.g. 2000-5000
            sampled posting descriptions).
        min_count_threshold: words below this frequency are excluded —
            mirrors the same OOV-robustness tradeoff documented in
            Assignment 4's StyleGram (too low a threshold lets noise in
            as "known"; too high drops real jargon that appears a
            handful of times across the sample).
    """
    counts: Counter[str] = Counter()
    for text in job_posting_sample_texts:
        for token in re.findall(r"[A-Za-z]+", text.lower()):
            counts[token] += 1

    kept = {word: count for word, count in counts.items() if count >= min_count_threshold}
    total = sum(kept.values())
    if total == 0:
        return {}
    return {word: count / total for word, count in kept.items()}
