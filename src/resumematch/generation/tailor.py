"""Tailoring for one posting: resume bullet rewrites and a cover letter,
both checked in code so they never add facts the resume doesn't state.

The model writes; code decides what is shown. Same idea as the gap
analysis (gap_analysis.check_claims), applied to generated prose:

  Bullet rewrites  a rewrite may only rephrase its original bullet:
                   - every number in it must be in the original bullet;
                   - every new word must already be on the resume, or be
                     a filler word, a verb or adverb form (-ed, -ing, -ly)
                     or one of a few generic nouns (team, project);
                   - it must keep at least half of the original's content words.
  Cover letter     each sentence is checked the same way against the resume
                   and the posting together, and a sentence that cites a
                   verified match must mention it.

A new tool, certificate or skill fails these checks, which is the point:
the measured failure of the base model is claiming things the resume does
not support (docs/engineering-log.md, two-sided-grounding). The checks
are lexical, so they also reject some harmless rewording; every rejection
is shown with its reason, never silently dropped.
"""

import json
import re
from dataclasses import dataclass, field

from resumematch.generation.gap_analysis import (
    _FILLER,
    _canonical,
    _mentions,
    _normalize,
    _stem,
    key_terms,
)
from resumematch.generation.ollama_client import OllamaClient
from resumematch.generation.text_clean import clean_generated
from resumematch.preprocess.resume_sections import _is_heading

MAX_BULLETS = 6
MAX_EVIDENCE = 8
MAX_SENTENCES = 12

_STOP = _FILLER | set("""
is are was were be been being am i me my we our you your he she they them their it its this that these those
who whom which what when where while so than then also both each every all any some more most such very
into onto over under across through during before after about against between up down out off again
not no can could will would shall should may might must do does did done has had having make made
dear hiring manager team sincerely regards thank thanks you yours excited opportunity role position
apply applying application interest interested company organization look forward hearing
""".split())

# Words that claim a level of skill. They are filler when reading a requirement
# ("strong SQL skills"), but in a rewrite they add a claim, so they need support.
_CLAIM_WORDS = set("""
expertise expert proven strong excellent advanced deep demonstrated proficient proficiency extensive
skilled solid track record exceptional outstanding seasoned
""".split())
_STOP = _STOP - _CLAIM_WORDS

# Generic nouns a rewrite may add without claiming a new skill.
_GENERIC_NOUNS = set("""
team teams project projects process processes result results impact performance efficiency workflow
workflows solution solutions user users customer customers client clients requirement requirements goal
goals operation operations initiative initiatives task tasks improvement improvements quality accuracy
delivery outcome outcomes reliability scale scalability time cost costs effort efforts work stakeholder
stakeholders partner partners colleague colleagues group groups organization business product products
service services feature features system systems role responsibility responsibilities
""".split())

# Action verbs in base and -s form: a rewrite may say "Lead" for "Led". Past
# and -ing forms are already allowed by suffix.
_ACTION_VERBS = set("""
lead manage build design develop implement create deliver drive improve reduce increase coordinate
collaborate partner support serve provide ensure maintain own run oversee plan prepare analyze review
monitor streamline optimize automate migrate establish launch train mentor educate teach guide communicate
present resolve troubleshoot deploy integrate scale protect secure process handle track report document
test evaluate assess identify define apply achieve cut raise grow expand enable help respond care assist
conduct perform execute organize schedule facilitate contribute introduce write refactor architect
modernize consolidate standardize spearhead participate
""".split())
_ACTION_VERBS |= {v + "s" for v in _ACTION_VERBS} | {v + "es" for v in _ACTION_VERBS}

_NUMBER = re.compile(r"\d[\d,.]*\d|\d")
# A word starts with a letter that doesn't follow a digit: "99th" and "$85M" are numbers, not words.
_WORD = re.compile(r"(?<![0-9A-Za-z])[A-Za-z][A-Za-z0-9+#'-]*")

# A job or school line, not an achievement: it has a year or "Present" and no full stop.
_JOB_LINE = re.compile(r"(?:\b(?:19|20)\d{2}\b|\bPresent\b)[^.]*$")

# Resume sections whose lines are achievement bullets.
_BULLET_SECTIONS = ("experience", "employment", "work", "history", "project", "leadership", "volunteer")


# --- Text helpers ----------------------------------------------------------------

def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text)}


def _stems(text: str) -> set[str]:
    words = _normalize(text).split() + _canonical(_normalize(text)).split()
    return {_stem(w) for w in words}


def _content_words(text: str) -> list[str]:
    return [w for w in _normalize(text).split() if w not in _STOP and not w.isdigit() and len(w) > 1]


def _plain_case(word: str) -> bool:
    """lowercase, or capitalized only at the first letter ("Migrated")."""
    return word[1:].islower() or len(word) == 1


def _allowed_new_word(word: str) -> bool:
    w = word.lower()
    return (w in _STOP or w in _GENERIC_NOUNS or _stem(w) in _GENERIC_NOUNS or w in _ACTION_VERBS
            or (_plain_case(word) and w.endswith(("ed", "ing", "ly")) and w.replace("-", "").isalpha()))


def _technical(word: str, sentence_start: bool) -> bool:
    """Names a tool, product or organization: inner capitals (PostgreSQL),
    digits or symbols (S3, C++), or a capital letter mid-sentence (Kubernetes)."""
    if any(c.isdigit() or c in "+#" for c in word) or not _plain_case(word):
        return True
    return word[:1].isupper() and not sentence_start


def unsupported_words(text: str, source: str, original: str | None = None, technical_only: bool = False) -> list[str]:
    """Words in text that the sources don't support: what a rewrite or a
    letter sentence would add. In text order, each once, as written.

    source: where ordinary words may come from (the resume, plus the posting
        for a letter).
    original: for a rewrite, its bullet. Technical words (tools, products,
        organizations) must then come from the bullet itself, so a rewrite
        can't move a skill from another part of the resume into a bullet
        about different work.
    technical_only: check only technical words (used for cover letters,
        where ordinary phrasing is free but tools and names are not)."""
    known = _stems(source) | (_stems(original) if original else set())
    strict = _stems(original) if original is not None else known
    out: list[str] = []
    start = True
    for match in _WORD.finditer(text):
        raw = match.group(0)
        word = raw.strip("'-")
        sentence_start = start
        start = text[match.end():match.end() + 2].startswith((". ", "! ", "? "))
        norm = _normalize(word)
        if not norm:
            continue
        parts = norm.split()
        technical = _technical(word, sentence_start)
        if technical_only and not technical:
            continue
        pool = strict if technical else known
        if all(_stem(p) in pool or p in _STOP for p in parts) or (pool is known and _allowed_new_word(word)):
            continue
        if word not in out:
            out.append(word)
    return out


_ESCAPED = re.compile(r"\\u([0-9a-fA-F]{4})")


def _clean(text) -> str:
    """Model text, tidied: an escape the model wrote out literally ("\\u2019")
    becomes its character, so it isn't read as the number 2019."""
    return clean_generated(_ESCAPED.sub(lambda m: chr(int(m.group(1), 16)), str(text)).strip())


# --- Bullets ---------------------------------------------------------------------

def resume_bullets(resume_text: str, min_words: int = 6) -> list[str]:
    """Achievement lines from experience-like sections, with PDF line wraps
    joined ("...p99 latency" + "under 180 ms." is one bullet). Falls back to
    every long line when no such section heading is found."""
    lines = [ln.strip() for ln in resume_text.splitlines()]
    in_section, found_section = False, False
    joined: list[tuple[bool, str]] = []
    for line in lines:
        if not line:
            continue
        if _is_heading(line):
            in_section = any(k in line.lower() for k in _BULLET_SECTIONS)
            found_section = found_section or in_section
            joined.append((False, ""))   # a heading ends the previous bullet
            continue
        line = line.lstrip("-*•▪●– ").strip()
        prev_in, prev = joined[-1] if joined else (False, "")
        continues = prev and not prev.endswith((".", "!", "?")) and (line[:1].islower() or line[:1].isdigit())
        if continues and prev_in == in_section:
            joined[-1] = (in_section, f"{prev} {line}")
        else:
            joined.append((in_section, line))
    pool = [t for flag, t in joined if t and (flag or not found_section)]
    return [t for t in pool if len(t.split()) >= min_words and not _JOB_LINE.search(t)
            and len(re.sub(r"\([^)]*\)", " ", t).split()) >= 4]   # not "Ledger Kata (Java 17, Spring Boot)"


def pick_bullets(bullets: list[str], posting_text: str, limit: int = MAX_BULLETS) -> list[str]:
    """The bullets that share the most key terms with the posting, in resume order."""
    posting = _stems(posting_text)
    scored = [(sum(_stem(t) in posting for t in set(key_terms(b))), i) for i, b in enumerate(bullets)]
    chosen = sorted(i for _score, i in sorted(scored, key=lambda s: (-s[0], s[1]))[:limit])
    return [bullets[i] for i in chosen]


@dataclass
class Rewrite:
    original: str
    text: str
    requirement: str            # the posting requirement it speaks to, as the model named it
    issue: str | None = None    # why it was rejected (None = shown)


def check_rewrite(rw: Rewrite, resume_text: str) -> None:
    """Sets rw.issue when the rewrite adds a fact (in place)."""
    if _normalize(rw.text) == _normalize(rw.original):
        rw.issue = "no change from the original"
        return
    extra_numbers = _numbers(rw.text) - _numbers(rw.original)
    if extra_numbers:
        rw.issue = f"adds a number not in the original bullet ({', '.join(sorted(extra_numbers))})"
        return
    added = unsupported_words(rw.text, resume_text, rw.original)
    if added:
        on_resume = [w for w in added if _stems(w) <= _stems(resume_text)]
        where = "this bullet doesn't mention" if len(on_resume) == len(added) else "your resume doesn't mention"
        rw.issue = f"adds {', '.join(added[:3])}, which {where}"
        return
    original_words = set(_content_words(rw.original))
    kept = original_words & {_stem(w) for w in _content_words(rw.text)} | (original_words & set(_content_words(rw.text)))
    if original_words and len(kept) * 2 < len(original_words):
        rw.issue = "changes what the bullet says, not only how"


BULLETS_PROMPT = """You are helping a candidate adapt resume bullets to one job posting.

JOB POSTING: {title}{company}
Requirements from the posting:
{posting}

RESUME BULLETS:
{bullets}

Rewrite each bullet so it speaks to this posting. Rules:
- Keep every fact exactly as the bullet states it: the same tools, numbers, employers and scope.
- Never add a skill, tool, certificate, number or result the bullet does not state.
- You may start with a stronger verb, reorder the sentence, and use the posting's words for something the bullet already says.
- Do not add adjectives or a closing phrase about impact or quality ("ensuring reliability", "improving overall efficiency", "high-performance", "comprehensive").
- If the bullet already fits the posting, change it as little as possible.
- One sentence per bullet, at most 40 words.
For each rewrite give the bullet number and the posting requirement it speaks to.
"""


def bullets_schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {"rewrites": {"type": "array", "minItems": n, "maxItems": n, "items": {
            "type": "object",
            "properties": {"bullet": {"type": "integer", "minimum": 1, "maximum": n},
                           "text": {"type": "string", "maxLength": 320},
                           "requirement": {"type": "string", "maxLength": 150}},
            "required": ["bullet", "text", "requirement"]}}},
        "required": ["rewrites"],
    }


def parse_rewrites(raw: str, bullets: list[str], resume_text: str) -> list[Rewrite]:
    try:
        items = json.loads(raw)["rewrites"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Model response is not the expected JSON ({error})") from error
    out, seen = [], set()
    for item in items:
        try:
            n, text, req = int(item["bullet"]), _clean(item["text"]), str(item["requirement"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if not (1 <= n <= len(bullets)) or n in seen or not text:
            continue
        seen.add(n)
        rw = Rewrite(bullets[n - 1], text, req)
        check_rewrite(rw, resume_text)
        out.append(rw)
    return out


# --- Cover letter ------------------------------------------------------------------

@dataclass
class Evidence:
    requirement: str            # "" for a resume line used as-is (no posting requirement attached)
    resume_quote: str


@dataclass
class LetterSentence:
    text: str
    paragraph: int
    evidence: int               # 1-based index into the evidence list, 0 = none
    issue: str | None = None


def verify_evidence(items: list[Evidence], resume_text: str, posting_text: str) -> list[Evidence]:
    """Keeps the evidence items that still hold: the quote is in the resume
    and the requirement is in the posting. They come from the browser (the
    report it already has), so they are checked again, not trusted."""
    resume = f" {_normalize(resume_text)} "
    kept = []
    for e in items[:MAX_EVIDENCE * 2]:
        quote = _normalize(e.resume_quote)
        if not quote or f" {quote} " not in resume:
            continue
        terms = key_terms(e.requirement)
        if not e.requirement or (terms and _mentions(terms, posting_text)):
            kept.append(e)
    return kept[:MAX_EVIDENCE]


def check_sentence(s: LetterSentence, evidence: list[Evidence], resume_text: str, posting_text: str,
                   job_names: str = "") -> None:
    """Numbers may come from the resume or the posting. Tools, products,
    acronyms and organizations may come only from the resume, the job title
    or the company name: a letter must not claim a tool because the posting
    asks for it. Ordinary phrasing is not checked; the letter is a draft."""
    extra_numbers = _numbers(s.text) - _numbers(f"{resume_text}\n{posting_text}\n{job_names}")
    if extra_numbers:
        s.issue = f"states a number that is in neither your resume nor the posting ({', '.join(sorted(extra_numbers))})"
        return
    # A sentence about the job may name the posting's products ("the Clinical
    # Data Hub"); one about the candidate's experience may not ("my experience
    # with Snowflake" when the resume has no Snowflake).
    source = f"{resume_text}\n{job_names}" if _EXPERIENCE_CUE.search(s.text) else f"{resume_text}\n{job_names}\n{posting_text}"
    added = unsupported_words(s.text, source, technical_only=True)
    if added:
        s.issue = f"mentions {', '.join(added[:3])}, which your resume doesn't mention"
        return
    if s.evidence:
        item = evidence[s.evidence - 1]
        terms = key_terms(item.requirement) + key_terms(item.resume_quote)  # requirement may be empty
        if not (_stems(" ".join(terms)) & _stems(s.text)):
            s.issue = "cites a match it doesn't talk about"


LETTER_PROMPT = """Write a short cover letter from the candidate for this job.

JOB: {title}{company}
Requirements from the posting:
{posting}

EVIDENCE (numbered; a verified match shows the requirement and the resume text that meets it,
a resume line shows only the resume text):
{evidence}

CANDIDATE'S RESUME:
{resume}

Rules:
- Three short paragraphs: why this job, two or three matches, a brief close. At most 10 sentences.
- Every sentence about the candidate's experience must cite one evidence item by its number and say only what its resume text says.
- Never add a skill, tool, employer, number or result that is not in the resume.
- Plain, direct sentences. No exaggeration.
Give each sentence with its paragraph number and the match it cites (0 for none).
"""


_EXPERIENCE_CUE = re.compile(
    r"\b(experience|experienced|skilled|skills|proficient|proficiency|background|expertise|familiar|worked|"
    r"built|used|using|developed|managed|led|know|knowledge|years)\b", re.IGNORECASE)

MAX_OPENING = 2
MAX_CLOSING = 2


def arrange_letter(sentences: list[LetterSentence]) -> tuple[list[list[LetterSentence]], int]:
    """([opening, body, closing], left out). The 3B model ignores paragraph
    numbers and repeats itself, so code builds the letter: the first general
    sentences open it, every sentence that cites evidence forms the body,
    the last general sentences close it. Repeats (same first six words) and
    extra general sentences are left out and counted."""
    shown = [s for s in sentences if s.issue is None]
    seen, unique = set(), []
    for s in shown:
        key = " ".join(_normalize(s.text).split()[:6])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    general = [i for i, s in enumerate(unique) if not s.evidence]
    opening_idx = [i for i in general if not any(unique[j].evidence for j in range(i))][:MAX_OPENING]
    rest = [i for i in general if i not in opening_idx]
    closing_idx = rest[-MAX_CLOSING:] if rest else []
    body = [s for s in unique if s.evidence]
    paragraphs = [[unique[i] for i in opening_idx], body, [unique[i] for i in closing_idx]]
    used = len(opening_idx) + len(body) + len(closing_idx)
    return [p for p in paragraphs if p], len(shown) - used


def letter_schema(n_evidence: int) -> dict:
    return {
        "type": "object",
        "properties": {"sentences": {"type": "array", "minItems": 4, "maxItems": MAX_SENTENCES, "items": {
            "type": "object",
            "properties": {"text": {"type": "string", "maxLength": 300},
                           "paragraph": {"type": "integer", "minimum": 1, "maximum": 3},
                           "evidence": {"type": "integer", "minimum": 0, "maximum": n_evidence}},
            "required": ["text", "paragraph", "evidence"]}}},
        "required": ["sentences"],
    }


def parse_letter(raw: str, evidence: list[Evidence], resume_text: str, posting_text: str,
                 job_names: str = "") -> list[LetterSentence]:
    try:
        items = json.loads(raw)["sentences"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Model response is not the expected JSON ({error})") from error
    out = []
    for item in items:
        try:
            s = LetterSentence(_clean(item["text"]), int(item["paragraph"]), int(item["evidence"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not s.text:
            continue
        if not (0 <= s.evidence <= len(evidence)):
            s.evidence = 0
        check_sentence(s, evidence, resume_text, posting_text, job_names)
        out.append(s)
    return out


# --- Running ------------------------------------------------------------------------

@dataclass
class TailorResult:
    kind: str                                   # "bullets" or "letter"
    rewrites: list[Rewrite] = field(default_factory=list)
    sentences: list[LetterSentence] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    attempts: int = 1


def _posting_block(requirement_text: str, max_words: int = 260) -> str:
    words = requirement_text.split()
    return " ".join(words[:max_words])


def _generate(client: OllamaClient, prompt: str, schema: dict, parse, max_tokens: int, attempts: int = 2):
    last: ValueError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return parse(client.generate(prompt, temperature=0.2, max_tokens=max_tokens, schema=schema)), attempt
        except ValueError as error:
            last = error
    assert last is not None
    raise last


def suggest_rewrites(resume_text: str, title: str, company: str | None, requirement_text: str,
                     client: OllamaClient) -> TailorResult:
    bullets = pick_bullets(resume_bullets(resume_text), requirement_text)
    if not bullets:
        return TailorResult("bullets")
    prompt = BULLETS_PROMPT.format(
        title=title, company=f" at {company}" if company else "", posting=_posting_block(requirement_text),
        bullets="\n".join(f"[{i}] {b}" for i, b in enumerate(bullets, 1)))
    rewrites, attempts = _generate(client, prompt, bullets_schema(len(bullets)),
                                   lambda raw: parse_rewrites(raw, bullets, resume_text), max_tokens=900)
    return TailorResult("bullets", rewrites=rewrites, attempts=attempts)


def draft_letter(resume_text: str, title: str, company: str | None, requirement_text: str,
                 evidence: list[Evidence], client: OllamaClient) -> TailorResult:
    """Evidence: the verified matches for this posting, then the resume's own
    bullets closest to it, so a posting with few verified matches still gets
    a letter built only from things the resume says."""
    matches = verify_evidence(evidence, resume_text, requirement_text)
    quoted = {_normalize(e.resume_quote) for e in matches}
    lines = [Evidence("", b) for b in pick_bullets(resume_bullets(resume_text), requirement_text, limit=4)
             if not any(q in _normalize(b) for q in quoted)]
    evidence = (matches + lines)[:MAX_EVIDENCE]
    if not evidence:
        return TailorResult("letter")
    prompt = LETTER_PROMPT.format(
        title=title, company=f" at {company}" if company else "", posting=_posting_block(requirement_text),
        evidence="\n".join(f"[{i}] {e.requirement}: \"{e.resume_quote}\"" if e.requirement else f"[{i}] resume line: \"{e.resume_quote}\""
                            for i, e in enumerate(evidence, 1)),
        resume=resume_text[:6000])
    sentences, attempts = _generate(client, prompt, letter_schema(len(evidence)),
                                    lambda raw: parse_letter(raw, evidence, resume_text, requirement_text,
                                                             f"{title} {company or ''}"),
                                    max_tokens=900)
    return TailorResult("letter", sentences=sentences, evidence=evidence, attempts=attempts)
