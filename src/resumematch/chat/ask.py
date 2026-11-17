"""Ask your report: questions answered from the resume and the report's
postings, with every sentence checked against what it cites.

Retrieval-augmented generation in its plain form, with the same rule as
the rest of the system: the model writes, code decides what is shown.

  1. Pieces. Always: each report posting's checked requirements (what the
     report verified, with resume quotes) and its facts line (location,
     work type, salary). Ranked by the question: the resume's lines (PDF
     wraps joined) and short passages of each posting. For questions that
     reach past the report ("other postings", "in general"): passages from
     a fresh search of the index.
  2. Ranking. The question and the ranked pieces are embedded with the
     search model; the closest 8 (and up to 3 from the search) are sent with
     the always-included pieces, the question and the previous turn.
  3. Routing. Questions the report's data answers exactly are answered by
     code, not the model: fit and closest-match questions from the verified
     checks, and salary, remote, location, work type and level questions
     from the facts. A question far from every piece, with none of its key
     words in them, is declined without calling the model.
  4. Answer (everything else). JSON schema: a few sentences, each citing
     piece numbers.
  5. Checks, per sentence: it cites at least one piece; its numbers are in
     what it cites (or the question); its tool, product and organization
     names are in what it cites, the question or the report's posting names;
     it shares a content word with what it cites; and if it says what the
     candidate has, the words it took from a posting are on the resume.
     Failing sentences are removed and listed.

If nothing passes, the reply is a plain "couldn't find that", never a guess.
Not caught: a sentence built only from resume words that still misreads
them ("You have the CPA certification" for "CPA candidate"). Sources are
shown under every answer for that reason.
"""

import json
import re
from dataclasses import dataclass, field

from resumematch.generation.gap_analysis import _normalize
from resumematch.generation.ollama_client import OllamaClient
from resumematch.generation.tailor import (
    _STOP,
    _clean,
    _content_words,
    _numbers,
    _stems,
    unsupported_words,
)
from resumematch.preprocess.boilerplate import split_sentences

MAX_REPORT_PIECES = 8
MAX_SEARCH_PIECES = 3
MAX_PIECE_WORDS = 60
MAX_HISTORY = 1          # earlier answers were echoed into new ones with 3 turns
# Best question-piece similarity below this: not about the resume or the postings.
# Measured on the nurse report: 6 on-topic questions 0.644 to 0.781, 5 off-topic
# ones ("Tell me a joke", "What is the capital of France?") 0.492 to 0.581.
MIN_RELEVANCE = 0.62
OFF_TOPIC = "That doesn't seem to be about your resume or these postings. Ask about a requirement, a posting or your experience."
MAX_QUESTION_CHARS = 400

NO_ANSWER = "I couldn't find that in your resume or these postings."

# Questions that reach past the report's postings.
_WIDER = re.compile(r"\b(other|others|in general|generally|most|market|typical|usually|commonly|elsewhere|"
                    r"companies|employers|industry|across)\b", re.IGNORECASE)


@dataclass
class Piece:
    id: int
    kind: str            # "resume" | "posting" | "check" | "facts" | "search"
    label: str           # "Your resume", "Posting 2", "Other posting (...)"
    text: str
    url: str | None = None
    name: str = ""       # the posting's title and company, which a sentence may name


@dataclass
class Sentence:
    text: str
    sources: list[int]
    issue: str | None = None


@dataclass
class ChatAnswer:
    sentences: list[Sentence]
    pieces: list[Piece]
    no_answer: bool = False
    attempts: int = 1
    by_code: bool = False     # answered from the report's data, not by the model

    @property
    def shown(self) -> list[Sentence]:
        return [s for s in self.sentences if s.issue is None]


# --- Pieces ----------------------------------------------------------------

def _trim(text: str, max_words: int = MAX_PIECE_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")


def resume_pieces(resume_text: str) -> list[str]:
    """Resume lines with PDF wraps joined; lines under 3 words dropped."""
    out: list[str] = []
    for line in (ln.strip().lstrip("-*•▪● ").strip() for ln in resume_text.splitlines()):
        if not line:
            continue
        if out and not out[-1].endswith((".", "!", "?", ":")) and (line[:1].islower() or line[:1].isdigit()):
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return [_trim(p) for p in out if len(p.split()) >= 3]


def posting_pieces(text: str) -> list[str]:
    """Passages of about two sentences, at most MAX_PIECE_WORDS words."""
    out, current = [], []
    for sentence in split_sentences(text):
        if current and len(" ".join(current + [sentence]).split()) > MAX_PIECE_WORDS:
            out.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        out.append(" ".join(current))
    return [_trim(p) for p in out if len(p.split()) >= 3]


def facts_piece(details: dict | None) -> str | None:
    """Every field stated, including absences ("salary: not listed"), so
    "which jobs list a salary?" can be answered from it."""
    if not details:
        return None
    return (f"location: {details.get('location') or 'not listed'}; work type: {details.get('work_type') or 'not listed'}; "
            f"level: {details.get('level') or 'not listed'}; remote work: {'allowed' if details.get('remote') else 'not mentioned'}; "
            f"salary: {details.get('salary') or 'not listed'}.")


@dataclass
class ReportPosting:
    job_id: str
    title: str
    company: str | None
    url: str | None
    text: str                     # the posting's requirement sentences (job_report.requirement_excerpt)
    details: dict | None = None
    closest: bool = False         # the report's closest match
    # The report's checked requirements: (requirement, status, resume quote).
    checks: list[tuple[str, str, str | None]] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.title} at {self.company}" if self.company else self.title


def checks_piece(checks: list[tuple[str, str, str | None]]) -> str | None:
    """The report's verdict for one posting, as one piece."""
    if not checks:
        return None
    shown = [f'{r} (your resume: "{_trim(q, 12)}")' if q else r for r, st, q in checks if st == "shown"]
    partly = [r for r, st, _q in checks if st == "partly"]
    missing = [r for r, st, _q in checks if st == "not_found"]
    parts = []
    if shown:
        parts.append("shown on your resume: " + "; ".join(shown))
    if partly:
        parts.append("partly shown: " + "; ".join(partly))
    if missing:
        parts.append("not found on your resume: " + "; ".join(missing))
    return _trim(". ".join(parts) + ".", 90)


def wants_wider_search(question: str) -> bool:
    return bool(_WIDER.search(question))


def gather_pieces(question: str, resume_text: str, postings: list[ReportPosting], embed,
                  search=None) -> tuple[list[Piece], float]:
    """The pieces most related to the question, numbered from 1.
    embed(list[str]) -> vectors (normalized); search(vector) -> chunk hits
    (VectorStore.query shape), used only for wider questions."""
    fixed: list[tuple] = []        # (kind, label, text, url, name): always sent
    candidates: list[tuple] = []   # ranked by the question
    candidates += [("resume", "Your resume", p, None, "") for p in resume_pieces(resume_text)]
    for n, post in enumerate(postings, 1):
        tag = f"Posting {n}"                # the prompt's legend says which posting is which
        check = checks_piece(post.checks)
        if check:
            fixed.append(("check", tag, check, post.url, post.label))
        facts = facts_piece(post.details)
        if facts:
            fixed.append(("facts", tag, facts, post.url, post.label))
        candidates += [("posting", tag, p, post.url, post.label) for p in posting_pieces(post.text)]
    if not candidates and not fixed:
        return [], 0.0
    vectors = embed([question] + [c[2] for c in candidates])
    q, rest = vectors[0], vectors[1:]
    scores = [sum(a * b for a, b in zip(q, v, strict=False)) for v in rest]
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])[:MAX_REPORT_PIECES]
    chosen = fixed + [candidates[i] for i in sorted(order)]   # report checks and facts first, then the rest in order
    if search is not None and wants_wider_search(question):
        known = {p.job_id for p in postings}
        seen: set[str] = set()
        for hit in search(q):
            if hit["job_id"] in known or hit["job_id"] in seen:
                continue
            seen.add(hit["job_id"])
            name = f"{hit['title']} at {hit['company_name']}" if hit.get("company_name") else hit["title"]
            chosen.append(("search", f"Other posting ({name})", _trim(hit["text"]), hit.get("job_posting_url"), name))
            if len(seen) == MAX_SEARCH_PIECES:
                break
    best = max(scores, default=0.0)
    # A question whose own key words are in the resume or postings is on topic even
    # when its embedding is far ("Which posting asks for SAP?" scored below the cutoff).
    q_terms = {t for t in _content_words(question) if len(t) > 2}
    corpus = _stems(" ".join(c[2] for c in candidates + fixed))
    if q_terms and any(_stems(t) & corpus for t in q_terms):
        best = max(best, MIN_RELEVANCE)
    return [Piece(i, kind, label, text, url, name) for i, (kind, label, text, url, name) in enumerate(chosen, 1)], best


# --- Prompt and checks --------------------------------------------------------

PROMPT = """You answer a job seeker's questions using only the numbered pieces below: lines from their resume
and passages from job postings.

QUESTION: {question}

POSTINGS IN THE REPORT:
{legend}

PIECES:
{pieces}
{history}
QUESTION AGAIN: {question}

Rules:
- Answer in 1 to 4 short sentences in your own words; don't copy pieces whole. Give the piece numbers each
  sentence uses in "sources", not in the sentence.
- A "report check" piece is what was verified against the resume; prefer it for questions about fit.
- Use only what the pieces say. Never add a skill, tool, number, employer or fact that isn't in a piece.
- Speak to the job seeker as "you".
- If the pieces don't answer the question, set "not_in_pieces" to true and say so in one sentence.
"""


def schema(n: int) -> dict:
    return {
        "type": "object",
        # Answer before the flag: with the flag first, the model decided to give up
        # before writing anything.
        "properties": {
            "sentences": {"type": "array", "minItems": 1, "maxItems": 5, "items": {
                "type": "object",
                "properties": {"text": {"type": "string", "maxLength": 300},
                               "sources": {"type": "array", "maxItems": 4,
                                           "items": {"type": "integer", "minimum": 1, "maximum": max(n, 1)}}},
                "required": ["text", "sources"]}},
            "not_in_pieces": {"type": "boolean"},
        },
        "required": ["sentences", "not_in_pieces"],
    }


# "(piece 4)", "[2]", "in pieces 6 and 7": citations the model writes into the text.
_CITATION = re.compile(
    r"\s*\(?\b(?:in |see |from )?(?:pieces?|postings?|excerpts?)\s*\[?\d+\]?(?:\s*(?:,|and)\s*\[?\d+\]?)*\)?"
    r"|\s*\[\d+(?:\s*,\s*\d+)*\]", re.IGNORECASE)


_POSTING_REF = re.compile(r"\b[Pp]ostings?\s+(\d+)((?:\s*(?:,|and)\s*\d+)*)")


def name_postings(text: str, names: dict[int, str]) -> str:
    """"Postings 1 and 2 mention" -> "Staff Accountant at IFG Chicago and Accountant at Rely mention".
    Deleting the reference left sentences starting "and mention"."""
    def sub(m):
        nums = [int(m.group(1))] + [int(n) for n in re.findall(r"\d+", m.group(2) or "")]
        known = [names[n] for n in nums if n in names]
        return " and ".join(known) if known else m.group(0)
    return _POSTING_REF.sub(sub, text)


# The piece header copied into an answer: "(Staff Accountant at IFG Chicago, report check)".
_COPIED_HEADER = re.compile(r"\s*\([^()]*,\s*(?:report check|facts)\)", re.IGNORECASE)


def strip_citations(text: str) -> str:
    out = _CITATION.sub("", text).strip()
    out = re.sub(r"\s+([.,;!?])", r"\1", out)
    return out


# A sentence that says what the candidate has done or has. "You do not have" is
# about an absence and is not checked against the resume.
_CLAIM = re.compile(
    r"\byou(?:'ve| have| had| hold| bring| worked| led| managed| used| are (?:experienced|skilled|certified|licensed|trained|familiar))\b"
    r"|\byour (?:experience|background|skills?|training|resume (?:shows|lists|mentions|includes))\b", re.IGNORECASE)
_NEGATED = re.compile(r"\b(?:not|n't|no|never|lack|without)\b", re.IGNORECASE)


def claims_about_candidate(text: str) -> bool:
    m = _CLAIM.search(text)
    return bool(m) and not _NEGATED.search(text[max(0, m.start() - 10):m.end() + 25])


def check_sentence(s: Sentence, pieces: dict[int, Piece], question: str, resume_text: str = "",
                   posting_names: str = "") -> None:
    cited = [pieces[i] for i in s.sources if i in pieces]
    if not cited:
        s.issue = "cites nothing"
        return
    source = " ".join(f"{p.text} {p.label} {p.name}" for p in cited)
    extra = _numbers(s.text) - _numbers(source) - _numbers(question) - _numbers(posting_names)
    if extra:
        s.issue = f"states a number not in what it cites ({', '.join(sorted(extra))})"
        return
    added = unsupported_words(s.text, f"{source}\n{question}\n{posting_names}", technical_only=True)
    if added:
        s.issue = f"mentions {', '.join(added[:3])}, which isn't in what it cites"
        return
    words = {w for w in _content_words(s.text) if len(w) > 2}
    if words and not ({_s for w in words for _s in _stems(w)} & _stems(source)):
        s.issue = "doesn't use what it cites"
        return
    if resume_text and claims_about_candidate(s.text):
        # Measured failure: "You have experience in a Level II Trauma center" and
        # "...a teaching hospital", from a posting and from the report's "not found"
        # list, for a resume that says "Level I trauma center".
        posting_words = set(_normalize(" ".join(p.text for p in cited if p.kind != "resume")).split())
        resume = _stems(resume_text)
        borrowed = [w for w in _normalize(s.text).split()
                    if w not in _STOP and w not in ("resume", "report") and w in posting_words and not (_stems(w) & resume)]
        if borrowed:
            s.issue = f"says you have {', '.join(dict.fromkeys(borrowed))}, which your resume doesn't show"


def _history_block(history: list[tuple[str, str]]) -> str:
    if not history:
        return ""
    turns = "\n".join(f"Q: {q[:MAX_QUESTION_CHARS]}\nA: {a[:600]}" for q, a in history[-MAX_HISTORY:])
    return f"\nEARLIER IN THIS CONVERSATION:\n{turns}\n"


def parse(raw: str, pieces: list[Piece], question: str, resume_text: str = "",
          names: dict[int, str] | None = None) -> tuple[list[Sentence], bool]:
    try:
        data = json.loads(raw)
        items, flag = data["sentences"], bool(data.get("not_in_pieces", False))
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Model response is not the expected JSON ({error})") from error
    by_id = {p.id: p for p in pieces}
    out, seen = [], set()
    for item in items:
        try:
            text = strip_citations(name_postings(_clean(item["text"]), names or {})).lstrip(":;,- ").strip()
            text = _COPIED_HEADER.sub("", text).strip()
            text = text[:1].upper() + text[1:]
            sources = sorted({int(i) for i in item.get("sources", [])})
        except (KeyError, TypeError, ValueError):
            continue
        key = " ".join(text.lower().split())
        if not text or key in seen or "_" in text and " " not in text:   # repeats, "not_in_pieces"
            continue
        seen.add(key)
        s = Sentence(text, sources)
        check_sentence(s, by_id, question, resume_text, " ".join((names or {}).values()))
        out.append(s)
    return out, flag


_KIND_NAMES = {"resume": "your resume", "posting": "{label}", "check": "{label}, report check",
               "facts": "{label}, facts", "search": "{label}"}


def legend(postings: list[ReportPosting]) -> str:
    return "\n".join(f"Posting {n}{' (your closest match)' if p.closest else ''}: {p.label}"
                     for n, p in enumerate(postings, 1)) or "(none)"


# Questions the report's data answers exactly; code answers them, not the model.
# Measured: asked "Which of these jobs list a salary?", the model listed jobs
# whose facts said "salary: not listed"; asked why the closest match fits, it
# wrote only claims the resume doesn't support.
_FACTS_Q = re.compile(r"\b(salary|salaries|pay|paid|compensation|wage|remote|on.?site|hybrid|location|located|where|"
                      r"full.?time|part.?time|contract|work type|level|seniority|senior|entry)\b", re.IGNORECASE)
_FIT_Q = re.compile(r"\b(fit|closest|best match|good match|why .*match|match me|suited)\b", re.IGNORECASE)


def _piece_id(pieces: list[Piece], kind: str, tag: str) -> list[int]:
    return [p.id for p in pieces if p.kind == kind and p.label == tag]


def facts_answer(postings: list[ReportPosting], pieces: list[Piece]) -> list[Sentence]:
    out = []
    for n, post in enumerate(postings, 1):
        d = post.details or {}
        if not d:
            continue
        bits = [d.get("location") or "location not listed", d.get("work_type") or "work type not listed",
                d.get("level") or "level not listed",
                "remote work allowed" if d.get("remote") else "remote work not mentioned",
                f"salary {d['salary']}" if d.get("salary") else "salary not listed"]
        out.append(Sentence(f"{post.label}: {', '.join(bits)}.", _piece_id(pieces, "facts", f"Posting {n}")))
    return out


def fit_answer(postings: list[ReportPosting], pieces: list[Piece]) -> list[Sentence]:
    n, post = next(((i, p) for i, p in enumerate(postings, 1) if p.closest), (1, postings[0]) if postings else (0, None))
    if post is None:
        return []
    shown = [(r, q) for r, st, q in post.checks if st == "shown"]
    missing = [r for r, st, _q in post.checks if st != "shown"]
    src = _piece_id(pieces, "check", f"Posting {n}")
    out = [Sentence(f"Your closest match is {post.label}.", src)]
    if shown:
        out.append(Sentence("Your resume shows " + "; ".join(f'{r} ("{_trim(q, 14)}")' if q else r for r, q in shown) + ".", src))
    if missing:
        out.append(Sentence("It doesn't yet show " + "; ".join(missing) + ".", src))
    if not post.checks:
        out.append(Sentence("No requirement of this posting could be checked against your resume.", src))
    return out


def answer(question: str, resume_text: str, postings: list[ReportPosting], client: OllamaClient, embed,
           search=None, history: list[tuple[str, str]] | None = None) -> ChatAnswer:
    question = " ".join(question.split())[:MAX_QUESTION_CHARS]
    pieces, relevance = gather_pieces(question, resume_text, postings, embed, search)
    if not pieces:
        return ChatAnswer([], [], no_answer=True)
    if postings and _FIT_Q.search(question):
        return ChatAnswer(fit_answer(postings, pieces), pieces, by_code=True)
    if postings and _FACTS_Q.search(question) and not wants_wider_search(question):
        sentences = facts_answer(postings, pieces)
        if sentences:
            return ChatAnswer(sentences, pieces, by_code=True)
    if relevance < MIN_RELEVANCE:
        return ChatAnswer([Sentence(OFF_TOPIC, [])], pieces, by_code=True, no_answer=True)
    prompt = PROMPT.format(
        legend=legend(postings),
        pieces="\n".join(f"[{p.id}] ({_KIND_NAMES[p.kind].format(label=p.label)}) {p.text}" for p in pieces),
        history=_history_block(history or []), question=question)
    last: ValueError | None = None
    for attempt in (1, 2):
        try:
            raw = client.generate(prompt, temperature=0.2, max_tokens=500, schema=schema(len(pieces)))
            names = {n: p.label for n, p in enumerate(postings, 1)}
            sentences, flag = parse(raw, pieces, question, resume_text, names)
            shown = [s for s in sentences if s.issue is None]
            # The flag alone doesn't hide sentences that passed every check.
            return ChatAnswer(sentences, pieces, no_answer=not shown, attempts=attempt)
        except ValueError as error:
            last = error
    assert last is not None
    raise last

