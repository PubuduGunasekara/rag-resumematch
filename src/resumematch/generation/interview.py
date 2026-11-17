"""Interview prep for one posting: likely questions, and what on the resume
to draw on for each.

Split of work, as elsewhere: the model only phrases questions. What to
draw on comes from code: the verified resume line for a requirement the
resume shows, the words it does show for a partly shown one, and an
honest-answer hint for a gap. Nothing the candidate is told to say is
generated.

Checks on each question: it ends with "?", mentions its requirement, and
names no tool or organization that is in neither the posting nor the
resume. A requirement whose question fails or is missing gets a plain
template question, so every requirement is covered whatever the model does.
"""

import json
from dataclasses import dataclass, field

from resumematch.generation.gap_analysis import _found, _mentions, _normalize, key_terms
from resumematch.generation.job_report import is_soft
from resumematch.generation.ollama_client import OllamaClient
from resumematch.generation.tailor import _GENERIC_NOUNS, _clean, _stems, unsupported_words
from resumematch.generation.tailor import _STOP as _STOP_WORDS

MAX_REQUIREMENTS = 10
MAX_GENERAL = 2

_ORDINARY_STARTS = {"cloud", "strong", "excellent", "proven", "working", "advanced", "basic", "solid", "experience",
                    "knowledge", "ability", "proficiency", "familiarity", "understanding", "demonstrated", "current"}

# Too general to be worth pointing at ("your resume mentions data").
_TOO_GENERAL = _GENERIC_NOUNS | {"data", "information", "technical", "technology", "technologies", "skill", "tool",
                                 "database", "databases", "framework", "frameworks", "application", "applications",
                                 "platform", "platforms", "software", "development", "environment"}


@dataclass
class PrepItem:
    requirement: str
    status: str                       # "shown" | "partly" | "not_found" | "soft" | "general"
    resume_quote: str | None = None
    partly: list[str] = field(default_factory=list)
    credential: bool = False
    question: str = ""
    generated: bool = True            # False: the model's question failed a check, a template was used
    issue: str | None = None          # why the model's question was not used

    @property
    def hint(self) -> str:
        """What to draw on, worded here so every interface says the same."""
        if self.status == "shown":
            quote = self.resume_quote or ""
            quote = quote if len(quote) <= 180 else quote[:177].rsplit(" ", 1)[0] + "..."
            return f'Draw on: "{quote}". Have one specific example ready: what you did, how, and the result.'
        if self.status == "partly":
            return (f"Your resume mentions {', '.join(self.partly)}, not all of it. Start with that, then say how "
                    f"you would close the gap.")
        if self.status == "soft":
            return "Prepare a short example from your own experience."
        if self.status == "general":
            return "About the role and employer: read the posting again before the interview."
        if self.credential:
            return "This is a credential. Say plainly whether you have it or are working toward it."
        # The check compares wording, so "not found" may be a synonym the resume
        # uses ("Docker, Kubernetes" for "containerization").
        return ("Your resume doesn't use these words. If you have this under another name, say so with an "
                "example; if not, answer honestly and mention related experience only if you have it.")


@dataclass
class ReportRequirement:
    """One requirement as the browser's report has it; checked again here."""
    text: str
    resume_quote: str | None = None
    credential: bool = False


def prepare_items(requirements: list[ReportRequirement], resume_text: str, posting_text: str) -> list[PrepItem]:
    """Requirements that are really in the posting, each with a status
    recomputed from the resume (the browser's copy is not trusted):
    "shown" needs a quote that is in the resume and mentions the requirement."""
    resume = f" {_normalize(resume_text)} "
    items: list[PrepItem] = []
    seen = set()
    for r in requirements:
        terms = key_terms(r.text)
        key = " ".join(terms)
        if not terms or key in seen or not _mentions(terms, posting_text):
            continue
        seen.add(key)
        quote = (r.resume_quote or "").strip()
        if quote and f" {_normalize(quote)} " in resume and _found(terms, quote):
            items.append(PrepItem(r.text, "shown", resume_quote=quote))
        elif is_soft(r.text):
            items.append(PrepItem(r.text, "soft"))
        else:
            # Any matching word counts here (the report needs two): "Docker and
            # microservices" with Docker on the resume should say so, not
            # "your resume doesn't show this".
            partly = [t for t in _found(terms, resume_text) if t not in _TOO_GENERAL and len(t) > 2]
            items.append(PrepItem(r.text, "partly" if partly else "not_found", partly=partly, credential=r.credential))
    # Gaps first: those are the questions to rehearse; then shown, then traits.
    order = {"not_found": 0, "partly": 1, "shown": 2, "soft": 3}
    return sorted(items, key=lambda i: order[i.status])[:MAX_REQUIREMENTS]


def template_question(item: PrepItem) -> str:
    text = item.requirement.rstrip(".").strip()
    first = text.split()[0].lower() if text.split() else ""
    # Lowercase only an ordinary first word ("Cloud technologies"), never a name ("Kafka").
    ordinary = first in _TOO_GENERAL or first in _ORDINARY_STARTS or first in _STOP_WORDS
    lowered = text[:1].lower() + text[1:] if ordinary and text[1:2].islower() else text
    if item.status == "soft":
        return f"Can you give an example of a time you showed {lowered}?"
    if "experience" in lowered.lower():          # "ECMO experience" -> "your ECMO experience"
        return f"Can you tell me about your {lowered}?"
    return f"Can you tell me about your experience with {lowered}?"


PROMPT = """You are preparing a candidate for a job interview.

JOB: {title}{company}
Requirements from the posting:
{posting}

For each numbered requirement below, write the question an interviewer for this job would most likely
ask about it. Then write {general} general questions about the role itself (requirement number 0).

{items}

Rules:
- One question each, ending with a question mark, at most 30 words.
- A question about a requirement must name that requirement.
- Do not mention tools, products or companies that are not in the posting.
"""


def schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {"questions": {"type": "array", "minItems": n, "maxItems": n + MAX_GENERAL, "items": {
            "type": "object",
            "properties": {"requirement": {"type": "integer", "minimum": 0, "maximum": n},
                           "question": {"type": "string", "maxLength": 240}},
            "required": ["requirement", "question"]}}},
        "required": ["questions"],
    }


def check_question(question: str, item: PrepItem | None, source: str) -> str | None:
    """Why the question can't be used, or None."""
    if not question.endswith("?"):
        return "not a question"
    added = unsupported_words(question, source, technical_only=True)
    if added:
        return f"mentions {', '.join(added[:3])}, which is in neither the posting nor your resume"
    if item is not None and not (_stems(" ".join(key_terms(item.requirement))) & _stems(question)):
        return "doesn't ask about this requirement"
    return None


def parse(raw: str, items: list[PrepItem], source: str) -> list[PrepItem]:
    """Fills each item's question (model or template) and returns the
    general questions that passed."""
    try:
        questions = json.loads(raw)["questions"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Model response is not the expected JSON ({error})") from error
    general: list[PrepItem] = []
    asked = set()
    for q in questions:
        try:
            n, text = int(q["requirement"]), _clean(q["question"])
        except (KeyError, TypeError, ValueError):
            continue
        key = " ".join(_normalize(text).split()[:8])
        if not text or key in asked:
            continue
        if n == 0:
            if len(general) < MAX_GENERAL and check_question(text, None, source) is None:
                asked.add(key)
                general.append(PrepItem("About the role", "general", question=text))
            continue
        if not (1 <= n <= len(items)) or items[n - 1].question:
            continue
        item = items[n - 1]
        issue = check_question(text, item, source)
        if issue is None:
            asked.add(key)
            item.question = text
        else:
            item.issue = issue
    for item in items:
        if not item.question:
            item.question, item.generated = template_question(item), False
    return general


@dataclass
class InterviewPrep:
    items: list[PrepItem]
    general: list[PrepItem]
    attempts: int = 1


def prepare(resume_text: str, title: str, company: str | None, requirement_text: str,
            requirements: list[ReportRequirement], client: OllamaClient) -> InterviewPrep:
    items = prepare_items(requirements, resume_text, requirement_text)
    if not items:
        return InterviewPrep([], [])
    source = f"{requirement_text}\n{resume_text}\n{title} {company or ''}"
    prompt = PROMPT.format(
        title=title, company=f" at {company}" if company else "", general=MAX_GENERAL,
        posting=" ".join(requirement_text.split()[:260]),
        items="\n".join(f"[{i}] {it.requirement}" for i, it in enumerate(items, 1)))
    for attempt in (1, 2):
        try:
            raw = client.generate(prompt, temperature=0.3, max_tokens=900, schema=schema(len(items)))
            general = parse(raw, items, source)
            return InterviewPrep(items, general, attempt)
        except ValueError:
            for item in items:          # a failed parse must not leave half-filled items
                item.question, item.generated, item.issue = "", True, None
    # The model failed twice: templates still cover every requirement.
    for item in items:
        item.question, item.generated, item.issue = template_question(item), False, "the model's answer couldn't be used"
    return InterviewPrep(items, [], 2)

