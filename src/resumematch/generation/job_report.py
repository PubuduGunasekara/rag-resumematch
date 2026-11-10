"""Per-job report: one checked analysis per retrieved posting, sorted into
best fit, potential fits, and not a match.

Everything that decides a tier or a score is computed here in code from
claims that passed check_claims. The LLM only proposes requirements; it
never rates the candidate. A score is "6 of 8 requirements we could
verify", which a user (or a grader) can audit line by line.

UI-agnostic on purpose: Streamlit and any other front end render the
same JobReport.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from resumematch.generation.gap_analysis import _BENEFIT, CitedClaim, GapAnalysisResult, _found, _mentions, _stem, key_terms
from resumematch.generation.text_clean import clean_generated
from resumematch.preprocess.boilerplate import _REQUIREMENT_CUE, split_sentences

# Read from the sentence of the posting that states the requirement.
_PREFERRED = re.compile(
    r"\b(preferred|preferably|a plus|is a plus|nice to have|desired|desirable|bonus|ideally|"
    r"advantageous|an advantage|familiarity)\b", re.IGNORECASE)
_REQUIRED = re.compile(
    r"\b(required|requires|require|must|minimum|at least|mandatory|essential|need to|needs to|"
    r"qualifications?)\b", re.IGNORECASE)

# Missing one of these can't be fixed by rewording a resume.
_CREDENTIAL = re.compile(
    r"\b(degree|bachelor'?s?|master'?s?|msn|bsn|mba|phd|doctorate|licen[sc]e|licensure|licensed|"
    r"nurse practitioners?|np|aprn|cpa|board certified|board certification|security clearance|clearance|"
    r"accredited (educational |education |training )?program)\b",
    re.IGNORECASE)


# Requirements no resume line can confirm or refute ("Expert clinical
# judgment", "Full command of the English language"). Shown apart and
# never counted: a word check would always call them "not found".
_SOFT = set("""
judgment judgement communication communicate interpersonal initiative teamwork collaboration
collaborative organizational organization organized english verbal written problem solving
critical thinking self directed directive motivated detail oriented adaptable flexible
learning learner professional professionalism attitude passion passionate customer focus
innovation integrity reliable dependable multitask multitasking prioritize priorities time
management physical lift stand computer proficiency computerized data information education
training ongoing patient centered care model work ethic team player demonstrate effective
expert be full command strong excellent
""".split())


def is_soft(requirement: str) -> bool:
    terms = key_terms(requirement)
    return bool(terms) and sum(t in _SOFT for t in terms) * 3 >= len(terms) * 2  # two thirds soft


@dataclass
class Requirement:
    text: str                    # cleaned model wording, e.g. "Cardiogenic shock experience"
    met: bool
    importance: str              # "required" | "preferred" | "unspecified"
    credential: bool             # a degree/license: can't be added by editing the resume
    resume_quote: str | None     # exact resume text that shows it (met only)
    posting_sentence: str | None  # the posting sentence that asks for it
    soft: bool = False           # general trait, shown apart, never counted
    partly: list[str] = field(default_factory=list)  # for a gap: the words of it the resume DOES show

    @property
    def status(self) -> str:
        """"shown" | "partly" | "not_found". Never "missing": a word check
        can confirm something is on a resume, not prove it's absent."""
        return "shown" if self.met else ("partly" if self.partly else "not_found")


@dataclass
class JobAnalysis:
    rank: int                    # retrieval rank, 1 = closest
    job_id: str
    title: str
    company: str | None
    url: str | None
    details: dict                # location, work_type, level, remote, salary (may be empty)
    requirements: list[Requirement]
    tier: str = "potential"      # "best" | "fit" | "potential" | "no"
    reason: str = ""             # one plain sentence shown on the card
    rejected: int = 0            # model claims that failed verification

    fusion_score: float = 0.0    # retrieval similarity (RRF over resume sections)

    @property
    def counted(self) -> list[Requirement]:
        return [r for r in self.requirements if not r.soft]

    @property
    def met(self) -> int:
        return sum(r.met for r in self.counted)

    @property
    def checked(self) -> int:
        return len(self.counted)

    @property
    def gaps(self) -> list[Requirement]:
        return [r for r in self.counted if not r.met]


@dataclass
class LearnItem:
    text: str
    jobs: int                    # how many of the analyzed jobs ask for it
    credential: bool


@dataclass
class JobReport:
    jobs: list[JobAnalysis]
    learn_first: list[LearnItem]
    headline: str
    summary: str
    best: JobAnalysis | None = field(default=None)
    analysis: GapAnalysisResult | None = None  # the checked LLM result, for a "how this was checked" panel


# Sentences about the employer, not the candidate. Dropped before the
# model sees a posting: with whole postings, llama3.2:3b listed "Child
# care", "Student loan subsidies" and "Supportive work environment" as
# missing requirements.
_ABOUT_EMPLOYER = re.compile(
    r"\b(we offer|we provide|our (mission|values|culture|team|company|people)|about (us|the company)|"
    r"join (us|our)|we are (proud|committed|an?)|why (work|join)|what we offer|you.ll (enjoy|receive|get))\b",
    re.IGNORECASE)


def requirement_excerpt(chunks: list[dict], max_words: int = 260) -> list[dict]:
    """One posting's chunks reduced to the sentences that could state a
    requirement, as a single excerpt (same citation fields).

    Employer sentences (perks, mission, "we offer") are dropped; then,
    if it's still long, sentences with a requirement cue ("must",
    "required", "preferred", "experience", "degree") are kept first, up
    to max_words, in document order. Five of these fit one prompt.
    Falls back to the original chunks if filtering leaves almost nothing."""
    sentences = list(dict.fromkeys(s for c in chunks for s in split_sentences(c["text"])))
    kept = [s for s in sentences if not _BENEFIT.search(s) and not _ABOUT_EMPLOYER.search(s)]
    if len(" ".join(kept).split()) < 40:
        return chunks
    cued = [s for s in kept if _REQUIRED.search(s) or _PREFERRED.search(s) or _REQUIREMENT_CUE.search(s)]
    chosen, words = set(), 0
    for sentence in cued + [s for s in kept if s not in cued]:
        n = len(sentence.split())
        if words + n > max_words and chosen:
            continue
        chosen.add(sentence)
        words += n
    text = "\n".join(s for s in kept if s in chosen)
    return [{**chunks[0], "section": "requirements", "text": text}]


_CLAUSE = re.compile(r"[;:,()]|\s(?:and|or)\s|\.\s")


def _posting_sentence(requirement: str, posting_text: str) -> str | None:
    terms = key_terms(requirement)
    if not terms:
        return None
    candidates = [s for s in split_sentences(posting_text) if _mentions(terms, s)]
    return min(candidates, key=len) if candidates else None  # shortest = most specific


def classify_importance(sentence: str | None, requirement: str | None = None) -> str:
    """Reads the cue nearest the requirement: in "BLS required; ACLS
    preferred" the clause holding ACLS decides, not whichever cue the
    sentence happens to contain first. Falls back to the whole sentence."""
    if not sentence:
        return "unspecified"
    if requirement:
        terms = key_terms(requirement)
        parts = _CLAUSE.split(sentence)
        for i, part in enumerate(parts):
            if terms and _mentions(terms, part):
                # The cue may sit in the clause itself or right after it ("ACLS, preferred").
                window = " ".join(parts[i:i + 2])
                if _PREFERRED.search(window) and not _REQUIRED.search(part):
                    return "preferred"
                if _REQUIRED.search(window):
                    return "required"
                break
    if _PREFERRED.search(sentence):
        return "preferred"
    if _REQUIRED.search(sentence):
        return "required"
    return "unspecified"


def build_requirements(claims: list[tuple[CitedClaim, bool]], posting_text: str,
                       resume_text: str = "") -> list[Requirement]:
    """claims: (verified claim, met?) for ONE posting. Wording variants
    of one requirement ("Six months of RN ICU experience" and the same
    with "required") are kept once; a met version beats a gap."""
    out, seen = [], set()
    for claim, met in sorted(claims, key=lambda cm: not cm[1]):
        key = frozenset(_stem(t) for t in key_terms(claim.requirement))
        if key in seen:
            continue
        seen.add(key)
        text = clean_generated(claim.requirement)
        sentence = _posting_sentence(claim.requirement, posting_text)
        out.append(Requirement(
            text=text, met=met, importance=classify_importance(sentence, claim.requirement),
            credential=bool(_CREDENTIAL.search(claim.requirement)),
            resume_quote=claim.resume_evidence if met else None, posting_sentence=sentence,
            soft=is_soft(claim.requirement),
            # "APQP tools including FMEA, SPC" for a resume with APQP and FMEA:
            # at least half its terms shown -> "partly", not "not found".
            partly=[] if met else _partly_shown(claim.requirement, resume_text),
        ))
    return out


def _partly_shown(requirement: str, resume_text: str) -> list[str]:
    terms = key_terms(requirement)
    found = _found(terms, resume_text) if resume_text else []
    # At least two words, so "Multidisciplinary team" isn't "partly shown"
    # because the resume says "team" somewhere.
    return found if len(found) >= 2 and len(found) * 2 >= len(terms) else []


def assign_tiers(jobs: list[JobAnalysis]) -> JobAnalysis | None:
    """Sets .tier and .reason on every job; returns the best fit (or None).

    Why not "share of requirements met": the checks can confirm a
    requirement IS on the resume, but can't prove one is absent ("CRRT"
    on the resume, "renal failure" in the posting). On the nurse resume
    a ratio rule called the RN ICU posting, an obvious fit, "not a
    match" (1 of 9). So tiers come from things that ARE reliable:

    no:        a required degree/license/certification-level credential
               isn't found (a missing NP license can't be fixed by
               rewording a resume). Preferred credentials never block.
    best:      the closest posting by retrieval (whole-resume similarity,
               fused across sections) without such a gap.
    potential: every other posting without such a gap.
    The checklist counts are shown as evidence, never as a verdict.
    """
    for job in jobs:
        hard = [r for r in job.gaps if r.credential and r.importance != "preferred"]
        if hard:
            what = hard[0].text[0].lower() + hard[0].text[1:]
            job.tier, job.reason = "no", f"Asks for {what}, which isn't on your resume."
        else:
            job.tier = "potential"
            n = len(job.gaps)
            job.reason = ("Everything we could check is on your resume." if job.checked and not n else
                          f"{n} {'requirement' if n == 1 else 'requirements'} not found on your resume yet." if n else
                          "Few specific requirements could be checked for this one.")
    open_jobs = [j for j in jobs if j.tier != "no"]
    best = min(open_jobs, key=lambda j: j.rank) if open_jobs else None
    if best:
        best.tier = "best"
    return best


def learn_first(jobs: list[JobAnalysis], limit: int = 5) -> list[LearnItem]:
    """Gaps grouped across jobs by their key terms ("Cardiogenic shock"
    and "Cardiogenic shock experience" are one item), most-asked first."""
    groups: dict[tuple, dict] = {}
    for job in jobs:
        if job.tier == "no":
            continue  # "learn first" is for roles you can realistically go for
        for gap in job.gaps:
            key = tuple(sorted({_stem(t) for t in key_terms(gap.text)}))
            if not key:
                continue
            g = groups.setdefault(key, {"texts": [], "jobs": set(), "credential": gap.credential})
            g["texts"].append(gap.text)
            g["jobs"].add(job.job_id)
    items = [LearnItem(min(g["texts"], key=len), len(g["jobs"]), g["credential"]) for g in groups.values()]
    items.sort(key=lambda i: -i.jobs)  # stable: equal counts keep first-seen order
    return items[:limit]


NO_EVIDENCE_HEADLINE = "We couldn't connect this document to job requirements"


def _headline(best: JobAnalysis | None, jobs: list[JobAnalysis]) -> tuple[str, str]:
    if not any(j.met or any(r.partly for r in j.gaps) for j in jobs):
        return NO_EVIDENCE_HEADLINE, (
            f"We searched for the {len(jobs)} closest postings, but nothing they ask for could be confirmed in "
            "this document. If this is your resume, make sure it lists your skills, tools and experience in plain "
            "text. The postings are below so you can read them yourself.")
    others = sum(j.tier == "potential" for j in jobs)
    blocked = sum(j.tier == "no" for j in jobs)
    parts = []
    if best:
        parts.append("one stands out as the closest match")
    if others:
        parts.append(f"{others} more {'is' if others == 1 else 'are'} worth a look")
    if blocked:
        parts.append(f"{blocked} {'asks' if blocked == 1 else 'ask'} for a credential your resume doesn't show")
    listed = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    summary = f"We found the {len(jobs)} postings closest to your resume. {listed[0].upper()}{listed[1:]}."
    if best:
        company = f" at {best.company}" if best.company else ""
        return f"Your closest match is {best.title}{company}", summary
    return "These roles ask for credentials your resume doesn't show yet", summary


def build_report(
    retrieved: list[dict],
    resume_text: str,
    analyze: Callable[[str, list[dict]], GapAnalysisResult],
    get_chunks: Callable[[str], list[dict]],
    details: dict[str, dict] | None = None,
) -> JobReport:
    """
    One LLM call over all retrieved postings, then per-job checklists by
    grouping the verified claims by the posting each one cites.

    Why not one call per job: measured on the nurse and engineer resumes,
    five per-job calls took 60-68 s and produced mostly noise (company
    blurbs like "Best Place to Work" as requirements, 5-14 rejected
    claims per job, results that changed from run to run). One pooled
    call over filtered requirement text is ~1/5 the time, and its claims
    were already cited to a specific posting.

    retrieved:  postings from retrieve_for_resume (best chunk each).
    analyze:    (resume_text, excerpts) -> checked GapAnalysisResult
                (run_gap_analysis with the client bound).
    get_chunks: job_id -> all chunks of that posting (VectorStore.get_posting_chunks).
    """
    chunks_by_job = {p["job_id"]: get_chunks(p["job_id"]) or [p] for p in retrieved}
    excerpts = [requirement_excerpt(chunks_by_job[p["job_id"]])[0] for p in retrieved]
    gap = analyze(resume_text, excerpts)

    claims_by_job: dict[str, list[tuple[CitedClaim, bool]]] = {p["job_id"]: [] for p in retrieved}
    for claim, met in [(c, True) for c in gap.matched] + [(c, False) for c in gap.missing]:
        claims_by_job.setdefault(claim.job_id, []).append((claim, met))
    rejected_by_job: dict[str, int] = {}
    for claim in gap.unverified:
        rejected_by_job[claim.job_id] = rejected_by_job.get(claim.job_id, 0) + 1

    jobs = []
    for rank, posting in enumerate(retrieved, start=1):
        job_id = posting["job_id"]
        full_text = "\n".join(c["text"] for c in chunks_by_job[job_id])
        jobs.append(JobAnalysis(
            rank=rank, job_id=job_id, title=posting["title"],
            company=posting.get("company_name"), url=posting.get("job_posting_url"),
            details=(details or {}).get(str(job_id), {}),
            requirements=build_requirements(claims_by_job.get(job_id, []), full_text, resume_text),
            rejected=rejected_by_job.get(job_id, 0), fusion_score=posting.get("fusion_score", 0.0),
        ))
    best = assign_tiers(jobs)
    headline, summary = _headline(best, jobs)
    if headline == NO_EVIDENCE_HEADLINE and best:
        best.tier = "potential"  # no evidence: don't present any posting as "the closest match"
        best = None
    order = {"best": 0, "potential": 1, "no": 2}
    jobs.sort(key=lambda j: (order[j.tier], j.rank))
    return JobReport(jobs=jobs, learn_first=learn_first(jobs), headline=headline, summary=summary,
                     best=best, analysis=gap)
