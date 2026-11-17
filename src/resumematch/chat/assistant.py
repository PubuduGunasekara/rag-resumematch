"""The assistant: one chat box that works from sign-in, before any resume.

It routes each message, in code, to one of five things:

  help     what the system does and what to do next (written here, per role)
  role     "what do data engineer postings ask for?": the role report's terms
  action   "draft a cover letter for the second posting": the site runs the
           existing tailoring endpoints; nothing is generated here
  qa       a question about the report: chat/ask.py, with its checks
  need     something that needs a resume or a report that isn't there yet

Only "qa" calls the language model. The rest is routing and the report's
own data, so the assistant can't claim a feature that doesn't exist or
invent what a job title asks for.
"""

import re
from dataclasses import dataclass, field

RESUME_ROLES = ("seeker", "advisor", "admin")
RECRUIT_ROLES = ("recruiter", "admin")

_GREETING = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|yo)\b[\s!.,]*$", re.IGNORECASE)
_HELP = re.compile(r"\b(help|what can (you|i|this)|what do you do|how (does|do) (this|it|i)|get(ting)? started|"
                   r"where do i (start|begin)|what is this|features?|what should i do)\b", re.IGNORECASE)
_LETTER = re.compile(r"\bcover.?letter\b", re.IGNORECASE)
_BULLETS = re.compile(r"\b(rewrite|reword|improve|edit|tailor|polish|fix|strengthen)\b.*\b(resume|bullets?|cv)\b|"
                      r"\bresume (edits?|suggestions?|rewrites?)\b|\bbullet (rewrites?|points?)\b", re.IGNORECASE)
_INTERVIEW = re.compile(r"\binterview", re.IGNORECASE)
_COMPARE = re.compile(r"\b(compare|revised|new version|updated resume|second version)\b", re.IGNORECASE)
_UPLOAD = re.compile(r"\b(upload|check my resume|analy[sz]e my resume|scan my resume|start over|new resume)\b", re.IGNORECASE)
_RANK = re.compile(r"\b(rank|candidates?|applicants?|shortlist|screen)\b", re.IGNORECASE)
_ROLE = [
    re.compile(r"what (?:do|does|would) (?:an? |the )?(.+?) (?:postings?|jobs?|roles?|positions?|employers?) "
               r"(?:usually |typically |commonly |most often )?(?:ask|require|need|want|look|expect)", re.IGNORECASE),
    re.compile(r"(?:skills?|requirements?|qualifications?) (?:for|of|needed for|to be) (?:an? |the )?(.+?)(?: jobs?| roles?| positions?)?[?.!]*$",
               re.IGNORECASE),
    re.compile(r"what (?:do i|does (?:someone|one)|would i) need to (?:be|become|work as) (?:an? |the )?(.+?)[?.!]*$", re.IGNORECASE),
]
_ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
             "fifth": 5, "5th": 5, "top": 1, "closest": 0, "best": 0}
_NOT_A_TITLE = {"these", "those", "the", "this", "my", "other", "others", "most", "all", "any", "such", "similar"}


@dataclass
class Reply:
    kind: str                                   # "help" | "role" | "action" | "need" | "qa"
    text: list[str] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)       # what the site should do or offer
    suggestions: list[str] = field(default_factory=list)    # follow-up questions to show as chips


def role_title(question: str) -> str | None:
    """The job title a question asks about, or None. "these postings" is not a title."""
    for pattern in _ROLE:
        m = pattern.search(question)
        if m:
            title = " ".join(m.group(1).split()).strip(" ?.!,")
            words = title.lower().split()
            if title and len(title) <= 60 and not (set(words) & _NOT_A_TITLE):
                return title
    return None


def pick_job(question: str, jobs: list[dict]) -> dict | None:
    """The posting a message refers to: "the second posting", a title or
    company word, else the closest match, else the first."""
    if not jobs:
        return None
    q = question.lower()
    for word, n in _ORDINALS.items():
        if re.search(rf"\b{word}\b", q):
            if n == 0:
                return next((j for j in jobs if j.get("closest")), jobs[0])
            if n <= len(jobs):
                return jobs[n - 1]
    m = re.search(r"\b(?:posting|job|number|#)\s*#?(\d)\b", q)
    if m and 1 <= int(m.group(1)) <= len(jobs):
        return jobs[int(m.group(1)) - 1]
    q_words = set(re.findall(r"[a-z]{4,}", q))
    best, best_n = None, 0
    for job in jobs:
        words = set(re.findall(r"[a-z]{4,}", f"{job.get('title', '')} {job.get('company') or ''}".lower()))
        n = len(q_words & words - {"posting", "letter", "cover", "resume", "interview", "questions"})
        if n > best_n:
            best, best_n = job, n
    return best or next((j for j in jobs if j.get("closest")), jobs[0])


def help_reply(role: str, has_report: bool) -> Reply:
    """What this role can do, and the next step. Only real features."""
    if role == "recruiter":
        return Reply("help", [
            "I can help you rank candidates for a job and explore what postings for a job title ask for.",
            "To rank candidates, open Rank candidates, paste the job description and upload up to 10 resumes. "
            "Blind review hides names, contact details, schools and graduation years until you reveal them.",
            "You can also ask me what postings for a job title ask for, for example: what do data engineer postings ask for?",
        ], [{"type": "open", "href": "/recruit", "label": "Rank candidates"},
            {"type": "open", "href": "/role", "label": "Explore a role"}],
            ["What do data engineer postings ask for?"])
    if not has_report:
        return Reply("help", [
            "Upload your resume and I'll find the closest postings among the indexed LinkedIn postings, then check "
            "each of their requirements against what your resume says.",
            "After that you can: see which requirements your resume shows, with the line that shows each; get "
            "suggested edits to your resume bullets and a cover letter draft for any posting; prepare for the "
            "interview; and upload a revised resume to see what changed.",
            "Everything I show is checked against your resume and the posting it cites. Before you upload, I can "
            "already tell you what postings for a job title ask for.",
        ], [{"type": "upload", "label": "Upload your resume"}, {"type": "open", "href": "/role", "label": "Explore a role"}],
            ["What do registered nurse postings ask for?", "What happens to my resume after the check?"])
    return Reply("help", [
        "Your report is ready. You can ask me about it, or ask me to start something.",
        "For example: why is my closest match a good fit, which of these jobs list a salary, draft a cover letter "
        "for the second posting, suggest resume edits, or prepare interview questions.",
    ], [{"type": "open", "href": "/matches", "label": "See your matches"},
        {"type": "open", "href": "/tailor", "label": "Tailor your resume"}],
        ["Why is my closest match a good fit?", "Draft a cover letter for my closest match",
         "What experience do I have that these jobs ask for?"])


PRIVACY = ("Your resume is read by a model running on this server, not sent to an outside service. The uploaded file "
           "is deleted as soon as its text is read, and the report lives only in this browser tab.")


def route(question: str, role: str, has_resume: bool, jobs: list[dict]) -> Reply | None:
    """A reply decided in code, or None when the question should go to the
    report Q&A (chat/ask.py). jobs: [{"job_id", "title", "company", "closest"}]."""
    q = " ".join(question.split())
    can_resume = role in RESUME_ROLES
    if _GREETING.match(q) or (_HELP.search(q) and not _LETTER.search(q) and not _INTERVIEW.search(q)):
        return help_reply(role, bool(jobs))
    if re.search(r"\b(privacy|private|stored?|saved?|happens to my resume|deleted?|who sees)\b", q, re.IGNORECASE):
        return Reply("help", [PRIVACY])
    title = role_title(q)
    if title:
        return Reply("role", actions=[{"type": "role", "title": title}])
    if role in RECRUIT_ROLES and not can_resume and _RANK.search(q):
        return Reply("help", ["Open Rank candidates, paste the job description and upload the resumes. I'll check every "
                              "resume against the same requirements in the same way."],
                     [{"type": "open", "href": "/recruit", "label": "Rank candidates"}])
    wants = ("letter" if _LETTER.search(q) else "bullets" if _BULLETS.search(q)
             else "interview" if _INTERVIEW.search(q) else "compare" if _COMPARE.search(q) else None)
    if wants and not can_resume:
        return Reply("need", ["That works on a job seeker's own resume, which the recruiter role doesn't have. "
                              "You can rank candidates or explore a role."],
                     [{"type": "open", "href": "/recruit", "label": "Rank candidates"}])
    if wants and not jobs:
        return Reply("need", ["I need your resume first. Upload it and I'll find the closest postings; then I can do that "
                              "for any of them."], [{"type": "upload", "label": "Upload your resume"}])
    if wants == "compare":
        return Reply("action", ["Open Compare and choose your revised file. I'll check it against the same postings and "
                                "show which requirements changed."],
                     [{"type": "open", "href": "/compare", "label": "Compare a revised version"}])
    if wants:
        job = pick_job(q, jobs)
        name = f"{job['title']} at {job['company']}" if job.get("company") else job["title"]
        doing = {"letter": "Drafting a cover letter for", "bullets": "Suggesting resume edits for",
                 "interview": "Preparing interview questions for"}[wants]
        end = "" if name.endswith(".") else "."   # "Rely Services, Inc." already ends a sentence
        return Reply("action", [f"{doing} {name}{end} It opens when it's ready; anything that adds a fact your resume "
                                f"doesn't state is removed and listed."],
                     [{"type": wants, "job_id": job["job_id"], "label": name}])
    if _UPLOAD.search(q) and can_resume:
        return Reply("need" if not has_resume else "help",
                     ["Use the upload box on the Start page. A PDF or Word file up to 5 MB works."],
                     [{"type": "upload", "label": "Upload your resume"}])
    if not jobs:
        if not can_resume:
            return help_reply(role, False)
        return Reply("need", ["I can answer that once your resume is checked. Upload it to start, or ask me what postings "
                              "for a job title ask for."],
                     [{"type": "upload", "label": "Upload your resume"}],
                     ["What do data engineer postings ask for?", "What can you do?"])
    return None


# --- Before sign-in ----------------------------------------------------------------

_SIGN_IN = re.compile(r"\b(sign.?in|log.?in|login|password|account|demo|credentials?|register|sign.?up)\b", re.IGNORECASE)
GUEST_SUGGESTIONS = ["What can ResumeMatch do?", "How do I sign in?", "What happens to my resume?"]


def guest_reply(message: str, password_public: bool) -> Reply:
    """The assistant on the login page: what the system is and how to get in.
    Nothing here needs a session, and nothing reaches the model."""
    q = " ".join(message.split())
    how_to = ("This is a demo with one account per role: job seeker, recruiter, career advisor and admin. "
              + ("Press \"Fill in\" next to a role on this page, then Sign in."
                 if password_public else "All four share the demo password set on this server."))
    if _SIGN_IN.search(q):
        return Reply("help", [how_to], suggestions=["What can ResumeMatch do?", "What happens to my resume?"])
    if re.search(r"\b(privacy|private|stored?|saved?|happens to my resume|deleted?|who sees)\b", q, re.IGNORECASE):
        return Reply("help", [PRIVACY, "Sign in to try it."], suggestions=["How do I sign in?"])
    if _GREETING.match(q) or _HELP.search(q) or re.search(r"\b(what is|resumematch|about)\b", q, re.IGNORECASE):
        return Reply("help", [
            "Hi. ResumeMatch checks your resume against real LinkedIn job postings and shows which requirements of the "
            "closest ones your resume meets, with the line of your resume that shows each.",
            "Signed in, you can also get suggested resume edits, a cover letter draft and interview prep for any posting, "
            "see what postings for a job title ask for, and ask me questions about your results. Recruiters can rank "
            "candidates for a job with blind review.",
            "Sign in to access these. " + how_to,
        ], suggestions=["How do I sign in?", "What happens to my resume?"])
    return Reply("need", ["Sign in to access that. Once you're in, I can answer questions about your resume and the postings, "
                          "and start any feature for you. " + how_to], suggestions=GUEST_SUGGESTIONS)
