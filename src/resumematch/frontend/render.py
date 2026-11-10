"""HTML for the results page, as plain functions of the report.

Kept apart from streamlit_app.py so it can be tested and reused (the PDF
report renders the same markup). Every string a user reads is written
here or comes from the report: no em dashes, no emoji, sentence case.
Source quotes (resume, postings) are shown exactly as written.
"""

import html
import re

from resumematch.generation.job_report import JobAnalysis, JobReport, LearnItem, Requirement

PAPER, INK, GRAPHITE, HAIRLINE = "#FAF8F4", "#1F1D1A", "#6B655C", "#E4DFD6"
MOSS, OCHRE, BRICK, SAGE = "#4A6B3A", "#A8742A", "#9A4A3A", "#E9EDD9"

FONTS_URL = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500"
             "&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500&display=swap")

CSS = f"""
@import url('{FONTS_URL}');
:root {{ --paper:{PAPER}; --ink:{INK}; --graphite:{GRAPHITE}; --hairline:{HAIRLINE};
        --moss:{MOSS}; --ochre:{OCHRE}; --brick:{BRICK}; --sage:{SAGE}; }}
.rm, .rm * {{ font-family:'IBM Plex Sans', system-ui, sans-serif; color:var(--ink); }}
.rm .serif {{ font-family:'Source Serif 4', Georgia, serif; font-weight:500; }}
.rm .tag {{ font-size:11px; letter-spacing:.07em; text-transform:uppercase; color:var(--graphite); margin:0 0 8px; }}
.rm .muted {{ color:var(--graphite); }}
.rm h1.serif {{ font-size:32px; line-height:1.2; margin:0 0 12px; }}
.rm h2.serif {{ font-size:23px; line-height:1.25; margin:0; }}
.rm .lede {{ font-size:16px; line-height:1.65; color:var(--graphite); max-width:600px; margin:0 0 8px; }}
.rm .facts {{ display:flex; flex-wrap:wrap; gap:0; margin:4px 0 14px; font-size:13px; color:var(--graphite); }}
.rm .facts span {{ padding:0 10px; border-left:1px solid var(--hairline); }}
.rm .facts span:first-child {{ padding-left:0; border-left:none; }}
.rm .seg {{ display:inline-flex; gap:3px; vertical-align:middle; margin-right:10px; }}
.rm .seg i {{ display:block; width:11px; height:11px; border:1px solid #C9C2B6; }}
.rm .seg i.f {{ background:var(--moss); border-color:var(--moss); }}
.rm .seg i.p {{ background:linear-gradient(90deg, var(--moss) 50%, transparent 50%); border-color:var(--moss); }}
.rm .lead {{ border-left:3px solid var(--moss); padding:2px 0 4px 20px; margin:6px 0 34px; }}
.rm .row {{ display:flex; justify-content:space-between; gap:16px; padding:9px 0; border-top:1px solid var(--hairline); font-size:14px; }}
.rm .row .q {{ display:block; font-size:12.5px; color:var(--graphite); margin-top:3px; }}
.rm .st {{ font-size:13px; white-space:nowrap; }}
.rm .st.shown {{ color:var(--moss); }} .rm .st.partly {{ color:var(--ochre); }} .rm .st.not_found {{ color:var(--ochre); }}
.rm .req {{ font-size:12px; color:var(--graphite); }}
.rm .reason {{ font-size:14px; margin:8px 0 0; }}
.rm .reason.no {{ color:var(--brick); }}
.rm a {{ color:var(--ink); text-decoration:underline; text-underline-offset:3px; text-decoration-color:#C9C2B6; }}
.rm a:hover {{ text-decoration-color:var(--ink); }}
.rm .resume {{ font-size:14px; line-height:1.85; background:#F3EFE8; padding:16px 18px; border-radius:3px; }}
.rm mark {{ background:var(--sage); padding:0 2px; border-radius:2px; }}
.rm .learn {{ display:grid; grid-template-columns:minmax(0,1fr) 120px; gap:16px; padding:9px 0; border-top:1px solid var(--hairline); font-size:14px; align-items:center; }}
.rm .bar {{ height:5px; background:var(--hairline); }} .rm .bar div {{ height:5px; background:var(--ink); }}
.rm .soft {{ font-size:13px; color:var(--graphite); margin-top:10px; }}
.rm .rule {{ border-top:1px solid var(--hairline); margin:34px 0 0; }}
"""

_STATUS = {"shown": "Shown", "partly": "Partly shown", "not_found": "Not found"}


def esc(text: str | None) -> str:
    return html.escape(text or "")


def facts(details: dict) -> str:
    items = [details.get("location"), details.get("work_type"), details.get("level"),
             "Remote allowed" if details.get("remote") else None, details.get("salary")]
    items = [i for i in items if i]
    return f'<div class="facts">{"".join(f"<span>{esc(i)}</span>" for i in items)}</div>' if items else ""


def segments(job: JobAnalysis) -> str:
    cells = []
    for r in job.counted:
        cells.append('<i class="f"></i>' if r.met else '<i class="p"></i>' if r.partly else "<i></i>")
    return f'<span class="seg" aria-hidden="true">{"".join(cells)}</span>' if cells else ""


def score_line(job: JobAnalysis) -> str:
    if not job.checked:
        return '<span class="muted" style="font-size:14px">No specific requirements could be checked</span>'
    partly = sum(bool(r.partly) for r in job.gaps)
    extra = f", {partly} partly" if partly else ""
    noun = "requirement" if job.checked == 1 else "requirements"
    return (f'{segments(job)}<span style="font-size:14px">{job.met} of {job.checked} '
            f'{noun} shown on your resume{extra}</span>')


def requirement_row(r: Requirement) -> str:
    importance = {"required": "required", "preferred": "preferred"}.get(r.importance, "")
    tag = f' <span class="req">{importance}</span>' if importance else ""
    quote = ""
    if r.met and r.resume_quote:
        quote = f'<span class="q">Your resume: "{esc(r.resume_quote)}"</span>'
    elif r.partly:
        quote = f'<span class="q">Your resume mentions {esc(", ".join(r.partly))}, not the rest</span>'
    return (f'<div class="row"><div>{esc(r.text)}{tag}{quote}</div>'
            f'<span class="st {r.status}">{_STATUS[r.status]}</span></div>')


def checklist(job: JobAnalysis) -> str:
    rows = "".join(requirement_row(r) for r in job.counted)
    soft = [r.text for r in job.requirements if r.soft]
    soft_html = (f'<p class="soft">It also asks for general strengths: {esc("; ".join(soft))}. '
                 'Your interview and cover letter are the place for these.</p>') if soft else ""
    if not rows:
        rows = ('<p class="muted" style="font-size:14px">The model found no requirements in this posting that could '
                'be checked against your resume. Open the posting to read it yourself.</p>')
    return rows + soft_html


def posting_link(job: JobAnalysis) -> str:
    if not job.url:
        return ""
    return (f'<p style="font-size:13px;margin:12px 0 0"><a href="{esc(job.url)}" target="_blank" rel="noopener">'
            'Open the original posting</a> <span class="muted">(it may have closed)</span></p>')


def header(report: JobReport, total_postings: int) -> str:
    return (f'<div class="rm"><p class="tag">Resume report</p><h1 class="serif">{esc(report.headline)}</h1>'
            f'<p class="lede">{esc(report.summary)} Searched {total_postings:,} LinkedIn postings.</p></div>')


def best_fit(job: JobAnalysis) -> str:
    company = f"{esc(job.company)}" if job.company else ""
    return (f'<div class="rm"><p class="tag" style="margin-top:28px">Closest match</p><div class="lead">'
            f'<h2 class="serif">{esc(job.title)}</h2><p class="muted" style="margin:2px 0 0;font-size:15px">{company}</p>'
            f'{facts(job.details)}{score_line(job)}<p class="reason">{esc(job.reason)}</p>'
            f'<div style="margin-top:16px">{checklist(job)}</div>{posting_link(job)}</div></div>')


def job_body(job: JobAnalysis) -> str:
    reason_class = "reason no" if job.tier == "no" else "reason"
    return (f'<div class="rm">{facts(job.details)}{score_line(job)}<p class="{reason_class}">{esc(job.reason)}</p>'
            f'<div style="margin-top:14px">{checklist(job)}</div>{posting_link(job)}</div>')


def job_label(job: JobAnalysis) -> str:
    """Plain text for an expander label (Streamlit renders it as markdown)."""
    company = f", {job.company}" if job.company else ""
    score = f"{job.met} of {job.checked} shown" if job.checked else "not checked"
    return f"{job.title}{company}   ({score})".replace("*", "").replace("_", " ")


def _quote_pattern(quote: str) -> re.Pattern | None:
    words = re.findall(r"[A-Za-z0-9+#]+", quote)
    return re.compile(r"[^A-Za-z0-9+#]+".join(map(re.escape, words)), re.IGNORECASE) if words else None


def highlight_spans(resume_text: str, quotes: list[str]) -> list[list[int]]:
    """[start, end) character ranges of every quote in the resume, merged.
    Quotes were verified against normalized text (case, punctuation and
    line breaks ignored), so they're located the same way. Used by the
    Streamlit page and sent as-is to the Next.js site by the API."""
    spans = []
    for quote in quotes:
        pattern = _quote_pattern(quote)
        if pattern:
            spans += [m.span() for m in pattern.finditer(resume_text)]
    spans.sort()
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def highlighted_resume(resume_text: str, quotes: list[str]) -> str:
    """The resume as written, with every quote that proved a requirement marked."""
    out, pos = [], 0
    merged = highlight_spans(resume_text, quotes)
    for start, end in merged:
        out += [esc(resume_text[pos:start]), f"<mark>{esc(resume_text[start:end])}</mark>"]
        pos = end
    out.append(esc(resume_text[pos:]))
    body = "".join(out).strip().replace("\n", "<br>")
    return f'<div class="rm"><div class="resume">{body}</div></div>'


def learn_first(items: list[LearnItem], total_jobs: int) -> str:
    if not items:
        return ('<div class="rm"><p class="muted" style="font-size:14px">Nothing came up in more than one place. '
                'The checklists above are the best guide.</p></div>')
    rows = "".join(
        f'<div class="learn"><span>{esc(i.text)}'
        f'{"<span class=req> credential</span>" if i.credential else ""}</span>'
        f'<span><div class="bar"><div style="width:{round(100 * i.jobs / max(total_jobs, 1))}%"></div></div>'
        f'<span class="req">{i.jobs} of {total_jobs} postings</span></span></div>'
        for i in items)
    return f'<div class="rm">{rows}</div>'


def section_tag(text: str, top: int = 34) -> str:
    return f'<div class="rm"><p class="tag" style="margin-top:{top}px">{esc(text)}</p></div>'
