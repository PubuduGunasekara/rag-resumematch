"""Streamlit front end: the quick local interface.

Run with: streamlit run src/resumematch/frontend/streamlit_app.py

Thin on purpose: it calls pipeline.run_pipeline once per upload, keeps
the result in session state, and renders it with frontend/render.py.
No retrieval or prompt logic belongs here.
"""

import hashlib
import json
import tempfile
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

# OLLAMA_HOST/OLLAMA_MODEL/EMBEDDING_MODEL from .env for local runs; real
# env vars (docker-compose) take precedence.
load_dotenv()

from resumematch.frontend import render  # noqa: E402 (after load_dotenv)
from resumematch.health import check  # noqa: E402
from resumematch.pipeline import IndexNotBuiltError, ResumeRejected, run_pipeline  # noqa: E402

DOMAIN_VOCAB_PATH = Path("data/processed/domain_vocab.json")
DETAILS_PATH = Path("data/processed/posting_details.json")
TOTAL_POSTINGS = 107_528  # distinct postings in the full index (BUILD_COMPLETE.json)

st.set_page_config(page_title="ResumeMatch", layout="centered", initial_sidebar_state="collapsed")

# Streamlit's own chrome, restyled to the same paper and ink theme.
st.markdown(f"""<style>{render.CSS}
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"] {{ display:none !important; }}
.stApp {{ background:{render.PAPER}; }}
.block-container {{ max-width:780px; padding-top:56px; padding-bottom:80px; }}
html, body, .stApp, .stMarkdown, button, input, label, p {{ font-family:'IBM Plex Sans', system-ui, sans-serif; color:{render.INK}; }}
[data-testid="stFileUploaderDropzone"] {{ background:#F3EFE8; border:1px dashed #C9C2B6; border-radius:3px; }}
[data-testid="stFileUploaderDropzone"] button {{ background:{render.INK}; color:{render.PAPER}; border:none; border-radius:3px; }}
[data-testid="stExpander"] details {{ border:none; border-top:1px solid {render.HAIRLINE}; border-radius:0; background:transparent; }}
[data-testid="stExpander"] summary {{ padding:12px 0; }}
[data-testid="stExpander"] summary p {{ font-size:15px; font-weight:500; }}
[data-testid="stExpander"] summary:hover p {{ text-decoration:underline; text-underline-offset:3px; }}
[data-testid="stExpanderDetails"] {{ padding:0 0 12px; }}
div[data-baseweb="select"] > div {{ background:#F3EFE8; border-color:{render.HAIRLINE}; border-radius:3px; }}
[data-testid="stSpinner"] p {{ color:{render.GRAPHITE}; }}
.stAlert {{ border-radius:3px; }}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_json(path: str):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def run(file_bytes: bytes, suffix: str, correction: bool, allow_unusual: bool = False):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        return run_pipeline(path, domain_vocabulary=load_json(str(DOMAIN_VOCAB_PATH)), apply_correction=correction,
                            details=load_json(str(DETAILS_PATH)) or {}, allow_unusual=allow_unusual)
    finally:
        Path(path).unlink(missing_ok=True)  # the resume is never kept


def friendly_error(message: str, detail: str | None = None):
    st.markdown(f'<div class="rm"><p class="reason no">{render.esc(message)}</p></div>', unsafe_allow_html=True)
    if detail:
        with st.expander("Details"):
            st.text(detail)
    st.stop()


# Landing
st.markdown(
    '<div class="rm"><p class="tag">ResumeMatch</p>'
    '<h1 class="serif">Check your resume against real job postings</h1>'
    '<p class="lede">Upload a resume. ResumeMatch finds the closest postings among 107,528 from LinkedIn, checks each '
    'requirement against what your resume actually says, and shows you what to work on before you apply.</p></div>',
    unsafe_allow_html=True,
)
@st.cache_data(ttl=30)
def startup_problems():
    from resumematch.pipeline import load_config, resolve_persist_dir
    return check(resolve_persist_dir(load_config()))


problems = startup_problems()
for problem in problems:
    st.markdown(f'<div class="rm"><p class="reason{" no" if problem.blocking else ""}" style="font-size:14px">'
                f'{render.esc(problem.message)}</p></div>', unsafe_allow_html=True)
if any(p.blocking for p in problems):
    st.stop()

uploaded = st.file_uploader("Your resume, PDF or Word", type=["pdf", "docx"], label_visibility="collapsed")
with st.expander("Options"):
    correction = st.toggle("Fix typos before searching", value=True,
                           help="Typos are fixed only for the search, using a vocabulary built from every posting. "
                                "Your resume is checked as written.")
st.markdown('<div class="rm"><p class="muted" style="font-size:12.5px;margin-top:4px">Your resume is processed '
            'on this machine by a local model and isn\'t stored.</p></div>', unsafe_allow_html=True)

if uploaded is None:
    st.stop()

key = hashlib.sha256(uploaded.getvalue()).hexdigest() + str(correction)
override = st.session_state.get("override") == key  # the user chose "Check it anyway" for this file
if st.session_state.get("key") != key:
    with st.spinner("Reading your resume and checking it against similar postings. This takes about 20 seconds."):
        try:
            st.session_state.result = run(uploaded.getvalue(), Path(uploaded.name).suffix, correction, override)
            st.session_state.key = key
        except ResumeRejected as rejected:
            st.markdown(f'<div class="rm"><p class="reason no">{render.esc(rejected.message)}</p></div>',
                        unsafe_allow_html=True)
            if rejected.soft and st.button("Check it anyway"):
                st.session_state.override = key
                st.rerun()
            st.stop()
        except IndexNotBuiltError:
            friendly_error("The job postings index hasn't been built yet. Run scripts/build_index.py first.")
        except requests.ConnectionError:
            friendly_error("Couldn't reach the local model. Start Ollama (brew services start ollama) and try again.")
        except ValueError as error:
            friendly_error("The model's answer couldn't be used, even after a retry. Upload the file again to rerun it.",
                           str(error))
        except Exception:  # noqa: BLE001 - never show a traceback to a user
            friendly_error("Something went wrong while checking this file. Try again, and if it keeps happening, "
                           "try exporting your resume again.")

result = st.session_state.result
report = result.report

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
st.markdown(render.header(report, TOTAL_POSTINGS), unsafe_allow_html=True)

if report.best:
    st.markdown(render.best_fit(report.best), unsafe_allow_html=True)
others_tag = "Also worth a look" if report.best else "The closest postings"

others = [j for j in report.jobs if j.tier == "potential"]
blocked = [j for j in report.jobs if j.tier == "no"]
if others:
    st.markdown(render.section_tag(others_tag, top=10 if report.best else 34), unsafe_allow_html=True)
    for job in others:
        with st.expander(render.job_label(job)):
            st.markdown(render.job_body(job), unsafe_allow_html=True)
if blocked:
    st.markdown(render.section_tag("Not a match right now"), unsafe_allow_html=True)
    for job in blocked:
        with st.expander(render.job_label(job)):
            st.markdown(render.job_body(job), unsafe_allow_html=True)

st.markdown(render.section_tag("What to work on, across these postings"), unsafe_allow_html=True)
st.markdown(render.learn_first(report.learn_first, len(report.jobs)), unsafe_allow_html=True)

st.markdown(render.section_tag("Your resume, with the evidence marked"), unsafe_allow_html=True)
with_quotes = [j for j in report.jobs if any(r.met for r in j.requirements)]
if with_quotes:
    choice = st.selectbox("Show evidence for", with_quotes, format_func=lambda j: f"{j.title}, {j.company or ''}".rstrip(", "),
                          label_visibility="collapsed")
    quotes = [r.resume_quote for r in choice.requirements if r.met and r.resume_quote]
else:
    quotes = []
st.markdown(render.highlighted_resume(result.resume_text, quotes), unsafe_allow_html=True)

# Every step of the retrieval and checking, visible to the reader.
st.markdown(render.section_tag("How this was checked"), unsafe_allow_html=True)
with st.expander(f"Search: {len(result.sections)} resume sections, {len(result.retrieved)} postings"):
    st.markdown(
        '<div class="rm"><p style="font-size:14px">Each section of your resume was searched separately (the embedding '
        'model reads at most 512 tokens at a time), and results were combined per posting with reciprocal rank '
        f'fusion. Sections: {render.esc(", ".join(s.heading for s in result.sections))}.</p></div>',
        unsafe_allow_html=True)
    for i, p in enumerate(result.retrieved, start=1):
        st.markdown(f"{i}. {p['title']}, {p.get('company_name') or 'unknown company'} "
                    f"(found by {', '.join(p['matched_sections'][:3])})")
with st.expander(f"Typo fixes used for the search: {len(result.corrections)}"):
    if result.corrections:
        st.table([{"In your resume": c.original_token, "Searched as": c.corrected_token} for c in result.corrections])
    else:
        st.write("None. Your resume was searched exactly as written.")
analysis = report.analysis
with st.expander(f"Model claims that failed a check: {len(analysis.unverified)}"):
    st.markdown('<div class="rm"><p style="font-size:14px">Every claim from the model is checked in code before it '
                'is shown. These failed and were left out.</p></div>', unsafe_allow_html=True)
    for c in analysis.unverified:
        st.markdown(f"- {c.requirement} (claimed {c.claimed_as}): {c.issue}")

st.markdown(
    '<div class="rm"><div class="rule"></div><p class="muted" style="font-size:12px;margin-top:12px">'
    'Postings from LinkedIn Job Postings 2023 to 2024 by Arsh Koneru, CC BY-SA 4.0. Listings may have closed. '
    'Checks compare wording, so a requirement marked not found may still be something you have. Add it if it\'s true.'
    '</p></div>', unsafe_allow_html=True)
