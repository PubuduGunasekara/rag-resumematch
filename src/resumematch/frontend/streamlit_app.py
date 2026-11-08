"""Streamlit front end — the required UI per the project spec.

Run with: streamlit run src/resumematch/frontend/streamlit_app.py

Kept deliberately thin: this file renders state and calls pipeline.py;
it does not contain retrieval/generation logic itself. If you find
yourself writing chunking or prompt logic in this file, it belongs in
preprocess/ or generation/ instead.
"""

import json
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# OLLAMA_HOST/OLLAMA_MODEL/EMBEDDING_MODEL from .env for local runs; real
# env vars (docker-compose) take precedence.
load_dotenv()

from resumematch.pipeline import run_pipeline, IndexNotBuiltError  # noqa: E402 (after load_dotenv)

# Written by scripts/build_index.py from a sample of indexed postings —
# see VOCAB_OUTPUT_PATH there. Kept as a plain JSON file rather than
# rebuilt here so the Streamlit process never has to touch the raw
# dataset at all.
DOMAIN_VOCAB_PATH = Path("data/processed/domain_vocab.json")

st.set_page_config(page_title="ResumeMatch", layout="wide")
st.title("ResumeMatch")
st.caption(
    "Upload a resume. Retrieves matching job postings from a 124,000+ posting "
    "corpus (LinkedIn Job Postings 2023-2024, Arsh Koneru, CC BY-SA 4.0) and "
    "produces a cited gap analysis against the closest matches."
)


@st.cache_resource
def get_domain_vocabulary():
    """
    Built once per app process, not per upload — see pipeline.py's
    docstring for why. Loads the vocabulary scripts/build_index.py
    persisted from a sample of indexed postings; falls back to None
    (pipeline.py then skips correction rather than crashing) if the
    index hasn't been built yet.
    """
    if not DOMAIN_VOCAB_PATH.exists():
        return None
    return json.loads(DOMAIN_VOCAB_PATH.read_text())


apply_correction = st.sidebar.toggle(
    "Typo correction",
    value=True,
    help=(
        "Fix typos/PDF-extraction artifacts in the resume before retrieval, "
        "using edit distance against a vocabulary built from all job postings. "
        "Turn off to compare retrieval with and without it."
    ),
)
if get_domain_vocabulary() is None:
    st.sidebar.caption("No vocabulary found yet (run scripts/build_index.py), so correction is skipped.")

uploaded_file = st.file_uploader("Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    with st.spinner("Retrieving matching postings and running gap analysis..."):
        try:
            pipeline_result = run_pipeline(
                tmp_path, domain_vocabulary=get_domain_vocabulary(), apply_correction=apply_correction
            )
        except IndexNotBuiltError:
            st.error(
                "The job-postings index hasn't been built yet. Run "
                "`python scripts/build_index.py` before using this app."
            )
            st.stop()
    result = pipeline_result.gap_analysis

    if apply_correction:
        corrections = pipeline_result.corrections
        label = f"Typo corrections applied before retrieval ({len(corrections)})"
        with st.expander(label, expanded=bool(corrections)):
            if corrections:
                st.table([
                    {"In resume": c.original_token, "Searched as": c.corrected_token,
                     "Edit distance": c.edit_distance}
                    for c in corrections
                ])
            else:
                st.write("No typos found. The resume was searched exactly as written.")

    with st.expander(f"Postings retrieved ({len(pipeline_result.retrieved)})"):
        st.caption(
            f"The resume was searched as {len(pipeline_result.sections)} separate sections "
            "(each within the embedding model's 512-token window), and results were "
            "combined per posting with reciprocal rank fusion."
        )
        for i, posting in enumerate(pipeline_result.retrieved, start=1):
            link = f"[{posting['title']}]({posting['job_posting_url']})" if posting["job_posting_url"] else posting["title"]
            st.markdown(f"**[{i}]** {link} · matched by: {', '.join(posting['matched_sections'])}")

    st.subheader("Matched")
    for claim in result.matched:
        with st.container(border=True):
            st.markdown(f"**{claim.requirement}**")
            st.caption(f"Supported by posting `{claim.job_id}`")
            if claim.job_posting_url:
                st.markdown(f"[View posting ↗]({claim.job_posting_url})")
            st.text(claim.supporting_excerpt)

    st.subheader("Missing")
    for claim in result.missing:
        with st.container(border=True):
            st.markdown(f"**{claim.requirement}**")
            st.caption(f"Required by posting `{claim.job_id}` — not evidenced in resume")
            if claim.job_posting_url:
                st.markdown(f"[View posting ↗]({claim.job_posting_url})")

    st.subheader("Summary")
    st.write(result.overall_summary)

    with st.expander("Raw model output (debugging)"):
        st.text(result.raw_response)
