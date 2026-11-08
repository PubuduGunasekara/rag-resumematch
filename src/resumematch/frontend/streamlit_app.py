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

from resumematch.pipeline import run_pipeline, IndexNotBuiltError

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


uploaded_file = st.file_uploader("Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    with st.spinner("Retrieving matching postings and running gap analysis..."):
        try:
            result = run_pipeline(tmp_path, domain_vocabulary=get_domain_vocabulary())
        except IndexNotBuiltError:
            st.error(
                "The job-postings index hasn't been built yet. Run "
                "`python scripts/build_index.py` before using this app."
            )
            st.stop()

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
