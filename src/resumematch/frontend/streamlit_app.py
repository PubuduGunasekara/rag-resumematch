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

import requests
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
        except requests.ConnectionError:
            st.error(
                "Couldn't reach the local LLM (Ollama). Is it running? "
                "Start it with `ollama serve` (or `brew services start ollama`)."
            )
            st.stop()
        except ValueError as error:
            # parse failure / truncated output, after the format-reminder retry
            st.error(
                "The language model's answer wasn't in a usable format, even after a retry. "
                "Upload the resume again to re-run it."
            )
            with st.expander("Details"):
                st.text(str(error))
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
            company = f" at {posting['company_name']}" if posting.get("company_name") else ""
            st.markdown(f"**[{i}]** {link}{company} · matched by: {', '.join(posting['matched_sections'])}")

    st.subheader("Matched")
    st.caption(
        "Each match was checked in code: the quoted text appears in the resume, it mentions the "
        "requirement, and the requirement appears in the cited posting."
    )
    if not result.matched:
        st.write("No verified matches.")
    for claim in result.matched:
        with st.container(border=True):
            st.markdown(f"**{claim.requirement}**")
            st.markdown(f"Resume: “{claim.resume_evidence}”")
            st.caption(f"Required by posting `{claim.job_id}`" + (" (citation corrected by check)" if claim.recited else ""))
            if claim.job_posting_url:
                st.markdown(f"[View posting ↗]({claim.job_posting_url})")
            with st.expander("Posting excerpt"):
                st.text(claim.supporting_excerpt)


    st.subheader("Missing")
    st.caption("Each gap was checked: the cited posting mentions it, and the resume doesn't.")
    if not result.missing:
        st.write("No verified gaps.")
    for claim in result.missing:
        with st.container(border=True):
            st.markdown(f"**{claim.requirement}**")
            st.caption(f"Required by posting `{claim.job_id}` — not evidenced in resume"
                       + (" (citation corrected by check)" if claim.recited else ""))
            if claim.job_posting_url:
                st.markdown(f"[View posting ↗]({claim.job_posting_url})")

    if result.unverified:
        with st.expander(f"Model claims that failed verification ({len(result.unverified)})"):
            st.caption("Not counted above. Each line says what the model claimed and which check it failed.")
            for claim in result.unverified:
                quote = f" — quoted “{claim.resume_evidence}”" if claim.resume_evidence else ""
                st.markdown(
                    f"- **{claim.requirement}** (claimed {claim.claimed_as}{quote}): {claim.issue} · posting `{claim.job_id}`"
                )

    st.subheader("Summary")
    st.caption("Written by the model and not verified; the checked lists above take precedence.")
    st.write(result.overall_summary)

    with st.expander("Raw model output (debugging)"):
        if result.attempts > 1:
            st.caption(f"The first answer was unusable; this is attempt {result.attempts}.")
        st.text(result.raw_response)
