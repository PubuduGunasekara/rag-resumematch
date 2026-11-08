#!/bin/sh
# Satisfies the spec's "AutomatedDownload" requirement: the container
# downloads and processes the corpus itself on first run, rather than
# expecting a pre-baked dataset to already be mounted.
#
# Idempotent: if chroma_db already has data (a re-run, or a volume
# that persisted from a previous `docker run`), skip straight to
# launching the app instead of re-indexing 124k postings again.
set -e

CHROMA_DIR="${CHROMA_PERSIST_DIR:-./chroma_db}"

if [ ! -d "$CHROMA_DIR" ] || [ -z "$(ls -A "$CHROMA_DIR" 2>/dev/null)" ]; then
  echo "No existing index found at $CHROMA_DIR — building it now."
  echo "(First run only. This downloads 124k+ job postings and will take a while.)"
  python scripts/build_index.py
else
  echo "Existing index found at $CHROMA_DIR — skipping build. Delete the volume to force a rebuild."
fi

exec streamlit run src/resumematch/frontend/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
