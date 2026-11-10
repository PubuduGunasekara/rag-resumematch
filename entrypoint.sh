#!/bin/sh
# Container start-up:
#   1. wait for Ollama and pull OLLAMA_MODEL if missing
#   2. build the index if there is no COMPLETE one (the spec's
#      "AutomatedDownload": a fresh clone downloads and indexes the corpus
#      itself, ~40 min on a laptop CPU)
#   3. write posting details (location, pay) if missing
#   4. start Streamlit
#
# The index (./chroma_db) and its vocabulary/boilerplate files
# (./data/processed) are bind-mounted from the host, so an index built
# once -- on your laptop, or copied to the GCP VM -- is reused instead of
# rebuilt, and survives container restarts.
set -e

CHROMA_DIR="${CHROMA_PERSIST_DIR:-/app/chroma_db}"

python scripts/wait_for_ollama.py

if [ -f "$CHROMA_DIR/BUILD_COMPLETE.json" ] && [ -f data/processed/domain_vocab.json ]; then
  echo "Complete index found at $CHROMA_DIR:"
  cat "$CHROMA_DIR/BUILD_COMPLETE.json"
else
  echo "No complete index at $CHROMA_DIR -- building it now (needs KAGGLE_USERNAME/KAGGLE_KEY)."
  echo "This downloads ~124k job postings and takes a while (~40 min on a laptop CPU)."
  # A partial index from a crashed build must not be appended to.
  rm -rf "${CHROMA_DIR:?}"/*
  python scripts/build_index.py ${BUILD_INDEX_ARGS:-}
fi

# Location, pay etc. for the job cards; an index built before this file
# existed won't have it. Seconds, and optional: the app works without it.
if [ ! -f data/processed/posting_details.json ]; then
  python scripts/build_posting_details.py || echo "Skipping posting details (dataset not available)."
fi

exec streamlit run src/resumematch/frontend/streamlit_app.py \
  --server.address=0.0.0.0 --server.port=8501 --browser.gatherUsageStats=false
