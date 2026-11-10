"""Writes data/processed/posting_details.json (location, work type,
seniority, remote, pay per posting) for an index built before
build_index.py started writing it. Takes a few seconds; no re-indexing.

The results page works without this file; job cards just show no
location or pay.

Usage:
    python scripts/build_posting_details.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv

from resumematch.ingest.job_postings_loader import download_dataset, load_posting_details

OUT = Path("data/processed/posting_details.json")


def main():
    load_dotenv()
    csv_dir = download_dataset()  # cached by kagglehub after the first download
    details = load_posting_details(csv_dir)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(details))
    print(f"Wrote details for {len(details):,} postings to {OUT}.")


if __name__ == "__main__":
    main()
