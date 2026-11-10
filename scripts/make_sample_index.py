"""Builds the downloadable sample index: a small, ready-to-use index so
people can try ResumeMatch without a Kaggle account or the 531 MB dataset.

Steps: build an index of the first N distinct postings with contact
details redacted; keep the posting details for those postings only; add
the CC BY-SA 4.0 license and attribution; check that no email or phone
number is left; pack it as a .tar.gz and print its size and SHA-256.

Needs the dataset (Kaggle credentials in .env). Usage:
    python scripts/make_sample_index.py                    # 2,000 postings into dist/
    python scripts/make_sample_index.py --postings 5000 --version v2

Publish the archive as a GitHub Release asset and put its SHA-256 in
scripts/fetch_sample_index.py.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import date
from pathlib import Path

import chromadb
from chromadb.config import Settings

ATTRIBUTION = """# Attribution

This archive contains data derived from **LinkedIn Job Postings (2023-2024)**, collected by
ArshKA (https://github.com/ArshKA/LinkedIn-Job-Scraper) with additional postings contributed by
zoeyyuzou, published on Kaggle at https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
(version 13) under the Creative Commons Attribution-ShareAlike 4.0 International license.

This archive is distributed under the same license, CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). See DATA_LICENSE.txt.

## Changes made to the original data

- Only the first {postings:,} distinct postings of `postings.csv` are included.
- Postings with no description and exact duplicate descriptions were removed.
- Email addresses and phone numbers in posting text were replaced with "[email removed]" and
  "[phone removed]".
- Repeated boilerplate sentences (equal opportunity and similar statements) were removed.
- Descriptions were split into chunks of up to 300 words and embedded with the
  `BAAI/bge-small-en-v1.5` model; each chunk is stored with its job ID, title, company and
  posting URL in a ChromaDB index.
- `posting_details.json` keeps location, work type, experience level, remote flag and pay for the
  included postings. `domain_vocab.json` is a word-frequency list built from all postings (after
  contact redaction). `boilerplate_sentences.json` lists the removed sentences.

Built on {built} by `scripts/make_sample_index.py` in https://github.com/PubuduGunasekara/rag-resumematch
(commit {commit}).
"""

DATA_LICENSE = """The data in this archive is licensed under the Creative Commons
Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).

Full license text: https://creativecommons.org/licenses/by-sa/4.0/legalcode
Summary: https://creativecommons.org/licenses/by-sa/4.0/

You may share and adapt this data for any purpose, provided you give appropriate credit (see
ATTRIBUTION.md), indicate if changes were made, and distribute your contributions under the same
license. The ResumeMatch source code is separately licensed under the MIT License.
"""

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<![\w-])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\w-])")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--postings", type=int, default=2000)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--out", type=Path, default=Path("dist"))
    args = parser.parse_args()

    name = f"resumematch-sample-index-{args.version}"
    stage = args.out / name
    if stage.exists():
        shutil.rmtree(stage)
    processed = stage / "data" / "processed"
    env = {**os.environ, "CHROMA_PERSIST_DIR": str(stage / "chroma_db")}
    subprocess.run([sys.executable, "scripts/build_index.py", "--limit", str(args.postings),
                    "--processed-dir", str(processed), "--redact-contacts"], env=env, check=True)

    collection = chromadb.PersistentClient(path=str(stage / "chroma_db"), settings=Settings(anonymized_telemetry=False)) \
        .get_collection("job_posting_chunks")
    stored = collection.get(include=["documents", "metadatas"])
    job_ids = {m["job_id"] for m in stored["metadatas"]}
    details = json.loads((processed / "posting_details.json").read_text())
    (processed / "posting_details.json").write_text(json.dumps({j: details[j] for j in job_ids if j in details}))

    leftovers = [d for d in stored["documents"] if EMAIL.search(d) or PHONE.search(d)]
    leftovers += [w for w in json.loads((processed / "domain_vocab.json").read_text()) if "@" in w]
    if leftovers:
        raise SystemExit(f"{len(leftovers)} items still contain contact details; not packing. First: {leftovers[0][:80]!r}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    (stage / "ATTRIBUTION.md").write_text(ATTRIBUTION.format(postings=len(job_ids), built=date.today().isoformat(),
                                                             commit=commit or "unknown"))
    (stage / "DATA_LICENSE.txt").write_text(DATA_LICENSE)
    info = {"version": args.version, "postings": len(job_ids), "chunks": collection.count(),
            "embedding_model": "BAAI/bge-small-en-v1.5", "chromadb": chromadb.__version__,
            "hnsw": collection.metadata, "contacts_redacted": True}
    (stage / "SAMPLE_INFO.json").write_text(json.dumps(info, indent=2))

    archive = args.out / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for item in ["chroma_db", "data", "ATTRIBUTION.md", "DATA_LICENSE.txt", "SAMPLE_INFO.json"]:
            tar.add(stage / item, arcname=item)
    print(json.dumps(info, indent=2))
    print(f"\n{archive}: {archive.stat().st_size / 1e6:.1f} MB")
    print(f"sha256: {sha256(archive)}")


if __name__ == "__main__":
    main()
