"""Downloads the prebuilt sample index (2,000 postings) so ResumeMatch runs
without a Kaggle account or the 531 MB dataset.

The archive is a GitHub Release asset. This script checks its SHA-256,
refuses archives with unsafe paths, and unpacks it into chroma_db/ and
data/processed/. The data is CC BY-SA 4.0; its ATTRIBUTION.md and
DATA_LICENSE.txt are unpacked into data/processed/ (kept out of git).

Usage:
    python scripts/fetch_sample_index.py                 # download from the release
    python scripts/fetch_sample_index.py --file dist/resumematch-sample-index-v1.tar.gz
    python scripts/fetch_sample_index.py --force         # replace an existing index (moved aside)
"""

import argparse
import hashlib
import io
import tarfile
import time
from pathlib import Path, PurePosixPath

import requests

VERSION = "v1"
URL = ("https://github.com/PubuduGunasekara/rag-resumematch/releases/download/"
       f"sample-index-{VERSION}/resumematch-sample-index-{VERSION}.tar.gz")
SHA256 = "56b52bfd63997a8ec7c033bd4fa4878911a3b19255e72ded1fff9dceff556e99"
ALLOWED_TOP = {"chroma_db", "data", "ATTRIBUTION.md", "DATA_LICENSE.txt", "SAMPLE_INFO.json"}
NOTES = {"ATTRIBUTION.md", "DATA_LICENSE.txt", "SAMPLE_INFO.json"}  # placed in data/processed/
PROCESSED_FILES = ["domain_vocab.json", "posting_details.json", "boilerplate_sentences.json"]


def safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Regular files and folders only, inside the expected top-level names."""
    members = []
    for m in tar.getmembers():
        path = PurePosixPath(m.name)
        if path.is_absolute() or ".." in path.parts or path.parts[0] not in ALLOWED_TOP:
            raise SystemExit(f"Unsafe path in archive: {m.name!r}. Not extracting.")
        if not (m.isfile() or m.isdir()):
            raise SystemExit(f"Unexpected entry type in archive: {m.name!r}. Not extracting.")
        if m.name in NOTES:
            m.name = f"data/processed/{m.name}"
        members.append(m)
    return members


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, help="use a local archive instead of downloading")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--sha256", default=SHA256)
    parser.add_argument("--dest", type=Path, default=Path("."), help="repository root to unpack into")
    parser.add_argument("--force", action="store_true", help="move an existing index aside and replace it")
    args = parser.parse_args()

    index = args.dest / "chroma_db"
    processed = args.dest / "data" / "processed"
    existing = [p for p in [index, *(processed / f for f in PROCESSED_FILES)] if p.exists()]
    if existing and not args.force:
        raise SystemExit(f"{existing[0]} already exists. Use --force to move the existing files aside and replace them.")

    if args.file:
        data = args.file.read_bytes()
    else:
        print(f"Downloading {args.url} ...")
        response = requests.get(args.url, timeout=120)
        response.raise_for_status()
        data = response.content
    digest = hashlib.sha256(data).hexdigest()
    if digest != args.sha256:
        raise SystemExit(f"Checksum mismatch: expected {args.sha256}, got {digest}. Not extracting.")

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = safe_members(tar)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for path in existing:
            path.rename(path.with_name(f"{path.name}.before-sample-{stamp}"))
        args.dest.mkdir(parents=True, exist_ok=True)
        tar.extractall(args.dest, members=members, filter="data")

    print(f"Sample index ready in {index} ({len(data) / 1e6:.1f} MB archive, checksum verified).")
    print(f"The data is CC BY-SA 4.0: see {processed / 'ATTRIBUTION.md'} and {processed / 'DATA_LICENSE.txt'}.")
    if existing:
        print(f"Your previous files were moved aside with the suffix .before-sample-{stamp}.")


if __name__ == "__main__":
    main()
